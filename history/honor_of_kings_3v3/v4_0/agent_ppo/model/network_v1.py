#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""Active 3v3 network components.

The historical ``V1`` class suffixes remain as platform-facing import names;
the checkpoint signature identifies the active hybrid multi-value network.
All tensors use the canonical layout ``[batch, actor=2, time, ...]``.
"""

import math
from collections import OrderedDict

import torch
import torch.nn as nn

from agent_ppo.conf.conf import FrozenV1Config
from agent_ppo.feature import schema_v1


def _orthogonal_init(module, gain=math.sqrt(2.0)):
    if isinstance(module, (nn.Linear, nn.Conv2d)):
        nn.init.orthogonal_(module.weight, gain=gain)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    return module


class LinearNormRelu(nn.Module):
    """Frozen default block: Linear -> LayerNorm -> ReLU."""

    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear = _orthogonal_init(nn.Linear(input_dim, output_dim))
        self.norm = nn.LayerNorm(output_dim)
        self.activation = nn.ReLU()

    def forward(self, value):
        return self.activation(self.norm(self.linear(value)))


class LinearReluNorm(nn.Module):
    """PSCN block, whose frozen formula applies LayerNorm after ReLU."""

    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear = _orthogonal_init(nn.Linear(input_dim, output_dim))
        self.norm = nn.LayerNorm(output_dim)
        self.activation = nn.ReLU()

    def forward(self, value):
        return self.norm(self.activation(self.linear(value)))


class LinearNorm(nn.Module):
    """Linear projection with LayerNorm for attention Key/Value tokens."""

    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear = _orthogonal_init(nn.Linear(input_dim, output_dim))
        self.norm = nn.LayerNorm(output_dim)

    def forward(self, value):
        return self.norm(self.linear(value))


class FrozenMLP(nn.Module):
    def __init__(self, dimensions, final_relu=True):
        super().__init__()
        layers = []
        for index, (input_dim, output_dim) in enumerate(zip(dimensions[:-1], dimensions[1:])):
            is_last = index == len(dimensions) - 2
            layers.append(
                LinearNormRelu(input_dim, output_dim)
                if final_relu or not is_last
                else LinearNorm(input_dim, output_dim)
            )
        self.layers = nn.Sequential(*layers)

    def forward(self, value):
        return self.layers(value)


class LightCNNV1(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = _orthogonal_init(nn.Conv2d(6, 18, kernel_size=5, padding=2))
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv2 = _orthogonal_init(nn.Conv2d(18, 12, kernel_size=3, padding=1))
        self.projection = LinearNormRelu(12 * 8 * 8, 128)

    def forward(self, image):
        leading = image.shape[:-3]
        flattened = image.reshape(-1, 6, 17, 17)
        encoded = torch.relu(self.conv1(flattened))
        encoded = self.pool(encoded)
        encoded = torch.relu(self.conv2(encoded))
        if encoded.shape[-3:] != (12, 8, 8):
            raise RuntimeError(f"LightCNN v1 convolution shape drifted: {tuple(encoded.shape[-3:])}")
        encoded = self.projection(encoded.flatten(start_dim=1))
        return encoded.reshape(*leading, 128)


def masked_max_mean(tokens, mask):
    """Return concatenated max and mean; all-empty rows are exactly zero."""

    if tokens.shape[:-1] != mask.shape:
        raise ValueError(
            f"masked_max_mean shape mismatch: tokens={tuple(tokens.shape)}, mask={tuple(mask.shape)}"
        )
    mask = mask.to(dtype=torch.bool)
    expanded = mask.unsqueeze(-1)
    count = expanded.sum(dim=-2)
    mean = (tokens * expanded.to(tokens.dtype)).sum(dim=-2) / count.clamp_min(1).to(tokens.dtype)
    minimum = torch.finfo(tokens.dtype).min
    maximum = tokens.masked_fill(~expanded, minimum).max(dim=-2).values
    empty = count == 0
    maximum = torch.where(empty, torch.zeros_like(maximum), maximum)
    mean = torch.where(empty, torch.zeros_like(mean), mean)
    return torch.cat((maximum, mean), dim=-1)


class FeatureEncoderV1(nn.Module):
    """Exact v1 feature splitter, entity encoders, and pooling contract."""

    PUBLIC_EXCLUDED_FIELDS = {
        "heroes": ("is_query_main", "toward_query_hero"),
        "lane_state": ("query_has_minion_cover",),
        "resource_anchors": ("query_eta_norm",),
    }
    FULL_PUBLIC_BLOCKS = (
        "soldiers",
        "organs",
        "monsters",
        "tyrant_team_context",
        "collaboration_state",
        "global_state",
    )

    def __init__(self, assert_public_raw_equal=True):
        super().__init__()
        self.assert_public_raw_equal = bool(assert_public_raw_equal)
        self.image_encoder = LightCNNV1()
        self.hero_encoder = FrozenMLP((393, 256, 128))
        self.relation_dynamic_encoder = FrozenMLP((24 + 72, 128, 64))
        self.soldier_encoder = FrozenMLP((56, 64, 32))
        self.organ_encoder = FrozenMLP((48, 64, 32))
        self.monster_encoder = FrozenMLP((60, 64, 32))
        self.bullet_encoder = FrozenMLP((32, 64, 32))
        self.target_encoder = FrozenMLP((48, 128, 64), final_relu=False)
        self.resource_encoder = FrozenMLP((32, 64, 32))

    @staticmethod
    def split_feature(feature):
        if feature.ndim != 4:
            raise ValueError(
                f"FeatureEncoderV1 expects [batch,actor,time,{FrozenV1Config.FEATURE_DIM}], "
                f"got {tuple(feature.shape)}"
            )
        if feature.shape[1] != FrozenV1Config.CONTROLLED_ACTOR_COUNT:
            raise ValueError(f"v1 actor dimension must be 2, got {feature.shape[1]}")
        if feature.shape[-1] != FrozenV1Config.FEATURE_DIM:
            raise ValueError(
                f"v1 feature dimension must be {FrozenV1Config.FEATURE_DIM}, "
                f"got {feature.shape[-1]}"
            )
        blocks = OrderedDict()
        for spec in schema_v1.FEATURE_BLOCKS:
            blocks[spec.name] = feature[..., spec.start : spec.stop].reshape(
                *feature.shape[:-1], *spec.shape
            )
        return blocks

    @staticmethod
    def _field_keep_mask(block_name, device):
        field_count = schema_v1.FEATURE_BLOCK_BY_NAME[block_name].shape[-1]
        keep = torch.ones(field_count, dtype=torch.bool, device=device)
        field_map = schema_v1.FIELD_MAPS[block_name]
        for name in FeatureEncoderV1.PUBLIC_EXCLUDED_FIELDS.get(block_name, ()):
            keep[field_map[name].start] = False
        return keep

    def _assert_public_blocks(self, blocks):
        if not self.assert_public_raw_equal:
            return
        for name in self.FULL_PUBLIC_BLOCKS:
            first, second = blocks[name][:, 0], blocks[name][:, 1]
            if not torch.equal(first, second):
                difference = (first - second).abs().max().item()
                raise ValueError(
                    f"v1 public raw block differs across controlled actors: {name}, max_abs={difference}"
                )
        for name in self.PUBLIC_EXCLUDED_FIELDS:
            keep = self._field_keep_mask(name, blocks[name].device)
            first = blocks[name][:, 0, ..., keep]
            second = blocks[name][:, 1, ..., keep]
            if not torch.equal(first, second):
                difference = (first - second).abs().max().item()
                raise ValueError(
                    f"v1 public fields differ across controlled actors: {name}, max_abs={difference}"
                )

    @staticmethod
    def _mask(block, block_name, field_name):
        index = schema_v1.FIELD_MAPS[block_name][field_name].start
        return block[..., index] > 0.5

    def forward(self, feature, validate_runtime=True):
        if not torch.is_floating_point(feature):
            feature = feature.float()
        if validate_runtime and not torch.isfinite(feature).all():
            raise ValueError("FeatureEncoderV1 input contains NaN/Inf")
        blocks = self.split_feature(feature)
        if validate_runtime:
            self._assert_public_blocks(blocks)

        image_embedding = self.image_encoder(blocks["image"])
        hero_tokens = self.hero_encoder(blocks["heroes"])
        relation_dynamic = torch.cat(
            (blocks["hero_relations"], blocks["dynamic_lane_position"]), dim=-1
        )
        relation_dynamic_tokens = self.relation_dynamic_encoder(relation_dynamic)
        soldier_tokens = self.soldier_encoder(blocks["soldiers"])
        organ_tokens = self.organ_encoder(blocks["organs"])
        monster_tokens = self.monster_encoder(blocks["monsters"])
        bullet_tokens = self.bullet_encoder(blocks["bullets"])
        target_bank = self.target_encoder(blocks["targets"])
        resource_tokens = self.resource_encoder(blocks["resource_anchors"])

        masks = {
            "heroes": self._mask(blocks["heroes"], "heroes", "slot_exists"),
            "soldiers": self._mask(blocks["soldiers"], "soldiers", "slot_exists"),
            "organs": self._mask(blocks["organs"], "organs", "slot_exists"),
            "monsters": self._mask(blocks["monsters"], "monsters", "slot_exists"),
            "bullets": self._mask(blocks["bullets"], "bullets", "slot_exists"),
            "targets": self._mask(blocks["targets"], "targets", "slot_valid"),
            "resources": self._mask(
                blocks["resource_anchors"], "resource_anchors", "slot_exists"
            ),
            "collaboration_targets": self._mask(
                blocks["targets"], "targets", "collaboration_target_legal"
            ),
        }

        soldier_summary = masked_max_mean(soldier_tokens, masks["soldiers"])
        monster_summary = masked_max_mean(monster_tokens, masks["monsters"])
        bullet_summary = masked_max_mean(bullet_tokens, masks["bullets"])
        target_summary = masked_max_mean(target_bank, masks["targets"])

        team_input = torch.cat(
            (
                hero_tokens.flatten(start_dim=-2),
                soldier_summary,
                organ_tokens.flatten(start_dim=-2),
                monster_summary,
                resource_tokens.flatten(start_dim=-2),
                blocks["lane_state"].flatten(start_dim=-2),
                blocks["tyrant_team_context"].flatten(start_dim=-2),
                blocks["collaboration_state"].flatten(start_dim=-2),
                blocks["global_state"].flatten(start_dim=-2),
            ),
            dim=-1,
        )
        private_input = torch.cat(
            (
                image_embedding,
                relation_dynamic_tokens.flatten(start_dim=-2),
                bullet_summary,
                target_summary,
                blocks["tyrant_local_position"].flatten(start_dim=-2),
                blocks["jungle_plan"].flatten(start_dim=-2),
                blocks["lane_plan"].flatten(start_dim=-2),
                blocks["main_context"].flatten(start_dim=-2),
            ),
            dim=-1,
        )
        if team_input.shape[-1] != schema_v1.TEAM_INPUT_DIM:
            raise RuntimeError(f"team input dimension drifted: {team_input.shape[-1]}")
        if private_input.shape[-1] != schema_v1.PRIVATE_INPUT_DIM:
            raise RuntimeError(f"private input dimension drifted: {private_input.shape[-1]}")
        return {
            "blocks": blocks,
            "masks": masks,
            "image_embedding": image_embedding,
            "hero_tokens": hero_tokens,
            "relation_dynamic_tokens": relation_dynamic_tokens,
            "soldier_tokens": soldier_tokens,
            "organ_tokens": organ_tokens,
            "monster_tokens": monster_tokens,
            "bullet_tokens": bullet_tokens,
            "target_bank": target_bank,
            "resource_tokens": resource_tokens,
            "team_input": team_input,
            "private_input": private_input,
        }


class ResidualPSCNV1(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.input_projection = LinearReluNorm(input_dim, 256)
        self.hidden1 = LinearReluNorm(256, 256)
        self.hidden2 = LinearReluNorm(128, 128)
        self.hidden3 = LinearReluNorm(64, 64)
        self.residual_projection = _orthogonal_init(nn.Linear(256, 256))
        self.output_norm = nn.LayerNorm(256)

    def forward(self, value):
        x0 = self.input_projection(value)
        h1 = self.hidden1(x0)
        p1, r1 = h1.split((128, 128), dim=-1)
        h2 = self.hidden2(r1)
        p2, r2 = h2.split((64, 64), dim=-1)
        p3 = self.hidden3(r2)
        return self.output_norm(x0 + self.residual_projection(torch.cat((p1, p2, p3), dim=-1)))


class ActorBackboneV1(nn.Module):
    """Frozen encoder, independent team/private PSCNs, and true GRU."""

    def __init__(self, assert_public_raw_equal=True):
        super().__init__()
        self.encoder = FeatureEncoderV1(assert_public_raw_equal=assert_public_raw_equal)
        self.team_pscn = ResidualPSCNV1(schema_v1.TEAM_INPUT_DIM)
        self.private_pscn = ResidualPSCNV1(schema_v1.PRIVATE_INPUT_DIM)
        self.actor_projection = LinearReluNorm(512, 256)
        self.static_path = FrozenMLP((256, 128, 128))
        self.pre_gru = LinearNormRelu(256, 128)
        self.gru = nn.GRU(128, FrozenV1Config.RNN_HIDDEN_SIZE, num_layers=1, batch_first=True)
        for name, parameter in self.gru.named_parameters():
            if "weight" in name:
                nn.init.orthogonal_(parameter)
            elif "bias" in name:
                nn.init.zeros_(parameter)
        self.policy_fusion = LinearNormRelu(256, 256)

    def _run_gru(
        self,
        gru_input,
        initial_hidden,
        reset_mask=None,
        has_internal_reset=None,
    ):
        batch, actors, time, width = gru_input.shape
        sequence = gru_input.reshape(batch * actors, time, width)
        hidden = initial_hidden.reshape(batch * actors, FrozenV1Config.RNN_HIDDEN_SIZE).unsqueeze(0)
        if reset_mask is None:
            output, next_hidden = self.gru(sequence, hidden)
        else:
            if reset_mask.shape != (batch, actors, time):
                raise ValueError(
                    f"reset mask must be {(batch, actors,time)}, got {tuple(reset_mask.shape)}"
                )
            reset = reset_mask.reshape(batch * actors, time).to(torch.bool)
            # Most replay sequences have no episode/death reset after their
            # first step. Apply a possible step-zero reset to the initial state
            # and use one fused GRU call; retain the exact stepwise path whenever
            # a reset occurs inside the sequence.
            if has_internal_reset is None:
                has_internal_reset = time > 1 and bool(
                    torch.any(reset[:, 1:]).item()
                )
            else:
                has_internal_reset = bool(has_internal_reset) and time > 1
            if not has_internal_reset:
                keep = (~reset[:, 0]).to(sequence.dtype).view(
                    1, batch * actors, 1
                )
                output, next_hidden = self.gru(sequence, hidden * keep)
            else:
                outputs = []
                current_hidden = hidden
                for step in range(time):
                    keep = (~reset[:, step]).to(sequence.dtype).view(
                        1, batch * actors, 1
                    )
                    current_hidden = current_hidden * keep
                    step_output, current_hidden = self.gru(
                        sequence[:, step : step + 1], current_hidden
                    )
                    outputs.append(step_output)
                output = torch.cat(outputs, dim=1)
                next_hidden = current_hidden
        return (
            output.reshape(batch, actors, time, FrozenV1Config.RNN_HIDDEN_SIZE),
            next_hidden.squeeze(0).reshape(batch, actors, FrozenV1Config.RNN_HIDDEN_SIZE),
        )

    def forward(
        self,
        feature,
        initial_hidden,
        reset_mask=None,
        has_internal_reset=None,
        validate_runtime=True,
    ):
        if feature.shape[2] not in (1, FrozenV1Config.RNN_TIME_STEPS):
            raise ValueError(f"v1 supports time=1 or 16, got {feature.shape[2]}")
        expected_hidden = (
            feature.shape[0],
            FrozenV1Config.CONTROLLED_ACTOR_COUNT,
            FrozenV1Config.RNN_HIDDEN_SIZE,
        )
        if tuple(initial_hidden.shape) != expected_hidden:
            raise ValueError(
                f"initial GRU hidden must be {expected_hidden}, got {tuple(initial_hidden.shape)}"
            )
        encoded = self.encoder(feature, validate_runtime=validate_runtime)
        team_feature = self.team_pscn(encoded["team_input"])
        private_feature = self.private_pscn(encoded["private_input"])
        actor_context = self.actor_projection(torch.cat((team_feature, private_feature), dim=-1))
        static_feature = self.static_path(actor_context)
        gru_input = self.pre_gru(actor_context)
        gru_output, next_hidden = self._run_gru(
            gru_input,
            initial_hidden,
            reset_mask,
            has_internal_reset=has_internal_reset,
        )
        policy_feature = self.policy_fusion(torch.cat((static_feature, gru_output), dim=-1))
        encoded.update(
            {
                "team_feature": team_feature,
                "private_feature": private_feature,
                "actor_context": actor_context,
                "static_feature": static_feature,
                "gru_input": gru_input,
                "gru_output": gru_output,
                "policy_feature": policy_feature,
                "next_hidden": next_hidden,
            }
        )
        return encoded


class PolicyHeadV1(nn.Module):
    def __init__(self, output_dim):
        super().__init__()
        self.hidden = LinearNormRelu(256, 64)
        self.output = _orthogonal_init(nn.Linear(64, output_dim), gain=0.01)

    def forward(self, policy_feature):
        return self.output(self.hidden(policy_feature))


class HeroRouteHeadsV1(nn.Module):
    def __init__(self):
        super().__init__()
        self.ordinary_heads = nn.ModuleList(
            PolicyHeadV1(output_dim)
            for output_dim in FrozenV1Config.HERO_ACTION_HEADS[:-1]
        )
        self.button_embedding = nn.Embedding(FrozenV1Config.HERO_ACTION_HEADS[0], 16)
        nn.init.normal_(self.button_embedding.weight, mean=0.0, std=0.02)
        self.target_query = LinearNorm(256 + 16, 64)

    def forward(self, policy_feature, target_keys):
        ordinary = [head(policy_feature) for head in self.ordinary_heads]
        button_ids = torch.arange(
            FrozenV1Config.HERO_ACTION_HEADS[0], device=policy_feature.device
        )
        embeddings = self.button_embedding(button_ids)
        expanded_policy = policy_feature.unsqueeze(-2).expand(
            *policy_feature.shape[:-1], FrozenV1Config.HERO_ACTION_HEADS[0], 256
        )
        expanded_embedding = embeddings.reshape(
            *((1,) * (policy_feature.ndim - 1)),
            FrozenV1Config.HERO_ACTION_HEADS[0],
            16,
        ).expand(*policy_feature.shape[:-1], FrozenV1Config.HERO_ACTION_HEADS[0], 16)
        query = self.target_query(torch.cat((expanded_policy, expanded_embedding), dim=-1))
        target_logits = torch.einsum("...bd,...td->...bt", query, target_keys) / math.sqrt(64.0)
        return ordinary, target_logits


class HeroPolicyRouterV1(nn.Module):
    """Deterministic 132/141/193 routing with no learned gate or fallback."""

    ROUTE_IDS = tuple(hero_id for hero_id, _ in FrozenV1Config.HERO_POLICY_ROUTES)
    ROUTE_NAMES = tuple(name for _, name in FrozenV1Config.HERO_POLICY_ROUTES)

    def __init__(self):
        super().__init__()
        self.routes = nn.ModuleDict(
            (name, HeroRouteHeadsV1()) for name in self.ROUTE_NAMES
        )
        self.target_key = LinearNorm(64, 64)
        self.target_value = LinearNorm(64, 64)

    @classmethod
    def decode_hero_ids(cls, main_context, validate_runtime=True):
        if main_context.shape[-1] != 32:
            raise ValueError(f"main_context must end in 32, got {main_context.shape[-1]}")
        slots = main_context[..., :6]
        active = slots > 0.5
        if validate_runtime and not torch.all(active.sum(dim=-1) == 1):
            raise ValueError("each v1 actor sample must contain exactly one query hero slot")
        indices = active.to(torch.int64).argmax(dim=-1)
        if validate_runtime and torch.any(indices >= 3):
            raise ValueError("controlled actor routed to an enemy hero slot")
        route_ids = torch.tensor(cls.ROUTE_IDS, device=indices.device, dtype=torch.long)
        return route_ids[indices]

    @classmethod
    def validate_hero_ids(cls, hero_ids):
        valid = torch.zeros_like(hero_ids, dtype=torch.bool)
        for hero_id in cls.ROUTE_IDS:
            valid |= hero_ids == hero_id
        if not torch.all(valid):
            invalid = torch.unique(hero_ids[~valid]).detach().cpu().tolist()
            raise ValueError(f"unroutable hero config IDs in v1 batch: {invalid}")

    def forward(
        self,
        policy_feature,
        target_bank,
        hero_ids,
        validate_runtime=True,
    ):
        if tuple(policy_feature.shape[:-1]) != tuple(hero_ids.shape):
            raise ValueError(
                f"hero route shape mismatch: policy={tuple(policy_feature.shape)}, ids={tuple(hero_ids.shape)}"
            )
        if target_bank.shape[:-2] != policy_feature.shape[:-1] or target_bank.shape[-2:] != (39, 64):
            raise ValueError(
                f"target bank must align as [...,39,64], got {tuple(target_bank.shape)}"
            )
        if validate_runtime:
            self.validate_hero_ids(hero_ids)
        target_keys = self.target_key(target_bank)
        target_values = self.target_value(target_bank)
        route_outputs = [
            self.routes[name](policy_feature, target_keys) for name in self.ROUTE_NAMES
        ]
        ordinary = []
        for head_index in range(4):
            stacked = torch.stack(
                [output[0][head_index] for output in route_outputs], dim=-2
            )
            weights = torch.stack(
                [(hero_ids == hero_id) for hero_id in self.ROUTE_IDS], dim=-1
            ).to(stacked.dtype)
            ordinary.append((stacked * weights.unsqueeze(-1)).sum(dim=-2))
        stacked_target = torch.stack([output[1] for output in route_outputs], dim=-3)
        target_weights = torch.stack(
            [(hero_ids == hero_id) for hero_id in self.ROUTE_IDS], dim=-1
        ).to(stacked_target.dtype)
        target_logits = (stacked_target * target_weights.unsqueeze(-1).unsqueeze(-1)).sum(
            dim=-3
        )
        return {
            "ordinary_logits": ordinary,
            "target_logits": target_logits,
            "target_keys": target_keys,
            "target_values": target_values,
            "hero_ids": hero_ids,
        }


class ActorValueHeadV1(nn.Module):
    """Hybrid final critic plus five independently supervised auxiliary values.

    ``self`` is query-specific, while ``team`` and ``enemy`` are explicitly
    averaged across the two controlled observations and then broadcast back.
    The three hybrid components are latent factors supervised only through the
    final scalar return.  The five auxiliary channels receive observable
    reward-detail lambda returns and never enter the PPO policy advantage.
    """

    def __init__(self):
        super().__init__()
        self.team_lambda = float(FrozenV1Config.HYBRID_VALUE_TEAM_LAMBDA)
        self.self_hidden = LinearNormRelu(384, 128)
        self.team_hidden = LinearNormRelu(256, 128)
        self.enemy_hidden = LinearNormRelu(448, 128)
        self.self_output = _orthogonal_init(nn.Linear(128, 1), gain=0.1)
        self.team_output = _orthogonal_init(nn.Linear(128, 1), gain=0.1)
        self.enemy_output = _orthogonal_init(nn.Linear(128, 1), gain=0.1)
        self.auxiliary_hidden = LinearNormRelu(3 * 128, 128)
        self.auxiliary_output = _orthogonal_init(
            nn.Linear(128, FrozenV1Config.AUXILIARY_VALUE_COUNT), gain=0.1
        )

    @staticmethod
    def _actor_shared(value):
        return value.mean(dim=1, keepdim=True).expand(
            -1, FrozenV1Config.CONTROLLED_ACTOR_COUNT, -1, -1
        )

    def forward(self, backbone):
        self_context = torch.cat(
            (backbone["private_feature"], backbone["gru_output"]), dim=-1
        )
        friendly_hero_summary = masked_max_mean(
            backbone["hero_tokens"][..., :3, :],
            backbone["masks"]["heroes"][..., :3],
        )
        team_context = self._actor_shared(friendly_hero_summary)

        enemy_hero_summary = masked_max_mean(
            backbone["hero_tokens"][..., 3:6, :],
            backbone["masks"]["heroes"][..., 3:6],
        )
        enemy_organ_summary = masked_max_mean(
            backbone["organ_tokens"][..., 3:6, :],
            backbone["masks"]["organs"][..., 3:6],
        )
        global_state = backbone["blocks"]["global_state"].squeeze(-2)
        enemy_context = self._actor_shared(
            torch.cat(
                (enemy_hero_summary, enemy_organ_summary, global_state), dim=-1
            )
        )

        if self_context.shape[-1] != 384:
            raise RuntimeError(f"self value input dimension drifted: {self_context.shape[-1]}")
        if team_context.shape[-1] != 256:
            raise RuntimeError(f"team value input dimension drifted: {team_context.shape[-1]}")
        if enemy_context.shape[-1] != 448:
            raise RuntimeError(f"enemy value input dimension drifted: {enemy_context.shape[-1]}")

        self_hidden = self.self_hidden(self_context)
        team_hidden = self.team_hidden(team_context)
        enemy_hidden = self.enemy_hidden(enemy_context)
        self_value = self.self_output(self_hidden)
        team_value = self.team_output(team_hidden)
        enemy_value = self.enemy_output(enemy_hidden)
        actor_value = (
            (1.0 - self.team_lambda) * self_value
            + self.team_lambda * team_value
            - enemy_value
        )
        auxiliary_values = self.auxiliary_output(
            self.auxiliary_hidden(
                torch.cat((self_hidden, team_hidden, enemy_hidden), dim=-1)
            )
        )
        return {
            "actor_value": actor_value,
            "actor_value_self": self_value,
            "actor_value_team": team_value,
            "actor_value_enemy": enemy_value,
            "actor_value_components": torch.cat(
                (self_value, team_value, enemy_value), dim=-1
            ),
            "actor_auxiliary_values": auxiliary_values,
        }


class ActorNetworkV1(nn.Module):
    """Stages 4-7: frozen actor backbone, routing, target attention, value."""

    def __init__(self, assert_public_raw_equal=True):
        super().__init__()
        self.backbone = ActorBackboneV1(
            assert_public_raw_equal=assert_public_raw_equal
        )
        self.policy_router = HeroPolicyRouterV1()
        self.value_head = ActorValueHeadV1()

    def forward(
        self,
        feature,
        initial_hidden,
        reset_mask=None,
        hero_ids=None,
        has_internal_reset=None,
        validate_runtime=True,
    ):
        backbone = self.backbone(
            feature,
            initial_hidden,
            reset_mask=reset_mask,
            has_internal_reset=has_internal_reset,
            validate_runtime=validate_runtime,
        )
        if hero_ids is None:
            main_context = backbone["blocks"]["main_context"].squeeze(-2)
            hero_ids = self.policy_router.decode_hero_ids(
                main_context,
                validate_runtime=validate_runtime,
            )
        routed = self.policy_router(
            backbone["policy_feature"],
            backbone["target_bank"],
            hero_ids,
            validate_runtime=validate_runtime,
        )
        routed.update(self.value_head(backbone))
        routed.update(backbone)
        return routed


class SemanticPositionBankV1(nn.Module):
    """Build the semantic position bank, including all twenty jungle camps."""

    POSITION_COUNT = FrozenV1Config.COMMAND_HEADS[-1]

    def __init__(self):
        super().__init__()
        self.encoder = FrozenMLP((3 + self.POSITION_COUNT, 64, 64), final_relu=False)
        self.key = LinearNorm(64, 64)
        self.value = LinearNorm(64, 64)

    @staticmethod
    def _field(block, block_name, field_name):
        return block[..., schema_v1.FIELD_MAPS[block_name][field_name].start]

    def forward(self, blocks):
        # Public source fields are checked by FeatureEncoderV1.  Actor 0 is
        # therefore a deterministic carrier for the team-level position bank.
        heroes = blocks["heroes"][:, 0]
        lane = blocks["lane_state"][:, 0, ..., 0, :]
        organs = blocks["organs"][:, 0]
        monsters = blocks["monsters"][:, 0]
        batch, time = heroes.shape[:2]
        coordinates = heroes.new_zeros(batch, time, self.POSITION_COUNT, 2)
        valid = torch.zeros(
            batch, time, self.POSITION_COUNT, dtype=torch.bool, device=heroes.device
        )

        hero_x = self._field(heroes[..., :3, :], "heroes", "side_x_norm")
        hero_z = self._field(heroes[..., :3, :], "heroes", "side_z_norm")
        hero_valid = (
            self._field(heroes[..., :3, :], "heroes", "location_valid") > 0.5
        ) & (self._field(heroes[..., :3, :], "heroes", "alive") > 0.5)
        coordinates[..., 1:4, 0] = hero_x
        coordinates[..., 1:4, 1] = hero_z
        valid[..., 1:4] = hero_valid

        own_front_valid = self._field(lane, "lane_state", "own_front_valid") > 0.5
        coordinates[..., 4, 0] = self._field(
            lane, "lane_state", "own_front_side_x_norm"
        )
        coordinates[..., 4, 1] = self._field(
            lane, "lane_state", "own_front_side_z_norm"
        )
        valid[..., 4] = own_front_valid

        clash_valid = self._field(lane, "lane_state", "clash_position_valid") > 0.5
        enemy_front_valid = (
            self._field(lane, "lane_state", "enemy_front_observed") > 0.5
        ) | (self._field(lane, "lane_state", "enemy_front_inferred") > 0.5)
        clash_x = self._field(lane, "lane_state", "clash_side_x_norm")
        enemy_x = self._field(lane, "lane_state", "enemy_front_side_x_norm")
        enemy_z = self._field(lane, "lane_state", "enemy_front_side_z_norm")
        coordinates[..., 5, 0] = torch.where(clash_valid, clash_x, enemy_x)
        coordinates[..., 5, 1] = torch.where(
            clash_valid, torch.zeros_like(enemy_z), enemy_z
        )
        valid[..., 5] = clash_valid | enemy_front_valid
        no_lane_anchor = ~valid[..., 5]
        coordinates[..., 5, 0] = torch.where(
            no_lane_anchor, torch.zeros_like(coordinates[..., 5, 0]), coordinates[..., 5, 0]
        )
        coordinates[..., 5, 1] = torch.where(
            no_lane_anchor, torch.zeros_like(coordinates[..., 5, 1]), coordinates[..., 5, 1]
        )
        valid[..., 5] = True

        organ_x = self._field(organs, "organs", "side_x_norm")
        organ_z = self._field(organs, "organs", "side_z_norm")
        organ_alive = self._field(organs, "organs", "alive") > 0.5
        coordinates[..., 6, 0] = organ_x[..., 0]
        coordinates[..., 6, 1] = organ_z[..., 0]
        valid[..., 6] = organ_alive[..., 0]
        high_or_crystal = torch.where(
            organ_alive[..., 1].unsqueeze(-1),
            torch.stack((organ_x[..., 1], organ_z[..., 1]), dim=-1),
            torch.stack((organ_x[..., 2], organ_z[..., 2]), dim=-1),
        )
        coordinates[..., 7, :] = high_or_crystal
        valid[..., 7] = organ_alive[..., 1] | organ_alive[..., 2]

        coordinates[..., 8, 0] = self._field(
            monsters[..., 19, :], "monsters", "side_x_norm"
        )
        coordinates[..., 8, 1] = self._field(
            monsters[..., 19, :], "monsters", "side_z_norm"
        )
        valid[..., 8] = (
            self._field(monsters[..., 19, :], "monsters", "location_valid") > 0.5
        )

        coordinates[..., 9, 0] = -66300.0 / 70000.0
        coordinates[..., 9, 1] = (19580.0 - 6115.0) / 35000.0
        valid[..., 9] = True

        # Monster feature coordinates fall back to fixed spawn anchors whenever
        # the current actor is not visible, so these slots cover the complete
        # map rather than only camps in the current frame-state vision.
        monster_x = self._field(monsters, "monsters", "side_x_norm")
        monster_z = self._field(monsters, "monsters", "side_z_norm")
        monster_valid = self._field(monsters, "monsters", "location_valid") > 0.5
        coordinates[..., 10:30, 0] = monster_x
        coordinates[..., 10:30, 1] = monster_z
        valid[..., 10:30] = monster_valid

        semantic = torch.eye(
            self.POSITION_COUNT, dtype=coordinates.dtype, device=coordinates.device
        ).reshape(1, 1, self.POSITION_COUNT, self.POSITION_COUNT)
        semantic = semantic.expand(batch, time, self.POSITION_COUNT, self.POSITION_COUNT)
        raw_tokens = torch.cat(
            (coordinates, valid.to(coordinates.dtype).unsqueeze(-1), semantic), dim=-1
        )
        encoded = self.encoder(raw_tokens)
        return {
            "coordinates_norm": coordinates,
            "valid": valid,
            "tokens": encoded,
            "keys": self.key(encoded),
            "values": self.value(encoded),
        }


class CommandNetworkV1(nn.Module):
    """Single team-level command policy and value; no command GRU."""

    ROUTE_IDS = HeroPolicyRouterV1.ROUTE_IDS

    def __init__(self):
        super().__init__()
        self.official_projection = LinearNormRelu(128, 256)
        self.command_feature = FrozenMLP((1024, 256, 256))
        self.intent_head = PolicyHeadV1(FrozenV1Config.COMMAND_HEADS[0])
        self.intent_embedding = nn.Embedding(FrozenV1Config.COMMAND_HEADS[0], 16)
        nn.init.normal_(self.intent_embedding.weight, mean=0.0, std=0.02)
        self.command_target_key = LinearNorm(64, 64)
        self.command_target_value = LinearNorm(64, 64)
        self.command_target_query = LinearNorm(256 + 16, 64)
        self.position_bank = SemanticPositionBankV1()
        self.command_position_query = LinearNorm(256 + 16, 64)
        self.value_hidden = LinearNormRelu(256, 128)
        self.value_output = _orthogonal_init(nn.Linear(128, 1), gain=1.0)

    @staticmethod
    def _official_ids(main_context, validate_runtime=True):
        slots = main_context[..., 9:12] > 0.5
        if validate_runtime and not torch.all(slots.sum(dim=-1) == 1):
            raise ValueError("each v1 sample must contain exactly one official hero ID")
        indices = slots.to(torch.int64).argmax(dim=-1)
        route_ids = torch.tensor(
            CommandNetworkV1.ROUTE_IDS, device=indices.device, dtype=torch.long
        )
        return route_ids[indices]

    def _fixed_hero_context(self, actor_output, validate_runtime=True):
        policy = actor_output["policy_feature"]
        hero_ids = actor_output["hero_ids"]
        batch, actors, time, _ = policy.shape
        if actors != 2:
            raise ValueError(f"command network requires exactly two actor features, got {actors}")
        counts = torch.stack(
            [(hero_ids == route_id).sum(dim=1) for route_id in self.ROUTE_IDS],
            dim=-1,
        )
        if validate_runtime and (
            not torch.all((counts == 0) | (counts == 1))
            or not torch.all(counts.sum(dim=-1) == 2)
        ):
            raise ValueError("two controlled command actors must occupy distinct hero routes")
        route_tensor = torch.tensor(
            self.ROUTE_IDS, device=hero_ids.device, dtype=hero_ids.dtype
        )
        official_ids = (counts == 0).to(hero_ids.dtype).mul(route_tensor).sum(dim=-1)
        main_context = actor_output["blocks"]["main_context"].squeeze(-2)
        official_slots = main_context[..., 9:12] > 0.5
        official_slot_count = official_slots.sum(dim=-1)
        valid_official = official_slot_count == 1
        if validate_runtime and torch.any(valid_official):
            official_per_actor = self._official_ids(
                main_context[valid_official], validate_runtime=True
            )
            expected = official_ids.unsqueeze(1).expand(batch, actors, time)[valid_official]
            if not torch.equal(official_per_actor, expected):
                raise ValueError("official hero ID conflicts with controlled route complement")
        friend_tokens = actor_output["hero_tokens"][..., :3, :].mean(dim=1)
        official_projected = self.official_projection(friend_tokens)
        fixed = policy.new_zeros(batch, time, 3, 256)
        occupancy = torch.zeros(
            batch, time, 3, dtype=torch.int64, device=policy.device
        )
        for route_index, route_id in enumerate(self.ROUTE_IDS):
            controlled_mask = hero_ids == route_id
            controlled_count = controlled_mask.sum(dim=1)
            controlled_value = (
                policy * controlled_mask.unsqueeze(-1).to(policy.dtype)
            ).sum(dim=1)
            official_mask = official_ids == route_id
            source_count = controlled_count + official_mask.to(controlled_count.dtype)
            if validate_runtime and not torch.all(source_count == 1):
                raise ValueError(
                    f"fixed hero command slot {route_id} must have exactly one source"
                )
            fixed[..., route_index, :] = torch.where(
                controlled_count.unsqueeze(-1) == 1,
                controlled_value,
                official_projected[..., route_index, :],
            )
            occupancy[..., route_index] = source_count
        return fixed, occupancy, official_ids

    def _conditioned_queries(self, command_feature, projection):
        intent_ids = torch.arange(
            FrozenV1Config.COMMAND_HEADS[0], device=command_feature.device
        )
        embeddings = self.intent_embedding(intent_ids)
        expanded_feature = command_feature.unsqueeze(-2).expand(
            *command_feature.shape[:-1], FrozenV1Config.COMMAND_HEADS[0], 256
        )
        expanded_embedding = embeddings.reshape(
            *((1,) * (command_feature.ndim - 1)),
            FrozenV1Config.COMMAND_HEADS[0],
            16,
        ).expand(
            *command_feature.shape[:-1], FrozenV1Config.COMMAND_HEADS[0], 16
        )
        return projection(torch.cat((expanded_feature, expanded_embedding), dim=-1))

    def forward(self, actor_output, validate_runtime=True):
        fixed_context, occupancy, official_ids = self._fixed_hero_context(
            actor_output,
            validate_runtime=validate_runtime,
        )
        team_feature = actor_output["team_feature"].mean(dim=1)
        command_input = torch.cat(
            (team_feature, fixed_context.flatten(start_dim=-2)), dim=-1
        )
        if command_input.shape[-1] != 1024:
            raise RuntimeError(f"command input dimension drifted: {command_input.shape[-1]}")
        feature = self.command_feature(command_input)
        intent_logits = self.intent_head(feature)

        target_bank = actor_output["target_bank"]
        target_keys = self.command_target_key(target_bank).mean(dim=1)
        target_values = self.command_target_value(target_bank).mean(dim=1)
        target_query = self._conditioned_queries(feature, self.command_target_query)
        target_logits = torch.einsum("...id,...td->...it", target_query, target_keys) / math.sqrt(64.0)

        position_bank = self.position_bank(actor_output["blocks"])
        position_query = self._conditioned_queries(feature, self.command_position_query)
        position_logits = torch.einsum(
            "...id,...pd->...ip", position_query, position_bank["keys"]
        ) / math.sqrt(64.0)
        value = self.value_output(self.value_hidden(feature))
        return {
            "command_feature": feature,
            "intent_logits": intent_logits,
            "target_logits": target_logits,
            "position_logits": position_logits,
            "command_value": value,
            "target_keys": target_keys,
            "target_values": target_values,
            "position_bank": position_bank,
            "fixed_hero_context": fixed_context,
            "fixed_hero_occupancy": occupancy,
            "official_ids": official_ids,
            "team_feature": team_feature,
        }


class NetworkV1(nn.Module):
    """Complete active actor, hybrid critic, and team-command network."""

    def __init__(self, assert_public_raw_equal=True):
        super().__init__()
        self.actor = ActorNetworkV1(
            assert_public_raw_equal=assert_public_raw_equal
        )
        self.command = CommandNetworkV1()

    def forward(
        self,
        feature,
        initial_hidden,
        reset_mask=None,
        hero_ids=None,
        has_internal_reset=None,
        validate_runtime=True,
    ):
        actor_output = self.actor(
            feature,
            initial_hidden,
            reset_mask=reset_mask,
            hero_ids=hero_ids,
            has_internal_reset=has_internal_reset,
            validate_runtime=validate_runtime,
        )
        command_output = self.command(
            actor_output,
            validate_runtime=validate_runtime,
        )
        return {"actor": actor_output, "command": command_output}


__all__ = (
    "ActorBackboneV1",
    "ActorNetworkV1",
    "ActorValueHeadV1",
    "FeatureEncoderV1",
    "FrozenMLP",
    "LightCNNV1",
    "LinearNorm",
    "LinearNormRelu",
    "LinearReluNorm",
    "ResidualPSCNV1",
    "HeroPolicyRouterV1",
    "HeroRouteHeadsV1",
    "PolicyHeadV1",
    "CommandNetworkV1",
    "SemanticPositionBankV1",
    "NetworkV1",
    "masked_max_mean",
)
