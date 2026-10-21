#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""PPO learner with hybrid actor value, auxiliary values, and team commands."""

import os
import time
from contextlib import nullcontext

import numpy as np
import torch

from agent_ppo.conf.conf import FrozenV1Config, V1TrainingConfig
from agent_ppo.feature import schema_v1
from agent_ppo.feature.definition_v1 import unpack_sample_v1
from agent_ppo.model.action_v1 import evaluate_actor_actions_v1
from agent_ppo.model.command_v1 import evaluate_command_actions_v1
from agent_ppo.model.network_v1 import HeroPolicyRouterV1


def _masked_mean(value, mask):
    mask = mask.to(value.dtype)
    return (value * mask).sum() / mask.sum().clamp_min(1.0)


def _masked_std(value, mask):
    mean = _masked_mean(value, mask)
    return torch.sqrt(_masked_mean((value - mean).square(), mask).clamp_min(0.0))


def _safe_ratio(new_log_probability, old_log_probability):
    return torch.exp((new_log_probability - old_log_probability).clamp(-20.0, 20.0))


def _normalize_advantage(value, mean, std):
    if mean is None or std is None:
        return value
    return (value - float(mean)) / max(float(std), V1TrainingConfig.ADVANTAGE_EPSILON)


def _explained_variance(prediction, target, mask):
    mask = mask > 0.5
    if not torch.any(mask):
        return prediction.new_zeros(())
    prediction = prediction[mask]
    target = target[mask]
    target_variance = torch.var(target, unbiased=False)
    if target_variance <= 1e-8:
        return prediction.new_zeros(())
    return 1.0 - torch.var(target - prediction, unbiased=False) / target_variance


class PPOLossV1:
    def __init__(self, config=V1TrainingConfig):
        self.config = config

    @staticmethod
    def _stack_actor_fields(unpacked):
        names = tuple(unpacked["actors"][0].keys())
        return {
            name: torch.stack(
                [actor[name] for actor in unpacked["actors"]], dim=1
            )
            for name in names
        }

    @staticmethod
    def _hero_ids_from_first_step(feature):
        block = schema_v1.FEATURE_BLOCK_BY_NAME["main_context"]
        main_context = feature[:, :, 0, block.start : block.stop].reshape(
            feature.shape[0], 2, 32
        )
        hero_ids = HeroPolicyRouterV1.decode_hero_ids(main_context)
        return hero_ids.unsqueeze(-1).expand(
            feature.shape[0], 2, FrozenV1Config.RNN_TIME_STEPS
        )

    @staticmethod
    def _padding_safe_actor_legal(legal, valid):
        result = legal.clone()
        invalid = ~(valid > 0.5)
        if torch.any(invalid):
            result[invalid] = 0
            offset = 0
            for width in FrozenV1Config.HERO_ACTION_HEADS:
                flat = result[..., offset : offset + width]
                flat[..., 0] = torch.where(
                    invalid, torch.ones_like(flat[..., 0]), flat[..., 0]
                )
                offset += width
        return result

    @staticmethod
    def _padding_safe_command_legal(legal, value_valid):
        result = legal.clone()
        invalid = ~(value_valid > 0.5)
        if torch.any(invalid):
            result[invalid] = 0
            result[..., 0] = torch.where(
                invalid, torch.ones_like(result[..., 0]), result[..., 0]
            )
        return result

    def actor_loss(self, actor_output, actor, normalization=None):
        valid = actor["is_train"].squeeze(-1)
        legal = self._padding_safe_actor_legal(
            actor["conditioned_legal_action"], valid
        )
        actions = actor["actions"].long()
        evaluated = evaluate_actor_actions_v1(
            actor_output, legal, actions
        )
        old_parts = actor["old_action_probabilities"].split(
            FrozenV1Config.HERO_ACTION_HEADS, dim=-1
        )
        old_selected = torch.stack(
            [
                probabilities.gather(
                    -1, actions[..., index].unsqueeze(-1)
                ).squeeze(-1)
                for index, probabilities in enumerate(old_parts)
            ],
            dim=-1,
        )
        old_log = old_selected.clamp_min(
            self.config.PROBABILITY_EPSILON
        ).log()
        new_log = evaluated["selected_log_probabilities"]
        ratio = _safe_ratio(new_log, old_log)
        advantage = actor["advantage"].squeeze(-1)
        normalization = normalization or {}
        advantage = _normalize_advantage(
            advantage,
            normalization.get("actor_mean"),
            normalization.get("actor_std"),
        ).unsqueeze(-1)
        head_mask = actor["sub_action_mask"] * valid.unsqueeze(-1)
        unclipped = ratio * advantage
        clipped = ratio.clamp(
            1.0 - self.config.CLIP_PARAM, 1.0 + self.config.CLIP_PARAM
        ) * advantage
        policy_loss = -_masked_mean(torch.minimum(unclipped, clipped), head_mask)
        entropy = _masked_mean(evaluated["entropy"], head_mask)
        value = actor_output["actor_value"].squeeze(-1)
        returns = actor["return"].squeeze(-1)
        value_loss = 0.5 * _masked_mean((value - returns).square(), valid)
        auxiliary_values = actor_output["actor_auxiliary_values"]
        auxiliary_returns = actor["auxiliary_returns"]
        expected_auxiliary_shape = (*value.shape, FrozenV1Config.AUXILIARY_VALUE_COUNT)
        if tuple(auxiliary_values.shape) != expected_auxiliary_shape:
            raise ValueError(
                "actor auxiliary value shape mismatch: expected "
                f"{expected_auxiliary_shape}, got {tuple(auxiliary_values.shape)}"
            )
        if tuple(auxiliary_returns.shape) != expected_auxiliary_shape:
            raise ValueError(
                "actor auxiliary return shape mismatch: expected "
                f"{expected_auxiliary_shape}, got {tuple(auxiliary_returns.shape)}"
            )
        auxiliary_losses = []
        auxiliary_metrics = {}
        for category_index, category_name in enumerate(
            FrozenV1Config.AUXILIARY_VALUE_CATEGORIES
        ):
            prediction = auxiliary_values[..., category_index]
            target = auxiliary_returns[..., category_index]
            category_loss = 0.5 * _masked_mean(
                (prediction - target).square(), valid
            )
            auxiliary_losses.append(category_loss)
            auxiliary_metrics[f"aux_{category_name}_loss"] = category_loss
            auxiliary_metrics[f"aux_{category_name}_explained_variance"] = (
                _explained_variance(prediction, target, valid)
            )
            auxiliary_metrics[f"aux_{category_name}_target_mean"] = _masked_mean(
                target, valid
            )
            auxiliary_metrics[f"aux_{category_name}_target_std"] = _masked_std(
                target, valid
            )
            auxiliary_metrics[f"aux_{category_name}_prediction_mean"] = (
                _masked_mean(prediction, valid)
            )
        auxiliary_value_loss = torch.stack(auxiliary_losses).mean()

        components = actor_output["actor_value_components"]
        if tuple(components.shape) != (*value.shape, 3):
            raise ValueError(
                "hybrid value components must end in 3, got "
                f"{tuple(components.shape)}"
            )
        component_reg_loss = 0.5 * _masked_mean(
            components.square().mean(dim=-1), valid
        )
        self_value, team_value, enemy_value = components.unbind(dim=-1)
        team_lambda = FrozenV1Config.HYBRID_VALUE_TEAM_LAMBDA
        recomposed_value = (
            (1.0 - team_lambda) * self_value
            + team_lambda * team_value
            - enemy_value
        )
        identity_error = _masked_mean((recomposed_value - value).abs(), valid)
        cancellation_numerator = (
            (1.0 - team_lambda) * self_value.abs()
            + team_lambda * team_value.abs()
            + enemy_value.abs()
        )
        cancellation_ratio = _masked_mean(
            cancellation_numerator, valid
        ) / _masked_mean(value.abs(), valid).clamp_min(1e-6)
        approx_kl = _masked_mean(old_log - new_log, head_mask)
        clip_fraction = _masked_mean(
            (torch.abs(ratio - 1.0) > self.config.CLIP_PARAM).to(ratio.dtype),
            head_mask,
        )
        result = {
            "policy_loss": policy_loss,
            "value_loss": value_loss,
            "auxiliary_value_loss": auxiliary_value_loss,
            "component_reg_loss": component_reg_loss,
            "hybrid_identity_error": identity_error,
            "hybrid_cancellation_ratio": cancellation_ratio,
            "entropy": entropy,
            "approx_kl": approx_kl,
            "clip_fraction": clip_fraction,
            "valid_count": valid.sum(),
            "policy_weight_count": head_mask.sum(),
            "explained_variance": _explained_variance(value, returns, valid),
        }
        result.update(auxiliary_metrics)
        for component_name, component_value in zip(
            FrozenV1Config.HYBRID_VALUE_COMPONENTS,
            (self_value, team_value, enemy_value),
        ):
            result[f"value_{component_name}_mean"] = _masked_mean(
                component_value, valid
            )
            result[f"value_{component_name}_std"] = _masked_std(
                component_value, valid
            )
        hero_ids = actor_output["hero_ids"].squeeze(-1)
        squared_error = (value - returns).square()
        for hero_id in HeroPolicyRouterV1.ROUTE_IDS:
            route_valid = valid * (hero_ids == hero_id).to(valid.dtype)
            route_head_mask = head_mask * (hero_ids == hero_id).unsqueeze(-1)
            result[f"hero_{hero_id}_valid_count"] = route_valid.sum()
            result[f"hero_{hero_id}_policy_weight_count"] = (
                route_head_mask.sum()
            )
            result[f"hero_{hero_id}_entropy"] = _masked_mean(
                evaluated["entropy"], route_head_mask
            )
            result[f"hero_{hero_id}_value_loss"] = 0.5 * _masked_mean(
                squared_error, route_valid
            )
        return result

    def command_loss(self, command_output, command, normalization=None):
        # Every real command step has the intent head active, including
        # cooldown/no-command steps. Padding has an all-zero active-head mask.
        # Actor ``is_train`` instead expresses per-hero actionability and may
        # legitimately differ when only one controlled hero is dead.
        value_valid = (
            command["active_head_mask"].sum(dim=-1) > 0.5
        ).to(command_output["command_value"].dtype)
        legal = self._padding_safe_command_legal(
            command["conditioned_legal_masks"], value_valid
        )
        actions = command["actions"].long()
        active_head_mask = command["active_head_mask"]
        evaluated = evaluate_command_actions_v1(
            command_output,
            legal,
            actions,
            active_head_mask,
        )
        new_joint_log = (
            evaluated["selected_log_probabilities"] * active_head_mask
        ).sum(dim=-1)
        old_joint_log = (
            command["old_log_probs"] * active_head_mask
        ).sum(dim=-1)
        ratio = _safe_ratio(new_joint_log, old_joint_log)
        advantage = command["advantage"].squeeze(-1)
        normalization = normalization or {}
        advantage = _normalize_advantage(
            advantage,
            normalization.get("command_mean"),
            normalization.get("command_std"),
        )
        policy_mask = command["is_train"].squeeze(-1) * value_valid
        unclipped = ratio * advantage
        clipped = ratio.clamp(
            1.0 - self.config.CLIP_PARAM, 1.0 + self.config.CLIP_PARAM
        ) * advantage
        policy_loss = -_masked_mean(
            torch.minimum(unclipped, clipped), policy_mask
        )
        entropy_per_step = (
            evaluated["entropy"] * active_head_mask
        ).sum(dim=-1)
        entropy = _masked_mean(entropy_per_step, policy_mask)
        value = command_output["command_value"].squeeze(-1)
        returns = command["return"].squeeze(-1)
        value_loss = 0.5 * _masked_mean(
            (value - returns).square(), value_valid
        )
        approx_kl = _masked_mean(old_joint_log - new_joint_log, policy_mask)
        clip_fraction = _masked_mean(
            (torch.abs(ratio - 1.0) > self.config.CLIP_PARAM).to(ratio.dtype),
            policy_mask,
        )
        intent_actions = actions[..., 0]
        unset_rate = _masked_mean(
            (intent_actions == 1).to(value.dtype), policy_mask
        )
        no_command_rate = _masked_mean(
            (intent_actions == 0).to(value.dtype), policy_mask
        )
        return {
            "policy_loss": policy_loss,
            "value_loss": value_loss,
            "entropy": entropy,
            "approx_kl": approx_kl,
            "clip_fraction": clip_fraction,
            "policy_valid_count": policy_mask.sum(),
            "value_valid_count": value_valid.sum(),
            "unset_rate": unset_rate,
            "no_command_rate": no_command_rate,
            "explained_variance": _explained_variance(
                value, returns, value_valid
            ),
        }

    @staticmethod
    def _loss_output_fp32(output):
        return {
            "actor": {
                "ordinary_logits": [
                    value.float() for value in output["actor"]["ordinary_logits"]
                ],
                "target_logits": output["actor"]["target_logits"].float(),
                "actor_value": output["actor"]["actor_value"].float(),
                "actor_value_components": output["actor"][
                    "actor_value_components"
                ].float(),
                "actor_auxiliary_values": output["actor"][
                    "actor_auxiliary_values"
                ].float(),
                "hero_ids": output["actor"]["hero_ids"],
            },
            "command": {
                "intent_logits": output["command"]["intent_logits"].float(),
                "target_logits": output["command"]["target_logits"].float(),
                "position_logits": output["command"]["position_logits"].float(),
                "command_value": output["command"]["command_value"].float(),
            },
        }

    def __call__(
        self,
        model,
        sample_batch,
        normalization=None,
        loss_weights=None,
        autocast_context=None,
    ):
        unpacked = unpack_sample_v1(sample_batch)
        actor = self._stack_actor_fields(unpacked)
        feature = actor["feature"]
        initial_hidden = actor["initial_hidden"]
        reset_mask = actor["rnn_reset"].squeeze(-1)
        hero_ids = self._hero_ids_from_first_step(feature)
        if autocast_context is None:
            output = model(
                feature,
                initial_hidden,
                reset_mask=reset_mask,
                hero_ids=hero_ids,
            )
        else:
            with autocast_context:
                output = model(
                    feature,
                    initial_hidden,
                    reset_mask=reset_mask,
                    hero_ids=hero_ids,
                )
        loss_output = self._loss_output_fp32(output)
        actor_metrics = self.actor_loss(
            loss_output["actor"], actor, normalization=normalization
        )
        command_metrics = self.command_loss(
            loss_output["command"],
            unpacked["command"],
            normalization=normalization,
        )
        weights = {
            "actor_policy": 1.0,
            "actor_value": 1.0,
            "actor_aux_value": 1.0,
            "command_policy": 1.0,
            "command_value": 1.0,
            **(loss_weights or {}),
        }
        total_loss = (
            weights["actor_policy"] * actor_metrics["policy_loss"]
            + weights["actor_value"] * self.config.VALUE_COEF * actor_metrics["value_loss"]
            + weights["actor_aux_value"]
            * self.config.AUXILIARY_VALUE_COEF
            * actor_metrics["auxiliary_value_loss"]
            + weights["actor_value"]
            * self.config.HYBRID_COMPONENT_REG_COEF
            * actor_metrics["component_reg_loss"]
            - weights["actor_policy"] * self.config.ENTROPY_COEF * actor_metrics["entropy"]
            + weights["command_policy"] * self.config.COMMAND_COEF * command_metrics["policy_loss"]
            + weights["command_value"] * self.config.COMMAND_VALUE_COEF * command_metrics["value_loss"]
            - weights["command_policy"] * self.config.COMMAND_ENTROPY_COEF * command_metrics["entropy"]
        )
        if not torch.isfinite(total_loss):
            raise FloatingPointError("v1 PPO total loss is NaN/Inf")
        metrics = {
            "total_loss": total_loss.detach(),
            "actor_policy_loss": actor_metrics["policy_loss"].detach(),
            "actor_value_loss": actor_metrics["value_loss"].detach(),
            "actor_auxiliary_value_loss": actor_metrics[
                "auxiliary_value_loss"
            ].detach(),
            "actor_hybrid_component_reg_loss": actor_metrics[
                "component_reg_loss"
            ].detach(),
            "actor_hybrid_identity_error": actor_metrics[
                "hybrid_identity_error"
            ].detach(),
            "actor_hybrid_cancellation_ratio": actor_metrics[
                "hybrid_cancellation_ratio"
            ].detach(),
            "actor_entropy": actor_metrics["entropy"].detach(),
            "actor_approx_kl": actor_metrics["approx_kl"].detach(),
            "actor_clip_fraction": actor_metrics["clip_fraction"].detach(),
            "actor_valid_count": actor_metrics["valid_count"].detach(),
            "actor_policy_weight_count": actor_metrics[
                "policy_weight_count"
            ].detach(),
            "actor_explained_variance": actor_metrics[
                "explained_variance"
            ].detach(),
            "command_policy_loss": command_metrics["policy_loss"].detach(),
            "command_value_loss": command_metrics["value_loss"].detach(),
            "command_entropy": command_metrics["entropy"].detach(),
            "command_approx_kl": command_metrics["approx_kl"].detach(),
            "command_clip_fraction": command_metrics["clip_fraction"].detach(),
            "command_policy_valid_count": command_metrics[
                "policy_valid_count"
            ].detach(),
            "command_value_valid_count": command_metrics[
                "value_valid_count"
            ].detach(),
            "command_unset_rate": command_metrics["unset_rate"].detach(),
            "command_no_command_rate": command_metrics[
                "no_command_rate"
            ].detach(),
            "command_explained_variance": command_metrics[
                "explained_variance"
            ].detach(),
        }
        for component_name in FrozenV1Config.HYBRID_VALUE_COMPONENTS:
            for statistic in ("mean", "std"):
                source = f"value_{component_name}_{statistic}"
                metrics[f"actor_{source}"] = actor_metrics[source].detach()
        for category_name in FrozenV1Config.AUXILIARY_VALUE_CATEGORIES:
            for statistic in (
                "loss",
                "explained_variance",
                "target_mean",
                "target_std",
                "prediction_mean",
            ):
                source = f"aux_{category_name}_{statistic}"
                metrics[f"actor_{source}"] = actor_metrics[source].detach()
        for hero_id in HeroPolicyRouterV1.ROUTE_IDS:
            for suffix in (
                "valid_count",
                "policy_weight_count",
                "entropy",
                "value_loss",
            ):
                key = f"hero_{hero_id}_{suffix}"
                metrics[key] = actor_metrics[key].detach()
        return total_loss, metrics, output


class AlgorithmV1:
    def __init__(self, model, optimizer, device=None, logger=None, monitor=None):
        self.model = model
        self.optimizer = optimizer
        self.device = torch.device(device or next(model.parameters()).device)
        self.logger = logger
        self.monitor = monitor
        self.loss = PPOLossV1()
        self.parameters = [
            parameter
            for group in optimizer.param_groups
            for parameter in group["params"]
        ]
        self.train_step = 0
        self.last_report_monitor_time = 0.0
        self.micro_batch_size = int(V1TrainingConfig.MICRO_BATCH_SIZE)
        self.amp_enabled = bool(
            V1TrainingConfig.AMP_ENABLED and self.device.type == "cuda"
        )
        self.amp_dtype = (
            torch.bfloat16
            if self.amp_enabled and torch.cuda.is_bf16_supported()
            else torch.float16
        )
        scaler_enabled = self.amp_enabled and self.amp_dtype == torch.float16
        if hasattr(torch.amp, "GradScaler"):
            self.grad_scaler = torch.amp.GradScaler(
                "cuda",
                enabled=scaler_enabled,
            )
        else:
            # The official train environment uses an older PyTorch release where
            # GradScaler is still exposed only through torch.cuda.amp.
            self.grad_scaler = torch.cuda.amp.GradScaler(enabled=scaler_enabled)

    @staticmethod
    def _sample_array(sample_data):
        sample = sample_data.sample
        if isinstance(sample, torch.Tensor):
            return sample.detach().cpu().numpy()
        return np.asarray(sample, dtype=np.float32)

    def _batch_statistics(self, list_sample_data):
        totals = {
            "actor_sum": 0.0,
            "actor_square_sum": 0.0,
            "actor_count": 0.0,
            "command_sum": 0.0,
            "command_square_sum": 0.0,
            "command_count": 0.0,
            "actor_policy": 0.0,
            "actor_value": 0.0,
            "actor_aux_value": 0.0,
            "command_policy": 0.0,
            "command_value": 0.0,
        }
        for sample_data in list_sample_data:
            unpacked = unpack_sample_v1(self._sample_array(sample_data))
            for actor in unpacked["actors"]:
                valid = actor["is_train"].reshape(-1).astype(np.float64)
                advantage = actor["advantage"].reshape(-1).astype(np.float64)
                active = valid > 0.5
                selected = advantage[active]
                totals["actor_sum"] += float(selected.sum())
                totals["actor_square_sum"] += float(np.square(selected).sum())
                totals["actor_count"] += float(active.sum())
                totals["actor_value"] += float(valid.sum())
                totals["actor_aux_value"] += float(valid.sum())
                totals["actor_policy"] += float(
                    (
                        actor["sub_action_mask"].astype(np.float64)
                        * valid.reshape(-1, 1)
                    ).sum()
                )
            command_valid = (
                unpacked["command"]["active_head_mask"]
                .astype(np.float64)
                .sum(axis=-1)
                .reshape(-1)
                > 0.5
            ).astype(np.float64)
            command_is_train = unpacked["command"]["is_train"].reshape(-1).astype(
                np.float64
            )
            policy_mask = command_valid * command_is_train
            command_advantage = unpacked["command"]["advantage"].reshape(-1).astype(
                np.float64
            )
            selected = command_advantage[policy_mask > 0.5]
            totals["command_sum"] += float(selected.sum())
            totals["command_square_sum"] += float(np.square(selected).sum())
            totals["command_count"] += float((policy_mask > 0.5).sum())
            totals["command_policy"] += float(policy_mask.sum())
            totals["command_value"] += float(command_valid.sum())

        def mean_std(prefix):
            count = totals[f"{prefix}_count"]
            if not V1TrainingConfig.ADVANTAGE_NORMALIZATION or count <= 0:
                return None, None
            mean = totals[f"{prefix}_sum"] / count
            variance = max(
                totals[f"{prefix}_square_sum"] / count - mean * mean,
                0.0,
            )
            return mean, max(variance**0.5, V1TrainingConfig.ADVANTAGE_EPSILON)

        actor_mean, actor_std = mean_std("actor")
        command_mean, command_std = mean_std("command")
        return {
            "normalization": {
                "actor_mean": actor_mean,
                "actor_std": actor_std,
                "command_mean": command_mean,
                "command_std": command_std,
            },
            "counts": {
                key: totals[key]
                for key in (
                    "actor_policy",
                    "actor_value",
                    "actor_aux_value",
                    "command_policy",
                    "command_value",
                )
            },
        }

    def format_data(self, list_sample_data):
        values = []
        for sample_data in list_sample_data:
            sample = sample_data.sample
            if isinstance(sample, torch.Tensor):
                values.append(sample.to(device=self.device, dtype=torch.float32))
            else:
                values.append(
                    torch.as_tensor(
                        np.asarray(sample), device=self.device, dtype=torch.float32
                    )
                )
        result = torch.stack(values)
        if result.shape[-1] != FrozenV1Config.SAMPLE_DIM:
            raise ValueError(f"learner received non-v1 sample: {tuple(result.shape)}")
        return result

    def learn(self, list_sample_data):
        if not list_sample_data:
            raise ValueError("learner received an empty v1 batch")
        global_statistics = self._batch_statistics(list_sample_data)
        global_counts = global_statistics["counts"]
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        aggregate = {}
        total_loss_value = 0.0
        for start in range(0, len(list_sample_data), self.micro_batch_size):
            micro_samples = list_sample_data[start : start + self.micro_batch_size]
            local_statistics = self._batch_statistics(micro_samples)
            local_counts = local_statistics["counts"]
            loss_weights = {
                key: (
                    local_counts[key] / global_counts[key]
                    if global_counts[key] > 0
                    else 0.0
                )
                for key in global_counts
            }
            sample_batch = self.format_data(micro_samples)
            # Older PyTorch releases validate the requested CPU autocast dtype
            # even when enabled=False.  Avoid constructing a float16 CPU
            # autocast context at all; CPU train_test should run in FP32.
            autocast_context = (
                torch.autocast(
                    device_type="cuda",
                    dtype=self.amp_dtype,
                    enabled=True,
                )
                if self.amp_enabled
                else nullcontext()
            )
            total_loss, metrics, _ = self.loss(
                self.model,
                sample_batch,
                normalization=global_statistics["normalization"],
                loss_weights=loss_weights,
                autocast_context=autocast_context,
            )
            self.grad_scaler.scale(total_loss).backward()
            total_loss_value += float(total_loss.detach().cpu())

            actor_policy_weight = loss_weights["actor_policy"]
            actor_value_weight = loss_weights["actor_value"]
            actor_aux_value_weight = loss_weights["actor_aux_value"]
            command_policy_weight = loss_weights["command_policy"]
            command_value_weight = loss_weights["command_value"]
            metric_weights = {
                "actor_policy_loss": actor_policy_weight,
                "actor_entropy": actor_policy_weight,
                "actor_approx_kl": actor_policy_weight,
                "actor_clip_fraction": actor_policy_weight,
                "actor_value_loss": actor_value_weight,
                "actor_explained_variance": actor_value_weight,
                "actor_auxiliary_value_loss": actor_aux_value_weight,
                "actor_hybrid_component_reg_loss": actor_value_weight,
                "actor_hybrid_identity_error": actor_value_weight,
                "actor_hybrid_cancellation_ratio": actor_value_weight,
                "command_policy_loss": command_policy_weight,
                "command_entropy": command_policy_weight,
                "command_approx_kl": command_policy_weight,
                "command_clip_fraction": command_policy_weight,
                "command_unset_rate": command_policy_weight,
                "command_no_command_rate": command_policy_weight,
                "command_value_loss": command_value_weight,
                "command_explained_variance": command_value_weight,
            }
            for component_name in FrozenV1Config.HYBRID_VALUE_COMPONENTS:
                for statistic in ("mean", "std"):
                    metric_weights[
                        f"actor_value_{component_name}_{statistic}"
                    ] = actor_value_weight
            for category_name in FrozenV1Config.AUXILIARY_VALUE_CATEGORIES:
                for statistic in (
                    "loss",
                    "explained_variance",
                    "target_mean",
                    "target_std",
                    "prediction_mean",
                ):
                    metric_weights[
                        f"actor_aux_{category_name}_{statistic}"
                    ] = actor_aux_value_weight
            for hero_id in HeroPolicyRouterV1.ROUTE_IDS:
                hero_valid = float(metrics[f"hero_{hero_id}_valid_count"].cpu())
                hero_policy = float(
                    metrics[f"hero_{hero_id}_policy_weight_count"].cpu()
                )
                aggregate[f"hero_{hero_id}_valid_count"] = (
                    aggregate.get(f"hero_{hero_id}_valid_count", 0.0) + hero_valid
                )
                aggregate[f"hero_{hero_id}_policy_weight_count"] = (
                    aggregate.get(f"hero_{hero_id}_policy_weight_count", 0.0)
                    + hero_policy
                )
                # Final per-hero means are completed after all micro-batches.
                aggregate[f"hero_{hero_id}_entropy_sum"] = (
                    aggregate.get(f"hero_{hero_id}_entropy_sum", 0.0)
                    + float(metrics[f"hero_{hero_id}_entropy"].cpu()) * hero_policy
                )
                aggregate[f"hero_{hero_id}_value_loss_sum"] = (
                    aggregate.get(f"hero_{hero_id}_value_loss_sum", 0.0)
                    + float(metrics[f"hero_{hero_id}_value_loss"].cpu()) * hero_valid
                )
            for name, weight in metric_weights.items():
                aggregate[name] = aggregate.get(name, 0.0) + float(
                    metrics[name].cpu()
                ) * weight
            aggregate["actor_valid_count"] = aggregate.get(
                "actor_valid_count", 0.0
            ) + float(metrics["actor_valid_count"].cpu())
            aggregate["actor_policy_weight_count"] = aggregate.get(
                "actor_policy_weight_count", 0.0
            ) + float(metrics["actor_policy_weight_count"].cpu())
            aggregate["command_policy_valid_count"] = aggregate.get(
                "command_policy_valid_count", 0.0
            ) + float(metrics["command_policy_valid_count"].cpu())
            aggregate["command_value_valid_count"] = aggregate.get(
                "command_value_valid_count", 0.0
            ) + float(metrics["command_value_valid_count"].cpu())
            del sample_batch, total_loss

        self.grad_scaler.unscale_(self.optimizer)
        if V1TrainingConfig.USE_GRAD_CLIP:
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                self.parameters, V1TrainingConfig.GRAD_CLIP_RANGE
            )
        else:
            gradient_norm = torch.zeros((), device=self.device)
        self.grad_scaler.step(self.optimizer)
        self.grad_scaler.update()
        self.train_step += 1
        result = dict(aggregate)
        result["total_loss"] = total_loss_value
        for hero_id in HeroPolicyRouterV1.ROUTE_IDS:
            valid = result[f"hero_{hero_id}_valid_count"]
            policy = result[f"hero_{hero_id}_policy_weight_count"]
            result[f"hero_{hero_id}_entropy"] = result.pop(
                f"hero_{hero_id}_entropy_sum"
            ) / max(policy, 1.0)
            result[f"hero_{hero_id}_value_loss"] = result.pop(
                f"hero_{hero_id}_value_loss_sum"
            ) / max(valid, 1.0)
        result["gradient_norm"] = float(gradient_norm.detach().cpu())
        result["train_step"] = self.train_step
        result["micro_batch_size"] = self.micro_batch_size
        result["amp_enabled"] = float(self.amp_enabled)
        if self.monitor and time.time() - self.last_report_monitor_time >= 60:
            self.monitor.put_data({os.getpid(): result})
            self.last_report_monitor_time = time.time()
        return result


__all__ = ("AlgorithmV1", "PPOLossV1")
