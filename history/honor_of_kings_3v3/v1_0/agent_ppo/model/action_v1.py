#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""Frozen ordinary-action legality and button-conditioned target utilities."""

import torch

from agent_ppo.conf.conf import FrozenV1Config


FIXED_HEAD_DIM = sum(FrozenV1Config.HERO_ACTION_HEADS[:-1])


def split_raw_legal_action_v1(raw_legal_action):
    if raw_legal_action.shape[-1] != FrozenV1Config.RAW_LEGAL_ACTION_DIM:
        raise ValueError(
            f"raw legal action must end in 629, got {raw_legal_action.shape[-1]}"
        )
    offset = 0
    fixed = []
    for width in FrozenV1Config.HERO_ACTION_HEADS[:-1]:
        fixed.append(raw_legal_action[..., offset : offset + width])
        offset += width
    target = raw_legal_action[..., offset:].reshape(
        *raw_legal_action.shape[:-1],
        FrozenV1Config.HERO_ACTION_HEADS[0],
        FrozenV1Config.HERO_ACTION_HEADS[-1],
    )
    return (*fixed, target)


def select_button_target_row_v1(value, button_action):
    if value.shape[-2:] != (
        FrozenV1Config.HERO_ACTION_HEADS[0],
        FrozenV1Config.HERO_ACTION_HEADS[-1],
    ):
        raise ValueError(f"button-target tensor must end in [13,39], got {tuple(value.shape[-2:])}")
    if tuple(value.shape[:-2]) != tuple(button_action.shape):
        raise ValueError(
            f"button selection shape mismatch: value={tuple(value.shape)}, button={tuple(button_action.shape)}"
        )
    if torch.any(button_action < 0) or torch.any(
        button_action >= FrozenV1Config.HERO_ACTION_HEADS[0]
    ):
        raise ValueError("button action out of [0,13) range")
    index = button_action.long().unsqueeze(-1).unsqueeze(-1).expand(
        *button_action.shape, 1, FrozenV1Config.HERO_ACTION_HEADS[-1]
    )
    return value.gather(dim=-2, index=index).squeeze(-2)


def condition_raw_legal_action_v1(raw_legal_action, button_action):
    button, move, offset_x, offset_z, target_matrix = split_raw_legal_action_v1(
        raw_legal_action
    )
    target = select_button_target_row_v1(target_matrix, button_action)
    result = torch.cat((button, move, offset_x, offset_z, target), dim=-1)
    if result.shape[-1] != FrozenV1Config.CONDITIONED_LEGAL_ACTION_DIM:
        raise RuntimeError(f"conditioned legal action dimension drifted: {result.shape[-1]}")
    return result


def split_conditioned_legal_action_v1(conditioned_legal_action):
    if conditioned_legal_action.shape[-1] != FrozenV1Config.CONDITIONED_LEGAL_ACTION_DIM:
        raise ValueError(
            f"conditioned legal action must end in 161, got {conditioned_legal_action.shape[-1]}"
        )
    return conditioned_legal_action.split(FrozenV1Config.HERO_ACTION_HEADS, dim=-1)


def masked_probabilities_v1(logits, legal_mask, label):
    if logits.shape != legal_mask.shape:
        raise ValueError(
            f"{label} logits/legal shape mismatch: {tuple(logits.shape)} vs {tuple(legal_mask.shape)}"
        )
    legal = legal_mask > 0.5
    legal_count = legal.sum(dim=-1)
    if torch.any(legal_count == 0):
        indices = torch.nonzero(legal_count == 0, as_tuple=False).detach().cpu().tolist()
        raise ValueError(f"{label} has all-illegal rows at {indices[:8]}")
    masked = logits.masked_fill(~legal, torch.finfo(logits.dtype).min)
    probabilities = torch.softmax(masked, dim=-1)
    probabilities = probabilities * legal.to(probabilities.dtype)
    probabilities = probabilities / probabilities.sum(dim=-1, keepdim=True)
    if not torch.isfinite(probabilities).all():
        raise FloatingPointError(f"{label} masked probabilities contain NaN/Inf")
    return probabilities


def _sample_from_probabilities(probabilities, deterministic):
    if deterministic:
        return probabilities.argmax(dim=-1)
    return torch.multinomial(
        probabilities.reshape(-1, probabilities.shape[-1]), 1
    ).reshape(probabilities.shape[:-1])


def sample_actor_actions_v1(network_output, raw_legal_action, deterministic=False):
    ordinary_logits = network_output["ordinary_logits"]
    target_logits_all_buttons = network_output["target_logits"]
    if len(ordinary_logits) != 4:
        raise ValueError("ActorNetworkV1 must supply four non-target ordinary logits")
    button_legal, move_legal, offset_x_legal, offset_z_legal, target_legal_matrix = (
        split_raw_legal_action_v1(raw_legal_action)
    )
    legal_parts = (button_legal, move_legal, offset_x_legal, offset_z_legal)
    labels = ("button", "move", "offset_x", "offset_z")
    probabilities = [
        masked_probabilities_v1(logits, legal, label)
        for logits, legal, label in zip(ordinary_logits, legal_parts, labels)
    ]
    actions = [
        _sample_from_probabilities(probability, deterministic)
        for probability in probabilities
    ]
    button_action = actions[0]
    target_logits = select_button_target_row_v1(
        target_logits_all_buttons, button_action
    )
    target_legal = select_button_target_row_v1(target_legal_matrix, button_action)
    target_probability = masked_probabilities_v1(
        target_logits, target_legal, "button-conditioned target"
    )
    target_action = _sample_from_probabilities(target_probability, deterministic)
    actions.append(target_action)
    probabilities.append(target_probability)
    action_tensor = torch.stack(actions, dim=-1)
    conditioned_legal = torch.cat((*legal_parts, target_legal), dim=-1)
    probability_vector = torch.cat(probabilities, dim=-1)
    selected_probabilities = torch.stack(
        [
            probability.gather(-1, action.unsqueeze(-1)).squeeze(-1)
            for probability, action in zip(probabilities, actions)
        ],
        dim=-1,
    )
    return {
        "actions": action_tensor,
        "probabilities": probabilities,
        "probability_vector": probability_vector,
        "selected_probabilities": selected_probabilities,
        "selected_log_probabilities": selected_probabilities.clamp_min(1e-12).log(),
        "conditioned_legal_action": conditioned_legal,
        "selected_target_logits": target_logits,
    }


def evaluate_actor_actions_v1(
    network_output, conditioned_legal_action, action_tensor
):
    if action_tensor.shape[-1] != FrozenV1Config.ACTOR_ACTION_COUNT:
        raise ValueError(f"actor actions must end in 5, got {action_tensor.shape[-1]}")
    legal_parts = split_conditioned_legal_action_v1(conditioned_legal_action)
    button_action = action_tensor[..., 0].long()
    logits = list(network_output["ordinary_logits"])
    logits.append(
        select_button_target_row_v1(network_output["target_logits"], button_action)
    )
    labels = ("button", "move", "offset_x", "offset_z", "target")
    probabilities = [
        masked_probabilities_v1(head_logits, legal, label)
        for head_logits, legal, label in zip(logits, legal_parts, labels)
    ]
    selected = [
        probability.gather(-1, action_tensor[..., index].long().unsqueeze(-1)).squeeze(-1)
        for index, probability in enumerate(probabilities)
    ]
    selected_probability = torch.stack(selected, dim=-1)
    entropy = torch.stack(
        [
            -(probability * probability.clamp_min(1e-12).log()).sum(dim=-1)
            for probability in probabilities
        ],
        dim=-1,
    )
    return {
        "probabilities": probabilities,
        "probability_vector": torch.cat(probabilities, dim=-1),
        "selected_probabilities": selected_probability,
        "selected_log_probabilities": selected_probability.clamp_min(1e-12).log(),
        "entropy": entropy,
    }


__all__ = (
    "condition_raw_legal_action_v1",
    "evaluate_actor_actions_v1",
    "masked_probabilities_v1",
    "sample_actor_actions_v1",
    "select_button_target_row_v1",
    "split_conditioned_legal_action_v1",
    "split_raw_legal_action_v1",
)
