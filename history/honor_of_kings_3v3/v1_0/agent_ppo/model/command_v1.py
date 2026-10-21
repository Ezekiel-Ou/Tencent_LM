#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""Sampling and evaluation for the frozen team command action heads."""

import torch

from agent_ppo.conf.conf import FrozenV1Config
from agent_ppo.feature import schema_v1
from agent_ppo.model.action_v1 import masked_probabilities_v1


TARGET_INTENTS = frozenset(schema_v1.TARGET_ACTIVE_COMMAND_INTENTS)
POSITION_INTENTS = frozenset(schema_v1.POSITION_ACTIVE_COMMAND_INTENTS)
COMMAND_SAMPLER_VERSION = "guarded_v2"


def select_intent_row_v1(value, intent_action, width):
    if value.shape[-2:] != (FrozenV1Config.COMMAND_HEADS[0], width):
        raise ValueError(
            f"intent-conditioned tensor must end in [9,{width}], got {tuple(value.shape[-2:])}"
        )
    if tuple(value.shape[:-2]) != tuple(intent_action.shape):
        raise ValueError(
            f"intent selection shape mismatch: value={tuple(value.shape)}, intent={tuple(intent_action.shape)}"
        )
    if torch.any(intent_action < 0) or torch.any(
        intent_action >= FrozenV1Config.COMMAND_HEADS[0]
    ):
        raise ValueError("command intent out of [0,9) range")
    index = intent_action.long().unsqueeze(-1).unsqueeze(-1).expand(
        *intent_action.shape, 1, width
    )
    return value.gather(-2, index).squeeze(-2)


def _sample(probabilities, deterministic):
    if deterministic:
        return probabilities.argmax(dim=-1)
    return torch.multinomial(
        probabilities.reshape(-1, probabilities.shape[-1]), 1
    ).reshape(probabilities.shape[:-1])


def _coherent_intent_legal_v1(
    intent_legal, target_legal_by_intent, position_legal_by_intent
):
    """Remove parent intents whose conditioned child head has no legal action."""

    if intent_legal.shape[-1] != FrozenV1Config.COMMAND_HEADS[0]:
        raise ValueError("command intent legal mask must end in 9")
    expected_target_shape = (*intent_legal.shape, FrozenV1Config.COMMAND_HEADS[1])
    expected_position_shape = (*intent_legal.shape, FrozenV1Config.COMMAND_HEADS[2])
    if target_legal_by_intent.shape != expected_target_shape:
        raise ValueError(
            "command target legal mask shape mismatch: "
            f"{tuple(target_legal_by_intent.shape)} vs {expected_target_shape}"
        )
    if position_legal_by_intent.shape != expected_position_shape:
        raise ValueError(
            "command position legal mask shape mismatch: "
            f"{tuple(position_legal_by_intent.shape)} vs {expected_position_shape}"
        )

    coherent = torch.isfinite(intent_legal) & (intent_legal > 0.5)
    target_available = torch.isfinite(target_legal_by_intent) & (
        target_legal_by_intent > 0.5
    )
    position_available = torch.isfinite(position_legal_by_intent) & (
        position_legal_by_intent > 0.5
    )
    coherent = coherent.clone()
    for intent in TARGET_INTENTS:
        coherent[..., intent] &= target_available[..., intent, :].any(dim=-1)
    for intent in POSITION_INTENTS:
        coherent[..., intent] &= position_available[..., intent, :].any(dim=-1)

    # Respect a policy mask which requires command renewal. Fall back to
    # no-command only when malformed/empty masks would otherwise leave no
    # sampleable action.
    no_legal_intent = ~coherent.any(dim=-1)
    coherent[..., 0] |= no_legal_intent
    return coherent.to(intent_legal.dtype)


def _optional_head(logits, legal, active, label, deterministic):
    if logits.shape != legal.shape or logits.shape[:-1] != active.shape:
        raise ValueError(f"{label} optional head shapes do not align")
    flat_logits = logits.reshape(-1, logits.shape[-1])
    flat_legal = legal.reshape(-1, legal.shape[-1])
    flat_active = active.reshape(-1)
    flat_probabilities = torch.zeros_like(flat_logits)
    flat_actions = torch.zeros(
        flat_logits.shape[0], dtype=torch.long, device=logits.device
    )
    flat_log_prob = torch.zeros(
        flat_logits.shape[0], dtype=logits.dtype, device=logits.device
    )
    flat_entropy = torch.zeros_like(flat_log_prob)
    flat_legal_available = (
        torch.isfinite(flat_legal) & (flat_legal > 0.5)
    ).any(dim=-1)
    invalid_active = flat_active & ~flat_legal_available
    effective_active = flat_active & flat_legal_available
    if torch.any(effective_active):
        active_probabilities = masked_probabilities_v1(
            flat_logits[effective_active], flat_legal[effective_active], label
        )
        active_actions = _sample(active_probabilities, deterministic)
        selected = active_probabilities.gather(
            -1, active_actions.unsqueeze(-1)
        ).squeeze(-1)
        entropy = -(
            active_probabilities
            * active_probabilities.clamp_min(1e-12).log()
        ).sum(dim=-1)
        flat_probabilities[effective_active] = active_probabilities
        flat_actions[effective_active] = active_actions
        flat_log_prob[effective_active] = selected.clamp_min(1e-12).log()
        flat_entropy[effective_active] = entropy
    return {
        "probabilities": flat_probabilities.reshape_as(logits),
        "action": flat_actions.reshape(active.shape),
        "log_probability": flat_log_prob.reshape(active.shape),
        "entropy": flat_entropy.reshape(active.shape),
        "invalid_active": invalid_active.reshape(active.shape),
    }


def run_command_guard_self_test_v1():
    """Fail fast if the deployed command sampler cannot contain an illegal child row."""

    intent_dim, target_dim, position_dim = FrozenV1Config.COMMAND_HEADS
    expected_contract = (11, 39, 30)
    if (intent_dim, target_dim, position_dim) != expected_contract:
        raise RuntimeError(
            "command guard self-test contract mismatch: "
            f"{FrozenV1Config.COMMAND_HEADS} vs {expected_contract}"
        )

    position_intent = min(POSITION_INTENTS)
    command_output = {
        "intent_logits": torch.zeros(1, 1, intent_dim),
        "target_logits": torch.zeros(1, 1, intent_dim, target_dim),
        "position_logits": torch.zeros(1, 1, intent_dim, position_dim),
    }
    command_output["intent_logits"][..., position_intent] = 10.0
    intent_legal = torch.zeros(1, 1, intent_dim)
    intent_legal[..., position_intent] = 1.0
    target_legal = torch.zeros(1, 1, intent_dim, target_dim)
    position_legal = torch.zeros(1, 1, intent_dim, position_dim)
    sampled = sample_command_actions_v1(
        command_output,
        intent_legal,
        target_legal,
        position_legal,
        command_is_train=torch.ones(1, 1),
        deterministic=True,
    )
    if sampled["actions"].tolist() != [[[0, 0, 0]]]:
        raise RuntimeError("command guard self-test did not fall back to no-command")
    if sampled["command_is_train"].item() != 0:
        raise RuntimeError("command guard self-test fallback remained trainable")

    child_guard = _optional_head(
        torch.zeros(1, position_dim),
        torch.zeros(1, position_dim),
        torch.ones(1, dtype=torch.bool),
        "command position self-test",
        deterministic=True,
    )
    if not bool(child_guard["invalid_active"].item()):
        raise RuntimeError("command guard self-test did not contain all-illegal child row")
    if child_guard["log_probability"].item() != 0:
        raise RuntimeError("command guard self-test fallback log-probability is not zero")

    return (
        "COMMAND_GUARD_SELF_TEST_OK "
        f"contract={FrozenV1Config.COMMAND_HEADS} sampler={COMMAND_SAMPLER_VERSION}"
    )


def sample_command_actions_v1(
    command_output,
    intent_legal,
    target_legal_by_intent,
    position_legal_by_intent,
    command_is_train=None,
    deterministic=False,
):
    intent_logits = command_output["intent_logits"]
    target_logits_by_intent = command_output["target_logits"]
    position_logits_by_intent = command_output["position_logits"]
    intent_legal = _coherent_intent_legal_v1(
        intent_legal, target_legal_by_intent, position_legal_by_intent
    )
    intent_probabilities = masked_probabilities_v1(
        intent_logits, intent_legal, "command intent"
    )
    intent_action = _sample(intent_probabilities, deterministic)
    intent_selected = intent_probabilities.gather(
        -1, intent_action.unsqueeze(-1)
    ).squeeze(-1)
    intent_log_probability = intent_selected.clamp_min(1e-12).log()
    intent_entropy = -(
        intent_probabilities * intent_probabilities.clamp_min(1e-12).log()
    ).sum(dim=-1)

    target_logits = select_intent_row_v1(
        target_logits_by_intent, intent_action, 39
    )
    position_logits = select_intent_row_v1(
        position_logits_by_intent, intent_action, FrozenV1Config.COMMAND_HEADS[2]
    )
    target_legal = select_intent_row_v1(
        target_legal_by_intent, intent_action, 39
    )
    position_legal = select_intent_row_v1(
        position_legal_by_intent, intent_action, FrozenV1Config.COMMAND_HEADS[2]
    )
    target_active = torch.zeros_like(intent_action, dtype=torch.bool)
    position_active = torch.zeros_like(intent_action, dtype=torch.bool)
    for intent in TARGET_INTENTS:
        target_active |= intent_action == intent
    for intent in POSITION_INTENTS:
        position_active |= intent_action == intent
    target = _optional_head(
        target_logits,
        target_legal,
        target_active,
        "command target",
        deterministic,
    )
    position = _optional_head(
        position_logits,
        position_legal,
        position_active,
        "command position",
        deterministic,
    )
    actions = torch.stack(
        (intent_action, target["action"], position["action"]), dim=-1
    )
    log_probabilities = torch.stack(
        (
            intent_log_probability,
            target["log_probability"],
            position["log_probability"],
        ),
        dim=-1,
    )
    active_head_mask = torch.stack(
        (
            torch.ones_like(intent_action, dtype=torch.bool),
            target_active,
            position_active,
        ),
        dim=-1,
    )
    repaired = target["invalid_active"] | position["invalid_active"]
    conditioned_legal = torch.cat(
        (intent_legal, target_legal, position_legal), dim=-1
    )
    if conditioned_legal.shape[-1] != FrozenV1Config.COMMAND_LEGAL_MASK_DIM:
        raise RuntimeError("conditioned command legal mask dimension drifted")
    if torch.any(repaired):
        # Last-resort inference containment: an inconsistent active child head
        # must not terminate the actor/workflow/Reverb chain.  The repaired
        # row executes the official no-command protocol and is excluded from
        # command PPO so the policy is never trained on an action it did not
        # execute.
        actions = torch.where(repaired.unsqueeze(-1), torch.zeros_like(actions), actions)
        log_probabilities = torch.where(
            repaired.unsqueeze(-1),
            torch.zeros_like(log_probabilities),
            log_probabilities,
        )
        fallback_active = torch.zeros_like(active_head_mask)
        fallback_active[..., 0] = True
        active_head_mask = torch.where(
            repaired.unsqueeze(-1), fallback_active, active_head_mask
        )
        fallback_legal = torch.zeros_like(conditioned_legal)
        fallback_legal[..., 0] = 1
        conditioned_legal = torch.where(
            repaired.unsqueeze(-1), fallback_legal, conditioned_legal
        )
        fallback_intent_probabilities = torch.zeros_like(intent_probabilities)
        fallback_intent_probabilities[..., 0] = 1
        intent_probabilities = torch.where(
            repaired.unsqueeze(-1),
            fallback_intent_probabilities,
            intent_probabilities,
        )
        intent_entropy = torch.where(
            repaired, torch.zeros_like(intent_entropy), intent_entropy
        )
    coherent_is_train = (intent_legal[..., 1:] > 0.5).any(dim=-1)
    if command_is_train is None:
        command_is_train = coherent_is_train
    else:
        command_is_train = (command_is_train > 0.5) & coherent_is_train
    command_is_train = command_is_train & ~repaired
    return {
        "actions": actions,
        "old_log_probabilities": log_probabilities,
        "conditioned_legal_mask": conditioned_legal,
        "active_head_mask": active_head_mask.to(intent_logits.dtype),
        "command_is_train": command_is_train.to(intent_logits.dtype),
        "intent_probabilities": intent_probabilities,
        "target_probabilities": target["probabilities"],
        "position_probabilities": position["probabilities"],
        "entropy": torch.stack(
            (intent_entropy, target["entropy"], position["entropy"]), dim=-1
        ),
        "mask_repaired": repaired.to(intent_logits.dtype),
    }


def split_conditioned_command_legal_v1(conditioned_legal_mask):
    if conditioned_legal_mask.shape[-1] != FrozenV1Config.COMMAND_LEGAL_MASK_DIM:
        raise ValueError(
            f"conditioned command legal mask must end in 58, got {conditioned_legal_mask.shape[-1]}"
        )
    return conditioned_legal_mask.split(FrozenV1Config.COMMAND_HEADS, dim=-1)


def evaluate_command_actions_v1(
    command_output,
    conditioned_legal_mask,
    actions,
    active_head_mask,
):
    if actions.shape[-1] != 3 or active_head_mask.shape != actions.shape:
        raise ValueError("command actions and active head mask must end in 3")
    intent_legal, target_legal, position_legal = split_conditioned_command_legal_v1(
        conditioned_legal_mask
    )
    intent_action = actions[..., 0].long()
    intent_probabilities = masked_probabilities_v1(
        command_output["intent_logits"], intent_legal, "command intent"
    )
    intent_selected = intent_probabilities.gather(
        -1, intent_action.unsqueeze(-1)
    ).squeeze(-1)
    target_logits = select_intent_row_v1(
        command_output["target_logits"], intent_action, 39
    )
    position_logits = select_intent_row_v1(
        command_output["position_logits"], intent_action, FrozenV1Config.COMMAND_HEADS[2]
    )
    target = _optional_head(
        target_logits,
        target_legal,
        active_head_mask[..., 1] > 0.5,
        "command target",
        deterministic=True,
    )
    position = _optional_head(
        position_logits,
        position_legal,
        active_head_mask[..., 2] > 0.5,
        "command position",
        deterministic=True,
    )
    target_selected = target["probabilities"].gather(
        -1, actions[..., 1].long().unsqueeze(-1)
    ).squeeze(-1)
    position_selected = position["probabilities"].gather(
        -1, actions[..., 2].long().unsqueeze(-1)
    ).squeeze(-1)
    target_log = torch.where(
        active_head_mask[..., 1] > 0.5,
        target_selected.clamp_min(1e-12).log(),
        torch.zeros_like(target_selected),
    )
    position_log = torch.where(
        active_head_mask[..., 2] > 0.5,
        position_selected.clamp_min(1e-12).log(),
        torch.zeros_like(position_selected),
    )
    intent_entropy = -(
        intent_probabilities * intent_probabilities.clamp_min(1e-12).log()
    ).sum(dim=-1)
    return {
        "selected_log_probabilities": torch.stack(
            (intent_selected.clamp_min(1e-12).log(), target_log, position_log),
            dim=-1,
        ),
        "entropy": torch.stack(
            (intent_entropy, target["entropy"], position["entropy"]), dim=-1
        ),
        "intent_probabilities": intent_probabilities,
        "target_probabilities": target["probabilities"],
        "position_probabilities": position["probabilities"],
    }


__all__ = (
    "evaluate_command_actions_v1",
    "sample_command_actions_v1",
    "select_intent_row_v1",
    "split_conditioned_command_legal_v1",
)
