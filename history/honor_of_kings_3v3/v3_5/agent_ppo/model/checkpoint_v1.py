#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""Strict hybrid multi-value checkpoint serialization and compatibility checks."""

import hashlib
import json
import os

import torch

from agent_ppo.feature.schema_v1 import checkpoint_signature_v1


CHECKPOINT_FORMAT = "agent_ppo_hybrid_multivalue"
CHECKPOINT_FORMAT_VERSION = 2

# These are deliberately narrow, one-way migrations for tensor-compatible
# checkpoints produced before confirmed opening or ordinary-action corrections.
# Every fingerprint binds the complete checkpoint signature; the mismatch
# allowlist prevents these exceptions from accepting any other drift.
_COMPATIBLE_SIGNATURE_MIGRATIONS = {
    "d70ba859f676b77973895e7784382ef68d1097bc5e0149dd74f5185b546ea86a": {
        "target_fingerprint": (
            "87c82985450ce4dbeba09a28c24812566b32e58cfd344d8e1fb6acc4030d27f1"
        ),
        "mismatch_keys": (
            "feature_version",
            "jungle_opening_legalizer_signature",
            "lane_opening_legalizer_signature",
            "legalizer_version",
            "ordinary_action_legalizer_version",
        ),
        "name": "pre_six_camp_opening_to_six_camp_opening_v1",
    },
    "2814385704a4d02163f20c08b914c516855b0e8cf74a6e8df75d136b180e21c0": {
        "target_fingerprint": (
            "87c82985450ce4dbeba09a28c24812566b32e58cfd344d8e1fb6acc4030d27f1"
        ),
        "mismatch_keys": (
            "feature_version",
            "jungle_opening_legalizer_signature",
            "lane_opening_legalizer_signature",
            "legalizer_version",
            "ordinary_action_legalizer_version",
        ),
        "name": "six_camp_common_attack_to_skill_route_v1",
    },
    "94190f4d925caea5b1538053603ced8982eadb30f90daaf5d7aae14e306fd848": {
        "target_fingerprint": (
            "87c82985450ce4dbeba09a28c24812566b32e58cfd344d8e1fb6acc4030d27f1"
        ),
        "mismatch_keys": (
            "feature_version",
            "jungle_opening_legalizer_signature",
            "lane_opening_legalizer_signature",
            "legalizer_version",
            "ordinary_action_legalizer_version",
        ),
        "name": "opening_skill_route_to_official_region_route_v1",
    },
    "07034eea528d6060edaaed2bc7052145b753d3b338ecc0ef583e478e482a2745": {
        "target_fingerprint": (
            "87c82985450ce4dbeba09a28c24812566b32e58cfd344d8e1fb6acc4030d27f1"
        ),
        "mismatch_keys": (
            "feature_version",
            "jungle_opening_legalizer_signature",
            "lane_opening_legalizer_signature",
            "legalizer_version",
            "ordinary_action_legalizer_version",
        ),
        "name": "official_region_route_to_adaptive_timeout_v1",
    },
    "31fb2f82761c1e8164795861d5ccbb3ef8af8a0625cc46e37498d1fe19856e95": {
        "target_fingerprint": (
            "87c82985450ce4dbeba09a28c24812566b32e58cfd344d8e1fb6acc4030d27f1"
        ),
        "mismatch_keys": (
            "feature_version",
            "jungle_opening_legalizer_signature",
            "lane_opening_legalizer_signature",
            "legalizer_version",
            "ordinary_action_legalizer_version",
        ),
        "name": "adaptive_timeout_to_monotonic_descent_v1",
    },
    "23fe8203be80173ee8536274de1f2bafc30c1518496cf1b987e85edffa3d3b5c": {
        "target_fingerprint": (
            "87c82985450ce4dbeba09a28c24812566b32e58cfd344d8e1fb6acc4030d27f1"
        ),
        "mismatch_keys": (
            "feature_version",
            "jungle_opening_legalizer_signature",
            "lane_opening_legalizer_signature",
            "legalizer_version",
            "ordinary_action_legalizer_version",
        ),
        "name": "monotonic_descent_to_confirmed_clear_skip_v1",
    },
    "83055549effa845646cbcc248b7e370a1e23efeffed9bef6d8e889a3b62d5daa": {
        "target_fingerprint": (
            "87c82985450ce4dbeba09a28c24812566b32e58cfd344d8e1fb6acc4030d27f1"
        ),
        "mismatch_keys": (
            "feature_version",
            "jungle_opening_legalizer_signature",
            "lane_opening_legalizer_signature",
            "legalizer_version",
            "ordinary_action_legalizer_version",
        ),
        "name": "confirmed_clear_skip_to_flash_gate_v1",
    },
    "1d0c0e4cc45a4fe5e5b1f4c1c4bf16a3fcd767b05a64e7a46ad958e09f4ae2e6": {
        "target_fingerprint": (
            "87c82985450ce4dbeba09a28c24812566b32e58cfd344d8e1fb6acc4030d27f1"
        ),
        "mismatch_keys": (
            "feature_version",
            "jungle_opening_legalizer_signature",
            "lane_opening_legalizer_signature",
            "legalizer_version",
            "ordinary_action_legalizer_version",
        ),
        "name": "lane_role_response_guard_and_kai_combat_lock_v1",
    },
    "b7206313dcd6828501500ba7a694cf12038f2d8eb412cb8cea341c4a0c76f0b2": {
        "target_fingerprint": (
            "87c82985450ce4dbeba09a28c24812566b32e58cfd344d8e1fb6acc4030d27f1"
        ),
        "mismatch_keys": ("ordinary_action_legalizer_version",),
        "name": "flash_button7_to_official_button8_v1",
    },
}

# PPO 3.5 keeps every tensor/layout dimension but intentionally changes feature
# semantics, legalizers and reward/runtime state machines.  All previously
# audited migrations therefore terminate at the latest behavior fingerprint.
# The last 3.0 Flash-button-8 checkpoint and the first PPO 3.5 candidate each
# get a narrow one-way entry.
_PPO35_FINGERPRINT = (
    "99fe8fa1a65c1ed97ef0028585d6a77fc535058ab19e214f63cd6c4b480c30aa"
)
_PPO35_REWARD_STATE_MACHINE_V2_FINGERPRINT = (
    "5a8fb0615cf776ac289c52bf4ac16c2d8b060e73aeccf92817a9060c6104df1a"
)
_PPO35_REWARD_STATE_MACHINE_V1_FINGERPRINT = (
    "3d2f9d6cf3f698a5e2fa94d2993ef83325563d62cf167e8691c521c4bd4bd374"
)
_PPO30_FLASH8_FINGERPRINT = (
    "87c82985450ce4dbeba09a28c24812566b32e58cfd344d8e1fb6acc4030d27f1"
)
for _migration in _COMPATIBLE_SIGNATURE_MIGRATIONS.values():
    _migration["target_fingerprint"] = _PPO35_FINGERPRINT
    _migration["mismatch_keys"] = tuple(_migration["mismatch_keys"]) + (
        "reward_engineering_version",
    )
_COMPATIBLE_SIGNATURE_MIGRATIONS[
    "b7206313dcd6828501500ba7a694cf12038f2d8eb412cb8cea341c4a0c76f0b2"
]["mismatch_keys"] = (
    "feature_version",
    "legalizer_version",
    "ordinary_action_legalizer_version",
    "reward_engineering_version",
)
_COMPATIBLE_SIGNATURE_MIGRATIONS[_PPO30_FLASH8_FINGERPRINT] = {
    "target_fingerprint": _PPO35_FINGERPRINT,
    "mismatch_keys": (
        "feature_version",
        "legalizer_version",
        "ordinary_action_legalizer_version",
        "reward_engineering_version",
    ),
    "name": "ppo30_flash8_to_ppo35_task_safety_reward_v1",
}
_COMPATIBLE_SIGNATURE_MIGRATIONS[
    _PPO35_REWARD_STATE_MACHINE_V1_FINGERPRINT
] = {
    "target_fingerprint": _PPO35_FINGERPRINT,
    "mismatch_keys": (
        "legalizer_version",
        "reward_engineering_version",
    ),
    "name": "ppo35_reward_state_machine_v1_to_v3",
}
_COMPATIBLE_SIGNATURE_MIGRATIONS[
    _PPO35_REWARD_STATE_MACHINE_V2_FINGERPRINT
] = {
    "target_fingerprint": _PPO35_FINGERPRINT,
    "mismatch_keys": (
        "legalizer_version",
        "reward_engineering_version",
    ),
    "name": "ppo35_reward_state_machine_v2_to_v3",
}


def _json_safe(value):
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def signature_fingerprint_v1(signature=None):
    signature = checkpoint_signature_v1() if signature is None else signature
    payload = json.dumps(
        _json_safe(signature), ensure_ascii=True, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _signature_mismatch_keys(expected, signature):
    if not isinstance(signature, dict):
        return (f"signature_type={type(signature).__name__}",)
    return tuple(
        key
        for key in sorted(set(expected) | set(signature))
        if _json_safe(expected.get(key)) != _json_safe(signature.get(key))
    )


def _signature_migration_name(source_fingerprint, target_fingerprint, mismatches):
    migration = _COMPATIBLE_SIGNATURE_MIGRATIONS.get(source_fingerprint)
    if migration is None:
        return None
    if migration["target_fingerprint"] != target_fingerprint:
        return None
    if tuple(mismatches) != tuple(migration["mismatch_keys"]):
        return None
    return migration["name"]


def validate_checkpoint_signature_v1(signature):
    expected = checkpoint_signature_v1()
    expected_fingerprint = signature_fingerprint_v1(expected)
    actual_fingerprint = signature_fingerprint_v1(signature)
    if actual_fingerprint == expected_fingerprint:
        return actual_fingerprint

    mismatches = _signature_mismatch_keys(expected, signature)
    migration_name = _signature_migration_name(
        actual_fingerprint,
        expected_fingerprint,
        mismatches,
    )
    if migration_name is None:
        raise ValueError(
            "checkpoint hybrid multi-value signature mismatch: "
            + ", ".join(mismatches[:16])
        )
    return actual_fingerprint


def build_checkpoint_payload_v1(model, optimizer=None, train_step=0, metadata=None):
    signature = checkpoint_signature_v1()
    payload = {
        "checkpoint_format": CHECKPOINT_FORMAT,
        "checkpoint_format_version": CHECKPOINT_FORMAT_VERSION,
        "signature": signature,
        "signature_fingerprint": signature_fingerprint_v1(signature),
        "model_state_dict": model.state_dict(),
        "train_step": int(train_step),
        "metadata": dict(metadata or {}),
    }
    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()
    return payload


def save_checkpoint_v1(path, model, optimizer=None, train_step=0, metadata=None):
    payload = build_checkpoint_payload_v1(
        model,
        optimizer=optimizer,
        train_step=train_step,
        metadata=metadata,
    )
    temporary = f"{path}.tmp-{os.getpid()}"
    torch.save(payload, temporary)
    os.replace(temporary, path)
    return payload["signature_fingerprint"]


def load_checkpoint_v1(
    path,
    model,
    optimizer=None,
    map_location=None,
    load_optimizer=False,
):
    payload = torch.load(path, map_location=map_location, weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError("checkpoint is not a hybrid multi-value payload")
    if payload.get("checkpoint_format") != CHECKPOINT_FORMAT:
        raise ValueError(
            "legacy or unknown checkpoint format is not loadable as hybrid multi-value"
        )
    if payload.get("checkpoint_format_version") != CHECKPOINT_FORMAT_VERSION:
        raise ValueError("unsupported hybrid multi-value checkpoint format version")
    source_fingerprint = validate_checkpoint_signature_v1(payload.get("signature"))
    if payload.get("signature_fingerprint") != source_fingerprint:
        raise ValueError("checkpoint signature fingerprint is corrupt")
    target_fingerprint = signature_fingerprint_v1()
    compatibility_migration = None
    if source_fingerprint != target_fingerprint:
        mismatches = _signature_mismatch_keys(
            checkpoint_signature_v1(),
            payload.get("signature"),
        )
        compatibility_migration = _signature_migration_name(
            source_fingerprint,
            target_fingerprint,
            mismatches,
        )
        if compatibility_migration is None:
            raise ValueError("checkpoint signature migration is not authorized")
    state_dict = payload.get("model_state_dict")
    if not isinstance(state_dict, dict):
        raise ValueError("checkpoint model_state_dict is missing")
    model.load_state_dict(state_dict, strict=True)
    if load_optimizer:
        if optimizer is None:
            raise ValueError("load_optimizer=True requires an optimizer")
        if "optimizer_state_dict" not in payload:
            raise ValueError("checkpoint optimizer_state_dict is missing")
        optimizer.load_state_dict(payload["optimizer_state_dict"])
    return {
        "train_step": int(payload.get("train_step", 0)),
        "metadata": dict(payload.get("metadata") or {}),
        "signature_fingerprint": source_fingerprint,
        "compatibility_migration": compatibility_migration,
    }


__all__ = (
    "CHECKPOINT_FORMAT",
    "CHECKPOINT_FORMAT_VERSION",
    "build_checkpoint_payload_v1",
    "load_checkpoint_v1",
    "save_checkpoint_v1",
    "signature_fingerprint_v1",
    "validate_checkpoint_signature_v1",
)

