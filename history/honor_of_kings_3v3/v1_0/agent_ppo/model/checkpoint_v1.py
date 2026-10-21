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


def validate_checkpoint_signature_v1(signature):
    expected = checkpoint_signature_v1()
    expected_fingerprint = signature_fingerprint_v1(expected)
    actual_fingerprint = signature_fingerprint_v1(signature)
    if actual_fingerprint != expected_fingerprint:
        mismatches = []
        if not isinstance(signature, dict):
            mismatches.append(f"signature_type={type(signature).__name__}")
        else:
            for key in sorted(set(expected) | set(signature)):
                if _json_safe(expected.get(key)) != _json_safe(signature.get(key)):
                    mismatches.append(key)
        raise ValueError(
            "checkpoint hybrid multi-value signature mismatch: "
            + ", ".join(mismatches[:16])
        )
    return expected_fingerprint


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
    fingerprint = validate_checkpoint_signature_v1(payload.get("signature"))
    if payload.get("signature_fingerprint") != fingerprint:
        raise ValueError("checkpoint signature fingerprint is corrupt")
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
        "signature_fingerprint": fingerprint,
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
