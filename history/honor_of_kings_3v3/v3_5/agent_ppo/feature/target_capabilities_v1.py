#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""Single-source command target capabilities for PPO 3.5.

Target slots remain part of the frozen observation/action contract even when a
protocol capability is disabled.  In particular, ordinary monster runtime IDs
stay observable for diagnostics and actor targeting, while collaboration
``SetTarget`` is permanently disabled for them.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class TargetCapabilityV1:
    target_type: str
    set_target: bool
    set_position: bool
    constraint: str


TARGET_CAPABILITIES_V1 = {
    "ordinary_monster": TargetCapabilityV1(
        "ordinary_monster",
        False,
        True,
        "all_official_heroes_all_phases",
    ),
    "enemy_hero": TargetCapabilityV1(
        "enemy_hero",
        True,
        True,
        "alive_visible_and_spawn_safe",
    ),
    "lane_soldier": TargetCapabilityV1(
        "lane_soldier",
        True,
        True,
        "bound_lane_and_runtime_lifecycle",
    ),
    "structure": TargetCapabilityV1(
        "structure",
        True,
        True,
        "wave_and_tower_safety",
    ),
    "tyrant": TargetCapabilityV1(
        "tyrant",
        True,
        True,
        "verified_neutral_objective",
    ),
    "treasure_thief": TargetCapabilityV1(
        "treasure_thief",
        False,
        True,
        "set_target_requires_separate_platform_evidence",
    ),
    "semantic_position": TargetCapabilityV1(
        "semantic_position",
        False,
        True,
        "position_bank_only",
    ),
}


def monster_target_type(config_id):
    config_id = int(config_id)
    if config_id == 122:
        return "tyrant"
    if config_id == 59:
        return "treasure_thief"
    return "ordinary_monster"


def monster_set_target_allowed(config_id):
    return bool(TARGET_CAPABILITIES_V1[monster_target_type(config_id)].set_target)


def monster_set_position_allowed(config_id):
    return bool(TARGET_CAPABILITIES_V1[monster_target_type(config_id)].set_position)


__all__ = (
    "TARGET_CAPABILITIES_V1",
    "TargetCapabilityV1",
    "monster_set_position_allowed",
    "monster_set_target_allowed",
    "monster_target_type",
)
