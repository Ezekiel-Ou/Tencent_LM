#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""PPO-consistent opening deployment for the two lane heroes.

The planner itself lives in :mod:`v1_feature_process`; this module only turns
an already observed lane plan into a pre-sampling legal-action mask.  The
conditioned mask and the probabilities produced from it are therefore stored
in the ordinary PPO sample without any post-sampling action replacement.
"""

import math

import numpy as np

from agent_ppo.conf.conf import FrozenV1Config
from agent_ppo.feature.collision_avoidance_v1 import (
    MOVE_BUTTON,
    MOVE_DIRECTION_COUNT,
    MOVE_HEAD_OFFSET,
    move_direction_vector,
)


LANE_HERO_CONFIG_IDS = (132, 141)
OPENING_DIRECTION_FAN = 3
WAYPOINT_ARRIVAL_DISTANCE = 1800.0
COMMON_ATTACK_BUTTON = 3
TARGET_MATRIX_OFFSET = sum(FrozenV1Config.HERO_ACTION_HEADS[:-1])
TARGET_COUNT = FrozenV1Config.HERO_ACTION_HEADS[-1]


def _ranked_move_directions(legal_move, desired_dx, desired_dz):
    legal_move = np.asarray(legal_move, dtype=np.float32).reshape(-1)
    if legal_move.shape != (MOVE_DIRECTION_COUNT,):
        raise ValueError(f"lane opening move mask must be [25], got {legal_move.shape}")
    length = math.hypot(float(desired_dx), float(desired_dz))
    if length <= WAYPOINT_ARRIVAL_DISTANCE:
        return (0,) if legal_move[0] > 0.5 else tuple()
    desired_x = float(desired_dx) / length
    desired_z = float(desired_dz) / length
    candidates = [index for index in range(1, 25) if legal_move[index] > 0.5]
    return tuple(
        sorted(
            candidates,
            key=lambda index: -(
                move_direction_vector(index)[0] * desired_x
                + move_direction_vector(index)[1] * desired_z
            ),
        )[:OPENING_DIRECTION_FAN]
    )


class LaneOpeningLegalizerV1:
    """Constrain only uninterrupted opening movement before actor sampling."""

    VERSION = FrozenV1Config.ORDINARY_ACTION_LEGALIZER_VERSION

    def __init__(self):
        self.reset()

    def reset(self):
        self.mask_application_count = 0
        self.attack_application_count = 0
        self.released_count = 0
        self.last_diagnostics = ()
        self._released_runtime_ids = set()

    def apply(self, raw_legal_action, config_ids, runtime_plans, collision_diagnostics):
        result = np.asarray(raw_legal_action, dtype=np.float32).copy()
        expected = (
            FrozenV1Config.CONTROLLED_ACTOR_COUNT,
            FrozenV1Config.RAW_LEGAL_ACTION_DIM,
        )
        if result.shape != expected:
            raise ValueError(f"invalid lane opening raw legal shape: {result.shape}")
        if len(config_ids) != expected[0] or len(runtime_plans) != expected[0]:
            raise ValueError("lane opening inputs must follow the two controlled actors")

        collision_by_runtime = {
            int(item.get("runtime_id", 0)): item
            for item in (collision_diagnostics or ())
        }
        diagnostics = []
        for actor_index, (config_id, plan) in enumerate(zip(config_ids, runtime_plans)):
            plan = dict(plan or {})
            runtime_id = int(plan.get("runtime_id", 0))
            collision = collision_by_runtime.get(runtime_id, {})
            escape_active = bool(collision.get("escape_active", False))
            active = bool(
                int(config_id) in LANE_HERO_CONFIG_IDS
                and plan.get("hard_mask_active", False)
                and not escape_active
            )
            selected = tuple()
            reason = "inactive"
            if escape_active:
                reason = "collision_escape"
            elif active:
                button_mask = result[actor_index, : FrozenV1Config.HERO_ACTION_HEADS[0]]
                move_mask = result[
                    actor_index,
                    MOVE_HEAD_OFFSET : MOVE_HEAD_OFFSET + MOVE_DIRECTION_COUNT,
                ]
                target_slot = int(plan.get("target_slot", -1))
                attack_target_offset = (
                    TARGET_MATRIX_OFFSET
                    + COMMON_ATTACK_BUTTON * TARGET_COUNT
                    + target_slot
                )
                attack_legal = bool(
                    plan.get("phase") == "engage_wave"
                    and 28 <= target_slot < TARGET_COUNT
                    and button_mask[COMMON_ATTACK_BUTTON] > 0.5
                    and result[actor_index, attack_target_offset] > 0.5
                )
                if attack_legal:
                    result[
                        actor_index, : FrozenV1Config.HERO_ACTION_HEADS[0]
                    ] = 0.0
                    result[actor_index, COMMON_ATTACK_BUTTON] = 1.0
                    target_row_start = (
                        TARGET_MATRIX_OFFSET + COMMON_ATTACK_BUTTON * TARGET_COUNT
                    )
                    result[
                        actor_index,
                        target_row_start : target_row_start + TARGET_COUNT,
                    ] = 0.0
                    result[actor_index, target_row_start + target_slot] = 1.0
                    self.mask_application_count += 1
                    self.attack_application_count += 1
                    reason = "first_wave_attack"
                elif button_mask[MOVE_BUTTON] <= 0.5:
                    active = False
                    reason = "move_button_illegal"
                else:
                    selected = _ranked_move_directions(
                        move_mask,
                        plan.get("waypoint_dx", 0.0),
                        plan.get("waypoint_dz", 0.0),
                    )
                    if not selected:
                        active = False
                        reason = "no_legal_route_direction"
                    else:
                        result[
                            actor_index, : FrozenV1Config.HERO_ACTION_HEADS[0]
                        ] = 0.0
                        result[actor_index, MOVE_BUTTON] = 1.0
                        result[
                            actor_index,
                            MOVE_HEAD_OFFSET : MOVE_HEAD_OFFSET
                            + MOVE_DIRECTION_COUNT,
                        ] = 0.0
                        for direction in selected:
                            result[actor_index, MOVE_HEAD_OFFSET + direction] = 1.0
                        self.mask_application_count += 1
                        reason = "opening_route"

            if plan.get("phase") == "released" and runtime_id not in self._released_runtime_ids:
                self._released_runtime_ids.add(runtime_id)
                self.released_count += 1
            diagnostics.append(
                {
                    "runtime_id": runtime_id,
                    "config_id": int(config_id),
                    "opening_mask_active": active,
                    "phase": plan.get("phase", "inactive"),
                    "interrupt": plan.get("interrupt", "none"),
                    "selected_directions": selected,
                    "reason": reason,
                }
            )

        self.last_diagnostics = tuple(diagnostics)
        return result, self.last_diagnostics


LANE_OPENING_LEGALIZER_SIGNATURE = (
    ("lane_hero_config_ids", LANE_HERO_CONFIG_IDS),
    ("assigned_row_z", ((132, 3150.0), (141, -3150.0))),
    ("opening_direction_fan", OPENING_DIRECTION_FAN),
    ("waypoint_arrival_distance", WAYPOINT_ARRIVAL_DISTANCE),
    ("release_condition", "actual_enemy_soldier_contact"),
    (
        "post_contact_control",
        "ppo_for_controlled_official_until_contact_and_first_wave_clear",
    ),
    (
        "official_route_guard",
        "physical_monotonic_initial_plus_one_retry_then_successor",
    ),
    ("ordering", "after_channel4_collision_before_sampling"),
)


__all__ = (
    "LANE_HERO_CONFIG_IDS",
    "LANE_OPENING_LEGALIZER_SIGNATURE",
    "LaneOpeningLegalizerV1",
)
