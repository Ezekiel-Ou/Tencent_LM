#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""Interruptible opening and post-camp departure rules for controlled Kai.

The rule changes legal actions before PPO sampling. Travel, pre-spawn waiting,
and common attacks on the protected upper/lower opening camps are therefore represented
by the exact conditioned mask stored in the PPO sample.
"""

import numpy as np

from agent_ppo.conf.conf import FrozenV1Config
from agent_ppo.feature.collision_avoidance_v1 import (
    MOVE_BUTTON,
    MOVE_DIRECTION_COUNT,
    MOVE_HEAD_OFFSET,
)
from agent_ppo.feature.lane_opening_v1 import _ranked_move_directions
from agent_ppo.feature.lane_opening_v1 import (
    COMMON_ATTACK_BUTTON,
    TARGET_COUNT,
    TARGET_MATRIX_OFFSET,
)


KAI_CONFIG_ID = 193


class JungleOpeningLegalizerV1:
    """Force controlled Kai through upper jungle and into the first lower camp."""

    VERSION = FrozenV1Config.ORDINARY_ACTION_LEGALIZER_VERSION

    def __init__(self):
        self.reset()

    def reset(self):
        self.mask_application_count = 0
        self.attack_application_count = 0
        self.exit_application_count = 0
        self.interrupt_count = 0
        self.last_diagnostics = ()

    def apply(self, raw_legal_action, config_ids, runtime_plans, collision_diagnostics):
        result = np.asarray(raw_legal_action, dtype=np.float32).copy()
        expected = (
            FrozenV1Config.CONTROLLED_ACTOR_COUNT,
            FrozenV1Config.RAW_LEGAL_ACTION_DIM,
        )
        if result.shape != expected:
            raise ValueError(f"invalid jungle opening raw legal shape: {result.shape}")
        if len(config_ids) != expected[0] or len(runtime_plans) != expected[0]:
            raise ValueError("jungle opening inputs must follow the controlled actors")

        collision_by_runtime = {
            int(item.get("runtime_id", 0)): item
            for item in (collision_diagnostics or ())
        }
        diagnostics = []
        for actor_index, (config_id, source_plan) in enumerate(
            zip(config_ids, runtime_plans)
        ):
            plan = dict(source_plan or {})
            runtime_id = int(plan.get("runtime_id", 0))
            collision = collision_by_runtime.get(runtime_id, {})
            escape_active = bool(collision.get("escape_active", False))
            is_kai = int(config_id) == KAI_CONFIG_ID
            active = bool(is_kai and plan.get("hard_mask_active") and not escape_active)
            selected = tuple()
            reason = "inactive"
            if is_kai and escape_active:
                reason = "collision_escape"
                self.interrupt_count += 1
            elif is_kai and plan.get("interrupt", "none") != "none":
                reason = str(plan.get("interrupt"))
                self.interrupt_count += 1
            elif active:
                button_mask = result[actor_index, : FrozenV1Config.HERO_ACTION_HEADS[0]]
                move_mask = result[
                    actor_index,
                    MOVE_HEAD_OFFSET : MOVE_HEAD_OFFSET + MOVE_DIRECTION_COUNT,
                ]
                target_slot = int(plan.get("target_slot", -1))
                target_offset = (
                    TARGET_MATRIX_OFFSET
                    + COMMON_ATTACK_BUTTON * TARGET_COUNT
                    + target_slot
                )
                attack_legal = bool(
                    plan.get("force_attack", False)
                    and 8 <= target_slot < 28
                    and button_mask[COMMON_ATTACK_BUTTON] > 0.5
                    and result[actor_index, target_offset] > 0.5
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
                    reason = (
                        "lower_jungle_attack"
                        if plan.get("phase") == "lower_clear"
                        else "upper_jungle_attack"
                    )
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
                            MOVE_HEAD_OFFSET : MOVE_HEAD_OFFSET + MOVE_DIRECTION_COUNT,
                        ] = 0.0
                        for direction in selected:
                            result[actor_index, MOVE_HEAD_OFFSET + direction] = 1.0
                        self.mask_application_count += 1
                        if plan.get("camp_exit_active", False):
                            self.exit_application_count += 1
                            reason = "camp_exit_rule"
                        else:
                            reason = (
                                "lower_jungle_rule"
                                if plan.get("phase") == "lower_clear"
                                else "upper_jungle_rule"
                            )
            elif is_kai and plan.get("phase") in (
                "upper_deer",
                "upper_wolf",
                "lower_clear",
            ):
                reason = "opening_inactive"

            diagnostics.append(
                {
                    "runtime_id": runtime_id,
                    "config_id": int(config_id),
                    "opening_mask_active": active,
                    "camp_exit_active": bool(plan.get("camp_exit_active", False)),
                    "phase": plan.get("phase", "inactive"),
                    "interrupt": plan.get("interrupt", "none"),
                    "selected_directions": selected,
                    "reason": reason,
                }
            )
        self.last_diagnostics = tuple(diagnostics)
        return result, self.last_diagnostics


JUNGLE_OPENING_LEGALIZER_SIGNATURE = (
    ("kai_config_id", KAI_CONFIG_ID),
    ("phases", ("upper_deer", "upper_wolf", "lower_clear")),
    ("ordering", "after_channel4_collision_before_lane_opening_and_sampling"),
    ("combat_hold", "until_first_lower_camp_engaged_or_emergency"),
    ("combat_action", "forced_legal_common_attack"),
    ("camp_exit_distance", 6000.0),
    ("cleared_camp_wait", "excluded_until_living_actor_observed"),
    ("topology_cursor", "monotonic_until_goal_or_interrupt_changes"),
)


__all__ = (
    "JUNGLE_OPENING_LEGALIZER_SIGNATURE",
    "JungleOpeningLegalizerV1",
    "KAI_CONFIG_ID",
)
