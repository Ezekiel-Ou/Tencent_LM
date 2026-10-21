#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""Pre-sampling Flash gate and button-conditioned offset masks."""

import math

import numpy as np

from agent_ppo.conf.conf import FrozenV1Config
from agent_ppo.feature.collision_avoidance_v1 import direction_clearance


# The official 13-way button protocol uses 8 for the talent/summoner skill.
# Button 7 is skill 4 and is not Flash for Marco Polo or Diao Chan.
FLASH_BUTTON = 8
OFFSET_X_START = FrozenV1Config.HERO_ACTION_HEADS[0] + FrozenV1Config.HERO_ACTION_HEADS[1]
OFFSET_Z_START = OFFSET_X_START + FrozenV1Config.HERO_ACTION_HEADS[2]
OFFSET_COUNT = FrozenV1Config.HERO_ACTION_HEADS[2]
OFFSET_CENTER = 21
OFFSET_STEP = 1000.0
FLASH_DISTANCE = 7000.0


def _nearest_move_index(dx, dz):
    if dx == 0.0 and dz == 0.0:
        return 0
    angle = math.degrees(math.atan2(dz, dx)) % 360.0
    return 1 + int(round(((180.0 - angle) % 360.0) / 15.0)) % 24


class FlashLegalizerV1:
    """Default-deny Flash while preserving the platform's authoritative mask."""

    def __init__(self):
        self.official_legal_count = 0
        self.gate_open_count = 0
        self.blocked_count = 0
        self.no_safe_landing_count = 0
        self.low_hp_count = 0
        self.tower_count = 0
        self.chase_count = 0
        self.marco_ultimate_count = 0

    def apply(self, raw_legal_action, plans):
        result = np.asarray(raw_legal_action, dtype=np.float32).copy()
        if result.shape != (
            FrozenV1Config.CONTROLLED_ACTOR_COUNT,
            FrozenV1Config.RAW_LEGAL_ACTION_DIM,
        ):
            raise ValueError(f"invalid raw legal action shape: {result.shape}")
        if len(plans) != FrozenV1Config.CONTROLLED_ACTOR_COUNT:
            raise ValueError("flash plans must match controlled actor count")

        offset_masks = np.ones(
            (FrozenV1Config.CONTROLLED_ACTOR_COUNT, 2, OFFSET_COUNT),
            dtype=np.float32,
        )
        diagnostics = []
        for actor_index, plan in enumerate(plans):
            official_flash_legal = result[actor_index, FLASH_BUTTON] > 0.5
            if not bool(plan.get("flash_gate_active", True)):
                diagnostics.append(
                    {
                        "runtime_id": int(plan.get("runtime_id", 0)),
                        "official_legal": bool(official_flash_legal),
                        "allowed": False,
                        "reason": str(plan.get("reason", "not_flash")),
                        "mode": "not_flash",
                    }
                )
                continue
            self.official_legal_count += int(official_flash_legal)
            allowed = bool(plan.get("allow", False) and official_flash_legal)
            reason = str(plan.get("reason", "blocked"))
            if not allowed:
                result[actor_index, FLASH_BUTTON] = 0.0
                self.blocked_count += int(official_flash_legal)
                diagnostics.append(
                    {
                        "runtime_id": int(plan.get("runtime_id", 0)),
                        "official_legal": bool(official_flash_legal),
                        "allowed": False,
                        "reason": reason,
                        "mode": "blocked",
                    }
                )
                continue

            dx = float(plan.get("direction_dx", 0.0))
            dz = float(plan.get("direction_dz", 0.0))
            length = math.hypot(dx, dz)
            obstacle = plan.get("obstacle")
            if length <= 1e-6 or obstacle is None:
                result[actor_index, FLASH_BUTTON] = 0.0
                reason = "no_safe_landing"
                allowed = False
            else:
                dx = dx / length * FLASH_DISTANCE
                dz = dz / length * FLASH_DISTANCE
                move_index = _nearest_move_index(dx, dz)
                if direction_clearance(np.asarray(obstacle), move_index) <= 0.0:
                    result[actor_index, FLASH_BUTTON] = 0.0
                    reason = "uncertain_wall"
                    allowed = False

            if allowed:
                x_index = int(np.clip(OFFSET_CENTER + round(dx / OFFSET_STEP), 1, 41))
                z_index = int(np.clip(OFFSET_CENTER + round(dz / OFFSET_STEP), 1, 41))
                platform_x = result[
                    actor_index, OFFSET_X_START : OFFSET_X_START + OFFSET_COUNT
                ]
                platform_z = result[
                    actor_index, OFFSET_Z_START : OFFSET_Z_START + OFFSET_COUNT
                ]
                if platform_x[x_index] <= 0.5 or platform_z[z_index] <= 0.5:
                    result[actor_index, FLASH_BUTTON] = 0.0
                    reason = "official_offset_illegal"
                    allowed = False
                else:
                    offset_masks[actor_index, 0].fill(0.0)
                    offset_masks[actor_index, 1].fill(0.0)
                    offset_masks[actor_index, 0, x_index] = 1.0
                    offset_masks[actor_index, 1, z_index] = 1.0

            if allowed:
                self.gate_open_count += 1
                mode = str(plan.get("mode", ""))
                self.low_hp_count += int(mode == "escape")
                self.tower_count += int(mode == "tower_escape")
                self.chase_count += int(mode == "chase")
                self.marco_ultimate_count += int(mode == "marco_ultimate")
            elif reason in (
                "no_safe_landing",
                "uncertain_wall",
                "official_offset_illegal",
                "landing_out_of_bounds",
                "landing_beyond_spawn_boundary",
            ):
                self.no_safe_landing_count += 1

            diagnostics.append(
                {
                    "runtime_id": int(plan.get("runtime_id", 0)),
                    "official_legal": bool(official_flash_legal),
                    "allowed": bool(allowed),
                    "reason": reason,
                    "mode": str(plan.get("mode", "blocked")) if allowed else "blocked",
                }
            )
        return result, offset_masks, tuple(diagnostics)


__all__ = ("FLASH_BUTTON", "FlashLegalizerV1")
