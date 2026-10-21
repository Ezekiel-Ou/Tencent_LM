#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""PPO-consistent spawn-boundary and critical tower-risk masks."""

import math

import numpy as np

from agent_ppo.conf.conf import FrozenV1Config
from agent_ppo.feature.collision_avoidance_v1 import (
    MOVE_DIRECTION_COUNT,
    MOVE_HEAD_OFFSET,
    move_direction_vector,
)
from agent_ppo.feature.lane_opening_v1 import TARGET_COUNT, TARGET_MATRIX_OFFSET


DISPLACEMENT_SKILL_BUTTON = 5
DISPLACEMENT_SKILL_CONFIG_IDS = (132, 141)
UNCERTAIN_DISPLACEMENT_DISTANCE = 7000.0


class SafetyLegalizerV1:
    """Apply hard topology masks before sampling and SampleData capture."""

    VERSION = FrozenV1Config.ORDINARY_ACTION_LEGALIZER_VERSION

    def __init__(self):
        self.spawn_move_block_count = 0
        self.spawn_target_block_count = 0
        self.spawn_displacement_block_count = 0
        self.spawn_recovery_count = 0
        self.tower_critical_move_block_count = 0
        self.tower_critical_displacement_block_count = 0
        self.all_illegal_recovery_count = 0

    @staticmethod
    def _movement_step(plan):
        speed = max(0.0, float(plan.get("move_speed", 0.0)))
        return max(1200.0, speed * float(plan.get("decision_seconds", 0.396)))

    def apply(self, raw_legal_action, config_ids, plans):
        result = np.asarray(raw_legal_action, dtype=np.float32).copy()
        expected = (
            FrozenV1Config.CONTROLLED_ACTOR_COUNT,
            FrozenV1Config.RAW_LEGAL_ACTION_DIM,
        )
        if result.shape != expected:
            raise ValueError(f"invalid safety raw legal shape: {result.shape}")
        if len(config_ids) != expected[0] or len(plans) != expected[0]:
            raise ValueError("safety inputs must follow controlled actor order")

        diagnostics = []
        for actor_index, (config_id, source_plan) in enumerate(
            zip(config_ids, plans)
        ):
            plan = dict(source_plan or {})
            position = plan.get("position")
            boundary = float(plan.get("spawn_boundary_x", float("inf")))
            move_mask = result[
                actor_index,
                MOVE_HEAD_OFFSET : MOVE_HEAD_OFFSET + MOVE_DIRECTION_COUNT,
            ]
            blocked_spawn = 0
            blocked_tower = 0
            over_boundary = bool(
                position is not None and float(position[0]) > boundary
            )
            tower_position = plan.get("critical_tower_position")
            critical = bool(
                plan.get("tower_critical", False)
                and position is not None
                and tower_position is not None
            )
            movement_step = self._movement_step(plan)
            for move_index in range(1, MOVE_DIRECTION_COUNT):
                if move_mask[move_index] <= 0.5:
                    continue
                dx, dz = move_direction_vector(move_index)
                projected_x = (
                    float(position[0]) + dx * movement_step
                    if position is not None
                    else -float("inf")
                )
                spawn_unsafe = bool(
                    position is not None
                    and (
                        (over_boundary and dx > 1e-6)
                        or (not over_boundary and projected_x > boundary)
                    )
                )
                tower_unsafe = False
                if critical:
                    tx = float(tower_position[0]) - float(position[0])
                    tz = float(tower_position[1]) - float(position[1])
                    length = math.hypot(tx, tz)
                    tower_unsafe = bool(
                        length > 1e-6 and (dx * tx + dz * tz) / length > 0.25
                    )
                if spawn_unsafe or tower_unsafe:
                    move_mask[move_index] = 0.0
                    blocked_spawn += int(spawn_unsafe)
                    blocked_tower += int(tower_unsafe)

            # Stopping is the deterministic fallback and is always less deep
            # than a prohibited movement. Preserve it if the platform exposes
            # it, and recover defensively if another legalizer removed every
            # direction.
            if not np.any(move_mask > 0.5):
                move_mask[0] = 1.0
                self.all_illegal_recovery_count += 1
            if over_boundary:
                self.spawn_recovery_count += 1

            enemy_slots = tuple(int(slot) for slot in plan.get("unsafe_enemy_slots", ()))
            blocked_targets = 0
            for target_slot in enemy_slots:
                if not 0 <= target_slot < TARGET_COUNT:
                    continue
                for button in range(FrozenV1Config.HERO_ACTION_HEADS[0]):
                    offset = (
                        TARGET_MATRIX_OFFSET + button * TARGET_COUNT + target_slot
                    )
                    blocked_targets += int(result[actor_index, offset] > 0.5)
                    result[actor_index, offset] = 0.0

            target_matrix = result[
                actor_index,
                TARGET_MATRIX_OFFSET:
            ].reshape(
                FrozenV1Config.HERO_ACTION_HEADS[0],
                TARGET_COUNT,
            )
            button_mask = result[
                actor_index,
                : FrozenV1Config.HERO_ACTION_HEADS[0],
            ]
            empty_target_rows = np.logical_not(
                np.any(target_matrix > 0.5, axis=1)
            )
            button_mask[empty_target_rows] = 0.0

            displacement_blocked = False
            if int(config_id) in DISPLACEMENT_SKILL_CONFIG_IDS and position is not None:
                within_spawn_belt = (
                    float(position[0])
                    >= boundary - UNCERTAIN_DISPLACEMENT_DISTANCE
                )
                if (
                    within_spawn_belt or critical
                ) and result[actor_index, DISPLACEMENT_SKILL_BUTTON] > 0.5:
                    # The current protocol does not expose the true landing of
                    # these skill-2 displacements before sampling.  The approved
                    # conservative rule therefore masks the sub-action in the
                    # boundary/critical safety belt.
                    result[actor_index, DISPLACEMENT_SKILL_BUTTON] = 0.0
                    displacement_blocked = True

            # A target safety edit can invalidate the only row behind an
            # otherwise legal button.  Never let that malformed row reach the
            # sampler.  If the upstream platform mask was itself globally
            # malformed, recover to the protocol no-op button/target pair.
            legal_buttons = result[
                actor_index,
                : FrozenV1Config.HERO_ACTION_HEADS[0],
            ]
            if not np.any(legal_buttons > 0.5):
                target_matrix[0].fill(0.0)
                target_matrix[0, 0] = 1.0
                legal_buttons[0] = 1.0
                self.all_illegal_recovery_count += 1

            self.spawn_move_block_count += blocked_spawn
            self.spawn_target_block_count += blocked_targets
            self.spawn_displacement_block_count += int(
                displacement_blocked
                and position is not None
                and float(position[0])
                >= boundary - UNCERTAIN_DISPLACEMENT_DISTANCE
            )
            self.tower_critical_move_block_count += blocked_tower
            self.tower_critical_displacement_block_count += int(
                displacement_blocked and critical
            )
            diagnostics.append(
                {
                    "runtime_id": int(plan.get("runtime_id", 0)),
                    "spawn_boundary_x": boundary,
                    "over_boundary": over_boundary,
                    "spawn_move_blocks": blocked_spawn,
                    "spawn_target_blocks": blocked_targets,
                    "tower_critical": critical,
                    "tower_move_blocks": blocked_tower,
                    "displacement_blocked": displacement_blocked,
                    "has_legal_move": bool(np.any(move_mask > 0.5)),
                }
            )
        return result, tuple(diagnostics)


__all__ = (
    "DISPLACEMENT_SKILL_BUTTON",
    "DISPLACEMENT_SKILL_CONFIG_IDS",
    "SafetyLegalizerV1",
    "UNCERTAIN_DISPLACEMENT_DISTANCE",
)
