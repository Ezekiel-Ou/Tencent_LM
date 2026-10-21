#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""PPO 4.0 pre-sampling ordinary-action legalizers.

All edits happen on the serialized legal mask before actor sampling.  The
executed action and the PPO old probability therefore always describe the
same conditioned action space.
"""

import numpy as np

from agent_ppo.conf.conf import FrozenV1Config, V1TrainingConfig
from agent_ppo.feature.collision_avoidance_v1 import (
    MOVE_BUTTON,
    MOVE_DIRECTION_COUNT,
    MOVE_HEAD_OFFSET,
)
from agent_ppo.feature.lane_opening_v1 import (
    COMMON_ATTACK_BUTTON,
    TARGET_COUNT,
    TARGET_MATRIX_OFFSET,
    _ranked_move_directions,
)


KAI_CONFIG_ID = 193
SMITE_BUTTON = 8
TYRANT_TARGET_SLOT = 27
ORDINARY_MONSTER_TARGET_START = 8
ORDINARY_MONSTER_TARGET_STOP = 27


def _validated_copy(raw_legal_action, plans, name):
    result = np.asarray(raw_legal_action, dtype=np.float32).copy()
    expected = (
        FrozenV1Config.CONTROLLED_ACTOR_COUNT,
        FrozenV1Config.RAW_LEGAL_ACTION_DIM,
    )
    if result.shape != expected:
        raise ValueError(f"invalid {name} raw legal shape: {result.shape}")
    if len(plans) != expected[0]:
        raise ValueError(f"{name} plans must follow controlled actor order")
    return result


def _views(result, actor_index):
    buttons = result[
        actor_index, : FrozenV1Config.HERO_ACTION_HEADS[0]
    ]
    moves = result[
        actor_index,
        MOVE_HEAD_OFFSET : MOVE_HEAD_OFFSET + MOVE_DIRECTION_COUNT,
    ]
    targets = result[actor_index, TARGET_MATRIX_OFFSET:].reshape(
        FrozenV1Config.HERO_ACTION_HEADS[0], TARGET_COUNT
    )
    return buttons, moves, targets


def _synchronize_buttons(buttons, targets):
    empty = np.logical_not(np.any(targets > 0.5, axis=1))
    # Button 0 (no-op) and button 2 (move) do not require a semantic target.
    empty[0] = False
    empty[MOVE_BUTTON] = False
    buttons[empty] = 0.0
    if not np.any(buttons > 0.5):
        buttons[0] = 1.0
        targets[0].fill(0.0)
        targets[0, 0] = 1.0


class RespawnReentryLegalizerV1:
    """Bound a controlled Kai to the topology waypoint until engagement."""

    VERSION = FrozenV1Config.ORDINARY_ACTION_LEGALIZER_VERSION

    def __init__(self):
        self.mask_application_count = 0
        self.engagement_count = 0
        self.timer_wait_count = 0
        self.navigation_count = 0
        self.safe_hold_count = 0

    def apply(self, raw_legal_action, config_ids, plans):
        result = _validated_copy(raw_legal_action, plans, "respawn reentry")
        diagnostics = []
        for actor_index, (config_id, source) in enumerate(
            zip(config_ids, plans)
        ):
            plan = dict(source or {})
            active = bool(
                int(config_id) == KAI_CONFIG_ID
                and plan.get("hard_mask_active", False)
            )
            reason = "inactive"
            selected = ()
            target_slot = int(plan.get("target_slot", -1))
            if active:
                self.mask_application_count += 1
                buttons, moves, targets = _views(result, actor_index)
                attack_legal = bool(
                    8 <= target_slot < ORDINARY_MONSTER_TARGET_STOP
                    and buttons[COMMON_ATTACK_BUTTON] > 0.5
                    and targets[COMMON_ATTACK_BUTTON, target_slot] > 0.5
                )
                if attack_legal:
                    buttons.fill(0.0)
                    buttons[COMMON_ATTACK_BUTTON] = 1.0
                    targets[COMMON_ATTACK_BUTTON].fill(0.0)
                    targets[COMMON_ATTACK_BUTTON, target_slot] = 1.0
                    reason = "engage_camp"
                    self.engagement_count += 1
                elif plan.get("wait_at_anchor", False):
                    buttons.fill(0.0)
                    buttons[0] = 1.0
                    reason = "timer_wait"
                    self.timer_wait_count += 1
                else:
                    selected = _ranked_move_directions(
                        moves,
                        float(plan.get("waypoint_dx", 0.0)),
                        float(plan.get("waypoint_dz", 0.0)),
                    )
                    if selected and buttons[MOVE_BUTTON] > 0.5:
                        buttons.fill(0.0)
                        buttons[MOVE_BUTTON] = 1.0
                        moves.fill(0.0)
                        for direction in selected:
                            moves[int(direction)] = 1.0
                        reason = "topology_navigation"
                        self.navigation_count += 1
                    else:
                        buttons.fill(0.0)
                        buttons[0] = 1.0
                        reason = "safe_hold"
                        self.safe_hold_count += 1
                _synchronize_buttons(buttons, targets)
            diagnostics.append(
                {
                    "runtime_id": int(plan.get("runtime_id", 0)),
                    "active": active,
                    "reason": reason,
                    "target_slot": target_slot,
                    "selected_directions": tuple(int(v) for v in selected),
                }
            )
        return result, tuple(diagnostics)


class TyrantSmiteLegalizerV1:
    """Apply the confirmed enemy-Kai/Tyrant Smite competition gate."""

    VERSION = FrozenV1Config.ORDINARY_ACTION_LEGALIZER_VERSION

    def __init__(self):
        self.lock_count = 0
        self.open_count = 0
        self.force_count = 0
        self.force_pending_count = 0

    def apply(self, raw_legal_action, config_ids, plans):
        result = _validated_copy(raw_legal_action, plans, "tyrant smite")
        diagnostics = []
        for actor_index, (config_id, source) in enumerate(
            zip(config_ids, plans)
        ):
            plan = dict(source or {})
            buttons, _moves, targets = _views(result, actor_index)
            active = bool(
                int(config_id) == KAI_CONFIG_ID
                and plan.get("active", False)
            )
            state = "inactive"
            if active:
                smite_legal = bool(
                    buttons[SMITE_BUTTON] > 0.5
                    and targets[SMITE_BUTTON, TYRANT_TARGET_SLOT] > 0.5
                )
                if plan.get("force", False):
                    if smite_legal:
                        buttons.fill(0.0)
                        buttons[SMITE_BUTTON] = 1.0
                        targets[SMITE_BUTTON].fill(0.0)
                        targets[SMITE_BUTTON, TYRANT_TARGET_SLOT] = 1.0
                        state = "force"
                        self.force_count += 1
                    else:
                        state = "force_pending"
                        self.force_pending_count += 1
                elif plan.get("lock", False):
                    targets[SMITE_BUTTON, TYRANT_TARGET_SLOT] = 0.0
                    _synchronize_buttons(buttons, targets)
                    state = "lock"
                    self.lock_count += 1
                else:
                    state = "open"
                    self.open_count += 1
            diagnostics.append(
                {
                    "runtime_id": int(plan.get("runtime_id", 0)),
                    "active": active,
                    "state": state,
                }
            )
        return result, tuple(diagnostics)


class LateConversionLegalizerV1:
    """Remove late ordinary farming and optionally fan movement to conversion."""

    VERSION = FrozenV1Config.ORDINARY_ACTION_LEGALIZER_VERSION

    def __init__(self):
        self.mask_application_count = 0
        self.ordinary_targets_blocked_total = 0
        self.movement_fan_count = 0

    def apply(self, raw_legal_action, config_ids, plans):
        result = _validated_copy(raw_legal_action, plans, "late conversion")
        diagnostics = []
        for actor_index, (config_id, source) in enumerate(
            zip(config_ids, plans)
        ):
            plan = dict(source or {})
            active = bool(
                int(config_id) == KAI_CONFIG_ID
                and plan.get("active", False)
            )
            blocked = 0
            selected = ()
            if active:
                self.mask_application_count += 1
                buttons, moves, targets = _views(result, actor_index)
                self_defense_slots = {
                    int(slot)
                    for slot in plan.get("self_defense_slots", ())
                    if ORDINARY_MONSTER_TARGET_START
                    <= int(slot)
                    < ORDINARY_MONSTER_TARGET_STOP
                }
                preserved = targets[:, sorted(self_defense_slots)].copy()
                blocked = int(
                    np.count_nonzero(
                        targets[
                            :,
                            ORDINARY_MONSTER_TARGET_START:
                            ORDINARY_MONSTER_TARGET_STOP,
                        ]
                        > 0.5
                    )
                )
                self.ordinary_targets_blocked_total += blocked
                targets[
                    :,
                    ORDINARY_MONSTER_TARGET_START:
                    ORDINARY_MONSTER_TARGET_STOP,
                ] = 0.0
                if self_defense_slots:
                    targets[:, sorted(self_defense_slots)] = preserved
                if not plan.get("enemy_attacking_tyrant", False):
                    targets[:, TYRANT_TARGET_SLOT] = 0.0
                _synchronize_buttons(buttons, targets)

                if (
                    plan.get("destination_valid", False)
                    and float(plan.get("destination_distance", 0.0))
                    > V1TrainingConfig.LATE_DESTINATION_ARRIVAL_DISTANCE
                    and not plan.get("combat_target_available", False)
                    and buttons[MOVE_BUTTON] > 0.5
                ):
                    selected = _ranked_move_directions(
                        moves,
                        float(plan.get("destination_dx", 0.0)),
                        float(plan.get("destination_dz", 0.0)),
                    )[: V1TrainingConfig.LATE_MOVE_DIRECTION_FAN]
                    if selected:
                        buttons.fill(0.0)
                        buttons[MOVE_BUTTON] = 1.0
                        moves.fill(0.0)
                        for direction in selected:
                            moves[int(direction)] = 1.0
                        self.movement_fan_count += 1
            diagnostics.append(
                {
                    "runtime_id": int(plan.get("runtime_id", 0)),
                    "active": active,
                    "ordinary_targets_blocked": blocked,
                    "selected_directions": tuple(int(v) for v in selected),
                }
            )
        return result, tuple(diagnostics)


__all__ = (
    "LateConversionLegalizerV1",
    "RespawnReentryLegalizerV1",
    "TyrantSmiteLegalizerV1",
)
