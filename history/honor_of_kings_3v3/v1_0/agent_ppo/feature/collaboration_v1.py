#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""Frozen v1 command legality, semantic masks, and protocol legalization."""

from dataclasses import dataclass
import math

import numpy as np

from agent_ppo.conf.conf import FrozenV1Config
from agent_ppo.feature import schema_v1


TARGET_INTENTS = frozenset(schema_v1.TARGET_ACTIVE_COMMAND_INTENTS)
POSITION_INTENTS = frozenset(schema_v1.POSITION_ACTIVE_COMMAND_INTENTS)
OBJECTIVE_TARGET_SLOTS = tuple(
    8 + index
    for index, (config_id, _relation) in enumerate(
        schema_v1.MONSTER_TARGET_CONFIG_AND_SIDE
    )
    if config_id in (49, 44, 59, 122)
)
LANE_NEUTRAL_OBJECTIVE_TARGET_SLOTS = tuple(
    8 + index
    for index, (config_id, _relation) in enumerate(
        schema_v1.MONSTER_TARGET_CONFIG_AND_SIDE
    )
    if config_id in (59, 122)
)
JUNGLE_TARGET_SLOTS = tuple(range(8, 28))


@dataclass(frozen=True)
class CommandRuntimeContextV1:
    camp: int
    frame_no: int
    state_valid: bool
    official_runtime_id: int
    official_config_id: int
    official_alive: bool
    team_emergency: bool
    cooldown_ready: bool
    current_command_active: bool
    unset_allowed: bool
    unset_reason: str
    target_runtime_ids: tuple
    collaboration_target_legal: tuple
    position_raw_coordinates: tuple
    position_valid: tuple
    jungle_target_legal: tuple
    jungle_position_legal: tuple
    jungle_route_active: bool = False
    jungle_route_phase: str = ""
    jungle_route_interrupt: str = "none"
    jungle_route_target_slot: int = -1
    jungle_route_position_slot: int = -1
    jungle_route_intruder_runtime: int = 0
    jungle_lane_target_slot: int = -1
    jungle_route_mode: str = "farm"
    jungle_camp_exit_active: bool = False
    jungle_opening_rotation_active: bool = False
    jungle_group_position_slot: int = -1
    jungle_defense_position_slot: int = 7
    lane_route_active: bool = False
    lane_route_phase: str = "inactive"
    lane_route_interrupt: str = "none"
    lane_route_target_slot: int = -1
    lane_route_position_slot: int = 5
    lane_route_intruder_runtime: int = 0
    lane_jungle_unlocked: bool = False

    def __post_init__(self):
        if self.camp not in (1, 2):
            raise ValueError(f"invalid command context camp: {self.camp}")
        if len(self.target_runtime_ids) != FrozenV1Config.COMMAND_HEADS[1]:
            raise ValueError("command target runtime map must contain 39 slots")
        if len(self.collaboration_target_legal) != FrozenV1Config.COMMAND_HEADS[1]:
            raise ValueError("command collaboration target mask must contain 39 slots")
        if len(self.position_raw_coordinates) != FrozenV1Config.COMMAND_HEADS[2]:
            raise ValueError("command position bank size drifted")
        if len(self.position_valid) != FrozenV1Config.COMMAND_HEADS[2]:
            raise ValueError("command position validity size drifted")
        if len(self.jungle_target_legal) != 20:
            raise ValueError("jungle target mask must contain 20 slots")
        if len(self.jungle_position_legal) != 20:
            raise ValueError("jungle position mask must contain 20 slots")


@dataclass(frozen=True)
class CommandLegalMasksV1:
    intent: np.ndarray
    target_by_intent: np.ndarray
    position_by_intent: np.ndarray
    command_is_train: bool

    def __post_init__(self):
        if self.intent.shape != (FrozenV1Config.COMMAND_HEADS[0],):
            raise ValueError(f"intent legal mask drifted: {self.intent.shape}")
        if self.target_by_intent.shape != (
            FrozenV1Config.COMMAND_HEADS[0], FrozenV1Config.COMMAND_HEADS[1]
        ):
            raise ValueError(f"target legal mask drifted: {self.target_by_intent.shape}")
        if self.position_by_intent.shape != (
            FrozenV1Config.COMMAND_HEADS[0], FrozenV1Config.COMMAND_HEADS[2]
        ):
            raise ValueError(f"position legal mask drifted: {self.position_by_intent.shape}")

    def conditioned(self, intent_action):
        intent_action = int(intent_action)
        if not 0 <= intent_action < FrozenV1Config.COMMAND_HEADS[0]:
            raise ValueError(f"command intent out of range: {intent_action}")
        result = np.concatenate(
            (
                self.intent,
                self.target_by_intent[intent_action],
                self.position_by_intent[intent_action],
            )
        ).astype(np.float32, copy=False)
        if result.shape != (FrozenV1Config.COMMAND_LEGAL_MASK_DIM,):
            raise RuntimeError(f"conditioned command legal mask drifted: {result.shape}")
        return result


@dataclass(frozen=True)
class LegalizedCommandV1:
    action: tuple
    valid: bool
    reason: str
    intent: int
    target_slot: int
    position_slot: int


def build_command_legal_masks_v1(context):
    if not isinstance(context, CommandRuntimeContextV1):
        raise TypeError("build_command_legal_masks_v1 requires CommandRuntimeContextV1")
    intent = np.zeros(FrozenV1Config.COMMAND_HEADS[0], dtype=np.float32)
    target = np.zeros(
        (FrozenV1Config.COMMAND_HEADS[0], FrozenV1Config.COMMAND_HEADS[1]),
        dtype=np.float32,
    )
    position = np.zeros(
        (FrozenV1Config.COMMAND_HEADS[0], FrozenV1Config.COMMAND_HEADS[2]),
        dtype=np.float32,
    )
    intent[0] = 1.0

    collaboration_legal = np.asarray(
        context.collaboration_target_legal, dtype=np.float32
    )
    base_position = np.asarray(context.position_valid, dtype=np.float32)
    protocol_ready = bool(
        context.state_valid and context.official_alive and context.cooldown_ready
    )

    if (
        context.state_valid
        and context.current_command_active
        and (context.unset_allowed or not context.official_alive)
    ):
        intent[1] = 1.0  # Unset is not constrained by ordinary cooldown/duration.

    target[2, 1:4] = collaboration_legal[1:4]
    target[3, 38] = collaboration_legal[38]
    target[4, list(OBJECTIVE_TARGET_SLOTS)] = collaboration_legal[
        list(OBJECTIVE_TARGET_SLOTS)
    ]
    target[9, list(JUNGLE_TARGET_SLOTS)] = np.asarray(
        context.jungle_target_legal, dtype=np.float32
    )

    for command_intent, legal_slots in schema_v1.COMMAND_POSITION_LEGAL_BY_INTENT.items():
        for slot in legal_slots:
            position[command_intent, slot] = base_position[slot]
    position[10, 10:30] *= np.asarray(
        context.jungle_position_legal, dtype=np.float32
    )

    # Kai's macro plan constrains the official-agent command interface. The
    # controlled-Kai counterpart uses the ordinary-action legalizer through
    # the first lower engagement. Recovery, defense, invasion response and
    # Tyrant contest replace the route for the current legal command window.
    if context.jungle_route_active and context.official_config_id == 193:
        target.fill(0)
        position.fill(0)
        interrupt = context.jungle_route_interrupt
        if interrupt == "contest_enemy":
            for slot, runtime_id in enumerate(context.target_runtime_ids):
                if runtime_id == context.jungle_route_intruder_runtime:
                    target[2, slot] = collaboration_legal[slot]
        elif interrupt in ("retreat", "recover"):
            safe_slot = 9 if interrupt == "recover" else 7
            position[8, safe_slot] = base_position[safe_slot]
        elif interrupt == "structure_defense":
            slot = context.jungle_defense_position_slot
            if 0 <= slot < FrozenV1Config.COMMAND_HEADS[2]:
                position[7, slot] = base_position[slot]
        elif context.jungle_route_phase == "lane_soak_wave":
            position[5, 5] = base_position[5]
        elif context.jungle_route_mode == "push":
            target[3, 38] = collaboration_legal[38]
            if not target[3, 38]:
                position[5, 5] = base_position[5]
        elif context.jungle_route_mode == "group":
            slot = context.jungle_group_position_slot
            if 0 <= slot < FrozenV1Config.COMMAND_HEADS[2]:
                position[6, slot] = base_position[slot]
        else:
            target_slot = context.jungle_route_target_slot
            position_slot = context.jungle_route_position_slot
            ordinary_opening_camp = bool(
                context.jungle_route_phase in ("upper_deer", "upper_wolf")
                or (
                    context.jungle_opening_rotation_active
                    and context.jungle_route_phase == "lower_clear"
                )
            )
            # The official platform rejected every observed SetTarget for
            # ordinary configs 42/43/44/45 while accepting SetPosition and
            # non-monster targets. Keep the official Kai under an accepted
            # camp-anchor command and let its built-in micro attack nearby
            # monsters. Tyrant and post-opening learned objectives retain
            # their target semantics.
            if (
                not ordinary_opening_camp
                and 0 <= target_slot < FrozenV1Config.COMMAND_HEADS[1]
            ):
                target[4 if target_slot == 27 else 9, target_slot] = collaboration_legal[target_slot]
            if 0 <= position_slot < FrozenV1Config.COMMAND_HEADS[2]:
                position[10, position_slot] = base_position[position_slot]
    elif context.lane_route_active and context.official_config_id in (132, 141):
        target.fill(0)
        position.fill(0)
        interrupt = context.lane_route_interrupt
        if interrupt in ("enemy_hero", "recent_hit"):
            for slot, runtime_id in enumerate(context.target_runtime_ids):
                if runtime_id == context.lane_route_intruder_runtime:
                    target[2, slot] = collaboration_legal[slot]
        elif interrupt == "low_hp":
            position[8, 9] = base_position[9]
        elif interrupt == "structure_defense":
            position[7, 6] = base_position[6]
        elif interrupt != "dead":
            # The learned semantic remains push_lane.  Far from the wave it
            # resolves to SetPosition(slot 5); near a legal first-wave soldier
            # the rule legalizer resolves the same intent to SetTarget.
            position[5, context.lane_route_position_slot] = base_position[
                context.lane_route_position_slot
            ]

    if (
        context.official_config_id in (132, 141)
        and not context.lane_jungle_unlocked
    ):
        # Marco and Diao Chan are lane roles while the enemy outer tower is
        # alive. They may still group for Tyrant or another neutral objective,
        # but cannot be sent alone to a jungle camp by intents 9/10. This gate
        # also applies after the first-wave command guard has ended.
        target[9].fill(0.0)
        position[10].fill(0.0)
        blocked_objective_slots = set(OBJECTIVE_TARGET_SLOTS) - set(
            LANE_NEUTRAL_OBJECTIVE_TARGET_SLOTS
        )
        if blocked_objective_slots:
            target[4, sorted(blocked_objective_slots)] = 0.0

    forced_opening_control = bool(
        (
            context.lane_route_active
            and context.official_config_id in (132, 141)
            and context.lane_route_interrupt
            in ("none", "enemy_hero", "recent_hit", "structure_defense")
        )
        or (
            context.jungle_route_active
            and context.official_config_id == 193
            and (
                context.jungle_route_phase in ("upper_deer", "upper_wolf")
                or (
                    context.jungle_opening_rotation_active
                    and context.jungle_route_phase
                    in ("lane_soak_wave", "lower_clear")
                )
                or context.jungle_camp_exit_active
            )
            and context.jungle_route_interrupt == "none"
        )
    )
    if context.jungle_camp_exit_active and context.official_config_id == 193:
        # Preserve the macro's next-camp waypoint. Slot 5 is the lane clash,
        # not a valid generic jungle departure destination.
        target.fill(0)
    if forced_opening_control:
        # During the protected opening, Unset would immediately return the
        # official hero toward spring. Enemy contact and pre-wave structure
        # defense are transient commands and must not make cancellation legal.
        intent[1] = 0.0
    if (
        forced_opening_control
        and context.official_config_id == 193
        and not context.jungle_camp_exit_active
        and any(np.any(target[index] > 0.5) for index in (4, 9))
    ):
        # At an opening camp, attacking the real monster has priority over
        # repeatedly refreshing the already reached position waypoint.
        position.fill(0)

    if protocol_ready:
        for command_intent in TARGET_INTENTS:
            intent[command_intent] = float(bool(np.any(target[command_intent] > 0.5)))
        for command_intent in POSITION_INTENTS:
            intent[command_intent] = float(
                bool(np.any(position[command_intent] > 0.5))
            )
        # Keep is always a real PPO choice. Cancel is exposed only by the
        # safety gate. A confirmed opening route with no active command may
        # force a bootstrap command so the official hero does not walk back.
        forced_opening_bootstrap = bool(
            not context.current_command_active
            and (context.jungle_route_active or context.lane_route_active)
            and np.any(intent[2:] > 0.5)
        )
        if forced_opening_control and np.any(intent[2:] > 0.5):
            # Opening deployment is confirmed rule behavior.  Do not let an
            # untrained command policy cancel it or keep an obsolete waypoint
            # once the protocol cooldown permits a refresh.
            intent[0] = 0.0
            intent[1] = 0.0
        elif forced_opening_bootstrap:
            intent[0] = 0.0
    else:
        target.fill(0)
        position.fill(0)
        if forced_opening_control and context.current_command_active:
            # No-command preserves the active SetPosition/SetTarget command;
            # Unset would send the official hero back toward its spring.
            intent.fill(0)
            intent[0] = 1.0

    # Forced singleton actions are executable but contain no policy decision.
    command_is_train = bool(np.count_nonzero(intent > 0.5) >= 2)
    return CommandLegalMasksV1(intent, target, position, command_is_train)


class CommandRuleLegalizerV1:
    """Map a learned semantic action to the official four-field protocol."""

    VERSION = FrozenV1Config.LEGALIZER_VERSION

    def legalize(self, intent, target_slot, position_slot, context, masks=None):
        if not isinstance(context, CommandRuntimeContextV1):
            raise TypeError("CommandRuleLegalizerV1 requires CommandRuntimeContextV1")
        masks = masks or build_command_legal_masks_v1(context)
        intent = int(intent)
        target_slot = int(target_slot)
        position_slot = int(position_slot)

        def invalid(reason):
            return LegalizedCommandV1(
                (0, 0, 0, 0), False, reason, intent, target_slot, position_slot
            )

        if not 0 <= intent < FrozenV1Config.COMMAND_HEADS[0] or masks.intent[intent] <= 0.5:
            return invalid("intent_not_legal")
        if intent == 0:
            return LegalizedCommandV1(
                (0, 0, 0, 0), True, "no_command", intent, 0, 0
            )
        if intent == 1:
            return LegalizedCommandV1((3, 0, 0, 0), True, "unset", intent, 0, 0)
        jungle_lane_target = bool(
            intent == 5
            and context.jungle_route_active
            and context.jungle_route_phase == "lane_soak_wave"
            and context.official_config_id == 193
            and 28 <= context.jungle_lane_target_slot <= 37
        )
        if (
            intent == 5
            and (
                (
                    context.lane_route_active
                    and context.official_config_id in (132, 141)
                    and 28 <= context.lane_route_target_slot <= 37
                )
                or jungle_lane_target
            )
        ):
            lane_target_slot = int(
                context.jungle_lane_target_slot
                if jungle_lane_target
                else context.lane_route_target_slot
            )
            # The soldier can die or leave vision between feature construction
            # and protocol legalization. Intent 5 is also a position intent, so
            # a stale target falls through to the legal position command.
            if context.collaboration_target_legal[lane_target_slot]:
                runtime_id = int(context.target_runtime_ids[lane_target_slot])
                if runtime_id > 0:
                    return LegalizedCommandV1(
                        (1, runtime_id, 0, 0),
                        True,
                        "push_lane_set_target",
                        intent,
                        lane_target_slot,
                        0,
                    )
        if intent in TARGET_INTENTS:
            if not 0 <= target_slot < 39:
                return invalid("target_slot_out_of_range")
            if masks.target_by_intent[intent, target_slot] <= 0.5:
                return invalid("target_slot_not_legal_for_intent")
            runtime_id = int(context.target_runtime_ids[target_slot])
            if runtime_id <= 0:
                return invalid("target_runtime_missing")
            return LegalizedCommandV1(
                (1, runtime_id, 0, 0),
                True,
                "set_target",
                intent,
                target_slot,
                0,
            )
        if intent in POSITION_INTENTS:
            if not 0 <= position_slot < FrozenV1Config.COMMAND_HEADS[2]:
                return invalid("position_slot_out_of_range")
            if masks.position_by_intent[intent, position_slot] <= 0.5:
                return invalid("position_slot_not_legal_for_intent")
            coordinate = context.position_raw_coordinates[position_slot]
            if coordinate is None or len(coordinate) != 2:
                return invalid("position_coordinate_missing")
            x, z = float(coordinate[0]), float(coordinate[1])
            if not math.isfinite(x) or not math.isfinite(z) or abs(x) >= 90000 or abs(z) >= 90000:
                return invalid("position_coordinate_invalid")
            return LegalizedCommandV1(
                (2, 0, int(round(x)), int(round(z))),
                True,
                "set_position",
                intent,
                0,
                position_slot,
            )
        return invalid("unhandled_intent")


__all__ = (
    "CommandLegalMasksV1",
    "CommandRuleLegalizerV1",
    "CommandRuntimeContextV1",
    "LegalizedCommandV1",
    "OBJECTIVE_TARGET_SLOTS",
    "JUNGLE_TARGET_SLOTS",
    "LANE_NEUTRAL_OBJECTIVE_TARGET_SLOTS",
    "POSITION_INTENTS",
    "TARGET_INTENTS",
    "build_command_legal_masks_v1",
)
