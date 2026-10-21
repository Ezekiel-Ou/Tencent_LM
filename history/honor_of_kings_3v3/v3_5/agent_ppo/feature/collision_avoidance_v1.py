#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""Channel-4 movement legality and PPO-consistent active unsticking."""

from dataclasses import dataclass
from functools import lru_cache
import math

import numpy as np

from agent_ppo.conf.conf import FrozenV1Config
from agent_ppo.feature.feature_process import v1_utils as u


MOVE_BUTTON = 2
MOVE_HEAD_OFFSET = FrozenV1Config.HERO_ACTION_HEADS[0]
MOVE_DIRECTION_COUNT = FrozenV1Config.HERO_ACTION_HEADS[1]
IMAGE_CENTER = 8
OBSTACLE_CHANNEL = 4
OBSTACLE_THRESHOLD = 0.5
RAY_RADIUS = 3
STUCK_DISPLACEMENT = 120.0
PROGRESS_DISPLACEMENT = 300.0
STUCK_TRIGGER_FRAMES = 24
ESCAPE_DURATION_FRAMES = 18
MACRO_ESCAPE_COOLDOWN_FRAMES = 60
COMBAT_CANCEL_DISTANCE = 12000.0


def move_direction_vector(move_index):
    """Return the calibrated camp-normalized `(side_dx, dz)` unit vector."""

    move_index = int(move_index)
    if move_index == 0:
        return (0.0, 0.0)
    if not 1 <= move_index <= 24:
        raise ValueError(f"move index out of range: {move_index}")
    angle = math.radians((180.0 - 15.0 * (move_index - 1)) % 360.0)
    return (math.cos(angle), math.sin(angle))


@lru_cache(maxsize=None)
def _ray_cells(move_index, radius=RAY_RADIUS):
    """Return immutable grid rays cached by direction and radius."""

    dx, dz = move_direction_vector(move_index)
    perpendicular_x, perpendicular_z = -dz, dx
    cells = []
    for step in range(1, int(radius) + 1):
        lateral_offsets = (-0.55, 0.0, 0.55) if step <= 2 else (0.0,)
        step_cells = []
        for lateral in lateral_offsets:
            local_x = dx * step + perpendicular_x * lateral
            local_z = dz * step + perpendicular_z * lateral
            row = IMAGE_CENTER - int(round(local_z))
            column = IMAGE_CENTER + int(round(local_x))
            if 0 <= row < 17 and 0 <= column < 17:
                cell = (row, column)
                if cell not in step_cells:
                    step_cells.append(cell)
        cells.append(tuple(step_cells))
    return tuple(cells)


def direction_clearance(obstacle, move_index):
    obstacle = np.asarray(obstacle, dtype=np.float32)
    if obstacle.shape != (17, 17):
        raise ValueError(f"obstacle channel must be [17,17], got {obstacle.shape}")
    weights = (1.0, 0.6, 0.25)
    score = 0.0
    weight_sum = 0.0
    for step, cells in enumerate(_ray_cells(move_index)):
        for row, column in cells:
            weight = weights[step]
            score += weight * (1.0 - float(obstacle[row, column] > OBSTACLE_THRESHOLD))
            weight_sum += weight
    return score / max(weight_sum, 1e-6)


def obstacle_move_mask(obstacle, official_move_mask):
    """Remove directions whose near-field corridor intersects channel 4."""

    obstacle = np.asarray(obstacle, dtype=np.float32)
    official = np.asarray(official_move_mask, dtype=np.float32).reshape(-1)
    if obstacle.shape != (17, 17) or official.shape != (MOVE_DIRECTION_COUNT,):
        raise ValueError("collision mask requires obstacle [17,17] and move mask [25]")
    result = (official > 0.5).astype(np.float32)
    if not np.isfinite(obstacle).all():
        return result, tuple()
    blocked = []
    for move_index in range(1, MOVE_DIRECTION_COUNT):
        if result[move_index] <= 0.5:
            continue
        ray = _ray_cells(move_index)
        immediate = [obstacle[row, column] for row, column in ray[0]]
        second = [obstacle[row, column] for row, column in ray[1]]
        immediate_blocked = any(value > OBSTACLE_THRESHOLD for value in immediate)
        second_blocked = bool(second) and sum(
            value > OBSTACLE_THRESHOLD for value in second
        ) >= max(1, len(second) - 1)
        if immediate_blocked or second_blocked:
            result[move_index] = 0.0
            blocked.append(move_index)
    if not np.any(result > 0.5):
        # Fail closed to a stationary action instead of producing an all-zero
        # PPO mask. If stationary is unavailable, retain the clearest official
        # direction so action sampling remains defined.
        if official[0] > 0.5:
            result[0] = 1.0
        else:
            legal_directions = [index for index in range(1, 25) if official[index] > 0.5]
            if legal_directions:
                best = max(
                    legal_directions,
                    key=lambda index: direction_clearance(obstacle, index),
                )
                result[best] = 1.0
                if best in blocked:
                    blocked.remove(best)
    return result, tuple(blocked)


@dataclass
class _HeroCollisionState:
    runtime_id: int
    last_frame: int = None
    last_position: tuple = None
    action_frame: int = None
    last_button: int = 0
    last_move: int = 0
    stagnant_frames: int = 0
    escape_until_frame: int = -1
    escape_move: int = 0
    escape_turn: int = 1
    escape_source: str = "none"
    macro_escape_cooldown_until: int = -1


class CollisionAvoidanceV1:
    """Stateful pre-sampling legalizer shared by actor and learner samples."""

    VERSION = FrozenV1Config.ORDINARY_ACTION_LEGALIZER_VERSION

    def __init__(self, camp):
        self.reset(camp)

    def reset(self, camp=None):
        if camp is not None:
            self.camp = u.normalize_camp(camp)
        if self.camp not in (1, 2):
            raise ValueError(f"invalid collision-avoidance camp: {self.camp}")
        self.states = {}
        self.escape_count = 0
        self.masked_direction_total = 0
        self.mask_application_count = 0
        self.last_diagnostics = ()

    @staticmethod
    def _controlled_actor_by_runtime(observation):
        return {
            u.actor_runtime_id(actor): actor
            for actor in observation.get("frame_state", {}).get("hero_states", [])
            if u.actor_runtime_id(actor)
        }

    def _combat_nearby(self, observation, actor, position):
        if actor is None or position is None:
            return False
        for other in observation.get("frame_state", {}).get("hero_states", []):
            if u.actor_camp(other) in (0, self.camp) or not u.is_alive(other):
                continue
            if not u.visible_to_camp(other, self.camp, is_friend=False):
                continue
            other_position = u.location_of(other)
            if other_position is not None and u.distance(position, other_position) <= COMBAT_CANCEL_DISTANCE:
                return True
        return False

    @staticmethod
    def _choose_escape_move(state, obstacle, official_move, collision_move):
        blocked_move = state.last_move if 1 <= state.last_move <= 24 else 1
        sign = state.escape_turn
        offsets = (6 * sign, -6 * sign, 8 * sign, -8 * sign, 12)
        candidates = []
        for offset in offsets:
            move_index = ((blocked_move - 1 + offset) % 24) + 1
            if collision_move[move_index] > 0.5:
                candidates.append(move_index)
        if not candidates:
            reverse = ((blocked_move - 1 + 12) % 24) + 1
            if official_move[reverse] > 0.5:
                candidates.append(reverse)
        if not candidates:
            candidates = [
                index for index in range(1, 25) if official_move[index] > 0.5
            ]
        if not candidates:
            return 0
        rank = {move_index: index for index, move_index in enumerate(candidates)}
        return max(
            candidates,
            key=lambda index: (
                direction_clearance(obstacle, index),
                -rank[index],
            ),
        )

    def apply(
        self,
        observation,
        hero_infos,
        raw_legal_action,
        force_escape_runtime_ids=(),
    ):
        result = np.asarray(raw_legal_action, dtype=np.float32).copy()
        if result.shape != (
            FrozenV1Config.CONTROLLED_ACTOR_COUNT,
            FrozenV1Config.RAW_LEGAL_ACTION_DIM,
        ):
            raise ValueError(f"invalid collision raw legal shape: {result.shape}")
        frame_state = observation.get("frame_state", {}) or {}
        frame_no = u.frame_number(frame_state)
        actor_by_runtime = self._controlled_actor_by_runtime(observation)
        diagnostics = []
        active_runtime_ids = set()
        forced_escape_ids = {int(value) for value in force_escape_runtime_ids}

        for actor_index, hero_info in enumerate(hero_infos):
            runtime_id = u.as_int(u.safe_get(hero_info, "runtime_id", default=0), 0)
            active_runtime_ids.add(runtime_id)
            actor = actor_by_runtime.get(runtime_id)
            position = u.location_of(actor) if actor else None
            alive = bool(actor) and u.is_alive(actor)
            state = self.states.get(runtime_id)
            if state is None:
                state = _HeroCollisionState(runtime_id=runtime_id)
                self.states[runtime_id] = state

            image = np.asarray(
                u.safe_get(hero_info, "feature", default=[]), dtype=np.float32
            ).reshape(-1)
            obstacle = (
                image.reshape(FrozenV1Config.IMAGE_SHAPE)[OBSTACLE_CHANNEL]
                if image.size == FrozenV1Config.IMAGE_FEATURE_DIM
                else np.zeros((17, 17), dtype=np.float32)
            )
            official_move = result[
                actor_index,
                MOVE_HEAD_OFFSET : MOVE_HEAD_OFFSET + MOVE_DIRECTION_COUNT,
            ].copy()
            collision_move, blocked = obstacle_move_mask(obstacle, official_move)

            displacement = float("inf")
            elapsed = 0
            same_observation = state.last_frame == frame_no
            if not same_observation:
                self.masked_direction_total += len(blocked)
                self.mask_application_count += 1
            if not same_observation and state.last_frame is not None:
                elapsed = frame_no - state.last_frame
                displacement_value = u.distance(position, state.last_position)
                displacement = (
                    float("inf") if displacement_value is None else displacement_value
                )
                attempted_move = bool(
                    state.action_frame == state.last_frame
                    and state.last_button == MOVE_BUTTON
                    and state.last_move > 0
                )
                if attempted_move and 0 < elapsed <= 60 and displacement < STUCK_DISPLACEMENT:
                    state.stagnant_frames += elapsed
                elif displacement >= PROGRESS_DISPLACEMENT or not attempted_move:
                    state.stagnant_frames = 0

            combat_nearby = self._combat_nearby(observation, actor, position)
            if not alive or combat_nearby or (
                displacement >= PROGRESS_DISPLACEMENT
                and state.escape_source != "macro"
            ):
                state.escape_until_frame = -1
                state.escape_move = 0
                state.escape_source = "none"
                if not alive or combat_nearby:
                    state.stagnant_frames = 0

            if frame_no > state.escape_until_frame:
                state.escape_move = 0
                state.escape_source = "none"

            escaping = bool(alive and frame_no <= state.escape_until_frame and state.escape_move > 0)
            activated = False
            macro_stuck_requested = bool(
                runtime_id in forced_escape_ids and alive and not combat_nearby
            )
            macro_stuck_forced = bool(
                macro_stuck_requested
                and frame_no >= state.macro_escape_cooldown_until
            )
            if (
                alive
                and not combat_nearby
                and not escaping
                and (
                    state.stagnant_frames >= STUCK_TRIGGER_FRAMES
                    or macro_stuck_forced
                )
            ):
                state.escape_turn *= -1
                state.escape_move = self._choose_escape_move(
                    state, obstacle, official_move, collision_move
                )
                if state.escape_move > 0:
                    state.escape_until_frame = frame_no + ESCAPE_DURATION_FRAMES
                    state.escape_source = (
                        "macro" if macro_stuck_forced else "local"
                    )
                    if macro_stuck_forced:
                        state.macro_escape_cooldown_until = (
                            frame_no + MACRO_ESCAPE_COOLDOWN_FRAMES
                        )
                    state.stagnant_frames = 0
                    escaping = True
                    activated = True
                    self.escape_count += 1

            if escaping and collision_move[state.escape_move] <= 0.5:
                state.escape_move = self._choose_escape_move(
                    state, obstacle, official_move, collision_move
                )
                escaping = state.escape_move > 0

            result[
                actor_index,
                MOVE_HEAD_OFFSET : MOVE_HEAD_OFFSET + MOVE_DIRECTION_COUNT,
            ] = collision_move
            if escaping and official_move[state.escape_move] > 0.5 and result[actor_index, MOVE_BUTTON] > 0.5:
                result[actor_index, : FrozenV1Config.HERO_ACTION_HEADS[0]] = 0.0
                result[actor_index, MOVE_BUTTON] = 1.0
                result[
                    actor_index,
                    MOVE_HEAD_OFFSET : MOVE_HEAD_OFFSET + MOVE_DIRECTION_COUNT,
                ] = 0.0
                result[actor_index, MOVE_HEAD_OFFSET + state.escape_move] = 1.0
            else:
                escaping = False

            if not same_observation:
                state.last_frame = frame_no
                state.last_position = position
            diagnostics.append(
                {
                    "runtime_id": runtime_id,
                    "blocked_direction_count": len(blocked),
                    "collision_mask_active": bool(blocked),
                    "stagnant_frames": state.stagnant_frames,
                    "escape_active": escaping,
                    "escape_activated": activated,
                    "escape_move": state.escape_move if escaping else 0,
                    "escape_source": state.escape_source if escaping else "none",
                    "combat_cancelled": combat_nearby,
                    "macro_stuck_forced": macro_stuck_forced,
                    "macro_stuck_requested": macro_stuck_requested,
                    "macro_escape_cooldown_until": state.macro_escape_cooldown_until,
                }
            )

        for runtime_id in list(self.states):
            if runtime_id not in active_runtime_ids:
                self.states.pop(runtime_id, None)
        self.last_diagnostics = tuple(diagnostics)
        return result, self.last_diagnostics

    def record_actions(self, frame_no, runtime_ids, actions):
        actions = np.asarray(actions, dtype=np.int64)
        if actions.shape != (FrozenV1Config.CONTROLLED_ACTOR_COUNT, 5):
            raise ValueError(f"collision action record must be [2,5], got {actions.shape}")
        for runtime_id, action in zip(runtime_ids, actions):
            state = self.states.get(int(runtime_id))
            if state is None:
                state = _HeroCollisionState(runtime_id=int(runtime_id))
                self.states[int(runtime_id)] = state
            state.action_frame = int(frame_no)
            state.last_button = int(action[0])
            state.last_move = int(action[1])


COLLISION_AVOIDANCE_SIGNATURE = (
    ("obstacle_channel", OBSTACLE_CHANNEL),
    ("ray_radius", RAY_RADIUS),
    ("stuck_displacement", STUCK_DISPLACEMENT),
    ("progress_displacement", PROGRESS_DISPLACEMENT),
    ("stuck_trigger_frames", STUCK_TRIGGER_FRAMES),
    ("escape_duration_frames", ESCAPE_DURATION_FRAMES),
    ("macro_escape_cooldown_frames", MACRO_ESCAPE_COOLDOWN_FRAMES),
    ("combat_cancel_distance", COMBAT_CANCEL_DISTANCE),
    ("macro_route_stuck_force", True),
)


__all__ = (
    "COLLISION_AVOIDANCE_SIGNATURE",
    "CollisionAvoidanceV1",
    "direction_clearance",
    "move_direction_vector",
    "obstacle_move_mask",
)
