#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""Feature Schema v1 builder.

The builder computes one public frame context and assembles two independent
active-contract policy observations for the controlled heroes.
"""

import copy
import math
from collections import defaultdict, deque

import numpy as np

from agent_ppo.conf.conf import FrozenV1Config, V1TrainingConfig
from agent_ppo.feature import schema_v1
from agent_ppo.feature.feature_process import v1_utils as u
from agent_ppo.feature.target_capabilities_v1 import (
    monster_set_position_allowed,
    monster_set_target_allowed,
    monster_target_type,
)


RECENT_SHORT_FRAMES = 30
RECENT_EVENT_FRAMES = 90
LAST_SEEN_FRAMES = 900
HERO_MOTION_MAX_FRAMES = 18
BULLET_MOTION_MAX_FRAMES = 12
LANE_VELOCITY_FRAMES = 30
LANE_TREND_FRAMES = 300
COMMAND_HISTORY_FRAMES = 1200
COMMAND_ACCEPTANCE_GRACE_FRAMES = 18
LANE_CONTACT_DISTANCE = 9000.0
LANE_CONTACT_ROW_TOLERANCE = 3000.0
LANE_ENEMY_HERO_INTERRUPT_DISTANCE = 12000.0
LANE_OFFICIAL_TARGET_COMMAND_DISTANCE = 26000.0
LANE_OFFICIAL_ROW_REACQUIRE_Z_DISTANCE = 5000.0
LANE_FIRST_WAVE_MISSING_GRACE_FRAMES = 30
LANE_TARGET_LOCK_GRACE_FRAMES = 30
LANE_MAX_BACKTRACK_DISTANCE = 3000.0
LANE_ROUTE_WAYPOINT_REACHED_DISTANCE = 2200.0
LANE_ROW_SIDE_Z = {132: 3150.0, 141: -3150.0}
LANE_ROW_NAMES = {132: "upper", 141: "lower"}
LANE_MONITOR_ENTRY_SIDE_X = -18000.0
LANE_MONITOR_ROW_TOLERANCE = 9000.0
LANE_MONITOR_WAVE_DISTANCE = 15000.0
LANE_OFFICIAL_FALLBACK_ENTRY_SIDE_X = -12000.0
LANE_ROUTE_WAYPOINTS = {
    # Both heroes share the measured base exit, split onto their assigned row,
    # then confirm a row-locked entry before any dynamic wave waypoint is used.
    # This prevents the official pathfinder from selecting the upper jungle
    # corridor for a long direct command to the lower wave.
    132: ((-50000.0, 8000.0), (-30000.0, 3150.0), (-18000.0, 3150.0)),
    141: ((-50000.0, 8000.0), (-30000.0, -3150.0), (-18000.0, -3150.0)),
}
JUNGLE_CAMP_EXIT_DISTANCE = 6000.0
JUNGLE_CAMP_CLEAR_NEAR_DISTANCE = 10000.0
JUNGLE_ROUTE_NODE_REACHED_DISTANCE = 2500.0
JUNGLE_ROUTE_PROGRESS_EPSILON = 300.0
JUNGLE_ROUTE_NO_PROGRESS_SECONDS = 3.0
JUNGLE_ROUTE_WRONG_WAY_SECONDS = 1.5
OFFICIAL_KAI_UPPER_ENTRY_NODE = 0
OFFICIAL_KAI_UPPER_ENTRY_REACHED_DISTANCE = 5000.0
OFFICIAL_KAI_UPPER_CONFIRMED_X_RANGE = (-32000.0, -10000.0)
OFFICIAL_KAI_UPPER_CONFIRMED_Z_MIN = 7500.0
OFFICIAL_KAI_BASE_EXIT_PASS_X = -54000.0
OFFICIAL_KAI_CORRIDOR_PASS_X = -33000.0
OFFICIAL_KAI_LOWER_ENTRY_PASS_X_RANGE = (-40000.0, 0.0)
OFFICIAL_KAI_LOWER_ENTRY_PASS_Z_MAX = 3500.0
# Side-coordinate waypoints reconstructed from the topology probes.  The
# fourth point is raw z=15000 converted to the lane-centred side coordinate.
# It is a real traversable point between the lane entrance and deer, not an
# interpolated camp anchor.
OFFICIAL_KAI_UPPER_ROUTE_WAYPOINTS = (
    (-50000.0, 8000.0),
    (-30000.0, 3500.0),
    (-25000.0, 0.0),
    (-25000.0, 15000.0 - u.LANE_CENTER_Z),
)
OFFICIAL_KAI_UPPER_ROUTE_STAGE_NAMES = (
    "upper_base_exit",
    "upper_lane_corridor",
    "upper_lane_entry",
    "upper_inner_15000",
    "upper_route_complete",
)

# Controlled Kai retains the monotonic six-camp contract.  Official Kai uses
# the same physical metadata only through upper wolf/deer and pioneer; PPO 3.5
# routes later camps through the learned farm option.  Archer camps each
# contain one config-32 major and one config-33 companion; the two camps are
# distinguished by physical anchor, not by config id.
KAI_OPENING_CAMP_ORDER = (
    ("wolf", "upper_wolf", 2, 44, (44, 45), 2),
    ("deer", "upper_deer", 1, 42, (42, 43), 2),
    ("pioneer", "lower_clear", 5, 30, (30, 31), 2),
    ("outer_archer", "lower_clear", 7, 33, (32, 33), 2),
    ("inner_archer", "lower_clear", 8, 32, (32, 33), 2),
    ("red_statue", "lower_clear", 6, 49, (49,), 1),
)
KAI_OPENING_CAMP_BY_NAME = {
    item[0]: {
        "phase": item[1],
        "goal_node": item[2],
        "anchor_config": item[3],
        "member_configs": item[4],
        "expected_members": item[5],
    }
    for item in KAI_OPENING_CAMP_ORDER
}
KAI_OPENING_WAYPOINT_REACHED_DISTANCE = 3000.0
KAI_OPENING_MONSTER_LEASH_DISTANCE = 4500.0
KAI_NEARBY_ENEMY_INTERRUPT_DISTANCE = 10000.0
KAI_OPENING_MAX_FRAME = 1800

MONSTER_RESPAWN_SECONDS = {
    30: 60.192,
    31: 60.192,
    32: 60.192,
    33: 60.192,
    42: 60.192,
    43: 60.192,
    44: 120.384,
    45: 120.384,
    49: 60.192,
    59: 60.192,
    122: 90.0,
}


def _put(vector, field_map, name, value):
    # ``vector`` is already float32, so NumPy performs the same scalar cast on
    # assignment.  Avoid constructing a separate np.float32 scalar for every
    # one of the thousands of feature fields emitted per observation.
    vector[field_map[name].start] = value


def _put_slice(vector, field_map, first_name, values):
    start = field_map[first_name].start
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    vector[start : start + values.size] = values


def _position(actor, own_camp):
    return u.side_position(u.location_of(actor), own_camp)


def _raw_legal_matrix(hero_info):
    legal = np.asarray(u.safe_get(hero_info, "legal_action", default=[]), dtype=np.float32).reshape(-1)
    if legal.size != FrozenV1Config.RAW_LEGAL_ACTION_DIM:
        return np.zeros((13, 39), dtype=np.float32), np.zeros(13, dtype=np.float32)
    fixed = legal[:122]
    return legal[122:].reshape(13, 39), fixed[:13]


def _sub_action(hero_info, button):
    masks = u.safe_get(hero_info, "sub_action_mask", default={}) or {}
    value = masks.get(str(button), masks.get(button, [0] * 5)) if isinstance(masks, dict) else [0] * 5
    result = np.asarray(value, dtype=np.float32).reshape(-1)
    return result[:5] if result.size >= 5 else np.zeros(5, dtype=np.float32)


def _value(actor, *names, default=0.0):
    return u.as_float(u.actor_get(actor, *names, default=default), default)


def _integer(actor, *names, default=0):
    return u.as_int(u.actor_get(actor, *names, default=default), default)


def _hp_ratio(actor):
    return u.ratio(_value(actor, "hp"), _value(actor, "max_hp", "maxHp"))


def _target_runtime(actor):
    return _integer(actor, "attack_target", "attackTarget")


def _distance_or_inf(first, second):
    value = u.distance(first, second)
    return float("inf") if value is None else value


def _normalized_velocity_toward(motion, origin, target):
    if not motion.get("valid") or origin is None or target is None:
        return 0.0
    dx, dz = target[0] - origin[0], target[1] - origin[1]
    length = math.hypot(dx, dz)
    if length <= 1e-6:
        return 0.0
    projection = (motion["vx"] * dx + motion["vz"] * dz) / length
    return u.signed_clip(projection / 7000.0)


def _seconds_from_ms(value):
    return max(0.0, u.as_float(value)) / 1000.0


def _forward_vector(actor, own_camp):
    forward = u.actor_get(actor, "forward", default=None)
    if not isinstance(forward, dict):
        return None
    x = u.safe_get(forward, "x", default=None)
    z = u.safe_get(forward, "z", default=None)
    if x is None or z is None:
        return None
    x, z = u.as_float(x, float("nan")), u.as_float(z, float("nan"))
    if not math.isfinite(x) or not math.isfinite(z):
        return None
    side_x = x if own_camp == 1 else -x
    length = math.hypot(side_x, z)
    return None if length <= 1e-6 else (side_x / length, z / length)


def _is_facing(actor, origin, target, own_camp):
    forward = _forward_vector(actor, own_camp)
    if forward is None or origin is None or target is None:
        return False
    dx, dz = target[0] - origin[0], target[1] - origin[1]
    length = math.hypot(dx, dz)
    return length > 1e-6 and (forward[0] * dx + forward[1] * dz) / length >= 0.5


class FeatureProcessV1:
    """Stateful, per-environment Feature Schema v1 processor."""

    def __init__(self, camp):
        self.reset(camp)

    def reset(self, camp=None):
        if camp is not None:
            self.camp = u.normalize_camp(camp)
        if self.camp not in (1, 2):
            raise ValueError(f"invalid camp for FeatureProcessV1: {self.camp}")

        self.last_frame_no = None
        self.last_seen = {}
        self.hero_motion = {}
        self.bullet_motion = {}
        self.previous_buffs = {}
        self.previous_money = {}
        self.previous_hero_counters = {}
        self.previous_reward_kda_counters = {}
        self.recent_events = defaultdict(deque)
        self.lane_history = deque()
        self.organ_hp_history = defaultdict(deque)
        self.resource_history = {}
        self.monster_camp_history = {}
        self.official_task_state = None
        self.official_task_reward_pending = 0.0
        self.official_task_rewarded_lifecycles = set()
        self.official_task_failed_lifecycles = set()
        self.official_task_counters = defaultdict(float)
        self.jungle_route_history = {
            "frame": None,
            "position": None,
            "stuck_frames": 0,
            "no_progress_frames": 0,
            "wrong_way_frames": 0,
            "best_path_distance": None,
            "goal_signature": None,
            "path_distance": None,
            "committed_route_signature": None,
            "committed_route_path": (),
            "committed_route_cursor": 0,
            "lane_decision_made": False,
            "lane_soak_active": False,
            "lane_soak_start_frame": None,
            "lane_wave_runtime_ids": set(),
            "lane_soak_exit_reason": "",
            "camp_exit_active": False,
            "camp_exit_anchor": None,
            "last_completed_camp_anchor": None,
            "last_completed_camp_frame": None,
            "lower_farm_started": False,
            "opening_stage_index": 0,
            "opening_released": False,
            "opening_release_reason": "none",
            "opening_route_key": None,
            "opening_waypoint_index": 0,
            # Official Kai's built-in pathfinder is not deterministic for a
            # single spring-to-deer SetPosition.  Latch the measured upper
            # lane entry before allowing the camp command; controlled Kai
            # continues to use the ordinary hard-opening legalizer.
            "official_upper_entry_reached": False,
            "official_upper_inner_reached": False,
            "official_upper_route_stage": 0,
            "official_upper_route_interrupted": False,
            "opening_fallback_active": False,
            "opening_fallback_source_signature": None,
            "opening_fallback_camp_name": "",
            "opening_fallback_lifecycle_id": 0,
            "opening_fallback_position": None,
            "opening_fallback_target_slot": None,
            "official_noncompliant": False,
            "learned_resource_target_key": None,
            "resource_candidate_distances": {},
            "resource_candidate_slots": (),
            "last_route_progress_frame": None,
            "last_route_progress_signature": None,
        }
        self._route_shaping_consumed_frame = None
        self.lane_opening_history = {}
        self.lane_attendance_history = {}
        self.lane_monitor_counters = defaultdict(float)
        self.no_target_history = {}
        self.long_inactivity_history = {}
        self._behavior_shaping_consumed_frame = None
        self.tyrant_reward_state = {
            "runtime_id": 0,
            "rewarded": False,
            "observed_dead": False,
            "kill_counts": {1: 0, 2: 0},
        }
        self.rewarded_kai_wolf_deaths = set()
        self.initial_wolf_runtime_ids = set()
        self.initial_wolf_member_deaths = set()
        self.rewarded_treasure_thief_deaths = set()
        self.speed_buff_reward_state = {}
        self.crystal_outcome_rewarded = False
        self.structure_hp_ratios = {}
        self.previous_structure_potential = None
        self.post_kill_push_state = {
            "own": {"expires": -1, "tokens": 0},
            "enemy": {"expires": -1, "tokens": 0},
        }
        self.post_kill_last_death_frame = {}
        self.scenario_rewarded_deaths = set()
        self.temporal_focus_state = {}
        self.defense_pressure_state = {
            1: {
                "structure_id": 0,
                "last_evaluation_frame": None,
                "window_total": 0.0,
            },
            2: {
                "structure_id": 0,
                "last_evaluation_frame": None,
                "window_total": 0.0,
            },
        }
        self.previous_crystal_danger_potential = None
        self.command_history = deque()
        # Hold a just-emitted command through the short platform state-echo
        # delay. A confirmed rejection releases the local throttle, while its
        # target runtime stays blocked for that entity lifecycle.
        self.local_last_command_frame = 0
        self.pending_collaboration_command = None
        self.active_command_attribution = None
        self.last_command_decision = None
        # Forced opening commands need a behavioral response lifecycle in
        # addition to the platform acceptance lifecycle above.  A goal gets
        # one initial command and at most one recovery retry.  A second
        # failure locks that exact role/lifecycle goal and forces a different
        # command-bearing fallback; it never turns into an Unset/no-command
        # handoff that would send the official hero back toward spring.
        self.official_command_guard = {
            "attempts": {},
            "retry_requested": set(),
            "failed_signatures": set(),
            "response_window": None,
            "fallback_pending": False,
            "fallback_source_signature": None,
            "fallback_confirmed": False,
            "last_failure_frame": None,
            "last_failure_reason": "none",
            "last_success_frame": None,
        }
        self.official_no_command_idle_state = {
            "start_frame": None,
            "start_position": None,
            "last_position": None,
            "path_distance": 0.0,
            "start_damage": None,
            "next_evaluation_frame": None,
        }
        self.rejected_command_target_runtimes = set()
        self.tower_target_streak = {}
        self.tower_risk_state = {}
        self.tower_risk_counters = defaultdict(float)
        self.defense_wave_state = None
        self.defense_wave_counters = defaultdict(float)
        self._memo_frame_no = None
        self._memo_features = None
        self._memo_context = None

    def process_feature(self, observation, copy=True):
        """Return features in controlled-actor observation order.

        The public/default path keeps the historical list-of-independent-arrays
        contract.  The Agent hot path is trusted not to mutate the result and
        requests the cached contiguous ``(2, FEATURE_DIM)`` array directly,
        avoiding two row copies followed by another ``np.stack`` copy.
        """

        frame_state = u.safe_get(observation, "frame_state", default={}) or {}
        current_frame = u.frame_number(frame_state)
        if self.last_frame_no is not None and current_frame < self.last_frame_no:
            self.reset(self.camp)
        if current_frame == self._memo_frame_no and self._memo_features is not None:
            if copy:
                return [feature.copy() for feature in self._memo_features]
            return self._memo_features

        controlled_infos = self._controlled_infos(observation)
        context = self._build_context(observation, controlled_infos)
        common_blocks = {
            "soldiers": self._build_soldiers(context),
            "monsters": self._build_monsters(context),
            "tyrant_team_context": self._build_tyrant_team_context(context),
            "collaboration_state": self._build_collaboration_state(context),
            "global_state": self._build_global_state(context),
        }

        features = []
        for actor_index, hero_info in enumerate(controlled_infos):
            query_runtime = u.as_int(u.safe_get(hero_info, "runtime_id", default=0), 0)
            query_config = u.as_int(u.safe_get(hero_info, "config_id", default=0), 0)
            query = context["hero_by_runtime"].get(query_runtime)
            if query is None:
                query = next(
                    (
                        hero
                        for hero in context["friend_heroes"]
                        if u.actor_config_id(hero["actor"]) == query_config and hero["controlled"]
                    ),
                    None,
                )
            if query is None:
                raise ValueError(f"controlled hero {query_config}/{query_runtime} is missing from frame_state")

            blocks = {
                "image": self._official_image(hero_info),
                "heroes": self._build_heroes(context, query),
                "soldiers": common_blocks["soldiers"],
                "organs": self._build_organs(context, query),
                "monsters": common_blocks["monsters"],
                "bullets": self._build_bullets(context, query),
                "targets": self._build_targets(context, query, hero_info),
                "hero_relations": self._build_hero_relations(context, query),
                "dynamic_lane_position": self._build_dynamic_lane_position(context),
                "tyrant_local_position": self._build_tyrant_local_position(context, query),
                "tyrant_team_context": common_blocks["tyrant_team_context"],
                "lane_state": self._build_lane_state(context, query),
                "resource_anchors": self._build_resource_anchors(context, query),
                "collaboration_state": common_blocks["collaboration_state"],
                "global_state": common_blocks["global_state"],
                "jungle_plan": self._build_jungle_plan(context, query),
                "lane_plan": self._build_lane_plan(context, query),
                "main_context": self._build_main_context(context, query, hero_info),
            }
            feature = self._flatten_blocks(blocks)
            features.append(feature)

        if len(features) != FrozenV1Config.CONTROLLED_ACTOR_COUNT:
            raise RuntimeError(f"expected two v1 actor features, got {len(features)}")

        features = np.stack(features, axis=0).astype(np.float32, copy=False)
        if features.shape != (
            FrozenV1Config.CONTROLLED_ACTOR_COUNT,
            FrozenV1Config.FEATURE_DIM,
        ):
            raise RuntimeError(f"invalid stacked v1 feature shape: {features.shape}")

        self.last_frame_no = current_frame
        self._memo_frame_no = current_frame
        self._memo_features = features
        self._memo_context = context
        if copy:
            return [feature.copy() for feature in features]
        return features

    def _build_jungle_plan(self, context, query):
        """Expose the topology-aware Kai macro state to each controlled actor."""

        vector = np.zeros((1, 67), dtype=np.float32)
        row = vector[0]
        fields = schema_v1.JUNGLE_PLAN_FIELD_MAP
        plan = context["jungle_macro"]
        query_position = query["side_position"]
        waypoint = plan["next_waypoint"]
        if query_position is not None and waypoint is not None:
            dx = waypoint[0] - query_position[0]
            dz = waypoint[1] - query_position[1]
            distance = math.hypot(dx, dz)
        else:
            dx = dz = distance = 0.0
        values = {
            "plan_valid": plan["valid"],
            "kai_is_official": plan["kai_is_official"],
            "team_emergency_active": plan["team_emergency"],
            "query_is_kai": u.actor_config_id(query["actor"]) == 193,
            f"phase_{plan['phase']}": 1.0,
            f"mode_{plan['mode']}": 1.0,
            f"interrupt_{plan['interrupt']}": 1.0,
            f"tyrant_buff_tier_{plan['buff_tier']}": 1.0,
            "kai_level_norm": u.clip(plan["kai_level"] / 15.0),
            "kai_level_at_least_2": plan["kai_level"] >= 2,
            "kai_level_at_least_3": plan["kai_level"] >= 3,
            "upper_deer_cleared_once": plan["deer_cleared"],
            "upper_wolf_cleared_once": plan["wolf_cleared"],
            "lower_remaining_norm": u.clip(plan["lower_remaining"] / 4.0),
            "tyrant_alive": plan["tyrant"]["alive"],
            "tyrant_spawn_within_arrival": plan["tyrant_spawn_within_arrival"],
            "enemy_attacking_tyrant": plan["tyrant"]["enemy_attacking"],
            "enemy_jungle_locked": False,
            "own_jungle_invaded": plan["own_jungle_invaded"],
            "contest_favorable": plan["contest_favorable"],
            "ally_eta_norm": u.clip(plan["ally_eta"] / 30.0),
            "enemy_eta_norm": u.clip(plan["enemy_eta"] / 30.0),
            "next_waypoint_dx_norm": u.signed_clip(dx / 30000.0),
            "next_waypoint_dz_norm": u.signed_clip(dz / 30000.0),
            "next_waypoint_distance_norm": u.clip(distance / 50000.0),
            "path_distance_norm": u.clip(plan["path_distance"] / 100000.0),
            "route_progress": plan["route_progress"],
            "route_stuck": plan["route_stuck"],
        }
        for name, value in values.items():
            _put(row, fields, name, value)
        _put(row, fields, f"goal_node_{plan['goal_node']}", 1.0)
        _put(row, fields, f"next_node_{plan['next_node']}", 1.0)
        return vector

    def _build_lane_plan(self, context, query):
        """Expose the assigned row, opening phase and interrupt state."""

        vector = np.zeros((1, 64), dtype=np.float32)
        row = vector[0]
        fields = schema_v1.LANE_PLAN_FIELD_MAP
        config_id = u.actor_config_id(query["actor"])
        plan = context["lane_macro"].get(config_id)
        if plan is None:
            _put(row, fields, "phase_inactive", 1.0)
            _put(row, fields, "interrupt_none", 1.0)
            return vector

        query_position = query["side_position"]
        waypoint = plan["next_waypoint"]
        if query_position is not None and waypoint is not None:
            dx = waypoint[0] - query_position[0]
            dz = waypoint[1] - query_position[1]
            waypoint_distance = math.hypot(dx, dz)
        else:
            dx = dz = waypoint_distance = 0.0
        own_front = plan["own_front"]
        enemy_front = plan["enemy_front"]
        clash = plan["clash_position"]
        stable_clash = context["lane_summary"]["clash_position"]
        target = plan["target_soldier"]
        target_position = _position(target, self.camp) if target else None
        target_distance = _distance_or_inf(query_position, target_position)
        query_side = u.normalized_side_position(query_position)
        values = {
            "plan_valid": plan["valid"],
            "query_is_lane_hero": config_id in LANE_ROW_SIDE_Z,
            "query_is_official": query["official"],
            "opening_active": plan["opening_active"],
            "hard_mask_active": plan["hard_mask_active"],
            "contact_reached": plan["contact_reached"],
            "assigned_upper": plan["row"] == "upper",
            "assigned_lower": plan["row"] == "lower",
            "own_wave_present": own_front is not None,
            "enemy_wave_present": enemy_front is not None,
            f"phase_{plan['phase']}": 1.0,
            f"interrupt_{plan['interrupt']}": 1.0,
            "waypoint_valid": waypoint is not None,
            "next_waypoint_dx_norm": u.signed_clip(dx / 30000.0),
            "next_waypoint_dz_norm": u.signed_clip(dz / 30000.0),
            "next_waypoint_distance_norm": u.clip(waypoint_distance / 50000.0),
            "route_progress": plan["route_progress"],
            "query_side_x_norm": query_side[0],
            "query_side_z_norm": query_side[1],
            "assigned_row_z_norm": u.signed_clip(plan["row_z"] / u.SIDE_Z_SCALE),
            "own_row_front_valid": own_front is not None,
            "own_row_front_side_x_norm": u.signed_clip(own_front[0] / u.SIDE_X_SCALE) if own_front else 0.0,
            "own_row_front_side_z_norm": u.signed_clip(own_front[1] / u.SIDE_Z_SCALE) if own_front else 0.0,
            "enemy_row_front_valid": enemy_front is not None,
            "enemy_row_front_side_x_norm": u.signed_clip(enemy_front[0] / u.SIDE_X_SCALE) if enemy_front else 0.0,
            "enemy_row_front_side_z_norm": u.signed_clip(enemy_front[1] / u.SIDE_Z_SCALE) if enemy_front else 0.0,
            "row_clash_valid": clash is not None,
            "row_clash_side_x_norm": u.signed_clip(clash[0] / u.SIDE_X_SCALE) if clash else 0.0,
            "row_clash_side_z_norm": u.signed_clip(clash[1] / u.SIDE_Z_SCALE) if clash else 0.0,
            "distance_to_own_wave_norm": u.clip(_distance_or_inf(query_position, own_front) / 50000.0),
            "distance_to_enemy_wave_norm": u.clip(_distance_or_inf(query_position, enemy_front) / 50000.0),
            "nearest_enemy_hero_distance_norm": u.clip(plan["nearest_enemy_hero_distance"] / 50000.0),
            "query_hp_ratio": _hp_ratio(query["actor"]),
            "enemy_hero_near": plan["enemy_hero_near"],
            "recent_enemy_hero_hit": plan["recent_enemy_hero_hit"],
            "structure_defense_required": plan["structure_defense"],
            "own_outer_alive": context["organs"][0]["alive"],
            "first_wave_captured": plan["first_wave_captured"],
            "waiting_for_wave": plan["waiting_for_wave"],
            "following_wave": plan["phase"] in ("follow_wave", "engage_wave"),
            "official_can_target_soldier": plan["target_slot"] >= 28,
            "target_soldier_slot_norm": u.clip((plan["target_slot"] - 28) / 9.0) if plan["target_slot"] >= 28 else 0.0,
            "target_soldier_hp_ratio": _hp_ratio(target) if target else 0.0,
            "target_soldier_distance_norm": u.clip(target_distance / 50000.0),
            "target_soldier_in_attack_range": target_distance <= max(3500.0, _value(query["actor"], "attack_range", "attackRange")),
            "stable_clash_valid": stable_clash is not None,
            "stable_clash_side_x_norm": u.signed_clip(stable_clash[0] / u.SIDE_X_SCALE) if stable_clash else 0.0,
            "first_wave_contact_soon": plan["first_wave_contact_soon"],
            "released_by_first_wave_clear": plan["release_reason"] == "first_wave_cleared",
            "target_lock_active": plan["target_lock_active"],
            "route_node_base_exit": plan["route_node"] == 0,
            "route_node_corridor": plan["route_node"] == 1,
            "route_node_stage_or_wave": plan["route_node"] == 2,
        }
        for name, value in values.items():
            _put(row, fields, name, value)
        return vector

    def lane_opening_runtime_plans(self):
        """Return the exact per-controlled-actor plans used by the mask."""

        if self._memo_context is None:
            raise RuntimeError("lane_opening_runtime_plans requires process_feature")
        result = []
        for info in self._memo_context["controlled_infos"]:
            config_id = u.as_int(u.safe_get(info, "config_id", default=0), 0)
            runtime_id = u.as_int(u.safe_get(info, "runtime_id", default=0), 0)
            plan = dict(self._memo_context["lane_macro"].get(config_id, {}))
            plan["runtime_id"] = runtime_id
            result.append(plan)
        return tuple(result)

    def jungle_opening_runtime_plans(self):
        """Return the current controlled-Kai plan used by the opening mask."""

        if self._memo_context is None:
            raise RuntimeError("jungle_opening_runtime_plans requires process_feature")
        macro = self._memo_context["jungle_macro"]
        result = []
        for info in self._memo_context["controlled_infos"]:
            config_id = u.as_int(u.safe_get(info, "config_id", default=0), 0)
            runtime_id = u.as_int(u.safe_get(info, "runtime_id", default=0), 0)
            plan = {
                "runtime_id": runtime_id,
                "phase": macro["phase"],
                "interrupt": macro["interrupt"],
                "hard_mask_active": bool(
                    config_id == 193
                    and macro["valid"]
                    and (
                        macro["phase"] in ("upper_deer", "upper_wolf")
                        or (
                            macro["opening_rotation_active"]
                            and macro["phase"] == "lower_clear"
                        )
                        or macro["camp_exit_active"]
                    )
                    and macro["interrupt"] == "none"
                ),
                "camp_exit_active": bool(macro["camp_exit_active"]),
                "route_stuck": bool(config_id == 193 and macro["route_stuck"]),
                "force_attack": bool(
                    config_id == 193
                    and macro["opening_target_attackable"]
                    and macro["target_slot"] is not None
                ),
                "wait_for_spawn": bool(
                    config_id == 193 and macro["opening_wait_for_spawn"]
                ),
                "leash_recover": bool(
                    config_id == 193
                    and macro.get("opening_leash_recover", False)
                ),
                "target_slot": (
                    8 + int(macro["target_slot"])
                    if macro["target_slot"] is not None
                    else -1
                ),
                "waypoint_dx": macro["next_waypoint_dx"],
                "waypoint_dz": macro["next_waypoint_dz"],
                "target_dx": macro["opening_target_dx"],
                "target_dz": macro["opening_target_dz"],
            }
            result.append(plan)
        return tuple(result)

    def _enemy_spawn_boundary_x(self, context):
        enemy_crystal_x = float(context["organs"][5]["side_position"][0])
        enemy_spring_config = 43 if self.camp == 1 else 42
        enemy_spring_x = float(
            u.side_position(
                u.ORGAN_ANCHORS[enemy_spring_config], self.camp
            )[0]
        )
        return (enemy_crystal_x + enemy_spring_x) / 2.0

    def safety_runtime_plans(self):
        """Return PPO 3.5 topology/tower safety inputs in actor order."""

        if self._memo_context is None:
            raise RuntimeError("safety_runtime_plans requires process_feature")
        context = self._memo_context
        boundary = self._enemy_spawn_boundary_x(context)
        result = []
        for info in context["controlled_infos"]:
            runtime_id = u.as_int(
                u.safe_get(info, "runtime_id", default=0), 0
            )
            query = context["hero_by_runtime"].get(runtime_id)
            unsafe_enemy_slots = tuple(
                1 + index
                for index, enemy in enumerate(context["enemy_heroes"])
                if enemy["alive"]
                and enemy["visible"]
                and enemy["side_position"] is not None
                and enemy["side_position"][0] > boundary
            )
            tower_state = self.tower_risk_state.get(runtime_id, {})
            current_critical_tower = next(
                (
                    organ
                    for organ in context["organs"][3:5]
                    if query is not None
                    and query["side_position"] is not None
                    and organ["alive"]
                    and organ["actor"] is not None
                    and _value(
                        organ["actor"],
                        "attack_range",
                        "attackRange",
                    )
                    > 0.0
                    and _distance_or_inf(
                        query["side_position"],
                        organ["side_position"],
                    )
                    <= _value(
                        organ["actor"],
                        "attack_range",
                        "attackRange",
                    )
                    and not self._tower_has_reliable_minion_cover(
                        context, organ
                    )
                    and _target_runtime(organ["actor"]) == runtime_id
                    and _hp_ratio(query["actor"])
                    <= V1TrainingConfig.TOWER_RISK_CRITICAL_HP_RATIO
                ),
                None,
            )
            result.append(
                {
                    "runtime_id": runtime_id,
                    "position": (
                        query["side_position"] if query is not None else None
                    ),
                    "move_speed": (
                        _value(
                            query["actor"], "mov_spd", "movSpd"
                        )
                        if query is not None
                        else 0.0
                    ),
                    "decision_seconds": 6.0 * u.RAW_FRAME_SECONDS,
                    "spawn_boundary_x": boundary,
                    "unsafe_enemy_slots": unsafe_enemy_slots,
                    "tower_critical": bool(
                        current_critical_tower is not None
                        or tower_state.get("phase") == "critical"
                    ),
                    "critical_tower_position": tower_state.get(
                        "tower_position",
                        current_critical_tower["side_position"]
                        if current_critical_tower is not None
                        else None,
                    ),
                }
            )
        return tuple(result)

    def flash_runtime_plans(self):
        """Return transient Flash gate inputs without changing feature/sample shapes."""

        if self._memo_context is None:
            raise RuntimeError("flash_runtime_plans requires process_feature")
        context = self._memo_context
        frame_no = context["frame_no"]
        spawn_boundary_x = self._enemy_spawn_boundary_x(context)
        result = []
        for info in context["controlled_infos"]:
            runtime_id = u.as_int(u.safe_get(info, "runtime_id", default=0), 0)
            config_id = u.as_int(u.safe_get(info, "config_id", default=0), 0)
            query = context["hero_by_runtime"].get(runtime_id)
            actor = query["actor"] if query is not None else {}
            position = query["side_position"] if query is not None else None
            self_hp = _hp_ratio(actor)
            summoner = u.skill_slots(actor)[5]
            summoner_id = u.as_int(
                u.safe_get(summoner or {}, "config_id", "configId", default=0), 0
            )
            enemies = [
                enemy
                for enemy in context["enemy_heroes"]
                if enemy["alive"]
                and enemy["visible"]
                and enemy["side_position"] is not None
                and position is not None
            ]
            nearest = min(
                enemies,
                key=lambda enemy: _distance_or_inf(position, enemy["side_position"]),
                default=None,
            )
            enemy_distance = (
                _distance_or_inf(position, nearest["side_position"])
                if nearest is not None
                else math.inf
            )
            enemy_hp = _hp_ratio(nearest["actor"]) if nearest is not None else 1.0
            summary = self._event_summary(query, frame_no) if query is not None else {
                "short_types": set(),
                "recent_other": set(),
            }
            recent_hero_hurt = "hurt_by_hero" in summary["short_types"]
            recent_tower_hurt = "hurt_by_tower" in summary["short_types"]
            threatening_tower = next(
                (
                    organ
                    for organ in context["organs"][3:5]
                    if organ["alive"]
                    and organ["actor"]
                    and _target_runtime(organ["actor"]) == runtime_id
                ),
                None,
            )
            tower_targeted = threatening_tower is not None
            tower_threat = bool(tower_targeted or recent_tower_hurt)
            nearby_friends = sum(
                friend["alive"]
                and friend["side_position"] is not None
                and _distance_or_inf(position, friend["side_position"]) <= 12000.0
                for friend in context["friend_heroes"]
            )
            nearby_enemies = sum(
                _distance_or_inf(position, enemy["side_position"]) <= 12000.0
                for enemy in enemies
            )
            outnumbered = nearby_enemies > nearby_friends
            lane_plan = context["lane_macro"].get(config_id, {})
            jungle = context["jungle_macro"]
            ordinary_opening = bool(
                lane_plan.get("opening_active")
                and lane_plan.get("hard_mask_active")
                or (
                    config_id == 193
                    and jungle.get("opening_rotation_active")
                    and jungle.get("interrupt") == "none"
                )
            )

            skill3_events = self.recent_events.get(("friend", config_id), ())
            recent_skill3 = any(
                event["type"] == "skill_used"
                and event["slot"] == 3
                and frame_no - event["frame"] <= 36
                for event in skill3_events
            )
            skill3 = u.skill_slots(actor)[3]
            combo_active = u.as_float(
                u.safe_get(
                    skill3 or {},
                    "combo_effect_time",
                    "comboEffectTime",
                    default=0,
                ),
                0,
            ) > 0
            marco_ultimate = config_id == 132 and (combo_active or recent_skill3)

            mode = "blocked"
            reason = "no_threat"
            direction_dx = direction_dz = 0.0
            allow = False
            if summoner_id != 80115:
                reason = "not_flash"
            elif ordinary_opening:
                reason = "ordinary_opening"
            elif position is None:
                reason = "position_unknown"
            elif not enemies and not tower_threat:
                reason = "no_visible_enemy_or_tower_threat"
            else:
                low_vs_enemy = (
                    self_hp <= 0.45
                    and (nearest is None or self_hp <= enemy_hp + 0.20 or enemy_hp <= 0.25)
                )
                enemy_low = enemy_hp <= 0.25 or (
                    enemy_hp <= 0.35 and self_hp - enemy_hp >= 0.15
                )
                chase = bool(
                    nearest is not None
                    and 6000.0 <= enemy_distance <= 14000.0
                    and enemy_low
                    and self_hp >= enemy_hp - 0.10
                    and not outnumbered
                )
                if marco_ultimate and nearest is not None:
                    allow, mode, reason = True, "marco_ultimate", "marco_ultimate"
                elif tower_threat and self_hp <= 0.45:
                    allow, mode, reason = True, "tower_escape", "low_hp_tower_threat"
                elif chase:
                    allow, mode, reason = True, "chase", "low_hp_enemy_chase"
                elif (
                    nearest is not None
                    and (recent_hero_hurt or enemy_distance <= 10000.0)
                    and low_vs_enemy
                ):
                    allow, mode, reason = True, "escape", "low_hp_hero_fight"

                if allow and mode == "tower_escape":
                    tower_state = self.tower_risk_state.get(runtime_id, {})
                    tower_position = (
                        threatening_tower["side_position"]
                        if threatening_tower is not None
                        else tower_state.get("tower_position")
                    )
                    if tower_position is not None:
                        direction_dx = position[0] - tower_position[0]
                        direction_dz = position[1] - tower_position[1]
                    elif nearest is not None:
                        ex, ez = nearest["side_position"]
                        direction_dx = position[0] - ex
                        direction_dz = position[1] - ez
                    else:
                        direction_dx, direction_dz = -1.0, 0.0
                elif allow and nearest is not None:
                    ex, ez = nearest["side_position"]
                    direction_dx, direction_dz = ex - position[0], ez - position[1]
                    if mode == "escape":
                        direction_dx, direction_dz = -direction_dx, -direction_dz
                elif allow:
                    # Tower-only emergency: retreat toward the own side/base.
                    direction_dx, direction_dz = -1.0, 0.0

            if allow:
                length = math.hypot(direction_dx, direction_dz)
                if length <= 1e-6:
                    allow, reason = False, "no_safe_landing"
                else:
                    landing_x = position[0] + direction_dx / length * 7000.0
                    landing_z = position[1] + direction_dz / length * 7000.0
                    if (
                        abs(landing_x) > u.SIDE_X_SCALE
                        or abs(landing_z) > u.SIDE_Z_SCALE
                    ):
                        allow, reason = False, "landing_out_of_bounds"
                    elif landing_x > spawn_boundary_x:
                        allow, reason = False, "landing_beyond_spawn_boundary"

            obstacle = self._official_image(info)[4]
            result.append(
                {
                    "runtime_id": runtime_id,
                    "flash_gate_active": bool(
                        config_id in (132, 141)
                        and summoner_id == 80115
                    ),
                    "allow": allow,
                    "mode": mode,
                    "reason": reason,
                    "direction_dx": direction_dx,
                    "direction_dz": direction_dz,
                    "obstacle": obstacle,
                    "self_hp": self_hp,
                    "enemy_hp": enemy_hp if nearest is not None else None,
                    "enemy_distance": enemy_distance,
                    "tower_threat": tower_threat,
                    "marco_ultimate": marco_ultimate,
                    "spawn_boundary_x": spawn_boundary_x,
                    "position": position,
                }
            )
        return tuple(result)

    def command_runtime_context(self):
        """Return the exact runtime mapping consumed by the v1 legalizer.

        ``process_feature`` must be called first for the same environment step;
        this prevents command masks from being built from a different frame.
        """

        if self._memo_context is None or self._memo_frame_no is None:
            raise RuntimeError("command_runtime_context requires process_feature for the current frame")
        from agent_ppo.feature.collaboration_v1 import CommandRuntimeContextV1

        context = self._memo_context
        first_runtime = u.as_int(
            u.safe_get(context["controlled_infos"][0], "runtime_id", default=0), 0
        )
        query = context["hero_by_runtime"].get(first_runtime)
        if query is None:
            raise RuntimeError("first controlled hero is missing from command runtime context")
        descriptors = self._target_descriptors(context, query)
        target_runtime_ids = []
        target_legal = []
        official_alive = context["official_hero"]["alive"]
        for descriptor in descriptors:
            actor = descriptor["actor"]
            snapshot = descriptor.get("snapshot")
            kind = descriptor["kind"]
            runtime_id = u.actor_runtime_id(actor) if actor else 0
            if snapshot is not None:
                alive = snapshot["alive"]
                visible = snapshot["visible"]
                current_valid = snapshot["current_valid"]
            else:
                alive = bool(actor) and u.is_alive(actor)
                visible = bool(actor) and u.visible_to_camp(
                    actor,
                    self.camp,
                    is_friend=u.actor_camp(actor) == self.camp,
                )
                current_valid = bool(actor) and visible
            legal_kind = kind in (
                "enemy_hero",
                "monster",
                "soldier",
                "outer_tower",
                "high_tower",
                "crystal",
            )
            if kind == "monster" and descriptor.get("fixed") is not None:
                legal_kind = monster_set_target_allowed(
                    descriptor["fixed"]["config_id"]
                )
            abilities = list(u.actor_get(actor, "abilities", default=[]) or []) if actor else []
            forbid_select = len(abilities) > 15 and bool(abilities[15])
            target_runtime_ids.append(runtime_id)
            target_legal.append(
                bool(
                    legal_kind
                    and runtime_id
                    and alive
                    and visible
                    and current_valid
                    and official_alive
                    and not forbid_select
                    and runtime_id
                    not in self.rejected_command_target_runtimes
                )
            )

        friend_by_config = {
            u.actor_config_id(hero["actor"]): hero for hero in context["friend_heroes"]
        }
        side_positions = [None] * FrozenV1Config.COMMAND_HEADS[2]
        valid = [False] * FrozenV1Config.COMMAND_HEADS[2]
        for slot, config_id in enumerate(u.HERO_CONFIG_IDS, 1):
            hero = friend_by_config[config_id]
            if hero["alive"] and hero["side_position"] is not None:
                side_positions[slot] = hero["side_position"]
                valid[slot] = True
        lane = context["lane_summary"]
        if lane["friend_front"] is not None:
            side_positions[4] = lane["friend_front"][1]
            valid[4] = True
        if lane["clash_position"] is not None:
            side_positions[5] = lane["clash_position"]
            valid[5] = True
        elif lane["enemy_front"] is not None:
            side_positions[5] = lane["enemy_front"][1]
            valid[5] = True
        else:
            # Keep lane heroes out of the spring before the first waves become
            # observable. Slot 5 stages at map centre until a real clash/front
            # replaces it.
            side_positions[5] = (0.0, 0.0)
            valid[5] = True
        official_config_id = u.actor_config_id(context["official_hero"]["actor"])
        official_position = context["official_hero"]["side_position"]
        official_lane_plan = context["lane_macro"].get(official_config_id)
        if (
            official_lane_plan
            and official_lane_plan["command_active"]
            and official_lane_plan["interrupt"] == "none"
            and official_lane_plan["command_waypoint"] is not None
        ):
            # Reuse the push-lane position slot for the measured row-specific
            # opening waypoint.  The command head remains checkpoint-explicit
            # and no free-form coordinate action is introduced.
            side_positions[5] = official_lane_plan["command_waypoint"]
            lane_position_signature = self._lane_position_goal_signature(
                u.actor_runtime_id(context["official_hero"]["actor"]),
                int(official_lane_plan.get("route_node", 2)),
                official_lane_plan["command_waypoint"],
            )
            active_position = context["collaboration"]["command_position"]
            same_active_lane_position = bool(
                context["collaboration"]["active"]
                and context["collaboration"]["type"] == "set_position"
                and active_position is not None
                and _distance_or_inf(
                    active_position,
                    official_lane_plan["command_waypoint"],
                )
                <= 100.0
            )
            lane_target_runtime = (
                u.actor_runtime_id(
                    official_lane_plan.get("target_soldier")
                )
                if official_lane_plan.get("target_soldier") is not None
                else 0
            )
            lane_target_signature = (
                self._lane_target_goal_signature(
                    u.actor_runtime_id(context["official_hero"]["actor"]),
                    lane_target_runtime,
                )
                if lane_target_runtime
                else None
            )
            same_active_lane_target = bool(
                lane_target_runtime
                and context["collaboration"]["active"]
                and context["collaboration"]["type"] == "set_target"
                and context["collaboration"]["target_runtime"]
                == lane_target_runtime
            )
            lane_target_available = bool(
                official_lane_plan.get("row_entry_reached")
                and lane_target_runtime
                and official_lane_plan.get("target_distance", float("inf"))
                <= LANE_OFFICIAL_TARGET_COMMAND_DISTANCE
                and not self._official_goal_failed(
                    lane_target_signature
                )
                and (
                    not (
                        same_active_lane_target
                    )
                    or self._official_goal_retry_requested(
                        lane_target_signature
                    )
                )
            )
            valid[5] = bool(
                not (
                    same_active_lane_target
                    and not self._official_goal_retry_requested(
                        lane_target_signature
                    )
                )
                and (
                    lane_target_available
                    or (
                        not self._official_goal_failed(
                            lane_position_signature
                        )
                        and (
                            not same_active_lane_position
                            or self._official_goal_retry_requested(
                                lane_position_signature
                            )
                        )
                    )
                )
            )
        own_outer, own_high, own_crystal = context["organs"][:3]
        if own_outer["alive"]:
            side_positions[6] = own_outer["side_position"]
            valid[6] = True
        high_or_crystal = own_high if own_high["alive"] else own_crystal
        if high_or_crystal["alive"]:
            side_positions[7] = high_or_crystal["side_position"]
            valid[7] = True
        tyrant = self._tyrant_status(context)
        tyrant_window = (
            tyrant["alive"]
            or tyrant["spawn_within_30"]
            or tyrant["team_attacking"]
            or tyrant["enemy_attacking"]
        )
        if tyrant_window and tyrant["position"] is not None:
            side_positions[8] = tyrant["position"]
            valid[8] = True
        own_spring_config = 42 if self.camp == 1 else 43
        side_positions[9] = u.side_position(
            u.ORGAN_ANCHORS[own_spring_config], self.camp
        )
        valid[9] = True
        jungle_target_legal = []
        jungle_position_legal = []
        for monster_index, (monster, spawn) in enumerate(
            zip(context["monsters"], context["monster_spawn"])
        ):
            position_slot = 10 + monster_index
            side_positions[position_slot] = u.side_position(
                monster["anchor"], self.camp
            )
            timer_raw = (
                u.safe_get(spawn, "spawn_timer", "spawnTimer", default=None)
                if spawn
                else None
            )
            timer_ms = u.as_float(timer_raw, float("inf"))
            timer_valid = math.isfinite(timer_ms) and timer_ms >= 0.0
            # Ordinary tasks require a currently living camp.  Positive
            # countdowns are never a reason to camp an empty anchor. Tyrant
            # retains its separately verified preparation window.
            alive_or_due_soon = bool(
                monster["alive"]
                or (
                    monster["config_id"]
                    == V1TrainingConfig.TYRANT_CONFIG_ID
                    and timer_valid
                    and timer_ms <= 15000.0
                )
            )
            position_ok = bool(
                monster_set_position_allowed(monster["config_id"])
                and alive_or_due_soon
            )
            anchor = tuple(float(value) for value in monster["anchor"])
            lifecycle_id = int(
                self.monster_camp_history.get(anchor, {}).get(
                    "lifecycle_id", 0
                )
            )
            if (
                "farm_camp",
                anchor,
                lifecycle_id,
            ) in self.official_task_failed_lifecycles:
                position_ok = False
            target_ok = bool(
                monster["alive"]
                and monster["visible"]
                and u.actor_runtime_id(monster["actor"])
                and target_legal[8 + monster_index]
                and monster_set_target_allowed(monster["config_id"])
            )
            valid[position_slot] = position_ok
            jungle_position_legal.append(position_ok)
            jungle_target_legal.append(target_ok)
        active_task = self.official_task_state
        if active_task is not None:
            task_slot = int(active_task.get("position_slot", -1))
            task_waypoint = active_task.get("command_waypoint")
            if (
                0 <= task_slot < len(side_positions)
                and task_waypoint is not None
            ):
                side_positions[task_slot] = tuple(
                    float(value) for value in task_waypoint
                )
                valid[task_slot] = True
                if 10 <= task_slot < 30:
                    jungle_position_legal[task_slot - 10] = True
        macro = context["jungle_macro"]
        route_target_slot = macro["target_slot"]
        route_position_slot = -1
        route_command_target_slot = -1
        jungle_lane_target_slot = -1
        if (
            u.actor_config_id(context["official_hero"]["actor"]) == 193
            and route_target_slot is not None
        ):
            route_position_slot = 10 + route_target_slot
            route_command_target_slot = 8 + route_target_slot
            # Reuse the semantic camp slot as the next topology waypoint.  No
            # command-head expansion is required and the next cooldown refresh
            # advances the official hero along the measured route.
            side_positions[route_position_slot] = macro["next_waypoint"]
            active_position = context["collaboration"]["command_position"]
            same_active_waypoint = bool(
                context["collaboration"]["active"]
                and context["collaboration"]["type"] == "set_position"
                and active_position is not None
                and _distance_or_inf(
                    active_position, macro["next_waypoint"]
                )
                <= 100.0
            )
            route_goal_signature = (
                self._jungle_position_goal_signature(
                    u.actor_runtime_id(
                        context["official_hero"]["actor"]
                    ),
                    macro.get("opening_camp_name", ""),
                    int(
                        macro.get("opening_camp_lifecycle_id", 0)
                    ),
                    macro["next_waypoint"],
                    fallback=bool(
                        macro.get("opening_fallback_active")
                    ),
                )
                if macro.get("opening_rotation_active")
                or macro.get("opening_fallback_active")
                else None
            )
            route_position_legal = bool(
                macro["valid"]
                and not self._official_goal_failed(
                    route_goal_signature
                )
                and (
                    not same_active_waypoint
                    or self._official_goal_retry_requested(
                        route_goal_signature
                    )
                )
            )
            if macro.get("opening_fallback_active"):
                # A command-bearing noncompliance handoff must remain legal
                # even when the fallback camp is currently empty/long-cooldown.
                # It is a safe position goal, not a claim that the camp cleared
                # or is ready to farm.
                valid[route_position_slot] = route_position_legal
                jungle_position_legal[route_target_slot] = (
                    route_position_legal
                )
            valid[route_position_slot] = route_position_legal
            jungle_position_legal[route_target_slot] = (
                route_position_legal
            )
            if macro["opening_rotation_active"]:
                # Archer camps reuse cfg32/cfg33. Override the frozen semantic
                # slot with the actor from the current physical camp so the
                # command never points at the same-cfg monster in the other
                # camp. This changes no head or feature dimension.
                opening_actor = macro["opening_target_actor"]
                command_slot = route_command_target_slot
                target_runtime_ids[command_slot] = 0
                target_legal[command_slot] = False
                jungle_target_legal[route_target_slot] = False
                if opening_actor is not None:
                    # Platform evidence rejects ordinary-monster SetTarget.
                    # Keep the runtime visible for diagnostics, but command
                    # official Kai only through the accepted SetPosition
                    # route and let its built-in micro choose camp members.
                    target_runtime_ids[command_slot] = (
                        u.actor_runtime_id(opening_actor)
                    )
        if (
            official_config_id == 193
            and macro["phase"] == "lane_soak_wave"
            and macro["valid"]
        ):
            wave_ids = set(macro["lane_wave_runtime_ids"])
            candidates = []
            for index, soldier in enumerate(context["soldiers"][10:]):
                slot = 28 + index
                if (
                    soldier is not None
                    and u.actor_runtime_id(soldier) in wave_ids
                    and target_legal[slot]
                    and _distance_or_inf(
                        context["official_hero"]["side_position"],
                        _position(soldier, self.camp),
                    )
                    <= 12000.0
                ):
                    candidates.append((
                        _distance_or_inf(
                            context["official_hero"]["side_position"],
                            _position(soldier, self.camp),
                        ),
                        slot,
                    ))
            if candidates:
                jungle_lane_target_slot = min(candidates)[1]
        if self.official_task_state is None:
            # Route diagnostics may reuse a semantic camp slot after the base
            # capability pass.  Reapply lifecycle failure locks last so no
            # macro override can reopen a task that exhausted its recovery.
            for monster_index, monster in enumerate(context["monsters"]):
                anchor = tuple(
                    float(value) for value in monster["anchor"]
                )
                lifecycle_id = int(
                    self.monster_camp_history.get(anchor, {}).get(
                        "lifecycle_id", 0
                    )
                )
                if (
                    "farm_camp",
                    anchor,
                    lifecycle_id,
                ) in self.official_task_failed_lifecycles:
                    position_slot = 10 + monster_index
                    valid[position_slot] = False
                    jungle_position_legal[monster_index] = False
        spawn_boundary_x = self._enemy_spawn_boundary_x(context)
        for slot, side_position in enumerate(side_positions):
            if (
                side_position is not None
                and side_position[0] > spawn_boundary_x
            ):
                valid[slot] = False
                if 10 <= slot < 30:
                    jungle_position_legal[slot - 10] = False
        for slot, descriptor in enumerate(descriptors):
            if descriptor["kind"] != "enemy_hero":
                continue
            snapshot = descriptor.get("snapshot")
            actor = descriptor.get("actor")
            target_position = (
                snapshot.get("side_position")
                if snapshot is not None
                else _position(actor, self.camp) if actor is not None else None
            )
            if (
                target_position is not None
                and target_position[0] > spawn_boundary_x
            ):
                target_legal[slot] = False
        raw_positions = tuple(
            u.raw_position(position, self.camp) if position is not None else None
            for position in side_positions
        )
        collaboration = context["collaboration"]
        official_position = context["official_hero"]["side_position"]
        generic_guard_active = bool(
            not (
                official_lane_plan
                and official_lane_plan.get("command_active")
            )
            and not (
                official_config_id == 193
                and (
                    macro.get("opening_rotation_active")
                    or macro.get("opening_fallback_active")
                )
            )
            and self.official_task_state is None
        )
        if generic_guard_active:
            official_runtime = u.actor_runtime_id(
                context["official_hero"]["actor"]
            )
            for slot, runtime_id in enumerate(target_runtime_ids):
                if not runtime_id:
                    continue
                signature = (
                    "generic",
                    int(official_runtime),
                    "target",
                    int(runtime_id),
                )
                same_active = bool(
                    collaboration["active"]
                    and collaboration["type"] == "set_target"
                    and collaboration["target_runtime"] == runtime_id
                )
                legal = bool(
                    not self._official_goal_failed(signature)
                    and (
                        not same_active
                        or self._official_goal_retry_requested(signature)
                    )
                )
                if not legal:
                    target_legal[slot] = False
                    if 8 <= slot < 28:
                        jungle_target_legal[slot - 8] = False
            for slot, side_position in enumerate(side_positions):
                if side_position is None:
                    continue
                signature = (
                    "generic",
                    int(official_runtime),
                    "position",
                    self._rounded_goal_position(side_position),
                )
                same_active = bool(
                    collaboration["active"]
                    and collaboration["type"] == "set_position"
                    and collaboration["command_position"] is not None
                    and _distance_or_inf(
                        collaboration["command_position"], side_position
                    )
                    <= 100.0
                )
                legal = bool(
                    not self._official_goal_failed(signature)
                    and (
                        not same_active
                        or self._official_goal_retry_requested(signature)
                    )
                )
                if not legal:
                    valid[slot] = False
                    if 10 <= slot < 30:
                        jungle_position_legal[slot - 10] = False
        group_candidates = []
        for slot, config_id in enumerate(u.HERO_CONFIG_IDS, 1):
            hero = friend_by_config[config_id]
            if (
                hero["alive"]
                and u.actor_runtime_id(hero["actor"])
                != u.actor_runtime_id(context["official_hero"]["actor"])
            ):
                group_candidates.append((
                    _distance_or_inf(official_position, hero["side_position"]),
                    slot,
                ))
        jungle_group_position_slot = (
            min(group_candidates)[1] if group_candidates else -1
        )
        jungle_defense_position_slot = 6 if context["organs"][0]["alive"] else 7
        enemy_near_official = any(
            enemy["alive"]
            and _distance_or_inf(official_position, enemy["side_position"]) <= 15000.0
            for enemy in context["enemy_heroes"]
        )
        structure_under_attack = any(
            organ["alive"] and organ["actor"] and _target_runtime(organ["actor"])
            for organ in context["organs"][:3]
        )
        team_emergency = bool(
            enemy_near_official
            or structure_under_attack
            or macro["team_emergency"]
            or (
                official_lane_plan
                and official_lane_plan["interrupt"] != "none"
            )
        )
        active_target_legal = bool(
            collaboration["type"] == "set_target"
            and collaboration["target_runtime"]
            and any(
                runtime_id == collaboration["target_runtime"] and legal
                for runtime_id, legal in zip(target_runtime_ids, target_legal)
            )
        )
        official_hp = _hp_ratio(context["official_hero"]["actor"])
        active_goal = (
            collaboration["target_position"]
            if collaboration["type"] == "set_target"
            else collaboration["command_position"]
        )
        unsafe_spawn_boundary_command = bool(
            collaboration["active"]
            and active_goal is not None
            and active_goal[0] > spawn_boundary_x
        )
        spawn_boundary_recovery_active = bool(
            official_alive
            and official_position is not None
            and (
                official_position[0] > spawn_boundary_x
                or unsafe_spawn_boundary_command
            )
        )
        unsafe_enemy_side_command = bool(
            collaboration["active"]
            and official_alive
            and official_position is not None
            and official_position[0] > 0.0
            and active_goal is not None
            and active_goal[0] > 0.0
            and context["organs"][3]["alive"]
        )
        unset_reason = "none"
        if collaboration["active"]:
            if not official_alive:
                unset_reason = "official_dead"
            elif (
                collaboration["type"] == "set_target"
                and not active_target_legal
            ):
                unset_reason = "target_invalid"
            elif official_hp <= 0.28:
                unset_reason = "low_hp_active_command"
            elif unsafe_spawn_boundary_command and (
                not collaboration["cooldown_ready"] or not valid[7]
            ):
                unset_reason = "unsafe_spawn_boundary"
            elif unsafe_enemy_side_command:
                unset_reason = "unsafe_enemy_side_command"
        unset_allowed = unset_reason != "none"
        return CommandRuntimeContextV1(
            camp=self.camp,
            frame_no=context["frame_no"],
            state_valid=collaboration["state_valid"],
            official_runtime_id=u.actor_runtime_id(context["official_hero"]["actor"]),
            official_config_id=u.actor_config_id(context["official_hero"]["actor"]),
            official_alive=official_alive,
            team_emergency=team_emergency,
            cooldown_ready=collaboration["cooldown_ready"],
            current_command_active=collaboration["active"],
            unset_allowed=unset_allowed,
            unset_reason=unset_reason,
            target_runtime_ids=tuple(target_runtime_ids),
            collaboration_target_legal=tuple(target_legal),
            position_raw_coordinates=raw_positions,
            position_valid=tuple(valid),
            jungle_target_legal=tuple(jungle_target_legal),
            jungle_position_legal=tuple(jungle_position_legal),
            jungle_route_active=bool(
                macro["valid"]
                and official_alive
                and (
                    macro.get("opening_rotation_active")
                    or macro.get("opening_fallback_active")
                    or macro.get("interrupt") != "none"
                )
            ),
            jungle_route_phase=macro["phase"],
            jungle_route_interrupt=macro["interrupt"],
            jungle_route_target_slot=route_command_target_slot,
            jungle_route_position_slot=route_position_slot,
            jungle_route_intruder_runtime=macro["intruder_runtime"],
            jungle_lane_target_slot=jungle_lane_target_slot,
            jungle_route_mode=macro["mode"],
            jungle_camp_exit_active=bool(macro["camp_exit_active"]),
            jungle_opening_rotation_active=bool(
                macro["opening_rotation_active"]
            ),
            jungle_group_position_slot=jungle_group_position_slot,
            jungle_defense_position_slot=jungle_defense_position_slot,
            lane_route_active=bool(
                official_lane_plan
                and official_lane_plan["command_active"]
                and official_alive
            ),
            lane_route_phase=official_lane_plan["phase"]
            if official_lane_plan
            else "inactive",
            lane_route_interrupt=official_lane_plan["interrupt"]
            if official_lane_plan
            else "none",
            lane_route_target_slot=official_lane_plan["target_slot"]
            if official_lane_plan
            and official_lane_plan["row_entry_reached"]
            and official_lane_plan["phase"]
            in ("stage_outer", "engage_wave", "follow_wave", "released")
            and official_lane_plan["target_distance"]
            <= LANE_OFFICIAL_TARGET_COMMAND_DISTANCE
            and not self._official_goal_failed(
                self._lane_target_goal_signature(
                    u.actor_runtime_id(
                        context["official_hero"]["actor"]
                    ),
                    u.actor_runtime_id(
                        official_lane_plan["target_soldier"]
                    )
                    if official_lane_plan["target_soldier"]
                    is not None
                    else 0,
                )
            )
            and (
                not (
                    collaboration["active"]
                    and collaboration["type"] == "set_target"
                    and collaboration["target_runtime"]
                    == (
                        u.actor_runtime_id(
                            official_lane_plan["target_soldier"]
                        )
                        if official_lane_plan["target_soldier"]
                        is not None
                        else 0
                    )
                )
                or self._official_goal_retry_requested(
                    self._lane_target_goal_signature(
                        u.actor_runtime_id(
                            context["official_hero"]["actor"]
                        ),
                        u.actor_runtime_id(
                            official_lane_plan["target_soldier"]
                        )
                        if official_lane_plan["target_soldier"]
                        is not None
                        else 0,
                    )
                )
            )
            else -1,
            lane_route_position_slot=5,
            lane_route_intruder_runtime=official_lane_plan["intruder_runtime"]
            if official_lane_plan
            else 0,
            lane_jungle_unlocked=not context["organs"][3]["alive"],
            generic_command_fallback_pending=bool(
                self.official_command_guard["fallback_pending"]
                and isinstance(
                    self.official_command_guard[
                        "fallback_source_signature"
                    ],
                    tuple,
                )
                and self.official_command_guard[
                    "fallback_source_signature"
                ][:1]
                == ("generic",)
            ),
            official_task_active=bool(
                self.official_task_state is not None
            ),
            official_task_intent=int(
                (self.official_task_state or {}).get("intent", 0)
            ),
            official_task_position_slot=int(
                (self.official_task_state or {}).get(
                    "position_slot", -1
                )
            ),
            official_task_phase=str(
                (self.official_task_state or {}).get(
                    "phase", "inactive"
                )
            ),
            official_task_force_refresh=bool(
                (self.official_task_state or {}).get(
                    "force_refresh", False
                )
            ),
            spawn_boundary_recovery_active=spawn_boundary_recovery_active,
            spawn_boundary_safe_position_slot=7,
        )

    @staticmethod
    def _rounded_goal_position(position):
        if position is None:
            return None
        return tuple(int(round(float(value))) for value in position)

    def _lane_position_goal_signature(
        self, official_runtime, route_node, side_position
    ):
        return (
            "lane",
            int(official_runtime),
            "position",
            int(route_node),
            self._rounded_goal_position(side_position),
        )

    @staticmethod
    def _lane_target_goal_signature(official_runtime, target_runtime):
        return (
            "lane",
            int(official_runtime),
            "target",
            int(target_runtime),
        )

    def _jungle_position_goal_signature(
        self,
        official_runtime,
        camp_name,
        lifecycle_id,
        side_position,
        fallback=False,
    ):
        return (
            "jungle_fallback" if fallback else "jungle",
            int(official_runtime),
            str(camp_name),
            int(lifecycle_id),
            "position",
            self._rounded_goal_position(side_position),
        )

    def _generic_goal_signature_from_action(
        self, official_runtime, command_action
    ):
        action_type = int(command_action[0])
        if action_type == 1:
            return (
                "generic",
                int(official_runtime),
                "target",
                int(command_action[1]),
            )
        if action_type == 2:
            side_position = u.side_position(
                (float(command_action[2]), float(command_action[3])),
                self.camp,
            )
            return (
                "generic",
                int(official_runtime),
                "position",
                self._rounded_goal_position(side_position),
            )
        return None

    def _forced_goal_signature_from_action(self, command_action):
        """Bind a forced protocol action to its role/lifecycle goal."""

        if self._memo_context is None or not command_action:
            return None
        action_type = int(command_action[0])
        if action_type not in (1, 2):
            return None
        context = self._memo_context
        official = context["official_hero"]
        official_runtime = u.actor_runtime_id(official["actor"])
        official_config = u.actor_config_id(official["actor"])
        lane_plan = (context.get("lane_macro") or {}).get(official_config)
        if (
            official_config in LANE_ROW_SIDE_Z
            and lane_plan
            and lane_plan.get("command_active")
        ):
            if action_type == 1:
                return self._lane_target_goal_signature(
                    official_runtime, int(command_action[1])
                )
            side_position = u.side_position(
                (float(command_action[2]), float(command_action[3])),
                self.camp,
            )
            return self._lane_position_goal_signature(
                official_runtime,
                int(lane_plan.get("route_node", 2)),
                side_position,
            )

        macro = context.get("jungle_macro") or {}
        if official_config == 193 and (
            macro.get("opening_rotation_active")
            or macro.get("opening_fallback_active")
        ):
            if action_type != 2:
                return None
            side_position = u.side_position(
                (float(command_action[2]), float(command_action[3])),
                self.camp,
            )
            return self._jungle_position_goal_signature(
                official_runtime,
                macro.get("opening_camp_name", ""),
                int(macro.get("opening_camp_lifecycle_id", 0)),
                side_position,
                fallback=bool(macro.get("opening_fallback_active")),
            )
        return self._generic_goal_signature_from_action(
            official_runtime, command_action
        )

    def _official_goal_failed(self, signature):
        return bool(
            signature is not None
            and signature
            in self.official_command_guard["failed_signatures"]
        )

    def _official_goal_retry_requested(self, signature):
        return bool(
            signature is not None
            and signature
            in self.official_command_guard["retry_requested"]
        )

    def record_collaboration_command(
        self,
        frame_no,
        command_action,
        intent=0,
        is_train=False,
        target_slot=0,
        position_slot=0,
    ):
        """Record a command until platform state accepts or rejects it.

        ``is_train`` distinguishes a real policy choice from a forced opening
        action.  Only a real choice can create a reward-attribution window;
        both kinds still use the same platform echo/cooldown protection.
        """

        action_type = int(command_action[0]) if command_action else 0
        self.last_command_decision = {
            "frame": int(frame_no),
            "action_type": action_type,
            "intent": int(intent),
            "is_train": bool(is_train),
            "target_slot": int(target_slot),
            "position_slot": int(position_slot),
        }
        if self._memo_context is not None and action_type == 2:
            official_config = u.actor_config_id(
                self._memo_context["official_hero"]["actor"]
            )
            lane_plan = self._memo_context["lane_macro"].get(
                official_config, {}
            )
            if lane_plan.get("deer_assist_active"):
                previous_count = int(
                    self.official_task_counters.get(
                        "official_deer_command", 0
                    )
                )
                self.official_task_counters[
                    "official_deer_command"
                ] += 1.0
                if previous_count > 0:
                    self.official_task_counters[
                        "official_deer_renewal"
                    ] += 1.0
            elif lane_plan.get("deer_return_active"):
                self.official_task_counters[
                    "official_deer_successor"
                ] += 1.0
        if (
            bool(is_train)
            and int(intent) == 10
            and action_type == 2
            and self.official_task_state is None
        ):
            self._start_official_farm_task(frame_no, position_slot)
        elif (
            not bool(is_train)
            and action_type == 2
            and self.official_task_state is not None
            and int(position_slot)
            == int(self.official_task_state.get("position_slot", -1))
        ):
            self.official_task_state["renewal_count"] = int(
                self.official_task_state.get("renewal_count", 0)
            ) + 1
            self.official_task_counters["task_renewal"] += 1.0
            if self.official_task_state.get("force_refresh", False):
                self.official_task_state["force_refresh"] = False
                self.official_task_counters[
                    "task_forced_refresh"
                ] += 1.0
        if action_type in (1, 2):
            goal_signature = self._forced_goal_signature_from_action(
                command_action
            )
            attempt = 0
            if goal_signature is not None:
                attempts = self.official_command_guard["attempts"]
                attempt = int(attempts.get(goal_signature, 0)) + 1
                attempts[goal_signature] = attempt
                self.official_command_guard["retry_requested"].discard(
                    goal_signature
                )
            self.local_last_command_frame = max(
                int(self.local_last_command_frame), int(frame_no)
            )
            self.pending_collaboration_command = {
                "frame": int(frame_no),
                "action_type": action_type,
                "target_runtime": int(command_action[1]),
                "raw_position": (
                    float(command_action[2]), float(command_action[3])
                ),
                "intent": int(intent),
                "is_train": bool(is_train),
                "goal_signature": goal_signature,
                "attempt": attempt,
            }
        elif action_type == 3:
            # Unset deliberately ends responsibility for the preceding
            # command.  The platform state still decides cooldown/legality.
            self.active_command_attribution = None

    def consume_jungle_route_shaping(self, expected_frame=None):
        """Return current route shaping once, in controlled-actor order."""

        if self._memo_context is None:
            return (0.0,) * FrozenV1Config.CONTROLLED_ACTOR_COUNT
        frame_no = self._memo_context["frame_no"]
        if expected_frame is not None and int(expected_frame) != frame_no:
            return (0.0,) * FrozenV1Config.CONTROLLED_ACTOR_COUNT
        if self._route_shaping_consumed_frame == frame_no:
            return (0.0,) * FrozenV1Config.CONTROLLED_ACTOR_COUNT
        self._route_shaping_consumed_frame = frame_no
        reward = float(self._memo_context["jungle_macro"]["route_shaping_reward"])
        return tuple(
            reward
            if u.as_int(u.safe_get(info, "config_id", default=0), 0) == 193
            else 0.0
            for info in self._memo_context["controlled_infos"]
        )

    def _lane_attendance_status(self, context, config_id, hero):
        """Return the early-lane soft objective without forcing an action."""

        inactive = {
            "active": False,
            "near": False,
            "goal": None,
            "distance": float("inf"),
        }
        if config_id not in LANE_ROW_SIDE_Z or hero is None or not hero["alive"]:
            return inactive
        plan = context["lane_macro"].get(config_id)
        history = self.lane_opening_history.get(config_id, {})
        if (
            plan is None
            or not history.get("released", False)
            or context["frame_no"] > V1TrainingConfig.LANE_ATTENDANCE_END_FRAME
            or _hp_ratio(hero["actor"])
            <= V1TrainingConfig.NO_ATTACKABLE_TARGET_RECOVERY_HP_RATIO
            or plan["interrupt"] != "none"
            or not context["organs"][0]["alive"]
            or not context["organs"][3]["alive"]
        ):
            return inactive
        tyrant = self._tyrant_status(context)
        if (
            tyrant["alive"]
            or tyrant["spawn_within_30"]
            or tyrant["team_attacking"]
            or tyrant["enemy_attacking"]
        ):
            return inactive
        if any(
            organ["alive"] and organ["actor"] and _target_runtime(organ["actor"])
            for organ in context["organs"][:3]
        ):
            return inactive

        row = context["lane_summary"]["rows"][LANE_ROW_NAMES[config_id]]
        own_front = row["friend_front"][1] if row["friend_front"] else None
        enemy_front = row["enemy_front"][1] if row["enemy_front"] else None
        clash = row["clash_position"]
        if own_front is not None:
            goal = (min(own_front[0] - 2500.0, 2500.0), own_front[1])
        elif clash is not None:
            goal = (clash[0] - 2500.0, clash[1])
        elif enemy_front is not None:
            goal = (enemy_front[0] - 5000.0, enemy_front[1])
        else:
            goal = (0.0, LANE_ROW_SIDE_Z[config_id])
        distance = _distance_or_inf(hero["side_position"], goal)
        return {
            "active": bool(hero["side_position"] is not None),
            "near": distance <= V1TrainingConfig.LANE_ATTENDANCE_NEAR_DISTANCE,
            "goal": goal,
            "distance": distance,
        }

    def _consume_lane_attendance_reward(self, context, config_id, hero, status):
        runtime_id = u.actor_runtime_id(hero["actor"]) if hero else 0
        state = self.lane_attendance_history.get(config_id)
        if state is None or state.get("runtime_id") != runtime_id:
            state = {
                "runtime_id": runtime_id,
                "frame": None,
                "position": None,
            }
            self.lane_attendance_history[config_id] = state
        position = hero["side_position"] if hero else None
        frame_no = context["frame_no"]
        reward = 0.0
        if status["active"] and position is not None and state["position"] is not None:
            elapsed = frame_no - state["frame"] if state["frame"] is not None else 0
            if 0 < elapsed <= V1TrainingConfig.NO_ATTACKABLE_TARGET_MAX_ELAPSED_FRAMES:
                previous_distance = _distance_or_inf(state["position"], status["goal"])
                current_distance = status["distance"]
                if max(previous_distance, current_distance) > V1TrainingConfig.LANE_ATTENDANCE_NEAR_DISTANCE:
                    progress = (previous_distance - current_distance) / float(
                        V1TrainingConfig.LANE_ATTENDANCE_PROGRESS_DISTANCE
                    )
                    reward = u.signed_clip(progress) * float(
                        V1TrainingConfig.LANE_ATTENDANCE_PROGRESS_REWARD
                    )
        if status["active"] and position is not None:
            state["frame"] = frame_no
            state["position"] = position
        else:
            state["frame"] = None
            state["position"] = None
        return float(reward)

    def _strategic_wait_active(self, context, config_id, hero, attendance):
        actor = hero["actor"]
        position = hero["side_position"]
        plan = context["lane_macro"].get(config_id)
        if plan and (
            plan["waiting_for_wave"]
            or plan["structure_defense"]
        ):
            return True
        if bool(u.actor_get(actor, "is_in_grass", "isInGrass", default=False)):
            return True
        if attendance["active"] and attendance["near"]:
            return True
        for organ in context["organs"][:3]:
            if (
                organ["alive"]
                and organ["actor"]
                and _target_runtime(organ["actor"])
                and _distance_or_inf(position, organ["side_position"])
                <= V1TrainingConfig.STRATEGIC_WAIT_RADIUS
            ):
                return True
        tyrant = self._tyrant_status(context)
        if (
            tyrant["position"] is not None
            and _distance_or_inf(position, tyrant["position"])
            <= V1TrainingConfig.STRATEGIC_WAIT_RADIUS
            and (
                tyrant["spawn_within_30"]
                or tyrant["team_attacking"]
                or tyrant["enemy_attacking"]
            )
        ):
            return True
        if config_id == 193:
            macro = context.get("jungle_macro", {})
            target_slot = macro.get("target_slot")
            live_committed_resource = bool(
                macro.get("valid")
                and macro.get("interrupt") == "none"
                and target_slot is not None
                and 0 <= int(target_slot) < len(context["monsters"])
                and context["monsters"][int(target_slot)]["alive"]
            )
            if (
                macro.get("opening_rotation_active")
                or (
                    live_committed_resource
                    and macro.get("route_progress_confirmed", False)
                    and float(macro.get("path_distance", 0.0))
                    > V1TrainingConfig.STRATEGIC_WAIT_RADIUS
                )
            ):
                return True
        for monster, spawn in zip(context["monsters"], context["monster_spawn"]):
            if monster["alive"]:
                continue
            if monster["config_id"] != V1TrainingConfig.TYRANT_CONFIG_ID:
                continue
            anchor_key = tuple(float(value) for value in monster["anchor"])
            if (
                self._spawn_seconds(spawn)
                <= V1TrainingConfig.STRATEGIC_CAMP_WAIT_SECONDS
                and _distance_or_inf(
                    position, u.side_position(monster["anchor"], self.camp)
                )
                <= V1TrainingConfig.STRATEGIC_WAIT_RADIUS
            ):
                return True
        return False

    def _consume_no_target_penalty(
        self,
        context,
        runtime_id,
        eligible,
        normal_attackable,
        strategic_wait,
    ):
        state = self.no_target_history.get(runtime_id)
        if state is None:
            state = {"last_frame": None, "no_target_frames": 0}
            self.no_target_history[runtime_id] = state
        frame_no = context["frame_no"]
        elapsed = frame_no - state["last_frame"] if state["last_frame"] is not None else 0
        state["last_frame"] = frame_no
        if not eligible or normal_attackable or strategic_wait:
            state["no_target_frames"] = 0
            return 0.0, False, bool(strategic_wait and eligible and not normal_attackable), False
        if elapsed > 0:
            state["no_target_frames"] += min(
                elapsed,
                V1TrainingConfig.NO_ATTACKABLE_TARGET_MAX_ELAPSED_FRAMES,
            )
        if state["no_target_frames"] < V1TrainingConfig.NO_ATTACKABLE_TARGET_GRACE_FRAMES:
            return 0.0, False, False, True
        normalized_elapsed = min(
            max(elapsed, 0),
            V1TrainingConfig.NO_ATTACKABLE_TARGET_MAX_ELAPSED_FRAMES,
        ) / float(V1TrainingConfig.NO_ATTACKABLE_TARGET_REFERENCE_FRAMES)
        if normalized_elapsed <= 0.0:
            return 0.0, False, False, True
        penalty = -float(V1TrainingConfig.NO_ATTACKABLE_TARGET_PENALTY) * normalized_elapsed
        return penalty, True, False, False

    @staticmethod
    def _hero_kda_counters(actor):
        return {
            "kill": max(0, _integer(actor, "kill_cnt", "killCnt")),
            "death": max(0, _integer(actor, "dead_cnt", "deadCnt")),
            "assist": max(0, _integer(actor, "assist_cnt", "assistCnt")),
        }

    def _build_zero_sum_kda_rewards(self, context):
        """Build per-hero K/D/A increments against the enemy-team mean.

        The same calculation from the opposite camp negates the team-average
        reward exactly.  Cumulative counters are used so a boundary that spans
        several raw frames still observes every event once.
        """

        deltas = {}
        for actor in context["raw_heroes"]:
            camp = u.actor_camp(actor)
            config_id = u.actor_config_id(actor)
            if camp not in (1, 2) or config_id not in u.HERO_CONFIG_IDS:
                continue
            key = (camp, config_id)
            counters = self._hero_kda_counters(actor)
            previous = self.previous_reward_kda_counters.get(key)
            if previous is None or any(
                counters[name] < previous[name]
                for name in ("kill", "death", "assist")
            ):
                delta = {"kill": 0, "death": 0, "assist": 0}
            else:
                delta = {
                    name: counters[name] - previous[name]
                    for name in ("kill", "death", "assist")
                }
            self.previous_reward_kda_counters[key] = counters
            deltas[key] = delta

        empty = {"kill": 0, "death": 0, "assist": 0}
        friend_deltas = [
            deltas.get((self.camp, config_id), empty)
            for config_id in u.HERO_CONFIG_IDS
        ]
        enemy_deltas = [
            deltas.get((3 - self.camp, config_id), empty)
            for config_id in u.HERO_CONFIG_IDS
        ]
        enemy_means = {
            name: sum(delta[name] for delta in enemy_deltas) / 3.0
            for name in ("kill", "death", "assist")
        }
        death_multiplier = (
            V1TrainingConfig.DEATH_MULTIPLIER_BEFORE_CANNON
            if context["frame_no"]
            < V1TrainingConfig.DEATH_MULTIPLIER_CANNON_FRAME
            else V1TrainingConfig.DEATH_MULTIPLIER_AFTER_CANNON
        )

        reward_by_runtime = {}
        reward_by_config = {}
        for config_id, delta in zip(u.HERO_CONFIG_IDS, friend_deltas):
            reward = (
                V1TrainingConfig.KILL_REWARD
                * (delta["kill"] - enemy_means["kill"])
                + V1TrainingConfig.DEATH_REWARD
                * death_multiplier
                * (delta["death"] - enemy_means["death"])
                + V1TrainingConfig.ASSIST_REWARD
                * (delta["assist"] - enemy_means["assist"])
            )
            reward_by_config[config_id] = float(reward)
            actor = next(
                (
                    candidate
                    for candidate in context["raw_heroes"]
                    if u.actor_camp(candidate) == self.camp
                    and u.actor_config_id(candidate) == config_id
                ),
                None,
            )
            runtime_id = u.actor_runtime_id(actor)
            if runtime_id:
                reward_by_runtime[runtime_id] = float(reward)

        return reward_by_runtime, reward_by_config

    @staticmethod
    def _killer_camp_for_event(context, event):
        killer = u.safe_get(event, "killer", "killer_actor", default={}) or {}
        killer_camp = u.actor_camp(killer)
        if killer_camp in (1, 2):
            return killer_camp
        killer_runtime = u.actor_runtime_id(killer)
        if killer_runtime:
            killer_actor = context["actor_by_runtime"].get(killer_runtime)
            killer_camp = u.actor_camp(killer_actor)
            if killer_camp in (1, 2):
                return killer_camp
        death = u.safe_get(event, "death", "dead", default={}) or {}
        hurt_list = list(
            u.actor_get(
                death, "single_hurt_list", "singleHurtList", default=[]
            )
            or []
        )
        if hurt_list:
            last_hurt = max(
                hurt_list,
                key=lambda item: u.as_int(
                    u.safe_get(
                        item, "frameNo", "frame_no", "frame", default=-1
                    ),
                    -1,
                ),
            )
            hurt_runtime = u.as_int(
                u.safe_get(last_hurt, "runtime_id", "runtimeId", default=0),
                0,
            )
            hurt_actor = context["actor_by_runtime"].get(hurt_runtime)
            hurt_camp = u.actor_camp(hurt_actor)
            if hurt_camp in (1, 2):
                return hurt_camp
        return 0

    def _consume_tyrant_team_reward(self, context, dead_actions):
        state = self.tyrant_reward_state
        tyrant = context.get("tyrant") or {}
        if tyrant.get("alive", False):
            runtime_id = u.actor_runtime_id(tyrant.get("actor"))
            if state["observed_dead"] or (
                runtime_id
                and state["runtime_id"]
                and runtime_id != state["runtime_id"]
            ):
                state["rewarded"] = False
            state["observed_dead"] = False
            if runtime_id:
                state["runtime_id"] = runtime_id

        reward = 0.0
        for event in dead_actions:
            death = u.safe_get(event, "death", "dead", default={}) or {}
            if u.actor_config_id(death) != V1TrainingConfig.TYRANT_CONFIG_ID:
                continue
            death_runtime = u.actor_runtime_id(death)
            if state["rewarded"] and (
                not death_runtime or death_runtime == state["runtime_id"]
            ):
                state["observed_dead"] = True
                continue
            killer_camp = self._killer_camp_for_event(context, event)
            if killer_camp not in (1, 2):
                continue
            kill_counts = state["kill_counts"]
            kill_counts[killer_camp] += 1
            if (
                kill_counts[killer_camp]
                <= V1TrainingConfig.TYRANT_REWARD_MAX_KILLS_PER_CAMP
            ):
                reward += (
                    V1TrainingConfig.TYRANT_TEAM_REWARD
                    if killer_camp == self.camp
                    else -V1TrainingConfig.TYRANT_TEAM_REWARD
                )
            state["rewarded"] = True
            state["observed_dead"] = True
            if death_runtime:
                state["runtime_id"] = death_runtime
        return float(reward)

    def _consume_kai_wolf_team_reward(self, context, dead_actions):
        """Reward post-opening wolf lifecycles, independent of killer hero."""

        for actor in context["monsters_raw"]:
            config_id = u.actor_config_id(actor)
            actor_camp = u.actor_camp(actor)
            if config_id not in V1TrainingConfig.KAI_WOLF_MEMBER_REWARDS:
                continue
            if actor_camp not in (1, 2):
                continue
            relation = "own" if actor_camp == self.camp else "enemy"
            anchor = tuple(
                float(value)
                for value in u.fixed_monster_anchor(44, relation, self.camp)
            )
            camp_state = self.monster_camp_history.get(anchor, {})
            if int(camp_state.get("clear_count", 0)) == 0:
                runtime_id = u.actor_runtime_id(actor)
                if runtime_id:
                    self.initial_wolf_runtime_ids.add(runtime_id)

        reward = 0.0
        count = 0
        for event in dead_actions:
            death = u.safe_get(event, "death", "dead", default={}) or {}
            config_id = u.actor_config_id(death)
            member_reward = V1TrainingConfig.KAI_WOLF_MEMBER_REWARDS.get(
                config_id
            )
            if member_reward is None:
                continue

            runtime_id = u.actor_runtime_id(death)
            signature = (
                ("runtime", runtime_id)
                if runtime_id
                else (
                    "fallback",
                    config_id,
                    u.actor_camp(death),
                    context["frame_no"],
                )
            )
            if signature in self.rewarded_kai_wolf_deaths:
                continue

            killer_camp = self._killer_camp_for_event(context, event)
            if killer_camp not in (1, 2):
                continue

            death_camp = u.actor_camp(death)
            member_key = (death_camp, config_id)
            relation = (
                "own"
                if death_camp == self.camp
                else "enemy"
                if death_camp in (1, 2)
                else None
            )
            clear_count = 0
            if relation is not None:
                anchor = tuple(
                    float(value)
                    for value in u.fixed_monster_anchor(
                        44, relation, self.camp
                    )
                )
                clear_count = int(
                    self.monster_camp_history.get(anchor, {}).get(
                        "clear_count", 0
                    )
                )
            initial_lifecycle = bool(
                runtime_id in self.initial_wolf_runtime_ids
                or (
                    member_key not in self.initial_wolf_member_deaths
                    and clear_count <= 1
                )
            )
            if initial_lifecycle:
                self.initial_wolf_member_deaths.add(member_key)
                self.rewarded_kai_wolf_deaths.add(signature)
                continue

            self.rewarded_kai_wolf_deaths.add(signature)
            reward += (
                member_reward
                if killer_camp == self.camp
                else -member_reward
            )
            count += 1
        return float(reward), int(count)

    def _consume_treasure_thief_team_reward(self, context, dead_actions):
        reward = 0.0
        count = 0
        for event in dead_actions:
            death = u.safe_get(event, "death", "dead", default={}) or {}
            if (
                u.actor_config_id(death)
                != V1TrainingConfig.TREASURE_THIEF_CONFIG_ID
            ):
                continue
            runtime_id = u.actor_runtime_id(death)
            signature = (
                ("runtime", runtime_id)
                if runtime_id
                else (
                    "fallback",
                    V1TrainingConfig.TREASURE_THIEF_CONFIG_ID,
                    context["frame_no"],
                )
            )
            if signature in self.rewarded_treasure_thief_deaths:
                continue
            killer_camp = self._killer_camp_for_event(context, event)
            if killer_camp not in (1, 2):
                continue
            self.rewarded_treasure_thief_deaths.add(signature)
            reward += (
                V1TrainingConfig.TREASURE_THIEF_TEAM_REWARD
                if killer_camp == self.camp
                else -V1TrainingConfig.TREASURE_THIEF_TEAM_REWARD
            )
            count += 1
        return float(reward), int(count)

    @staticmethod
    def _speed_buff_signatures(actor):
        signatures = set()
        for item in u.buff_items(actor):
            if (
                u.as_int(
                    u.safe_get(item, "config_id", "configId", default=0),
                    0,
                )
                != V1TrainingConfig.SPEED_BUFF_CONFIG_ID
            ):
                continue
            signatures.add(
                (
                    V1TrainingConfig.SPEED_BUFF_CONFIG_ID,
                    u.as_int(
                        u.safe_get(
                            item,
                            "start_time",
                            "startTime",
                            default=0,
                        ),
                        0,
                    ),
                    u.as_int(
                        u.safe_get(item, "times", default=0),
                        0,
                    ),
                )
            )
        return signatures

    def _consume_speed_buff_team_reward(self, context):
        reward = 0.0
        count = 0
        for hero in context["heroes"]:
            if not hero["current_valid"]:
                continue
            actor = hero["actor"]
            runtime_id = u.actor_runtime_id(actor)
            actor_camp = u.actor_camp(actor)
            if not runtime_id or actor_camp not in (1, 2):
                continue
            key = (actor_camp, runtime_id)
            current = self._speed_buff_signatures(actor)
            previous = self.speed_buff_reward_state.get(key)
            self.speed_buff_reward_state[key] = current
            if previous is None:
                continue
            acquisitions = current - previous
            if not acquisitions:
                continue
            direction = 1.0 if actor_camp == self.camp else -1.0
            reward += (
                direction
                * V1TrainingConfig.SPEED_BUFF_TEAM_REWARD
                * len(acquisitions)
            )
            count += len(acquisitions)
        return float(reward), int(count)

    def _consume_crystal_outcome_reward(self, context, dead_actions):
        if self.crystal_outcome_rewarded:
            return 0.0
        for event in dead_actions:
            death = u.safe_get(event, "death", "dead", default={}) or {}
            config_id = u.actor_config_id(death)
            sub_type = u.normalize_sub_type(
                u.actor_get(death, "sub_type", "subType", default=None)
            )
            if config_id not in (68, 72) and sub_type != "ACTOR_SUB_CRYSTAL":
                continue
            crystal_camp = u.actor_camp(death)
            if crystal_camp not in (1, 2):
                crystal_camp = 1 if config_id == 68 else 2 if config_id == 72 else 0
            if crystal_camp not in (1, 2):
                continue
            self.crystal_outcome_rewarded = True
            return float(
                V1TrainingConfig.TERMINAL_CRYSTAL_REWARD
                if crystal_camp != self.camp
                else -V1TrainingConfig.TERMINAL_CRYSTAL_REWARD
            )
        for organ in context.get("organs", ()):
            if organ.get("type") != "crystal" or organ.get("actor") is None:
                continue
            if organ.get("alive", False):
                continue
            crystal_camp = u.actor_camp(organ["actor"])
            if crystal_camp not in (1, 2):
                continue
            self.crystal_outcome_rewarded = True
            return float(
                V1TrainingConfig.TERMINAL_CRYSTAL_REWARD
                if crystal_camp != self.camp
                else -V1TrainingConfig.TERMINAL_CRYSTAL_REWARD
            )
        return 0.0

    @staticmethod
    def _structure_curve(hp_ratio):
        hp_ratio = u.clip(hp_ratio)
        missing_hp = 1.0 - hp_ratio
        return hp_ratio - V1TrainingConfig.STRUCTURE_POTENTIAL_BETA * (
            missing_hp ** V1TrainingConfig.STRUCTURE_POTENTIAL_POWER
        )

    def _tower_has_reliable_minion_cover(self, context, organ):
        if not organ["alive"] or organ["actor"] is None:
            return False
        attack_range = _value(
            organ["actor"], "attack_range", "attackRange"
        )
        if attack_range <= 0.0:
            return False
        friend_soldiers = tuple(context["lane_summary"]["friend"])
        friend_runtime_ids = {
            u.actor_runtime_id(soldier)
            for soldier in friend_soldiers
            if u.is_alive(soldier)
        }
        if _target_runtime(organ["actor"]) in friend_runtime_ids:
            return True
        return any(
            u.is_alive(soldier)
            and _position(soldier, self.camp) is not None
            and _distance_or_inf(
                organ["side_position"],
                _position(soldier, self.camp),
            )
            <= attack_range
            for soldier in friend_soldiers
        )

    def _consume_tower_risk_reward(self, context):
        """Advance warning/danger/critical and charge unique tower hits."""

        rewards = np.zeros(
            FrozenV1Config.CONTROLLED_ACTOR_COUNT, dtype=np.float32
        )
        hits = np.zeros_like(rewards)
        phases = []
        frame_no = context["frame_no"]
        controlled_runtime_ids = {
            u.as_int(u.safe_get(info, "runtime_id", default=0), 0)
            for info in context["controlled_infos"]
        }
        for stale_runtime in tuple(self.tower_risk_state):
            if stale_runtime not in controlled_runtime_ids:
                self.tower_risk_state.pop(stale_runtime, None)
                self.tower_risk_counters["reset_runtime_change"] += 1.0
        alive_tower_runtimes = {
            u.actor_runtime_id(organ["actor"])
            for organ in context["organs"][3:5]
            if organ["alive"] and organ["actor"] is not None
        }
        for actor_index, info in enumerate(context["controlled_infos"]):
            runtime_id = u.as_int(
                u.safe_get(info, "runtime_id", default=0), 0
            )
            hero = context["hero_by_runtime"].get(runtime_id)
            if (
                hero is None
                or not hero["alive"]
                or hero["side_position"] is None
            ):
                if self.tower_risk_state.pop(runtime_id, None) is not None:
                    self.tower_risk_counters[
                        "reset_actor_invalid"
                    ] += 1.0
                phases.append("inactive")
                continue
            state = self.tower_risk_state.get(runtime_id)
            if (
                state is not None
                and int(state.get("tower_runtime", 0))
                not in alive_tower_runtimes
            ):
                self.tower_risk_state.pop(runtime_id, None)
                self.tower_risk_counters[
                    "reset_tower_invalid"
                ] += 1.0
                state = None
            candidates = []
            for organ in context["organs"][3:5]:
                if not organ["alive"] or organ["actor"] is None:
                    continue
                attack_range = _value(
                    organ["actor"], "attack_range", "attackRange"
                )
                if attack_range <= 0.0:
                    continue
                distance = _distance_or_inf(
                    hero["side_position"], organ["side_position"]
                )
                if distance <= attack_range:
                    candidates.append((distance - attack_range, organ))
            if not candidates:
                if state is not None:
                    outside_since = state.get("outside_since")
                    if outside_since is None:
                        state["outside_since"] = frame_no
                    elif (
                        frame_no - int(outside_since)
                        >= V1TrainingConfig.TOWER_RISK_EXIT_FRAMES
                    ):
                        self.tower_risk_state.pop(runtime_id, None)
                        self.tower_risk_counters[
                            "reset_outside"
                        ] += 1.0
                        state = None
                phases.append(
                    str(state.get("phase", "inactive"))
                    if state is not None
                    else "inactive"
                )
                continue

            _distance, tower = min(candidates, key=lambda item: item[0])
            tower_runtime = u.actor_runtime_id(tower["actor"])
            if self._tower_has_reliable_minion_cover(context, tower):
                if self.tower_risk_state.pop(runtime_id, None) is not None:
                    self.tower_risk_counters[
                        "reset_minion_cover"
                    ] += 1.0
                phases.append("inactive")
                continue
            if (
                state is None
                or int(state.get("tower_runtime", 0)) != tower_runtime
            ):
                state = {
                    "tower_runtime": tower_runtime,
                    "tower_config_id": tower["config_id"],
                    "tower_position": tower["side_position"],
                    "enter_frame": frame_no,
                    "outside_since": None,
                    "phase": "warning",
                    "hit_count": 0,
                    "penalty_total": 0.0,
                    "seen_hit_signatures": set(),
                }
                self.tower_risk_state[runtime_id] = state
                self.tower_risk_counters["event_start"] += 1.0
            else:
                state["outside_since"] = None
                state["tower_position"] = tower["side_position"]

            previous_phase = str(state["phase"])
            tower_locked = _target_runtime(tower["actor"]) == runtime_id
            if tower_locked:
                state["phase"] = (
                    "critical"
                    if _hp_ratio(hero["actor"])
                    <= V1TrainingConfig.TOWER_RISK_CRITICAL_HP_RATIO
                    else "danger"
                )
            elif state["phase"] == "warning":
                state["phase"] = "warning"
            if str(state["phase"]) != previous_phase:
                self.tower_risk_counters[
                    f"enter_{state['phase']}"
                ] += 1.0
            self.tower_risk_counters[
                f"phase_{state['phase']}_frames"
            ] += 1.0

            key = ("friend", u.actor_config_id(hero["actor"]))
            tower_hits = [
                event
                for event in self.recent_events.get(key, ())
                if event["frame"] == frame_no
                and event["type"] == "hurt_by_tower"
                and event["amount"] > 0.0
                and (
                    event["other_runtime"] == tower_runtime
                    or event["other_runtime"] == 0
                )
            ]
            for _event in tower_hits:
                hit_signature = (
                    int(_event["frame"]),
                    int(tower_runtime),
                    round(float(_event["amount"]), 3),
                )
                if hit_signature in state["seen_hit_signatures"]:
                    self.tower_risk_counters[
                        "duplicate_hit_suppressed"
                    ] += 1.0
                    continue
                state["seen_hit_signatures"].add(hit_signature)
                hit_index = int(state["hit_count"])
                magnitude = float(
                    V1TrainingConfig.TOWER_RISK_HIT_PENALTIES[
                        min(
                            hit_index,
                            len(
                                V1TrainingConfig.TOWER_RISK_HIT_PENALTIES
                            )
                            - 1,
                        )
                    ]
                )
                remaining = max(
                    0.0,
                    float(
                        V1TrainingConfig.TOWER_RISK_EVENT_PENALTY_CAP
                    )
                    - float(state["penalty_total"]),
                )
                applied = min(magnitude, remaining)
                if applied > 0.0:
                    rewards[actor_index] -= applied
                    state["penalty_total"] += applied
                    hits[actor_index] += 1.0
                    self.tower_risk_counters[
                        "confirmed_hit"
                    ] += 1.0
                state["hit_count"] += 1
                hit_phase = (
                    "critical"
                    if _hp_ratio(hero["actor"])
                    <= V1TrainingConfig.TOWER_RISK_CRITICAL_HP_RATIO
                    else "danger"
                )
                if hit_phase != str(state["phase"]):
                    self.tower_risk_counters[
                        f"enter_{hit_phase}"
                    ] += 1.0
                state["phase"] = hit_phase
            phases.append(str(state["phase"]))
        return (
            rewards,
            hits,
            tuple(phases),
        )

    def _consume_structure_team_reward(self, context, dead_actions):
        """Return bounded nonlinear outer/high-tower potential difference.

        Missing structure actors keep their last confirmed HP.  Only a current
        actor observation or a structure death event may lower the cached
        ratio, so fog/partial NPC payloads cannot manufacture tower reward.
        The same raw state viewed from the opposite camp negates the potential.
        """

        previous_ratios = dict(self.structure_hp_ratios)
        for organ in context.get("organs", ()):
            actor = organ.get("actor")
            if actor is None:
                continue
            max_hp = _value(actor, "max_hp", "maxHp")
            if max_hp <= 0.0:
                continue
            self.structure_hp_ratios[organ["config_id"]] = u.clip(
                _value(actor, "hp") / max_hp
            )
        for event in dead_actions:
            death = u.safe_get(event, "death", "dead", default={}) or {}
            config_id = u.actor_config_id(death)
            if config_id in (66, 67, 68, 70, 71, 72):
                self.structure_hp_ratios[config_id] = 0.0

        own_structures = tuple(context["organs"][:3])
        enemy_structures = tuple(context["organs"][3:6])
        own_towers = own_structures[:2]
        enemy_towers = enemy_structures[:2]
        own_ids = tuple(organ["config_id"] for organ in own_towers)
        enemy_ids = tuple(organ["config_id"] for organ in enemy_towers)
        own_push_ids = tuple(
            organ["config_id"] for organ in own_structures
        )
        enemy_push_ids = tuple(
            organ["config_id"] for organ in enemy_structures
        )

        def confirmed_damage(config_ids):
            return float(
                sum(
                    max(
                        0.0,
                        previous_ratios[config_id]
                        - self.structure_hp_ratios[config_id],
                    )
                    for config_id in config_ids
                    if config_id in previous_ratios
                    and config_id in self.structure_hp_ratios
                )
            )

        own_tower_damage = confirmed_damage(own_ids)
        enemy_tower_damage = confirmed_damage(enemy_ids)
        own_push_conversion_damage = confirmed_damage(own_push_ids)
        enemy_push_conversion_damage = confirmed_damage(enemy_push_ids)
        tower_ids = own_ids + enemy_ids
        if not all(config_id in self.structure_hp_ratios for config_id in tower_ids):
            self.previous_structure_potential = None
            return (
                0.0,
                own_tower_damage,
                enemy_tower_damage,
                own_push_conversion_damage,
                enemy_push_conversion_damage,
                (0.0,) * FrozenV1Config.CONTROLLED_ACTOR_COUNT,
                {
                    "positive_team_reward": 0.0,
                    "negative_team_reward": 0.0,
                    "coverage": 0.0,
                    "fallback": False,
                    "hit_counts": (0, 0),
                },
            )

        beta = float(V1TrainingConfig.STRUCTURE_POTENTIAL_BETA)
        scale = float(V1TrainingConfig.STRUCTURE_TOWER_TOTAL_REWARD) / (
            1.0 + beta
        )
        initialized = all(
            config_id in previous_ratios for config_id in tower_ids
        )
        positive_reward = (
            scale
            * sum(
                self._structure_curve(previous_ratios[config_id])
                - self._structure_curve(
                    self.structure_hp_ratios[config_id]
                )
                for config_id in enemy_ids
            )
            if initialized
            else 0.0
        )
        negative_reward = (
            scale
            * sum(
                self._structure_curve(
                    self.structure_hp_ratios[config_id]
                )
                - self._structure_curve(previous_ratios[config_id])
                for config_id in own_ids
            )
            if initialized
            else 0.0
        )
        positive_reward = max(0.0, float(positive_reward))
        negative_reward = min(0.0, float(negative_reward))
        reward = positive_reward + negative_reward
        potential = scale * (
            sum(
                self._structure_curve(self.structure_hp_ratios[config_id])
                for config_id in own_ids
            )
            - sum(
                self._structure_curve(self.structure_hp_ratios[config_id])
                for config_id in enemy_ids
            )
        )
        self.previous_structure_potential = potential

        actor_contributions = np.zeros(
            FrozenV1Config.CONTROLLED_ACTOR_COUNT, dtype=np.float64
        )
        hit_counts = np.zeros_like(actor_contributions)
        attributable_drop = 0.0
        total_enemy_drop = 0.0
        for tower in enemy_towers:
            config_id = tower["config_id"]
            drop = (
                max(
                    0.0,
                    previous_ratios.get(config_id, 0.0)
                    - self.structure_hp_ratios.get(config_id, 0.0),
                )
                if initialized
                else 0.0
            )
            if drop <= 1e-12 or tower["actor"] is None:
                continue
            total_enemy_drop += drop
            tower_runtime = u.actor_runtime_id(tower["actor"])
            max_hp = max(
                0.0, _value(tower["actor"], "max_hp", "maxHp")
            )
            per_actor = np.zeros_like(actor_contributions)
            participants = []
            for actor_index, info in enumerate(
                context["controlled_infos"]
            ):
                config = u.as_int(
                    u.safe_get(info, "config_id", default=0), 0
                )
                events = self.recent_events.get(("friend", config), ())
                damage = sum(
                    event["amount"]
                    for event in events
                    if event["frame"] == context["frame_no"]
                    and event["type"] == "damage_dealt"
                    and event["other_runtime"] == tower_runtime
                )
                hit = any(
                    event["frame"] == context["frame_no"]
                    and event["type"] == "hit_organ"
                    and event["other_runtime"] == tower_runtime
                    for event in events
                )
                if damage > 0.0:
                    per_actor[actor_index] = (
                        damage / max_hp if max_hp > 0.0 else 0.0
                    )
                if hit or damage > 0.0:
                    participants.append(actor_index)
                    hit_counts[actor_index] += 1.0
            confirmed = min(drop, float(per_actor.sum()))
            if confirmed <= 1e-12 and participants:
                confirmed = drop
                per_actor[participants] = drop / float(
                    len(participants)
                )
            elif confirmed > 0.0 and per_actor.sum() > confirmed:
                per_actor *= confirmed / float(per_actor.sum())
            attributable_drop += confirmed
            actor_contributions += per_actor

        coverage = (
            u.clip(attributable_drop / total_enemy_drop)
            if total_enemy_drop > 1e-12
            else 0.0
        )
        contribution_total = float(actor_contributions.sum())
        if contribution_total > 1e-12:
            proportions = actor_contributions / contribution_total
        else:
            proportions = np.zeros_like(actor_contributions)
        actor_rewards = np.full(
            FrozenV1Config.CONTROLLED_ACTOR_COUNT,
            negative_reward
            + V1TrainingConfig.STRUCTURE_TEAM_FLOOR_RATIO
            * positive_reward,
            dtype=np.float64,
        )
        actor_rewards += (
            coverage * positive_reward * proportions
            + (1.0 - coverage)
            * V1TrainingConfig.STRUCTURE_PERSONAL_ATTRIBUTION_RATIO
            * positive_reward
        )
        if positive_reward > 0.0:
            expected_total = 2.0 * positive_reward + 2.0 * negative_reward
            drift = expected_total - float(actor_rewards.sum())
            actor_rewards += drift / float(
                FrozenV1Config.CONTROLLED_ACTOR_COUNT
            )
        return (
            float(reward),
            own_tower_damage,
            enemy_tower_damage,
            own_push_conversion_damage,
            enemy_push_conversion_damage,
            tuple(float(value) for value in actor_rewards),
            {
                "positive_team_reward": positive_reward,
                "negative_team_reward": negative_reward,
                "coverage": float(coverage),
                "fallback": bool(
                    positive_reward > 0.0 and coverage < 1.0
                ),
                "hit_counts": tuple(int(value) for value in hit_counts),
            },
        )

    def _consume_defense_wave_reward(self, context, dead_actions):
        """Attribute one bounded defensive-wave window to the controlled actors."""

        actor_count = FrozenV1Config.CONTROLLED_ACTOR_COUNT
        rewards = np.zeros(actor_count, dtype=np.float32)
        frame_no = context["frame_no"]
        controlled_hard_masks = []
        for info in context["controlled_infos"]:
            config_id = u.as_int(
                u.safe_get(info, "config_id", default=0), 0
            )
            lane_plan = context["lane_macro"].get(config_id, {})
            controlled_hard_masks.append(
                bool(
                    lane_plan.get("hard_mask_active")
                    or (
                        config_id == 193
                        and context["jungle_macro"].get(
                            "opening_rotation_active"
                        )
                        and context["jungle_macro"].get("interrupt")
                        == "none"
                    )
                )
            )
        if all(controlled_hard_masks):
            self.defense_wave_state = None
            return rewards, {"completed": False, "window_active": False}
        enemy_soldiers = {
            u.actor_runtime_id(soldier): soldier
            for soldier in context["soldiers"][10:]
            if soldier is not None and u.actor_runtime_id(soldier)
        }
        eligible_structures = []
        for organ in context["organs"][:3]:
            if not organ["alive"] or organ["actor"] is None:
                continue
            attack_range = _value(
                organ["actor"], "attack_range", "attackRange"
            )
            if attack_range > 0.0:
                eligible_structures.append(
                    (organ, u.actor_runtime_id(organ["actor"]), attack_range)
                )

        # Preserve an active window on its original structure.  For a new
        # window, choose the structure whose own range/target state makes an
        # enemy soldier the strongest current threat.  Selecting the first
        # living structure would silently miss high-ground/crystal waves while
        # the outer tower still exists.
        state = self.defense_wave_state
        structure_entry = None
        if state is not None:
            structure_entry = next(
                (
                    entry
                    for entry in eligible_structures
                    if entry[1] == int(state["structure_runtime"])
                ),
                None,
            )
            if structure_entry is None:
                self.defense_wave_state = None
                state = None
        if state is None:
            structure_candidates = []
            for organ, runtime_id, attack_range in eligible_structures:
                for soldier in enemy_soldiers.values():
                    if not u.is_alive(soldier):
                        continue
                    position = _position(soldier, self.camp)
                    if position is None:
                        continue
                    distance = _distance_or_inf(
                        position, organ["side_position"]
                    )
                    targeting_structure = (
                        _target_runtime(soldier) == runtime_id
                    )
                    if targeting_structure or distance <= attack_range:
                        level_rank = 0
                    elif (
                        distance
                        <= attack_range
                        + V1TrainingConfig.DEFENSE_WAVE_FRONT_MARGIN
                    ):
                        level_rank = 1
                    else:
                        continue
                    structure_candidates.append(
                        (
                            level_rank,
                            distance - attack_range,
                            int(organ["slot"]),
                            (organ, runtime_id, attack_range),
                        )
                    )
            if structure_candidates:
                structure_entry = min(
                    structure_candidates,
                    key=lambda item: item[:3],
                )[3]
        if structure_entry is None:
            return rewards, {"completed": False, "window_active": False}
        structure, structure_runtime, attack_range = structure_entry

        current_threats = {}
        for runtime_id, soldier in enemy_soldiers.items():
            if not u.is_alive(soldier):
                continue
            position = _position(soldier, self.camp)
            if position is None:
                continue
            distance = _distance_or_inf(
                position, structure["side_position"]
            )
            targeting_structure = (
                _target_runtime(soldier) == structure_runtime
            )
            if targeting_structure or distance <= attack_range:
                level = "under"
            elif (
                distance
                <= attack_range
                + V1TrainingConfig.DEFENSE_WAVE_FRONT_MARGIN
            ):
                level = "front"
            else:
                continue
            current_threats[runtime_id] = (soldier, level)

        if state is None and current_threats:
            level = (
                "under"
                if any(value[1] == "under" for value in current_threats.values())
                else "front"
            )
            state = {
                "structure_runtime": structure_runtime,
                "structure_config_id": structure["config_id"],
                "structure_position": structure["side_position"],
                "structure_attack_range": attack_range,
                "level": level,
                "start_frame": frame_no,
                "last_threat_frame": frame_no,
                "no_threat_since": None,
                "soldiers": {},
                "responded": set(),
                "participants": set(),
                "ignored": set(),
                "ignore_candidates": set(),
                "ignore_exemption_keys": set(),
                "other_progress_suppressed": set(),
                "last_progress_frame": None,
                "last_hit_frame": {},
                "last_progress_frame_by_actor": {},
                "progress_total": 0.0,
                "positive_total": 0.0,
            }
            self.defense_wave_state = state
            self.defense_wave_counters[f"window_{level}"] += 1.0
            self.defense_wave_counters[
                f"window_structure_{structure['type']}"
            ] += 1.0
        if state is None:
            return rewards, {"completed": False, "window_active": False}
        if any(value[1] == "under" for value in current_threats.values()):
            state["level"] = "under"
        scale = (
            1.0
            if state["level"] == "under"
            else V1TrainingConfig.DEFENSE_WAVE_FRONT_SCALE
        )
        positive_cap = V1TrainingConfig.DEFENSE_WAVE_POSITIVE_CAP * scale

        dead_runtime_ids = {
            u.actor_runtime_id(
                u.safe_get(event, "death", "dead", default={}) or {}
            )
            for event in dead_actions
        }
        invalidation_reason = None

        def missing_grace_expired(ledger):
            missing_since = ledger.get("missing_since")
            return bool(
                missing_since is not None
                and frame_no - int(missing_since)
                >= V1TrainingConfig.DEFENSE_WAVE_MISSING_GRACE_FRAMES
            )

        def soldier_identity_discontinuous(ledger, soldier):
            config_id = u.actor_config_id(soldier)
            camp = u.actor_camp(soldier)
            hp = max(0.0, _value(soldier, "hp"))
            max_hp = max(hp, _value(soldier, "max_hp", "maxHp"))
            if (
                ledger.get("resolved", False)
                and ledger.get("resolved_reason")
                in ("dead_event", "observed_dead")
            ):
                return True
            if (
                config_id != int(ledger["config_id"])
                or camp != int(ledger["camp"])
                or abs(max_hp - float(ledger["max_hp"])) > 1e-3
            ):
                return True
            missing_since = ledger.get("missing_since")
            previous_position = ledger.get("last_seen_position")
            position = _position(soldier, self.camp)
            if (
                missing_since is None
                or previous_position is None
                or position is None
            ):
                return False
            elapsed = max(
                1,
                frame_no - int(ledger.get("last_seen_frame", frame_no)),
            )
            move_speed = max(
                0.0,
                _value(
                    soldier,
                    "move_speed",
                    "moveSpeed",
                    "mov_spd",
                    "movSpd",
                ),
            )
            plausible_distance = (
                move_speed * elapsed * u.RAW_FRAME_SECONDS
                + V1TrainingConfig.DEFENSE_WAVE_FRONT_MARGIN
            )
            hp_reset = (
                hp
                > float(ledger.get("last_seen_hp", hp))
                + 0.25 * max(1.0, max_hp)
            )
            impossible_position_jump = (
                _distance_or_inf(position, previous_position)
                > plausible_distance
            )
            # Either signal independently disproves continuity. Requiring both
            # lets a reused runtime inherit the old ledger when HP is unchanged.
            return bool(hp_reset or impossible_position_jump)

        for runtime_id, (soldier, level) in current_threats.items():
            hp = max(0.0, _value(soldier, "hp"))
            max_hp = max(hp, _value(soldier, "max_hp", "maxHp"))
            position = _position(soldier, self.camp)
            ledger = state["soldiers"].setdefault(
                runtime_id,
                {
                    "config_id": u.actor_config_id(soldier),
                    "camp": u.actor_camp(soldier),
                    "entry_hp": max(hp, 1.0),
                    "last_hp": hp,
                    "last_seen_hp": hp,
                    "last_seen_frame": frame_no,
                    "last_seen_position": position,
                    "missing_since": None,
                    "progress_paid": 0.0,
                    "resolved": False,
                    "resolved_reason": None,
                    "level": level,
                    "max_hp": max_hp,
                },
            )
            if soldier_identity_discontinuous(ledger, soldier):
                invalidation_reason = "runtime_reuse"
                break
            if missing_grace_expired(ledger):
                invalidation_reason = "observation_invalidated"
                break
            ledger["level"] = (
                "under"
                if level == "under" or ledger["level"] == "under"
                else "front"
            )
            ledger["current_hp"] = hp
            ledger["max_hp"] = max_hp
            ledger["last_seen_hp"] = hp
            ledger["last_seen_frame"] = frame_no
            ledger["last_seen_position"] = position
            ledger["missing_since"] = None
            ledger["resolved"] = False
            ledger["resolved_reason"] = None
        for runtime_id, ledger in state["soldiers"].items():
            if invalidation_reason is not None:
                break
            if runtime_id in dead_runtime_ids:
                ledger["resolved"] = True
                ledger["resolved_reason"] = "dead_event"
                ledger["missing_since"] = None
                continue
            soldier = enemy_soldiers.get(runtime_id)
            if soldier is not None:
                if soldier_identity_discontinuous(ledger, soldier):
                    invalidation_reason = "runtime_reuse"
                    break
                if missing_grace_expired(ledger):
                    invalidation_reason = "observation_invalidated"
                    break
                position = _position(soldier, self.camp)
                hp = max(0.0, _value(soldier, "hp"))
                ledger["last_seen_hp"] = hp
                ledger["last_seen_frame"] = frame_no
                ledger["last_seen_position"] = position
                ledger["missing_since"] = None
                if not u.is_alive(soldier):
                    ledger["resolved"] = True
                    ledger["resolved_reason"] = "observed_dead"
                    continue
                if (
                    position is not None
                    and _distance_or_inf(
                        position, structure["side_position"]
                    )
                    > attack_range
                    + V1TrainingConfig.DEFENSE_WAVE_FRONT_MARGIN
                ):
                    ledger["resolved"] = True
                    ledger["resolved_reason"] = "observed_exit"
                continue
            if ledger.get("resolved", False):
                continue
            if ledger.get("missing_since") is None:
                ledger["missing_since"] = frame_no
                self.defense_wave_counters[
                    "soldier_missing"
                ] += 1.0
            missing_frames = frame_no - int(ledger["missing_since"])
            self.defense_wave_counters["missing_peak_frames"] = max(
                float(
                    self.defense_wave_counters.get(
                        "missing_peak_frames", 0.0
                    )
                ),
                float(missing_frames),
            )
            if (
                missing_frames
                >= V1TrainingConfig.DEFENSE_WAVE_MISSING_GRACE_FRAMES
            ):
                invalidation_reason = "observation_invalidated"
                break

        if invalidation_reason is not None:
            self.defense_wave_counters["window_invalidated"] += 1.0
            self.defense_wave_counters[
                f"window_invalidated_{invalidation_reason}"
            ] += 1.0
            self.defense_wave_state = None
            return (
                rewards
                * float(V1TrainingConfig.DEFENSE_WAVE_REWARD_SCALE)
            ), {
                "completed": False,
                "window_active": False,
                "invalidated": True,
                "invalidation_reason": invalidation_reason,
            }

        per_actor_events = []
        for info in context["controlled_infos"]:
            config_id = u.as_int(
                u.safe_get(info, "config_id", default=0), 0
            )
            per_actor_events.append(
                tuple(
                    event
                    for event in self.recent_events.get(
                        ("friend", config_id), ()
                    )
                    if event["frame"] == frame_no
                )
            )

        for runtime_id, (soldier, _level) in current_threats.items():
            ledger = state["soldiers"][runtime_id]
            participants = []
            contribution = np.zeros(actor_count, dtype=np.float64)
            for actor_index, events in enumerate(per_actor_events):
                if controlled_hard_masks[actor_index]:
                    continue
                damage = sum(
                    event["amount"]
                    for event in events
                    if event["type"] == "damage_dealt"
                    and event["other_runtime"] == runtime_id
                )
                hit = any(
                    event["type"] == "hit_soldier"
                    and event["other_runtime"] == runtime_id
                    for event in events
                )
                if damage > 0.0:
                    contribution[actor_index] = damage
                if hit or damage > 0.0:
                    participants.append(actor_index)
                    state["participants"].add(actor_index)
                    state["last_hit_frame"][actor_index] = frame_no
                    if actor_index not in state["responded"]:
                        remaining = max(
                            0.0,
                            positive_cap - float(state["positive_total"]),
                        )
                        response_reward = min(
                            V1TrainingConfig.DEFENSE_WAVE_RESPONSE_REWARD
                            * scale,
                            remaining,
                        )
                        rewards[actor_index] += response_reward
                        state["positive_total"] += response_reward
                        state["responded"].add(actor_index)
                        self.defense_wave_counters[
                            "first_response"
                        ] += 1.0

            current_hp = max(0.0, _value(soldier, "hp"))
            hp_drop = max(0.0, float(ledger["last_hp"]) - current_hp)
            ledger["last_hp"] = current_hp
            if hp_drop <= 1e-12 or not participants:
                continue
            attributed_hp_drop = (
                min(hp_drop, float(contribution.sum()))
                if contribution.sum() > 0.0
                else hp_drop
            )
            if attributed_hp_drop <= 1e-12:
                continue
            per_soldier_remaining = max(
                0.0,
                V1TrainingConfig.DEFENSE_WAVE_SOLDIER_PROGRESS_CAP
                * scale
                - float(ledger["progress_paid"]),
            )
            window_remaining = max(
                0.0,
                V1TrainingConfig.DEFENSE_WAVE_PROGRESS_CAP * scale
                - float(state["progress_total"]),
            )
            progress = min(
                per_soldier_remaining,
                window_remaining,
                V1TrainingConfig.DEFENSE_WAVE_SOLDIER_PROGRESS_CAP
                * scale
                * attributed_hp_drop
                / max(1.0, float(ledger["entry_hp"])),
            )
            if progress <= 0.0:
                continue
            if contribution.sum() > 0.0:
                shares = contribution / contribution.sum()
            else:
                shares = np.zeros(actor_count, dtype=np.float64)
                shares[participants] = 1.0 / float(len(participants))
            rewards += (progress * shares).astype(np.float32)
            ledger["progress_paid"] += progress
            state["progress_total"] += progress
            state["positive_total"] += progress
            state["last_progress_frame"] = frame_no
            for actor_index, share in enumerate(shares):
                if share > 0.0:
                    state["last_progress_frame_by_actor"][
                        actor_index
                    ] = frame_no
            self.defense_wave_counters["progress_reward"] += progress

        all_resolved = bool(state["soldiers"]) and all(
            ledger.get("resolved", False)
            for ledger in state["soldiers"].values()
        )
        if current_threats or not all_resolved:
            state["last_threat_frame"] = frame_no
            state["no_threat_since"] = None
        elif state["no_threat_since"] is None:
            state["no_threat_since"] = frame_no

        completed = bool(
            state["no_threat_since"] is not None
            and frame_no - int(state["no_threat_since"])
            >= V1TrainingConfig.DEFENSE_WAVE_CLEAR_FRAMES
        )
        if completed:
            if state["participants"]:
                remaining = max(
                    0.0, positive_cap - float(state["positive_total"])
                )
                total_completion = min(
                    2.0
                    * V1TrainingConfig.DEFENSE_WAVE_COMPLETION_REWARD
                    * scale,
                    remaining,
                )
                rewards += total_completion / float(actor_count)
                state["positive_total"] += total_completion
                self.defense_wave_counters[
                    "completion_reward"
                ] += total_completion
            self.defense_wave_counters["window_completed"] += 1.0
            self.defense_wave_state = None
            return (
                rewards
                * float(V1TrainingConfig.DEFENSE_WAVE_REWARD_SCALE)
            ), {
                "completed": True,
                "window_active": False,
            }

        age = frame_no - int(state["start_frame"])
        if age >= V1TrainingConfig.DEFENSE_WAVE_IGNORE_AGE_FRAMES:
            friendly_runtime_ids = {
                u.actor_runtime_id(hero["actor"])
                for hero in context["friend_heroes"]
                if hero["alive"]
            }
            enemy_structure_threat = any(
                enemy["alive"]
                and enemy["visible"]
                and enemy["side_position"] is not None
                and _distance_or_inf(
                    enemy["side_position"],
                    structure["side_position"],
                )
                <= attack_range
                and _target_runtime(enemy["actor"])
                in (
                    structure_runtime,
                    *friendly_runtime_ids,
                )
                for enemy in context["enemy_heroes"]
            )
            for actor_index, info in enumerate(context["controlled_infos"]):
                if actor_index in state["ignored"]:
                    continue
                runtime_id = u.as_int(
                    u.safe_get(info, "runtime_id", default=0), 0
                )
                hero = context["hero_by_runtime"].get(runtime_id)
                if (
                    hero is None
                    or not hero["alive"]
                    or hero["side_position"] is None
                ):
                    continue
                in_defense_zone = (
                    _distance_or_inf(
                        hero["side_position"], structure["side_position"]
                    )
                    <= attack_range
                    + (
                        0.0
                        if state["level"] == "under"
                        else V1TrainingConfig.DEFENSE_WAVE_FRONT_MARGIN
                    )
                )
                hero_attack_range = max(
                    2500.0,
                    _value(
                        hero["actor"], "attack_range", "attackRange"
                    ),
                )
                attackable_threat = any(
                    _distance_or_inf(
                        hero["side_position"], _position(soldier, self.camp)
                    )
                    <= hero_attack_range
                    for soldier, _level in current_threats.values()
                )
                events = per_actor_events[actor_index]
                hit_enemy_hero = any(
                    event["type"] == "hit_hero" for event in events
                )
                recent_hurt = "hurt_by_hero" in self._event_summary(
                    hero, frame_no
                )["short_types"]
                crystal_runtime = u.actor_runtime_id(
                    context["organs"][5]["actor"]
                )
                attacking_crystal = any(
                    event["type"] in ("hit_organ", "damage_dealt")
                    and event["other_runtime"] == crystal_runtime
                    for event in events
                )
                tyrant_runtime = u.actor_runtime_id(
                    context["tyrant"]["actor"]
                )
                tyrant_clears = int(
                    self.tyrant_reward_state["kill_counts"].get(
                        self.camp, 0
                    )
                )
                tyrant_exempt = bool(
                    u.actor_config_id(hero["actor"]) == 193
                    and tyrant_clears < 3
                    and any(
                        event["type"] in (
                            "hit_monster",
                            "damage_dealt",
                        )
                        and event["other_runtime"] == tyrant_runtime
                        for event in events
                    )
                )
                lane_plan = context["lane_macro"].get(
                    u.actor_config_id(hero["actor"]), {}
                )
                opening_exempt = bool(
                    lane_plan.get("hard_mask_active")
                    or (
                        u.actor_config_id(hero["actor"]) == 193
                        and context["jungle_macro"].get(
                            "opening_rotation_active"
                        )
                        and context["jungle_macro"].get("interrupt")
                        == "none"
                    )
                )
                own_last_hit = state["last_hit_frame"].get(actor_index)
                own_last_progress = state[
                    "last_progress_frame_by_actor"
                ].get(actor_index)
                own_recent_response = any(
                    value is not None
                    and frame_no - int(value)
                    <= V1TrainingConfig.DEFENSE_WAVE_IGNORE_AGE_FRAMES
                    for value in (own_last_hit, own_last_progress)
                )
                other_progress = any(
                    other_index != actor_index
                    and frame_no - int(progress_frame)
                    <= V1TrainingConfig.DEFENSE_WAVE_IGNORE_AGE_FRAMES
                    for other_index, progress_frame in state[
                        "last_progress_frame_by_actor"
                    ].items()
                )
                if (
                    in_defense_zone
                    and attackable_threat
                    and not own_recent_response
                    and hit_enemy_hero
                ):
                    if other_progress:
                        if (
                            actor_index
                            not in state["other_progress_suppressed"]
                        ):
                            state["other_progress_suppressed"].add(
                                actor_index
                            )
                            self.defense_wave_counters[
                                "other_progress_suppressed"
                            ] += 1.0
                        if enemy_structure_threat:
                            key = (actor_index, "division")
                            if key not in state["ignore_exemption_keys"]:
                                state["ignore_exemption_keys"].add(key)
                                self.defense_wave_counters[
                                    "ignore_exemption"
                                ] += 1.0
                                self.defense_wave_counters[
                                    "ignore_exempt_division"
                                ] += 1.0
                        continue
                    if actor_index not in state["ignore_candidates"]:
                        state["ignore_candidates"].add(actor_index)
                        self.defense_wave_counters[
                            "ignore_candidate"
                        ] += 1.0
                    exemption_reason = None
                    if recent_hurt:
                        exemption_reason = "recent_hurt"
                    elif attacking_crystal:
                        exemption_reason = "crystal"
                    elif tyrant_exempt:
                        exemption_reason = "tyrant"
                    elif opening_exempt:
                        exemption_reason = "opening"
                    elif (
                        _hp_ratio(hero["actor"])
                        <= V1TrainingConfig.DEFENSE_WAVE_LOW_HP_EXEMPT_RATIO
                    ):
                        exemption_reason = "low_hp"
                    if exemption_reason is not None:
                        key = (actor_index, exemption_reason)
                        if key not in state["ignore_exemption_keys"]:
                            state["ignore_exemption_keys"].add(key)
                            self.defense_wave_counters[
                                "ignore_exemption"
                            ] += 1.0
                            self.defense_wave_counters[
                                f"ignore_exempt_{exemption_reason}"
                            ] += 1.0
                        continue
                    penalty = (
                        -V1TrainingConfig.DEFENSE_WAVE_IGNORE_PENALTY
                        * scale
                    )
                    rewards[actor_index] += penalty
                    state["ignored"].add(actor_index)
                    self.defense_wave_counters[
                        "ignore_penalty"
                    ] += 1.0
        return (
            rewards
            * float(V1TrainingConfig.DEFENSE_WAVE_REWARD_SCALE)
        ), {
            "completed": False,
            "window_active": True,
            "level": state["level"],
        }

    def _consume_post_kill_push_reward(
        self,
        context,
        dead_actions,
        own_push_conversion_damage,
        enemy_push_conversion_damage,
    ):
        """Reward converting a hero kill into confirmed structure HP damage."""

        frame_no = context["frame_no"]
        for side in ("own", "enemy"):
            state = self.post_kill_push_state[side]
            if frame_no > state["expires"]:
                state["tokens"] = 0

        for event in dead_actions:
            death = u.safe_get(event, "death", "dead", default={}) or {}
            config_id = u.actor_config_id(death)
            actor_type = u.normalize_actor_type(
                u.actor_get(death, "actor_type", "actorType", default=None)
            )
            if config_id not in u.HERO_CONFIG_IDS and actor_type != "ACTOR_HERO":
                continue
            death_camp = u.actor_camp(death)
            if death_camp not in (1, 2):
                continue
            runtime_id = u.actor_runtime_id(death)
            death_key = runtime_id or (death_camp, config_id)
            previous_frame = self.post_kill_last_death_frame.get(death_key)
            if previous_frame is not None and frame_no - previous_frame <= 30:
                continue
            self.post_kill_last_death_frame[death_key] = frame_no
            side = "own" if death_camp != self.camp else "enemy"
            state = self.post_kill_push_state[side]
            state["tokens"] = min(
                V1TrainingConfig.POST_KILL_PUSH_MAX_TOKENS,
                state["tokens"] + 1,
            )
            state["expires"] = max(
                state["expires"],
                frame_no + V1TrainingConfig.POST_KILL_PUSH_WINDOW_FRAMES,
            )

        reward = 0.0
        own_window = self.post_kill_push_state["own"]
        if (
            enemy_push_conversion_damage > 1e-12
            and own_window["tokens"] > 0
            and frame_no <= own_window["expires"]
        ):
            reward += V1TrainingConfig.POST_KILL_PUSH_REWARD
            own_window["tokens"] -= 1
        enemy_window = self.post_kill_push_state["enemy"]
        if (
            own_push_conversion_damage > 1e-12
            and enemy_window["tokens"] > 0
            and frame_no <= enemy_window["expires"]
        ):
            reward -= V1TrainingConfig.POST_KILL_PUSH_REWARD
            enemy_window["tokens"] -= 1
        return float(reward)

    def _scenario_death_signature(self, context, event, death):
        runtime_id = u.actor_runtime_id(death)
        actor = context["actor_by_runtime"].get(runtime_id)
        death_count = _integer(
            actor or death, "dead_cnt", "deadCnt", default=0
        )
        if runtime_id and death_count > 0:
            return ("hero_death", runtime_id, death_count)
        event_frame = u.as_int(
            u.safe_get(
                event,
                "frame_no",
                "frameNo",
                "frame",
                default=u.safe_get(
                    death,
                    "frame_no",
                    "frameNo",
                    "frame",
                    default=context["frame_no"],
                ),
            ),
            context["frame_no"],
        )
        return (
            "hero_death",
            runtime_id,
            u.actor_camp(death),
            u.actor_config_id(death),
            event_frame,
        )

    def _consume_temporal_focus_fire_reward(self, context, dead_actions):
        """Reward a second real participant joining the same local target."""

        frame_no = context["frame_no"]
        contributions = defaultdict(set)
        for actor in context["raw_heroes"]:
            attacker_camp = u.actor_camp(actor)
            attacker_runtime = u.actor_runtime_id(actor)
            attacker_position = u.location_of(actor)
            if (
                attacker_camp not in (1, 2)
                or not attacker_runtime
                or not u.is_alive(actor)
                or attacker_position is None
            ):
                continue
            target_runtimes = set()
            for hurt in (
                u.actor_get(
                    actor,
                    "hurt_hero_info",
                    "hurtHeroInfo",
                    default=[],
                )
                or []
            ):
                if u.as_float(
                    u.safe_get(hurt, "hurt", default=0.0), 0.0
                ) <= 0.0:
                    continue
                target_runtimes.add(
                    u.as_int(
                        u.safe_get(
                            hurt,
                            "hurt_target",
                            "hurtTarget",
                            default=0,
                        ),
                        0,
                    )
                )
            for hit in (
                u.actor_get(
                    actor,
                    "hit_target_info",
                    "hitTargetInfo",
                    default=[],
                )
                or []
            ):
                target_runtimes.add(
                    u.as_int(
                        u.safe_get(
                            hit,
                            "hit_target",
                            "hitTarget",
                            default=0,
                        ),
                        0,
                    )
                )
            for target_runtime in target_runtimes:
                target_actor = context["actor_by_runtime"].get(
                    target_runtime
                )
                if (
                    not target_runtime
                    or u.target_type(target_actor) != "hero"
                    or u.actor_camp(target_actor) == attacker_camp
                    or _distance_or_inf(
                        attacker_position,
                        u.location_of(target_actor),
                    )
                    > V1TrainingConfig.TEMPORAL_FOCUS_LOCAL_RADIUS
                ):
                    continue
                contributions[(attacker_camp, target_runtime)].add(
                    attacker_runtime
                )

        for event in dead_actions:
            death = u.safe_get(event, "death", "dead", default={}) or {}
            target_runtime = u.actor_runtime_id(death)
            target_camp = u.actor_camp(death)
            killer_camp = self._killer_camp_for_event(context, event)
            if (
                target_runtime
                and target_camp in (1, 2)
                and killer_camp in (1, 2)
                and target_camp != killer_camp
            ):
                contributions[(killer_camp, target_runtime)].update(
                    self._event_actor_runtimes(event)
                )

        for key in list(self.temporal_focus_state):
            if (
                frame_no - self.temporal_focus_state[key]["last_frame"]
                > V1TrainingConfig.TEMPORAL_FOCUS_WINDOW_FRAMES
            ):
                self.temporal_focus_state.pop(key, None)

        reward = 0.0
        count = 0
        for key, current_participants in contributions.items():
            attacker_camp, target_runtime = key
            valid_participants = {
                runtime_id
                for runtime_id in current_participants
                if u.actor_camp(
                    context["actor_by_runtime"].get(runtime_id)
                )
                == attacker_camp
            }
            if not valid_participants:
                continue
            state = self.temporal_focus_state.get(key)
            if (
                state is None
                or frame_no - state["last_frame"]
                > V1TrainingConfig.TEMPORAL_FOCUS_WINDOW_FRAMES
            ):
                state = {
                    "last_frame": frame_no,
                    "participants": set(),
                    "rewarded": False,
                }
                self.temporal_focus_state[key] = state
            state["participants"].update(valid_participants)
            state["last_frame"] = frame_no
            if (
                not state["rewarded"]
                and len(state["participants"])
                >= V1TrainingConfig.TEMPORAL_FOCUS_MIN_PARTICIPANTS
            ):
                reward += (
                    V1TrainingConfig.TEMPORAL_FOCUS_REWARD
                    if attacker_camp == self.camp
                    else -V1TrainingConfig.TEMPORAL_FOCUS_REWARD
                )
                count += 1
                state["rewarded"] = True
        return float(reward), int(count)

    def _defense_pressure_for_camp(self, context, target_camp):
        organ_ids = (66, 67, 68) if target_camp == 1 else (70, 71, 72)
        ratios = self.structure_hp_ratios
        active_id = next(
            (
                config_id
                for config_id in organ_ids[:2]
                if ratios.get(config_id, 0.0) > 1e-12
            ),
            0,
        )
        state = self.defense_pressure_state[target_camp]
        if not active_id:
            state.update(
                {
                    "structure_id": 0,
                    "last_evaluation_frame": None,
                    "window_total": 0.0,
                }
            )
            return 0.0, False

        structure = next(
            (
                actor
                for actor in context["organs_raw"]
                if u.actor_config_id(actor) == active_id
            ),
            None,
        )
        structure_position = u.location_of(structure)
        if structure is None or structure_position is None:
            return 0.0, False
        attack_range = max(
            0.0,
            _value(structure, "attack_range", "attackRange"),
        )
        nearest_distance = min(
            (
                _distance_or_inf(
                    u.location_of(soldier), structure_position
                )
                for soldier in context["soldiers_raw"]
                if u.actor_camp(soldier) == 3 - target_camp
                and u.is_alive(soldier)
                and u.location_of(soldier) is not None
            ),
            default=float("inf"),
        )
        if nearest_distance <= attack_range:
            base_penalty = (
                V1TrainingConfig.DEFENSE_PRESSURE_UNDER_TOWER_PENALTY
            )
        elif (
            nearest_distance
            <= attack_range
            + V1TrainingConfig.DEFENSE_PRESSURE_FRONT_MARGIN
        ):
            base_penalty = (
                V1TrainingConfig.DEFENSE_PRESSURE_FRONT_PENALTY
            )
        else:
            state.update(
                {
                    "structure_id": active_id,
                    "last_evaluation_frame": None,
                    "window_total": 0.0,
                }
            )
            return 0.0, False

        if state["structure_id"] != active_id:
            state.update(
                {
                    "structure_id": active_id,
                    "last_evaluation_frame": None,
                    "window_total": 0.0,
                }
            )
        frame_no = context["frame_no"]
        if (
            state["last_evaluation_frame"] is not None
            and frame_no - state["last_evaluation_frame"]
            < V1TrainingConfig.DEFENSE_PRESSURE_EVALUATION_FRAMES
        ):
            return 0.0, False

        hp_ratio = ratios.get(active_id, 1.0)
        if (
            hp_ratio
            <= V1TrainingConfig.DEFENSE_PRESSURE_CRITICAL_HP_THRESHOLD
        ):
            multiplier = (
                V1TrainingConfig.DEFENSE_PRESSURE_CRITICAL_HP_MULTIPLIER
            )
        elif (
            hp_ratio
            <= V1TrainingConfig.DEFENSE_PRESSURE_LOW_HP_THRESHOLD
        ):
            multiplier = (
                V1TrainingConfig.DEFENSE_PRESSURE_LOW_HP_MULTIPLIER
            )
        else:
            multiplier = 1.0
        remaining_budget = max(
            0.0,
            V1TrainingConfig.DEFENSE_PRESSURE_WINDOW_CAP
            - state["window_total"],
        )
        charged = min(base_penalty * multiplier, remaining_budget)
        state["last_evaluation_frame"] = frame_no
        state["window_total"] += charged
        return float(charged), charged > 1e-12

    def _consume_defense_pressure_reward(self, context):
        own_charge, own_applied = self._defense_pressure_for_camp(
            context, self.camp
        )
        enemy_charge, enemy_applied = self._defense_pressure_for_camp(
            context, 3 - self.camp
        )
        return (
            float(enemy_charge - own_charge),
            int(own_applied) + int(enemy_applied),
        )

    @staticmethod
    def _event_actor_runtimes(event):
        runtimes = set()
        killer = u.safe_get(
            event, "killer", "killer_actor", default={}
        ) or {}
        killer_runtime = u.actor_runtime_id(killer)
        if killer_runtime:
            runtimes.add(killer_runtime)
        for item in list(
            u.safe_get(event, "assist_set", "assistSet", default=[]) or []
        ):
            runtime_id = (
                u.actor_runtime_id(item)
                if isinstance(item, dict)
                else u.as_int(item, 0)
            )
            if runtime_id:
                runtimes.add(runtime_id)
        return runtimes

    @staticmethod
    def _death_attacker_runtimes(context, event, death):
        runtimes = FeatureProcessV1._event_actor_runtimes(event)
        for item in list(
            u.actor_get(
                death,
                "single_hurt_list",
                "singleHurtList",
                default=[],
            )
            or []
        ):
            runtime_id = u.as_int(
                u.safe_get(item, "runtime_id", "runtimeId", default=0),
                0,
            )
            actor = context["actor_by_runtime"].get(runtime_id)
            if runtime_id and u.target_type(actor) == "hero":
                runtimes.add(runtime_id)
        return runtimes

    def _recent_support_evidence(
        self, snapshot, target_runtimes, frame_no
    ):
        key = (
            snapshot["relation"],
            u.actor_config_id(snapshot["actor"]),
        )
        contributed = False
        higher_value_objective = False
        for event in self.recent_events.get(key, ()):
            if (
                frame_no - event["frame"]
                > V1TrainingConfig.TEAMMATE_SUPPORT_WINDOW_FRAMES
            ):
                continue
            event_type = event["type"]
            other_runtime = event["other_runtime"]
            if event_type == "hit_organ":
                higher_value_objective = True
            if (
                other_runtime in target_runtimes
                and event_type
                in (
                    "damage_dealt",
                    "hit_hero",
                    "hurt_by_hero",
                    "damage_taken",
                )
            ):
                contributed = True
        return contributed, higher_value_objective

    def _consume_teamfight_scenario_rewards(self, context, dead_actions):
        """Return high-precision rescue penalties and three-person kill reward."""

        controlled_runtime_ids = [
            u.as_int(u.safe_get(info, "runtime_id", default=0), 0)
            for info in context["controlled_infos"]
        ]
        abandonment_rewards = np.zeros(
            FrozenV1Config.CONTROLLED_ACTOR_COUNT, dtype=np.float32
        )
        abandonment_counts = np.zeros_like(abandonment_rewards)
        support_rewards = np.zeros_like(abandonment_rewards)
        support_counts = np.zeros_like(abandonment_rewards)
        group_attack_reward = 0.0
        group_attack_count = 0
        command_abandonment_penalty = 0.0
        command_abandonment_count = 0
        command_support_reward = 0.0
        command_support_count = 0

        for event in dead_actions:
            death = u.safe_get(event, "death", "dead", default={}) or {}
            config_id = u.actor_config_id(death)
            actor_type = u.normalize_actor_type(
                u.actor_get(
                    death, "actor_type", "actorType", default=None
                )
            )
            if config_id not in u.HERO_CONFIG_IDS and actor_type != "ACTOR_HERO":
                continue
            signature = self._scenario_death_signature(
                context, event, death
            )
            if signature in self.scenario_rewarded_deaths:
                continue
            self.scenario_rewarded_deaths.add(signature)

            victim_runtime = u.actor_runtime_id(death)
            victim_snapshot = context["hero_by_runtime"].get(victim_runtime)
            victim_actor = (
                victim_snapshot["actor"] if victim_snapshot else death
            )
            victim_camp = u.actor_camp(victim_actor)
            killer_camp = self._killer_camp_for_event(context, event)
            if victim_camp not in (1, 2) or killer_camp not in (1, 2):
                continue

            participant_runtimes = self._event_actor_runtimes(event)
            if victim_runtime:
                for snapshot in context["heroes"]:
                    actor = snapshot["actor"]
                    runtime_id = u.actor_runtime_id(actor)
                    if (
                        u.actor_camp(actor) != killer_camp
                        or not runtime_id
                    ):
                        continue
                    contributed, _ = self._recent_support_evidence(
                        snapshot, {victim_runtime}, context["frame_no"]
                    )
                    if contributed:
                        participant_runtimes.add(runtime_id)
            participant_runtimes = {
                runtime_id
                for runtime_id in participant_runtimes
                if u.actor_camp(
                    context["actor_by_runtime"].get(runtime_id)
                )
                == killer_camp
            }
            if (
                victim_camp != killer_camp
                and len(participant_runtimes)
                >= V1TrainingConfig.GROUP_ATTACK_MIN_PARTICIPANTS
            ):
                group_attack_reward += (
                    V1TrainingConfig.GROUP_ATTACK_REWARD
                    if killer_camp == self.camp
                    else -V1TrainingConfig.GROUP_ATTACK_REWARD
                )
                group_attack_count += 1

            if victim_camp != self.camp:
                continue
            victim_position = (
                victim_snapshot["side_position"]
                if victim_snapshot
                else None
            )
            if victim_position is None:
                raw_position = u.location_of(death)
                victim_position = (
                    u.side_position(raw_position, self.camp)
                    if raw_position is not None
                    else None
                )
            attacker_runtimes = self._death_attacker_runtimes(
                context, event, death
            )
            if victim_position is None or not attacker_runtimes:
                continue

            for snapshot in context["friend_heroes"]:
                actor = snapshot["actor"]
                runtime_id = u.actor_runtime_id(actor)
                if (
                    not runtime_id
                    or runtime_id == victim_runtime
                    or not snapshot["alive"]
                    or snapshot["side_position"] is None
                    or _hp_ratio(actor)
                    <= V1TrainingConfig.TEAMMATE_SUPPORT_LOW_HP_EXEMPT_RATIO
                    or _distance_or_inf(
                        snapshot["side_position"], victim_position
                    )
                    > V1TrainingConfig.TEAMMATE_SUPPORT_RADIUS
                ):
                    continue
                contributed, higher_value_objective = (
                    self._recent_support_evidence(
                        snapshot,
                        attacker_runtimes,
                        context["frame_no"],
                    )
                )
                if contributed:
                    if runtime_id in controlled_runtime_ids:
                        actor_index = controlled_runtime_ids.index(
                            runtime_id
                        )
                        support_rewards[actor_index] += (
                            V1TrainingConfig.TEAMMATE_SUPPORT_REWARD
                        )
                        support_counts[actor_index] += 1.0
                    elif (
                        snapshot["official"]
                        and self.active_command_attribution is not None
                    ):
                        command_support_reward += (
                            V1TrainingConfig.TEAMMATE_SUPPORT_REWARD
                        )
                        command_support_count += 1
                    continue
                if higher_value_objective:
                    continue
                if runtime_id in controlled_runtime_ids:
                    actor_index = controlled_runtime_ids.index(runtime_id)
                    abandonment_rewards[actor_index] -= (
                        V1TrainingConfig.TEAMMATE_ABANDONMENT_PENALTY
                    )
                    abandonment_counts[actor_index] += 1.0
                elif (
                    snapshot["official"]
                    and self.active_command_attribution is not None
                ):
                    command_abandonment_penalty -= (
                        V1TrainingConfig.TEAMMATE_ABANDONMENT_PENALTY
                    )
                    command_abandonment_count += 1

        return {
            "abandonment_rewards": abandonment_rewards,
            "abandonment_counts": abandonment_counts,
            "support_rewards": support_rewards,
            "support_counts": support_counts,
            "group_attack_reward": float(group_attack_reward),
            "group_attack_count": int(group_attack_count),
            "command_abandonment_penalty": float(
                command_abandonment_penalty
            ),
            "command_abandonment_count": int(
                command_abandonment_count
            ),
            "command_support_reward": float(command_support_reward),
            "command_support_count": int(command_support_count),
        }

    def _crystal_danger_pressure(self, context, target_camp):
        organ_ids = (66, 67, 68) if target_camp == 1 else (70, 71, 72)
        outer_id, high_id, crystal_id = organ_ids
        ratios = self.structure_hp_ratios
        if (
            outer_id not in ratios
            or high_id not in ratios
            or crystal_id not in ratios
            or ratios[outer_id] > 1e-12
            or ratios[high_id] > 1e-12
            or ratios[crystal_id] <= 1e-12
        ):
            return 0.0

        crystal_position = u.ORGAN_ANCHORS[crystal_id]
        radius = V1TrainingConfig.CRYSTAL_DANGER_RADIUS

        def proximity(actor):
            if (
                actor is None
                or not u.is_alive(actor)
                or u.actor_camp(actor) != 3 - target_camp
            ):
                return 0.0
            distance = _distance_or_inf(
                u.location_of(actor), crystal_position
            )
            closeness = u.clip((radius - distance) / radius)
            return closeness * closeness

        hero_pressure = max(
            (proximity(actor) for actor in context["raw_heroes"]),
            default=0.0,
        )
        minion_pressure = max(
            (proximity(actor) for actor in context["soldiers_raw"]),
            default=0.0,
        )
        crystal_ratio = ratios[crystal_id]
        if (
            crystal_ratio
            <= V1TrainingConfig.CRYSTAL_DANGER_CRITICAL_HP_THRESHOLD
        ):
            hp_multiplier = (
                V1TrainingConfig.CRYSTAL_DANGER_CRITICAL_HP_MULTIPLIER
            )
        elif (
            crystal_ratio
            <= V1TrainingConfig.CRYSTAL_DANGER_LOW_HP_THRESHOLD
        ):
            hp_multiplier = (
                V1TrainingConfig.CRYSTAL_DANGER_LOW_HP_MULTIPLIER
            )
        else:
            hp_multiplier = 1.0
        return float(
            hp_multiplier
            * (
                V1TrainingConfig.CRYSTAL_DANGER_HERO_BUDGET
                * hero_pressure
                + V1TrainingConfig.CRYSTAL_DANGER_MINION_BUDGET
                * minion_pressure
            )
        )

    def _consume_crystal_danger_reward(self, context):
        """Return zero-sum distance potential for an exposed living crystal."""

        own_pressure = self._crystal_danger_pressure(context, self.camp)
        enemy_pressure = self._crystal_danger_pressure(
            context, 3 - self.camp
        )
        potential = enemy_pressure - own_pressure
        reward = (
            0.0
            if self.previous_crystal_danger_potential is None
            else potential - self.previous_crystal_danger_potential
        )
        self.previous_crystal_danger_potential = potential
        return float(reward)

    def _long_inactivity_tyrant_exempt(self, context, hero):
        status = self._tyrant_status(context)
        spawn_seconds = status["spawn_seconds"]
        return bool(
            not status["alive"]
            and spawn_seconds is not None
            and 0.0 <= spawn_seconds
            <= V1TrainingConfig.LONG_INACTIVITY_TYRANT_SPAWN_SECONDS
            and status["position"] is not None
            and hero["side_position"] is not None
            and _distance_or_inf(hero["side_position"], status["position"])
            <= V1TrainingConfig.LONG_INACTIVITY_TYRANT_RADIUS
        )

    def _long_inactivity_objective(self, context, config_id, hero):
        """Return a stable objective signature and current distance if known."""

        position = hero["side_position"]
        actor = hero["actor"]
        target_runtime = _target_runtime(actor)
        target_actor = context["actor_by_runtime"].get(target_runtime)
        target_position = (
            _position(target_actor, self.camp) if target_actor else None
        )
        if target_runtime and target_position is not None:
            return (
                ("runtime", target_runtime),
                _distance_or_inf(position, target_position),
            )

        if config_id == 193:
            macro = context.get("jungle_macro") or {}
            waypoint = macro.get("next_waypoint")
            if macro.get("valid") and waypoint is not None:
                return (
                    (
                        "jungle",
                        macro.get("phase"),
                        macro.get("target_slot"),
                    ),
                    _distance_or_inf(position, waypoint),
                )
        lane_plan = (context.get("lane_macro") or {}).get(config_id)
        if lane_plan and lane_plan.get("opening_active"):
            waypoint = lane_plan.get("next_waypoint")
            if waypoint is not None:
                return (
                    ("lane", config_id, lane_plan.get("route_node")),
                    _distance_or_inf(position, waypoint),
                )

        candidates = []
        for snapshot in context.get("enemy_heroes", ()):
            if snapshot["alive"] and snapshot["side_position"] is not None:
                candidates.append(
                    (
                        ("runtime", u.actor_runtime_id(snapshot["actor"])),
                        snapshot["side_position"],
                    )
                )
        for actor_candidate in context.get("soldiers_raw", ()):
            if (
                u.actor_camp(actor_candidate) != self.camp
                and u.is_alive(actor_candidate)
            ):
                candidate_position = _position(actor_candidate, self.camp)
                if candidate_position is not None:
                    candidates.append(
                        (
                            ("runtime", u.actor_runtime_id(actor_candidate)),
                            candidate_position,
                        )
                    )
        for monster in context.get("monsters", ()):
            if monster["alive"] and monster["actor"] is not None:
                candidate_position = _position(monster["actor"], self.camp)
                if candidate_position is not None:
                    candidates.append(
                        (
                            ("runtime", u.actor_runtime_id(monster["actor"])),
                            candidate_position,
                        )
                    )
        target_organ = context.get("target38")
        if target_organ and target_organ["alive"]:
            candidates.append(
                (
                    ("organ", target_organ["config_id"]),
                    target_organ["side_position"],
                )
            )
        if not candidates:
            return None, None
        signature, target_position = min(
            candidates,
            key=lambda item: _distance_or_inf(position, item[1]),
        )
        return signature, _distance_or_inf(position, target_position)

    @staticmethod
    def _reset_long_inactivity_state(state):
        state["history"].clear()
        state["next_evaluation_frame"] = None
        state["abnormal_steps"] = 0
        state["last_damage"] = None

    def _consume_long_inactivity_penalty(
        self, context, config_id, hero, structure_progress=False
    ):
        """Evaluate a trailing five-second behavior window after 23 seconds."""

        runtime_id = u.actor_runtime_id(hero["actor"]) if hero else 0
        state = self.long_inactivity_history.get(runtime_id)
        if state is None:
            state = {
                "history": deque(),
                "next_evaluation_frame": None,
                "abnormal_steps": 0,
                "last_damage": None,
            }
            self.long_inactivity_history[runtime_id] = state

        frame_no = context["frame_no"]
        if frame_no < V1TrainingConfig.LONG_INACTIVITY_START_FRAME:
            self._reset_long_inactivity_state(state)
            return 0.0, False, False, ()

        tyrant_exempt = bool(
            hero is not None
            and hero["alive"]
            and hero["side_position"] is not None
            and self._long_inactivity_tyrant_exempt(context, hero)
        )
        recent_types = (
            self._event_summary(hero, frame_no)["short_types"]
            if hero is not None
            else set()
        )
        under_hero_pressure = "hurt_by_hero" in recent_types
        eligible = bool(
            hero is not None
            and runtime_id
            and hero["alive"]
            and hero["side_position"] is not None
        )
        if not eligible or under_hero_pressure or tyrant_exempt:
            self._reset_long_inactivity_state(state)
            return 0.0, False, tyrant_exempt, ()

        raw_damage = u.actor_get(
            hero["actor"], "total_hurt", "totalHurt", default=None
        )
        damage = None
        if raw_damage is not None:
            candidate = u.as_float(raw_damage, float("nan"))
            if math.isfinite(candidate) and candidate >= 0.0:
                damage = candidate
        if (
            damage is not None
            and state["last_damage"] is not None
            and damage < state["last_damage"]
        ):
            self._reset_long_inactivity_state(state)
        state["last_damage"] = damage
        objective_signature, objective_distance = (
            self._long_inactivity_objective(context, config_id, hero)
        )
        history = state["history"]
        if not history:
            state["next_evaluation_frame"] = (
                frame_no + V1TrainingConfig.LONG_INACTIVITY_WINDOW_FRAMES
            )
        history.append(
            {
                "frame": frame_no,
                "position": hero["side_position"],
                "damage": damage,
                "objective_signature": objective_signature,
                "objective_distance": objective_distance,
            }
        )
        window = V1TrainingConfig.LONG_INACTIVITY_WINDOW_FRAMES
        while (
            len(history) >= 2
            and frame_no - history[1]["frame"] >= window
        ):
            history.popleft()
        next_evaluation = state["next_evaluation_frame"]
        if next_evaluation is None or frame_no < next_evaluation:
            return 0.0, False, False, ()
        interval = V1TrainingConfig.LONG_INACTIVITY_EVALUATION_INTERVAL_FRAMES
        evaluation_count = 1 + (frame_no - next_evaluation) // interval
        state["next_evaluation_frame"] += evaluation_count * interval
        if len(history) < 2 or frame_no - history[0]["frame"] < window:
            state["next_evaluation_frame"] = frame_no + interval
            return 0.0, False, False, ()

        history_items = list(history)
        path_distance = sum(
            _distance_or_inf(previous["position"], current["position"])
            for previous, current in zip(history_items, history_items[1:])
        )
        net_distance = _distance_or_inf(
            history_items[0]["position"], history_items[-1]["position"]
        )
        max_excursion = max(
            _distance_or_inf(
                history_items[0]["position"], item["position"]
            )
            for item in history_items
        )
        stationary = bool(
            path_distance
            <= V1TrainingConfig.LONG_INACTIVITY_STATIC_PATH_DISTANCE
            and max_excursion
            <= V1TrainingConfig.LONG_INACTIVITY_STATIC_MAX_EXCURSION
        )

        sampled = [history_items[0]]
        for item in history_items[1:-1]:
            if item["frame"] - sampled[-1]["frame"] >= interval:
                sampled.append(item)
        if history_items[-1]["frame"] != sampled[-1]["frame"]:
            sampled.append(history_items[-1])
        segments = []
        for previous, current in zip(sampled, sampled[1:]):
            dx = current["position"][0] - previous["position"][0]
            dz = current["position"][1] - previous["position"][1]
            length = math.hypot(dx, dz)
            if length >= V1TrainingConfig.LONG_INACTIVITY_WANDER_MIN_SEGMENT:
                segments.append((dx, dz, length))
        reversal_count = sum(
            (
                first[0] * second[0] + first[1] * second[1]
            )
            / (first[2] * second[2])
            <= V1TrainingConfig.LONG_INACTIVITY_WANDER_REVERSE_COSINE
            for first, second in zip(segments, segments[1:])
        )
        revisited = any(
            later["frame"] - earlier["frame"]
            >= V1TrainingConfig.LONG_INACTIVITY_WANDER_REVISIT_FRAMES
            and _distance_or_inf(earlier["position"], later["position"])
            <= V1TrainingConfig.LONG_INACTIVITY_WANDER_REVISIT_RADIUS
            for index, earlier in enumerate(sampled)
            for later in sampled[index + 1 :]
        )
        movement_efficiency = net_distance / max(path_distance, 1.0)
        geometry_wandering = bool(
            path_distance
            > V1TrainingConfig.LONG_INACTIVITY_STATIC_PATH_DISTANCE
            and net_distance
            <= V1TrainingConfig.LONG_INACTIVITY_WANDER_NET_DISTANCE
            and (
                movement_efficiency
                <= V1TrainingConfig.LONG_INACTIVITY_WANDER_EFFICIENCY
                or reversal_count
                >= V1TrainingConfig.LONG_INACTIVITY_WANDER_MIN_REVERSALS
                or revisited
            )
        )

        oldest_damage = history_items[0]["damage"]
        newest_damage = history_items[-1]["damage"]
        damage_progress = bool(
            oldest_damage is not None
            and newest_damage is not None
            and newest_damage - oldest_damage > 1e-6
        )
        current_signature = history_items[-1]["objective_signature"]
        current_distance = history_items[-1]["objective_distance"]
        matching_objective = next(
            (
                item
                for item in history_items
                if current_signature is not None
                and item["objective_signature"] == current_signature
                and item["objective_distance"] is not None
            ),
            None,
        )
        objective_progress = bool(
            matching_objective is not None
            and current_distance is not None
            and matching_objective["objective_distance"] - current_distance
            >= V1TrainingConfig.LONG_INACTIVITY_OBJECTIVE_PROGRESS_DISTANCE
        )
        wandering = bool(
            geometry_wandering
            and not damage_progress
            and not objective_progress
            and not structure_progress
        )
        effective_rotation = bool(
            net_distance
            >= V1TrainingConfig.LONG_INACTIVITY_ROTATION_NET_DISTANCE
            and movement_efficiency
            >= V1TrainingConfig.LONG_INACTIVITY_ROTATION_EFFICIENCY
        )
        no_damage = bool(
            oldest_damage is not None
            and newest_damage is not None
            and newest_damage - oldest_damage <= 1e-6
            and not effective_rotation
        )
        reasons = tuple(
            name
            for name, active in (
                ("stationary", stationary),
                ("wandering", wandering),
                ("no_damage", no_damage),
            )
            if active
        )
        if not reasons:
            state["abnormal_steps"] = 0
            return 0.0, False, False, ()

        penalty = 0.0
        for _ in range(evaluation_count):
            k = state["abnormal_steps"]
            magnitude = V1TrainingConfig.LONG_INACTIVITY_PENALTY_MIN + (
                V1TrainingConfig.LONG_INACTIVITY_PENALTY_MAX
                - V1TrainingConfig.LONG_INACTIVITY_PENALTY_MIN
            ) * (
                1.0
                - math.exp(
                    -float(k) / V1TrainingConfig.LONG_INACTIVITY_PENALTY_TAU
                )
            )
            penalty -= min(
                magnitude, V1TrainingConfig.LONG_INACTIVITY_PENALTY_MAX
            )
            state["abnormal_steps"] += 1
        return float(penalty), True, False, reasons

    def _build_behavior_shaping(self, context):
        """Return checkpoint-compatible custom rewards for this boundary."""

        infos = context["controlled_infos"]
        controlled_runtime_ids = [
            u.as_int(u.safe_get(info, "runtime_id", default=0), 0)
            for info in infos
        ]
        controlled_config_ids = [
            u.as_int(u.safe_get(info, "config_id", default=0), 0)
            for info in infos
        ]
        rewards = np.zeros(FrozenV1Config.CONTROLLED_ACTOR_COUNT, dtype=np.float32)
        last_hit_counts = np.zeros_like(rewards)
        last_hit_rewards = np.zeros_like(rewards)
        no_target_counts = np.zeros_like(rewards)
        no_target_penalties = np.zeros_like(rewards)
        no_target_grace_counts = np.zeros_like(rewards)
        strategic_wait_counts = np.zeros_like(rewards)
        long_inactivity_counts = np.zeros_like(rewards)
        long_inactivity_penalties = np.zeros_like(rewards)
        long_inactivity_tyrant_exemptions = np.zeros_like(rewards)
        long_inactivity_stationary_counts = np.zeros_like(rewards)
        long_inactivity_wandering_counts = np.zeros_like(rewards)
        long_inactivity_no_damage_counts = np.zeros_like(rewards)
        attendance_counts = np.zeros_like(rewards)
        attendance_rewards = np.zeros_like(rewards)

        frame_action = u.safe_get(
            context["frame_state"], "frame_action", "frameAction", default={}
        ) or {}
        dead_actions = list(u.safe_get(
            frame_action, "dead_action", "deadAction", default=[]
        ) or [])
        kda_by_runtime, kda_by_config = self._build_zero_sum_kda_rewards(context)
        kda_rewards = np.asarray(
            [
                kda_by_runtime.get(runtime_id, kda_by_config.get(config_id, 0.0))
                for runtime_id, config_id in zip(
                    controlled_runtime_ids, controlled_config_ids
                )
            ],
            dtype=np.float32,
        )
        tyrant_team_reward = self._consume_tyrant_team_reward(
            context, dead_actions
        )
        (
            kai_wolf_team_reward,
            kai_wolf_reward_count,
        ) = self._consume_kai_wolf_team_reward(context, dead_actions)
        (
            treasure_thief_team_reward,
            treasure_thief_reward_count,
        ) = self._consume_treasure_thief_team_reward(
            context, dead_actions
        )
        (
            speed_buff_team_reward,
            speed_buff_reward_count,
        ) = self._consume_speed_buff_team_reward(context)
        crystal_outcome_reward = self._consume_crystal_outcome_reward(
            context, dead_actions
        )
        (
            structure_team_reward,
            own_tower_damage,
            enemy_tower_damage,
            own_push_conversion_damage,
            enemy_push_conversion_damage,
            structure_actor_rewards,
            structure_attribution,
        ) = self._consume_structure_team_reward(context, dead_actions)
        structure_actor_rewards = np.asarray(
            structure_actor_rewards, dtype=np.float32
        )
        post_kill_push_reward = self._consume_post_kill_push_reward(
            context,
            dead_actions,
            own_push_conversion_damage,
            enemy_push_conversion_damage,
        )
        crystal_danger_reward = self._consume_crystal_danger_reward(
            context
        )
        (
            defense_pressure_reward,
            defense_pressure_count,
        ) = self._consume_defense_pressure_reward(context)
        (
            tower_risk_rewards,
            tower_risk_hit_counts,
            tower_risk_phases,
        ) = self._consume_tower_risk_reward(context)
        (
            defense_wave_rewards,
            defense_wave_status,
        ) = self._consume_defense_wave_reward(context, dead_actions)
        (
            temporal_focus_team_reward,
            temporal_focus_count,
        ) = self._consume_temporal_focus_fire_reward(
            context, dead_actions
        )
        teamfight_scenarios = self._consume_teamfight_scenario_rewards(
            context, dead_actions
        )
        teammate_abandonment_rewards = np.asarray(
            teamfight_scenarios["abandonment_rewards"],
            dtype=np.float32,
        )
        teammate_support_rewards = np.asarray(
            teamfight_scenarios["support_rewards"],
            dtype=np.float32,
        )
        group_attack_team_reward = float(
            teamfight_scenarios["group_attack_reward"]
        )
        kda_training_rewards = (
            kda_rewards
            + teammate_abandonment_rewards
            + teammate_support_rewards
            + temporal_focus_team_reward
            + group_attack_team_reward
        )
        command_attribution = self._consume_command_attribution_reward(
            context, dead_actions
        )
        (
            command_no_command_idle_penalty,
            command_no_command_idle_applied,
        ) = self._consume_official_no_command_idle_penalty(context)
        pushing_shared_reward = (
            tyrant_team_reward
            + post_kill_push_reward
            + crystal_danger_reward
            + defense_pressure_reward
        )
        pushing_actor_rewards = (
            structure_actor_rewards
            + float(pushing_shared_reward)
            + np.asarray(tower_risk_rewards, dtype=np.float32)
            + np.asarray(defense_wave_rewards, dtype=np.float32)
        )
        farming_team_reward = (
            kai_wolf_team_reward
            + treasure_thief_team_reward
            + speed_buff_team_reward
        )
        rewards += (
            kda_training_rewards
            + pushing_actor_rewards
            + crystal_outcome_reward
            + farming_team_reward
        )
        command_farm_camp_reward = float(
            self.official_task_reward_pending
        )
        self.official_task_reward_pending = 0.0

        for event in dead_actions:
            death = u.safe_get(event, "death", "dead", default={}) or {}
            if u.normalize_sub_type(
                u.actor_get(death, "sub_type", "subType", default=None)
            ) != "ACTOR_SUB_SOLDIER":
                continue
            killer = u.safe_get(event, "killer", "killer_actor", default={}) or {}
            killer_runtime = u.actor_runtime_id(killer)
            if not killer_runtime:
                hurt_list = list(
                    u.actor_get(
                        death,
                        "single_hurt_list",
                        "singleHurtList",
                        default=[],
                    )
                    or []
                )
                if hurt_list:
                    last_hurt = max(
                        hurt_list,
                        key=lambda item: u.as_int(
                            u.safe_get(
                                item,
                                "frameNo",
                                "frame_no",
                                "frame",
                                default=-1,
                            ),
                            -1,
                        ),
                    )
                    killer_runtime = u.as_int(
                        u.safe_get(
                            last_hurt, "runtime_id", "runtimeId", default=0
                        ),
                        0,
                    )
            if killer_runtime not in controlled_runtime_ids:
                continue
            actor_index = controlled_runtime_ids.index(killer_runtime)
            if controlled_config_ids[actor_index] not in LANE_ROW_SIDE_Z:
                continue
            rewards[actor_index] += V1TrainingConfig.LANE_LAST_HIT_REWARD
            last_hit_rewards[actor_index] += V1TrainingConfig.LANE_LAST_HIT_REWARD
            last_hit_counts[actor_index] += 1.0

        for actor_index, (info, runtime_id) in enumerate(
            zip(infos, controlled_runtime_ids)
        ):
            hero = context["hero_by_runtime"].get(runtime_id)
            config_id = controlled_config_ids[actor_index]
            attendance = self._lane_attendance_status(
                context, config_id, hero
            )
            attendance_reward = self._consume_lane_attendance_reward(
                context, config_id, hero, attendance
            )
            attendance_rewards[actor_index] = attendance_reward
            rewards[actor_index] += attendance_reward
            if abs(attendance_reward) > 1e-12:
                attendance_counts[actor_index] += 1.0
            alive_and_ready = bool(
                hero is not None
                and hero["alive"]
                and context["frame_no"]
                >= V1TrainingConfig.NO_ATTACKABLE_TARGET_START_FRAME
                and _hp_ratio(hero["actor"])
                > V1TrainingConfig.NO_ATTACKABLE_TARGET_RECOVERY_HP_RATIO
            )
            legal_matrix, button_legal = _raw_legal_matrix(info)
            normal_attackable = bool(
                button_legal[3] > 0.5
                and (
                    np.any(legal_matrix[3, 1:4] > 0.5)
                    or np.any(legal_matrix[3, 8:] > 0.5)
                )
            )
            strategic_wait = bool(
                alive_and_ready
                and self._strategic_wait_active(
                    context, config_id, hero, attendance
                )
            )
            penalty, applied, exempted, in_grace = self._consume_no_target_penalty(
                context,
                runtime_id,
                alive_and_ready,
                normal_attackable,
                strategic_wait,
            )
            rewards[actor_index] += penalty
            no_target_penalties[actor_index] = penalty
            if applied:
                no_target_counts[actor_index] += 1.0
            if exempted:
                strategic_wait_counts[actor_index] += 1.0
            if in_grace:
                no_target_grace_counts[actor_index] += 1.0
            (
                inactivity_penalty,
                inactivity_applied,
                inactivity_tyrant_exempt,
                inactivity_reasons,
            ) = self._consume_long_inactivity_penalty(
                context,
                config_id,
                hero,
                structure_progress=enemy_tower_damage > 1e-12,
            )
            rewards[actor_index] += inactivity_penalty
            long_inactivity_penalties[actor_index] = inactivity_penalty
            if inactivity_applied:
                long_inactivity_counts[actor_index] += 1.0
                long_inactivity_stationary_counts[actor_index] += float(
                    "stationary" in inactivity_reasons
                )
                long_inactivity_wandering_counts[actor_index] += float(
                    "wandering" in inactivity_reasons
                )
                long_inactivity_no_damage_counts[actor_index] += float(
                    "no_damage" in inactivity_reasons
                )
            if inactivity_tyrant_exempt:
                long_inactivity_tyrant_exemptions[actor_index] += 1.0

        return {
            "controlled_config_ids": tuple(
                int(value) for value in controlled_config_ids
            ),
            "rewards": tuple(float(value) for value in rewards),
            "last_hit_counts": tuple(int(value) for value in last_hit_counts),
            "last_hit_rewards": tuple(float(value) for value in last_hit_rewards),
            "farming_auxiliary_rewards": tuple(
                float(value + farming_team_reward)
                for value in last_hit_rewards
            ),
            "kda_auxiliary_rewards": tuple(
                float(value) for value in kda_training_rewards
            ),
            "pushing_auxiliary_rewards": tuple(
                float(value) for value in pushing_actor_rewards
            ),
            "win_auxiliary_rewards": (
                float(crystal_outcome_reward),
            )
            * FrozenV1Config.CONTROLLED_ACTOR_COUNT,
            "kda_rewards_by_runtime": dict(kda_by_runtime),
            "kda_rewards_by_config": dict(kda_by_config),
            "tyrant_team_reward": float(tyrant_team_reward),
            "kai_wolf_team_reward": float(kai_wolf_team_reward),
            "kai_wolf_reward_count": int(kai_wolf_reward_count),
            "treasure_thief_team_reward": float(
                treasure_thief_team_reward
            ),
            "treasure_thief_reward_count": int(
                treasure_thief_reward_count
            ),
            "speed_buff_team_reward": float(speed_buff_team_reward),
            "speed_buff_reward_count": int(speed_buff_reward_count),
            "structure_team_reward": float(structure_team_reward),
            "structure_actor_rewards": tuple(
                float(value) for value in structure_actor_rewards
            ),
            "structure_positive_team_reward": float(
                structure_attribution["positive_team_reward"]
            ),
            "structure_negative_team_reward": float(
                structure_attribution["negative_team_reward"]
            ),
            "structure_attribution_coverage": float(
                structure_attribution["coverage"]
            ),
            "structure_attribution_fallback": bool(
                structure_attribution["fallback"]
            ),
            "structure_attributed_hit_counts": tuple(
                int(value)
                for value in structure_attribution["hit_counts"]
            ),
            "own_tower_damage": float(own_tower_damage),
            "enemy_tower_damage": float(enemy_tower_damage),
            "own_push_conversion_damage": float(
                own_push_conversion_damage
            ),
            "enemy_push_conversion_damage": float(
                enemy_push_conversion_damage
            ),
            "post_kill_push_reward": float(post_kill_push_reward),
            "crystal_danger_reward": float(crystal_danger_reward),
            "defense_pressure_reward": float(
                defense_pressure_reward
            ),
            "defense_pressure_count": int(defense_pressure_count),
            "tower_risk_rewards": tuple(
                float(value) for value in tower_risk_rewards
            ),
            "tower_risk_hit_counts": tuple(
                int(value) for value in tower_risk_hit_counts
            ),
            "tower_risk_phases": tuple(tower_risk_phases),
            "defense_wave_rewards": tuple(
                float(value) for value in defense_wave_rewards
            ),
            "defense_wave_completed": bool(
                defense_wave_status.get("completed", False)
            ),
            "defense_wave_active": bool(
                defense_wave_status.get("window_active", False)
            ),
            "defense_wave_invalidated": bool(
                defense_wave_status.get("invalidated", False)
            ),
            "defense_wave_invalidation_reason": str(
                defense_wave_status.get("invalidation_reason", "none")
            ),
            "command_farm_camp_reward": command_farm_camp_reward,
            "temporal_focus_team_reward": float(
                temporal_focus_team_reward
            ),
            "temporal_focus_count": int(temporal_focus_count),
            "teammate_abandonment_rewards": tuple(
                float(value) for value in teammate_abandonment_rewards
            ),
            "teammate_abandonment_counts": tuple(
                int(value)
                for value in teamfight_scenarios["abandonment_counts"]
            ),
            "teammate_support_rewards": tuple(
                float(value) for value in teammate_support_rewards
            ),
            "teammate_support_counts": tuple(
                int(value)
                for value in teamfight_scenarios["support_counts"]
            ),
            "group_attack_team_reward": float(
                group_attack_team_reward
            ),
            "group_attack_count": int(
                teamfight_scenarios["group_attack_count"]
            ),
            "command_teammate_abandonment_penalty": float(
                teamfight_scenarios[
                    "command_abandonment_penalty"
                ]
            ),
            "command_teammate_abandonment_count": int(
                teamfight_scenarios[
                    "command_abandonment_count"
                ]
            ),
            "command_teammate_support_reward": float(
                teamfight_scenarios["command_support_reward"]
            ),
            "command_teammate_support_count": int(
                teamfight_scenarios["command_support_count"]
            ),
            "crystal_outcome_reward": float(crystal_outcome_reward),
            "command_attribution_active": bool(
                command_attribution["active"]
            ),
            "command_attribution_intent": int(
                command_attribution["intent"]
            ),
            "command_official_kda_reward": float(
                command_attribution["official_kda_reward"]
            ),
            "command_goal_progress_reward": float(
                command_attribution["goal_progress_reward"]
            ),
            "command_nonresponse_penalty": float(
                command_attribution["nonresponse_penalty"]
            ),
            "command_nonresponse_applied": bool(
                command_attribution["nonresponse_applied"]
            ),
            "command_no_command_idle_penalty": float(
                command_no_command_idle_penalty
            ),
            "command_no_command_idle_applied": bool(
                command_no_command_idle_applied
            ),
            "no_target_counts": tuple(int(value) for value in no_target_counts),
            "no_target_penalties": tuple(
                float(value) for value in no_target_penalties
            ),
            "no_target_grace_counts": tuple(
                int(value) for value in no_target_grace_counts
            ),
            "strategic_wait_counts": tuple(
                int(value) for value in strategic_wait_counts
            ),
            "long_inactivity_counts": tuple(
                int(value) for value in long_inactivity_counts
            ),
            "long_inactivity_penalties": tuple(
                float(value) for value in long_inactivity_penalties
            ),
            "long_inactivity_tyrant_exemptions": tuple(
                int(value) for value in long_inactivity_tyrant_exemptions
            ),
            "long_inactivity_stationary_counts": tuple(
                int(value) for value in long_inactivity_stationary_counts
            ),
            "long_inactivity_wandering_counts": tuple(
                int(value) for value in long_inactivity_wandering_counts
            ),
            "long_inactivity_no_damage_counts": tuple(
                int(value) for value in long_inactivity_no_damage_counts
            ),
            "lane_attendance_counts": tuple(
                int(value) for value in attendance_counts
            ),
            "lane_attendance_rewards": tuple(
                float(value) for value in attendance_rewards
            ),
        }

    def consume_behavior_shaping(self, expected_frame=None):
        """Consume last-hit/NOP shaping once for the current state boundary."""

        empty = {
            "controlled_config_ids": (),
            "rewards": (0.0,) * FrozenV1Config.CONTROLLED_ACTOR_COUNT,
            "last_hit_counts": (0,) * FrozenV1Config.CONTROLLED_ACTOR_COUNT,
            "last_hit_rewards": (0.0,) * FrozenV1Config.CONTROLLED_ACTOR_COUNT,
            "farming_auxiliary_rewards": (0.0,)
            * FrozenV1Config.CONTROLLED_ACTOR_COUNT,
            "kda_auxiliary_rewards": (0.0,)
            * FrozenV1Config.CONTROLLED_ACTOR_COUNT,
            "pushing_auxiliary_rewards": (0.0,)
            * FrozenV1Config.CONTROLLED_ACTOR_COUNT,
            "win_auxiliary_rewards": (0.0,)
            * FrozenV1Config.CONTROLLED_ACTOR_COUNT,
            "kda_rewards_by_runtime": {},
            "kda_rewards_by_config": {},
            "tyrant_team_reward": 0.0,
            "kai_wolf_team_reward": 0.0,
            "kai_wolf_reward_count": 0,
            "treasure_thief_team_reward": 0.0,
            "treasure_thief_reward_count": 0,
            "speed_buff_team_reward": 0.0,
            "speed_buff_reward_count": 0,
            "structure_team_reward": 0.0,
            "structure_actor_rewards": (0.0,)
            * FrozenV1Config.CONTROLLED_ACTOR_COUNT,
            "structure_positive_team_reward": 0.0,
            "structure_negative_team_reward": 0.0,
            "structure_attribution_coverage": 0.0,
            "structure_attribution_fallback": False,
            "structure_attributed_hit_counts": (0,)
            * FrozenV1Config.CONTROLLED_ACTOR_COUNT,
            "own_tower_damage": 0.0,
            "enemy_tower_damage": 0.0,
            "own_push_conversion_damage": 0.0,
            "enemy_push_conversion_damage": 0.0,
            "post_kill_push_reward": 0.0,
            "crystal_danger_reward": 0.0,
            "defense_pressure_reward": 0.0,
            "defense_pressure_count": 0,
            "tower_risk_rewards": (0.0,)
            * FrozenV1Config.CONTROLLED_ACTOR_COUNT,
            "tower_risk_hit_counts": (0,)
            * FrozenV1Config.CONTROLLED_ACTOR_COUNT,
            "tower_risk_phases": ("inactive",)
            * FrozenV1Config.CONTROLLED_ACTOR_COUNT,
            "defense_wave_rewards": (0.0,)
            * FrozenV1Config.CONTROLLED_ACTOR_COUNT,
            "defense_wave_completed": False,
            "defense_wave_active": False,
            "defense_wave_invalidated": False,
            "defense_wave_invalidation_reason": "none",
            "command_farm_camp_reward": 0.0,
            "temporal_focus_team_reward": 0.0,
            "temporal_focus_count": 0,
            "teammate_abandonment_rewards": (0.0,)
            * FrozenV1Config.CONTROLLED_ACTOR_COUNT,
            "teammate_abandonment_counts": (0,)
            * FrozenV1Config.CONTROLLED_ACTOR_COUNT,
            "teammate_support_rewards": (0.0,)
            * FrozenV1Config.CONTROLLED_ACTOR_COUNT,
            "teammate_support_counts": (0,)
            * FrozenV1Config.CONTROLLED_ACTOR_COUNT,
            "group_attack_team_reward": 0.0,
            "group_attack_count": 0,
            "command_teammate_abandonment_penalty": 0.0,
            "command_teammate_abandonment_count": 0,
            "command_teammate_support_reward": 0.0,
            "command_teammate_support_count": 0,
            "crystal_outcome_reward": 0.0,
            "command_attribution_active": False,
            "command_attribution_intent": 0,
            "command_official_kda_reward": 0.0,
            "command_goal_progress_reward": 0.0,
            "command_nonresponse_penalty": 0.0,
            "command_nonresponse_applied": False,
            "command_no_command_idle_penalty": 0.0,
            "command_no_command_idle_applied": False,
            "no_target_counts": (0,) * FrozenV1Config.CONTROLLED_ACTOR_COUNT,
            "no_target_penalties": (0.0,)
            * FrozenV1Config.CONTROLLED_ACTOR_COUNT,
            "no_target_grace_counts": (0,)
            * FrozenV1Config.CONTROLLED_ACTOR_COUNT,
            "strategic_wait_counts": (0,)
            * FrozenV1Config.CONTROLLED_ACTOR_COUNT,
            "long_inactivity_counts": (0,)
            * FrozenV1Config.CONTROLLED_ACTOR_COUNT,
            "long_inactivity_penalties": (0.0,)
            * FrozenV1Config.CONTROLLED_ACTOR_COUNT,
            "long_inactivity_tyrant_exemptions": (0,)
            * FrozenV1Config.CONTROLLED_ACTOR_COUNT,
            "long_inactivity_stationary_counts": (0,)
            * FrozenV1Config.CONTROLLED_ACTOR_COUNT,
            "long_inactivity_wandering_counts": (0,)
            * FrozenV1Config.CONTROLLED_ACTOR_COUNT,
            "long_inactivity_no_damage_counts": (0,)
            * FrozenV1Config.CONTROLLED_ACTOR_COUNT,
            "lane_attendance_counts": (0,)
            * FrozenV1Config.CONTROLLED_ACTOR_COUNT,
            "lane_attendance_rewards": (0.0,)
            * FrozenV1Config.CONTROLLED_ACTOR_COUNT,
        }
        if self._memo_context is None:
            return empty
        frame_no = self._memo_context["frame_no"]
        if expected_frame is not None and int(expected_frame) != frame_no:
            return empty
        if self._behavior_shaping_consumed_frame == frame_no:
            return empty
        self._behavior_shaping_consumed_frame = frame_no
        return dict(self._memo_context["behavior_shaping"])

    def _controlled_infos(self, observation):
        infos = list(u.safe_get(observation, "heroes", default=[]) or [])
        official_runtime_ids = self._official_runtime_ids(observation)
        controlled = [
            info
            for info in infos
            if u.as_int(u.safe_get(info, "runtime_id", default=0), 0) not in official_runtime_ids
        ]
        if len(controlled) != FrozenV1Config.CONTROLLED_ACTOR_COUNT:
            raise ValueError(
                "Feature Schema v1 requires exactly two controlled heroes; "
                f"observation supplied {len(controlled)} after official filtering"
            )
        config_ids = [u.as_int(u.safe_get(info, "config_id", default=0), 0) for info in controlled]
        if len(set(config_ids)) != 2 or any(config_id not in u.HERO_CONFIG_IDS for config_id in config_ids):
            raise ValueError(f"invalid controlled lineup for v1: {config_ids}")
        return controlled

    def controlled_infos(self, observation):
        """Public protocol-validated controlled-actor view in sample order."""

        frame_state = u.safe_get(observation, "frame_state", default={}) or {}
        if (
            u.frame_number(frame_state) == self._memo_frame_no
            and self._memo_context is not None
        ):
            return tuple(self._memo_context["controlled_infos"])
        return tuple(self._controlled_infos(observation))

    def _official_runtime_ids(self, observation):
        result = set()
        info = u.safe_get(observation, "official_model_info", default={}) or {}
        values = info.values() if isinstance(info, dict) else info if isinstance(info, list) else []
        for item in values:
            if not isinstance(item, dict):
                continue
            runtime_id = u.as_int(u.safe_get(item, "runtime_id", "runtimeId", default=0), 0)
            if runtime_id:
                result.add(runtime_id)
        return result

    def _official_image(self, hero_info):
        image = np.asarray(u.safe_get(hero_info, "feature", default=[]), dtype=np.float32).reshape(-1)
        if image.size != FrozenV1Config.IMAGE_FEATURE_DIM:
            raise ValueError(f"official image must contain 1734 floats, got {image.size}")
        image = image.reshape(FrozenV1Config.IMAGE_SHAPE)
        if not np.isfinite(image).all():
            raise ValueError("official image contains NaN/Inf")
        return image

    def _flatten_blocks(self, blocks):
        flat = []
        for block in schema_v1.FEATURE_BLOCKS:
            value = np.asarray(blocks[block.name], dtype=np.float32)
            if tuple(value.shape) != block.shape:
                raise RuntimeError(f"{block.name} shape mismatch: expected {block.shape}, got {value.shape}")
            flat.append(value.reshape(-1))
        result = np.concatenate(flat).astype(np.float32, copy=False)
        if result.size != FrozenV1Config.FEATURE_DIM or not np.isfinite(result).all():
            raise RuntimeError("invalid v1 feature vector")
        return result

    def _build_context(self, observation, controlled_infos):
        frame_state = u.safe_get(observation, "frame_state", default={}) or {}
        current_frame = u.frame_number(frame_state)
        raw_heroes = list(u.safe_get(frame_state, "hero_states", "heroStates", default=[]) or [])
        npcs = list(u.safe_get(frame_state, "npc_states", "npcStates", default=[]) or [])
        bullets = list(u.safe_get(frame_state, "bullets", default=[]) or [])
        cakes = list(u.safe_get(frame_state, "cakes", default=[]) or [])
        spawn_infos = list(
            u.safe_get(frame_state, "monster_spawn_infos", "monsterSpawnInfos", default=[]) or []
        )
        controlled_runtime_ids = {
            runtime_id
            for info in controlled_infos
            if (
                runtime_id := u.as_int(
                    u.safe_get(info, "runtime_id", default=0), 0
                )
            )
        }
        controlled_config_ids = {
            u.as_int(u.safe_get(info, "config_id", default=0), 0)
            for info in controlled_infos
        }
        official_config_ids = set(u.HERO_CONFIG_IDS) - controlled_config_ids
        official_runtime_ids = self._official_runtime_ids(observation)

        friend_raw = [hero for hero in raw_heroes if u.actor_camp(hero) == self.camp]
        enemy_raw = [hero for hero in raw_heroes if u.actor_camp(hero) not in (0, self.camp)]
        if not official_runtime_ids:
            official_runtime_ids = {
                u.actor_runtime_id(hero)
                for hero in friend_raw
                if u.actor_runtime_id(hero) not in controlled_runtime_ids
            }

        friend_runtime_by_config = {}
        for info in list(u.safe_get(observation, "heroes", default=[]) or []):
            config_id = u.as_int(
                u.safe_get(info, "config_id", default=0), 0
            )
            runtime_id = u.as_int(
                u.safe_get(info, "runtime_id", default=0), 0
            )
            if config_id in u.HERO_CONFIG_IDS and runtime_id:
                friend_runtime_by_config[config_id] = runtime_id
        if len(official_config_ids) == 1 and len(official_runtime_ids) == 1:
            friend_runtime_by_config.setdefault(
                next(iter(official_config_ids)),
                next(iter(official_runtime_ids)),
            )

        dead_runtime_ids = self._dead_runtime_ids(frame_state)
        fixed_heroes = []
        raw_by_key = {(u.actor_camp(hero), u.actor_config_id(hero)): hero for hero in raw_heroes}
        for relation, camp in (("friend", self.camp), ("enemy", 3 - self.camp)):
            for config_id in u.HERO_CONFIG_IDS:
                actor = raw_by_key.get((camp, config_id), {})
                snapshot = self._hero_snapshot(
                    actor,
                    relation,
                    config_id,
                    camp,
                    current_frame,
                    dead_runtime_ids,
                    controlled_runtime_ids,
                    official_runtime_ids,
                    controlled_config_ids,
                    official_config_ids,
                    (
                        friend_runtime_by_config.get(config_id, 0)
                        if relation == "friend"
                        else 0
                    ),
                )
                fixed_heroes.append(snapshot)

        hero_by_runtime = {
            u.actor_runtime_id(snapshot["actor"]): snapshot
            for snapshot in fixed_heroes
            if u.actor_runtime_id(snapshot["actor"])
        }
        actor_by_runtime = u.actor_index_by_runtime(raw_heroes, npcs)
        organs_raw = [
            npc
            for npc in npcs
            if u.normalize_actor_type(u.actor_get(npc, "actor_type", "actorType", default=None)) == "ACTOR_ORGAN"
        ]
        soldiers_raw = [
            npc
            for npc in npcs
            if u.normalize_sub_type(u.actor_get(npc, "sub_type", "subType", default=None))
            == "ACTOR_SUB_SOLDIER"
        ]
        monsters_raw = [
            npc
            for npc in npcs
            if u.normalize_actor_type(u.actor_get(npc, "actor_type", "actorType", default=None))
            == "ACTOR_MONSTER"
            and u.normalize_sub_type(u.actor_get(npc, "sub_type", "subType", default=None))
            != "ACTOR_SUB_SOLDIER"
        ]

        context = {
            "observation": observation,
            "frame_state": frame_state,
            "frame_no": current_frame,
            "controlled_infos": controlled_infos,
            "controlled_runtime_ids": controlled_runtime_ids,
            "official_runtime_ids": official_runtime_ids,
            "heroes": fixed_heroes,
            "friend_heroes": fixed_heroes[:3],
            "enemy_heroes": fixed_heroes[3:],
            "hero_by_runtime": hero_by_runtime,
            "raw_heroes": raw_heroes,
            "npcs": npcs,
            "soldiers_raw": soldiers_raw,
            "organs_raw": organs_raw,
            "monsters_raw": monsters_raw,
            "bullets_raw": bullets,
            "cakes_raw": cakes,
            "spawn_infos": spawn_infos,
            "actor_by_runtime": actor_by_runtime,
            "dead_runtime_ids": dead_runtime_ids,
        }
        context["official_hero"] = next(
            (snapshot for snapshot in context["friend_heroes"] if snapshot["official"]), None
        )
        if context["official_hero"] is None:
            raise ValueError("official teammate is missing from Feature Schema v1 context")

        context["organs"] = self._fixed_organs(context)
        for organ in context["organs"]:
            target_runtime = _target_runtime(organ["actor"]) if organ["actor"] else 0
            previous = self.tower_target_streak.get(organ["config_id"])
            if target_runtime and previous and previous["target"] == target_runtime:
                streak = previous["streak"] + 1
            else:
                streak = 1 if target_runtime else 0
            self.tower_target_streak[organ["config_id"]] = {
                "target": target_runtime,
                "streak": streak,
                "frame": current_frame,
            }
            organ["hit_streak"] = streak
        context["soldiers"] = self._fixed_soldiers(context)
        context["monsters"] = self._fixed_monsters(context)
        context["target38"] = next(
            (organ for organ in context["organs"][3:] if organ["alive"]), None
        )
        context["tyrant"] = context["monsters"][-1]
        context["monster_spawn"] = self._spawn_info_by_slot(context)
        context["bullet_dynamics"] = self._bullet_dynamics(context)
        context["lane_summary"] = self._lane_summary(context)
        context["resources"] = self._resource_summary(context)
        # Authoritative camp lifecycle and current combat evidence must be
        # available before the task observer decides completion/invalidation.
        self._update_monster_camp_history(context)
        self._update_recent_events(context)
        self._update_official_task_state(context)
        context["collaboration"] = self._collaboration_summary(context)
        context["lane_macro"] = self._build_lane_macro_context(context)
        self._update_lane_monitoring(context)
        context["jungle_macro"] = self._build_jungle_macro_context(context)
        context["behavior_shaping"] = self._build_behavior_shaping(context)
        return context

    def _update_lane_monitoring(self, context):
        """Accumulate observation-only indicators for normal lane attendance.

        These counters do not enter features, rewards, actions, SampleData or
        checkpoints.  They intentionally distinguish being on the assigned
        row, staying near a wave, and showing an actual enemy-minion engagement
        signal; none of the three alone is reported as guaranteed lane income.
        """

        for config_id in LANE_ROW_SIDE_Z:
            hero = context["friend_heroes"][u.HERO_CONFIG_IDS.index(config_id)]
            if not hero["alive"] or hero["side_position"] is None:
                continue
            role = "official" if hero["official"] else "controlled"
            prefixes = (f"role_{role}", f"hero_{config_id}")
            position = hero["side_position"]
            plan = context["lane_macro"][config_id]
            row = context["lane_summary"]["rows"][LANE_ROW_NAMES[config_id]]
            friendly_positions = [
                _position(soldier, self.camp) for soldier in row["friend"]
            ]
            enemy_positions = [
                _position(soldier, self.camp) for soldier in row["enemy"]
            ]
            friendly_positions = [value for value in friendly_positions if value is not None]
            enemy_positions = [value for value in enemy_positions if value is not None]
            all_wave_positions = friendly_positions + enemy_positions
            on_lane = bool(
                position[0] >= LANE_MONITOR_ENTRY_SIDE_X
                and abs(position[1] - LANE_ROW_SIDE_Z[config_id])
                <= LANE_MONITOR_ROW_TOLERANCE
            )
            near_wave = bool(
                all_wave_positions
                and min(_distance_or_inf(position, value) for value in all_wave_positions)
                <= LANE_MONITOR_WAVE_DISTANCE
            )
            enemy_runtime_ids = {
                u.actor_runtime_id(soldier) for soldier in row["enemy"]
            }
            attack_range = max(
                6000.0,
                1.25 * _value(hero["actor"], "attack_range", "attackRange"),
            )
            enemy_in_engagement_range = bool(
                enemy_positions
                and min(_distance_or_inf(position, value) for value in enemy_positions)
                <= attack_range
            )
            eating_signal = bool(
                on_lane
                and near_wave
                and (
                    _target_runtime(hero["actor"]) in enemy_runtime_ids
                    or enemy_in_engagement_range
                )
            )
            for prefix in prefixes:
                self.lane_monitor_counters[f"{prefix}_alive_steps"] += 1.0
                self.lane_monitor_counters[f"{prefix}_on_lane_steps"] += float(on_lane)
                self.lane_monitor_counters[f"{prefix}_near_wave_steps"] += float(near_wave)
                self.lane_monitor_counters[f"{prefix}_eating_signal_steps"] += float(
                    eating_signal
                )
                if plan["opening_active"]:
                    self.lane_monitor_counters[f"{prefix}_opening_steps"] += 1.0
                    self.lane_monitor_counters[
                        f"{prefix}_opening_progress_sum"
                    ] += float(plan["route_progress"])

    def lane_monitor_metrics(self):
        """Return finite episode-to-date rates for dashboard reporting."""

        result = {}
        prefixes = (
            "role_controlled",
            "role_official",
            "hero_132",
            "hero_141",
        )
        for prefix in prefixes:
            alive = self.lane_monitor_counters[f"{prefix}_alive_steps"]
            opening = self.lane_monitor_counters[f"{prefix}_opening_steps"]
            public_name = prefix.replace("role_", "")
            result[f"lane_{public_name}_alive_steps"] = float(alive)
            result[f"lane_{public_name}_on_lane_rate"] = float(
                self.lane_monitor_counters[f"{prefix}_on_lane_steps"]
                / max(alive, 1.0)
            )
            result[f"lane_{public_name}_near_wave_rate"] = float(
                self.lane_monitor_counters[f"{prefix}_near_wave_steps"]
                / max(alive, 1.0)
            )
            result[f"lane_{public_name}_eating_signal_rate"] = float(
                self.lane_monitor_counters[f"{prefix}_eating_signal_steps"]
                / max(alive, 1.0)
            )
            result[f"lane_{public_name}_opening_progress_mean"] = float(
                self.lane_monitor_counters[f"{prefix}_opening_progress_sum"]
                / max(opening, 1.0)
            )
        result["lane_first_wave_clear_count"] = float(
            sum(
                history.get("first_wave_cleared", False)
                for history in self.lane_opening_history.values()
            )
        )
        result["lane_opening_emergency_release_count"] = float(
            sum(
                history.get("release_reason")
                in ("dead", "low_hp", "recent_hit", "enemy_hero", "structure_defense")
                for history in self.lane_opening_history.values()
            )
        )
        return result

    def _dead_runtime_ids(self, frame_state):
        result = set()
        frame_action = u.safe_get(frame_state, "frame_action", "frameAction", default={}) or {}
        for event in u.safe_get(frame_action, "dead_action", "deadAction", default=[]) or []:
            death = u.safe_get(event, "death", default={}) or {}
            runtime_id = u.as_int(u.safe_get(death, "runtime_id", "runtimeId", default=0), 0)
            if runtime_id:
                result.add(runtime_id)
        return result

    def _hero_snapshot(
        self,
        actor,
        relation,
        expected_config_id,
        expected_camp,
        current_frame,
        dead_runtime_ids,
        controlled_runtime_ids,
        official_runtime_ids,
        controlled_config_ids,
        official_config_ids,
        expected_runtime_id,
    ):
        config_id = u.actor_config_id(actor) or expected_config_id
        runtime_id = u.actor_runtime_id(actor) or expected_runtime_id
        cache_key = (relation, config_id)
        is_friend = relation == "friend"
        visible = bool(actor) and u.visible_to_camp(actor, self.camp, is_friend=is_friend)
        position_valid = u.location_of(actor) is not None
        current_valid = bool(actor) and (is_friend or visible)
        last_seen_used = False
        last_seen_age = 0
        source_actor = actor or {
            "config_id": config_id,
            "runtime_id": runtime_id,
            "camp": expected_camp,
        }

        if not is_friend:
            if visible and position_valid:
                self.last_seen[cache_key] = {
                    "frame": current_frame,
                    "runtime_id": runtime_id,
                    "actor": copy.deepcopy(actor),
                }
            elif cache_key in self.last_seen:
                cached = self.last_seen[cache_key]
                age = current_frame - cached["frame"]
                if 0 <= age <= LAST_SEEN_FRAMES and (not runtime_id or runtime_id == cached["runtime_id"]):
                    source_actor = cached["actor"]
                    runtime_id = cached["runtime_id"]
                    position_valid = u.location_of(source_actor) is not None
                    last_seen_used = True
                    last_seen_age = age
                else:
                    self.last_seen.pop(cache_key, None)
            if not visible and not last_seen_used:
                source_actor = {
                    "config_id": config_id,
                    "runtime_id": runtime_id,
                    "camp": expected_camp,
                }
                position_valid = False

        alive = u.is_alive(source_actor)
        if runtime_id in dead_runtime_ids:
            alive = False
        controlled = bool(
            runtime_id in controlled_runtime_ids
            or (is_friend and config_id in controlled_config_ids)
        )
        official = bool(
            runtime_id in official_runtime_ids
            or (is_friend and config_id in official_config_ids)
        )
        snapshot = {
            "actor": source_actor,
            "relation": relation,
            "slot_exists": config_id in u.HERO_CONFIG_IDS,
            "alive": alive,
            "visible": visible,
            "current_valid": current_valid,
            "last_seen_used": last_seen_used,
            "last_seen_age": last_seen_age,
            "position_valid": position_valid,
            "controlled": controlled,
            "official": official,
            "opponent": not is_friend,
        }
        snapshot["side_position"] = _position(source_actor, self.camp) if position_valid else None
        snapshot["motion"] = self._hero_motion(snapshot, current_frame, cache_key)
        snapshot["buff_delta"] = self._buff_delta(snapshot, current_frame, cache_key)
        current_money_total = _value(source_actor, "money_cnt", "moneyCnt")
        previous_money = self.previous_money.get(cache_key)
        snapshot["money_delta"] = (
            max(0.0, current_money_total - previous_money["value"])
            if current_valid
            and previous_money is not None
            and previous_money["runtime"] == runtime_id
            and current_frame > previous_money["frame"]
            else 0.0
        )
        snapshot["money_delta_valid"] = bool(
            current_valid
            and previous_money is not None
            and previous_money["runtime"] == runtime_id
            and current_frame > previous_money["frame"]
        )
        if current_valid:
            self.previous_money[cache_key] = {
                "frame": current_frame,
                "runtime": runtime_id,
                "value": current_money_total,
            }
        return snapshot

    def _hero_motion(self, snapshot, current_frame, cache_key):
        actor = snapshot["actor"]
        runtime_id = u.actor_runtime_id(actor)
        current = snapshot["side_position"]
        previous = self.hero_motion.get(cache_key)
        result = {"valid": False, "vx": 0.0, "vz": 0.0, "speed": 0.0}
        if previous is not None:
            elapsed = current_frame - previous["frame"]
            if (
                0 < elapsed <= HERO_MOTION_MAX_FRAMES
                and current is not None
                and previous["position"] is not None
                and snapshot["alive"]
                and previous["alive"]
                and snapshot["visible"]
                and previous["visible"]
                and runtime_id == previous["runtime_id"]
            ):
                seconds = elapsed * u.RAW_FRAME_SECONDS
                vx = (current[0] - previous["position"][0]) / seconds
                vz = (current[1] - previous["position"][1]) / seconds
                result = {
                    "valid": True,
                    "vx": vx,
                    "vz": vz,
                    "speed": math.hypot(vx, vz),
                    "previous_position": previous["position"],
                }
        if snapshot["visible"] and snapshot["position_valid"]:
            self.hero_motion[cache_key] = {
                "frame": current_frame,
                "runtime_id": runtime_id,
                "position": current,
                "alive": snapshot["alive"],
                "visible": snapshot["visible"],
            }
        elif not snapshot["alive"]:
            self.hero_motion.pop(cache_key, None)
        return result

    def _buff_delta(self, snapshot, current_frame, cache_key):
        if not snapshot["current_valid"]:
            return {"new": 0, "lost": 0, "valid": False}
        current = {
            u.as_int(u.safe_get(item, "config_id", "configId", default=0), 0)
            for item in u.buff_items(snapshot["actor"])
        }
        current.discard(0)
        previous = self.previous_buffs.get(cache_key)
        self.previous_buffs[cache_key] = current
        if previous is None:
            return {"new": 0, "lost": 0, "valid": False}
        return {"new": len(current - previous), "lost": len(previous - current), "valid": True}

    def _fixed_soldiers(self, context):
        friend = u.ordered_by_runtime(
            actor
            for actor in context["soldiers_raw"]
            if u.actor_camp(actor) == self.camp and u.is_alive(actor)
        )[:10]
        enemy = u.ordered_by_runtime(
            actor
            for actor in context["soldiers_raw"]
            if u.actor_camp(actor) not in (0, self.camp) and u.is_alive(actor)
        )[:10]
        return friend + [None] * (10 - len(friend)) + enemy + [None] * (10 - len(enemy))

    def _fixed_organs(self, context):
        raw_by_config = {u.actor_config_id(actor): actor for actor in context["organs_raw"]}
        result = []
        for slot, config_id in enumerate(u.fixed_organ_config_ids(self.camp)):
            actor = raw_by_config.get(config_id)
            result.append(
                {
                    "slot": slot,
                    "config_id": config_id,
                    "actor": actor,
                    "anchor": u.ORGAN_ANCHORS[config_id],
                    "side_position": u.side_position(u.ORGAN_ANCHORS[config_id], self.camp),
                    "alive": bool(actor) and u.is_alive(actor),
                    "friend": slot < 3,
                    "type": ("outer", "high", "crystal")[slot % 3],
                }
            )
        return result

    def _fixed_monsters(self, context):
        result = []
        for slot, ((config_id, relation), slot_name) in enumerate(
            zip(schema_v1.MONSTER_TARGET_CONFIG_AND_SIDE, schema_v1.MONSTER_TARGET_SLOTS)
        ):
            anchor = u.fixed_monster_anchor(config_id, relation, self.camp)
            if config_id in (32, 33) and relation in ("own", "enemy"):
                # Both physical archer camps contain cfg32+cfg33.  Preserve
                # the frozen two-slot contract by treating the cfg32 semantic
                # slot as the inner camp and cfg33 as the outer camp, then
                # expose that camp's current living member.  This continues
                # to the companion after the major dies and cannot borrow a
                # same-cfg actor from the other physical camp.
                camp_name = (
                    "inner_archer" if config_id == 32 else "outer_archer"
                )
                candidates = list(
                    self._physical_camp_actors(
                        context, camp_name, relation=relation
                    )
                )
                member_order = KAI_OPENING_CAMP_BY_NAME[camp_name][
                    "member_configs"
                ]
                actor = min(
                    candidates,
                    key=lambda candidate: (
                        not u.is_alive(candidate),
                        member_order.index(u.actor_config_id(candidate)),
                        _distance_or_inf(
                            u.location_of(candidate), anchor
                        ),
                        u.actor_runtime_id(candidate),
                    ),
                    default=None,
                )
            else:
                candidates = [
                    actor
                    for actor in context["monsters_raw"]
                    if u.actor_config_id(actor) == config_id
                    and u.side_relation_for_monster(actor, self.camp)
                    == relation
                ]
                actor = min(
                    candidates,
                    key=lambda candidate: _distance_or_inf(
                        u.location_of(candidate), anchor
                    ),
                    default=None,
                )
            result.append(
                {
                    "slot": slot,
                    "name": slot_name,
                    "config_id": config_id,
                    "relation": relation,
                    "actor": actor,
                    "anchor": anchor,
                    "alive": bool(actor) and u.is_alive(actor),
                    "visible": bool(actor) and u.visible_to_camp(actor, self.camp, is_friend=True),
                }
            )
        return result

    def _spawn_info_by_slot(self, context):
        result = []
        for monster in context["monsters"]:
            spawn_config_id = u.monster_spawn_config_id(monster["config_id"])
            candidates = [
                info
                for info in context["spawn_infos"]
                if u.as_int(u.safe_get(info, "monster_id", "monsterId", default=0), 0)
                == spawn_config_id
            ]
            anchor = monster["anchor"]
            selected = min(
                candidates,
                key=lambda info: _distance_or_inf(
                    (
                        u.as_float(u.safe_get(info, "born_pos_x", "bornPosX", default=0)),
                        u.as_float(u.safe_get(info, "born_pos_z", "bornPosZ", default=0)),
                    ),
                    anchor,
                ),
                default=None,
            )
            result.append(selected)
        return result

    @staticmethod
    def _spawn_seconds(spawn):
        value = (
            u.safe_get(spawn, "spawn_timer", "spawnTimer", default=None)
            if spawn
            else None
        )
        seconds = u.as_float(value, float("inf")) / 1000.0
        return seconds if math.isfinite(seconds) and seconds >= 0.0 else float("inf")

    def _opening_camp_anchor(self, camp_name):
        spec = KAI_OPENING_CAMP_BY_NAME[camp_name]
        return tuple(
            float(value)
            for value in u.fixed_monster_anchor(
                spec["anchor_config"], "own", self.camp
            )
        )

    def _physical_camp_actors(self, context, camp_name, relation="own"):
        """Return every current raw actor belonging to one physical camp.

        Configs 32/33 each occur in both archer camps on the same side.  They
        must therefore be assigned by the nearest measured config-32 spawn
        anchor rather than collapsed by ``(config_id, side)``.
        """

        spec = KAI_OPENING_CAMP_BY_NAME[camp_name]
        anchor = tuple(
            float(value)
            for value in u.fixed_monster_anchor(
                spec["anchor_config"], relation, self.camp
            )
        )
        archer_anchors = None
        if camp_name in ("inner_archer", "outer_archer"):
            archer_anchors = {
                name: tuple(
                    float(value)
                    for value in u.fixed_monster_anchor(
                        KAI_OPENING_CAMP_BY_NAME[name]["anchor_config"],
                        relation,
                        self.camp,
                    )
                )
                for name in ("inner_archer", "outer_archer")
            }
        result = []
        for actor in context["monsters_raw"]:
            config_id = u.actor_config_id(actor)
            if (
                config_id not in spec["member_configs"]
                or u.side_relation_for_monster(actor, self.camp) != relation
            ):
                continue
            position = u.location_of(actor)
            if position is None:
                continue
            if archer_anchors is not None:
                nearest = min(
                    archer_anchors,
                    key=lambda name: _distance_or_inf(
                        position, archer_anchors[name]
                    ),
                )
                if nearest != camp_name:
                    continue
            result.append(actor)
        return tuple(result)

    @staticmethod
    def _opening_target_slot(config_id):
        try:
            return schema_v1.MONSTER_TARGET_CONFIG_AND_SIDE.index(
                (int(config_id), "own")
            )
        except ValueError:
            return None

    def _opening_camp_status(self, context, camp_name, kai_position=None):
        spec = KAI_OPENING_CAMP_BY_NAME[camp_name]
        anchor = self._opening_camp_anchor(camp_name)
        actors = self._physical_camp_actors(context, camp_name)
        alive = tuple(actor for actor in actors if u.is_alive(actor))
        state = self.monster_camp_history.get(anchor, {})
        target_actor = min(
            alive,
            key=lambda actor: (
                spec["member_configs"].index(u.actor_config_id(actor)),
                _distance_or_inf(kai_position, _position(actor, self.camp)),
            ),
            default=None,
        )
        target_slot = self._opening_target_slot(
            u.actor_config_id(target_actor)
            if target_actor is not None
            else spec["member_configs"][0]
        )
        return {
            "name": camp_name,
            "anchor": anchor,
            "side_anchor": u.side_position(anchor, self.camp),
            "actors": actors,
            "alive_actors": alive,
            "target_actor": target_actor,
            "target_slot": target_slot,
            "cleared": bool(state.get("cleared_once", False)),
            "lifecycle_id": int(state.get("lifecycle_id", 0)),
            "seen_count": len(state.get("cycle_seen_runtimes", ())),
            "expected_members": spec["expected_members"],
        }

    def _opening_route_waypoints(self, camp_name, official=False):
        """Return measured, side-normalized waypoints for one opening leg."""

        anchor = u.side_position(self._opening_camp_anchor(camp_name), self.camp)
        routes = {
            # Leave the lane at the measured friendly upper entrance, advance
            # through raw (-25000, 15000), then go directly to wolf. Deer is
            # intentionally not used as a transit waypoint because wolf is the
            # first protected camp.
            "wolf": (*OFFICIAL_KAI_UPPER_ROUTE_WAYPOINTS, anchor),
            "deer": (anchor,),
            # Return from deer through the measured upper entrance, cross the
            # lane only on the friendly side, then descend through the measured
            # lower entrance to pioneer. Later camps follow the outer perimeter
            # by their exact physical anchors.
            "pioneer": (
                # A previously cleared deer skips deer combat, not the
                # topology connection. This first anchor lets Kai return
                # safely from wolf before descending into lower jungle.
                u.side_position(
                    self._opening_camp_anchor("deer"), self.camp
                ),
                (-25000.0, 15000.0 - u.LANE_CENTER_Z),
                (-25000.0, 0.0),
                (-23000.0, 0.0),
                u.side_position(
                    self._opening_camp_anchor("pioneer"), self.camp
                ),
            ),
            "outer_archer": (anchor,),
            "inner_archer": (anchor,),
            "red_statue": (anchor,),
        }
        if official and camp_name == "pioneer":
            # The official pathfinder repeatedly failed the former
            # wolf->deer->inner return and could drift through mid/enemy side.
            # Once both upper camps are clear, only the measured own lower
            # entrance is required before the pioneer anchor. Progress past
            # that entrance is accepted by a monotonic region gate below.
            return (
                (-23000.0, 0.0),
                u.side_position(
                    self._opening_camp_anchor("pioneer"), self.camp
                ),
            )
        return tuple(routes[camp_name])

    def _update_monster_camp_history(self, context):
        """Track physical clears, including both members of every double camp."""

        camp_specs = []
        for relation in ("own", "enemy"):
            for camp_name, _phase, _node, anchor_config, configs, expected in (
                KAI_OPENING_CAMP_ORDER
            ):
                key = tuple(
                    float(value)
                    for value in u.fixed_monster_anchor(
                        anchor_config, relation, self.camp
                    )
                )
                actors = self._physical_camp_actors(
                    context, camp_name, relation=relation
                )
                camp_specs.append(
                    (
                        key,
                        actors,
                        expected,
                        u.monster_spawn_config_id(anchor_config),
                    )
                )
        for config_id in (59, 122):
            key = tuple(
                float(value)
                for value in u.fixed_monster_anchor(
                    config_id, "center", self.camp
                )
            )
            actors = tuple(
                actor
                for actor in context["monsters_raw"]
                if u.actor_config_id(actor) == config_id
            )
            camp_specs.append((key, actors, 1, config_id))

        friendly_positions = tuple(
            hero["side_position"]
            for hero in context["friend_heroes"]
            if hero["alive"] and hero["side_position"] is not None
        )
        for key, actors, expected_members, spawn_config_id in camp_specs:
            spawn_candidates = [
                info
                for info in context["spawn_infos"]
                if u.as_int(
                    u.safe_get(
                        info, "monster_id", "monsterId", default=0
                    ),
                    0,
                )
                == spawn_config_id
            ]
            spawn = min(
                spawn_candidates,
                key=lambda info: _distance_or_inf(
                    (
                        u.as_float(
                            u.safe_get(
                                info,
                                "born_pos_x",
                                "bornPosX",
                                default=0,
                            )
                        ),
                        u.as_float(
                            u.safe_get(
                                info,
                                "born_pos_z",
                                "bornPosZ",
                                default=0,
                            )
                        ),
                    ),
                    key,
                ),
                default=None,
            )
            spawn_timer_value = (
                u.safe_get(
                    spawn, "spawn_timer", "spawnTimer", default=None
                )
                if spawn is not None
                else None
            )
            spawn_timer_ms = u.as_float(
                spawn_timer_value, float("nan")
            )
            spawn_timer_positive = bool(
                math.isfinite(spawn_timer_ms) and spawn_timer_ms > 0.0
            )
            state = self.monster_camp_history.setdefault(
                key,
                {
                    "ever_alive": False,
                    "cleared_once": False,
                    "live_cycle_seen": False,
                    "lifecycle_id": 0,
                    "cycle_seen_runtimes": set(),
                    "expected_members": expected_members,
                    "clear_count": 0,
                    "last_clear_frame": None,
                    "last_clear_evidence": "none",
                    "spawn_timer_zero_seen": False,
                    "last_finite_spawn_timer_ms": None,
                    "timer_death_transition_count": 0,
                },
            )
            state.setdefault("cycle_seen_runtimes", set())
            state.setdefault("lifecycle_id", 0)
            state.setdefault("spawn_timer_zero_seen", False)
            state.setdefault("last_finite_spawn_timer_ms", None)
            state.setdefault("timer_death_transition_count", 0)
            state["expected_members"] = expected_members
            alive_actors = tuple(actor for actor in actors if u.is_alive(actor))
            any_alive = bool(alive_actors)
            if any_alive:
                if not state["live_cycle_seen"]:
                    state["cycle_seen_runtimes"] = set()
                    state["lifecycle_id"] = int(
                        state.get("lifecycle_id", 0)
                    ) + 1
                state["ever_alive"] = True
                state["live_cycle_seen"] = True
            state["cycle_seen_runtimes"].update(
                u.actor_runtime_id(actor)
                for actor in actors
                if u.actor_runtime_id(actor)
            )
            side_anchor = u.side_position(key, self.camp)
            observer_near = any(
                _distance_or_inf(position, side_anchor)
                <= JUNGLE_CAMP_CLEAR_NEAR_DISTANCE
                for position in friendly_positions
            )
            state["observed_runtime_ids"] = tuple(
                sorted(
                    u.actor_runtime_id(actor)
                    for actor in actors
                    if u.actor_runtime_id(actor)
                )
            )
            state["alive_runtime_ids"] = tuple(
                sorted(
                    u.actor_runtime_id(actor)
                    for actor in alive_actors
                    if u.actor_runtime_id(actor)
                )
            )
            state["observer_near"] = observer_near
            state["spawn_timer_ms"] = (
                float(spawn_timer_ms)
                if math.isfinite(spawn_timer_ms)
                else None
            )
            state["spawn_timer_positive"] = spawn_timer_positive
            state["last_observation_frame"] = context["frame_no"]
            previous_finite_timer = state["last_finite_spawn_timer_ms"]
            timer_death_transition = bool(
                spawn_timer_positive
                and state["spawn_timer_zero_seen"]
                and previous_finite_timer is not None
                and previous_finite_timer <= 0.0
            )
            state["timer_death_transition"] = timer_death_transition
            # A positive timer at episode start is the initial spawn countdown,
            # not a clear. Only a physical timer that has first reached zero
            # (camp spawned) and then jumps positive (whole camp entered
            # respawn) is authoritative clear evidence. This remains valid
            # when a teammate or opponent cleared the camp outside our vision.
            confirmed_clear = timer_death_transition
            if confirmed_clear:
                state["cleared_once"] = True
                state["live_cycle_seen"] = False
                state["last_clear_evidence"] = "spawn_timer_zero_to_positive"
                state["timer_death_transition_count"] += 1
                if state["last_clear_frame"] != context["frame_no"]:
                    state["clear_count"] += 1
                    state["last_clear_frame"] = context["frame_no"]
            if math.isfinite(spawn_timer_ms):
                if spawn_timer_ms <= 0.0:
                    state["spawn_timer_zero_seen"] = True
                state["last_finite_spawn_timer_ms"] = float(
                    spawn_timer_ms
                )

    def _start_official_farm_task(self, frame_no, position_slot):
        """Commit one learned SetPosition choice as a cross-step farm option."""

        if self._memo_context is None:
            return False
        monster_index = int(position_slot) - 10
        if not 0 <= monster_index < len(self._memo_context["monsters"]):
            return False
        official = self._memo_context["official_hero"]
        if u.actor_config_id(official["actor"]) != 193 or not official["alive"]:
            return False
        monster = self._memo_context["monsters"][monster_index]
        if (
            not monster_set_position_allowed(monster["config_id"])
            or monster_target_type(monster["config_id"]) != "ordinary_monster"
            or not monster["alive"]
        ):
            return False
        anchor = tuple(float(value) for value in monster["anchor"])
        camp_state = self.monster_camp_history.get(anchor, {})
        lifecycle_id = int(camp_state.get("lifecycle_id", 0))
        lifecycle_key = ("farm_camp", anchor, lifecycle_id)
        if lifecycle_key in self.official_task_failed_lifecycles:
            return False
        official_position = official["side_position"]
        goal = u.side_position(anchor, self.camp)
        distance = _distance_or_inf(official_position, goal)
        self.official_task_state = {
            "task_type": "farm_camp",
            "intent": 10,
            "position_slot": int(position_slot),
            "monster_index": monster_index,
            "anchor": anchor,
            "lifecycle_id": lifecycle_id,
            "official_runtime_id": u.actor_runtime_id(official["actor"]),
            "start_frame": int(frame_no),
            "task_start_frame": int(frame_no),
            "phase_frame": int(frame_no),
            "recovery_start_frame": int(frame_no),
            "phase": "planned",
            "initial_distance": distance,
            "best_distance": distance,
            "navigation_response": False,
            "navigation_response_frame": None,
            "last_navigation_progress_frame": int(frame_no),
            "last_navigation_progress_distance": distance,
            "camp_arrival_frame": None,
            "engaged": False,
            "first_engagement_frame": None,
            "engagement_frame": None,
            "last_consistent_frame": int(frame_no),
            "navigation_recovery_count": 0,
            "engagement_recovery_count": 0,
            "persistence_recovery_count": 0,
            "renewal_count": 0,
            "command_waypoint": goal,
            "using_alternative_entrance": False,
            "force_refresh": False,
            "failure_reason": "none",
        }
        self.official_task_counters["farm_camp_start"] += 1.0
        return True

    def _finish_official_task(self, frame_no, outcome, reward=False):
        task = self.official_task_state
        if task is None:
            return
        lifecycle_key = (
            task.get("task_type"),
            task.get("anchor"),
            int(task.get("lifecycle_id", 0)),
        )
        if outcome in (
            "protocol_rejected",
            "navigation_nonresponse",
            "engagement_nonresponse",
            "persistence_failure",
        ):
            self.official_task_failed_lifecycles.add(lifecycle_key)
            self.official_task_counters[
                "failed_lifecycle_lock"
            ] += 1.0
        if reward and lifecycle_key not in self.official_task_rewarded_lifecycles:
            self.official_task_rewarded_lifecycles.add(lifecycle_key)
            self.official_task_reward_pending += float(
                V1TrainingConfig.COMMAND_FARM_CAMP_COMPLETION_REWARD
            )
            self.official_task_counters["farm_camp_rewarded"] += 1.0
        if outcome == "completed":
            self.official_task_counters["task_completion"] += 1.0
        else:
            self.official_task_counters["task_abort"] += 1.0
        self.official_task_counters[f"task_end_{outcome}"] += 1.0
        self.official_task_counters["last_task_duration_frames"] = float(
            int(frame_no)
            - int(
                task.get(
                    "task_start_frame",
                    task.get("start_frame", frame_no),
                )
            )
        )
        self.official_task_state = None

    @staticmethod
    def _official_task_goal_node(monster_index):
        """Map the frozen monster bank onto the existing topology graph."""

        return {
            10: 1,
            11: 1,
            14: 2,
            16: 2,
            15: 3,
            17: 3,
            18: 3,
            2: 5,
            3: 5,
            0: 6,
            8: 7,
            9: 7,
            6: 8,
            7: 8,
            19: 9,
        }.get(int(monster_index))

    def _official_task_alternative_entrance(
        self, context, monster_index, official_position
    ):
        """Return one safe graph waypoint before retrying the camp anchor."""

        goal_node = self._official_task_goal_node(monster_index)
        if goal_node is None:
            return None
        positions = self._route_node_positions(context)
        path, next_node, _remaining = self._shortest_route(
            positions,
            official_position,
            goal_node,
        )
        if len(path) > 1 and next_node != goal_node:
            return positions[next_node]
        entrance_node = (
            0
            if goal_node in (1, 2, 3)
            else 4
            if goal_node in (5, 6, 7, 8, 9)
            else None
        )
        return (
            positions[entrance_node]
            if entrance_node is not None
            else None
        )

    def _update_official_task_state(self, context):
        """Observe navigation, engagement, persistence and authoritative result."""

        task = self.official_task_state
        if task is None:
            return
        frame_no = context["frame_no"]
        official = context["official_hero"]
        if (
            not official["alive"]
            or u.actor_runtime_id(official["actor"])
            != int(task["official_runtime_id"])
        ):
            self._finish_official_task(frame_no, "task_invalidated")
            return
        if _hp_ratio(official["actor"]) <= 0.28:
            self._finish_official_task(frame_no, "emergency_preempted")
            return

        # High-ground/crystal pressure, a nearby enemy, or direct recent hero
        # damage is a genuine higher-priority preemption.
        critical_structure_pressure = any(
            organ["alive"]
            and organ["actor"]
            and _target_runtime(organ["actor"])
            for organ in context["organs"][1:3]
        )
        nearby_enemy = any(
            enemy["alive"]
            and enemy["visible"]
            and _distance_or_inf(
                official["side_position"], enemy["side_position"]
            )
            <= KAI_NEARBY_ENEMY_INTERRUPT_DISTANCE
            for enemy in context["enemy_heroes"]
        )
        recent_hero_hurt = "hurt_by_hero" in self._event_summary(
            official, frame_no
        )["short_types"]
        tyrant_emergency = self._tyrant_status(context)["enemy_attacking"]
        if (
            critical_structure_pressure
            or nearby_enemy
            or recent_hero_hurt
            or tyrant_emergency
        ):
            self._finish_official_task(frame_no, "emergency_preempted")
            return

        anchor = tuple(task["anchor"])
        camp_state = self.monster_camp_history.get(anchor)
        if camp_state is None:
            self._finish_official_task(frame_no, "task_invalidated")
            return
        current_lifecycle = int(camp_state.get("lifecycle_id", 0))
        if (
            current_lifecycle != int(task["lifecycle_id"])
            and camp_state.get("alive_runtime_ids")
        ):
            self._finish_official_task(frame_no, "task_invalidated")
            return

        goal = u.side_position(anchor, self.camp)
        distance = _distance_or_inf(official["side_position"], goal)
        command_waypoint = task.get("command_waypoint")
        if (
            task.get("using_alternative_entrance", False)
            and command_waypoint is not None
            and _distance_or_inf(
                official["side_position"], command_waypoint
            )
            <= JUNGLE_ROUTE_NODE_REACHED_DISTANCE
        ):
            task["command_waypoint"] = goal
            task["using_alternative_entrance"] = False
            task["force_refresh"] = True
            task["phase"] = "successor"
            task["phase_frame"] = frame_no
            task["last_navigation_progress_frame"] = frame_no
            task["last_navigation_progress_distance"] = distance
            self.official_task_counters[
                "navigation_successor"
            ] += 1.0
        navigation_waypoint = task.get("command_waypoint") or goal
        navigation_distance = _distance_or_inf(
            official["side_position"], navigation_waypoint
        )
        # Initial 1,500-unit response is historical evidence, not permission
        # to wait forever. Keep a rolling pre-arrival progress watchdog.
        if (
            task.get("camp_arrival_frame") is None
            and not task["engaged"]
            and navigation_distance
            <= float(
                task.get(
                    "last_navigation_progress_distance",
                    navigation_distance,
                )
            )
            - JUNGLE_ROUTE_PROGRESS_EPSILON
        ):
            task["last_navigation_progress_frame"] = frame_no
            task["last_navigation_progress_distance"] = (
                navigation_distance
            )
        task["best_distance"] = min(float(task["best_distance"]), distance)
        if (
            float(task["initial_distance"]) - float(task["best_distance"])
            >= V1TrainingConfig.COMMAND_RESPONSE_MIN_PROGRESS_DISTANCE
            or distance <= JUNGLE_CAMP_CLEAR_NEAR_DISTANCE
        ):
            if not task["navigation_response"]:
                self.official_task_counters["navigation_response"] += 1.0
                task["navigation_response_frame"] = frame_no
                self.official_task_counters[
                    "last_navigation_latency_frames"
                ] = float(
                    frame_no
                    - int(
                        task.get(
                            "task_start_frame",
                            task["start_frame"],
                        )
                    )
                )
            task["navigation_response"] = True
            task["phase"] = "navigating"
        if (
            distance <= JUNGLE_CAMP_CLEAR_NEAR_DISTANCE
            and task.get("camp_arrival_frame") is None
        ):
            task["camp_arrival_frame"] = frame_no
            self.official_task_counters["camp_arrival"] += 1.0
            self.official_task_counters[
                "last_camp_arrival_latency_frames"
            ] = float(
                frame_no
                - int(
                    task.get(
                        "task_start_frame",
                        task["start_frame"],
                    )
                )
            )

        alive_runtimes = set(camp_state.get("alive_runtime_ids", ()))
        key = ("friend", u.actor_config_id(official["actor"]))
        correct_hit = any(
            event["frame"] == frame_no
            and event["other_runtime"] in alive_runtimes
            and event["type"] in ("hit_monster", "damage_dealt")
            for event in self.recent_events.get(key, ())
        )
        if _target_runtime(official["actor"]) in alive_runtimes:
            correct_hit = True
        if correct_hit:
            if task.get("camp_arrival_frame") is None:
                task["camp_arrival_frame"] = frame_no
                self.official_task_counters["camp_arrival"] += 1.0
                self.official_task_counters[
                    "last_camp_arrival_latency_frames"
                ] = float(
                    frame_no
                    - int(
                        task.get(
                            "task_start_frame",
                            task["start_frame"],
                        )
                    )
                )
            if not task["engaged"]:
                task["first_engagement_frame"] = frame_no
                task["engagement_frame"] = frame_no
                self.official_task_counters["engagement_response"] += 1.0
                arrival_frame = task.get("camp_arrival_frame")
                self.official_task_counters[
                    "last_arrival_to_engagement_frames"
                ] = float(
                    0
                    if arrival_frame is None
                    else frame_no - int(arrival_frame)
                )
            task["engaged"] = True
            task["phase"] = "engaged"
            task["phase_frame"] = frame_no
            task["last_consistent_frame"] = frame_no

        if camp_state.get("last_clear_frame") == frame_no:
            self._finish_official_task(
                frame_no,
                "completed" if task["engaged"] else "task_invalidated",
                reward=bool(task["engaged"]),
            )
            return
        if (
            camp_state.get("spawn_timer_positive")
            and not camp_state.get("alive_runtime_ids")
            and camp_state.get("last_clear_frame") != frame_no
        ):
            self._finish_official_task(frame_no, "task_invalidated")
            return

        age = frame_no - int(
            task.get("task_start_frame", task["start_frame"])
        )
        if age >= V1TrainingConfig.COMMAND_TASK_HARD_DEADLINE_FRAMES:
            if task["engaged"] or task.get("first_engagement_frame") is not None:
                outcome = "persistence_failure"
            elif (
                task.get("camp_arrival_frame") is not None
                or int(task.get("engagement_recovery_count", 0)) > 0
            ):
                outcome = "engagement_nonresponse"
            else:
                outcome = "navigation_nonresponse"
            self._finish_official_task(frame_no, outcome)
            return
        if (
            not task["engaged"]
            and task.get("camp_arrival_frame") is None
            and frame_no
            - int(
                task.get(
                    "last_navigation_progress_frame",
                    task["recovery_start_frame"],
                )
            )
            >= V1TrainingConfig.COMMAND_TASK_NAVIGATION_WINDOW_FRAMES
        ):
            if int(task["navigation_recovery_count"]) < 1:
                task["navigation_recovery_count"] = 1
                task["phase_frame"] = frame_no
                task["recovery_start_frame"] = frame_no
                task["camp_arrival_frame"] = None
                task["initial_distance"] = distance
                task["best_distance"] = distance
                alternative = self._official_task_alternative_entrance(
                    context,
                    task["monster_index"],
                    official["side_position"],
                )
                task["command_waypoint"] = alternative or goal
                task["last_navigation_progress_frame"] = frame_no
                task["last_navigation_progress_distance"] = (
                    _distance_or_inf(
                        official["side_position"],
                        task["command_waypoint"],
                    )
                )
                task["using_alternative_entrance"] = bool(
                    alternative is not None
                )
                task["force_refresh"] = True
                task["phase"] = "nonresponse_recovery"
                self.official_task_counters["navigation_recovery"] += 1.0
            else:
                self._finish_official_task(
                    frame_no, "navigation_nonresponse"
                )
                return
        if (
            task["navigation_response"]
            and not task["engaged"]
            and task.get("camp_arrival_frame") is not None
            and frame_no - int(task["camp_arrival_frame"])
            >= V1TrainingConfig.COMMAND_TASK_ENGAGEMENT_WINDOW_FRAMES
        ):
            if int(task["engagement_recovery_count"]) < 1:
                task["engagement_recovery_count"] = 1
                task["command_waypoint"] = goal
                task["using_alternative_entrance"] = False
                task["force_refresh"] = True
                task["phase"] = "nonresponse_recovery"
                task["phase_frame"] = frame_no
                task["recovery_start_frame"] = frame_no
                task["camp_arrival_frame"] = None
                self.official_task_counters[
                    "engagement_recovery"
                ] += 1.0
            else:
                self._finish_official_task(
                    frame_no, "engagement_nonresponse"
                )
                return
        if (
            task["engaged"]
            and frame_no - int(task["last_consistent_frame"])
            >= V1TrainingConfig.COMMAND_TASK_PERSISTENCE_WINDOW_FRAMES
        ):
            wrong_target = bool(
                _target_runtime(official["actor"])
                and _target_runtime(official["actor"]) not in alive_runtimes
            )
            left_region = distance > JUNGLE_CAMP_CLEAR_NEAR_DISTANCE
            if wrong_target or left_region or alive_runtimes:
                if int(task["persistence_recovery_count"]) < 1:
                    task["persistence_recovery_count"] = 1
                    task["command_waypoint"] = goal
                    task["using_alternative_entrance"] = False
                    task["force_refresh"] = True
                    task["phase"] = "nonresponse_recovery"
                    task["phase_frame"] = frame_no
                    task["recovery_start_frame"] = frame_no
                    task["last_consistent_frame"] = frame_no
                    self.official_task_counters[
                        "persistence_recovery"
                    ] += 1.0
                else:
                    self._finish_official_task(
                        frame_no, "persistence_failure"
                    )

    def _camp_cleared_once(self, context, monster_slot):
        anchor = tuple(float(value) for value in context["monsters"][monster_slot]["anchor"])
        return bool(self.monster_camp_history.get(anchor, {}).get("cleared_once", False))

    @staticmethod
    def _camp_member_slots(context, monster_slot):
        """Return all target slots sharing one measured physical camp anchor."""

        anchor = tuple(float(value) for value in context["monsters"][monster_slot]["anchor"])
        return tuple(
            index
            for index, monster in enumerate(context["monsters"])
            if tuple(float(value) for value in monster["anchor"]) == anchor
        )

    def _live_camp_target(self, context, monster_slot, hero_position):
        """Choose a living member, including a camp's lesser companion."""

        candidates = []
        for slot in self._camp_member_slots(context, monster_slot):
            monster = context["monsters"][slot]
            if not monster["alive"] or monster["actor"] is None:
                continue
            candidates.append(
                (
                    _distance_or_inf(
                        hero_position, _position(monster["actor"], self.camp)
                    ),
                    slot,
                )
            )
        return min(candidates)[1] if candidates else None

    def _route_node_positions(self, context):
        positions = [
            (anchor[0], anchor[1] - u.LANE_CENTER_Z)
            for anchor in u.JUNGLE_ROUTE_STATIC_RAW_ANCHORS
        ]
        # Route goals must use each physical camp's measured Debug anchor.
        # Camp 2 is close to, but not an exact x mirror of, camp 1.
        monster_nodes = {1: 10, 2: 14, 5: 2, 6: 0, 7: 8, 8: 6}
        for node, monster_slot in monster_nodes.items():
            positions[node] = u.side_position(
                context["monsters"][monster_slot]["anchor"], self.camp
            )
        positions[9] = u.side_position(context["tyrant"]["anchor"], self.camp)
        lane = context["lane_summary"]
        lane_position = lane["clash_position"]
        if lane_position is None and lane["enemy_front"] is not None:
            lane_position = lane["enemy_front"][1]
        positions.append(lane_position or (0.0, 0.0))
        own_outer, own_high, own_crystal = context["organs"][:3]
        safe = own_outer if own_outer["alive"] else own_high if own_high["alive"] else own_crystal
        positions.append(safe["side_position"])
        return tuple(positions)

    @staticmethod
    def _shortest_route(
        positions,
        start_position,
        goal_node,
        node_reached_distance=JUNGLE_ROUTE_NODE_REACHED_DISTANCE,
        completed_start_nodes=(),
    ):
        """Return the debug-topology path, next node, and remaining distance."""

        if start_position is None:
            return (goal_node,), goal_node, 0.0
        static_count = len(u.JUNGLE_ROUTE_STATIC_RAW_ANCHORS)
        # Region-gated attachment is the wall model: a lane position enters
        # upper jungle through node 0 and lower/Tyrant through node 4 instead
        # of attaching to whichever camp is closest through a wall.
        if abs(start_position[1]) <= 6500.0:
            if goal_node in (1, 2, 3):
                upper_entry = positions[0]
                committed_to_upper = bool(
                    start_position[1] >= 1200.0
                    and abs(start_position[0] - upper_entry[0]) <= 10000.0
                )
                start_node = 1 if committed_to_upper else 0
            elif goal_node in (5, 6, 7, 8, 9):
                lower_entry = positions[4]
                committed_to_lower = bool(
                    start_position[1] <= -1200.0
                    and abs(start_position[0] - lower_entry[0]) <= 10000.0
                )
                start_node = 5 if committed_to_lower else 4
            else:
                start_node = goal_node
        elif start_position[1] > 6500.0:
            upper_start = 1 if goal_node in (1, 2, 3) else 0
            start_node = min(
                range(upper_start, 4),
                key=lambda index: _distance_or_inf(start_position, positions[index]),
            )
        else:
            lower_start = 5 if goal_node in (5, 6, 7, 8, 9) else 4
            start_node = min(
                range(lower_start, static_count),
                key=lambda index: _distance_or_inf(start_position, positions[index]),
            )
        edges = list(u.JUNGLE_ROUTE_EDGES) + [
            (0, 10),
            (4, 10),
            (10, 11),
            (0, 11),
            (4, 11),
        ]
        adjacency = defaultdict(list)
        for first, second in edges:
            weight = _distance_or_inf(positions[first], positions[second])
            adjacency[first].append((second, weight))
            adjacency[second].append((first, weight))
        distances = {start_node: _distance_or_inf(start_position, positions[start_node])}
        previous = {}
        pending = set(range(len(positions)))
        while pending:
            reachable = [node for node in pending if node in distances]
            if not reachable:
                break
            node = min(reachable, key=distances.get)
            pending.remove(node)
            if node == goal_node:
                break
            for neighbor, weight in adjacency[node]:
                candidate = distances[node] + weight
                if candidate < distances.get(neighbor, float("inf")):
                    distances[neighbor] = candidate
                    previous[neighbor] = node
        if goal_node not in distances:
            return (goal_node,), goal_node, _distance_or_inf(start_position, positions[goal_node])
        path = [goal_node]
        while path[-1] != start_node:
            path.append(previous[path[-1]])
        path.reverse()
        next_node = path[0]
        if (
            (
                next_node in completed_start_nodes
                or _distance_or_inf(start_position, positions[next_node])
                <= node_reached_distance
            )
            and len(path) > 1
        ):
            next_node = path[1]
        return tuple(path), next_node, distances[goal_node]

    def _committed_shortest_route(
        self,
        positions,
        start_position,
        goal_node,
        route_signature,
        node_reached_distance=JUNGLE_ROUTE_NODE_REACHED_DISTANCE,
        completed_start_nodes=(),
    ):
        """Advance monotonically through one topology path until the goal changes.

        Reattaching to the nearest graph node on every decision can reverse a
        route after a waypoint has been crossed.  In particular, an upper-wolf
        departure could alternate between the wolf and deer nodes forever.
        The committed cursor may advance, but never moves back to an earlier
        node in the same route.
        """

        history = self.jungle_route_history
        path = tuple(history.get("committed_route_path", ()))
        cursor = int(history.get("committed_route_cursor", 0))
        needs_replan = bool(
            start_position is None
            or history.get("committed_route_signature") != route_signature
            or not path
            or path[-1] != goal_node
            or not 0 <= cursor < len(path)
            or any(not 0 <= node < len(positions) for node in path)
        )
        if needs_replan:
            path, next_node, _distance = self._shortest_route(
                positions,
                start_position,
                goal_node,
                node_reached_distance=node_reached_distance,
                completed_start_nodes=completed_start_nodes,
            )
            cursor = path.index(next_node)
            history["committed_route_signature"] = route_signature
            history["committed_route_path"] = tuple(path)
            history["committed_route_cursor"] = cursor
        else:
            while (
                cursor < len(path) - 1
                and (
                    path[cursor] in completed_start_nodes
                    or _distance_or_inf(
                        start_position, positions[path[cursor]]
                    )
                    <= node_reached_distance
                )
            ):
                cursor += 1
            history["committed_route_cursor"] = cursor

        next_node = path[cursor]
        remaining_distance = _distance_or_inf(start_position, positions[next_node])
        for index in range(cursor, len(path) - 1):
            remaining_distance += _distance_or_inf(
                positions[path[index]], positions[path[index + 1]]
            )
        return tuple(path[cursor:]), next_node, remaining_distance

    def _clear_committed_route(self):
        history = self.jungle_route_history
        history["committed_route_signature"] = None
        history["committed_route_path"] = ()
        history["committed_route_cursor"] = 0

    @staticmethod
    def _nearby_eta(heroes, destination, excluded_runtime=0):
        candidates = []
        for hero in heroes:
            if not hero["alive"] or not hero["visible"] or hero["side_position"] is None:
                continue
            if u.actor_runtime_id(hero["actor"]) == excluded_runtime:
                continue
            speed = max(1.0, _value(hero["actor"], "mov_spd", "movSpd"))
            candidates.append(_distance_or_inf(hero["side_position"], destination) / speed)
        return min(candidates, default=30.0)

    def _build_jungle_macro_context(self, context):
        """Build the rule-bounded, interruptible macro state for Kai."""

        kai = next(hero for hero in context["friend_heroes"] if u.actor_config_id(hero["actor"]) == 193)
        kai_actor = kai["actor"]
        kai_position = kai["side_position"]
        kai_runtime = u.actor_runtime_id(kai_actor)
        kai_level = _value(kai_actor, "level")
        kai_hp = _hp_ratio(kai_actor)
        history = self.jungle_route_history
        team_buff_tier = 0
        for hero in context["friend_heroes"]:
            ids = {
                u.as_int(u.safe_get(item, "config_id", "configId", default=0), 0)
                for item in u.buff_items(hero["actor"])
            }
            team_buff_tier = max(
                team_buff_tier,
                3 if 60007 in ids else 2 if 60006 in ids else 1 if 60005 in ids else 0,
            )

        opening_statuses = {
            name: self._opening_camp_status(context, name, kai_position)
            for name, *_rest in KAI_OPENING_CAMP_ORDER
        }
        if (
            kai["official"]
            and history.get("opening_fallback_active", False)
            and self.official_command_guard["fallback_confirmed"]
        ):
            # The failed hard-opening goal is released only after a different
            # valid successor command has entered platform micro_state.
            history["opening_fallback_active"] = False
            history["opening_released"] = True
            history["opening_release_reason"] = "official_noncompliant"
            history["official_noncompliant"] = True
            history["lower_farm_started"] = True
            self.official_command_guard["fallback_pending"] = False
            self.official_command_guard["fallback_source_signature"] = None
            self.official_command_guard["fallback_confirmed"] = False
        wolf_cleared = opening_statuses["wolf"]["cleared"]
        deer_cleared = opening_statuses["deer"]["cleared"]
        opening_stage_index = int(history.get("opening_stage_index", 0))
        opening_camp_name = None
        if history.get("opening_released", False):
            # Release is terminal for the protected opening. Do not let the
            # official wolf/deer proximity selector keep mutating a policy
            # feature or emitting false stage-change diagnostics afterward.
            opening_stage_index = int(
                history.get("opening_stage_index", opening_stage_index)
            )
        elif kai["official"] and not (wolf_cleared and deer_cleared):
            # The official policy owns ordinary combat target selection. Once
            # it has physically entered upper jungle, cooperate with a living
            # deer target instead of repeatedly replacing its movement with a
            # wolf command. Both camps remain mandatory before lower routing.
            if wolf_cleared:
                opening_camp_name = "deer"
            elif deer_cleared:
                opening_camp_name = "wolf"
            else:
                target_runtime = _target_runtime(kai_actor)
                deer_runtime_ids = {
                    u.actor_runtime_id(actor)
                    for actor in opening_statuses["deer"]["alive_actors"]
                }
                in_confirmed_upper = bool(
                    kai_position is not None
                    and OFFICIAL_KAI_UPPER_CONFIRMED_X_RANGE[0]
                    <= kai_position[0]
                    <= OFFICIAL_KAI_UPPER_CONFIRMED_X_RANGE[1]
                    and kai_position[1] >= OFFICIAL_KAI_UPPER_CONFIRMED_Z_MIN
                )
                deer_is_natural_target = bool(
                    target_runtime in deer_runtime_ids
                    or (
                        in_confirmed_upper
                        and opening_statuses["deer"]["target_actor"] is not None
                        and (
                            opening_statuses["wolf"]["target_actor"] is None
                            or _distance_or_inf(
                                kai_position,
                                opening_statuses["deer"]["side_anchor"],
                            )
                            < _distance_or_inf(
                                kai_position,
                                opening_statuses["wolf"]["side_anchor"],
                            )
                        )
                    )
                )
                opening_camp_name = (
                    "deer" if deer_is_natural_target else "wolf"
                )
            opening_stage_index = (
                1 if opening_camp_name == "deer" else 0
            )
        else:
            if kai["official"] and wolf_cleared and deer_cleared:
                opening_stage_index = max(2, opening_stage_index)
            while (
                opening_stage_index < len(KAI_OPENING_CAMP_ORDER)
                and opening_statuses[
                    KAI_OPENING_CAMP_ORDER[opening_stage_index][0]
                ]["cleared"]
            ):
                opening_stage_index += 1
            if opening_stage_index < len(KAI_OPENING_CAMP_ORDER):
                opening_camp_name = KAI_OPENING_CAMP_ORDER[
                    opening_stage_index
                ][0]
        history["opening_stage_index"] = opening_stage_index
        protected_stage_count = (
            3 if kai["official"] else len(KAI_OPENING_CAMP_ORDER)
        )
        if opening_stage_index >= protected_stage_count:
            history["opening_released"] = True
            history["opening_release_reason"] = (
                "protected_first_lower_cleared"
                if kai["official"]
                else "six_camps_cleared"
            )
            history["lower_farm_started"] = True
        elif (
            not history.get("opening_released", False)
            and context["frame_no"] >= KAI_OPENING_MAX_FRAME
        ):
            history["opening_released"] = True
            history["opening_release_reason"] = "opening_timeout"
            history["lower_farm_started"] = True
        opening_rotation_active = bool(
            kai["alive"]
            and not history.get("opening_released", False)
            and opening_stage_index < len(KAI_OPENING_CAMP_ORDER)
        )
        if not opening_rotation_active:
            opening_camp_name = None
        opening_camp = (
            opening_statuses[opening_camp_name]
            if opening_camp_name is not None
            else None
        )

        own_camp_anchors = {
            tuple(float(value) for value in monster["anchor"])
            for monster in context["monsters"]
            if monster["relation"] == "own"
        }
        newly_cleared = [
            key
            for key, state in self.monster_camp_history.items()
            if key in own_camp_anchors
            and state.get("last_clear_frame") == context["frame_no"]
        ]
        if (
            not opening_rotation_active
            and kai["alive"]
            and kai_position is not None
            and newly_cleared
        ):
            clear_anchor = min(
                newly_cleared,
                key=lambda key: _distance_or_inf(
                    kai_position, u.side_position(key, self.camp)
                ),
            )
            if (
                _distance_or_inf(
                    kai_position, u.side_position(clear_anchor, self.camp)
                )
                <= JUNGLE_CAMP_CLEAR_NEAR_DISTANCE
            ):
                history["camp_exit_active"] = True
                history["camp_exit_anchor"] = clear_anchor
                history["last_completed_camp_anchor"] = clear_anchor
                history["last_completed_camp_frame"] = context["frame_no"]
        if history["camp_exit_active"]:
            exit_anchor = history.get("camp_exit_anchor")
            if (
                not kai["alive"]
                or kai_position is None
                or exit_anchor is None
                or _distance_or_inf(
                    kai_position, u.side_position(exit_anchor, self.camp)
                )
                >= JUNGLE_CAMP_EXIT_DISTANCE
            ):
                history["camp_exit_active"] = False
                history["camp_exit_anchor"] = None
        tyrant = self._tyrant_status(context)
        lower_nodes_and_slots = ((5, 2), (6, 0), (7, 8), (8, 6))
        lower_candidates = []
        move_speed = max(1.0, _value(kai_actor, "mov_spd", "movSpd"))
        for node, primary_slot in lower_nodes_and_slots:
            monster = context["monsters"][primary_slot]
            member_slots = self._camp_member_slots(context, primary_slot)
            live_target = self._live_camp_target(
                context, primary_slot, kai_position
            )
            timer_seconds = min(
                self._spawn_seconds(context["monster_spawn"][slot])
                for slot in member_slots
            )
            distance = _distance_or_inf(
                kai_position, u.side_position(monster["anchor"], self.camp)
            )
            arrival = distance / move_speed
            camp_key = tuple(float(value) for value in monster["anchor"])
            cleared_before = bool(
                self.monster_camp_history.get(camp_key, {}).get(
                    "cleared_once", False
                )
            )
            if live_target is not None:
                lower_candidates.append((0.0, arrival, node, live_target))
            elif not cleared_before and timer_seconds <= arrival + 8.0:
                # Waiting is allowed only for a never-cleared camp that will
                # spawn during travel. A cleared camp becomes eligible again
                # only after a living actor is observed there.
                lower_candidates.append(
                    (timer_seconds, arrival, node, primary_slot)
                )
        lower_remaining = len(lower_candidates)

        # After the protected opening, both Kai variants observe the same
        # simultaneous high-value candidate set.  The controlled actor keeps
        # the actual choice in its frozen monster/resource slots and ordinary
        # action heads.  The planner may infer a commitment from a real attack
        # target or unambiguous physical approach, but never imposes a fixed
        # wolf/thief priority or nearest-resource rule.
        high_value_candidates = []
        if history.get("opening_released", False):
            own_wolf_slot = 14
            wolf = context["monsters"][own_wolf_slot]
            wolf_anchor = tuple(
                float(value) for value in wolf["anchor"]
            )
            wolf_clears = int(
                self.monster_camp_history.get(wolf_anchor, {}).get(
                    "clear_count", 0
                )
            )
            wolf_target = self._live_camp_target(
                context, own_wolf_slot, kai_position
            )
            if wolf_clears >= 1 and wolf_target is not None:
                wolf_distance = _distance_or_inf(
                    kai_position,
                    u.side_position(wolf["anchor"], self.camp),
                )
                high_value_candidates.append(
                    {
                        "key": "own_wolf_respawn",
                        "distance": wolf_distance,
                        "arrival": wolf_distance / move_speed,
                        "goal_node": 2,
                        "target_slot": wolf_target,
                        "runtime_ids": {
                            u.actor_runtime_id(
                                context["monsters"][slot]["actor"]
                            )
                            for slot in self._camp_member_slots(
                                context, own_wolf_slot
                            )
                            if context["monsters"][slot]["alive"]
                        },
                    }
                )

            enemy_wolf_slot = 15
            enemy_wolf = context["monsters"][enemy_wolf_slot]
            enemy_wolf_target = self._live_camp_target(
                context, enemy_wolf_slot, kai_position
            )
            if enemy_wolf_target is not None:
                enemy_wolf_distance = _distance_or_inf(
                    kai_position,
                    u.side_position(enemy_wolf["anchor"], self.camp),
                )
                high_value_candidates.append(
                    {
                        "key": "enemy_wolf",
                        "distance": enemy_wolf_distance,
                        "arrival": enemy_wolf_distance / move_speed,
                        "goal_node": 3,
                        "target_slot": enemy_wolf_target,
                        "runtime_ids": {
                            u.actor_runtime_id(
                                context["monsters"][slot]["actor"]
                            )
                            for slot in self._camp_member_slots(
                                context, enemy_wolf_slot
                            )
                            if context["monsters"][slot]["alive"]
                        },
                    }
                )

            thief_slot = 18
            thief = context["monsters"][thief_slot]
            thief_target = self._live_camp_target(
                context, thief_slot, kai_position
            )
            if thief_target is not None:
                thief_distance = _distance_or_inf(
                    kai_position,
                    u.side_position(thief["anchor"], self.camp),
                )
                high_value_candidates.append(
                    {
                        "key": "treasure_thief",
                        "distance": thief_distance,
                        "arrival": thief_distance / move_speed,
                        "goal_node": 3,
                        "target_slot": thief_target,
                        "runtime_ids": {
                            u.actor_runtime_id(
                                context["monsters"][slot]["actor"]
                            )
                            for slot in self._camp_member_slots(
                                context, thief_slot
                            )
                            if context["monsters"][slot]["alive"]
                        },
                    }
                )

        selected_high_value = None
        if high_value_candidates:
            attack_runtime = _target_runtime(kai_actor)
            selected_high_value = next(
                (
                    candidate
                    for candidate in high_value_candidates
                    if attack_runtime
                    and attack_runtime in candidate["runtime_ids"]
                ),
                None,
            )
            candidate_by_key = {
                candidate["key"]: candidate
                for candidate in high_value_candidates
            }
            committed_key = history.get(
                "learned_resource_target_key"
            )
            if selected_high_value is None and committed_key in candidate_by_key:
                selected_high_value = candidate_by_key[committed_key]
            if selected_high_value is None:
                previous_distances = dict(
                    history.get("resource_candidate_distances", {})
                )
                improvements = [
                    (
                        float(previous_distances[candidate["key"]])
                        - float(candidate["distance"]),
                        candidate,
                    )
                    for candidate in high_value_candidates
                    if candidate["key"] in previous_distances
                ]
                improvements.sort(key=lambda item: item[0])
                if improvements:
                    best_improvement, best_candidate = improvements[-1]
                    second_improvement = (
                        improvements[-2][0]
                        if len(improvements) > 1
                        else -float("inf")
                    )
                    if (
                        best_improvement
                        >= JUNGLE_ROUTE_PROGRESS_EPSILON
                        and best_improvement - second_improvement
                        >= JUNGLE_ROUTE_PROGRESS_EPSILON / 2.0
                    ):
                        selected_high_value = best_candidate
            if selected_high_value is not None:
                history["learned_resource_target_key"] = (
                    selected_high_value["key"]
                )
            elif committed_key not in candidate_by_key:
                history["learned_resource_target_key"] = None
            history["resource_candidate_distances"] = {
                candidate["key"]: float(candidate["distance"])
                for candidate in high_value_candidates
            }
            history["resource_candidate_slots"] = tuple(
                int(candidate["target_slot"])
                for candidate in high_value_candidates
            )
        else:
            history["learned_resource_target_key"] = None
            history["resource_candidate_distances"] = {}
            history["resource_candidate_slots"] = ()

        tyrant_distance = _distance_or_inf(kai_position, tyrant["position"])
        tyrant_arrival = tyrant_distance / move_speed
        tyrant_spawn_within_arrival = bool(
            tyrant["alive"]
            or (
                tyrant["spawn_timer_valid"]
                and tyrant["spawn_seconds"] <= tyrant_arrival + 12.0
            )
        )

        # The lane visit is a bounded, one-wave opportunity after upper jungle,
        # not a level gate. Only currently observed enemy soldiers between the
        # two outer-tower anchors qualify. The initial wave IDs are frozen so a
        # following wave cannot keep Kai in lane indefinitely.
        own_outer_x = context["organs"][0]["side_position"][0]
        enemy_outer_x = context["organs"][3]["side_position"][0]
        enemy_between_outers = [
            soldier
            for soldier in context["soldiers"][10:]
            if soldier is not None
            and u.is_alive(soldier)
            and _position(soldier, self.camp) is not None
            and own_outer_x <= _position(soldier, self.camp)[0] <= enemy_outer_x
        ]
        upper_cleared = bool(deer_cleared and wolf_cleared)
        lower_remaining = sum(
            not opening_statuses[name]["cleared"]
            for name in (
                "pioneer",
                "outer_archer",
                "inner_archer",
                "red_statue",
            )
        )
        # The confirmed six-camp opening no longer contains a lane-soak
        # branch. Keep the legacy fields inert for feature/checkpoint
        # compatibility.
        history["lane_decision_made"] = True
        history["lane_soak_active"] = False
        history["lane_wave_runtime_ids"] = set()
        history["lane_soak_exit_reason"] = "removed_by_six_camp_opening"

        own_active_structure_index, own_active_structure = next(
            (
                (index, organ)
                for index, organ in enumerate(context["organs"][:3])
                if organ["alive"]
            ),
            (2, context["organs"][2]),
        )
        enemy_active_structure = next(
            (organ for organ in context["organs"][3:] if organ["alive"]),
            context["organs"][5],
        )
        own_structure_position = own_active_structure["side_position"]
        enemy_structure_position = enemy_active_structure["side_position"]
        own_structure_runtime = u.actor_runtime_id(own_active_structure["actor"])
        enemy_soldiers_near_structure = sum(
            soldier is not None
            and _distance_or_inf(
                _position(soldier, self.camp), own_structure_position
            )
            <= 12000.0
            for soldier in context["soldiers"][10:]
        )
        enemy_heroes_near_structure = sum(
            enemy["alive"]
            and enemy["visible"]
            and _distance_or_inf(enemy["side_position"], own_structure_position)
            <= 15000.0
            for enemy in context["enemy_heroes"]
        )
        enemy_actively_targeting_structure = bool(
            own_structure_runtime
            and any(
                actor is not None
                and _target_runtime(actor) == own_structure_runtime
                for actor in (
                    [hero["actor"] for hero in context["enemy_heroes"]]
                    + list(context["soldiers"][10:])
                )
            )
        )
        lane_hero_dead = any(
            u.actor_config_id(hero["actor"]) in LANE_ROW_SIDE_Z
            and not hero["alive"]
            for hero in context["friend_heroes"]
        )
        structure_pressure_signal = bool(
            enemy_actively_targeting_structure
            or enemy_heroes_near_structure >= 1
            or enemy_soldiers_near_structure >= 2
        )
        critical_base_structure = own_active_structure_index >= 1
        enemy_structure_pressure = bool(
            structure_pressure_signal
            and (
                critical_base_structure
                or lane_hero_dead
            )
        )
        structure_pressure_reason = (
            "critical_base"
            if enemy_structure_pressure and critical_base_structure
            else "lane_death_outer"
            if enemy_structure_pressure
            else "outer_covered"
            if structure_pressure_signal
            else "none"
        )
        friendly_wave_at_enemy_structure = any(
            soldier is not None
            and _distance_or_inf(
                _position(soldier, self.camp), enemy_structure_position
            )
            <= 12000.0
            for soldier in context["soldiers"][:10]
        )
        friendly_push_count = sum(
            hero["alive"]
            and _distance_or_inf(
                hero["side_position"], enemy_structure_position
            )
            <= 20000.0
            for hero in context["friend_heroes"]
        )
        enemy_alive_count = sum(hero["alive"] for hero in context["enemy_heroes"])
        push_window = bool(
            not opening_rotation_active
            and context["frame_no"] >= 900
            and enemy_active_structure["alive"]
            and friendly_wave_at_enemy_structure
            and (friendly_push_count >= 2 or enemy_alive_count <= 2)
        )

        favorable_teamfight = False
        for enemy in context["enemy_heroes"]:
            if not enemy["alive"] or not enemy["visible"]:
                continue
            destination = enemy["side_position"]
            local_friends = sum(
                hero["alive"]
                and _distance_or_inf(hero["side_position"], destination)
                <= 16000.0
                for hero in context["friend_heroes"]
            )
            local_enemies = sum(
                hero["alive"]
                and hero["visible"]
                and _distance_or_inf(hero["side_position"], destination)
                <= 16000.0
                for hero in context["enemy_heroes"]
            )
            if local_friends >= 2 and local_friends >= local_enemies:
                favorable_teamfight = True
                break
        if opening_rotation_active:
            # Controlled Kai keeps wolf -> deer -> four protected lower camps;
            # official Kai leaves this branch after pioneer. Ordinary push or
            # group opportunities must not strand either variant between its
            # consecutive protected stages.
            favorable_teamfight = False
        if history["lane_soak_active"] and (push_window or favorable_teamfight):
            history["lane_soak_active"] = False
            history["lane_soak_exit_reason"] = "team_macro_window"

        if opening_rotation_active:
            opening_spec = KAI_OPENING_CAMP_BY_NAME[opening_camp_name]
            phase = opening_spec["phase"]
            goal_node = opening_spec["goal_node"]
            target_slot = opening_camp["target_slot"]
        elif (
            team_buff_tier < 3
            and tyrant_spawn_within_arrival
        ):
            phase, goal_node, target_slot = "tyrant_prepare", 9, 19
        elif high_value_candidates:
            if selected_high_value is None:
                # Keep all three frozen target/resource slots available to
                # PPO.  Until behavior reveals a choice, do not manufacture a
                # route commitment that could shape or exempt one candidate.
                goal_node, target_slot = 10, None
            else:
                goal_node = int(selected_high_value["goal_node"])
                target_slot = int(selected_high_value["target_slot"])
            phase = "control_cycle"
        elif lower_candidates:
            _timer, _arrival, goal_node, target_slot = min(lower_candidates)
            phase = "lower_clear" if team_buff_tier == 0 else "control_cycle"
        else:
            phase, goal_node, target_slot = "control_cycle", 9, 19

        macro_mode = "farm"
        # A real push or grouped fight has priority over optional farming and
        # optional Tyrant stacking. Enemy-started Tyrant remains an interrupt.
        if push_window:
            macro_mode = "push"
            phase, goal_node, target_slot = "control_cycle", 10, None
        elif favorable_teamfight:
            macro_mode = "group"
            phase, goal_node, target_slot = "control_cycle", 10, None

        # Losing a lane hero while the outer tower is pressured is not enough
        # to abandon the protected opening (six camps for controlled Kai,
        # upper pair plus pioneer for official Kai). Keep that signal available
        # for the post-opening macro, but only high-tower/crystal pressure may
        # interrupt and permanently release an active opening.
        structure_emergency = bool(
            enemy_structure_pressure
            and (
                not opening_rotation_active
                or critical_base_structure
            )
        )
        invaders = [
            enemy
            for enemy in context["enemy_heroes"]
            if enemy["alive"]
            and enemy["visible"]
            and enemy["side_position"] is not None
            and enemy["side_position"][0] < 0.0
            and abs(enemy["side_position"][1]) > 6500.0
        ]
        nearby_invaders = [
            enemy
            for enemy in invaders
            if _distance_or_inf(kai_position, enemy["side_position"])
            <= KAI_NEARBY_ENEMY_INTERRUPT_DISTANCE
        ]
        # During the protected opening, only a nearby enemy physically inside
        # our jungle is an invasion emergency. Merely pathing onto the enemy
        # side must not turn a routing failure into a permanent release.
        actionable_intruders = (
            nearby_invaders
            if opening_rotation_active or lane_hero_dead
            else []
        )
        intruder = min(
            actionable_intruders,
            key=lambda enemy: _distance_or_inf(kai_position, enemy["side_position"]),
            default=None,
        )
        ally_eta = enemy_eta = 30.0
        contest_favorable = False
        if intruder is not None:
            destination = intruder["side_position"]
            ally_eta = self._nearby_eta(context["friend_heroes"], destination, kai_runtime)
            enemy_eta = self._nearby_eta(
                context["enemy_heroes"], destination, u.actor_runtime_id(intruder["actor"])
            )
            local_friends = [
                hero for hero in context["friend_heroes"]
                if hero["alive"] and _distance_or_inf(hero["side_position"], destination) <= 16000.0
            ]
            local_enemies = [
                hero for hero in context["enemy_heroes"]
                if hero["alive"] and hero["visible"]
                and _distance_or_inf(hero["side_position"], destination) <= 16000.0
            ]
            friend_power = sum(_hp_ratio(hero["actor"]) * (1.0 + _value(hero["actor"], "level") / 15.0) for hero in local_friends)
            enemy_power = sum(_hp_ratio(hero["actor"]) * (1.0 + _value(hero["actor"], "level") / 15.0) for hero in local_enemies)
            contest_favorable = bool(kai_hp >= 0.45 and friend_power >= 0.9 * max(enemy_power, 0.1))
            if (
                kai_position is not None
                and kai_position[0] > 0.0
                and context["organs"][3]["alive"]
            ):
                contest_favorable = False

        interrupt = "none"
        if kai_hp < 0.28:
            interrupt, goal_node, target_slot = "recover", 11, None
        elif structure_emergency:
            interrupt, goal_node, target_slot = "structure_defense", 11, None
        elif tyrant["enemy_attacking"] and kai_level >= 3:
            interrupt, goal_node, target_slot = "tyrant_emergency", 9, 19
        elif intruder is not None:
            if contest_favorable:
                interrupt = "contest_enemy"
                # Static route anchors are stored in camp-1 raw coordinates,
                # which are also the canonical side-coordinate x values.
                goal_node = min(
                    range(len(u.JUNGLE_ROUTE_STATIC_RAW_ANCHORS)),
                    key=lambda node: _distance_or_inf(
                        (
                            u.JUNGLE_ROUTE_STATIC_RAW_ANCHORS[node][0],
                            u.JUNGLE_ROUTE_STATIC_RAW_ANCHORS[node][1]
                            - u.LANE_CENTER_Z,
                        ),
                        intruder["side_position"],
                    ),
                )
                target_slot = None
            else:
                interrupt, goal_node, target_slot = "retreat", 11, None

        if opening_rotation_active and interrupt != "none":
            history["opening_released"] = True
            history["opening_release_reason"] = interrupt
            history["lower_farm_started"] = True
            opening_rotation_active = False
            opening_camp_name = None
            opening_camp = None

        positions = self._route_node_positions(context)
        route_commit_signature = (phase, macro_mode, interrupt, goal_node)
        staged_waypoint = None
        official_upper_staged_active = False
        deer_ready = False
        opening_staged_active = False
        official_lower_topology_active = bool(
            opening_rotation_active
            and interrupt == "none"
            and kai["official"]
            and phase == "lower_clear"
            and opening_camp_name != "pioneer"
        )
        official_completed_lower_nodes = ()
        if official_lower_topology_active and opening_stage_index > 2:
            previous_camp_name = KAI_OPENING_CAMP_ORDER[
                opening_stage_index - 1
            ][0]
            previous_goal_node = KAI_OPENING_CAMP_BY_NAME[
                previous_camp_name
            ]["goal_node"]
            if opening_statuses[previous_camp_name]["cleared"]:
                # A timer-confirmed clear is stronger evidence than being a
                # few units inside an exact topology radius. Never send the
                # official hero back to the camp it just completed.
                official_completed_lower_nodes = (previous_goal_node,)
        if (
            opening_rotation_active
            and interrupt == "none"
            and not official_lower_topology_active
        ):
            route_key = opening_camp_name
            route_changed = history.get("opening_route_key") != route_key
            if route_changed:
                history["opening_route_key"] = route_key
                history["opening_waypoint_index"] = 0
            waypoints = self._opening_route_waypoints(
                route_key, official=kai["official"]
            )
            waypoint_reached_distance = (
                OFFICIAL_KAI_UPPER_ENTRY_REACHED_DISTANCE
                if kai["official"]
                else KAI_OPENING_WAYPOINT_REACHED_DISTANCE
            )
            waypoint_index = min(
                int(history.get("opening_waypoint_index", 0)),
                len(waypoints) - 1,
            )
            if (
                route_changed
                and _distance_or_inf(kai_position, waypoints[-1])
                <= waypoint_reached_distance
            ):
                # Late/partial observations may first see Kai already at the
                # camp itself. Accept only the final camp anchor as direct
                # evidence; proximity to an intermediate/later transit point
                # may not skip required route topology.
                waypoint_index = len(waypoints) - 1
            # Only measured hero position can advance a route. Command-state
            # echo/acceptance is deliberately not treated as movement. Route
            # points advance sequentially; proximity to a later point cannot
            # skip a required topology connection.
            while (
                waypoint_index < len(waypoints) - 1
                and _distance_or_inf(
                    kai_position, waypoints[waypoint_index]
                )
                <= waypoint_reached_distance
            ):
                waypoint_index += 1
            if kai["official"] and kai_position is not None:
                side_x, side_z = kai_position
                if (
                    route_key == "wolf"
                    and waypoint_index == 0
                    and side_x >= OFFICIAL_KAI_BASE_EXIT_PASS_X
                    and side_z >= 0.0
                ):
                    waypoint_index = 1
                if (
                    route_key == "wolf"
                    and waypoint_index == 1
                    and side_x >= OFFICIAL_KAI_CORRIDOR_PASS_X
                    and side_z >= 0.0
                ):
                    waypoint_index = 2
                if (
                    route_key == "wolf"
                    and waypoint_index < len(waypoints) - 1
                    and OFFICIAL_KAI_UPPER_CONFIRMED_X_RANGE[0]
                    <= side_x
                    <= OFFICIAL_KAI_UPPER_CONFIRMED_X_RANGE[1]
                    and side_z >= OFFICIAL_KAI_UPPER_CONFIRMED_Z_MIN
                ):
                    # Official pathfinding may enter the verified upper
                    # corridor above the exact lane-side waypoint. This
                    # physical region proves correct upper entry and is
                    # stronger than forcing it back to a stale 3k circle.
                    waypoint_index = len(waypoints) - 1
                elif route_key == "pioneer":
                    # From either upper camp, descend through the own lower
                    # entrance. Crossing it on our side is stronger evidence
                    # than exact arrival and prevents a stale waypoint from
                    # pulling Kai back upward or toward mid.
                    if (
                        waypoint_index == 0
                        and OFFICIAL_KAI_LOWER_ENTRY_PASS_X_RANGE[0]
                        <= side_x
                        <= OFFICIAL_KAI_LOWER_ENTRY_PASS_X_RANGE[1]
                        and side_z <= OFFICIAL_KAI_LOWER_ENTRY_PASS_Z_MAX
                    ):
                        waypoint_index = len(waypoints) - 1
            history["opening_waypoint_index"] = waypoint_index
            staged_waypoint = waypoints[waypoint_index]
            next_node = goal_node
            committed_path = (goal_node,)
            remaining = waypoints[waypoint_index:]
            path_distance = _distance_or_inf(kai_position, remaining[0]) + sum(
                _distance_or_inf(first, second)
                for first, second in zip(remaining, remaining[1:])
            )
            opening_staged_active = True
            self._clear_committed_route()
            if route_key == "wolf":
                history["official_upper_route_stage"] = waypoint_index
                history["official_upper_entry_reached"] = waypoint_index >= 3
                history["official_upper_inner_reached"] = waypoint_index >= 4
        elif kai["official"] and phase in ("upper_deer", "upper_wolf"):
            if interrupt != "none":
                history["official_upper_route_interrupted"] = True
            elif not kai["alive"]:
                history["official_upper_route_stage"] = 0
                history["official_upper_entry_reached"] = False
                history["official_upper_inner_reached"] = False
                history["official_upper_route_interrupted"] = False
            elif kai_position is not None:
                stage = int(history["official_upper_route_stage"])
                if history["official_upper_route_interrupted"]:
                    if kai_position[1] < -3500.0:
                        stage = 1
                    elif (
                        abs(kai_position[1]) <= 3500.0
                        and _distance_or_inf(
                            kai_position,
                            OFFICIAL_KAI_UPPER_ROUTE_WAYPOINTS[2],
                        )
                        > OFFICIAL_KAI_UPPER_ENTRY_REACHED_DISTANCE
                    ):
                        stage = 2
                    history["official_upper_route_interrupted"] = False
                elif stage >= 2 and kai_position[1] < -3500.0:
                    # This is a real lower-corridor displacement, not normal
                    # upward travel after crossing the entry.
                    stage = 1

                # Exact arrival advances the fixed route stages. Command
                # acceptance alone never advances them.
                while stage < len(OFFICIAL_KAI_UPPER_ROUTE_WAYPOINTS):
                    waypoint = OFFICIAL_KAI_UPPER_ROUTE_WAYPOINTS[stage]
                    if (
                        _distance_or_inf(kai_position, waypoint)
                        > OFFICIAL_KAI_UPPER_ENTRY_REACHED_DISTANCE
                    ):
                        break
                    stage += 1

                # Sampling can observe the official hero after it has already
                # crossed a waypoint. Treat measured physical progress into
                # the upper corridor as stronger evidence than Euclidean
                # distance back to the entrance.
                in_upper_x_band = -28500.0 <= kai_position[0] <= -16000.0
                if in_upper_x_band and kai_position[1] >= 3500.0:
                    stage = max(stage, 3)
                if in_upper_x_band and kai_position[1] >= 8000.0:
                    stage = max(stage, 4)

                # Once deer is cleared, its measured anchor becomes the final
                # wall-safe transit point before wolf. A hero already near
                # deer or wolf may skip that redundant transit.
                if phase == "upper_wolf" and stage >= 4:
                    if (
                        _distance_or_inf(kai_position, positions[1]) <= 7000.0
                        or _distance_or_inf(kai_position, positions[2]) <= 8000.0
                    ):
                        stage = 5

                history["official_upper_route_stage"] = stage
                history["official_upper_entry_reached"] = stage >= 3
                history["official_upper_inner_reached"] = stage >= 4

            deer_members = self._camp_member_slots(context, 10)
            deer_alive = any(
                context["monsters"][slot]["alive"] for slot in deer_members
            )
            deer_spawn_due = min(
                self._spawn_seconds(context["monster_spawn"][slot])
                for slot in deer_members
            ) <= 0.05
            deer_ready = bool(deer_alive or deer_spawn_due)

            if interrupt == "none":
                route_stage = int(history["official_upper_route_stage"])
                direct_goal = bool(
                    (phase == "upper_deer" and route_stage >= 4 and deer_ready)
                    or (phase == "upper_wolf" and route_stage >= 5)
                )
                if direct_goal:
                    next_node = goal_node
                    committed_path = (goal_node,)
                    path_distance = _distance_or_inf(
                        kai_position, positions[goal_node]
                    )
                else:
                    # Deer phase holds raw-z=15000 until spawn. Wolf recovery
                    # reuses entry -> inner -> deer before the direct wolf leg.
                    if phase == "upper_deer":
                        waypoint_index = min(
                            route_stage,
                            len(OFFICIAL_KAI_UPPER_ROUTE_WAYPOINTS) - 1,
                        )
                        staged_waypoint = OFFICIAL_KAI_UPPER_ROUTE_WAYPOINTS[
                            waypoint_index
                        ]
                        remaining_waypoints = list(
                            OFFICIAL_KAI_UPPER_ROUTE_WAYPOINTS[waypoint_index:]
                        ) + [positions[1]]
                    else:
                        wolf_route = list(
                            OFFICIAL_KAI_UPPER_ROUTE_WAYPOINTS
                        ) + [positions[1]]
                        waypoint_index = min(
                            route_stage, len(wolf_route) - 1
                        )
                        staged_waypoint = wolf_route[waypoint_index]
                        remaining_waypoints = wolf_route[waypoint_index:] + [
                            positions[2]
                        ]
                    next_node = OFFICIAL_KAI_UPPER_ENTRY_NODE
                    committed_path = (
                        OFFICIAL_KAI_UPPER_ENTRY_NODE,
                        goal_node,
                    )
                    path_distance = _distance_or_inf(
                        kai_position, remaining_waypoints[0]
                    ) + sum(
                        _distance_or_inf(first, second)
                        for first, second in zip(
                            remaining_waypoints,
                            remaining_waypoints[1:],
                        )
                    )
                    official_upper_staged_active = True
                self._clear_committed_route()
            else:
                committed_path, next_node, path_distance = (
                    self._committed_shortest_route(
                        positions,
                        kai_position,
                        goal_node,
                        route_commit_signature,
                    )
                )
        elif interrupt == "none" and phase == "upper_deer":
            # Controlled Kai's direct opening-to-deer route is already
            # platform-confirmed on both camps.
            self._clear_committed_route()
            next_node = goal_node
            path_distance = _distance_or_inf(kai_position, positions[goal_node])
            committed_path = (goal_node,)
        elif (
            interrupt == "none"
            and phase == "upper_wolf"
            and kai_position is not None
            and kai_position[1] >= 6500.0
        ):
            # A post-opening/recovery wolf goal may stay direct while Kai is
            # physically in upper jungle. Otherwise the committed graph must
            # reacquire the measured upper entrance.
            self._clear_committed_route()
            next_node = goal_node
            path_distance = _distance_or_inf(kai_position, positions[goal_node])
            committed_path = (goal_node,)
        else:
            committed_path, next_node, path_distance = self._committed_shortest_route(
                positions,
                kai_position,
                goal_node,
                route_commit_signature,
                node_reached_distance=(
                    OFFICIAL_KAI_UPPER_ENTRY_REACHED_DISTANCE
                    if official_lower_topology_active
                    else JUNGLE_ROUTE_NODE_REACHED_DISTANCE
                ),
                completed_start_nodes=official_completed_lower_nodes,
            )
        elapsed = (
            context["frame_no"] - history["frame"]
            if history["frame"] is not None
            else 0
        )
        moved = _distance_or_inf(kai_position, history["position"])
        if elapsed > 0 and path_distance > 3500.0 and moved < 150.0:
            history["stuck_frames"] += elapsed
        else:
            history["stuck_frames"] = 0
        history["frame"] = context["frame_no"]
        history["position"] = kai_position
        goal_signature = (phase, macro_mode, interrupt, goal_node)
        previous_path_distance = history.get("path_distance")
        if history.get("goal_signature") != goal_signature:
            history["best_path_distance"] = path_distance
            history["no_progress_frames"] = 0
            history["wrong_way_frames"] = 0
        else:
            best_path_distance = history.get("best_path_distance")
            if (
                best_path_distance is None
                or path_distance
                <= best_path_distance - JUNGLE_ROUTE_PROGRESS_EPSILON
            ):
                history["best_path_distance"] = min(
                    path_distance,
                    best_path_distance
                    if best_path_distance is not None
                    else path_distance,
                )
                history["no_progress_frames"] = 0
            elif elapsed > 0 and path_distance > 3500.0:
                history["no_progress_frames"] += elapsed
            if (
                elapsed > 0
                and previous_path_distance is not None
                and path_distance
                >= previous_path_distance + JUNGLE_ROUTE_PROGRESS_EPSILON
                and path_distance > 3500.0
            ):
                history["wrong_way_frames"] += elapsed
            elif (
                previous_path_distance is None
                or path_distance
                <= previous_path_distance + JUNGLE_ROUTE_PROGRESS_EPSILON
            ):
                history["wrong_way_frames"] = 0
        route_stuck_triggered = bool(
            history["stuck_frames"] * u.RAW_FRAME_SECONDS >= 2.0
            or history["no_progress_frames"] * u.RAW_FRAME_SECONDS
            >= JUNGLE_ROUTE_NO_PROGRESS_SECONDS
            or history["wrong_way_frames"] * u.RAW_FRAME_SECONDS
            >= JUNGLE_ROUTE_WRONG_WAY_SECONDS
        )
        # `route_stuck` is an event, not a persistent state.  A persistent
        # value caused collision recovery to be restarted after every
        # successful escape step, producing a four-direction loop.  Reset the
        # progress baseline when the event fires so another recovery is only
        # requested after a fresh no-progress interval.
        route_stuck = bool(route_stuck_triggered)
        if route_stuck:
            history["stuck_frames"] = 0
            history["no_progress_frames"] = 0
            history["wrong_way_frames"] = 0
            history["best_path_distance"] = path_distance
        route_shaping_reward = 0.0
        live_committed_navigation = bool(
            target_slot is not None
            and 0 <= int(target_slot) < len(context["monsters"])
            and context["monsters"][int(target_slot)]["alive"]
            and interrupt == "none"
        )
        if (
            live_committed_navigation
            and history.get("goal_signature") == goal_signature
            and previous_path_distance is not None
            and elapsed > 0
            and previous_path_distance - path_distance
            >= JUNGLE_ROUTE_PROGRESS_EPSILON
        ):
            history["last_route_progress_frame"] = context["frame_no"]
            history["last_route_progress_signature"] = goal_signature
        route_progress_confirmed = bool(
            live_committed_navigation
            and history.get("last_route_progress_signature")
            == goal_signature
            and history.get("last_route_progress_frame") is not None
            and context["frame_no"]
            - int(history["last_route_progress_frame"])
            <= V1TrainingConfig.NO_ATTACKABLE_TARGET_MAX_ELAPSED_FRAMES
        )
        if (
            not kai["official"]
            and kai["alive"]
            and live_committed_navigation
            and history.get("goal_signature") == goal_signature
            and previous_path_distance is not None
            and elapsed > 0
        ):
            progress = (previous_path_distance - path_distance) / float(
                V1TrainingConfig.JUNGLE_ROUTE_PROGRESS_DISTANCE
            )
            route_shaping_reward += u.signed_clip(progress) * float(
                V1TrainingConfig.JUNGLE_ROUTE_PROGRESS_REWARD
            )
        enemy_side_incursion = bool(
            kai_position is not None
            and kai_position[0] > 0.0
            and abs(kai_position[1]) > 6500.0
            and context["organs"][3]["alive"]
            and interrupt == "none"
        )
        # Enemy-side entry and route-stuck remain explicit diagnostics and
        # recovery triggers, but PPO 3.5 removes both negative reward terms.
        history["goal_signature"] = goal_signature
        history["path_distance"] = path_distance
        if opening_staged_active or official_upper_staged_active:
            next_waypoint = staged_waypoint
        else:
            next_waypoint = positions[next_node]
        opening_camp_lifecycle_id = (
            int(opening_camp.get("lifecycle_id", 0))
            if opening_camp is not None
            else 0
        )
        opening_fallback_active = bool(
            history.get("opening_fallback_active", False)
        )
        if (
            kai["official"]
            and opening_rotation_active
            and interrupt == "none"
            and not opening_fallback_active
            and opening_camp_name
        ):
            current_signature = self._jungle_position_goal_signature(
                kai_runtime,
                opening_camp_name,
                opening_camp_lifecycle_id,
                next_waypoint,
            )
            if self._official_goal_failed(current_signature):
                fallback_candidates = []
                for name, *_rest in KAI_OPENING_CAMP_ORDER:
                    if name == opening_camp_name:
                        continue
                    status = opening_statuses[name]
                    candidate_signature = (
                        self._jungle_position_goal_signature(
                            kai_runtime,
                            name,
                            int(status.get("lifecycle_id", 0)),
                            status["side_anchor"],
                            fallback=True,
                        )
                    )
                    if self._official_goal_failed(candidate_signature):
                        continue
                    fallback_candidates.append(
                        (
                            0 if status["alive_actors"] else 1,
                            _distance_or_inf(
                                kai_position, status["side_anchor"]
                            ),
                            name,
                            status,
                        )
                    )
                safe_positions = [
                    organ["side_position"]
                    for organ in context["organs"][:3]
                    if organ["alive"]
                    and organ["side_position"] is not None
                ]
                safe_positions.extend(
                    friend["side_position"]
                    for friend in context["friend_heroes"]
                    if not friend["official"]
                    and friend["alive"]
                    and friend["side_position"] is not None
                )
                for safe_index, safe_position in enumerate(safe_positions):
                    safe_name = f"safe_handoff_{safe_index}"
                    safe_signature = (
                        self._jungle_position_goal_signature(
                            kai_runtime,
                            safe_name,
                            0,
                            safe_position,
                            fallback=True,
                        )
                    )
                    if self._official_goal_failed(safe_signature):
                        continue
                    fallback_candidates.append(
                        (
                            2,
                            _distance_or_inf(
                                kai_position, safe_position
                            ),
                            safe_name,
                            {
                                "lifecycle_id": 0,
                                "target_slot": target_slot,
                                "side_anchor": safe_position,
                                "alive_actors": (),
                            },
                        )
                    )
                if fallback_candidates:
                    _, _, fallback_name, fallback_status = min(
                        fallback_candidates
                    )
                    opening_fallback_active = True
                    history["opening_fallback_active"] = True
                    history[
                        "opening_fallback_source_signature"
                    ] = current_signature
                    history[
                        "opening_fallback_camp_name"
                    ] = fallback_name
                    history[
                        "opening_fallback_lifecycle_id"
                    ] = int(fallback_status.get("lifecycle_id", 0))
                    history["opening_fallback_position"] = (
                        fallback_status["side_anchor"]
                    )
                    history["opening_fallback_target_slot"] = (
                        fallback_status["target_slot"]
                    )
                    history["official_noncompliant"] = True
                    self.official_command_guard[
                        "fallback_pending"
                    ] = True
                    self.official_command_guard[
                        "fallback_source_signature"
                    ] = current_signature
                    self.official_command_guard[
                        "fallback_confirmed"
                    ] = False
                    opening_camp_name = fallback_name
                    opening_camp = fallback_status
                    opening_camp_lifecycle_id = int(
                        fallback_status.get("lifecycle_id", 0)
                    )
                    target_slot = fallback_status["target_slot"]
                    next_waypoint = fallback_status["side_anchor"]
                    if fallback_name in KAI_OPENING_CAMP_BY_NAME:
                        next_node = KAI_OPENING_CAMP_BY_NAME[
                            fallback_name
                        ]["goal_node"]
                    path_distance = _distance_or_inf(
                        kai_position, next_waypoint
                    )

        if opening_fallback_active:
            fallback_name = str(
                history.get("opening_fallback_camp_name", "")
            )
            if fallback_name in opening_statuses:
                fallback_status = opening_statuses[fallback_name]
                opening_camp_name = fallback_name
                opening_camp = fallback_status
                opening_camp_lifecycle_id = int(
                    history.get(
                        "opening_fallback_lifecycle_id",
                        fallback_status.get("lifecycle_id", 0),
                    )
                )
                target_slot = fallback_status["target_slot"]
                next_waypoint = fallback_status["side_anchor"]
                next_node = KAI_OPENING_CAMP_BY_NAME[
                    fallback_name
                ]["goal_node"]
                path_distance = _distance_or_inf(
                    kai_position, next_waypoint
                )
            elif fallback_name.startswith("safe_handoff_"):
                fallback_position = history.get(
                    "opening_fallback_position"
                )
                if fallback_position is not None:
                    opening_camp_name = fallback_name
                    opening_camp = None
                    opening_camp_lifecycle_id = 0
                    target_slot = history.get(
                        "opening_fallback_target_slot", target_slot
                    )
                    next_waypoint = fallback_position
                    path_distance = _distance_or_inf(
                        kai_position, next_waypoint
                    )
        opening_monster_anchor_distance = 0.0
        opening_leash_recover = False
        if (
            opening_rotation_active
            and opening_camp is not None
            and not kai["official"]
            and opening_camp["alive_actors"]
        ):
            member_distances = [
                _distance_or_inf(
                    _position(actor, self.camp),
                    opening_camp["side_anchor"],
                )
                for actor in opening_camp["alive_actors"]
                if _position(actor, self.camp) is not None
            ]
            opening_monster_anchor_distance = max(
                member_distances, default=0.0
            )
            opening_leash_recover = bool(
                opening_monster_anchor_distance
                > KAI_OPENING_MONSTER_LEASH_DISTANCE
            )
            if opening_leash_recover:
                # Recovery is based on the monster leaving its physical camp,
                # never merely on Kai being more than 4,500 from the anchor.
                next_waypoint = opening_camp["side_anchor"]
                path_distance = _distance_or_inf(
                    kai_position, next_waypoint
                )
        next_waypoint_dx = (
            next_waypoint[0] - kai_position[0] if kai_position is not None else 0.0
        )
        next_waypoint_dz = (
            next_waypoint[1] - kai_position[1] if kai_position is not None else 0.0
        )
        opening_target_actor = (
            opening_camp["target_actor"]
            if opening_rotation_active and opening_camp is not None
            else None
        )
        opening_target_position = (
            _position(opening_target_actor, self.camp)
            if opening_target_actor is not None
            else None
        )
        opening_target_attackable = False
        opening_wait_for_spawn = False
        opening_target_dx = 0.0
        opening_target_dz = 0.0
        if opening_rotation_active and opening_camp is not None:
            # Some protocol/debug frames omit hero attack_range. A missing
            # field must not leave Kai permanently movement-locked at range.
            attack_range = max(
                2500.0, _value(kai_actor, "attack_range", "attackRange")
            )
            opening_target_attackable = bool(
                not opening_leash_recover
                and
                opening_target_actor is not None
                and u.is_alive(opening_target_actor)
                and opening_target_position is not None
                and _distance_or_inf(kai_position, opening_target_position)
                <= attack_range
            )
            opening_wait_for_spawn = bool(
                not opening_leash_recover
                and
                opening_target_actor is None
                and _distance_or_inf(
                    kai_position, opening_camp["side_anchor"]
                )
                <= KAI_OPENING_WAYPOINT_REACHED_DISTANCE
            )
            if opening_target_position is not None and kai_position is not None:
                opening_target_dx = (
                    opening_target_position[0] - kai_position[0]
                )
                opening_target_dz = (
                    opening_target_position[1] - kai_position[1]
                )
        return {
            "valid": bool(kai["alive"] and kai_position is not None),
            "kai_is_official": kai["official"],
            "kai_level": kai_level,
            "phase": phase,
            "mode": macro_mode,
            "interrupt": interrupt,
            "buff_tier": team_buff_tier,
            "deer_cleared": deer_cleared,
            "wolf_cleared": wolf_cleared,
            "opening_rotation_active": opening_rotation_active,
            "opening_stage_index": int(history["opening_stage_index"]),
            "opening_camp_name": opening_camp_name or "",
            "opening_camp_lifecycle_id": int(
                opening_camp_lifecycle_id
            ),
            "opening_fallback_active": bool(opening_fallback_active),
            "official_noncompliant": bool(
                history.get("official_noncompliant", False)
            ),
            "opening_release_reason": history["opening_release_reason"],
            "official_upper_entry_reached": bool(
                history["official_upper_entry_reached"]
            ),
            "official_upper_inner_reached": bool(
                history["official_upper_inner_reached"]
            ),
            "official_upper_route_stage": int(
                history["official_upper_route_stage"]
            ),
            "official_upper_route_interrupted": bool(
                history["official_upper_route_interrupted"]
            ),
            "official_upper_route_stage_name": OFFICIAL_KAI_UPPER_ROUTE_STAGE_NAMES[
                min(
                    int(history["official_upper_route_stage"]),
                    len(OFFICIAL_KAI_UPPER_ROUTE_STAGE_NAMES) - 1,
                )
            ],
            "lower_farm_started": bool(history["lower_farm_started"]),
            "lower_remaining": lower_remaining,
            "high_value_candidate_slots": tuple(
                history.get("resource_candidate_slots", ())
            ),
            "learned_resource_target_key": history.get(
                "learned_resource_target_key"
            ),
            "tyrant": tyrant,
            "tyrant_spawn_within_arrival": tyrant_spawn_within_arrival,
            "enemy_outer_destroyed": not context["organs"][3]["alive"],
            "own_jungle_invaded": bool(invaders),
            "contest_favorable": contest_favorable,
            "ally_eta": ally_eta,
            "enemy_eta": enemy_eta,
            "team_emergency": interrupt != "none",
            "structure_pressure_reason": structure_pressure_reason,
            "goal_node": goal_node,
            "next_node": next_node,
            "committed_route_path": tuple(committed_path),
            "committed_route_cursor": int(
                history.get("committed_route_cursor", 0)
            ),
            "next_waypoint": next_waypoint,
            "next_waypoint_dx": next_waypoint_dx,
            "next_waypoint_dz": next_waypoint_dz,
            "opening_target_attackable": opening_target_attackable,
            "opening_wait_for_spawn": opening_wait_for_spawn,
            "opening_leash_recover": opening_leash_recover,
            "opening_monster_anchor_distance": float(
                opening_monster_anchor_distance
            ),
            "opening_target_actor": opening_target_actor,
            "opening_target_dx": opening_target_dx,
            "opening_target_dz": opening_target_dz,
            "path_distance": path_distance,
            "route_progress": u.clip(1.0 - path_distance / 100000.0),
            "route_progress_confirmed": route_progress_confirmed,
            "route_stuck": route_stuck,
            "camp_exit_active": history["camp_exit_active"],
            "camp_exit_anchor": history["camp_exit_anchor"],
            "last_completed_camp_anchor": history[
                "last_completed_camp_anchor"
            ],
            "lane_wave_runtime_ids": tuple(history["lane_wave_runtime_ids"]),
            "lane_soak_exit_reason": history["lane_soak_exit_reason"],
            "route_shaping_reward": route_shaping_reward,
            "enemy_side_incursion": enemy_side_incursion,
            "target_slot": target_slot,
            "intruder_runtime": u.actor_runtime_id(intruder["actor"]) if intruder else 0,
        }

    def _bullet_dynamics(self, context):
        current_frame = context["frame_no"]
        result = {}
        active_ids = set()
        for bullet in context["bullets_raw"]:
            runtime_id = u.actor_runtime_id(bullet)
            position = _position(bullet, self.camp)
            if not runtime_id or position is None:
                continue
            active_ids.add(runtime_id)
            previous = self.bullet_motion.get(runtime_id)
            dynamic = {"valid": False, "vx": 0.0, "vz": 0.0, "speed": 0.0}
            if previous is not None:
                elapsed = current_frame - previous["frame"]
                if 0 < elapsed <= BULLET_MOTION_MAX_FRAMES:
                    seconds = elapsed * u.RAW_FRAME_SECONDS
                    vx = (position[0] - previous["position"][0]) / seconds
                    vz = (position[1] - previous["position"][1]) / seconds
                    dynamic = {"valid": True, "vx": vx, "vz": vz, "speed": math.hypot(vx, vz)}
            result[runtime_id] = dynamic
            self.bullet_motion[runtime_id] = {"frame": current_frame, "position": position}
        for runtime_id in list(self.bullet_motion):
            if runtime_id not in active_ids and current_frame - self.bullet_motion[runtime_id]["frame"] > BULLET_MOTION_MAX_FRAMES:
                self.bullet_motion.pop(runtime_id, None)
        return result

    def _lane_summary(self, context):
        """Build the shared, side-normalized lane snapshot and update history."""

        friend = [actor for actor in context["soldiers"][:10] if actor is not None and u.is_alive(actor)]
        enemy = [actor for actor in context["soldiers"][10:] if actor is not None and u.is_alive(actor)]

        def front_of(actors, friend_side):
            positioned = [(actor, _position(actor, self.camp)) for actor in actors]
            positioned = [(actor, position) for actor, position in positioned if position is not None]
            if not positioned:
                return None
            key = (lambda item: item[1][0])
            return (max if friend_side else min)(positioned, key=key)

        def actors_in_row(actors, row_name):
            result = []
            for actor in actors:
                position = _position(actor, self.camp)
                if position is None:
                    continue
                nearest_config = min(
                    LANE_ROW_SIDE_Z,
                    key=lambda config_id: abs(
                        position[1] - LANE_ROW_SIDE_Z[config_id]
                    ),
                )
                actor_row = LANE_ROW_NAMES[nearest_config]
                if actor_row == row_name:
                    result.append(actor)
            return result

        rows = {}
        for row_name in ("upper", "lower"):
            row_friend = actors_in_row(friend, row_name)
            row_enemy = actors_in_row(enemy, row_name)
            row_friend_front = front_of(row_friend, True)
            row_enemy_front = front_of(row_enemy, False)
            row_clash = None
            if row_friend_front is not None and row_enemy_front is not None:
                row_clash = (
                    (row_friend_front[1][0] + row_enemy_front[1][0]) / 2.0,
                    (row_friend_front[1][1] + row_enemy_front[1][1]) / 2.0,
                )
            rows[row_name] = {
                "friend": row_friend,
                "enemy": row_enemy,
                "friend_front": row_friend_front,
                "enemy_front": row_enemy_front,
                "clash_position": row_clash,
            }

        friend_front = front_of(friend, True)
        enemy_front = front_of(enemy, False)
        enemy_observed = enemy_front is not None
        enemy_inferred = False
        enemy_confidence = 1.0 if enemy_observed else 0.0
        if enemy_front is None:
            for historical in reversed(self.lane_history):
                age = context["frame_no"] - historical["frame"]
                if age > LANE_TREND_FRAMES:
                    break
                if historical["enemy_front"] is not None:
                    enemy_front = (None, historical["enemy_front"])
                    enemy_inferred = True
                    enemy_confidence = u.clip(1.0 - age / LANE_TREND_FRAMES)
                    break

        own_active = next((organ for organ in context["organs"][:3] if organ["alive"]), context["organs"][2])
        enemy_active = next((organ for organ in context["organs"][3:] if organ["alive"]), context["organs"][5])
        own_x = own_active["side_position"][0]
        enemy_x = enemy_active["side_position"][0]

        def progress(position):
            if position is None or enemy_x <= own_x:
                return 0.0
            return u.clip((position[0] - own_x) / (enemy_x - own_x))

        friend_position = friend_front[1] if friend_front else None
        enemy_position = enemy_front[1] if enemy_front else None
        clash_position = None
        row_clashes = [
            row["clash_position"]
            for row in rows.values()
            if row["clash_position"] is not None
        ]
        if row_clashes:
            # The collaboration bank needs a stable lane centre.  Averaging
            # matched upper/lower rows avoids pairing one row's friendly front
            # with the other row's enemy front and producing a z-jittering
            # command target.
            clash_position = (
                sum(position[0] for position in row_clashes) / len(row_clashes),
                sum(position[1] for position in row_clashes) / len(row_clashes),
            )
        elif friend_position is not None and enemy_position is not None:
            clash_position = (
                (friend_position[0] + enemy_position[0]) / 2.0,
                0.0,
            )

        def soldier_type(actor):
            config_id = u.actor_config_id(actor)
            if config_id in (16, 19):
                return "melee"
            if config_id in (17, 20):
                return "ranged"
            if config_id in (18, 21):
                return "cannon"
            return "super"

        summary = {
            "friend": friend,
            "enemy": enemy,
            "friend_front": friend_front,
            "enemy_front": enemy_front,
            "enemy_front_observed": enemy_observed,
            "enemy_front_inferred": enemy_inferred,
            "enemy_front_confidence": enemy_confidence,
            "own_active": own_active,
            "enemy_active": enemy_active,
            "own_boundary_x": own_x,
            "enemy_boundary_x": enemy_x,
            "friend_progress": progress(friend_position),
            "enemy_progress": progress(enemy_position),
            "clash_position": clash_position,
            "rows": rows,
            "friend_type_counts": {name: sum(soldier_type(actor) == name for actor in friend) for name in ("melee", "ranged", "cannon", "super")},
            "enemy_type_counts": {name: sum(soldier_type(actor) == name for actor in enemy) for name in ("melee", "ranged", "cannon", "super")},
        }

        historical = {
            "frame": context["frame_no"],
            "friend_front": friend_position,
            "enemy_front": enemy_position if enemy_observed else None,
            "friend_hp": sum(_value(actor, "hp") for actor in friend),
            "enemy_hp": sum(_value(actor, "hp") for actor in enemy),
            "friend_count": len(friend),
            "enemy_count": len(enemy),
        }
        self.lane_history.append(historical)
        while self.lane_history and context["frame_no"] - self.lane_history[0]["frame"] > LANE_TREND_FRAMES:
            self.lane_history.popleft()

        for organ in context["organs"]:
            history = self.organ_hp_history[organ["config_id"]]
            hp = _value(organ["actor"], "hp") if organ["actor"] else 0.0
            max_hp = _value(organ["actor"], "max_hp", "maxHp") if organ["actor"] else 0.0
            history.append((context["frame_no"], hp, max_hp))
            while history and context["frame_no"] - history[0][0] > LANE_TREND_FRAMES:
                history.popleft()
        return summary

    def _build_lane_macro_context(self, context):
        """Build the four-stage first-wave rule for Marco and Diao Chan.

        Controlled deployment ends at real attackable-wave contact. An
        official lane hero remains command-guarded through its captured first
        wave; enemy combat is transient and wall-clock time never releases it.
        """

        friend_by_config = {
            u.actor_config_id(hero["actor"]): hero
            for hero in context["friend_heroes"]
        }
        plans = {}
        own_outer_x = context["organs"][0]["side_position"][0]
        enemy_outer = context["organs"][3]
        enemy_outer_x = enemy_outer["side_position"][0]
        frame_no = context["frame_no"]

        for config_id in LANE_ROW_SIDE_Z:
            hero = friend_by_config[config_id]
            actor = hero["actor"]
            runtime_id = u.actor_runtime_id(actor)
            position = hero["side_position"]
            alive = hero["alive"]
            row_name = LANE_ROW_NAMES[config_id]
            row_z = LANE_ROW_SIDE_Z[config_id]
            row = context["lane_summary"]["rows"][row_name]
            own_front_pair = row["friend_front"]
            enemy_front_pair = row["enemy_front"]
            own_front = own_front_pair[1] if own_front_pair else None
            enemy_front = enemy_front_pair[1] if enemy_front_pair else None
            clash = row["clash_position"]

            history = self.lane_opening_history.get(config_id)
            if history is None or history.get("runtime_id") != runtime_id:
                history = {
                    "runtime_id": runtime_id,
                    "released": False,
                    "release_reason": "none",
                    "release_frame": None,
                    "first_wave_cleared": False,
                    "first_enemy_wave_ids": set(),
                    "wave_missing_since": None,
                    "contact_reached": False,
                    "locked_target_runtime": 0,
                    "locked_target_last_seen": None,
                    "max_goal_x": None,
                    "max_progress_x": position[0] if position is not None else None,
                    "row_entry_reached": False,
                    "official_route_cursor": 0,
                    "official_entry_failed": False,
                    "official_noncompliant": False,
                    "deer_assist_started_frame": None,
                    "deer_assist_completed": False,
                    "deer_assist_end_reason": "none",
                    "deer_clear_frame": None,
                    "upper_return_cursor": 0,
                }
                self.lane_opening_history[config_id] = history
            deer_anchor = tuple(
                float(value)
                for value in self._opening_camp_anchor("deer")
            )
            deer_state = self.monster_camp_history.get(deer_anchor, {})
            deer_task_ready = bool(
                deer_state.get("alive_runtime_ids")
                or (
                    deer_state.get("spawn_timer_ms") is not None
                    and float(deer_state["spawn_timer_ms"]) <= 0.0
                )
            )
            if hero["official"] and not history["deer_assist_completed"]:
                if deer_state.get("cleared_once"):
                    history["deer_assist_completed"] = True
                    history["deer_assist_end_reason"] = "authoritative_clear"
                    history["deer_clear_frame"] = frame_no
                elif (
                    deer_task_ready
                    and history["deer_assist_started_frame"] is None
                    and not history["contact_reached"]
                ):
                    history["deer_assist_started_frame"] = frame_no
            upper_handoff_semantics = bool(
                hero["official"]
                and not history["released"]
                and (
                    history["deer_assist_started_frame"] is not None
                    and not history["deer_assist_completed"]
                    or history["deer_assist_end_reason"]
                    == "authoritative_clear"
                )
            )
            if upper_handoff_semantics:
                row_name = "upper"
                row_z = 3150.0
                row = context["lane_summary"]["rows"][row_name]
                own_front_pair = row["friend_front"]
                enemy_front_pair = row["enemy_front"]
                own_front = (
                    own_front_pair[1] if own_front_pair else None
                )
                enemy_front = (
                    enemy_front_pair[1] if enemy_front_pair else None
                )
                clash = row["clash_position"]
            if position is not None and not history["released"]:
                previous_progress = history.get("max_progress_x")
                history["max_progress_x"] = (
                    position[0]
                    if previous_progress is None
                    else max(previous_progress, position[0])
                )

            visible_enemies = [
                enemy
                for enemy in context["enemy_heroes"]
                if enemy["alive"]
                and enemy["visible"]
                and enemy["current_valid"]
                and enemy["side_position"] is not None
            ]
            nearest_enemy = min(
                visible_enemies,
                key=lambda enemy: _distance_or_inf(position, enemy["side_position"]),
                default=None,
            )
            nearest_enemy_distance = (
                _distance_or_inf(position, nearest_enemy["side_position"])
                if nearest_enemy is not None
                else float("inf")
            )
            enemy_hero_near = (
                nearest_enemy_distance <= LANE_ENEMY_HERO_INTERRUPT_DISTANCE
            )
            recent_enemy_hit = "hurt_by_hero" in self._event_summary(
                hero, frame_no
            )["short_types"]
            structure_defense = any(
                enemy["side_position"][0]
                <= context["organs"][0]["side_position"][0] + 9000.0
                for enemy in visible_enemies
            )

            interrupt = "none"
            if not alive:
                interrupt = "dead"
            elif _hp_ratio(actor) <= V1TrainingConfig.NO_ATTACKABLE_TARGET_RECOVERY_HP_RATIO:
                interrupt = "low_hp"
            elif recent_enemy_hit:
                interrupt = "recent_hit"
            elif enemy_hero_near:
                interrupt = "enemy_hero"
            elif structure_defense:
                interrupt = "structure_defense"

            if (
                hero["official"]
                and not history["deer_assist_completed"]
                and interrupt in (
                    "dead",
                    "low_hp",
                    "structure_defense",
                )
            ):
                history["deer_assist_completed"] = True
                history["deer_assist_end_reason"] = interrupt

            current_row_ids = {
                u.actor_runtime_id(soldier)
                for soldier in row["enemy"]
                if u.is_alive(soldier)
                and _position(soldier, self.camp) is not None
                and own_outer_x
                <= _position(soldier, self.camp)[0]
                <= enemy_outer_x
            }
            if not history["contact_reached"] and current_row_ids:
                history["first_enemy_wave_ids"].update(current_row_ids)

            first_wave_cleared = False
            if history["first_enemy_wave_ids"]:
                remaining_ids = history["first_enemy_wave_ids"] & current_row_ids
                if remaining_ids:
                    history["wave_missing_since"] = None
                elif history["wave_missing_since"] is None:
                    history["wave_missing_since"] = frame_no
                elif (
                    frame_no - history["wave_missing_since"]
                    >= LANE_FIRST_WAVE_MISSING_GRACE_FRAMES
                ):
                    first_wave_cleared = True
                    history["first_wave_cleared"] = True

            target_candidates = []
            for soldier in row["enemy"]:
                runtime = u.actor_runtime_id(soldier)
                if (
                    hero["official"]
                    and self._official_goal_failed(
                        self._lane_target_goal_signature(
                            runtime_id, runtime
                        )
                    )
                ):
                    continue
                if (
                    history["first_enemy_wave_ids"]
                    and runtime not in history["first_enemy_wave_ids"]
                ):
                    continue
                if not u.visible_to_camp(soldier, self.camp, is_friend=False):
                    continue
                soldier_position = _position(soldier, self.camp)
                if (
                    soldier_position is None
                    or not own_outer_x <= soldier_position[0] <= enemy_outer_x
                ):
                    continue
                under_enemy_outer = bool(
                    enemy_outer["alive"]
                    and _distance_or_inf(
                        soldier_position, enemy_outer["side_position"]
                    )
                    <= u.TOWER_DISTANCE_SCALE
                )
                own_wave_covers_tower = bool(
                    own_front is not None
                    and _distance_or_inf(own_front, enemy_outer["side_position"])
                    <= u.TOWER_DISTANCE_SCALE
                )
                if under_enemy_outer and not own_wave_covers_tower:
                    continue
                slot = next(
                    (
                        28 + index
                        for index, fixed in enumerate(context["soldiers"][10:])
                        if fixed and u.actor_runtime_id(fixed) == runtime
                    ),
                    -1,
                )
                if slot >= 28:
                    target_candidates.append(
                        (
                            _hp_ratio(soldier),
                            _distance_or_inf(position, soldier_position),
                            -soldier_position[0],
                            runtime,
                            slot,
                            soldier,
                        )
                    )
            target_candidates.sort(key=lambda item: item[:3])
            candidates_by_runtime = {
                item[3]: item for item in target_candidates
            }
            locked_runtime = history["locked_target_runtime"]
            selected = candidates_by_runtime.get(locked_runtime)
            if selected is not None:
                history["locked_target_last_seen"] = frame_no
            elif (
                locked_runtime
                and history["locked_target_last_seen"] is not None
                and frame_no - history["locked_target_last_seen"]
                <= LANE_TARGET_LOCK_GRACE_FRAMES
            ):
                selected = None
            else:
                selected = target_candidates[0] if target_candidates else None
                history["locked_target_runtime"] = selected[3] if selected else 0
                history["locked_target_last_seen"] = frame_no if selected else None

            target_soldier = selected[5] if selected else None
            target_slot = selected[4] if selected else -1
            target_position = (
                _position(target_soldier, self.camp) if target_soldier else None
            )
            target_distance = _distance_or_inf(position, target_position)
            official_row_reacquire = bool(
                hero["official"]
                and history["row_entry_reached"]
                and not history["contact_reached"]
                and not history["first_wave_cleared"]
                and position is not None
                and abs(position[1] - row_z)
                > LANE_OFFICIAL_ROW_REACQUIRE_Z_DISTANCE
                and target_distance > LANE_OFFICIAL_TARGET_COMMAND_DISTANCE
            )
            contact_now = bool(
                target_soldier is not None
                and target_distance <= LANE_CONTACT_DISTANCE
                and position is not None
                and abs(position[1] - row_z) <= LANE_CONTACT_ROW_TOLERANCE
            )
            history["contact_reached"] = bool(
                history["contact_reached"] or contact_now
            )

            release_reason = "none"
            transient_official_combat = bool(
                hero["official"]
                and interrupt in ("enemy_hero", "recent_hit")
            )
            transient_official_route_interrupt = bool(
                hero["official"]
                and interrupt
                in ("enemy_hero", "recent_hit", "structure_defense")
                and not history["contact_reached"]
            )
            if contact_now and not hero["official"]:
                # Controlled lane heroes have completed their deterministic
                # deployment once they actually reach an attackable first-wave
                # soldier. PPO owns combat/last-hit decisions from this frame.
                release_reason = "wave_contact"
            elif interrupt != "none" and not transient_official_route_interrupt:
                release_reason = interrupt
            elif first_wave_cleared and history["contact_reached"]:
                release_reason = "first_wave_cleared"
            if not history["released"] and release_reason != "none":
                history.update(
                    released=True,
                    release_reason=release_reason,
                    release_frame=frame_no,
                )

            waypoint = None
            route_node = 2
            waiting_for_wave = False
            phase = "inactive"
            route_waypoints = LANE_ROUTE_WAYPOINTS[
                132 if upper_handoff_semantics else config_id
            ]
            row_entry_waypoint = route_waypoints[2]
            if position is None or not alive:
                phase = "inactive"
            elif history["released"]:
                phase = "released"
            else:
                if hero["official"]:
                    # The official model receives soft collaboration commands,
                    # so preserve an explicit monotonic physical route cursor.
                    # A command echo never advances this cursor.
                    official_route = route_waypoints
                    fallback_entry = (
                        LANE_OFFICIAL_FALLBACK_ENTRY_SIDE_X,
                        row_z,
                    )
                    cursor = min(
                        int(history.get("official_route_cursor", 0)),
                        len(official_route),
                    )

                    def route_reached(index):
                        if index >= len(official_route):
                            return True
                        goal = official_route[index]
                        if (
                            _distance_or_inf(position, goal)
                            <= LANE_ROUTE_WAYPOINT_REACHED_DISTANCE
                        ):
                            return True
                        if index == 0:
                            return bool(
                                position[0] >= goal[0]
                                and position[1] >= 0.0
                            )
                        tolerance = (
                            LANE_OFFICIAL_ROW_REACQUIRE_Z_DISTANCE
                            if index == 1
                            else LANE_ROUTE_WAYPOINT_REACHED_DISTANCE
                        )
                        return bool(
                            position[0] >= goal[0]
                            and abs(position[1] - row_z) <= tolerance
                        )

                    while cursor < len(official_route) and route_reached(
                        cursor
                    ):
                        cursor += 1

                    if cursor < len(official_route):
                        candidate = official_route[cursor]
                        candidate_signature = (
                            self._lane_position_goal_signature(
                                runtime_id,
                                min(cursor, 2),
                                candidate,
                            )
                        )
                        if self._official_goal_failed(candidate_signature):
                            if cursor == 2:
                                history["official_entry_failed"] = True
                            cursor += 1

                    history["official_route_cursor"] = cursor
                    if cursor < len(official_route):
                        waypoint = official_route[cursor]
                        route_node = min(cursor, 2)
                        phase = (
                            "base_exit"
                            if cursor == 0
                            else "corridor"
                            if cursor == 1
                            else "stage_outer"
                        )
                    elif history["official_entry_failed"]:
                        fallback_signature = (
                            self._lane_position_goal_signature(
                                runtime_id, 2, fallback_entry
                            )
                        )
                        fallback_reached = bool(
                            _distance_or_inf(position, fallback_entry)
                            <= LANE_ROUTE_WAYPOINT_REACHED_DISTANCE
                            or (
                                position[0] >= fallback_entry[0]
                                and abs(position[1] - row_z)
                                <= LANE_ROUTE_WAYPOINT_REACHED_DISTANCE
                            )
                        )
                        if fallback_reached:
                            history["row_entry_reached"] = True
                        elif not self._official_goal_failed(
                            fallback_signature
                        ):
                            waypoint = fallback_entry
                            route_node = 2
                            phase = "stage_outer"
                        else:
                            history["official_noncompliant"] = True
                    else:
                        history["row_entry_reached"] = True

                    if (
                        history["official_noncompliant"]
                        and waypoint is None
                        and not history["row_entry_reached"]
                    ):
                        safe_candidates = [
                            organ["side_position"]
                            for organ in context["organs"][:3]
                            if organ["alive"]
                            and organ["side_position"] is not None
                        ]
                        safe_candidates.extend(
                            friend["side_position"]
                            for friend in context["friend_heroes"]
                            if not friend["official"]
                            and friend["alive"]
                            and friend["side_position"] is not None
                        )
                        waypoint = next(
                            (
                                candidate
                                for candidate in safe_candidates
                                if not self._official_goal_failed(
                                    self._lane_position_goal_signature(
                                        runtime_id, 2, candidate
                                    )
                                )
                            ),
                            safe_candidates[0]
                            if safe_candidates
                            else row_entry_waypoint,
                        )
                        route_node = 2
                        phase = "stage_outer"
                    elif (
                        waypoint is None
                        and history["row_entry_reached"]
                        and official_row_reacquire
                    ):
                        waypoint = row_entry_waypoint
                        route_node = 2
                        phase = "stage_outer"
                else:
                    if (
                        _distance_or_inf(position, row_entry_waypoint)
                        <= LANE_ROUTE_WAYPOINT_REACHED_DISTANCE
                        or (
                            position[0] >= row_entry_waypoint[0]
                            and abs(position[1] - row_z)
                            <= LANE_ROUTE_WAYPOINT_REACHED_DISTANCE
                        )
                    ):
                        history["row_entry_reached"] = True
                    if (
                        position[0] < -50000.0
                        and _distance_or_inf(position, route_waypoints[0])
                        > LANE_ROUTE_WAYPOINT_REACHED_DISTANCE
                    ):
                        phase = "base_exit"
                        route_node = 0
                        waypoint = route_waypoints[0]
                    elif (
                        position[0] < -30000.0
                        and _distance_or_inf(position, route_waypoints[1])
                        > LANE_ROUTE_WAYPOINT_REACHED_DISTANCE
                    ):
                        phase = "corridor"
                        route_node = 1
                        waypoint = route_waypoints[1]
                    elif not history["row_entry_reached"]:
                        phase = "stage_outer"
                        route_node = 2
                        waypoint = row_entry_waypoint

            if (
                position is not None
                and alive
                and not history["released"]
                and waypoint is None
            ):
                attack_range = max(
                    3500.0, _value(actor, "attack_range", "attackRange")
                )
                if target_position is not None:
                    phase = "engage_wave"
                    standoff = min(5000.0, max(2500.0, attack_range * 0.65))
                    raw_waypoint = (target_position[0] - standoff, row_z)
                elif own_front is not None:
                    phase = "follow_wave"
                    raw_waypoint = (
                        min(own_front[0] - 2500.0, enemy_outer_x - 2500.0),
                        own_front[1],
                    )
                else:
                    phase = "stage_outer"
                    raw_waypoint = (-12000.0, row_z)

                previous_goal = history.get("max_goal_x")
                history["max_goal_x"] = (
                    raw_waypoint[0]
                    if previous_goal is None
                    else max(previous_goal, raw_waypoint[0])
                )
                frontier_floor = max(
                    history["max_goal_x"] - LANE_MAX_BACKTRACK_DISTANCE,
                    history["max_progress_x"] - LANE_MAX_BACKTRACK_DISTANCE,
                )
                waypoint = (
                    max(raw_waypoint[0], frontier_floor),
                    raw_waypoint[1],
                )
                waiting_for_wave = bool(
                    target_position is None
                    and _distance_or_inf(position, waypoint) <= 2200.0
                )

            deer_assist_active = bool(
                hero["official"]
                and alive
                and position is not None
                and not history["deer_assist_completed"]
                and history["deer_assist_started_frame"] is not None
            )
            deer_return_active = False
            if deer_assist_active and interrupt == "none":
                waypoint = u.side_position(deer_anchor, self.camp)
                phase = "stage_outer"
                route_node = 2
                waiting_for_wave = False
            elif (
                hero["official"]
                and history["deer_assist_completed"]
                and history["deer_assist_end_reason"]
                == "authoritative_clear"
                and not history["first_wave_cleared"]
                and not history["released"]
                and interrupt == "none"
            ):
                return_waypoints = (
                    (-30000.0, 3150.0),
                    (-18000.0, 3150.0),
                )
                cursor = min(
                    int(history.get("upper_return_cursor", 0)),
                    len(return_waypoints),
                )
                while (
                    cursor < len(return_waypoints)
                    and (
                        _distance_or_inf(
                            position, return_waypoints[cursor]
                        )
                        <= LANE_ROUTE_WAYPOINT_REACHED_DISTANCE
                        or (
                            position[0] >= return_waypoints[cursor][0]
                            and abs(position[1] - 3150.0)
                            <= LANE_OFFICIAL_ROW_REACQUIRE_Z_DISTANCE
                        )
                    )
                ):
                    cursor += 1
                history["upper_return_cursor"] = cursor
                if cursor < len(return_waypoints):
                    waypoint = return_waypoints[cursor]
                    phase = "corridor" if cursor == 0 else "stage_outer"
                    route_node = 1 if cursor == 0 else 2
                    waiting_for_wave = False
                    deer_return_active = True
                else:
                    history["row_entry_reached"] = True

            command_waypoint = waypoint
            if (
                hero["official"]
                and position is not None
                and interrupt == "none"
                and history["row_entry_reached"]
                and phase in ("engage_wave", "follow_wave")
                and own_front is not None
            ):
                # Keep base-exit/corridor/row-entry commands on the measured
                # fixed route. Only after physical row entry may the assigned
                # minion front replace the static waypoint.
                escort_x = min(
                    own_front[0] - 1500.0,
                    position[0] + 18000.0,
                    enemy_outer_x - 5000.0,
                )
                escort_x = max(escort_x, position[0] - 2500.0)
                command_waypoint = (escort_x, own_front[1])

            lane_position_signature = (
                self._lane_position_goal_signature(
                    runtime_id, route_node, command_waypoint
                )
                if hero["official"] and command_waypoint is not None
                else None
            )
            lane_target_signature = (
                self._lane_target_goal_signature(
                    runtime_id,
                    u.actor_runtime_id(target_soldier),
                )
                if hero["official"] and target_soldier is not None
                else None
            )
            opening_active = bool(
                alive and position is not None and not history["released"]
            )
            official_wave_guard_active = bool(
                hero["official"]
                and alive
                and position is not None
                and history["first_enemy_wave_ids"]
                and not history["first_wave_cleared"]
            )
            hard_mask_active = bool(opening_active and interrupt == "none")
            route_progress = (
                u.clip((position[0] + 66000.0) / 66000.0)
                if position is not None
                else 0.0
            )
            plans[config_id] = {
                "valid": bool(alive and position is not None),
                "runtime_id": runtime_id,
                "official": hero["official"],
                "row": row_name,
                "row_z": row_z,
                "phase": phase,
                "interrupt": interrupt,
                "opening_active": opening_active,
                # An official lane hero is not trained by this policy. Keep
                # SetPosition/SetTarget constrained to its captured first wave
                # after controlled micro would already have been released;
                # otherwise an untrained command head can legally send it to a
                # jungle slot immediately after lane contact.
                "command_active": bool(opening_active or official_wave_guard_active),
                "hard_mask_active": hard_mask_active,
                "contact_reached": history["contact_reached"],
                "transient_official_combat": transient_official_combat,
                "first_wave_captured": bool(history["first_enemy_wave_ids"]),
                "first_wave_cleared": bool(history["first_wave_cleared"]),
                "target_lock_active": bool(history["locked_target_runtime"]),
                "row_entry_reached": bool(history["row_entry_reached"]),
                "official_route_cursor": int(
                    history.get("official_route_cursor", 0)
                ),
                "official_noncompliant": bool(
                    history.get("official_noncompliant", False)
                ),
                "deer_assist_active": deer_assist_active,
                "deer_return_active": deer_return_active,
                "deer_assist_end_reason": history[
                    "deer_assist_end_reason"
                ],
                "deer_clear_frame": history["deer_clear_frame"],
                "command_retry_requested": bool(
                    self._official_goal_retry_requested(
                        lane_target_signature
                    )
                    or self._official_goal_retry_requested(
                        lane_position_signature
                    )
                ),
                "release_reason": history["release_reason"],
                "next_waypoint": waypoint,
                "command_waypoint": command_waypoint,
                "waypoint_dx": waypoint[0] - position[0] if waypoint and position else 0.0,
                "waypoint_dz": waypoint[1] - position[1] if waypoint and position else 0.0,
                "route_node": route_node,
                "route_progress": route_progress,
                "own_front": own_front,
                "enemy_front": enemy_front,
                "clash_position": clash,
                "target_soldier": target_soldier,
                "target_slot": target_slot,
                "target_distance": target_distance,
                "nearest_enemy_hero_distance": nearest_enemy_distance,
                "intruder_runtime": u.actor_runtime_id(nearest_enemy["actor"])
                if nearest_enemy else 0,
                "enemy_hero_near": enemy_hero_near,
                "recent_enemy_hero_hit": recent_enemy_hit,
                "structure_defense": structure_defense,
                "waiting_for_wave": waiting_for_wave,
                "first_wave_contact_soon": bool(
                    own_front is not None
                    and enemy_front is not None
                    and abs(enemy_front[0] - own_front[0]) <= 14000.0
                ),
            }
        return plans

    def _mark_official_goal_failure(
        self, signature, attempt, frame_no, reason
    ):
        if signature is None:
            return
        guard = self.official_command_guard
        guard["last_failure_frame"] = int(frame_no)
        guard["last_failure_reason"] = str(reason)
        if int(attempt) < 2:
            guard["retry_requested"].add(signature)
            return
        guard["retry_requested"].discard(signature)
        guard["failed_signatures"].add(signature)
        guard["fallback_pending"] = True
        guard["fallback_source_signature"] = signature
        guard["fallback_confirmed"] = False

    def _clear_official_goal_attempt(self, signature):
        if signature is None:
            return
        guard = self.official_command_guard
        if signature not in guard["failed_signatures"]:
            guard["attempts"].pop(signature, None)
            guard["retry_requested"].discard(signature)

    def _update_official_command_guard_response(
        self,
        context,
        active,
        command_type,
        target_runtime,
        raw_position,
        target_actor,
        target_position,
        command_position,
    ):
        guard = self.official_command_guard
        window = guard.get("response_window")
        if window is None:
            return

        action_type = int(window["action_type"])
        same_goal = bool(
            active
            and (
                (action_type == 1 and command_type == "set_target")
                or (action_type == 2 and command_type == "set_position")
            )
        )
        if action_type == 1:
            same_goal = bool(
                same_goal
                and int(target_runtime)
                == int(window.get("target_runtime", 0))
            )
            goal_position = target_position
        else:
            expected_raw = window.get("raw_position")
            same_goal = bool(
                same_goal
                and raw_position is not None
                and expected_raw is not None
                and math.hypot(
                    raw_position[0] - float(expected_raw[0]),
                    raw_position[1] - float(expected_raw[1]),
                )
                <= 100.0
            )
            goal_position = command_position

        official = context["official_hero"]
        official_position = official.get("side_position")
        response_observed = False
        goal_distance = float("inf")
        if official_position is not None and goal_position is not None:
            distance = _distance_or_inf(official_position, goal_position)
            goal_distance = distance
            if window.get("initial_distance") is None:
                window["initial_distance"] = distance
                window["best_distance"] = distance
            else:
                window["best_distance"] = min(
                    float(window["best_distance"]), distance
                )
                response_observed = bool(
                    float(window["initial_distance"])
                    - float(window["best_distance"])
                    >= V1TrainingConfig.COMMAND_RESPONSE_MIN_PROGRESS_DISTANCE
                )
            if action_type == 2 and distance <= 1200.0:
                response_observed = True
        if (
            action_type == 1
            and int(window.get("target_runtime", 0)) > 0
        ):
            target_runtime_id = int(window["target_runtime"])
            attack_range = max(
                3500.0,
                _value(
                    official["actor"], "attack_range", "attackRange"
                ),
            )
            dealt_damage = any(
                u.as_int(
                    u.safe_get(
                        hurt,
                        "hurt_target",
                        "hurtTarget",
                        default=0,
                    ),
                    0,
                )
                == target_runtime_id
                and u.as_float(
                    u.safe_get(hurt, "hurt", default=0.0), 0.0
                )
                > 0.0
                for hurt in (
                    u.actor_get(
                        official["actor"],
                        "hurt_hero_info",
                        "hurtHeroInfo",
                        default=[],
                    )
                    or []
                )
            )
            hit_target = any(
                u.as_int(
                    u.safe_get(
                        hit,
                        "hit_target",
                        "hitTarget",
                        default=0,
                    ),
                    0,
                )
                == target_runtime_id
                for hit in (
                    u.actor_get(
                        official["actor"],
                        "hit_target_info",
                        "hitTargetInfo",
                        default=[],
                    )
                    or []
                )
            )
            response_observed = bool(
                goal_distance <= attack_range + 1000.0
                or dealt_damage
                or hit_target
            )

        if response_observed:
            guard["last_success_frame"] = int(context["frame_no"])
            self._clear_official_goal_attempt(
                window.get("goal_signature")
            )
            guard["response_window"] = None
            return

        age = int(context["frame_no"]) - int(window["accepted_frame"])
        if age < V1TrainingConfig.COMMAND_RESPONSE_WINDOW_FRAMES:
            return
        if not same_goal:
            # A different accepted command supersedes this response window.
            # It is successor evidence, not nonresponse evidence for the old
            # goal.
            self._clear_official_goal_attempt(
                window.get("goal_signature")
            )
            guard["response_window"] = None
            return

        # A target disappearing before the response window ends is not a
        # reusable failed target.  The lane/camp lifecycle will select a new
        # runtime or advance from authoritative clear evidence.
        if action_type == 1 and (
            target_actor is None
            or not u.is_alive(target_actor)
        ):
            self._clear_official_goal_attempt(
                window.get("goal_signature")
            )
            guard["response_window"] = None
            return
        self._mark_official_goal_failure(
            window.get("goal_signature"),
            int(window.get("attempt", 1)),
            context["frame_no"],
            "behavior_nonresponse",
        )
        guard["response_window"] = None

    def _collaboration_summary(self, context):
        frame_no = context["frame_no"]
        raw_value = u.safe_get(
            context["frame_state"], "collaboration_state", "collaborationState", default=None
        )
        raw = raw_value or {}
        micro_states = list(u.safe_get(raw, "micro_state", "microState", default=[]) or [])
        official_runtime = u.actor_runtime_id(context["official_hero"]["actor"])
        candidates = []
        for state in micro_states:
            receiver = u.as_int(u.safe_get(state, "receiver", default=0), 0)
            sender = u.as_int(u.safe_get(state, "sender", default=0), 0)
            if receiver == official_runtime and sender in context["controlled_runtime_ids"]:
                candidates.append(state)
        current = max(
            candidates,
            key=lambda item: u.as_int(u.safe_get(item, "end_frame", "endFrame", default=0), 0),
            default=None,
        )

        micro_cd = max(0, u.as_int(u.safe_get(raw, "micro_cd", "microCd", default=0), 0))
        max_duration = max(0, u.as_int(u.safe_get(raw, "max_duration", "maxDuration", default=0), 0))
        last_by_sender = {}
        for item in u.safe_get(raw, "micro_last", "microLast", default=[]) or []:
            sender = u.as_int(u.safe_get(item, "sender", default=0), 0)
            sent_frame = u.as_int(u.safe_get(item, "frame_no", "frameNo", default=0), 0)
            if sender in context["controlled_runtime_ids"]:
                last_by_sender[sender] = max(last_by_sender.get(sender, 0), sent_frame)
        platform_last_send_frame = max(last_by_sender.values(), default=0)

        command_type = "none"
        target_runtime = 0
        raw_position = None
        end_frame = 0
        sender = 0
        receiver = 0
        if current is not None:
            target_runtime = u.as_int(
                u.safe_get(current, "target_runtime_id", "targetRuntimeId", default=0), 0
            )
            position_obj = u.safe_get(current, "target_position", "targetPosition", default={}) or {}
            x = u.safe_get(position_obj, "x", default=None)
            z = u.safe_get(position_obj, "z", default=None)
            if x is not None and z is not None:
                candidate = (u.as_float(x, float("nan")), u.as_float(z, float("nan")))
                if all(math.isfinite(value) and abs(value) < 90000 for value in candidate):
                    raw_position = candidate
            command_type = "set_target" if target_runtime else "set_position" if raw_position is not None else "none"
            end_frame = u.as_int(u.safe_get(current, "end_frame", "endFrame", default=0), 0)
            sender = u.as_int(u.safe_get(current, "sender", default=0), 0)
            receiver = u.as_int(u.safe_get(current, "receiver", default=0), 0)

        active = current is not None and end_frame >= frame_no and command_type != "none"
        pending = self.pending_collaboration_command
        if pending is not None:
            pending_type = int(pending["action_type"])
            state_matches = False
            if pending_type == 1:
                state_matches = bool(
                    active
                    and command_type == "set_target"
                    and target_runtime == int(pending.get("target_runtime", 0))
                )
            elif pending_type == 2 and active and command_type == "set_position":
                pending_position = pending.get("raw_position")
                state_matches = bool(
                    raw_position is not None
                    and pending_position is not None
                    and math.hypot(
                        raw_position[0] - float(pending_position[0]),
                        raw_position[1] - float(pending_position[1]),
                    )
                    <= 100.0
                )
            accepted = bool(
                platform_last_send_frame >= int(pending["frame"])
                and state_matches
            )
            expired = (
                frame_no - int(pending["frame"])
                >= COMMAND_ACCEPTANCE_GRACE_FRAMES
            )
            if accepted:
                goal_signature = pending.get("goal_signature")
                if goal_signature is not None:
                    self.official_command_guard["response_window"] = {
                        "goal_signature": goal_signature,
                        "attempt": int(pending.get("attempt", 1)),
                        "action_type": pending_type,
                        "target_runtime": int(
                            pending.get("target_runtime", 0)
                        ),
                        "raw_position": pending.get("raw_position"),
                        "accepted_frame": int(frame_no),
                        "initial_distance": None,
                        "best_distance": None,
                    }
                    if (
                        self.official_command_guard["fallback_pending"]
                        and goal_signature
                        != self.official_command_guard[
                            "fallback_source_signature"
                        ]
                    ):
                        self.official_command_guard[
                            "fallback_confirmed"
                        ] = True
                        if (
                            u.actor_config_id(
                                context["official_hero"]["actor"]
                            )
                            in LANE_ROW_SIDE_Z
                            or (
                                isinstance(
                                    self.official_command_guard[
                                        "fallback_source_signature"
                                    ],
                                    tuple,
                                )
                                and self.official_command_guard[
                                    "fallback_source_signature"
                                ][:1]
                                == ("generic",)
                            )
                        ):
                            self.official_command_guard[
                                "fallback_pending"
                            ] = False
                            self.official_command_guard[
                                "fallback_source_signature"
                            ] = None
                if bool(pending.get("is_train", False)):
                    self.active_command_attribution = {
                        "intent": int(pending.get("intent", 0)),
                        "action_type": pending_type,
                        "target_runtime": int(
                            pending.get("target_runtime", 0)
                        ),
                        "raw_position": pending.get("raw_position"),
                        "accepted_frame": frame_no,
                        "last_potential": None,
                        "progress_total": 0.0,
                        "initial_distance": None,
                        "best_distance": None,
                        "response_observed": False,
                        "target_lock_rewarded": False,
                        "target_range_rewarded": False,
                        "target_hit_rewarded": False,
                        "nonresponse_penalized": False,
                    }
                else:
                    # Forced opening actions are executable control, not a
                    # sampled command-policy decision.
                    self.active_command_attribution = None
                self.pending_collaboration_command = None
            elif expired:
                if (
                    bool(pending.get("is_train", False))
                    and int(pending.get("intent", 0)) == 10
                    and self.official_task_state is not None
                ):
                    self._finish_official_task(
                        frame_no, "protocol_rejected"
                    )
                pending_target = int(pending.get("target_runtime", 0))
                if (
                    pending_type == 1
                    and pending_target > 0
                    and pending_target in context["actor_by_runtime"]
                    and (
                        pending.get("goal_signature") is None
                        or int(pending.get("attempt", 1)) >= 2
                    )
                ):
                    self.rejected_command_target_runtimes.add(pending_target)
                if pending.get("goal_signature") is not None:
                    self._mark_official_goal_failure(
                        pending.get("goal_signature"),
                        int(pending.get("attempt", 1)),
                        frame_no,
                        "platform_rejected",
                    )
                # Rejected commands do not update micro_last and therefore do
                # not consume platform cooldown. Release the provisional local
                # hold so the next legal command can fall back to SetPosition.
                self.local_last_command_frame = platform_last_send_frame
                self.pending_collaboration_command = None
        self.rejected_command_target_runtimes.intersection_update(
            int(runtime_id) for runtime_id in context["actor_by_runtime"]
        )
        local_last_send_frame = int(self.local_last_command_frame)
        last_send_frame = max(platform_last_send_frame, local_last_send_frame)

        target_actor = context["actor_by_runtime"].get(target_runtime)
        target_snapshot = context["hero_by_runtime"].get(target_runtime)
        if target_snapshot is not None:
            target_actor = target_snapshot["actor"]
        target_position = (
            target_snapshot["side_position"]
            if target_snapshot is not None
            else _position(target_actor, self.camp)
            if target_actor is not None
            else None
        )
        command_position = u.side_position(raw_position, self.camp)
        self._update_official_command_guard_response(
            context=context,
            active=active,
            command_type=command_type,
            target_runtime=target_runtime,
            raw_position=raw_position,
            target_actor=target_actor,
            target_position=target_position,
            command_position=command_position,
        )

        attribution = self.active_command_attribution
        attribution_for_boundary = attribution
        if attribution is not None:
            expected_type = (
                "set_target"
                if int(attribution["action_type"]) == 1
                else "set_position"
            )
            same_goal = bool(active and command_type == expected_type)
            if expected_type == "set_target":
                same_goal = same_goal and target_runtime == int(
                    attribution["target_runtime"]
                )
            else:
                expected_position = attribution.get("raw_position")
                same_goal = bool(
                    same_goal
                    and raw_position is not None
                    and expected_position is not None
                    and math.hypot(
                        raw_position[0] - float(expected_position[0]),
                        raw_position[1] - float(expected_position[1]),
                    )
                    <= 100.0
                )
            official_alive = bool(context["official_hero"]["alive"])
            target_still_valid = True
            if expected_type == "set_target":
                target_still_valid = bool(
                    target_actor is not None
                    and u.is_alive(target_actor)
                    and u.visible_to_camp(
                        target_actor,
                        self.camp,
                        is_friend=u.actor_camp(target_actor) == self.camp,
                    )
                )
            if not same_goal or not official_alive or not target_still_valid:
                # Keep this boundary's snapshot so a terminal target/death
                # event can still be credited once, then close the window.
                self.active_command_attribution = None

        signature = (command_type, sender, receiver, target_runtime, raw_position, end_frame)
        previous = self.command_history[-1] if self.command_history else None
        if previous is None or previous["signature"] != signature:
            self.command_history.append(
                {
                    "frame": frame_no,
                    "signature": signature,
                    "type": command_type,
                    "target_runtime": target_runtime,
                    "valid": active or command_type == "none",
                    "overwritten": bool(previous and previous["type"] != "none" and command_type != "none"),
                    "unset": bool(previous and previous["type"] != "none" and command_type == "none"),
                }
            )
        while self.command_history and frame_no - self.command_history[0]["frame"] > COMMAND_HISTORY_FRAMES:
            self.command_history.popleft()
        last_history = self.command_history[-1] if self.command_history else None

        return {
            "raw": raw,
            "state_valid": isinstance(raw_value, dict),
            "type": command_type if active else "none",
            "active": active,
            "sender": sender,
            "receiver": receiver,
            "target_runtime": target_runtime,
            "target_actor": target_actor,
            "target_snapshot": target_snapshot,
            "target_position": target_position,
            "command_position": command_position,
            "end_frame": end_frame,
            "end_frame_valid": end_frame > 0,
            "remaining_frames": max(0, end_frame - frame_no),
            "micro_cd": micro_cd,
            "max_duration": max_duration,
            "last_send_frame": last_send_frame,
            "platform_last_send_frame": platform_last_send_frame,
            "local_last_send_frame": local_last_send_frame,
            "pending_command_frame": int(
                self.pending_collaboration_command["frame"]
            )
            if self.pending_collaboration_command is not None
            else 0,
            "pending_command_type": int(
                self.pending_collaboration_command["action_type"]
            )
            if self.pending_collaboration_command is not None
            else 0,
            "rejected_target_runtime_count": len(
                self.rejected_command_target_runtimes
            ),
            "guard_attempt_count": max(
                self.official_command_guard["attempts"].values(),
                default=0,
            ),
            "guard_retry_pending": bool(
                self.official_command_guard["retry_requested"]
            ),
            "guard_failed_goal_count": len(
                self.official_command_guard["failed_signatures"]
            ),
            "guard_fallback_pending": bool(
                self.official_command_guard["fallback_pending"]
            ),
            "guard_fallback_confirmed": bool(
                self.official_command_guard["fallback_confirmed"]
            ),
            "guard_last_failure_frame": self.official_command_guard[
                "last_failure_frame"
            ],
            "guard_last_failure_reason": self.official_command_guard[
                "last_failure_reason"
            ],
            "cooldown_ready": last_send_frame <= 0 or frame_no - last_send_frame >= micro_cd,
            "age_frames": max(0, frame_no - last_send_frame) if last_send_frame else 0,
            "last_history": last_history,
            "attribution": attribution_for_boundary,
        }

    @staticmethod
    def _reset_official_idle_state(state):
        state["start_frame"] = None
        state["start_position"] = None
        state["last_position"] = None
        state["path_distance"] = 0.0
        state["start_damage"] = None
        state["next_evaluation_frame"] = None

    def _official_command_wait_exempt(self, context):
        """Return whether stationary official behavior is strategically valid."""

        official = context["official_hero"]
        actor = official["actor"]
        position = official["side_position"]
        if not official["alive"] or position is None:
            return True
        if bool(u.actor_get(actor, "is_in_grass", "isInGrass", default=False)):
            return True
        config_id = u.actor_config_id(actor)
        if config_id == 193:
            macro = context.get("jungle_macro") or {}
            if macro.get("opening_wait_for_spawn"):
                return True
        lane_plan = (context.get("lane_macro") or {}).get(config_id)
        if lane_plan and lane_plan.get("waiting_for_wave"):
            return True
        tyrant = self._tyrant_status(context)
        if (
            tyrant["position"] is not None
            and _distance_or_inf(position, tyrant["position"])
            <= V1TrainingConfig.STRATEGIC_WAIT_RADIUS
            and (
                tyrant["spawn_within_30"]
                or tyrant["team_attacking"]
                or tyrant["enemy_attacking"]
            )
        ):
            return True
        if _hp_ratio(actor) <= V1TrainingConfig.NO_ATTACKABLE_TARGET_RECOVERY_HP_RATIO:
            own_outer, own_high, own_crystal = context["organs"][:3]
            safe = (
                own_outer
                if own_outer["alive"]
                else own_high
                if own_high["alive"]
                else own_crystal
            )
            if (
                safe["side_position"] is not None
                and _distance_or_inf(position, safe["side_position"])
                <= 12000.0
            ):
                return True
        return False

    def _official_has_meaningful_command_opportunity(self, context):
        """Conservatively detect whether no-command leaves useful work undone."""

        official = context["official_hero"]
        position = official["side_position"]
        if not official["alive"] or position is None:
            return False
        config_id = u.actor_config_id(official["actor"])
        if config_id == 193:
            macro = context.get("jungle_macro") or {}
            if macro.get("valid") and (
                macro.get("opening_rotation_active")
                or macro.get("mode") in ("push", "group")
                or macro.get("target_slot") is not None
            ):
                return True
        lane_plan = (context.get("lane_macro") or {}).get(config_id)
        if lane_plan and (
            lane_plan.get("command_active")
            or lane_plan.get("target_soldier") is not None
            or lane_plan.get("structure_defense")
        ):
            return True
        if any(
            enemy["alive"]
            and enemy["visible"]
            and enemy["side_position"] is not None
            and _distance_or_inf(position, enemy["side_position"]) <= 20000.0
            for enemy in context["enemy_heroes"]
        ):
            return True
        if any(
            soldier is not None
            and u.is_alive(soldier)
            and _distance_or_inf(
                position, _position(soldier, self.camp)
            )
            <= 26000.0
            for soldier in context["soldiers"][10:]
        ):
            return True
        if any(
            organ["alive"]
            and organ["actor"]
            and _target_runtime(organ["actor"])
            for organ in context["organs"][:3]
        ):
            return True
        tyrant = self._tyrant_status(context)
        if (
            tyrant["position"] is not None
            and (
                tyrant["alive"]
                or tyrant["spawn_within_30"]
                or tyrant["team_attacking"]
                or tyrant["enemy_attacking"]
            )
        ):
            return True
        teammate_distances = [
            _distance_or_inf(position, hero["side_position"])
            for hero in context["friend_heroes"]
            if hero["alive"]
            and u.actor_runtime_id(hero["actor"])
            != u.actor_runtime_id(official["actor"])
            and hero["side_position"] is not None
        ]
        return bool(
            teammate_distances and min(teammate_distances) >= 18000.0
        )

    def _consume_official_no_command_idle_penalty(self, context):
        """Penalize repeated trainable no-command choices during real idleness."""

        state = self.official_no_command_idle_state
        frame_no = context["frame_no"]
        decision = self.last_command_decision
        collaboration = context["collaboration"]
        official = context["official_hero"]
        eligible_decision = bool(
            decision
            and decision.get("is_train")
            and int(decision.get("intent", -1)) == 0
            and int(decision.get("action_type", -1)) == 0
            and 0 < frame_no - int(decision.get("frame", frame_no))
            <= V1TrainingConfig.COMMAND_NO_COMMAND_IDLE_DECISION_MAX_AGE_FRAMES
        )
        eligible = bool(
            eligible_decision
            and official["alive"]
            and official["side_position"] is not None
            and not collaboration["active"]
            and collaboration["pending_command_frame"] == 0
            and collaboration["cooldown_ready"]
            and not self._official_command_wait_exempt(context)
            and self._official_has_meaningful_command_opportunity(context)
        )
        if not eligible:
            self._reset_official_idle_state(state)
            return 0.0, False

        position = official["side_position"]
        damage = u.as_float(
            u.actor_get(
                official["actor"], "total_hurt", "totalHurt", default=0.0
            )
        )
        if state["start_frame"] is None:
            state["start_frame"] = frame_no
            state["start_position"] = position
            state["last_position"] = position
            state["start_damage"] = damage
            state["next_evaluation_frame"] = (
                frame_no
                + V1TrainingConfig.COMMAND_NO_COMMAND_IDLE_WINDOW_FRAMES
            )
            return 0.0, False

        state["path_distance"] += _distance_or_inf(
            state["last_position"], position
        )
        state["last_position"] = position
        net_progress = _distance_or_inf(state["start_position"], position)
        damage_progress = damage - float(state["start_damage"])
        if (
            net_progress
            >= V1TrainingConfig.COMMAND_NO_COMMAND_IDLE_MIN_PROGRESS_DISTANCE
            or damage_progress > 1e-6
        ):
            self._reset_official_idle_state(state)
            return 0.0, False
        if frame_no < int(state["next_evaluation_frame"]):
            return 0.0, False

        penalty = -float(
            V1TrainingConfig.COMMAND_NO_COMMAND_IDLE_PENALTY
        )
        state["start_frame"] = frame_no
        state["start_position"] = position
        state["last_position"] = position
        state["path_distance"] = 0.0
        state["start_damage"] = damage
        state["next_evaluation_frame"] = (
            frame_no + V1TrainingConfig.COMMAND_NO_COMMAND_IDLE_WINDOW_FRAMES
        )
        return penalty, True

    def _consume_command_attribution_reward(self, context, dead_actions):
        """Return official outcomes plus bounded command-response shaping."""

        attribution = context["collaboration"].get("attribution")
        if attribution is None:
            return {
                "active": False,
                "intent": 0,
                "official_kda_reward": 0.0,
                "goal_progress_reward": 0.0,
                "nonresponse_penalty": 0.0,
                "nonresponse_applied": False,
            }

        official = context["official_hero"]
        official_runtime = u.actor_runtime_id(official["actor"])
        intent = int(attribution.get("intent", 0))
        target_runtime = int(attribution.get("target_runtime", 0))
        official_kda_reward = 0.0
        death_multiplier = (
            V1TrainingConfig.DEATH_MULTIPLIER_BEFORE_CANNON
            if context["frame_no"]
            < V1TrainingConfig.DEATH_MULTIPLIER_CANNON_FRAME
            else V1TrainingConfig.DEATH_MULTIPLIER_AFTER_CANNON
        )
        for event in dead_actions:
            death = u.safe_get(event, "death", "dead", default={}) or {}
            death_runtime = u.actor_runtime_id(death)
            if death_runtime == official_runtime:
                official_kda_reward += (
                    V1TrainingConfig.DEATH_REWARD * death_multiplier
                )
                continue
            if u.actor_config_id(death) not in u.HERO_CONFIG_IDS:
                continue
            death_camp = u.actor_camp(death)
            if death_camp not in (1, 2) and death_runtime:
                death_camp = u.actor_camp(
                    context["actor_by_runtime"].get(death_runtime)
                )
            if death_camp != 3 - self.camp:
                continue
            # A focus command only owns the requested enemy.  Other accepted
            # intents may own an official participation event during their
            # active option window.
            if intent == 2 and target_runtime and death_runtime != target_runtime:
                continue
            killer = u.safe_get(
                event, "killer", "killer_actor", default={}
            ) or {}
            killer_runtime = u.actor_runtime_id(killer)
            assists = list(
                u.safe_get(
                    event, "assist_set", "assistSet", default=[]
                )
                or []
            )
            assist_runtimes = {u.actor_runtime_id(item) for item in assists}
            if killer_runtime == official_runtime:
                official_kda_reward += V1TrainingConfig.KILL_REWARD
            elif official_runtime in assist_runtimes:
                official_kda_reward += V1TrainingConfig.ASSIST_REWARD

        action_type = int(attribution.get("action_type", 0))
        goal_position = None
        if action_type == 2:
            goal_position = u.side_position(
                attribution.get("raw_position"), self.camp
            )
        elif action_type == 1:
            target_actor = context["actor_by_runtime"].get(target_runtime)
            target_snapshot = context["hero_by_runtime"].get(target_runtime)
            if target_snapshot is not None:
                goal_position = target_snapshot["side_position"]
            elif target_actor is not None:
                goal_position = _position(target_actor, self.camp)
        official_position = official.get("side_position")
        progress_reward = 0.0
        nonresponse_penalty = 0.0
        nonresponse_applied = False

        def add_bounded_response(delta):
            nonlocal progress_reward
            budget = float(V1TrainingConfig.COMMAND_GOAL_PROGRESS_BUDGET)
            old_total = float(attribution.get("progress_total", 0.0))
            new_total = float(np.clip(old_total + float(delta), -budget, budget))
            applied = new_total - old_total
            attribution["progress_total"] = new_total
            progress_reward += applied
            return applied

        if goal_position is not None and official_position is not None:
            distance = _distance_or_inf(official_position, goal_position)
            budget = float(V1TrainingConfig.COMMAND_GOAL_PROGRESS_BUDGET)
            potential = budget * (
                1.0 - u.clip(distance / u.GLOBAL_DISTANCE_SCALE)
            )
            previous = attribution.get("last_potential")
            if previous is not None:
                add_bounded_response(potential - float(previous))
            attribution["last_potential"] = potential
            if attribution.get("initial_distance") is None:
                attribution["initial_distance"] = distance
                attribution["best_distance"] = distance
            else:
                attribution["best_distance"] = min(
                    float(attribution["best_distance"]), distance
                )
                if (
                    float(attribution["initial_distance"])
                    - float(attribution["best_distance"])
                    >= V1TrainingConfig.COMMAND_RESPONSE_MIN_PROGRESS_DISTANCE
                ):
                    attribution["response_observed"] = True

        if action_type == 2 and goal_position is not None and official_position is not None:
            if _distance_or_inf(official_position, goal_position) <= 1200.0:
                attribution["response_observed"] = True

        if action_type == 1 and target_runtime:
            locked = _target_runtime(official["actor"]) == target_runtime
            attack_range = max(
                0.0,
                _value(official["actor"], "attack_range", "attackRange"),
            )
            in_range = bool(
                goal_position is not None
                and official_position is not None
                and attack_range > 0.0
                and _distance_or_inf(official_position, goal_position)
                <= attack_range
            )
            hit_target = any(
                event["frame"] >= int(attribution["accepted_frame"])
                and event["other_runtime"] == target_runtime
                and event["type"] == "damage_dealt"
                and event["amount"] > 0.0
                for event in self.recent_events.get(
                    ("friend", u.actor_config_id(official["actor"])), ()
                )
            )
            if locked and not attribution.get("target_lock_rewarded"):
                add_bounded_response(
                    V1TrainingConfig.COMMAND_TARGET_LOCK_REWARD
                )
                attribution["target_lock_rewarded"] = True
            if in_range and not attribution.get("target_range_rewarded"):
                add_bounded_response(
                    V1TrainingConfig.COMMAND_TARGET_RANGE_REWARD
                )
                attribution["target_range_rewarded"] = True
            if hit_target and not attribution.get("target_hit_rewarded"):
                add_bounded_response(
                    V1TrainingConfig.COMMAND_TARGET_FIRST_HIT_REWARD
                )
                attribution["target_hit_rewarded"] = True
            # Lock/range are useful early response signals, but only real
            # approach progress or a hit prevents the later nonresponse
            # penalty. This catches a hero that locks and circles forever.
            if hit_target:
                attribution["response_observed"] = True

        response_age = context["frame_no"] - int(
            attribution.get("accepted_frame", context["frame_no"])
        )
        wait_exempt = bool(
            action_type == 2 and self._official_command_wait_exempt(context)
        )
        if (
            response_age >= V1TrainingConfig.COMMAND_RESPONSE_WINDOW_FRAMES
            and not attribution.get("response_observed")
            and not attribution.get("nonresponse_penalized")
            and not wait_exempt
        ):
            applied = add_bounded_response(
                -V1TrainingConfig.COMMAND_NO_RESPONSE_PENALTY
            )
            nonresponse_penalty = min(0.0, applied)
            # A saturated response budget can clip the requested penalty to
            # zero.  Do not report a penalty event when no penalty was applied.
            nonresponse_applied = nonresponse_penalty < 0.0
            attribution["nonresponse_penalized"] = True

        return {
            "active": True,
            "intent": intent,
            "official_kda_reward": float(official_kda_reward),
            "goal_progress_reward": float(progress_reward),
            "nonresponse_penalty": float(nonresponse_penalty),
            "nonresponse_applied": bool(nonresponse_applied),
        }

    def _append_event(self, cache_key, frame, event_type, amount=0.0, other_runtime=0, slot=None):
        event = {
            "frame": frame,
            "type": event_type,
            "amount": max(0.0, u.as_float(amount)),
            "other_runtime": u.as_int(other_runtime, 0),
            "slot": slot,
        }
        events = self.recent_events[cache_key]
        if events and all(
            events[-1].get(name) == event[name]
            for name in (
                "frame",
                "type",
                "amount",
                "other_runtime",
                "slot",
            )
        ):
            return
        events.append(event)

    def _update_recent_events(self, context):
        frame_no = context["frame_no"]
        for snapshot in context["heroes"]:
            if not snapshot["current_valid"]:
                continue
            actor = snapshot["actor"]
            key = (snapshot["relation"], u.actor_config_id(actor))
            runtime_id = u.actor_runtime_id(actor)

            for hit in u.actor_get(actor, "hit_target_info", "hitTargetInfo", default=[]) or []:
                target = u.as_int(u.safe_get(hit, "hit_target", "hitTarget", default=0), 0)
                target_actor = context["actor_by_runtime"].get(target)
                kind = u.target_type(target_actor)
                if kind == "hero":
                    event_type = "hit_hero"
                elif kind == "soldier":
                    event_type = "hit_soldier"
                elif kind in ("outer_tower", "high_tower", "crystal", "organ"):
                    event_type = "hit_organ"
                else:
                    event_type = "hit_monster"
                slot = u.normalized_skill_slot(u.safe_get(hit, "slot_type", "slotType", default=None))
                self._append_event(key, frame_no, event_type, other_runtime=target, slot=slot)

            for hurt in u.actor_get(actor, "hurt_hero_info", "hurtHeroInfo", default=[]) or []:
                target = u.as_int(u.safe_get(hurt, "hurt_target", "hurtTarget", default=0), 0)
                amount = u.as_float(u.safe_get(hurt, "hurt", default=0), 0)
                self._append_event(key, frame_no, "damage_dealt", amount, target)

            for taken in u.actor_get(actor, "take_hurt_infos", "takeHurtInfos", default=[]) or []:
                attacker = u.as_int(u.safe_get(taken, "atker", default=0), 0)
                amount = u.as_float(u.safe_get(taken, "hurt_value", "hurtValue", default=0), 0)
                source_actor = context["actor_by_runtime"].get(attacker)
                source_type = u.target_type(source_actor)
                explicit_type = str(u.safe_get(taken, "source_type", "sourceType", default="")).lower()
                if source_type == "hero" or "hero" in explicit_type:
                    event_type = "hurt_by_hero"
                elif source_type == "soldier" or "soldier" in explicit_type:
                    event_type = "hurt_by_soldier"
                elif source_type in ("outer_tower", "high_tower", "crystal", "organ") or "organ" in explicit_type or "tower" in explicit_type:
                    event_type = "hurt_by_tower"
                else:
                    event_type = "hurt_by_monster"
                self._append_event(key, frame_no, event_type, amount, attacker)
                self._append_event(key, frame_no, "damage_taken", amount, attacker)

            for slot_index, skill in enumerate(u.skill_slots(actor)):
                if skill is not None and u.as_int(
                    u.safe_get(skill, "succ_used_in_frame", "succUsedInFrame", default=0), 0
                ):
                    self._append_event(key, frame_no, "skill_used", slot=slot_index)

            counters = {
                "kill": _integer(actor, "kill_cnt", "killCnt"),
                "assist": _integer(actor, "assist_cnt", "assistCnt"),
                "death": _integer(actor, "dead_cnt", "deadCnt"),
            }
            previous = self.previous_hero_counters.get(key)
            if previous is not None and previous["runtime"] == runtime_id:
                for name in ("kill", "assist", "death"):
                    if counters[name] > previous[name]:
                        self._append_event(key, frame_no, name)
            self.previous_hero_counters[key] = {"runtime": runtime_id, **counters}

        for key, events in list(self.recent_events.items()):
            while events and frame_no - events[0]["frame"] > RECENT_EVENT_FRAMES:
                events.popleft()
            if not events:
                self.recent_events.pop(key, None)

    def _event_summary(self, snapshot, frame_no):
        key = (snapshot["relation"], u.actor_config_id(snapshot["actor"]))
        events = self.recent_events.get(key, ())
        short = [event for event in events if frame_no - event["frame"] <= RECENT_SHORT_FRAMES]
        long_window = [event for event in events if frame_no - event["frame"] <= RECENT_EVENT_FRAMES]
        last_skill = next(
            (event["slot"] for event in reversed(short) if event["type"] == "skill_used"), None
        )
        return {
            "short_types": {event["type"] for event in short},
            "long_types": {event["type"] for event in long_window},
            "damage_dealt": sum(event["amount"] for event in long_window if event["type"] == "damage_dealt"),
            "damage_taken": sum(event["amount"] for event in long_window if event["type"] == "damage_taken"),
            "last_skill": last_skill,
            "recent_other": {
                (event["type"], event["other_runtime"])
                for event in long_window
                if event["other_runtime"]
            },
        }

    def _build_heroes(self, context, query):
        vectors = np.zeros((6, 393), dtype=np.float32)
        for index, snapshot in enumerate(context["heroes"]):
            vectors[index] = self._build_one_hero(context, query, snapshot)
        return vectors

    def _build_one_hero(self, context, query, snapshot):
        fields = schema_v1.HERO_FIELD_MAP
        vector = np.zeros(393, dtype=np.float32)
        actor = snapshot["actor"]
        config_id = u.actor_config_id(actor)
        runtime_id = u.actor_runtime_id(actor)
        query_runtime = u.actor_runtime_id(query["actor"])
        is_friend = snapshot["relation"] == "friend"
        revive_time = _value(actor, "revive_time", "reviveTime")

        for name, value in {
            "slot_exists": snapshot["slot_exists"],
            "alive": snapshot["alive"],
            "visible_current": snapshot["visible"],
            "current_observation_valid": snapshot["current_valid"],
            "last_seen_used": snapshot["last_seen_used"],
            "last_seen_age_norm": u.clip(snapshot["last_seen_age"] / LAST_SEEN_FRAMES),
            "is_friend": is_friend,
            "is_enemy": not is_friend,
            "controlled_by_us": snapshot["controlled"],
            "official_teammate": snapshot["official"],
            "opponent_controlled": snapshot["opponent"],
            "is_query_main": runtime_id != 0 and runtime_id == query_runtime,
            "hero_id_132": config_id == 132,
            "hero_id_141": config_id == 141,
            "hero_id_193": config_id == 193,
            "respawning": not snapshot["alive"] and revive_time > 0,
        }.items():
            _put(vector, fields, name, value)

        hp = _value(actor, "hp")
        max_hp = _value(actor, "max_hp", "maxHp")
        ep = _value(actor, "ep")
        max_ep = _value(actor, "max_ep", "maxEp")
        hp_known = snapshot["current_valid"] or snapshot["last_seen_used"]
        ep_known = hp_known and max_ep > 0
        behavior = str(u.actor_get(actor, "behav_mode", "behavMode", default="")).lower()
        behavior_index = 4
        if not snapshot["alive"]:
            behavior_index = 0
        elif behavior in ("2", "move", "moving", "behav_move"):
            behavior_index = 1
        elif behavior in ("3", "4", "6", "attack", "attacking", "behav_attack"):
            behavior_index = 2
        elif behavior in ("7", "8", "cast", "casting", "skill", "behav_skill"):
            behavior_index = 3
        behavior_names = ("behavior_dead", "behavior_moving", "behavior_attacking", "behavior_casting", "behavior_other")
        for name, value in {
            "hp_ratio": u.ratio(hp, max_hp) if hp_known else 0,
            "max_hp_norm": u.clip(max_hp / 12000.0) if hp_known else 0,
            "hp_known": hp_known,
            "low_hp": hp_known and u.ratio(hp, max_hp) <= 0.3,
            "revive_time_norm": u.clip(revive_time / 60.0),
            "ep_ratio": u.ratio(ep, max_ep) if ep_known else 0,
            "max_ep_norm": u.clip(max_ep / 10000.0) if ep_known else 0,
            "ep_known": ep_known,
            "uses_ep": max_ep > 0,
            "hp_recover_norm": u.clip(_value(actor, "hp_recover", "hpRecover") / 1000.0),
            "ep_recover_norm": u.clip(_value(actor, "ep_recover", "epRecover") / 1000.0),
        }.items():
            _put(vector, fields, name, value)
        _put_slice(vector, fields, behavior_names[0], u.one_hot(behavior_index, 5))

        combat_specs = (
            ("phy_atk_norm", ("phy_atk", "phyAtk"), 100.0, 900.0),
            ("phy_def_norm", ("phy_def", "phyDef"), 0.0, 1200.0),
            ("mgc_atk_norm", ("mgc_atk", "mgcAtk"), 0.0, 1000.0),
            ("mgc_def_norm", ("mgc_def", "mgcDef"), 0.0, 700.0),
            ("mov_spd_norm", ("mov_spd", "movSpd"), 1250.0, 7000.0),
            ("atk_spd_norm", ("atk_spd", "atkSpd"), 0.0, 10000.0),
            ("phy_armor_hurt_norm", ("phy_armor_hurt", "phyArmorHurt"), 0.0, 240.0),
            ("mgc_armor_hurt_norm", ("mgc_armor_hurt", "mgcArmorHurt"), 0.0, 170.0),
            ("crit_rate_norm", ("crit_rate", "critRate"), 0.0, 5000.0),
            ("crit_effect_norm", ("crit_effect", "critEffect"), 9500.0, 17000.0),
            ("phy_vamp_norm", ("phy_vamp", "phyVamp"), 0.0, 4000.0),
            ("mgc_vamp_norm", ("mgc_vamp", "mgcVamp"), 0.0, 3400.0),
            ("cd_reduce_norm", ("cd_reduce", "cdReduce"), 0.0, 3600.0),
            ("ctrl_reduce_norm", ("ctrl_reduce", "ctrlReduce"), 0.0, 3500.0),
            ("attack_range_norm", ("attack_range", "attackRange"), 0.0, 13000.0),
            ("sight_area_norm", ("sight_area", "sightArea"), 0.0, 20000.0),
        )
        for name, aliases, low, high in combat_specs:
            _put(vector, fields, name, u.min_max(_value(actor, *aliases), low, high))

        for slot_index, skill in enumerate(u.skill_slots(actor)):
            if skill is None:
                continue
            prefix = schema_v1.SKILL_SLOT_NAMES[slot_index]
            cooldown = u.as_float(u.safe_get(skill, "cooldown", default=0), 0)
            cooldown_max = u.as_float(
                u.safe_get(skill, "cooldown_max", "cooldownMax", default=0), 0
            )
            skill_values = {
                "exists": 1,
                "level_norm": u.clip(u.as_float(u.safe_get(skill, "level", default=0), 0) / 15.0),
                "usable": bool(u.safe_get(skill, "usable", default=False)),
                "cooldown_ratio": u.ratio(cooldown, cooldown_max),
                "cooldown_max_norm": u.clip(cooldown_max / 120000.0),
                "succ_used_in_frame": bool(
                    u.safe_get(skill, "succ_used_in_frame", "succUsedInFrame", default=0)
                ),
                "used_times_norm": u.clip(
                    u.as_float(u.safe_get(skill, "used_times", "usedTimes", default=0), 0) / 100.0
                ),
                "hit_hero_times_norm": u.clip(
                    u.as_float(
                        u.safe_get(skill, "hit_hero_times", "hitHeroTimes", default=0), 0
                    )
                    / 100.0
                ),
                "has_next_config": bool(
                    u.safe_get(skill, "next_config_id", "nextConfigId", default=0)
                ),
                "combo_effect_time_norm": u.clip(
                    u.as_float(
                        u.safe_get(skill, "combo_effect_time", "comboEffectTime", default=0), 0
                    )
                    / 12000.0
                ),
            }
            for suffix, value in skill_values.items():
                _put(vector, fields, f"{prefix}_{suffix}", value)

        level = _value(actor, "level")
        exp = _value(actor, "exp")
        next_exp = _value(actor, "next_level_exp", "nextLevelExp", "max_exp", "maxExp")
        money = _value(actor, "money")
        money_total = _value(actor, "money_cnt", "moneyCnt")
        equip_state = u.actor_get(actor, "equip_state", "equipState", default={}) or {}
        equips = list(u.safe_get(equip_state, "equips", "equipments", default=[]) or [])
        equip_value = _value(actor, "equipment_value", "equipmentValue")
        if equip_value <= 0:
            equip_value = sum(
                u.as_float(u.safe_get(item, "price", "buy_price", "buyPrice", default=0), 0)
                for item in equips
            )
        for name, value in {
            "level_norm": u.clip(level / 15.0),
            "exp_progress_norm": u.ratio(exp, next_exp),
            "money_current_norm": u.clip(money / 2000.0),
            "money_total_norm": u.clip(money_total / 15000.0),
            "money_delta_norm": u.clip(snapshot["money_delta"] / 1200.0),
            "equipment_value_norm": u.clip(equip_value / 15000.0),
            "equipment_slot_count_norm": u.clip(len(equips) / 6.0),
            "kill_cnt_norm": u.clip(_value(actor, "kill_cnt", "killCnt") / 10.0),
            "dead_cnt_norm": u.clip(_value(actor, "dead_cnt", "deadCnt") / 10.0),
            "assist_cnt_norm": u.clip(_value(actor, "assist_cnt", "assistCnt") / 15.0),
            "total_hurt_to_hero_log_norm": u.log_norm(
                _value(actor, "total_hurt_to_hero", "totalHurtToHero"), 100000
            ),
            "total_be_hurt_by_hero_log_norm": u.log_norm(
                _value(actor, "total_be_hurt_by_hero", "totalBeHurtByHero"), 100000
            ),
        }.items():
            _put(vector, fields, name, value)

        abilities = list(u.actor_get(actor, "abilities", default=[]) or [])
        for ability_index, name in enumerate(schema_v1.ABILITY_NAMES):
            _put(vector, fields, f"ability_{name}", bool(abilities[ability_index]) if ability_index < len(abilities) else 0)
        _put(vector, fields, "abilities_valid", len(abilities) >= len(schema_v1.ABILITY_NAMES))

        self._fill_hero_buff_mark(vector, actor, snapshot, context)
        self._fill_hero_position_movement(vector, snapshot, query, context)
        self._fill_hero_recent_tactical(vector, snapshot, query, context)
        return vector

    def _fill_hero_buff_mark(self, vector, actor, snapshot, context):
        fields = schema_v1.HERO_FIELD_MAP
        buffs = u.buff_items(actor)
        buff_by_id = defaultdict(list)
        for item in buffs:
            config_id = u.as_int(u.safe_get(item, "config_id", "configId", default=0), 0)
            if config_id:
                buff_by_id[config_id].append(item)
        buff_ids = set(buff_by_id)
        whitelist = set(schema_v1.BUFF_RAW_WHITELIST)
        for config_id in schema_v1.BUFF_RAW_WHITELIST:
            _put(vector, fields, f"buff_raw_{config_id}_present", config_id in buff_ids)

        semantic_ids = (
            (90000,),
            (900009,),
            (90000, 900009),
            (90201,),
            (90014,),
            (90204, 902070),
            (90206,),
            (90202,),
            (801049,),
            (60005, 60006, 60007),
            (60006, 60007),
            (60007,),
        )
        semantic_durations = (30, 30, 30, 15, 4.5, 30, 2, 30, 30, 900, 900, 900)
        now_ms = context["frame_no"] * u.RAW_FRAME_SECONDS * 1000.0
        for name, ids, duration in zip(
            schema_v1.BUFF_STRONG_SEMANTIC_FLAGS, semantic_ids, semantic_durations
        ):
            if name in ("has_recover_combo", "has_90204_902070_dark_wolf_long"):
                present = all(config_id in buff_ids for config_id in ids)
            else:
                present = any(config_id in buff_ids for config_id in ids)
            _put(vector, fields, name, present)
            starts = [
                u.as_float(u.safe_get(item, "start_time", "startTime", default=now_ms), now_ms)
                for config_id in ids
                for item in buff_by_id.get(config_id, [])
            ]
            age = max(0.0, now_ms - min(starts)) / 1000.0 if present and starts else 0.0
            _put(vector, fields, f"{name}_age_norm", u.clip(age / duration))

        family_present = {family: False for family in schema_v1.BUFF_FAMILIES}
        unknown_count = 0
        for config_id in buff_ids:
            text_id = str(config_id)
            family = next(
                (prefix for prefix in schema_v1.BUFF_FAMILIES[:-1] if text_id.startswith(prefix)),
                "other",
            )
            family_present[family] = True
            if config_id not in whitelist:
                unknown_count += 1
                _put(
                    vector,
                    fields,
                    f"buff_unknown_hash_{schema_v1.stable_unknown_buff_bucket(config_id)}_present",
                    1,
                )
        for family, present in family_present.items():
            _put(vector, fields, f"buff_family_{family}_present", present)
        for name, value in {
            "buff_count_norm": u.clip(len(buff_ids) / 16.0),
            "new_buff_count_norm": u.clip(snapshot["buff_delta"]["new"] / 16.0),
            "lost_buff_count_norm": u.clip(snapshot["buff_delta"]["lost"] / 16.0),
            "unknown_buff_count_norm": u.clip(unknown_count / 8.0),
            "has_no_buff": not buff_ids,
        }.items():
            _put(vector, fields, name, value)

        marks = u.mark_items(actor)
        _put(vector, fields, "mark_known", snapshot["current_valid"] or snapshot["last_seen_used"])
        marco = [item for item in marks if u.as_int(u.safe_get(item, "config_id", "configId", default=0), 0) == 13200]
        diaochan = [
            item
            for item in marks
            if u.as_int(u.safe_get(item, "config_id", "configId", default=0), 0) in (14190, 14191)
        ]
        _put(vector, fields, "marco_13200_present", bool(marco))
        _put(vector, fields, "marco_layer_norm", u.clip(max((u.as_float(u.safe_get(item, "layer", default=0), 0) for item in marco), default=0) / 9.0))
        _put(vector, fields, "diaochan_mark_present", bool(diaochan))
        _put(vector, fields, "diaochan_layer_norm", u.clip(max((u.as_float(u.safe_get(item, "layer", default=0), 0) for item in diaochan), default=0) / 3.0))
        origins = [
            u.as_int(
                u.safe_get(
                    item,
                    "origin_actor_id",
                    "origin_actorId",
                    "originActorId",
                    default=0,
                ),
                0,
            )
            for item in marks
        ]
        actor_runtime = u.actor_runtime_id(actor)
        friend_ids = {u.actor_runtime_id(item["actor"]) for item in context["friend_heroes"]}
        enemy_ids = {u.actor_runtime_id(item["actor"]) for item in context["enemy_heroes"]}
        _put(vector, fields, "mark_origin_is_self", actor_runtime in origins)
        _put(vector, fields, "mark_origin_is_friend", any(origin in friend_ids and origin != actor_runtime for origin in origins))
        _put(vector, fields, "mark_origin_is_enemy", any(origin in enemy_ids for origin in origins))
        _put(vector, fields, "mark_origin_unknown", any(origin not in friend_ids | enemy_ids for origin in origins))
        _put(vector, fields, "other_mark_present", any(u.as_int(u.safe_get(item, "config_id", "configId", default=0), 0) not in (13200, 14190, 14191) for item in marks))

    def _fill_hero_position_movement(self, vector, snapshot, query, context):
        fields = schema_v1.HERO_FIELD_MAP
        position = snapshot["side_position"]
        query_position = query["side_position"]
        normalized = u.normalized_side_position(position)
        coarse_x, coarse_z = u.coarse_position_buckets(position)
        _put(vector, fields, "location_valid", snapshot["position_valid"])
        _put(vector, fields, "movement_cache_valid", snapshot["motion"]["valid"])
        _put(vector, fields, "side_x_norm", normalized[0])
        _put(vector, fields, "side_z_norm", normalized[1])
        _put_slice(vector, fields, "side_x_coarse_bucket_0", coarse_x)
        _put_slice(vector, fields, "side_z_coarse_bucket_0", coarse_z)
        _put_slice(vector, fields, "region_own_base", u.region_one_hot(position))
        if position is not None:
            own_base = context["organs"][2]["side_position"]
            enemy_base = context["organs"][5]["side_position"]
            _put(vector, fields, "distance_to_lane_center_norm", u.clip(abs(position[1]) / u.SIDE_Z_SCALE))
            _put(vector, fields, "distance_to_own_base_norm", u.clip(_distance_or_inf(position, own_base) / u.GLOBAL_DISTANCE_SCALE))
            _put(vector, fields, "distance_to_enemy_base_norm", u.clip(_distance_or_inf(position, enemy_base) / u.GLOBAL_DISTANCE_SCALE))

        motion = snapshot["motion"]
        _put(vector, fields, "movement_valid", motion["valid"])
        if not motion["valid"]:
            return
        _put(vector, fields, "delta_side_x_per_frame_norm", u.signed_clip(motion["vx"] / 7000.0))
        _put(vector, fields, "delta_side_z_per_frame_norm", u.signed_clip(motion["vz"] / 7000.0))
        _put(vector, fields, "speed_per_frame_norm", u.clip(motion["speed"] / 7000.0))

        friend_positions = [
            hero["side_position"]
            for hero in context["friend_heroes"]
            if hero is not snapshot and hero["side_position"] is not None and hero["alive"]
        ]
        enemy_positions = [
            hero["side_position"]
            for hero in context["enemy_heroes"]
            if hero["side_position"] is not None and hero["alive"]
        ]
        nearest_friend = min(friend_positions, key=lambda other: _distance_or_inf(position, other), default=None)
        nearest_enemy = min(enemy_positions, key=lambda other: _distance_or_inf(position, other), default=None)
        enemy_towers = [organ for organ in context["organs"][3:] if organ["alive"]]
        nearest_enemy_tower = min(
            (organ["side_position"] for organ in enemy_towers),
            key=lambda other: _distance_or_inf(position, other),
            default=None,
        )
        if nearest_enemy_tower is not None:
            away_point = (2 * position[0] - nearest_enemy_tower[0], 2 * position[1] - nearest_enemy_tower[1])
        else:
            away_point = None
        targets = {
            "toward_enemy_crystal": context["organs"][5]["side_position"],
            "toward_own_crystal": context["organs"][2]["side_position"],
            "toward_query_hero": query_position if snapshot is not query else None,
            "toward_nearest_enemy": nearest_enemy,
            "toward_nearest_friend": nearest_friend,
            "toward_tyrant": u.side_position(context["tyrant"]["anchor"], self.camp),
            "toward_lane_center": (position[0], 0.0) if position is not None else None,
            "away_from_enemy_tower": away_point,
        }
        for name, target in targets.items():
            _put(vector, fields, name, _normalized_velocity_toward(motion, position, target) > 0.05)

    def _fill_hero_recent_tactical(self, vector, snapshot, query, context):
        fields = schema_v1.HERO_FIELD_MAP
        event = self._event_summary(snapshot, context["frame_no"])
        short = event["short_types"]
        long_types = event["long_types"]
        flag_types = {
            "recent_hit_hero": "hit_hero",
            "recent_hit_soldier": "hit_soldier",
            "recent_hit_monster": "hit_monster",
            "recent_hit_organ": "hit_organ",
            "recent_hurt_by_hero": "hurt_by_hero",
            "recent_hurt_by_soldier": "hurt_by_soldier",
            "recent_hurt_by_monster": "hurt_by_monster",
            "recent_hurt_by_tower": "hurt_by_tower",
            "recent_skill_used": "skill_used",
        }
        for name, event_type in flag_types.items():
            _put(vector, fields, name, event_type in short)
        _put_slice(
            vector,
            fields,
            "last_skill_slot_0",
            u.one_hot(event["last_skill"], 6) if event["last_skill"] is not None else np.zeros(6),
        )
        _put(vector, fields, "recent_damage_dealt_norm", u.log_norm(event["damage_dealt"], 20000))
        _put(vector, fields, "recent_damage_taken_norm", u.log_norm(event["damage_taken"], 20000))
        _put(vector, fields, "recent_kill", "kill" in long_types)
        _put(vector, fields, "recent_assist", "assist" in long_types)
        _put(vector, fields, "recent_death", "death" in long_types)

        actor = snapshot["actor"]
        runtime_id = u.actor_runtime_id(actor)
        position = snapshot["side_position"]
        enemy_towers = [
            organ for organ in context["organs"][3:5] if organ["alive"]
        ]
        own_towers = [
            organ for organ in context["organs"][:2] if organ["alive"]
        ]
        enemy_distances = [
            _distance_or_inf(position, organ["side_position"])
            for organ in enemy_towers
        ]
        own_distances = [
            _distance_or_inf(position, organ["side_position"])
            for organ in own_towers
        ]
        in_enemy_range = any(
            distance <= _value(organ["actor"], "attack_range", "attackRange")
            for distance, organ in zip(enemy_distances, enemy_towers)
            if _value(organ["actor"], "attack_range", "attackRange") > 0
        )
        in_own_range = any(
            distance <= _value(organ["actor"], "attack_range", "attackRange")
            for distance, organ in zip(own_distances, own_towers)
            if _value(organ["actor"], "attack_range", "attackRange") > 0
        )
        targeted = any(_target_runtime(organ["actor"]) == runtime_id for organ in enemy_towers)
        friend_soldiers = tuple(context["lane_summary"]["friend"])
        friend_soldier_ids = {
            u.actor_runtime_id(soldier) for soldier in friend_soldiers
        }
        related_enemy_tower = min(
            enemy_towers,
            key=lambda organ: (
                _distance_or_inf(position, organ["side_position"])
                - max(
                    0.0,
                    _value(
                        organ["actor"],
                        "attack_range",
                        "attackRange",
                    ),
                )
            ),
            default=None,
        )
        enemy_tower_minion_target = bool(
            related_enemy_tower is not None
            and _target_runtime(related_enemy_tower["actor"])
            in friend_soldier_ids
        )
        # "Cover" is tower-relative, not hero-relative. A nearby minion that
        # is outside every related tower's actual range cannot tank that tower.
        # Missing/zero attack_range is deliberately conservative.
        related_attack_range = (
            _value(
                related_enemy_tower["actor"],
                "attack_range",
                "attackRange",
            )
            if related_enemy_tower is not None
            else 0.0
        )
        own_cover = bool(
            related_enemy_tower is not None
            and related_attack_range > 0.0
            and any(
                _position(soldier, self.camp) is not None
                and _distance_or_inf(
                    related_enemy_tower["side_position"],
                    _position(soldier, self.camp),
                )
                <= related_attack_range
                for soldier in friend_soldiers
                if u.is_alive(soldier)
            )
        )
        values = {
            "is_in_grass": bool(u.actor_get(actor, "is_in_grass", "isInGrass", default=False)),
            "in_enemy_tower_range": in_enemy_range,
            "targeted_by_enemy_tower": targeted,
            "in_own_tower_range": in_own_range,
            "enemy_tower_has_minion_target": enemy_tower_minion_target,
            "own_minion_cover": own_cover,
            "nearest_enemy_tower_distance_norm": u.clip(min(enemy_distances, default=u.TOWER_DISTANCE_SCALE) / u.TOWER_DISTANCE_SCALE),
            "nearest_own_tower_distance_norm": u.clip(min(own_distances, default=u.TOWER_DISTANCE_SCALE) / u.TOWER_DISTANCE_SCALE),
        }
        for name, value in values.items():
            _put(vector, fields, name, value)

    @staticmethod
    def _soldier_type(actor):
        config_id = u.actor_config_id(actor)
        if config_id in (16, 19):
            return "melee"
        if config_id in (17, 20):
            return "ranged"
        if config_id in (18, 21):
            return "cannon"
        return "super"

    def _build_soldiers(self, context):
        result = np.zeros((20, 56), dtype=np.float32)
        fields = schema_v1.SOLDIER_FIELD_MAP
        own_anchors = tuple(organ["side_position"][0] for organ in context["organs"][:3])
        enemy_anchors = tuple(organ["side_position"][0] for organ in context["organs"][3:])
        hero_config_by_runtime = {
            u.actor_runtime_id(hero["actor"]): u.actor_config_id(hero["actor"])
            for hero in context["heroes"]
        }
        for index, actor in enumerate(context["soldiers"]):
            if actor is None:
                continue
            vector = result[index]
            position = _position(actor, self.camp)
            visible = u.visible_to_camp(actor, self.camp, is_friend=u.actor_camp(actor) == self.camp)
            normalized = u.normalized_side_position(position)
            coarse_x, coarse_z = u.coarse_position_buckets(position)
            _put(vector, fields, "location_valid", position is not None)
            _put(vector, fields, "position_current_valid", position is not None and visible)
            _put(vector, fields, "side_x_norm", normalized[0])
            _put(vector, fields, "side_z_norm", normalized[1])
            _put_slice(vector, fields, "side_x_coarse_bucket_0", coarse_x)
            _put_slice(vector, fields, "side_z_coarse_bucket_0", coarse_z)
            if position is not None:
                _put_slice(
                    vector,
                    fields,
                    "lane_zone_own_base",
                    u.lane_zone(position[0], own_anchors, enemy_anchors),
                )
            is_friend = u.actor_camp(actor) == self.camp
            for name, value in {
                "slot_exists": True,
                "alive": u.is_alive(actor),
                "visible_current": visible,
                "belongs_to_official_target_order": not is_friend and index >= 10,
                "is_friend": is_friend,
                "is_enemy": not is_friend,
            }.items():
                _put(vector, fields, name, value)
            soldier_type = self._soldier_type(actor)
            for name in ("melee", "ranged", "cannon", "super"):
                _put(vector, fields, f"type_{name}", soldier_type == name)
            hp = _value(actor, "hp")
            max_hp = _value(actor, "max_hp", "maxHp")
            for name, value in {
                "hp_ratio": u.ratio(hp, max_hp),
                "max_hp_norm": u.clip(max_hp / 20000.0),
                "hp_known": max_hp > 0,
                "low_hp": max_hp > 0 and u.ratio(hp, max_hp) <= 0.3,
                "attack_range_norm": u.clip(_value(actor, "attack_range", "attackRange") / 13000.0),
                "phy_atk_norm": u.min_max(_value(actor, "phy_atk", "phyAtk"), 0, 900),
                "kill_income_norm": u.clip(_value(actor, "kill_income", "killIncome") / 200.0),
            }.items():
                _put(vector, fields, name, value)
            target_runtime = _target_runtime(actor)
            target_actor = context["actor_by_runtime"].get(target_runtime)
            target_type = u.target_type(target_actor)
            target_config = hero_config_by_runtime.get(target_runtime)
            if not target_runtime:
                target_index = 0
            elif target_config in u.HERO_CONFIG_IDS:
                target_index = 1 + u.HERO_CONFIG_IDS.index(target_config)
            elif target_type == "soldier":
                target_index = 4
            elif target_type in ("outer_tower", "high_tower", "crystal", "organ"):
                target_index = 5
            elif target_type == "monster":
                target_index = 6
            else:
                target_index = 7
            _put_slice(vector, fields, "attack_target_none", u.one_hot(target_index, 8))
        return result

    def _build_organs(self, context, query):
        result = np.zeros((6, 48), dtype=np.float32)
        fields = schema_v1.ORGAN_FIELD_MAP
        friend_hero_ids = {u.actor_runtime_id(hero["actor"]) for hero in context["friend_heroes"]}
        enemy_hero_ids = {u.actor_runtime_id(hero["actor"]) for hero in context["enemy_heroes"]}
        friend_soldier_ids = {u.actor_runtime_id(actor) for actor in context["soldiers"][:10] if actor}
        enemy_soldier_ids = {u.actor_runtime_id(actor) for actor in context["soldiers"][10:] if actor}
        for index, organ in enumerate(context["organs"]):
            vector = result[index]
            actor = organ["actor"]
            anchor_side = organ["side_position"]
            normalized = u.normalized_side_position(anchor_side)
            coarse_x, coarse_z = u.coarse_position_buckets(anchor_side)
            _put(vector, fields, "anchor_location_valid", anchor_side is not None)
            _put(vector, fields, "side_x_norm", normalized[0])
            _put(vector, fields, "side_z_norm", normalized[1])
            _put_slice(vector, fields, "side_x_coarse_bucket_0", coarse_x)
            _put_slice(vector, fields, "side_z_coarse_bucket_0", coarse_z)
            visible = bool(actor) and u.visible_to_camp(actor, self.camp, is_friend=organ["friend"])
            for name, value in {
                "slot_exists": True,
                "alive": organ["alive"],
                "visible_current": visible,
                "current_actor_position_valid": bool(actor) and u.location_of(actor) is not None,
                "is_friend": organ["friend"],
                "is_enemy": not organ["friend"],
                "type_outer": organ["type"] == "outer",
                "type_high": organ["type"] == "high",
                "type_crystal": organ["type"] == "crystal",
            }.items():
                _put(vector, fields, name, value)
            hp = _value(actor, "hp") if actor else 0
            max_hp = _value(actor, "max_hp", "maxHp") if actor else 0
            for name, value in {
                "hp_ratio": u.ratio(hp, max_hp),
                "max_hp_norm": u.clip(max_hp / 12000.0),
                "hp_known": max_hp > 0,
                "low_hp": max_hp > 0 and u.ratio(hp, max_hp) <= 0.3,
                "attack_range_norm": u.clip(_value(actor, "attack_range", "attackRange") / 13000.0) if actor else 0,
                "attack_norm": u.min_max(_value(actor, "phy_atk", "phyAtk", "attack"), 0, 1500) if actor else 0,
                "kill_income_norm": u.clip(_value(actor, "kill_income", "killIncome") / 500.0) if actor else 0,
            }.items():
                _put(vector, fields, name, value)
            target_runtime = _target_runtime(actor) if actor else 0
            if not target_runtime:
                target_type_index = 0
            elif target_runtime in friend_hero_ids:
                target_type_index = 1
            elif target_runtime in enemy_hero_ids:
                target_type_index = 2
            elif target_runtime in friend_soldier_ids | enemy_soldier_ids:
                target_type_index = 3
            else:
                target_type_index = 4
            _put_slice(vector, fields, "attack_target_none", u.one_hot(target_type_index, 5))
            attack_range = _value(actor, "attack_range", "attackRange") if actor else 0
            if attack_range > 0:
                _put(
                    vector,
                    fields,
                    "friend_soldier_in_range",
                    any(_distance_or_inf(anchor_side, _position(soldier, self.camp)) <= attack_range for soldier in context["lane_summary"]["friend"]),
                )
                _put(
                    vector,
                    fields,
                    "enemy_soldier_in_range",
                    any(_distance_or_inf(anchor_side, _position(soldier, self.camp)) <= attack_range for soldier in context["lane_summary"]["enemy"]),
                )
            _put(vector, fields, "tower_hit_streak_norm", u.clip(organ.get("hit_streak", 0) / 10.0))
            _put(
                vector,
                fields,
                "current_target_has_minion_cover",
                target_runtime in (enemy_soldier_ids if organ["friend"] else friend_soldier_ids),
            )
        return result

    def _build_monsters(self, context):
        result = np.zeros((20, 60), dtype=np.float32)
        fields = schema_v1.MONSTER_FIELD_MAP
        friend_ids = {u.actor_runtime_id(hero["actor"]) for hero in context["friend_heroes"]}
        for index, monster in enumerate(context["monsters"]):
            vector = result[index]
            actor = monster["actor"]
            current_position = _position(actor, self.camp) if actor else None
            anchor_position = u.side_position(monster["anchor"], self.camp)
            position = current_position if current_position is not None else anchor_position
            normalized = u.normalized_side_position(position)
            coarse_x, coarse_z = u.coarse_position_buckets(position)
            _put(vector, fields, "location_valid", position is not None)
            _put(vector, fields, "position_current_valid", current_position is not None and monster["visible"])
            _put(vector, fields, "side_x_norm", normalized[0])
            _put(vector, fields, "side_z_norm", normalized[1])
            _put_slice(vector, fields, "side_x_coarse_bucket_0", coarse_x)
            _put_slice(vector, fields, "side_z_coarse_bucket_0", coarse_z)
            _put_slice(vector, fields, "region_own_base", u.region_one_hot(position))
            for name, value in {
                "slot_exists": True,
                "alive": monster["alive"],
                "visible_current": monster["visible"],
                "position_source_is_current_actor": current_position is not None,
                "side_relation_own": monster["relation"] == "own",
                "side_relation_enemy": monster["relation"] == "enemy",
                "side_relation_center": monster["relation"] == "center",
            }.items():
                _put(vector, fields, name, value)
            type_ids = (49, 30, 31, 32, 33, 42, 43, 44, 45, 59, 122)
            _put_slice(
                vector,
                fields,
                "monster_type_49",
                u.one_hot(type_ids.index(monster["config_id"]), len(type_ids)),
            )
            hp = _value(actor, "hp") if actor else 0
            max_hp = _value(actor, "max_hp", "maxHp") if actor else 0
            _put(vector, fields, "hp_ratio", u.ratio(hp, max_hp))
            _put(vector, fields, "max_hp_norm", u.clip(max_hp / 20000.0))
            _put(vector, fields, "hp_known", max_hp > 0)
            spawn = context["monster_spawn"][index]
            timer_raw = u.safe_get(spawn, "spawn_timer", "spawnTimer", default=None) if spawn else None
            timer_valid = timer_raw is not None and math.isfinite(u.as_float(timer_raw, float("nan")))
            _put(vector, fields, "spawn_timer_valid", timer_valid)
            period = MONSTER_RESPAWN_SECONDS.get(monster["config_id"], 90.0)
            _put(vector, fields, "spawn_timer_norm", u.clip(_seconds_from_ms(timer_raw) / period) if timer_valid else 0)
            team_attacking = bool(actor) and _target_runtime(actor) in friend_ids
            monster_runtime = u.actor_runtime_id(actor) if actor else 0
            if monster_runtime:
                for hero in context["friend_heroes"]:
                    events = self._event_summary(hero, context["frame_no"])["recent_other"]
                    if any(other == monster_runtime and event_type.startswith("hit_") for event_type, other in events):
                        team_attacking = True
                        break
            _put(vector, fields, "team_attacking_monster", team_attacking)
        return result

    def _build_bullets(self, context, query):
        result = np.zeros((10, 32), dtype=np.float32)
        fields = schema_v1.BULLET_FIELD_MAP
        query_position = query["side_position"]
        query_runtime = u.actor_runtime_id(query["actor"])
        enemy_camp = 3 - self.camp
        hero_candidates = []
        tower_candidates = []
        for bullet in context["bullets_raw"]:
            source_runtime = u.as_int(
                u.safe_get(bullet, "source_actor", "sourceActor", default=0), 0
            )
            source = context["actor_by_runtime"].get(source_runtime)
            source_type = u.target_type(source)
            source_camp = u.actor_camp(source) if source else u.normalize_camp(
                u.safe_get(bullet, "camp", default=0)
            )
            if source_camp not in (0, enemy_camp):
                continue
            position = _position(bullet, self.camp)
            if position is None:
                continue
            item = (bullet, source, position)
            if source_type in ("outer_tower", "high_tower", "crystal", "organ"):
                tower_candidates.append(item)
            else:
                hero_candidates.append(item)
        sort_key = lambda item: (_distance_or_inf(query_position, item[2]), u.actor_runtime_id(item[0]))
        selected = sorted(hero_candidates, key=sort_key)[:5] + sorted(tower_candidates, key=sort_key)[:5]
        for index, (bullet, source, position) in enumerate(selected):
            vector = result[index]
            source_runtime = u.as_int(
                u.safe_get(bullet, "source_actor", "sourceActor", default=0), 0
            )
            source_config = u.actor_config_id(source) if source else 0
            source_type = u.target_type(source)
            dynamic = context["bullet_dynamics"].get(
                u.actor_runtime_id(bullet), {"valid": False, "vx": 0.0, "vz": 0.0, "speed": 0.0}
            )
            _put(vector, fields, "slot_exists", True)
            _put(vector, fields, "source_valid", source is not None)
            _put(vector, fields, "position_valid", True)
            _put(vector, fields, "velocity_valid", dynamic["valid"])
            for hero_id in u.HERO_CONFIG_IDS:
                _put(vector, fields, f"source_type_hero{hero_id}", source_type == "hero" and source_config == hero_id)
            _put(
                vector,
                fields,
                "source_type_tower",
                source_type in ("outer_tower", "high_tower", "crystal", "organ"),
            )
            slot = u.normalized_skill_slot(
                u.safe_get(bullet, "slot_type", "slotType", default=None)
            )
            if slot == 0:
                slot_index = 0
            elif slot in (1, 2, 3):
                slot_index = slot
            elif slot in (4, 5):
                slot_index = 4
            else:
                slot_index = 5
            _put_slice(vector, fields, "source_slot_common", u.one_hot(slot_index, 6))
            geometry = u.relative_geometry(query_position, position)
            _put_slice(vector, fields, "relative_dx_norm", geometry[:5])
            if dynamic["valid"]:
                _put(vector, fields, "velocity_side_x_norm", u.signed_clip(dynamic["vx"] / 20000.0))
                _put(vector, fields, "velocity_side_z_norm", u.signed_clip(dynamic["vz"] / 20000.0))
                _put(vector, fields, "speed_norm", u.clip(dynamic["speed"] / 20000.0))

            query_motion = query["motion"]
            relative_valid = bool(dynamic["valid"] and query_motion["valid"] and query_position is not None)
            _put(vector, fields, "relative_motion_valid", relative_valid)
            if relative_valid:
                rvx = dynamic["vx"] - query_motion["vx"]
                rvz = dynamic["vz"] - query_motion["vz"]
                _put(vector, fields, "relative_velocity_dx_norm", u.signed_clip(rvx / 20000.0))
                _put(vector, fields, "relative_velocity_dz_norm", u.signed_clip(rvz / 20000.0))
                rx, rz = position[0] - query_position[0], position[1] - query_position[1]
                dist = math.hypot(rx, rz)
                if dist > 1e-6:
                    closing = -(rvx * rx + rvz * rz) / dist
                    tangential = (rx * rvz - rz * rvx) / dist
                else:
                    closing, tangential = 0.0, 0.0
                _put(vector, fields, "radial_closing_speed_norm", u.signed_clip(closing / 20000.0))
                _put(vector, fields, "tangential_speed_norm", u.signed_clip(tangential / 20000.0))
                speed_sq = rvx * rvx + rvz * rvz
                closest_t = -(rx * rvx + rz * rvz) / speed_sq if speed_sq > 1e-6 else -1.0
                future_t = min(3.0, max(0.0, closest_t))
                closest_distance = math.hypot(rx + rvx * future_t, rz + rvz * future_t)
                if 0.0 <= closest_t <= 3.0:
                    _put(vector, fields, "time_to_closest_approach_norm", u.clip(closest_t / 3.0))
                    _put(vector, fields, "closest_approach_distance_norm", u.clip(closest_distance / 5000.0))
                    _put(vector, fields, "collision_course", closest_distance <= 1200.0)
            bullet_target = u.as_int(
                u.safe_get(bullet, "target_runtime_id", "targetRuntimeId", "target", default=0), 0
            )
            _put(
                vector,
                fields,
                "targets_query_hero",
                bullet_target == query_runtime or (source is not None and _target_runtime(source) == query_runtime),
            )
            _put(
                vector,
                fields,
                "unknown_skill",
                not u.as_int(u.safe_get(bullet, "skill_id", "skillId", default=0), 0) or slot is None,
            )
        return result

    def _target_descriptors(self, context, query):
        descriptors = [{"kind": "none", "slot_valid": True, "actor": None, "snapshot": None}]
        for snapshot in context["enemy_heroes"]:
            descriptors.append({"kind": "enemy_hero", "slot_valid": True, "actor": snapshot["actor"], "snapshot": snapshot})
        for snapshot in context["friend_heroes"]:
            descriptors.append({"kind": "friend_hero", "slot_valid": True, "actor": snapshot["actor"], "snapshot": snapshot})
        descriptors.append({"kind": "self", "slot_valid": True, "actor": query["actor"], "snapshot": query})
        for monster in context["monsters"]:
            descriptors.append(
                {
                    "kind": "monster",
                    "slot_valid": True,
                    "actor": monster["actor"],
                    "snapshot": None,
                    "fixed": monster,
                }
            )
        for soldier in context["soldiers"][10:]:
            descriptors.append(
                {"kind": "soldier", "slot_valid": soldier is not None, "actor": soldier, "snapshot": None}
            )
        target38 = context["target38"]
        target38_kind = (
            {"outer": "outer_tower", "high": "high_tower", "crystal": "crystal"}.get(
                target38["type"], "other"
            )
            if target38 is not None
            else "other"
        )
        descriptors.append(
            {
                "kind": target38_kind,
                "slot_valid": target38 is not None,
                "actor": target38["actor"] if target38 is not None else None,
                "snapshot": None,
                "fixed": target38,
            }
        )
        if len(descriptors) != 39:
            raise RuntimeError(f"target descriptor count drifted: {len(descriptors)}")
        return descriptors

    def _build_targets(self, context, query, hero_info):
        result = np.zeros((39, 48), dtype=np.float32)
        fields = schema_v1.TARGET_FIELD_MAP
        legal_matrix, _ = _raw_legal_matrix(hero_info)
        any_button_target_legal = np.any(legal_matrix, axis=0)
        chosen_skill_target_legal = np.any(legal_matrix[4:], axis=0)
        query_actor = query["actor"]
        query_position = query["side_position"]
        query_runtime = u.actor_runtime_id(query_actor)
        query_attack_range = _value(query_actor, "attack_range", "attackRange")
        official = context["official_hero"]
        official_actor = official["actor"]
        official_position = official["side_position"]
        official_alive = official["alive"]
        official_speed = _value(official_actor, "mov_spd", "movSpd")
        official_effective_range = _value(
            official_actor, "attack_range", "attackRange"
        )
        friend_ids = {u.actor_runtime_id(hero["actor"]) for hero in context["friend_heroes"]}
        query_attack_target = _target_runtime(query_actor)
        query_region = (
            u.semantic_region(query_position) if query_position is not None else None
        )
        command_target = context["collaboration"]["target_runtime"]
        type_names = (
            "none",
            "enemy_hero",
            "friend_hero",
            "self",
            "monster",
            "soldier",
            "outer_tower",
            "high_tower",
            "crystal",
            "other",
        )
        for index, descriptor in enumerate(self._target_descriptors(context, query)):
            vector = result[index]
            actor = descriptor["actor"]
            snapshot = descriptor.get("snapshot")
            fixed = descriptor.get("fixed")
            kind = descriptor["kind"]
            runtime_id = u.actor_runtime_id(actor) if actor else 0
            if snapshot is not None:
                position = snapshot["side_position"]
                alive = snapshot["alive"]
                visible = snapshot["visible"]
                current_valid = snapshot["current_valid"]
                last_seen = snapshot["last_seen_used"]
                entity_exists = snapshot["slot_exists"]
            else:
                position = _position(actor, self.camp) if actor else None
                if position is None and fixed is not None and fixed.get("anchor") is not None:
                    position = u.side_position(fixed["anchor"], self.camp)
                alive = bool(actor) and u.is_alive(actor)
                visible = bool(actor) and u.visible_to_camp(
                    actor, self.camp, is_friend=u.actor_camp(actor) == self.camp
                )
                current_valid = bool(actor) and visible
                last_seen = False
                entity_exists = bool(actor)
            if kind == "none":
                position = None
                alive = visible = current_valid = last_seen = entity_exists = False

            abilities = list(u.actor_get(actor, "abilities", default=[]) or []) if actor else []
            forbid_select = len(abilities) > 15 and bool(abilities[15])
            selectable = bool(descriptor["slot_valid"] and (kind == "none" or alive and current_valid and not forbid_select))
            _put(vector, fields, "slot_valid", descriptor["slot_valid"])
            _put(vector, fields, "entity_exists", entity_exists)
            _put(vector, fields, "alive", alive)
            _put(vector, fields, "visible_current", visible)
            _put(vector, fields, "position_valid", position is not None)
            _put(vector, fields, "selectable", selectable)
            _put(vector, fields, "current_observation_valid", current_valid)
            _put(vector, fields, "last_seen_used", last_seen)
            type_index = type_names.index(kind) if kind in type_names else 9
            _put_slice(vector, fields, "type_none", u.one_hot(type_index, 10))
            geometry = u.relative_geometry(query_position, position)
            _put_slice(vector, fields, "relative_dx_norm", geometry[:5])
            normalized_target = u.normalized_side_position(position)
            _put(vector, fields, "target_side_x_norm", normalized_target[0])
            _put(vector, fields, "target_side_z_norm", normalized_target[1])
            distance = _distance_or_inf(query_position, position)
            _put(vector, fields, "in_local_range", distance <= u.LOCAL_DISTANCE_SCALE)
            _put(
                vector,
                fields,
                "same_semantic_region",
                query_position is not None
                and position is not None
                and query_region == u.semantic_region(position),
            )
            _put(vector, fields, "topology_reachable", query_position is not None and position is not None)
            hp = _value(actor, "hp") if actor else 0
            max_hp = _value(actor, "max_hp", "maxHp") if actor else 0
            _put(vector, fields, "hp_ratio", u.ratio(hp, max_hp))
            _put(vector, fields, "hp_known", max_hp > 0 and (current_valid or last_seen))
            _put(vector, fields, "low_hp", max_hp > 0 and u.ratio(hp, max_hp) <= 0.3)
            attack_range = _value(actor, "attack_range", "attackRange") if actor else 0
            threatens_query = (
                alive
                and kind in ("enemy_hero", "monster", "soldier", "outer_tower", "high_tower", "crystal")
                and (_target_runtime(actor) == query_runtime or (attack_range > 0 and distance <= attack_range))
            )
            _put(vector, fields, "threat_to_query", threatens_query)
            _put(vector, fields, "attacking_friend", actor is not None and _target_runtime(actor) in friend_ids)
            marks = u.mark_items(actor) if actor else []
            marco_layers = [
                u.as_float(u.safe_get(item, "layer", default=0), 0)
                for item in marks
                if u.as_int(u.safe_get(item, "config_id", "configId", default=0), 0) == 13200
            ]
            diaochan_layers = [
                u.as_float(u.safe_get(item, "layer", default=0), 0)
                for item in marks
                if u.as_int(u.safe_get(item, "config_id", "configId", default=0), 0) in (14190, 14191)
            ]
            _put(vector, fields, "marco_mark_layer_norm", u.clip(max(marco_layers, default=0) / 9.0))
            _put(vector, fields, "diaochan_mark_layer_norm", u.clip(max(diaochan_layers, default=0) / 3.0))
            resource_value = 0.0
            if kind == "monster" and fixed is not None:
                resource_value = {122: 1.0, 44: 0.8, 49: 0.7, 59: 0.6}.get(fixed["config_id"], 0.4)
            elif kind == "crystal":
                resource_value = 1.0
            elif kind == "high_tower":
                resource_value = 0.8
            elif kind == "outer_tower":
                resource_value = 0.6
            _put(vector, fields, "resource_or_structure_value_norm", resource_value)
            _put(
                vector,
                fields,
                "any_button_target_legal",
                bool(any_button_target_legal[index]),
            )
            # Official which_button index 3 is Normal Attack; indices 0 and 1
            # are no-op actions.  Keep the target feature aligned with the
            # protocol row used by action sampling.
            _put(vector, fields, "common_attack_target_legal", bool(legal_matrix[3, index]))
            _put(
                vector,
                fields,
                "chosen_skill_target_legal",
                bool(chosen_skill_target_legal[index]),
            )
            _put(
                vector,
                fields,
                "is_query_current_attack_target",
                query_attack_target == runtime_id and runtime_id != 0,
            )
            structure = kind in ("outer_tower", "high_tower", "crystal")
            _put(
                vector,
                fields,
                "has_minion_cover_if_structure",
                bool(
                    structure
                    and fixed is not None
                    and self._tower_has_reliable_minion_cover(
                        context, fixed
                    )
                ),
            )
            _put(
                vector,
                fields,
                "attack_range_margin_norm",
                u.signed_clip((query_attack_range - distance) / 15000.0)
                if selectable and query_position is not None and position is not None
                else 0,
            )
            enemy_or_objective = kind in (
                "enemy_hero",
                "monster",
                "soldier",
                "outer_tower",
                "high_tower",
                "crystal",
            )
            if kind == "monster" and fixed is not None:
                enemy_or_objective = monster_set_target_allowed(
                    fixed["config_id"]
                )
            coll_legal = bool(
                enemy_or_objective
                and alive
                and visible
                and current_valid
                and runtime_id
                and official_alive
                and not forbid_select
            )
            _put(vector, fields, "collaboration_target_legal", coll_legal)
            _put(vector, fields, "official_hero_alive", official_alive)
            official_distance = _distance_or_inf(official_position, position)
            _put(vector, fields, "official_hero_distance_norm", u.clip(official_distance / u.GLOBAL_DISTANCE_SCALE))
            _put(
                vector,
                fields,
                "official_hero_in_effective_range",
                official_alive
                and official_effective_range > 0
                and official_distance <= official_effective_range,
            )
            _put(vector, fields, "is_official_current_command_target", runtime_id != 0 and runtime_id == command_target)
            _put(
                vector,
                fields,
                "official_to_target_eta_norm",
                u.clip((official_distance / official_speed) / 30.0)
                if official_position is not None and position is not None and official_speed > 0
                else 0,
            )
        return result

    def _build_hero_relations(self, context, query):
        result = np.zeros((6, 24), dtype=np.float32)
        fields = schema_v1.HERO_RELATION_FIELD_MAP
        query_actor = query["actor"]
        query_runtime = u.actor_runtime_id(query_actor)
        query_position = query["side_position"]
        query_hp = _hp_ratio(query_actor)
        query_level = _value(query_actor, "level")
        query_gold = _value(query_actor, "money_cnt", "moneyCnt")
        query_range = _value(query_actor, "attack_range", "attackRange")
        query_events = self._event_summary(query, context["frame_no"])
        for index, other in enumerate(context["heroes"]):
            vector = result[index]
            other_actor = other["actor"]
            other_runtime = u.actor_runtime_id(other_actor)
            other_position = other["side_position"]
            position_valid = query_position is not None and other_position is not None
            confidence = (
                1.0
                if other["visible"] or other["relation"] == "friend"
                else u.clip(1.0 - other["last_seen_age"] / LAST_SEEN_FRAMES)
                if other["last_seen_used"]
                else 0.0
            )
            relation_valid = bool(query["alive"] and other["slot_exists"] and position_valid and confidence > 0)
            _put(vector, fields, "relation_valid", relation_valid)
            _put(vector, fields, "other_observation_confidence", confidence)
            _put(vector, fields, "is_query_self", other_runtime != 0 and other_runtime == query_runtime)
            _put(vector, fields, "is_friend", other["relation"] == "friend")
            _put(vector, fields, "is_enemy", other["relation"] == "enemy")
            geometry = u.relative_geometry(query_position, other_position)
            _put_slice(vector, fields, "relative_dx_norm", geometry[:5])
            _put(vector, fields, "hp_ratio_diff", u.signed_clip(query_hp - _hp_ratio(other_actor)))
            _put(vector, fields, "level_diff_norm", u.signed_clip((query_level - _value(other_actor, "level")) / 15.0))
            _put(vector, fields, "gold_diff_norm", u.signed_clip((query_gold - _value(other_actor, "money_cnt", "moneyCnt")) / 15000.0))
            distance = _distance_or_inf(query_position, other_position)
            other_range = _value(other_actor, "attack_range", "attackRange")
            _put(vector, fields, "query_can_attack_other", relation_valid and query_range > 0 and distance <= query_range)
            _put(vector, fields, "other_can_attack_query", relation_valid and other_range > 0 and distance <= other_range)
            _put(vector, fields, "query_attacking_other", other_runtime != 0 and _target_runtime(query_actor) == other_runtime)
            _put(vector, fields, "other_attacking_query", query_runtime != 0 and _target_runtime(other_actor) == query_runtime)
            _put(
                vector,
                fields,
                "query_recently_damaged_other",
                any(event_type in ("hit_hero", "damage_dealt") and runtime == other_runtime for event_type, runtime in query_events["recent_other"]),
            )
            other_events = self._event_summary(other, context["frame_no"])
            _put(
                vector,
                fields,
                "other_recently_damaged_query",
                any(event_type in ("hit_hero", "damage_dealt") and runtime == query_runtime for event_type, runtime in other_events["recent_other"]),
            )
            _put(vector, fields, "other_moving_toward_query", _normalized_velocity_toward(other["motion"], other_position, query_position) > 0.05)
            _put(vector, fields, "query_moving_toward_other", _normalized_velocity_toward(query["motion"], query_position, other_position) > 0.05)
            _put(vector, fields, "query_facing_other", _is_facing(query_actor, query_position, other_position, self.camp))
            _put(vector, fields, "other_facing_query", _is_facing(other_actor, other_position, query_position, self.camp))
            _put(
                vector,
                fields,
                "same_semantic_region",
                position_valid and u.semantic_region(query_position) == u.semantic_region(other_position),
            )
        return result

    def _build_dynamic_lane_position(self, context):
        result = np.zeros((6, 72), dtype=np.float32)
        fields = schema_v1.DYNAMIC_LANE_POSITION_FIELD_MAP
        lane = context["lane_summary"]
        own_active = lane["own_active"]
        enemy_active = lane["enemy_active"]
        own_x = lane["own_boundary_x"]
        enemy_x = lane["enemy_boundary_x"]
        for index, hero in enumerate(context["heroes"]):
            position = hero["side_position"]
            valid = bool(
                hero["position_valid"]
                and position is not None
                and abs(position[1]) <= 6500.0
                and enemy_x > own_x
            )
            if not valid:
                continue
            vector = result[index]
            _put(vector, fields, "dynamic_lane_valid", 1)
            for name in ("outer", "high", "crystal"):
                _put(vector, fields, f"own_active_boundary_{name}", own_active["type"] == name)
                _put(vector, fields, f"enemy_active_boundary_{name}", enemy_active["type"] == name)
            _put(vector, fields, "own_active_boundary_side_x_norm", u.signed_clip(own_x / u.SIDE_X_SCALE))
            _put(vector, fields, "enemy_active_boundary_side_x_norm", u.signed_clip(enemy_x / u.SIDE_X_SCALE))
            progress = u.clip((position[0] - own_x) / (enemy_x - own_x))
            _put(vector, fields, "dynamic_lane_progress", progress)
            scaled = progress * 60.0
            bucket = min(60, int(math.floor(scaled + 1e-8)))
            residual = scaled - bucket if bucket < 60 else 0.0
            _put_slice(vector, fields, "dynamic_lane_bucket_0", u.one_hot(bucket, 61))
            _put(vector, fields, "bucket_residual", u.clip(residual))
        return result

    def _tyrant_status(self, context):
        tyrant = context["tyrant"]
        actor = tyrant["actor"]
        spawn = context["monster_spawn"][-1]
        timer_raw = u.safe_get(spawn, "spawn_timer", "spawnTimer", default=None) if spawn else None
        timer_valid = timer_raw is not None and math.isfinite(u.as_float(timer_raw, float("nan")))
        timer_seconds = _seconds_from_ms(timer_raw) if timer_valid else None
        tyrant_runtime = u.actor_runtime_id(actor) if actor else 0
        friend_ids = {u.actor_runtime_id(hero["actor"]) for hero in context["friend_heroes"]}
        enemy_ids = {u.actor_runtime_id(hero["actor"]) for hero in context["enemy_heroes"]}
        team_attacking = bool(actor) and _target_runtime(actor) in friend_ids
        enemy_attacking = bool(actor) and _target_runtime(actor) in enemy_ids
        for hero in context["heroes"]:
            if not tyrant_runtime:
                break
            events = self._event_summary(hero, context["frame_no"])["recent_other"]
            hit = any(other == tyrant_runtime and event_type.startswith("hit_") for event_type, other in events)
            if hero["relation"] == "friend":
                team_attacking = team_attacking or hit
            else:
                enemy_attacking = enemy_attacking or hit
        return {
            "alive": tyrant["alive"],
            "spawn_timer_valid": timer_valid,
            "spawn_seconds": timer_seconds,
            "spawn_within_30": timer_seconds is not None and timer_seconds <= 30.0,
            "team_attacking": team_attacking,
            "enemy_attacking": enemy_attacking,
            "position": _position(actor, self.camp) if actor else u.side_position(tyrant["anchor"], self.camp),
            "runtime": tyrant_runtime,
        }

    def _build_tyrant_local_position(self, context, query):
        vector = np.zeros((1, 54), dtype=np.float32)
        fields = schema_v1.TYRANT_LOCAL_POSITION_FIELD_MAP
        status = self._tyrant_status(context)
        active = status["alive"] or status["spawn_within_30"] or status["team_attacking"] or status["enemy_attacking"]
        if not active or query["side_position"] is None or status["position"] is None:
            return vector
        row = vector[0]
        _put(row, fields, "tyrant_fine_valid", 1)
        _put(row, fields, "tyrant_alive", status["alive"])
        _put(row, fields, "tyrant_spawn_within_30s", status["spawn_within_30"])
        _put(row, fields, "team_attacking_tyrant", status["team_attacking"])
        _put(row, fields, "enemy_attacking_tyrant", status["enemy_attacking"])
        dx = status["position"][0] - query["side_position"][0]
        dz = status["position"][1] - query["side_position"][1]
        distance = math.hypot(dx, dz)
        _put(row, fields, "relative_dx_norm", u.signed_clip(dx / 6000.0))
        _put(row, fields, "relative_dz_norm", u.signed_clip(dz / 6000.0))
        _put(row, fields, "distance_norm", u.clip(distance / 6000.0))
        _put_slice(row, fields, "local_x_bucket_0", u.one_hot(u.uniform_bucket(dx, -6000, 6000, 21), 21))
        _put_slice(row, fields, "local_z_bucket_0", u.one_hot(u.uniform_bucket(dz, -6000, 6000, 21), 21))
        if distance <= 3500:
            topology = 3
        elif distance > 7000:
            topology = 0
        elif query["side_position"][0] < status["position"][0]:
            topology = 1
        else:
            topology = 2
        _put_slice(row, fields, "topology_outside", u.one_hot(topology, 4))
        return vector

    @staticmethod
    def _skill_readiness(snapshot):
        learned = [slot for slot in u.skill_slots(snapshot["actor"]) if slot is not None]
        if not learned:
            return 0.0
        usable = sum(bool(u.safe_get(slot, "usable", default=False)) for slot in learned)
        return usable / len(learned)

    @staticmethod
    def _smite_ready(snapshot):
        slot = u.skill_slots(snapshot["actor"])[5]
        if slot is None:
            return False
        config_id = u.as_int(u.safe_get(slot, "config_id", "configId", default=0), 0)
        return config_id in (80104, 80116, 801162) and bool(
            u.safe_get(slot, "usable", default=False)
        )

    def _build_tyrant_team_context(self, context):
        vector = np.zeros((1, 24), dtype=np.float32)
        row = vector[0]
        fields = schema_v1.TYRANT_TEAM_CONTEXT_FIELD_MAP
        status = self._tyrant_status(context)
        target = status["position"]
        controlled = []
        for info in context["controlled_infos"]:
            runtime_id = u.as_int(u.safe_get(info, "runtime_id", default=0), 0)
            snapshot = context["hero_by_runtime"].get(runtime_id)
            if snapshot is not None:
                controlled.append(snapshot)
        team = controlled + [context["official_hero"]]
        if len(team) != 3:
            raise RuntimeError("tyrant team context requires two controlled heroes plus official hero")
        distances = []
        etas = []
        for index, hero in enumerate(team):
            position = hero["side_position"]
            distance = u.distance(position, target)
            speed = _value(hero["actor"], "mov_spd", "movSpd")
            eta = distance / speed if distance is not None and speed > 0 else None
            distances.append(distance)
            etas.append(eta)
            prefix = ("controlled_1", "controlled_2", "official")[index]
            _put(row, fields, f"{prefix}_distance_to_tyrant", u.clip(distance / u.GLOBAL_DISTANCE_SCALE) if distance is not None else 0)
            _put(row, fields, f"{prefix}_eta_to_tyrant", u.clip(eta / 30.0) if eta is not None else 0)
            _put(row, fields, f"{prefix}_near_tyrant", distance is not None and distance <= 10000)
            _put(row, fields, f"{prefix}_moving_toward_tyrant", _normalized_velocity_toward(hero["motion"], position, target) > 0.05)
            _put(
                row,
                fields,
                f"{prefix}_combat_ready",
                hero["alive"] and _hp_ratio(hero["actor"]) >= 0.3 and self._skill_readiness(hero) >= 0.25,
            )
        enemy_distances = []
        enemy_etas = []
        enemy_near_weight = 0.0
        enemy_smite_confidence = 0.0
        for enemy in context["enemy_heroes"]:
            confidence = 1.0 if enemy["visible"] else u.clip(1.0 - enemy["last_seen_age"] / LAST_SEEN_FRAMES) if enemy["last_seen_used"] else 0.0
            distance = u.distance(enemy["side_position"], target)
            speed = _value(enemy["actor"], "mov_spd", "movSpd")
            eta = distance / speed if distance is not None and speed > 0 else None
            if distance is not None:
                enemy_distances.append(distance)
                if distance <= 10000:
                    enemy_near_weight += confidence
            if eta is not None:
                enemy_etas.append(eta)
            if self._smite_ready(enemy):
                enemy_smite_confidence = max(enemy_smite_confidence, confidence)
        friend_near = sum(distance is not None and distance <= 10000 for distance in distances)
        _put(row, fields, "friend_count_near_tyrant_norm", friend_near / 3.0)
        _put(row, fields, "enemy_count_near_tyrant_norm", u.clip(enemy_near_weight / 3.0))
        valid_friend_etas = [eta for eta in etas if eta is not None]
        if valid_friend_etas and enemy_etas:
            _put(row, fields, "tyrant_eta_advantage", u.signed_clip((min(enemy_etas) - min(valid_friend_etas)) / 30.0))
        _put(row, fields, "tyrant_local_number_advantage", u.signed_clip((friend_near - enemy_near_weight) / 3.0))
        _put(row, fields, "friend_smite_ready", any(self._smite_ready(hero) for hero in team))
        _put(row, fields, "enemy_smite_ready_confidence", enemy_smite_confidence)
        reach_window = status["spawn_seconds"] if status["spawn_seconds"] is not None and status["spawn_seconds"] <= 30 else 30.0
        _put(row, fields, "all_three_can_reach_before_window", all(eta is not None and eta <= reach_window for eta in etas))
        collaboration = context["collaboration"]
        _put(
            row,
            fields,
            "official_command_available",
            context["official_hero"]["alive"] and collaboration["cooldown_ready"],
        )
        goal_is_tyrant = collaboration["target_runtime"] == status["runtime"] and status["runtime"] != 0
        if collaboration["command_position"] is not None and target is not None:
            goal_is_tyrant = goal_is_tyrant or _distance_or_inf(collaboration["command_position"], target) <= 5000
        _put(row, fields, "official_command_goal_is_tyrant", goal_is_tyrant)
        return vector

    def _lane_front_velocity(self, context, key, current_position):
        if current_position is None:
            return None
        for historical in reversed(list(self.lane_history)[:-1]):
            elapsed = context["frame_no"] - historical["frame"]
            previous = historical[key]
            if elapsed <= 0:
                continue
            if elapsed > LANE_VELOCITY_FRAMES:
                break
            if previous is not None:
                return (current_position[0] - previous[0]) / (elapsed * u.RAW_FRAME_SECONDS)
        return None

    def _organ_recent_damage(self, organ):
        history = self.organ_hp_history.get(organ["config_id"], ())
        if len(history) < 2:
            return 0.0
        _, old_hp, old_max = history[0]
        _, current_hp, current_max = history[-1]
        denominator = current_max if current_max > 0 else old_max
        return u.clip((old_hp - current_hp) / denominator) if denominator > 0 else 0.0

    def _build_lane_state(self, context, query):
        vector = np.zeros((1, 96), dtype=np.float32)
        row = vector[0]
        fields = schema_v1.LANE_STATE_FIELD_MAP
        lane = context["lane_summary"]
        friend, enemy = lane["friend"], lane["enemy"]

        friend_hp = sum(_value(actor, "hp") for actor in friend)
        enemy_hp = sum(_value(actor, "hp") for actor in enemy)
        friend_max_hp = sum(_value(actor, "max_hp", "maxHp") for actor in friend)
        enemy_max_hp = sum(_value(actor, "max_hp", "maxHp") for actor in enemy)
        friend_attack = sum(_value(actor, "phy_atk", "phyAtk") for actor in friend)
        enemy_attack = sum(_value(actor, "phy_atk", "phyAtk") for actor in enemy)
        friend_value = sum(_value(actor, "kill_income", "killIncome") for actor in friend)
        enemy_value = sum(_value(actor, "kill_income", "killIncome") for actor in enemy)
        composition = {
            "friend_alive_count_norm": len(friend) / 10.0,
            "enemy_alive_count_norm": len(enemy) / 10.0,
            "friend_melee_count_norm": lane["friend_type_counts"]["melee"] / 10.0,
            "enemy_melee_count_norm": lane["enemy_type_counts"]["melee"] / 10.0,
            "friend_ranged_count_norm": lane["friend_type_counts"]["ranged"] / 10.0,
            "enemy_ranged_count_norm": lane["enemy_type_counts"]["ranged"] / 10.0,
            "friend_cannon_count_norm": lane["friend_type_counts"]["cannon"] / 10.0,
            "enemy_cannon_count_norm": lane["enemy_type_counts"]["cannon"] / 10.0,
            "friend_super_count_norm": lane["friend_type_counts"]["super"] / 10.0,
            "enemy_super_count_norm": lane["enemy_type_counts"]["super"] / 10.0,
            "friend_total_hp_norm": u.ratio(friend_hp, friend_max_hp),
            "enemy_total_hp_norm": u.ratio(enemy_hp, enemy_max_hp),
            "friend_total_attack_norm": u.clip(friend_attack / 9000.0),
            "enemy_total_attack_norm": u.clip(enemy_attack / 9000.0),
            "friend_cannon_present": lane["friend_type_counts"]["cannon"] > 0,
            "enemy_cannon_present": lane["enemy_type_counts"]["cannon"] > 0,
            "friend_wave_present": bool(friend),
            "enemy_wave_present": bool(enemy),
            "friend_wave_value_norm": u.clip(friend_value / 2000.0),
            "enemy_wave_value_norm": u.clip(enemy_value / 2000.0),
        }
        for name, value in composition.items():
            _put(row, fields, name, value)

        friend_front = lane["friend_front"]
        enemy_front = lane["enemy_front"]
        friend_actor, friend_position = friend_front if friend_front else (None, None)
        enemy_actor, enemy_position = enemy_front if enemy_front else (None, None)
        own_x, enemy_x = lane["own_boundary_x"], lane["enemy_boundary_x"]
        friend_velocity = self._lane_front_velocity(context, "friend_front", friend_position)
        enemy_velocity = self._lane_front_velocity(context, "enemy_front", enemy_position)
        if friend_position is not None:
            friend_norm = u.normalized_side_position(friend_position)
            for name, value in {
                "own_front_valid": True,
                "own_front_side_x_norm": friend_norm[0],
                "own_front_side_z_norm": friend_norm[1],
                "own_front_dynamic_progress": lane["friend_progress"],
                "own_front_velocity_norm": u.signed_clip(friend_velocity / 7000.0) if friend_velocity is not None else 0,
                "own_front_to_enemy_boundary_norm": u.clip(abs(enemy_x - friend_position[0]) / u.GLOBAL_DISTANCE_SCALE),
                "own_front_to_own_boundary_norm": u.clip(abs(friend_position[0] - own_x) / u.GLOBAL_DISTANCE_SCALE),
                "own_front_hp_ratio": _hp_ratio(friend_actor) if friend_actor else 0,
                "own_front_is_cannon": bool(friend_actor) and self._soldier_type(friend_actor) == "cannon",
                "own_front_under_enemy_tower": _distance_or_inf(friend_position, lane["enemy_active"]["side_position"]) <= u.TOWER_DISTANCE_SCALE,
            }.items():
                _put(row, fields, name, value)
        if enemy_position is not None:
            enemy_norm = u.normalized_side_position(enemy_position)
            for name, value in {
                "enemy_front_observed": lane["enemy_front_observed"],
                "enemy_front_inferred": lane["enemy_front_inferred"],
                "enemy_front_confidence": lane["enemy_front_confidence"],
                "enemy_front_side_x_norm": enemy_norm[0],
                "enemy_front_side_z_norm": enemy_norm[1],
                "enemy_front_dynamic_progress": lane["enemy_progress"],
                "enemy_front_velocity_norm": u.signed_clip(enemy_velocity / 7000.0) if enemy_velocity is not None else 0,
                "enemy_front_to_own_boundary_norm": u.clip(abs(enemy_position[0] - own_x) / u.GLOBAL_DISTANCE_SCALE),
                "enemy_front_to_enemy_boundary_norm": u.clip(abs(enemy_x - enemy_position[0]) / u.GLOBAL_DISTANCE_SCALE),
                "enemy_front_hp_ratio": _hp_ratio(enemy_actor) if enemy_actor else 0,
                "enemy_front_is_cannon": bool(enemy_actor) and self._soldier_type(enemy_actor) == "cannon",
                "enemy_front_under_own_tower": _distance_or_inf(enemy_position, lane["own_active"]["side_position"]) <= u.TOWER_DISTANCE_SCALE,
            }.items():
                _put(row, fields, name, value)
        if friend_position is not None and enemy_position is not None:
            gap = abs(enemy_position[0] - friend_position[0])
            _put(row, fields, "front_gap_norm", u.clip(gap / u.GLOBAL_DISTANCE_SCALE))
            _put(row, fields, "lane_front_advantage", u.signed_clip(lane["friend_progress"] - lane["enemy_progress"]))

        clash = lane["clash_position"]
        near_radius = 8000.0
        friend_soldiers_near = sum(clash is not None and _distance_or_inf(_position(actor, self.camp), clash) <= near_radius for actor in friend)
        enemy_soldiers_near = sum(clash is not None and _distance_or_inf(_position(actor, self.camp), clash) <= near_radius for actor in enemy)
        friend_heroes_near = sum(clash is not None and hero["alive"] and _distance_or_inf(hero["side_position"], clash) <= near_radius for hero in context["friend_heroes"])
        enemy_heroes_near = 0.0
        for hero in context["enemy_heroes"]:
            confidence = 1.0 if hero["visible"] else u.clip(1.0 - hero["last_seen_age"] / LAST_SEEN_FRAMES) if hero["last_seen_used"] else 0.0
            if clash is not None and hero["alive"] and _distance_or_inf(hero["side_position"], clash) <= near_radius:
                enemy_heroes_near += confidence
        own_outer, own_high, own_crystal, enemy_outer, enemy_high, enemy_crystal = context["organs"]
        friend_soldier_ids = {u.actor_runtime_id(actor) for actor in friend}
        enemy_soldier_ids = {u.actor_runtime_id(actor) for actor in enemy}
        enemy_tower_targeting = any(_target_runtime(organ["actor"]) in friend_soldier_ids for organ in context["organs"][3:] if organ["actor"])
        own_tower_targeting = any(_target_runtime(organ["actor"]) in enemy_soldier_ids for organ in context["organs"][:3] if organ["actor"])
        pressure = {
            "friend_soldiers_near_clash_norm": friend_soldiers_near / 10.0,
            "enemy_soldiers_near_clash_norm": enemy_soldiers_near / 10.0,
            "friend_heroes_near_clash_norm": friend_heroes_near / 3.0,
            "enemy_heroes_near_clash_norm": u.clip(enemy_heroes_near / 3.0),
            "waves_engaged": friend_position is not None and enemy_position is not None and _distance_or_inf(friend_position, enemy_position) <= 5000,
            "clash_position_valid": clash is not None,
            "clash_side_x_norm": u.normalized_side_position(clash)[0],
            "local_number_advantage": u.signed_clip((friend_soldiers_near + friend_heroes_near - enemy_soldiers_near - enemy_heroes_near) / 3.0),
            "friend_wave_under_enemy_outer": friend_position is not None and _distance_or_inf(friend_position, enemy_outer["side_position"]) <= u.TOWER_DISTANCE_SCALE,
            "friend_wave_under_enemy_high": friend_position is not None and _distance_or_inf(friend_position, enemy_high["side_position"]) <= u.TOWER_DISTANCE_SCALE,
            "friend_wave_near_enemy_crystal": friend_position is not None and _distance_or_inf(friend_position, enemy_crystal["side_position"]) <= u.TOWER_DISTANCE_SCALE,
            "enemy_wave_under_own_outer": enemy_position is not None and _distance_or_inf(enemy_position, own_outer["side_position"]) <= u.TOWER_DISTANCE_SCALE,
            "enemy_wave_under_own_high": enemy_position is not None and _distance_or_inf(enemy_position, own_high["side_position"]) <= u.TOWER_DISTANCE_SCALE,
            "enemy_wave_near_own_crystal": enemy_position is not None and _distance_or_inf(enemy_position, own_crystal["side_position"]) <= u.TOWER_DISTANCE_SCALE,
            "enemy_tower_targeting_friend_soldier": enemy_tower_targeting,
            "own_tower_targeting_enemy_soldier": own_tower_targeting,
            "friend_wave_has_tower_aggro": enemy_tower_targeting,
            "enemy_wave_has_tower_aggro": own_tower_targeting,
            "query_has_minion_cover": any(_distance_or_inf(query["side_position"], _position(actor, self.camp)) <= 8000 for actor in friend) if query["side_position"] is not None else False,
            "enemy_has_minion_cover": enemy_heroes_near > 0 and bool(enemy),
        }
        for name, value in pressure.items():
            _put(row, fields, name, value)

        phase = {
            "both_outer_alive": own_outer["alive"] and enemy_outer["alive"],
            "own_outer_down": not own_outer["alive"],
            "enemy_outer_down": not enemy_outer["alive"],
            "both_outer_down": not own_outer["alive"] and not enemy_outer["alive"],
            "own_high_exposed": not own_outer["alive"] and own_high["alive"],
            "enemy_high_exposed": not enemy_outer["alive"] and enemy_high["alive"],
            "own_crystal_exposed": not own_high["alive"],
            "enemy_crystal_exposed": not enemy_high["alive"],
            "can_pressure_enemy_outer": enemy_outer["alive"] and pressure["friend_wave_under_enemy_outer"],
            "can_pressure_enemy_high": not enemy_outer["alive"] and enemy_high["alive"] and pressure["friend_wave_under_enemy_high"],
            "can_pressure_enemy_crystal": not enemy_high["alive"] and pressure["friend_wave_near_enemy_crystal"],
            "enemy_pressure_own_outer": own_outer["alive"] and pressure["enemy_wave_under_own_outer"],
            "enemy_pressure_own_high": not own_outer["alive"] and own_high["alive"] and pressure["enemy_wave_under_own_high"],
            "enemy_pressure_own_crystal": not own_high["alive"] and pressure["enemy_wave_near_own_crystal"],
        }
        phase["wave_supports_push"] = any(phase[name] for name in ("can_pressure_enemy_outer", "can_pressure_enemy_high", "can_pressure_enemy_crystal"))
        phase["wave_requires_defense"] = any(phase[name] for name in ("enemy_pressure_own_outer", "enemy_pressure_own_high", "enemy_pressure_own_crystal"))
        for name, value in phase.items():
            _put(row, fields, name, value)

        if len(self.lane_history) >= 2:
            old = self.lane_history[0]
            current = self.lane_history[-1]
            old_friend = old["friend_front"]
            old_enemy = old["enemy_front"]
            if friend_position is not None and old_friend is not None:
                _put(row, fields, "own_front_delta_norm", u.signed_clip((friend_position[0] - old_friend[0]) / u.GLOBAL_DISTANCE_SCALE))
            if enemy_position is not None and old_enemy is not None:
                _put(row, fields, "enemy_front_delta_norm", u.signed_clip((enemy_position[0] - old_enemy[0]) / u.GLOBAL_DISTANCE_SCALE))
            if friend_position is not None and enemy_position is not None and old_friend is not None and old_enemy is not None:
                old_gap = abs(old_enemy[0] - old_friend[0])
                new_gap = abs(enemy_position[0] - friend_position[0])
                _put(row, fields, "front_gap_delta_norm", u.signed_clip((new_gap - old_gap) / u.GLOBAL_DISTANCE_SCALE))
            _put(row, fields, "friend_wave_hp_delta_norm", u.signed_clip((current["friend_hp"] - old["friend_hp"]) / max(1.0, friend_max_hp)))
            _put(row, fields, "enemy_wave_hp_delta_norm", u.signed_clip((current["enemy_hp"] - old["enemy_hp"]) / max(1.0, enemy_max_hp)))
            _put(row, fields, "friend_clear_speed_norm", u.clip((old["enemy_hp"] - current["enemy_hp"]) / max(1.0, enemy_max_hp)))
            _put(row, fields, "enemy_clear_speed_norm", u.clip((old["friend_hp"] - current["friend_hp"]) / max(1.0, friend_max_hp)))
            _put(row, fields, "friend_wave_recently_spawned", current["friend_count"] > old["friend_count"])
            _put(row, fields, "enemy_wave_recently_spawned", current["enemy_count"] > old["enemy_count"])
            _put(row, fields, "lane_trend_valid", True)
        for name, organ in zip(
            (
                "enemy_outer_recent_damage_norm",
                "enemy_high_recent_damage_norm",
                "enemy_crystal_recent_damage_norm",
                "own_outer_recent_damage_norm",
                "own_high_recent_damage_norm",
                "own_crystal_recent_damage_norm",
            ),
            (enemy_outer, enemy_high, enemy_crystal, own_outer, own_high, own_crystal),
        ):
            _put(row, fields, name, self._organ_recent_damage(organ))
        return vector

    def _resource_summary(self, context):
        raw_by_config = {u.actor_config_id(cake): cake for cake in context["cakes_raw"]}
        configs = u.fixed_resource_config_ids(self.camp)
        associated = (None, context["organs"][0], context["organs"][1], context["organs"][3], context["organs"][4])
        resources = []
        for index, (config_id, organ) in enumerate(zip(configs, associated)):
            actor = raw_by_config.get(config_id)
            available = actor is not None
            disabled = organ is not None and not organ["alive"]
            history = self.resource_history.get(config_id)
            if history is None:
                history = {
                    "available": available,
                    "last_disappear": None,
                    "last_appear": context["frame_no"] if available else None,
                }
                self.resource_history[config_id] = history
            else:
                if history["available"] and not available:
                    history["last_disappear"] = context["frame_no"]
                elif not history["available"] and available:
                    history["last_appear"] = context["frame_no"]
                    history["last_disappear"] = None
                history["available"] = available
            if disabled:
                timer_valid = False
                remaining = 0
            elif available:
                timer_valid = True
                remaining = 0
            elif history["last_disappear"] is not None:
                age = context["frame_no"] - history["last_disappear"]
                timer_valid = 0 <= age <= 912
                remaining = max(0, 912 - age)
            else:
                timer_valid = False
                remaining = 0
            resources.append(
                {
                    "index": index,
                    "config_id": config_id,
                    "actor": actor,
                    "anchor_raw": u.CAKE_ANCHORS[config_id],
                    "anchor": u.side_position(u.CAKE_ANCHORS[config_id], self.camp),
                    "available": available and not disabled,
                    "visible": available,
                    "disabled": disabled,
                    "associated_organ": organ,
                    "timer_valid": timer_valid,
                    "remaining_frames": remaining,
                }
            )
        return resources

    def _build_resource_anchors(self, context, query):
        result = np.zeros((5, 32), dtype=np.float32)
        fields = schema_v1.RESOURCE_ANCHOR_FIELD_MAP
        controlled = []
        for info in context["controlled_infos"]:
            runtime_id = u.as_int(u.safe_get(info, "runtime_id", default=0), 0)
            snapshot = context["hero_by_runtime"].get(runtime_id)
            if snapshot is not None:
                controlled.append(snapshot)
        team = controlled + [context["official_hero"]]
        if len(team) != 3:
            raise RuntimeError("resource anchors require a complete three-hero friendly team")
        friend_buff_ids = {
            u.as_int(u.safe_get(item, "config_id", "configId", default=0), 0)
            for hero in context["friend_heroes"]
            for item in u.buff_items(hero["actor"])
        }
        enemy_buff_ids = {
            u.as_int(u.safe_get(item, "config_id", "configId", default=0), 0)
            for hero in context["enemy_heroes"]
            for item in u.buff_items(hero["actor"])
        }
        for index, resource in enumerate(context["resources"]):
            vector = result[index]
            anchor = resource["anchor"]
            normalized = u.normalized_side_position(anchor)
            is_speed = index == 0
            is_own = index in (1, 2)
            is_enemy = index in (3, 4)
            for name, value in {
                "slot_exists": True,
                "visible_current": resource["visible"],
                "available": resource["available"],
                "permanently_disabled": resource["disabled"],
                "is_speed_buff": is_speed,
                "is_health_pack": not is_speed,
                "is_own_side": is_own,
                "is_enemy_side": is_enemy,
                "is_neutral": is_speed,
                "anchor_side_x_norm": normalized[0],
                "anchor_side_z_norm": normalized[1],
                "respawn_timer_valid": resource["timer_valid"],
                "respawn_timer_norm": u.clip(resource["remaining_frames"] / 912.0),
            }.items():
                _put(vector, fields, name, value)

            friend_distances = []
            friend_etas = []
            for team_index, hero in enumerate(team):
                distance = u.distance(hero["side_position"], anchor)
                speed = _value(hero["actor"], "mov_spd", "movSpd")
                eta = distance / speed if distance is not None and speed > 0 else None
                friend_distances.append(distance)
                friend_etas.append(eta)
                field_name = (
                    "controlled_hero_1_distance_norm",
                    "controlled_hero_2_distance_norm",
                    "official_hero_distance_norm",
                )[team_index]
                _put(vector, fields, field_name, u.clip(distance / u.GLOBAL_DISTANCE_SCALE) if distance is not None else 0)
            enemy_distances = []
            enemy_etas = []
            enemy_near = 0.0
            low_enemy = False
            for enemy in context["enemy_heroes"]:
                confidence = 1.0 if enemy["visible"] else u.clip(1.0 - enemy["last_seen_age"] / LAST_SEEN_FRAMES) if enemy["last_seen_used"] else 0.0
                distance = u.distance(enemy["side_position"], anchor)
                speed = _value(enemy["actor"], "mov_spd", "movSpd")
                if distance is not None and confidence > 0:
                    enemy_distances.append((distance, confidence))
                    if distance <= 10000:
                        enemy_near += confidence
                        low_enemy = low_enemy or (_hp_ratio(enemy["actor"]) <= 0.3 and confidence > 0.5)
                    if speed > 0:
                        enemy_etas.append((distance / speed, confidence))
            valid_friend_distances = [value for value in friend_distances if value is not None]
            valid_friend_etas = [value for value in friend_etas if value is not None]
            friend_near = sum(value is not None and value <= 10000 for value in friend_distances)
            low_friend = any(
                distance is not None and distance <= 10000 and _hp_ratio(hero["actor"]) <= 0.3
                for distance, hero in zip(friend_distances, team)
            )
            min_friend_distance = min(valid_friend_distances, default=u.GLOBAL_DISTANCE_SCALE)
            min_enemy_distance = min((distance for distance, _ in enemy_distances), default=u.GLOBAL_DISTANCE_SCALE)
            min_friend_eta = min(valid_friend_etas, default=None)
            min_enemy_eta = min((eta for eta, _ in enemy_etas), default=None)
            eta_valid = min_friend_eta is not None and min_enemy_eta is not None
            effect_id = 90201 if is_speed else 90014
            for name, value in {
                "nearest_friend_distance_norm": u.clip(min_friend_distance / u.GLOBAL_DISTANCE_SCALE),
                "nearest_enemy_distance_norm": u.clip(min_enemy_distance / u.GLOBAL_DISTANCE_SCALE),
                "friend_count_near_norm": friend_near / 3.0,
                "enemy_count_near_norm": u.clip(enemy_near / 3.0),
                "low_hp_friend_near": low_friend,
                "low_hp_enemy_near": low_enemy,
                "effect_held_by_friend": effect_id in friend_buff_ids,
                "effect_held_by_enemy": effect_id in enemy_buff_ids,
                "friend_eta_min_norm": u.clip(min_friend_eta / 30.0) if min_friend_eta is not None else 1.0,
                "enemy_eta_min_norm": u.clip(min_enemy_eta / 30.0) if min_enemy_eta is not None else 1.0,
                "eta_advantage": u.signed_clip((min_enemy_eta - min_friend_eta) / 30.0) if eta_valid else 0,
                "contested": friend_near > 0 and enemy_near > 0,
                "safe_access": friend_near >= enemy_near and bool(valid_friend_distances),
                "eta_comparison_valid": eta_valid,
                "associated_tower_alive": bool(resource["associated_organ"] and resource["associated_organ"]["alive"]),
            }.items():
                _put(vector, fields, name, value)
            query_distance = u.distance(query["side_position"], anchor)
            query_speed = _value(query["actor"], "mov_spd", "movSpd")
            _put(
                vector,
                fields,
                "query_eta_norm",
                u.clip((query_distance / query_speed) / 30.0)
                if query_distance is not None and query_speed > 0
                else 0,
            )
        return result

    def _runtime_to_target_slot(self, context, runtime_id):
        if not runtime_id:
            return None
        for offset, hero in enumerate(context["enemy_heroes"], 1):
            if u.actor_runtime_id(hero["actor"]) == runtime_id:
                return offset
        for offset, hero in enumerate(context["friend_heroes"], 4):
            if u.actor_runtime_id(hero["actor"]) == runtime_id:
                return offset
        for offset, monster in enumerate(context["monsters"], 8):
            if monster["actor"] and u.actor_runtime_id(monster["actor"]) == runtime_id:
                return offset
        for offset, soldier in enumerate(context["soldiers"][10:], 28):
            if soldier and u.actor_runtime_id(soldier) == runtime_id:
                return offset
        if context["target38"] and u.actor_runtime_id(context["target38"]["actor"]) == runtime_id:
            return 38
        return None

    def _current_collaboration_opportunities(self, context):
        enemy_heroes = [
            hero
            for hero in context["enemy_heroes"]
            if hero["alive"] and hero["visible"] and hero["current_valid"]
        ]
        soldiers = [
            soldier
            for soldier in context["soldiers"][10:]
            if soldier and u.is_alive(soldier) and u.visible_to_camp(soldier, self.camp)
        ]
        structures = [organ for organ in context["organs"][3:] if organ["alive"]]
        objectives = [
            monster
            for monster in context["monsters"]
            if monster["alive"] and monster["visible"] and monster["config_id"] in (44, 49, 59, 122)
        ]
        return {
            "enemy_heroes": enemy_heroes,
            "soldiers": soldiers,
            "structures": structures,
            "objectives": objectives,
        }

    def _build_collaboration_state(self, context):
        vector = np.zeros((1, 64), dtype=np.float32)
        row = vector[0]
        fields = schema_v1.COLLABORATION_STATE_FIELD_MAP
        collaboration = context["collaboration"]
        official = context["official_hero"]
        official_actor = official["actor"]
        official_runtime = u.actor_runtime_id(official_actor)
        controlled_runtime_ids = [
            u.as_int(u.safe_get(info, "runtime_id", default=0), 0)
            for info in context["controlled_infos"]
        ]
        type_index = {"none": 0, "set_target": 1, "set_position": 2, "unset": 3}.get(
            collaboration["type"], 0
        )
        _put_slice(row, fields, "current_type_none", u.one_hot(type_index, 4))
        target_snapshot = collaboration["target_snapshot"]
        target_actor = collaboration["target_actor"]
        target_runtime = collaboration["target_runtime"]
        target_slot = self._runtime_to_target_slot(context, target_runtime)
        target_alive = (
            target_snapshot["alive"]
            if target_snapshot is not None
            else bool(target_actor) and u.is_alive(target_actor)
        )
        target_visible = (
            target_snapshot["visible"]
            if target_snapshot is not None
            else bool(target_actor)
            and u.visible_to_camp(
                target_actor,
                self.camp,
                is_friend=u.actor_camp(target_actor) == self.camp,
            )
        )
        target_position = collaboration["target_position"]
        current_legal = bool(
            collaboration["active"]
            and (
                collaboration["type"] == "set_position"
                and collaboration["command_position"] is not None
                or collaboration["type"] == "set_target"
                and target_slot is not None
                and target_alive
                and target_visible
            )
        )
        for name, value in {
            "current_command_active": collaboration["active"],
            "sender_is_controlled_hero_1": collaboration["sender"] == controlled_runtime_ids[0],
            "sender_is_controlled_hero_2": collaboration["sender"] == controlled_runtime_ids[1],
            "sender_unknown": collaboration["active"] and collaboration["sender"] not in controlled_runtime_ids,
            "receiver_is_official_hero": collaboration["receiver"] == official_runtime,
            "receiver_alive": official["alive"],
            "target_runtime_maps_to_slot": target_slot is not None,
            "target_alive": target_alive,
            "target_visible": target_visible,
            "target_position_valid": target_position is not None,
            "target_is_last_seen": bool(target_snapshot and target_snapshot["last_seen_used"]),
            "command_end_frame_valid": collaboration["end_frame_valid"],
            "remaining_duration_norm": u.clip(collaboration["remaining_frames"] / max(1, collaboration["max_duration"])),
            "micro_cd_norm": u.clip(collaboration["micro_cd"] / 1200.0),
            "cooldown_ready": collaboration["cooldown_ready"],
            "max_duration_norm": u.clip(collaboration["max_duration"] / 1200.0),
            "command_age_norm": u.clip(collaboration["age_frames"] / max(1, collaboration["max_duration"])),
            "target_changed_or_invalid": collaboration["active"] and not current_legal,
            "command_persists_through_receiver_death": collaboration["active"] and not official["alive"],
            "current_state_valid": collaboration["state_valid"],
        }.items():
            _put(row, fields, name, value)

        official_position = official["side_position"]
        goal = target_position if collaboration["type"] == "set_target" else collaboration["command_position"]
        target_distance = u.distance(official_position, target_position)
        position_distance = u.distance(official_position, collaboration["command_position"])
        toward_goal = _normalized_velocity_toward(official["motion"], official_position, goal)
        official_events = self._event_summary(official, context["frame_no"])
        recent_hit_target = any(
            runtime == target_runtime and event_type.startswith("hit_")
            for event_type, runtime in official_events["recent_other"]
        )
        reached_range = False
        if collaboration["type"] == "set_target" and target_distance is not None:
            effective_range = _value(official_actor, "attack_range", "attackRange")
            reached_range = effective_range > 0 and target_distance <= effective_range
        elif collaboration["type"] == "set_position" and position_distance is not None:
            reached_range = position_distance <= 1200
        real_matches = (
            collaboration["type"] == "set_target"
            and target_runtime != 0
            and _target_runtime(official_actor) == target_runtime
            or collaboration["type"] == "set_position"
            and toward_goal > 0.05
        )
        for name, value in {
            "official_slot_valid": official["slot_exists"],
            "official_alive": official["alive"],
            "official_respawning": not official["alive"] and _value(official_actor, "revive_time", "reviveTime") > 0,
            "official_position_valid": official_position is not None,
            "official_to_target_distance_norm": u.clip(target_distance / u.GLOBAL_DISTANCE_SCALE) if target_distance is not None else 0,
            "official_to_position_distance_norm": u.clip(position_distance / u.GLOBAL_DISTANCE_SCALE) if position_distance is not None else 0,
            "official_approaching_command_goal": toward_goal > 0.05,
            "official_moving_away_from_goal": toward_goal < -0.05,
            "official_attacking_command_target": target_runtime != 0 and _target_runtime(official_actor) == target_runtime,
            "official_recent_hit_command_target": recent_hit_target,
            "official_recent_damage_to_target_norm": u.log_norm(
                sum(
                    event["amount"]
                    for event in self.recent_events.get(("friend", u.actor_config_id(official_actor)), ())
                    if event["other_runtime"] == target_runtime and event["type"] == "damage_dealt"
                ),
                20000,
            ),
            "official_recent_kill_on_command_target": False,
            "official_recent_assist_on_command_target": False,
            "official_real_cmd_matches_command": real_matches,
            "official_reached_effective_range": reached_range,
            "official_response_valid": collaboration["active"] and official_position is not None,
        }.items():
            _put(row, fields, name, value)

        last_history = collaboration["last_history"]
        frames_since_last = (
            context["frame_no"] - collaboration["last_send_frame"]
            if collaboration["last_send_frame"] > 0
            else 0
        )
        repeated = False
        if len(self.command_history) >= 2:
            repeated = (
                self.command_history[-1]["target_runtime"] != 0
                and self.command_history[-1]["target_runtime"]
                == self.command_history[-2]["target_runtime"]
            )
        for name, value in {
            "last_send_valid": collaboration["last_send_frame"] > 0,
            "frames_since_last_send_norm": u.clip(frames_since_last / COMMAND_HISTORY_FRAMES),
            "last_send_was_valid": bool(last_history and last_history["valid"]),
            "last_send_was_invalid": bool(last_history and not last_history["valid"]),
            "last_command_overwritten": bool(last_history and last_history["overwritten"]),
            "last_command_unset": bool(last_history and last_history["unset"]),
            "last_target_died": bool(last_history and last_history["target_runtime"] and not target_alive),
            "repeated_same_target": repeated,
        }.items():
            _put(row, fields, name, value)

        opportunities = self._current_collaboration_opportunities(context)
        valid_target = any(opportunities.values())
        friend_positions = [
            hero["side_position"]
            for hero in context["friend_heroes"]
            if hero["alive"] and hero["side_position"] is not None
        ]
        pair_distances = [
            _distance_or_inf(first, second)
            for first_index, first in enumerate(friend_positions)
            for second in friend_positions[first_index + 1 :]
        ]
        grouped = len(friend_positions) >= 2 and max(pair_distances, default=0) <= 15000
        official_team_distances = [
            _distance_or_inf(official_position, position)
            for position in friend_positions
            if position != official_position
        ]
        far = bool(official_team_distances) and min(official_team_distances) >= 25000
        danger = official["alive"] and (
            _hp_ratio(official_actor) <= 0.3
            or sum(
                enemy["visible"]
                and _distance_or_inf(official_position, enemy["side_position"]) <= 10000
                for enemy in context["enemy_heroes"]
            )
            >= 2
        )
        can_position = official["alive"] and collaboration["cooldown_ready"]
        readiness = {
            "can_send_set_target": official["alive"] and collaboration["cooldown_ready"] and valid_target,
            "can_send_set_position": can_position,
            "can_send_unset": collaboration["active"],
            "current_command_still_legal": current_legal,
            "current_command_goal_completed": collaboration["active"] and (not target_alive if collaboration["type"] == "set_target" else reached_range),
            "official_far_from_team": far,
            "official_grouped_with_team": grouped,
            "official_in_danger": danger,
            "enemy_hero_target_available": bool(opportunities["enemy_heroes"]),
            "structure_target_available": bool(opportunities["structures"]),
            "soldier_target_available": bool(opportunities["soldiers"]),
            "objective_target_available": bool(opportunities["objectives"]),
            "gather_position_available": len(friend_positions) >= 2,
            "defend_position_available": any(organ["alive"] for organ in context["organs"][:3]),
            "push_position_available": bool(context["lane_summary"]["friend"]),
            "objective_position_available": self._tyrant_status(context)["alive"] or any(resource["available"] for resource in context["resources"]),
        }
        for name, value in readiness.items():
            _put(row, fields, name, value)
        context["collaboration_readiness"] = readiness
        return vector

    @staticmethod
    def _hero_confidence(hero):
        if hero["relation"] == "friend" or hero["visible"]:
            return 1.0
        if hero["last_seen_used"]:
            return u.clip(1.0 - hero["last_seen_age"] / LAST_SEEN_FRAMES)
        return 0.0

    def _team_grouping(self, heroes):
        positioned = [hero["side_position"] for hero in heroes if hero["alive"] and hero["side_position"] is not None]
        if len(positioned) < 2:
            return False, False
        distances = [
            _distance_or_inf(first, second)
            for index, first in enumerate(positioned)
            for second in positioned[index + 1 :]
        ]
        maximum = max(distances, default=0)
        return maximum <= 15000, maximum >= 25000

    def _hero_attribute_power(self, hero):
        actor = hero["actor"]
        values = (
            u.min_max(_value(actor, "phy_atk", "phyAtk"), 100, 900),
            u.min_max(_value(actor, "mgc_atk", "mgcAtk"), 0, 1000),
            u.min_max(_value(actor, "phy_def", "phyDef"), 0, 1200),
            u.min_max(_value(actor, "mgc_def", "mgcDef"), 0, 700),
            u.min_max(_value(actor, "mov_spd", "movSpd"), 1250, 7000),
            u.min_max(_value(actor, "atk_spd", "atkSpd"), 0, 10000),
        )
        return sum(values) / len(values)

    @staticmethod
    def _hero_buff_ids(hero):
        return {
            u.as_int(u.safe_get(item, "config_id", "configId", default=0), 0)
            for item in u.buff_items(hero["actor"])
        }

    def _build_global_state(self, context):
        vector = np.zeros((1, 128), dtype=np.float32)
        row = vector[0]
        fields = schema_v1.GLOBAL_STATE_FIELD_MAP
        game_seconds = max(0.0, context["frame_no"] * u.RAW_FRAME_SECONDS)
        stage_index = min(3, int(game_seconds // 180.0))
        stage_progress = (game_seconds - stage_index * 180.0) / 180.0
        tyrant = self._tyrant_status(context)
        speed_resource = context["resources"][0]
        health_resources = context["resources"][1:]
        speed_countdown = speed_resource["remaining_frames"] * u.RAW_FRAME_SECONDS
        health_valid = [resource for resource in health_resources if resource["timer_valid"]]
        health_countdown = min(
            (resource["remaining_frames"] * u.RAW_FRAME_SECONDS for resource in health_valid),
            default=0.0,
        )
        clock = {
            "game_time_norm": u.clip(game_seconds / 900.0),
            "early_game": stage_index == 0,
            "mid_game": stage_index == 1,
            "late_game": stage_index == 2,
            "super_late_game": stage_index == 3,
            "stage_progress": u.clip(stage_progress),
            "match_clock_valid": context["frame_no"] >= 0,
            "timeout_pressure": u.clip((game_seconds - 840.0) / 60.0),
            "wave_cycle_sin": math.sin(2.0 * math.pi * game_seconds / 30.0),
            "wave_cycle_cos": math.cos(2.0 * math.pi * game_seconds / 30.0),
            "tyrant_countdown_norm": u.clip((tyrant["spawn_seconds"] or 0.0) / 90.0),
            "tyrant_countdown_valid": tyrant["spawn_timer_valid"],
            "speed_buff_countdown_norm": u.clip(speed_countdown / 60.192),
            "speed_buff_countdown_valid": speed_resource["timer_valid"],
            "nearest_health_pack_countdown_norm": u.clip(health_countdown / 60.192),
            "nearest_health_pack_countdown_valid": bool(health_valid),
        }
        for name, value in clock.items():
            _put(row, fields, name, value)

        friend = context["friend_heroes"]
        enemy = context["enemy_heroes"]
        friend_weights = [1.0] * 3
        enemy_weights = [self._hero_confidence(hero) for hero in enemy]

        def weighted_sum(heroes, weights, getter):
            return sum(getter(hero) * weight for hero, weight in zip(heroes, weights))

        def weighted_mean(heroes, weights, getter):
            denominator = sum(weights)
            return weighted_sum(heroes, weights, getter) / denominator if denominator > 0 else 0.0

        friend_gold = weighted_sum(friend, friend_weights, lambda hero: _value(hero["actor"], "money_cnt", "moneyCnt"))
        enemy_gold = weighted_sum(enemy, enemy_weights, lambda hero: _value(hero["actor"], "money_cnt", "moneyCnt"))
        controlled_ids = context["controlled_runtime_ids"]
        controlled_gold = sum(
            _value(hero["actor"], "money_cnt", "moneyCnt")
            for hero in friend
            if u.actor_runtime_id(hero["actor"]) in controlled_ids
        )
        official_gold = _value(context["official_hero"]["actor"], "money_cnt", "moneyCnt")
        friend_level = sum(_value(hero["actor"], "level") for hero in friend)
        enemy_level = weighted_sum(enemy, enemy_weights, lambda hero: _value(hero["actor"], "level"))
        friend_hp_power = sum(_hp_ratio(hero["actor"]) for hero in friend) / 3.0
        enemy_hp_power = weighted_mean(enemy, enemy_weights, lambda hero: _hp_ratio(hero["actor"]))
        friend_attr = sum(self._hero_attribute_power(hero) for hero in friend) / 3.0
        enemy_attr = weighted_mean(enemy, enemy_weights, self._hero_attribute_power)
        friend_skill_ready = sum(self._skill_readiness(hero) for hero in friend)
        enemy_skill_ready = weighted_sum(enemy, enemy_weights, self._skill_readiness)
        friend_ultimate = sum(
            bool(u.safe_get(u.skill_slots(hero["actor"])[3] or {}, "usable", default=False))
            for hero in friend
        )
        enemy_ultimate = sum(
            bool(u.safe_get(u.skill_slots(hero["actor"])[3] or {}, "usable", default=False)) * weight
            for hero, weight in zip(enemy, enemy_weights)
        )
        friend_buffs = [self._hero_buff_ids(hero) for hero in friend]
        enemy_buffs = [self._hero_buff_ids(hero) for hero in enemy]
        key_ids = {90201, 90204, 902070, 60005, 60006, 60007}
        friend_key = sum(bool(ids & key_ids) for ids in friend_buffs) / 3.0
        enemy_key = sum(bool(ids & key_ids) * weight for ids, weight in zip(enemy_buffs, enemy_weights)) / 3.0
        friend_power = float(
            np.mean(
                [
                    np.mean(
                        (
                            _hp_ratio(hero["actor"]),
                            _value(hero["actor"], "level") / 15.0,
                            _value(hero["actor"], "money_cnt", "moneyCnt") / 15000.0,
                            self._skill_readiness(hero),
                            self._hero_attribute_power(hero),
                            bool(self._hero_buff_ids(hero) & key_ids),
                        )
                    )
                    for hero in friend
                ]
            )
        )
        enemy_power_values = [
            np.mean(
                (
                    _hp_ratio(hero["actor"]),
                    _value(hero["actor"], "level") / 15.0,
                    _value(hero["actor"], "money_cnt", "moneyCnt") / 15000.0,
                    self._skill_readiness(hero),
                    self._hero_attribute_power(hero),
                    bool(self._hero_buff_ids(hero) & key_ids),
                )
            )
            for hero in enemy
        ]
        enemy_power = (
            sum(value * weight for value, weight in zip(enemy_power_values, enemy_weights))
            / sum(enemy_weights)
            if sum(enemy_weights) > 0
            else 0.0
        )
        economy = {
            "friend_gold_sum_norm": u.clip(friend_gold / 45000.0),
            "enemy_gold_sum_norm": u.clip(enemy_gold / 45000.0),
            "gold_advantage_norm": u.signed_clip((friend_gold - enemy_gold) / 45000.0),
            "friend_gold_mean_norm": u.clip(friend_gold / 3.0 / 15000.0),
            "enemy_gold_mean_norm": u.clip(enemy_gold / max(1.0, sum(enemy_weights)) / 15000.0),
            "controlled_gold_sum_norm": u.clip(controlled_gold / 30000.0),
            "official_gold_norm": u.clip(official_gold / 15000.0),
            "richest_friend_gold_norm": u.clip(max((_value(hero["actor"], "money_cnt", "moneyCnt") for hero in friend), default=0) / 15000.0),
            "richest_enemy_gold_norm": u.clip(max((_value(hero["actor"], "money_cnt", "moneyCnt") * weight for hero, weight in zip(enemy, enemy_weights)), default=0) / 15000.0),
            "friend_level_sum_norm": u.clip(friend_level / 45.0),
            "enemy_level_sum_norm": u.clip(enemy_level / 45.0),
            "level_advantage_norm": u.signed_clip((friend_level - enemy_level) / 45.0),
            "friend_exp_progress_mean": sum(u.ratio(_value(hero["actor"], "exp"), _value(hero["actor"], "next_level_exp", "nextLevelExp", "max_exp", "maxExp")) for hero in friend) / 3.0,
            "enemy_exp_progress_mean": weighted_mean(enemy, enemy_weights, lambda hero: u.ratio(_value(hero["actor"], "exp"), _value(hero["actor"], "next_level_exp", "nextLevelExp", "max_exp", "maxExp"))),
            "friend_hp_power_norm": friend_hp_power,
            "enemy_hp_power_norm": enemy_hp_power,
            "friend_attribute_power_norm": friend_attr,
            "enemy_attribute_power_norm": enemy_attr,
            "power_advantage_norm": u.signed_clip(float(friend_power) - enemy_power),
            "friend_skill_ready_count_norm": u.clip(friend_skill_ready / 3.0),
            "enemy_skill_ready_count_norm": u.clip(enemy_skill_ready / 3.0),
            "friend_ultimate_ready_count_norm": friend_ultimate / 3.0,
            "enemy_ultimate_ready_count_norm": u.clip(enemy_ultimate / 3.0),
            "key_buff_advantage": u.signed_clip(friend_key - enemy_key),
        }
        for name, value in economy.items():
            _put(row, fields, name, value)

        friend_alive = sum(hero["alive"] for hero in friend)
        enemy_alive = sum(hero["alive"] * weight for hero, weight in zip(enemy, enemy_weights))
        enemy_dead = sum(
            (not hero["alive"]) * weight for hero, weight in zip(enemy, enemy_weights)
        )
        friend_revives = [_value(hero["actor"], "revive_time", "reviveTime") for hero in friend if not hero["alive"]]
        enemy_revives = [
            _value(hero["actor"], "revive_time", "reviveTime")
            for hero, weight in zip(enemy, enemy_weights)
            if not hero["alive"] and weight > 0
        ]
        team_positions = [hero["side_position"] for hero in friend if hero["alive"] and hero["side_position"] is not None]
        centroid = (
            (sum(position[0] for position in team_positions) / len(team_positions), sum(position[1] for position in team_positions) / len(team_positions))
            if team_positions
            else None
        )
        local_friend = sum(_distance_or_inf(hero["side_position"], centroid) <= 15000 for hero in friend if hero["alive"] and centroid is not None)
        local_enemy = sum(
            (_distance_or_inf(hero["side_position"], centroid) <= 15000) * weight
            for hero, weight in zip(enemy, enemy_weights)
            if hero["alive"] and centroid is not None
        )
        alive = {
            "friend_alive_count_norm": friend_alive / 3.0,
            "enemy_alive_count_norm": u.clip(enemy_alive / 3.0),
            "alive_advantage": u.signed_clip((friend_alive - enemy_alive) / 3.0),
            "controlled_alive_count_norm": sum(hero["alive"] and u.actor_runtime_id(hero["actor"]) in controlled_ids for hero in friend) / 2.0,
            "official_hero_alive": context["official_hero"]["alive"],
            "friend_dead_count_norm": (3 - friend_alive) / 3.0,
            "enemy_dead_count_norm": u.clip(enemy_dead / 3.0),
            "max_friend_revive_time_norm": u.clip(max(friend_revives, default=0) / 60.0),
            "min_friend_revive_time_norm": u.clip(min(friend_revives, default=0) / 60.0),
            "max_enemy_revive_time_norm": u.clip(max(enemy_revives, default=0) / 60.0),
            "min_enemy_revive_time_norm": u.clip(min(enemy_revives, default=0) / 60.0),
            "friend_low_hp_count_norm": sum(hero["alive"] and _hp_ratio(hero["actor"]) <= 0.3 for hero in friend) / 3.0,
            "enemy_low_hp_count_norm": u.clip(sum(hero["alive"] and _hp_ratio(hero["actor"]) <= 0.3 and weight for hero, weight in zip(enemy, enemy_weights)) / 3.0),
            "local_number_advantage_query": u.signed_clip((local_friend - local_enemy) / 3.0),
            "force_window_by_alive_advantage": friend_alive > enemy_alive,
            "danger_window_by_alive_disadvantage": friend_alive < enemy_alive,
        }
        for name, value in alive.items():
            _put(row, fields, name, value)

        lane = context["lane_summary"]
        own_outer, own_high, own_crystal, enemy_outer, enemy_high, enemy_crystal = context["organs"]
        hp_ratio = lambda organ: _hp_ratio(organ["actor"]) if organ["actor"] else 0.0
        structure = {
            "own_outer_alive": own_outer["alive"],
            "enemy_outer_alive": enemy_outer["alive"],
            "own_high_alive": own_high["alive"],
            "enemy_high_alive": enemy_high["alive"],
            "own_crystal_alive": own_crystal["alive"],
            "enemy_crystal_alive": enemy_crystal["alive"],
            "own_outer_hp_ratio": hp_ratio(own_outer),
            "enemy_outer_hp_ratio": hp_ratio(enemy_outer),
            "own_high_hp_ratio": hp_ratio(own_high),
            "enemy_high_hp_ratio": hp_ratio(enemy_high),
            "own_crystal_hp_ratio": hp_ratio(own_crystal),
            "enemy_crystal_hp_ratio": hp_ratio(enemy_crystal),
            "outer_tower_advantage": u.signed_clip(hp_ratio(own_outer) - hp_ratio(enemy_outer)),
            "high_tower_advantage": u.signed_clip(hp_ratio(own_high) - hp_ratio(enemy_high)),
            "crystal_hp_advantage": u.signed_clip(hp_ratio(own_crystal) - hp_ratio(enemy_crystal)),
            "own_crystal_exposed": not own_high["alive"],
            "enemy_crystal_exposed": not enemy_high["alive"],
            "own_lane_front_norm": lane["friend_progress"],
            "enemy_lane_front_norm": lane["enemy_progress"],
            "lane_front_advantage": u.signed_clip(lane["friend_progress"] - lane["enemy_progress"]),
        }
        structure["wave_supports_push"] = bool(lane["friend_front"] and _distance_or_inf(lane["friend_front"][1], lane["enemy_active"]["side_position"]) <= u.TOWER_DISTANCE_SCALE)
        structure["wave_requires_defense"] = bool(lane["enemy_front"] and _distance_or_inf(lane["enemy_front"][1], lane["own_active"]["side_position"]) <= u.TOWER_DISTANCE_SCALE)
        structure["structure_pressure_advantage"] = u.signed_clip(float(structure["wave_supports_push"]) - float(structure["wave_requires_defense"]))
        structure["crystal_race_active"] = structure["own_crystal_exposed"] and structure["enemy_crystal_exposed"]
        for name, value in structure.items():
            _put(row, fields, name, value)

        friend_stage = max(
            (3 if 60007 in ids else 2 if 60006 in ids else 1 if 60005 in ids else 0 for ids in friend_buffs),
            default=0,
        )
        enemy_stage = max(
            (3 if 60007 in ids else 2 if 60006 in ids else 1 if 60005 in ids else 0 for ids in enemy_buffs),
            default=0,
        )
        own_wolf, enemy_wolf = context["monsters"][14], context["monsters"][15]
        objectives = {
            "tyrant_alive": tyrant["alive"],
            "tyrant_spawn_within_30s": tyrant["spawn_within_30"],
            "tyrant_team_attacking": tyrant["team_attacking"],
            "tyrant_enemy_attacking": tyrant["enemy_attacking"],
            "friend_tyrant_stage_1": friend_stage >= 1,
            "friend_tyrant_stage_2": friend_stage >= 2,
            "friend_tyrant_stage_3": friend_stage >= 3,
            "enemy_tyrant_stage_1": enemy_stage >= 1,
            "enemy_tyrant_stage_2": enemy_stage >= 2,
            "enemy_tyrant_stage_3": enemy_stage >= 3,
            "dark_wolf_own_available_or_soon": own_wolf["alive"],
            "dark_wolf_enemy_available_or_soon": enemy_wolf["alive"],
            "friend_has_dark_wolf_buff": any({90204, 902070}.issubset(ids) for ids in friend_buffs),
            "enemy_has_dark_wolf_buff": any({90204, 902070}.issubset(ids) for ids in enemy_buffs),
            "speed_buff_available_or_soon": speed_resource["available"] or speed_resource["timer_valid"] and speed_resource["remaining_frames"] * u.RAW_FRAME_SECONDS <= 30,
            "friend_has_speed_buff": any(90201 in ids for ids in friend_buffs),
            "enemy_has_speed_buff": any(90201 in ids for ids in enemy_buffs),
            "own_outer_health_pack_available": context["resources"][1]["available"],
            "own_high_health_pack_available": context["resources"][2]["available"],
            "enemy_outer_health_pack_available": context["resources"][3]["available"],
            "enemy_high_health_pack_available": context["resources"][4]["available"],
        }
        tyrant_position = tyrant["position"]
        objective_friend = sum(hero["alive"] and _distance_or_inf(hero["side_position"], tyrant_position) <= 10000 for hero in friend)
        objective_enemy = sum(
            hero["alive"] and _distance_or_inf(hero["side_position"], tyrant_position) <= 10000 and weight
            for hero, weight in zip(enemy, enemy_weights)
        )
        objectives["objective_local_number_advantage"] = u.signed_clip((objective_friend - objective_enemy) / 3.0)
        objectives["objective_contest_risk"] = u.clip(0.5 + (objective_enemy - objective_friend) / 6.0 + max(0.0, -economy["power_advantage_norm"]) / 2.0)
        objectives["objective_control_advantage"] = u.signed_clip(objectives["objective_local_number_advantage"] + (friend_stage - enemy_stage) / 3.0)
        for name, value in objectives.items():
            _put(row, fields, name, value)

        friend_grouped, friend_split = self._team_grouping(friend)
        enemy_grouped, enemy_split = self._team_grouping([hero for hero, weight in zip(enemy, enemy_weights) if weight > 0])
        friend_hps = [_hp_ratio(hero["actor"]) for hero in friend if hero["alive"]]
        enemy_hps = [_hp_ratio(hero["actor"]) for hero, weight in zip(enemy, enemy_weights) if hero["alive"] and weight > 0]
        friend_damage_ready = sum(
            any(bool(u.safe_get(slot or {}, "usable", default=False)) for slot in u.skill_slots(hero["actor"])[1:4])
            for hero in friend
            if hero["alive"]
        )
        enemy_damage_ready = sum(
            any(bool(u.safe_get(slot or {}, "usable", default=False)) for slot in u.skill_slots(hero["actor"])[1:4]) * weight
            for hero, weight in zip(enemy, enemy_weights)
            if hero["alive"]
        )
        friend_escape_ready = sum(
            any(bool(u.safe_get(slot or {}, "usable", default=False)) for slot in u.skill_slots(hero["actor"])[2:4])
            for hero in friend
            if hero["alive"]
        )
        enemy_escape_ready = sum(
            any(bool(u.safe_get(slot or {}, "usable", default=False)) for slot in u.skill_slots(hero["actor"])[2:4]) * weight
            for hero, weight in zip(enemy, enemy_weights)
            if hero["alive"]
        )
        isolated_enemy = False
        for enemy_hero in enemy:
            if not enemy_hero["alive"] or not enemy_hero["visible"] or enemy_hero["side_position"] is None:
                continue
            teammate_distances = [
                _distance_or_inf(enemy_hero["side_position"], other["side_position"])
                for other in enemy
                if other is not enemy_hero and other["alive"] and other["side_position"] is not None
            ]
            friend_near = sum(_distance_or_inf(enemy_hero["side_position"], hero["side_position"]) <= 12000 for hero in friend if hero["alive"])
            isolated_enemy = isolated_enemy or (min(teammate_distances, default=u.GLOBAL_DISTANCE_SCALE) >= 15000 and friend_near >= 2)
        endangered_friend = any(hero["alive"] and _hp_ratio(hero["actor"]) <= 0.25 for hero in friend)
        readiness = context.get("collaboration_readiness", {})
        collaboration = context["collaboration"]
        teamfight = {
            "friend_grouped": friend_grouped,
            "enemy_grouped": enemy_grouped,
            "friend_split": friend_split,
            "enemy_split": enemy_split,
            "friend_avg_hp": sum(friend_hps) / len(friend_hps) if friend_hps else 0,
            "enemy_avg_hp": sum(enemy_hps) / len(enemy_hps) if enemy_hps else 0,
            "lowest_friend_hp": min(friend_hps, default=0),
            "lowest_enemy_hp": min(enemy_hps, default=0),
            "friend_damage_ready_count_norm": friend_damage_ready / 3.0,
            "enemy_damage_ready_count_norm": u.clip(enemy_damage_ready / 3.0),
            "friend_escape_ready_count_norm": friend_escape_ready / 3.0,
            "enemy_escape_ready_count_norm": u.clip(enemy_escape_ready / 3.0),
            "fight_readiness_advantage": u.signed_clip((friend_damage_ready - enemy_damage_ready) / 3.0 + (friend_hp_power - enemy_hp_power) / 2.0),
            "disengage_pressure": u.clip((enemy_damage_ready - friend_escape_ready) / 3.0 + max(0, enemy_hp_power - friend_hp_power)),
            "isolated_enemy_exists": isolated_enemy,
            "endangered_friend_exists": endangered_friend,
            "official_available_for_command": context["official_hero"]["alive"] and collaboration["cooldown_ready"],
            "command_cooldown_ready": collaboration["cooldown_ready"],
            "current_command_active": collaboration["active"],
            "current_command_legal": readiness.get("current_command_still_legal", False),
            "official_grouped_with_team": readiness.get("official_grouped_with_team", False),
            "official_far_from_team": readiness.get("official_far_from_team", False),
            "valid_collaboration_target_exists": any(readiness.get(name, False) for name in ("enemy_hero_target_available", "structure_target_available", "soldier_target_available", "objective_target_available")),
            "valid_collaboration_position_exists": any(readiness.get(name, False) for name in ("gather_position_available", "defend_position_available", "push_position_available", "objective_position_available")),
        }
        for name, value in teamfight.items():
            _put(row, fields, name, value)
        return vector

    def _build_main_context(self, context, query, hero_info):
        vector = np.zeros((1, 32), dtype=np.float32)
        row = vector[0]
        fields = schema_v1.MAIN_CONTEXT_FIELD_MAP
        query_config = u.actor_config_id(query["actor"])
        query_slot = u.HERO_CONFIG_IDS.index(query_config) if query_config in u.HERO_CONFIG_IDS else None
        if query_slot is not None:
            _put_slice(row, fields, "query_hero_slot_friend132", u.one_hot(query_slot, 6))
        controlled_configs = tuple(
            sorted(
                u.as_int(u.safe_get(info, "config_id", default=0), 0)
                for info in context["controlled_infos"]
            )
        )
        if controlled_configs in schema_v1.CONTROLLED_LINEUPS:
            _put_slice(
                row,
                fields,
                "controlled_lineup_132_141",
                u.one_hot(schema_v1.CONTROLLED_LINEUPS.index(controlled_configs), 3),
            )
        official_config = u.actor_config_id(context["official_hero"]["actor"])
        if official_config in u.HERO_CONFIG_IDS:
            _put_slice(
                row,
                fields,
                "official_hero_id_132",
                u.one_hot(u.HERO_CONFIG_IDS.index(official_config), 3),
            )
        _put(row, fields, "query_slot_valid", query_slot is not None)
        _put(row, fields, "query_alive", query["alive"])
        _put(row, fields, "query_visible", query["visible"])
        _put(row, fields, "query_is_controlled_by_us", query["controlled"])
        legal_matrix, button_legal = _raw_legal_matrix(hero_info)
        _put_slice(row, fields, "legal_button_0", button_legal)
        _put(row, fields, "any_enemy_hero_target_available", bool(np.any(legal_matrix[:, 1:4])))
        _put(row, fields, "any_nonhero_unit_target_available", bool(np.any(legal_matrix[:, 8:38])))
        _put(row, fields, "structure_target_available", bool(np.any(legal_matrix[:, 38])))
        return vector
