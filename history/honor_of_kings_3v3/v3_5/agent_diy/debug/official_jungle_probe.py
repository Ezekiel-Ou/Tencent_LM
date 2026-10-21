#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""Controlled platform experiments for official lane-hero jungle behavior.

This module deliberately lives under ``agent_diy``.  It never changes the
frozen PPO masks.  Forced scenarios bypass the production collaboration
legalizer only at the final wire command so platform acceptance and official
model behavior can be measured independently.
"""

from collections import Counter
from dataclasses import dataclass
import math
import os

from agent_ppo.feature.feature_process import v1_utils as u


LINE_HERO_CONFIG_IDS = (132, 141)
HERO_NAMES = {132: "marco", 141: "diaochan", 193: "kai"}
COMMAND_INTENT_NAMES = (
    "no_command",
    "unset",
    "focus_enemy_hero",
    "attack_structure",
    "take_key_objective",
    "push_lane",
    "gather_team",
    "defend_structure",
    "retreat_safe",
    "attack_jungle_monster",
    "move_or_wait_jungle_camp",
)
JUNGLE_REGIONS = frozenset(("upper_jungle", "lower_jungle"))
ORDINARY_MONSTER_CONFIG_IDS = frozenset((30, 31, 32, 33, 42, 43, 44, 45, 49))

NONE_COMMAND = [0, 0, 0, 0]
SET_TARGET = 1
SET_POSITION = 2
UNSET = 3
COMMAND_ECHO_GRACE_FRAMES = 30
EXPECTED_COMMAND_COOLDOWN_FRAMES = 100
EXPECTED_COMMAND_DURATION_FRAMES = 300
TARGET_APPROACH_STANDOFF = 3500.0
TARGET_SEND_MAX_DISTANCE = 5000.0
MONSTER_LEASH_DISTANCE = 4500.0
MONSTER_PULL_DISTANCE_THRESHOLDS = (4500.0, 6000.0, 7500.0, 9000.0)
HP_RESET_RISE = 0.25
HP_RESET_NEAR_FULL = 0.95
POSITION_GOAL_MATCH_DISTANCE = 100.0
WAYPOINT_REACHED_DISTANCE = 5000.0
OWN_SPRING_SIDE_POSITION = (-66300.0, 13465.0)
SPRING_RETURN_DISTANCE = 6000.0

# Side-normalized, platform-verified upper-jungle route. The final camp anchor
# is appended from live camp state so the same sequence works on both sides.
DEER_STAGED_ROUTE = (
    ("upper_base_exit", (-50000.0, 8000.0)),
    ("upper_lane_corridor", (-30000.0, 3500.0)),
    ("upper_lane_entry", (-25000.0, 0.0)),
    ("upper_inner_15000", (-25000.0, 8885.0)),
)


@dataclass(frozen=True)
class OfficialJungleScenario:
    name: str
    official_config_id: int
    mode: str
    camp_name: str = ""
    controlled_policy: str = "freeze_all"

    @property
    def freeze_controlled_heroes(self):
        return self.controlled_policy == "freeze_all"

    @property
    def run_controlled_kai_only(self):
        return self.controlled_policy == "kai_only"


SCENARIO_CATALOG = (
    OfficialJungleScenario(
        "baseline_marco",
        132,
        "observe_policy",
        controlled_policy="policy_all",
    ),
    OfficialJungleScenario(
        "baseline_diaochan",
        141,
        "observe_policy",
        controlled_policy="policy_all",
    ),
    OfficialJungleScenario("deer_far_once_marco", 132, "far_position_once", "deer"),
    OfficialJungleScenario("deer_far_once_diaochan", 141, "far_position_once", "deer"),
    OfficialJungleScenario(
        "deer_far_refresh_marco", 132, "far_position_refresh", "deer"
    ),
    OfficialJungleScenario(
        "deer_far_refresh_diaochan",
        141,
        "far_position_refresh",
        "deer",
    ),
    OfficialJungleScenario(
        "deer_staged_waypoints_marco",
        132,
        "staged_waypoints",
        "deer",
    ),
    OfficialJungleScenario(
        "deer_staged_waypoints_diaochan",
        141,
        "staged_waypoints",
        "deer",
    ),
    OfficialJungleScenario(
        "deer_far_refresh_until_contact_marco",
        132,
        "far_refresh_until_contact",
        "deer",
    ),
    OfficialJungleScenario(
        "deer_far_refresh_until_contact_diaochan",
        141,
        "far_refresh_until_contact",
        "deer",
    ),
    OfficialJungleScenario(
        "deer_far_unset_on_contact_marco",
        132,
        "far_unset_on_contact",
        "deer",
    ),
    OfficialJungleScenario(
        "deer_far_unset_on_contact_diaochan",
        141,
        "far_unset_on_contact",
        "deer",
    ),
    OfficialJungleScenario(
        "deer_far_refresh_until_clear_marco",
        132,
        "far_refresh_until_clear",
        "deer",
    ),
    OfficialJungleScenario(
        "deer_far_refresh_until_clear_diaochan",
        141,
        "far_refresh_until_clear",
        "deer",
    ),
    OfficialJungleScenario(
        "deer_team_refresh_until_clear_marco",
        132,
        "team_refresh_until_clear",
        "deer",
        "kai_only",
    ),
    OfficialJungleScenario(
        "deer_team_refresh_until_clear_diaochan",
        141,
        "team_refresh_until_clear",
        "deer",
        "kai_only",
    ),
    OfficialJungleScenario(
        "deer_team_renew_on_expiry_until_clear_marco",
        132,
        "team_renew_on_expiry_until_clear",
        "deer",
        "kai_only",
    ),
    OfficialJungleScenario(
        "deer_team_renew_on_expiry_until_clear_diaochan",
        141,
        "team_renew_on_expiry_until_clear",
        "deer",
        "kai_only",
    ),
    OfficialJungleScenario(
        "deer_staged_refresh_until_contact_marco",
        132,
        "staged_refresh_until_contact",
        "deer",
    ),
    OfficialJungleScenario(
        "deer_staged_refresh_until_contact_diaochan",
        141,
        "staged_refresh_until_contact",
        "deer",
    ),
    OfficialJungleScenario(
        "deer_position_anchor_marco", 132, "set_position_anchor", "deer"
    ),
    OfficialJungleScenario(
        "deer_position_anchor_diaochan", 141, "set_position_anchor", "deer"
    ),
    OfficialJungleScenario(
        "deer_position_standoff_marco",
        132,
        "set_position_standoff",
        "deer",
    ),
    OfficialJungleScenario(
        "deer_position_standoff_diaochan",
        141,
        "set_position_standoff",
        "deer",
    ),
    OfficialJungleScenario(
        "deer_set_target_marco", 132, "approach_then_set_target", "deer"
    ),
    OfficialJungleScenario(
        "deer_set_target_diaochan",
        141,
        "approach_then_set_target",
        "deer",
    ),
    OfficialJungleScenario(
        "wolf_set_target_marco", 132, "approach_then_set_target", "wolf"
    ),
    OfficialJungleScenario(
        "wolf_set_target_diaochan",
        141,
        "approach_then_set_target",
        "wolf",
    ),
)
SCENARIO_BY_NAME = {scenario.name: scenario for scenario in SCENARIO_CATALOG}
DEFAULT_SCENARIO_NAMES = (
    "deer_team_refresh_until_clear_marco",
    "deer_team_refresh_until_clear_diaochan",
    "deer_team_renew_on_expiry_until_clear_marco",
    "deer_team_renew_on_expiry_until_clear_diaochan",
)


def selected_scenarios(value=None):
    """Return the requested scenario sequence, preserving caller order."""

    raw = os.environ.get("DEBUG_SCENARIOS", "") if value is None else value
    names = [item.strip() for item in str(raw).split(",") if item.strip()]
    if not names:
        return tuple(SCENARIO_BY_NAME[name] for name in DEFAULT_SCENARIO_NAMES)
    unknown = [name for name in names if name not in SCENARIO_BY_NAME]
    if unknown:
        raise ValueError(
            "unknown DEBUG_SCENARIOS: "
            + ", ".join(unknown)
            + "; valid values: "
            + ", ".join(SCENARIO_BY_NAME)
        )
    return tuple(SCENARIO_BY_NAME[name] for name in names)


def region_name(side_position):
    index = u.semantic_region(side_position)
    if 0 <= index < len(u.REGION_NAMES):
        return u.REGION_NAMES[index]
    return "unknown"


class OfficialJungleProbe:
    """Override or observe one official-model jungle experiment."""

    def __init__(self, target_attempt_limit=4):
        self.target_attempt_limit = max(1, int(target_attempt_limit))
        self.reset(None)

    def reset(self, scenario):
        self.scenario = scenario
        self.phase = "inactive"
        self.pending_target = None
        self.target_attempt_counts = Counter()
        self.target_results = []
        self.position_command_count = 0
        self.target_command_count = 0
        self.jungle_entry_frames = []
        self.ordinary_monster_attacks = []
        self.cause_counts = Counter()
        self.last_region = "unknown"
        self.last_attack_runtime = 0
        self.last_policy_command = None
        self.camp_engaged = False
        self.camp_cleared = False
        self.max_monster_anchor_distance = 0.0
        self.monster_leash_failure = False
        self.monster_hp_reset = False
        self.last_monster_hp_ratio = {}
        self.minimum_monster_hp_ratio = {}
        self.sent_initial_far_position = False
        self.waypoint_index = 0
        self.waypoint_transitions = []
        self.current_waypoint_name = ""
        self.current_waypoint_goal = None
        self.position_command_frames = []
        self.position_command_goals = []
        self.same_position_refresh_count = 0
        self.matching_position_hold_steps = 0
        self.expired_position_reissue_count = 0
        self.start_camp_distance = None
        self.minimum_camp_distance = float("inf")
        self.final_camp_distance = None
        self.first_camp_arrival_frame = 0
        self.first_camp_engage_frame = 0
        self.first_camp_clear_frame = 0
        self.first_monster_leash_frame = 0
        self.first_monster_hp_reset_frame = 0
        self.position_control_stop_frame = 0
        self.position_control_stop_reason = ""
        self.position_control_stop_distance = None
        self.position_control_stop_active = False
        self.position_control_stop_remaining_frames = 0
        self.position_control_stop_active_goal = None
        self.position_commands_after_stop = 0
        self.unset_command_count = 0
        self.unset_command_frames = []
        self.unset_sent = False
        self.policy_handoff_frame = 0
        self.first_successor_command_frame = 0
        self.first_successor_command = None
        self.first_inactive_after_stop_frame = 0
        self.first_spring_return_after_stop_frame = 0
        self.maximum_post_stop_camp_distance = 0.0
        self.first_unobserved_after_engage_frame = 0
        self.unobserved_after_engage_steps = 0
        self.first_reappearance_after_unobserved_frame = 0
        self.reappearance_after_unobserved_count = 0
        self.camp_unobserved_after_engage = False
        self.last_visible_camp_frame = 0
        self.last_camp_lifecycle = {}
        self.monster_pull_threshold_first_frames = {}
        self.monster_pull_threshold_steps = Counter()
        self.monster_pull_threshold_current_streak = Counter()
        self.monster_pull_threshold_longest_streak = Counter()
        self.first_monster_hp_reset_anchor_distance = None
        self.controlled_kai_present = False
        self.first_kai_wolf_attack_frame = 0
        self.first_wolf_clear_frame = 0
        self.deer_hp_ratios_at_wolf_clear = {}
        self.first_kai_deer_arrival_frame = 0
        self.first_kai_deer_attack_frame = 0
        self.first_lower_route_frame = 0
        self.first_kai_pioneer_attack_frame = 0
        self.first_kai_leash_recover_frame = 0
        self.kai_leash_recover_steps = 0
        self.first_official_lane_row_entry_after_clear_frame = 0
        self.deer_clear_team_snapshot = {}
        self.last_team_kai_metrics = {}
        self.observed_micro_cd = set()
        self.observed_max_duration = set()

    def apply(
        self,
        proposed_command,
        observation,
        camp,
        config_ids,
        feature_process,
        policy_decision,
    ):
        """Return ``(actual_wire_command, diagnostic)`` for the current step."""

        proposed = [int(value) for value in proposed_command]
        scenario = self.scenario
        context = getattr(feature_process, "_memo_context", None) or {}
        official = context.get("official_hero") or {}
        official_actor = official.get("actor")
        official_config_id = (
            u.actor_config_id(official_actor) if official_actor is not None else 0
        )
        frame = u.frame_number(observation.get("frame_state", {}) or {})
        active = bool(
            scenario is not None
            and official_actor is not None
            and official_config_id == scenario.official_config_id
            and official_config_id in LINE_HERO_CONFIG_IDS
        )
        diagnostic = {
            "active": active,
            "scenario": scenario.name if scenario else "none",
            "mode": scenario.mode if scenario else "none",
            "phase": self.phase,
            "frame": frame,
            "official_config_id": official_config_id,
            "controlled_config_ids": tuple(int(value) for value in config_ids),
            "proposed_command": tuple(proposed),
            "policy_decision": dict(policy_decision or {}),
            "override_applied": False,
            "reason": "inactive",
        }
        if not active:
            self.phase = "inactive"
            return proposed, diagnostic

        if proposed[0] != 0 and policy_decision:
            self.last_policy_command = dict(policy_decision)

        collaboration = context.get("collaboration") or {}
        micro_cd = int(collaboration.get("micro_cd", 0))
        max_duration = int(collaboration.get("max_duration", 0))
        if micro_cd > 0:
            self.observed_micro_cd.add(micro_cd)
        if max_duration > 0:
            self.observed_max_duration.add(max_duration)

        official_position = official.get("side_position")
        official_region = region_name(official_position)
        attack_target = self._official_attack_target(context, official_actor)
        entering_jungle = bool(
            official_region in JUNGLE_REGIONS and self.last_region not in JUNGLE_REGIONS
        )
        if entering_jungle:
            self.jungle_entry_frames.append(frame)
        new_ordinary_monster_attack = bool(
            attack_target["config_id"] in ORDINARY_MONSTER_CONFIG_IDS
            and attack_target["runtime_id"] != self.last_attack_runtime
        )
        if new_ordinary_monster_attack:
            self.ordinary_monster_attacks.append(
                {
                    "frame": frame,
                    "runtime_id": attack_target["runtime_id"],
                    "config_id": attack_target["config_id"],
                    "region": official_region,
                }
            )

        cause = self._classify_jungle_cause(
            context=context,
            official_region=official_region,
            attack_target=attack_target,
            policy_decision=policy_decision,
        )
        if official_region in JUNGLE_REGIONS:
            self.cause_counts[cause] += 1

        camp_status = None
        camp_metrics = {}
        team_kai_metrics = {}
        if scenario.camp_name:
            camp_status = feature_process._opening_camp_status(
                context,
                scenario.camp_name,
                kai_position=official_position,
            )
            self._update_position_progress(
                frame=frame,
                official_position=official_position,
                camp_anchor=camp_status.get("side_anchor"),
            )
            camp_metrics = self._update_camp_metrics(
                camp_status=camp_status,
                camp_lifecycle=self._camp_lifecycle_snapshot(
                    feature_process,
                    camp_status,
                ),
                official_attack_runtime=attack_target["runtime_id"],
                camp=int(camp),
                frame=frame,
            )
            team_kai_metrics = self._update_team_kai_metrics(
                context=context,
                feature_process=feature_process,
                camp=int(camp),
                frame=frame,
            )

        if scenario.mode == "observe_policy":
            self.phase = "observe_policy"
            result = proposed
            reason = "pass_production_policy"
        elif scenario.mode == "far_position_once":
            self.phase = "far_position_once"
            side_goal = tuple(float(value) for value in camp_status["side_anchor"])
            raw_goal = u.raw_position(side_goal, int(camp))
            result, reason = self._far_position_once_command(
                collaboration=collaboration,
                raw_goal=raw_goal,
            )
            self._set_current_waypoint("deer_anchor", side_goal)
        elif scenario.mode == "far_position_refresh":
            self.phase = "far_position_refresh"
            side_goal = tuple(float(value) for value in camp_status["side_anchor"])
            raw_goal = u.raw_position(side_goal, int(camp))
            result, reason = self._far_position_refresh_command(
                collaboration=collaboration,
                raw_goal=raw_goal,
            )
            self._set_current_waypoint("deer_anchor", side_goal)
        elif scenario.mode == "far_refresh_until_contact":
            self.phase = "far_refresh_until_contact"
            side_goal = tuple(float(value) for value in camp_status["side_anchor"])
            self._set_current_waypoint("deer_anchor", side_goal)
            if self._maybe_stop_position_control(frame, collaboration):
                result = list(NONE_COMMAND)
                reason = "stop_position_after_" + self.position_control_stop_reason
            else:
                raw_goal = u.raw_position(side_goal, int(camp))
                result, reason = self._far_position_refresh_command(
                    collaboration=collaboration,
                    raw_goal=raw_goal,
                )
        elif scenario.mode == "far_unset_on_contact":
            self.phase = "far_unset_on_contact"
            side_goal = tuple(float(value) for value in camp_status["side_anchor"])
            self._set_current_waypoint("deer_anchor", side_goal)
            if self._maybe_stop_position_control(frame, collaboration):
                result, reason = self._unset_position_command(
                    collaboration=collaboration,
                    reason="unset_position_after_" + self.position_control_stop_reason,
                )
            else:
                raw_goal = u.raw_position(side_goal, int(camp))
                result, reason = self._far_position_refresh_command(
                    collaboration=collaboration,
                    raw_goal=raw_goal,
                )
        elif scenario.mode in (
            "far_refresh_until_clear",
            "team_refresh_until_clear",
            "team_renew_on_expiry_until_clear",
        ):
            self.phase = scenario.mode
            side_goal = tuple(float(value) for value in camp_status["side_anchor"])
            self._set_current_waypoint("deer_anchor", side_goal)
            if self.camp_cleared:
                self._set_position_control_stop(
                    frame=frame,
                    reason="authoritative_camp_clear",
                    collaboration=collaboration,
                )
                if scenario.mode.startswith("team_"):
                    result, reason = self._team_handoff_after_clear(
                        proposed=proposed,
                        collaboration=collaboration,
                        frame=frame,
                    )
                else:
                    result, reason = self._unset_position_command(
                        collaboration=collaboration,
                        reason="unset_position_after_authoritative_camp_clear",
                    )
            else:
                raw_goal = u.raw_position(side_goal, int(camp))
                if scenario.mode == "team_renew_on_expiry_until_clear":
                    result, reason = self._far_position_renew_on_expiry_command(
                        collaboration=collaboration,
                        side_goal=side_goal,
                        raw_goal=raw_goal,
                    )
                else:
                    result, reason = self._far_position_refresh_command(
                        collaboration=collaboration,
                        raw_goal=raw_goal,
                    )
        elif scenario.mode == "staged_waypoints":
            self.phase = "staged_waypoints"
            result, reason = self._staged_waypoint_command(
                collaboration=collaboration,
                official_position=official_position,
                camp_anchor=camp_status["side_anchor"],
                frame=frame,
                camp=int(camp),
            )
        elif scenario.mode == "staged_refresh_until_contact":
            self.phase = "staged_refresh_until_contact"
            if self._maybe_stop_position_control(frame, collaboration):
                result = list(NONE_COMMAND)
                reason = "stop_position_after_" + self.position_control_stop_reason
            else:
                result, reason = self._staged_waypoint_command(
                    collaboration=collaboration,
                    official_position=official_position,
                    camp_anchor=camp_status["side_anchor"],
                    frame=frame,
                    camp=int(camp),
                    refresh_matching=True,
                )
        elif scenario.mode in ("set_position_anchor", "set_position_standoff"):
            self.phase = "hold_camp_position"
            side_goal = tuple(float(value) for value in camp_status["side_anchor"])
            if scenario.mode == "set_position_standoff":
                side_goal = self._standoff_goal(
                    side_goal,
                    TARGET_APPROACH_STANDOFF,
                )
            raw_goal = u.raw_position(side_goal, int(camp))
            result, reason = self._position_command(
                collaboration=collaboration,
                side_goal=side_goal,
                raw_goal=raw_goal,
            )
            self._set_current_waypoint("legacy_camp_position", side_goal)
        elif scenario.mode == "approach_then_set_target":
            result, reason = self._target_experiment_command(
                context=context,
                camp_status=camp_status,
                official_position=official_position,
                frame=frame,
                camp=int(camp),
            )
        else:
            raise ValueError(f"unsupported official jungle debug mode: {scenario.mode}")

        self._update_post_stop_state(
            frame=frame,
            collaboration=collaboration,
            official_position=official_position,
        )
        if result[0] == SET_POSITION:
            self.position_command_count += 1
            self._record_position_command(
                frame=frame,
                command=result,
                reason=reason,
                camp=int(camp),
            )
        elif result[0] == SET_TARGET:
            self.target_command_count += 1
        elif result[0] == UNSET:
            self.unset_command_count += 1
            self.unset_command_frames.append(int(frame))

        self.last_region = official_region
        self.last_attack_runtime = attack_target["runtime_id"]
        diagnostic.update(
            {
                "phase": self.phase,
                "reason": reason,
                "override_applied": result != proposed,
                "actual_command": tuple(result),
                "official_position": self._rounded_or_none(official_position),
                "official_region": official_region,
                "entered_jungle_this_step": entering_jungle,
                "jungle_cause": cause,
                "official_attack_target": attack_target,
                "new_ordinary_monster_attack": new_ordinary_monster_attack,
                "last_policy_command": dict(self.last_policy_command or {}),
                "enemy_outer_alive": bool(context["organs"][3]["alive"]),
                "collaboration": self._collaboration_diagnostic(collaboration),
                "camp": camp_metrics,
                "team_kai": team_kai_metrics,
                "position_control": self._position_control_diagnostic(),
                "target_attempts": tuple(self.target_results),
                "pending_target": dict(self.pending_target or {}),
            }
        )
        return result, diagnostic

    def summary(self):
        scenario = self.scenario
        accepted = [
            item for item in self.target_results if item["status"] == "accepted"
        ]
        rejected = [
            item for item in self.target_results if item["status"] == "rejected"
        ]
        return {
            "scenario": scenario.name if scenario else "none",
            "official_config_id": scenario.official_config_id if scenario else 0,
            "mode": scenario.mode if scenario else "none",
            "camp_name": scenario.camp_name if scenario else "",
            "position_command_count": self.position_command_count,
            "position_command_frames": tuple(self.position_command_frames),
            "position_command_goals": tuple(self.position_command_goals),
            "same_position_refresh_count": self.same_position_refresh_count,
            "matching_position_hold_steps": self.matching_position_hold_steps,
            "expired_position_reissue_count": (
                self.expired_position_reissue_count
            ),
            "position_command_intervals": tuple(
                current - previous
                for previous, current in zip(
                    self.position_command_frames,
                    self.position_command_frames[1:],
                )
            ),
            "distinct_position_goal_count": len(
                {
                    (
                        int(round(item["side_goal"][0] / 100.0)),
                        int(round(item["side_goal"][1] / 100.0)),
                    )
                    for item in self.position_command_goals
                    if item.get("side_goal") is not None
                }
            ),
            "waypoint_index": self.waypoint_index,
            "waypoint_transitions": tuple(self.waypoint_transitions),
            "current_waypoint_name": self.current_waypoint_name,
            "current_waypoint_goal": self._rounded_or_none(self.current_waypoint_goal),
            "start_camp_distance": self._rounded_distance(self.start_camp_distance),
            "minimum_camp_distance": self._rounded_distance(self.minimum_camp_distance),
            "final_camp_distance": self._rounded_distance(self.final_camp_distance),
            "first_camp_arrival_frame": self.first_camp_arrival_frame,
            "reached_camp": bool(self.first_camp_arrival_frame),
            "first_camp_engage_frame": self.first_camp_engage_frame,
            "first_camp_clear_frame": self.first_camp_clear_frame,
            "first_monster_leash_frame": self.first_monster_leash_frame,
            "first_monster_hp_reset_frame": self.first_monster_hp_reset_frame,
            "position_control_stop_frame": self.position_control_stop_frame,
            "position_control_stop_reason": self.position_control_stop_reason,
            "position_control_stop_distance": self._rounded_distance(
                self.position_control_stop_distance
            ),
            "position_control_stop_active": (self.position_control_stop_active),
            "position_control_stop_remaining_frames": (
                self.position_control_stop_remaining_frames
            ),
            "position_control_stop_active_goal": self._rounded_or_none(
                self.position_control_stop_active_goal
            ),
            "position_commands_after_stop": self.position_commands_after_stop,
            "unset_command_count": self.unset_command_count,
            "unset_command_frames": tuple(self.unset_command_frames),
            "policy_handoff_frame": self.policy_handoff_frame,
            "first_successor_command_frame": (
                self.first_successor_command_frame
            ),
            "first_successor_command": (
                tuple(self.first_successor_command)
                if self.first_successor_command is not None
                else None
            ),
            "first_inactive_after_stop_frame": (self.first_inactive_after_stop_frame),
            "stop_to_inactive_frames": (
                self.first_inactive_after_stop_frame - self.position_control_stop_frame
                if self.first_inactive_after_stop_frame
                and self.position_control_stop_frame
                else None
            ),
            "unset_to_inactive_frames": (
                self.first_inactive_after_stop_frame - self.unset_command_frames[0]
                if self.first_inactive_after_stop_frame and self.unset_command_frames
                else None
            ),
            "first_spring_return_after_stop_frame": (
                self.first_spring_return_after_stop_frame
            ),
            "maximum_post_stop_camp_distance": self._rounded_distance(
                self.maximum_post_stop_camp_distance
            ),
            "stop_to_clear_frames": (
                self.first_camp_clear_frame - self.position_control_stop_frame
                if self.first_camp_clear_frame and self.position_control_stop_frame
                else None
            ),
            "observed_micro_cd": tuple(sorted(self.observed_micro_cd)),
            "observed_max_duration": tuple(sorted(self.observed_max_duration)),
            "formal_command_config_match": bool(
                self.observed_micro_cd == {EXPECTED_COMMAND_COOLDOWN_FRAMES}
                and self.observed_max_duration == {EXPECTED_COMMAND_DURATION_FRAMES}
            ),
            "target_command_count": self.target_command_count,
            "target_accepted_count": len(accepted),
            "target_rejected_count": len(rejected),
            "target_results": tuple(self.target_results),
            "target_pending_at_end": dict(self.pending_target or {}),
            "entered_jungle": bool(self.jungle_entry_frames),
            "jungle_entry_frames": tuple(self.jungle_entry_frames),
            "ordinary_monster_attack_count": len(self.ordinary_monster_attacks),
            "ordinary_monster_attacks": tuple(self.ordinary_monster_attacks),
            "jungle_cause_counts": dict(self.cause_counts),
            "camp_engaged": self.camp_engaged,
            "camp_cleared": self.camp_cleared,
            "camp_clear_is_authoritative": bool(
                self.camp_cleared
                and self.last_camp_lifecycle.get("authoritative_cleared", False)
            ),
            "camp_cleared_after_engage": bool(
                self.first_camp_clear_frame
                and self.first_camp_engage_frame
                and self.first_camp_clear_frame >= self.first_camp_engage_frame
            ),
            "last_camp_lifecycle": dict(self.last_camp_lifecycle),
            "last_visible_camp_frame": self.last_visible_camp_frame,
            "first_unobserved_after_engage_frame": (
                self.first_unobserved_after_engage_frame
            ),
            "unobserved_after_engage_steps": (self.unobserved_after_engage_steps),
            "first_reappearance_after_unobserved_frame": (
                self.first_reappearance_after_unobserved_frame
            ),
            "reappearance_after_unobserved_count": (
                self.reappearance_after_unobserved_count
            ),
            "max_monster_anchor_distance": round(self.max_monster_anchor_distance, 1),
            "monster_leash_failure": self.monster_leash_failure,
            "monster_pull_threshold_first_frames": {
                str(int(threshold)): int(frame)
                for threshold, frame in sorted(
                    self.monster_pull_threshold_first_frames.items()
                )
            },
            "monster_pull_threshold_steps": {
                str(int(threshold)): int(
                    self.monster_pull_threshold_steps[threshold]
                )
                for threshold in MONSTER_PULL_DISTANCE_THRESHOLDS
            },
            "monster_pull_threshold_longest_streak_steps": {
                str(int(threshold)): int(
                    self.monster_pull_threshold_longest_streak[threshold]
                )
                for threshold in MONSTER_PULL_DISTANCE_THRESHOLDS
            },
            "monster_hp_reset": self.monster_hp_reset,
            "first_monster_hp_reset_anchor_distance": (
                self._rounded_distance(
                    self.first_monster_hp_reset_anchor_distance
                )
            ),
            "cleared_without_leash": bool(
                self.camp_cleared
                and not self.monster_leash_failure
                and not self.monster_hp_reset
            ),
            "controlled_policy": (
                scenario.controlled_policy if scenario else "none"
            ),
            "controlled_kai_present": self.controlled_kai_present,
            "first_kai_wolf_attack_frame": self.first_kai_wolf_attack_frame,
            "first_wolf_clear_frame": self.first_wolf_clear_frame,
            "deer_hp_ratios_at_wolf_clear": dict(
                self.deer_hp_ratios_at_wolf_clear
            ),
            "official_deer_engaged_before_wolf_clear": bool(
                self.first_camp_engage_frame
                and self.first_wolf_clear_frame
                and self.first_camp_engage_frame < self.first_wolf_clear_frame
            ),
            "deer_cleared_before_wolf_clear": bool(
                self.first_camp_clear_frame
                and self.first_wolf_clear_frame
                and self.first_camp_clear_frame <= self.first_wolf_clear_frame
            ),
            "first_kai_deer_arrival_frame": (
                self.first_kai_deer_arrival_frame
            ),
            "first_kai_deer_attack_frame": (
                self.first_kai_deer_attack_frame
            ),
            "first_lower_route_frame": self.first_lower_route_frame,
            "first_kai_pioneer_attack_frame": (
                self.first_kai_pioneer_attack_frame
            ),
            "first_kai_leash_recover_frame": (
                self.first_kai_leash_recover_frame
            ),
            "kai_leash_recover_steps": self.kai_leash_recover_steps,
            "first_official_lane_row_entry_after_clear_frame": (
                self.first_official_lane_row_entry_after_clear_frame
            ),
            "deer_clear_team_snapshot": dict(
                self.deer_clear_team_snapshot
            ),
            "wolf_clear_to_deer_clear_frames": (
                self.first_camp_clear_frame - self.first_wolf_clear_frame
                if self.first_camp_clear_frame
                and self.first_wolf_clear_frame
                and self.first_camp_clear_frame >= self.first_wolf_clear_frame
                else None
            ),
            "kai_deer_attack_to_clear_frames": (
                self.first_camp_clear_frame - self.first_kai_deer_attack_frame
                if self.first_camp_clear_frame
                and self.first_kai_deer_attack_frame
                and self.first_camp_clear_frame
                >= self.first_kai_deer_attack_frame
                else None
            ),
            "deer_clear_to_pioneer_attack_frames": (
                self.first_kai_pioneer_attack_frame
                - self.first_camp_clear_frame
                if self.first_kai_pioneer_attack_frame
                and self.first_camp_clear_frame
                and self.first_kai_pioneer_attack_frame
                >= self.first_camp_clear_frame
                else None
            ),
            "last_team_kai_metrics": dict(self.last_team_kai_metrics),
            "minimum_monster_hp_ratio": {
                str(runtime): round(value, 4)
                for runtime, value in self.minimum_monster_hp_ratio.items()
            },
        }

    def _far_position_once_command(self, collaboration, raw_goal):
        if self.sent_initial_far_position:
            return list(NONE_COMMAND), "far_position_already_sent_once"
        if not collaboration.get("cooldown_ready", False):
            return list(NONE_COMMAND), "wait_initial_position_cooldown"
        self.sent_initial_far_position = True
        return [
            SET_POSITION,
            0,
            int(round(raw_goal[0])),
            int(round(raw_goal[1])),
        ], "send_far_position_once"

    def _far_position_refresh_command(self, collaboration, raw_goal):
        if not collaboration.get("cooldown_ready", False):
            return list(NONE_COMMAND), "wait_refresh_cooldown"
        reason = (
            "refresh_same_far_position"
            if self.position_command_count
            else "send_initial_far_position"
        )
        return [
            SET_POSITION,
            0,
            int(round(raw_goal[0])),
            int(round(raw_goal[1])),
        ], reason

    def _far_position_renew_on_expiry_command(
        self,
        collaboration,
        side_goal,
        raw_goal,
    ):
        current_position = collaboration.get("command_position")
        matching_active = bool(
            collaboration.get("active")
            and collaboration.get("type") == "set_position"
            and current_position is not None
            and u.distance(current_position, side_goal)
            <= POSITION_GOAL_MATCH_DISTANCE
        )
        if matching_active:
            self.matching_position_hold_steps += 1
            return list(NONE_COMMAND), "hold_matching_position_until_expiry"
        if not collaboration.get("cooldown_ready", False):
            return list(NONE_COMMAND), "wait_expired_position_cooldown"
        if self.position_command_count:
            self.expired_position_reissue_count += 1
            reason = (
                "replace_nonmatching_position"
                if collaboration.get("active")
                else "reissue_expired_position"
            )
        else:
            reason = "send_initial_far_position"
        return [
            SET_POSITION,
            0,
            int(round(raw_goal[0])),
            int(round(raw_goal[1])),
        ], reason

    def _team_handoff_after_clear(
        self,
        proposed,
        collaboration,
        frame,
    ):
        if not self.policy_handoff_frame:
            self.policy_handoff_frame = int(frame)
        if proposed[0] != 0 and not self.first_successor_command_frame:
            self.first_successor_command_frame = int(frame)
            self.first_successor_command = tuple(int(value) for value in proposed)
        reason = (
            "direct_successor_command_after_clear"
            if proposed[0] != 0
            else (
                "hold_old_deer_command_until_successor_cooldown"
                if collaboration.get("active", False)
                else "wait_production_successor_after_clear"
            )
        )
        return list(proposed), reason

    def _staged_waypoint_command(
        self,
        collaboration,
        official_position,
        camp_anchor,
        frame,
        camp,
        refresh_matching=False,
    ):
        route = list(DEER_STAGED_ROUTE)
        route.append(
            (
                "deer_anchor",
                tuple(float(value) for value in camp_anchor),
            )
        )
        self.waypoint_index = min(
            self.waypoint_index,
            len(route) - 1,
        )
        while (
            official_position is not None
            and self.waypoint_index < len(route) - 1
            and u.distance(
                official_position,
                route[self.waypoint_index][1],
            )
            <= WAYPOINT_REACHED_DISTANCE
        ):
            previous_index = self.waypoint_index
            self.waypoint_index += 1
            self.waypoint_transitions.append(
                {
                    "frame": int(frame),
                    "from_index": previous_index,
                    "from_name": route[previous_index][0],
                    "to_index": self.waypoint_index,
                    "to_name": route[self.waypoint_index][0],
                }
            )

        waypoint_name, side_goal = route[self.waypoint_index]
        self._set_current_waypoint(waypoint_name, side_goal)
        current_position = collaboration.get("command_position")
        matching_active = bool(
            collaboration.get("active")
            and collaboration.get("type") == "set_position"
            and current_position is not None
            and u.distance(current_position, side_goal) <= POSITION_GOAL_MATCH_DISTANCE
        )
        if matching_active and not refresh_matching:
            return list(NONE_COMMAND), "hold_active_staged_waypoint"
        if not collaboration.get("cooldown_ready", False):
            return list(NONE_COMMAND), "wait_staged_waypoint_cooldown"

        raw_goal = u.raw_position(side_goal, camp)
        if not self.position_command_count:
            reason = "send_initial_staged_waypoint"
        elif matching_active:
            reason = "refresh_current_staged_waypoint"
        elif collaboration.get("active"):
            reason = "advance_to_next_staged_waypoint"
        else:
            reason = "reissue_expired_staged_waypoint"
        return [
            SET_POSITION,
            0,
            int(round(raw_goal[0])),
            int(round(raw_goal[1])),
        ], reason

    def _set_current_waypoint(self, name, side_goal):
        self.current_waypoint_name = str(name)
        self.current_waypoint_goal = tuple(float(value) for value in side_goal)

    def _maybe_stop_position_control(self, frame, collaboration):
        if self.position_control_stop_frame:
            return True
        reason = ""
        if self.camp_engaged:
            reason = "camp_engaged"
        elif (
            self.final_camp_distance is not None
            and self.final_camp_distance <= WAYPOINT_REACHED_DISTANCE
        ):
            reason = "camp_arrival_radius"
        if not reason:
            return False
        self._set_position_control_stop(
            frame=frame,
            reason=reason,
            collaboration=collaboration,
        )
        return True

    def _set_position_control_stop(self, frame, reason, collaboration):
        if self.position_control_stop_frame:
            return
        self.position_control_stop_frame = int(frame)
        self.position_control_stop_reason = str(reason)
        self.position_control_stop_distance = self.final_camp_distance
        self.position_control_stop_active = bool(collaboration.get("active", False))
        self.position_control_stop_remaining_frames = int(
            collaboration.get("remaining_frames", 0)
        )
        active_goal = collaboration.get("command_position")
        self.position_control_stop_active_goal = (
            tuple(float(value) for value in active_goal)
            if active_goal is not None
            else None
        )

    def _unset_position_command(self, collaboration, reason):
        if self.unset_sent:
            return list(NONE_COMMAND), "position_unset_already_sent"
        if not collaboration.get("active", False):
            return list(NONE_COMMAND), "position_already_inactive_at_stop"
        self.unset_sent = True
        return [UNSET, 0, 0, 0], str(reason)

    def _update_post_stop_state(
        self,
        frame,
        collaboration,
        official_position,
    ):
        if not self.position_control_stop_frame:
            return
        if not self.first_inactive_after_stop_frame and not collaboration.get(
            "active", False
        ):
            self.first_inactive_after_stop_frame = int(frame)
        if self.final_camp_distance is not None:
            self.maximum_post_stop_camp_distance = max(
                self.maximum_post_stop_camp_distance,
                float(self.final_camp_distance),
            )
        if (
            not self.first_spring_return_after_stop_frame
            and official_position is not None
            and u.distance(official_position, OWN_SPRING_SIDE_POSITION)
            <= SPRING_RETURN_DISTANCE
        ):
            self.first_spring_return_after_stop_frame = int(frame)

    def _record_position_command(self, frame, command, reason, camp):
        if (
            self.position_control_stop_frame
            and int(frame) >= self.position_control_stop_frame
        ):
            self.position_commands_after_stop += 1
        raw_goal = (float(command[2]), float(command[3]))
        side_goal = u.side_position(raw_goal, camp)
        if (
            self.position_command_goals
            and self.position_command_goals[-1].get("side_goal") is not None
            and u.distance(
                self.position_command_goals[-1]["side_goal"],
                side_goal,
            )
            <= POSITION_GOAL_MATCH_DISTANCE
        ):
            self.same_position_refresh_count += 1
        self.position_command_frames.append(int(frame))
        self.position_command_goals.append(
            {
                "frame": int(frame),
                "waypoint_index": self.waypoint_index,
                "waypoint_name": self.current_waypoint_name,
                "raw_goal": self._rounded_or_none(raw_goal),
                "side_goal": self._rounded_or_none(side_goal),
                "reason": str(reason),
            }
        )

    def _update_position_progress(
        self,
        frame,
        official_position,
        camp_anchor,
    ):
        if official_position is None or camp_anchor is None:
            return
        distance = u.distance(official_position, camp_anchor)
        if self.start_camp_distance is None:
            self.start_camp_distance = distance
        self.minimum_camp_distance = min(
            self.minimum_camp_distance,
            distance,
        )
        self.final_camp_distance = distance
        if not self.first_camp_arrival_frame and distance <= WAYPOINT_REACHED_DISTANCE:
            self.first_camp_arrival_frame = int(frame)

    def _position_control_diagnostic(self):
        return {
            "waypoint_index": self.waypoint_index,
            "waypoint_name": self.current_waypoint_name,
            "waypoint_goal": self._rounded_or_none(self.current_waypoint_goal),
            "position_command_count": self.position_command_count,
            "same_position_refresh_count": (self.same_position_refresh_count),
            "matching_position_hold_steps": self.matching_position_hold_steps,
            "expired_position_reissue_count": (
                self.expired_position_reissue_count
            ),
            "start_camp_distance": self._rounded_distance(self.start_camp_distance),
            "minimum_camp_distance": self._rounded_distance(self.minimum_camp_distance),
            "current_camp_distance": self._rounded_distance(self.final_camp_distance),
            "first_camp_arrival_frame": self.first_camp_arrival_frame,
            "first_camp_engage_frame": self.first_camp_engage_frame,
            "first_camp_clear_frame": self.first_camp_clear_frame,
            "position_control_stop_frame": self.position_control_stop_frame,
            "position_control_stop_reason": self.position_control_stop_reason,
            "position_control_stop_distance": self._rounded_distance(
                self.position_control_stop_distance
            ),
            "position_control_stop_active": (self.position_control_stop_active),
            "position_control_stop_remaining_frames": (
                self.position_control_stop_remaining_frames
            ),
            "position_control_stop_active_goal": self._rounded_or_none(
                self.position_control_stop_active_goal
            ),
            "position_commands_after_stop": self.position_commands_after_stop,
            "unset_command_count": self.unset_command_count,
            "unset_command_frames": tuple(self.unset_command_frames),
            "policy_handoff_frame": self.policy_handoff_frame,
            "first_successor_command_frame": (
                self.first_successor_command_frame
            ),
            "first_inactive_after_stop_frame": (self.first_inactive_after_stop_frame),
            "first_spring_return_after_stop_frame": (
                self.first_spring_return_after_stop_frame
            ),
            "maximum_post_stop_camp_distance": self._rounded_distance(
                self.maximum_post_stop_camp_distance
            ),
            "observed_micro_cd": tuple(sorted(self.observed_micro_cd)),
            "observed_max_duration": tuple(sorted(self.observed_max_duration)),
        }

    def _target_experiment_command(
        self,
        context,
        camp_status,
        official_position,
        frame,
        camp,
    ):
        collaboration = context.get("collaboration") or {}
        self._resolve_pending_target(
            collaboration=collaboration,
            frame=frame,
        )
        if any(item["status"] == "accepted" for item in self.target_results):
            self.phase = "target_accepted_observe"
            return list(NONE_COMMAND), "hold_after_target_acceptance"
        if self.pending_target is not None:
            self.phase = "target_pending"
            return list(NONE_COMMAND), "wait_target_echo"
        if len(self.target_results) >= self.target_attempt_limit:
            self.phase = "target_attempts_complete"
            return list(NONE_COMMAND), "target_attempt_limit_reached"

        target_actor = self._target_candidate(camp_status, camp)
        target_runtime = (
            u.actor_runtime_id(target_actor) if target_actor is not None else 0
        )
        target_position = (
            u.side_position(u.location_of(target_actor), camp)
            if target_actor is not None and u.location_of(target_actor) is not None
            else None
        )
        target_distance = (
            u.distance(official_position, target_position)
            if official_position is not None and target_position is not None
            else float("inf")
        )
        cooldown_ready = bool(collaboration.get("cooldown_ready", False))
        if (
            target_actor is not None
            and target_runtime
            and target_distance <= TARGET_SEND_MAX_DISTANCE
            and cooldown_ready
        ):
            attempt = int(self.target_attempt_counts[target_runtime]) + 1
            self.target_attempt_counts[target_runtime] = attempt
            self.pending_target = {
                "frame": frame,
                "runtime_id": target_runtime,
                "config_id": u.actor_config_id(target_actor),
                "distance": round(target_distance, 1),
                "attempt": attempt,
            }
            self.phase = "target_pending"
            return [
                SET_TARGET,
                target_runtime,
                0,
                0,
            ], "send_raw_monster_set_target"

        side_anchor = tuple(float(value) for value in camp_status["side_anchor"])
        side_goal = self._standoff_goal(
            side_anchor,
            TARGET_APPROACH_STANDOFF,
        )
        raw_goal = u.raw_position(side_goal, camp)
        self.phase = "approach_for_set_target"
        return self._position_command(
            collaboration=collaboration,
            side_goal=side_goal,
            raw_goal=raw_goal,
        )

    def _resolve_pending_target(self, collaboration, frame):
        pending = self.pending_target
        if pending is None:
            return
        accepted = bool(
            collaboration.get("active")
            and collaboration.get("type") == "set_target"
            and int(collaboration.get("target_runtime", 0))
            == int(pending["runtime_id"])
            and int(collaboration.get("platform_last_send_frame", 0))
            >= int(pending["frame"])
        )
        if accepted:
            result = dict(pending)
            result.update(status="accepted", observed_frame=frame)
            self.target_results.append(result)
            self.pending_target = None
        elif frame - int(pending["frame"]) >= COMMAND_ECHO_GRACE_FRAMES:
            result = dict(pending)
            result.update(status="rejected", observed_frame=frame)
            self.target_results.append(result)
            self.pending_target = None

    def _target_candidate(self, camp_status, camp):
        alive = [
            actor
            for actor in camp_status.get("alive_actors", ())
            if u.is_alive(actor)
            and u.actor_runtime_id(actor)
            and u.visible_to_camp(actor, camp, is_friend=False)
        ]
        return min(
            alive,
            key=lambda actor: (
                self.target_attempt_counts[u.actor_runtime_id(actor)],
                u.actor_config_id(actor),
                u.actor_runtime_id(actor),
            ),
            default=None,
        )

    @staticmethod
    def _position_command(collaboration, side_goal, raw_goal):
        current_position = collaboration.get("command_position")
        matches = bool(
            collaboration.get("active")
            and collaboration.get("type") == "set_position"
            and current_position is not None
            and u.distance(current_position, side_goal) <= 100.0
        )
        if matches:
            return list(NONE_COMMAND), "hold_matching_set_position"
        if collaboration.get("cooldown_ready", False):
            return [
                SET_POSITION,
                0,
                int(round(raw_goal[0])),
                int(round(raw_goal[1])),
            ], "send_forced_camp_set_position"
        return list(NONE_COMMAND), "wait_command_cooldown"

    @staticmethod
    def _camp_lifecycle_snapshot(feature_process, camp_status):
        raw_anchor = tuple(float(value) for value in camp_status.get("anchor", ()))
        history = getattr(feature_process, "monster_camp_history", None) or {}
        state = history.get(raw_anchor, {}) if raw_anchor else {}
        return {
            "authoritative_cleared": bool(camp_status.get("cleared", False)),
            "lifecycle_id": int(camp_status.get("lifecycle_id", 0)),
            "seen_count": int(camp_status.get("seen_count", 0)),
            "expected_members": int(camp_status.get("expected_members", 0)),
            "observed_runtime_ids": tuple(
                int(value) for value in state.get("observed_runtime_ids", ())
            ),
            "alive_runtime_ids": tuple(
                int(value) for value in state.get("alive_runtime_ids", ())
            ),
            "observer_near": bool(state.get("observer_near", False)),
            "spawn_timer_ms": state.get("spawn_timer_ms"),
            "spawn_timer_positive": bool(state.get("spawn_timer_positive", False)),
            "timer_death_transition": bool(state.get("timer_death_transition", False)),
            "last_clear_evidence": str(state.get("last_clear_evidence", "none")),
            "last_clear_frame": state.get("last_clear_frame"),
            "clear_count": int(state.get("clear_count", 0)),
        }

    def _update_camp_metrics(
        self,
        camp_status,
        camp_lifecycle,
        official_attack_runtime,
        camp,
        frame,
    ):
        anchor = tuple(float(value) for value in camp_status["side_anchor"])
        alive = tuple(camp_status.get("alive_actors", ()))
        alive_runtimes = set()
        members = []
        measurement_active = not self.camp_cleared
        maximum_step_displacement = 0.0
        for actor in alive:
            runtime = u.actor_runtime_id(actor)
            alive_runtimes.add(runtime)
            position_raw = u.location_of(actor)
            position = (
                u.side_position(position_raw, camp)
                if position_raw is not None
                else None
            )
            displacement = (
                u.distance(position, anchor) if position is not None else None
            )
            if displacement is not None and measurement_active:
                maximum_step_displacement = max(
                    maximum_step_displacement,
                    displacement,
                )
                self.max_monster_anchor_distance = max(
                    self.max_monster_anchor_distance,
                    displacement,
                )
                if displacement > MONSTER_LEASH_DISTANCE:
                    self.monster_leash_failure = True
                    if not self.first_monster_leash_frame:
                        self.first_monster_leash_frame = int(frame)
            hp_ratio = float(
                u.ratio(
                    u.actor_get(actor, "hp", default=0),
                    u.actor_get(actor, "max_hp", "maxHp", default=0),
                )
            )
            previous_hp = self.last_monster_hp_ratio.get(runtime)
            if (
                measurement_active
                and
                previous_hp is not None
                and previous_hp <= 1.0 - HP_RESET_RISE
                and hp_ratio - previous_hp >= HP_RESET_RISE
                and hp_ratio >= HP_RESET_NEAR_FULL
            ):
                self.monster_hp_reset = True
                if not self.first_monster_hp_reset_frame:
                    self.first_monster_hp_reset_frame = int(frame)
                    self.first_monster_hp_reset_anchor_distance = displacement
            if measurement_active:
                self.last_monster_hp_ratio[runtime] = hp_ratio
                self.minimum_monster_hp_ratio[runtime] = min(
                    self.minimum_monster_hp_ratio.get(runtime, 1.0),
                    hp_ratio,
                )
            if measurement_active and hp_ratio < 0.999:
                self.camp_engaged = True
            members.append(
                {
                    "runtime_id": runtime,
                    "config_id": u.actor_config_id(actor),
                    "hp_ratio": round(hp_ratio, 4),
                    "anchor_distance": (
                        round(displacement, 1) if displacement is not None else None
                    ),
                    "position": self._rounded_or_none(position),
                }
            )
        for threshold in MONSTER_PULL_DISTANCE_THRESHOLDS:
            if not measurement_active:
                continue
            if maximum_step_displacement > threshold:
                self.monster_pull_threshold_steps[threshold] += 1
                self.monster_pull_threshold_current_streak[threshold] += 1
                self.monster_pull_threshold_longest_streak[threshold] = max(
                    self.monster_pull_threshold_longest_streak[threshold],
                    self.monster_pull_threshold_current_streak[threshold],
                )
                self.monster_pull_threshold_first_frames.setdefault(
                    threshold,
                    int(frame),
                )
            else:
                self.monster_pull_threshold_current_streak[threshold] = 0
        if official_attack_runtime in alive_runtimes:
            self.camp_engaged = True
        if self.camp_engaged and not self.first_camp_engage_frame:
            self.first_camp_engage_frame = int(frame)
        authoritative_clear = bool(camp_lifecycle.get("authoritative_cleared", False))
        if alive:
            self.last_visible_camp_frame = int(frame)
            if self.camp_unobserved_after_engage:
                self.reappearance_after_unobserved_count += 1
                if not self.first_reappearance_after_unobserved_frame:
                    self.first_reappearance_after_unobserved_frame = int(frame)
            self.camp_unobserved_after_engage = False
        elif self.camp_engaged and not authoritative_clear:
            self.unobserved_after_engage_steps += 1
            if not self.first_unobserved_after_engage_frame:
                self.first_unobserved_after_engage_frame = int(frame)
            self.camp_unobserved_after_engage = True
        if authoritative_clear:
            self.camp_cleared = True
        if self.camp_cleared and not self.first_camp_clear_frame:
            self.first_camp_clear_frame = int(frame)
        self.last_camp_lifecycle = dict(camp_lifecycle)
        return {
            "name": self.scenario.camp_name,
            "raw_anchor": self._rounded_or_none(camp_status.get("anchor")),
            "side_anchor": self._rounded_or_none(anchor),
            "alive_count": len(alive),
            "members": tuple(members),
            "engaged": self.camp_engaged,
            "cleared": self.camp_cleared,
            "authoritative_cleared": authoritative_clear,
            "temporarily_unobserved_after_engage": bool(
                self.camp_engaged and not alive and not authoritative_clear
            ),
            "last_visible_frame": self.last_visible_camp_frame,
            "first_unobserved_after_engage_frame": (
                self.first_unobserved_after_engage_frame
            ),
            "unobserved_after_engage_steps": (self.unobserved_after_engage_steps),
            "first_reappearance_after_unobserved_frame": (
                self.first_reappearance_after_unobserved_frame
            ),
            "reappearance_after_unobserved_count": (
                self.reappearance_after_unobserved_count
            ),
            "lifecycle": dict(camp_lifecycle),
            "max_anchor_distance": round(self.max_monster_anchor_distance, 1),
            "leash_failure": self.monster_leash_failure,
            "pull_threshold_steps": {
                str(int(threshold)): int(
                    self.monster_pull_threshold_steps[threshold]
                )
                for threshold in MONSTER_PULL_DISTANCE_THRESHOLDS
            },
            "hp_reset": self.monster_hp_reset,
            "first_engage_frame": self.first_camp_engage_frame,
            "first_clear_frame": self.first_camp_clear_frame,
            "first_leash_frame": self.first_monster_leash_frame,
            "first_hp_reset_frame": self.first_monster_hp_reset_frame,
        }

    def _update_team_kai_metrics(
        self,
        context,
        feature_process,
        camp,
        frame,
    ):
        controlled_kai = next(
            (
                hero
                for hero in context.get("friend_heroes", ())
                if hero.get("controlled")
                and u.actor_config_id(hero.get("actor")) == 193
            ),
            None,
        )
        if controlled_kai is None:
            self.last_team_kai_metrics = {}
            return {}

        self.controlled_kai_present = True
        kai_actor = controlled_kai.get("actor")
        kai_position = controlled_kai.get("side_position")
        kai_attack_runtime = u.as_int(
            u.actor_get(
                kai_actor,
                "attack_target",
                "attack_target_runtime_id",
                "attackTarget",
                default=0,
            ),
            0,
        )
        statuses = {
            name: feature_process._opening_camp_status(
                context,
                name,
                kai_position=kai_position,
            )
            for name in ("wolf", "deer", "pioneer")
        }
        live_runtimes = {
            name: {
                u.actor_runtime_id(actor)
                for actor in status.get("alive_actors", ())
                if u.actor_runtime_id(actor)
            }
            for name, status in statuses.items()
        }
        if (
            not self.first_kai_wolf_attack_frame
            and kai_attack_runtime in live_runtimes["wolf"]
        ):
            self.first_kai_wolf_attack_frame = int(frame)
        if statuses["wolf"].get("cleared") and not self.first_wolf_clear_frame:
            self.first_wolf_clear_frame = int(frame)
            self.deer_hp_ratios_at_wolf_clear = {
                str(u.actor_runtime_id(actor)): round(
                    float(
                        u.ratio(
                            u.actor_get(actor, "hp", default=0),
                            u.actor_get(actor, "max_hp", "maxHp", default=0),
                        )
                    ),
                    4,
                )
                for actor in statuses["deer"].get("alive_actors", ())
                if u.actor_runtime_id(actor)
            }
        deer_anchor = statuses["deer"].get("side_anchor")
        kai_deer_distance = (
            u.distance(kai_position, deer_anchor)
            if kai_position is not None and deer_anchor is not None
            else math.inf
        )
        if (
            not self.first_kai_deer_arrival_frame
            and kai_deer_distance <= WAYPOINT_REACHED_DISTANCE
        ):
            self.first_kai_deer_arrival_frame = int(frame)
        if (
            not self.first_kai_deer_attack_frame
            and kai_attack_runtime in live_runtimes["deer"]
        ):
            self.first_kai_deer_attack_frame = int(frame)
        if (
            not self.first_kai_pioneer_attack_frame
            and kai_attack_runtime in live_runtimes["pioneer"]
        ):
            self.first_kai_pioneer_attack_frame = int(frame)

        macro = context.get("jungle_macro") or {}
        if (
            not self.first_lower_route_frame
            and (
                int(macro.get("opening_stage_index", 0)) >= 2
                or macro.get("opening_camp_name") in (
                    "pioneer",
                    "outer_archer",
                    "inner_archer",
                    "red_statue",
                )
            )
        ):
            self.first_lower_route_frame = int(frame)
        if macro.get("opening_leash_recover", False):
            self.kai_leash_recover_steps += 1
            if not self.first_kai_leash_recover_frame:
                self.first_kai_leash_recover_frame = int(frame)

        official_config_id = (
            self.scenario.official_config_id if self.scenario else 0
        )
        lane_plan = (context.get("lane_macro") or {}).get(
            official_config_id,
            {},
        )
        if (
            self.camp_cleared
            and lane_plan.get("row_entry_reached", False)
            and not self.first_official_lane_row_entry_after_clear_frame
        ):
            self.first_official_lane_row_entry_after_clear_frame = int(frame)
        official_actor = (context.get("official_hero") or {}).get("actor")
        own_outer = (context.get("organs") or [{}])[0]
        if self.camp_cleared and not self.deer_clear_team_snapshot:
            own_outer_actor = own_outer.get("actor")
            self.deer_clear_team_snapshot = {
                "frame": int(frame),
                "wolf_authoritative_cleared": bool(
                    statuses["wolf"].get("cleared", False)
                ),
                "controlled_kai_hp_ratio": round(
                    float(
                        u.ratio(
                            u.actor_get(kai_actor, "hp", default=0),
                            u.actor_get(
                                kai_actor,
                                "max_hp",
                                "maxHp",
                                default=0,
                            ),
                        )
                    ),
                    4,
                ),
                "official_hp_ratio": round(
                    float(
                        u.ratio(
                            u.actor_get(official_actor, "hp", default=0),
                            u.actor_get(
                                official_actor,
                                "max_hp",
                                "maxHp",
                                default=0,
                            ),
                        )
                    ),
                    4,
                ),
                "own_outer_hp_ratio": round(
                    float(
                        u.ratio(
                            u.actor_get(own_outer_actor, "hp", default=0),
                            u.actor_get(
                                own_outer_actor,
                                "max_hp",
                                "maxHp",
                                default=0,
                            ),
                        )
                    ),
                    4,
                ),
                "official_lane_phase": str(
                    lane_plan.get("phase", "unknown")
                ),
                "official_lane_row_entry_reached": bool(
                    lane_plan.get("row_entry_reached", False)
                ),
                "official_first_wave_cleared": bool(
                    lane_plan.get("first_wave_cleared", False)
                ),
            }

        metrics = {
            "controlled_kai_runtime_id": u.actor_runtime_id(kai_actor),
            "controlled_kai_position": self._rounded_or_none(kai_position),
            "controlled_kai_hp_ratio": round(
                float(
                    u.ratio(
                        u.actor_get(kai_actor, "hp", default=0),
                        u.actor_get(kai_actor, "max_hp", "maxHp", default=0),
                    )
                ),
                4,
            ),
            "controlled_kai_attack_runtime": kai_attack_runtime,
            "controlled_kai_attack_camp": next(
                (
                    name
                    for name, runtimes in live_runtimes.items()
                    if kai_attack_runtime in runtimes
                ),
                "none",
            ),
            "jungle_phase": str(macro.get("phase", "unknown")),
            "opening_stage_index": int(
                macro.get("opening_stage_index", 0)
            ),
            "opening_camp_name": str(
                macro.get("opening_camp_name", "")
            ),
            "opening_leash_recover": bool(
                macro.get("opening_leash_recover", False)
            ),
            "opening_monster_anchor_distance": round(
                float(macro.get("opening_monster_anchor_distance", 0.0)),
                1,
            ),
            "wolf_authoritative_cleared": bool(
                statuses["wolf"].get("cleared", False)
            ),
            "deer_authoritative_cleared": bool(
                statuses["deer"].get("cleared", False)
            ),
            "pioneer_authoritative_cleared": bool(
                statuses["pioneer"].get("cleared", False)
            ),
            "kai_deer_distance": self._rounded_distance(
                kai_deer_distance
            ),
            "official_lane_phase": str(lane_plan.get("phase", "unknown")),
            "official_lane_row_entry_reached": bool(
                lane_plan.get("row_entry_reached", False)
            ),
            "official_lane_release_reason": str(
                lane_plan.get("release_reason", "none")
            ),
        }
        self.last_team_kai_metrics = dict(metrics)
        return metrics

    def _classify_jungle_cause(
        self,
        context,
        official_region,
        attack_target,
        policy_decision,
    ):
        if official_region not in JUNGLE_REGIONS:
            return "not_in_jungle"
        scenario = self.scenario
        if scenario and scenario.mode != "observe_policy":
            return "debug_forced_" + scenario.mode

        decision = self.last_policy_command or policy_decision or {}
        intent = int(decision.get("intent", -1))
        if intent == 9:
            return "policy_attack_jungle_monster"
        if intent == 10:
            return "policy_move_or_wait_jungle_camp"
        if intent == 6:
            selected_region = decision.get("selected_position_region")
            if selected_region in JUNGLE_REGIONS:
                return "policy_gather_team_to_jungle"
            return "policy_gather_team_path_or_micro"
        if intent == 4:
            return "policy_key_objective"
        if intent == 5:
            return "policy_push_lane_path_or_micro"

        collaboration = context.get("collaboration") or {}
        if not collaboration.get("active"):
            return "official_micro_without_active_command"
        if (
            collaboration.get("type") == "set_target"
            and attack_target["config_id"] in ORDINARY_MONSTER_CONFIG_IDS
        ):
            return "active_target_and_official_monster_attack"
        lane_plan = (context.get("lane_macro") or {}).get(
            self.scenario.official_config_id,
            {},
        )
        if lane_plan.get("command_active"):
            return "lane_guard_path_or_official_micro"
        return "official_micro_under_other_command"

    @staticmethod
    def _official_attack_target(context, official_actor):
        runtime = u.as_int(
            u.actor_get(
                official_actor,
                "attack_target",
                "attack_target_runtime_id",
                "attackTarget",
                default=0,
            ),
            0,
        )
        target = (context.get("actor_by_runtime") or {}).get(runtime)
        return {
            "runtime_id": runtime,
            "config_id": u.actor_config_id(target) if target is not None else 0,
            "actor_type": u.target_type(target) if target is not None else None,
        }

    @staticmethod
    def _standoff_goal(anchor, distance):
        length = math.hypot(anchor[0], anchor[1])
        if length <= 1e-6:
            return anchor
        scale = float(distance) / length
        return (
            anchor[0] * (1.0 + scale),
            anchor[1] * (1.0 + scale),
        )

    @staticmethod
    def _collaboration_diagnostic(collaboration):
        attribution = collaboration.get("attribution") or {}
        return {
            "active": bool(collaboration.get("active", False)),
            "type": str(collaboration.get("type", "none")),
            "target_runtime": int(collaboration.get("target_runtime", 0)),
            "command_position": OfficialJungleProbe._rounded_or_none(
                collaboration.get("command_position")
            ),
            "cooldown_ready": bool(collaboration.get("cooldown_ready", False)),
            "micro_cd": int(collaboration.get("micro_cd", 0)),
            "max_duration": int(collaboration.get("max_duration", 0)),
            "end_frame": int(collaboration.get("end_frame", 0)),
            "remaining_frames": int(collaboration.get("remaining_frames", 0)),
            "platform_last_send_frame": int(
                collaboration.get("platform_last_send_frame", 0)
            ),
            "attribution_intent": int(attribution.get("intent", -1)),
            "attribution_intent_name": (
                COMMAND_INTENT_NAMES[int(attribution["intent"])]
                if attribution.get("intent") is not None
                and 0 <= int(attribution["intent"]) < len(COMMAND_INTENT_NAMES)
                else "none"
            ),
        }

    @staticmethod
    def _rounded_distance(value):
        if value is None or not math.isfinite(float(value)):
            return None
        return round(float(value), 1)

    @staticmethod
    def _rounded_or_none(values):
        if values is None:
            return None
        return tuple(round(float(value), 1) for value in values)


__all__ = (
    "COMMAND_ECHO_GRACE_FRAMES",
    "COMMAND_INTENT_NAMES",
    "DEER_STAGED_ROUTE",
    "DEFAULT_SCENARIO_NAMES",
    "EXPECTED_COMMAND_COOLDOWN_FRAMES",
    "EXPECTED_COMMAND_DURATION_FRAMES",
    "JUNGLE_REGIONS",
    "LINE_HERO_CONFIG_IDS",
    "MONSTER_LEASH_DISTANCE",
    "MONSTER_PULL_DISTANCE_THRESHOLDS",
    "OfficialJungleProbe",
    "OfficialJungleScenario",
    "SCENARIO_CATALOG",
    "SCENARIO_BY_NAME",
    "WAYPOINT_REACHED_DISTANCE",
    "selected_scenarios",
)
