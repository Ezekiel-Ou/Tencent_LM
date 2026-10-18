#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""Official-platform recorder for official-model lane skill attribution."""

from collections import Counter
from dataclasses import dataclass
import json
import math
import os
import time

from agent_ppo.feature.feature_process import v1_utils as u

from agent_ppo_debug.debug.official_skill_direction import (
    OfficialSkillDirectionTracker,
)


HERO_NAMES = {132: "marco", 141: "diaochan", 193: "kai"}
NONE_COMMAND = [0, 0, 0, 0]
COMMAND_NAMES = {0: "none", 1: "set_target", 2: "set_position", 3: "unset"}
LANE_REGIONS = frozenset(
    (
        "own_tower_zone",
        "own_lane",
        "mid_lane",
        "enemy_lane",
        "enemy_tower_zone",
    )
)


@dataclass(frozen=True)
class LaneSkillScenario:
    name: str
    official_config_id: int


SCENARIOS = tuple(
    LaneSkillScenario(f"lane_{name}", config_id)
    for config_id, name in HERO_NAMES.items()
)
SCENARIO_BY_NAME = {scenario.name: scenario for scenario in SCENARIOS}


def selected_scenarios(value=None):
    raw = os.environ.get("DEBUG_SCENARIOS", "") if value is None else value
    names = [part.strip() for part in str(raw).split(",") if part.strip()]
    if not names:
        return SCENARIOS
    aliases = {
        "baseline_marco": "lane_marco",
        "baseline_diaochan": "lane_diaochan",
        "baseline_kai": "lane_kai",
    }
    names = [aliases.get(name, name) for name in names]
    unknown = [name for name in names if name not in SCENARIO_BY_NAME]
    if unknown:
        raise ValueError(
            "unknown DEBUG_SCENARIOS: "
            + ", ".join(unknown)
            + "; valid values: "
            + ", ".join(SCENARIO_BY_NAME)
        )
    return tuple(SCENARIO_BY_NAME[name] for name in names)


def _safe(value):
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(key): _safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_safe(item) for item in value]
    if hasattr(value, "item"):
        try:
            return _safe(value.item())
        except (TypeError, ValueError):
            pass
    return str(value)


def _compact_collaboration_state(frame_state):
    state = u.safe_get(frame_state, "collaboration_state", default={}) or {}
    active = []
    for item in u.safe_get(state, "micro_state", "microState", default=[]) or []:
        position = u.safe_get(
            item, "target_position", "targetPosition", default={}
        ) or {}
        active.append(
            {
                "sender": u.as_int(u.safe_get(item, "sender", default=0), 0),
                "receiver": u.as_int(u.safe_get(item, "receiver", default=0), 0),
                "end_frame": u.as_int(
                    u.safe_get(item, "end_frame", "endFrame", default=0), 0
                ),
                "target_runtime_id": u.as_int(
                    u.safe_get(
                        item,
                        "target_runtime_id",
                        "targetRuntimeId",
                        default=0,
                    ),
                    0,
                ),
                "target_position": {
                    "x": u.safe_get(position, "x", default=None),
                    "z": u.safe_get(position, "z", default=None),
                },
            }
        )
    return {
        "micro_cd": u.as_int(
            u.safe_get(state, "micro_cd", "microCd", default=0), 0
        ),
        "max_duration": u.as_int(
            u.safe_get(state, "max_duration", "maxDuration", default=0), 0
        ),
        "active": active,
        "last_send": [
            {
                "sender": u.as_int(u.safe_get(item, "sender", default=0), 0),
                "frame": u.as_int(
                    u.safe_get(item, "frame_no", "frameNo", default=0), 0
                ),
            }
            for item in u.safe_get(
                state, "micro_last", "microLast", default=[]
            )
            or []
        ],
    }


def policy_command_snapshot(agent, proposed_command):
    proposed = [int(value) for value in proposed_command]
    act_data = getattr(agent, "act_data", None)
    decision = None
    if act_data is not None:
        decision = getattr(act_data, "stochastic_command", None)
        if decision is None:
            decision = getattr(act_data, "deterministic_command", None)
    action = (
        [int(value) for value in decision.action]
        if decision is not None
        else [0, 0, 0]
    )
    obs_data = getattr(agent, "obs_data", None)
    command_context = (
        getattr(obs_data, "command_context", None) if obs_data is not None else None
    )
    target_runtime = 0
    raw_position = None
    if command_context is not None:
        target_slot = action[1]
        position_slot = action[2]
        if 0 <= target_slot < len(command_context.target_runtime_ids):
            target_runtime = int(command_context.target_runtime_ids[target_slot])
        if 0 <= position_slot < len(command_context.position_raw_coordinates):
            raw_position = command_context.position_raw_coordinates[position_slot]
    return {
        "semantic_action": action,
        "selected_target_runtime_id": target_runtime,
        "selected_position_raw": (
            [round(float(value), 1) for value in raw_position]
            if raw_position is not None
            else None
        ),
        "protocol_action": proposed,
    }


class OfficialLaneSkillTraceRecorder:
    """Write the complete factors around official-model lane skill casts."""

    def __init__(self, agent_type, logger=None, output_root="debug_outputs"):
        run_name = os.environ.get("DEBUG_RUN_NAME") or time.strftime(
            "official_lane_skill_%Y%m%d_%H%M%S"
        )
        safe_name = run_name.replace("/", "_").replace("\\", "_").strip(" .")
        output_dir = os.path.abspath(os.path.join(output_root, safe_name))
        os.makedirs(output_dir, exist_ok=True)
        suffix = f"{agent_type}_{os.getpid()}_{id(self) % 10000}"
        self.path = os.path.join(
            output_dir, f"official_lane_skill_trace_{suffix}.jsonl"
        )
        self.handle = open(self.path, "a", encoding="utf-8")
        self.logger = logger
        self.tracker = OfficialSkillDirectionTracker()
        self.episode = -1
        self.event_counts = Counter()
        self.skipped_non_lane_steps = 0
        self.closed = False
        if logger:
            logger.info(f"DIY official lane skill trace output: {self.path}")

    def start_episode(self, observation, camp, scenario):
        self.episode += 1
        self.tracker.reset()
        self.event_counts = Counter()
        self.skipped_non_lane_steps = 0
        frame = u.frame_number(observation.get("frame_state", {}) or {})
        self._write(
            {
                "schema": "official_lane_skill_direction_v1",
                "record_type": "episode_start",
                "episode": self.episode,
                "frame": frame,
                "camp": int(camp),
                "scenario": _safe(scenario.__dict__),
            },
            log_line=True,
        )

    def record_step(
        self,
        observation,
        camp,
        scenario,
        proposed_command,
        actual_command,
        controlled_actions,
        policy_decision,
        feature_process,
    ):
        frame_state = observation.get("frame_state", {}) or {}
        frame = u.frame_number(frame_state)
        context = getattr(feature_process, "_memo_context", None) or {}
        official = context.get("official_hero") or {}
        official_actor = official.get("actor")
        official_raw_position = u.location_of(official_actor)
        official_side_position = (
            u.side_position(official_raw_position, int(camp))
            if official_raw_position is not None
            else None
        )
        region_index = u.semantic_region(official_side_position)
        region = (
            u.REGION_NAMES[region_index]
            if 0 <= region_index < len(u.REGION_NAMES)
            else "unknown"
        )
        if region not in LANE_REGIONS:
            self.tracker.break_observation_continuity()
            self.skipped_non_lane_steps += 1
            return
        by_runtime = dict(context.get("actor_by_runtime") or {})
        evidence = self.tracker.observe(
            frame_state=frame_state,
            official_actor=official_actor,
            camp=camp,
            by_runtime=by_runtime,
            collaboration=context.get("collaboration") or {},
        )
        events = []
        if evidence["skill_commands"]:
            events.append("official_skill_real_cmd")
        if evidence["successful_skill_slots"]:
            events.append("official_skill_success")
        if evidence["hits"]:
            events.append("official_skill_hit")
        if evidence["bullets"]:
            events.append("official_projectile_visible")
        if evidence["aim_assessment"]["geometric_reverse_candidates"]:
            events.append("geometric_reverse_candidate")
        for event in events:
            self.event_counts[event] += 1

        record = {
            "schema": "official_lane_skill_direction_v1",
            "record_type": "step",
            "episode": self.episode,
            "frame": frame,
            "camp": int(camp),
            "scenario": scenario.name,
            "policy_decision": _safe(policy_decision),
            "proposed_command": {
                "protocol_action": [int(value) for value in proposed_command],
                "type": COMMAND_NAMES.get(int(proposed_command[0]), "unknown"),
            },
            "actual_command": {
                "protocol_action": [int(value) for value in actual_command],
                "type": COMMAND_NAMES.get(int(actual_command[0]), "unknown"),
            },
            "platform_collaboration_state": _compact_collaboration_state(frame_state),
            "controlled_actions": {
                str(config_id): [int(value) for value in action]
                for config_id, action in controlled_actions.items()
            },
            "official": {
                "config_id": u.actor_config_id(official_actor),
                "runtime_id": u.actor_runtime_id(official_actor),
                "alive": bool(official_actor is not None and u.is_alive(official_actor)),
                "behav_mode": u.as_int(
                    u.actor_get(
                        official_actor, "behav_mode", "behavMode", default=0
                    ),
                    0,
                ),
                "region": region,
            },
            "skill_direction": evidence,
            "events": events,
        }
        self._write(record, log_line=bool(events))

    def end_episode(self, frame):
        self._write(
            {
                "schema": "official_lane_skill_direction_v1",
                "record_type": "episode_summary",
                "episode": self.episode,
                "frame": int(frame),
                "skill_direction_summary": self.tracker.summary(),
                "event_counts": dict(self.event_counts),
                "skipped_non_lane_steps": self.skipped_non_lane_steps,
            },
            log_line=True,
        )

    def _write(self, record, log_line=False):
        line = json.dumps(_safe(record), ensure_ascii=False, sort_keys=True)
        self.handle.write(line + "\n")
        self.handle.flush()
        if log_line and self.logger:
            self.logger.info("DIY_OFFICIAL_LANE_SKILL " + line)

    def close(self):
        if not self.closed:
            self.handle.close()
            self.closed = True


__all__ = (
    "LaneSkillScenario",
    "OfficialLaneSkillTraceRecorder",
    "policy_command_snapshot",
    "selected_scenarios",
)
