#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""JSONL evidence recorder for official-model jungle experiments."""

from collections import Counter
import json
import math
import os
import time

from agent_ppo.feature.feature_process import v1_utils as u

from agent_diy.debug.official_jungle_probe import (
    COMMAND_ECHO_GRACE_FRAMES,
    COMMAND_INTENT_NAMES,
    JUNGLE_REGIONS,
    region_name,
)


COMMAND_NAMES = {
    0: "none",
    1: "set_target",
    2: "set_position",
    3: "unset",
}


def _safe(value):
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, dict):
        return {str(key): _safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_safe(item) for item in value]
    if hasattr(value, "item"):
        return _safe(value.item())
    if hasattr(value, "tolist"):
        return _safe(value.tolist())
    return str(value)


def _position(actor, camp):
    raw = u.location_of(actor) if actor is not None else None
    if raw is None:
        return {"raw": None, "side": None, "region": "unknown"}
    side = u.side_position(raw, int(camp))
    return {
        "raw": [round(float(raw[0]), 1), round(float(raw[1]), 1)],
        "side": [round(float(side[0]), 1), round(float(side[1]), 1)],
        "region": region_name(side),
    }


def _compact_collaboration_state(frame_state):
    state = u.safe_get(frame_state, "collaboration_state", default={}) or {}
    active = []
    for item in u.safe_get(state, "micro_state", "microState", default=[]) or []:
        raw_position = (
            u.safe_get(
                item,
                "target_position",
                "targetPosition",
                default={},
            )
            or {}
        )
        position_x = u.safe_get(raw_position, "x", default=None)
        position_z = u.safe_get(raw_position, "z", default=None)
        active.append(
            {
                "sender": u.as_int(u.safe_get(item, "sender", default=0), 0),
                "receiver": u.as_int(u.safe_get(item, "receiver", default=0), 0),
                "end_frame": u.as_int(
                    u.safe_get(item, "end_frame", "endFrame", default=0),
                    0,
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
                    "x": (float(position_x) if position_x is not None else None),
                    "z": (float(position_z) if position_z is not None else None),
                },
            }
        )
    last_send = []
    for item in u.safe_get(state, "micro_last", "microLast", default=[]) or []:
        last_send.append(
            {
                "sender": u.as_int(u.safe_get(item, "sender", default=0), 0),
                "frame": u.as_int(
                    u.safe_get(item, "frame_no", "frameNo", default=0),
                    0,
                ),
            }
        )
    return {
        "micro_cd": u.as_int(
            u.safe_get(state, "micro_cd", "microCd", default=0),
            0,
        ),
        "max_duration": u.as_int(
            u.safe_get(
                state,
                "max_duration",
                "maxDuration",
                default=0,
            ),
            0,
        ),
        "active": active,
        "last_send": last_send,
    }


class CommandEchoTracker:
    """Separate wire acceptance from official-model behavioral response."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.pending = None
        self.pending_unset = None
        self.last_accepted = None
        self.accepted_count = 0
        self.rejected_count = 0
        self.superseded_count = 0
        self.unset_accepted_count = 0
        self.unset_rejected_count = 0

    def update(self, frame, command, collaboration_state):
        command = [int(value) for value in command]
        result = {
            "status": "none",
            "observed_frame": int(frame),
            "submitted_frame": None,
            "action_type": 0,
            "target_runtime_id": 0,
            "position": [0, 0],
        }
        if self.pending is not None:
            pending = self.pending
            result.update(
                {
                    "status": "pending",
                    "submitted_frame": pending["frame"],
                    "action_type": pending["action_type"],
                    "target_runtime_id": pending["target_runtime_id"],
                    "position": list(pending["position"]),
                }
            )
            last_send = max(
                (
                    int(item.get("frame", 0))
                    for item in collaboration_state.get("last_send", ())
                ),
                default=0,
            )
            matched = any(
                self._matches(item, pending)
                for item in collaboration_state.get("active", ())
            )
            if matched and last_send >= pending["frame"]:
                result["status"] = "accepted"
                result["accepted_frame"] = int(frame)
                self.accepted_count += 1
                self.last_accepted = dict(pending)
                self.pending = None
            elif frame - pending["frame"] >= COMMAND_ECHO_GRACE_FRAMES:
                result["status"] = "rejected"
                self.rejected_count += 1
                self.pending = None

        if self.pending_unset is not None:
            pending_unset = self.pending_unset
            previous = pending_unset.get("previous")
            still_active = (
                any(
                    self._matches(item, previous)
                    for item in collaboration_state.get("active", ())
                )
                if previous is not None
                else bool(collaboration_state.get("active", ()))
            )
            result.update(
                {
                    "status": "pending",
                    "submitted_frame": pending_unset["frame"],
                    "action_type": 3,
                    "target_runtime_id": 0,
                    "position": [0, 0],
                }
            )
            if not still_active:
                result["status"] = "accepted"
                result["accepted_frame"] = int(frame)
                self.accepted_count += 1
                self.unset_accepted_count += 1
                self.last_accepted = None
                self.pending_unset = None
            elif frame - pending_unset["frame"] >= COMMAND_ECHO_GRACE_FRAMES:
                result["status"] = "rejected"
                self.rejected_count += 1
                self.unset_rejected_count += 1
                self.pending_unset = None

        if command[0] in (1, 2):
            if self.pending is not None:
                result["status"] = "superseded"
                self.superseded_count += 1
            self.pending = {
                "frame": int(frame),
                "action_type": command[0],
                "target_runtime_id": command[1],
                "position": (command[2], command[3]),
            }
            if result["status"] == "none":
                result.update(
                    {
                        "status": "submitted",
                        "submitted_frame": int(frame),
                        "action_type": command[0],
                        "target_runtime_id": command[1],
                        "position": [command[2], command[3]],
                    }
                )
        elif command[0] == 3:
            if self.pending is not None:
                self.pending = None
                self.superseded_count += 1
            if self.pending_unset is not None:
                self.superseded_count += 1
            self.pending_unset = {
                "frame": int(frame),
                "previous": (
                    dict(self.last_accepted) if self.last_accepted is not None else None
                ),
            }
            result.update(
                {
                    "status": "submitted",
                    "submitted_frame": int(frame),
                    "action_type": 3,
                    "target_runtime_id": 0,
                    "position": [0, 0],
                }
            )
        return result

    @staticmethod
    def _matches(active, pending):
        if pending["action_type"] == 1:
            return int(active.get("target_runtime_id", 0)) == int(
                pending["target_runtime_id"]
            )
        position = active.get("target_position") or {}
        if not isinstance(position, dict):
            return False
        x = position.get("x")
        z = position.get("z")
        return bool(
            x is not None
            and z is not None
            and math.hypot(
                float(x) - pending["position"][0],
                float(z) - pending["position"][1],
            )
            <= 100.0
        )

    def summary(self):
        return {
            "wire_accepted_count": self.accepted_count,
            "wire_rejected_count": self.rejected_count,
            "wire_superseded_count": self.superseded_count,
            "wire_unset_accepted_count": self.unset_accepted_count,
            "wire_unset_rejected_count": self.unset_rejected_count,
            "wire_pending_at_end": dict(self.pending or self.pending_unset or {}),
        }


class OfficialJungleTraceRecorder:
    """Write per-step facts and event-oriented summaries to one JSONL file."""

    def __init__(self, agent_type, logger=None, output_root="debug_outputs"):
        run_name = os.environ.get("DEBUG_RUN_NAME") or time.strftime(
            "official_jungle_%Y%m%d_%H%M%S"
        )
        safe_name = run_name.replace("/", "_").replace("\\", "_").strip(" .")
        output_dir = os.path.abspath(os.path.join(output_root, safe_name))
        os.makedirs(output_dir, exist_ok=True)
        suffix = f"{agent_type}_{os.getpid()}_{id(self) % 10000}"
        self.path = os.path.join(
            output_dir,
            f"official_jungle_trace_{suffix}.jsonl",
        )
        self.handle = open(self.path, "a", encoding="utf-8")
        self.logger = logger
        self.stdout_every_step = os.environ.get("DEBUG_LOG_EVERY_STEP", "0") == "1"
        self.echo = CommandEchoTracker()
        self.episode = -1
        self.closed = False
        self.last_region = "unknown"
        self.last_attack_runtime = 0
        self.event_counts = Counter()
        if self.logger:
            self.logger.info(f"DIY official jungle trace output: {self.path}")

    def start_episode(self, observation, camp, scenario):
        self.episode += 1
        self.echo.reset()
        self.last_region = "unknown"
        self.last_attack_runtime = 0
        self.event_counts = Counter()
        frame = u.frame_number(observation.get("frame_state", {}) or {})
        self._write(
            {
                "schema": "official_position_control_debug_v5",
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
        controlled_config_ids,
        proposed_command,
        actual_command,
        controlled_actions,
        policy_decision,
        feature_process,
        diagnostic,
    ):
        frame_state = observation.get("frame_state", {}) or {}
        frame = u.frame_number(frame_state)
        context = getattr(feature_process, "_memo_context", None) or {}
        official = context.get("official_hero") or {}
        official_actor = official.get("actor")
        official_position = _position(official_actor, int(camp))
        actors = list(
            u.safe_get(frame_state, "hero_states", "heroStates", default=[]) or []
        )
        monsters = list(
            u.safe_get(frame_state, "npc_states", "npcStates", default=[]) or []
        )
        by_runtime = {
            u.actor_runtime_id(actor): actor
            for actor in (*actors, *monsters)
            if u.actor_runtime_id(actor)
        }
        attack_runtime = u.as_int(
            u.actor_get(
                official_actor,
                "attack_target",
                "attack_target_runtime_id",
                "attackTarget",
                default=0,
            ),
            0,
        )
        attack_target = by_runtime.get(attack_runtime)
        attack = {
            "runtime_id": attack_runtime,
            "config_id": (
                u.actor_config_id(attack_target) if attack_target is not None else 0
            ),
            "actor_type": (
                u.target_type(attack_target) if attack_target is not None else None
            ),
        }
        collaboration_state = _compact_collaboration_state(frame_state)
        echo = self.echo.update(
            frame=frame,
            command=actual_command,
            collaboration_state=collaboration_state,
        )
        events = []
        current_region = official_position["region"]
        if current_region in JUNGLE_REGIONS and self.last_region not in JUNGLE_REGIONS:
            events.append("official_entered_jungle")
        if attack_runtime and attack_runtime != self.last_attack_runtime:
            events.append("official_attack_target_changed")
            if attack_target is not None and u.target_type(attack_target) == "monster":
                events.append("official_started_attacking_monster")
        if echo["status"] in ("accepted", "rejected", "superseded"):
            events.append("wire_command_" + echo["status"])
        for event in events:
            self.event_counts[event] += 1

        record = {
            "schema": "official_position_control_debug_v5",
            "record_type": "step",
            "episode": self.episode,
            "frame": frame,
            "camp": int(camp),
            "scenario": scenario.name,
            "controlled_config_ids": [int(value) for value in controlled_config_ids],
            "controlled_actions": {
                str(config_id): [int(value) for value in action]
                for config_id, action in controlled_actions.items()
            },
            "enemy_outer_alive": bool(context["organs"][3]["alive"]),
            "policy_decision": _safe(policy_decision),
            "proposed_command": {
                "protocol_action": [int(value) for value in proposed_command],
                "type": COMMAND_NAMES.get(int(proposed_command[0]), "unknown"),
            },
            "actual_command": {
                "protocol_action": [int(value) for value in actual_command],
                "type": COMMAND_NAMES.get(int(actual_command[0]), "unknown"),
            },
            "platform_state_before": collaboration_state,
            "wire_echo": echo,
            "official": {
                "config_id": u.actor_config_id(official_actor),
                "runtime_id": u.actor_runtime_id(official_actor),
                "alive": bool(
                    official_actor is not None and u.is_alive(official_actor)
                ),
                "hp_ratio": round(
                    float(
                        u.ratio(
                            u.actor_get(
                                official_actor,
                                "hp",
                                default=0,
                            ),
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
                "behav_mode": u.as_int(
                    u.actor_get(
                        official_actor,
                        "behav_mode",
                        "behavMode",
                        default=0,
                    ),
                    0,
                ),
                "position": official_position,
                "attack_target": attack,
            },
            "probe": _safe(diagnostic),
            "events": events,
        }
        log_line = bool(events or self.stdout_every_step)
        self._write(record, log_line=log_line)
        self.last_region = current_region
        self.last_attack_runtime = attack_runtime

    def end_episode(self, frame, probe_summary):
        self._write(
            {
                "schema": "official_position_control_debug_v5",
                "record_type": "episode_summary",
                "episode": self.episode,
                "frame": int(frame),
                "probe_summary": _safe(probe_summary),
                "wire_summary": self.echo.summary(),
                "event_counts": dict(self.event_counts),
            },
            log_line=True,
        )

    def _write(self, record, log_line):
        line = json.dumps(
            _safe(record),
            ensure_ascii=False,
            sort_keys=True,
        )
        self.handle.write(line + "\n")
        self.handle.flush()
        if log_line and self.logger:
            self.logger.info("DIY_OFFICIAL_JUNGLE " + line)

    def close(self):
        if self.closed:
            return
        self.handle.close()
        self.closed = True


def policy_command_snapshot(agent, proposed_command):
    """Capture the sampled semantic intent before a DIY wire override."""

    proposed = [int(value) for value in proposed_command]
    act_data = getattr(agent, "act_data", None)
    decision = None
    if act_data is not None:
        decision = getattr(act_data, "stochastic_command", None)
        if decision is None:
            decision = getattr(act_data, "deterministic_command", None)
    action = (
        [int(value) for value in decision.action] if decision is not None else [0, 0, 0]
    )
    intent, target_slot, position_slot = action
    obs_data = getattr(agent, "obs_data", None)
    command_context = (
        getattr(obs_data, "command_context", None) if obs_data is not None else None
    )
    selected_runtime = 0
    selected_raw_position = None
    if command_context is not None:
        if 0 <= target_slot < len(command_context.target_runtime_ids):
            selected_runtime = int(command_context.target_runtime_ids[target_slot])
        if 0 <= position_slot < len(command_context.position_raw_coordinates):
            selected_raw_position = command_context.position_raw_coordinates[
                position_slot
            ]

    feature_process = getattr(agent, "feature_processes", None)
    context = getattr(feature_process, "_memo_context", None) or {}
    selected_actor = (context.get("actor_by_runtime") or {}).get(selected_runtime)
    selected_side_position = (
        u.side_position(selected_raw_position, int(agent.hero_camp))
        if selected_raw_position is not None
        else None
    )
    return {
        "intent": intent,
        "intent_name": (
            COMMAND_INTENT_NAMES[intent]
            if 0 <= intent < len(COMMAND_INTENT_NAMES)
            else f"unknown_{intent}"
        ),
        "target_slot": target_slot,
        "position_slot": position_slot,
        "selected_target_runtime_id": selected_runtime,
        "selected_target_config_id": (
            u.actor_config_id(selected_actor) if selected_actor is not None else 0
        ),
        "selected_position_raw": (
            [round(float(value), 1) for value in selected_raw_position]
            if selected_raw_position is not None
            else None
        ),
        "selected_position_side": (
            [round(float(value), 1) for value in selected_side_position]
            if selected_side_position is not None
            else None
        ),
        "selected_position_region": region_name(selected_side_position),
        "protocol_action": proposed,
    }


__all__ = (
    "CommandEchoTracker",
    "OfficialJungleTraceRecorder",
    "policy_command_snapshot",
)
