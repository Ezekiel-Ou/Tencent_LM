#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""Extract official-model skill execution evidence from raw observations.

This module is diagnostic-only.  It records what the platform says the
official hero actually executed (``Hero.real_cmd``), then keeps command,
target, facing, successful-use, hit and projectile evidence separate so a
video symptom is not prematurely attributed to the collaboration policy.
"""

from collections import Counter, deque
import math

from agent_ppo.feature.feature_process import v1_utils as u


SKILL_COMMAND_FIELDS = (
    ("obj_skill", ("obj_skill", "objSkill")),
    ("dir_skill", ("dir_skill", "dirSkill")),
    ("pos_skill", ("pos_skill", "posSkill")),
    ("charge_skill", ("charge_skill", "chargeSkill")),
)

# Official ``CmdPkg`` is a tagged union.  The platform JSON may still contain
# default-valued dictionaries for inactive union members, so presence testing
# is not a valid discriminator.  These values follow 开发指南/数据协议.md and
# the platform CmdType enum.
COMMAND_KIND_BY_TYPE = {
    1: "move_pos",
    2: "move_dir",
    3: "stop",
    4: "attack_common",
    5: "attack_topos",
    6: "attack_actor",
    7: "obj_skill",
    8: "dir_skill",
    9: "pos_skill",
    10: "learn_skill",
    11: "buy_equip",
    12: "sell_equip",
    13: "charge_skill",
}

COMMAND_FIELD_ALIASES = {
    "move_pos": ("move_pos", "movePos"),
    "move_dir": ("move_dir", "moveDir"),
    "attack_common": ("attack_common", "attackCommon"),
    "attack_topos": ("attack_topos", "attackTopos"),
    "attack_actor": ("attack_actor", "attackActor"),
    "obj_skill": ("obj_skill", "objSkill"),
    "dir_skill": ("dir_skill", "dirSkill"),
    "pos_skill": ("pos_skill", "posSkill"),
    "learn_skill": ("learn_skill", "learnSkill"),
    "buy_equip": ("buy_equip", "buyEquip"),
    "sell_equip": ("sell_equip", "sellEquip"),
    "charge_skill": ("charge_skill", "chargeSkill"),
}

# In current official lineups these direction skills create trackable bullets.
# Other dir_skill entries are displacements or area skills and must not steal a
# nearby bullet merely because the platform bullet omits its skill id.
PROJECTILE_DIRECTION_SKILL_IDS = frozenset((13210, 14110, 19310))


def _finite_pair(value):
    if not isinstance(value, dict):
        return None
    x = u.safe_get(value, "x", default=None)
    z = u.safe_get(value, "z", default=None)
    if x is None or z is None:
        return None
    x = u.as_float(x, float("nan"))
    z = u.as_float(z, float("nan"))
    if not math.isfinite(x) or not math.isfinite(z):
        return None
    return (x, z)


def _rounded_pair(value):
    if value is None:
        return None
    return [round(float(value[0]), 3), round(float(value[1]), 3)]


def _side_vector(raw_vector, camp):
    if raw_vector is None:
        return None
    return (
        raw_vector[0] if int(camp) == 1 else -raw_vector[0],
        raw_vector[1],
    )


def _unit_vector(vector):
    if vector is None:
        return None
    length = math.hypot(vector[0], vector[1])
    if length <= 1e-6:
        return None
    return (vector[0] / length, vector[1] / length)


def _actor_descriptor(actor):
    if actor is None:
        return {
            "runtime_id": 0,
            "config_id": 0,
            "actor_type": None,
        }
    return {
        "runtime_id": u.actor_runtime_id(actor),
        "config_id": u.actor_config_id(actor),
        "actor_type": u.target_type(actor),
    }


def _position_descriptor(raw_position, camp):
    return {
        "raw": _rounded_pair(raw_position),
        "side": _rounded_pair(
            u.side_position(raw_position, int(camp))
            if raw_position is not None
            else None
        ),
    }


def _reference_geometry(origin_side, target_side, forward_side):
    if origin_side is None or target_side is None:
        return {
            "valid": False,
            "delta_side": None,
            "distance": None,
            "bearing_side_degrees": None,
            "forward_cosine": None,
            "target_is_behind_forward": None,
        }
    dx = float(target_side[0]) - float(origin_side[0])
    dz = float(target_side[1]) - float(origin_side[1])
    distance = math.hypot(dx, dz)
    target_unit = _unit_vector((dx, dz))
    forward_unit = _unit_vector(forward_side)
    cosine = (
        forward_unit[0] * target_unit[0] + forward_unit[1] * target_unit[1]
        if forward_unit is not None and target_unit is not None
        else None
    )
    return {
        "valid": True,
        "delta_side": _rounded_pair((dx, dz)),
        "distance": round(distance, 3),
        "bearing_side_degrees": round(
            math.degrees(math.atan2(dz, dx)), 3
        ),
        "forward_cosine": round(cosine, 6) if cosine is not None else None,
        "target_is_behind_forward": cosine < 0.0 if cosine is not None else None,
    }


def _angular_difference(first, second):
    if first is None or second is None:
        return None
    return round(abs((float(first) - float(second) + 180.0) % 360.0 - 180.0), 3)


def _nearby_actor_snapshots(by_runtime, official_actor, camp, limit=12):
    official_runtime = u.actor_runtime_id(official_actor)
    official_raw = u.location_of(official_actor)
    official_side = (
        u.side_position(official_raw, int(camp))
        if official_raw is not None
        else None
    )
    snapshots = []
    for runtime_id, actor in by_runtime.items():
        if not runtime_id or runtime_id == official_runtime:
            continue
        raw_position = u.location_of(actor)
        if raw_position is None:
            continue
        side_position = u.side_position(raw_position, int(camp))
        geometry = _reference_geometry(official_side, side_position, None)
        actor_camp = u.actor_camp(actor)
        relation = (
            "friendly"
            if actor_camp == int(camp)
            else "enemy"
            if actor_camp not in (0, int(camp))
            else "neutral"
        )
        snapshots.append(
            {
                "runtime_id": int(runtime_id),
                "config_id": u.actor_config_id(actor),
                "actor_type": u.target_type(actor),
                "relation": relation,
                "position": _position_descriptor(raw_position, camp),
                "distance": geometry["distance"],
                "bearing_side_degrees": geometry["bearing_side_degrees"],
                "hp_ratio": round(
                    float(
                        u.ratio(
                            u.actor_get(actor, "hp", default=0),
                            u.actor_get(actor, "max_hp", "maxHp", default=0),
                        )
                    ),
                    4,
                ),
            }
        )
    snapshots.sort(
        key=lambda item: (
            float("inf") if item["distance"] is None else item["distance"],
            item["runtime_id"],
        )
    )
    return snapshots[: max(1, int(limit))]


def _aim_assessment(
    official_side_position,
    skill_commands,
    bullets,
    references,
    by_runtime,
    camp,
):
    explicit_runtime_ids = {
        int(command.get("actor", {}).get("runtime_id", 0))
        for command in skill_commands
        if int(command.get("actor", {}).get("runtime_id", 0))
    }
    attack_runtime = int(
        references.get("attack_target", {}).get("actor", {}).get("runtime_id", 0)
    )
    target_runtime_ids = set(explicit_runtime_ids)
    if attack_runtime:
        target_runtime_ids.add(attack_runtime)

    target_positions = {}
    for runtime_id in target_runtime_ids:
        raw = u.location_of(by_runtime.get(runtime_id))
        if raw is not None:
            target_positions[runtime_id] = u.side_position(raw, int(camp))

    command_assessments = []
    for command in skill_commands:
        actor_runtime = int(command.get("actor", {}).get("runtime_id", 0))
        actor_side = target_positions.get(actor_runtime)
        destination_side = command.get("destination", {}).get("side")
        command_assessments.append(
            {
                "kind": command.get("kind"),
                "skill_id": command.get("skill_id", 0),
                "slot_index": command.get("slot_index"),
                "explicit_target_runtime_id": actor_runtime,
                "explicit_target_geometry": _reference_geometry(
                    official_side_position, actor_side, None
                ),
                "destination_geometry": _reference_geometry(
                    official_side_position, destination_side, None
                ),
                "degree_raw": command.get("degree_raw"),
                "degree_calibrated": False,
            }
        )

    return {
        "command_aims": command_assessments,
        "projectile_aims": [],
        "geometric_reverse_candidates": [],
        "reverse_candidate_is_conclusion": False,
        "candidate_rule": "only_cast_linked_initial_projectiles",
    }


def _command_payload(command, aliases):
    return u.safe_get(command, *aliases, default=None)


def _parse_real_command(command, camp, by_runtime):
    raw_type = u.as_int(
        u.safe_get(command, "command_type", "commandType", default=0),
        0,
    )
    tagged_kind = COMMAND_KIND_BY_TYPE.get(raw_type, "unknown")
    result = {
        "command_type_raw": raw_type,
        "kind": tagged_kind,
        "payload_selected_by": "command_type" if tagged_kind != "unknown" else "none",
    }

    aliases = COMMAND_FIELD_ALIASES.get(tagged_kind, ())
    payload = _command_payload(command, aliases) if aliases else None
    if tagged_kind in dict(SKILL_COMMAND_FIELDS):
        if not isinstance(payload, dict):
            result["payload_missing"] = True
            payload = {}
        result["skill_id"] = u.as_int(
            u.safe_get(payload, "skill_id", "skillID", "skillId", default=0),
            0,
        )
        raw_slot = u.safe_get(payload, "slot_type", "slotType", default=None)
        result["slot_type_raw"] = raw_slot
        result["slot_index"] = u.normalized_skill_slot(raw_slot)
        actor_runtime = u.as_int(
            u.safe_get(payload, "actor_id", "actorID", "actorId", default=0),
            0,
        )
        result["actor"] = _actor_descriptor(by_runtime.get(actor_runtime))
        if actor_runtime and result["actor"]["runtime_id"] == 0:
            result["actor"]["runtime_id"] = actor_runtime
        degree = u.safe_get(payload, "degree", default=None)
        result["degree_raw"] = (
            u.as_float(degree, 0.0) if degree is not None else None
        )
        destination = u.safe_get(
            payload, "dest_pos", "destPos", "position", default=None
        )
        result["destination"] = _position_descriptor(
            _finite_pair(destination), camp
        )
        state = u.safe_get(payload, "state", default=None)
        result["charge_state_raw"] = state
        return result

    if tagged_kind == "unknown":
        populated = []
        for kind, field_aliases in COMMAND_FIELD_ALIASES.items():
            candidate = _command_payload(command, field_aliases)
            if isinstance(candidate, dict) and any(
                value not in (None, 0, 0.0, "", False, {}, [])
                for value in candidate.values()
            ):
                populated.append(kind)
        result["populated_payload_candidates"] = populated
    return result


class OfficialSkillDirectionTracker:
    """Build per-frame skill evidence and compact episode counters."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.previous_bullets = {}
        self.bullet_first_seen = {}
        self.bullet_cast_links = {}
        self.emitted_projectile_links = set()
        self.pending_direction_casts = deque(maxlen=64)
        self.target_history = deque(maxlen=6)
        self.counts = Counter()

    def observe(self, frame_state, official_actor, camp, by_runtime, collaboration):
        frame = u.frame_number(frame_state)
        official_runtime = u.actor_runtime_id(official_actor)
        official_raw_position = u.location_of(official_actor)
        official_side_position = (
            u.side_position(official_raw_position, int(camp))
            if official_raw_position is not None
            else None
        )
        raw_forward = _finite_pair(
            u.actor_get(official_actor, "forward", default=None)
        )
        side_forward = _side_vector(raw_forward, camp)

        raw_commands = u.actor_get(
            official_actor, "real_cmd", "realCmd", default=[]
        ) or []
        commands = [
            _parse_real_command(command, camp, by_runtime)
            for command in raw_commands
            if isinstance(command, dict)
        ]
        skill_commands = [
            command
            for command in commands
            if command.get("kind") in {
                "obj_skill", "dir_skill", "pos_skill", "charge_skill"
            }
        ]

        skill_slots = []
        successful_slots = []
        for slot_index, slot in enumerate(u.skill_slots(official_actor)):
            if slot is None:
                continue
            successful = u.as_int(
                u.safe_get(
                    slot,
                    "succ_used_in_frame",
                    "succUsedInFrame",
                    default=0,
                ),
                0,
            )
            item = {
                "slot_index": slot_index,
                "slot_type_raw": u.safe_get(
                    slot, "slot_type", "slotType", default=None
                ),
                "skill_id": u.as_int(
                    u.safe_get(slot, "config_id", "configId", default=0), 0
                ),
                "usable": bool(u.safe_get(slot, "usable", default=False)),
                "cooldown": u.as_float(
                    u.safe_get(slot, "cooldown", default=0), 0
                ),
                "used_times": u.as_int(
                    u.safe_get(slot, "used_times", "usedTimes", default=0), 0
                ),
                "succ_used_in_frame": successful,
                "hit_hero_times": u.as_int(
                    u.safe_get(
                        slot, "hit_hero_times", "hitHeroTimes", default=0
                    ),
                    0,
                ),
            }
            skill_slots.append(item)
            if successful > 0:
                successful_slots.append(item)

        hits = []
        for hit in u.actor_get(
            official_actor,
            "hit_target_info",
            "hitTargetInfo",
            default=[],
        ) or []:
            target_runtime = u.as_int(
                u.safe_get(hit, "hit_target", "hitTarget", default=0), 0
            )
            hits.append(
                {
                    "target": _actor_descriptor(by_runtime.get(target_runtime)),
                    "skill_id": u.as_int(
                        u.safe_get(hit, "skill_id", "skillId", default=0), 0
                    ),
                    "slot_type_raw": u.safe_get(
                        hit, "slot_type", "slotType", default=None
                    ),
                    "slot_index": u.normalized_skill_slot(
                        u.safe_get(hit, "slot_type", "slotType", default=None)
                    ),
                    "continuous_hit_count": u.as_int(
                        u.safe_get(
                            hit,
                            "conti_hit_count",
                            "contiHitCount",
                            default=0,
                        ),
                        0,
                    ),
                }
            )
            if target_runtime and hits[-1]["target"]["runtime_id"] == 0:
                hits[-1]["target"]["runtime_id"] = target_runtime

        bullets = self._official_bullets(
            frame_state, frame, official_runtime, camp
        )
        references = self._references(
            official_side_position,
            side_forward,
            official_actor,
            collaboration,
            by_runtime,
            camp,
        )
        nearby_actors = _nearby_actor_snapshots(
            by_runtime,
            official_actor,
            camp,
        )
        aim_assessment = _aim_assessment(
            official_side_position,
            skill_commands,
            bullets,
            references,
            by_runtime,
            camp,
        )
        target_snapshot = self._target_snapshot(
            frame,
            references,
            nearby_actors,
        )
        self.target_history.append(target_snapshot)
        self._record_direction_casts(
            frame,
            camp,
            skill_commands,
            aim_assessment["command_aims"],
            target_snapshot,
        )
        projectile_links = self._link_projectiles(frame, bullets)
        aim_assessment["projectile_aims"] = projectile_links
        aim_assessment["geometric_reverse_candidates"] = [
            link
            for link in projectile_links
            if link["primary_target_alignment"]["is_reverse_candidate"]
        ]
        aim_assessment["recent_target_history"] = list(self.target_history)
        attack_runtime = int(
            references["attack_target"]["actor"].get("runtime_id", 0)
        )
        collaboration_runtime = int(
            references["collaboration_goal"]["target"].get("runtime_id", 0)
        )
        nearest_enemy_runtime = next(
            (
                int(actor["runtime_id"])
                for actor in nearby_actors
                if actor["relation"] == "enemy"
            ),
            0,
        )
        hit_runtime_ids = {
            int(hit["target"].get("runtime_id", 0))
            for hit in hits
            if int(hit["target"].get("runtime_id", 0))
        }
        dependency_signals = []
        for command in skill_commands:
            explicit_runtime = int(
                command.get("actor", {}).get("runtime_id", 0)
            )
            dependency_signals.append(
                {
                    "kind": command.get("kind"),
                    "skill_id": command.get("skill_id", 0),
                    "slot_index": command.get("slot_index"),
                    "explicit_target_runtime_id": explicit_runtime,
                    "matches_attack_target": bool(
                        explicit_runtime and explicit_runtime == attack_runtime
                    ),
                    "matches_collaboration_target": bool(
                        explicit_runtime
                        and explicit_runtime == collaboration_runtime
                    ),
                    "matches_nearest_enemy": bool(
                        explicit_runtime
                        and explicit_runtime == nearest_enemy_runtime
                    ),
                    "hit_explicit_target_this_frame": bool(
                        explicit_runtime and explicit_runtime in hit_runtime_ids
                    ),
                }
            )

        if raw_commands:
            self.counts["frames_with_real_cmd"] += 1
        if skill_commands:
            self.counts["frames_with_skill_real_cmd"] += 1
        for command in skill_commands:
            self.counts[command["kind"]] += 1
        for signal in dependency_signals:
            if signal["matches_attack_target"]:
                self.counts["skill_actor_matches_attack_target"] += 1
            if signal["matches_collaboration_target"]:
                self.counts["skill_actor_matches_collaboration_target"] += 1
            if signal["matches_nearest_enemy"]:
                self.counts["skill_actor_matches_nearest_enemy"] += 1
            if signal["hit_explicit_target_this_frame"]:
                self.counts["skill_actor_hit_same_frame"] += 1
        if successful_slots:
            self.counts["frames_with_successful_skill"] += 1
            if not skill_commands:
                self.counts["successful_skill_without_real_cmd"] += 1
        if hits:
            self.counts["frames_with_skill_hit"] += 1
        if bullets:
            self.counts["frames_with_official_bullet"] += 1
        if aim_assessment["geometric_reverse_candidates"]:
            self.counts["frames_with_geometric_reverse_candidate"] += 1

        return {
            "frame": int(frame),
            "official_runtime_id": official_runtime,
            "real_cmd_field_present": u.actor_get(
                official_actor, "real_cmd", "realCmd", default=None
            )
            is not None,
            "real_cmd_count": len(raw_commands),
            "real_cmd_raw": raw_commands,
            "commands": commands,
            "skill_commands": skill_commands,
            "skill_slots": skill_slots,
            "successful_skill_slots": successful_slots,
            "hits": hits,
            "bullets": bullets,
            "official_position": _position_descriptor(
                official_raw_position, camp
            ),
            "official_forward": {
                "raw": _rounded_pair(raw_forward),
                "side": _rounded_pair(side_forward),
                "side_unit": _rounded_pair(_unit_vector(side_forward)),
            },
            "references": references,
            "nearby_actors": nearby_actors,
            "aim_assessment": aim_assessment,
            "dependency_signals": dependency_signals,
        }

    def break_observation_continuity(self):
        """Prevent casts or bullets from linking across an unrecorded gap."""

        self.previous_bullets = {}
        self.bullet_first_seen = {}
        self.bullet_cast_links = {}
        self.pending_direction_casts.clear()
        self.target_history.clear()

    @staticmethod
    def _target_snapshot(frame, references, nearby_actors):
        attack = references["attack_target"]
        collaboration = references["collaboration_goal"]
        nearest_enemy = next(
            (
                actor
                for actor in nearby_actors
                if actor["relation"] == "enemy"
            ),
            None,
        )
        return {
            "frame": int(frame),
            "attack_target": {
                "runtime_id": int(attack["actor"].get("runtime_id", 0)),
                "config_id": int(attack["actor"].get("config_id", 0)),
                "actor_type": attack["actor"].get("actor_type"),
                "bearing_side_degrees": attack["geometry"].get(
                    "bearing_side_degrees"
                ),
                "distance": attack["geometry"].get("distance"),
            },
            "collaboration_goal": {
                "active": bool(collaboration.get("active", False)),
                "type": collaboration.get("type", "none"),
                "runtime_id": int(
                    collaboration["target"].get("runtime_id", 0)
                ),
                "bearing_side_degrees": collaboration["geometry"].get(
                    "bearing_side_degrees"
                ),
                "distance": collaboration["geometry"].get("distance"),
            },
            "nearest_enemy": {
                "runtime_id": int(nearest_enemy["runtime_id"]),
                "config_id": int(nearest_enemy["config_id"]),
                "actor_type": nearest_enemy.get("actor_type"),
                "bearing_side_degrees": nearest_enemy.get(
                    "bearing_side_degrees"
                ),
                "distance": nearest_enemy.get("distance"),
            }
            if nearest_enemy is not None
            else {
                "runtime_id": 0,
                "config_id": 0,
                "actor_type": None,
                "bearing_side_degrees": None,
                "distance": None,
            },
        }

    def _record_direction_casts(
        self,
        frame,
        camp,
        skill_commands,
        command_aims,
        target_snapshot,
    ):
        for index, command in enumerate(skill_commands):
            if command.get("kind") != "dir_skill":
                continue
            degree = command.get("degree_raw")
            if degree is None:
                continue
            raw_angle = (-float(degree) + 180.0) % 360.0 - 180.0
            side_angle = (
                raw_angle
                if int(camp) == 1
                else (180.0 - raw_angle + 180.0) % 360.0 - 180.0
            )
            history = [dict(item) for item in self.target_history]
            attack_ids = [
                int(item["attack_target"]["runtime_id"])
                for item in history
            ]
            nonzero_attack_ids = [value for value in attack_ids if value]
            command_aim = command_aims[index]
            explicit_geometry = command_aim["explicit_target_geometry"]
            self.pending_direction_casts.append(
                {
                    "cast_id": f"{int(frame)}:{index}",
                    "frame": int(frame),
                    "camp": int(camp),
                    "skill_id": int(command.get("skill_id", 0)),
                    "slot_index": command.get("slot_index"),
                    "degree_raw": float(degree),
                    "expected_projectile_angle_raw": round(raw_angle, 3),
                    "expected_projectile_angle_side": round(side_angle, 3),
                    "target_snapshot": target_snapshot,
                    "explicit_target": {
                        "runtime_id": int(
                            command_aim["explicit_target_runtime_id"]
                        ),
                        "config_id": int(
                            command.get("actor", {}).get("config_id", 0)
                        ),
                        "actor_type": command.get("actor", {}).get(
                            "actor_type"
                        ),
                        "bearing_side_degrees": explicit_geometry.get(
                            "bearing_side_degrees"
                        ),
                        "distance": explicit_geometry.get("distance"),
                    },
                    "recent_target_history": history,
                    "attack_target_changed_recently": len(
                        set(nonzero_attack_ids)
                    )
                    > 1,
                    "linked_bullet_runtime_id": 0,
                }
            )

    def _link_projectiles(self, frame, bullets):
        for bullet in bullets:
            runtime_id = int(bullet.get("runtime_id", 0))
            if not runtime_id or not bullet.get("is_new"):
                continue
            first_frame = int(bullet.get("first_observed_frame", frame))
            candidates = [
                cast
                for cast in self.pending_direction_casts
                if not cast["linked_bullet_runtime_id"]
                and 0 <= first_frame - cast["frame"] <= 12
                and (
                    (
                        bullet.get("skill_id")
                        and int(bullet["skill_id"]) == cast["skill_id"]
                    )
                    or (
                        not bullet.get("skill_id")
                        and cast["skill_id"]
                        in PROJECTILE_DIRECTION_SKILL_IDS
                    )
                )
            ]
            if not candidates:
                continue
            cast = max(candidates, key=lambda item: item["frame"])
            cast["linked_bullet_runtime_id"] = runtime_id
            self.bullet_cast_links[runtime_id] = cast

        links = []
        for bullet in bullets:
            runtime_id = int(bullet.get("runtime_id", 0))
            cast = self.bullet_cast_links.get(runtime_id)
            velocity_raw = bullet.get("velocity_per_raw_frame", {}).get("raw")
            velocity_side = bullet.get("velocity_per_raw_frame", {}).get("side")
            if (
                cast is None
                or velocity_raw is None
                or velocity_side is None
                or runtime_id in self.emitted_projectile_links
            ):
                continue
            raw_angle = math.degrees(
                math.atan2(float(velocity_raw[1]), float(velocity_raw[0]))
            )
            side_angle = math.degrees(
                math.atan2(float(velocity_side[1]), float(velocity_side[0]))
            )
            target = cast["target_snapshot"]["attack_target"]
            explicit_target = cast["explicit_target"]
            primary_target = (
                explicit_target
                if explicit_target["runtime_id"]
                else target
            )
            collaboration = cast["target_snapshot"]["collaboration_goal"]
            nearest = cast["target_snapshot"]["nearest_enemy"]
            target_difference = _angular_difference(
                side_angle, target["bearing_side_degrees"]
            )
            primary_difference = _angular_difference(
                side_angle, primary_target["bearing_side_degrees"]
            )
            collaboration_difference = _angular_difference(
                side_angle, collaboration["bearing_side_degrees"]
            )
            nearest_difference = _angular_difference(
                side_angle, nearest["bearing_side_degrees"]
            )
            execution_error = _angular_difference(
                raw_angle, cast["expected_projectile_angle_raw"]
            )
            link = {
                "cast_id": cast["cast_id"],
                "cast_frame": cast["frame"],
                "bullet_runtime_id": runtime_id,
                "bullet_first_observed_frame": bullet[
                    "first_observed_frame"
                ],
                "velocity_observed_frame": int(frame),
                "skill_id": cast["skill_id"],
                "slot_index": cast["slot_index"],
                "degree_raw": cast["degree_raw"],
                "projectile_angle_raw": round(raw_angle, 3),
                "projectile_angle_side": round(side_angle, 3),
                "degree_execution_error": execution_error,
                "attack_target_alignment": {
                    "runtime_id": target["runtime_id"],
                    "config_id": target["config_id"],
                    "actor_type": target["actor_type"],
                    "angular_difference": target_difference,
                    "is_aligned": bool(
                        target_difference is not None
                        and target_difference <= 30.0
                    ),
                    "is_reverse_candidate": bool(
                        target_difference is not None
                        and target_difference >= 120.0
                        and execution_error is not None
                        and execution_error <= 5.0
                    ),
                },
                "primary_target_alignment": {
                    "source": (
                        "explicit_skill_actor"
                        if explicit_target["runtime_id"]
                        else "attack_target"
                    ),
                    "runtime_id": primary_target["runtime_id"],
                    "config_id": primary_target["config_id"],
                    "actor_type": primary_target["actor_type"],
                    "angular_difference": primary_difference,
                    "is_aligned": bool(
                        primary_difference is not None
                        and primary_difference <= 30.0
                    ),
                    "is_reverse_candidate": bool(
                        primary_difference is not None
                        and primary_difference >= 120.0
                        and execution_error is not None
                        and execution_error <= 5.0
                    ),
                },
                "collaboration_goal_alignment": {
                    "type": collaboration["type"],
                    "runtime_id": collaboration["runtime_id"],
                    "angular_difference": collaboration_difference,
                },
                "nearest_enemy_alignment": {
                    "runtime_id": nearest["runtime_id"],
                    "config_id": nearest["config_id"],
                    "actor_type": nearest["actor_type"],
                    "angular_difference": nearest_difference,
                },
                "attack_target_changed_recently": cast[
                    "attack_target_changed_recently"
                ],
                "recent_target_history": cast["recent_target_history"],
                "reverse_candidate_is_conclusion": False,
            }
            links.append(link)
            self.emitted_projectile_links.add(runtime_id)
            self.counts["linked_initial_projectiles"] += 1
            if link["primary_target_alignment"]["is_reverse_candidate"]:
                self.counts["linked_reverse_candidates"] += 1

        while (
            self.pending_direction_casts
            and int(frame) - self.pending_direction_casts[0]["frame"] > 36
        ):
            self.pending_direction_casts.popleft()
        return links

    def _official_bullets(self, frame_state, frame, official_runtime, camp):
        active = {}
        result = []
        for bullet in u.safe_get(frame_state, "bullets", default=[]) or []:
            source_runtime = u.as_int(
                u.safe_get(
                    bullet, "source_actor", "sourceActor", default=0
                ),
                0,
            )
            if not official_runtime or source_runtime != official_runtime:
                continue
            runtime_id = u.actor_runtime_id(bullet)
            raw_position = u.location_of(bullet)
            if not runtime_id or raw_position is None:
                continue
            active[runtime_id] = (int(frame), raw_position)
            previous = self.previous_bullets.get(runtime_id)
            first_observed_frame = self.bullet_first_seen.setdefault(
                runtime_id, int(frame)
            )
            raw_velocity = None
            if previous is not None and int(frame) > previous[0]:
                elapsed = int(frame) - previous[0]
                raw_velocity = (
                    (raw_position[0] - previous[1][0]) / elapsed,
                    (raw_position[1] - previous[1][1]) / elapsed,
                )
            result.append(
                {
                    "runtime_id": runtime_id,
                    "first_observed_frame": first_observed_frame,
                    "is_new": first_observed_frame == int(frame),
                    "skill_id": u.as_int(
                        u.safe_get(bullet, "skill_id", "skillId", default=0),
                        0,
                    ),
                    "slot_type_raw": u.safe_get(
                        bullet, "slot_type", "slotType", default=None
                    ),
                    "slot_index": u.normalized_skill_slot(
                        u.safe_get(bullet, "slot_type", "slotType", default=None)
                    ),
                    "position": _position_descriptor(raw_position, camp),
                    "velocity_per_raw_frame": {
                        "raw": _rounded_pair(raw_velocity),
                        "side": _rounded_pair(_side_vector(raw_velocity, camp)),
                    },
                }
            )
        self.previous_bullets = active
        return result

    @staticmethod
    def _references(
        official_side_position,
        side_forward,
        official_actor,
        collaboration,
        by_runtime,
        camp,
    ):
        result = {}
        attack_runtime = u.as_int(
            u.actor_get(
                official_actor,
                "attack_target",
                "attackTarget",
                default=0,
            ),
            0,
        )
        attack_actor = by_runtime.get(attack_runtime)
        attack_raw = u.location_of(attack_actor)
        attack_side = (
            u.side_position(attack_raw, int(camp))
            if attack_raw is not None
            else None
        )
        result["attack_target"] = {
            "actor": _actor_descriptor(attack_actor),
            "position": _position_descriptor(attack_raw, camp),
            "geometry": _reference_geometry(
                official_side_position, attack_side, side_forward
            ),
        }
        if attack_runtime and result["attack_target"]["actor"]["runtime_id"] == 0:
            result["attack_target"]["actor"]["runtime_id"] = attack_runtime

        command_type = str(collaboration.get("type", "none"))
        command_target = collaboration.get("target_actor")
        command_target_raw = u.location_of(command_target)
        if command_target_raw is None:
            target_runtime = u.as_int(
                collaboration.get("target_runtime", 0), 0
            )
            command_target = by_runtime.get(target_runtime)
            command_target_raw = u.location_of(command_target)
        command_goal_side = (
            u.side_position(command_target_raw, int(camp))
            if command_type == "set_target" and command_target_raw is not None
            else collaboration.get("command_position")
            if command_type == "set_position"
            else None
        )
        result["collaboration_goal"] = {
            "active": bool(collaboration.get("active", False)),
            "type": command_type,
            "target": _actor_descriptor(command_target),
            "position_side": _rounded_pair(command_goal_side),
            "geometry": _reference_geometry(
                official_side_position, command_goal_side, side_forward
            ),
        }
        return result

    def summary(self):
        return dict(self.counts)


__all__ = ("OfficialSkillDirectionTracker",)
