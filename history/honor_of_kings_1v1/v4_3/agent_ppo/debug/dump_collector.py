"""
DumpCollector: range/cardinality tracker for agent_ppo Phase A debug runs.

Purpose: feed every frame's observation in, accumulate distinct values,
min/max, and histograms; periodically dump full observations to per-frame
JSONs; write summary.json at the end. The summary is the artifact the
human pastes back, and is what Phase B feature-engineering numbers will
be calibrated against.

Defensive about field layout: tries flat (`hero["camp"]`, the 2026
official baseline) first, falls back to nested (`hero["actor_state"]["camp"]`,
the hok_semi 2025 layout). The detected layout is recorded in summary
under `protocol_check.hero_state_layout`.
"""

from __future__ import annotations

import json
import os
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional, Tuple


KNOWN_ACTOR_SUBTYPES = {
    (0, 0): "hero",
    (1, 0): "monster",
    (1, 11): "soldier",
    (2, 21): "tower",
    (2, 23): "crystal",
    (2, 24): "spring",
}

STRING_ACTOR_TYPES = {
    "ACTOR_HERO": 0,
    "ACTOR_MONSTER": 1,
    "ACTOR_ORGAN": 2,
    "ACTOR_TYPE_HERO": 0,
    "ACTOR_TYPE_MONSTER": 1,
    "ACTOR_TYPE_ORGAN": 2,
}

STRING_SUB_TYPES = {
    "ACTOR_SUB_NONE": 0,
    "ACTOR_SUB_MONSTER": 0,
    "ACTOR_SUB_SOLDIER": 11,
    "ACTOR_SUB_TOWER": 21,
    "ACTOR_SUB_CRYSTAL": 23,
    "ACTOR_SUB_TOWER_SPRING": 24,
    "ACTOR_SUB_SPRING": 24,
}

TRUSTED_COORD_KINDS = ("hero", "soldier", "tower", "crystal", "spring", "monster", "cake")
COORD_OUTLIER_ABS_THRESHOLD = 60000


def _get_any(d: Any, keys: List[str], default=None):
    if not isinstance(d, dict):
        return default
    for key in keys:
        if key in d:
            return d[key]
    return default


def _hero_field(hero: dict, key: str, alt_keys: Optional[List[str]] = None, default=None):
    """Read a hero field that may live at the flat root or nested under actor_state."""
    candidates = [key] + (alt_keys or [])
    for k in candidates:
        if k in hero:
            return hero[k]
    actor_state = hero.get("actor_state") or hero.get("actorState")
    if isinstance(actor_state, dict):
        for k in candidates:
            if k in actor_state:
                return actor_state[k]
    return default


def _npc_field(npc: dict, key: str, alt_keys: Optional[List[str]] = None, default=None):
    return _hero_field(npc, key, alt_keys=alt_keys, default=default)


def _field_or_values(obj: dict, key: str, alt_keys: Optional[List[str]] = None, default=None):
    value = _hero_field(obj, key, alt_keys=alt_keys, default=None)
    if value is not None:
        return value
    values = _hero_field(obj, "values")
    if isinstance(values, dict):
        candidates = [key] + (alt_keys or [])
        for candidate in candidates:
            if candidate in values:
                return values[candidate]
    return default


def _coerce_int_or_none(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _enum_int(value, mapping: Optional[Dict[str, int]] = None):
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str):
        text = value.strip()
        if mapping and text in mapping:
            return mapping[text]
        try:
            return int(text)
        except ValueError:
            return None
    return None


def _read_location(obj: dict):
    loc = _hero_field(obj, "location")
    if not isinstance(loc, dict):
        return None, None, None
    return (
        _coerce_int_or_none(loc.get("x")),
        _coerce_int_or_none(loc.get("y")),
        _coerce_int_or_none(loc.get("z")),
    )


def _bounded_append(items: list, value: dict, limit: int) -> None:
    if len(items) < limit:
        items.append(value)


def _money_bucket(delta: int) -> str:
    if delta is None:
        return "unknown"
    if delta < 0:
        return "negative"
    if delta == 0:
        return "0"
    if delta < 20:
        return "1-19"
    if delta < 50:
        return "20-49"
    if delta < 100:
        return "50-99"
    if delta < 200:
        return "100-199"
    if delta < 500:
        return "200-499"
    return ">=500"


def _add_limited_tuple(bucket: Dict[Any, set], key: Any, values, limit: int = 32) -> None:
    """Keep bounded unique tuple samples for masks/flag vectors."""
    try:
        tup = tuple(values)
    except TypeError:
        return
    if len(bucket[key]) < limit or tup in bucket[key]:
        bucket[key].add(tup)


class _Range:
    __slots__ = ("min", "max", "count")

    def __init__(self):
        self.min = None
        self.max = None
        self.count = 0

    def update(self, v):
        if v is None:
            return
        try:
            v = float(v)
        except (TypeError, ValueError):
            return
        if self.min is None or v < self.min:
            self.min = v
        if self.max is None or v > self.max:
            self.max = v
        self.count += 1

    def to_dict(self):
        return {"min": self.min, "max": self.max, "count": self.count}


class _CoordAcc:
    """Coordinate tracker split by entity source to diagnose map-axis remapping."""

    def __init__(self, outlier_abs_threshold: int = COORD_OUTLIER_ABS_THRESHOLD):
        self.x = _Range()
        self.y = _Range()
        self.z = _Range()
        self.long_plus = _Range()   # x + z, diagonal candidate
        self.long_minus = _Range()  # x - z, opposite diagonal candidate
        self.width_plus = _Range()  # x - z if long axis is x+z
        self.width_minus = _Range() # x + z if long axis is x-z
        self.count = 0
        self.first_frame = None
        self.last_frame = None
        self.samples = []
        self.outlier_samples = []
        self._outlier_abs_threshold = outlier_abs_threshold

    def update(self, x, y, z, frame_no=None, meta: Optional[dict] = None) -> None:
        if x is None or z is None:
            return
        self.count += 1
        self.x.update(x)
        self.y.update(y)
        self.z.update(z)
        self.long_plus.update(x + z)
        self.long_minus.update(x - z)
        self.width_plus.update(x - z)
        self.width_minus.update(x + z)
        if frame_no is not None:
            if self.first_frame is None or frame_no < self.first_frame:
                self.first_frame = int(frame_no)
            if self.last_frame is None or frame_no > self.last_frame:
                self.last_frame = int(frame_no)
        sample = {"frame": frame_no, "x": x, "y": y, "z": z}
        if meta:
            sample.update({k: v for k, v in meta.items() if v is not None})
        _bounded_append(self.samples, sample, 16)
        if abs(x) > self._outlier_abs_threshold or abs(z) > self._outlier_abs_threshold:
            _bounded_append(self.outlier_samples, sample, 64)

    def merge_from(self, other: "_CoordAcc") -> None:
        for rname in ("x", "y", "z", "long_plus", "long_minus", "width_plus", "width_minus"):
            r = getattr(other, rname)
            getattr(self, rname).update(r.min)
            getattr(self, rname).update(r.max)
        self.count += other.count
        if other.first_frame is not None:
            if self.first_frame is None or other.first_frame < self.first_frame:
                self.first_frame = other.first_frame
        if other.last_frame is not None:
            if self.last_frame is None or other.last_frame > self.last_frame:
                self.last_frame = other.last_frame
        for item in other.samples:
            _bounded_append(self.samples, item, 16)
        for item in other.outlier_samples:
            _bounded_append(self.outlier_samples, item, 64)

    @staticmethod
    def _span(r: _Range):
        if r.min is None or r.max is None:
            return None
        return r.max - r.min

    def to_dict(self):
        plus_span = self._span(self.long_plus)
        minus_span = self._span(self.long_minus)
        if plus_span is None and minus_span is None:
            suggested_axis = "unknown"
        elif minus_span is None or (plus_span is not None and plus_span >= minus_span):
            suggested_axis = "x_plus_z"
        else:
            suggested_axis = "x_minus_z"
        return {
            "count": self.count,
            "first_frame": self.first_frame,
            "last_frame": self.last_frame,
            "x": self.x.to_dict(),
            "y": self.y.to_dict(),
            "z": self.z.to_dict(),
            "diag_x_plus_z": self.long_plus.to_dict(),
            "diag_x_minus_z": self.long_minus.to_dict(),
            "span_x_plus_z": plus_span,
            "span_x_minus_z": minus_span,
            "suggested_long_axis": suggested_axis,
            "samples": list(self.samples),
            "outlier_samples_abs_gt_60000": list(self.outlier_samples),
        }


class _BuffAcc:
    """Structured buff/mark tracker grouped by holder and config id."""

    def __init__(self):
        self.count = 0
        self.first_frame = None
        self.last_frame = None
        self.times = _Range()
        self.start_time = _Range()
        self.layer = _Range()
        self.origin_actor_ids = set()
        self.holder_runtime_ids = set()
        self.samples = []

    def update(self, frame_no=None, holder_runtime_id=None, times=None,
               start_time=None, layer=None, origin_actor_id=None, sample=None) -> None:
        self.count += 1
        if frame_no is not None:
            frame_no = int(frame_no)
            if self.first_frame is None or frame_no < self.first_frame:
                self.first_frame = frame_no
            if self.last_frame is None or frame_no > self.last_frame:
                self.last_frame = frame_no
        if holder_runtime_id is not None:
            self.holder_runtime_ids.add(holder_runtime_id)
        if origin_actor_id is not None:
            self.origin_actor_ids.add(origin_actor_id)
        self.times.update(times)
        self.start_time.update(start_time)
        self.layer.update(layer)
        if sample:
            _bounded_append(self.samples, sample, 12)

    def to_dict(self):
        return {
            "count": self.count,
            "first_frame": self.first_frame,
            "last_frame": self.last_frame,
            "times": self.times.to_dict(),
            "start_time": self.start_time.to_dict(),
            "layer": self.layer.to_dict(),
            "origin_actor_ids_seen": sorted(self.origin_actor_ids),
            "holder_runtime_ids_seen": sorted(self.holder_runtime_ids),
            "samples": list(self.samples),
        }


class _HeroAcc:
    """Per-hero-config_id accumulator."""

    def __init__(self):
        self.location_x = _Range()
        self.location_z = _Range()
        self.forward_x = _Range()
        self.forward_z = _Range()
        self.hp = _Range()
        self.max_hp = _Range()
        self.ep = _Range()
        self.max_ep = _Range()
        self.attack_range = _Range()
        self.attack_target_seen = set()
        self.kill_income = _Range()
        self.sight_area = _Range()
        self.mov_spd = _Range()
        self.atk_spd = _Range()
        self.phy_atk = _Range()
        self.phy_def = _Range()
        self.mgc_atk = _Range()
        self.mgc_def = _Range()
        self.phy_armor_hurt = _Range()
        self.mgc_armor_hurt = _Range()
        self.crit_rate = _Range()
        self.crit_effe = _Range()
        self.phy_vamp = _Range()
        self.mgc_vamp = _Range()
        self.cd_reduce = _Range()
        self.ctrl_reduce = _Range()
        self.level = _Range()
        self.exp = _Range()
        self.money = _Range()
        self.frame_money = _Range()
        self.kill_cnt = _Range()
        self.dead_cnt = _Range()
        self.assist_cnt = _Range()
        self.revive_time = _Range()
        self.hp_recover = _Range()
        self.ep_recover = _Range()
        self.is_in_grass_seen = set()
        self.behave_seen = set()
        self.passive_skill_ids = set()
        self.buff_skill_ids = set()
        self.buff_mark_id_to_max_layer = {}
        self.summoner_skill_ids = set()
        self.abilities_lengths_seen = set()
        self.abilities_patterns_seen = set()
        self.camp_visible_patterns_seen = set()
        self.max_hp_by_level = {}
        self.max_ep_by_level = {}
        self.skill_cooldown_max_per_slot = {}
        self.skill_cooldown_cur_per_slot = {}
        self.skill_level_per_slot = {}
        self.skill_usable_per_slot = defaultdict(set)
        self.skill_config_ids_per_slot = defaultdict(set)
        self.skill_used_times_per_slot = {}
        self.skill_hit_hero_times_per_slot = {}
        self.skill_succ_used_in_frame_per_slot = {}
        self.skill_next_config_ids_per_slot = defaultdict(set)
        self.skill_combo_effect_time_per_slot = {}
        self.hit_target_slot_counts = defaultdict(int)
        self.hit_target_skill_ids = set()
        self.hit_target_runtime_ids = set()
        self.hit_target_conti_hit_count = _Range()
        self.take_hurt_value = _Range()
        self.take_hurt_atkers = set()
        self.take_hurt_skill_slots = set()
        self.take_hurt_source_types = set()
        self.take_hurt_source_ids = set()
        self.real_cmd_types = defaultdict(int)
        self.real_cmd_keys_seen = set()
        self.real_cmd_skill_ids = set()
        self.real_cmd_slot_types = set()
        self.real_cmd_actor_ids = set()
        self.equip_config_ids = set()
        self.equip_buy_price = _Range()
        self.equip_amount = _Range()
        self.equip_active_skill_ids = set()
        self.equip_active_cooldown = _Range()
        self.equip_passive_skill_ids = set()
        self.equip_passive_cooldown = _Range()
        self.money_delta_histogram = defaultdict(int)
        self._last_money_total = None

    def to_dict(self):
        return {
            "location_x": self.location_x.to_dict(),
            "location_z": self.location_z.to_dict(),
            "forward_x": self.forward_x.to_dict(),
            "forward_z": self.forward_z.to_dict(),
            "hp": self.hp.to_dict(),
            "max_hp": self.max_hp.to_dict(),
            "ep": self.ep.to_dict(),
            "max_ep": self.max_ep.to_dict(),
            "attack_range": self.attack_range.to_dict(),
            "attack_target_values_seen": sorted(self.attack_target_seen, key=str),
            "kill_income": self.kill_income.to_dict(),
            "sight_area": self.sight_area.to_dict(),
            "mov_spd": self.mov_spd.to_dict(),
            "atk_spd": self.atk_spd.to_dict(),
            "phy_atk": self.phy_atk.to_dict(),
            "phy_def": self.phy_def.to_dict(),
            "mgc_atk": self.mgc_atk.to_dict(),
            "mgc_def": self.mgc_def.to_dict(),
            "phy_armor_hurt": self.phy_armor_hurt.to_dict(),
            "mgc_armor_hurt": self.mgc_armor_hurt.to_dict(),
            "crit_rate": self.crit_rate.to_dict(),
            "crit_effe": self.crit_effe.to_dict(),
            "phy_vamp": self.phy_vamp.to_dict(),
            "mgc_vamp": self.mgc_vamp.to_dict(),
            "cd_reduce": self.cd_reduce.to_dict(),
            "ctrl_reduce": self.ctrl_reduce.to_dict(),
            "level": self.level.to_dict(),
            "money_total": self.money.to_dict(),
            "money_per_frame_field": self.frame_money.to_dict(),
            "kill_cnt": self.kill_cnt.to_dict(),
            "dead_cnt": self.dead_cnt.to_dict(),
            "assist_cnt": self.assist_cnt.to_dict(),
            "revive_time": self.revive_time.to_dict(),
            "exp": self.exp.to_dict(),
            "hp_recover": self.hp_recover.to_dict(),
            "ep_recover": self.ep_recover.to_dict(),
            "is_in_grass_values_seen": sorted(self.is_in_grass_seen, key=str),
            "behave_seen": sorted(self.behave_seen, key=str),
            "passive_skill_ids": sorted(self.passive_skill_ids),
            "buff_skill_ids_seen": sorted(self.buff_skill_ids),
            "buff_mark_id_to_max_layer": {str(k): v for k, v in self.buff_mark_id_to_max_layer.items()},
            "summoner_skill_ids_seen": sorted(self.summoner_skill_ids),
            "abilities_lengths_seen": sorted(self.abilities_lengths_seen),
            "abilities_patterns_seen": [list(v) for v in sorted(self.abilities_patterns_seen)],
            "camp_visible_patterns_seen": [list(v) for v in sorted(self.camp_visible_patterns_seen)],
            "max_hp_by_level": {str(k): v for k, v in sorted(self.max_hp_by_level.items())},
            "max_ep_by_level": {str(k): v for k, v in sorted(self.max_ep_by_level.items())},
            "skill_cooldown_max_per_slot": {
                str(k): v.to_dict() for k, v in sorted(self.skill_cooldown_max_per_slot.items())
            },
            "skill_cooldown_cur_per_slot": {
                str(k): v.to_dict() for k, v in sorted(self.skill_cooldown_cur_per_slot.items())
            },
            "skill_level_per_slot": {
                str(k): v.to_dict() for k, v in sorted(self.skill_level_per_slot.items())
            },
            "skill_usable_per_slot": {
                str(k): sorted(v, key=str) for k, v in sorted(self.skill_usable_per_slot.items())
            },
            "skill_config_ids_per_slot": {
                str(k): sorted(v) for k, v in sorted(self.skill_config_ids_per_slot.items())
            },
            "skill_used_times_per_slot": {
                str(k): v.to_dict() for k, v in sorted(self.skill_used_times_per_slot.items())
            },
            "skill_hit_hero_times_per_slot": {
                str(k): v.to_dict() for k, v in sorted(self.skill_hit_hero_times_per_slot.items())
            },
            "skill_succ_used_in_frame_per_slot": {
                str(k): v.to_dict() for k, v in sorted(self.skill_succ_used_in_frame_per_slot.items())
            },
            "skill_next_config_ids_per_slot": {
                str(k): sorted(v) for k, v in sorted(self.skill_next_config_ids_per_slot.items())
            },
            "skill_combo_effect_time_per_slot": {
                str(k): v.to_dict() for k, v in sorted(self.skill_combo_effect_time_per_slot.items())
            },
            "hit_target_slot_counts": dict(self.hit_target_slot_counts),
            "hit_target_skill_ids_seen": sorted(self.hit_target_skill_ids),
            "hit_target_runtime_ids_seen": sorted(self.hit_target_runtime_ids),
            "hit_target_conti_hit_count": self.hit_target_conti_hit_count.to_dict(),
            "take_hurt_value": self.take_hurt_value.to_dict(),
            "take_hurt_atkers_seen": sorted(self.take_hurt_atkers),
            "take_hurt_skill_slots_seen": sorted(self.take_hurt_skill_slots),
            "take_hurt_source_types_seen": sorted(self.take_hurt_source_types),
            "take_hurt_source_ids_seen": sorted(self.take_hurt_source_ids),
            "real_cmd_types": dict(self.real_cmd_types),
            "real_cmd_keys_seen": sorted(self.real_cmd_keys_seen),
            "real_cmd_skill_ids_seen": sorted(self.real_cmd_skill_ids),
            "real_cmd_slot_types_seen": sorted(self.real_cmd_slot_types),
            "real_cmd_actor_ids_seen": sorted(self.real_cmd_actor_ids),
            "equip_config_ids_seen": sorted(self.equip_config_ids),
            "equip_buy_price": self.equip_buy_price.to_dict(),
            "equip_amount": self.equip_amount.to_dict(),
            "equip_active_skill_ids_seen": sorted(self.equip_active_skill_ids),
            "equip_active_cooldown": self.equip_active_cooldown.to_dict(),
            "equip_passive_skill_ids_seen": sorted(self.equip_passive_skill_ids),
            "equip_passive_cooldown": self.equip_passive_cooldown.to_dict(),
            "money_delta_histogram": dict(self.money_delta_histogram),
        }


class _HeroCampDebugAcc:
    """Compact per hero-config + camp tracker for final feature schema decisions."""

    def __init__(self):
        self.count = 0
        self.first_frame = None
        self.last_frame = None
        self.runtime_ids = set()
        self.hp = _Range()
        self.max_hp = _Range()
        self.level = _Range()
        self.exp = _Range()
        self.money_total = _Range()
        self.money_frame = _Range()
        self.money_delta_histogram = defaultdict(int)
        self._last_money_total = None
        self.abilities_lengths_seen = set()
        self.abilities_patterns_seen = set()
        self.camp_visible_patterns_seen = set()
        self.skill_slot_types_per_slot = defaultdict(set)
        self.skill_config_ids_per_slot = defaultdict(set)
        self.skill_cooldown_cur_per_slot = {}
        self.skill_cooldown_max_per_slot = {}
        self.skill_level_per_slot = {}
        self.skill_usable_per_slot = defaultdict(set)
        self.summoner_slot_candidates = defaultdict(set)

    def update(self, hero: dict, frame_no: Optional[int] = None) -> None:
        self.count += 1
        if frame_no is not None:
            frame_no = int(frame_no)
            if self.first_frame is None or frame_no < self.first_frame:
                self.first_frame = frame_no
            if self.last_frame is None or frame_no > self.last_frame:
                self.last_frame = frame_no

        runtime_id = _coerce_int_or_none(_hero_field(hero, "runtime_id", ["runtimeId"]))
        if runtime_id is not None:
            self.runtime_ids.add(runtime_id)

        self.hp.update(_coerce_int_or_none(_hero_field(hero, "hp")))
        self.max_hp.update(_coerce_int_or_none(_hero_field(hero, "max_hp", ["maxHp", "max_Hp"])))
        self.level.update(_coerce_int_or_none(_hero_field(hero, "level")))
        self.exp.update(_coerce_int_or_none(_hero_field(hero, "exp")))

        money_total = _coerce_int_or_none(_hero_field(hero, "moneyCnt", ["money_cnt", "money_total"]))
        money_now = _coerce_int_or_none(_hero_field(hero, "money"))
        self.money_total.update(money_total)
        self.money_frame.update(money_now)
        if money_total is not None:
            if self._last_money_total is None:
                self._last_money_total = money_total
            else:
                delta = money_total - self._last_money_total
                self.money_delta_histogram[_money_bucket(delta)] += 1
                self._last_money_total = money_total

        abilities = _hero_field(hero, "abilities")
        if isinstance(abilities, (list, tuple)):
            self.abilities_lengths_seen.add(len(abilities))
            _add_limited_tuple({0: self.abilities_patterns_seen}, 0, [int(bool(v)) for v in abilities], limit=16)
        camp_visible = _hero_field(hero, "camp_visible", ["campVisible"])
        if isinstance(camp_visible, (list, tuple)):
            _add_limited_tuple({0: self.camp_visible_patterns_seen}, 0, [int(bool(v)) for v in camp_visible], limit=16)

        skill_state = _hero_field(hero, "skill_state", ["skillState"])
        if not isinstance(skill_state, dict):
            return
        slot_states = skill_state.get("slot_states") or skill_state.get("slotStates") or []
        for idx, slot in enumerate(slot_states):
            if not isinstance(slot, dict):
                continue
            slot_type = _get_any(slot, ["slot_type", "slotType"])
            if slot_type is not None:
                self.skill_slot_types_per_slot[idx].add(str(slot_type))
            config_id = _coerce_int_or_none(_get_any(slot, ["configId", "config_id", "skill_id", "skillId"]))
            if config_id is not None:
                self.skill_config_ids_per_slot[idx].add(config_id)
            for key, target in [
                ("cooldown", self.skill_cooldown_cur_per_slot),
                ("cooldown_max", self.skill_cooldown_max_per_slot),
                ("level", self.skill_level_per_slot),
            ]:
                val = _coerce_int_or_none(_get_any(slot, [key, _snake_to_camel(key)]))
                if val is None:
                    continue
                rng = target.get(idx)
                if rng is None:
                    rng = _Range()
                    target[idx] = rng
                rng.update(val)
            if "usable" in slot:
                self.skill_usable_per_slot[idx].add(bool(slot.get("usable")))
            if self._looks_like_summoner_slot(slot_type, config_id):
                self.summoner_slot_candidates[idx].add((str(slot_type), config_id))

    @staticmethod
    def _looks_like_summoner_slot(slot_type, config_id) -> bool:
        if slot_type in ("SLOT_SKILL_5", 5, "5", "SLOT_SUMMONER_SKILL", "summoner"):
            return True
        return config_id in {80102, 80103, 80104, 80105, 80107, 80108, 80109, 80110, 80115, 80121}

    def to_dict(self):
        return {
            "count": self.count,
            "first_frame": self.first_frame,
            "last_frame": self.last_frame,
            "runtime_ids_seen": sorted(self.runtime_ids),
            "hp": self.hp.to_dict(),
            "max_hp": self.max_hp.to_dict(),
            "level": self.level.to_dict(),
            "exp": self.exp.to_dict(),
            "money_total": self.money_total.to_dict(),
            "money_per_frame_field": self.money_frame.to_dict(),
            "money_delta_histogram": dict(self.money_delta_histogram),
            "abilities_lengths_seen": sorted(self.abilities_lengths_seen),
            "abilities_patterns_seen": [list(v) for v in sorted(self.abilities_patterns_seen)],
            "camp_visible_patterns_seen": [list(v) for v in sorted(self.camp_visible_patterns_seen)],
            "skill_slot_types_per_slot": {
                str(k): sorted(v, key=str) for k, v in sorted(self.skill_slot_types_per_slot.items())
            },
            "skill_config_ids_per_slot": {
                str(k): sorted(v) for k, v in sorted(self.skill_config_ids_per_slot.items())
            },
            "skill_cooldown_cur_per_slot": {
                str(k): v.to_dict() for k, v in sorted(self.skill_cooldown_cur_per_slot.items())
            },
            "skill_cooldown_max_per_slot": {
                str(k): v.to_dict() for k, v in sorted(self.skill_cooldown_max_per_slot.items())
            },
            "skill_level_per_slot": {
                str(k): v.to_dict() for k, v in sorted(self.skill_level_per_slot.items())
            },
            "skill_usable_per_slot": {
                str(k): sorted(v, key=str) for k, v in sorted(self.skill_usable_per_slot.items())
            },
            "summoner_slot_candidates": {
                str(k): sorted([list(item) for item in v], key=str)
                for k, v in sorted(self.summoner_slot_candidates.items())
            },
        }


def _snake_to_camel(key: str) -> str:
    parts = key.split("_")
    return parts[0] + "".join(part[:1].upper() + part[1:] for part in parts[1:])


class _OrganAcc:
    """Per-camp tower/crystal accumulator."""

    def __init__(self):
        self.location_x = _Range()
        self.location_z = _Range()
        self.forward_x = _Range()
        self.forward_z = _Range()
        self.hp = _Range()
        self.max_hp = _Range()
        self.ep = _Range()
        self.max_ep = _Range()
        self.attack_range = _Range()
        self.sight_area = _Range()
        self.mov_spd = _Range()
        self.atk_spd = _Range()
        self.phy_atk = _Range()
        self.phy_def = _Range()
        self.mgc_atk = _Range()
        self.mgc_def = _Range()
        self.phy_armor_hurt = _Range()
        self.mgc_armor_hurt = _Range()
        self.crit_rate = _Range()
        self.crit_effe = _Range()
        self.phy_vamp = _Range()
        self.mgc_vamp = _Range()
        self.cd_reduce = _Range()
        self.ctrl_reduce = _Range()
        self.kill_income = _Range()
        self.hp_recover = _Range()
        self.ep_recover = _Range()
        self.attack_target_seen = set()
        self.behave_seen = set()
        self.config_ids = set()
        self.abilities_lengths_seen = set()
        self.abilities_patterns_seen = set()
        self.camp_visible_patterns_seen = set()
        self.hurt_hero_targets_seen = set()
        self.hurt_hero_value = _Range()

    def to_dict(self):
        return {
            "location_x": self.location_x.to_dict(),
            "location_z": self.location_z.to_dict(),
            "forward_x": self.forward_x.to_dict(),
            "forward_z": self.forward_z.to_dict(),
            "hp": self.hp.to_dict(),
            "max_hp": self.max_hp.to_dict(),
            "ep": self.ep.to_dict(),
            "max_ep": self.max_ep.to_dict(),
            "attack_range": self.attack_range.to_dict(),
            "sight_area": self.sight_area.to_dict(),
            "mov_spd": self.mov_spd.to_dict(),
            "atk_spd": self.atk_spd.to_dict(),
            "phy_atk": self.phy_atk.to_dict(),
            "phy_def": self.phy_def.to_dict(),
            "mgc_atk": self.mgc_atk.to_dict(),
            "mgc_def": self.mgc_def.to_dict(),
            "phy_armor_hurt": self.phy_armor_hurt.to_dict(),
            "mgc_armor_hurt": self.mgc_armor_hurt.to_dict(),
            "crit_rate": self.crit_rate.to_dict(),
            "crit_effe": self.crit_effe.to_dict(),
            "phy_vamp": self.phy_vamp.to_dict(),
            "mgc_vamp": self.mgc_vamp.to_dict(),
            "cd_reduce": self.cd_reduce.to_dict(),
            "ctrl_reduce": self.ctrl_reduce.to_dict(),
            "kill_income": self.kill_income.to_dict(),
            "hp_recover": self.hp_recover.to_dict(),
            "ep_recover": self.ep_recover.to_dict(),
            "attack_target_values_seen": sorted(self.attack_target_seen, key=str),
            "behave_seen": sorted(self.behave_seen, key=str),
            "config_ids_seen": sorted(self.config_ids),
            "abilities_lengths_seen": sorted(self.abilities_lengths_seen),
            "abilities_patterns_seen": [list(v) for v in sorted(self.abilities_patterns_seen)],
            "camp_visible_patterns_seen": [list(v) for v in sorted(self.camp_visible_patterns_seen)],
            "hurt_hero_targets_seen": sorted(self.hurt_hero_targets_seen),
            "hurt_hero_value": self.hurt_hero_value.to_dict(),
        }


class DumpCollector:
    def __init__(self, dump_dir: str, sample_frame_interval: int = 200,
                 max_frame_dumps: int = 80, logger=None):
        self.dump_dir = dump_dir
        self.frames_dir = os.path.join(dump_dir, "frames")
        self.sample_frame_interval = max(1, int(sample_frame_interval))
        self.max_frame_dumps = max(0, int(max_frame_dumps))
        self.logger = logger
        os.makedirs(self.dump_dir, exist_ok=True)
        if self.max_frame_dumps > 0:
            os.makedirs(self.frames_dir, exist_ok=True)

        # Counters
        self.episodes_observed = 0
        self.frames_observed = 0
        self.frame_dumps_written = 0
        self.frame_no_max = 0

        # Protocol detection
        self.hero_layout_votes = defaultdict(int)   # 'flat' or 'nested'
        self.sub_type_dtype_votes = defaultdict(int)
        self.camp_dtype_votes = defaultdict(int)
        self.actor_type_dtype_votes = defaultdict(int)
        self.legal_action_lengths = set()
        self.sub_action_mask_shape = set()
        self.legal_action_first_button_mask_seen = set()  # tuples of first 12 values

        # Distinct sets
        self.actor_types_seen = set()
        self.sub_types_seen = set()
        self.camp_values_seen = set()
        self.npc_config_id_to_sub_type = {}     # config_id -> sub_type
        self.npc_actor_subtype_counts = defaultdict(int)
        self.actor_subtype_kind_counts = defaultdict(lambda: defaultdict(int))
        self.kind_first_seen_frame = {}
        self.runtime_actor_index = {}
        self.bullet_slot_types_seen = set()
        self.bullet_skill_ids_seen = set()
        self.bullet_camp_seen = set()
        self.bullet_runtime_ids_seen = set()
        self.bullet_source_actor_ids_seen = set()
        self.bullet_slot_type_counts = defaultdict(int)
        self.bullet_source_kind_counts = defaultdict(int)
        self.bullet_source_actor_samples = {}
        self.bullet_slot_to_skill = defaultdict(set)
        self.bullet_camp_slot_counts = defaultdict(lambda: defaultdict(int))
        self.behave_per_actor_kind = defaultdict(set)  # 'hero'/'soldier'/'tower'/'crystal'/'monster'/'spring' -> set
        self.dead_action_keys_seen = set()
        self.dead_action_death_type_counts = defaultdict(int)
        self.dead_action_killer_type_counts = defaultdict(int)
        self.dead_action_hurt_type_counts = defaultdict(int)
        self.dead_action_hurt_value = _Range()
        self.dead_action_income_exp = _Range()
        self.dead_action_income_money = _Range()
        self.dead_action_single_hurt_slot_counts = defaultdict(int)
        self.dead_action_single_hurt_value = _Range()
        self.skill_use_events_seen = defaultdict(int)
        self.skill_use_slot_types = set()
        self.buff_skill_ids_seen = set()
        self.buff_mark_id_to_max_layer = {}
        self.buff_origin_actor_ids_seen = set()
        self.buff_seen_by_kind = defaultdict(set)
        self.buff_detail_by_holder = {}
        self.hero_buff_keys_by_config_camp = defaultdict(set)
        self.buff_state_key_patterns = defaultdict(int)
        self.buff_mark_field_key_patterns = defaultdict(int)
        self.buff_mark_container_counts = defaultdict(int)
        self.buff_mark_holder_counts = defaultdict(int)
        self.buff_mark_raw_samples = []
        self.cake_first_appear_frame = None
        self.cake_seen = False
        self.cake_camp_locations = []   # list of [x, z] tuples seen

        # Map bounds (legacy mixed source) plus source-split coordinate diagnostics.
        self.map_x_range = _Range()
        self.map_z_range = _Range()
        self.map_sum_range = _Range()
        self.map_diff_range = _Range()
        self.coord_stats = defaultdict(_CoordAcc)
        self.coord_stats_by_camp = defaultdict(_CoordAcc)
        self.valid_coord_stats = defaultdict(_CoordAcc)
        self.valid_coord_stats_by_camp = defaultdict(_CoordAcc)

        # Sub-action mask snapshot per button
        self.sub_action_mask_per_button = {}
        self.sub_action_mask_unique_per_button = defaultdict(set)
        self.legal_action_target_mask_unique_per_button = defaultdict(set)

        # Hero / organ / monster acc
        self.heroes: Dict[int, _HeroAcc] = {}
        self.hero_by_config_camp_debug: Dict[str, _HeroCampDebugAcc] = {}
        self.tower: Dict[int, _OrganAcc] = {}    # camp -> _OrganAcc
        self.crystal: Dict[int, _OrganAcc] = {}
        self.spring: Dict[int, _OrganAcc] = {}
        self.monster_per_config: Dict[int, _OrganAcc] = {}
        self.monster_seen = False

        # Soldier per camp+config
        self.soldier_per_config_camp: Dict[Tuple[int, int], _OrganAcc] = {}
        self.frame_state_keys_logged = False

    # ===== entry points =====

    def on_episode_start(self) -> None:
        self.episodes_observed += 1
        # reset per-hero last_money so first delta is well-defined
        for hero_acc in self.heroes.values():
            hero_acc._last_money_total = None
        for hero_acc in self.hero_by_config_camp_debug.values():
            hero_acc._last_money_total = None

    def on_frame(self, observation: Optional[dict], frame_no: int, env_id: Any = "unknown") -> None:
        if not isinstance(observation, dict):
            return
        self.frames_observed += 1
        if frame_no is not None and frame_no > self.frame_no_max:
            self.frame_no_max = int(frame_no)

        legal_action = observation.get("legal_action")
        if isinstance(legal_action, (list, tuple)):
            self.legal_action_lengths.add(len(legal_action))
            self._record_button_mask_snapshot(legal_action)
            self._record_target_mask_snapshots(legal_action)
        sub_action_mask = observation.get("sub_action_mask")
        if isinstance(sub_action_mask, (list, tuple)) and sub_action_mask:
            self.sub_action_mask_shape.add((len(sub_action_mask), len(sub_action_mask[0]) if sub_action_mask[0] else 0))
            for button_idx, mask_row in enumerate(sub_action_mask):
                if button_idx not in self.sub_action_mask_per_button and isinstance(mask_row, (list, tuple)):
                    self.sub_action_mask_per_button[button_idx] = list(mask_row)
                if isinstance(mask_row, (list, tuple)):
                    _add_limited_tuple(self.sub_action_mask_unique_per_button, button_idx, mask_row)

        frame_state = observation.get("frame_state")
        if not isinstance(frame_state, dict):
            if frame_no is not None and frame_no % 300 == 0:
                self.logger.info(f"[FRAME_DEBUG] frame_no={frame_no} frame_state_missing_keys={list(observation.keys()) if isinstance(observation, dict) else 'not_dict'}")
            return
        if not self.frame_state_keys_logged:
            self.frame_state_keys_logged = True
            keys = sorted(frame_state.keys())
            has_bullets = "bullets" in frame_state
            bullets_count = len(frame_state.get("bullets", []) or []) if has_bullets else 0
            self.logger.info(f"[FRAME_DEBUG] first_frame_state_keys={keys} has_bullets={has_bullets} bullets_count={bullets_count}")
        if frame_no is not None and frame_no % 300 == 0:
            has_bullets = "bullets" in frame_state
            bullets_count = len(frame_state.get("bullets", []) or []) if has_bullets else 0
            self.logger.info(f"[FRAME_DEBUG] frame_no={frame_no} has_bullets={has_bullets} bullets_count={bullets_count}")

        for hero in frame_state.get("hero_states", []) or []:
            self._consume_hero(hero, frame_no)
        for npc in frame_state.get("npc_states", []) or []:
            self._consume_npc(npc, frame_no)
        bullets = frame_state.get("bullets", []) or []
        if bullets:
            if not self.bullet_slot_types_seen:
                self.logger.info(f"[BULLET_DEBUG] first_bullets_seen_at_frame={frame_no} count={len(bullets)} sample={bullets[0]}")
            for bullet in bullets:
                self._consume_bullet(bullet, frame_no)
        else:
            if frame_no is not None and frame_no % 500 == 0 and not self.bullet_slot_types_seen:
                self.logger.info(f"[BULLET_DEBUG] no_bullets_yet_at_frame={frame_no}")
        cakes = frame_state.get("cakes")
        if cakes:
            for cake in cakes:
                self._consume_cake(cake, frame_no)
        frame_action = frame_state.get("frame_action")
        if isinstance(frame_action, dict):
            for key in frame_action.keys():
                self.dead_action_keys_seen.add(str(key))
            self._consume_frame_action(frame_action)

        # Sample full observation occasionally
        if self.frame_dumps_written < self.max_frame_dumps and frame_no is not None:
            if int(frame_no) % self.sample_frame_interval == 0:
                self._dump_frame(observation, env_id, frame_no)

    def write_summary(self, path: Optional[str] = None) -> str:
        if path is None:
            path = os.path.join(self.dump_dir, "summary.json")
        summary = self._build_summary()
        with open(path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False, default=_json_default)
        if self.logger is not None:
            self.logger.info(f"DumpCollector wrote summary to {path}")
            self._log_summary_to_logger(summary)
        return path

    def _log_summary_to_logger(self, summary: dict) -> None:
        self.logger.info(
            f"[SUMMARY] episodes={summary['episodes_observed']} frames={summary['frames_observed']} "
            f"frame_no_max={summary['frame_no_max_observed']} dumps={summary['frame_dumps_written']}"
        )
        pc = summary.get("protocol_check", {})
        self.logger.info(
            f"[SUMMARY] layout={pc.get('hero_state_layout','?')} sub_type_dtype={pc.get('sub_type_dtype','?')} "
            f"camp_dtype={pc.get('camp_dtype','?')} actor_type_dtype={pc.get('actor_type_dtype','?')}"
        )
        self.logger.info(
            f"[SUMMARY] actor_types={summary.get('actor_types_seen','?')}"
        )
        self.logger.info(
            f"[SUMMARY] sub_types={summary.get('sub_types_seen','?')}"
        )
        self.logger.info(
            f"[SUMMARY] actor_subtype_kind_counts={summary.get('actor_subtype_kind_counts','?')}"
        )
        buff_details = summary.get("buff_detail_by_holder", {}) or {}
        self.logger.info(
            f"[SUMMARY] buff_detail_count={len(buff_details)} "
            f"hero_buff_groups={summary.get('hero_buff_keys_by_config_camp', {})}"
        )
        self.logger.info(
            f"[FREEZE] buff_skill_ids={summary.get('buff_skill_ids_seen', [])} "
            f"buff_mark_layers={summary.get('buff_mark_id_to_max_layer', {})} "
            f"buff_by_kind={summary.get('buff_seen_by_kind', {})}"
        )
        self.logger.info(
            f"[FREEZE] buff_state_key_patterns={summary.get('buff_state_key_patterns', {})} "
            f"buff_mark_container_counts={summary.get('buff_mark_container_counts', {})} "
            f"buff_mark_field_key_patterns={summary.get('buff_mark_field_key_patterns', {})} "
            f"buff_mark_holder_counts={summary.get('buff_mark_holder_counts', {})}"
        )
        mark_samples = summary.get("buff_mark_raw_samples", []) or []
        if mark_samples:
            self.logger.info(f"[FREEZE] buff_mark_raw_samples={mark_samples[:24]}")
        for key, detail in (summary.get("hero_by_config_camp_debug", {}) or {}).items():
            self.logger.info(
                f"[FREEZE] hero_money_by_config_camp {key}: "
                f"frames=({detail.get('first_frame')},{detail.get('last_frame')}) "
                f"count={detail.get('count')} runtime_ids={detail.get('runtime_ids_seen')} "
                f"hp=({detail.get('hp',{}).get('min')},{detail.get('hp',{}).get('max')}) "
                f"max_hp=({detail.get('max_hp',{}).get('min')},{detail.get('max_hp',{}).get('max')}) "
                f"level=({detail.get('level',{}).get('min')},{detail.get('level',{}).get('max')}) "
                f"exp=({detail.get('exp',{}).get('min')},{detail.get('exp',{}).get('max')}) "
                f"money_total=({detail.get('money_total',{}).get('min')},"
                f"{detail.get('money_total',{}).get('max')},"
                f"{detail.get('money_total',{}).get('count')}) "
                f"money_frame=({detail.get('money_per_frame_field',{}).get('min')},"
                f"{detail.get('money_per_frame_field',{}).get('max')},"
                f"{detail.get('money_per_frame_field',{}).get('count')}) "
                f"money_delta={detail.get('money_delta_histogram', {})} "
                f"camp_visible={detail.get('camp_visible_patterns_seen', [])} "
                f"abilities={detail.get('abilities_patterns_seen', [])}"
            )
            self.logger.info(
                f"[FREEZE] hero_summoner_by_config_camp {key}: "
                f"summoner_candidates={detail.get('summoner_slot_candidates', {})} "
                f"slot_types={detail.get('skill_slot_types_per_slot', {})} "
                f"config_ids={detail.get('skill_config_ids_per_slot', {})} "
                f"cooldown_cur={detail.get('skill_cooldown_cur_per_slot', {})} "
                f"cooldown_max={detail.get('skill_cooldown_max_per_slot', {})} "
                f"usable={detail.get('skill_usable_per_slot', {})} "
                f"levels={detail.get('skill_level_per_slot', {})}"
            )
        for key, detail in list(buff_details.items())[:80]:
            self.logger.info(
                f"[SUMMARY] buff_detail {key}: count={detail.get('count')} "
                f"frames=({detail.get('first_frame')},{detail.get('last_frame')}) "
                f"times=({detail.get('times',{}).get('min')},{detail.get('times',{}).get('max')}) "
                f"layer=({detail.get('layer',{}).get('min')},{detail.get('layer',{}).get('max')}) "
                f"origins={detail.get('origin_actor_ids_seen')}"
            )
        self._log_building_summary(summary)
        has_bullets = bool(summary.get('bullet_slot_types_seen'))
        self.logger.info(
            f"[SUMMARY] bullet_exists={has_bullets} "
            f"bullet_slot_types={summary.get('bullet_slot_types_seen','?')} "
            f"bullet_skill_ids={summary.get('bullet_skill_ids_seen','?')} "
            f"bullet_camps={summary.get('bullet_camp_values_seen','?')}"
        )
        self.logger.info(
            f"[SUMMARY] bullet_slot_type_counts={summary.get('bullet_slot_type_counts','?')}"
        )
        self.logger.info(
            f"[SUMMARY] bullet_slot_to_skill={summary.get('bullet_slot_to_skill_mapping','?')}"
        )
        self.logger.info(
            f"[SUMMARY] bullet_camp_slot_counts={summary.get('bullet_camp_slot_counts','?')}"
        )
        bullet_source_samples = summary.get("bullet_source_actor_samples", {}) or {}
        bullet_source_sample_keys = list(bullet_source_samples.keys())[:16]
        self.logger.info(
            f"[SUMMARY] bullet_source_kind_counts={summary.get('bullet_source_kind_counts','?')} "
            f"bullet_source_actor_sample_keys={bullet_source_sample_keys}"
        )
        self.logger.info(
            f"[SUMMARY] cake_seen={summary.get('cake_seen','?')} "
            f"cake_first_frame={summary.get('cake_first_appear_frame','?')}"
        )
        for kind, values in summary.get("behave_seen_per_actor_kind", {}).items():
            self.logger.info(f"[SUMMARY] behave_{kind}={values}")
        for cid, h in summary.get("heroes", {}).items():
            hp = h.get("hp", {})
            ep = h.get("max_ep", {})
            ar = h.get("attack_range", {})
            sa = h.get("sight_area", {})
            ms = h.get("mov_spd", {})
            lv = h.get("level", {})
            self.logger.info(
                f"[SUMMARY] hero_{cid}: hp=({hp.get('min')},{hp.get('max')}) "
                f"max_ep=({ep.get('min')},{ep.get('max')}) "
                f"atk_range=({ar.get('min')},{ar.get('max')}) "
                f"sight=({sa.get('min')},{sa.get('max')}) "
                f"mov_spd=({ms.get('min')},{ms.get('max')}) "
                f"level=({lv.get('min')},{lv.get('max')})"
            )
            self.logger.info(
                f"[SUMMARY] hero_{cid}: phy_atk=({h.get('phy_atk',{}).get('min')},{h.get('phy_atk',{}).get('max')}) "
                f"phy_def=({h.get('phy_def',{}).get('min')},{h.get('phy_def',{}).get('max')}) "
                f"mgc_atk=({h.get('mgc_atk',{}).get('min')},{h.get('mgc_atk',{}).get('max')}) "
                f"mgc_def=({h.get('mgc_def',{}).get('min')},{h.get('mgc_def',{}).get('max')}) "
                f"atk_spd=({h.get('atk_spd',{}).get('min')},{h.get('atk_spd',{}).get('max')})"
            )
            self.logger.info(
                f"[SUMMARY] hero_{cid}: behave={h.get('behave_seen','?')} "
                f"grass={h.get('is_in_grass_values_seen','?')} "
                f"passive={h.get('passive_skill_ids','?')} "
                f"buff_skills={h.get('buff_skill_ids_seen','?')} "
                f"buff_marks={h.get('buff_mark_id_to_max_layer','?')} "
                f"summoner={h.get('summoner_skill_ids_seen','?')}"
            )
            exp = h.get("exp", {})
            if exp.get("count"):
                self.logger.info(
                    f"[SUMMARY] hero_{cid}: exp=({exp.get('min')},{exp.get('max')}) "
                    f"hp_recover=({h.get('hp_recover',{}).get('min')},{h.get('hp_recover',{}).get('max')}) "
                    f"ep_recover=({h.get('ep_recover',{}).get('min')},{h.get('ep_recover',{}).get('max')})"
                )
            cd = h.get("skill_cooldown_max_per_slot", {})
            if cd:
                self.logger.info(f"[SUMMARY] hero_{cid}: cooldown_max_per_slot={cd}")
            cd_cur = h.get("skill_cooldown_cur_per_slot", {})
            skill_cfg = h.get("skill_config_ids_per_slot", {})
            skill_usable = h.get("skill_usable_per_slot", {})
            skill_next = h.get("skill_next_config_ids_per_slot", {})
            if skill_cfg or cd or cd_cur or skill_usable or skill_next:
                self.logger.info(
                    f"[FREEZE] hero_{cid}_skill_table "
                    f"config_ids={skill_cfg} cooldown_cur={cd_cur} cooldown_max={cd} "
                    f"usable={skill_usable} next_config_ids={skill_next}"
                )
            sl = h.get("skill_level_per_slot", {})
            if sl:
                self.logger.info(f"[SUMMARY] hero_{cid}: skill_level_per_slot={sl}")
            mh = h.get("money_delta_histogram", {})
            if mh:
                self.logger.info(f"[SUMMARY] hero_{cid}: money_delta_histogram={mh}")
            mhl = h.get("max_hp_by_level", {})
            if mhl:
                self.logger.info(f"[SUMMARY] hero_{cid}: max_hp_by_level={mhl}")
            mel = h.get("max_ep_by_level", {})
            if mel:
                self.logger.info(f"[SUMMARY] hero_{cid}: max_ep_by_level={mel}")
        for key, acc in summary.get("soldier_per_config_camp", {}).items():
            hp = acc.get("hp", {})
            ar = acc.get("attack_range", {})
            sa = acc.get("sight_area", {})
            self.logger.info(
                f"[SUMMARY] soldier_{key}: hp=({hp.get('min')},{hp.get('max')}) "
                f"atk_range=({ar.get('min')},{ar.get('max')}) sight=({sa.get('min')},{sa.get('max')}) "
                f"behave={acc.get('behave_seen','?')} "
                f"atk_target={acc.get('attack_target_values_seen','?')}"
            )
        self.logger.info(f"[SUMMARY] monster_seen={summary.get('monster_seen','?')}")
        for cid, acc in summary.get("monster_per_config", {}).items():
            hp = acc.get("hp", {})
            ar = acc.get("attack_range", {})
            sa = acc.get("sight_area", {})
            self.logger.info(
                f"[SUMMARY] monster_{cid}: hp=({hp.get('min')},{hp.get('max')}) "
                f"atk_range=({ar.get('min')},{ar.get('max')}) sight=({sa.get('min')},{sa.get('max')}) "
                f"behave={acc.get('behave_seen','?')} config_ids={acc.get('config_ids_seen','?')}"
            )
        self.logger.info(
            f"[SUMMARY] legal_action_lengths={summary.get('legal_action_lengths_seen','?')}"
        )
        self.logger.info(
            f"[SUMMARY] sub_action_mask_shape={summary.get('sub_action_mask_shape_seen','?')}"
        )
        self.logger.info(
            f"[SUMMARY] legal_target_mask_unique={summary.get('legal_action_target_mask_unique_per_button','?')}"
        )
        sue = summary.get("skill_use_events", {})
        if sue:
            self.logger.info(f"[SUMMARY] skill_use_events={sue}")
        sust = summary.get("skill_use_slot_types", [])
        if sust:
            self.logger.info(f"[SUMMARY] skill_use_slot_types={sust}")

    def _log_building_summary(self, summary: dict) -> None:
        for label, key in [("tower", "tower_per_camp"), ("crystal", "crystal_per_camp"), ("spring", "spring_per_camp")]:
            entries = summary.get(key, {}) or {}
            self.logger.info(f"[SUMMARY] {key}_count={len(entries)}")
            for camp, acc in entries.items():
                hp = acc.get("hp", {})
                ar = acc.get("attack_range", {})
                sa = acc.get("sight_area", {})
                self.logger.info(
                    f"[SUMMARY] {label}_camp={camp}: hp=({hp.get('min')},{hp.get('max')}) "
                    f"atk_range=({ar.get('min')},{ar.get('max')}) sight=({sa.get('min')},{sa.get('max')}) "
                    f"behave={acc.get('behave_seen','?')} "
                    f"atk_target={acc.get('attack_target_values_seen','?')} "
                    f"config_ids={acc.get('config_ids_seen','?')}"
                )

    # ===== consumers =====

    def _consume_hero(self, hero: dict, frame_no: Optional[int] = None) -> None:
        if not isinstance(hero, dict):
            return
        # Detect layout
        if "camp" in hero:
            self.hero_layout_votes["flat"] += 1
        actor_state = hero.get("actor_state") or hero.get("actorState")
        if isinstance(actor_state, dict) and "camp" in actor_state:
            self.hero_layout_votes["nested"] += 1

        config_id = _coerce_int_or_none(_hero_field(hero, "config_id", ["configId"]))
        if config_id is None:
            return
        acc = self.heroes.get(config_id)
        if acc is None:
            acc = _HeroAcc()
            self.heroes[config_id] = acc

        # camp dtype
        camp = _hero_field(hero, "camp")
        self._note_dtype(camp, self.camp_dtype_votes, self.camp_values_seen)
        camp_debug_key = f"hero={config_id}|camp={camp}"
        camp_debug_acc = self.hero_by_config_camp_debug.get(camp_debug_key)
        if camp_debug_acc is None:
            camp_debug_acc = _HeroCampDebugAcc()
            self.hero_by_config_camp_debug[camp_debug_key] = camp_debug_acc
        camp_debug_acc.update(hero, frame_no)
        runtime_id = _coerce_int_or_none(_hero_field(hero, "runtime_id", ["runtimeId"]))
        if runtime_id is not None:
            self.runtime_actor_index[runtime_id] = {
                "kind": "hero", "camp": camp, "config_id": config_id,
                "actor_type": _hero_field(hero, "actor_type", ["actorType"]),
                "sub_type": _hero_field(hero, "sub_type", ["subType"]),
            }
        self._note_kind_seen("hero", frame_no)

        abilities = _hero_field(hero, "abilities")
        if isinstance(abilities, (list, tuple)):
            acc.abilities_lengths_seen.add(len(abilities))
            _add_limited_tuple({0: acc.abilities_patterns_seen}, 0, [int(bool(v)) for v in abilities], limit=16)
        camp_visible = _hero_field(hero, "camp_visible", ["campVisible"])
        if isinstance(camp_visible, (list, tuple)):
            _add_limited_tuple({0: acc.camp_visible_patterns_seen}, 0, [int(bool(v)) for v in camp_visible], limit=16)

        # actor_type dtype
        actor_type = _hero_field(hero, "actor_type", ["actorType"])
        if actor_type is not None:
            self.actor_types_seen.add(actor_type)
            self._note_dtype(actor_type, self.actor_type_dtype_votes, set())
        sub_type = _hero_field(hero, "sub_type", ["subType"])
        if sub_type is not None:
            self.sub_types_seen.add(sub_type)
            self._note_dtype(sub_type, self.sub_type_dtype_votes, set())
        self._record_actor_subtype("hero", actor_type, sub_type)

        # location
        loc = _hero_field(hero, "location")
        if isinstance(loc, dict):
            x, y, z = _read_location(hero)
            acc.location_x.update(x)
            acc.location_z.update(z)
            self.map_x_range.update(x)
            self.map_z_range.update(z)
            if x is not None and z is not None:
                self.map_sum_range.update(x + z)
                self.map_diff_range.update(z - x)
            self._record_coord(
                "hero",
                x, y, z, frame_no,
                {
                    "camp": camp, "config_id": config_id, "runtime_id": runtime_id,
                    "actor_type": actor_type, "sub_type": sub_type,
                },
            )

        forward = _hero_field(hero, "forward")
        if isinstance(forward, dict):
            acc.forward_x.update(_coerce_int_or_none(forward.get("x")))
            acc.forward_z.update(_coerce_int_or_none(forward.get("z")))

        # vitals
        hp = _coerce_int_or_none(_hero_field(hero, "hp"))
        max_hp = _coerce_int_or_none(_hero_field(hero, "max_hp", ["maxHp", "max_Hp"]))
        acc.hp.update(hp)
        acc.max_hp.update(max_hp)

        ep = None
        max_ep = None
        values = _hero_field(hero, "values")
        if isinstance(values, dict):
            ep = _coerce_int_or_none(values.get("ep"))
            max_ep = _coerce_int_or_none(values.get("max_ep"))
            for rec_key in ["hp_recover", "hpRecover", "hp_recover_speed", "hpRecoverSpeed"]:
                if rec_key in values:
                    acc.hp_recover.update(_coerce_int_or_none(values[rec_key]))
                    break
            for rec_key in ["ep_recover", "epRecover", "ep_recover_speed", "epRecoverSpeed"]:
                if rec_key in values:
                    acc.ep_recover.update(_coerce_int_or_none(values[rec_key]))
                    break
        if ep is None:
            ep = _coerce_int_or_none(_hero_field(hero, "ep"))
        if max_ep is None:
            max_ep = _coerce_int_or_none(_hero_field(hero, "max_ep", ["maxEp"]))
        acc.ep.update(ep)
        acc.max_ep.update(max_ep)
        if acc.hp_recover.count == 0:
            for alt_key in ["hp_recover", "hpRecover", "recover_hp", "recoverHp"]:
                val = _coerce_int_or_none(_hero_field(hero, alt_key))
                if val is not None:
                    acc.hp_recover.update(val)
                    break
        if acc.ep_recover.count == 0:
            for alt_key in ["ep_recover", "epRecover", "recover_ep", "recoverEp"]:
                val = _coerce_int_or_none(_hero_field(hero, alt_key))
                if val is not None:
                    acc.ep_recover.update(val)
                    break

        # combat stats
        for field, target in [
            ("attack_range", acc.attack_range),
            ("sight_area", acc.sight_area),
            ("mov_spd", acc.mov_spd),
            ("atk_spd", acc.atk_spd),
            ("phy_atk", acc.phy_atk),
            ("phy_def", acc.phy_def),
            ("mgc_atk", acc.mgc_atk),
            ("mgc_def", acc.mgc_def),
            ("phy_armor_hurt", acc.phy_armor_hurt),
            ("mgc_armor_hurt", acc.mgc_armor_hurt),
            ("crit_rate", acc.crit_rate),
            ("crit_effe", acc.crit_effe),
            ("phy_vamp", acc.phy_vamp),
            ("mgc_vamp", acc.mgc_vamp),
            ("cd_reduce", acc.cd_reduce),
            ("ctrl_reduce", acc.ctrl_reduce),
            ("kill_income", acc.kill_income),
        ]:
            target.update(_coerce_int_or_none(_field_or_values(hero, field)))
        attack_target = _hero_field(hero, "attack_target", ["attackTarget"])
        if attack_target is not None:
            acc.attack_target_seen.add(attack_target)

        # progression
        level = _coerce_int_or_none(_hero_field(hero, "level"))
        acc.level.update(level)
        acc.exp.update(_coerce_int_or_none(_hero_field(hero, "exp")))
        if level is not None and max_hp is not None:
            cur = acc.max_hp_by_level.get(level)
            acc.max_hp_by_level[level] = max(cur, max_hp) if cur is not None else max_hp
        if level is not None and max_ep is not None:
            cur = acc.max_ep_by_level.get(level)
            acc.max_ep_by_level[level] = max(cur, max_ep) if cur is not None else max_ep

        # money
        money_total = _coerce_int_or_none(_hero_field(hero, "moneyCnt", ["money_cnt", "money_total"]))
        money_now = _coerce_int_or_none(_hero_field(hero, "money"))
        acc.money.update(money_total)
        acc.frame_money.update(money_now)
        if money_total is not None:
            if acc._last_money_total is None:
                acc._last_money_total = money_total
            else:
                delta = money_total - acc._last_money_total
                acc.money_delta_histogram[_money_bucket(delta)] += 1
                acc._last_money_total = money_total

        # KDA / revive
        acc.kill_cnt.update(_coerce_int_or_none(_hero_field(hero, "killCnt", ["kill_cnt"])))
        acc.dead_cnt.update(_coerce_int_or_none(_hero_field(hero, "deadCnt", ["dead_cnt"])))
        acc.assist_cnt.update(_coerce_int_or_none(_hero_field(hero, "assistCnt", ["assist_cnt"])))
        acc.revive_time.update(_coerce_int_or_none(_hero_field(hero, "revive_time")))

        # behave / grass
        behave = _hero_field(hero, "behav_mode", ["behave"])
        if behave is not None:
            acc.behave_seen.add(str(behave))
            self.behave_per_actor_kind["hero"].add(str(behave))
        grass = _hero_field(hero, "isInGrass", ["is_in_grass"])
        if grass is not None:
            acc.is_in_grass_seen.add(grass)

        for s_field in ["summoner_skill_id", "summonerSkillId", "summoner_skill", "summonerSkill"]:
            s_val = _coerce_int_or_none(_hero_field(hero, s_field))
            if s_val is not None:
                acc.summoner_skill_ids.add(s_val)
                break

        skill_state = _hero_field(hero, "skill_state", ["skillState"])
        if isinstance(skill_state, dict):
            slot_states = skill_state.get("slot_states") or skill_state.get("slotStates") or []
            for idx, slot in enumerate(slot_states):
                if not isinstance(slot, dict):
                    continue
                config_id = _coerce_int_or_none(slot.get("configId") or slot.get("config_id") or slot.get("skill_id"))
                if config_id is not None:
                    acc.skill_config_ids_per_slot[idx].add(config_id)
                cd_cur = _coerce_int_or_none(slot.get("cooldown"))
                if cd_cur is not None:
                    rng = acc.skill_cooldown_cur_per_slot.get(idx)
                    if rng is None:
                        rng = _Range()
                        acc.skill_cooldown_cur_per_slot[idx] = rng
                    rng.update(cd_cur)
                cd_max = _coerce_int_or_none(slot.get("cooldown_max") or slot.get("cooldownMax"))
                if cd_max is not None:
                    rng = acc.skill_cooldown_max_per_slot.get(idx)
                    if rng is None:
                        rng = _Range()
                        acc.skill_cooldown_max_per_slot[idx] = rng
                    rng.update(cd_max)
                slot_level = _coerce_int_or_none(slot.get("level"))
                if slot_level is not None:
                    rng_lv = acc.skill_level_per_slot.get(idx)
                    if rng_lv is None:
                        rng_lv = _Range()
                        acc.skill_level_per_slot[idx] = rng_lv
                    rng_lv.update(slot_level)
                if "usable" in slot:
                    acc.skill_usable_per_slot[idx].add(bool(slot.get("usable")))
                for field_name, container in [
                    ("usedTimes", acc.skill_used_times_per_slot),
                    ("hitHeroTimes", acc.skill_hit_hero_times_per_slot),
                    ("succUsedInFrame", acc.skill_succ_used_in_frame_per_slot),
                    ("comboEffectTime", acc.skill_combo_effect_time_per_slot),
                ]:
                    val = _coerce_int_or_none(slot.get(field_name))
                    if val is None:
                        continue
                    rng = container.get(idx)
                    if rng is None:
                        rng = _Range()
                        container[idx] = rng
                    rng.update(val)
                next_config_id = _coerce_int_or_none(slot.get("nextConfigID") or slot.get("nextConfigId"))
                if next_config_id is not None:
                    acc.skill_next_config_ids_per_slot[idx].add(next_config_id)
                slot_type = slot.get("slot_type") or slot.get("slotType")
                if slot_type in ("SLOT_SKILL_5", 5, "SLOT_SUMMONER_SKILL", "summoner"):
                    for cid_field in ["configId", "config_id", "skill_id", "skillId", "id"]:
                        s_cid = _coerce_int_or_none(slot.get(cid_field))
                        if s_cid is not None:
                            acc.summoner_skill_ids.add(s_cid)
                            break

        # passive skills
        for ps in (_hero_field(hero, "passive_skill", ["passiveSkill"]) or []):
            if isinstance(ps, dict):
                pid = _coerce_int_or_none(ps.get("passive_skillid") or ps.get("passive_skill_id"))
                if pid is not None:
                    acc.passive_skill_ids.add(pid)

        self._consume_buff_state(
            hero,
            "hero",
            acc,
            frame_no=frame_no,
            holder_meta={"hero_config_id": config_id, "camp": camp, "runtime_id": runtime_id},
        )

        self._consume_hero_hit_targets(hero, acc)
        self._consume_hero_take_hurts(hero, acc)
        self._consume_hero_real_cmd(hero, acc)
        self._consume_hero_equips(hero, acc)

    def _consume_buff_state(self, actor: dict, kind: str, acc=None,
                            frame_no: Optional[int] = None,
                            holder_meta: Optional[dict] = None) -> None:
        buff_state = _hero_field(actor, "buff_state", ["buffState"])
        if not isinstance(buff_state, dict):
            return
        self.buff_state_key_patterns["|".join(sorted(str(k) for k in buff_state.keys()))] += 1
        holder_meta = holder_meta or {}
        holder_runtime_id = _coerce_int_or_none(holder_meta.get("runtime_id"))
        for s in buff_state.get("buff_skills", []) or []:
            if not isinstance(s, dict):
                continue
            cid = _coerce_int_or_none(s.get("configId") or s.get("config_id"))
            times = _coerce_int_or_none(s.get("times"))
            start_time = _coerce_int_or_none(s.get("startTime") or s.get("start_time"))
            if cid is not None:
                self.buff_skill_ids_seen.add(cid)
                self.buff_seen_by_kind[kind].add(cid)
                if acc is not None and hasattr(acc, "buff_skill_ids"):
                    acc.buff_skill_ids.add(cid)
                self._record_buff_detail(
                    holder_kind=kind,
                    holder_meta=holder_meta,
                    buff_type="skill",
                    config_id=cid,
                    frame_no=frame_no,
                    holder_runtime_id=holder_runtime_id,
                    times=times,
                    start_time=start_time,
                    layer=None,
                    origin_actor_id=None,
                )
        mark_entries = []
        mark_container_key = None
        for key in ("buff_marks", "buffMarks", "marks", "mark_states", "markStates"):
            if key not in buff_state:
                continue
            value = buff_state.get(key)
            count_key = f"{key}:non_list"
            if isinstance(value, list):
                count_key = f"{key}:len={len(value)}"
            elif value is None:
                count_key = f"{key}:none"
            self.buff_mark_container_counts[count_key] += 1
            if isinstance(value, list):
                mark_entries = value
                mark_container_key = key
                break
        holder_key = self._buff_holder_key(kind, holder_meta)
        if mark_entries:
            self.buff_mark_holder_counts[holder_key] += len(mark_entries)
        for m in mark_entries or []:
            if not isinstance(m, dict):
                continue
            self.buff_mark_field_key_patterns["|".join(sorted(str(k) for k in m.keys()))] += 1
            cid = _coerce_int_or_none(_get_any(
                m,
                ["configId", "configID", "config_id", "configid", "buff_mark_id", "buffMarkId", "mark_id", "markId", "id"],
            ))
            layer = _coerce_int_or_none(_get_any(m, ["layer", "layers", "level", "count"]))
            origin = _coerce_int_or_none(_get_any(
                m,
                ["origin_actorId", "originActorId", "origin_actor_id", "source_actor", "sourceActor", "actor_id", "actorId"],
            ))
            _bounded_append(
                self.buff_mark_raw_samples,
                {
                    "frame": frame_no,
                    "holder": holder_key,
                    "container": mark_container_key,
                    "parsed_config_id": cid,
                    "parsed_layer": layer,
                    "parsed_origin": origin,
                    "raw": dict(m),
                },
                64,
            )
            if origin is not None:
                self.buff_origin_actor_ids_seen.add(origin)
            if cid is not None:
                prev = self.buff_mark_id_to_max_layer.get(cid, 0)
                if layer is not None and layer > prev:
                    self.buff_mark_id_to_max_layer[cid] = layer
                elif cid not in self.buff_mark_id_to_max_layer:
                    self.buff_mark_id_to_max_layer[cid] = prev
                if acc is not None and hasattr(acc, "buff_mark_id_to_max_layer"):
                    prev_acc = acc.buff_mark_id_to_max_layer.get(cid, 0)
                    if layer is not None and layer > prev_acc:
                        acc.buff_mark_id_to_max_layer[cid] = layer
                    elif cid not in acc.buff_mark_id_to_max_layer:
                        acc.buff_mark_id_to_max_layer[cid] = prev_acc
                self._record_buff_detail(
                    holder_kind=kind,
                    holder_meta=holder_meta,
                    buff_type="mark",
                    config_id=cid,
                    frame_no=frame_no,
                    holder_runtime_id=holder_runtime_id,
                    times=None,
                    start_time=None,
                    layer=layer,
                    origin_actor_id=origin,
                )

    def _record_buff_detail(self, holder_kind: str, holder_meta: dict, buff_type: str,
                            config_id: int, frame_no: Optional[int], holder_runtime_id,
                            times, start_time, layer, origin_actor_id) -> None:
        hero_config_id = holder_meta.get("hero_config_id")
        holder_config_id = holder_meta.get("config_id")
        camp = holder_meta.get("camp")
        if holder_kind == "hero":
            holder_key = f"hero={hero_config_id}|camp={camp}"
        else:
            holder_key = f"{holder_kind}_config={holder_config_id}|camp={camp}"
        key = f"{holder_key}|type={buff_type}|config_id={config_id}"
        acc = self.buff_detail_by_holder.setdefault(key, _BuffAcc())
        sample = {
            "frame": frame_no,
            "holder_kind": holder_kind,
            "hero_config_id": hero_config_id,
            "holder_config_id": holder_config_id,
            "camp": camp,
            "buff_type": buff_type,
            "config_id": config_id,
            "times": times,
            "start_time": start_time,
            "layer": layer,
            "origin_actor_id": origin_actor_id,
        }
        acc.update(
            frame_no=frame_no,
            holder_runtime_id=holder_runtime_id,
            times=times,
            start_time=start_time,
            layer=layer,
            origin_actor_id=origin_actor_id,
            sample=sample,
        )
        if holder_kind == "hero":
            self.hero_buff_keys_by_config_camp[f"hero={hero_config_id}|camp={camp}"].add(key)

    @staticmethod
    def _buff_holder_key(holder_kind: str, holder_meta: dict) -> str:
        if holder_kind == "hero":
            return f"hero={holder_meta.get('hero_config_id')}|camp={holder_meta.get('camp')}"
        return f"{holder_kind}_config={holder_meta.get('config_id')}|camp={holder_meta.get('camp')}"

    def _consume_hero_hit_targets(self, hero: dict, acc: _HeroAcc) -> None:
        for hit in (_hero_field(hero, "hit_target_info", ["hitTargetInfo"]) or []):
            if not isinstance(hit, dict):
                continue
            target = _coerce_int_or_none(hit.get("hit_target") or hit.get("hitTarget"))
            skill_id = _coerce_int_or_none(hit.get("skill_id") or hit.get("skillId"))
            slot_type = hit.get("slot_type") or hit.get("slotType")
            conti = _coerce_int_or_none(hit.get("conti_hit_count") or hit.get("contiHitCount"))
            if target is not None:
                acc.hit_target_runtime_ids.add(target)
            if skill_id is not None:
                acc.hit_target_skill_ids.add(skill_id)
            if slot_type is not None:
                acc.hit_target_slot_counts[str(slot_type)] += 1
            acc.hit_target_conti_hit_count.update(conti)

    def _consume_hero_take_hurts(self, hero: dict, acc: _HeroAcc) -> None:
        for hurt in (_hero_field(hero, "take_hurt_infos", ["takeHurtInfos"]) or []):
            if not isinstance(hurt, dict):
                continue
            atker = _coerce_int_or_none(hurt.get("atker"))
            hurt_value = _coerce_int_or_none(hurt.get("hurtValue") or hurt.get("hurt_value"))
            skill_slot = _coerce_int_or_none(hurt.get("skillSlot") or hurt.get("skill_slot"))
            source_type = hurt.get("sourceType") if "sourceType" in hurt else hurt.get("source_type")
            source_id = _coerce_int_or_none(hurt.get("sourceID") or hurt.get("source_id"))
            if atker is not None:
                acc.take_hurt_atkers.add(atker)
            acc.take_hurt_value.update(hurt_value)
            if skill_slot is not None:
                acc.take_hurt_skill_slots.add(skill_slot)
            if source_type is not None:
                acc.take_hurt_source_types.add(source_type)
            if source_id is not None:
                acc.take_hurt_source_ids.add(source_id)

    def _consume_hero_real_cmd(self, hero: dict, acc: _HeroAcc) -> None:
        for cmd in (_hero_field(hero, "real_cmd", ["realCmd"]) or []):
            if not isinstance(cmd, dict):
                continue
            command_type = cmd.get("command_type") or cmd.get("commandType")
            if command_type is not None:
                acc.real_cmd_types[str(command_type)] += 1
            for key, value in cmd.items():
                acc.real_cmd_keys_seen.add(str(key))
                if not isinstance(value, dict):
                    continue
                skill_id = _coerce_int_or_none(value.get("skillID") or value.get("skillId") or value.get("skill_id"))
                slot_type = _coerce_int_or_none(value.get("slotType") or value.get("slot_type"))
                actor_id = _coerce_int_or_none(value.get("actorID") or value.get("actorId") or value.get("actor_id"))
                if skill_id is not None:
                    acc.real_cmd_skill_ids.add(skill_id)
                if slot_type is not None:
                    acc.real_cmd_slot_types.add(slot_type)
                if actor_id is not None:
                    acc.real_cmd_actor_ids.add(actor_id)

    def _consume_hero_equips(self, hero: dict, acc: _HeroAcc) -> None:
        equip_state = _hero_field(hero, "equip_state", ["equipState"])
        if not isinstance(equip_state, dict):
            return
        for equip in equip_state.get("equips", []) or []:
            if not isinstance(equip, dict):
                continue
            config_id = _coerce_int_or_none(equip.get("configId") or equip.get("config_id"))
            if config_id is not None:
                acc.equip_config_ids.add(config_id)
            acc.equip_buy_price.update(_coerce_int_or_none(equip.get("buyPrice") or equip.get("buy_price")))
            acc.equip_amount.update(_coerce_int_or_none(equip.get("amount")))
            for active in equip.get("active_skill", []) or equip.get("activeSkill", []) or []:
                if not isinstance(active, dict):
                    continue
                skill_id = _coerce_int_or_none(active.get("active_skillid") or active.get("active_skill_id"))
                if skill_id is not None:
                    acc.equip_active_skill_ids.add(skill_id)
                acc.equip_active_cooldown.update(_coerce_int_or_none(active.get("cooldown")))
            for passive in equip.get("passive_skill", []) or equip.get("passiveSkill", []) or []:
                if not isinstance(passive, dict):
                    continue
                skill_id = _coerce_int_or_none(passive.get("passive_skillid") or passive.get("passive_skill_id"))
                if skill_id is not None:
                    acc.equip_passive_skill_ids.add(skill_id)
                acc.equip_passive_cooldown.update(_coerce_int_or_none(passive.get("cooldown")))

    def _consume_npc(self, npc: dict, frame_no: Optional[int] = None) -> None:
        if not isinstance(npc, dict):
            return
        config_id = _coerce_int_or_none(_npc_field(npc, "config_id", ["configId"]))
        sub_type = _npc_field(npc, "sub_type", ["subType"])
        actor_type = _npc_field(npc, "actor_type", ["actorType"])
        camp = _npc_field(npc, "camp")

        if sub_type is not None:
            self.sub_types_seen.add(sub_type)
            self._note_dtype(sub_type, self.sub_type_dtype_votes, set())
        if actor_type is not None:
            self.actor_types_seen.add(actor_type)
            self._note_dtype(actor_type, self.actor_type_dtype_votes, set())
        if camp is not None:
            self._note_dtype(camp, self.camp_dtype_votes, self.camp_values_seen)
        if config_id is not None and sub_type is not None:
            self.npc_config_id_to_sub_type.setdefault(config_id, sub_type)

        kind = self._classify_npc(actor_type, sub_type, config_id)
        self._note_kind_seen(kind, frame_no)
        self._record_actor_subtype(kind, actor_type, sub_type)
        runtime_id = _coerce_int_or_none(_npc_field(npc, "runtime_id", ["runtimeId"]))
        if runtime_id is not None:
            self.runtime_actor_index[runtime_id] = {
                "kind": kind, "camp": camp, "config_id": config_id,
                "actor_type": actor_type, "sub_type": sub_type,
            }

        loc = _npc_field(npc, "location")
        x = z = None
        if isinstance(loc, dict):
            x, y, z = _read_location(npc)
            self.map_x_range.update(x)
            self.map_z_range.update(z)
            if x is not None and z is not None:
                self.map_sum_range.update(x + z)
                self.map_diff_range.update(z - x)
            self._record_coord(
                kind,
                x, y, z, frame_no,
                {
                    "camp": camp, "config_id": config_id, "runtime_id": runtime_id,
                    "actor_type": actor_type, "sub_type": sub_type,
                },
            )

        forward = _npc_field(npc, "forward")
        forward_x = forward_z = None
        if isinstance(forward, dict):
            forward_x = _coerce_int_or_none(forward.get("x"))
            forward_z = _coerce_int_or_none(forward.get("z"))

        hp = _coerce_int_or_none(_npc_field(npc, "hp"))
        max_hp = _coerce_int_or_none(_npc_field(npc, "max_hp", ["maxHp"]))
        ep = _coerce_int_or_none(_field_or_values(npc, "ep"))
        max_ep = _coerce_int_or_none(_field_or_values(npc, "max_ep", ["maxEp"]))
        attack_range = _coerce_int_or_none(_field_or_values(npc, "attack_range"))
        sight_area = _coerce_int_or_none(_field_or_values(npc, "sight_area"))
        attack_target = _npc_field(npc, "attack_target")
        behave = _npc_field(npc, "behav_mode", ["behave"])

        if behave is not None:
            self.behave_per_actor_kind[kind].add(str(behave))

        if kind == "tower":
            acc = self.tower.setdefault(camp, _OrganAcc())
        elif kind == "crystal":
            acc = self.crystal.setdefault(camp, _OrganAcc())
        elif kind == "spring":
            acc = self.spring.setdefault(camp, _OrganAcc())
        elif kind == "soldier":
            acc = self.soldier_per_config_camp.setdefault((camp, config_id or -1), _OrganAcc())
        elif kind == "monster":
            self.monster_seen = True
            acc = self.monster_per_config.setdefault(config_id or -1, _OrganAcc())
        else:
            return
        acc.location_x.update(x)
        acc.location_z.update(z)
        acc.forward_x.update(forward_x)
        acc.forward_z.update(forward_z)
        acc.hp.update(hp)
        acc.max_hp.update(max_hp)
        acc.ep.update(ep)
        acc.max_ep.update(max_ep)
        acc.attack_range.update(attack_range)
        acc.sight_area.update(sight_area)
        for field, target in [
            ("mov_spd", acc.mov_spd),
            ("atk_spd", acc.atk_spd),
            ("phy_atk", acc.phy_atk),
            ("phy_def", acc.phy_def),
            ("mgc_atk", acc.mgc_atk),
            ("mgc_def", acc.mgc_def),
            ("phy_armor_hurt", acc.phy_armor_hurt),
            ("mgc_armor_hurt", acc.mgc_armor_hurt),
            ("crit_rate", acc.crit_rate),
            ("crit_effe", acc.crit_effe),
            ("phy_vamp", acc.phy_vamp),
            ("mgc_vamp", acc.mgc_vamp),
            ("cd_reduce", acc.cd_reduce),
            ("ctrl_reduce", acc.ctrl_reduce),
            ("kill_income", acc.kill_income),
            ("hp_recover", acc.hp_recover),
            ("ep_recover", acc.ep_recover),
        ]:
            target.update(_coerce_int_or_none(_field_or_values(npc, field)))
        abilities = _npc_field(npc, "abilities")
        if isinstance(abilities, (list, tuple)):
            acc.abilities_lengths_seen.add(len(abilities))
            _add_limited_tuple({0: acc.abilities_patterns_seen}, 0, [int(bool(v)) for v in abilities], limit=16)
        camp_visible = _npc_field(npc, "camp_visible", ["campVisible"])
        if isinstance(camp_visible, (list, tuple)):
            _add_limited_tuple({0: acc.camp_visible_patterns_seen}, 0, [int(bool(v)) for v in camp_visible], limit=16)
        if attack_target is not None:
            acc.attack_target_seen.add(attack_target)
        if behave is not None:
            acc.behave_seen.add(str(behave))
        if config_id is not None:
            acc.config_ids.add(config_id)
        for hurt in (_npc_field(npc, "hurt_hero_info", ["hurtHeroInfo"]) or []):
            if not isinstance(hurt, dict):
                continue
            target = _coerce_int_or_none(hurt.get("hurt_target") or hurt.get("hurtTarget"))
            if target is not None:
                acc.hurt_hero_targets_seen.add(target)
            acc.hurt_hero_value.update(_coerce_int_or_none(hurt.get("hurt")))
        self._consume_buff_state(
            npc,
            kind,
            frame_no=frame_no,
            holder_meta={"config_id": config_id, "camp": camp, "runtime_id": runtime_id},
        )

    def _consume_bullet(self, bullet: dict, frame_no: Optional[int] = None) -> None:
        if not isinstance(bullet, dict):
            return
        slot_type = bullet.get("slot_type") or bullet.get("slotType")
        skill_id = _coerce_int_or_none(bullet.get("skill_id"))
        camp = bullet.get("camp")
        runtime_id = _coerce_int_or_none(bullet.get("runtime_id") or bullet.get("runtimeId"))
        source_actor = _coerce_int_or_none(bullet.get("source_actor") or bullet.get("sourceActor"))
        if runtime_id is not None:
            self.bullet_runtime_ids_seen.add(runtime_id)
        if source_actor is not None:
            self.bullet_source_actor_ids_seen.add(source_actor)
            source_info = self.runtime_actor_index.get(source_actor)
            if source_info:
                source_kind = source_info.get("kind", "unknown")
                self.bullet_source_kind_counts[source_kind] += 1
                self.bullet_source_actor_samples.setdefault(str(source_actor), source_info)
            else:
                self.bullet_source_kind_counts["unknown"] += 1
        if slot_type is not None:
            self.bullet_slot_types_seen.add(slot_type)
            self.bullet_slot_type_counts[slot_type] += 1
            if skill_id is not None:
                self.bullet_slot_to_skill[slot_type].add(skill_id)
            if camp is not None:
                self.bullet_camp_slot_counts[camp][slot_type] += 1
        if skill_id is not None:
            self.bullet_skill_ids_seen.add(skill_id)
        if camp is not None:
            self.bullet_camp_seen.add(camp)
        loc = bullet.get("location")
        if isinstance(loc, dict):
            x = _coerce_int_or_none(loc.get("x"))
            y = _coerce_int_or_none(loc.get("y"))
            z = _coerce_int_or_none(loc.get("z"))
            self.map_x_range.update(x)
            self.map_z_range.update(z)
            self._record_coord(
                "bullet",
                x, y, z, frame_no,
                {
                    "camp": camp, "runtime_id": runtime_id, "source_actor": source_actor,
                    "slot_type": slot_type, "skill_id": skill_id,
                },
            )

    def _consume_frame_action(self, frame_action: dict) -> None:
        dead_action = frame_action.get("dead_action")
        if isinstance(dead_action, list):
            for event in dead_action:
                if not isinstance(event, dict):
                    continue
                death = event.get("death")
                if isinstance(death, dict):
                    key = f"{death.get('actor_type')}|{death.get('sub_type')}|{death.get('config_id')}"
                    self.dead_action_death_type_counts[key] += 1
                killer = event.get("killer")
                if isinstance(killer, dict):
                    key = f"{killer.get('actor_type')}|{killer.get('sub_type')}|{killer.get('config_id')}"
                    self.dead_action_killer_type_counts[key] += 1
                    hurt_info = killer.get("hurt_info")
                    if isinstance(hurt_info, list):
                        for hi in hurt_info:
                            if isinstance(hi, dict):
                                slot = hi.get("slot_type")
                                if slot is not None:
                                    self.skill_use_slot_types.add(str(slot))
                                hurt_type = hi.get("hurt_type")
                                if hurt_type is not None:
                                    self.dead_action_hurt_type_counts[str(hurt_type)] += 1
                                self.dead_action_hurt_value.update(_coerce_int_or_none(hi.get("hurt_val")))
                    income_info = killer.get("income_info")
                    if isinstance(income_info, dict):
                        self.dead_action_income_exp.update(_coerce_int_or_none(income_info.get("exp")))
                        self.dead_action_income_money.update(_coerce_int_or_none(income_info.get("money")))
                    single_hurt_list = killer.get("single_hurt_list")
                    if isinstance(single_hurt_list, list):
                        for single_hurt in single_hurt_list:
                            if not isinstance(single_hurt, dict):
                                continue
                            slot = single_hurt.get("slot_type")
                            if slot is not None:
                                self.dead_action_single_hurt_slot_counts[str(slot)] += 1
                            hurt_info = single_hurt.get("hurt_info")
                            if isinstance(hurt_info, dict):
                                self.dead_action_single_hurt_value.update(
                                    _coerce_int_or_none(hurt_info.get("hurt_val"))
                                )
        for key, value in frame_action.items():
            if isinstance(value, list):
                self.skill_use_events_seen[key] += len(value)

    def _consume_cake(self, cake: dict, frame_no: Optional[int]) -> None:
        if not isinstance(cake, dict):
            return
        self.cake_seen = True
        if self.cake_first_appear_frame is None and frame_no is not None:
            self.cake_first_appear_frame = int(frame_no)
        collider = cake.get("collider")
        loc = collider.get("location") if isinstance(collider, dict) else None
        if isinstance(loc, dict):
            x = _coerce_int_or_none(loc.get("x"))
            y = _coerce_int_or_none(loc.get("y"))
            z = _coerce_int_or_none(loc.get("z"))
            if x is not None and z is not None:
                tup = [x, z]
                if tup not in self.cake_camp_locations:
                    self.cake_camp_locations.append(tup)
                self._record_coord(
                    "cake",
                    x, y, z, frame_no,
                    {"config_id": cake.get("configId") or cake.get("config_id"), "radius": collider.get("radius")},
                )

    # ===== helpers =====

    def _note_kind_seen(self, kind: str, frame_no: Optional[int]) -> None:
        if frame_no is None or kind in self.kind_first_seen_frame:
            return
        self.kind_first_seen_frame[kind] = int(frame_no)

    def _record_actor_subtype(self, kind: str, actor_type, sub_type) -> None:
        actor_i = _enum_int(actor_type, STRING_ACTOR_TYPES)
        sub_i = _enum_int(sub_type, STRING_SUB_TYPES)
        key = f"actor={actor_i if actor_i is not None else actor_type}|sub={sub_i if sub_i is not None else sub_type}"
        self.npc_actor_subtype_counts[key] += 1
        self.actor_subtype_kind_counts[key][kind] += 1

    def _record_coord(self, kind: str, x, y, z, frame_no: Optional[int], meta: Optional[dict]) -> None:
        self.coord_stats[kind].update(x, y, z, frame_no, meta)
        camp = meta.get("camp") if isinstance(meta, dict) else None
        if camp is not None:
            self.coord_stats_by_camp[f"{kind}|camp={camp}"].update(x, y, z, frame_no, meta)
        if self._is_valid_map_coord(x, z):
            self.valid_coord_stats[kind].update(x, y, z, frame_no, meta)
            if camp is not None:
                self.valid_coord_stats_by_camp[f"{kind}|camp={camp}"].update(x, y, z, frame_no, meta)

    @staticmethod
    def _is_valid_map_coord(x, z) -> bool:
        if x is None or z is None:
            return False
        return abs(x) <= COORD_OUTLIER_ABS_THRESHOLD and abs(z) <= COORD_OUTLIER_ABS_THRESHOLD

    @staticmethod
    def _classify_npc(actor_type, sub_type, config_id) -> str:
        actor_i = _enum_int(actor_type, STRING_ACTOR_TYPES)
        sub_i = _enum_int(sub_type, STRING_SUB_TYPES)
        if (actor_i, sub_i) in KNOWN_ACTOR_SUBTYPES:
            return KNOWN_ACTOR_SUBTYPES[(actor_i, sub_i)]
        if sub_i == 21:
            return "tower"
        if sub_i == 23:
            return "crystal"
        if sub_i == 24:
            return "spring"
        if sub_i == 11:
            return "soldier"
        if actor_i == 1 and sub_i == 0:
            return "monster"
        if actor_i == 2:
            return "organ_unknown"
        if actor_i == 1:
            return "npc_unknown"
        return "unknown"

    @staticmethod
    def _note_dtype(value, vote_dict: defaultdict, value_set: set):
        if value is None:
            return
        vote_dict[type(value).__name__] += 1
        try:
            value_set.add(value)
        except TypeError:
            value_set.add(str(value))

    def _record_button_mask_snapshot(self, legal_action) -> None:
        try:
            head = tuple(int(v) for v in legal_action[:12])
        except (TypeError, ValueError):
            return
        self.legal_action_first_button_mask_seen.add(head)

    def _record_target_mask_snapshots(self, legal_action) -> None:
        try:
            length = len(legal_action)
        except TypeError:
            return
        try:
            if length >= 12 + 16 * 4 + 12 * 9:
                target_mask = [int(v) for v in legal_action[-12 * 9:]]
                for button_idx in range(12):
                    start = button_idx * 9
                    _add_limited_tuple(
                        self.legal_action_target_mask_unique_per_button,
                        button_idx,
                        target_mask[start:start + 9],
                    )
            elif length >= 12 + 16 * 4 + 9:
                _add_limited_tuple(
                    self.legal_action_target_mask_unique_per_button,
                    "compressed",
                    [int(v) for v in legal_action[-9:]],
                )
        except (TypeError, ValueError):
            return

    def _dump_frame(self, observation: dict, env_id: Any, frame_no: int) -> None:
        env_id_str = str(env_id).replace("/", "_").replace("\\", "_")
        path = os.path.join(self.frames_dir, f"episode_{env_id_str}_frame_{frame_no}.json")
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(observation, f, ensure_ascii=False, default=_json_default)
            self.frame_dumps_written += 1
        except OSError as exc:
            if self.logger is not None:
                self.logger.warning(f"DumpCollector failed to write {path}: {exc}")

    def _build_coordinate_analysis(self) -> dict:
        trusted_raw = _CoordAcc()
        trusted_valid = _CoordAcc()
        for kind in TRUSTED_COORD_KINDS:
            acc = self.coord_stats.get(kind)
            if acc is not None:
                trusted_raw.merge_from(acc)
            valid_acc = self.valid_coord_stats.get(kind)
            if valid_acc is not None:
                trusted_valid.merge_from(valid_acc)
        anchors = self._build_anchor_points(self.valid_coord_stats_by_camp)
        return {
            "trusted_entity_kinds": list(TRUSTED_COORD_KINDS),
            "reference_expectations": {
                "base_or_spawn_abs_coord_about": 40000,
                "outer_tower_abs_coord_about": 13000,
                "cake_abs_coord_about": 15000,
                "outlier_abs_threshold": COORD_OUTLIER_ABS_THRESHOLD,
            },
            "note": (
                "Use valid_actor_projection for map remapping. raw_actor_projection keeps "
                "sentinel/outlier coordinates for debugging. Bullet bounds are split out."
            ),
            "valid_actor_projection": trusted_valid.to_dict(),
            "raw_actor_projection": trusted_raw.to_dict(),
            "trusted_actor_projection": trusted_valid.to_dict(),
            "anchor_points_by_kind_and_camp": anchors,
            "base_distance_from_spring": self._distance_between_first_two_anchors(anchors.get("spring", {})),
            "tower_distance_from_outer_towers": self._distance_between_first_two_anchors(anchors.get("tower", {})),
            "by_kind": {str(k): v.to_dict() for k, v in sorted(self.coord_stats.items(), key=lambda kv: str(kv[0]))},
            "valid_by_kind": {
                str(k): v.to_dict() for k, v in sorted(self.valid_coord_stats.items(), key=lambda kv: str(kv[0]))
            },
            "by_kind_and_camp": {
                str(k): v.to_dict() for k, v in sorted(self.coord_stats_by_camp.items(), key=lambda kv: str(kv[0]))
            },
            "valid_by_kind_and_camp": {
                str(k): v.to_dict() for k, v in sorted(self.valid_coord_stats_by_camp.items(), key=lambda kv: str(kv[0]))
            },
        }

    def _build_anchor_points(self, source: Optional[dict] = None) -> dict:
        source = source or self.valid_coord_stats_by_camp
        anchors = defaultdict(dict)
        for key, acc in source.items():
            if "|" not in str(key) or acc.x.min is None or acc.z.min is None:
                continue
            kind, camp_part = str(key).split("|", 1)
            camp = camp_part.replace("camp=", "", 1)
            anchors[kind][camp] = {
                "x_mid": (acc.x.min + acc.x.max) / 2 if acc.x.max is not None else acc.x.min,
                "z_mid": (acc.z.min + acc.z.max) / 2 if acc.z.max is not None else acc.z.min,
                "x_range": acc.x.to_dict(),
                "z_range": acc.z.to_dict(),
                "count": acc.count,
                "first_frame": acc.first_frame,
            }
        return {k: dict(v) for k, v in anchors.items()}

    @staticmethod
    def _distance_between_first_two_anchors(points: dict):
        if not isinstance(points, dict) or len(points) < 2:
            return None
        ordered = sorted(points.items(), key=lambda kv: str(kv[0]))[:2]
        (_, a), (_, b) = ordered
        dx = a["x_mid"] - b["x_mid"]
        dz = a["z_mid"] - b["z_mid"]
        return (dx * dx + dz * dz) ** 0.5

    def _build_summary(self) -> dict:
        layout = "unknown"
        if self.hero_layout_votes:
            layout = max(self.hero_layout_votes.items(), key=lambda kv: kv[1])[0]
        sub_type_dtype = self._top_dtype(self.sub_type_dtype_votes)
        camp_dtype = self._top_dtype(self.camp_dtype_votes)
        actor_type_dtype = self._top_dtype(self.actor_type_dtype_votes)

        return {
            "schema_version": 1,
            "episodes_observed": self.episodes_observed,
            "frames_observed": self.frames_observed,
            "frame_no_max_observed": self.frame_no_max,
            "frame_dumps_written": self.frame_dumps_written,
            "protocol_check": {
                "hero_state_layout": layout,
                "hero_layout_votes": dict(self.hero_layout_votes),
                "sub_type_dtype": sub_type_dtype,
                "camp_dtype": camp_dtype,
                "actor_type_dtype": actor_type_dtype,
            },
            "legal_action_lengths_seen": sorted(self.legal_action_lengths),
            "sub_action_mask_shape_seen": [list(s) for s in sorted(self.sub_action_mask_shape)],
            "legal_action_button_mask_unique_first12": [
                list(t) for t in sorted(self.legal_action_first_button_mask_seen)
            ],
            "sub_action_mask_per_button": {
                str(k): v for k, v in sorted(self.sub_action_mask_per_button.items())
            },
            "sub_action_mask_unique_per_button": {
                str(k): [list(t) for t in sorted(v)]
                for k, v in sorted(self.sub_action_mask_unique_per_button.items())
            },
            "legal_action_target_mask_unique_per_button": {
                str(k): [list(t) for t in sorted(v)]
                for k, v in sorted(self.legal_action_target_mask_unique_per_button.items())
            },
            "actor_types_seen": sorted(self.actor_types_seen, key=str),
            "sub_types_seen": sorted(self.sub_types_seen, key=str),
            "camp_values_seen": sorted(self.camp_values_seen, key=str),
            "npc_config_id_to_sub_type": {
                str(k): v for k, v in sorted(self.npc_config_id_to_sub_type.items())
            },
            "known_actor_subtype_mapping": {
                f"actor={a}|sub={s}": kind for (a, s), kind in KNOWN_ACTOR_SUBTYPES.items()
            },
            "npc_actor_subtype_counts": dict(sorted(self.npc_actor_subtype_counts.items(), key=lambda kv: str(kv[0]))),
            "actor_subtype_kind_counts": {
                str(k): dict(v) for k, v in sorted(self.actor_subtype_kind_counts.items(), key=lambda kv: str(kv[0]))
            },
            "kind_first_seen_frame": dict(sorted(self.kind_first_seen_frame.items(), key=lambda kv: str(kv[0]))),
            "bullet_slot_types_seen": sorted(self.bullet_slot_types_seen, key=str),
            "bullet_skill_ids_seen": sorted(self.bullet_skill_ids_seen),
            "bullet_camp_values_seen": sorted(self.bullet_camp_seen, key=str),
            "bullet_runtime_ids_seen_sample": sorted(self.bullet_runtime_ids_seen)[:128],
            "bullet_source_actor_ids_seen": sorted(self.bullet_source_actor_ids_seen),
            "bullet_source_kind_counts": dict(self.bullet_source_kind_counts),
            "bullet_source_actor_samples": self.bullet_source_actor_samples,
            "bullet_slot_type_counts": dict(self.bullet_slot_type_counts),
            "bullet_slot_to_skill_mapping": {
                str(k): sorted(v) for k, v in sorted(self.bullet_slot_to_skill.items())
            },
            "bullet_camp_slot_counts": {
                str(c): dict(d) for c, d in sorted(self.bullet_camp_slot_counts.items())
            },
            "behave_seen_per_actor_kind": {
                kind: sorted(values, key=str) for kind, values in self.behave_per_actor_kind.items()
            },
            "frame_action_top_keys_seen": sorted(self.dead_action_keys_seen),
            "dead_action_death_type_counts": dict(self.dead_action_death_type_counts),
            "dead_action_killer_type_counts": dict(self.dead_action_killer_type_counts),
            "dead_action_hurt_type_counts": dict(self.dead_action_hurt_type_counts),
            "dead_action_hurt_value": self.dead_action_hurt_value.to_dict(),
            "dead_action_income_exp": self.dead_action_income_exp.to_dict(),
            "dead_action_income_money": self.dead_action_income_money.to_dict(),
            "dead_action_single_hurt_slot_counts": dict(self.dead_action_single_hurt_slot_counts),
            "dead_action_single_hurt_value": self.dead_action_single_hurt_value.to_dict(),
            "skill_use_events": dict(self.skill_use_events_seen),
            "skill_use_slot_types": sorted(self.skill_use_slot_types, key=str),
            "buff_skill_ids_seen": sorted(self.buff_skill_ids_seen),
            "buff_mark_id_to_max_layer": {str(k): v for k, v in sorted(self.buff_mark_id_to_max_layer.items())},
            "buff_origin_actor_ids_seen": sorted(self.buff_origin_actor_ids_seen),
            "buff_seen_by_kind": {
                str(k): sorted(v) for k, v in sorted(self.buff_seen_by_kind.items(), key=lambda kv: str(kv[0]))
            },
            "buff_state_key_patterns": dict(sorted(self.buff_state_key_patterns.items())),
            "buff_mark_container_counts": dict(sorted(self.buff_mark_container_counts.items())),
            "buff_mark_field_key_patterns": dict(sorted(self.buff_mark_field_key_patterns.items())),
            "buff_mark_holder_counts": dict(sorted(self.buff_mark_holder_counts.items())),
            "buff_mark_raw_samples": list(self.buff_mark_raw_samples),
            "buff_detail_by_holder": {
                str(k): v.to_dict() for k, v in sorted(self.buff_detail_by_holder.items(), key=lambda kv: str(kv[0]))
            },
            "hero_buff_keys_by_config_camp": {
                str(k): sorted(v) for k, v in sorted(self.hero_buff_keys_by_config_camp.items(), key=lambda kv: str(kv[0]))
            },
            "map_xz_bounds_observed": {
                "x_min": self.map_x_range.min, "x_max": self.map_x_range.max,
                "z_min": self.map_z_range.min, "z_max": self.map_z_range.max,
                "sum_min": self.map_sum_range.min, "sum_max": self.map_sum_range.max,
                "diff_min": self.map_diff_range.min, "diff_max": self.map_diff_range.max,
            },
            "coordinate_analysis": self._build_coordinate_analysis(),
            "heroes": {str(cid): acc.to_dict() for cid, acc in sorted(self.heroes.items())},
            "hero_by_config_camp_debug": {
                str(k): acc.to_dict()
                for k, acc in sorted(self.hero_by_config_camp_debug.items(), key=lambda kv: str(kv[0]))
            },
            "tower_per_camp": {str(c): acc.to_dict() for c, acc in sorted(self.tower.items(), key=lambda kv: str(kv[0]))},
            "crystal_per_camp": {str(c): acc.to_dict() for c, acc in sorted(self.crystal.items(), key=lambda kv: str(kv[0]))},
            "spring_per_camp": {str(c): acc.to_dict() for c, acc in sorted(self.spring.items(), key=lambda kv: str(kv[0]))},
            "soldier_per_config_camp": {
                f"camp={c}|config_id={cid}": acc.to_dict()
                for (c, cid), acc in sorted(self.soldier_per_config_camp.items(), key=lambda kv: str(kv[0]))
            },
            "monster_seen": self.monster_seen,
            "monster_per_config": {
                str(cid): acc.to_dict() for cid, acc in sorted(self.monster_per_config.items(), key=lambda kv: str(kv[0]))
            },
            "cake_seen": self.cake_seen,
            "cake_first_appear_frame": self.cake_first_appear_frame,
            "cake_locations_observed": list(self.cake_camp_locations),
        }

    @staticmethod
    def _top_dtype(votes: defaultdict) -> str:
        if not votes:
            return "unknown"
        return max(votes.items(), key=lambda kv: kv[1])[0]


def _json_default(o):
    # Make numpy / set / bytes serializable for the frame dumps.
    try:
        import numpy as _np
        if isinstance(o, _np.ndarray):
            return o.tolist()
        if isinstance(o, (_np.integer,)):
            return int(o)
        if isinstance(o, (_np.floating,)):
            return float(o)
    except ImportError:
        pass
    if isinstance(o, set):
        return sorted(o, key=str)
    if isinstance(o, bytes):
        try:
            return o.decode("utf-8", errors="replace")
        except Exception:
            return repr(o)
    return repr(o)


# ----- self-test -----

def _self_test():
    """Cheap structural test: build a fake observation and check the summary keys."""
    fake_hero_blue = {
        "player_id": 100,
        "config_id": 112,
        "runtime_id": 1001,
        "actor_type": "ACTOR_HERO",
        "sub_type": 0,
        "camp": "PLAYERCAMP_1",
        "behav_mode": "State_Idle",
        "location": {"x": -42000, "z": 0, "y": 0},
        "hp": 800,
        "max_hp": 800,
        "values": {"ep": 0, "max_ep": 100, "hp_recover": 5, "ep_recover": 1},
        "attack_range": 700,
        "sight_area": 8000,
        "mov_spd": 380,
        "atk_spd": 0,
        "phy_atk": 100,
        "phy_def": 60,
        "mgc_atk": 0,
        "mgc_def": 50,
        "level": 1,
        "money": 800,
        "moneyCnt": 800,
        "killCnt": 0,
        "deadCnt": 0,
        "assistCnt": 0,
        "isInGrass": False,
        "skill_state": {
            "slot_states": [
                {"slot_type": "SLOT_SKILL_0", "cooldown_max": 0, "level": 1},
                {"slot_type": "SLOT_SKILL_1", "cooldown_max": 6000, "level": 0},
                {"slot_type": "SLOT_SKILL_2", "cooldown_max": 8000, "level": 0},
                {"slot_type": "SLOT_SKILL_3", "cooldown_max": 30000, "level": 0},
                {"slot_type": "SLOT_SKILL_4", "cooldown_max": 0, "level": 1},
                {"slot_type": "SLOT_SKILL_5", "configId": 80115, "cooldown_max": 90000, "level": 1},
                {"slot_type": "SLOT_SKILL_6", "cooldown_max": 0, "level": 1},
            ]
        },
        "buff_state": {
            "buff_skills": [{"configId": 90015, "times": 1, "startTime": 12}],
            "buff_marks": [{"configId": 91001, "origin_actorId": 1001, "layer": 2}],
        },
        "passive_skill": [{"passive_skillid": 11200}],
    }
    fake_tower_blue = {
        "config_id": 8001, "runtime_id": 2001, "actor_type": "ACTOR_ORGAN",
        "sub_type": "ACTOR_SUB_TOWER", "camp": "PLAYERCAMP_1",
        "location": {"x": -30000, "z": 0, "y": 0},
        "hp": 9999, "max_hp": 9999, "attack_range": 7000, "behav_mode": "Attack_Move",
        "attack_target": 0,
    }
    fake_obs = {
        "env_id": "demo_env",
        "player_id": 1001,
        "player_camp": 1,
        "legal_action": [1] * 184,
        "sub_action_mask": [[1, 0, 0, 0, 0, 0]] * 12,
        "frame_state": {
            "frame_no": 0,
            "hero_states": [fake_hero_blue],
            "npc_states": [fake_tower_blue],
            "bullets": [{"slot_type": "SLOT_SKILL_0", "skill_id": 0, "camp": "PLAYERCAMP_1"}],
            "cakes": [],
            "frame_action": {"dead_action": []},
            "map_state": False,
        },
    }
    coll = DumpCollector("agent_ppo/debug/dumps", sample_frame_interval=1, max_frame_dumps=0)
    coll.on_episode_start()
    coll.on_frame(fake_obs, frame_no=0, env_id="demo_env")
    data = coll._build_summary()
    expected_top_level = {
        "schema_version", "episodes_observed", "frames_observed", "frame_no_max_observed",
        "protocol_check", "actor_types_seen", "sub_types_seen", "heroes",
        "hero_by_config_camp_debug", "tower_per_camp", "monster_seen", "cake_seen",
    }
    missing = expected_top_level - set(data.keys())
    assert not missing, f"summary missing keys: {missing}"
    assert "112" in data["heroes"], "hero 112 not recorded"
    assert data["hero_by_config_camp_debug"], "hero camp debug not recorded"
    assert data["tower_per_camp"], "tower not recorded"
    assert data["buff_detail_by_holder"], "buff details not recorded"
    print("DumpCollector self-test OK; summary keys:", sorted(data.keys()))


if __name__ == "__main__":
    _self_test()
