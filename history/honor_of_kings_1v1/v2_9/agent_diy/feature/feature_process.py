#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Structured observation builder for the frozen 4833-dim agent_diy schema.

The layout is defined by docs/feature_engineering_design.md:
global, self/enemy hero, self/enemy soldiers, monster, self/enemy tower,
enemy bullets, target candidates.
"""

import hashlib
import math
from collections import deque

import numpy as np

from agent_diy.conf.conf import Args, Config, GameConfig


HERO_TYPE = {0, "0", "ACTOR_TYPE_HERO"}
MONSTER_TYPE = {1, "1", "ACTOR_TYPE_MONSTER", "ACTOR_MONSTER"}
ORGAN_TYPE = {2, "2", "ACTOR_TYPE_ORGAN", "ACTOR_ORGAN"}
HERO_SUB_TYPES = {0, "0", "ACTOR_SUB_HERO"}
MONSTER_SUB_TYPES = {0, "0", "ACTOR_SUB_MONSTER"}
SOLDIER_SUB_TYPES = {11, "11", "ACTOR_SUB_SOLDIER"}
TOWER_SUB_TYPES = {21, "21", "ACTOR_SUB_TOWER"}
CRYSTAL_SUB_TYPES = {23, "23", "ACTOR_SUB_CRYSTAL"}
SPRING_SUB_TYPES = {24, "24", "ACTOR_SUB_TOWER_SPRING"}

BEHAVIOR_COMMON = [0, 1, 2]
MONSTER_BEHAVIORS = [0, 23, 26]
SLOT_KEYS = {
    0: "attack",
    1: "skill1",
    2: "skill2",
    3: "skill3",
    4: "recover",
    5: "summoner",
}


def _clip(value, lo, hi):
    return max(lo, min(hi, value))


def _safe_float(value, default=0.0):
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value, default=0):
    try:
        if value is None:
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def _safe_div(num, den, default=0.0):
    den = _safe_float(den, 0.0)
    if den == 0.0:
        return default
    return _safe_float(num, 0.0) / den


def _get(obj, key, default=0):
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _get_any(obj, keys, default=0):
    for key in keys:
        value = _get(obj, key, None)
        if value is not None:
            return value
    return default


def _one_hot(index, size):
    values = [0.0] * size
    if 0 <= index < size:
        values[index] = 1.0
    return values


def _bucket(value, edges):
    for idx, edge in enumerate(edges):
        if value <= edge:
            return idx
    return len(edges)


def _stable_hash(value, buckets):
    payload = str(value).encode("utf-8")
    return int(hashlib.md5(payload).hexdigest(), 16) % buckets


class FeatureProcess:
    def __init__(self, camp, logger=None):
        self.logger = logger
        self.reset(camp)

    def reset(self, camp):
        self.main_camp = camp
        self.transform_camp2_to_camp1 = self._camp_key(camp) == 2
        self.frame_no = 0
        self.last_money = {"self": None, "enemy": None}
        self.last_hp = {"self": None, "enemy": None}
        self.last_tower_hp = {"self": None, "enemy": None}
        self.last_buff_set = {"self": set(), "enemy": set()}
        self.recent_hit_events = deque(maxlen=120)
        self.recent_take_hurt_events = deque(maxlen=120)
        self.recent_skill_success = {"self": {}, "enemy": {}}
        self.recent_recover_or_cake_attempt = {"self": -999999, "enemy": -999999}
        self.recent_recover_attempt_hp = {"self": 0.0, "enemy": 0.0}
        self.recent_recover_attempt_type = {"self": "", "enemy": ""}
        self.next_cake_frame = {"self": 1778, "enemy": 1778}
        self.last_cake_exists = {"self": False, "enemy": False}
        self.bullet_cache = {}
        self.runtime_index = {}
        self.self_pos = None
        self.enemy_pos = None
        self.self_hero = None
        self.enemy_hero = None
        self.self_tower = None
        self.enemy_tower = None
        self.self_soldiers = []
        self.enemy_soldiers = []
        self.target_enemy_soldiers = []
        self.monster = None
        self.legal_action = None
        self.raw_target_legal = None

    def process_feature(self, observation):
        frame_state = observation["frame_state"]
        player_id = observation.get("player_id")
        player_camp = observation.get("player_camp", observation.get("camp", self.main_camp))
        if player_camp is not None and self._camp_key(player_camp) != self._camp_key(self.main_camp):
            self.reset(player_camp)

        self.frame_no = _safe_int(frame_state.get("frame_no", frame_state.get("frameNo", 0)))
        self.legal_action = np.asarray(observation.get("legal_action", []), dtype=np.float32).reshape(-1)
        self.raw_target_legal = self._target_legal_from_observation(self.legal_action)

        heroes = list(frame_state.get("hero_states", []))
        npcs = list(frame_state.get("npc_states", []))
        bullets = list(frame_state.get("bullets", []))
        cakes = list(frame_state.get("cakes", []))

        self.self_hero, self.enemy_hero = self._split_heroes(heroes, player_id)
        self.self_pos = self._position(self.self_hero)
        self.enemy_pos = self._position(self.enemy_hero)
        self.self_tower, self.enemy_tower = self._split_towers(npcs)
        self.self_soldiers, self.enemy_soldiers, self.target_enemy_soldiers = self._split_soldiers(npcs)
        self.monster = self._select_monster(npcs)
        self.self_cake, self.enemy_cake = self._split_cakes(cakes)
        self._update_cake_attempts()

        self._build_runtime_index(heroes, npcs)
        self._update_recent_events()
        hero_bullets, tower_bullets = self._split_enemy_bullets(bullets)

        feature = []
        feature.extend(self._process_global())
        feature.extend(self._process_hero(self.self_hero, is_enemy=False))
        feature.extend(self._process_hero(self.enemy_hero, is_enemy=True))
        for soldier in self.self_soldiers:
            feature.extend(self._process_soldier(soldier, is_enemy=False, target_index=None))
        for soldier in self.enemy_soldiers:
            target_index = self.target_enemy_soldiers.index(soldier) + 3 if soldier in self.target_enemy_soldiers else None
            feature.extend(self._process_soldier(soldier, is_enemy=True, target_index=target_index))
        feature.extend(self._process_monster(self.monster))
        feature.extend(self._process_tower(self.self_tower, is_enemy=False, cake=self.self_cake))
        feature.extend(self._process_tower(self.enemy_tower, is_enemy=True, cake=self.enemy_cake))
        for bullet in hero_bullets:
            feature.extend(self._process_bullet(bullet))
        for _ in range(Args.BULLET_HERO_THREAT_SLOTS - len(hero_bullets)):
            feature.extend([0.0] * Args.DIM_BULLET)
        for bullet in tower_bullets:
            feature.extend(self._process_bullet(bullet))
        for _ in range(Args.BULLET_TOWER_SLOTS - len(tower_bullets)):
            feature.extend([0.0] * Args.DIM_BULLET)
        for target_index, entity in enumerate(self._target_entities()):
            feature.extend(self._process_target(entity, target_index))

        self._update_history_after_frame(bullets)
        if len(feature) != Config.FEATURE_DIM:
            raise ValueError(f"agent_diy feature dim mismatch: {len(feature)} != {Config.FEATURE_DIM}")
        return np.asarray(feature, dtype=np.float32).tolist()

    def _camp_key(self, camp):
        if camp in (1, "1", "PLAYERCAMP_1", "blue_camp"):
            return 1
        if camp in (2, "2", "PLAYERCAMP_2", "red_camp"):
            return 2
        return camp

    def _same_camp(self, camp_a, camp_b):
        return self._camp_key(camp_a) == self._camp_key(camp_b)

    def _is_enemy_camp(self, camp):
        camp_key = self._camp_key(camp)
        return camp_key in (1, 2) and camp_key != self._camp_key(self.main_camp)

    def _runtime_id(self, obj):
        return _get_any(obj or {}, ["runtime_id", "runtimeId", "player_id", "playerId", "id"], None)

    def _config_id(self, obj):
        return _safe_int(_get_any(obj or {}, ["config_id", "configId"], 0), 0)

    def _actor_type(self, obj):
        return _get_any(obj or {}, ["actor_type", "actorType"], None)

    def _sub_type(self, obj):
        return _get_any(obj or {}, ["sub_type", "subType"], None)

    def _location_obj(self, obj):
        if obj is None:
            return None
        collider = _get(obj, "collider", None)
        if collider:
            return _get(collider, "location", None)
        return _get(obj, "location", None)

    def _project_location(self, obj_or_loc):
        loc = obj_or_loc
        if obj_or_loc is not None and not ("x" in obj_or_loc if isinstance(obj_or_loc, dict) else hasattr(obj_or_loc, "x")):
            loc = self._location_obj(obj_or_loc)
        if not loc:
            return None
        x = _safe_float(_get(loc, "x", None), None)
        z = _safe_float(_get(loc, "z", None), None)
        if x is None or z is None:
            return None
        if abs(x) > Args.RAW_COORD_ABS_LIMIT or abs(z) > Args.RAW_COORD_ABS_LIMIT:
            return None
        if self.transform_camp2_to_camp1:
            x, z = -x, -z
        lane = (x + z) / Args.SQRT2
        width = (x - z) / Args.SQRT2
        return lane, width

    def _position(self, obj):
        return self._project_location(obj)

    def _dist(self, a, b, default=1e9):
        if a is None or b is None:
            return default
        return math.dist(a, b)

    def _dist_ratio(self, a, b, scale=45000.0):
        return _clip(self._dist(a, b) / scale, 0.0, 1.0)

    def _valid_location(self, obj):
        return self._position(obj) is not None

    def _split_heroes(self, heroes, player_id):
        self_hero, enemy_hero = None, None
        for hero in heroes:
            if player_id is not None and self._runtime_id(hero) == player_id:
                self_hero = hero
                break
        if self_hero is None:
            for hero in heroes:
                if self._same_camp(_get(hero, "camp", None), self.main_camp):
                    self_hero = hero
                    break
        for hero in heroes:
            if hero is not self_hero:
                enemy_hero = hero
                break
        return self_hero, enemy_hero

    def _is_tower(self, npc):
        return self._actor_type(npc) in ORGAN_TYPE and self._sub_type(npc) in TOWER_SUB_TYPES

    def _is_soldier(self, npc):
        return self._actor_type(npc) in MONSTER_TYPE and self._sub_type(npc) in SOLDIER_SUB_TYPES

    def _is_monster(self, npc):
        return self._actor_type(npc) in MONSTER_TYPE and self._sub_type(npc) in MONSTER_SUB_TYPES

    def _split_towers(self, npcs):
        self_towers, enemy_towers = [], []
        for npc in npcs:
            if not self._is_tower(npc):
                continue
            if self._same_camp(_get(npc, "camp", None), self.main_camp):
                self_towers.append(npc)
            else:
                enemy_towers.append(npc)
        self_tower = self._select_nearest_anchor(self_towers, Args.SELF_TOWER_ANCHOR)
        enemy_tower = self._select_nearest_anchor(enemy_towers, Args.ENEMY_TOWER_ANCHOR)
        return self_tower, enemy_tower

    def _select_nearest_anchor(self, units, anchor):
        if not units:
            return None
        return sorted(units, key=lambda unit: self._dist(self._position(unit), anchor))[0]

    def _soldier_sort_key(self, soldier, is_enemy):
        pos = self._position(soldier)
        hp_ratio = _safe_div(_get_any(soldier, ["hp", "HP"], 0), _get_any(soldier, ["max_hp", "maxHp", "maxHP"], 0), 0.0)
        lane = pos[0] if pos else 0.0
        frontline_score = -abs(lane)
        progress = lane if is_enemy else -lane
        return (-frontline_score, -progress, self._dist(pos, self.self_pos), hp_ratio, self._runtime_id(soldier) or 0)

    def _split_soldiers(self, npcs):
        self_soldiers, enemy_soldiers = [], []
        for npc in npcs:
            if not self._is_soldier(npc):
                continue
            if self._same_camp(_get(npc, "camp", None), self.main_camp):
                self_soldiers.append(npc)
            else:
                enemy_soldiers.append(npc)
        self_soldiers = sorted(self_soldiers, key=lambda unit: self._soldier_sort_key(unit, False))
        enemy_targets = sorted(enemy_soldiers, key=lambda unit: self._soldier_sort_key(unit, True))
        self_slots = self_soldiers[: Args.SOLDIER_PER_SIDE]
        enemy_slots = enemy_targets[: Args.SOLDIER_PER_SIDE]
        self_slots += [None] * (Args.SOLDIER_PER_SIDE - len(self_slots))
        enemy_slots += [None] * (Args.SOLDIER_PER_SIDE - len(enemy_slots))
        return self_slots, enemy_slots, enemy_targets[:4]

    def _select_monster(self, npcs):
        monsters = [npc for npc in npcs if self._is_monster(npc)]
        if not monsters:
            return None
        return sorted(monsters, key=lambda unit: self._dist(self._position(unit), self.self_pos))[0]

    def _split_cakes(self, cakes):
        self_cake, enemy_cake = None, None
        for cake in cakes:
            pos = self._position(cake)
            if pos is None:
                continue
            if self._dist(pos, Args.SELF_CAKE_ANCHOR) <= self._dist(pos, Args.ENEMY_CAKE_ANCHOR):
                self_cake = cake if self_cake is None else self_cake
            else:
                enemy_cake = cake if enemy_cake is None else enemy_cake
        return self_cake, enemy_cake

    def _build_runtime_index(self, heroes, npcs):
        self.runtime_index = {}
        for hero in heroes:
            runtime_id = self._runtime_id(hero)
            if runtime_id is not None:
                self.runtime_index[runtime_id] = {"kind": "hero", "entity": hero}
        for npc in npcs:
            runtime_id = self._runtime_id(npc)
            if runtime_id is None:
                continue
            if self._is_soldier(npc):
                kind = "soldier"
            elif self._is_tower(npc):
                kind = "tower"
            elif self._is_monster(npc):
                kind = "monster"
            elif self._actor_type(npc) in ORGAN_TYPE:
                kind = "organ"
            else:
                kind = "unknown"
            self.runtime_index[runtime_id] = {"kind": kind, "entity": npc}

    def _split_enemy_bullets(self, bullets):
        enemy_or_unknown = []
        tower_bullets = []
        for bullet in bullets:
            camp = _get(bullet, "camp", None)
            if self._same_camp(camp, self.main_camp):
                continue
            source = self.runtime_index.get(_get(bullet, "source_actor", None), {})
            source_kind = source.get("kind", "unknown")
            if source_kind in ("tower", "organ"):
                tower_bullets.append(bullet)
            else:
                enemy_or_unknown.append(bullet)
        enemy_or_unknown = sorted(enemy_or_unknown, key=self._bullet_threat_score, reverse=True)
        tower_bullets = sorted(tower_bullets, key=lambda bullet: self._dist(self._position(bullet), self.self_pos))
        return enemy_or_unknown[: Args.BULLET_HERO_THREAT_SLOTS], tower_bullets[: Args.BULLET_TOWER_SLOTS]

    def _target_legal_from_observation(self, legal_action):
        if legal_action.shape[0] == Config.LEGAL_ACTION_DIM:
            return legal_action[-Config.LABEL_SIZE_LIST[-1] :]
        if legal_action.shape[0] == Config.RAW_LEGAL_ACTION_DIM:
            target = legal_action[-Config.LABEL_SIZE_LIST[0] * Config.LABEL_SIZE_LIST[-1] :]
            return target.reshape(Config.LABEL_SIZE_LIST[0], Config.LABEL_SIZE_LIST[-1]).max(axis=0)
        return np.ones([Config.LABEL_SIZE_LIST[-1]], dtype=np.float32)

    def _legal_summary(self):
        if self.legal_action is None or self.legal_action.shape[0] == 0:
            button = np.ones([Config.LABEL_SIZE_LIST[0]], dtype=np.float32)
            target = np.ones([Config.LABEL_SIZE_LIST[-1]], dtype=np.float32)
        elif self.legal_action.shape[0] == Config.LEGAL_ACTION_DIM:
            button = self.legal_action[: Config.LABEL_SIZE_LIST[0]]
            target = self.legal_action[-Config.LABEL_SIZE_LIST[-1] :]
        elif self.legal_action.shape[0] == Config.RAW_LEGAL_ACTION_DIM:
            button = self.legal_action[: Config.LABEL_SIZE_LIST[0]]
            target = self.raw_target_legal
        else:
            button = np.ones([Config.LABEL_SIZE_LIST[0]], dtype=np.float32)
            target = np.ones([Config.LABEL_SIZE_LIST[-1]], dtype=np.float32)
        target_summary = [
            float(target[1] > 0),
            float(target[2] > 0),
            float(np.max(target[3:7]) > 0),
            float(target[7] > 0 or target[8] > 0),
        ]
        return list((button > 0).astype(np.float32)) + target_summary

    def _process_position(self, pos):
        if pos is None:
            return [0.0] * Args.DIM_POSITION
        lane, width = pos
        values = []
        values.extend(self._axis_one_hot(lane, -Args.CENTER_LANE_HALF, Args.CENTER_LANE_HALF, Args.CENTER_LANE_UNIT, Args.DIM_CENTER_LANE))
        values.extend(self._axis_one_hot(width, -Args.CENTER_WIDTH_HALF, Args.CENTER_WIDTH_HALF, Args.CENTER_WIDTH_UNIT, Args.DIM_CENTER_WIDTH))
        values.extend(self._axis_one_hot(lane, -Args.GLOBAL_LANE_HALF, Args.GLOBAL_LANE_HALF, Args.GLOBAL_LANE_UNIT, Args.DIM_GLOBAL_LANE))
        values.extend(self._axis_one_hot(width, -Args.GLOBAL_WIDTH_HALF, Args.GLOBAL_WIDTH_HALF, Args.GLOBAL_WIDTH_UNIT, Args.DIM_GLOBAL_WIDTH))
        center_lane_norm = _clip(lane / Args.CENTER_LANE_HALF, -1.0, 1.0)
        center_width_norm = _clip(width / Args.CENTER_WIDTH_HALF, -1.0, 1.0)
        global_lane_norm = _clip(lane / Args.GLOBAL_LANE_HALF, -1.0, 1.0)
        global_width_norm = _clip(width / Args.GLOBAL_WIDTH_HALF, -1.0, 1.0)
        center_dist = _clip(math.sqrt(center_lane_norm ** 2 + center_width_norm ** 2), 0.0, 1.0)
        global_dist = _clip(math.sqrt(global_lane_norm ** 2 + global_width_norm ** 2), 0.0, 1.0)
        values.extend([center_lane_norm, center_width_norm, global_lane_norm, global_width_norm, center_dist, global_dist])
        assert len(values) == Args.DIM_POSITION
        return values

    def _axis_one_hot(self, value, lo, hi, unit, dim):
        inner_bins = dim - 2
        if value < lo:
            return _one_hot(0, dim)
        if value > hi:
            return _one_hot(dim - 1, dim)
        idx = int(math.floor((value - lo) / unit)) + 1
        return _one_hot(max(1, min(idx, inner_bins)), dim)

    def _process_unit_core(self, unit):
        if unit is None:
            return [0.0] * Args.DIM_UNIT_CORE
        pos = self._position(unit)
        exist_visible = [1.0, float(pos is not None)]
        camp = _get(unit, "camp", None)
        if self._same_camp(camp, self.main_camp):
            camp_relation = [1.0, 0.0, 0.0]
        elif self._camp_key(camp) in (1, 2):
            camp_relation = [0.0, 1.0, 0.0]
        else:
            camp_relation = [0.0, 0.0, 1.0]
        values = exist_visible + camp_relation + self._hp_state(unit) + self._attack_target_type(unit)
        values.extend(self._process_position(pos))
        assert len(values) == Args.DIM_UNIT_CORE
        return values

    def _hp_state(self, unit):
        hp = _safe_float(_get_any(unit, ["hp", "HP"], None), None)
        max_hp = _safe_float(_get_any(unit, ["max_hp", "maxHp", "maxHP"], None), None)
        known = hp is not None and max_hp is not None and max_hp > 0
        hp = hp or 0.0
        max_hp = max_hp or 0.0
        ratio = _clip(hp / max_hp, 0.0, 1.0) if known else 0.0
        ratio_bucket = 0 if hp <= 0 else _bucket(ratio, [0.2, 0.4, 0.6, 0.8]) + 1
        hp_abs_bucket = _bucket(hp, [500, 1500, 3000, 6000])
        values = [
            ratio,
            _clip(max_hp / Args.HERO_HP_SCALE, 0.0, 1.0),
            float(hp > 0),
            float(known),
        ]
        values.extend(_one_hot(ratio_bucket, 6))
        values.extend(_one_hot(hp_abs_bucket, 5))
        values.append(float(0.0 < ratio <= 0.25))
        assert len(values) == 16
        return values

    def _attack_target_type(self, unit):
        target = _get(unit, "attack_target", None)
        info = self.runtime_index.get(target)
        if not target or info is None:
            return [1.0, 0.0, 0.0, 0.0, 0.0]
        entity = info["entity"]
        kind = info["kind"]
        if kind == "hero":
            if self._same_camp(_get(entity, "camp", None), self.main_camp):
                return [0.0, 1.0, 0.0, 0.0, 0.0]
            return [0.0, 0.0, 1.0, 0.0, 0.0]
        if kind == "soldier":
            return [0.0, 0.0, 0.0, 1.0, 0.0]
        return [0.0, 0.0, 0.0, 0.0, 1.0]

    def _process_global(self):
        self_hp = self._hp_ratio(self.self_hero)
        enemy_hp = self._hp_ratio(self.enemy_hero)
        self_tower_hp = self._hp_ratio(self.self_tower)
        enemy_tower_hp = self._hp_ratio(self.enemy_tower)
        values = []
        values.extend(self._hero_id_one_hot(self.self_hero))
        values.extend(self._hero_id_one_hot(self.enemy_hero))
        values.extend([float(self._camp_key(self.main_camp) == 1), float(self.transform_camp2_to_camp1)])
        values.extend(self._time_features())
        values.extend([self_tower_hp, enemy_tower_hp, _clip(self_tower_hp - enemy_tower_hp, -1.0, 1.0), 0.0, 0.0, 0.0])
        self_in_enemy = self._in_tower_range(self.self_hero, self.enemy_tower, is_enemy_tower=True)
        enemy_in_self = self._in_tower_range(self.enemy_hero, self.self_tower, is_enemy_tower=False)
        self_targeted = self._targeted_by_tower(self.self_hero, self.enemy_tower)
        enemy_targeted = self._targeted_by_tower(self.enemy_hero, self.self_tower)
        self_enemy_dist = self._dist_ratio(self.self_pos, self._tower_pos(self.enemy_tower, True))
        enemy_self_dist = self._dist_ratio(self.enemy_pos, self._tower_pos(self.self_tower, False))
        values.extend(
            [
                self_hp,
                enemy_hp,
                _clip(self_hp - enemy_hp, -1.0, 1.0),
                float(self_in_enemy),
                float(enemy_in_self),
                float(self_targeted),
                float(enemy_targeted),
                self_enemy_dist,
                enemy_self_dist,
                float(enemy_in_self) - float(self_in_enemy),
            ]
        )
        values.extend(self._level_economy_features())
        values.extend(self._legal_summary())
        assert len(values) == Args.DIM_GLOBAL
        return values

    def _hero_id_one_hot(self, hero):
        config_id = self._config_id(hero)
        return [float(config_id == 112), float(config_id == 133)]

    def _time_features(self):
        frame = float(self.frame_no)
        stages = [frame < 920, 920 <= frame < 1778, 1778 <= frame < 6254, frame >= 6254]
        countdowns = [
            _clip((920.0 - frame) / 920.0, -1.0, 1.0),
            _clip((1778.0 - frame) / 1778.0, -1.0, 1.0),
            _clip((6254.0 - frame) / 6254.0, -1.0, 1.0),
        ]
        return (
            [_clip(frame / Args.FRAME_MAX_FALLBACK, 0.0, 1.0)]
            + [float(x) for x in stages]
            + countdowns
            + [float(frame >= Args.FRAME_MAX_FALLBACK * 0.8), float(frame >= 6254)]
        )

    def _level_economy_features(self):
        self_level = _safe_float(_get(self.self_hero or {}, "level", 0), 0.0)
        enemy_level = _safe_float(_get(self.enemy_hero or {}, "level", 0), 0.0)
        self_exp = self._level_exp_ratio(self.self_hero)
        enemy_exp = self._level_exp_ratio(self.enemy_hero)
        self_money_total = self._money_total(self.self_hero)
        enemy_money_total = self._money_total(self.enemy_hero)
        self_delta = self._money_delta(self_money_total, "self")
        enemy_delta = self._money_delta(enemy_money_total, "enemy")
        self_money_frame = self._money_frame(self_money_total, "self")
        enemy_money_frame = self._money_frame(enemy_money_total, "enemy")
        return [
            _clip(self_level / Args.LEVEL_MAX, 0.0, 1.0),
            _clip(enemy_level / Args.LEVEL_MAX, 0.0, 1.0),
            _clip((self_level - enemy_level) / Args.LEVEL_MAX, -1.0, 1.0),
            self_exp,
            enemy_exp,
            _clip(self_exp - enemy_exp, -1.0, 1.0),
            _clip(self_money_total / Args.MONEY_TOTAL_SCALE, 0.0, 1.0),
            _clip(enemy_money_total / Args.MONEY_TOTAL_SCALE, 0.0, 1.0),
            _clip((self_money_total - enemy_money_total) / Args.MONEY_TOTAL_SCALE, -1.0, 1.0),
            _clip(self_money_frame / Args.MONEY_FRAME_SCALE, 0.0, 1.0),
            _clip(enemy_money_frame / Args.MONEY_FRAME_SCALE, 0.0, 1.0),
            _clip((self_delta - enemy_delta) / Args.MONEY_DELTA_SCALE, -1.0, 1.0),
        ]

    def _level_exp_ratio(self, hero):
        if hero is None:
            return 0.0
        level = _safe_int(_get(hero, "level", 1), 1)
        exp = _safe_float(_get(hero, "exp", 0), 0.0)
        max_exp = GameConfig.LEVEL_MAX_EXP.get(level)
        if max_exp is None:
            max_exp = max(GameConfig.LEVEL_MAX_EXP.values())
        return _clip(exp / max(float(max_exp), 1.0), 0.0, 1.0)

    def _money_total(self, hero):
        return _safe_float(_get_any(hero or {}, ["moneyCnt", "money_cnt", "money"], 0), 0.0)

    def _money_frame(self, current_total, side):
        return max(self._money_delta(current_total, side), 0.0)

    def _money_delta(self, current, side):
        last = self.last_money.get(side)
        return 0.0 if last is None else current - last

    def _process_hero(self, hero, is_enemy):
        if hero is None:
            return [0.0] * Args.DIM_HERO
        side = "enemy" if is_enemy else "self"
        values = []
        values.extend(self._process_unit_core(hero))
        config_id = self._config_id(hero)
        values.extend([float(config_id == 112), float(config_id == 133), float(not is_enemy), float(is_enemy)])
        values.extend(self._behavior_one_hot(_get(hero, "behav_mode", None), Args.HERO_BEHAVE))
        values.extend(self._ep_state(hero))
        values.extend(self._combat_attrs(hero))
        for slot_idx in Args.HERO_SKILL_SLOT_ORDER:
            values.extend(self._skill_slot_feature(self._slot_state(hero, slot_idx), slot_idx, config_id, side))
        values.extend(self._level_one_hot(hero))
        values.extend(self._hero_money_feature(hero, side))
        values.append(float(bool(_get(hero, "is_in_grass", False))))
        values.extend(self._hero_tower_relation(hero))
        values.extend(self._hero_buff_feature(hero, side))
        values.extend(self._recent_event_feature(hero, side))
        values.extend(self._retreat_context(hero, side))
        assert len(values) == Args.DIM_HERO, f"hero feature dim mismatch: {len(values)}"
        return values

    def _behavior_one_hot(self, value, candidates):
        idx = candidates.index(value) if value in candidates else len(candidates)
        return _one_hot(idx, len(candidates) + 1)

    def _ep_state(self, hero):
        ep = _safe_float(_get(hero, "ep", 0), 0.0)
        max_ep = _safe_float(_get(hero, "max_ep", 0), 0.0)
        ratio = _clip(ep / Args.EP_MAX_SIZE, 0.0, 1.0)
        max_ratio = _clip(max_ep / Args.EP_MAX_SIZE, 0.0, 1.0)
        bucket = 0 if ep <= 0 else _bucket(_safe_div(ep, max(max_ep, Args.EP_MAX_SIZE), 0.0), [0.2, 0.4, 0.6, 0.8]) + 1
        values = [ratio, max_ratio, float(ep > 0 or max_ep > 0), float(max_ep > 0)]
        values.extend(_one_hot(bucket, 6))
        return values

    def _combat_attrs(self, hero):
        attack_range = _safe_float(_get(hero, "attack_range", 0), 0.0)
        phy_atk = _safe_float(_get(hero, "phy_atk", 0), 0.0)
        values = [
            _clip(attack_range / 12000.0, 0.0, 1.0),
            _clip(phy_atk / 1000.0, 0.0, 1.0),
            _clip(_safe_float(_get(hero, "phy_def", 0), 0.0) / 1000.0, 0.0, 1.0),
            _clip(_safe_float(_get(hero, "mov_spd", 0), 0.0) / 10000.0, 0.0, 1.0),
            _clip(_safe_float(_get(hero, "atk_spd", 0), 0.0) / 10000.0, 0.0, 1.0),
            _clip(_safe_float(_get(hero, "hp_recover", 0), 0.0) / 500.0, 0.0, 1.0),
            float(attack_range > 0),
            float(any(_get(hero, key, None) is not None for key in ["phy_atk", "phy_def", "mov_spd", "atk_spd"])),
        ]
        values.extend(_one_hot(_bucket(attack_range, [1000, 3000, 8800]), 4))
        values.extend(_one_hot(_bucket(phy_atk, [150, 300, 600]), 4))
        return values

    def _slot_state(self, hero, slot_idx):
        skill_state = _get_any(hero or {}, ["skill_state", "skillState"], {}) or {}
        slots = _get_any(skill_state, ["slot_states", "slotStates"], []) or []
        for slot in slots:
            if self._slot_key(_get_any(slot, ["slot_type", "slotType"], None)) == slot_idx:
                return slot
        return None

    def _slot_key(self, slot_type):
        if isinstance(slot_type, int):
            return slot_type
        text = str(slot_type)
        if text.startswith("SLOT_SKILL_"):
            return _safe_int(text.split("_")[-1], -1)
        return _safe_int(slot_type, -1)

    def _skill_slot_feature(self, slot, slot_idx, hero_id, side):
        values = []
        values.extend(_one_hot(slot_idx, 6))
        config_id = self._config_id(slot)
        values.extend(
            [
                float(str(config_id).startswith("112")),
                float(str(config_id).startswith("133")),
                float(slot_idx == 4 or config_id == 80102),
                float(config_id in GameConfig.SUMMONER_SKILL_IDS),
                float(config_id == 80115),
                float(config_id == 80109),
                float(config_id != 0 and config_id not in GameConfig.SUMMONER_SKILL_IDS and not str(config_id).startswith(("112", "133"))),
            ]
        )
        level = _safe_float(_get(slot or {}, "level", 0), 0.0)
        usable = bool(_get(slot or {}, "usable", False))
        cooldown = _safe_float(_get(slot or {}, "cooldown", 0), 0.0)
        cooldown_max = _safe_float(_get_any(slot or {}, ["cooldown_max", "cooldownMax"], 0), 0.0)
        cd_bucket = 0 if cooldown <= 0 else _bucket(cooldown, [1000, 3000, 6000, 15000]) + 1
        values.extend(
            [
                _clip(level / Args.LEVEL_MAX, 0.0, 1.0),
                float(usable),
                _clip(_safe_div(cooldown, max(cooldown_max, 1.0), 0.0), 0.0, 1.0),
                _clip(cooldown_max / 120000.0, 0.0, 1.0),
                float(slot is not None),
                float(level > 0 or usable),
            ]
        )
        values.extend(_one_hot(cd_bucket, 6))
        succ = _safe_float(_get_any(slot or {}, ["succUsedInFrame", "succ_used_in_frame"], 0), 0.0)
        recent_hit = self._recent_flag(side, "hit_any")
        recent_interrupted = self._recent_recover_interrupted(side)
        values.extend(
            [
                float(succ > 0),
                _clip(_safe_float(_get_any(slot or {}, ["usedTimes", "used_times"], 0), 0.0) / 20.0, 0.0, 1.0),
                _clip(_safe_float(_get_any(slot or {}, ["hitHeroTimes", "hit_hero_times"], 0), 0.0) / 20.0, 0.0, 1.0),
                recent_hit,
                recent_interrupted,
            ]
        )
        values.extend(
            [
                float(_safe_int(_get_any(slot or {}, ["nextConfigID", "nextConfigId", "next_config_id"], 0), 0) > 0),
                float(_safe_float(_get_any(slot or {}, ["comboEffectTime", "combo_effect_time"], 0), 0.0) > 0),
                _clip(_safe_float(_get_any(slot or {}, ["comboEffectTime", "combo_effect_time"], 0), 0.0) / 10000.0, 0.0, 1.0),
            ]
        )
        values.extend(self._skill_type_tags(hero_id, slot_idx, config_id))
        assert len(values) == Args.DIM_SKILL_SLOT
        if succ > 0:
            self.recent_skill_success[side][slot_idx] = self.frame_no
        if succ > 0 and (slot_idx == 4 or config_id == 80102):
            self._record_recover_or_cake_attempt(side, "recover")
        return values

    def _skill_type_tags(self, hero_id, slot_idx, config_id):
        tags = [0.0] * 7
        damage, mobility, recover, defense, tower, long_range, passive = range(7)
        if slot_idx in (0, 1, 2, 3):
            tags[damage] = 1.0
        if hero_id == 112 and slot_idx in (1, 2, 3):
            tags[passive] = 1.0
        if hero_id == 112 and slot_idx == 2:
            tags[long_range] = 1.0
        if hero_id == 133 and slot_idx == 2:
            tags[defense] = 1.0
        if hero_id == 133 and slot_idx == 3:
            tags[damage] = 1.0
        if slot_idx == 4 or config_id == 80102:
            tags[recover] = 1.0
        if config_id in (80109, 80115):
            tags[mobility] = 1.0
        if config_id == 80107:
            tags[defense] = 1.0
        if config_id == 80105:
            tags[tower] = 1.0
        if config_id == 80110:
            tags[damage] = 1.0
        return tags

    def _level_one_hot(self, hero):
        level = _safe_int(_get(hero, "level", 1), 1)
        return _one_hot(_clip(level, 1, Args.LEVEL_MAX) - 1, Args.LEVEL_MAX)

    def _hero_money_feature(self, hero, side):
        money_total = self._money_total(hero)
        delta = self._money_delta(money_total, side)
        money_frame = self._money_frame(money_total, side)
        values = [
            _clip(money_total / Args.MONEY_TOTAL_SCALE, 0.0, 1.0),
            _clip(money_frame / Args.MONEY_FRAME_SCALE, 0.0, 1.0),
            _clip(delta / Args.MONEY_DELTA_SCALE, -1.0, 1.0),
        ]
        values.extend(_one_hot(_bucket(money_total, [1000, 2000, 4000, 7000, 10000]), 6))
        values.extend(_one_hot(_bucket(max(delta, 0.0), [10, 30, 60, 120]), 5))
        values.extend(_one_hot(_bucket(max(money_frame, 0.0), [100, 300, 600]), 4))
        assert len(values) == 18
        return values

    def _hero_tower_relation(self, hero):
        hero_pos = self._position(hero)
        enemy_tower_pos = self._tower_pos(self.enemy_tower, True)
        self_tower_pos = self._tower_pos(self.self_tower, False)
        return [
            self._dist_ratio(hero_pos, enemy_tower_pos),
            self._dist_ratio(hero_pos, self_tower_pos),
            float(self._in_tower_range(hero, self.enemy_tower, is_enemy_tower=True)),
            float(self._in_tower_range(hero, self.self_tower, is_enemy_tower=False)),
            float(self._targeted_by_tower(hero, self.enemy_tower)),
            float(self._targeted_by_tower(hero, self.self_tower)),
        ]

    def _hero_buff_feature(self, hero, side):
        buff_state = _get_any(hero or {}, ["buff_state", "buffState"], {}) or {}
        buff_skills = _get_any(buff_state, ["buff_skills", "buffSkills"], []) or []
        buff_ids = [self._config_id(buff) for buff in buff_skills if self._config_id(buff) > 0]
        current_set = set(buff_ids)
        last_set = self.last_buff_set.get(side, set())
        values = [0.0] * 96
        for buff_id in buff_ids:
            if buff_id in Args.BUFF_WHITELIST_96:
                values[Args.BUFF_WHITELIST_96.index(buff_id)] = 1.0
        family_112 = [0.0] * 8
        family_133 = [0.0] * 8
        family_common = [0.0] * 8
        unknown_hash = [0.0] * 8
        for buff_id in buff_ids:
            text = str(buff_id)
            if text.startswith("112"):
                idx = self._prefix_family(text, ["1120", "1121", "1122", "1123", "1128", "1129"], 6, 7)
                family_112[idx] = 1.0
            elif text.startswith("133"):
                idx = self._prefix_family(text, ["1330", "1331", "1332", "1333", "1339"], 5, 6)
                family_133[idx] = 1.0
            elif text.startswith(("100", "110", "900", "911", "914", "500")):
                idx = self._prefix_family(text, ["100", "110", "900", "911", "914", "500"], 6, 7)
                family_common[idx] = 1.0
            else:
                unknown_hash[_stable_hash(buff_id, 8)] = 1.0
        hero_specific_count = sum(1 for buff_id in buff_ids if str(buff_id).startswith(("112", "133")))
        values.extend(family_112)
        values.extend(family_133)
        values.extend(family_common)
        values.extend(unknown_hash)
        values.extend(
            [
                _clip(len(buff_ids) / 20.0, 0.0, 1.0),
                _clip(hero_specific_count / 10.0, 0.0, 1.0),
                _clip(len(current_set - last_set) / 8.0, 0.0, 1.0),
                _clip(len(last_set - current_set) / 8.0, 0.0, 1.0),
                float(any(str(buff_id).startswith("112") for buff_id in buff_ids)),
                float(any(str(buff_id).startswith("133") for buff_id in buff_ids)),
                float(any(str(buff_id).startswith(("100", "110", "900", "911", "914", "500")) for buff_id in buff_ids)),
                float(len(buff_ids) == 0),
            ]
        )
        assert len(values) == Args.DIM_HERO_BUFF
        return values

    def _prefix_family(self, text, prefixes, other_idx, unknown_idx):
        for idx, prefix in enumerate(prefixes):
            if text.startswith(prefix):
                return idx
        return other_idx if text else unknown_idx

    def _recent_event_feature(self, hero, side):
        runtime = self._runtime_id(hero)
        recent = [
            event
            for event in list(self.recent_hit_events) + list(self.recent_take_hurt_events)
            if event["side"] == side and self.frame_no - event["frame"] <= 15
        ]
        hit_hero = any(event.get("kind") == "hit_hero" for event in recent)
        hit_soldier = any(event.get("kind") == "hit_soldier" for event in recent)
        hit_tower = any(event.get("kind") == "hit_tower" for event in recent)
        hurt_hero = any(event.get("kind") == "hurt_hero" for event in recent)
        hurt_tower = any(event.get("kind") == "hurt_tower" for event in recent)
        hurt_soldier = any(event.get("kind") == "hurt_soldier" for event in recent)
        last_slot = -1
        last_hurt_ratio = 0.0
        for event in reversed(recent):
            if "slot" in event:
                last_slot = event["slot"]
                break
        for event in reversed(recent):
            if "hurt" in event:
                last_hurt_ratio = _clip(event["hurt"] / 3000.0, 0.0, 1.0)
                break
        values = [
            float(hit_hero),
            float(hit_soldier),
            float(hit_tower),
            float(hurt_hero),
            float(hurt_tower),
            float(hurt_soldier),
            float(runtime is not None),
            self._recent_skill_flag(side),
        ]
        values.extend(_one_hot(last_slot if 0 <= last_slot <= 5 else 6, 7))
        values.extend([last_hurt_ratio, float(self._recent_recover_interrupted(side))])
        values.extend(_one_hot(_bucket(len(recent), [1, 3, 6, 10]), 5))
        values.extend([self._recent_flag(side, "hit_any"), self._recent_flag(side, "hurt_any")])
        assert len(values) == 24
        return values

    def _retreat_context(self, hero, side):
        hp = self._hp_ratio(hero)
        hero_pos = self._position(hero)
        enemy_pos = self.enemy_pos if side == "self" else self.self_pos
        base = Args.SELF_BASE_ANCHOR if side == "self" else Args.ENEMY_BASE_ANCHOR
        move_legal = 0.0
        if self.legal_action is not None and self.legal_action.shape[0] >= 10:
            move_legal = float(self.legal_action[2] > 0)
        enemy_dist = self._dist(hero_pos, enemy_pos)
        recent_damage = self._recent_flag(side, "hurt_any")
        lane_sign = 0.0
        if hero_pos is not None:
            lane_sign = _clip((base[0] - hero_pos[0]) / Args.GLOBAL_LANE_HALF, -1.0, 1.0)
        values = [
            float(hp <= 0.30),
            float(hp <= 0.15),
            float(enemy_dist <= Args.TOWER_ATTACK_RANGE_FALLBACK),
            recent_damage,
            move_legal,
            self._dist_ratio(hero_pos, base),
            lane_sign,
            float(hp <= 0.30 and enemy_dist > Args.TOWER_ATTACK_RANGE_FALLBACK and recent_damage == 0.0),
        ]
        return values

    def _process_soldier(self, soldier, is_enemy, target_index):
        if soldier is None:
            return [0.0] * Args.DIM_SOLDIER
        values = self._process_unit_core(soldier)
        config_id = self._config_id(soldier)
        if config_id in Args.SOLDIER_CONFIG_GROUPS["melee"]:
            type_idx = 0
        elif config_id in Args.SOLDIER_CONFIG_GROUPS["ranged"]:
            type_idx = 1
        elif config_id in Args.SOLDIER_CONFIG_GROUPS["cannon"]:
            type_idx = 2
        else:
            type_idx = 3
        values.extend(_one_hot(type_idx, 4))
        values.extend(_one_hot(_bucket(_safe_int(_get(soldier, "behav_mode", 0), 0), [0, 2, 23]), 4))
        pos = self._position(soldier)
        tower = self.enemy_tower if not is_enemy else self.self_tower
        values.extend(
            [
                self._dist_ratio(pos, self._tower_pos(tower, is_enemy=False)),
            float(self._in_tower_range(soldier, tower, is_enemy_tower=not is_enemy)),
                float(self._targeted_by_tower(soldier, tower)),
                float(_get(soldier, "attack_target", None) is not None),
            ]
        )
        values.extend(
            [
                float(target_index in (3, 4, 5, 6)),
                float(target_index == 3),
                float(target_index == 4),
                float(target_index == 5),
            ]
        )
        values.extend([_clip(_safe_float(_get(soldier, "kill_income", 0), 0.0) / 100.0, 0.0, 1.0), float(self._hp_ratio(soldier) <= 0.25)])
        assert len(values) == Args.DIM_SOLDIER
        return values

    def _process_monster(self, monster):
        if monster is None:
            return [0.0] * Args.DIM_MONSTER
        values = self._process_unit_core(monster)
        config_id = self._config_id(monster)
        values.extend(
            [
                float(config_id == 6827),
                float(config_id != 0 and config_id != 6827),
                float(config_id == 0),
                float(config_id != 0 and _stable_hash(config_id, 2) == 1),
            ]
        )
        values.extend(_one_hot(MONSTER_BEHAVIORS.index(_get(monster, "behav_mode", None)) if _get(monster, "behav_mode", None) in MONSTER_BEHAVIORS else 3, 4))
        target_legal = self.raw_target_legal[8] if self.raw_target_legal is not None and len(self.raw_target_legal) > 8 else 1.0
        values.extend([1.0, float(target_legal > 0)])
        hp = self._hp_ratio(monster)
        kill_income = _clip(_safe_float(_get(monster, "kill_income", 0), 0.0) / 300.0, 0.0, 1.0)
        values.extend(
            [
                kill_income,
                0.0,
                0.0,
                float(0.0 < hp <= 0.25),
                float(kill_income > 0.0),
                float(self._dist(self.self_pos, self._position(monster)) <= 12000 and self._dist(self.enemy_pos, self._position(monster)) <= 12000),
            ]
        )
        assert len(values) == Args.DIM_MONSTER
        return values

    def _process_tower(self, tower, is_enemy, cake):
        if tower is None:
            core = [0.0] * Args.DIM_UNIT_CORE
        else:
            core = self._process_unit_core(tower)
        anchor = Args.ENEMY_TOWER_ANCHOR if is_enemy else Args.SELF_TOWER_ANCHOR
        values = list(core)
        values.extend(self._tower_anchor_feature(anchor, is_enemy))
        sub_type = self._sub_type(tower)
        if sub_type in TOWER_SUB_TYPES:
            organ_idx = 0
        elif sub_type in CRYSTAL_SUB_TYPES:
            organ_idx = 1
        elif sub_type in SPRING_SUB_TYPES:
            organ_idx = 2
        else:
            organ_idx = 3
        values.extend(_one_hot(organ_idx, 4))
        attack_range = self._tower_attack_range(tower)
        sight_area = _safe_float(_get(tower or {}, "sight_area", 0), 0.0)
        values.extend(
            [
                _clip(attack_range / 12000.0, 0.0, 1.0),
                _clip(sight_area / 12000.0, 0.0, 1.0),
                float(attack_range > 0),
                float(sight_area > 0),
            ]
        )
        values.extend(self._tower_aggro_feature(tower, is_enemy, attack_range))
        values.extend(self._cake_feature(cake, is_enemy))
        values.extend(self._tower_pressure_feature(tower, is_enemy, attack_range))
        assert len(values) == Args.DIM_TOWER
        return values

    def _tower_anchor_feature(self, anchor, is_enemy):
        self_dist = self._dist_ratio(self.self_pos, anchor)
        enemy_dist = self._dist_ratio(self.enemy_pos, anchor)
        self_lane = self.self_pos[0] if self.self_pos else 0.0
        enemy_lane = self.enemy_pos[0] if self.enemy_pos else 0.0
        low, high = sorted([Args.SELF_TOWER_ANCHOR[0], Args.ENEMY_TOWER_ANCHOR[0]])
        lane_forward = _clip((self_lane - Args.SELF_TOWER_ANCHOR[0]) / max(1.0, Args.ENEMY_TOWER_ANCHOR[0] - Args.SELF_TOWER_ANCHOR[0]), 0.0, 1.0)
        return [
            _clip(anchor[0] / Args.GLOBAL_LANE_HALF, -1.0, 1.0),
            _clip(anchor[1] / Args.GLOBAL_WIDTH_HALF, -1.0, 1.0),
            self_dist,
            enemy_dist,
            float(is_enemy),
            float(not is_enemy),
            float(low <= self_lane <= high),
            float(low <= enemy_lane <= high),
            lane_forward,
            _clip(abs((self.self_pos[1] if self.self_pos else 0.0) - anchor[1]) / Args.CENTER_WIDTH_HALF, 0.0, 1.0),
            _clip(abs((self.enemy_pos[1] if self.enemy_pos else 0.0) - anchor[1]) / Args.CENTER_WIDTH_HALF, 0.0, 1.0),
            1.0,
        ]

    def _tower_aggro_feature(self, tower, is_enemy, attack_range):
        target = _get(tower or {}, "attack_target", None)
        info = self.runtime_index.get(target, {})
        target_kind = info.get("kind", "unknown")
        entity = info.get("entity")
        target_vec = [0.0] * 4
        if not target or not info:
            target_vec[0] = 1.0
        elif target_kind == "hero" and self._same_camp(_get(entity, "camp", None), self.main_camp):
            target_vec[1] = 1.0
        elif target_kind == "hero":
            target_vec[2] = 1.0
        else:
            target_vec[3] = 1.0
        tower_pos = self._tower_pos(tower, is_enemy)
        return target_vec + [
            float(self._dist(self.self_pos, tower_pos) <= attack_range),
            float(self._dist(self.enemy_pos, tower_pos) <= attack_range),
            float(self._targeted_by_tower(self.self_hero, tower)),
            float(self._targeted_by_tower(self.enemy_hero, tower)),
            float(is_enemy and self._recent_flag("self", "hit_hero") and self._dist(self.self_pos, tower_pos) <= attack_range),
            float((not is_enemy) and self._recent_flag("enemy", "hit_hero") and self._dist(self.enemy_pos, tower_pos) <= attack_range),
        ]

    def _cake_feature(self, cake, is_enemy):
        anchor = Args.ENEMY_CAKE_ANCHOR if is_enemy else Args.SELF_CAKE_ANCHOR
        side = "enemy" if is_enemy else "self"
        exists = cake is not None
        if exists:
            self.next_cake_frame[side] = self.frame_no + int(Args.CAKE_RESPAWN_SECONDS * 30)
        respawn = _clip((self.next_cake_frame[side] - self.frame_no) / max(1.0, Args.CAKE_RESPAWN_SECONDS * 30), 0.0, 1.0) if not exists else 0.0
        self_dist = self._dist_ratio(self.self_pos, anchor)
        enemy_dist = self._dist_ratio(self.enemy_pos, anchor)
        return [
            float(exists),
            self_dist,
            enemy_dist,
            float(self._hp_ratio(self.self_hero) <= 0.50),
            float(enemy_dist > self_dist and not self._in_tower_range(self.self_hero, self.enemy_tower, is_enemy_tower=True)),
            respawn,
        ]

    def _tower_pressure_feature(self, tower, is_enemy, attack_range):
        hp = self._hp_ratio(tower)
        tower_pos = self._tower_pos(tower, is_enemy)
        self_in = self._dist(self.self_pos, tower_pos) <= attack_range
        enemy_in = self._dist(self.enemy_pos, tower_pos) <= attack_range
        hero_attack_range = _safe_float(_get(self.self_hero or {}, "attack_range", 0), Args.TOWER_ATTACK_RANGE_FALLBACK)
        self_can_attack = self._dist(self.self_pos, tower_pos) <= hero_attack_range
        self_targeted = self._targeted_by_tower(self.self_hero, tower)
        soldier_targeted = False
        target = _get(tower or {}, "attack_target", None)
        info = self.runtime_index.get(target, {})
        if info.get("kind") == "soldier":
            soldier_targeted = True
        push_value = is_enemy and hp <= 0.5 and self_can_attack and not self_targeted and self._dist(self.enemy_pos, self.self_pos) > Args.TOWER_ATTACK_RANGE_FALLBACK
        return [hp, float(0.0 < hp <= 0.25), float(self_can_attack), float(self_targeted), float(soldier_targeted), float(push_value)]

    def _process_bullet(self, bullet):
        if bullet is None:
            return [0.0] * Args.DIM_BULLET
        values = [1.0]
        camp = _get(bullet, "camp", None)
        if self._is_enemy_camp(camp):
            values.extend([1.0, 0.0, 0.0])
        elif self._same_camp(camp, self.main_camp):
            values.extend([0.0, 1.0, 0.0])
        else:
            values.extend([0.0, 0.0, 1.0])
        source_id = _get(bullet, "source_actor", None)
        source = self.runtime_index.get(source_id, {})
        source_kind = source.get("kind", "unknown")
        kind_idx = {"hero": 0, "tower": 1, "soldier": 2, "monster": 3}.get(source_kind, 4)
        values.extend(_one_hot(kind_idx, 5))
        source_entity = source.get("entity")
        source_config = self._config_id(source_entity)
        values.extend([float(source_config == 112), float(source_config == 133)])
        slot_type = self._slot_key(_get_any(bullet, ["slot_type", "slotType"], -1))
        values.extend(_one_hot(slot_type if 0 <= slot_type <= 3 else 4, 5))
        skill_id = _safe_int(_get(bullet, "skill_id", 0), 0)
        values.append(float(skill_id != 0))

        bullet_pos = self._position(bullet)
        delta_pos = None
        if bullet_pos is not None and self.self_pos is not None:
            delta_pos = (bullet_pos[0] - self.self_pos[0], bullet_pos[1] - self.self_pos[1])
        if delta_pos is None:
            values.extend([0.0] * (Args.DIM_CENTER_LANE + Args.DIM_CENTER_WIDTH))
        else:
            values.extend(self._axis_one_hot(delta_pos[0], -Args.CENTER_LANE_HALF, Args.CENTER_LANE_HALF, Args.CENTER_LANE_UNIT, Args.DIM_CENTER_LANE))
            values.extend(self._axis_one_hot(delta_pos[1], -Args.CENTER_WIDTH_HALF, Args.CENTER_WIDTH_HALF, Args.CENTER_WIDTH_UNIT, Args.DIM_CENTER_WIDTH))
        values.extend(
            [
                self._dist_ratio(bullet_pos, self.self_pos),
                self._dist_ratio(bullet_pos, self.enemy_pos),
                self._dist_ratio(bullet_pos, self._tower_pos(self.self_tower, False)),
                self._dist_ratio(bullet_pos, self._tower_pos(self.enemy_tower, True)),
            ]
        )
        values.extend(self._direction_sign(delta_pos))
        trajectory = self._bullet_trajectory_feature(bullet, bullet_pos)
        values.extend(trajectory)
        values.extend(self._bullet_source_context(source_entity, source_kind, bullet_pos, slot_type))
        values.extend(self._bullet_hit_risk(delta_pos, trajectory))
        assert len(values) == Args.DIM_BULLET, f"bullet feature dim mismatch: {len(values)}"
        return values

    def _direction_sign(self, delta):
        if delta is None:
            return [0.0] * 6
        dlane, dwidth = delta
        return [
            float(dlane > 0),
            float(dlane < 0),
            float(abs(dlane) <= 500),
            float(dwidth > 0),
            float(dwidth < 0),
            float(abs(dwidth) <= 250),
        ]

    def _bullet_trajectory_feature(self, bullet, pos):
        runtime_id = self._runtime_id(bullet)
        cache = self.bullet_cache.get(runtime_id)
        values = [0.0] * 18
        if runtime_id is None or cache is None or pos is None:
            return values
        dt = max(1, self.frame_no - cache["frame_no"])
        if dt > 2:
            return values
        v_lane = (pos[0] - cache["lane"]) / dt
        v_width = (pos[1] - cache["width"]) / dt
        speed = math.sqrt(v_lane ** 2 + v_width ** 2)
        rel_lane = pos[0] - (self.self_pos[0] if self.self_pos else 0.0)
        rel_width = pos[1] - (self.self_pos[1] if self.self_pos else 0.0)
        approaching = rel_lane * v_lane + rel_width * v_width < 0
        closest = self._closest_approach_distance((rel_lane, rel_width), (v_lane, v_width))
        values[0] = 1.0
        values[1] = _clip(speed / 5000.0, 0.0, 1.0)
        values[2:8] = self._direction_sign((v_lane, v_width))
        values[8] = float(approaching)
        values[9] = _clip(closest / Args.CENTER_LANE_HALF, 0.0, 1.0)
        time_bucket = _bucket(_safe_div(math.sqrt(rel_lane ** 2 + rel_width ** 2), speed, 999.0), [1, 2, 4, 8, 16])
        values[10:16] = _one_hot(time_bucket, 6)
        values[16] = _clip(v_lane / 5000.0, -1.0, 1.0)
        values[17] = _clip(v_width / 5000.0, -1.0, 1.0)
        return values

    def _closest_approach_distance(self, rel, vel):
        vx, vz = vel
        denom = vx * vx + vz * vz
        if denom <= 1e-6:
            return math.sqrt(rel[0] ** 2 + rel[1] ** 2)
        t = _clip(-(rel[0] * vx + rel[1] * vz) / denom, 0.0, 10.0)
        return math.sqrt((rel[0] + vx * t) ** 2 + (rel[1] + vz * t) ** 2)

    def _bullet_source_context(self, source_entity, source_kind, bullet_pos, slot_type):
        source_pos = self._position(source_entity)
        hero_id = self._config_id(source_entity)
        side = "enemy" if source_entity is not None and self._is_enemy_camp(_get(source_entity, "camp", None)) else "self"
        recent_skill = any(self.frame_no - frame <= 3 for frame in self.recent_skill_success.get(side, {}).values())
        recent_attack = self._recent_flag(side, "hit_any")
        continuous_bullets = sum(1 for item in self.bullet_cache.values() if item.get("source_actor") == self._runtime_id(source_entity))
        return [
            self._dist_ratio(source_pos, self.self_pos),
            float(hero_id == 112),
            float(hero_id == 133),
            float(recent_skill),
            float(recent_attack),
            _clip(continuous_bullets / 6.0, 0.0, 1.0),
            float(source_entity is None),
            float(source_kind == "hero"),
            float(source_kind == "tower"),
            float(slot_type in (1, 2, 3)),
            float(self._dist(bullet_pos, self.self_pos) <= Args.TOWER_ATTACK_RANGE_FALLBACK),
            float(side == "enemy"),
        ]

    def _bullet_hit_risk(self, delta, trajectory):
        if delta is None:
            return [0.0] * 8
        dist = math.sqrt(delta[0] ** 2 + delta[1] ** 2)
        values = _one_hot(_bucket(dist, [1000, 3000, 6000, 10000]), 5)
        values.extend(
            [
                float(abs(delta[1]) <= 1200 and 0 <= delta[0] <= 6000),
                float(delta[0] < -500),
                float(trajectory[8] > 0 and dist <= 6000),
            ]
        )
        return values

    def _target_entities(self):
        target_soldiers = list(self.target_enemy_soldiers[:4])
        target_soldiers += [None] * (4 - len(target_soldiers))
        return [None, self.enemy_hero, self.self_hero] + target_soldiers + [self.enemy_tower, self.monster]

    def _process_target(self, entity, target_index):
        values = []
        legal = 1.0
        if self.raw_target_legal is not None and target_index < len(self.raw_target_legal):
            legal = float(self.raw_target_legal[target_index] > 0)
        exist = float(target_index == 0 or entity is not None)
        values.extend([exist, legal])
        # type_idx 5-way one-hot. Implementation maps:
        #   0 = none/unknown
        #   1 = enemy_hero  (target_index == 1)
        #   2 = self_hero   (target_index == 2)
        #   3 = soldier     (target_index in 3..6)
        #   4 = tower/monster (target_index in {7, 8})
        # Note: this swaps positions 1 and 2 vs the textual order in
        # docs/feature_engineering_design.md section 3.4/10
        # ([none, self_hero, enemy_hero, soldier, organ_or_monster]).
        # Functionally equivalent (model learns the mapping from data); kept
        # as-is to preserve checkpoint compatibility.
        type_idx = 0
        if target_index in (1, 2):
            type_idx = 1 if target_index == 1 else 2
        elif 3 <= target_index <= 6:
            type_idx = 3
        elif target_index == 7:
            type_idx = 4
        elif target_index == 8:
            type_idx = 4
        values.extend(_one_hot(type_idx, 5))
        hp = self._hp_ratio(entity)
        values.extend([hp, float(hp > 0.0), float(0.0 < hp <= 0.25)])
        target_pos = self._position(entity)
        delta = None
        if self.self_pos is not None and target_pos is not None:
            delta = (target_pos[0] - self.self_pos[0], target_pos[1] - self.self_pos[1])
        if delta is None:
            values.extend([0.0] * 6)
            distance = 1e9
        else:
            distance = math.sqrt(delta[0] ** 2 + delta[1] ** 2)
            values.extend(
                [
                    _clip(distance / Args.GLOBAL_LANE_HALF, 0.0, 1.0),
                    _clip(delta[0] / Args.GLOBAL_LANE_HALF, -1.0, 1.0),
                    _clip(delta[1] / Args.CENTER_WIDTH_HALF, -1.0, 1.0),
                    float(delta[0] > 0),
                    float(delta[1] > 0),
                    float(abs(delta[1]) <= 3000),
                ]
            )
        attack_range = _safe_float(_get(self.self_hero or {}, "attack_range", 0), Args.TOWER_ATTACK_RANGE_FALLBACK)
        in_ranges = [
            distance <= attack_range,
            distance <= self._skill_range(1),
            distance <= self._skill_range(2),
            distance <= self._skill_range(3),
            distance <= self._skill_range(5),
        ]
        in_ranges.append(any(in_ranges))
        values.extend([float(x) for x in in_ranges])
        in_enemy_tower = self._dist(target_pos, self._tower_pos(self.enemy_tower, True)) <= self._tower_attack_range(self.enemy_tower)
        values.extend(
            [
                float(in_enemy_tower),
                self._dist_ratio(target_pos, self._tower_pos(self.enemy_tower, True)),
                float(target_index == 7 and hp <= 0.25),
                float(self._targeted_by_tower(self.self_hero, self.enemy_tower)),
            ]
        )
        runtime = self._runtime_id(entity) or 0
        config = self._config_id(entity) if entity is not None else target_index
        values.extend(
            [
                _clip((runtime % 1000) / 1000.0, 0.0, 1.0),
                _clip((config % 1000) / 1000.0, 0.0, 1.0),
                float(entity is not None and self._same_camp(_get(entity, "camp", None), self.main_camp)),
                float(entity is not None and self._is_enemy_camp(_get(entity, "camp", None))),
            ]
        )
        assert len(values) == Args.DIM_TARGET
        return values

    def _skill_range(self, slot_idx):
        hero_id = self._config_id(self.self_hero)
        if hero_id == 112 and slot_idx == 2:
            return Args.GLOBAL_LANE_HALF
        if slot_idx == 5:
            return 8800.0
        if slot_idx == 1:
            return 8800.0
        if slot_idx == 2:
            return 12000.0
        if slot_idx == 3:
            return 12000.0
        return 3000.0

    def _in_tower_range(self, unit, tower, is_enemy_tower=None):
        if is_enemy_tower is None:
            is_enemy_tower = self._is_enemy_camp(_get(tower or {}, "camp", None))
        return self._dist(self._position(unit), self._tower_pos(tower, is_enemy_tower)) <= self._tower_attack_range(tower)

    def _targeted_by_tower(self, unit, tower):
        if unit is None or tower is None:
            return False
        return self._runtime_id(unit) == _get(tower, "attack_target", None)

    def _tower_attack_range(self, tower):
        return _safe_float(_get(tower or {}, "attack_range", 0), 0.0) or Args.TOWER_ATTACK_RANGE_FALLBACK

    def _tower_pos(self, tower, is_enemy):
        return self._position(tower) or (Args.ENEMY_TOWER_ANCHOR if is_enemy else Args.SELF_TOWER_ANCHOR)

    def _hp_ratio(self, unit):
        if unit is None:
            return 0.0
        return _clip(
            _safe_div(_get_any(unit, ["hp", "HP"], 0), _get_any(unit, ["max_hp", "maxHp", "maxHP"], 0), 0.0),
            0.0,
            1.0,
        )

    def _update_recent_events(self):
        for side, hero in (("self", self.self_hero), ("enemy", self.enemy_hero)):
            if hero is None:
                continue
            for hit in _get(hero, "hit_target_info", []) or []:
                target = _get(hit, "hit_target", None)
                info = self.runtime_index.get(target, {})
                kind = info.get("kind", "unknown")
                event_kind = {"hero": "hit_hero", "soldier": "hit_soldier", "tower": "hit_tower"}.get(kind, "hit_any")
                self.recent_hit_events.append({"frame": self.frame_no, "side": side, "kind": event_kind, "slot": self._slot_key(_get_any(hit, ["slot_type", "slotType"], -1))})
                self.recent_hit_events.append({"frame": self.frame_no, "side": side, "kind": "hit_any"})
            for hurt in _get(hero, "take_hurt_infos", []) or []:
                attacker = _get(hurt, "atker", None)
                info = self.runtime_index.get(attacker, {})
                kind = info.get("kind", "unknown")
                event_kind = {"hero": "hurt_hero", "soldier": "hurt_soldier", "tower": "hurt_tower"}.get(kind, "hurt_any")
                self.recent_take_hurt_events.append(
                    {
                        "frame": self.frame_no,
                        "side": side,
                        "kind": event_kind,
                        "slot": self._slot_key(_get_any(hurt, ["skillSlot", "skill_slot"], -1)),
                        "hurt": _safe_float(_get(hurt, "hurtValue", 0), 0.0),
                    }
                )
                self.recent_take_hurt_events.append({"frame": self.frame_no, "side": side, "kind": "hurt_any"})

    def _recent_flag(self, side, kind, window=15):
        events = list(self.recent_hit_events) + list(self.recent_take_hurt_events)
        return float(any(event["side"] == side and event["kind"] == kind and self.frame_no - event["frame"] <= window for event in events))

    def _recent_skill_flag(self, side, window=15):
        # recent_skill_success is a dict {side: {slot_idx: last_succ_frame}},
        # not part of the hit/take_hurt event queue, so it must be read directly.
        return float(any(
            self.frame_no - frame <= window
            for frame in self.recent_skill_success.get(side, {}).values()
        ))

    def _recent_recover_interrupted(self, side):
        attempt_frame = self.recent_recover_or_cake_attempt.get(side, -999999)
        if self.frame_no - attempt_frame > 15:
            return 0.0
        if not self._recent_hurt_after(side, attempt_frame):
            return 0.0
        hero = self.self_hero if side == "self" else self.enemy_hero
        current_hp = self._hp_ratio(hero)
        attempt_hp = self.recent_recover_attempt_hp.get(side, 0.0)
        return float(current_hp <= attempt_hp + 0.02)

    def _recent_hurt_after(self, side, attempt_frame, window=15):
        return any(
            event["side"] == side
            and event["kind"] == "hurt_any"
            and attempt_frame <= event["frame"] <= self.frame_no
            and self.frame_no - event["frame"] <= window
            for event in self.recent_take_hurt_events
        )

    def _record_recover_or_cake_attempt(self, side, attempt_type):
        hero = self.self_hero if side == "self" else self.enemy_hero
        self.recent_recover_or_cake_attempt[side] = self.frame_no
        self.recent_recover_attempt_hp[side] = self._hp_ratio(hero)
        self.recent_recover_attempt_type[side] = attempt_type

    def _update_cake_attempts(self):
        for side, cake, anchor in (
            ("self", self.self_cake, Args.SELF_CAKE_ANCHOR),
            ("enemy", self.enemy_cake, Args.ENEMY_CAKE_ANCHOR),
        ):
            existed = self.last_cake_exists.get(side, False)
            exists = cake is not None
            if not existed or exists:
                continue
            hero = self.self_hero if side == "self" else self.enemy_hero
            other = self.enemy_hero if side == "self" else self.self_hero
            hero_dist = self._dist(self._position(hero), anchor)
            other_dist = self._dist(self._position(other), anchor)
            if hero_dist <= 3500 and hero_dist < other_dist:
                self._record_recover_or_cake_attempt(side, "cake")

    def _bullet_threat_score(self, bullet):
        pos = self._position(bullet)
        near_self = self._dist(pos, self.self_pos) <= 6000
        source = self.runtime_index.get(_get(bullet, "source_actor", None), {})
        source_is_enemy_hero = source.get("kind") == "hero" and self._is_enemy_camp(_get(source.get("entity"), "camp", None))
        trajectory = self._bullet_trajectory_feature(bullet, pos)
        return 3.0 * trajectory[8] + 2.0 * float(source_is_enemy_hero or source.get("kind") is None) + 2.0 * float(near_self) + trajectory[1] + self._recent_skill_flag("enemy")

    def _update_history_after_frame(self, bullets):
        for side, hero in (("self", self.self_hero), ("enemy", self.enemy_hero)):
            self.last_money[side] = self._money_total(hero)
            self.last_hp[side] = self._hp_ratio(hero)
            buff_state = _get_any(hero or {}, ["buff_state", "buffState"], {}) or {}
            buff_skills = _get_any(buff_state, ["buff_skills", "buffSkills"], []) or []
            self.last_buff_set[side] = set(
                self._config_id(buff)
                for buff in buff_skills
                if self._config_id(buff) > 0
            )
        self.last_tower_hp["self"] = self._hp_ratio(self.self_tower)
        self.last_tower_hp["enemy"] = self._hp_ratio(self.enemy_tower)
        self.last_cake_exists["self"] = self.self_cake is not None
        self.last_cake_exists["enemy"] = self.enemy_cake is not None
        for bullet in bullets:
            runtime_id = self._runtime_id(bullet)
            pos = self._position(bullet)
            if runtime_id is None or pos is None:
                continue
            self.bullet_cache[runtime_id] = {
                "lane": pos[0],
                "width": pos[1],
                "frame_no": self.frame_no,
                "source_actor": _get(bullet, "source_actor", None),
                "camp": _get(bullet, "camp", None),
                "slot_type": _get_any(bullet, ["slot_type", "slotType"], None),
            }
        old_keys = [key for key, value in self.bullet_cache.items() if self.frame_no - value["frame_no"] > 8]
        for key in old_keys:
            self.bullet_cache.pop(key, None)
