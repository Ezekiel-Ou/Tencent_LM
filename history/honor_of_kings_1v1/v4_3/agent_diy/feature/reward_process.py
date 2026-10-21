#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Reward manager adapted from hok_semi and remapped to the current hok1v1 dict protocol.
"""

import math

from agent_diy.conf.conf import Args, GameConfig


TOWER_SUB_TYPES = {21, "21", "ACTOR_SUB_TOWER"}
SOLDIER_SUB_TYPES = {1, "1", 11, "11", "ACTOR_SUB_SOLDIER"}
SKILL_HIT_HERO_REWARD_KEYS = {
    (112, 1): "luban_skill1_hit_enemy_hero",
    (112, 2): "luban_skill2_hit_enemy_hero",
    (112, 3): "luban_skill3_hit_enemy_hero",
    (133, 1): "direnjie_skill1_hit_enemy_hero",
    (133, 3): "direnjie_skill3_hit_enemy_hero",
}
SKILL_HIT_REWARD_KEYS = set(SKILL_HIT_HERO_REWARD_KEYS.values()) | {
    "luban_skill1_hit_enemy_soldier",
    "direnjie_skill3_followup_damage",
    "direnjie_skill3_miss",
}
SCENARIO_REWARD_KEYS = {
    "minion_tower_push",
    "enemy_dead_enemy_cake",
    "enemy_minion_tower_front",
    "enemy_minion_under_own_tower",
    "river_crab_pressure",
    "early_own_half_hero_offense",
    "opening_trade_pressure",
}


class RewardStruct:
    def __init__(self, m_weight=0.0):
        self.cur_frame_value = 0.0
        self.last_frame_value = 0.0
        self.value = 0.0
        self.weight = m_weight


def init_calc_frame_map():
    return {key: RewardStruct(weight) for key, weight in GameConfig.REWARD_WEIGHT_DICT.items()}


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


def _safe_div(num, den, default=0.0):
    try:
        den = float(den)
        if den == 0:
            return default
        return float(num) / den
    except (TypeError, ValueError):
        return default


def _hp(unit):
    return _get_any(unit or {}, ["hp", "HP"], 0)


def _max_hp(unit):
    return _get_any(unit or {}, ["max_hp", "maxHp", "maxHP"], 0)


def _hp_rate(unit, default=0.0):
    return _safe_div(_hp(unit), _max_hp(unit), default)


def _hp_value_rate(hp_rate):
    try:
        hp_rate = float(hp_rate)
    except (TypeError, ValueError):
        hp_rate = 0.0
    hp_rate = min(max(hp_rate, 0.0), 1.0)
    return math.pow(hp_rate, float(GameConfig.CAKE_HP_VALUE_POWER))


def _tower_hp_reward_value(tower):
    hp_rate = min(max(_hp_rate(tower or {}), 0.0), 1.0)
    threshold = float(GameConfig.TOWER_HP_LOW_SHAPING_THRESHOLD)
    low_total = float(GameConfig.TOWER_HP_LOW_SHAPING_TOTAL)
    power = float(GameConfig.TOWER_HP_LOW_SHAPING_POWER)
    if hp_rate >= threshold:
        return hp_rate
    damage_progress = (threshold - hp_rate) / threshold
    return threshold - low_total * math.pow(damage_progress, power)


def _loc(obj):
    collider = _get(obj or {}, "collider", {}) or {}
    location = _get(collider, "location", None) or _get(obj or {}, "location", {}) or {}
    return float(_get(location, "x", 0)), float(_get(location, "z", 0))


def _camp_key(camp):
    if camp in (1, "1", "PLAYERCAMP_1", "blue_camp"):
        return 1
    if camp in (2, "2", "PLAYERCAMP_2", "red_camp"):
        return 2
    return camp


class GameRewardManager:
    def __init__(self, main_hero_runtime_id):
        self.main_hero_player_id = main_hero_runtime_id
        self.main_hero_camp = -1
        self.m_reward_value = {}
        self.m_cur_calc_frame_map = init_calc_frame_map()
        self.m_main_calc_frame_map = init_calc_frame_map()
        self.m_enemy_calc_frame_map = init_calc_frame_map()
        self.time_scale_arg = GameConfig.TIME_SCALE_ARG
        self.m_each_level_max_exp = {}
        self.has_last_frame = False
        self.m_last_hit_debug = self._empty_last_hit_debug()
        self._main_ult_hit_frame = -99999
        self._enemy_ult_hit_frame = -99999
        self._cleanse_main_count = 0
        self._cleanse_enemy_count = 0
        self._skill2_misuse_count = 0
        # Skill-hit detection state. slot3_armed/credit/frame tracked per side so
        # Luban ult bullet-farm only credits once per cast cycle.
        self._skill_hit_main_count = 0
        self._skill_hit_enemy_count = 0
        self._skill_hit_reward_counts = self._empty_skill_hit_reward_counts()
        self.slot3_armed = {"main": False, "enemy": False}
        self.slot3_credit_used = {"main": False, "enemy": False}
        self.slot3_armed_frame = {"main": -10000, "enemy": -10000}
        self.slot3_fallback_credit_frame = {"main": -10000, "enemy": -10000}
        self.last_luban_skill1_action_context = None
        self.pending_luban_skill1_frame = None
        self.pending_luban_skill1_aim_pos = None
        self.pending_luban_skill1_target_runtime = None
        self.pending_luban_skill1_hit_soldiers = set()
        self.prev_skill_hit_hero_times = {"main": {}}
        self.prev_main_total_hurt_to_hero = None
        self.main_total_hurt_to_hero_delta = 0.0
        self.prev_enemy_soldier_hp = {}
        self.pending_direnjie_skill3_followup_frame = None
        self.pending_direnjie_skill3_followup_count = 0
        self._direnjie_skill3_followup_count_value = 0.0
        self.pending_direnjie_skill3_miss_frame = None
        self._direnjie_skill3_miss_count_value = 0.0
        # Cake pickup detection: track cake positions and main hero HP from previous frame.
        self.prev_cake_locs = set()
        self.prev_main_hp = None
        self._pending_cake_interrupt = None
        self._cake_pickup_count = 0
        self._cake_debug = self._empty_cake_debug()
        # Recover-skill-at-low-HP detection: record low-HP attempts and settle
        # after a short window only if HP actually rises.
        self._recover_low_hp_count = 0
        self._pending_recover = None
        self._recover_debug = self._empty_recover_debug()
        # Safe last_hit detection (last_hit while inside own tower attack range).
        self._safe_last_hit_count = 0
        # Duel summoner timing: 80110 is a "commit to a real hero trade"
        # reward. Light poke is neutral; obvious empty use is penalized.
        self._pending_duel_summoner = None
        self._duel_summoner_timing_value = 0.0
        self._duel_summoner_debug = self._empty_duel_summoner_debug()
        # Low-frequency scenario shaping around minion-tanked tower pushes and
        # enemy-dead cake invades.
        self._minion_tower_push_value = 0.0
        self._enemy_dead_enemy_cake_value = 0.0
        self._enemy_minion_tower_front_value = 0.0
        self._enemy_minion_under_own_tower_value = 0.0
        self._river_crab_pressure_value = 0.0
        self._early_own_half_hero_offense_value = 0.0
        self._early_own_half_offense_debug = self._empty_early_own_half_offense_debug()
        self._early_own_half_offense_intent_total = 0.0
        self._early_own_half_offense_damage_total = 0.0
        self._early_own_half_offense_total = 0.0
        self._opening_trade_pressure_value = 0.0
        self._opening_trade_debug = self._empty_opening_trade_debug()
        self._opening_trade_visible_total = 0.0
        self._opening_trade_intent_total = 0.0
        self._opening_trade_damage_total = 0.0
        self._opening_trade_total = 0.0
        self._pending_opening_trade = None
        self._opening_contact_exit_frame = None
        self.last_action_context = None
        self._prev_enemy_tower_hp_for_push = None
        self._prev_main_enemy_tower_dist = None
        self._last_tower_push_eval_frame = -10000
        self._prev_enemy_cake_dist = None
        self._last_enemy_dead_cake_eval_frame = -10000
        self._last_enemy_minion_defense_eval_frame = -10000
        self._enemy_minion_defense_debug = self._empty_enemy_minion_defense_debug()
        self._last_river_crab_pressure_frame = -10000
        self._river_crab_pressure_total = 0.0
        self._river_crab_pressure_debug = self._empty_river_crab_pressure_debug()
        self.prev_river_crab_hp = {}
        self._prev_enemy_dead_cnt_for_reward = 0
        self._last_enemy_hero_dead_frame_for_reward = -10000
        self._last_forward_stage = None
        self.init_max_exp_of_each_hero()

    def init_max_exp_of_each_hero(self):
        self.m_each_level_max_exp = dict(GameConfig.LEVEL_MAX_EXP)

    def result(self, frame_data):
        self.frame_data_process(frame_data)
        self.get_reward(frame_data, self.m_reward_value)
        return self.m_reward_value

    def record_action_context(self, frame_data, action):
        """Record pre-step action context used by delayed reward detectors."""
        self.last_luban_skill1_action_context = None
        self.last_action_context = None
        if not isinstance(action, (list, tuple)) or len(action) < 6:
            return
        self.last_action_context = {
            "button": int(action[0]),
            "target": int(action[5]),
        }
        if int(action[0]) != 4:
            return
        main_hero = self._find_main_hero(frame_data)
        if not self._is_luban(main_hero):
            return
        target_index = int(action[5])
        target_entity = self._target_entity(frame_data, main_hero, target_index)
        target_pos = self._entity_pos(target_entity)
        if target_pos is None:
            return
        self.last_luban_skill1_action_context = {
            "target_index": target_index,
            "target_runtime": self._actor_runtime(target_entity),
            "aim_pos": target_pos,
        }

    def set_cur_calc_frame_vec(self, calc_frame_map, frame_data, camp):
        main_hero, enemy_hero = self._find_heroes_by_camp(frame_data, camp)
        main_tower, enemy_tower = self._find_towers_by_camp(frame_data, camp)

        for reward_name, reward_struct in calc_frame_map.items():
            reward_struct.last_frame_value = reward_struct.cur_frame_value
            if main_hero is None:
                reward_struct.cur_frame_value = 0.0
                continue

            if reward_name == "money":
                reward_struct.cur_frame_value = _get(main_hero, "money_cnt", _get(main_hero, "money", 0))
            elif reward_name == "hp_point":
                hp_rate = _hp_rate(main_hero)
                reward_struct.cur_frame_value = math.sqrt(math.sqrt(max(hp_rate, 0.0)))
            elif reward_name == "ep_rate":
                hp = _hp(main_hero)
                reward_struct.cur_frame_value = 0.0 if hp <= 0 else _safe_div(_get(main_hero, "ep", 0), _get(main_hero, "max_ep", 0))
            elif reward_name == "kill":
                reward_struct.cur_frame_value = _get(main_hero, "kill_cnt", 0)
            elif reward_name == "death":
                reward_struct.cur_frame_value = _get(main_hero, "dead_cnt", 0)
            elif reward_name == "tower_hp_point":
                reward_struct.cur_frame_value = _tower_hp_reward_value(main_tower)
            elif reward_name == "last_hit":
                last_hit_value, last_hit_debug = self._count_last_hit(frame_data, main_hero, enemy_hero)
                reward_struct.cur_frame_value = last_hit_value
                if calc_frame_map is self.m_main_calc_frame_map:
                    self.m_last_hit_debug = last_hit_debug
            elif reward_name == "exp":
                reward_struct.cur_frame_value = self.calculate_exp_sum(main_hero)
            elif reward_name == "forward":
                reward_struct.cur_frame_value = self.calculate_forward(main_hero, frame_data)
            elif reward_name == "cleanse_success":
                if calc_frame_map is self.m_main_calc_frame_map:
                    reward_struct.cur_frame_value = float(
                        self._cleanse_main_count - self._skill2_misuse_count
                    )
                else:
                    reward_struct.cur_frame_value = 0.0
            elif reward_name in SKILL_HIT_REWARD_KEYS:
                if calc_frame_map is self.m_main_calc_frame_map:
                    reward_struct.cur_frame_value = float(self._skill_hit_reward_counts.get(reward_name, 0.0))
                else:
                    reward_struct.cur_frame_value = 0.0
            elif reward_name == "cake_pickup":
                if calc_frame_map is self.m_main_calc_frame_map:
                    reward_struct.cur_frame_value = float(self._cake_pickup_count)
                else:
                    reward_struct.cur_frame_value = 0.0
            elif reward_name == "recover_skill_low_hp":
                if calc_frame_map is self.m_main_calc_frame_map:
                    reward_struct.cur_frame_value = float(self._recover_low_hp_count)
                else:
                    reward_struct.cur_frame_value = 0.0
            elif reward_name == "safe_last_hit":
                if calc_frame_map is self.m_main_calc_frame_map:
                    # Reuse m_last_hit_debug populated by the last_hit branch above
                    # (REWARD_WEIGHT_DICT order guarantees last_hit precedes safe_last_hit).
                    main_kill_count = self.m_last_hit_debug.get("last_hit_main_count", 0)
                    if main_kill_count > 0 and self._is_under_own_tower(main_hero, main_tower):
                        reward_struct.cur_frame_value = float(main_kill_count)
                        self._safe_last_hit_count = int(main_kill_count)
                    else:
                        reward_struct.cur_frame_value = 0.0
                        self._safe_last_hit_count = 0
                else:
                    reward_struct.cur_frame_value = 0.0
            elif reward_name == "duel_summoner_timing":
                if calc_frame_map is self.m_main_calc_frame_map:
                    reward_struct.cur_frame_value = float(self._duel_summoner_timing_value)
                else:
                    reward_struct.cur_frame_value = 0.0
            elif reward_name == "minion_tower_push":
                if calc_frame_map is self.m_main_calc_frame_map:
                    reward_struct.cur_frame_value = float(self._minion_tower_push_value)
                else:
                    reward_struct.cur_frame_value = 0.0
            elif reward_name == "enemy_dead_enemy_cake":
                if calc_frame_map is self.m_main_calc_frame_map:
                    reward_struct.cur_frame_value = float(self._enemy_dead_enemy_cake_value)
                else:
                    reward_struct.cur_frame_value = 0.0
            elif reward_name == "enemy_minion_tower_front":
                if calc_frame_map is self.m_main_calc_frame_map:
                    reward_struct.cur_frame_value = float(self._enemy_minion_tower_front_value)
                else:
                    reward_struct.cur_frame_value = 0.0
            elif reward_name == "enemy_minion_under_own_tower":
                if calc_frame_map is self.m_main_calc_frame_map:
                    reward_struct.cur_frame_value = float(self._enemy_minion_under_own_tower_value)
                else:
                    reward_struct.cur_frame_value = 0.0
            elif reward_name == "river_crab_pressure":
                if calc_frame_map is self.m_main_calc_frame_map:
                    reward_struct.cur_frame_value = float(self._river_crab_pressure_value)
                else:
                    reward_struct.cur_frame_value = 0.0
            elif reward_name == "early_own_half_hero_offense":
                if calc_frame_map is self.m_main_calc_frame_map:
                    reward_struct.cur_frame_value = float(self._early_own_half_hero_offense_value)
                else:
                    reward_struct.cur_frame_value = 0.0
            elif reward_name == "opening_trade_pressure":
                if calc_frame_map is self.m_main_calc_frame_map:
                    reward_struct.cur_frame_value = float(self._opening_trade_pressure_value)
                else:
                    reward_struct.cur_frame_value = 0.0
            elif reward_name in ("win", "no_op_streak_penalty"):
                reward_struct.cur_frame_value = 0.0

    def calculate_exp_sum(self, hero):
        exp_sum = 0.0
        level = int(_get(hero, "level", 1) or 1)
        for level_idx in range(1, level):
            exp_sum += self.m_each_level_max_exp.get(level_idx, 0)
        exp_sum += _get(hero, "exp", 0)
        return exp_sum

    def calculate_forward(self, main_hero, frame_data=None):
        if main_hero is None:
            return 0.0
        hp_rate = _hp_rate(main_hero)
        if hp_rate <= 0.0:
            return 0.0
        frame_no = 0
        if frame_data is not None:
            frame_no = int(_get_any(frame_data, ["frame_no", "frameNo"], 0) or 0)
        lane = self._own_perspective_lane(main_hero)
        if lane is None:
            return 0.0

        if self._opening_enemy_contact(frame_data, main_hero):
            return 0.0

        target = self._opening_forward_target(frame_data, frame_no, main_hero)
        if target is None:
            return 0.0
        target_lane, target_width = target
        width = self._own_perspective_width(main_hero)
        if width is None:
            return 0.0
        return self._opening_position_score(lane, width, target_lane, target_width)

    def _enemy_hero_visible_to_main(self, frame_data, main_hero):
        if frame_data is None or main_hero is None:
            return False
        main_camp = _camp_key(_get(main_hero or {}, "camp", self.main_hero_camp))
        enemy_hero = None
        for hero in frame_data.get("hero_states", []) or []:
            if hero is main_hero:
                continue
            if _camp_key(_get(hero, "camp", None)) != main_camp:
                enemy_hero = hero
                break
        if enemy_hero is None:
            return False
        camp_visible = _get_any(enemy_hero, ["camp_visible", "campVisible"], None)
        if isinstance(camp_visible, (list, tuple)):
            visible_idx = int(main_camp or 0) - 1
            if 0 <= visible_idx < len(camp_visible):
                return bool(camp_visible[visible_idx])
        return True

    def _opening_position_score(self, lane, width, target_lane, target_width):
        lane_error = max(0.0, abs(float(lane) - float(target_lane)) - float(GameConfig.OPENING_TARGET_BAND))
        width_error = (
            0.0
            if target_width is None
            else max(0.0, abs(float(width) - float(target_width)) - float(GameConfig.OPENING_WIDTH_BAND))
        )
        score = -(
            lane_error / max(float(GameConfig.OPENING_LANE_SCALE), 1.0)
            + width_error / max(float(GameConfig.OPENING_WIDTH_SCALE), 1.0)
        )
        clip = float(GameConfig.OPENING_POSITION_CLIP)
        return max(min(score, 0.0), -clip)

    def _opening_forward_stage(self, frame_no, main_hero=None):
        frame_no = int(frame_no or 0)
        air_start_frame = self._opening_air_attack_start_frame(main_hero)
        air_end_frame = self._opening_air_attack_follow_frame(main_hero)
        if int(GameConfig.OPENING_WAVE_GUARD_WIDTH_PULL_FRAME) <= frame_no < air_start_frame:
            return "width_pull"
        if frame_no < int(GameConfig.OPENING_WAVE_GUARD_START_FRAME):
            return "approach_tower"
        if frame_no < air_start_frame:
            return "wait_tower"
        if frame_no < air_end_frame:
            return "air_attack"
        if frame_no < int(GameConfig.OPENING_FORWARD_END_FRAME):
            return "follow_wave"
        return "off"

    def _opening_forward_target(self, frame_data, frame_no, main_hero):
        stage = self._opening_forward_stage(frame_no, main_hero)
        if stage == "approach_tower":
            return (
                float(GameConfig.OPENING_TOWER_TARGET_LANE),
                self._opening_approach_target_width(main_hero),
            )
        if stage == "wait_tower":
            return (
                float(GameConfig.OPENING_WAVE_GUARD_WAIT_LANE),
                float(GameConfig.OPENING_WAVE_GUARD_TARGET_WIDTH),
            )
        if stage == "width_pull":
            return (
                float(GameConfig.OPENING_WAVE_GUARD_WAIT_LANE),
                float(GameConfig.OPENING_WAVE_GUARD_FOLLOW_WIDTH),
            )
        if stage == "follow_wave":
            current_lane = self._own_perspective_lane(main_hero)
            target_lane = (
                float(GameConfig.OPENING_WAVE_GUARD_WAIT_LANE)
                if current_lane is None
                else float(current_lane) + float(GameConfig.OPENING_WAVE_GUARD_FORWARD_DELTA)
            )
            return (
                target_lane,
                float(GameConfig.OPENING_WAVE_GUARD_FOLLOW_WIDTH),
            )
        return None

    def _opening_air_attack_start_frame(self, main_hero=None):
        hero_id = int(_get_any(main_hero or {}, ["config_id", "configId", "hero_id", "heroId"], 0) or 0)
        starts = getattr(GameConfig, "OPENING_AIR_ATTACK_START_FRAMES_BY_HERO", {}) or {}
        return int(starts.get(hero_id, GameConfig.OPENING_WAVE_GUARD_AIR_ATTACK_START_FRAME))

    def _opening_air_attack_follow_frame(self, main_hero=None):
        hero_id = int(_get_any(main_hero or {}, ["config_id", "configId", "hero_id", "heroId"], 0) or 0)
        counts = getattr(GameConfig, "OPENING_AIR_ATTACK_COUNTS_BY_HERO", {}) or {}
        count = int(counts.get(hero_id, len(GameConfig.OPENING_AIR_ATTACK_FRAMES)))
        return (
            self._opening_air_attack_start_frame(main_hero)
            + int(GameConfig.OPENING_AIR_ATTACK_INTERVAL_FRAMES) * max(count - 1, 0)
            + 1
        )

    def _own_perspective_lane(self, unit):
        collider = _get(unit or {}, "collider", {}) or {}
        location = _get(collider, "location", None) or _get(unit or {}, "location", None)
        if not location:
            return None
        try:
            x = float(_get(location, "x", None))
            z = float(_get(location, "z", None))
        except (TypeError, ValueError):
            return None
        if abs(x) > Args.RAW_COORD_ABS_LIMIT or abs(z) > Args.RAW_COORD_ABS_LIMIT:
            return None
        if _camp_key(_get(unit or {}, "camp", None)) == 2:
            x, z = -x, -z
        return (x + z) / Args.SQRT2

    def _own_perspective_width(self, unit):
        collider = _get(unit or {}, "collider", {}) or {}
        location = _get(collider, "location", None) or _get(unit or {}, "location", None)
        if not location:
            return None
        try:
            x = float(_get(location, "x", None))
            z = float(_get(location, "z", None))
        except (TypeError, ValueError):
            return None
        if abs(x) > Args.RAW_COORD_ABS_LIMIT or abs(z) > Args.RAW_COORD_ABS_LIMIT:
            return None
        if _camp_key(_get(unit or {}, "camp", None)) == 2:
            x, z = -x, -z
        return (x - z) / Args.SQRT2

    def _opening_approach_target_width(self, main_hero):
        width = self._own_perspective_width(main_hero)
        if width is None:
            return None
        width_limit = float(GameConfig.OPENING_WAVE_GUARD_APPROACH_WIDTH_LIMIT)
        if abs(width) <= width_limit:
            return None
        return width_limit if width > 0 else -width_limit

    def _opening_enemy_contact(self, frame_data, main_hero):
        return self._enemy_hero_visible_to_main(frame_data, main_hero) or self._enemy_minion_visible_to_main(
            frame_data, main_hero
        )

    def _enemy_minion_visible_to_main(self, frame_data, main_hero):
        if frame_data is None or main_hero is None:
            return False
        main_camp = _camp_key(_get(main_hero or {}, "camp", self.main_hero_camp))
        for npc in frame_data.get("npc_states", []) or []:
            if _get_any(npc, ["sub_type", "subType"], None) not in SOLDIER_SUB_TYPES:
                continue
            if _camp_key(_get(npc, "camp", None)) == main_camp:
                continue
            if float(_hp(npc) or 0) <= 0:
                continue
            camp_visible = _get_any(npc, ["camp_visible", "campVisible"], None)
            if isinstance(camp_visible, (list, tuple)):
                visible_idx = int(main_camp or 0) - 1
                if 0 <= visible_idx < len(camp_visible):
                    return bool(camp_visible[visible_idx])
            return True
        return False

    def _second_front_own_minion(self, frame_data):
        if frame_data is None:
            return None
        minions = []
        main_camp = _camp_key(getattr(self, "main_hero_camp", None))
        if main_camp not in (1, 2):
            return None
        for npc in frame_data.get("npc_states", []) or []:
            if _get_any(npc, ["sub_type", "subType"], None) not in SOLDIER_SUB_TYPES:
                continue
            if float(_hp(npc) or 0) <= 0:
                continue
            npc_camp = _camp_key(_get(npc, "camp", None))
            if npc_camp != main_camp:
                continue
            lane = self._lane_for_camp(npc, main_camp)
            if lane is None:
                continue
            minions.append((float(lane), npc))
        if not minions:
            return None
        minions.sort(key=lambda item: item[0], reverse=True)
        return minions[1][1] if len(minions) >= 2 else minions[0][1]

    def _back_own_minion(self, frame_data):
        if frame_data is None:
            return None
        back_minion = None
        back_lane = None
        main_camp = _camp_key(getattr(self, "main_hero_camp", None))
        if main_camp not in (1, 2):
            return None
        for npc in frame_data.get("npc_states", []) or []:
            if _get_any(npc, ["sub_type", "subType"], None) not in SOLDIER_SUB_TYPES:
                continue
            if float(_hp(npc) or 0) <= 0:
                continue
            npc_camp = _camp_key(_get(npc, "camp", None))
            if npc_camp != main_camp:
                continue
            lane = self._lane_for_camp(npc, main_camp)
            if lane is None:
                continue
            if back_lane is None or float(lane) < back_lane:
                back_lane = float(lane)
                back_minion = npc
        return back_minion

    def _front_own_minion(self, frame_data):
        if frame_data is None:
            return None
        front_minion = None
        front_lane = None
        main_camp = _camp_key(getattr(self, "main_hero_camp", None))
        if main_camp not in (1, 2):
            return None
        for npc in frame_data.get("npc_states", []) or []:
            if _get_any(npc, ["sub_type", "subType"], None) not in SOLDIER_SUB_TYPES:
                continue
            if float(_hp(npc) or 0) <= 0:
                continue
            npc_camp = _camp_key(_get(npc, "camp", None))
            if npc_camp != main_camp:
                continue
            lane = self._lane_for_camp(npc, main_camp)
            if lane is None:
                continue
            if front_lane is None or float(lane) > front_lane:
                front_lane = float(lane)
                front_minion = npc
        return front_minion

    def frame_data_process(self, frame_data):
        main_camp, enemy_camp = -1, -1
        main_hero, enemy_hero = None, None

        for hero in frame_data.get("hero_states", []):
            if self._is_main_hero(hero):
                main_camp = _get(hero, "camp")
                self.main_hero_camp = main_camp
                main_hero = hero
            else:
                enemy_camp = _get(hero, "camp")
                enemy_hero = hero
        frame_no = frame_data.get("frame_no", frame_data.get("frameNo", 0))
        self._cleanse_main_count, self._cleanse_enemy_count, self._skill2_misuse_count = self._compute_cleanse_events(
            main_hero, enemy_hero, frame_no
        )
        self.main_total_hurt_to_hero_delta = self._compute_main_total_hurt_to_hero_delta(main_hero)
        self._skill_hit_reward_counts = self._compute_skill_hit_events(frame_data, main_hero, enemy_hero)
        self._cake_pickup_count = self._detect_cake_pickup(frame_data, main_hero, enemy_hero, frame_no)
        self._recover_low_hp_count = self._detect_recover_low_hp(main_hero, enemy_hero, frame_no)
        self._duel_summoner_timing_value = self._detect_duel_summoner_timing(
            frame_data, main_hero, enemy_hero, frame_no
        )
        main_tower, enemy_tower = self._find_towers_by_camp(frame_data, main_camp)
        own_soldiers_in_enemy_tower = self._own_soldiers_in_enemy_tower_range(
            frame_data, main_hero, enemy_tower
        )
        self._minion_tower_push_value = self._detect_minion_tower_push(
            frame_data, main_hero, enemy_hero, enemy_tower, own_soldiers_in_enemy_tower
        )
        self._enemy_dead_enemy_cake_value = self._detect_enemy_dead_enemy_cake(
            frame_data, main_hero, enemy_hero, enemy_tower, own_soldiers_in_enemy_tower
        )
        self._river_crab_pressure_value = self._detect_river_crab_pressure(
            frame_data, main_hero, enemy_hero, main_tower, enemy_tower, own_soldiers_in_enemy_tower
        )
        self._early_own_half_hero_offense_value = self._detect_early_own_half_hero_offense(
            frame_data, main_hero, enemy_hero, frame_no
        )
        self._opening_trade_pressure_value = self._detect_opening_trade_pressure(
            frame_data, main_hero, enemy_hero, frame_no
        )
        (
            self._enemy_minion_tower_front_value,
            self._enemy_minion_under_own_tower_value,
        ) = self._detect_enemy_minion_own_tower_pressure(frame_data, main_hero, main_tower)
        self.set_cur_calc_frame_vec(self.m_main_calc_frame_map, frame_data, main_camp)
        self.set_cur_calc_frame_vec(self.m_enemy_calc_frame_map, frame_data, enemy_camp)

    def get_reward(self, frame_data, reward_dict):
        reward_dict.clear()
        frame_no = frame_data.get("frame_no", frame_data.get("frameNo", 0))
        reward_sum = 0.0
        main_hero = self._find_main_hero(frame_data)
        main_tower, enemy_tower = self._find_main_enemy_towers(frame_data, main_hero)

        for reward_name, reward_struct in self.m_cur_calc_frame_map.items():
            if reward_name == "ep_rate":
                reward_struct.cur_frame_value = self.m_main_calc_frame_map[reward_name].cur_frame_value
                reward_struct.last_frame_value = self.m_main_calc_frame_map[reward_name].last_frame_value
                reward_struct.value = (
                    reward_struct.cur_frame_value - reward_struct.last_frame_value
                    if reward_struct.last_frame_value > 0
                    else 0.0
                )
            elif reward_name == "exp" and main_hero is not None and _get(main_hero, "level", 1) >= 15:
                reward_struct.value = 0.0
            elif reward_name == "forward":
                stage = self._opening_forward_stage(frame_no, main_hero)
                if stage in ("approach_tower", "wait_tower", "width_pull", "follow_wave"):
                    reward_struct.value = self.m_main_calc_frame_map[reward_name].cur_frame_value
                else:
                    reward_struct.value = 0.0
                self._last_forward_stage = stage
            elif reward_name == "last_hit":
                reward_struct.value = self.m_main_calc_frame_map[reward_name].cur_frame_value
            elif reward_name == "cleanse_success":
                reward_struct.value = self.m_main_calc_frame_map[reward_name].cur_frame_value
            elif reward_name in SKILL_HIT_REWARD_KEYS:
                reward_struct.value = self.m_main_calc_frame_map[reward_name].cur_frame_value
            elif reward_name == "cake_pickup":
                reward_struct.value = self.m_main_calc_frame_map[reward_name].cur_frame_value
            elif reward_name == "recover_skill_low_hp":
                reward_struct.value = self.m_main_calc_frame_map[reward_name].cur_frame_value
            elif reward_name == "safe_last_hit":
                reward_struct.value = self.m_main_calc_frame_map[reward_name].cur_frame_value
            elif reward_name == "duel_summoner_timing":
                reward_struct.value = self.m_main_calc_frame_map[reward_name].cur_frame_value
            elif reward_name in SCENARIO_REWARD_KEYS:
                reward_struct.value = self.m_main_calc_frame_map[reward_name].cur_frame_value
            elif reward_name in ("win", "no_op_streak_penalty"):
                reward_struct.value = 0.0
            else:
                reward_struct.cur_frame_value = (
                    self.m_main_calc_frame_map[reward_name].cur_frame_value
                    - self.m_enemy_calc_frame_map[reward_name].cur_frame_value
                )
                reward_struct.last_frame_value = (
                    self.m_main_calc_frame_map[reward_name].last_frame_value
                    - self.m_enemy_calc_frame_map[reward_name].last_frame_value
                )
                reward_struct.value = reward_struct.cur_frame_value - reward_struct.last_frame_value

            time_scale = self._time_scale(reward_name, frame_no)

            if not self.has_last_frame:
                reward_struct.value = 0.0

            weight_multiplier = self._reward_weight_multiplier(
                reward_name, frame_no, main_hero, main_tower, enemy_tower
            )
            reward_dict[f"{reward_name}_origin"] = reward_struct.value
            reward_dict[f"{reward_name}_weight"] = (
                reward_struct.value * reward_struct.weight * time_scale * weight_multiplier
            )
            reward_sum += reward_dict[f"{reward_name}_weight"]

        reward_dict["reward_sum"] = reward_sum
        reward_dict.update(self.m_last_hit_debug)
        reward_dict.update(self._cake_debug)
        reward_dict.update(self._recover_debug)
        reward_dict.update(self._duel_summoner_debug)
        reward_dict.update(self._enemy_minion_defense_debug)
        reward_dict.update(self._river_crab_pressure_debug)
        reward_dict.update(self._early_own_half_offense_debug)
        reward_dict.update(self._opening_trade_debug)
        reward_dict["direnjie_skill3_followup_count"] = float(self._direnjie_skill3_followup_count_value)
        reward_dict["direnjie_skill3_miss_count"] = float(self._direnjie_skill3_miss_count_value)
        self.has_last_frame = True
        return reward_dict

    def _reward_weight_multiplier(self, reward_name, frame_no, main_hero, main_tower, enemy_tower):
        if reward_name != "death":
            return 1.0
        enemy_tower_hp_rate = _hp_rate(enemy_tower or {})
        if 0.0 < enemy_tower_hp_rate < GameConfig.DEATH_MULTIPLIER_ENEMY_TOWER_HP_THRESHOLD:
            return GameConfig.DEATH_MULTIPLIER_ENEMY_TOWER_LOW_HP
        if int(frame_no or 0) < GameConfig.CANNON_FRAME:
            return GameConfig.DEATH_MULTIPLIER_BEFORE_CANNON
        return GameConfig.DEATH_MULTIPLIER_AFTER_CANNON

    def _find_main_enemy_towers(self, frame_data, main_hero):
        main_tower, enemy_tower = None, None
        if main_hero is None:
            return main_tower, enemy_tower
        main_camp = _get(main_hero, "camp", None)
        for npc in frame_data.get("npc_states", []) or []:
            if _get_any(npc, ["sub_type", "subType"], None) not in TOWER_SUB_TYPES:
                continue
            if _camp_key(_get(npc, "camp", None)) == _camp_key(main_camp):
                main_tower = npc
            else:
                enemy_tower = npc
        return main_tower, enemy_tower

    def _time_scale(self, reward_name, frame_no):
        if reward_name in GameConfig.REWARD_WITHOUT_TIME_SCALE:
            return 1.0
        scale_arg = getattr(GameConfig, "REWARD_TIME_SCALE_OVERRIDES", {}).get(reward_name, self.time_scale_arg)
        if scale_arg == 0:
            return 1.0
        if scale_arg is None or scale_arg <= 0:
            return 1.0
        return math.pow(0.6, float(frame_no) / scale_arg)

    def _find_main_hero(self, frame_data):
        for hero in frame_data.get("hero_states", []):
            if self._is_main_hero(hero):
                return hero
        return None

    def _is_main_hero(self, hero):
        return (
            _get(hero, "runtime_id", None) == self.main_hero_player_id
            or _get(hero, "player_id", None) == self.main_hero_player_id
        )

    def _find_heroes_by_camp(self, frame_data, camp):
        main_hero, enemy_hero = None, None
        for hero in frame_data.get("hero_states", []):
            if _camp_key(_get(hero, "camp")) == _camp_key(camp):
                main_hero = hero
            else:
                enemy_hero = hero
        return main_hero, enemy_hero

    def _find_towers_by_camp(self, frame_data, camp):
        main_tower, enemy_tower = None, None
        for npc in frame_data.get("npc_states", []):
            if _get_any(npc, ["sub_type", "subType"], None) not in TOWER_SUB_TYPES:
                continue
            if _camp_key(_get(npc, "camp")) == _camp_key(camp):
                main_tower = npc
            else:
                enemy_tower = npc
        return main_tower, enemy_tower

    def _count_last_hit(self, frame_data, main_hero, enemy_hero):
        frame_action = _get_any(frame_data, ["frame_action", "frameAction"], {}) or {}
        dead_actions = _get_any(frame_action, ["dead_action", "deadAction"], [])
        if not isinstance(dead_actions, list):
            dead_actions = []
        main_runtime = self._actor_runtime(main_hero)
        enemy_runtime = self._actor_runtime(enemy_hero)
        debug = self._empty_last_hit_debug()
        debug["last_hit_dead_action_count"] = len(dead_actions)
        value = 0.0
        for dead_action in dead_actions:
            killer = _get_any(dead_action, ["killer", "killer_actor"], {}) if isinstance(dead_action, dict) else {}
            death = _get_any(dead_action, ["death", "dead"], {}) if isinstance(dead_action, dict) else {}
            if self._actor_sub_type(death) not in SOLDIER_SUB_TYPES:
                continue
            debug["last_hit_soldier_dead_count"] += 1
            killer_id = self._actor_runtime(killer)
            if killer_id is None:
                killer_id = self._last_hurt_runtime(death)
            if killer_id == main_runtime:
                value += 1.0
                debug["last_hit_main_count"] += 1
            elif killer_id == enemy_runtime:
                value -= 1.0
                debug["last_hit_enemy_count"] += 1
        return value, debug

    def _empty_last_hit_debug(self):
        return {
            "last_hit_dead_action_count": 0,
            "last_hit_soldier_dead_count": 0,
            "last_hit_main_count": 0,
            "last_hit_enemy_count": 0,
        }

    def _empty_recover_debug(self):
        return {
            "recover_attempt_count": 0,
            "recover_success_count": 0,
            "recover_interrupted_count": 0,
            "recover_high_hp_penalty_count": 0,
        }

    def _empty_cake_debug(self):
        return {
            "cake_success_count": 0,
            "cake_interrupted_count": 0,
            "cake_wasted_count": 0,
            "cake_high_hp_penalty_count": 0,
        }

    def _empty_duel_summoner_debug(self):
        return {
            "duel_summoner_cast_count": 0.0,
            "duel_summoner_good_count": 0.0,
            "duel_summoner_mid_count": 0.0,
            "duel_summoner_touch_count": 0.0,
            "duel_summoner_poke_count": 0.0,
            "duel_summoner_wasted_count": 0.0,
            "duel_summoner_80110_cast_count": 0.0,
            "duel_summoner_80110_good_count": 0.0,
        }

    def _empty_early_own_half_offense_debug(self):
        return {
            "early_own_half_offense_intent_count": 0.0,
            "early_own_half_offense_damage_count": 0.0,
            "early_own_half_offense_damage_value": 0.0,
        }

    def _empty_opening_trade_debug(self):
        return {
            "opening_trade_start_count": 0.0,
            "opening_trade_visible_count": 0.0,
            "opening_trade_intent_count": 0.0,
            "opening_trade_damage_count": 0.0,
            "opening_trade_80110_good_count": 0.0,
            "opening_trade_good_count": 0.0,
            "opening_trade_strong_count": 0.0,
            "opening_trade_bad_count": 0.0,
            "opening_trade_wasted_80110_count": 0.0,
            "opening_trade_reward_value": 0.0,
        }

    def _empty_enemy_minion_defense_debug(self):
        return {
            "enemy_minion_tower_front_count": 0.0,
            "enemy_minion_under_own_tower_count": 0.0,
            "enemy_minion_defense_event_count": 0.0,
            "enemy_minion_defense_multiplier": 0.0,
        }

    def _empty_river_crab_pressure_debug(self):
        return {
            "river_crab_pressure_count": 0.0,
        }

    def _actor_runtime(self, actor):
        if actor is None:
            return None
        actor_state = _get_any(actor, ["actor_state", "actorState"], None)
        if actor_state is not None:
            runtime_id = _get_any(actor_state, ["runtime_id", "runtimeId"], None)
            if runtime_id is not None:
                return runtime_id
        return _get_any(actor, ["runtime_id", "runtimeId"], None)

    def _actor_sub_type(self, actor):
        if actor is None:
            return None
        actor_state = _get_any(actor, ["actor_state", "actorState"], None)
        if actor_state is not None:
            sub_type = _get_any(actor_state, ["sub_type", "subType"], None)
            if sub_type is not None:
                return sub_type
        return _get_any(actor, ["sub_type", "subType"], None)

    def _last_hurt_runtime(self, death):
        single_hurt_list = _get_any(death, ["single_hurt_list", "singleHurtList"], [])
        if not isinstance(single_hurt_list, list) or not single_hurt_list:
            return None
        last_hurt = max(
            single_hurt_list,
            key=lambda item: _get_any(item, ["frameNo", "frame_no", "frame"], -1),
        )
        return _get_any(last_hurt, ["runtime_id", "runtimeId"], None)

    def _is_di_renjie(self, hero):
        if hero is None:
            return False
        hero_id = _get_any(hero, ["config_id", "configId", "hero_id", "heroId"], 0)
        try:
            return int(hero_id or 0) == 133
        except (TypeError, ValueError):
            return False

    def _used_skill2(self, hero):
        if hero is None:
            return False
        skill_state = _get_any(hero or {}, ["skill_state", "skillState"], {}) or {}
        slots = _get_any(skill_state, ["slot_states", "slotStates"], []) or []
        for slot in slots:
            slot_type = _get_any(slot, ["slot_type", "slotType"], None)
            if isinstance(slot_type, int):
                slot_idx = slot_type
            else:
                text = str(slot_type) if slot_type is not None else ""
                if text.startswith("SLOT_SKILL_"):
                    try:
                        slot_idx = int(text.split("_")[-1])
                    except (TypeError, ValueError):
                        slot_idx = -1
                else:
                    try:
                        slot_idx = int(slot_type) if slot_type is not None else -1
                    except (TypeError, ValueError):
                        slot_idx = -1
            if slot_idx == 2:
                succ = _get_any(slot, ["succUsedInFrame", "succ_used_in_frame"], 0)
                try:
                    return float(succ) > 0
                except (TypeError, ValueError):
                    return False
        return False

    def _compute_cleanse_events(self, main_hero, enemy_hero, frame_no):
        CLEANSE_WINDOW = getattr(GameConfig, "CLEANSE_WINDOW_FRAMES", 300)

        if not self._is_di_renjie(main_hero) or not self._is_di_renjie(enemy_hero):
            self._main_ult_hit_frame = -99999
            self._enemy_ult_hit_frame = -99999
            return 0, 0, 0

        main_runtime = self._actor_runtime(main_hero)
        enemy_runtime = self._actor_runtime(enemy_hero)

        for hurt in _get(main_hero, "take_hurt_infos", []) or []:
            atker = _get_any(hurt, ["atker", "attacker"], None)
            slot = self._slot_idx(_get_any(hurt, ["skillSlot", "skill_slot"], -1))
            if atker == enemy_runtime and slot == 3:
                self._main_ult_hit_frame = frame_no
                break

        for hurt in _get(enemy_hero, "take_hurt_infos", []) or []:
            atker = _get_any(hurt, ["atker", "attacker"], None)
            slot = self._slot_idx(_get_any(hurt, ["skillSlot", "skill_slot"], -1))
            if atker == main_runtime and slot == 3:
                self._enemy_ult_hit_frame = frame_no
                break

        main_used_skill2 = self._used_skill2(main_hero)
        enemy_used_skill2 = self._used_skill2(enemy_hero)

        main_cleansed = 0
        if frame_no - self._main_ult_hit_frame <= CLEANSE_WINDOW and main_used_skill2:
            main_cleansed = 1
            self._main_ult_hit_frame = -99999

        enemy_cleansed = 0
        if frame_no - self._enemy_ult_hit_frame <= CLEANSE_WINDOW and enemy_used_skill2:
            enemy_cleansed = 1
            self._enemy_ult_hit_frame = -99999

        main_skill2_misuse = 0
        if main_used_skill2 and not self._is_in_berserk_engagement(main_hero, enemy_hero):
            main_skill2_misuse = 1

        return main_cleansed, enemy_cleansed, main_skill2_misuse

    def _slot_idx(self, slot_type):
        if isinstance(slot_type, int):
            return slot_type
        text = str(slot_type) if slot_type is not None else ""
        if text.startswith("SLOT_SKILL_"):
            try:
                return int(text.split("_")[-1])
            except (TypeError, ValueError):
                return -1
        try:
            return int(slot_type) if slot_type is not None else -1
        except (TypeError, ValueError):
            return -1

    def _slot_succ_used(self, hero, target_slot):
        if hero is None:
            return False
        skill_state = _get_any(hero or {}, ["skill_state", "skillState"], {}) or {}
        slots = _get_any(skill_state, ["slot_states", "slotStates"], []) or []
        for slot in slots:
            if self._slot_idx(_get_any(slot, ["slot_type", "slotType"], None)) != target_slot:
                continue
            succ = _get_any(slot, ["succUsedInFrame", "succ_used_in_frame"], 0)
            try:
                return float(succ) > 0
            except (TypeError, ValueError):
                return False
        return False

    def _empty_skill_hit_reward_counts(self):
        return {key: 0.0 for key in SKILL_HIT_REWARD_KEYS}

    def _count_skill_hits_on(self, victim, attacker_runtime, attacker_side, attacker_config, reward_key_map, frame_no):
        # Walk victim.take_hurt_infos and count skill 1/2/3 hits attributed to attacker.
        # Slot 3 hits are capped via attacker_side's armed/credit_used state.
        counts = self._empty_skill_hit_reward_counts()
        if victim is None or attacker_runtime is None:
            return counts
        hurts = _get(victim, "take_hurt_infos", []) or []
        for hurt in hurts:
            atker = _get_any(hurt, ["atker", "attacker"], None)
            if atker is None or str(atker) != str(attacker_runtime):
                continue
            slot_int = self._slot_idx(_get_any(hurt, ["skillSlot", "skill_slot"], -1))
            reward_key = reward_key_map.get((attacker_config, slot_int))
            if not reward_key:
                continue
            try:
                hurt_val = float(_get_any(hurt, ["hurtValue", "hurt_value"], 0) or 0)
            except (TypeError, ValueError):
                continue
            if hurt_val <= 0:
                continue
            if slot_int == 3:
                if self._credit_slot3_hit(attacker_side, frame_no):
                    counts[reward_key] += 1.0
            else:
                counts[reward_key] += 1.0
        return counts

    def _count_skill_hit_targets(self, attacker, victim, attacker_side, attacker_config, reward_key_map, frame_no):
        counts = self._empty_skill_hit_reward_counts()
        if attacker is None or victim is None:
            return counts
        victim_runtime = self._actor_runtime(victim)
        if victim_runtime is None:
            return counts
        for hit in _get_any(attacker, ["hit_target_info", "hitTargetInfo"], []) or []:
            if not isinstance(hit, dict):
                continue
            target_runtime = _get_any(hit, ["hit_target", "hitTarget"], None)
            if target_runtime is None or str(target_runtime) != str(victim_runtime):
                continue
            slot_int = self._slot_idx(_get_any(hit, ["slot_type", "slotType"], -1))
            reward_key = reward_key_map.get((attacker_config, slot_int))
            if not reward_key:
                continue
            if slot_int == 3:
                if self._credit_slot3_hit(attacker_side, frame_no):
                    counts[reward_key] += 1.0
            else:
                counts[reward_key] += 1.0
        return counts

    def _count_skill_hit_hero_times_delta(self, attacker, attacker_side, attacker_config, reward_key_map, frame_no):
        counts = self._empty_skill_hit_reward_counts()
        if attacker is None:
            self.prev_skill_hit_hero_times[attacker_side] = {}
            return counts

        current = {}
        for slot in self._slot_states(attacker):
            slot_int = self._slot_idx(_get_any(slot, ["slot_type", "slotType"], -1))
            if slot_int < 0:
                continue
            try:
                hit_times = float(_get_any(slot, ["hitHeroTimes", "hit_hero_times"], 0) or 0)
            except (TypeError, ValueError):
                hit_times = 0.0
            current[slot_int] = max(current.get(slot_int, 0.0), hit_times)

        previous = self.prev_skill_hit_hero_times.get(attacker_side, {})
        self.prev_skill_hit_hero_times[attacker_side] = current
        if not previous:
            return counts

        for slot_int, hit_times in current.items():
            reward_key = reward_key_map.get((attacker_config, slot_int))
            if not reward_key:
                continue
            delta = max(0.0, hit_times - previous.get(slot_int, hit_times))
            if delta <= 0:
                continue
            if slot_int == 3:
                if self._credit_slot3_hit(attacker_side, frame_no, from_counter_delta=True):
                    counts[reward_key] += 1.0
            else:
                counts[reward_key] += delta
        return counts

    def _credit_slot3_hit(self, attacker_side, frame_no, from_counter_delta=False):
        if self.slot3_armed.get(attacker_side, False):
            if self.slot3_credit_used.get(attacker_side, False):
                return False
            self.slot3_credit_used[attacker_side] = True
            return True
        if from_counter_delta:
            self.slot3_fallback_credit_frame[attacker_side] = int(frame_no or 0)
            return True
        frame_no = int(frame_no or 0)
        last_credit = int(self.slot3_fallback_credit_frame.get(attacker_side, -10000))
        if frame_no - last_credit <= 10:
            return False
        self.slot3_fallback_credit_frame[attacker_side] = frame_no
        return True

    def _slot_states(self, hero):
        skill_state = _get_any(hero or {}, ["skill_state", "skillState"], {}) or {}
        return _get_any(skill_state, ["slot_states", "slotStates"], []) or []

    def _compute_skill_hit_events(self, frame_data, main_hero, enemy_hero):
        # Positive-only: reward selected skill hits. Di Renjie skill 2 is not in
        # SKILL_HIT_* maps; it is rewarded only via cleanse_success.
        SLOT3_TIMEOUT = 200
        frame_no = frame_data.get("frame_no", frame_data.get("frameNo", 0))
        counts = self._empty_skill_hit_reward_counts()

        if self.slot3_armed.get("main", False) and frame_no - self.slot3_armed_frame.get("main", -10000) > SLOT3_TIMEOUT:
            self.slot3_armed["main"] = False
            self.slot3_credit_used["main"] = False
        if self._slot_succ_used(main_hero, 3):
            self.slot3_armed["main"] = True
            self.slot3_credit_used["main"] = False
            self.slot3_armed_frame["main"] = frame_no
            if self._is_direnjie(main_hero):
                self.pending_direnjie_skill3_miss_frame = int(frame_no or 0)
        if self._is_luban(main_hero) and self._slot_succ_used(main_hero, 1):
            self.pending_luban_skill1_frame = frame_no
            context = self.last_luban_skill1_action_context or {}
            self.pending_luban_skill1_aim_pos = context.get("aim_pos")
            self.pending_luban_skill1_target_runtime = context.get("target_runtime")
            self.pending_luban_skill1_hit_soldiers = set()
            self.last_luban_skill1_action_context = None

        main_runtime = self._actor_runtime(main_hero)
        main_config = int(_get_any(main_hero or {}, ["config_id", "configId", "hero_id", "heroId"], 0) or 0)
        hurt_counts = self._count_skill_hits_on(
            enemy_hero, main_runtime, "main", main_config, SKILL_HIT_HERO_REWARD_KEYS, frame_no
        )
        hit_target_counts = self._count_skill_hit_targets(
            main_hero, enemy_hero, "main", main_config, SKILL_HIT_HERO_REWARD_KEYS, frame_no
        )
        hit_times_counts = self._count_skill_hit_hero_times_delta(
            main_hero, "main", main_config, SKILL_HIT_HERO_REWARD_KEYS, frame_no
        )
        hero_counts = self._empty_skill_hit_reward_counts()
        for key in hero_counts:
            hero_counts[key] = max(
                hurt_counts.get(key, 0.0),
                hit_target_counts.get(key, 0.0),
                hit_times_counts.get(key, 0.0),
            )
            counts[key] += hero_counts[key]
        if hero_counts.get("direnjie_skill3_hit_enemy_hero", 0.0) > 0:
            self.pending_direnjie_skill3_followup_frame = frame_no
            self.pending_direnjie_skill3_followup_count = 0
            self.pending_direnjie_skill3_miss_frame = None

        counts["luban_skill1_hit_enemy_soldier"] += self._detect_luban_skill1_soldier_hits(
            frame_data, main_hero, frame_no
        )
        counts["direnjie_skill3_followup_damage"] += self._detect_direnjie_skill3_followup_damage(
            enemy_hero, main_hero, main_runtime, frame_no
        )
        counts["direnjie_skill3_miss"] += self._detect_direnjie_skill3_miss(frame_no)
        return counts

    def _is_luban(self, hero):
        return int(_get_any(hero or {}, ["config_id", "configId", "hero_id", "heroId"], 0) or 0) == 112

    def _is_direnjie(self, hero):
        return int(_get_any(hero or {}, ["config_id", "configId", "hero_id", "heroId"], 0) or 0) == 133

    def _enemy_soldier_hp_map(self, frame_data):
        current = {}
        for npc in frame_data.get("npc_states", []) or []:
            if _get_any(npc, ["sub_type", "subType"], None) not in SOLDIER_SUB_TYPES:
                continue
            if _camp_key(_get(npc, "camp", None)) == _camp_key(self.main_hero_camp):
                continue
            runtime = self._actor_runtime(npc)
            if runtime is None:
                continue
            current[runtime] = {
                "hp": float(_hp(npc) or 0),
                "pos": self._entity_pos(npc),
            }
        return current

    def _detect_luban_skill1_soldier_hits(self, frame_data, main_hero, frame_no):
        current_hp = self._enemy_soldier_hp_map(frame_data)
        reward_count = 0.0
        pending_frame = self.pending_luban_skill1_frame
        if self._is_luban(main_hero) and pending_frame is not None:
            elapsed = int(frame_no or 0) - int(pending_frame or 0)
            if 0 <= elapsed <= GameConfig.LUBAN_SKILL1_SOLDIER_HIT_WINDOW:
                for runtime, info in current_hp.items():
                    if runtime in self.pending_luban_skill1_hit_soldiers:
                        continue
                    prev_info = self.prev_enemy_soldier_hp.get(runtime)
                    prev_hp = prev_info.get("hp") if isinstance(prev_info, dict) else prev_info
                    hp = info.get("hp", 0.0)
                    if prev_hp is not None and hp < prev_hp and self._matches_luban_skill1_aim(runtime, info.get("pos")):
                        reward_count += 1.0
                        self.pending_luban_skill1_hit_soldiers.add(runtime)
            elif elapsed > GameConfig.LUBAN_SKILL1_SOLDIER_HIT_WINDOW:
                self.pending_luban_skill1_frame = None
                self.pending_luban_skill1_aim_pos = None
                self.pending_luban_skill1_target_runtime = None
                self.pending_luban_skill1_hit_soldiers = set()
        self.prev_enemy_soldier_hp = current_hp
        return reward_count

    def _matches_luban_skill1_aim(self, runtime, soldier_pos):
        if self.pending_luban_skill1_target_runtime is not None and runtime == self.pending_luban_skill1_target_runtime:
            return True
        if soldier_pos is None or self.pending_luban_skill1_aim_pos is None:
            return False
        return math.dist(soldier_pos, self.pending_luban_skill1_aim_pos) <= GameConfig.LUBAN_SKILL1_SOLDIER_AIM_RADIUS

    def _target_entity(self, frame_data, main_hero, target_index):
        heroes = frame_data.get("hero_states", []) or []
        enemy_hero = None
        for hero in heroes:
            if hero is not main_hero:
                enemy_hero = hero
                break
        target_soldiers = self._target_enemy_soldiers(frame_data, main_hero)
        target_soldiers += [None] * max(0, 4 - len(target_soldiers))
        enemy_tower = self._enemy_tower(frame_data, main_hero)
        monster = self._nearest_monster(frame_data, main_hero)
        target_entities = [None, enemy_hero, main_hero] + target_soldiers[:4] + [enemy_tower, monster]
        if 0 <= target_index < len(target_entities):
            return target_entities[target_index]
        return None

    def _target_enemy_soldiers(self, frame_data, main_hero):
        soldiers = []
        main_camp = _get(main_hero or {}, "camp", None)
        main_pos = self._project_for_target_sort(main_hero)
        for npc in frame_data.get("npc_states", []) or []:
            if _get_any(npc, ["sub_type", "subType"], None) not in SOLDIER_SUB_TYPES:
                continue
            if _camp_key(_get(npc, "camp", None)) == _camp_key(main_camp):
                continue
            soldiers.append(npc)
        return sorted(soldiers, key=lambda unit: self._soldier_target_sort_key(unit, main_pos))[:4]

    def _soldier_target_sort_key(self, soldier, main_pos):
        pos = self._project_for_target_sort(soldier)
        hp_ratio = _hp_rate(soldier)
        lane = pos[0] if pos else 0.0
        frontline_score = -abs(lane)
        progress = lane
        dist_to_self = math.dist(pos, main_pos) if pos is not None and main_pos is not None else 1e9
        return (-frontline_score, -progress, dist_to_self, hp_ratio, self._actor_runtime(soldier) or 0)

    def _project_for_target_sort(self, entity):
        pos = self._entity_pos(entity)
        if pos is None:
            return None
        x, z = pos
        if _camp_key(self.main_hero_camp) == 2:
            x, z = -x, -z
        sqrt2 = math.sqrt(2.0)
        return (x + z) / sqrt2, (x - z) / sqrt2

    def _entity_pos(self, entity):
        if entity is None:
            return None
        collider = _get(entity, "collider", {}) or {}
        loc = _get(collider, "location", None) or _get(entity, "location", None)
        if not loc:
            return None
        return float(_get(loc, "x", 0) or 0), float(_get(loc, "z", 0) or 0)

    def _enemy_tower(self, frame_data, main_hero):
        main_camp = _get(main_hero or {}, "camp", None)
        for npc in frame_data.get("npc_states", []) or []:
            if _get_any(npc, ["sub_type", "subType"], None) not in TOWER_SUB_TYPES:
                continue
            if _camp_key(_get(npc, "camp", None)) != _camp_key(main_camp):
                return npc
        return None

    def _nearest_monster(self, frame_data, main_hero):
        main_pos = self._entity_pos(main_hero)
        monsters = []
        for npc in frame_data.get("npc_states", []) or []:
            sub_type = _get_any(npc, ["sub_type", "subType"], None)
            if sub_type in SOLDIER_SUB_TYPES or sub_type in TOWER_SUB_TYPES:
                continue
            pos = self._entity_pos(npc)
            if pos is not None:
                monsters.append((math.dist(pos, main_pos) if main_pos is not None else 1e9, npc))
        return sorted(monsters, key=lambda item: item[0])[0][1] if monsters else None

    def _detect_direnjie_skill3_followup_damage(self, enemy_hero, main_hero, main_runtime, frame_no):
        self._direnjie_skill3_followup_count_value = 0.0
        if self.pending_direnjie_skill3_followup_frame is None:
            return 0.0
        hit_frame = int(self.pending_direnjie_skill3_followup_frame or 0)
        elapsed = int(frame_no or 0) - hit_frame
        if elapsed <= 0:
            return 0.0
        followup_window = (
            GameConfig.DI_RENJIE_SKILL3_FOLLOWUP_WINDOW_EARLY
            if hit_frame < GameConfig.CANNON_FRAME
            else GameConfig.DI_RENJIE_SKILL3_FOLLOWUP_WINDOW
        )
        if elapsed > followup_window:
            self.pending_direnjie_skill3_followup_frame = None
            self.pending_direnjie_skill3_followup_count = 0
            return 0.0
        if self.pending_direnjie_skill3_followup_count >= GameConfig.DI_RENJIE_SKILL3_FOLLOWUP_CAP:
            return 0.0
        if self._enemy_hero_damaged_by_main(enemy_hero, main_hero, main_runtime):
            self.pending_direnjie_skill3_followup_count += 1
            self._direnjie_skill3_followup_count_value = 1.0
            return 1.0
        return 0.0

    def _enemy_hero_damaged_by_main(self, enemy_hero, main_hero, main_runtime):
        if enemy_hero is None or main_runtime is None:
            return False
        for hurt in _get(enemy_hero, "take_hurt_infos", []) or []:
            atker = _get_any(hurt, ["atker", "attacker"], None)
            if atker is None or str(atker) != str(main_runtime):
                continue
            try:
                if float(_get_any(hurt, ["hurtValue", "hurt_value"], 0) or 0) > 0:
                    return True
            except (TypeError, ValueError):
                continue
        enemy_runtime = self._actor_runtime(enemy_hero)
        if enemy_runtime is not None:
            for hit in _get_any(main_hero or {}, ["hit_target_info", "hitTargetInfo"], []) or []:
                if not isinstance(hit, dict):
                    continue
                target_runtime = _get_any(hit, ["hit_target", "hitTarget"], None)
                if target_runtime is not None and str(target_runtime) == str(enemy_runtime):
                    return True
        if self.main_total_hurt_to_hero_delta > 0:
            return True
        return False

    def _detect_direnjie_skill3_miss(self, frame_no):
        self._direnjie_skill3_miss_count_value = 0.0
        if self.pending_direnjie_skill3_miss_frame is None:
            return 0.0
        cast_frame = int(self.pending_direnjie_skill3_miss_frame or 0)
        elapsed = int(frame_no or 0) - cast_frame
        if elapsed < int(GameConfig.DI_RENJIE_SKILL3_MISS_WINDOW):
            return 0.0
        self.pending_direnjie_skill3_miss_frame = None
        self._direnjie_skill3_miss_count_value = 1.0
        return 1.0

    def _compute_main_total_hurt_to_hero_delta(self, main_hero):
        if main_hero is None:
            self.prev_main_total_hurt_to_hero = None
            return 0.0
        try:
            current = float(_get_any(main_hero, ["total_hurt_to_hero", "totalHurtToHero"], 0) or 0)
        except (TypeError, ValueError):
            current = 0.0
        previous = self.prev_main_total_hurt_to_hero
        self.prev_main_total_hurt_to_hero = current
        if previous is None:
            return 0.0
        return max(0.0, current - previous)

    def _detect_cake_pickup(self, frame_data, main_hero, enemy_hero, frame_no):
        # Detect: cake disappeared this frame + main hero is close enough and
        # closer than the enemy. The event value follows hp_point's hp value
        # function, while obvious high-HP waste remains a light penalty.
        self._cake_debug = self._empty_cake_debug()
        frame_no = int(frame_no or 0)
        cakes = frame_data.get("cakes", []) or []
        current_locs = set()
        for cake in cakes:
            collider = _get(cake or {}, "collider", {}) or {}
            loc = _get(collider, "location", {}) or _get(cake or {}, "location", {}) or {}
            x = float(_get(loc, "x", 0) or 0)
            z = float(_get(loc, "z", 0) or 0)
            current_locs.add((x, z))

        disappeared = self.prev_cake_locs - current_locs
        self.prev_cake_locs = current_locs

        if main_hero is None:
            self.prev_main_hp = None
            self._pending_cake_interrupt = None
            return 0.0

        main_pos = self._entity_pos(main_hero)
        if main_pos is None:
            self.prev_main_hp = _hp(main_hero)
            self._pending_cake_interrupt = None
            return 0.0
        main_hp = float(_hp(main_hero) or 0)
        main_max_hp = max(float(_max_hp(main_hero) or 1), 1.0)
        prev_main_hp = float(self.prev_main_hp if self.prev_main_hp is not None else main_hp)
        prev_hp_rate = min(max(prev_main_hp / main_max_hp, 0.0), 1.0)
        current_hp_rate = min(max(main_hp / main_max_hp, 0.0), 1.0)
        hp_gain = max(0.0, main_hp - prev_main_hp)
        hp_gain_rate = hp_gain / main_max_hp
        hp_value_delta = max(0.0, _hp_value_rate(current_hp_rate) - _hp_value_rate(prev_hp_rate))
        self.prev_main_hp = main_hp

        enemy_pos = self._entity_pos(enemy_hero) if enemy_hero else None
        hurt_by_enemy_hero = self._hurt_by_enemy_hero(main_hero, enemy_hero)
        count = 0.0
        pickup_detected = False
        success_detected = False
        for cake_pos in disappeared:
            d_main = ((main_pos[0] - cake_pos[0]) ** 2 + (main_pos[1] - cake_pos[1]) ** 2) ** 0.5
            if d_main > GameConfig.CAKE_PICKUP_PROXIMITY:
                continue
            if enemy_pos is not None:
                d_enemy = ((enemy_pos[0] - cake_pos[0]) ** 2 + (enemy_pos[1] - cake_pos[1]) ** 2) ** 0.5
                if d_enemy < d_main:
                    continue

            if (
                prev_hp_rate >= float(GameConfig.CAKE_HIGH_HP_THRESHOLD)
                and hp_gain_rate <= float(GameConfig.CAKE_SMALL_GAIN_HP_RATIO)
            ):
                reward_value = -float(GameConfig.CAKE_WASTE_PENALTY)
                self._cake_debug["cake_wasted_count"] += 1
                self._cake_debug["cake_high_hp_penalty_count"] += 1
            elif hp_gain < float(GameConfig.CAKE_MIN_SUCCESS_HP_GAIN):
                continue
            else:
                reward_value = float(GameConfig.CAKE_RESOURCE_BONUS) + float(GameConfig.CAKE_HP_DELTA_SCALE) * hp_value_delta
                self._cake_debug["cake_success_count"] += 1
                success_detected = True
            pickup_detected = True
            count += reward_value

        if pickup_detected:
            self._pending_cake_interrupt = {"pickup_frame": frame_no} if success_detected else None
            return count

        count += self._settle_pending_cake_interrupt(frame_no, hurt_by_enemy_hero)
        return count

    def _settle_pending_cake_interrupt(self, frame_no, hurt_by_enemy_hero):
        pending = self._pending_cake_interrupt
        if pending is None:
            return 0.0
        pickup_frame = int(pending.get("pickup_frame", frame_no))
        elapsed = frame_no - pickup_frame
        if 0 <= elapsed <= int(GameConfig.CAKE_INTERRUPT_FRAMES) and hurt_by_enemy_hero:
            self._cake_debug["cake_interrupted_count"] = 1
            self._pending_cake_interrupt = None
            return -float(GameConfig.CAKE_INTERRUPT_PENALTY)
        if elapsed > int(GameConfig.CAKE_INTERRUPT_FRAMES):
            self._pending_cake_interrupt = None
        return 0.0

    def _hurt_by_enemy_hero(self, main_hero, enemy_hero):
        enemy_runtime = self._actor_runtime(enemy_hero)
        if main_hero is None or enemy_runtime is None:
            return False
        for hurt in _get_any(main_hero, ["take_hurt_infos", "takeHurtInfos"], []) or []:
            atker = _get_any(hurt, ["atker", "attacker"], None)
            if atker is None or str(atker) != str(enemy_runtime):
                continue
            try:
                hurt_value = float(_get_any(hurt, ["hurtValue", "hurt_value"], 0) or 0)
            except (TypeError, ValueError):
                hurt_value = 0.0
            if hurt_value > 0:
                return True
        return False

    def _detect_duel_summoner_timing(self, frame_data, main_hero, enemy_hero, frame_no):
        self._duel_summoner_debug = self._empty_duel_summoner_debug()
        frame_no = int(frame_no or 0)
        used_skill_id = self._used_duel_summoner_skill(main_hero)
        if used_skill_id is not None:
            self._start_duel_summoner_check(main_hero, enemy_hero, frame_no, used_skill_id)

        if self._pending_duel_summoner is None:
            return 0.0

        self._update_duel_summoner_check(main_hero, enemy_hero)
        elapsed = frame_no - int(self._pending_duel_summoner.get("start_frame", frame_no))
        if elapsed < GameConfig.DUEL_SUMMONER_WINDOW:
            return 0.0

        value = self._finish_duel_summoner_check()
        self._pending_duel_summoner = None
        return value

    def _start_duel_summoner_check(self, main_hero, enemy_hero, frame_no, skill_id):
        dist = self._distance_to_entity(main_hero, enemy_hero)
        skill_id = int(skill_id)
        self._pending_duel_summoner = {
            "skill_id": skill_id,
            "start_frame": int(frame_no or 0),
            "main_max_hp": max(float(_max_hp(main_hero) or 0), 1.0),
            "enemy_max_hp": max(float(_max_hp(enemy_hero) or 0), 1.0),
            "damage_out": 0.0,
            "damage_in": 0.0,
            "interaction_count": 0,
            "min_distance": dist,
        }
        self._duel_summoner_debug["duel_summoner_cast_count"] = 1.0
        if skill_id == 80110:
            self._duel_summoner_debug["duel_summoner_80110_cast_count"] = 1.0

    def _update_duel_summoner_check(self, main_hero, enemy_hero):
        pending = self._pending_duel_summoner
        if pending is None:
            return

        dist = self._distance_to_entity(main_hero, enemy_hero)
        if dist is not None:
            previous = pending.get("min_distance")
            pending["min_distance"] = dist if previous is None else min(float(previous), dist)

        main_runtime = self._actor_runtime(main_hero)
        enemy_runtime = self._actor_runtime(enemy_hero)
        damage_out, events_out = self._hero_damage_from_runtime(enemy_hero, main_runtime)
        damage_in, events_in = self._hero_damage_from_runtime(main_hero, enemy_runtime)
        pending["damage_out"] += damage_out
        pending["damage_in"] += damage_in
        pending["interaction_count"] += events_out + events_in

    def _finish_duel_summoner_check(self):
        pending = self._pending_duel_summoner or {}
        skill_id = int(pending.get("skill_id", 0) or 0)
        damage_out = float(pending.get("damage_out", 0.0) or 0.0)
        damage_in = float(pending.get("damage_in", 0.0) or 0.0)
        total_damage = damage_out + damage_in
        interaction_count = int(pending.get("interaction_count", 0) or 0)
        min_distance = pending.get("min_distance")
        main_max_hp = max(float(pending.get("main_max_hp", 1.0) or 1.0), 1.0)
        enemy_max_hp = max(float(pending.get("enemy_max_hp", 1.0) or 1.0), 1.0)
        max_hp = max(main_max_hp, enemy_max_hp)

        in_duel_range = min_distance is not None and float(min_distance) <= GameConfig.DUEL_SUMMONER_RANGE
        good_damage_met = (
            damage_out >= enemy_max_hp * GameConfig.DUEL_SUMMONER_GOOD_DAMAGE_HP_RATIO
            or damage_in >= main_max_hp * GameConfig.DUEL_SUMMONER_GOOD_DAMAGE_HP_RATIO
            or total_damage >= max_hp * GameConfig.DUEL_SUMMONER_GOOD_DAMAGE_HP_RATIO
        )
        good_interaction_met = interaction_count >= GameConfig.DUEL_SUMMONER_GOOD_INTERACTION_COUNT

        if in_duel_range and good_damage_met and good_interaction_met:
            self._duel_summoner_debug["duel_summoner_good_count"] = 1.0
            if skill_id == 80110:
                self._duel_summoner_debug["duel_summoner_80110_good_count"] = 1.0
            return GameConfig.DUEL_SUMMONER_GOOD_REWARD

        mid_damage_met = (
            damage_out >= enemy_max_hp * GameConfig.DUEL_SUMMONER_MID_DAMAGE_HP_RATIO
            or damage_in >= main_max_hp * GameConfig.DUEL_SUMMONER_MID_DAMAGE_HP_RATIO
            or total_damage >= max_hp * GameConfig.DUEL_SUMMONER_MID_DAMAGE_HP_RATIO
        )
        mid_interaction_met = interaction_count >= GameConfig.DUEL_SUMMONER_MID_INTERACTION_COUNT
        if in_duel_range and (mid_damage_met or mid_interaction_met):
            self._duel_summoner_debug["duel_summoner_mid_count"] = 1.0
            return GameConfig.DUEL_SUMMONER_MID_REWARD

        if in_duel_range and (interaction_count > 0 or total_damage > 0):
            self._duel_summoner_debug["duel_summoner_touch_count"] = 1.0
            return GameConfig.DUEL_SUMMONER_TOUCH_REWARD

        if interaction_count > 0 or total_damage > 0:
            self._duel_summoner_debug["duel_summoner_poke_count"] = 1.0
            return 0.0

        if not in_duel_range:
            self._duel_summoner_debug["duel_summoner_wasted_count"] = 1.0
            return GameConfig.DUEL_SUMMONER_WASTED_REWARD
        return 0.0

    def _hero_damage_from_runtime(self, victim, attacker_runtime):
        if victim is None or attacker_runtime is None:
            return 0.0, 0
        damage = 0.0
        event_count = 0
        for hurt in _get_any(victim, ["take_hurt_infos", "takeHurtInfos"], []) or []:
            atker = _get_any(hurt, ["atker", "attacker"], None)
            if atker is None or str(atker) != str(attacker_runtime):
                continue
            try:
                hurt_value = float(_get_any(hurt, ["hurtValue", "hurt_value"], 0) or 0)
            except (TypeError, ValueError):
                continue
            if hurt_value <= 0:
                continue
            damage += hurt_value
            event_count += 1
        return damage, event_count

    def _used_duel_summoner_skill(self, hero):
        if hero is None:
            return None
        for skill_id in GameConfig.DUEL_SUMMONER_SKILL_IDS:
            if self._used_skill_by_config(hero, skill_id):
                return int(skill_id)
        return None

    def _unit_damaged_by_runtime(self, unit, runtime):
        for hurt in _get_any(unit or {}, ["take_hurt_infos", "takeHurtInfos"], []) or []:
            attacker = _get_any(hurt, ["atker", "attacker"], None)
            if attacker is None or runtime is None or str(attacker) != str(runtime):
                continue
            try:
                if float(_get_any(hurt, ["hurtValue", "hurt_value"], 0) or 0) > 0:
                    return True
            except (TypeError, ValueError):
                continue
        return False

    def _detect_early_own_half_hero_offense(self, frame_data, main_hero, enemy_hero, frame_no):
        self._early_own_half_offense_debug = self._empty_early_own_half_offense_debug()
        frame_no = int(frame_no or 0)
        self._update_opening_contact_exit_frame(frame_data, main_hero, enemy_hero, frame_no)

        if self._opening_contact_exit_frame is None:
            return 0.0
        if not (int(self._opening_contact_exit_frame) <= frame_no < int(GameConfig.OPENING_FORWARD_END_FRAME)):
            return 0.0
        if main_hero is None or enemy_hero is None:
            return 0.0
        if float(_hp(main_hero) or 0) <= 0 or float(_hp(enemy_hero) or 0) <= 0:
            return 0.0
        if not self._enemy_hero_visible_to_main(frame_data, main_hero):
            return 0.0

        main_camp = _camp_key(_get(main_hero or {}, "camp", self.main_hero_camp))
        enemy_lane = self._lane_for_camp(enemy_hero, main_camp)
        if enemy_lane is None or float(enemy_lane) >= float(GameConfig.EARLY_OFFENSE_ENEMY_LANE_MAX):
            return 0.0

        reward_value = 0.0
        if self._last_action_attacks_enemy_hero():
            reward_value += self._grant_early_offense_intent()

        damage = self._early_offense_damage_amount(main_hero, enemy_hero)
        if damage > 0.0:
            reward_value += self._grant_early_offense_damage(damage, enemy_hero)

        remaining_total = max(0.0, float(GameConfig.EARLY_OFFENSE_TOTAL_CAP) - self._early_own_half_offense_total)
        reward_value = min(reward_value, remaining_total)
        self._early_own_half_offense_total += reward_value
        return reward_value

    def _update_opening_contact_exit_frame(self, frame_data, main_hero, enemy_hero, frame_no):
        if self._opening_contact_exit_frame is not None:
            return
        if frame_no <= int(GameConfig.OPENING_WAVE_GUARD_CONTACT_EXIT_FRAME):
            return
        if frame_no >= int(GameConfig.OPENING_FORWARD_END_FRAME):
            return
        if main_hero is None:
            return
        if self._enemy_hero_visible_to_main(frame_data, main_hero) or self._enemy_minion_visible_to_main(
            frame_data, main_hero
        ):
            self._opening_contact_exit_frame = frame_no

    def _last_action_attacks_enemy_hero(self):
        action = self.last_action_context or {}
        button = int(action.get("button", -1))
        target = int(action.get("target", -1))
        return button in (3, 4, 5, 6, 10) and target == 1

    def _grant_early_offense_intent(self):
        remaining = max(0.0, float(GameConfig.EARLY_OFFENSE_INTENT_CAP) - self._early_own_half_offense_intent_total)
        value = min(float(GameConfig.EARLY_OFFENSE_INTENT_REWARD), remaining)
        if value > 0.0:
            self._early_own_half_offense_intent_total += value
            self._early_own_half_offense_debug["early_own_half_offense_intent_count"] = 1.0
        return value

    def _early_offense_damage_amount(self, main_hero, enemy_hero):
        main_runtime = self._actor_runtime(main_hero)
        direct_damage, _ = self._hero_damage_from_runtime(enemy_hero, main_runtime)
        if direct_damage > 0.0:
            return direct_damage
        return max(0.0, float(self.main_total_hurt_to_hero_delta or 0.0))

    def _grant_early_offense_damage(self, damage, enemy_hero):
        enemy_max_hp = max(float(_max_hp(enemy_hero) or 1.0), 1.0)
        raw_value = min(
            float(damage) / enemy_max_hp * float(GameConfig.EARLY_OFFENSE_DAMAGE_SCALE),
            float(GameConfig.EARLY_OFFENSE_DAMAGE_REWARD_CAP),
        )
        remaining = max(0.0, float(GameConfig.EARLY_OFFENSE_DAMAGE_CAP) - self._early_own_half_offense_damage_total)
        value = min(raw_value, remaining)
        if value > 0.0:
            self._early_own_half_offense_damage_total += value
            self._early_own_half_offense_debug["early_own_half_offense_damage_count"] = 1.0
            self._early_own_half_offense_debug["early_own_half_offense_damage_value"] = float(damage)
        return value

    def _detect_opening_trade_pressure(self, frame_data, main_hero, enemy_hero, frame_no):
        self._opening_trade_debug = self._empty_opening_trade_debug()
        frame_no = int(frame_no or 0)
        value = 0.0

        if self._pending_opening_trade is not None:
            self._update_opening_trade_check(main_hero, enemy_hero)
            if self._opening_trade_should_finish(frame_data, main_hero, enemy_hero, frame_no):
                value += self._finish_opening_trade_check(frame_data, main_hero, enemy_hero)
                self._pending_opening_trade = None

        if not self._opening_trade_context(frame_data, main_hero, enemy_hero, frame_no):
            self._opening_trade_debug["opening_trade_reward_value"] = value
            return value

        used_skill_id = self._used_duel_summoner_skill(main_hero)
        distance = self._distance_to_entity(main_hero, enemy_hero)
        in_range = distance is not None and float(distance) < float(GameConfig.OPENING_BERSERK_ENEMY_DISTANCE)
        attacks_enemy_hero = self._last_action_attacks_enemy_hero()
        should_start = in_range or attacks_enemy_hero or used_skill_id == GameConfig.BERSERK_SKILL_ID
        started_trade = False
        if should_start and self._pending_opening_trade is None:
            self._start_opening_trade_check(main_hero, enemy_hero, frame_no, used_skill_id)
            started_trade = True
        if started_trade:
            self._update_opening_trade_check(main_hero, enemy_hero)

        if in_range:
            value += self._grant_opening_trade_visible()
        if attacks_enemy_hero:
            value += self._grant_opening_trade_intent()

        damage_out, _, _, _ = self._opening_trade_damage_pair(main_hero, enemy_hero)
        if damage_out > 0.0:
            value += self._grant_opening_trade_damage(damage_out, enemy_hero)

        self._opening_trade_debug["opening_trade_reward_value"] = value
        return value

    def _opening_trade_context(self, frame_data, main_hero, enemy_hero, frame_no):
        if main_hero is None or enemy_hero is None:
            return False
        if float(_hp(main_hero) or 0) <= 0 or float(_hp(enemy_hero) or 0) <= 0:
            return False
        if not self._enemy_hero_visible_to_main(frame_data, main_hero):
            return False
        if int(frame_no or 0) < self._opening_air_attack_follow_frame(main_hero):
            return False
        if int(frame_no or 0) >= int(GameConfig.OPENING_FORWARD_END_FRAME):
            return False
        main_camp = _camp_key(_get(main_hero or {}, "camp", self.main_hero_camp))
        enemy_lane = self._lane_for_camp(enemy_hero, main_camp)
        return enemy_lane is not None and float(enemy_lane) < float(GameConfig.OPENING_TRADE_ENEMY_LANE_MAX)

    def _start_opening_trade_check(self, main_hero, enemy_hero, frame_no, skill_id):
        self._pending_opening_trade = {
            "start_frame": int(frame_no or 0),
            "main_max_hp": max(float(_max_hp(main_hero) or 0), 1.0),
            "enemy_max_hp": max(float(_max_hp(enemy_hero) or 0), 1.0),
            "damage_out": 0.0,
            "damage_in": 0.0,
            "interaction_count": 0,
            "min_distance": self._distance_to_entity(main_hero, enemy_hero),
            "used_80110": int(skill_id or 0) == int(GameConfig.BERSERK_SKILL_ID),
        }
        self._opening_trade_debug["opening_trade_start_count"] = 1.0

    def _update_opening_trade_check(self, main_hero, enemy_hero):
        pending = self._pending_opening_trade
        if pending is None:
            return

        distance = self._distance_to_entity(main_hero, enemy_hero)
        if distance is not None:
            previous = pending.get("min_distance")
            pending["min_distance"] = distance if previous is None else min(float(previous), distance)
        if self._used_duel_summoner_skill(main_hero) == GameConfig.BERSERK_SKILL_ID:
            pending["used_80110"] = True

        damage_out, damage_in, events_out, events_in = self._opening_trade_damage_pair(main_hero, enemy_hero)
        pending["damage_out"] += damage_out
        pending["damage_in"] += damage_in
        pending["interaction_count"] += events_out + events_in

    def _opening_trade_damage_pair(self, main_hero, enemy_hero):
        main_runtime = self._actor_runtime(main_hero)
        enemy_runtime = self._actor_runtime(enemy_hero)
        damage_out, events_out = self._hero_damage_from_runtime(enemy_hero, main_runtime)
        damage_in, events_in = self._hero_damage_from_runtime(main_hero, enemy_runtime)
        if damage_out <= 0.0 and self.main_total_hurt_to_hero_delta > 0:
            damage_out = float(self.main_total_hurt_to_hero_delta or 0.0)
            events_out = 1
        return damage_out, damage_in, events_out, events_in

    def _opening_trade_should_finish(self, frame_data, main_hero, enemy_hero, frame_no):
        pending = self._pending_opening_trade or {}
        elapsed = int(frame_no or 0) - int(pending.get("start_frame", frame_no))
        if elapsed >= int(GameConfig.OPENING_TRADE_WINDOW_FRAMES):
            return True
        if main_hero is not None and float(_hp(main_hero) or 0) <= 0:
            return True
        if enemy_hero is not None and float(_hp(enemy_hero) or 0) <= 0:
            return True
        return self._frame_action_has_enemy_hero_death(frame_data, main_hero, enemy_hero)

    def _finish_opening_trade_check(self, frame_data, main_hero, enemy_hero):
        pending = self._pending_opening_trade or {}
        damage_out = float(pending.get("damage_out", 0.0) or 0.0)
        damage_in = float(pending.get("damage_in", 0.0) or 0.0)
        interaction_count = int(pending.get("interaction_count", 0) or 0)
        min_distance = pending.get("min_distance")
        main_max_hp = max(float(pending.get("main_max_hp", 1.0) or 1.0), 1.0)
        enemy_max_hp = max(float(pending.get("enemy_max_hp", 1.0) or 1.0), 1.0)
        in_range = min_distance is not None and float(min_distance) < float(GameConfig.OPENING_BERSERK_ENEMY_DISTANCE)
        enemy_dead = (
            enemy_hero is not None and float(_hp(enemy_hero) or 0) <= 0
        ) or self._frame_action_has_enemy_hero_death(frame_data, main_hero, enemy_hero)

        value = 0.0
        had_interaction = interaction_count > 0 or damage_out > 0.0 or damage_in > 0.0
        if bool(pending.get("used_80110", False)) and had_interaction:
            value += self._cap_opening_trade_reward(float(GameConfig.OPENING_TRADE_80110_INTERACTION_REWARD))
            self._opening_trade_debug["opening_trade_80110_good_count"] = 1.0

        strong_damage_met = damage_out >= enemy_max_hp * float(GameConfig.OPENING_TRADE_STRONG_DAMAGE_HP_RATIO)
        good_damage_met = damage_out >= enemy_max_hp * float(GameConfig.OPENING_TRADE_GOOD_DAMAGE_HP_RATIO)
        good_interaction_met = interaction_count >= int(GameConfig.OPENING_TRADE_GOOD_INTERACTION_COUNT)
        bad_trade = (
            damage_in >= main_max_hp * float(GameConfig.OPENING_TRADE_BAD_DAMAGE_IN_HP_RATIO)
            and damage_in > damage_out * float(GameConfig.OPENING_TRADE_BAD_DAMAGE_IN_RATIO)
        )

        if enemy_dead or strong_damage_met:
            value += self._cap_opening_trade_reward(float(GameConfig.OPENING_TRADE_STRONG_REWARD))
            self._opening_trade_debug["opening_trade_strong_count"] = 1.0
        elif in_range and good_damage_met and good_interaction_met:
            value += self._cap_opening_trade_reward(float(GameConfig.OPENING_TRADE_GOOD_REWARD))
            self._opening_trade_debug["opening_trade_good_count"] = 1.0
        elif bad_trade:
            value += float(GameConfig.OPENING_TRADE_BAD_REWARD)
            self._opening_trade_debug["opening_trade_bad_count"] = 1.0
        elif bool(pending.get("used_80110", False)) and not had_interaction:
            value += float(GameConfig.OPENING_TRADE_WASTED_80110_REWARD)
            self._opening_trade_debug["opening_trade_wasted_80110_count"] = 1.0
        return value

    def _grant_opening_trade_visible(self):
        remaining = max(0.0, float(GameConfig.OPENING_TRADE_VISIBLE_CAP) - self._opening_trade_visible_total)
        value = min(float(GameConfig.OPENING_TRADE_VISIBLE_REWARD), remaining)
        value = self._cap_opening_trade_reward(value)
        if value > 0.0:
            self._opening_trade_visible_total += value
            self._opening_trade_debug["opening_trade_visible_count"] = 1.0
        return value

    def _grant_opening_trade_intent(self):
        remaining = max(0.0, float(GameConfig.OPENING_TRADE_INTENT_CAP) - self._opening_trade_intent_total)
        value = min(float(GameConfig.OPENING_TRADE_INTENT_REWARD), remaining)
        value = self._cap_opening_trade_reward(value)
        if value > 0.0:
            self._opening_trade_intent_total += value
            self._opening_trade_debug["opening_trade_intent_count"] = 1.0
        return value

    def _grant_opening_trade_damage(self, damage, enemy_hero):
        enemy_max_hp = max(float(_max_hp(enemy_hero) or 1.0), 1.0)
        raw_value = min(
            float(damage) / enemy_max_hp * float(GameConfig.OPENING_TRADE_DAMAGE_SCALE),
            float(GameConfig.OPENING_TRADE_DAMAGE_REWARD_CAP),
        )
        remaining_damage = max(0.0, float(GameConfig.OPENING_TRADE_DAMAGE_CAP) - self._opening_trade_damage_total)
        value = min(raw_value, remaining_damage)
        value = self._cap_opening_trade_reward(value)
        if value > 0.0:
            self._opening_trade_damage_total += value
            self._opening_trade_debug["opening_trade_damage_count"] = 1.0
        return value

    def _cap_opening_trade_reward(self, value):
        value = max(float(value or 0.0), 0.0)
        remaining_total = max(0.0, float(GameConfig.OPENING_TRADE_TOTAL_CAP) - self._opening_trade_total)
        value = min(value, remaining_total)
        self._opening_trade_total += value
        return value

    def _detect_minion_tower_push(self, frame_data, main_hero, enemy_hero, enemy_tower, own_soldier_count):
        frame_no = int(frame_data.get("frame_no", frame_data.get("frameNo", 0)) or 0)
        current_hp = float(_hp(enemy_tower or {}) or 0)
        current_dist = self._distance_to_enemy_tower(main_hero, enemy_tower)
        prev_hp = self._prev_enemy_tower_hp_for_push
        prev_dist = self._prev_main_enemy_tower_dist
        self._prev_enemy_tower_hp_for_push = current_hp if enemy_tower is not None else None
        self._prev_main_enemy_tower_dist = current_dist

        if frame_no - self._last_tower_push_eval_frame < GameConfig.TOWER_PUSH_EVAL_INTERVAL:
            return 0.0
        if not self._tower_push_context(main_hero, enemy_hero, enemy_tower, own_soldier_count):
            return 0.0

        self._last_tower_push_eval_frame = frame_no
        if self._is_actively_pushing_tower(frame_data, main_hero, enemy_tower, current_hp, prev_hp):
            return GameConfig.TOWER_PUSH_REWARD
        if (
            current_dist is not None
            and prev_dist is not None
            and current_dist < prev_dist - GameConfig.TOWER_PUSH_APPROACH_DELTA
        ):
            return GameConfig.TOWER_PUSH_APPROACH_REWARD
        return GameConfig.TOWER_PUSH_IDLE_PENALTY

    def _detect_enemy_dead_enemy_cake(self, frame_data, main_hero, enemy_hero, enemy_tower, own_soldier_count):
        frame_no = int(frame_data.get("frame_no", frame_data.get("frameNo", 0)) or 0)
        enemy_cake_pos = self._enemy_cake_pos(frame_data, main_hero, enemy_tower)
        main_pos = self._entity_pos(main_hero)
        current_dist = math.dist(main_pos, enemy_cake_pos) if main_pos is not None and enemy_cake_pos is not None else None
        prev_dist = self._prev_enemy_cake_dist
        self._prev_enemy_cake_dist = current_dist

        if frame_no - self._last_enemy_dead_cake_eval_frame < GameConfig.ENEMY_DEAD_CAKE_EVAL_INTERVAL:
            return 0.0
        if frame_no >= GameConfig.CANNON_FRAME:
            return 0.0
        if not self._is_enemy_dead_for_reward(frame_data, main_hero, enemy_hero):
            return 0.0
        if not self._enemy_tower_has_safe_soldier_cover(frame_data, main_hero, enemy_tower, own_soldier_count):
            return 0.0
        if enemy_cake_pos is None or current_dist is None:
            return 0.0
        if enemy_tower is None or float(_hp(enemy_tower) or 0) <= 0:
            return 0.0
        if main_hero is None or float(_hp(main_hero) or 0) <= 0:
            return 0.0

        self._last_enemy_dead_cake_eval_frame = frame_no
        if current_dist <= GameConfig.ENEMY_DEAD_CAKE_NEAR_RANGE:
            return GameConfig.ENEMY_DEAD_CAKE_NEAR_REWARD
        if (
            prev_dist is not None
            and current_dist < prev_dist - GameConfig.ENEMY_DEAD_CAKE_APPROACH_DELTA
        ):
            return GameConfig.ENEMY_DEAD_CAKE_APPROACH_REWARD
        if current_dist <= GameConfig.ENEMY_DEAD_CAKE_ZONE_RANGE:
            return GameConfig.ENEMY_DEAD_CAKE_ZONE_REWARD
        return 0.0

    def _detect_enemy_minion_own_tower_pressure(self, frame_data, main_hero, main_tower):
        self._enemy_minion_defense_debug = self._empty_enemy_minion_defense_debug()
        frame_no = int(frame_data.get("frame_no", frame_data.get("frameNo", 0)) or 0)
        if frame_no < GameConfig.CANNON_FRAME:
            return 0.0, 0.0
        if frame_no - self._last_enemy_minion_defense_eval_frame < GameConfig.ENEMY_MINION_DEFENSE_EVAL_INTERVAL:
            return 0.0, 0.0
        if main_hero is None or main_tower is None:
            return 0.0, 0.0
        if float(_hp(main_hero) or 0) <= 0 or float(_hp(main_tower) or 0) <= 0:
            return 0.0, 0.0

        tower_pos = self._entity_pos(main_tower)
        if tower_pos is None:
            return 0.0, 0.0
        tower_range = self._tower_attack_range(main_tower)
        front_range = tower_range * GameConfig.ENEMY_MINION_TOWER_FRONT_RANGE_MULTIPLIER
        main_camp = _camp_key(_get(main_hero, "camp", None))
        front_count = 0
        under_count = 0
        for npc in frame_data.get("npc_states", []) or []:
            if _get_any(npc, ["sub_type", "subType"], None) not in SOLDIER_SUB_TYPES:
                continue
            if _camp_key(_get(npc, "camp", None)) == main_camp:
                continue
            if float(_hp(npc) or 0) <= 0:
                continue
            pos = self._entity_pos(npc)
            if pos is None:
                continue
            dist = math.dist(pos, tower_pos)
            if dist <= tower_range:
                under_count += 1
            elif dist <= front_range:
                front_count += 1

        if front_count <= 0 and under_count <= 0:
            return 0.0, 0.0

        self._last_enemy_minion_defense_eval_frame = frame_no
        multiplier = self._enemy_minion_defense_multiplier(main_tower)
        front_value = self._capped_count_reward(
            front_count,
            GameConfig.ENEMY_MINION_TOWER_FRONT_REWARD,
            GameConfig.ENEMY_MINION_TOWER_FRONT_CAP,
        ) * multiplier
        under_value = self._capped_count_reward(
            under_count,
            GameConfig.ENEMY_MINION_UNDER_OWN_TOWER_REWARD,
            GameConfig.ENEMY_MINION_UNDER_OWN_TOWER_CAP,
        ) * multiplier
        self._enemy_minion_defense_debug = {
            "enemy_minion_tower_front_count": float(front_count),
            "enemy_minion_under_own_tower_count": float(under_count),
            "enemy_minion_defense_event_count": 1.0,
            "enemy_minion_defense_multiplier": float(multiplier),
        }
        return front_value, under_value

    def _detect_river_crab_pressure(
        self, frame_data, main_hero, enemy_hero, main_tower, enemy_tower, own_soldier_count
    ):
        self._river_crab_pressure_debug = self._empty_river_crab_pressure_debug()
        frame_no = int(frame_data.get("frame_no", frame_data.get("frameNo", 0)) or 0)
        current_hp = self._river_crab_hp_map(frame_data)
        try:
            if not self.has_last_frame:
                self.prev_river_crab_hp = current_hp
                return 0.0
            if self._river_crab_pressure_total >= float(GameConfig.RIVER_CRAB_PRESSURE_CAP):
                self.prev_river_crab_hp = current_hp
                return 0.0
            if frame_no - self._last_river_crab_pressure_frame < int(GameConfig.RIVER_CRAB_PRESSURE_EVAL_INTERVAL):
                self.prev_river_crab_hp = current_hp
                return 0.0
            if not self._river_crab_safe_context(
                frame_data, main_hero, enemy_hero, main_tower, enemy_tower, own_soldier_count
            ):
                self.prev_river_crab_hp = current_hp
                return 0.0

            main_runtime = self._actor_runtime(main_hero)
            if main_runtime is None:
                self.prev_river_crab_hp = current_hp
                return 0.0
            main_target = _get_any(main_hero or {}, ["attack_target", "attackTarget"], None)
            for runtime, info in current_hp.items():
                prev_info = self.prev_river_crab_hp.get(runtime, {})
                prev_hp = prev_info.get("hp") if isinstance(prev_info, dict) else None
                hp = float(info.get("hp", 0.0) or 0.0)
                hp_drop = prev_hp is not None and hp < float(prev_hp)
                direct_damage = self._unit_damaged_by_runtime(info.get("unit"), main_runtime)
                target_damage = hp_drop and main_target is not None and str(main_target) == str(runtime)
                if not (direct_damage or target_damage):
                    continue
                value = min(
                    float(GameConfig.RIVER_CRAB_PRESSURE_REWARD),
                    float(GameConfig.RIVER_CRAB_PRESSURE_CAP) - float(self._river_crab_pressure_total),
                )
                if value <= 0.0:
                    break
                self._river_crab_pressure_total += value
                self._last_river_crab_pressure_frame = frame_no
                self._river_crab_pressure_debug["river_crab_pressure_count"] = 1.0
                return value
            return 0.0
        finally:
            self.prev_river_crab_hp = current_hp

    def _river_crab_safe_context(
        self, frame_data, main_hero, enemy_hero, main_tower, enemy_tower, own_soldier_count
    ):
        if main_hero is None or main_tower is None or enemy_tower is None:
            return False
        if float(_hp(main_hero) or 0) <= 0:
            return False
        if own_soldier_count > 0:
            return False
        if not self._enemy_hero_far_from_main(main_hero, enemy_hero):
            return False
        return not self._has_enemy_soldier_between_towers(frame_data, main_hero, main_tower, enemy_tower)

    def _river_crab_hp_map(self, frame_data):
        current = {}
        for npc in frame_data.get("npc_states", []) or []:
            if not self._is_river_crab(npc):
                continue
            runtime = self._actor_runtime(npc)
            if runtime is None:
                continue
            current[runtime] = {
                "hp": float(_hp(npc) or 0.0),
                "unit": npc,
            }
        return current

    def _is_river_crab(self, unit):
        try:
            config_id = int(_get_any(unit or {}, ["config_id", "configId"], 0) or 0)
        except (TypeError, ValueError):
            return False
        return config_id in set(int(value) for value in GameConfig.RIVER_CRAB_CONFIG_IDS)

    def _enemy_hero_far_from_main(self, main_hero, enemy_hero):
        if enemy_hero is None or float(_hp(enemy_hero) or 0) <= 0:
            return True
        dist = self._distance_to_entity(main_hero, enemy_hero)
        return dist is not None and dist > float(GameConfig.RIVER_CRAB_HERO_SAFE_RANGE)

    def _has_enemy_soldier_between_towers(self, frame_data, main_hero, main_tower, enemy_tower):
        main_camp = _camp_key(_get(main_hero or {}, "camp", None))
        main_tower_lane = self._lane_for_camp(main_tower, main_camp)
        enemy_tower_lane = self._lane_for_camp(enemy_tower, main_camp)
        if main_tower_lane is None or enemy_tower_lane is None:
            return True
        lo, hi = sorted((main_tower_lane, enemy_tower_lane))
        for npc in frame_data.get("npc_states", []) or []:
            if _get_any(npc, ["sub_type", "subType"], None) not in SOLDIER_SUB_TYPES:
                continue
            if _camp_key(_get(npc, "camp", None)) == main_camp:
                continue
            if float(_hp(npc) or 0) <= 0:
                continue
            lane = self._lane_for_camp(npc, main_camp)
            if lane is None:
                return True
            if lo <= lane <= hi:
                return True
        return False

    def _lane_for_camp(self, unit, camp):
        pos = self._entity_pos(unit)
        if pos is None:
            return None
        x, z = pos
        if _camp_key(camp) == 2:
            x, z = -x, -z
        return (x + z) / Args.SQRT2

    def _enemy_minion_defense_multiplier(self, main_tower):
        hp_rate = _hp_rate(main_tower or {})
        if 0.0 < hp_rate < GameConfig.ENEMY_MINION_DEFENSE_TOWER_HP_LOW:
            return GameConfig.ENEMY_MINION_DEFENSE_LOW_MULTIPLIER
        if 0.0 < hp_rate < GameConfig.ENEMY_MINION_DEFENSE_TOWER_HP_MID:
            return GameConfig.ENEMY_MINION_DEFENSE_MID_MULTIPLIER
        return 1.0

    def _capped_count_reward(self, count, reward_per_unit, reward_cap):
        capped_count = min(max(int(count or 0), 0), GameConfig.ENEMY_MINION_DEFENSE_COUNT_CAP)
        value = float(reward_per_unit) * capped_count
        return max(float(reward_cap), value)

    def _enemy_tower_has_safe_soldier_cover(self, frame_data, main_hero, enemy_tower, own_soldier_count):
        if own_soldier_count < GameConfig.ENEMY_DEAD_CAKE_MIN_SOLDIERS:
            return False
        return self._tower_target_is_non_hero(frame_data, enemy_tower)

    def _tower_target_is_non_hero(self, frame_data, tower):
        target_runtime = _get(tower or {}, "attack_target", None)
        if target_runtime is None:
            return False
        for hero in frame_data.get("hero_states", []) or []:
            if self._actor_runtime(hero) == target_runtime and float(_hp(hero) or 0) > 0:
                return False
        return True

    def _tower_push_context(self, main_hero, enemy_hero, enemy_tower, own_soldier_count):
        if main_hero is None or enemy_tower is None:
            return False
        if float(_hp(main_hero) or 0) <= 0:
            return False
        if float(_hp(enemy_tower) or 0) <= 0:
            return False
        if own_soldier_count < GameConfig.TOWER_PUSH_MIN_SOLDIERS:
            return False
        if self._tower_targets_main_hero(main_hero, enemy_tower):
            return False
        return self._is_enemy_dead(enemy_hero) or self._is_enemy_far_from_tower(enemy_hero, enemy_tower)

    def _is_actively_pushing_tower(self, frame_data, main_hero, enemy_tower, current_hp, prev_hp):
        tower_runtime = self._actor_runtime(enemy_tower)
        if tower_runtime is not None and _get(main_hero or {}, "attack_target", None) == tower_runtime:
            return True
        if prev_hp is not None and current_hp < prev_hp:
            return True
        main_pos = self._entity_pos(main_hero)
        tower_pos = self._entity_pos(enemy_tower)
        if main_pos is None or tower_pos is None:
            return False
        attack_range = float(
            _get(main_hero or {}, "attack_range", GameConfig.TOWER_PUSH_HERO_ATTACK_RANGE_FALLBACK)
            or GameConfig.TOWER_PUSH_HERO_ATTACK_RANGE_FALLBACK
        )
        if math.dist(main_pos, tower_pos) > attack_range:
            return False
        return self._tower_targets_own_soldier(frame_data, main_hero, enemy_tower)

    def _tower_targets_main_hero(self, main_hero, tower):
        main_runtime = self._actor_runtime(main_hero)
        target_runtime = _get(tower or {}, "attack_target", None)
        return main_runtime is not None and target_runtime is not None and str(main_runtime) == str(target_runtime)

    def _own_soldiers_in_enemy_tower_range(self, frame_data, main_hero, enemy_tower):
        if main_hero is None or enemy_tower is None:
            return 0
        main_camp = _camp_key(_get(main_hero, "camp", None))
        tower_pos = self._entity_pos(enemy_tower)
        if tower_pos is None:
            return 0
        tower_range = self._tower_attack_range(enemy_tower)
        count = 0
        for npc in frame_data.get("npc_states", []) or []:
            if _get_any(npc, ["sub_type", "subType"], None) not in SOLDIER_SUB_TYPES:
                continue
            if _camp_key(_get(npc, "camp", None)) != main_camp:
                continue
            if float(_hp(npc) or 0) <= 0:
                continue
            pos = self._entity_pos(npc)
            if pos is not None and math.dist(pos, tower_pos) <= tower_range:
                count += 1
        return count

    def _tower_targets_own_soldier(self, frame_data, main_hero, tower):
        target_runtime = _get(tower or {}, "attack_target", None)
        if target_runtime is None:
            return False
        main_camp = _camp_key(_get(main_hero or {}, "camp", None))
        for npc in frame_data.get("npc_states", []) or []:
            if self._actor_runtime(npc) != target_runtime:
                continue
            if _get_any(npc, ["sub_type", "subType"], None) not in SOLDIER_SUB_TYPES:
                return False
            return _camp_key(_get(npc, "camp", None)) == main_camp
        return False

    def _enemy_cake(self, frame_data, enemy_tower):
        cakes = frame_data.get("cakes", []) or []
        if not cakes:
            return None
        tower_pos = self._entity_pos(enemy_tower)
        if tower_pos is None:
            return cakes[0]
        cakes_with_dist = []
        for cake in cakes:
            pos = self._entity_pos(cake)
            if pos is not None:
                cakes_with_dist.append((math.dist(pos, tower_pos), cake))
        return sorted(cakes_with_dist, key=lambda item: item[0])[0][1] if cakes_with_dist else None

    def _enemy_cake_pos(self, frame_data, main_hero, enemy_tower):
        enemy_cake = self._enemy_cake(frame_data, enemy_tower)
        pos = self._entity_pos(enemy_cake)
        if pos is not None:
            return pos
        return self._enemy_cake_anchor_raw(main_hero)

    def _enemy_cake_anchor_raw(self, main_hero):
        main_camp = _camp_key(_get(main_hero or {}, "camp", None))
        anchor = Args.CAKE_ANCHORS_BY_CAMP.get(main_camp, {}).get("enemy")
        if anchor is None:
            return None
        lane, width = anchor
        sqrt2 = math.sqrt(2.0)
        x = (float(lane) + float(width)) / sqrt2
        z = (float(lane) - float(width)) / sqrt2
        if main_camp == 2:
            x, z = -x, -z
        return x, z

    def _distance_to_enemy_tower(self, main_hero, enemy_tower):
        return self._distance_to_entity(main_hero, enemy_tower)

    def _distance_to_entity(self, source, target):
        source_pos = self._entity_pos(source)
        target_pos = self._entity_pos(target)
        if source_pos is None or target_pos is None:
            return None
        return math.dist(source_pos, target_pos)

    def _is_enemy_dead(self, enemy_hero):
        return enemy_hero is not None and float(_hp(enemy_hero) or 0) <= 0

    def _is_enemy_dead_for_reward(self, frame_data, main_hero, enemy_hero):
        frame_no = int(_get_any(frame_data or {}, ["frame_no", "frameNo"], 0) or 0)
        if self._frame_action_has_enemy_hero_death(frame_data, main_hero, enemy_hero):
            self._last_enemy_hero_dead_frame_for_reward = frame_no
            return True
        if enemy_hero is None:
            return self._in_recent_enemy_dead_window(frame_no)

        dead_cnt = int(_get_any(enemy_hero, ["dead_cnt", "deadCnt"], 0) or 0)
        dead_cnt_increased = dead_cnt > int(self._prev_enemy_dead_cnt_for_reward or 0)
        self._prev_enemy_dead_cnt_for_reward = dead_cnt
        revive_time = int(_get_any(enemy_hero, ["revive_time", "reviveTime"], 0) or 0)
        enemy_dead = dead_cnt_increased or revive_time > 0 or float(_hp(enemy_hero) or 0) <= 0
        if enemy_dead:
            self._last_enemy_hero_dead_frame_for_reward = frame_no
            return True
        return False

    def _in_recent_enemy_dead_window(self, frame_no):
        last_dead_frame = int(self._last_enemy_hero_dead_frame_for_reward or -10000)
        return 0 <= int(frame_no or 0) - last_dead_frame <= int(GameConfig.FORCE_HOME_POST_KILL_WINDOW_FRAMES)

    def _frame_action_has_enemy_hero_death(self, frame_data, main_hero, enemy_hero):
        frame_action = _get_any(frame_data or {}, ["frame_action", "frameAction"], {}) or {}
        dead_actions = _get_any(frame_action, ["dead_action", "deadAction"], []) or []
        if not isinstance(dead_actions, list):
            return False
        enemy_runtime = self._actor_runtime(enemy_hero)
        main_camp = _camp_key(_get(main_hero or {}, "camp", None))
        for dead_action in dead_actions:
            if not isinstance(dead_action, dict):
                continue
            death = _get_any(dead_action, ["death", "dead"], {}) or {}
            death_runtime = self._actor_runtime(death)
            if enemy_runtime is not None and death_runtime == enemy_runtime:
                return True
            death_camp = _camp_key(_get(death, "camp", None))
            death_config_id = int(_get_any(death, ["config_id", "configId"], 0) or 0)
            if main_camp in (1, 2) and death_camp in (1, 2) and death_camp != main_camp:
                if death_config_id in GameConfig.HERO_IDS:
                    return True
        return False

    def _is_enemy_far_from_tower(self, enemy_hero, enemy_tower):
        dist = self._distance_to_entity(enemy_hero, enemy_tower)
        return dist is not None and dist >= GameConfig.TOWER_PUSH_ENEMY_FAR_RANGE

    def _tower_attack_range(self, tower):
        return float(
            _get(tower or {}, "attack_range", GameConfig.TOWER_PUSH_TOWER_RANGE_FALLBACK)
            or GameConfig.TOWER_PUSH_TOWER_RANGE_FALLBACK
        )

    def _used_skill_by_config(self, hero, config_id):
        if hero is None:
            return False
        skill_state = _get_any(hero or {}, ["skill_state", "skillState"], {}) or {}
        slots = _get_any(skill_state, ["slot_states", "slotStates"], []) or []
        for slot in slots:
            slot_config = _get_any(slot, ["config_id", "configId"], 0)
            try:
                if int(slot_config or 0) != int(config_id):
                    continue
            except (TypeError, ValueError):
                continue
            succ = _get_any(slot, ["succUsedInFrame", "succ_used_in_frame"], 0)
            try:
                return float(succ or 0) > 0
            except (TypeError, ValueError):
                return False
        return False

    def _is_in_berserk_engagement(self, main_hero, enemy_hero):
        if main_hero is None or enemy_hero is None:
            return False
        if float(_hp(enemy_hero) or 0) <= 0:
            return False
        return math.dist(_loc(main_hero), _loc(enemy_hero)) <= GameConfig.BERSERK_ENGAGE_RANGE

    def _detect_recover_low_hp(self, main_hero, enemy_hero, frame_no):
        self._recover_debug = self._empty_recover_debug()
        if main_hero is None:
            self._pending_recover = None
            return 0

        frame_no = int(frame_no or 0)
        main_hp = float(_hp(main_hero) or 0)
        main_max_hp = float(_max_hp(main_hero) or 1)
        hp_rate = main_hp / max(main_max_hp, 1.0)
        buff_ids = self._hero_buff_ids(main_hero)
        has_start_buff = GameConfig.RECOVER_START_BUFF_ID in buff_ids
        has_effect_buff = GameConfig.RECOVER_EFFECT_BUFF_ID in buff_ids
        hurt_by_enemy_hero = self._hurt_by_enemy_hero(main_hero, enemy_hero)

        slot4_used = self._slot_succ_used(main_hero, 4)
        summoner_heal_used = self._used_skill_by_config(main_hero, 80102)
        if hp_rate > 0.9 and (slot4_used or summoner_heal_used or has_start_buff):
            self._pending_recover = None
            self._recover_debug["recover_high_hp_penalty_count"] = 1
            return -1

        if self._pending_recover is None and hp_rate < 0.5:
            if slot4_used or summoner_heal_used or has_start_buff:
                self._pending_recover = {
                    "attempt_frame": frame_no,
                    "main_hp_at_attempt": main_hp,
                    "saw_start_buff": bool(has_start_buff),
                    "saw_effect_buff": bool(has_effect_buff),
                }
                self._recover_debug["recover_attempt_count"] = 1

        if self._pending_recover is not None:
            self._pending_recover["saw_start_buff"] = bool(
                self._pending_recover.get("saw_start_buff", False) or has_start_buff
            )
            self._pending_recover["saw_effect_buff"] = bool(
                self._pending_recover.get("saw_effect_buff", False) or has_effect_buff
            )
            elapsed = frame_no - int(self._pending_recover["attempt_frame"])
            hp_gain = main_hp - float(self._pending_recover["main_hp_at_attempt"])
            success = hp_gain >= GameConfig.RECOVER_HP_GAIN_THRESHOLD
            if success and elapsed >= GameConfig.RECOVER_CONFIRM_FRAMES:
                self._pending_recover = None
                self._recover_debug["recover_success_count"] = 1
                return 1
            if 0 <= elapsed <= int(GameConfig.RECOVER_INTERRUPT_FRAMES) and hurt_by_enemy_hero and not success:
                self._pending_recover = None
                self._recover_debug["recover_interrupted_count"] = 1
                return -float(GameConfig.RECOVER_INTERRUPT_PENALTY)
            if elapsed > int(GameConfig.RECOVER_INTERRUPT_FRAMES):
                self._pending_recover = None
        return 0

    def _hero_buff_ids(self, hero):
        buff_state = _get_any(hero or {}, ["buff_state", "buffState"], {}) or {}
        buff_skills = _get_any(buff_state, ["buff_skills", "buffSkills"], []) or []
        ids = set()
        for buff in buff_skills:
            buff_id = _get_any(buff, ["config_id", "configId", "id"], 0)
            try:
                buff_id = int(buff_id or 0)
            except (TypeError, ValueError):
                buff_id = 0
            if buff_id > 0:
                ids.add(buff_id)
        return ids

    def _is_under_own_tower(self, main_hero, main_tower):
        if main_hero is None or main_tower is None:
            return False
        main_pos = self._entity_pos(main_hero)
        tower_pos = self._entity_pos(main_tower)
        if main_pos is None or tower_pos is None:
            return False
        dist = ((main_pos[0] - tower_pos[0]) ** 2 + (main_pos[1] - tower_pos[1]) ** 2) ** 0.5
        return dist <= self._tower_attack_range(main_tower)
