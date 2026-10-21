#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""

from agent_ppo.feature.feature_process.feature_normalizer import FeatureNormalizer
from agent_ppo.conf.conf import LegacyConfig as Config
import configparser
import os
import math


class HeroProcess:
    HERO_FIELD_ALIASES = {
        "killCnt": "kill_cnt",
        "deadCnt": "dead_cnt",
        "assistCnt": "assist_cnt",
        "moneyCnt": "money_cnt",
        "totalHurt": "total_hurt",
        "totalHurtToHero": "total_hurt_to_hero",
        "totalBeHurtByHero": "total_be_hurt_by_hero",
    }

    def __init__(self, camp):
        self.normalizer = FeatureNormalizer()
        self.main_camp = camp
        self.transform_camp2_to_camp1 = camp == "PLAYERCAMP_2"
        self.get_hero_config()
        self.map_feature_to_norm = self.normalizer.parse_config(self.hero_feature_config)
        self.unit_buff_num = Config.feature_hero_frd_shape[0]
        self.one_unit_feature_num = Config.feature_hero_frd_shape[1]
        self.view_dist = 15000
        self.hero_features = {}

    def get_hero_config(self):
        self.config = configparser.ConfigParser()
        self.config.optionxform = str
        current_dir = os.path.dirname(__file__)
        config_path = os.path.join(current_dir, "hero_feature_config.ini")
        self.config.read(config_path)

        # Get normalized configuration
        # 获取归一化的配置
        self.hero_feature_config = []
        for feature, config in self.config["feature_config"].items():
            self.hero_feature_config.append(f"{feature}:{config}")

        # Get feature function configuration
        # 获取特征函数的配置
        self.feature_func_map = {}
        for feature, func_name in self.config["feature_functions"].items():
            if hasattr(self, func_name):
                self.feature_func_map[feature] = getattr(self, func_name)
            else:
                raise ValueError(f"Unsupported function: {func_name}")

    def process_vec_hero(self, data_dicts):
        self.main_camp_hero_dict = data_dicts["hero"][0]
        self.enemy_camp_hero_dict = data_dicts["hero"][1]
        self.main_camp_organ_dict = data_dicts["organ"][0]
        self.enemy_camp_organ_dict = data_dicts["organ"][1]

        # Generate hero features for our camp
        # 生成我方阵营的英雄特征
        self.hero_features = {}
        main_camp_hero_vector_features = self.generate_one_type_hero_feature(self.main_camp_hero_dict, "main_camp")

        # Generate hero features for enemy camp
        # 生成敌方阵营的英雄特征
        enemy_camp_hero_vector_features = self.generate_one_type_hero_feature(self.enemy_camp_hero_dict, "enemy_camp")

        return main_camp_hero_vector_features, enemy_camp_hero_vector_features, self.hero_features

    def generate_one_type_hero_feature(self, one_type_hero_info, camp):
        vector_features = []
        num_heros_considered = 0
        for hero in one_type_hero_info.values():
            if num_heros_considered >= self.unit_buff_num:
                break
            hero_feature = []
            # Generate each specific feature through feature_func_map
            # 通过 feature_func_map 生成每个具体特征
            for feature_name, feature_func in self.feature_func_map.items():
                value = []
                self.feature_func_map[feature_name](hero, value, feature_name)
                # Normalize the specific features
                # 对具体特征进行正则化
                if feature_name not in self.map_feature_to_norm:
                    assert False
                for k in value:
                    norm_func, *params = self.map_feature_to_norm[feature_name]
                    normalized_value = norm_func(k, *params)
                    if isinstance(normalized_value, list):
                        hero_feature.extend(normalized_value)
                    else:
                        hero_feature.append(normalized_value)

            assert (
                len(hero_feature) == self.one_unit_feature_num
            ), f"hero_feature长度应为{self.one_unit_feature_num}, 实际为{len(hero_feature)}"
            vector_features += hero_feature
            if camp == "main_camp":
                self.hero_features[hero["config_id"]] = hero_feature
            num_heros_considered += 1

        if num_heros_considered < self.unit_buff_num:
            for _ in range(self.unit_buff_num - num_heros_considered):
                empty_feature = [0] * self.one_unit_feature_num
                vector_features += empty_feature

        return vector_features

    def _get_hero_value(self, hero, feature_name):
        return hero[self.HERO_FIELD_ALIASES.get(feature_name, feature_name)]

    def get_from_hero(self, hero, vector_feature, feature_name):
        value = self._get_hero_value(hero, feature_name)
        vector_feature.append(value)

    def get_from_actor_state(self, hero, vector_feature, feature_name):
        value = self._get_hero_value(hero, feature_name)
        vector_feature.append(value)

    def get_from_actor_state_values(self, hero, vector_feature, feature_name):
        value = self._get_hero_value(hero, feature_name)
        vector_feature.append(value)

    def get_hp_rate(self, hero, vector_feature, feature_name):
        value = 0
        if hero["max_hp"] > 0:
            value = hero["hp"] / hero["max_hp"]
        vector_feature.append(value)

    def get_ep_rate(self, hero, vector_feature, feature_name):
        value = 0
        if hero["max_ep"] > 0:
            value = hero["ep"] / hero["max_ep"]
        vector_feature.append(value)

    def get_summon_skill_cd(self, hero, vector_feature, feature_name):
        value = 0
        for skill_state in hero["skill_state"]["slot_states"]:
            if skill_state["slot_type"] == "SLOT_SKILL_5":
                value = skill_state["cooldown"]
        vector_feature.append(value)

    def get_common_attack_is_useable(self, hero, vector_feature, feature_name):
        value = 0
        for skill_state in hero["skill_state"]["slot_states"]:
            if skill_state["slot_type"] == "SLOT_SKILL_0":
                value = 1 if skill_state["usable"] else 0
        vector_feature.append(value)

    def get_skill_1_useable(self, hero, vector_feature, feature_name):
        value = 0
        for skill_state in hero["skill_state"]["slot_states"]:
            if skill_state["slot_type"] == "SLOT_SKILL_1":
                value = 1 if skill_state["usable"] else 0
        vector_feature.append(value)

    def get_skill_2_useable(self, hero, vector_feature, feature_name):
        value = 0
        for skill_state in hero["skill_state"]["slot_states"]:
            if skill_state["slot_type"] == "SLOT_SKILL_2":
                value = 1 if skill_state["usable"] else 0
        vector_feature.append(value)

    def get_skill_3_useable(self, hero, vector_feature, feature_name):
        value = 0
        for skill_state in hero["skill_state"]["slot_states"]:
            if skill_state["slot_type"] == "SLOT_SKILL_3":
                value = 1 if skill_state["usable"] else 0
        vector_feature.append(value)

    def get_skill_4_useable(self, hero, vector_feature, feature_name):
        value = 0
        for skill_state in hero["skill_state"]["slot_states"]:
            if skill_state["slot_type"] == "SLOT_SKILL_4":
                value = 1 if skill_state["usable"] else 0
        vector_feature.append(value)

    def get_hero_skill_1_cd(self, hero, vector_feature, feature_name):
        value = 0
        for skill_state in hero["skill_state"]["slot_states"]:
            if skill_state["slot_type"] == "SLOT_SKILL_1":
                value = skill_state["cooldown"]
        vector_feature.append(value)

    def get_hero_skill_2_cd(self, hero, vector_feature, feature_name):
        value = 0
        for skill_state in hero["skill_state"]["slot_states"]:
            if skill_state["slot_type"] == "SLOT_SKILL_2":
                value = skill_state["cooldown"]
        vector_feature.append(value)

    def get_hero_skill_3_cd(self, hero, vector_feature, feature_name):
        value = 0
        for skill_state in hero["skill_state"]["slot_states"]:
            if skill_state["slot_type"] == "SLOT_SKILL_3":
                value = skill_state["cooldown"]
        vector_feature.append(value)

    def get_hero_skill_4_cd(self, hero, vector_feature, feature_name):
        value = 0
        for skill_state in hero["skill_state"]["slot_states"]:
            if skill_state["slot_type"] == "SLOT_SKILL_4":
                value = skill_state["cooldown"]
        vector_feature.append(value)

    def get_summon_skill_type(self, hero, vector_feature, feature_name):
        summon_skill_type_map = {
            "80102": 1,
            "80109": 2,
            "80104": 3,
            "80116": 3,
            "80108": 4,
            "80110": 5,
            "80105": 6,
            "80103": 7,
            "80107": 8,
            "80121": 9,
            "80115": 10,
        }
        value = 0
        for skill_state in hero["skill_state"]["slot_states"]:
            if skill_state["slot_type"] == "SLOT_SKILL_5":
                value = summon_skill_type_map.get(str(skill_state["configId"]), 0)
        vector_feature.append(value)

    def get_dist_from_enemy_heros(self, main_hero, vector_feature, feature_name):
        for _, hero in self.enemy_camp_hero_dict.items():
            dist = self.cal_dist(main_hero["location"], hero["location"])
            dist = dist * 100
            vector_feature.append(dist)

    def cal_dist(self, pos1, pos2):
        dist = math.sqrt((pos1["x"] / 100.0 - pos2["x"] / 100.0) ** 2 + (pos1["z"] / 100.0 - pos2["z"] / 100.0) ** 2)
        return dist

    def get_hero_level(self, hero, vector_feature, feature_name):
        value = 0.0
        if hero["level"] > 0:
            value = hero["level"]
        vector_feature.append(value)

    def get_hero_main_job(self, hero, vector_feature, feature_name):
        value = 0.0
        if hero["config_id"] == 132:
            value = 1.0
        elif hero["config_id"] == 141:
            value = 2.0
        elif hero["config_id"] == 193:
            value = 3.0
        vector_feature.append(value)

    def is_alive(self, hero, vector_feature, feature_name):
        value = 0.0
        if hero["hp"] > 0:
            value = 1.0
        vector_feature.append(value)

    def get_location_x(self, hero, vector_feature, feature_name):
        value = hero["location"]["x"]
        if self.transform_camp2_to_camp1 and value != 100000:
            value = 0 - value
        vector_feature.append(value)

    def get_location_z(self, hero, vector_feature, feature_name):
        value = hero["location"]["z"]
        if self.transform_camp2_to_camp1 and value != 100000:
            value = 0 - value
        vector_feature.append(value)

    def get_is_hero_under_tower_atk(self, hero, vector_feature, feature_name):
        value = 0
        for _, organ in self.enemy_camp_organ_dict.items():
            if organ["attack_target"] == hero["runtime_id"]:
                value = 1
                break
        vector_feature.append(value)

    def get_friend_hero_in_view(self, hero, vector_feature, feature_name):
        value = 0
        for _, friend_hero in self.main_camp_hero_dict.items():
            if friend_hero["hp"] <= 0:
                continue
            dist = self.cal_dist(hero["location"], friend_hero["location"])
            dist = dist * 100
            if dist < self.view_dist:
                value += 1
        vector_feature.append(value)

    def get_enemy_hero_in_view(self, hero, vector_feature, feature_name):
        value = 0
        for _, enemy_hero in self.enemy_camp_hero_dict.items():
            if enemy_hero["hp"] <= 0:
                continue
            dist = self.cal_dist(hero["location"], enemy_hero["location"])
            dist = dist * 100
            if dist < self.view_dist:
                value += 1
        vector_feature.append(value)

    def get_hero_in_view_diff(self, hero, vector_feature, feature_name):
        value = 0
        for _, friend_hero in self.main_camp_hero_dict.items():
            if friend_hero["hp"] <= 0:
                continue
            dist = self.cal_dist(hero["location"], friend_hero["location"])
            dist = dist * 100
            if dist < self.view_dist:
                value += 1
        for _, enemy_hero in self.enemy_camp_hero_dict.items():
            if enemy_hero["hp"] <= 0:
                continue
            dist = self.cal_dist(hero["location"], enemy_hero["location"])
            dist = dist * 100
            if dist < self.view_dist:
                value -= 1
        vector_feature.append(value)
