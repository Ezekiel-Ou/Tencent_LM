#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""

from enum import Enum
from agent_ppo.feature.feature_process.feature_normalizer import FeatureNormalizer
from agent_ppo.conf.conf import LegacyConfig as Config
import configparser
import os
import math
from collections import OrderedDict


class OrganProcess:
    def __init__(self, camp):
        self.normalizer = FeatureNormalizer()
        self.main_camp = camp
        self.transform_camp2_to_camp1 = camp == "PLAYERCAMP_2"
        self.get_organ_config()
        self.map_feature_to_norm = self.normalizer.parse_config(self.organ_feature_config)
        self.view_dist = 15000
        self.unit_buff_num = Config.feature_organ_frd_shape[0]
        self.one_unit_feature_num = Config.feature_organ_frd_shape[1]

    def get_organ_config(self):
        self.config = configparser.ConfigParser()
        current_dir = os.path.dirname(__file__)
        config_path = os.path.join(current_dir, "organ_feature_config.ini")
        self.config.read(config_path)

        # Get normalized configuration
        # 获取归一化的配置
        self.organ_feature_config = []
        for feature, config in self.config["feature_config"].items():
            self.organ_feature_config.append(f"{feature}:{config}")

        # Get feature function configuration
        # 获取特征函数的配置
        self.feature_func_map = {}
        for feature, func_name in self.config["feature_functions"].items():
            if hasattr(self, func_name):
                self.feature_func_map[feature] = getattr(self, func_name)
            else:
                raise ValueError(f"Unsupported function: {func_name}")

    def process_vec_organ(self, data_dicts):
        self.main_camp_hero_dict = data_dicts["hero"][0]
        self.enemy_camp_hero_dict = data_dicts["hero"][1]
        self.main_camp_organ_dict = data_dicts["organ"][0]
        self.enemy_camp_organ_dict = data_dicts["organ"][1]
        self.main_camp_soldier_dict = data_dicts["soldier"][0]
        self.enemy_camp_soldier_dict = data_dicts["soldier"][1]

        vector_feature = []
        # Generate features for our team's towers
        # 生成我方阵营的防御塔特征
        main_camp_organ_vector_feature = self.generate_one_type_organ_feature(self.main_camp_organ_dict, "main_camp")
        assert (
            len(main_camp_organ_vector_feature) == self.unit_buff_num * self.one_unit_feature_num
        ), f"main_camp_organ_vector_feature长度应为{self.unit_buff_num * self.one_unit_feature_num}, 实际为{len(main_camp_organ_vector_feature)}"
        vector_feature.extend(main_camp_organ_vector_feature)

        # Generate features for enemy team's towers
        # 生成敌方阵营的防御塔特征
        enemy_camp_organ_vector_feature = self.generate_one_type_organ_feature(self.enemy_camp_organ_dict, "enemy_camp")
        vector_feature.extend(enemy_camp_organ_vector_feature)

        return vector_feature

    def generate_one_type_organ_feature(self, one_type_organ_info, camp):
        vector_feature = []
        num_organs_considered = 0

        def process_organ(organ):
            nonlocal num_organs_considered
            # Generate each specific feature through feature_func_map
            # 通过 feature_func_map 生成每个具体特征
            for feature_name, feature_func in self.feature_func_map.items():
                value = []
                self.feature_func_map[feature_name](organ, value)
                # Normalize the specific features
                # 对具体特征进行正则化
                if feature_name not in self.map_feature_to_norm:
                    assert False
                for k in value:
                    norm_func, *params = self.map_feature_to_norm[feature_name]
                    normalized_value = norm_func(k, *params)
                    if isinstance(normalized_value, list):
                        vector_feature.extend(normalized_value)
                    else:
                        vector_feature.append(normalized_value)
            num_organs_considered += 1

        if "crystal" in one_type_organ_info:
            organ = one_type_organ_info["crystal"]
            process_organ(organ)

        for tower_name in ["tower_66", "tower_67", "tower_70", "tower_71"]:
            if tower_name in one_type_organ_info:
                organ = one_type_organ_info[tower_name]
                process_organ(organ)

        if num_organs_considered < self.unit_buff_num:
            for _ in range((self.unit_buff_num - num_organs_considered) * self.one_unit_feature_num):
                vector_feature.append(0)

        return vector_feature

    def get_hp(self, organ, vector_feature):
        value = organ["hp"]
        vector_feature.append(value)

    def get_max_hp(self, organ, vector_feature):
        value = organ["max_hp"]
        vector_feature.append(value)

    def get_hp_rate(self, organ, vector_feature):
        value = 0
        if organ["max_hp"] > 0:
            value = organ["hp"] / organ["max_hp"]
        vector_feature.append(value)

    def get_atk(self, organ, vector_feature):
        value = organ["phy_atk"]
        vector_feature.append(value)

    def get_atk_range(self, organ, vector_feature):
        value = organ["attack_range"]
        vector_feature.append(value)

    def get_kill_income(self, organ, vector_feature):
        value = organ["kill_income"]
        vector_feature.append(value)

    def judge_in_view(self, main_hero_location, obj_location):
        if (
            (main_hero_location["x"] - obj_location["x"] >= 0 - self.view_dist)
            and (main_hero_location["x"] - obj_location["x"] <= self.view_dist)
            and (main_hero_location["z"] - obj_location["z"] >= 0 - self.view_dist)
            and (main_hero_location["z"] - obj_location["z"] <= self.view_dist)
        ):
            return True
        return False

    def cal_dist(self, pos1, pos2):
        dist = math.sqrt((pos1["x"] / 100.0 - pos2["x"] / 100.0) ** 2 + (pos1["z"] / 100.0 - pos2["z"] / 100.0) ** 2)
        return dist

    def is_alive(self, organ, vector_feature):
        value = 0.0
        if organ["hp"] > 0:
            value = 1.0
        vector_feature.append(value)

    def belong_to_main_camp(self, organ, vector_feature):
        value = 0.0
        if organ["camp"] == self.main_camp:
            value = 1.0
        vector_feature.append(value)

    def get_organ_type(self, organ, vector_feature):
        organ_sub_type = organ["sub_type"]
        if organ_sub_type == "ACTOR_SUB_TOWER":
            value = 21
        elif organ_sub_type == "ACTOR_SUB_CRYSTAL":
            value = 24
        else:
            value = 0
        vector_feature.append(value)

    def generate_distance_with_each_hero_feature(self, organ, vector_feature):
        for _, hero in self.main_camp_hero_dict.items():
            dist = self.cal_dist(hero["location"], organ["location"])
            dist = dist * 100
            vector_feature.append(dist)
        for _, hero in self.enemy_camp_hero_dict.items():
            dist = self.cal_dist(hero["location"], organ["location"])
            dist = dist * 100
            vector_feature.append(dist)

    def get_normal_organ_location_x(self, organ, vector_feature):
        value = organ["location"]["x"]
        if self.transform_camp2_to_camp1:
            value = 0 - value
        vector_feature.append(value)

    def get_normal_organ_location_z(self, organ, vector_feature):
        value = organ["location"]["z"]
        if self.transform_camp2_to_camp1:
            value = 0 - value
        vector_feature.append(value)

    def get_soldier_in_range(self, organ, vector_feature):
        value_in_range = 0.0
        if organ["camp"] == self.main_camp:
            for _, soldier in self.enemy_camp_soldier_dict.items():
                dist = self.cal_dist(organ["location"], soldier["location"])
                dist *= 100
                if dist < organ["attack_range"]:
                    value_in_range = 1.0
                    break
        else:
            for _, soldier in self.main_camp_soldier_dict.items():
                dist = self.cal_dist(organ["location"], soldier["location"])
                dist *= 100
                if dist < organ["attack_range"]:
                    value_in_range = 1.0
                    break
        vector_feature.append(value_in_range)
