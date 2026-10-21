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
from collections import OrderedDict


class SoldierProcess:
    def __init__(self, camp):
        self.normalizer = FeatureNormalizer()
        self.main_camp = camp
        self.main_camp_hero_dict = {}
        self.enemy_camp_hero_dict = {}
        self.main_camp_soldier_dict = {}
        self.enemy_camp_soldier_dict = {}
        self.transform_camp2_to_camp1 = camp == "PLAYERCAMP_2"
        self.get_soldier_config()
        self.map_feature_to_norm = self.normalizer.parse_config(self.soldier_feature_config)
        self.view_dist = 15000
        self.unit_buff_num = Config.feature_soldier_frd_shape[0]
        self.one_unit_feature_num = Config.feature_soldier_frd_shape[1]

    def get_soldier_config(self):
        self.config = configparser.ConfigParser()
        current_dir = os.path.dirname(__file__)
        config_path = os.path.join(current_dir, "soldier_feature_config.ini")
        self.config.read(config_path)

        # Get normalized configuration
        # 获取归一化的配置
        self.soldier_feature_config = []
        for feature, config in self.config["feature_config"].items():
            self.soldier_feature_config.append(f"{feature}:{config}")

        # Get feature function configuration
        # 获取特征函数的配置
        self.feature_func_map = {}
        for feature, func_name in self.config["feature_functions"].items():
            if hasattr(self, func_name):
                self.feature_func_map[feature] = getattr(self, func_name)
            else:
                raise ValueError(f"Unsupported function: {func_name}")

    def process_vec_soldier(self, data_dicts):
        self.main_camp_hero_dict = data_dicts["hero"][0]
        self.enemy_camp_hero_dict = data_dicts["hero"][1]
        self.main_camp_soldier_dict = data_dicts["soldier"][0]
        self.enemy_camp_soldier_dict = data_dicts["soldier"][1]

        vector_feature = []

        # Generate features for our camp soldiers
        # 生成我方阵营的小兵特征
        main_camp_soldier_vector_feature = self.generate_one_type_soldier_feature(
            self.main_camp_soldier_dict, "main_camp"
        )

        assert (
            len(main_camp_soldier_vector_feature) == self.unit_buff_num * self.one_unit_feature_num
        ), f"main_camp_soldier_vector_feature长度应为{self.unit_buff_num * self.one_unit_feature_num}, 实际为{len(main_camp_soldier_vector_feature)}"
        vector_feature.extend(main_camp_soldier_vector_feature)

        # Generate features for enemy camp soldiers
        # 生成敌方阵营的小兵特征
        enemy_camp_soldier_vector_feature = self.generate_one_type_soldier_feature(
            self.enemy_camp_soldier_dict, "enemy_camp"
        )
        vector_feature.extend(enemy_camp_soldier_vector_feature)

        return vector_feature

    def generate_one_type_soldier_feature(self, one_type_soldier_info, camp):
        vector_feature = []
        num_soldiers_considered = 0
        for soldier in one_type_soldier_info.values():
            # Get at most unit_buff_num soldier features
            # 最多获取 unit_buff_num 个士兵特征
            if num_soldiers_considered >= self.unit_buff_num:
                break
            # Generate each specific feature through feature_func_map
            # 通过 feature_func_map 生成每个具体特征
            for feature_name, feature_func in self.feature_func_map.items():
                value = []
                self.feature_func_map[feature_name](soldier, value)
                # Normalize the specific feature
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
            num_soldiers_considered += 1

        if num_soldiers_considered < self.unit_buff_num:
            self.no_soldier_feature(vector_feature, num_soldiers_considered)
        return vector_feature

    def no_soldier_feature(self, vector_feature, num_soldiers_considered):
        for _ in range((self.unit_buff_num - num_soldiers_considered) * self.one_unit_feature_num):
            vector_feature.append(0)

    def get_hp(self, soldier, vector_feature):
        value = soldier["hp"]
        vector_feature.append(value)

    def get_max_hp(self, soldier, vector_feature):
        value = soldier["max_hp"]
        vector_feature.append(value)

    def get_hp_rate(self, soldier, vector_feature):
        value = 0
        if soldier["max_hp"] > 0:
            value = soldier["hp"] / soldier["max_hp"]
        vector_feature.append(value)

    def get_atk(self, soldier, vector_feature):
        value = soldier["phy_atk"]
        vector_feature.append(value)

    def get_kill_income(self, soldier, vector_feature):
        value = soldier["kill_income"]
        vector_feature.append(value)

    def cal_dist(self, pos1, pos2):
        dist = math.sqrt((pos1["x"] / 100.0 - pos2["x"] / 100.0) ** 2 + (pos1["z"] / 100.0 - pos2["z"] / 100.0) ** 2)
        return dist

    def is_alive(self, soldier, vector_feature):
        value = 0.0
        if soldier["hp"] > 0:
            value = 1.0
        vector_feature.append(value)

    def belong_to_main_camp(self, soldier, vector_feature):
        value = 0.0
        if soldier["camp"] == self.main_camp:
            value = 1.0
        vector_feature.append(value)

    def get_soldier_type(self, soldier, vector_feature):
        soldier_config_id = soldier["config_id"]
        if soldier_config_id in [6023, 6024, 6088, 6089, 6320, 6321]:
            value = 1.0
        elif soldier_config_id in [6002, 6005, 6014, 6015, 6372, 6375, 6085, 6086]:
            value = 2.0
        else:
            value = 3.0
        vector_feature.append(value)

    def generate_soldier_with_each_hero_feature(self, soldier, vector_feature):
        for hero_id, hero in self.main_camp_hero_dict.items():
            dist = self.cal_dist(hero["location"], soldier["location"])
            dist = dist * 100
            vector_feature.append(dist)
        for hero_id, hero in self.enemy_camp_hero_dict.items():
            dist = self.cal_dist(hero["location"], soldier["location"])
            dist = dist * 100
            vector_feature.append(dist)

    def get_diy_feature(self, soldier, vector_feature):
        # Custom features
        # 自定义特征
        diy_feature = 0
        vector_feature.append(diy_feature)

    def get_location_x(self, soldier, vector_feature):
        value = soldier["location"]["x"]
        if self.transform_camp2_to_camp1 and value != 100000:
            value = 0 - value
        vector_feature.append(value)

    def get_location_z(self, soldier, vector_feature):
        value = soldier["location"]["z"]
        if self.transform_camp2_to_camp1 and value != 100000:
            value = 0 - value
        vector_feature.append(value)
