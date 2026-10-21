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


class MonsterProcess:
    def __init__(self, camp):
        self.normalizer = FeatureNormalizer()
        self.main_camp = camp
        self.transform_camp2_to_camp1 = camp == "PLAYERCAMP_2"
        self.get_monster_config()
        self.map_feature_to_norm = self.normalizer.parse_config(self.monster_feature_config)
        self.view_dist = 15000
        self.unit_buff_num = Config.feature_monster_shape[0]
        self.one_unit_feature_num = Config.feature_monster_shape[1]

    def get_monster_config(self):
        self.config = configparser.ConfigParser()
        current_dir = os.path.dirname(__file__)
        config_path = os.path.join(current_dir, "monster_feature_config.ini")
        self.config.read(config_path)

        # Get normalized configuration
        # 获取归一化的配置
        self.monster_feature_config = []
        for feature, config in self.config["feature_config"].items():
            self.monster_feature_config.append(f"{feature}:{config}")

        # Get feature function configuration
        # 获取特征函数的配置
        self.feature_func_map = {}
        for feature, func_name in self.config["feature_functions"].items():
            if hasattr(self, func_name):
                self.feature_func_map[feature] = getattr(self, func_name)
            else:
                raise ValueError(f"Unsupported function: {func_name}")

    def process_vec_monster(self, data_dicts):
        self.main_camp_hero_dict = data_dicts["hero"][0]
        self.enemy_camp_hero_dict = data_dicts["hero"][1]
        self.monster_dict = data_dicts["monster"][0]

        monster_vector_feature = self.generate_monster_feature(self.monster_dict)
        assert (
            len(monster_vector_feature) == self.unit_buff_num * self.one_unit_feature_num
        ), f"vector_feature长度应为{self.unit_buff_num * self.one_unit_feature_num}, 实际为{len(monster_vector_feature)}"

        return monster_vector_feature

    def generate_monster_feature(self, monster_info):
        vector_feature = []
        num_monsters_considered = 0
        for monster in monster_info.values():
            # Get at most unit_buff_num monster features
            # 最多获取 unit_buff_num 个野怪特征
            if num_monsters_considered >= self.unit_buff_num:
                break
            if monster is None:
                for _ in range(self.one_unit_feature_num):
                    vector_feature.append(0)
                continue
            # Generate each specific feature through feature_func_map
            # 通过 feature_func_map 生成每个具体特征
            for feature_name, feature_func in self.feature_func_map.items():
                value = []
                self.feature_func_map[feature_name](monster, value)
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
            num_monsters_considered += 1

        return vector_feature

    def get_hp(self, monster, vector_feature):
        value = monster["hp"]
        vector_feature.append(value)

    def get_max_hp(self, monster, vector_feature):
        value = monster["max_hp"]
        vector_feature.append(value)

    def get_hp_rate(self, monster, vector_feature):
        value = 0
        if monster["max_hp"] > 0:
            value = monster["hp"] / monster["max_hp"]
        vector_feature.append(value)

    def get_atk(self, monster, vector_feature):
        value = monster["phy_atk"]
        vector_feature.append(value)

    def get_kill_income(self, monster, vector_feature):
        value = monster["kill_income"]
        vector_feature.append(value)

    def cal_dist(self, pos1, pos2):
        dist = math.sqrt((pos1["x"] / 100.0 - pos2["x"] / 100.0) ** 2 + (pos1["z"] / 100.0 - pos2["z"] / 100.0) ** 2)
        return dist

    def is_alive(self, monster, vector_feature):
        value = 0.0
        if monster["hp"] > 0:
            value = 1.0
        vector_feature.append(value)

    def belong_to_main_camp(self, monster, vector_feature):
        value = 0.0
        if monster["camp"] == self.main_camp:
            value = 1.0
        vector_feature.append(value)

    def get_monster_type(self, monster, vector_feature):
        monster_config_id = monster["config_id"]
        if monster_config_id in [49]:
            value = 1.0
        elif monster_config_id in [30, 31, 32, 33]:
            value = 2.0
        elif monster_config_id in [42, 43, 44, 45]:
            value = 3.0
        elif monster_config_id in [59]:
            value = 4.0
        elif monster_config_id in [122]:
            value = 5.0
        vector_feature.append(value)

    def generate_monster_with_each_hero_feature(self, monster, vector_feature):
        for _, hero in self.main_camp_hero_dict.items():
            dist = self.cal_dist(hero["location"], monster["location"])
            dist = dist * 100
            vector_feature.append(dist)
        for _, hero in self.enemy_camp_hero_dict.items():
            dist = self.cal_dist(hero["location"], monster["location"])
            dist = dist * 100
            vector_feature.append(dist)

    def get_location_x(self, monster, vector_feature):
        value = monster["location"]["x"]
        if self.transform_camp2_to_camp1 and value != 100000:
            value = 0 - value
        vector_feature.append(value)

    def get_location_z(self, monster, vector_feature):
        value = monster["location"]["z"]
        if self.transform_camp2_to_camp1 and value != 100000:
            value = 0 - value
        vector_feature.append(value)
