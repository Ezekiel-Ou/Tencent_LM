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


class GlobalProcess:
    def __init__(self, camp):
        self.normalizer = FeatureNormalizer()
        self.main_camp = camp
        self.transform_camp2_to_camp1 = camp == "PLAYERCAMP_2"
        self.get_global_config()
        self.map_feature_to_norm = self.normalizer.parse_config(self.global_feature_config)
        self.global_feature_num = math.prod(Config.feature_global_info_shape)

    def get_global_config(self):
        self.global_config = configparser.ConfigParser()
        current_dir = os.path.dirname(__file__)
        config_path = os.path.join(current_dir, "global_feature_config.ini")
        self.global_config.read(config_path)

        # Get normalized configuration
        # 获取归一化的配置
        self.global_feature_config = []
        for feature, config in self.global_config["feature_config"].items():
            self.global_feature_config.append(f"{feature}:{config}")

        # Get feature function configuration
        # 获取特征函数的配置
        self.global_feature_func_map = {}
        for feature, func_name in self.global_config["feature_functions"].items():
            if hasattr(self, func_name):
                self.global_feature_func_map[feature] = getattr(self, func_name)
            else:
                raise ValueError(f"Unsupported function: {func_name}")

    def process_vec_global(self, data_dicts):
        self.main_camp_hero_dict = data_dicts["hero"][0]
        self.enemy_camp_hero_dict = data_dicts["hero"][1]
        self.main_camp_organ_dict = data_dicts["organ"][0]
        self.enemy_camp_organ_dict = data_dicts["organ"][1]

        global_vector_feature = self.generate_global_feature(data_dicts["frame_no"])
        assert (
            len(global_vector_feature) == self.global_feature_num
        ), f"vector_feature长度应为{self.global_feature_num}, 实际为{len(global_vector_feature)}"

        return global_vector_feature

    def generate_global_feature(self, frame_no):
        vector_feature = []

        # Generate each specific feature through feature_func_map
        # 通过 feature_func_map 生成每个具体特征
        for feature_name, feature_func in self.global_feature_func_map.items():
            value = []
            self.global_feature_func_map[feature_name](frame_no, value)
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

        return vector_feature

    def get_game_time(self, frame_no, vector_feature):
        frame_no = frame_no
        game_time = frame_no / 1800
        vector_feature.append(game_time)

    def get_friend_money(self, frame_no, vector_feature):
        money_count = 0
        for _, hero in self.main_camp_hero_dict.items():
            money_count += hero["money_cnt"]
        vector_feature.append(money_count)

    def get_enemy_money(self, frame_no, vector_feature):
        money_count = 0
        for _, hero in self.enemy_camp_hero_dict.items():
            money_count += hero["money_cnt"]
        vector_feature.append(money_count)

    def get_money_diff(self, frame_no, vector_feature):
        money_count = 0
        for _, hero in self.main_camp_hero_dict.items():
            money_count += hero["money_cnt"]
        for _, hero in self.enemy_camp_hero_dict.items():
            money_count -= hero["money_cnt"]
        vector_feature.append(money_count)

    def get_friend_kill(self, frame_no, vector_feature):
        kill_count = 0
        for _, hero in self.main_camp_hero_dict.items():
            kill_count += hero["kill_cnt"]
        vector_feature.append(kill_count)

    def get_enemy_kill(self, frame_no, vector_feature):
        kill_count = 0
        for _, hero in self.enemy_camp_hero_dict.items():
            kill_count += hero["kill_cnt"]
        vector_feature.append(kill_count)

    def get_kill_diff(self, frame_no, vector_feature):
        kill_count = 0
        for _, hero in self.main_camp_hero_dict.items():
            kill_count += hero["kill_cnt"]
        for _, hero in self.enemy_camp_hero_dict.items():
            kill_count -= hero["kill_cnt"]
        vector_feature.append(kill_count)

    def get_friend_organ(self, frame_no, vector_feature):
        organ_count = 0
        for _, organ in self.main_camp_organ_dict.items():
            organ_count += 1 if organ["hp"] > 0 else 0
        vector_feature.append(organ_count)

    def get_enemy_organ(self, frame_no, vector_feature):
        organ_count = 0
        for _, organ in self.enemy_camp_organ_dict.items():
            organ_count += 1 if organ["hp"] > 0 else 0
        vector_feature.append(organ_count)

    def get_organ_diff(self, frame_no, vector_feature):
        organ_count = 0
        for _, organ in self.main_camp_organ_dict.items():
            organ_count += 1 if organ["hp"] > 0 else 0
        for _, organ in self.enemy_camp_organ_dict.items():
            organ_count -= 1 if organ["hp"] > 0 else 0
        vector_feature.append(organ_count)
