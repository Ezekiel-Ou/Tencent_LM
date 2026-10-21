#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""


import numpy as np


class GameConfig:
    HERO_NUM = 3
    MODEL_SAVE_INTERVAL = 1800
    SEND_SAMPLE_NUM = 1920


# Configuration related to model and algorithms used
# 模型和算法使用的相关配置
class Config:
    # The input dimension of samples on the learner from Reverb varies depending on the algorithm used.
    # learner上reverb样本的输入维度, 注意不同的算法维度不一样
    HERO_NUM = GameConfig.HERO_NUM
    feature_img_shape = [6, 17, 17]
    feature_hero_frd_shape = [3, 86]
    feature_hero_emy_shape = [3, 86]
    feature_soldier_frd_shape = [10, 18]
    feature_soldier_emy_shape = [10, 18]
    feature_organ_frd_shape = [3, 21]
    feature_organ_emy_shape = [3, 21]
    feature_monster_shape = [20, 20]
    feature_global_info_shape = [1, 47]
    feature_hero_main_shape = [1, 86]

    HERO_SERI_VEC_SPLIT_SHAPE = [
        feature_img_shape,
        feature_hero_frd_shape,
        feature_hero_emy_shape,
        feature_soldier_frd_shape,
        feature_soldier_emy_shape,
        feature_organ_frd_shape,
        feature_organ_emy_shape,
        feature_monster_shape,
        feature_global_info_shape,
        feature_hero_main_shape,
    ]

    LSTM_TIME_STEPS = 16
    LSTM_UNIT_SIZE = 16
    HERO_LABEL_SIZE_LIST = [13, 25, 42, 42, 39]
    legal_action_shape = HERO_LABEL_SIZE_LIST
    feature_shape = [abs(sum([int(np.prod(x)) for x in HERO_SERI_VEC_SPLIT_SHAPE]))]

    SAMPLE_DIM = 0
