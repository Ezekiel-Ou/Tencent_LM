#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2025 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""

class Config:
    # 64*x+y
    TREASURE_POSITIONS = [
        1230, 604, 620, 2733, 2071,
        3192, 2298, 1527, 2657, 3497
    ]

    OPTIMAL_TREASURE_ORDER = [0, 1, 2, 4, 5, 9, 8, 3, 6, 7]

    NUM_STAGES = 11   # 0~10

    # 距离归一化上限，要和 definition.py 里 observation_process 保持一致
    MAX_DIST_NORM = 150.0

    # TSP 引导强度
    TSP_TARGET_REWARD = 0.35
    TSP_TARGET_PENALTY = 0.12

    # 机会宝箱阈值（原始格数）
    OPPORTUNITY_DIST = 2.0

    # 终点附近关闭宝箱引导（原始格数）
    NEAR_END_DIST = 8.0
    VERY_NEAR_END_DIST = 6.0

    # =========================
    # 1. Observation / Action
    # =========================

    # 状态维度：
    DIM_OF_OBSERVATION = 250

    # 动作维度：上下左右
    DIM_OF_ACTION = 4

    # =========================
    # 2. SAMPLE_DIM（⚠️必须和definition.py一致）
    # =========================

    # PPO sample结构：
    # obs + action + logprob + reward + done + value + next_obs
    # = 250 + 1 + 1 + 1 + 1 + 1 + 250 = 505
    SAMPLE_DIM = 2 * DIM_OF_OBSERVATION + 5   # = 505

    # =========================
    # 3. PPO Hyperparameters
    # =========================

    # 折扣因子
    GAMMA = 0.95
    # GAE参数
    LAMBDA = 0.95
    # PPO裁剪系数
    CLIP_EPS = 0.2
    # 学习率
    ACTOR_LR = 3e-4
    CRITIC_LR = 1e-3
    # 熵正则（鼓励探索）
    ENTROPY_COEF = 0.01
    # value loss 权重
    VALUE_COEF = 0.5
    # 梯度裁剪
    MAX_GRAD_NORM = 0.5
    # =========================
    # 4. 训练参数
    # =========================
    # 每轮收集多少步
    ROLLOUT_STEPS = 1024
    # PPO更新轮数
    PPO_EPOCHS = 4
    # mini-batch大小
    MINI_BATCH_SIZE = 256
    # 最大训练episode
    MAX_EPISODES = 2000
    # 日志 & 保存
    LOG_FREQ = 10
    SAVE_MODEL_FREQ = 100
    # ========================
    # 5. 网络结构
    # =========================
    HIDDEN_DIM = 128
    # =========================
    # 6. 数据类型（开悟要求）
    # =========================
    DATA_TYPE = "float32"