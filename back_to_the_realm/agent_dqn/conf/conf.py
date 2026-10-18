#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2025 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""


# Configuration, including dimension settings, algorithm parameter settings.
# The last few configurations in the file are for the Kaiwu platform to use and should not be changed.
# 配置，包含维度设置，算法参数设置，文件的最后一些配置是开悟平台使用不要改动
class Config:

    # Input dimension of reverb sample on learner. Note that different algorithms have different dimensions.
    # For example, the dimension of dqn in the sample code is 21624, and the dimension of target_dqn is 21624
    # **Note**, this item must be configured correctly and should be aligned with the NumpyData2SampleData function data in definition.py
    # Otherwise the sample dimension error may be reported
    # learner上reverb样本的输入维度, 注意不同的算法维度不一样, 比如示例代码中dqn的维度是21624, target_dqn的维度是21624
    # **注意**，此项应该与definition.py中的NumpyData2SampleData函数数据对齐，否则可能报样本维度错误
    SAMPLE_DIM = 21624

    # field of view radius
    # 视野域半径
    VIEW_SIZE = 25

    # Size of observation. Note that in our sample code the original feature dimension is 10808.
    # Here is the dimension after CNN processing and the original vector feature concatenation.
    # observation的维度，注意在我们的示例代码中原特征维度是10808，这里是经过CNN处理之后的维度与原始向量特征拼接后的维度
    DIM_OF_OBSERVATION = 128 + 404

    # Dimension of movement action direction
    # 移动动作方向的维度
    DIM_OF_ACTION_DIRECTION = 8

    # Dimension of flash action direction
    # 闪现动作方向的维度
    DIM_OF_TALENT = 8

    # norm_pos + one_hot_pos + end_pos_features + treasure_poss_features + [buff_availability, talent_availability]
    #  + obstacle_map + treasure_map + end_map + location_memory
    # 2 + 128*2 + 9  + 9*15 + 2 + 4*51*51 = 10808
    # # Describe how to perform feature segmentation. The features in the sample code are processed into vectors + feature maps.
    # The following configuration describes the dimensions of the two.
    # 描述如何进行特征分割，示例代码中的特征处理成向量+特征图，以下配置描述了两者的维度
    DESC_OBS_SPLIT = [404, (4, VIEW_SIZE * 2 + 1, VIEW_SIZE * 2 + 1)]  # sum = 10808

    # Exploration factor, see the calculation of epsilon in the function in the above comment
    # 探索因子, epsilon的计算见上面注释中的函数
    EPSILON_GREEDY_PROBABILITY = 600000

    # Discount factor GAMMA in RL
    # RL中的回报折扣GAMMA
    GAMMA = 0.95

    # epsilon initial value
    # epsilon初始值
    EPSILON = 1.0

    # epsilon minimum value
    # epsilon最小值
    EPSILON_MIN = 0.1

    # target network update frequency
    # target网络更新频率
    TARGET_UPDATE_FREQ = 500

    # soft update ratio for target network
    # target网络软更新系数
    TARGET_SOFT_TAU = 0.01

    # Number of nearest seen treasures considered for shaping.
    # 用于奖励塑形的最近已见宝箱数量
    TREASURE_TOPK = 1

    # Vision coverage exploration settings.
    # 视野覆盖探索设置：当前 51x51 视野累计覆盖整张 128x128 地图的比例
    VISION_COVERAGE_PHASE1_THRESHOLD = 0.75
    FINAL_REWARD_COVERAGE_THRESHOLD = 0.8
    LOWER_HALF_COVERAGE_THRESHOLD = 0.65
    PHASE1_OBJECTIVE_REWARD_SCALE = 0.35
    PHASE1_VISION_EXPLORATION_SCALE = 1.0
    PHASE2_VISION_EXPLORATION_SCALE = 0.25
    COVERAGE_MILESTONE_THRESHOLDS = (0.15, 0.30, 0.45, 0.60, 0.75)
    COVERAGE_MILESTONE_REWARD = 0.8
    LOWER_HALF_COVERAGE_ONCE_REWARD = 1.2
    RIGHT_LOWER_COVERAGE_TARGET = 0.35
    RIGHT_LOWER_COVERAGE_ONCE_REWARD = 1.0

    # Movement preference shaping: prioritize straight movement over diagonal movement.
    # 移动偏好奖励：优先直行，其次斜行
    PREFER_STRAIGHT_REWARD = 0.4
    PREFER_DIAGONAL_REWARD = 0.15

    # Small-area circling penalty settings.
    # 小范围转圈惩罚参数
    SMALL_LOOP_RADIUS = 2
    SMALL_LOOP_VISIT_THRESHOLD = 6
    SMALL_LOOP_PENALTY_SCALE = 0.2
    SMALL_LOOP_MAX_PENALTY = 2.0

    # Reward shaping thresholds and scales.
    # 奖励塑形相关阈值和缩放参数
    BUMP_DISTANCE_THRESHOLD = 500
    FINAL_REWARD_BASE = 5.0
    FINAL_REWARD_TREASURE_BONUS = 0.5
    END_DISTANCE_REWARD_SCALE = 1000.0
    END_DISTANCE_REWARD_MIN = -1.0
    END_DISTANCE_REWARD_MAX = 1.0
    TREASURE_DISTANCE_REWARD_SCALE = 800.0
    TREASURE_DISTANCE_REWARD_MIN = -1.2
    TREASURE_DISTANCE_REWARD_MAX = 1.2
    TREASURE_DISTANCE_DISCOVERY_BONUS = 0.35
    TREASURE_COLLECTED_REWARD_BASE = 1.2
    TREASURE_COLLECTED_REWARD_PER_CHEST = 0.3
    MIN_TREASURE_BEFORE_FINISH = 11
    END_APPROACH_WITH_FEW_TREASURES_SCALE = 1200.0
    END_APPROACH_WITH_FEW_TREASURES_MIN = -1.0
    END_APPROACH_WITH_FEW_TREASURES_MAX = 0.15
    EARLY_FINISH_PENALTY_BASE = -4.0
    EARLY_FINISH_PENALTY_PER_MISSING = -0.8
    EARLY_FINISH_PENALTY_MIN = -10.0
    NEW_VISIBLE_TREASURE_REWARD_PER_CHEST = 0.6
    NEW_VISIBLE_TREASURE_REWARD_MAX = 1.5
    TOPK_TREASURE_PROGRESS_REWARD_SCALE = 2.0
    TOPK_TREASURE_PROGRESS_REWARD_MIN = -1.0
    TOPK_TREASURE_PROGRESS_REWARD_MAX = 1.0
    TOPK_TREASURE_COLLECTED_REWARD_PER_CHEST = 0.8
    TOPK_TREASURE_COLLECTED_REWARD_MAX = 1.5
    BUFF_DISTANCE_REWARD_SCALE = 500.0
    BUFF_DISTANCE_REWARD_MIN = -1.0
    BUFF_DISTANCE_REWARD_MAX = 1.0
    BUFF_COLLECT_DISTANCE_THRESHOLD = 2
    BUFF_COLLECT_REWARD = 1.0
    POST_TREASURE_END_PROGRESS_SCALE = 1000.0
    POST_TREASURE_END_PROGRESS_OFFSET = 0.5
    POST_TREASURE_END_PENALTY_MIN = -1.5
    POST_TREASURE_END_PENALTY_MAX = -0.2
    FLICKER_BAD_DIST_THRESHOLD = 2500
    FLICKER_GOOD_DIST_THRESHOLD = 6000
    FLICKER_BUMP_PENALTY = -1.2
    FLICKER_LONG_MOVE_REWARD = 1.2
    FLICKER_PROGRESS_REWARD = 0.4
    FLICKER_NO_PROGRESS_PENALTY = -0.3
    EXPLORATION_NEW_GRID_REWARD = 1.2
    EXPLORATION_REPEAT_GRID_PENALTY_SCALE = 0.25
    EXPLORATION_REPEAT_GRID_PENALTY_MAX = 4.0
    GLOBAL_FIRST_VISIT_REWARD = 1.6
    GLOBAL_REPEAT_VISIT_PENALTY_SCALE = 0.08
    GLOBAL_REPEAT_VISIT_PENALTY_MAX = 1.2
    FRONTIER_PROGRESS_REWARD_SCALE = 0.12
    FRONTIER_PROGRESS_REWARD_MIN = -0.3
    FRONTIER_PROGRESS_REWARD_MAX = 0.3
    STEP_REWARD_BASE = 1.0
    STALL_DISTANCE_THRESHOLD = 3.0
    STALL_PENALTY = 1.0
    TREASURE_STALL_STEP_THRESHOLD = 150
    ALT_TOPK_TREASURE_INDEX = 2

    # Reward component weights.
    # 各奖励分量权重
    REWARD_WEIGHT_END_DIST = 0.8
    REWARD_WEIGHT_WIN = 1.0
    REWARD_WEIGHT_BUFF_DIST = 0.15
    REWARD_WEIGHT_BUFF = 0.5
    REWARD_WEIGHT_TREASURE_DIST = 0.9
    REWARD_WEIGHT_TREASURE = 1.6
    REWARD_WEIGHT_NEW_VISIBLE_TREASURE = 0.9
    REWARD_WEIGHT_TOPK_SEEN_TREASURE = 1.0
    REWARD_WEIGHT_FLICKER = 0.2
    REWARD_WEIGHT_POST_TREASURE_END = 0.6
    REWARD_WEIGHT_MOVE_STYLE = 0.12
    REWARD_WEIGHT_SMALL_LOOP = 0.3
    REWARD_WEIGHT_STEP = -0.001
    REWARD_WEIGHT_BUMP = -1.5
    REWARD_WEIGHT_MEMORY = -0.02
    REWARD_WEIGHT_EXPLORATION = 0.12
    REWARD_WEIGHT_VISION_EXPLORATION = 0.35
    REWARD_WEIGHT_FRONTIER_PROGRESS = 0.45
    REWARD_WEIGHT_LOWER_HALF_COVERAGE = 0.8
    REWARD_WEIGHT_RIGHT_LOWER_COVERAGE = 1.2
    REWARD_WEIGHT_EARLY_FINISH = 1.0
    REWARD_WEIGHT_STALL = 1.0

    # Initial learning rate
    # 初始的学习率
    START_LR = 1e-4

    # Configuration about kaiwu usage. The following configurations can be ignored
    # 关于开悟平台使用的配置，是可以忽略的配置，不需要改动
    SUB_ACTION_MASK_SHAPE = 0
    LSTM_HIDDEN_SHAPE = 0
    LSTM_CELL_SHAPE = 0
    OBSERVATION_SHAPE = 4500
    LEGAL_ACTION_SHAPE = 2
