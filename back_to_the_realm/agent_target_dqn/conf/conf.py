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

    MODEL_CNN_EMBED_DIM = 64
    MODEL_BACKBONE_HIDDEN_DIM = 128
    MODEL_HEAD_HIDDEN_DIM = 64

    # Input dimension of reverb sample on learner. Note that different algorithms have different dimensions.
    # For example, the dimension of dqn in the sample code is 21624, and the dimension of target_dqn is 21624
    # **Note**, this item must be configured correctly and should be aligned with the NumpyData2SampleData function data in definition.py
    # Otherwise the sample dimension error may be reported
    # learner上reverb样本的输入维度, 注意不同的算法维度不一样, 比如示例代码中dqn的维度是21624, target_dqn的维度是21624
    # **注意**，此项必须正确配置，应该与definition.py中的NumpyData2SampleData函数数据对齐，否则可能报样本维度错误
    SAMPLE_DIM = 21624

    # field of view radius
    # 视野域半径
    VIEW_SIZE = 25

    # Size of observation after CNN processing and vector feature concatenation.
    # The raw feature serialized into samples remains 10808.
    # observation的维度是CNN处理后的地图特征与原始向量特征拼接后的维度，序列化进样本的原始特征仍为10808。
    DIM_OF_OBSERVATION = MODEL_CNN_EMBED_DIM + 404

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
    EPSILON_GREEDY_PROBABILITY = 300000

    # Discount factor GAMMA in RL
    # RL中的回报折扣GAMMA
    GAMMA = 0.995

    # epsilon initial value
    # epsilon初始值
    EPSILON = 1.0

    # epsilon minimum value
    # epsilon最小值
    EPSILON_MIN = 0.05

    # target network update frequency
    # target网络更新频率
    TARGET_UPDATE_FREQ = 500

    # soft update ratio for target network
    # target网络软更新系数
    TARGET_SOFT_TAU = 0.01

    # Initial learning rate
    # 初始的学习率
    START_LR = 1e-4

    # Number of gradient updates per learner batch. DQN is off-policy, so a small amount of batch reuse improves
    # sample efficiency under a fixed training time budget.
    # 每个learner batch重复学习次数。DQN为离策略算法，少量复用batch可以提升固定训练时长下的样本利用率。
    LEARN_REPEAT = 2

    # Configuration about kaiwu usage. The following configurations can be ignored
    # 关于开悟平台使用的配置，是可以忽略的配置，不需要改动
    SUB_ACTION_MASK_SHAPE = 0
    LSTM_HIDDEN_SHAPE = 0
    LSTM_CELL_SHAPE = 0
    OBSERVATION_SHAPE = 4500
    LEGAL_ACTION_SHAPE = 2
