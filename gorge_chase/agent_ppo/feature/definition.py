#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Data definitions and GAE computation for Gorge Chase PPO (refactored).

数据类定义与 GAE 计算（重构版）。

设计：
  - ObsData 携带两份观测：map_tensor 与 sym_feat；另含 rule_hints（不进网络）
  - SampleData 的 map_tensor 以扁平 1D 存储（MAP_FLAT_LEN），便于框架序列化
    模型 forward 时再 reshape 回 (C, H, W)
  - SampleData 的所有字段都用 1D 整数维度声明，框架按 create_cls 规约自动处理序列化
    （不需要手写 SampleData2NumpyData / NumpyData2SampleData）
"""

import numpy as np

from common_python.utils.common_func import create_cls
from agent_ppo.conf.conf import Config


# ObsData: map_tensor=(C,H,W), sym_feat=(SYM_FEATURE_LEN,), legal_action=(16,)
ObsData = create_cls(
    "ObsData",
    map_tensor=None,
    sym_feat=None,
    legal_action=None,
    rule_hints=None,
)

# ActData: action(final), d_action(greedy), prob(16D), value, rl_action(pre-rule for logging)
ActData = create_cls(
    "ActData",
    action=None,
    d_action=None,
    prob=None,
    value=None,
    rl_action=None,
)

# SampleData: flat storage for distributed transfer
SampleData = create_cls(
    "SampleData",
    map_tensor=Config.MAP_FLAT_LEN,      # 6*21*21 = 2646
    sym_feat=Config.SYM_FEATURE_LEN,     # 110
    legal_action=Config.ACTION_NUM,      # 16
    act=1,
    reward=Config.VALUE_NUM,
    reward_sum=Config.VALUE_NUM,
    done=1,
    value=Config.VALUE_NUM,
    next_value=Config.VALUE_NUM,
    advantage=Config.VALUE_NUM,
    prob=Config.ACTION_NUM,              # 16
)


def sample_process(list_sample_data):
    """Fill next_value and compute GAE advantage.

    填充 next_value 并使用 GAE 计算优势函数。
    约定：最后一个样本的 next_value 由调用方预先设置：
      - 真终局（terminated）= 0
      - 截断终局（truncated）= critic bootstrap value
    """
    for i in range(len(list_sample_data) - 1):
        list_sample_data[i].next_value = list_sample_data[i + 1].value
    _calc_gae(list_sample_data)
    return list_sample_data


def _calc_gae(list_sample_data):
    """Compute GAE (Generalized Advantage Estimation).

    计算广义优势估计（GAE）。
    """
    gamma = Config.GAMMA
    lamda = Config.LAMDA
    gae = 0.0
    for sample in reversed(list_sample_data):
        delta = -sample.value + sample.reward + gamma * sample.next_value
        gae = gae * gamma * lamda + delta
        sample.advantage = gae
        sample.reward_sum = gae + sample.value
