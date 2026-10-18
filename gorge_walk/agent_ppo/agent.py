#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2025 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""

import os
import numpy as np
import torch

from kaiwu_agent.agent.base_agent import BaseAgent
from kaiwu_agent.utils.common_func import create_cls
from kaiwu_agent.agent.base_agent import (
    learn_wrapper,
    save_model_wrapper,
    load_model_wrapper,
    predict_wrapper,
    exploit_wrapper,
)

from agent_ppo.conf.conf import Config
from agent_ppo.algorithm.algorithm import Algorithm
from agent_ppo.feature.definition import observation_process as feature_observation_process

# 数据结构定义
ObsData = create_cls("ObsData", feature=None, legal_act=None)
ActData = create_cls("ActData", action=None, logprob=None, value=None)

class Agent(BaseAgent):
    def __init__(self, agent_type="player", device=None, logger=None, monitor=None) -> None:
        super().__init__(agent_type, device, logger, monitor)

        self.device = device
        self.logger = logger
        self.monitor = monitor
        
        # 必须初始化 action_size，供 observation_process 使用
        self.action_size = Config.DIM_OF_ACTION

        # PPO算法实例
        self.alg = Algorithm(
            device=device,
            logger=logger,
            monitor=monitor,
        )

        # 借鉴优点：评估阶段防卡死状态
        self._eval_prev_pos = None
        self._eval_stuck_steps = 0
        self._eval_recent_pos = []

    def _extract_pos_from_feature(self, feature):
        """ 从 250 维特征中还原坐标 (Row index 1:65, Col index 65:129) """
        row = int(np.argmax(feature[1:65]))
        col = int(np.argmax(feature[65:129]))
        return (row, col)

    @predict_wrapper
    def predict(self, list_obs_data):
        list_act_data = []
        for item in list_obs_data:
            # PPO 采样逻辑
            if isinstance(item, tuple):
                obs_data = item[0]  # 如果是元组 (ObsData, None)，取第一个
            else:
                obs_data = item     # 如果直接是 ObsData 对象，直接用
            # obs_data, _ = obs_tuple
            action, logprob, value = self.alg.predict(obs_data.feature, deterministic=False)
            list_act_data.append(ActData(action=action, logprob=logprob, value=value))

        return list_act_data, "ppo_v1"

    @exploit_wrapper
    def exploit(self, list_obs_data):
        """
        借鉴 DQN 优点：在评估阶段加入防卡死逻辑
        """
        list_act_data = []
        for item in list_obs_data:
            if isinstance(item, tuple):
                obs_data = item[0]  # 如果是元组 (ObsData, None)，取第一个
            else:
                obs_data = item     # 如果直接是 ObsData 对象，直接用
            # obs_data, _ = obs_tuple
            state = obs_data.feature
            # 1. 正常确定性预测
            action, logprob, value = self.alg.predict(state, deterministic=True)

            # 2. 优点借鉴：坐标监控与破局
            cur_pos = self._extract_pos_from_feature(state)
            
            # 检测是否陷入局部死循环
            self._eval_recent_pos.append(cur_pos)
            if len(self._eval_recent_pos) > 10: self._eval_recent_pos.pop(0)
            
            # 如果最近 10 步内同一个位置出现超过 4 次，说明在横跳
            if self._eval_recent_pos.count(cur_pos) > 4:
                # 尝试从合法动作里随机选一个，强行破局
                legal_indices = np.where(obs_data.legal_act > 0)[0]
                if len(legal_indices) > 1:
                    action = int(np.random.choice([a for a in legal_indices if a != action]))
                    if self.logger: self.logger.info(f"评估防卡死触发：强行切换动作至 {action}")

            list_act_data.append(ActData(action=action, logprob=logprob, value=value))

        return list_act_data, "ppo_v1"

    @learn_wrapper
    def learn(self, list_sample_data):
        self.alg.learn(list_sample_data)
    
    # def observation_process(self, raw_obs, extra_info=None):
    #     # 调用你写好的 250 维转换逻辑
    #     feature = feature_observation_process(raw_obs, extra_info)
        
    #     # 借鉴优点：根据视野实时计算合法动作掩码
    #     legal_act = np.ones(self.action_size, dtype=np.int64)
    #     if extra_info:
    #         view_radius = int(getattr(extra_info, "view", 2))
    #         side = view_radius * 2 + 1
    #         local_view = list(getattr(extra_info, "local_view", []))
            
    #         if len(local_view) == side * side:
    #             center = view_radius
    #             # 0:UP, 1:DOWN, 2:LEFT, 3:RIGHT (根据环境定义)
    #             if local_view[(center - 1) * side + center] == 0: legal_act[0] = 0
    #             if local_view[(center + 1) * side + center] == 0: legal_act[1] = 0
    #             if local_view[center * side + (center - 1)] == 0: legal_act[2] = 0
    #             if local_view[center * side + (center + 1)] == 0: legal_act[3] = 0

    #     return ObsData(feature=feature, legal_act=legal_act), None

    def observation_process(self, raw_obs, extra_info=None):
        """
        调用 definition.py 进行 250 维转换，并处理 legal_act
        """
        # 调用 definition 里的处理逻辑（feature_observation_process 是引用的别名）
        feature = feature_observation_process(raw_obs, extra_info)
        
        # 默认全部动作合法
        legal_act = np.ones(self.action_size, dtype=np.int64)
        
        if extra_info:
            # 这里的 logic 用于在推理时防止智能体撞墙
            view_radius = int(extra_info.view)
            side = view_radius * 2 + 1
            local_view = list(extra_info.local_view)
            
            if len(local_view) == side * side:
                center = view_radius
                # 将 local_view 展平索引转为 2D 理解
                # 0: UP, 1: DOWN, 2: LEFT, 3: RIGHT (根据环境定义确认顺序)
                # 假设视野中心为 [center, center]，上方为 [center-1, center]
                if local_view[(center - 1) * side + center] == 0: legal_act[0] = 0
                if local_view[(center + 1) * side + center] == 0: legal_act[1] = 0
                if local_view[center * side + (center - 1)] == 0: legal_act[2] = 0
                if local_view[center * side + (center + 1)] == 0: legal_act[3] = 0

        return ObsData(feature=feature, legal_act=legal_act), None 

    def _get_stage_target(self, feature):
        t_status = feature[240:250]
        collected = int(10 - np.sum(t_status))
        if collected < len(Config.OPTIMAL_TREASURE_ORDER):
            return Config.OPTIMAL_TREASURE_ORDER[collected]
        return None

    def action_process(self, act_data):
        if isinstance(act_data, list):
            act_data = act_data[0]
        return int(act_data.action)

    def reset(self):
        """
        每局开始时重置智能体状态。
        PPO 主要是基于当前观测做决策，目前不需要特殊的重置逻辑，
        但这个接口必须存在，否则 Workflow 会报错。
        """
        # 暂时什么都不用做
        pass

    @save_model_wrapper
    def save_model(self, path=None, id="latest"):
        """ 适配开悟平台的保存接口 """
        # path 通常是开悟平台提供的文件夹路径
        if not path:
            path = "./model"
        os.makedirs(path, exist_ok=True)
        
        full_path = os.path.join(path, f"ppo_model_{id}.pt")
        self.alg.save_model(path=full_path, id=id)

    @load_model_wrapper
    def load_model(self, path=None, id="latest"):
        """ 适配开悟平台的加载接口 """
        # 如果 path 指向的是文件夹，我们需要拼接文件名
        if os.path.isdir(path):
            full_path = os.path.join(path, f"ppo_model_{id}.pt")
        else:
            full_path = path

        if os.path.exists(full_path):
            self.alg.load_model(full_path)
        else:
            if self.logger:
                self.logger.warning(f"未找到模型文件: {full_path}")