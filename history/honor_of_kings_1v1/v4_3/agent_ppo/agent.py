#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""


import torch

torch.set_num_threads(1)
torch.set_num_interop_threads(1)

import os
import random
from agent_ppo.model.model import Model
from agent_ppo.feature.definition import *
import numpy as np
from kaiwudrl.interface.agent import BaseAgent

from agent_ppo.conf.conf import Config
from agent_ppo.feature.reward_process import GameRewardManager
from torch.optim.lr_scheduler import LambdaLR
from agent_ppo.algorithm.algorithm import Algorithm
from agent_ppo.feature.feature_process import FeatureProcess


# Available summoner skills / 可选召唤师技能
SUMMONER_SKILL_MAP = {
    80102: "治疗",
    80109: "疾跑",
    80104: "惩击",
    80108: "终结",
    80110: "狂暴",
    80105: "干扰",
    80103: "晕眩",
    80107: "净化",
    80121: "弱化",
    80115: "闪现",
}
SUMMONER_SKILL_IDS = list(SUMMONER_SKILL_MAP.keys())


class Agent(BaseAgent):
    def __init__(self, agent_type="player", device=None, logger=None, monitor=None):
        self.cur_model_name = ""
        self.device = device
        # Create Model and convert the model to achannel-last memory format to achieve better performance.
        # 创建模型, 将模型转换为通道后内存格式，以获得更好的性能。
        self.model = Model().to(self.device)
        self.model = self.model.to(memory_format=torch.channels_last)

        # config info
        # 配置信息
        self.lstm_unit_size = Config.LSTM_UNIT_SIZE
        self.lstm_hidden = np.zeros([self.lstm_unit_size])
        self.lstm_cell = np.zeros([self.lstm_unit_size])
        self.label_size_list = Config.LABEL_SIZE_LIST
        self.legal_action_size = Config.LEGAL_ACTION_SIZE_LIST
        self.seri_vec_split_shape = Config.SERI_VEC_SPLIT_SHAPE

        # env info
        # 环境信息
        self.hero_camp = 0
        self.player_id = 0
        self.env_id = None

        # learning info
        # 学习信息
        self.train_step = 0
        self.lr = Config.INIT_LEARNING_RATE_START
        parameters = self.model.parameters()
        self.optimizer = torch.optim.Adam(params=parameters, lr=self.lr, betas=(0.9, 0.999), eps=1e-8)
        self.parameters = [p for param_group in self.optimizer.param_groups for p in param_group["params"]]
        self.target_lr = Config.TARGET_LR
        self.target_step = Config.TARGET_STEP
        self.scheduler = LambdaLR(self.optimizer, lr_lambda=self.lr_lambda)

        # tools
        # 工具
        self.reward_manager = None
        self.logger = logger
        self.monitor = monitor

        self.algorithm = Algorithm(self.model, self.optimizer, self.scheduler, self.device, self.logger, self.monitor)

        super().__init__(agent_type, device, logger, monitor)

    def lr_lambda(self, step):
        # Define learning rate decay function
        # 定义学习率衰减函数
        if step > self.target_step:
            return self.target_lr / self.lr
        else:
            return 1.0 - ((1.0 - self.target_lr / self.lr) * step / self.target_step)

    def init_config(self, config_data):
        # Select summoner skill for each hero based on hero lineup of both camps
        # 根据双方阵营英雄阵容，为己方每个英雄选择召唤师技能
        my_heroes = config_data.get("my_heroes", [])
        select_skills = {}
        for hero_id in my_heroes:
            skill_id = random.choice(SUMMONER_SKILL_IDS)
            select_skills[hero_id] = skill_id
        return select_skills

    def reset(self, observation):
        # Reset function, called at the beginning of each episode
        # 重置函数，每局开始时调用
        self.hero_camp = observation["camp"]
        self.player_id = observation["player_id"]
        self.lstm_hidden = np.zeros([self.lstm_unit_size])
        self.lstm_cell = np.zeros([self.lstm_unit_size])
        self.reward_manager = GameRewardManager(self.player_id)
        self.feature_processes = FeatureProcess(self.hero_camp)

    def _model_inference(self, list_obs_data):
        # Using the network for inference
        # 使用网络进行推理
        feature = [obs_data.feature for obs_data in list_obs_data]
        legal_action = [obs_data.legal_action for obs_data in list_obs_data]
        lstm_cell = [obs_data.lstm_cell for obs_data in list_obs_data]
        lstm_hidden = [obs_data.lstm_hidden for obs_data in list_obs_data]

        input_list = [np.array(feature), np.array(lstm_cell), np.array(lstm_hidden)]
        torch_inputs = [torch.from_numpy(nparr).to(torch.float32) for nparr in input_list]
        for i, data in enumerate(torch_inputs):
            data = data.reshape(-1)
            torch_inputs[i] = data.float()

        feature, lstm_cell, lstm_hidden = torch_inputs
        feature_vec = feature.reshape(-1, self.seri_vec_split_shape[0][0])
        lstm_hidden_state = lstm_hidden.reshape(-1, self.lstm_unit_size)
        lstm_cell_state = lstm_cell.reshape(-1, self.lstm_unit_size)

        format_inputs = [feature_vec, lstm_hidden_state, lstm_cell_state]

        self.model.set_eval_mode()
        with torch.no_grad():
            output_list = self.model(format_inputs, inference=True)

        np_output = []
        for output in output_list:
            np_output.append(output.numpy())

        logits, value, _lstm_cell, _lstm_hidden = np_output[:4]

        _lstm_cell = _lstm_cell.squeeze(axis=0)
        _lstm_hidden = _lstm_hidden.squeeze(axis=0)

        list_act_data = list()
        for i in range(len(legal_action)):
            prob, d_prob, action, d_action = self._sample_masked_action(logits[i], legal_action[i])
            list_act_data.append(
                ActData(
                    action=action,
                    d_action=d_action,
                    prob=prob,
                    d_prob=d_prob,
                    value=value,
                    lstm_cell=_lstm_cell[i],
                    lstm_hidden=_lstm_hidden[i],
                )
            )
        return list_act_data

    def predict(self, observation):
        # Prediction function, usually called during training
        # Returns a random sampling action
        # 预测函数，通常在训练时调用，返回随机采样动作
        obs_data = self.observation_process(observation)
        act_data = self._model_inference([obs_data])[0]
        self.update_status(obs_data, act_data)
        action = self.action_process(observation, act_data, True)
        return action

    def exploit(self, observation):
        # Exploitation function, usually called during evaluation
        # Returns the action with the highest probability
        # 利用函数，在评估时调用，返回最大概率动作
        obs_data = self.observation_process(observation)
        act_data = self._model_inference([obs_data])[0]
        self.update_status(obs_data, act_data)
        d_action = self.action_process(observation, act_data, False)
        return d_action

    def observation_process(self, observation):
        feature = self.feature_processes.process_feature(observation)
        feature_vec, legal_action = (
            feature,
            observation["legal_action"],
        )
        return ObsData(
            feature=feature_vec, legal_action=legal_action, lstm_cell=self.lstm_cell, lstm_hidden=self.lstm_hidden
        )

    def action_process(self, observation, act_data, is_stochastic):
        if is_stochastic:
            # Use stochastic sampling action
            # 采用随机采样动作 action
            return act_data.action
        else:
            # Use the action with the highest probability
            # 采用最大概率动作 d_action
            return act_data.d_action

    def learn(self, list_sample_data):
        return self.algorithm.learn(list_sample_data)

    def save_model(self, path=None, id="1"):
        # To save the model, it can consist of multiple files, and it is important to ensure that
        #  each filename includes the "model.ckpt-id" field.
        # 保存模型, 可以是多个文件, 需要确保每个文件名里包括了model.ckpt-id字段
        model_file_path = f"{path}/model.ckpt-{str(id)}.pkl"
        torch.save(self.model.state_dict(), model_file_path)
        self.logger.info(f"save model {model_file_path} successfully")

    def load_model(self, path=None, id="1"):
        # When loading the model, you can load multiple files, and it is important to ensure that
        # each filename matches the one used during the save_model process.
        # 加载模型, 可以加载多个文件, 注意每个文件名需要和save_model时保持一致
        model_file_path = f"{path}/model.ckpt-{str(id)}.pkl"
        if self.cur_model_name == model_file_path:
            self.logger.info(f"current model is {model_file_path}, so skip load model")
        else:
            self.model.load_state_dict(
                torch.load(
                    model_file_path,
                    map_location=self.device,
                )
            )
            self.cur_model_name = model_file_path
            self.logger.info(f"load model {model_file_path} successfully")

    def load_opponent_agent(self, id="1"):
        # Framework provides loading opponent agent function, no need to implement function content
        # 框架提供的加载对手模型功能，无需实现函数内容
        pass

    def update_status(self, obs_data, act_data):
        self.obs_data = obs_data
        self.act_data = act_data
        self.lstm_cell = act_data.lstm_cell
        self.lstm_hidden = act_data.lstm_hidden

    def _sample_masked_action(self, logits, legal_action):
        """
        Sample actions from predicted logits and legal actions
        return: probability, stochastic and deterministic actions with additional list
        """
        """
        从预测的logits和合法动作中采样动作
        返回：以列表形式概率、随机和确定性动作
        """

        prob_list = []
        d_prob_list = []
        action_list = []
        d_action_list = []
        label_split_size = [sum(self.label_size_list[: index + 1]) for index in range(len(self.label_size_list))]
        legal_actions = np.split(legal_action, label_split_size[:-1])
        logits_split = np.split(logits, label_split_size[:-1])
        for index in range(0, len(self.label_size_list) - 1):
            probs = self._legal_soft_max(logits_split[index], legal_actions[index])
            prob_list += list(probs)
            d_prob_list += list(probs)
            sample_action = self._legal_sample(probs, use_max=False)
            action_list.append(sample_action)
            d_action = self._legal_sample(probs, use_max=True)
            d_action_list.append(d_action)

        # deals with the last prediction, target
        # 处理最后的预测，目标
        index = len(self.label_size_list) - 1
        target_legal_action_o = np.reshape(
            legal_actions[index],
            [
                self.legal_action_size[0],
                self.legal_action_size[-1] // self.legal_action_size[0],
            ],
        )
        one_hot_actions = np.eye(self.label_size_list[0])[action_list[0]]
        one_hot_actions = np.reshape(one_hot_actions, [self.label_size_list[0], 1])
        target_legal_action = np.sum(target_legal_action_o * one_hot_actions, axis=0)

        legal_actions[index] = target_legal_action
        probs = self._legal_soft_max(logits_split[-1], target_legal_action)
        prob_list += list(probs)
        sample_action = self._legal_sample(probs, use_max=False)
        action_list.append(sample_action)

        one_hot_actions = np.eye(self.label_size_list[0])[d_action_list[0]]
        one_hot_actions = np.reshape(one_hot_actions, [self.label_size_list[0], 1])
        target_legal_action_d = np.sum(target_legal_action_o * one_hot_actions, axis=0)

        probs = self._legal_soft_max(logits_split[-1], target_legal_action_d)
        d_prob_list += list(probs)

        d_action = self._legal_sample(probs, use_max=True)
        d_action_list.append(d_action)

        return [prob_list], [d_prob_list], action_list, d_action_list

    def _legal_soft_max(self, input_hidden, legal_action):
        _lsm_const_w, _lsm_const_e = 1e20, 1e-5
        _lsm_const_e = 0.00001

        tmp = input_hidden - _lsm_const_w * (1.0 - legal_action)
        tmp_max = np.max(tmp, keepdims=True)
        tmp = np.clip(tmp - tmp_max, -_lsm_const_w, 1)
        tmp = (np.exp(tmp) + _lsm_const_e) * legal_action
        probs = tmp / np.sum(tmp, keepdims=True)
        return probs

    def _legal_sample(self, probs, legal_action=None, use_max=False):
        # Sample with probability, input probs should be 1D array
        # 根据概率采样，输入的probs应该是一维数组
        if use_max:
            return np.argmax(probs)

        return np.argmax(np.random.multinomial(1, probs, size=1))
