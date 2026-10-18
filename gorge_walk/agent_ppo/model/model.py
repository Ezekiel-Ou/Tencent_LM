#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2025 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""

import os
import sys
from typing import List

import torch
from torch import nn
from torch.distributions import Categorical

from agent_ppo.conf.conf import Config
from kaiwu_agent.utils.common_func import attached

if os.path.basename(sys.argv[0]) == "learner.py":
    torch.set_num_interop_threads(2)
    torch.set_num_threads(2)
else:
    torch.set_num_interop_threads(4)
    torch.set_num_threads(4)


def make_fc_layer(in_features: int, out_features: int):
    """
    创建并初始化一个全连接层
    """
    fc_layer = nn.Linear(in_features, out_features)
    nn.init.orthogonal_(fc_layer.weight)
    nn.init.zeros_(fc_layer.bias)
    return fc_layer


class MLP(nn.Module):
    def __init__(
        self,
        fc_feat_dim_list: List[int],
        name: str,
        non_linearity: nn.Module = nn.ReLU,
        non_linearity_last: bool = False,
    ):
        super().__init__()
        self.fc_layers = nn.Sequential()
        for i in range(len(fc_feat_dim_list) - 1):
            fc_layer = make_fc_layer(fc_feat_dim_list[i], fc_feat_dim_list[i + 1])
            self.fc_layers.add_module(f"{name}_fc{i + 1}", fc_layer)
            if i + 1 < len(fc_feat_dim_list) - 1 or non_linearity_last:
                self.fc_layers.add_module(f"{name}_non_linear{i + 1}", non_linearity())

    def forward(self, data):
        return self.fc_layers(data)


class Model(nn.Module):
    """
    PPO 用 Actor-Critic 网络
    输入：
        state: [batch, DIM_OF_OBSERVATION]
    输出：
        logits: [batch, DIM_OF_ACTION]
        value : [batch]
    """

    def __init__(self, state_shape=None, action_shape=None, softmax=False):
        super().__init__()

        self.feature_len = Config.DIM_OF_OBSERVATION
        self.action_dim = Config.DIM_OF_ACTION
        self.hidden_dim = Config.HIDDEN_DIM

        # 共享特征层
        self.feature_layer = nn.Sequential(
            make_fc_layer(self.feature_len, 256),
            nn.Tanh(),
            make_fc_layer(256, 256),
            nn.Tanh(),
            make_fc_layer(256, self.hidden_dim),
            nn.Tanh(),
        )

        # Actor 头：输出动作 logits
        self.actor_layer = nn.Sequential(
            make_fc_layer(self.hidden_dim, 64),
            nn.Tanh(),
            make_fc_layer(64, self.action_dim),
        )

        # Critic 头：输出状态价值 V(s)
        self.critic_layer = nn.Sequential(
            make_fc_layer(self.hidden_dim, 64),
            nn.Tanh(),
            make_fc_layer(64, 1),
        )

    def forward(self, feature):
        """
        返回:
            logits: [batch, action_dim]
            value : [batch]
        """
        x = self.feature_layer(feature)
        logits = self.actor_layer(x)
        value = self.critic_layer(x).squeeze(-1)
        return logits, value

    @torch.no_grad()
    def predict(self, feature, deterministic=False):
        """
        推理接口
        输入:
            feature: [batch, obs_dim] 或 [obs_dim]
        返回:
            action, logprob, value
        """
        if feature.dim() == 1:
            feature = feature.unsqueeze(0)

        logits, value = self.forward(feature)
        dist = Categorical(logits=logits)

        if deterministic:
            action = torch.argmax(logits, dim=-1)
        else:
            action = dist.sample()

        logprob = dist.log_prob(action)
        return action, logprob, value

    def evaluate_actions(self, feature, action):
        """
        PPO 更新时使用
        输入:
            feature: [batch, obs_dim]
            action : [batch]
        返回:
            logprob: [batch]
            entropy: [batch]
            value  : [batch]
        """
        logits, value = self.forward(feature)
        dist = Categorical(logits=logits)
        logprob = dist.log_prob(action)
        entropy = dist.entropy()
        return logprob, entropy, value