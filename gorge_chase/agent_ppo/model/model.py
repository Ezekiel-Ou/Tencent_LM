#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
CNN + MLP dual-branch Actor-Critic for Gorge Chase PPO.

CNN（21×21×6 多通道）+ MLP（130D 符号特征）双分支 Actor-Critic。

输入：
  - map_tensor: (B, C, H, W) 或 (B, C*H*W) 扁平向量（算法侧传 flat，forward 内 reshape）
  - sym_feat:   (B, SYM_FEATURE_LEN)

输出：
  - logits: (B, ACTION_NUM)  # 16
  - value:  (B, VALUE_NUM)   # 1
"""

import torch
import torch.nn as nn

from agent_ppo.conf.conf import Config


def make_fc_layer(in_features, out_features):
    """Create a linear layer with orthogonal init (Actor-Critic common practice).

    正交初始化的线性层（RL 稳定性常用）。
    """
    fc = nn.Linear(in_features, out_features)
    nn.init.orthogonal_(fc.weight.data)
    nn.init.zeros_(fc.bias.data)
    return fc


def make_conv_layer(in_ch, out_ch, kernel=3, stride=1, padding=1):
    conv = nn.Conv2d(in_ch, out_ch, kernel_size=kernel, stride=stride, padding=padding)
    nn.init.orthogonal_(conv.weight.data)
    nn.init.zeros_(conv.bias.data)
    return conv


class Model(nn.Module):
    """Dual-branch Actor-Critic.

    CNN 分支提取 21×21 多通道空间特征；
    MLP 分支处理 130D 符号特征（rays/BFS/top-K 宝箱等 + 第二只怪刷新提示 + 防磨蹭提示）；
    融合后再过共享 MLP，最终分 Actor / Critic 双头。
    """

    def __init__(self, device=None):
        super().__init__()
        self.model_name = "gorge_chase_cnn_mlp"
        self.device = device

        C = Config.MAP_CHANNELS
        V = Config.MAP_VIEW
        S = Config.SYM_FEATURE_LEN
        A = Config.ACTION_NUM
        Vn = Config.VALUE_NUM

        self._map_shape = (C, V, V)

        # CNN 分支：21×21×6 -> (32,)
        self.cnn = nn.Sequential(
            make_conv_layer(C, 16, 3, 1, 1),
            nn.ReLU(inplace=True),
            make_conv_layer(16, 32, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                    # 21 -> 10
            make_conv_layer(32, 32, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),            # -> (B, 32, 1, 1)
            nn.Flatten(),                        # -> (B, 32)
        )

        # MLP 分支：130D -> 64D
        self.mlp = nn.Sequential(
            make_fc_layer(S, 128),
            nn.ReLU(inplace=True),
            make_fc_layer(128, 64),
            nn.ReLU(inplace=True),
        )

        # 融合层
        fused_dim = 32 + 64  # 96
        self.fusion = nn.Sequential(
            make_fc_layer(fused_dim, 128),
            nn.ReLU(inplace=True),
            make_fc_layer(128, 64),
            nn.ReLU(inplace=True),
        )

        # 双头
        self.actor_head = make_fc_layer(64, A)
        self.critic_head = make_fc_layer(64, Vn)

    def _reshape_map(self, map_tensor):
        if map_tensor.dim() == 2:
            # (B, C*H*W) -> (B, C, H, W)
            return map_tensor.view(-1, *self._map_shape)
        return map_tensor

    def forward(self, map_tensor, sym_feat, inference=False):
        m = self._reshape_map(map_tensor)
        cnn_feat = self.cnn(m)
        mlp_feat = self.mlp(sym_feat)
        fused = torch.cat([cnn_feat, mlp_feat], dim=1)
        h = self.fusion(fused)
        logits = self.actor_head(h)
        value = self.critic_head(h)
        return logits, value

    def set_train_mode(self):
        self.train()

    def set_eval_mode(self):
        self.eval()
