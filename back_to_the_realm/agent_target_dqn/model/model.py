#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2025 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""


import torch
import numpy as np
from torch import nn


class Model(nn.Module):
    def __init__(
        self,
        state_shape,
        action_shape=0,
        softmax=False,
        cnn_embed_dim=96,
        backbone_hidden_dim=192,
        head_hidden_dim=128,
    ):
        super().__init__()
        cnn_layer1 = [
            nn.Conv2d(4, 8, kernel_size=3, stride=2, padding=2),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
        ]
        cnn_layer2 = [
            nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
        ]
        cnn_layer3 = [
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        ]
        cnn_layer4 = [
            nn.Conv2d(32, 4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(4),
            nn.ReLU(inplace=True),
        ]
        cnn_flatten = [nn.Flatten(), nn.Linear(256, cnn_embed_dim), nn.ReLU(inplace=True)]

        self.cnn_layer = cnn_layer1 + cnn_layer2 + cnn_layer3 + cnn_layer4 + cnn_flatten
        self.cnn_model = nn.Sequential(*self.cnn_layer)

        self.backbone = nn.Sequential(
            nn.Linear(np.prod(state_shape), backbone_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(backbone_hidden_dim, head_hidden_dim),
            nn.ReLU(inplace=True),
        )

        action_dim = int(np.prod(action_shape)) if action_shape else 0
        self.value_head = nn.Linear(head_hidden_dim, 1) if action_dim else None
        self.advantage_head = nn.Linear(head_hidden_dim, action_dim) if action_dim else None
        self.softmax = nn.Softmax(dim=-1) if softmax else None

        self.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    # Forward inference
    # 前向推理
    def forward(self, s, state=None, info=None):
        feature_vec, feature_maps = s[0], s[1]
        feature_maps = self.cnn_model(feature_maps)

        feature_maps = feature_maps.view(feature_maps.shape[0], -1)

        concat_feature = torch.concat([feature_vec, feature_maps], dim=1)

        hidden = self.backbone(concat_feature)
        if self.advantage_head is None:
            logits = hidden
        else:
            value = self.value_head(hidden)
            advantage = self.advantage_head(hidden)
            logits = value + advantage - advantage.mean(dim=1, keepdim=True)

        if self.softmax is not None:
            logits = self.softmax(logits)
        return logits, state
