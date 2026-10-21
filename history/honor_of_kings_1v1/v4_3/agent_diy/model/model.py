#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
PPO policy/value model for the frozen 4833-dim agent_diy schema.
"""

import math
from typing import List

import numpy as np
import torch
import torch.nn as nn
from torch.nn import ModuleDict

from agent_diy.conf.conf import Args, Config


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.model_name = Config.NETWORK_NAME
        self.data_split_shape = Config.DATA_SPLIT_SHAPE
        self.lstm_time_steps = Config.LSTM_TIME_STEPS
        self.lstm_unit_size = Config.LSTM_UNIT_SIZE
        self.dim_public = Config.DIM_PUBLIC
        self.seri_vec_split_shape = Config.SERI_VEC_SPLIT_SHAPE
        self.m_var_beta = Config.BETA_START
        self.log_epsilon = Config.LOG_EPSILON
        self.label_size_list = Config.LABEL_SIZE_LIST
        self.is_reinforce_task_list = Config.IS_REINFORCE_TASK_LIST
        self.min_policy = Config.MIN_POLICY
        self.clip_param = Config.CLIP_PARAM
        self.dual_clip_param = Config.DUAL_CLIP_PARAM
        self.var_beta = self.m_var_beta
        self.target_embed_dim = Config.TARGET_EMBED_DIM
        self.cut_points = [value[0] for value in Config.data_shapes]
        self.legal_action_size = Config.LEGAL_ACTION_SIZE_LIST

        self.feature_dim = Config.SERI_VEC_SPLIT_SHAPE[0][0]
        self.legal_action_dim = np.sum(Config.LEGAL_ACTION_SIZE_LIST)
        self.lstm_hidden_dim = Config.LSTM_UNIT_SIZE

        self.unit_no_pos_dim = Args.DIM_UNIT_CORE - Args.DIM_POSITION
        self.position_mlp = MLP([Args.DIM_POSITION, 128, 64], "position_mlp")
        self.unit_no_pos_mlp = MLP([self.unit_no_pos_dim, 64, 32], "unit_no_pos_mlp")
        self.unit_embed_dim = self.unit_no_pos_dim + 32 + 64

        self.global_mlp = MLP([Args.DIM_GLOBAL, 128, 64], "global_mlp", non_linearity_last=True)
        self.hero_mlp = MLP(
            [self.unit_embed_dim + Args.DIM_HERO - Args.DIM_UNIT_CORE, 512, 256, 128],
            "hero_mlp",
            non_linearity_last=True,
        )
        self.soldier_mlp = MLP(
            [self.unit_embed_dim + Args.DIM_SOLDIER - Args.DIM_UNIT_CORE, 192, 96, 64],
            "soldier_mlp",
            non_linearity_last=True,
        )
        self.monster_mlp = MLP(
            [self.unit_embed_dim + Args.DIM_MONSTER - Args.DIM_UNIT_CORE, 192, 96, 64],
            "monster_mlp",
            non_linearity_last=True,
        )
        self.tower_mlp = MLP(
            [self.unit_embed_dim + Args.DIM_TOWER - Args.DIM_UNIT_CORE, 192, 96, 64],
            "tower_mlp",
            non_linearity_last=True,
        )
        self.bullet_mlp = MLP([Args.DIM_BULLET, 192, 96, 64], "bullet_mlp", non_linearity_last=True)
        self.target_mlp = MLP([Args.DIM_TARGET, 64, self.target_embed_dim], "target_mlp", non_linearity_last=True)

        concat_dim = 64 + 128 * 2 + 128 * 2 + 64 + 64 * 2 + 128 + self.target_embed_dim * 2
        self.concat_mlp = MLP([concat_dim, self.lstm_unit_size], "concat_mlp", non_linearity_last=True, use_layer_norm=True)
        self.concat_mlp_other = MLP([concat_dim, 512, self.lstm_unit_size], "concat_other_mlp", use_layer_norm=True)
        self.lstm_and_linear_mlp = MLP(
            [self.lstm_unit_size * 2, self.dim_public],
            "lstm_and_linear_mlp",
            non_linearity_last=True,
            use_layer_norm=True,
        )

        self.lstm = nn.LSTM(
            input_size=self.lstm_unit_size,
            hidden_size=self.lstm_unit_size,
            num_layers=1,
            bias=True,
            batch_first=True,
            dropout=Config.LSTM_DROPOUT,
            bidirectional=False,
        )

        self.label_mlp = ModuleDict(
            {
                f"hero_label{label_index}_mlp": MLP(
                    [self.dim_public, self.label_size_list[label_index]],
                    f"hero_label{label_index}_mlp",
                )
                for label_index in range(len(self.label_size_list) - 1)
            }
        )
        self.lstm_tar_embed_mlp = make_fc_layer(self.dim_public, self.target_embed_dim)
        self.target_query_mlp = make_fc_layer(self.target_embed_dim, self.target_embed_dim, use_bias=False)
        self.value_mlp = MLP([self.dim_public, 64, 1], "hero_value_mlp")

    def _encode_unit_feature(self, x, mlp):
        core = x[..., : Args.DIM_UNIT_CORE]
        extra = x[..., Args.DIM_UNIT_CORE :]
        unit_no_pos = core[..., : self.unit_no_pos_dim]
        position = core[..., self.unit_no_pos_dim : Args.DIM_UNIT_CORE]
        unit_no_pos_embed = self.unit_no_pos_mlp(unit_no_pos)
        position_embed = self.position_mlp(position)
        return mlp(torch.cat([unit_no_pos, unit_no_pos_embed, position_embed, extra], dim=-1))

    def _pool_instances(self, encoded, use_mean=True):
        max_pool = encoded.max(dim=1).values
        if not use_mean:
            return max_pool
        mean_pool = encoded.mean(dim=1)
        return torch.cat([max_pool, mean_pool], dim=-1)

    def _split_feature(self, feature_vec):
        split_sizes = [
            Args.DIM_GLOBAL,
            Args.DIM_HERO,
            Args.DIM_HERO,
            Args.DIM_SOLDIER * Args.SOLDIER_PER_SIDE,
            Args.DIM_SOLDIER * Args.SOLDIER_PER_SIDE,
            Args.DIM_MONSTER,
            Args.DIM_TOWER,
            Args.DIM_TOWER,
            Args.DIM_BULLET * Args.BULLET_SLOTS,
            Args.DIM_TARGET * Args.TARGET_SLOTS,
        ]
        return feature_vec.split(split_sizes, dim=1)

    def forward(self, data_list, inference=False):
        feature_vec, lstm_hidden_init, lstm_cell_init = data_list
        if feature_vec.shape[-1] != self.feature_dim:
            raise ValueError(f"model feature dim mismatch: {feature_vec.shape[-1]} != {self.feature_dim}")

        (
            global_vec,
            self_hero_vec,
            enemy_hero_vec,
            self_soldier_vec,
            enemy_soldier_vec,
            monster_vec,
            self_tower_vec,
            enemy_tower_vec,
            bullet_vec,
            target_vec,
        ) = self._split_feature(feature_vec)

        batch_steps = feature_vec.shape[0]
        global_result = self.global_mlp(global_vec)
        self_hero_result = self._encode_unit_feature(self_hero_vec, self.hero_mlp)
        enemy_hero_result = self._encode_unit_feature(enemy_hero_vec, self.hero_mlp)

        self_soldiers = self_soldier_vec.reshape(batch_steps, Args.SOLDIER_PER_SIDE, Args.DIM_SOLDIER)
        enemy_soldiers = enemy_soldier_vec.reshape(batch_steps, Args.SOLDIER_PER_SIDE, Args.DIM_SOLDIER)
        self_soldier_flat = self_soldiers.reshape(batch_steps * Args.SOLDIER_PER_SIDE, Args.DIM_SOLDIER)
        enemy_soldier_flat = enemy_soldiers.reshape(batch_steps * Args.SOLDIER_PER_SIDE, Args.DIM_SOLDIER)
        self_soldier_result = self._encode_unit_feature(self_soldier_flat, self.soldier_mlp).reshape(
            batch_steps, Args.SOLDIER_PER_SIDE, 64
        )
        enemy_soldier_result = self._encode_unit_feature(enemy_soldier_flat, self.soldier_mlp).reshape(
            batch_steps, Args.SOLDIER_PER_SIDE, 64
        )
        self_soldier_pool = self._pool_instances(self_soldier_result)
        enemy_soldier_pool = self._pool_instances(enemy_soldier_result)

        monster_result = self._encode_unit_feature(monster_vec, self.monster_mlp)
        self_tower_result = self._encode_unit_feature(self_tower_vec, self.tower_mlp)
        enemy_tower_result = self._encode_unit_feature(enemy_tower_vec, self.tower_mlp)

        bullets = bullet_vec.reshape(batch_steps, Args.BULLET_SLOTS, Args.DIM_BULLET)
        bullet_flat = bullets.reshape(batch_steps * Args.BULLET_SLOTS, Args.DIM_BULLET)
        bullet_result = self.bullet_mlp(bullet_flat).reshape(batch_steps, Args.BULLET_SLOTS, 64)
        bullet_pool = self._pool_instances(bullet_result)

        targets = target_vec.reshape(batch_steps, Args.TARGET_SLOTS, Args.DIM_TARGET)
        target_embedding = self.target_mlp(targets)
        target_summary = torch.cat(
            [target_embedding.max(dim=1).values, target_embedding.mean(dim=1)],
            dim=-1,
        )

        concat_result = torch.cat(
            [
                global_result,
                self_hero_result,
                enemy_hero_result,
                self_soldier_pool,
                enemy_soldier_pool,
                monster_result,
                self_tower_result,
                enemy_tower_result,
                bullet_pool,
                target_summary,
            ],
            dim=1,
        )

        fc_public_result = self.concat_mlp(concat_result)
        batch_size = lstm_hidden_init.shape[0]
        total_steps = fc_public_result.shape[0]
        if batch_size <= 0 or total_steps % batch_size != 0:
            raise ValueError(f"invalid lstm batch/step shape: batch={batch_size}, total_steps={total_steps}")
        seq_len = total_steps // batch_size
        reshape_fc_public_result = fc_public_result.reshape(batch_size, seq_len, self.lstm_unit_size)
        lstm_initial_state = (lstm_hidden_init.unsqueeze(0), lstm_cell_init.unsqueeze(0))
        lstm_outputs, state = self.lstm(reshape_fc_public_result, lstm_initial_state)
        lstm_hidden_output, lstm_cell_output = state[0], state[1]
        lstm_outputs = lstm_outputs.reshape(-1, self.lstm_unit_size)

        public_mlp_result = self.concat_mlp_other(concat_result)
        public_hidden = self.lstm_and_linear_mlp(torch.cat([lstm_outputs, public_mlp_result], dim=-1))
        public_hidden = public_hidden.reshape(-1, self.dim_public)

        result_list = []
        for label_index in range(len(self.label_size_list) - 1):
            label_logits = self.label_mlp[f"hero_label{label_index}_mlp"](public_hidden)
            if label_index == 0:
                button_bias = torch.as_tensor(
                    Config.BUTTON_LOGIT_BIAS,
                    dtype=label_logits.dtype,
                    device=label_logits.device,
                )
                label_logits = label_logits + button_bias
            result_list.append(label_logits)

        q = self.target_query_mlp(target_embedding)
        k = self.lstm_tar_embed_mlp(public_hidden).unsqueeze(2)
        target_logits = torch.matmul(q, k).reshape(-1, self.label_size_list[-1]) / math.sqrt(self.target_embed_dim)
        result_list.append(target_logits)

        value_result = self.value_mlp(public_hidden)
        result_list.append(value_result)

        logits = torch.flatten(torch.cat(result_list[:-1], 1), start_dim=1)
        value = result_list[-1]
        if inference:
            return [logits, value, lstm_cell_output, lstm_hidden_output]
        return result_list

    def compute_loss(self, data_list, rst_list):
        seri_vec = data_list[0].reshape(-1, self.data_split_shape[0])
        usq_reward = data_list[1].reshape(-1, self.data_split_shape[1])
        usq_advantage = data_list[2].reshape(-1, self.data_split_shape[2])
        usq_is_train = data_list[-3].reshape(-1, self.data_split_shape[-3])

        usq_label_list = data_list[3 : 3 + len(self.label_size_list)]
        for shape_index in range(len(self.label_size_list)):
            usq_label_list[shape_index] = usq_label_list[shape_index].reshape(
                -1, self.data_split_shape[3 + shape_index]
            ).long()

        old_label_probability_list = data_list[3 + len(self.label_size_list) : 3 + 2 * len(self.label_size_list)]
        for shape_index in range(len(self.label_size_list)):
            old_label_probability_list[shape_index] = old_label_probability_list[shape_index].reshape(
                -1, self.data_split_shape[3 + len(self.label_size_list) + shape_index]
            )

        usq_weight_list = data_list[3 + 2 * len(self.label_size_list) : 3 + 3 * len(self.label_size_list)]
        for shape_index in range(len(self.label_size_list)):
            usq_weight_list[shape_index] = usq_weight_list[shape_index].reshape(
                -1,
                self.data_split_shape[3 + 2 * len(self.label_size_list) + shape_index],
            )

        reward = usq_reward.squeeze(dim=1)
        advantage = usq_advantage.squeeze(dim=1)
        frame_is_train = usq_is_train.squeeze(dim=1)
        raw_advantage = advantage.clone()
        if Config.USE_ADVANTAGE_NORM:
            valid_adv = advantage[frame_is_train > 0]
            if valid_adv.numel() > 1:
                advantage = (advantage - valid_adv.mean()) / (valid_adv.std(unbiased=False) + 1e-8)

        label_list = [ele.squeeze(dim=1) for ele in usq_label_list]
        weight_list = [weight.squeeze(dim=1) for weight in usq_weight_list]
        label_result = rst_list[:-1]
        value_result = rst_list[-1]

        _, split_feature_legal_action = torch.split(
            seri_vec,
            [np.prod(self.seri_vec_split_shape[0]), np.prod(self.seri_vec_split_shape[1])],
            dim=1,
        )
        feature_legal_action_shape = list(self.seri_vec_split_shape[1])
        feature_legal_action_shape.insert(0, -1)
        feature_legal_action = split_feature_legal_action.reshape(feature_legal_action_shape)
        legal_action_flag_list = torch.split(feature_legal_action, self.label_size_list, dim=1)

        value_result_squeezed = value_result.squeeze(dim=1)
        value_loss_unclipped = torch.square(reward - value_result_squeezed)
        if Config.USE_VALUE_CLIP:
            old_value = reward - raw_advantage
            value_clipped = old_value + torch.clamp(
                value_result_squeezed - old_value,
                -Config.VALUE_CLIP_PARAM,
                Config.VALUE_CLIP_PARAM,
            )
            value_loss_clipped = torch.square(reward - value_clipped)
            value_loss = torch.maximum(value_loss_unclipped, value_loss_clipped)
        else:
            value_loss = value_loss_unclipped
        value_train_weight = frame_is_train.float()
        self.value_cost = 0.5 * torch.sum(value_loss * value_train_weight) / torch.maximum(
            torch.sum(value_train_weight),
            torch.tensor(1.0, device=reward.device),
        )

        label_probability_list = []
        epsilon = 1e-5
        self.policy_cost = torch.tensor(0.0, device=reward.device)
        boundary = torch.tensor(1e8, device=reward.device)
        ratio_sum = torch.tensor(0.0, device=reward.device)
        kl_sum = torch.tensor(0.0, device=reward.device)
        clip_sum = torch.tensor(0.0, device=reward.device)
        ratio_weight_sum = torch.tensor(0.0, device=reward.device)

        for task_index in range(len(self.is_reinforce_task_list)):
            if not self.is_reinforce_task_list[task_index]:
                continue

            one_hot_actions = nn.functional.one_hot(label_list[task_index].long(), self.label_size_list[task_index])
            legal_mask = legal_action_flag_list[task_index]
            legal_action_flag_list_max_mask = (1 - legal_mask) * boundary
            label_logits_subtract_max = torch.clamp(
                label_result[task_index]
                - torch.max(label_result[task_index] - legal_action_flag_list_max_mask, dim=1, keepdim=True).values,
                -boundary,
                1,
            )
            label_exp_logits = legal_mask * (torch.exp(label_logits_subtract_max) + self.min_policy)
            label_sum_exp_logits = torch.clamp(label_exp_logits.sum(1, keepdim=True), min=epsilon)
            label_probability = label_exp_logits / label_sum_exp_logits
            label_probability_list.append(label_probability)

            policy_p = (one_hot_actions * label_probability).sum(1)
            old_policy_p = (one_hot_actions * old_label_probability_list[task_index]).sum(1)
            policy_log_p = torch.log(policy_p + self.log_epsilon)
            old_policy_log_p = torch.log(old_policy_p + self.log_epsilon)
            ratio = torch.exp(policy_log_p - old_policy_log_p)

            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * advantage
            clipped = torch.minimum(surr1, surr2)
            dual_clipped = torch.where(advantage < 0, torch.maximum(clipped, self.dual_clip_param * advantage), clipped)
            train_weight = weight_list[task_index].float() * frame_is_train
            temp_policy_loss = -torch.sum(dual_clipped * train_weight) / torch.maximum(
                torch.sum(train_weight), torch.tensor(1.0, device=reward.device)
            )
            self.policy_cost = self.policy_cost + temp_policy_loss
            ratio_sum = ratio_sum + torch.sum(ratio.detach() * train_weight)
            kl_sum = kl_sum + torch.sum((old_policy_log_p - policy_log_p).detach() * train_weight)
            clip_sum = clip_sum + torch.sum((torch.abs(ratio.detach() - 1.0) > self.clip_param).float() * train_weight)
            ratio_weight_sum = ratio_weight_sum + torch.sum(train_weight)

        entropy_loss_list = []
        current_entropy_loss_index = 0
        for task_index in range(len(self.is_reinforce_task_list)):
            if self.is_reinforce_task_list[task_index]:
                temp_entropy_loss = -torch.sum(
                    label_probability_list[current_entropy_loss_index]
                    * legal_action_flag_list[task_index]
                    * torch.log(label_probability_list[current_entropy_loss_index] + epsilon),
                    dim=1,
                )
                train_weight = weight_list[task_index].float() * frame_is_train
                temp_entropy_loss = -torch.sum(temp_entropy_loss * train_weight) / torch.maximum(
                    torch.sum(train_weight), torch.tensor(1.0, device=reward.device)
                )
                entropy_loss_list.append(temp_entropy_loss)
                current_entropy_loss_index += 1
            else:
                entropy_loss_list.append(torch.tensor(0.0, device=reward.device))

        self.entropy_cost = torch.tensor(0.0, device=reward.device)
        for entropy_element in entropy_loss_list:
            self.entropy_cost = self.entropy_cost + entropy_element

        self.loss = self.value_cost + self.policy_cost + self.var_beta * self.entropy_cost
        ratio_den = torch.maximum(ratio_weight_sum, torch.tensor(1.0, device=reward.device))
        self.mean_ratio = ratio_sum / ratio_den
        self.approx_kl = kl_sum / ratio_den
        self.clip_fraction = clip_sum / ratio_den
        return self.loss, [self.loss, [self.value_cost, self.policy_cost, self.entropy_cost]]

    def set_train_mode(self):
        self.lstm_time_steps = Config.LSTM_TIME_STEPS
        self.train()

    def set_eval_mode(self):
        self.lstm_time_steps = 1
        self.eval()


def make_fc_layer(in_features: int, out_features: int, use_bias=True):
    fc_layer = nn.Linear(in_features, out_features, bias=use_bias)
    nn.init.orthogonal_(fc_layer.weight)
    if use_bias:
        nn.init.zeros_(fc_layer.bias)
    return fc_layer


class MLP(nn.Module):
    def __init__(
        self,
        fc_feat_dim_list: List[int],
        name: str,
        non_linearity: nn.Module = nn.ReLU,
        non_linearity_last: bool = False,
        use_layer_norm: bool = False,
    ):
        super(MLP, self).__init__()
        self.fc_layers = nn.Sequential()
        for i in range(len(fc_feat_dim_list) - 1):
            fc_layer = make_fc_layer(fc_feat_dim_list[i], fc_feat_dim_list[i + 1])
            self.fc_layers.add_module(f"{name}_fc{i + 1}", fc_layer)
            is_last = i + 1 == len(fc_feat_dim_list) - 1
            if not is_last or non_linearity_last:
                if use_layer_norm:
                    self.fc_layers.add_module(f"{name}_ln{i + 1}", nn.LayerNorm(fc_feat_dim_list[i + 1]))
                self.fc_layers.add_module(f"{name}_non_linear{i + 1}", non_linearity())

    def forward(self, data):
        return self.fc_layers(data)
