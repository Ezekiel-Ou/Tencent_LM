#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2025 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""


import time
import os
import numpy as np
import torch
from copy import deepcopy
from agent_target_dqn.conf.conf import Config
from agent_target_dqn.model.model import Model
from agent_target_dqn.feature.definition import ActData


class Algorithm:
    TARGET_FEATURE_START = 2 + 128 * 2
    TARGET_DIRECTION_BIAS = 2.0
    TARGET_TALENT_BIAS = 0.0
    EXPLORATION_DIRECTION_BIAS = 2.2
    EXPLORATION_TALENT_BIAS = 0.2
    EXPLOIT_TARGET_DIRECTION_BIAS = 8.0
    EXPLOIT_TARGET_TALENT_BIAS = 0.0
    EXPLOIT_EXPLORATION_DIRECTION_BIAS = 12.0
    EXPLOIT_EXPLORATION_TALENT_BIAS = 1.0
    DIRECTION_OFFSETS = (
        (1, 0),
        (1, 1),
        (0, 1),
        (-1, 1),
        (-1, 0),
        (-1, -1),
        (0, -1),
        (1, -1),
    )
    DIRECTION_FALLBACK_DELTAS = (0, -1, 1, -2, 2, -3, 3, 4)

    def __init__(self, device, monitor):
        self.device = device
        self.lr = Config.START_LR
        self.epsilon = Config.EPSILON
        self.epsilon_min = Config.EPSILON_MIN
        self.egp = Config.EPSILON_GREEDY_PROBABILITY
        self.target_update_freq = Config.TARGET_UPDATE_FREQ
        self.target_soft_tau = Config.TARGET_SOFT_TAU
        self.learn_repeat = Config.LEARN_REPEAT
        self._gamma = Config.GAMMA
        self.obs_split = Config.DESC_OBS_SPLIT
        self.obs_shape = Config.DIM_OF_OBSERVATION
        self.talent_direction = Config.DIM_OF_TALENT
        self.direction_space = Config.DIM_OF_ACTION_DIRECTION
        self.act_shape = Config.DIM_OF_ACTION_DIRECTION + Config.DIM_OF_TALENT

        self.train_step = 0
        self.predict_count = 0
        self.model = Model(
            state_shape=self.obs_shape,
            action_shape=self.act_shape,
            softmax=False,
            cnn_embed_dim=Config.MODEL_CNN_EMBED_DIM,
            backbone_hidden_dim=Config.MODEL_BACKBONE_HIDDEN_DIM,
            head_hidden_dim=Config.MODEL_HEAD_HIDDEN_DIM,
        )
        self.model.to(self.device)
        self.device = device
        self.monitor = monitor
        self.last_report_monitor_time = 0
        self.optim = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.target_model = deepcopy(self.model)
        self.target_model.eval()

    def learn(self, list_sample_data):
        t_data = list_sample_data
        batch = len(t_data)

        batch_feature_vec = [frame.obs[: self.obs_split[0]] for frame in t_data]
        batch_feature_map = [frame.obs[self.obs_split[0] :] for frame in t_data]
        batch_action = torch.as_tensor(
            np.asarray([self._scalar_value(frame.act) for frame in t_data], dtype=np.int64),
            dtype=torch.long,
            device=self.device,
        ).view(-1, 1)

        _batch_obs_legal = self._stack_to_tensor([frame._obs_legal for frame in t_data], expected_len=2)
        _batch_obs_legal = (
            torch.cat(
                (
                    _batch_obs_legal[:, 0].unsqueeze(1).expand(batch, self.direction_space),
                    _batch_obs_legal[:, 1].unsqueeze(1).expand(batch, self.talent_direction),
                ),
                1,
            )
            .bool()
        )

        rew = torch.as_tensor(
            np.asarray([self._scalar_value(frame.rew) for frame in t_data], dtype=np.float32),
            dtype=torch.float32,
            device=self.device,
        )
        _batch_feature_vec = [frame._obs[: self.obs_split[0]] for frame in t_data]
        _batch_feature_map = [frame._obs[self.obs_split[0] :] for frame in t_data]
        not_done = torch.as_tensor(
            np.asarray([0.0 if self._scalar_value(frame.done) == 1 else 1.0 for frame in t_data], dtype=np.float32),
            dtype=torch.float32,
            device=self.device,
        )

        batch_feature = [
            self._stack_to_tensor(batch_feature_vec),
            self._stack_to_tensor(batch_feature_map).view(batch, *self.obs_split[1]),
        ]
        _batch_feature = [
            self._stack_to_tensor(_batch_feature_vec),
            self._stack_to_tensor(_batch_feature_map).view(batch, *self.obs_split[1]),
        ]

        value_loss = 0.0
        q_value = 0.0
        reward = rew.mean().detach().item()
        model_grad_norm = 0.0
        for _ in range(self.learn_repeat):
            model = getattr(self, "model")
            target_model = getattr(self, "target_model")
            model.eval()
            target_model.eval()
            with torch.no_grad():
                # Double DQN: online network selects action, target network evaluates value.
                q_online, _ = model(_batch_feature, state=None)
                q_online = q_online.masked_fill(~_batch_obs_legal, -1e9)
                next_action = q_online.argmax(dim=1, keepdim=True)

                q_target, _ = target_model(_batch_feature, state=None)
                q_target = q_target.masked_fill(~_batch_obs_legal, -1e9)
                q_max = q_target.gather(1, next_action).view(-1).detach()

            target_q = rew + self._gamma * q_max * not_done

            self.optim.zero_grad(set_to_none=True)

            model.train()
            logits, h = model(batch_feature, state=None)

            pred_q = logits.gather(1, batch_action).view(-1)
            loss = torch.nn.functional.smooth_l1_loss(pred_q, target_q)
            loss.backward()

            model_grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            self.optim.step()

            self.train_step += 1

            # Mix periodic hard update and frequent soft update for better stability.
            if self.train_step % self.target_update_freq == 0:
                self.target_model.load_state_dict(self.model.state_dict())
            else:
                self._soft_update_target()

            value_loss = loss.detach().item()
            q_value = target_q.mean().detach().item()

        # Periodically report monitoring
        # 按照间隔上报监控
        now = time.time()
        if now - self.last_report_monitor_time >= 60:
            grad_norm = float(model_grad_norm)
            monitor_data = {
                "value_loss": value_loss,
                "q_value": q_value,
                "reward": reward,
                "grad_norm": grad_norm,
            }
            if self.monitor:
                self.monitor.put_data({os.getpid(): monitor_data})

            self.last_report_monitor_time = now

    def _to_numpy_1d(self, item):
        if torch.is_tensor(item):
            item = item.detach().cpu().numpy()
        array = np.asarray(item, dtype=np.float32)
        return array.reshape(-1)

    def _stack_to_tensor(self, data, expected_len=None):
        arrays = []
        for item in data:
            array = self._to_numpy_1d(item)
            if expected_len is not None:
                if array.size < expected_len:
                    array = np.pad(array, (0, expected_len - array.size), mode="constant")
                elif array.size > expected_len:
                    array = array[:expected_len]
            arrays.append(array)

        return torch.as_tensor(np.stack(arrays, axis=0), dtype=torch.float32, device=self.device)

    def _scalar_value(self, item):
        if torch.is_tensor(item):
            return item.detach().cpu().reshape(-1)[0].item()
        return np.asarray(item).reshape(-1)[0].item()

    def _apply_target_direction_bias(self, logits, feature_vec, legal_act, exploit_flag=False):
        target_start = self.TARGET_FEATURE_START
        target_direction = feature_vec[:, target_start : target_start + self.direction_space]
        has_target = target_direction.max(dim=1).values > 0.5
        if not has_target.any():
            return logits

        target_action = target_direction.argmax(dim=1, keepdim=True)
        target_distance = feature_vec[:, target_start + self.direction_space]
        is_exploration_target = torch.isclose(
            target_distance,
            torch.full_like(target_distance, 0.2),
            atol=1e-4,
        ).unsqueeze(1)
        bias = torch.zeros_like(logits)
        target_direction_bias = self.EXPLOIT_TARGET_DIRECTION_BIAS if exploit_flag else self.TARGET_DIRECTION_BIAS
        target_talent_bias = self.EXPLOIT_TARGET_TALENT_BIAS if exploit_flag else self.TARGET_TALENT_BIAS
        exploration_direction_bias = (
            self.EXPLOIT_EXPLORATION_DIRECTION_BIAS if exploit_flag else self.EXPLORATION_DIRECTION_BIAS
        )
        exploration_talent_bias = self.EXPLOIT_EXPLORATION_TALENT_BIAS if exploit_flag else self.EXPLORATION_TALENT_BIAS
        move_bias = torch.where(
            is_exploration_target,
            torch.full_like(target_action, exploration_direction_bias, dtype=torch.float32),
            torch.full_like(target_action, target_direction_bias, dtype=torch.float32),
        )
        bias.scatter_add_(1, target_action, move_bias)

        talent_action = target_action + self.direction_space
        talent_bias = torch.where(
            is_exploration_target,
            torch.full_like(talent_action, exploration_talent_bias, dtype=torch.float32),
            torch.full_like(talent_action, target_talent_bias, dtype=torch.float32),
        )
        bias.scatter_add_(1, talent_action, talent_bias)

        bias = bias * legal_act.float() * has_target.float().unsqueeze(1)
        return logits + bias

    def _is_direction_passable(self, feature_map_tensor, batch_index, direction):
        obstacle_grid = feature_map_tensor[batch_index, 0]
        center_x = obstacle_grid.shape[0] // 2
        center_z = obstacle_grid.shape[1] // 2
        dx, dz = self.DIRECTION_OFFSETS[int(direction) % self.direction_space]
        next_x = center_x + dx
        next_z = center_z + dz
        if next_x < 0 or next_z < 0 or next_x >= obstacle_grid.shape[0] or next_z >= obstacle_grid.shape[1]:
            return False
        return bool(obstacle_grid[next_x, next_z].item() > 0)

    def _nearest_passable_direction(self, feature_map_tensor, batch_index, target_direction):
        for delta in self.DIRECTION_FALLBACK_DELTAS:
            candidate = (int(target_direction) + delta) % self.direction_space
            if self._is_direction_passable(feature_map_tensor, batch_index, candidate):
                return candidate
        return None

    def _target_override_actions(self, logits, feature_vec, feature_map_tensor, legal_act):
        target_start = self.TARGET_FEATURE_START
        target_direction = feature_vec[:, target_start : target_start + self.direction_space]
        target_distance = feature_vec[:, target_start + self.direction_space]
        has_target = target_direction.max(dim=1).values > 0.5
        model_actions = logits.argmax(dim=1)
        actions = model_actions.clone()

        for batch_index in range(feature_vec.shape[0]):
            if not bool(has_target[batch_index].item()):
                continue

            target_dir = int(target_direction[batch_index].argmax().item())
            passable_dir = self._nearest_passable_direction(feature_map_tensor, batch_index, target_dir)
            if passable_dir is None:
                continue

            move_action = passable_dir
            talent_action = passable_dir + self.direction_space
            target_move_q = logits[batch_index, move_action]

            # Evaluation follows the stage target deterministically; the local passability fallback above handles blocked directions.
            actions[batch_index] = move_action

            is_exploration_target = bool(abs(float(target_distance[batch_index].item()) - 0.2) <= 1e-4)
            if is_exploration_target and bool(legal_act[batch_index, talent_action].item()):
                talent_q = logits[batch_index, talent_action]
                if talent_q > target_move_q + 1.5:
                    actions[batch_index] = talent_action

        return actions

    def predict_detail(self, list_obs_data, exploit_flag=False):
        batch = len(list_obs_data)
        feature_vec = [obs_data.feature[: self.obs_split[0]] for obs_data in list_obs_data]
        feature_map = [obs_data.feature[self.obs_split[0] :] for obs_data in list_obs_data]
        legal_act = [obs_data.legal_act for obs_data in list_obs_data]
        legal_act = self._stack_to_tensor(legal_act, expected_len=2)
        legal_act = (
            torch.cat(
                (
                    legal_act[:, 0].unsqueeze(1).expand(batch, self.direction_space),
                    legal_act[:, 1].unsqueeze(1).expand(batch, self.talent_direction),
                ),
                1,
            )
            .bool()
        )
        model = self.model
        model.eval()

        # Linear epsilon decay from EPSILON to EPSILON_MIN.
        decay_ratio = min(1.0, self.predict_count / max(1, self.egp))
        self.epsilon = Config.EPSILON - (Config.EPSILON - self.epsilon_min) * decay_ratio
        feature_vec_tensor = self._stack_to_tensor(feature_vec)

        with torch.no_grad():
            # epsilon greedy
            if not exploit_flag and np.random.rand(1) < self.epsilon:
                random_action = torch.rand(batch, self.act_shape, dtype=torch.float32, device=self.device)
                random_action = random_action.masked_fill(~legal_act, -1e9)
                act = random_action.argmax(dim=1).cpu().view(-1, 1).tolist()
            else:
                feature_map_tensor = self._stack_to_tensor(feature_map).view(batch, *self.obs_split[1])
                feature = [feature_vec_tensor, feature_map_tensor]
                logits, _ = model(feature, state=None)
                logits = self._apply_target_direction_bias(logits, feature_vec_tensor, legal_act, exploit_flag)
                logits = logits.masked_fill(~legal_act, -1e9)
                if exploit_flag:
                    action_tensor = self._target_override_actions(logits, feature_vec_tensor, feature_map_tensor, legal_act)
                else:
                    action_tensor = logits.argmax(dim=1)
                act = action_tensor.cpu().view(-1, 1).tolist()

        format_action = [[instance[0] % self.direction_space, instance[0] // self.direction_space] for instance in act]
        self.predict_count += 1
        return [ActData(move_dir=i[0], use_talent=i[1]) for i in format_action]

    def _soft_update_target(self):
        tau = self.target_soft_tau
        for target_param, param in zip(self.target_model.parameters(), self.model.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)
