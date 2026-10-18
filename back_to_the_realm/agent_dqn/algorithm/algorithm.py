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
from agent_dqn.conf.conf import Config
from agent_dqn.model.model import Model
from agent_dqn.feature.definition import ActData


class Algorithm:
    def __init__(self, device, monitor):
        self.device = device
        self.lr = Config.START_LR
        self.epsilon = Config.EPSILON
        self.epsilon_min = Config.EPSILON_MIN
        self.egp = Config.EPSILON_GREEDY_PROBABILITY
        self.target_update_freq = Config.TARGET_UPDATE_FREQ
        self.target_soft_tau = Config.TARGET_SOFT_TAU
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
        batch_action = torch.LongTensor(np.array([int(frame.act) for frame in t_data])).view(-1, 1).to(self.device)

        _batch_obs_legal = self._format_legal_actions([frame._obs_legal for frame in t_data])

        rew = torch.tensor(np.array([frame.rew for frame in t_data]), device=self.device)
        _batch_feature_vec = [frame._obs[: self.obs_split[0]] for frame in t_data]
        _batch_feature_map = [frame._obs[self.obs_split[0] :] for frame in t_data]
        not_done = torch.tensor(np.array([0 if frame.done == 1 else 1 for frame in t_data]), device=self.device)

        batch_feature = [
            self.__convert_to_tensor(batch_feature_vec),
            self.__convert_to_tensor(batch_feature_map).view(batch, *self.obs_split[1]),
        ]
        _batch_feature = [
            self.__convert_to_tensor(_batch_feature_vec),
            self.__convert_to_tensor(_batch_feature_map).view(batch, *self.obs_split[1]),
        ]

        model = getattr(self, "model")
        target_model = getattr(self, "target_model")
        model.eval()
        target_model.eval()
        with torch.no_grad():
            # Double DQN: online network selects action, target network evaluates value.
            q_online, _ = model(_batch_feature, state=None)
            q_online = q_online.masked_fill(~_batch_obs_legal, float("-inf"))
            next_action = q_online.argmax(dim=1, keepdim=True)

            q_target, _ = target_model(_batch_feature, state=None)
            q_target = q_target.masked_fill(~_batch_obs_legal, float("-inf"))
            q_max = q_target.gather(1, next_action).view(-1).detach()

        target_q = rew + self._gamma * q_max * not_done

        self.optim.zero_grad()

        model = getattr(self, "model")
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
        reward = rew.mean().detach().item()

        # Periodically report monitoring
        # 按照间隔上报监控
        now = time.time()
        if now - self.last_report_monitor_time >= 60:
            monitor_data = {
                "value_loss": value_loss,
                "q_value": q_value,
                "reward": reward,
                "diy_1": model_grad_norm,
            }
            if self.monitor:
                self.monitor.put_data({os.getpid(): monitor_data})

            self.last_report_monitor_time = now

    def __convert_to_tensor(self, data):
        if isinstance(data, list):
            data = [np.array(item, dtype=np.float32) for item in data]
        elif isinstance(data, np.ndarray):
            if data.dtype == np.object_:
                data = data.astype(np.float32)
            else:
                data = data.astype(np.float32)
        else:
            raise TypeError(f"Unsupported data type: {type(data)}")

        tensor = torch.stack([torch.tensor(item) for item in data]).to(self.device)
        return tensor

    def predict_detail(self, list_obs_data, exploit_flag=False):
        batch = len(list_obs_data)
        feature_vec = [obs_data.feature[: self.obs_split[0]] for obs_data in list_obs_data]
        feature_map = [obs_data.feature[self.obs_split[0] :] for obs_data in list_obs_data]
        legal_act = self._format_legal_actions([obs_data.legal_act for obs_data in list_obs_data])
        model = self.model
        model.eval()

        # Linear epsilon decay from EPSILON to EPSILON_MIN.
        decay_ratio = min(1.0, self.predict_count / max(1, self.egp))
        self.epsilon = Config.EPSILON - (Config.EPSILON - self.epsilon_min) * decay_ratio

        with torch.no_grad():
            # epsilon greedy
            if not exploit_flag and np.random.rand(1) < self.epsilon:
                random_action = np.random.rand(batch, self.act_shape)
                random_action = torch.tensor(random_action, dtype=torch.float32).to(self.device)
                random_action = random_action.masked_fill(~legal_act, 0)
                act = random_action.argmax(dim=1).cpu().view(-1, 1).tolist()
            else:
                feature = [
                    self.__convert_to_tensor(feature_vec),
                    self.__convert_to_tensor(feature_map).view(batch, *self.obs_split[1]),
                ]
                logits, _ = model(feature, state=None)
                logits = logits.masked_fill(~legal_act, float(torch.min(logits)))
                act = logits.argmax(dim=1).cpu().view(-1, 1).tolist()

        format_action = [[instance[0] % self.direction_space, instance[0] // self.direction_space] for instance in act]
        self.predict_count += 1
        return [ActData(move_dir=i[0], use_talent=i[1]) for i in format_action]

    def _soft_update_target(self):
        tau = self.target_soft_tau
        for target_param, param in zip(self.target_model.parameters(), self.model.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

    def _format_legal_actions(self, legal_act_list):
        legal_act_np = np.stack([self._legal_action_to_numpy(legal_act) for legal_act in legal_act_list]).astype(np.float32)
        legal_act = torch.from_numpy(legal_act_np).to(self.device)
        batch = legal_act.shape[0]
        return (
            torch.cat(
                (
                    legal_act[:, 0].unsqueeze(1).expand(batch, self.direction_space),
                    legal_act[:, 1].unsqueeze(1).expand(batch, self.talent_direction),
                ),
                1,
            )
            .bool()
            .to(self.device)
        )

    def _legal_action_to_numpy(self, legal_act):
        if isinstance(legal_act, torch.Tensor):
            legal_act = legal_act.detach().cpu().numpy()

        legal_act = np.asarray(legal_act, dtype=np.float32).reshape(-1)
        if legal_act.size != 2:
            raise ValueError(f"legal action should contain 2 values, got shape {legal_act.shape}")

        return legal_act
