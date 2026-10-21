#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
PPO learner wrapper for agent_diy.
"""

import os
import time

import numpy as np
import torch

from agent_diy.conf.conf import Config


class Algorithm:
    def __init__(self, model, optimizer, scheduler, device=None, logger=None, monitor=None):
        self.device = device
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.parameters = [p for param_group in self.optimizer.param_groups for p in param_group["params"]]
        self.train_step = 0

        self.logger = logger
        self.monitor = monitor

        self.cut_points = [value[0] for value in Config.data_shapes]
        self.data_split_shape = Config.DATA_SPLIT_SHAPE
        self.seri_vec_split_shape = Config.SERI_VEC_SPLIT_SHAPE
        self.lstm_unit_size = Config.LSTM_UNIT_SIZE
        self.last_report_monitor_time = 0

    def learn(self, list_sample_data):
        if not list_sample_data:
            return None

        samples = []
        for sample in list_sample_data:
            value = sample.sample
            if isinstance(value, torch.Tensor):
                samples.append(value.detach().cpu().numpy())
            else:
                samples.append(np.asarray(value, dtype=np.float32))
        _input_datas = torch.as_tensor(np.stack(samples), dtype=torch.float32, device=self.device)
        if _input_datas.shape[1] != Config.SAMPLE_DIM:
            raise ValueError(f"learner sample dim mismatch: {_input_datas.shape[1]} != {Config.SAMPLE_DIM}")

        results = {}
        data_list = list(_input_datas.split(self.cut_points, dim=1))
        for i, data in enumerate(data_list):
            data_list[i] = data.reshape(-1).float()

        seri_vec = data_list[0].reshape(-1, self.data_split_shape[0])
        feature, _ = seri_vec.split(
            [
                np.prod(self.seri_vec_split_shape[0]),
                np.prod(self.seri_vec_split_shape[1]),
            ],
            dim=1,
        )
        init_lstm_cell = data_list[-2]
        init_lstm_hidden = data_list[-1]

        feature_vec = feature.reshape(-1, self.seri_vec_split_shape[0][0])
        lstm_hidden_state = init_lstm_hidden.reshape(-1, self.lstm_unit_size)
        lstm_cell_state = init_lstm_cell.reshape(-1, self.lstm_unit_size)
        format_inputs = [feature_vec, lstm_hidden_state, lstm_cell_state]

        self.model.set_train_mode()
        self.optimizer.zero_grad()
        beta_end = getattr(Config, "BETA_END", Config.BETA_START * 0.1)
        self.model.var_beta = max(beta_end, Config.BETA_START * (0.9999 ** self.train_step))

        rst_list = self.model(format_inputs)
        total_loss, info_list = self.model.compute_loss(data_list, rst_list)
        results["total_loss"] = total_loss.item()

        total_loss.backward()
        if Config.USE_GRAD_CLIP:
            torch.nn.utils.clip_grad_norm_(self.parameters, Config.GRAD_CLIP_RANGE)

        self.optimizer.step()
        self.train_step += 1
        self.scheduler.step()

        info_values = []
        for info in info_list:
            if isinstance(info, list):
                info_values.append([i.item() for i in info])
            else:
                info_values.append(info.item())

        now = time.time()
        if now - self.last_report_monitor_time >= 60:
            _, (value_loss, policy_loss, entropy_loss) = info_values
            results["value_loss"] = round(value_loss, 4)
            results["policy_loss"] = round(policy_loss, 4)
            results["entropy_loss"] = round(entropy_loss, 4)
            results["mean_ratio"] = round(self._scalar(getattr(self.model, "mean_ratio", 0.0)), 4)
            results["approx_kl"] = round(self._scalar(getattr(self.model, "approx_kl", 0.0)), 6)
            results["clip_fraction"] = round(self._scalar(getattr(self.model, "clip_fraction", 0.0)), 4)
            if self.monitor:
                self.monitor.put_data({os.getpid(): results})
            self.last_report_monitor_time = now

        return results

    def _scalar(self, value):
        if isinstance(value, torch.Tensor):
            return float(value.detach().cpu().item())
        return float(value)
