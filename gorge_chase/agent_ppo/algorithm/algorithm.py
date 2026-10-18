#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
PPO algorithm for Gorge Chase PPO (refactored).

峡谷追猎 PPO 算法（重构版）：
  - 适配双观测输入 (map_tensor, sym_feat)
  - 熵系数线性退火（BETA_START -> BETA_END over BETA_ANNEAL_STEPS）
  - 新增 approx_kl / clip_frac 诊断指标

损失：
  total_loss = vf_coef * value_loss + policy_loss - beta * entropy_loss
"""

import os
import time

import torch

from agent_ppo.conf.conf import Config


class Algorithm:
    def __init__(self, model, optimizer, device=None, logger=None, monitor=None):
        self.device = device
        self.model = model
        self.optimizer = optimizer
        self.parameters = [p for pg in self.optimizer.param_groups for p in pg["params"]]
        self.logger = logger
        self.monitor = monitor

        self.label_size = Config.ACTION_NUM  # 16
        self.value_num = Config.VALUE_NUM
        self.vf_coef = Config.VF_COEF
        self.clip_param = Config.CLIP_PARAM

        self.beta_start = Config.BETA_START
        self.beta_end = Config.BETA_END
        self.beta_steps = max(int(Config.BETA_ANNEAL_STEPS), 1)
        self.var_beta = self.beta_start

        self.last_report_monitor_time = 0
        self.train_step = 0

    def _update_beta(self):
        progress = min(self.train_step / float(self.beta_steps), 1.0)
        self.var_beta = self.beta_start * (1.0 - progress) + self.beta_end * progress

    def learn(self, list_sample_data):
        """PPO update on a batch of SampleData.

        对一批 SampleData 执行 PPO 更新。
        """
        map_tensor = torch.stack([f.map_tensor for f in list_sample_data]).to(self.device)
        sym_feat = torch.stack([f.sym_feat for f in list_sample_data]).to(self.device)
        legal_action = torch.stack([f.legal_action for f in list_sample_data]).to(self.device)
        act = torch.stack([f.act for f in list_sample_data]).to(self.device).view(-1, 1)
        old_prob = torch.stack([f.prob for f in list_sample_data]).to(self.device)
        reward = torch.stack([f.reward for f in list_sample_data]).to(self.device)
        advantage = torch.stack([f.advantage for f in list_sample_data]).to(self.device)
        old_value = torch.stack([f.value for f in list_sample_data]).to(self.device)
        reward_sum = torch.stack([f.reward_sum for f in list_sample_data]).to(self.device)

        self.model.set_train_mode()
        self.optimizer.zero_grad()

        logits, value_pred = self.model(map_tensor, sym_feat)

        self._update_beta()

        total_loss, info_list = self._compute_loss(
            logits=logits,
            value_pred=value_pred,
            legal_action=legal_action,
            old_action=act,
            old_prob=old_prob,
            advantage=advantage,
            old_value=old_value,
            reward_sum=reward_sum,
            reward=reward,
        )

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters, Config.GRAD_CLIP_RANGE)
        self.optimizer.step()
        self.train_step += 1

        now = time.time()
        if now - self.last_report_monitor_time >= 60:
            results = {
                "total_loss": round(total_loss.item(), 4),
                "value_loss": round(info_list[0].item(), 4),
                "policy_loss": round(info_list[1].item(), 4),
                "entropy_loss": round(info_list[2].item(), 4),
                "approx_kl": round(info_list[3].item(), 6),
                "clip_frac": round(info_list[4].item(), 4),
                "beta": round(self.var_beta, 5),
                "reward": round(reward.mean().item(), 4),
            }
            if self.logger:
                self.logger.info(
                    f"[train] step:{self.train_step} "
                    f"total:{results['total_loss']} "
                    f"policy:{results['policy_loss']} "
                    f"value:{results['value_loss']} "
                    f"entropy:{results['entropy_loss']} "
                    f"kl:{results['approx_kl']} "
                    f"clip_frac:{results['clip_frac']} "
                    f"beta:{results['beta']}"
                )
            if self.monitor:
                self.monitor.put_data({os.getpid(): results})
            self.last_report_monitor_time = now

    def _compute_loss(
        self,
        logits,
        value_pred,
        legal_action,
        old_action,
        old_prob,
        advantage,
        old_value,
        reward_sum,
        reward,
    ):
        """Standard PPO loss: policy + value + entropy + diagnostics."""
        prob_dist = self._masked_softmax(logits, legal_action)

        # Policy loss
        one_hot = torch.nn.functional.one_hot(
            old_action[:, 0].long().clamp(0, self.label_size - 1),
            self.label_size,
        ).float()
        new_prob = (one_hot * prob_dist).sum(1, keepdim=True).clamp(1e-9)
        old_action_prob = (one_hot * old_prob).sum(1, keepdim=True).clamp(1e-9)
        ratio = new_prob / old_action_prob
        adv = advantage.view(-1, 1)
        policy_loss1 = -ratio * adv
        policy_loss2 = -ratio.clamp(1 - self.clip_param, 1 + self.clip_param) * adv
        policy_loss = torch.maximum(policy_loss1, policy_loss2).mean()

        # Value loss (clipped)
        vp = value_pred
        ov = old_value
        tdret = reward_sum
        value_clip = ov + (vp - ov).clamp(-self.clip_param, self.clip_param)
        value_loss = 0.5 * torch.maximum(
            torch.square(tdret - vp),
            torch.square(tdret - value_clip),
        ).mean()

        # Entropy
        entropy_loss = (-prob_dist * torch.log(prob_dist.clamp(1e-9, 1.0))).sum(1).mean()

        # Diagnostics
        with torch.no_grad():
            log_ratio = torch.log(ratio.clamp(1e-9))
            approx_kl = 0.5 * (log_ratio ** 2).mean()
            clip_frac = ((ratio - 1.0).abs() > self.clip_param).float().mean()

        total_loss = self.vf_coef * value_loss + policy_loss - self.var_beta * entropy_loss
        return total_loss, [value_loss, policy_loss, entropy_loss, approx_kl, clip_frac]

    def _masked_softmax(self, logits, legal_action):
        """Softmax with legal action masking (numerically stable)."""
        # 屏蔽非法动作：legal=0 的位置直接给一个极大负数
        neg_inf = -1e9
        masked = logits + (legal_action - 1.0) * -neg_inf  # (1-legal) * -neg_inf = (1-legal) * 1e9
        # 等价于：masked = logits.masked_fill(legal_action == 0, neg_inf)
        # 上式写成算子运算避免 boolean mask 在某些 torch 旧版行为
        masked = torch.where(legal_action > 0.5, logits, torch.full_like(logits, neg_inf))
        # 减最大值稳定
        m = masked.max(dim=1, keepdim=True).values
        safe = masked - m
        exp = torch.exp(safe) * (legal_action > 0.5).float()
        denom = exp.sum(dim=1, keepdim=True).clamp(1e-9)
        return exp / denom
