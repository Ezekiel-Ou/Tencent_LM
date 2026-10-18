#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2025 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""

import os
import time
import numpy as np
import torch
import torch.nn.functional as F

from agent_ppo.conf.conf import Config
from agent_ppo.model.model import Model


class Algorithm:
    def __init__(self, device, logger=None, monitor=None):
        self.device = device
        self.logger = logger
        self.monitor = monitor

        # 基本维度
        self.obs_dim = Config.DIM_OF_OBSERVATION
        self.act_dim = Config.DIM_OF_ACTION

        # PPO 超参数
        self.gamma = Config.GAMMA
        self.lam = Config.LAMBDA
        self.clip_eps = Config.CLIP_EPS
        self.actor_lr = Config.ACTOR_LR
        self.critic_lr = Config.CRITIC_LR
        self.entropy_coef = Config.ENTROPY_COEF
        self.value_coef = Config.VALUE_COEF
        self.max_grad_norm = Config.MAX_GRAD_NORM
        self.ppo_epochs = Config.PPO_EPOCHS
        self.mini_batch_size = Config.MINI_BATCH_SIZE

        # 模型
        self.model = Model().to(self.device)

        # 优化器
        self.optim = torch.optim.Adam(
            [
                {"params": self.model.feature_layer.parameters(), "lr": self.actor_lr},
                {"params": self.model.actor_layer.parameters(), "lr": self.actor_lr},
                {"params": self.model.critic_layer.parameters(), "lr": self.critic_lr},
            ]
        )

        self.train_step = 0
        self.predict_count = 0
        self.last_report_monitor_time = 0

    def _to_tensor(self, data, dtype=torch.float32):
        """
        将输入统一转成 tensor
        """
        if isinstance(data, np.ndarray):
            return torch.from_numpy(data).to(self.device, dtype=dtype)
        if torch.is_tensor(data):
            return data.to(self.device, dtype=dtype)
        return torch.tensor(data, dtype=dtype, device=self.device)

    def _build_batch(self, list_sample_data):
        """
        从 list_sample_data 中提取 PPO 所需批数据

        每个 sample 需要包含字段：
            state
            action
            logprob
            reward
            done
            value
            next_state
        """
        states = np.stack(
            [np.asarray(sample.state, dtype=np.float32) for sample in list_sample_data],
            axis=0,
        )
        actions = np.asarray(
            [int(sample.action) for sample in list_sample_data],
            dtype=np.int64,
        )
        old_logprobs = np.asarray(
            [float(sample.logprob) for sample in list_sample_data],
            dtype=np.float32,
        )
        rewards = np.asarray(
            [float(sample.reward) for sample in list_sample_data],
            dtype=np.float32,
        )
        dones = np.asarray(
            [float(sample.done) for sample in list_sample_data],
            dtype=np.float32,
        )
        values = np.asarray(
            [float(sample.value) for sample in list_sample_data],
            dtype=np.float32,
        )
        next_states = np.stack(
            [np.asarray(sample.next_state, dtype=np.float32) for sample in list_sample_data],
            axis=0,
        )

        states = self._to_tensor(states, torch.float32)
        actions = self._to_tensor(actions, torch.long)
        old_logprobs = self._to_tensor(old_logprobs, torch.float32)
        rewards = self._to_tensor(rewards, torch.float32)
        dones = self._to_tensor(dones, torch.float32)
        values = self._to_tensor(values, torch.float32)
        next_states = self._to_tensor(next_states, torch.float32)

        return states, actions, old_logprobs, rewards, dones, values, next_states

    @torch.no_grad()
    def _compute_gae(self, rewards, dones, values, next_states):
        """
        使用 GAE 计算 advantage 和 returns
        """
        _, next_values = self.model(next_states)
        next_values = next_values.detach()

        T = rewards.shape[0]
        advantages = torch.zeros_like(rewards, device=self.device)

        gae = 0.0
        for t in reversed(range(T)):
            if t == T - 1:
                next_value = next_values[t]
                next_nonterminal = 1.0 - dones[t]
            else:
                next_value = values[t + 1]
                next_nonterminal = 1.0 - dones[t]

            delta = rewards[t] + self.gamma * next_value * next_nonterminal - values[t]
            gae = delta + self.gamma * self.lam * next_nonterminal * gae
            advantages[t] = gae

        returns = advantages + values
        return advantages, returns

    def learn(self, list_sample_data):
        """
        PPO 的训练入口
        """
        if list_sample_data is None or len(list_sample_data) == 0:
            return

        states, actions, old_logprobs, rewards, dones, old_values, next_states = self._build_batch(
            list_sample_data
        )

        with torch.no_grad():
            advantages, returns = self._compute_gae(
                rewards=rewards,
                dones=dones,
                values=old_values,
                next_states=next_states,
            )
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        batch_size = states.shape[0]
        indices = np.arange(batch_size)

        total_actor_loss = 0.0
        total_critic_loss = 0.0
        total_entropy = 0.0
        update_count = 0

        self.model.train()

        for _ in range(self.ppo_epochs):
            np.random.shuffle(indices)

            for start in range(0, batch_size, self.mini_batch_size):
                end = start + self.mini_batch_size
                mb_idx = indices[start:end]

                mb_states = states[mb_idx]
                mb_actions = actions[mb_idx]
                mb_old_logprobs = old_logprobs[mb_idx]
                mb_advantages = advantages[mb_idx]
                mb_returns = returns[mb_idx]

                new_logprobs, entropy, new_values = self.model.evaluate_actions(
                    mb_states, mb_actions
                )

                ratio = torch.exp(new_logprobs - mb_old_logprobs)

                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(
                    ratio,
                    1.0 - self.clip_eps,
                    1.0 + self.clip_eps,
                ) * mb_advantages

                actor_loss = -torch.min(surr1, surr2).mean()
                critic_loss = F.mse_loss(new_values, mb_returns)
                entropy_bonus = entropy.mean()

                loss = actor_loss + self.value_coef * critic_loss - self.entropy_coef * entropy_bonus

                self.optim.zero_grad()
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.max_grad_norm,
                )
                self.optim.step()

                total_actor_loss += actor_loss.item()
                total_critic_loss += critic_loss.item()
                total_entropy += entropy_bonus.item()
                update_count += 1

        self.train_step += 1

        mean_actor_loss = total_actor_loss / max(update_count, 1)
        mean_critic_loss = total_critic_loss / max(update_count, 1)
        mean_entropy = total_entropy / max(update_count, 1)
        mean_reward = rewards.mean().item()
        mean_return = returns.mean().item()

        now = time.time()
        if now - self.last_report_monitor_time >= 60:
            monitor_data = {
                "actor_loss": mean_actor_loss,
                "critic_loss": mean_critic_loss,
                "entropy": mean_entropy,
                "reward": mean_reward,
                "return": mean_return,
            }
            if self.monitor:
                self.monitor.put_data({os.getpid(): monitor_data})
            self.last_report_monitor_time = now

        if self.logger:
            self.logger.info(
                f"[PPO] step={self.train_step}, "
                f"actor_loss={mean_actor_loss:.6f}, "
                f"critic_loss={mean_critic_loss:.6f}, "
                f"entropy={mean_entropy:.6f}, "
                f"reward={mean_reward:.6f}, "
                f"return={mean_return:.6f}"
            )

    @torch.no_grad()
    def predict(self, state, deterministic=False):
        """
        单样本预测动作

        输入:
            state: shape = [obs_dim]

        返回:
            action: int
            logprob: float
            value: float
        """
        self.model.eval()

        state = np.asarray(state, dtype=np.float32)
        obs = self._to_tensor(state, torch.float32)
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)

        action, logprob, value = self.model.predict(obs, deterministic=deterministic)

        return int(action.item()), float(logprob.item()), float(value.item())

    @torch.no_grad()
    def predict_batch(self, list_state, deterministic=False):
        """
        批量预测动作

        输入:
            list_state: List[state]

        返回:
            actions, logprobs, values
        """
        self.model.eval()

        obs = np.asarray(list_state, dtype=np.float32)
        obs = self._to_tensor(obs, torch.float32)

        actions, logprobs, values = self.model.predict(obs, deterministic=deterministic)

        return (
            actions.cpu().numpy().tolist(),
            logprobs.cpu().numpy().tolist(),
            values.cpu().numpy().tolist(),
        )

    def save_model(self, path=None, id="latest"):
        if path is None:
            os.makedirs("./model", exist_ok=True)
            path = f"./model/ppo_model_{id}.pt"

        torch.save(self.model.state_dict(), path)

        if self.logger:
            self.logger.info(f"save model success: {path}")

    def load_model(self, path):
        state_dict = torch.load(path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        if self.logger:
            self.logger.info(f"load model success: {path}")

