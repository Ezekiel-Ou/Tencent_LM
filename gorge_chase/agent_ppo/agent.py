#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Agent for Gorge Chase PPO (refactored).

峡谷追猎 PPO Agent 主类（重构版）。

关键变更：
  - ObsData 携带双观测 (map_tensor, sym_feat) 与 rule_hints
  - predict 末尾调用 rules.apply_rule_override（保留 RL 采样 prob 不变，避免 PPO ratio 失真）
  - _run_model 拆包两输入
"""

import torch

torch.set_num_threads(1)
torch.set_num_interop_threads(1)

import numpy as np

from kaiwudrl.interface.agent import BaseAgent

from agent_ppo.algorithm.algorithm import Algorithm
from agent_ppo.conf.conf import Config
from agent_ppo.feature.definition import ActData, ObsData
from agent_ppo.feature.preprocessor import Preprocessor
from agent_ppo.feature.rules import apply_rule_override
from agent_ppo.model.model import Model


class Agent(BaseAgent):
    def __init__(self, agent_type="player", device=None, logger=None, monitor=None):
        torch.manual_seed(0)
        self.device = device
        self.model = Model(device).to(self.device)
        self.optimizer = torch.optim.Adam(
            params=self.model.parameters(),
            lr=Config.INIT_LEARNING_RATE_START,
            betas=(0.9, 0.999),
            eps=1e-8,
        )
        self.algorithm = Algorithm(self.model, self.optimizer, self.device, logger, monitor)
        self.preprocessor = Preprocessor()
        self.last_action = -1
        self.logger = logger
        self.monitor = monitor
        self.rule_override_count = 0
        super().__init__(agent_type, device, logger, monitor)

    # ----------------------------------------------------------------
    # Episode lifecycle
    # ----------------------------------------------------------------
    def reset(self, env_obs=None):
        self.preprocessor.reset()
        self.last_action = -1
        self.rule_override_count = 0

    # ----------------------------------------------------------------
    # Observation pipeline
    # ----------------------------------------------------------------
    def observation_process(self, env_obs):
        feat_dict = self.preprocessor.feature_process(env_obs, self.last_action)
        obs_data = ObsData(
            map_tensor=feat_dict["map_tensor"],       # (C, H, W) float32
            sym_feat=feat_dict["sym_feat"],           # (SYM_FEATURE_LEN,) float32
            legal_action=feat_dict["legal_action_16d"],
            rule_hints=feat_dict["rule_hints"],
        )
        remain_info = {
            "reward": feat_dict["reward"],
            "wall_bump_count": self.preprocessor.wall_bump_count,
            "anti_stall_count": self.preprocessor.anti_stall_count,
            "treasure_discover_count": self.preprocessor.treasure_discover_count,
        }
        return obs_data, remain_info

    # ----------------------------------------------------------------
    # Inference (stochastic / greedy)
    # ----------------------------------------------------------------
    def predict(self, list_obs_data):
        obs_data = list_obs_data[0]
        logits_np, value_np, prob = self._run_model(
            obs_data.map_tensor, obs_data.sym_feat, obs_data.legal_action
        )

        rl_action = self._legal_sample(prob, use_max=False)
        d_action = self._legal_sample(prob, use_max=True)

        # 规则接管（只替换执行动作，不改 prob）
        final_action, overridden = apply_rule_override(
            rl_action, obs_data.legal_action, obs_data.rule_hints
        )
        if overridden:
            self.rule_override_count += 1

        return [ActData(
            action=[final_action],
            d_action=[d_action],
            prob=list(prob),
            value=value_np,
            rl_action=[rl_action],
        )]

    def exploit(self, env_obs):
        obs_data, _ = self.observation_process(env_obs)
        _, _, prob = self._run_model(
            obs_data.map_tensor, obs_data.sym_feat, obs_data.legal_action
        )
        d_action = self._legal_sample(prob, use_max=True)
        # 评估时也允许规则接管
        final_action, overridden = apply_rule_override(
            d_action, obs_data.legal_action, obs_data.rule_hints
        )
        self.last_action = int(final_action)
        return int(final_action)

    # ----------------------------------------------------------------
    # Training
    # ----------------------------------------------------------------
    def learn(self, list_sample_data):
        return self.algorithm.learn(list_sample_data)

    # ----------------------------------------------------------------
    # Checkpoint
    # ----------------------------------------------------------------
    def save_model(self, path=None, id="1"):
        model_file_path = f"{path}/model.ckpt-{str(id)}.pkl"
        state_dict_cpu = {k: v.clone().cpu() for k, v in self.model.state_dict().items()}
        torch.save(state_dict_cpu, model_file_path)
        if self.logger:
            self.logger.info(f"save model {model_file_path} successfully")

    def load_model(self, path=None, id="1"):
        model_file_path = f"{path}/model.ckpt-{str(id)}.pkl"
        try:
            self.model.load_state_dict(
                torch.load(model_file_path, map_location=self.device)
            )
            if self.logger:
                self.logger.info(f"load model {model_file_path} successfully")
        except FileNotFoundError:
            if self.logger:
                self.logger.warning(f"ckpt {model_file_path} not found, keeping random init")

    # ----------------------------------------------------------------
    # Action unpacking
    # ----------------------------------------------------------------
    def action_process(self, act_data, is_stochastic=True):
        action = act_data.action if is_stochastic else act_data.d_action
        self.last_action = int(action[0])
        return int(action[0])

    def evaluate_value(self, obs_data):
        """Estimate critic value for one observation without sampling an action."""
        _, value_np, _ = self._run_model(
            obs_data.map_tensor, obs_data.sym_feat, obs_data.legal_action
        )
        return np.array(value_np, dtype=np.float32).flatten()[:1]

    # ----------------------------------------------------------------
    # Helpers
    # ----------------------------------------------------------------
    def _run_model(self, map_tensor, sym_feat, legal_action):
        self.model.set_eval_mode()
        map_t = torch.tensor(np.asarray(map_tensor)[None], dtype=torch.float32).to(self.device)
        sym_t = torch.tensor(np.asarray(sym_feat)[None], dtype=torch.float32).to(self.device)

        with torch.no_grad():
            logits, value = self.model(map_t, sym_t, inference=True)

        logits_np = logits.cpu().numpy()[0]
        value_np = value.cpu().numpy()[0]

        legal_np = np.asarray(legal_action, dtype=np.float32)
        prob = self._legal_soft_max(logits_np, legal_np)
        return logits_np, value_np, prob

    def _legal_soft_max(self, logits, legal_action):
        """Masked softmax (numpy, numerically stable)."""
        neg_inf = -1e9
        masked = np.where(legal_action > 0.5, logits, neg_inf)
        m = np.max(masked)
        exp = np.exp(masked - m) * (legal_action > 0.5).astype(np.float32)
        total = exp.sum()
        if total <= 0:
            # 极端兜底：若 legal_action 全 0，平均分布
            return np.ones_like(logits, dtype=np.float32) / float(len(logits))
        return exp / total

    def _legal_sample(self, probs, use_max=False):
        probs = np.asarray(probs, dtype=np.float64)
        if use_max:
            return int(np.argmax(probs))
        # 归一化，multinomial 对舍入容忍度较低
        s = probs.sum()
        if s <= 0:
            return int(np.argmax(probs))
        probs = probs / s
        return int(np.argmax(np.random.multinomial(1, probs, size=1)))
