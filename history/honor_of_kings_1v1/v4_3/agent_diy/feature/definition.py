#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Data definitions and trajectory packing for agent_diy.
"""

import collections
import itertools
import random

import numpy as np
import torch
from common_python.utils.common_func import Frame, create_cls

from agent_diy.conf.conf import Config


def _lineup_iterator_shuffle_cycle(camps):
    while True:
        random.shuffle(camps)
        for camp in camps:
            yield camp


def lineup_iterator_roundrobin_camp_heroes(camp_heroes=None, lineup_weights=None):
    if not camp_heroes:
        raise ValueError("camp_heroes is empty")

    camps = []
    for lineups in itertools.product(camp_heroes, camp_heroes):
        weight = 1
        if lineup_weights:
            weight = int(
                lineup_weights.get(
                    tuple(lineups),
                    lineup_weights.get(str(tuple(lineups)), 1),
                )
                or 1
            )
        camps.extend([list(lineups)] * max(1, weight))
    return _lineup_iterator_shuffle_cycle(camps)


ObsData = create_cls("ObsData", feature=None, legal_action=None, lstm_cell=None, lstm_hidden=None)

ActData = create_cls(
    "ActData",
    action=None,
    d_action=None,
    prob=None,
    d_prob=None,
    value=None,
    lstm_cell=None,
    lstm_hidden=None,
)

SampleData = create_cls("SampleData", sample=Config.SAMPLE_DIM)

NONE_ACTION = [0, 15, 15, 15, 15, 0]


def sample_process(collector):
    return collector.sample_process()


def build_frame(agent, observation):
    obs_data, act_data = agent.obs_data, agent.act_data
    frame_state = observation["frame_state"]
    frame_no = frame_state.get("frame_no", frame_state.get("frameNo", 0))
    is_train = False
    for hero in frame_state.get("hero_states", []):
        hero_camp = hero.get("camp")
        hero_hp = hero.get("hp", 0)
        if str(hero_camp) == str(agent.hero_camp):
            is_train = hero_hp > 0
    # When a hard rule overrides the action, mark this frame
    # is_train=False so PPO (model.py:311/354/374) zeros its gradient
    # contribution — the stored prob/action no longer match what was executed.
    if getattr(agent, "rule_override_active", False) or getattr(agent, "recall_override_active", False):
        is_train = False

    feature_vec = np.asarray(obs_data.feature, dtype=np.float32).reshape([-1])
    if feature_vec.shape[0] != Config.FEATURE_DIM:
        raise ValueError(f"feature dim mismatch: {feature_vec.shape[0]} != {Config.FEATURE_DIM}")

    reward = observation["reward"]["reward_sum"]
    sub_action_mask = observation["sub_action_mask"]

    prob, value, action = act_data.prob, act_data.value, act_data.action
    lstm_cell, lstm_hidden = act_data.lstm_cell, act_data.lstm_hidden

    legal_action = _update_legal_action(observation["legal_action"], action)
    frame = Frame(
        frame_no=frame_no,
        feature=feature_vec,
        legal_action=legal_action.reshape([-1]),
        action=action,
        reward=reward,
        reward_sum=0,
        value=np.asarray(value).flatten()[0],
        next_value=0,
        advantage=0,
        prob=prob,
        sub_action=_get_sub_action_mask(sub_action_mask, action[0]),
        lstm_info=np.concatenate([np.asarray(lstm_cell).flatten(), np.asarray(lstm_hidden).flatten()]).reshape([-1]),
        is_train=False if action[0] < 0 else is_train,
    )
    return frame


def _get_sub_action_mask(sub_action_mask, button):
    if isinstance(sub_action_mask, dict):
        return sub_action_mask.get(str(button), sub_action_mask.get(button, [1] * len(Config.LABEL_SIZE_LIST)))
    return sub_action_mask[button]


def _update_legal_action(original_la, action):
    target_size = Config.LABEL_SIZE_LIST[-1]
    top_size = Config.LABEL_SIZE_LIST[0]
    original_la = np.asarray(original_la, dtype=np.float32)
    if original_la.shape[0] == Config.LEGAL_ACTION_DIM:
        return original_la
    if original_la.shape[0] != Config.RAW_LEGAL_ACTION_DIM:
        raise ValueError(f"raw legal_action dim mismatch: {original_la.shape[0]} != {Config.RAW_LEGAL_ACTION_DIM}")
    fix_part = original_la[: -target_size * top_size]
    target_la = original_la[-target_size * top_size :]
    target_la = target_la.reshape([top_size, target_size])[action[0]]
    return np.concatenate([fix_part, target_la], axis=0)


class FrameCollector:
    def __init__(self, num_agents):
        self._data_shapes = Config.data_shapes
        self._LSTM_FRAME = Config.LSTM_TIME_STEPS
        self.num_agents = num_agents
        self.rl_data_map = [collections.OrderedDict() for _ in range(num_agents)]
        self.m_replay_buffer = [[] for _ in range(num_agents)]
        self.gamma = Config.GAMMA
        self.lamda = Config.LAMDA

    def reset(self, num_agents):
        self.num_agents = num_agents
        self.rl_data_map = [collections.OrderedDict() for _ in range(self.num_agents)]
        self.m_replay_buffer = [[] for _ in range(self.num_agents)]

    def save_frame(self, rl_data_info, agent_id):
        reward = self._clip_reward(rl_data_info.reward)

        if len(self.rl_data_map[agent_id]) > 0:
            last_key = list(self.rl_data_map[agent_id].keys())[-1]
            last_rl_data_info = self.rl_data_map[agent_id][last_key]
            last_rl_data_info.next_value = rl_data_info.value
            last_rl_data_info.reward = reward

        rl_data_info.reward = 0
        self.rl_data_map[agent_id][rl_data_info.frame_no] = rl_data_info

    def save_last_frame(self, reward, agent_id):
        if len(self.rl_data_map[agent_id]) > 0:
            last_key = list(self.rl_data_map[agent_id].keys())[-1]
            last_rl_data_info = self.rl_data_map[agent_id][last_key]
            last_rl_data_info.next_value = 0
            last_rl_data_info.reward = reward

    def sample_process(self):
        self._calc_reward()
        self._format_data()
        return self.m_replay_buffer

    def _calc_reward(self):
        for i in range(self.num_agents):
            reversed_keys = list(self.rl_data_map[i].keys())
            reversed_keys.reverse()
            gae = 0.0
            for frame_no in reversed_keys:
                rl_info = self.rl_data_map[i][frame_no]
                delta = -rl_info.value + rl_info.reward + self.gamma * rl_info.next_value
                gae = gae * self.gamma * self.lamda + delta
                rl_info.advantage = gae
                rl_info.reward_sum = gae + rl_info.value

    def _reshape_lstm_batch_sample(self, sample_batch, sample_lstm):
        sample = np.zeros([np.prod(sample_batch.shape) + np.prod(sample_lstm.shape)], dtype=np.float32)
        idx, s_idx = 0, 0

        sample[-sample_lstm.shape[0] :] = sample_lstm
        for split_shape in self._data_shapes[:-2]:
            one_shape = split_shape[0] // self._LSTM_FRAME
            sample[s_idx : s_idx + split_shape[0]] = sample_batch[:, idx : idx + one_shape].reshape([-1])
            idx += one_shape
            s_idx += split_shape[0]

        if sample.shape[0] != Config.SAMPLE_DIM:
            raise ValueError(f"sample dim mismatch: {sample.shape[0]} != {Config.SAMPLE_DIM}")
        return sample.astype(np.float32)

    def _append_lstm_sample(self, agent_idx, sample_batch, sample_lstm, valid_steps):
        if not 0 < valid_steps <= self._LSTM_FRAME:
            raise ValueError(f"invalid lstm valid steps: {valid_steps}")
        if valid_steps < self._LSTM_FRAME:
            sample_batch[valid_steps:, :] = 0.0
        sample_array = self._reshape_lstm_batch_sample(sample_batch, sample_lstm)
        self.m_replay_buffer[agent_idx].append(SampleData(sample=torch.from_numpy(sample_array)))

    def _format_data(self):
        sample_one_size = np.sum(self._data_shapes[:-2]) // self._LSTM_FRAME
        sample_lstm_size = np.sum(self._data_shapes[-2:])
        sample_batch = np.zeros([self._LSTM_FRAME, sample_one_size], dtype=np.float32)

        for agent_idx in range(self.num_agents):
            sample_lstm = np.zeros([sample_lstm_size], dtype=np.float32)
            cnt = 0
            for frame_no in self.rl_data_map[agent_idx]:
                rl_info = self.rl_data_map[agent_idx][frame_no]
                idx = 0

                idx = self._assign(sample_batch[cnt], idx, rl_info.feature)
                idx = self._assign(sample_batch[cnt], idx, rl_info.legal_action)
                sample_batch[cnt, idx] = rl_info.reward_sum
                idx += 1
                sample_batch[cnt, idx] = rl_info.advantage
                idx += 1
                idx = self._assign(sample_batch[cnt], idx, rl_info.action, expected=6)

                for p in rl_info.prob:
                    idx = self._assign(sample_batch[cnt], idx, p)

                idx = self._assign(sample_batch[cnt], idx, rl_info.sub_action, expected=6)
                sample_batch[cnt, idx] = rl_info.is_train
                idx += 1

                assert idx == sample_one_size, f"Sample check failed, {idx}/{sample_one_size}"

                cnt += 1
                if cnt == self._LSTM_FRAME:
                    self._append_lstm_sample(agent_idx, sample_batch, sample_lstm, cnt)
                    sample_lstm = rl_info.lstm_info.astype(np.float32)
                    cnt = 0

            if cnt > 0:
                self._append_lstm_sample(agent_idx, sample_batch, sample_lstm, cnt)

    def _assign(self, target, idx, value, expected=None):
        arr = np.asarray(value, dtype=np.float32).reshape([-1])
        if expected is not None and arr.shape[0] != expected:
            raise ValueError(f"sample field dim mismatch: {arr.shape[0]} != {expected}")
        target[idx : idx + arr.shape[0]] = arr
        return idx + arr.shape[0]

    def _clip_reward(self, reward, max=100, min=-100):
        return max if reward > max else min if reward < min else reward

    def __len__(self):
        if not self.rl_data_map:
            return 0
        return max(len(agent_samples) for agent_samples in self.rl_data_map)
