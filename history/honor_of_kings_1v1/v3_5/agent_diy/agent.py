#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""

import hashlib
import json
import os
import random

import numpy as np
import torch
from torch.optim.lr_scheduler import LambdaLR

try:
    from kaiwu_agent.agent.base_agent import (
        BaseAgent,
        predict_wrapper,
        exploit_wrapper,
        learn_wrapper,
        save_model_wrapper,
        load_model_wrapper,
        reset_wrapper,
        load_opponent_agent_wrapper,
    )
    from kaiwu_agent.utils.common_func import attached
except ModuleNotFoundError:
    from kaiwudrl.interface.agent import BaseAgent

    def _identity_wrapper(func):
        return func

    predict_wrapper = _identity_wrapper
    exploit_wrapper = _identity_wrapper
    learn_wrapper = _identity_wrapper
    save_model_wrapper = _identity_wrapper
    load_model_wrapper = _identity_wrapper
    reset_wrapper = _identity_wrapper
    load_opponent_agent_wrapper = _identity_wrapper
    attached = _identity_wrapper

from agent_diy.algorithm.algorithm import Algorithm
from agent_diy.conf.conf import Args, Config, GameConfig
from agent_diy.feature.definition import NONE_ACTION, ActData, ObsData
from agent_diy.feature.feature_process import FeatureProcess
from agent_diy.feature.reward_process import GameRewardManager
from agent_diy.model.model import Model


torch.set_num_threads(1)
torch.set_num_interop_threads(1)


SUMMONER_SKILL_MAP = {
    80102: "治疗",
    80109: "疾跑",
    80104: "惩击",
    80108: "终结",
    80110: "狂暴",
    80105: "干扰",
    80103: "晕眩",
    80107: "净化",
    80121: "弱化",
    80115: "闪现",
}

TOWER_SUB_TYPES = {21, "21", "ACTOR_SUB_TOWER"}
SOLDIER_SUB_TYPES = {1, "1", 11, "11", "ACTOR_SUB_SOLDIER"}


@attached
class Agent(BaseAgent):
    def __init__(self, agent_type="player", device=None, logger=None, monitor=None):
        self.cur_model_name = ""
        self.device = device
        self.model = Model().to(self.device)
        self.model = self.model.to(memory_format=torch.channels_last)

        self.lstm_unit_size = Config.LSTM_UNIT_SIZE
        self.lstm_hidden = np.zeros([self.lstm_unit_size], dtype=np.float32)
        self.lstm_cell = np.zeros([self.lstm_unit_size], dtype=np.float32)
        self.label_size_list = Config.LABEL_SIZE_LIST
        self.legal_action_size = Config.LEGAL_ACTION_SIZE_LIST
        self.seri_vec_split_shape = Config.SERI_VEC_SPLIT_SHAPE

        self.hero_camp = 0
        self.player_id = 0
        self.env_id = None
        self.summoner_skill_counter = {}

        self.train_step = 0
        self.lr = Config.INIT_LEARNING_RATE_START
        self.optimizer = torch.optim.Adam(params=self.model.parameters(), lr=self.lr, betas=(0.9, 0.999), eps=1e-8)
        self.parameters = [p for param_group in self.optimizer.param_groups for p in param_group["params"]]
        self.target_lr = Config.TARGET_LR
        self.target_step = Config.TARGET_STEP
        self.scheduler = LambdaLR(self.optimizer, lr_lambda=self.lr_lambda)

        self.reward_manager = None
        self.feature_processes = None
        self.logger = logger
        self.monitor = monitor
        self.algorithm = Algorithm(self.model, self.optimizer, self.scheduler, self.device, self.logger, self.monitor)

        # Force-home rule state (used by _maybe_force_home).
        self.own_cake_exists = False
        self.own_cake_seen_once = False
        self.last_own_cake_pos = None
        self.last_own_cake_disappear_frame = -10000
        self.next_own_cake_frame = None
        self.prev_dead_cnt = 0
        self.prev_enemy_dead_cnt = 0
        self.last_enemy_hero_kill_frame = -10000
        self.last_recover_success_frame = -10000
        self.last_cake_eaten_frame = -10000
        self.force_home_pending_recover = None
        self.force_home_camp = None
        self.force_home_phase = None
        self.force_home_post_kill_active = False
        self.force_home_path_camp = None
        self.force_home_path_points = []
        self.force_home_path_ready = False
        self.force_home_path_index = None
        self.force_home_progress_phase = None
        self.force_home_progress_index = None
        self.force_home_progress_best_dist = None
        self.force_home_stuck_frames = 0
        self.rule_override_active = False
        self.recall_override_active = False
        self.cleanse_override_active = False
        self.rule_override_count = 0
        self.force_home_trigger_count = 0
        self.force_home_override_count = 0
        self.force_home_start_count = 0
        self.force_home_retreat_count = 0
        self.force_home_return_count = 0
        self.opening_unstuck_count = 0
        self.opening_unstuck_last_pos = None
        self.opening_unstuck_last_frame = None
        self.opening_unstuck_last_trigger_frame = -10000
        self.opening_unstuck_camp = None
        self.luban_skill1_aim_assist_count = 0
        self.cleanse_override_count = 0
        self.skill2_blocked_count = 0
        self.skill2_total_cast_count = 0
        self.skill2_cast_outside_window_count = 0
        self._enemy_ult_cast_frame = None
        self._pending_cleanse_frame = None
        self._fallback_cleanse_ult_cast_frame = None
        self._prev_enemy_slot3_hit_hero_times = None

        super().__init__(agent_type, device, logger, monitor)

    def lr_lambda(self, step):
        if step > self.target_step:
            return self.target_lr / self.lr
        return 1.0 - ((1.0 - self.target_lr / self.lr) * step / self.target_step)

    def init_config(self, config_data):
        my_heroes = config_data.get("my_heroes", [])
        opponent_heroes = config_data.get("opponent_heroes", [])
        opponent_hero = opponent_heroes[0] if opponent_heroes else None
        is_eval = bool(config_data.get("is_eval", False))
        forced_skill = config_data.get("forced_summoner_skill", None)
        select_skills = {}
        for hero_id in my_heroes:
            if forced_skill is not None:
                select_skills[hero_id] = int(forced_skill)
            elif is_eval:
                select_skills[hero_id] = self._default_summoner_skill(hero_id, opponent_hero)
            else:
                select_skills[hero_id] = self._select_train_summoner_skill(hero_id, opponent_hero)
        return select_skills

    def _default_summoner_skill(self, my_hero, opponent_hero):
        # Eval / match always uses the configured default (currently 80110 狂暴).
        # Training cycle path also falls back here when candidates are exhausted.
        return self._matchup_summoner_skill(my_hero, opponent_hero)

    def _matchup_summoner_skill(self, my_hero, opponent_hero):
        candidates = list(getattr(GameConfig, "DUEL_SUMMONER_SKILL_IDS", []))
        candidates = [int(skill_id) for skill_id in candidates if int(skill_id) in GameConfig.SUMMONER_SKILL_IDS]
        if not candidates:
            return GameConfig.DEFAULT_SUMMONER_SKILL

        try:
            matchup = (int(my_hero), int(opponent_hero or 0))
        except (TypeError, ValueError):
            return GameConfig.DEFAULT_SUMMONER_SKILL

        winrate_table = getattr(GameConfig, "SUMMONER_SKILL_MATCHUP_WINRATE", {})
        skill_scores = winrate_table.get(matchup, {})
        if not skill_scores:
            return GameConfig.DEFAULT_SUMMONER_SKILL

        best_skill = int(GameConfig.DEFAULT_SUMMONER_SKILL)
        best_score = float(skill_scores.get(best_skill, 0.0) or 0.0)
        for skill_id in candidates:
            score = float(skill_scores.get(skill_id, 0.0) or 0.0)
            if score > best_score:
                best_skill = int(skill_id)
                best_score = score
        return best_skill

    def _select_train_summoner_skill(self, my_hero, opponent_hero):
        default_skill = self._default_summoner_skill(my_hero, opponent_hero)
        candidates = list(GameConfig.SUMMONER_SKILL_CANDIDATES_BY_HERO.get(my_hero, []))
        if default_skill not in candidates:
            candidates.insert(0, default_skill)
        candidates = [skill_id for skill_id in candidates if skill_id in GameConfig.SUMMONER_SKILL_IDS]
        if not candidates:
            return default_skill

        mode = getattr(GameConfig, "SUMMONER_SKILL_TRAIN_MODE", "cycle")
        if mode == "fixed":
            return default_skill
        if mode == "random":
            return random.choice(candidates)

        counter_key = (int(my_hero), int(opponent_hero or 0))
        counter = self.summoner_skill_counter.get(counter_key, 0)
        self.summoner_skill_counter[counter_key] = counter + 1
        return candidates[counter % len(candidates)]

    @reset_wrapper
    def reset(self, observation):
        self.hero_camp = observation.get("player_camp", observation.get("camp", 0))
        self.player_id = observation.get("player_id", 0)
        self.lstm_hidden = np.zeros([self.lstm_unit_size], dtype=np.float32)
        self.lstm_cell = np.zeros([self.lstm_unit_size], dtype=np.float32)
        self.reward_manager = GameRewardManager(self.player_id)
        self.feature_processes = FeatureProcess(self.hero_camp, logger=self.logger)
        # Reset force-home rule state per episode.
        self.own_cake_exists = False
        self.own_cake_seen_once = False
        self.last_own_cake_pos = None
        self.last_own_cake_disappear_frame = -10000
        self.next_own_cake_frame = None
        self.prev_dead_cnt = 0
        self.prev_enemy_dead_cnt = 0
        self.last_enemy_hero_kill_frame = -10000
        self.last_recover_success_frame = -10000
        self.last_cake_eaten_frame = -10000
        self.force_home_pending_recover = None
        self.force_home_camp = None
        self.force_home_phase = None
        self.force_home_post_kill_active = False
        self.force_home_path_camp = None
        self.force_home_path_points = []
        self.force_home_path_ready = False
        self.force_home_path_index = None
        self.force_home_progress_phase = None
        self.force_home_progress_index = None
        self.force_home_progress_best_dist = None
        self.force_home_stuck_frames = 0
        self.rule_override_active = False
        self.recall_override_active = False
        self.cleanse_override_active = False
        self.rule_override_count = 0
        self.force_home_trigger_count = 0
        self.force_home_override_count = 0
        self.force_home_start_count = 0
        self.force_home_retreat_count = 0
        self.force_home_return_count = 0
        self.opening_unstuck_count = 0
        self.opening_unstuck_last_pos = None
        self.opening_unstuck_last_frame = None
        self.opening_unstuck_last_trigger_frame = -10000
        self.opening_unstuck_camp = None
        self.luban_skill1_aim_assist_count = 0
        self.cleanse_override_count = 0
        self.skill2_blocked_count = 0
        self.skill2_total_cast_count = 0
        self.skill2_cast_outside_window_count = 0
        self._enemy_ult_cast_frame = None
        self._pending_cleanse_frame = None
        self._fallback_cleanse_ult_cast_frame = None
        self._prev_enemy_slot3_hit_hero_times = None

    def _model_inference(self, list_obs_data):
        feature = [obs_data.feature for obs_data in list_obs_data]
        legal_action = [obs_data.legal_action for obs_data in list_obs_data]
        lstm_cell = [obs_data.lstm_cell for obs_data in list_obs_data]
        lstm_hidden = [obs_data.lstm_hidden for obs_data in list_obs_data]

        input_list = [np.array(feature), np.array(lstm_cell), np.array(lstm_hidden)]
        torch_inputs = [torch.from_numpy(nparr).to(torch.float32) for nparr in input_list]
        for i, data in enumerate(torch_inputs):
            torch_inputs[i] = data.reshape(-1).float().to(self.device)

        feature, lstm_cell, lstm_hidden = torch_inputs
        feature_vec = feature.reshape(-1, self.seri_vec_split_shape[0][0])
        lstm_hidden_state = lstm_hidden.reshape(-1, self.lstm_unit_size)
        lstm_cell_state = lstm_cell.reshape(-1, self.lstm_unit_size)

        format_inputs = [feature_vec, lstm_hidden_state, lstm_cell_state]

        self.model.set_eval_mode()
        with torch.no_grad():
            output_list = self.model(format_inputs, inference=True)

        np_output = [output.detach().cpu().numpy() for output in output_list]
        logits, value, next_lstm_cell, next_lstm_hidden = np_output[:4]

        next_lstm_cell = next_lstm_cell.squeeze(axis=0)
        next_lstm_hidden = next_lstm_hidden.squeeze(axis=0)

        list_act_data = []
        for i in range(len(legal_action)):
            prob, d_prob, action, d_action = self._sample_masked_action(logits[i], legal_action[i])
            list_act_data.append(
                ActData(
                    action=action,
                    d_action=d_action,
                    prob=prob,
                    d_prob=d_prob,
                    value=value[i],
                    lstm_cell=next_lstm_cell[i],
                    lstm_hidden=next_lstm_hidden[i],
                )
            )
        return list_act_data

    @predict_wrapper
    def predict(self, observation):
        self._reset_rule_override_flags()
        policy_observation = self._prepare_policy_observation(observation)
        obs_data = self.observation_process(policy_observation)
        act_data = self._model_inference([obs_data])[0]
        self._maybe_aim_luban_skill1(observation, act_data, is_stochastic=True)
        self.update_status(obs_data, act_data)
        action = self.action_process(observation, act_data, True)
        action = self._maybe_auto_cleanse(observation, action)
        action = self._maybe_opening_unstuck(observation, action)
        action = self._maybe_force_home(observation, action)
        self._track_skill2_action_stats(observation, action)
        return action

    @exploit_wrapper
    def exploit(self, observation):
        self._reset_rule_override_flags()
        policy_observation = self._prepare_policy_observation(observation)
        obs_data = self.observation_process(policy_observation)
        act_data = self._model_inference([obs_data])[0]
        self._maybe_aim_luban_skill1(observation, act_data, is_stochastic=False)
        self.update_status(obs_data, act_data)
        action = self.action_process(observation, act_data, False)
        action = self._maybe_auto_cleanse(observation, action)
        action = self._maybe_opening_unstuck(observation, action)
        action = self._maybe_force_home(observation, action)
        self._track_skill2_action_stats(observation, action)
        return action

    def observation_process(self, observation):
        if self.feature_processes is None:
            self.feature_processes = FeatureProcess(observation.get("player_camp", observation.get("camp", 0)))
        feature = self.feature_processes.process_feature(observation)
        if len(feature) != Config.FEATURE_DIM:
            raise ValueError(f"feature dim mismatch: {len(feature)} != {Config.FEATURE_DIM}")
        legal_action = observation["legal_action"]
        if len(legal_action) not in (Config.LEGAL_ACTION_DIM, Config.RAW_LEGAL_ACTION_DIM):
            raise ValueError(
                f"legal_action dim mismatch: {len(legal_action)} not in "
                f"({Config.LEGAL_ACTION_DIM}, {Config.RAW_LEGAL_ACTION_DIM})"
            )
        return ObsData(
            feature=feature,
            legal_action=legal_action,
            lstm_cell=self.lstm_cell,
            lstm_hidden=self.lstm_hidden,
        )

    def action_process(self, observation, act_data, is_stochastic):
        action = act_data.action if is_stochastic else act_data.d_action
        return self._normalize_action(action)

    def _maybe_aim_luban_skill1(self, observation, act_data, is_stochastic):
        if not getattr(GameConfig, "LUBAN_SKILL1_AIM_ASSIST", False):
            return
        attr = "action" if is_stochastic else "d_action"
        action = self._normalize_action(getattr(act_data, attr, None))
        # Only correct the sub-actions after the policy has already selected
        # Luban skill 1. The rule does not force skill usage.
        if len(action) != 6 or action[0] != 4:
            return

        frame_state = observation.get("frame_state", {}) or {}
        main_hero, enemy_hero, _ = self._find_my_hero_and_tower(frame_state)
        if self._hero_config_id(main_hero) != 112 or enemy_hero is None:
            return
        if self._distance_between_heroes(main_hero, enemy_hero) > GameConfig.LUBAN_SKILL1_AIM_RANGE:
            return

        center = int(GameConfig.LUBAN_SKILL1_AIM_CENTER)
        target = int(GameConfig.LUBAN_SKILL1_AIM_TARGET)
        if not self._is_luban_skill1_aim_legal(observation, center, target):
            return

        action[3] = center
        action[4] = center
        action[5] = target
        setattr(act_data, attr, action)
        self.rule_override_active = True
        self.rule_override_count += 1
        self.luban_skill1_aim_assist_count += 1

    def _is_luban_skill1_aim_legal(self, observation, center, target):
        legal_action = observation.get("legal_action", [])
        if legal_action is None:
            legal_action = []
        if hasattr(legal_action, "tolist"):
            legal_action = legal_action.tolist()
        if len(legal_action) not in (Config.LEGAL_ACTION_DIM, Config.RAW_LEGAL_ACTION_DIM):
            return False
        offsets = [0]
        for size in Config.LABEL_SIZE_LIST[:-1]:
            offsets.append(offsets[-1] + size)
        button = 4
        if int(legal_action[button] or 0) != 1:
            return False
        skill_x_offset = offsets[3]
        skill_z_offset = offsets[4]
        if not (0 <= center < Config.LABEL_SIZE_LIST[3]):
            return False
        if int(legal_action[skill_x_offset + center] or 0) != 1:
            return False
        if int(legal_action[skill_z_offset + center] or 0) != 1:
            return False
        target_offset = offsets[5]
        if len(legal_action) == Config.RAW_LEGAL_ACTION_DIM:
            target_offset += button * Config.LABEL_SIZE_LIST[-1]
        return 0 <= target < Config.LABEL_SIZE_LIST[-1] and int(legal_action[target_offset + target] or 0) == 1

    def _hero_config_id(self, hero):
        if not hero:
            return 0
        return int(hero.get("config_id", hero.get("configId", 0)) or 0)

    def _hero_location(self, hero):
        if not hero:
            return None
        collider = hero.get("collider", {}) or {}
        loc = collider.get("location", None) or hero.get("location", None)
        if not loc:
            return None
        return float(loc.get("x", 0) or 0), float(loc.get("z", 0) or 0)

    def _distance_between_heroes(self, hero_a, hero_b):
        loc_a = self._hero_location(hero_a)
        loc_b = self._hero_location(hero_b)
        if loc_a is None or loc_b is None:
            return float("inf")
        dx = loc_a[0] - loc_b[0]
        dz = loc_a[1] - loc_b[1]
        return (dx * dx + dz * dz) ** 0.5

    def _dist(self, a, b, default=1e9):
        if a is None or b is None:
            return default
        dx = float(a[0]) - float(b[0])
        dz = float(a[1]) - float(b[1])
        return (dx * dx + dz * dz) ** 0.5

    def _maybe_auto_cleanse(self, observation, action):
        frame_state = observation.get("frame_state", {}) or {}
        main_hero, enemy_hero, _ = self._find_my_hero_and_tower(frame_state)
        if main_hero is None or enemy_hero is None:
            self._pending_cleanse_frame = None
            return action

        if self._hero_config_id(main_hero) != 133 or self._hero_config_id(enemy_hero) != 133:
            self._pending_cleanse_frame = None
            return action

        frame_no = self._frame_no(frame_state)
        pending_cleanse_frame = self._pending_cleanse_frame
        if pending_cleanse_frame is not None and frame_no >= int(pending_cleanse_frame):
            self._pending_cleanse_frame = None
            if self._is_skill2_available(main_hero):
                cleanse_action = self._legalized_rule_action(observation, [5, 15, 15, 15, 15, 2])
                if cleanse_action is not None:
                    self.rule_override_active = True
                    self.cleanse_override_active = True
                    self.rule_override_count += 1
                    self.cleanse_override_count += 1
                    return cleanse_action

        fallback_action = self._maybe_force_cleanse_after_enemy_ult(observation, main_hero, frame_no)
        if fallback_action is not None:
            return fallback_action

        if not self._was_hit_by_enemy_ult(main_hero, enemy_hero):
            return action

        if self._pending_cleanse_frame is None:
            self._pending_cleanse_frame = frame_no + 1
        return action

    def _maybe_force_cleanse_after_enemy_ult(self, observation, main_hero, frame_no):
        last_cast = self._enemy_ult_cast_frame
        if last_cast is None:
            return None
        if self._fallback_cleanse_ult_cast_frame == last_cast:
            return None
        elapsed = int(frame_no or 0) - int(last_cast)
        trigger_frame = int(GameConfig.DI_RENJIE_SKILL2_UNMASK_AFTER_ULT_START)
        unmask_end = int(GameConfig.DI_RENJIE_SKILL2_UNMASK_AFTER_ULT_END)
        if elapsed < trigger_frame or elapsed > unmask_end:
            return None
        self._fallback_cleanse_ult_cast_frame = last_cast
        if not self._is_skill2_available(main_hero):
            return None
        cleanse_action = self._legalized_rule_action(observation, [5, 15, 15, 15, 15, 2])
        if cleanse_action is None:
            return None
        self.rule_override_active = True
        self.cleanse_override_active = True
        self.rule_override_count += 1
        self.cleanse_override_count += 1
        return cleanse_action

    def _maybe_opening_unstuck(self, observation, action):
        frame_state = observation.get("frame_state", {}) or {}
        frame_no = self._frame_no(frame_state)
        if not (
            int(GameConfig.OPENING_UNSTUCK_START_FRAME)
            <= frame_no
            <= int(GameConfig.OPENING_UNSTUCK_END_FRAME)
        ):
            self._reset_opening_unstuck_track()
            return action
        if self.cleanse_override_active:
            return action

        current_camp = self._resolve_current_camp(observation, frame_state)
        if current_camp not in (1, 2):
            self._reset_opening_unstuck_track()
            return action
        if self.opening_unstuck_camp not in (None, current_camp):
            self._reset_opening_unstuck_track()
        self.opening_unstuck_camp = current_camp

        main_hero, _, _ = self._find_my_hero_and_tower(frame_state)
        if main_hero is None or self._unit_hp(main_hero) <= 0:
            self._reset_opening_unstuck_track()
            return action

        pos = self._project_own_perspective(main_hero)
        if pos is None:
            self._reset_opening_unstuck_track()
            return action

        prev_pos = self.opening_unstuck_last_pos
        prev_frame = self.opening_unstuck_last_frame
        self.opening_unstuck_last_pos = (float(pos[0]), float(pos[1]))
        self.opening_unstuck_last_frame = frame_no

        if prev_pos is None or prev_frame is None or frame_no <= int(prev_frame):
            return action
        if frame_no - int(self.opening_unstuck_last_trigger_frame) < int(GameConfig.OPENING_UNSTUCK_COOLDOWN_FRAMES):
            return action
        if self._dist(pos, prev_pos) > float(GameConfig.OPENING_UNSTUCK_MIN_MOVE):
            return action

        target = self._opening_forward_target(pos)
        if target is None:
            return action
        move_action = self._move_action_towards(main_hero, target)
        if move_action is None:
            return action
        legal_move = self._legalized_rule_action(observation, move_action, active_heads=(1, 2))
        if legal_move is None:
            return action

        self.rule_override_active = True
        self.rule_override_count += 1
        self.opening_unstuck_count += 1
        self.opening_unstuck_last_trigger_frame = frame_no
        return legal_move

    def _reset_opening_unstuck_track(self):
        self.opening_unstuck_last_pos = None
        self.opening_unstuck_last_frame = None
        self.opening_unstuck_camp = None

    def _opening_forward_target(self, projected_pos):
        if projected_pos is None:
            return None
        lane = float(projected_pos[0]) + float(GameConfig.OPENING_UNSTUCK_FORWARD_DELTA)
        width = float(projected_pos[1])
        return self._unproject_own_perspective((lane, width))

    def _prepare_policy_observation(self, observation):
        self._track_enemy_ult_cast(observation)
        delay_observation = self._maybe_delay_cleanse_hit_frame(observation)
        if delay_observation is not observation:
            return delay_observation
        return self._maybe_block_skill2(observation)

    def _maybe_delay_cleanse_hit_frame(self, observation):
        frame_state = observation.get("frame_state", {}) or {}
        main_hero, enemy_hero, _ = self._find_my_hero_and_tower(frame_state)
        if main_hero is None or enemy_hero is None:
            return observation
        if self._hero_config_id(main_hero) != 133 or self._hero_config_id(enemy_hero) != 133:
            return observation
        if not self._was_hit_by_enemy_ult(main_hero, enemy_hero):
            return observation

        frame_no = self._frame_no(frame_state)
        if self._pending_cleanse_frame is None or frame_no >= int(self._pending_cleanse_frame):
            self._pending_cleanse_frame = frame_no + 1
        return self._mask_skill2_button(observation)

    def _mask_skill2_button(self, observation):
        legal_action = observation.get("legal_action", [])
        if legal_action is None:
            return observation
        legal_len = len(legal_action)
        if legal_len not in (Config.LEGAL_ACTION_DIM, Config.RAW_LEGAL_ACTION_DIM):
            return observation
        if int(legal_action[5] or 0) == 0:
            return observation

        if hasattr(legal_action, "copy"):
            masked_legal_action = legal_action.copy()
        else:
            masked_legal_action = list(legal_action)
        masked_legal_action[5] = 0

        masked_observation = dict(observation)
        masked_observation["legal_action"] = masked_legal_action
        return masked_observation

    def _was_hit_by_enemy_ult(self, main_hero, enemy_hero):
        main_runtime = self._actor_runtime(main_hero)
        enemy_runtime = self._actor_runtime(enemy_hero)
        current_hit_times = self._slot_hit_hero_times(enemy_hero, 3)
        previous_hit_times = self._prev_enemy_slot3_hit_hero_times
        for hurt in self._get_any(main_hero or {}, ["take_hurt_infos", "takeHurtInfos"], []) or []:
            if not isinstance(hurt, dict):
                continue
            atker = self._get_any(hurt, ["atker", "attacker"], None)
            slot = self._slot_idx(self._get_any(hurt, ["skillSlot", "skill_slot"], -1))
            if atker is not None and enemy_runtime is not None and str(atker) == str(enemy_runtime) and slot == 3:
                self._prev_enemy_slot3_hit_hero_times = current_hit_times
                return True
        if main_runtime is not None:
            for hit in self._get_any(enemy_hero or {}, ["hit_target_info", "hitTargetInfo"], []) or []:
                if not isinstance(hit, dict):
                    continue
                target_runtime = self._get_any(hit, ["hit_target", "hitTarget"], None)
                slot = self._slot_idx(self._get_any(hit, ["slot_type", "slotType"], -1))
                if target_runtime is not None and str(target_runtime) == str(main_runtime) and slot == 3:
                    self._prev_enemy_slot3_hit_hero_times = current_hit_times
                    return True
        self._prev_enemy_slot3_hit_hero_times = current_hit_times
        if current_hit_times is None or previous_hit_times is None:
            return False
        return current_hit_times > previous_hit_times

    def _slot_hit_hero_times(self, hero, target_slot):
        values = []
        for slot in self._slot_states(hero):
            if self._slot_idx(self._get_any(slot, ["slot_type", "slotType"])) != target_slot:
                continue
            try:
                values.append(float(self._get_any(slot, ["hitHeroTimes", "hit_hero_times"], 0) or 0))
            except (TypeError, ValueError):
                continue
        return max(values) if values else None

    def _track_enemy_ult_cast(self, observation):
        frame_state = observation.get("frame_state", {}) or {}
        main_hero, enemy_hero, _ = self._find_my_hero_and_tower(frame_state)
        if main_hero is None or enemy_hero is None:
            return
        if self._hero_config_id(main_hero) != 133 or self._hero_config_id(enemy_hero) != 133:
            return
        if self._slot_succ_used(enemy_hero, 3):
            frame_no = self._frame_no(frame_state)
            self._enemy_ult_cast_frame = frame_no

    def _maybe_block_skill2(self, observation):
        frame_state = observation.get("frame_state", {}) or {}
        main_hero, enemy_hero, _ = self._find_my_hero_and_tower(frame_state)
        if not self._should_block_skill2(frame_state, main_hero, enemy_hero):
            return observation

        legal_action = observation.get("legal_action", [])
        if legal_action is None:
            return observation
        legal_len = len(legal_action)
        if legal_len not in (Config.LEGAL_ACTION_DIM, Config.RAW_LEGAL_ACTION_DIM):
            return observation
        if int(legal_action[5] or 0) == 0:
            return observation

        if hasattr(legal_action, "copy"):
            masked_legal_action = legal_action.copy()
        else:
            masked_legal_action = list(legal_action)
        masked_legal_action[5] = 0

        masked_observation = dict(observation)
        masked_observation["legal_action"] = masked_legal_action
        self.skill2_blocked_count += 1
        return masked_observation

    def _should_block_skill2(self, frame_state, main_hero, enemy_hero):
        if main_hero is None or enemy_hero is None:
            return False
        if self._hero_config_id(main_hero) != 133 or self._hero_config_id(enemy_hero) != 133:
            return False
        enemy_hp = self._unit_hp(enemy_hero)
        if enemy_hp <= 0:
            return False
        enemy_level = int(self._get_any(enemy_hero, ["level"], 0) or 0)
        if enemy_level < 4:
            return False
        last_cast = self._enemy_ult_cast_frame
        frame_no = self._frame_no(frame_state)
        if last_cast is None:
            return True
        elapsed = frame_no - int(last_cast)
        unmask_start = int(GameConfig.DI_RENJIE_SKILL2_UNMASK_AFTER_ULT_START)
        unmask_end = int(GameConfig.DI_RENJIE_SKILL2_UNMASK_AFTER_ULT_END)
        return elapsed < unmask_start or elapsed > unmask_end

    def _track_skill2_action_stats(self, observation, action):
        action = self._normalize_action(action)
        if len(action) != 6 or int(action[0]) != 5:
            return
        self.skill2_total_cast_count += 1
        frame_state = observation.get("frame_state", {}) or {}
        main_hero, enemy_hero, _ = self._find_my_hero_and_tower(frame_state)
        if self._should_block_skill2(frame_state, main_hero, enemy_hero):
            self.skill2_cast_outside_window_count += 1

    def _actor_runtime(self, actor):
        if actor is None:
            return None
        actor_state = actor.get("actor_state") or actor.get("actorState")
        if actor_state is not None:
            runtime_id = actor_state.get("runtime_id") or actor_state.get("runtimeId")
            if runtime_id is not None:
                return runtime_id
        return actor.get("runtime_id") or actor.get("runtimeId")

    def _is_skill2_available(self, main_hero):
        slots = self._slot_states(main_hero)
        for slot in slots:
            if self._slot_idx(self._get_any(slot, ["slot_type", "slotType"])) != 2:
                continue
            usable = bool(slot.get("usable", False))
            cooldown = float(slot.get("cooldown", 0) or 0)
            return usable and cooldown <= 0
        return False

    def _slot_succ_used(self, hero, target_slot):
        for slot in self._slot_states(hero):
            if self._slot_idx(self._get_any(slot, ["slot_type", "slotType"])) != target_slot:
                continue
            try:
                return float(self._get_any(slot, ["succUsedInFrame", "succ_used_in_frame"], 0) or 0) > 0
            except (TypeError, ValueError):
                return False
        return False

    def _frame_no(self, frame_state):
        return int(frame_state.get("frame_no", frame_state.get("frameNo", 0)) or 0)

    def _slot_states(self, hero):
        skill_state = self._get_any(hero or {}, ["skill_state", "skillState"], {}) or {}
        return self._get_any(skill_state, ["slot_states", "slotStates"], []) or []

    def _get_any(self, obj, keys, default=None):
        for key in keys:
            if isinstance(obj, dict):
                value = obj.get(key, None)
            else:
                value = getattr(obj, key, None)
            if value is not None:
                return value
        return default

    def _reset_rule_override_flags(self):
        self.rule_override_active = False
        self.recall_override_active = False
        self.cleanse_override_active = False

    def _legalized_rule_action(self, observation, preferred_action, active_heads=None):
        legal_action = observation.get("legal_action", [])
        if legal_action is None:
            legal_action = []
        if hasattr(legal_action, "tolist"):
            legal_action = legal_action.tolist()
        if len(legal_action) not in (Config.LEGAL_ACTION_DIM, Config.RAW_LEGAL_ACTION_DIM):
            return None

        preferred_action = self._normalize_action(preferred_action)
        button = preferred_action[0]
        if not (0 <= button < Config.LABEL_SIZE_LIST[0]):
            return None
        if int(legal_action[button] or 0) != 1:
            return None

        offsets = [0]
        for size in Config.LABEL_SIZE_LIST[:-1]:
            offsets.append(offsets[-1] + size)

        action = list(preferred_action)
        active_heads = set(range(1, len(Config.LABEL_SIZE_LIST))) if active_heads is None else set(active_heads)
        for head_idx in range(1, len(Config.LABEL_SIZE_LIST)):
            if head_idx not in active_heads:
                continue
            if head_idx == len(Config.LABEL_SIZE_LIST) - 1 and len(legal_action) == Config.RAW_LEGAL_ACTION_DIM:
                size = Config.LABEL_SIZE_LIST[-1]
                target_base = offsets[-1]
                mask = legal_action[target_base + button * size : target_base + (button + 1) * size]
            else:
                size = Config.LABEL_SIZE_LIST[head_idx]
                mask = legal_action[offsets[head_idx] : offsets[head_idx] + size]
            action[head_idx] = self._preferred_or_first_legal(
                action[head_idx],
                mask,
                prefer_nearest=head_idx in (1, 2, 3, 4),
            )
            if action[head_idx] is None:
                return None
        return [int(value) for value in action]

    def _preferred_or_first_legal(self, preferred_idx, mask, prefer_nearest=False):
        if hasattr(mask, "tolist"):
            mask = mask.tolist()
        legal_indices = [idx for idx, value in enumerate(mask) if int(value or 0) == 1]
        if not legal_indices:
            return None
        if 0 <= int(preferred_idx) < len(mask) and int(mask[int(preferred_idx)] or 0) == 1:
            return int(preferred_idx)
        if prefer_nearest:
            preferred_idx = int(preferred_idx)
            return min(legal_indices, key=lambda idx: (abs(idx - preferred_idx), idx))
        return int(legal_indices[0])

    def _maybe_force_home(self, observation, action):
        # Force walking back to base when low HP and safe, then return to the
        # first-tower area after healing. The task has no usable recall button.
        self.recall_override_active = False
        if self.cleanse_override_active:
            return action

        frame_state = observation.get("frame_state", {}) or {}
        frame_no = frame_state.get("frame_no", frame_state.get("frameNo", 0)) or 0
        current_camp = self._resolve_current_camp(observation, frame_state)
        if current_camp not in (1, 2):
            self._clear_force_home_phase(clear_camp=True)
            return action
        if self.force_home_path_camp not in (None, current_camp):
            self._reset_force_home_path()
        self.force_home_path_camp = current_camp
        self.force_home_camp = current_camp

        main_hero, enemy_hero, main_tower = self._find_my_hero_and_tower(frame_state)
        if main_hero is None:
            self._clear_force_home_phase(clear_camp=True)
            return action

        if self._hero_money_total(main_hero) >= GameConfig.FORCE_HOME_DISABLE_MONEY_TOTAL:
            self._clear_force_home_phase(clear_camp=True)
            return action

        self._update_own_cake_state(frame_state, frame_no, main_hero)
        self._update_force_home_recover_state(main_hero, frame_no)

        hp = self._unit_hp(main_hero)
        hp_rate = self._unit_hp_rate(main_hero)
        if hp <= 0:
            self._clear_force_home_phase(clear_camp=True)
            return action

        dead_cnt = int(self._get_any(main_hero, ["dead_cnt", "deadCnt"], 0) or 0)
        if dead_cnt > self.prev_dead_cnt:
            self.prev_dead_cnt = dead_cnt
            self._clear_force_home_phase(clear_camp=True)
            return action
        self.prev_dead_cnt = dead_cnt

        enemy_dead_for_gate = self._enemy_dead_for_force_home(frame_state, enemy_hero)

        if self.force_home_phase is None:
            self._record_force_home_path(main_hero, frame_no)

        if self.force_home_phase == "retreat_spring" and hp_rate >= GameConfig.FORCE_HOME_HP_RECOVERED:
            self.force_home_phase = "return"
            self.force_home_path_index = None
            self._reset_force_home_progress()
        if self.force_home_phase == "return" and hp_rate <= GameConfig.FORCE_HOME_HP_TRIGGER_PRE_CANNON:
            self.force_home_phase = "retreat_spring"
            self.force_home_path_index = None
            self._reset_force_home_progress()

        if self.force_home_phase == "retreat_spring":
            if (
                not self.force_home_post_kill_active
                and self._own_perspective_lane(main_hero) > GameConfig.FORCE_HOME_DEEP_LANE
            ):
                if (
                    self.own_cake_exists
                    or self._is_recover_skill_available(main_hero)
                ):
                    self._clear_force_home_phase(clear_camp=True)
                    return action
            target = self._force_home_retreat_target(main_hero)
            if target is None:
                self._clear_force_home_phase(clear_camp=True)
                return action
            return self._force_walk_to(
                observation,
                main_hero,
                target,
                action,
                phase="retreat",
            )

        if self.force_home_phase == "return":
            target = self._force_home_return_target(main_hero)
            if target is None:
                self._clear_force_home_phase(clear_camp=True)
                return action
            if self._own_perspective_lane(main_hero) >= GameConfig.FORCE_HOME_RETURN_EXIT_LANE:
                self._clear_force_home_phase(clear_camp=True)
                return action
            return self._force_walk_to(
                observation,
                main_hero,
                target,
                action,
                phase="return",
            )

        if not self._should_start_force_home(
            observation,
            frame_state,
            frame_no,
            hp_rate,
            main_hero,
            enemy_hero,
            main_tower,
            enemy_dead_for_gate,
        ):
            return action

        if not self._has_force_home_path():
            return action

        self.force_home_trigger_count += 1
        self.force_home_phase = "retreat_spring"
        self.force_home_post_kill_active = self._is_post_kill_lane_cleared(frame_no, frame_state)
        current_idx = self._nearest_force_home_path_index(self._project_own_perspective(main_hero))
        self.force_home_path_index = max(0, current_idx - 1) if current_idx is not None else 0
        target = self._force_home_retreat_target(main_hero)
        if target is None:
            self._clear_force_home_phase(clear_camp=True)
            return action
        return self._force_walk_to(
            observation,
            main_hero,
            target,
            action,
            phase="retreat",
            count_start=True,
        )

    def _should_start_force_home(
        self,
        observation,
        frame_state,
        frame_no,
        hp_rate,
        main_hero,
        enemy_hero,
        main_tower,
        enemy_dead_for_gate,
    ):
        if self._hero_money_total(main_hero) >= GameConfig.FORCE_HOME_DISABLE_MONEY_TOTAL:
            return False
        post_kill_lane_cleared = self._is_post_kill_lane_cleared(frame_no, frame_state)
        if main_tower is None:
            return False
        if not self._tower_hp_above(main_tower, GameConfig.FORCE_HOME_TOWER_HP_MIN):
            return False
        if self._enemy_minions_under_own_tower(frame_state, main_tower) > GameConfig.FORCE_HOME_TOWER_AREA_ENEMY_MINIONS_MAX:
            return False
        if post_kill_lane_cleared:
            return hp_rate < GameConfig.FORCE_HOME_POST_KILL_HP_TRIGGER
        if self.own_cake_exists:
            return False
        if self._is_recover_skill_available(main_hero):
            return False
        if self._in_force_home_recover_cooldown(frame_no, hp_rate):
            return False
        if not self._enemy_invisible_or_far(main_hero, enemy_hero):
            return False
        if enemy_dead_for_gate and self._enemy_minion_in_lane_range(
            frame_state,
            GameConfig.FORCE_HOME_ENEMY_DEAD_LANE_LO,
            GameConfig.FORCE_HOME_ENEMY_DEAD_LANE_HI,
        ):
            return False
        if hp_rate >= GameConfig.FORCE_HOME_HP_TRIGGER_PRE_CANNON:
            return False
        return True

    def _hero_money_total(self, hero):
        return float(self._get_any(hero or {}, ["moneyCnt", "money_cnt", "money"], 0) or 0)

    def _in_post_kill_force_home_window(self, frame_no):
        try:
            frame_no = int(frame_no or 0)
        except (TypeError, ValueError):
            frame_no = 0
        last_kill_frame = int(getattr(self, "last_enemy_hero_kill_frame", -10000) or -10000)
        return 0 <= frame_no - last_kill_frame <= int(GameConfig.FORCE_HOME_POST_KILL_WINDOW_FRAMES)

    def _is_post_kill_lane_cleared(self, frame_no, frame_state):
        return self._in_post_kill_force_home_window(frame_no) and not self._enemy_minion_in_lane_range(
            frame_state,
            GameConfig.FORCE_HOME_ENEMY_DEAD_LANE_LO,
            GameConfig.FORCE_HOME_ENEMY_DEAD_LANE_HI,
        )

    def _tower_hp_above(self, tower, threshold):
        return self._unit_hp_rate(tower) > float(threshold)

    def _enemy_dead_for_force_home(self, frame_state, enemy_hero):
        frame_no = int(self._get_any(frame_state or {}, ["frame_no", "frameNo"], 0) or 0)
        if self._frame_action_has_enemy_hero_death(frame_state, enemy_hero):
            self.last_enemy_hero_kill_frame = frame_no
            return True
        if enemy_hero is None:
            return True
        dead_cnt = int(self._get_any(enemy_hero, ["dead_cnt", "deadCnt"], 0) or 0)
        dead_cnt_increased = dead_cnt > int(self.prev_enemy_dead_cnt or 0)
        self.prev_enemy_dead_cnt = dead_cnt
        revive_time = int(self._get_any(enemy_hero, ["revive_time", "reviveTime"], 0) or 0)
        enemy_dead = dead_cnt_increased or revive_time > 0 or self._unit_hp(enemy_hero) <= 0
        if enemy_dead:
            self.last_enemy_hero_kill_frame = frame_no
        return enemy_dead

    def _frame_action_has_enemy_hero_death(self, frame_state, enemy_hero):
        frame_action = self._get_any(frame_state or {}, ["frame_action", "frameAction"], {}) or {}
        dead_actions = self._get_any(frame_action, ["dead_action", "deadAction"], []) or []
        if not isinstance(dead_actions, list):
            return False
        enemy_runtime = self._actor_runtime(enemy_hero)
        main_camp = self._camp_key(getattr(self, "force_home_camp", None) or self.hero_camp)
        for dead_action in dead_actions:
            if not isinstance(dead_action, dict):
                continue
            death = self._get_any(dead_action, ["death", "dead"], {}) or {}
            death_runtime = self._actor_runtime(death)
            if enemy_runtime is not None and death_runtime == enemy_runtime:
                return True
            death_camp = self._camp_key(self._get_any(death, ["camp"], None))
            death_config_id = int(self._get_any(death, ["config_id", "configId"], 0) or 0)
            if main_camp in (1, 2) and death_camp in (1, 2) and death_camp != main_camp:
                if death_config_id in GameConfig.HERO_IDS:
                    return True
        return False

    def _enemy_minions_under_own_tower(self, frame_state, main_tower):
        tower_pos = self._own_raw_location(main_tower)
        if tower_pos is None:
            return 0
        tower_range = self._tower_attack_range(main_tower)
        count = 0
        for npc in frame_state.get("npc_states", []) or []:
            if not self._is_enemy_minion(npc):
                continue
            npc_pos = self._own_raw_location(npc)
            if npc_pos is None:
                continue
            if self._dist(tower_pos, npc_pos) <= tower_range:
                count += 1
        return count

    def _enemy_minion_in_lane_range(self, frame_state, lane_lo, lane_hi):
        lane_lo = float(lane_lo)
        lane_hi = float(lane_hi)
        for npc in frame_state.get("npc_states", []) or []:
            if not self._is_enemy_minion(npc):
                continue
            pos = self._project_own_perspective(npc)
            if pos is None:
                continue
            if lane_lo <= float(pos[0]) <= lane_hi:
                return True
        return False

    def _is_enemy_minion(self, npc):
        if npc is None or self._unit_hp(npc) <= 0:
            return False
        sub_type = self._actor_sub_type(npc)
        if sub_type not in SOLDIER_SUB_TYPES:
            return False
        main_camp = self._camp_key(getattr(self, "force_home_camp", None) or self.hero_camp)
        npc_camp = self._camp_key(npc.get("camp"))
        return main_camp in (1, 2) and npc_camp in (1, 2) and npc_camp != main_camp

    def _actor_sub_type(self, actor):
        actor_state = self._get_any(actor or {}, ["actor_state", "actorState"], {}) or {}
        sub_type = self._get_any(actor_state, ["sub_type", "subType"], None)
        if sub_type is not None:
            return sub_type
        return self._get_any(actor or {}, ["sub_type", "subType"], None)

    def _tower_attack_range(self, tower):
        return float(
            self._get_any(tower or {}, ["attack_range", "attackRange"], Args.TOWER_ATTACK_RANGE_FALLBACK)
            or Args.TOWER_ATTACK_RANGE_FALLBACK
        )

    def _is_action_button_legal(self, observation, button):
        legal_action = observation.get("legal_action", [])
        if legal_action is None:
            return False
        if hasattr(legal_action, "tolist"):
            legal_action = legal_action.tolist()
        return 0 <= int(button) < len(legal_action) and int(legal_action[int(button)] or 0) == 1

    def _in_force_home_recover_cooldown(self, frame_no, hp_rate):
        if hp_rate < GameConfig.FORCE_HOME_HP_TRIGGER_PRE_CANNON:
            return False
        last_recover_frame = max(int(self.last_recover_success_frame), int(self.last_cake_eaten_frame))
        return int(frame_no or 0) - last_recover_frame < int(GameConfig.FORCE_HOME_RECOVER_COOLDOWN_FRAMES)

    def _clear_force_home_phase(self, clear_camp=False):
        self.force_home_phase = None
        self.force_home_post_kill_active = False
        self.force_home_path_index = None
        self._reset_force_home_progress()
        if clear_camp:
            self.force_home_camp = None

    def _reset_force_home_path(self):
        self.force_home_path_camp = None
        self.force_home_path_points = []
        self.force_home_path_ready = False
        self.force_home_path_index = None
        self._reset_force_home_progress()

    def _reset_force_home_progress(self):
        self.force_home_progress_phase = None
        self.force_home_progress_index = None
        self.force_home_progress_best_dist = None
        self.force_home_stuck_frames = 0

    def _record_force_home_path(self, main_hero, frame_no):
        if self.force_home_phase is not None or self.force_home_path_ready:
            return
        try:
            frame_no = int(frame_no or 0)
        except (TypeError, ValueError):
            frame_no = 0
        if frame_no > GameConfig.FORCE_HOME_PATH_RECORD_FRAMES:
            self.force_home_path_ready = self._is_force_home_path_valid()
            return

        pos = self._project_own_perspective(main_hero)
        if pos is None:
            return
        points = list(self.force_home_path_points)
        crossed_return_lane = bool(points) and points[-1][0] < GameConfig.FORCE_HOME_RETURN_EXIT_LANE <= pos[0]
        turn_point = self._is_force_home_turn_point(points, pos)
        if (
            not points
            or crossed_return_lane
            or turn_point
            or self._dist(pos, points[-1]) >= GameConfig.FORCE_HOME_PATH_MIN_DISTANCE
        ):
            points.append((float(pos[0]), float(pos[1])))
            self.force_home_path_points = self._compress_force_home_path(points)
        if pos[0] >= GameConfig.FORCE_HOME_RETURN_EXIT_LANE:
            self.force_home_path_ready = self._is_force_home_path_valid()

    def _compress_force_home_path(self, points):
        max_points = int(GameConfig.FORCE_HOME_PATH_MAX_POINTS)
        if len(points) <= max_points:
            return points
        if max_points <= 2:
            return [points[0], points[-1]]
        keep = {0, len(points) - 1}
        ranked = sorted(
            range(1, len(points) - 1),
            key=lambda idx: self._force_home_path_point_importance(points, idx),
            reverse=True,
        )
        keep.update(ranked[: max_points - len(keep)])
        return [points[idx] for idx in sorted(keep)]

    def _force_home_path_point_importance(self, points, idx):
        prev_point = points[idx - 1]
        point = points[idx]
        next_point = points[idx + 1]
        prev_dist = self._dist(prev_point, point)
        next_dist = self._dist(point, next_point)
        turn_score = 0.0
        denom = prev_dist * next_dist
        if denom > 1e-6:
            v1 = (point[0] - prev_point[0], point[1] - prev_point[1])
            v2 = (next_point[0] - point[0], next_point[1] - point[1])
            cos_value = (v1[0] * v2[0] + v1[1] * v2[1]) / denom
            turn_score = max(0.0, 1.0 - cos_value) * 5000.0
        lane_score = 0.0
        exit_lane = GameConfig.FORCE_HOME_RETURN_EXIT_LANE
        if prev_point[0] < exit_lane <= point[0] or abs(point[0] - exit_lane) <= GameConfig.FORCE_HOME_PATH_MIN_DISTANCE:
            lane_score = 100000.0
        return lane_score + turn_score + min(prev_dist, next_dist)

    def _is_force_home_turn_point(self, points, pos):
        if len(points) < 2:
            return False
        prev_point = points[-2]
        point = points[-1]
        prev_dist = self._dist(prev_point, point)
        next_dist = self._dist(point, pos)
        if (
            prev_dist < GameConfig.FORCE_HOME_PATH_TURN_MIN_DISTANCE
            or next_dist < GameConfig.FORCE_HOME_PATH_TURN_MIN_DISTANCE
        ):
            return False
        v1 = (point[0] - prev_point[0], point[1] - prev_point[1])
        v2 = (pos[0] - point[0], pos[1] - point[1])
        cos_value = (v1[0] * v2[0] + v1[1] * v2[1]) / max(prev_dist * next_dist, 1e-6)
        return cos_value <= GameConfig.FORCE_HOME_PATH_TURN_COS

    def _is_force_home_path_valid(self):
        points = self.force_home_path_points
        if len(points) < int(GameConfig.FORCE_HOME_PATH_MIN_POINTS):
            return False
        first_lane = float(points[0][0])
        last_lane = float(points[-1][0])
        if first_lane > -30000.0:
            return False
        if last_lane < GameConfig.FORCE_HOME_PATH_VALID_EXIT_LANE:
            return False
        return last_lane - first_lane >= 12000.0

    def _has_force_home_path(self):
        if not self.force_home_path_ready:
            self.force_home_path_ready = self._is_force_home_path_valid()
        return bool(self.force_home_path_ready)

    def _nearest_force_home_path_index(self, pos):
        if pos is None or not self.force_home_path_points:
            return None
        return min(range(len(self.force_home_path_points)), key=lambda idx: self._dist(pos, self.force_home_path_points[idx]))

    def _force_home_retreat_target(self, main_hero):
        if not self._has_force_home_path():
            return None
        pos = self._project_own_perspective(main_hero)
        if self.force_home_path_index is None:
            current_idx = self._nearest_force_home_path_index(pos)
            self.force_home_path_index = max(0, current_idx - 1) if current_idx is not None else 0
        self._advance_force_home_path_index(pos, -1)
        return self._force_home_path_target(self.force_home_path_index)

    def _force_home_path_target(self, index):
        if not self.force_home_path_points:
            return None
        index = max(0, min(int(index), len(self.force_home_path_points) - 1))
        return self._unproject_own_perspective(self.force_home_path_points[index])

    def _force_walk_to(self, observation, main_hero, target, fallback_action, phase=None, count_start=False):
        move_action = self._move_action_towards(main_hero, target)
        if move_action is None:
            return fallback_action
        legal_move = self._legalized_rule_action(observation, move_action, active_heads=(1, 2))
        if legal_move is None:
            return fallback_action
        self.rule_override_active = True
        self.rule_override_count += 1
        self.force_home_override_count += 1
        if count_start:
            self.force_home_start_count += 1
        if phase == "retreat":
            self.force_home_retreat_count += 1
        elif phase == "return":
            self.force_home_return_count += 1
        return legal_move

    def _move_action_towards(self, main_hero, target):
        loc = self._own_raw_location(main_hero)
        target_loc = self._own_raw_location(target)
        if loc is None or target_loc is None:
            return None
        dx = target_loc[0] - loc[0]
        dz = target_loc[1] - loc[1]
        return [
            2,
            self._direction_bucket(dx),
            self._direction_bucket(dz),
            8,
            8,
            0,
        ]

    def _direction_bucket(self, delta):
        deadzone = float(GameConfig.FORCE_HOME_DIRECTION_DEADZONE)
        if delta > deadzone:
            return 15
        if delta < -deadzone:
            return 0
        return 8

    def _unit_hp(self, unit):
        return float(self._get_any(unit or {}, ["hp", "HP"], 0) or 0)

    def _unit_hp_rate(self, unit):
        hp = self._unit_hp(unit)
        max_hp = float(self._get_any(unit or {}, ["max_hp", "maxHp", "maxHP"], 1) or 1)
        return hp / max(max_hp, 1.0)

    def _own_perspective_lane(self, unit):
        pos = self._project_own_perspective(unit)
        if pos is None:
            return float("-inf")
        return float(pos[0])

    def _enemy_invisible_or_far(self, main_hero, enemy_hero):
        if enemy_hero is None:
            return True
        if not self._visible_to_own_camp(enemy_hero):
            return True
        main_loc = self._project_own_perspective(main_hero)
        enemy_loc = self._project_own_perspective(enemy_hero)
        if main_loc is None or enemy_loc is None:
            return True
        dx = main_loc[0] - enemy_loc[0]
        dz = main_loc[1] - enemy_loc[1]
        return (dx * dx + dz * dz) ** 0.5 > GameConfig.FORCE_HOME_ENEMY_SAFE_RANGE

    def _visible_to_own_camp(self, unit):
        visible = self._get_any(unit or {}, ["camp_visible", "campVisible"], None)
        if visible is None:
            return True
        camp_key = self._camp_key(getattr(self, "force_home_camp", None) or self.hero_camp)
        idx = 0 if camp_key == 1 else 1 if camp_key == 2 else None
        if idx is None or idx >= len(visible):
            return True
        return bool(visible[idx])

    def _resolve_current_camp(self, observation, frame_state):
        obs_camp = self._camp_key(observation.get("player_camp", observation.get("camp", None)))
        hero_camp = self._camp_key(self._observed_player_hero_camp(observation, frame_state))
        if obs_camp in (1, 2) and hero_camp in (1, 2) and obs_camp != hero_camp:
            return None
        if obs_camp in (1, 2):
            self.hero_camp = obs_camp
            return obs_camp
        if hero_camp in (1, 2):
            self.hero_camp = hero_camp
            return hero_camp
        fallback = self._camp_key(self.hero_camp)
        return fallback if fallback in (1, 2) else None

    def _observed_player_hero_camp(self, observation, frame_state):
        player_id = observation.get("player_id", getattr(self, "player_id", None))
        if player_id is not None:
            for hero in frame_state.get("hero_states", []) or []:
                if self._actor_runtime(hero) == player_id:
                    return hero.get("camp")
        return None

    def _force_home_return_target(self, main_hero=None):
        if self._has_force_home_path() and main_hero is not None:
            pos = self._project_own_perspective(main_hero)
            if self.force_home_path_index is None:
                current_idx = self._nearest_force_home_path_index(pos)
                self.force_home_path_index = current_idx if current_idx is not None else 0
            self._advance_force_home_path_index(pos, 1)
            return self._force_home_path_target(self.force_home_path_index)
        return None

    def _advance_force_home_path_index(self, pos, direction):
        if pos is None or not self.force_home_path_points or self.force_home_path_index is None:
            return
        direction = -1 if direction < 0 else 1
        last_idx = len(self.force_home_path_points) - 1
        radius = GameConfig.FORCE_HOME_PATH_WAYPOINT_RADIUS
        while 0 <= self.force_home_path_index + direction <= last_idx:
            target = self.force_home_path_points[self.force_home_path_index]
            if self._dist(pos, target) > radius:
                break
            self.force_home_path_index += direction
            self._reset_force_home_progress()
        self._skip_stuck_force_home_waypoint(pos, direction)

    def _skip_stuck_force_home_waypoint(self, pos, direction):
        if pos is None or not self.force_home_path_points or self.force_home_path_index is None:
            return
        next_idx = self.force_home_path_index + direction
        if next_idx < 0 or next_idx >= len(self.force_home_path_points):
            return
        dist = self._dist(pos, self.force_home_path_points[self.force_home_path_index])
        phase = self.force_home_phase
        if phase != self.force_home_progress_phase or self.force_home_path_index != self.force_home_progress_index:
            self.force_home_progress_phase = phase
            self.force_home_progress_index = self.force_home_path_index
            self.force_home_progress_best_dist = dist
            self.force_home_stuck_frames = 0
            return
        best_dist = self.force_home_progress_best_dist
        if best_dist is None or dist <= best_dist - GameConfig.FORCE_HOME_STUCK_MIN_PROGRESS:
            self.force_home_progress_best_dist = dist
            self.force_home_stuck_frames = 0
            return
        self.force_home_stuck_frames += 1
        if self.force_home_stuck_frames >= int(GameConfig.FORCE_HOME_STUCK_FRAMES):
            self.force_home_path_index = next_idx
            self._reset_force_home_progress()

    def _own_raw_location(self, obj_or_loc):
        if obj_or_loc is None:
            return None
        if isinstance(obj_or_loc, dict):
            loc = obj_or_loc
            if "x" not in loc:
                loc = (obj_or_loc.get("collider", {}) or {}).get("location", None) or obj_or_loc.get("location", None)
            if not loc:
                return None
            x = loc.get("x", None)
            z = loc.get("z", None)
        else:
            if len(obj_or_loc) < 2:
                return None
            x, z = obj_or_loc[0], obj_or_loc[1]
        if x is None or z is None:
            return None
        x = float(x)
        z = float(z)
        if abs(x) > Args.RAW_COORD_ABS_LIMIT or abs(z) > Args.RAW_COORD_ABS_LIMIT:
            return None
        if self._camp_key(getattr(self, "force_home_camp", None) or self.hero_camp) == 2:
            x, z = -x, -z
        return x, z

    def _project_own_perspective(self, obj_or_loc):
        loc = obj_or_loc
        if obj_or_loc is not None and not isinstance(obj_or_loc, dict):
            return None
        if isinstance(obj_or_loc, dict) and "x" not in obj_or_loc:
            loc = (obj_or_loc.get("collider", {}) or {}).get("location", None) or obj_or_loc.get("location", None)
        if not loc:
            return None
        x = loc.get("x", None)
        z = loc.get("z", None)
        if x is None or z is None:
            return None
        x = float(x)
        z = float(z)
        if abs(x) > Args.RAW_COORD_ABS_LIMIT or abs(z) > Args.RAW_COORD_ABS_LIMIT:
            return None
        if self._camp_key(getattr(self, "force_home_camp", None) or self.hero_camp) == 2:
            x, z = -x, -z
        lane = (x + z) / Args.SQRT2
        width = (x - z) / Args.SQRT2
        return lane, width

    def _unproject_own_perspective(self, projected):
        lane, width = float(projected[0]), float(projected[1])
        x = (lane + width) / Args.SQRT2
        z = (lane - width) / Args.SQRT2
        if self._camp_key(getattr(self, "force_home_camp", None) or self.hero_camp) == 2:
            x, z = -x, -z
        return x, z

    def _camp_key(self, camp):
        if camp in (1, "1", "PLAYERCAMP_1", "blue_camp"):
            return 1
        if camp in (2, "2", "PLAYERCAMP_2", "red_camp"):
            return 2
        return camp

    def _find_my_hero_and_tower(self, frame_state):
        main_hero = None
        enemy_hero = None
        main_towers = []
        main_camp = self._camp_key(self.hero_camp)
        for hero in frame_state.get("hero_states", []) or []:
            if self._camp_key(hero.get("camp")) == main_camp:
                main_hero = hero
            else:
                enemy_hero = hero
        for npc in frame_state.get("npc_states", []) or []:
            if self._actor_sub_type(npc) not in TOWER_SUB_TYPES:
                continue
            if self._camp_key(npc.get("camp")) == main_camp:
                main_towers.append(npc)
        main_tower = self._select_nearest_projected_anchor(main_towers, self._tower_anchor(False))
        return main_hero, enemy_hero, main_tower

    def _select_nearest_projected_anchor(self, units, anchor):
        if not units:
            return None
        return sorted(units, key=lambda unit: self._dist(self._project_own_perspective(unit), anchor))[0]

    def _side_anchor(self, anchors_by_camp, default_anchor, is_enemy):
        side = "enemy" if is_enemy else "self"
        camp_key = self._camp_key(getattr(self, "force_home_camp", None) or self.hero_camp)
        anchors = anchors_by_camp.get(camp_key, {})
        return anchors.get(side, default_anchor)

    def _tower_anchor(self, is_enemy):
        default_anchor = Args.ENEMY_TOWER_ANCHOR if is_enemy else Args.SELF_TOWER_ANCHOR
        return self._side_anchor(Args.TOWER_ANCHORS_BY_CAMP, default_anchor, is_enemy)

    def _cake_anchor(self, is_enemy):
        default_anchor = Args.ENEMY_CAKE_ANCHOR if is_enemy else Args.SELF_CAKE_ANCHOR
        return self._side_anchor(Args.CAKE_ANCHORS_BY_CAMP, default_anchor, is_enemy)

    def _update_own_cake_state(self, frame_state, frame_no, main_hero=None):
        frame_no = int(frame_no or 0)
        cakes = frame_state.get("cakes", []) or []
        found_own = False
        own_cake_pos = None
        for cake in cakes:
            pos = self._project_own_perspective(cake)
            if pos is None:
                continue
            if self._dist(pos, self._cake_anchor(False)) <= self._dist(pos, self._cake_anchor(True)):
                found_own = True
                own_cake_pos = pos
                break
        if found_own:
            self.own_cake_seen_once = True
            self.last_own_cake_pos = own_cake_pos
            self.next_own_cake_frame = None
        if self.own_cake_exists and not found_own:
            main_pos = self._project_own_perspective(main_hero)
            cake_pos = self.last_own_cake_pos or self._cake_anchor(False)
            if main_pos is not None and self._dist(main_pos, cake_pos) <= GameConfig.CAKE_PICKUP_PROXIMITY:
                self.last_cake_eaten_frame = int(frame_no or 0)
            self.last_own_cake_disappear_frame = frame_no
            self.next_own_cake_frame = frame_no + 2250
            self.last_own_cake_pos = None
        self.own_cake_exists = found_own

    def _is_own_cake_unavailable_for_force_home(self, frame_state, frame_no, main_hero=None):
        self._update_own_cake_state(frame_state, frame_no, main_hero)
        return not self.own_cake_exists

    def _update_force_home_recover_state(self, main_hero, frame_no):
        if main_hero is None:
            self.force_home_pending_recover = None
            return

        frame_no = int(frame_no or 0)
        main_hp = self._unit_hp(main_hero)
        hp_rate = self._unit_hp_rate(main_hero)
        buff_ids = self._hero_buff_ids(main_hero)
        has_start_buff = GameConfig.RECOVER_START_BUFF_ID in buff_ids
        has_effect_buff = GameConfig.RECOVER_EFFECT_BUFF_ID in buff_ids

        if self.force_home_pending_recover is None and hp_rate < 0.5:
            if self._slot_succ_used(main_hero, 4) or has_start_buff:
                self.force_home_pending_recover = {
                    "attempt_frame": frame_no,
                    "main_hp_at_attempt": main_hp,
                    "saw_start_buff": bool(has_start_buff),
                    "saw_effect_buff": bool(has_effect_buff),
                }

        if self.force_home_pending_recover is None:
            return

        self.force_home_pending_recover["saw_start_buff"] = bool(
            self.force_home_pending_recover.get("saw_start_buff", False) or has_start_buff
        )
        self.force_home_pending_recover["saw_effect_buff"] = bool(
            self.force_home_pending_recover.get("saw_effect_buff", False) or has_effect_buff
        )
        elapsed = frame_no - int(self.force_home_pending_recover["attempt_frame"])
        if elapsed < GameConfig.RECOVER_CONFIRM_FRAMES:
            return

        hp_gain = main_hp - float(self.force_home_pending_recover["main_hp_at_attempt"])
        self.force_home_pending_recover = None
        if hp_gain >= GameConfig.RECOVER_HP_GAIN_THRESHOLD:
            self.last_recover_success_frame = frame_no

    def _is_recover_skill_available(self, main_hero):
        slots = self._slot_states(main_hero)
        for slot in slots:
            if self._slot_idx(self._get_any(slot, ["slot_type", "slotType"])) != 4:
                continue
            usable = bool(slot.get("usable", False))
            cooldown = float(slot.get("cooldown", 0) or 0)
            return usable and cooldown <= 0
        return False

    def _used_skill_by_config(self, hero, config_id):
        for slot in self._slot_states(hero):
            slot_config = self._get_any(slot, ["config_id", "configId"], 0)
            try:
                if int(slot_config or 0) != int(config_id):
                    continue
            except (TypeError, ValueError):
                continue
            try:
                return float(self._get_any(slot, ["succUsedInFrame", "succ_used_in_frame"], 0) or 0) > 0
            except (TypeError, ValueError):
                return False
        return False

    def _hero_buff_ids(self, hero):
        buff_state = self._get_any(hero or {}, ["buff_state", "buffState"], {}) or {}
        buff_skills = self._get_any(buff_state, ["buff_skills", "buffSkills"], []) or []
        ids = set()
        for buff in buff_skills:
            buff_id = self._get_any(buff, ["config_id", "configId", "id"], 0)
            try:
                buff_id = int(buff_id or 0)
            except (TypeError, ValueError):
                buff_id = 0
            if buff_id > 0:
                ids.add(buff_id)
        return ids

    def _slot_idx(self, slot_type):
        if isinstance(slot_type, int):
            return slot_type
        text = str(slot_type) if slot_type is not None else ""
        if text.startswith("SLOT_SKILL_"):
            try:
                return int(text.split("_")[-1])
            except (TypeError, ValueError):
                return -1
        try:
            return int(slot_type) if slot_type is not None else -1
        except (TypeError, ValueError):
            return -1

    @learn_wrapper
    def learn(self, list_sample_data):
        return self.algorithm.learn(list_sample_data)

    @save_model_wrapper
    def save_model(self, path=None, id="1"):
        model_file_path = self._model_file_path(path, id, create_dir=True)
        checkpoint = self._build_checkpoint_payload()
        torch.save(checkpoint, model_file_path)
        self._save_checkpoint_metadata(model_file_path, checkpoint["checkpoint_meta"])
        self._log_info(f"save model {model_file_path} successfully")

    @load_model_wrapper
    def load_model(self, path=None, id="1"):
        model_file_path = self._model_file_path(path, id)
        if not os.path.exists(model_file_path):
            if str(id) == "latest":
                self._log_warning(f"latest model {model_file_path} not found, keep current parameters")
                return
            raise FileNotFoundError(f"model file {model_file_path} not found")

        if self.cur_model_name == model_file_path:
            return
        else:
            state_dict = self._load_checkpoint_state_dict(model_file_path)
            self.model.load_state_dict(state_dict)
            self.cur_model_name = model_file_path
            self._log_info(f"load model {model_file_path} successfully")

    @load_opponent_agent_wrapper
    def load_opponent_agent(self, id="1"):
        # The official framework wrapper resolves and loads model-pool opponents.
        pass

    def _model_file_path(self, path, id, create_dir=False):
        if path is None:
            path = os.path.join(os.getcwd(), "agent_diy", "ckpt")
        if create_dir:
            os.makedirs(path, exist_ok=True)
        return os.path.join(path, f"model.ckpt-{str(id)}.pkl")

    def _build_checkpoint_payload(self):
        state_dict = self.model.state_dict()
        meta = {
            "checkpoint_signature": Config.checkpoint_signature(),
            "checkpoint_signature_hash": Config.checkpoint_signature_hash(),
            "model_state_shape_hash": self._state_dict_shape_hash(state_dict),
        }
        return {
            "state_dict": state_dict,
            "checkpoint_meta": meta,
        }

    def _load_checkpoint_state_dict(self, model_file_path):
        checkpoint = self._torch_load(model_file_path)
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint and "checkpoint_meta" in checkpoint:
            self._validate_checkpoint_meta(checkpoint["checkpoint_meta"], model_file_path)
            return checkpoint["state_dict"]

        meta = self._load_checkpoint_metadata(model_file_path)
        if meta is not None:
            self._validate_checkpoint_meta(meta, model_file_path)
        else:
            self._log_warning(
                f"checkpoint {model_file_path} has no protocol metadata; treating it as a legacy state_dict"
            )
        return checkpoint

    def _save_checkpoint_metadata(self, model_file_path, meta):
        meta_file_path = self._checkpoint_metadata_path(model_file_path)
        with open(meta_file_path, "w", encoding="utf-8") as file:
            json.dump(meta, file, sort_keys=True, indent=2)

    def _load_checkpoint_metadata(self, model_file_path):
        meta_file_path = self._checkpoint_metadata_path(model_file_path)
        if not os.path.exists(meta_file_path):
            return None
        with open(meta_file_path, "r", encoding="utf-8") as file:
            return json.load(file)

    def _checkpoint_metadata_path(self, model_file_path):
        return f"{model_file_path}.meta.json"

    def _validate_checkpoint_meta(self, meta, model_file_path):
        expected_signature_hash = Config.checkpoint_signature_hash()
        actual_signature_hash = meta.get("checkpoint_signature_hash")
        if actual_signature_hash != expected_signature_hash:
            raise ValueError(
                "checkpoint protocol mismatch for "
                f"{model_file_path}: {actual_signature_hash} != {expected_signature_hash}"
            )

        expected_state_hash = self._state_dict_shape_hash(self.model.state_dict())
        actual_state_hash = meta.get("model_state_shape_hash")
        if actual_state_hash and actual_state_hash != expected_state_hash:
            raise ValueError(
                "checkpoint model shape mismatch for "
                f"{model_file_path}: {actual_state_hash} != {expected_state_hash}"
            )

    def _state_dict_shape_hash(self, state_dict):
        shapes = {
            key: list(value.shape)
            for key, value in state_dict.items()
            if hasattr(value, "shape")
        }
        payload = json.dumps(shapes, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _torch_load(self, model_file_path):
        try:
            return torch.load(model_file_path, map_location=self.device, weights_only=True)
        except TypeError:
            return torch.load(model_file_path, map_location=self.device)

    def _log_info(self, message):
        if self.logger:
            self.logger.info(message)

    def _log_warning(self, message):
        if self.logger:
            self.logger.warning(message)

    def update_status(self, obs_data, act_data):
        self.obs_data = obs_data
        self.act_data = act_data
        self.lstm_cell = act_data.lstm_cell
        self.lstm_hidden = act_data.lstm_hidden

    def _sample_masked_action(self, logits, legal_action):
        prob_list = []
        d_prob_list = []
        action_list = []
        d_action_list = []
        label_split_size = [sum(self.label_size_list[: index + 1]) for index in range(len(self.label_size_list))]
        legal_action = np.asarray(legal_action, dtype=np.float32)
        legal_actions = np.split(legal_action, label_split_size[:-1])
        logits_split = np.split(logits, label_split_size[:-1])

        for index in range(0, len(self.label_size_list) - 1):
            probs = self._legal_soft_max(logits_split[index], legal_actions[index])
            prob_list += list(probs)
            d_prob_list += list(probs)
            action_list.append(self._legal_sample(probs, use_max=False))
            d_action_list.append(self._legal_sample(probs, use_max=True))

        index = len(self.label_size_list) - 1
        if legal_action.shape[0] == Config.RAW_LEGAL_ACTION_DIM:
            target_legal_action_o = np.reshape(
                legal_actions[index],
                [self.legal_action_size[0], self.legal_action_size[-1] // self.legal_action_size[0]],
            )
            one_hot_actions = np.eye(self.label_size_list[0])[action_list[0]].reshape([self.label_size_list[0], 1])
            target_legal_action = np.sum(target_legal_action_o * one_hot_actions, axis=0)
        else:
            target_legal_action_o = None
            target_legal_action = legal_actions[index]

        probs = self._legal_soft_max(logits_split[-1], target_legal_action)
        prob_list += list(probs)
        action_list.append(self._legal_sample(probs, use_max=False))

        if target_legal_action_o is not None:
            one_hot_actions = np.eye(self.label_size_list[0])[d_action_list[0]].reshape([self.label_size_list[0], 1])
            target_legal_action_d = np.sum(target_legal_action_o * one_hot_actions, axis=0)
        else:
            target_legal_action_d = target_legal_action
        probs = self._legal_soft_max(logits_split[-1], target_legal_action_d)
        d_prob_list += list(probs)
        d_action_list.append(self._legal_sample(probs, use_max=True))

        return [prob_list], [d_prob_list], action_list, d_action_list

    def _legal_soft_max(self, input_hidden, legal_action):
        legal_action = np.asarray(legal_action, dtype=np.float64).reshape(-1)
        legal_action = (legal_action > 0).astype(np.float64)
        if legal_action.sum() <= 0:
            legal_action = np.ones_like(legal_action, dtype=np.float64)

        logits = np.asarray(input_hidden, dtype=np.float64).reshape(-1)
        masked = np.where(legal_action > 0, logits, -1e20)
        masked = masked - np.max(masked)
        exp_logits = np.exp(np.clip(masked, -80.0, 80.0)) * legal_action
        probs = exp_logits / max(exp_logits.sum(), 1e-12)
        probs = np.where(legal_action > 0, probs, 0.0)
        return self._normalize_probs(probs, legal_action=legal_action)

    def _legal_sample(self, probs, legal_action=None, use_max=False):
        if legal_action is None:
            legal_action = np.asarray(probs, dtype=np.float64) > 0
        probs = self._normalize_probs(probs, legal_action=legal_action)
        if use_max:
            return int(np.argmax(probs))
        return int(np.random.choice(len(probs), p=probs))

    def _normalize_probs(self, probs, legal_action=None):
        probs = np.asarray(probs, dtype=np.float64).reshape(-1)
        if legal_action is not None:
            legal_action = np.asarray(legal_action, dtype=np.float64).reshape(-1)
            legal_mask = legal_action > 0
        else:
            legal_mask = np.ones_like(probs, dtype=bool)
        probs = np.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
        probs = np.clip(probs, 0.0, 1.0)
        probs = np.where(legal_mask, probs, 0.0)
        total = probs.sum(dtype=np.float64)
        if total <= 0.0:
            probs = legal_mask.astype(np.float64)
            probs = probs / probs.sum(dtype=np.float64)
        else:
            probs = probs / total

        legal_indices = np.flatnonzero(legal_mask)
        repair_index = int(legal_indices[-1])
        other_indices = np.arange(len(probs)) != repair_index
        other_sum = probs[other_indices].sum(dtype=np.float64)
        if other_sum >= 1.0:
            probs[other_indices] = probs[other_indices] / (other_sum + 1e-12)
            probs[other_indices] *= 1.0 - 1e-12
            other_sum = probs[other_indices].sum(dtype=np.float64)
        probs[repair_index] = max(0.0, 1.0 - other_sum)
        probs = np.where(legal_mask, probs, 0.0)
        return probs

    def _normalize_action(self, action):
        if action is None:
            return list(NONE_ACTION)
        if isinstance(action, np.ndarray):
            action = action.tolist()
        if isinstance(action, tuple):
            action = list(action)
        if isinstance(action, list) and len(action) == 1:
            first = action[0]
            if first is None:
                return list(NONE_ACTION)
            if isinstance(first, np.ndarray):
                first = first.tolist()
            if isinstance(first, tuple):
                first = list(first)
            if isinstance(first, list):
                action = first
        if not isinstance(action, list) or len(action) != len(Config.LABEL_SIZE_LIST):
            if self.logger:
                self.logger.warning(f"invalid action {action}, fallback to NONE_ACTION")
            return list(NONE_ACTION)
        return [int(x) for x in action]
