#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Training workflow for agent_diy.
"""

import os
import time

from agent_diy.conf.conf import Config, GameConfig
from agent_diy.feature.definition import (
    FrameCollector,
    NONE_ACTION,
    build_frame,
    lineup_iterator_roundrobin_camp_heroes,
    sample_process,
)
from agent_diy.workflow.env_conf_manager import EnvConfManager
from common_python.utils.workflow_disaster_recovery import handle_disaster_recovery
from tools.model_pool_utils import get_valid_model_pool


def _normalize_env_action(action):
    if action is None:
        return list(NONE_ACTION)
    if hasattr(action, "tolist"):
        action = action.tolist()
    if isinstance(action, tuple):
        action = list(action)
    if isinstance(action, list) and len(action) == 1:
        first = action[0]
        if first is None:
            return list(NONE_ACTION)
        if hasattr(first, "tolist"):
            first = first.tolist()
        if isinstance(first, tuple):
            first = list(first)
        if isinstance(first, list):
            action = first
    if not isinstance(action, list) or len(action) != 6:
        return list(NONE_ACTION)
    return [int(x) for x in action]


def workflow(envs, agents, logger=None, monitor=None, *args, **kwargs):
    do_learns = [True, True]
    last_save_model_time = time.time()

    env_conf_manager = EnvConfManager(
        config_path="agent_diy/conf/train_env_conf.toml",
        logger=logger,
    )
    lineup_iterator = lineup_iterator_roundrobin_camp_heroes(GameConfig.HERO_IDS)

    episode_runner = EpisodeRunner(
        env=envs[0],
        agents=agents,
        logger=logger,
        monitor=monitor,
        env_conf_manager=env_conf_manager,
        lineup_iterator=lineup_iterator,
    )

    while True:
        for g_data in episode_runner.run_episodes():
            for index, (do_learn, agent) in enumerate(zip(do_learns, agents)):
                if do_learn and len(g_data[index]) > 0:
                    agent.send_sample_data(g_data[index])
            g_data.clear()

            now = time.time()
            if now - last_save_model_time > GameConfig.MODEL_SAVE_INTERVAL:
                agents[0].save_model()
                last_save_model_time = now


class EpisodeRunner:
    def __init__(self, env, agents, logger, monitor, env_conf_manager, lineup_iterator):
        self.env = env
        self.agents = agents
        self.logger = logger
        self.monitor = monitor
        self.env_conf_manager = env_conf_manager
        self.lineup_iterator = lineup_iterator
        self.agent_num = len(agents)
        self.episode_cnt = 0
        self.last_report_monitor_time = 0
        self.selected_summoner_skills = [None for _ in range(self.agent_num)]
        self.no_op_streaks = [0 for _ in range(self.agent_num)]

    def _call_init_config(self, usr_conf, is_eval=False):
        blue_hero_ids, red_hero_ids = EnvConfManager.extract_hero_ids_from_usr_conf(usr_conf)
        camp_keys = ["blue_camp", "red_camp"]
        self.selected_summoner_skills = [None for _ in range(self.agent_num)]

        for agent_idx, agent in enumerate(self.agents):
            if agent_idx == 0:
                my_hero_ids = blue_hero_ids
                opponent_hero_ids = red_hero_ids
                camp_key = camp_keys[0]
            else:
                my_hero_ids = red_hero_ids
                opponent_hero_ids = blue_hero_ids
                camp_key = camp_keys[1]

            config_data = {
                "my_camp": camp_key,
                "my_heroes": my_hero_ids,
                "opponent_heroes": opponent_hero_ids,
                "is_eval": is_eval,
            }
            select_skills = agent.init_config(config_data)
            EnvConfManager.inject_select_skills(usr_conf, camp_key, select_skills)
            self.selected_summoner_skills[agent_idx] = self._first_selected_skill(select_skills, my_hero_ids)

    def run_episodes(self):
        while True:
            lineup = next(self.lineup_iterator)
            if sorted(lineup) and any(hero_id not in GameConfig.HERO_IDS for hero_id in lineup):
                raise ValueError(f"unsupported lineup {lineup}; expected heroes {GameConfig.HERO_IDS}")
            usr_conf, is_eval, monitor_side = self.env_conf_manager.update_config(lineup)
            self._call_init_config(usr_conf, is_eval=is_eval)

            env_obs = self.env.reset(usr_conf=usr_conf)
            if handle_disaster_recovery(env_obs, self.logger):
                break

            observation = env_obs["observation"]
            self.reset_agents(observation)
            frame_collector = FrameCollector(self.agent_num)

            self.episode_cnt += 1
            reward_sum_list = [0] * self.agent_num
            is_train_test = os.environ.get("is_train_test", "False").lower() == "true"

            reward_item_sum_list = [
                {reward_name: 0.0 for reward_name in GameConfig.REWARD_WEIGHT_DICT}
                for _ in range(self.agent_num)
            ]
            reward_debug_sum_list = [
                {debug_name: 0.0 for debug_name in GameConfig.REWARD_DEBUG_KEY_LIST}
                for _ in range(self.agent_num)
            ]
            action_debug_sum_list = [
                {debug_name: 0.0 for debug_name in GameConfig.ACTION_DEBUG_KEY_LIST}
                for _ in range(self.agent_num)
            ]
            self.no_op_streaks = [0 for _ in range(self.agent_num)]
            for i, (do_sample, agent) in enumerate(zip(self.do_samples, self.agents)):
                if do_sample:
                    reward = agent.reward_manager.result(observation[str(i)]["frame_state"])
                    observation[str(i)]["reward"] = reward
                    reward_sum_list[i] += reward["reward_sum"]
                    self._accumulate_reward_items(reward_item_sum_list[i], reward)
                    self._accumulate_reward_debug_items(reward_debug_sum_list[i], reward)
                    self._accumulate_skill_usage_items(action_debug_sum_list[i], observation[str(i)]["frame_state"], agent)

            while True:
                actions = [list(NONE_ACTION) for _ in range(self.agent_num)]

                for index, (do_predict, do_sample, agent) in enumerate(
                    zip(self.do_predicts, self.do_samples, self.agents)
                ):
                    if do_predict:
                        assist_before = float(getattr(agent, "luban_skill1_aim_assist_count", 0))
                        raw_action = agent.predict(observation[str(index)]) if not is_eval else agent.exploit(observation[str(index)])
                        assist_after = float(getattr(agent, "luban_skill1_aim_assist_count", 0))
                        if do_sample:
                            assist_delta = max(0.0, assist_after - assist_before)
                            action_debug_sum_list[index]["luban_skill1_aim_assist_count"] += assist_delta
                        actions[index] = _normalize_env_action(raw_action)
                        if do_sample and getattr(agent, "reward_manager", None) is not None:
                            agent.reward_manager.record_action_context(observation[str(index)]["frame_state"], actions[index])
                        self._accumulate_action_debug_items(action_debug_sum_list[index], actions[index])

                        if not is_eval and do_sample:
                            frame = build_frame(agent, observation[str(index)])
                            frame_collector.save_frame(frame, agent_id=index)

                actions = [_normalize_env_action(action) for action in actions]
                env_reward, env_obs = self.env.step(actions)
                if handle_disaster_recovery(env_obs, self.logger):
                    break

                frame_no = env_obs["frame_no"]
                observation = env_obs["observation"]
                terminated = env_obs["terminated"]
                truncated = env_obs["truncated"]

                for i, (do_sample, agent) in enumerate(zip(self.do_samples, self.agents)):
                    if do_sample:
                        reward = agent.reward_manager.result(observation[str(i)]["frame_state"])
                        self._apply_no_op_streak_reward(reward, actions[i], i)
                        observation[str(i)]["reward"] = reward
                        reward_sum_list[i] += reward["reward_sum"]
                        self._accumulate_reward_items(reward_item_sum_list[i], reward)
                        self._accumulate_reward_debug_items(reward_debug_sum_list[i], reward)
                        self._accumulate_skill_usage_items(action_debug_sum_list[i], observation[str(i)]["frame_state"], agent)

                is_gameover = terminated or truncated or (is_train_test and frame_no >= 1000)
                if is_gameover:
                    for i, (do_sample, agent) in enumerate(zip(self.do_samples, self.agents)):
                        if not is_eval and do_sample:
                            terminal_value = self._apply_terminal_reward(
                                observation[str(i)],
                                reward_item_sum_list[i],
                                env_obs=env_obs,
                                terminated=terminated,
                                truncated=truncated,
                            )
                            reward_sum_list[i] += terminal_value
                            frame_collector.save_last_frame(
                                agent_id=i,
                                reward=observation[str(i)]["reward"]["reward_sum"],
                            )
                    self.logger.info(
                        f"episode_{self.episode_cnt} terminated in fno_{frame_no}, truncated:{truncated}, "
                        f"eval:{is_eval}, reward_sum:{reward_sum_list[monitor_side]}"
                    )

                    now = time.time()
                    if now - self.last_report_monitor_time >= 60:
                        monitor_data = {"episode_cnt": self.episode_cnt}
                        if self.monitor:
                            if is_eval:
                                monitor_data["reward"] = round(reward_sum_list[monitor_side], 2)
                            for reward_name, reward_value in reward_item_sum_list[monitor_side].items():
                                monitor_data[f"reward_{reward_name}"] = round(reward_value, 4)
                            for debug_name, debug_value in reward_debug_sum_list[monitor_side].items():
                                monitor_data[debug_name] = round(debug_value, 4)
                            for debug_name, debug_value in action_debug_sum_list[monitor_side].items():
                                monitor_data[debug_name] = round(debug_value, 4)
                            selected_skill = self.selected_summoner_skills[monitor_side]
                            for skill_id in GameConfig.SUMMONER_SKILL_MONITOR_IDS:
                                monitor_data[f"selected_summoner_{skill_id}"] = 1.0 if selected_skill == skill_id else 0.0
                            monitor_data["rule_override_count"] = float(
                                getattr(self.agents[monitor_side], "rule_override_count", 0)
                            )
                            monitor_data["force_home_trigger_count"] = float(
                                getattr(self.agents[monitor_side], "force_home_trigger_count", 0)
                            )
                            monitor_data["force_home_override_count"] = float(
                                getattr(self.agents[monitor_side], "force_home_override_count", 0)
                            )
                            monitor_data["force_home_start_count"] = float(
                                getattr(self.agents[monitor_side], "force_home_start_count", 0)
                            )
                            monitor_data["force_home_retreat_count"] = float(
                                getattr(self.agents[monitor_side], "force_home_retreat_count", 0)
                            )
                            monitor_data["force_home_return_count"] = float(
                                getattr(self.agents[monitor_side], "force_home_return_count", 0)
                            )
                            monitor_data["opening_unstuck_count"] = float(
                                getattr(self.agents[monitor_side], "opening_unstuck_count", 0)
                            )
                            monitor_data["cleanse_override_count"] = float(
                                getattr(self.agents[monitor_side], "cleanse_override_count", 0)
                            )
                            skill2_total = float(getattr(self.agents[monitor_side], "skill2_total_cast_count", 0))
                            cleanse_override = float(getattr(self.agents[monitor_side], "cleanse_override_count", 0))
                            monitor_data["skill2_blocked_count"] = float(
                                getattr(self.agents[monitor_side], "skill2_blocked_count", 0)
                            )
                            monitor_data["skill2_total_cast_count"] = skill2_total
                            monitor_data["skill2_cast_outside_window_count"] = float(
                                getattr(self.agents[monitor_side], "skill2_cast_outside_window_count", 0)
                            )
                            monitor_data["skill2_cleanse_rate"] = cleanse_override / max(1.0, skill2_total)
                            monitor_data["luban_skill1_aim_assist_count"] = float(
                                sum(
                                    action_debug_sum_list[i].get("luban_skill1_aim_assist_count", 0.0)
                                    for i, do_sample in enumerate(self.do_samples)
                                    if do_sample
                                )
                            )
                            self.monitor.put_data({os.getpid(): monitor_data})
                            self.last_report_monitor_time = now

                    if len(frame_collector) > 0 and not is_eval:
                        yield sample_process(frame_collector)
                    break

    def reset_agents(self, observation):
        opponent_agent = self.env_conf_manager.get_opponent_agent()
        monitor_side = self.env_conf_manager.get_monitor_side()
        is_train_test = os.environ.get("is_train_test", "False").lower() == "true"

        self.do_predicts = [True, True]
        self.do_samples = [True, True]

        for i, agent in enumerate(self.agents):
            if i == monitor_side:
                agent.load_model(id="latest")
            else:
                if opponent_agent == "common_ai":
                    self.do_predicts[i] = False
                    self.do_samples[i] = False
                elif opponent_agent == "selfplay":
                    agent.load_model(id="latest")
                else:
                    eval_candidate_model = get_valid_model_pool(self.logger)
                    if int(opponent_agent) not in eval_candidate_model:
                        raise ValueError(f"opponent_agent model_id {opponent_agent} not in {eval_candidate_model}")
                    if is_train_test:
                        self.logger.info("Run train_test, cannot get opponent agent, so replace with latest model")
                        agent.load_model(id="latest")
                    else:
                        agent.load_opponent_agent(id=opponent_agent)
                    self.do_samples[i] = False
            agent.reset(observation[str(i)])

    def _accumulate_reward_items(self, target, reward):
        for reward_name in GameConfig.REWARD_WEIGHT_DICT:
            target[reward_name] += float(reward.get(f"{reward_name}_weight", 0.0))

    def _apply_no_op_streak_reward(self, reward, action, agent_idx):
        button = int(action[0]) if isinstance(action, list) and action else 0
        if button in (0, 1):
            self.no_op_streaks[agent_idx] += 1
        else:
            self.no_op_streaks[agent_idx] = 0
        if self.no_op_streaks[agent_idx] < GameConfig.NO_OP_STREAK_THRESHOLD:
            return
        value = float(GameConfig.NO_OP_STREAK_REWARD)
        reward["no_op_streak_penalty_origin"] = value
        reward["no_op_streak_penalty_weight"] = value
        reward["reward_sum"] += value

    def _apply_terminal_reward(self, observation, reward_item_sum, env_obs=None, terminated=False, truncated=False):
        reward = observation.get("reward", {})
        terminal_value = self._terminal_reward_value(
            observation,
            env_obs=env_obs,
            terminated=terminated,
            truncated=truncated,
        )
        if terminal_value == 0.0:
            return 0.0
        reward["win_origin"] = terminal_value
        reward["win_weight"] = terminal_value
        reward["reward_sum"] = float(reward.get("reward_sum", 0.0)) + terminal_value
        reward_item_sum["win"] += terminal_value
        return terminal_value

    def _terminal_reward_value(self, observation, env_obs=None, terminated=False, truncated=False):
        inferred_win, inferred_source = self._infer_terminal_outcome_from_tower_hp(
            observation,
            terminated=terminated,
            truncated=truncated,
        )
        if inferred_win is not None:
            self._log_terminal_reward_debug(
                observation,
                source=inferred_source,
                inferred_win=inferred_win,
                env_obs=env_obs,
                terminated=terminated,
                truncated=truncated,
            )
            return float(GameConfig.TERMINAL_WIN_REWARD) if inferred_win else -float(GameConfig.TERMINAL_WIN_REWARD)

        top_level_win = observation.get("win", None)
        frame_state = observation.get("frame_state", {}) or {}
        frame_state_win = frame_state.get("win", None)
        parsed_win = self._parse_terminal_win_flag(top_level_win)
        win_source = "observation.win"
        if parsed_win is None:
            parsed_win = self._parse_terminal_win_flag(frame_state_win)
            win_source = "observation.frame_state.win"
        if parsed_win is not None:
            self._log_terminal_reward_debug(
                observation,
                source=win_source,
                inferred_win=parsed_win,
                env_obs=env_obs,
                terminated=terminated,
                truncated=truncated,
            )
            return float(GameConfig.TERMINAL_WIN_REWARD) if parsed_win else -float(GameConfig.TERMINAL_WIN_REWARD)

        env_win = self._terminal_win_from_env_metrics(env_obs, observation)
        if env_win is not None:
            self._log_terminal_reward_debug(
                observation,
                source="env_win_rate",
                inferred_win=env_win,
                env_obs=env_obs,
                terminated=terminated,
                truncated=truncated,
            )
            return float(GameConfig.TERMINAL_WIN_REWARD) if env_win else -float(GameConfig.TERMINAL_WIN_REWARD)

        self._log_terminal_reward_debug(
            observation,
            source="missing_terminal_outcome",
            inferred_win=None,
            env_obs=env_obs,
            terminated=terminated,
            truncated=truncated,
        )
        return 0.0

    def _infer_terminal_outcome_from_tower_hp(self, observation, terminated=False, truncated=False):
        if truncated and not terminated:
            return None, "truncated_without_termination"

        frame_state = observation.get("frame_state", {}) or {}
        hero_states = frame_state.get("hero_states", []) or []
        npc_states = frame_state.get("npc_states", []) or []
        main_camp = self._terminal_main_camp(observation, hero_states)
        if main_camp not in (1, 2):
            return None, "missing_main_camp"

        main_tower_hp, enemy_tower_hp = self._terminal_tower_hp_by_camp(npc_states, main_camp)
        if main_tower_hp is None or enemy_tower_hp is None:
            return None, "missing_tower_hp"
        if main_tower_hp <= 0.0 and enemy_tower_hp > 0.0:
            return False, "tower_hp_destroyed"
        if enemy_tower_hp <= 0.0 and main_tower_hp > 0.0:
            return True, "tower_hp_destroyed"
        if terminated and main_tower_hp != enemy_tower_hp:
            return enemy_tower_hp < main_tower_hp, "tower_hp_compare_on_terminated"
        return None, "tower_hp_inconclusive"

    def _terminal_main_camp(self, observation, hero_states):
        main_camp = self._camp_key(observation.get("player_camp", observation.get("camp", None)))
        if main_camp in (1, 2):
            return main_camp

        player_id = observation.get("player_id", None)
        for hero in hero_states:
            if player_id is None:
                break
            hero_player_id = self._get(hero, ["player_id", "playerId", "runtime_id", "runtimeId"], None)
            if hero_player_id == player_id:
                return self._camp_key(self._get(hero, "camp", None))
        return main_camp

    def _terminal_tower_hp_by_camp(self, npc_states, main_camp):
        main_tower_hp = None
        enemy_tower_hp = None
        normalized_main_camp = self._camp_key(main_camp)

        for npc in npc_states:
            sub_type = self._actor_sub_type(npc)
            if sub_type not in (21, "21", "ACTOR_SUB_TOWER"):
                continue
            hp = self._coerce_float(self._get(npc, ["hp", "HP"], 0.0))
            if hp is None:
                hp = 0.0
            npc_camp = self._camp_key(self._get(npc, "camp", None))
            if npc_camp == normalized_main_camp:
                main_tower_hp = hp
            elif npc_camp in (1, 2):
                enemy_tower_hp = hp

        return main_tower_hp, enemy_tower_hp

    def _actor_sub_type(self, actor):
        actor_state = self._get(actor, "actor_state", None) or self._get(actor, "actorState", None)
        if actor_state:
            sub_type = self._get(actor_state, ["sub_type", "subType"], None)
            if sub_type is not None:
                return sub_type
        return self._get(actor, ["sub_type", "subType"], None)

    def _parse_terminal_win_flag(self, win):
        if win is None:
            return None
        if isinstance(win, bool):
            return win
        if isinstance(win, (int, float)):
            if float(win) == 1.0:
                return True
            if float(win) == 0.0:
                return False
            return bool(win)
        if isinstance(win, str):
            lowered = win.strip().lower()
            if lowered in ("true", "win", "won", "victory", "1"):
                return True
            if lowered in ("false", "lose", "loss", "lost", "defeat", "0"):
                return False
        return None

    def _terminal_win_from_env_metrics(self, env_obs, observation):
        metrics = self._extract_terminal_env_metrics(env_obs=env_obs, observation=observation)
        win_rate = self._coerce_float(metrics.get("win_rate", None))
        if win_rate == 1.0:
            return True
        if win_rate == 0.0:
            return False
        return None

    def _extract_terminal_env_metrics(self, env_obs=None, observation=None):
        candidates = []
        if isinstance(env_obs, dict):
            candidates.append(env_obs)
            extra_info = env_obs.get("extra_info", None)
            if isinstance(extra_info, dict):
                candidates.append(extra_info)
            for value in env_obs.values():
                if isinstance(value, dict):
                    candidates.append(value)
        if isinstance(observation, dict):
            candidates.append(observation)
            frame_state = observation.get("frame_state", None)
            if isinstance(frame_state, dict):
                candidates.append(frame_state)

        metrics = {
            "win_rate": None,
            "self_tower_hp": None,
            "enemy_tower_hp": None,
        }
        for key in metrics:
            for candidate in candidates:
                if key in candidate:
                    metrics[key] = candidate.get(key)
                    break
        return metrics

    def _log_terminal_reward_debug(self, observation, source, inferred_win, env_obs=None, terminated=False, truncated=False):
        if not self.logger:
            return
        frame_state = observation.get("frame_state", {}) or {}
        main_camp = self._terminal_main_camp(observation, frame_state.get("hero_states", []) or [])
        main_tower_hp, enemy_tower_hp = self._terminal_tower_hp_by_camp(
            frame_state.get("npc_states", []) or [],
            main_camp,
        )
        env_metrics = self._extract_terminal_env_metrics(env_obs=env_obs, observation=observation)
        self.logger.info(
            "terminal reward debug: "
            f"source={source}, "
            f"inferred_win={repr(inferred_win)}, "
            f"main_camp={repr(main_camp)}, "
            f"tower_hp=({repr(main_tower_hp)}, {repr(enemy_tower_hp)}), "
            f"env_win_rate={repr(env_metrics.get('win_rate'))}, "
            f"env_self_tower_hp={repr(env_metrics.get('self_tower_hp'))}, "
            f"env_enemy_tower_hp={repr(env_metrics.get('enemy_tower_hp'))}, "
            f"terminated={terminated}, truncated={truncated}"
        )

    def _coerce_float(self, value):
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _accumulate_reward_debug_items(self, target, reward):
        for debug_name in GameConfig.REWARD_DEBUG_KEY_LIST:
            target[debug_name] += float(reward.get(debug_name, 0.0))

    def _accumulate_action_debug_items(self, target, action):
        if not isinstance(action, list) or not action:
            return
        button = int(action[0])
        if len(action) >= len(Config.LABEL_SIZE_LIST):
            target_index = int(action[len(Config.LABEL_SIZE_LIST) - 1])
            if 0 <= target_index < len(Config.TARGET_ORDER):
                target_name = Config.TARGET_ORDER[target_index]
                if target_name == "none":
                    target["target_none_count"] += 1.0
                elif target_name == "enemy_hero":
                    target["target_enemy_hero_count"] += 1.0
                    if button == 3:
                        target["attack_target_enemy_hero_count"] += 1.0
                    elif button in (4, 5, 6, 10):
                        target["skill_target_enemy_hero_count"] += 1.0
                elif target_name == "self_hero":
                    target["target_self_hero_count"] += 1.0
                elif target_name.startswith("enemy_soldier"):
                    target["target_enemy_soldier_count"] += 1.0
                    if button == 3:
                        target["attack_target_enemy_soldier_count"] += 1.0
                elif target_name == "enemy_tower":
                    target["target_enemy_tower_count"] += 1.0
                elif target_name == "monster":
                    target["target_monster_count"] += 1.0
        if button in (0, 1):
            target["action_noop_count"] += 1.0
        elif button == 2:
            target["action_move_count"] += 1.0
        elif button == 3:
            target["action_attack_count"] += 1.0
        elif button in (4, 5, 6, 10):
            target["action_skill_count"] += 1.0
        elif button == 7:
            target["action_recover_count"] += 1.0
        elif button == 8:
            target["action_summoner_count"] += 1.0
        elif button == 9:
            target["action_recall_count"] += 1.0
        elif button == 11:
            target["action_equipment_count"] += 1.0

    def _accumulate_skill_usage_items(self, target, frame_state, agent):
        main_hero = self._find_agent_hero(frame_state, agent)
        if not main_hero:
            return
        skill_state = self._get(main_hero, ["skill_state", "skillState"], {}) or {}
        slot_states = self._get(skill_state, ["slot_states", "slotStates"], []) or []
        for slot in slot_states:
            used_times = float(self._get(slot, "succUsedInFrame", self._get(slot, "succ_used_in_frame", 0)) or 0)
            if used_times <= 0:
                continue
            slot_key = self._slot_type_key(self._get(slot, ["slot_type", "slotType"], None))
            config_id = int(self._get(slot, "configId", self._get(slot, "config_id", 0)) or 0)
            if slot_key == "SLOT_SKILL_1":
                target["skill_1_used_count"] += used_times
            elif slot_key == "SLOT_SKILL_2":
                target["skill_2_used_count"] += used_times
            elif slot_key == "SLOT_SKILL_3":
                target["skill_3_used_count"] += used_times
            elif slot_key == "SLOT_SKILL_4":
                target["recover_used_count"] += used_times
            elif slot_key == "SLOT_SKILL_5" or config_id in GameConfig.SUMMONER_SKILL_IDS:
                target["summoner_skill_used_count"] += used_times

    def _find_agent_hero(self, frame_state, agent):
        player_id = getattr(agent, "player_id", None)
        hero_camp = getattr(agent, "hero_camp", None)
        for hero in frame_state.get("hero_states", []):
            if self._get(hero, "runtime_id", None) == player_id or self._get(hero, "player_id", None) == player_id:
                return hero
        for hero in frame_state.get("hero_states", []):
            if self._camp_key(self._get(hero, "camp", None)) == self._camp_key(hero_camp):
                return hero
        return None

    def _first_selected_skill(self, select_skills, hero_ids):
        if not isinstance(select_skills, dict):
            return int(select_skills) if select_skills is not None else None
        for hero_id in hero_ids:
            skill_id = select_skills.get(hero_id, select_skills.get(str(hero_id)))
            if skill_id is not None:
                return int(skill_id)
        return None

    def _slot_type_key(self, slot_type):
        if isinstance(slot_type, int):
            return f"SLOT_SKILL_{slot_type}"
        return str(slot_type)

    def _camp_key(self, camp):
        if camp in (1, "1", "PLAYERCAMP_1", "blue_camp"):
            return 1
        if camp in (2, "2", "PLAYERCAMP_2", "red_camp"):
            return 2
        return camp

    def _get(self, obj, key, default=0):
        if isinstance(key, (list, tuple)):
            for item in key:
                value = self._get(obj, item, None)
                if value is not None:
                    return value
            return default
        if isinstance(obj, dict):
            return obj.get(key, default)
        return getattr(obj, key, default)
