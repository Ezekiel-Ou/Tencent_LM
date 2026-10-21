#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""


import os
import sys
import time
import random
from agent_ppo.feature.definition import (
    sample_process,
    build_frame,
    FrameCollector,
    NONE_ACTION,
    lineup_iterator_roundrobin_camp_heroes,
)
from agent_ppo.conf.conf import GameConfig
from tools.env_conf_manager import EnvConfManager
from tools.model_pool_utils import get_valid_model_pool
from tools.metrics_utils import get_training_metrics
from common_python.utils.workflow_disaster_recovery import handle_disaster_recovery


def _emit_debug_log(logger, message):
    """Emit debug probes through both framework logger and stdout."""
    if logger is not None:
        log_fn = getattr(logger, "warning", None) or getattr(logger, "info", None)
        if log_fn is not None:
            log_fn(message)
    print(message, flush=True)


def workflow(envs, agents, logger=None, monitor=None, *args, **kwargs):
    # Whether the agent is training, corresponding to do_predicts
    # 智能体是否进行训练
    do_learns = [True, True]
    last_save_model_time = time.time()

    # Create environment configuration manager instance
    # 创建对局配置管理器实例
    env_conf_manager = EnvConfManager(
        config_path="agent_ppo/conf/train_env_conf.toml",
        logger=logger,
    )

    # Lineup iterator (112:Luban, 133:DiRenjie)
    # 阵容迭代器 (112:鲁班， 133:狄仁杰)
    lineup_iterator = lineup_iterator_roundrobin_camp_heroes([112, 133])

    # Create EpisodeRunner instance
    # 创建 EpisodeRunner 实例
    episode_runner = EpisodeRunner(
        env=envs[0],
        agents=agents,
        logger=logger,
        monitor=monitor,
        env_conf_manager=env_conf_manager,
        lineup_iterator=lineup_iterator,
    )

    # Phase A debug mode: bypass training, drive env with a fixed-action DebugAgent,
    # dump observation statistics, then sys.exit(0). Toggled via [debug] in toml.
    # 阶段 A 调试模式: 用固定动作 DebugAgent 驱动 env, 采集 observation 取值范围, 然后 sys.exit(0).
    debug_conf = env_conf_manager.get_current_config().get("debug", {}) if hasattr(env_conf_manager, "get_current_config") else {}
    enabled_debug_modes = [
        name
        for name in (
            "enable_debug_agent",
            "enable_luban_buff_debug",
            "enable_passive_buff_debug",
            "enable_luban_skill1_passive_buff_debug",
        )
        if debug_conf.get(name, False)
    ]
    if len(enabled_debug_modes) > 1:
        raise ValueError(f"[debug] debug modes are mutually exclusive: {enabled_debug_modes}")

    if debug_conf.get("enable_passive_buff_debug", False):
        _emit_debug_log(logger, f"[PASSIVE_BUFF_ENTER] agent_ppo workflow entering PASSIVE_BUFF_DEBUG mode, debug_conf={debug_conf}")
        episode_runner.run_passive_buff_debug_episodes(debug_conf)
        _emit_debug_log(logger, "[PASSIVE_BUFF_EXIT] agent_ppo workflow PASSIVE_BUFF_DEBUG mode finished, exiting.")
        sys.exit(0)

    if debug_conf.get("enable_luban_skill1_passive_buff_debug", False):
        _emit_debug_log(logger, f"[PASSIVE_BUFF_ENTER] agent_ppo workflow entering LUBAN_SKILL1_PASSIVE_BUFF_DEBUG mode, debug_conf={debug_conf}")
        episode_runner.run_luban_skill1_passive_buff_debug_episodes(debug_conf)
        _emit_debug_log(logger, "[PASSIVE_BUFF_EXIT] agent_ppo workflow LUBAN_SKILL1_PASSIVE_BUFF_DEBUG mode finished, exiting.")
        sys.exit(0)

    if debug_conf.get("enable_luban_buff_debug", False):
        if logger is not None:
            logger.info(f"agent_ppo workflow entering LUBAN_BUFF_DEBUG mode, debug_conf={debug_conf}")
        episode_runner.run_luban_buff_debug_episodes(debug_conf)
        if logger is not None:
            logger.info("agent_ppo workflow LUBAN_BUFF_DEBUG mode finished, exiting.")
        sys.exit(0)

    if debug_conf.get("enable_debug_agent", False):
        if logger is not None:
            logger.info(f"agent_ppo workflow entering DEBUG mode, debug_conf={debug_conf}")
        episode_runner.run_debug_episodes(debug_conf)
        if logger is not None:
            logger.info("agent_ppo workflow DEBUG mode finished, exiting.")
        sys.exit(0)

    while True:
        # Run episodes and collect data
        # 运行对局并收集数据
        for g_data in episode_runner.run_episodes():
            for index, (d_learn, agent) in enumerate(zip(do_learns, agents)):
                if d_learn and len(g_data[index]) > 0:
                    # The learner trains in a while true loop, here learn actually sends samples
                    # learner 采用 while true 训练，此处 learn 实际为发送样本
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

    def _call_init_config(self, usr_conf):
        """Call init_config on both agents to get summoner skill selections,
        then inject the results into usr_conf.
        调用双方 agent 的 init_config 获取召唤师技能选择，并注入 usr_conf。
        """
        blue_hero_ids, red_hero_ids = EnvConfManager.extract_hero_ids_from_usr_conf(usr_conf)

        # camp_keys[i] is the camp key for agents[i] based on monitor_side
        # monitor_side 的 agent 对应 blue/red 取决于 monitor_side 配置
        monitor_side = self.env_conf_manager.get_monitor_side()
        camp_keys = ["blue_camp", "red_camp"]

        for agent_idx, agent in enumerate(self.agents):
            # Determine which camp this agent controls
            # 确定该 agent 控制哪个阵营
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
            }

            select_skills = agent.init_config(config_data)
            EnvConfManager.inject_select_skills(usr_conf, camp_key, select_skills)
            self.logger.info(
                f"Agent[{agent_idx}] init_config: camp={camp_key}, select_skills={select_skills}"
            )

    def run_episodes(self):
        # Single environment process
        # 单局流程
        while True:
            # Retrieving training metrics
            # 获取训练中的指标
            training_metrics = get_training_metrics()
            if training_metrics:
                for key, value in training_metrics.items():
                    if key == "env":
                        for env_key, env_value in value.items():
                            self.logger.info(f"training_metrics {key} {env_key} is {env_value}")
                    else:
                        self.logger.info(f"training_metrics {key} is {value}")

            # Update environment configuration
            # Can use a list of length 2 to pass in the lineup id of the current game
            # 更新对局配置, 可以用长度为2的列表传入当前对局的阵容id
            lineup = next(self.lineup_iterator)
            usr_conf, is_eval, monitor_side = self.env_conf_manager.update_config(lineup)

            # Call init_config on agents to get summoner skill selections
            # 调用 agent 的 init_config 获取召唤师技能选择，注入 usr_conf
            self._call_init_config(usr_conf)

            # Start a new environment
            # 启动新对局，返回初始环境状态

            env_obs = self.env.reset(usr_conf=usr_conf)
            # Disaster recovery
            # 容灾
            if handle_disaster_recovery(env_obs, self.logger):
                break

            observation = env_obs["observation"]
            extra_info = env_obs["extra_info"]

            # Reset agents
            # 重置智能体
            self.reset_agents(observation)

            # Reset environment frame collector
            # 重置环境帧收集器
            frame_collector = FrameCollector(self.agent_num)

            # Game variables
            # 对局变量
            self.episode_cnt += 1
            frame_no = 0
            reward_sum_list = [0] * self.agent_num
            is_train_test = os.environ.get("is_train_test", "False").lower() == "true"
            self.logger.info(f"Episode {self.episode_cnt} start, usr_conf is {usr_conf}")

            # Reward initialization
            # 回报初始化
            for i, (do_sample, agent) in enumerate(zip(self.do_samples, self.agents)):
                if do_sample:
                    reward = agent.reward_manager.result(observation[str(i)]["frame_state"])
                    observation[str(i)]["reward"] = reward
                    reward_sum_list[i] += reward["reward_sum"]

            while True:
                # Initialize the default actions. If the agent does not make a decision, env.step uses the default action.
                # 初始化默认的actions，如果智能体不进行决策，则env.step使用默认action
                actions = [NONE_ACTION] * self.agent_num

                for index, (do_predict, do_sample, agent) in enumerate(
                    zip(self.do_predicts, self.do_samples, self.agents)
                ):
                    if do_predict:
                        if not is_eval:
                            actions[index] = agent.predict(observation[str(index)])
                        else:
                            actions[index] = agent.exploit(observation[str(index)])

                        # Only sample when do_sample=True and is_eval=False
                        # 评估对局数据不采样，不是训练中最新模型产生的数据不采样
                        if not is_eval and do_sample:
                            frame = build_frame(agent, observation[str(index)])
                            frame_collector.save_frame(frame, agent_id=index)

                # Step forward
                # 推进环境到下一帧，得到新的状态
                env_reward, env_obs = self.env.step(actions)
                # Disaster recovery
                # 容灾
                if handle_disaster_recovery(env_obs, self.logger):
                    break

                frame_no = env_obs["frame_no"]
                observation = env_obs["observation"]
                extra_info = env_obs["extra_info"]
                terminated = env_obs["terminated"]
                truncated = env_obs["truncated"]

                # Reward generation
                # 计算回报，作为当前环境状态observation的一部分
                for i, (do_sample, agent) in enumerate(zip(self.do_samples, self.agents)):
                    if do_sample:
                        reward = agent.reward_manager.result(observation[str(i)]["frame_state"])
                        observation[str(i)]["reward"] = reward
                        reward_sum_list[i] += reward["reward_sum"]

                # Normal end or timeout exit, run train_test will exit early
                # 正常结束或超时退出，运行train_test时会提前退出
                is_gameover = terminated or truncated or (is_train_test and frame_no >= 1000)
                if is_gameover:
                    self.logger.info(
                        f"episode_{self.episode_cnt} terminated in fno_{frame_no}, truncated:{truncated}, eval:{is_eval}, reward_sum:{reward_sum_list[monitor_side]}"
                    )
                    # Reward for saving the last state of the environment
                    # 保存环境最后状态的reward
                    for i, (do_sample, agent) in enumerate(zip(self.do_samples, self.agents)):
                        if not is_eval and do_sample:
                            frame_collector.save_last_frame(
                                agent_id=i,
                                reward=observation[str(i)]["reward"]["reward_sum"],
                            )

                    now = time.time()
                    if now - self.last_report_monitor_time >= 60:
                        monitor_data = {"episode_cnt": self.episode_cnt}
                        if self.monitor:
                            if is_eval:
                                monitor_data["reward"] = round(reward_sum_list[monitor_side], 2)
                            self.monitor.put_data({os.getpid(): monitor_data})
                            self.last_report_monitor_time = now

                    # Sample process
                    # 进行样本处理，准备训练
                    if len(frame_collector) > 0 and not is_eval:
                        list_agents_samples = sample_process(frame_collector)
                        yield list_agents_samples
                    break

    def reset_agents(self, observation):
        opponent_agent = self.env_conf_manager.get_opponent_agent()
        monitor_side = self.env_conf_manager.get_monitor_side()
        is_train_test = os.environ.get("is_train_test", "False").lower() == "true"

        # The 'do_predicts' specifies which agents are to perform model predictions.
        # do_predicts 指定哪些智能体要进行模型预测
        # The 'do_samples' specifies which agents are to perform training sampling.
        # do_samples 指定哪些智能体要进行训练采样
        self.do_predicts = [True, True]
        self.do_samples = [True, True]

        # Load model according to the configuration
        # 根据对局配置加载模型
        for i, agent in enumerate(self.agents):
            # Report the latest model in the training camp to the monitor
            # 训练中最新模型所在阵营上报监控
            if i == monitor_side:
                # monitor_side uses the latest model
                # monitor_side 使用最新模型
                agent.load_model(id="latest")
            else:
                if opponent_agent == "common_ai":
                    # common_ai does not need to load a model, no need to predict
                    # 如果对手是 common_ai 则不需要加载模型, 也不需要进行预测
                    self.do_predicts[i] = False
                    self.do_samples[i] = False
                elif opponent_agent == "selfplay":
                    # Training model, "latest" - latest model, "random" - random model from the model pool
                    # 加载训练过的模型，可以选择最新模型，也可以选择随机模型 "latest" - 最新模型, "random" - 模型池中随机模型
                    agent.load_model(id="latest")
                else:
                    # Opponent model, model_id is checked from kaiwu.json
                    # 选择kaiwu.json中设置的对手模型, model_id 即 opponent_agent，必须设置正确否则报错
                    eval_candidate_model = get_valid_model_pool(self.logger)
                    if int(opponent_agent) not in eval_candidate_model:
                        raise Exception(f"opponent_agent model_id {opponent_agent} not in {eval_candidate_model}")
                    else:
                        if is_train_test:
                            # Run train_test, cannot get opponent agent, so replace with latest model
                            # 运行 train_test 时, 无法获取到对手模型，因此将替换为最新模型
                            self.logger.info(f"Run train_test, cannot get opponent agent, so replace with latest model")
                            agent.load_model(id="latest")
                        else:
                            agent.load_opponent_agent(id=opponent_agent)
                        self.do_samples[i] = False
            # Reset agent
            # 重置agent
            agent.reset(observation[str(i)])

    # ===== Passive buff debug helpers =====

    def run_passive_buff_debug_episodes(self, debug_conf):
        """Drive 112/133 lineups with normal-attack-air only and log buff ids.

        This path bypasses model load, reward, sampling, and training. It is
        meant for platform-side passive-state identification from Aisrv logs and
        numeric monitor fields.
        """
        from agent_ppo.debug import (
            PassiveBuffDebugAgent,
            extract_passive_buff_state,
            find_own_hero,
            format_passive_buff_line,
            format_passive_summary_line,
            parse_passive_attack_no,
        )

        lineups = self._normalize_passive_debug_lineups(
            debug_conf.get("passive_buff_debug_lineups", [[112, 112], [133, 133]])
        )
        episodes_per_lineup = int(debug_conf.get("passive_buff_debug_episodes_per_lineup", 1))
        max_frames = int(debug_conf.get("passive_buff_debug_max_frames", 900))
        settle_steps = int(debug_conf.get("passive_buff_debug_settle_steps", 12))
        debug_agent = PassiveBuffDebugAgent(
            attack_interval_steps=int(debug_conf.get("passive_buff_debug_attack_interval_steps", 7)),
            max_attacks=int(debug_conf.get("passive_buff_debug_max_attacks", 12)),
        )

        is_train_test = os.environ.get("is_train_test", "False").lower() == "true"
        side_names = ["blue", "red"]
        global_ep = 0
        self._passive_debug_log(
            f"[PASSIVE_BUFF_CONFIG] lineups={lineups} episodes_per_lineup={episodes_per_lineup} "
            f"max_frames={max_frames} max_attacks={debug_agent.max_attacks} "
            f"attack_interval_steps={debug_agent.attack_interval_steps} settle_steps={settle_steps}"
        )

        for lineup in lineups:
            for _ in range(max(1, episodes_per_lineup)):
                debug_agent.reset()
                usr_conf, _, _ = self.env_conf_manager.update_config(lineup)
                self._inject_debug_summoner_skills(usr_conf)
                self._passive_debug_log(
                    f"[PASSIVE_BUFF_EPISODE_BEGIN] ep={global_ep} lineup={lineup} usr_conf={usr_conf}"
                )

                env_obs = self.env.reset(usr_conf=usr_conf)
                if handle_disaster_recovery(env_obs, self.logger):
                    self._passive_debug_log(
                        f"[PASSIVE_BUFF_ABORT] ep={global_ep} lineup={lineup} "
                        f"reason=reset_disaster_recovery env_obs={env_obs}"
                    )
                    return

                observation = env_obs.get("observation", {})
                last_buff_state = {}
                skill_ids_seen = {camp_idx: {} for camp_idx in range(self.agent_num)}
                mark_ids_seen = {camp_idx: {} for camp_idx in range(self.agent_num)}
                attacks_attempted = {camp_idx: 0 for camp_idx in range(self.agent_num)}

                for camp_idx in range(self.agent_num):
                    obs_i = observation.get(str(camp_idx)) if isinstance(observation, dict) else None
                    hero_id = lineup[camp_idx] if camp_idx < len(lineup) else None
                    last_buff_state[camp_idx] = extract_passive_buff_state(find_own_hero(obs_i, hero_id))

                self._passive_debug_log(
                    f"[PASSIVE_BUFF_START] ep={global_ep} lineup={lineup} "
                    f"max_attacks={debug_agent.max_attacks} "
                    f"attack_interval_steps={debug_agent.attack_interval_steps}"
                )

                step_no = 0
                frame_no = env_obs.get("frame_no", 0)
                while True:
                    actions = []
                    tags = []
                    for camp_idx in range(self.agent_num):
                        obs_i = observation.get(str(camp_idx)) if isinstance(observation, dict) else None
                        action, tag = debug_agent.act(obs_i or {}, step_no, agent_id=camp_idx)
                        actions.append(action)
                        tags.append(tag)
                        if self._is_passive_air_attack_attempt(action, tag):
                            attacks_attempted[camp_idx] += 1

                    env_reward, env_obs = self.env.step(actions)
                    if handle_disaster_recovery(env_obs, self.logger):
                        self._passive_debug_log(
                            f"[PASSIVE_BUFF_ABORT] ep={global_ep} lineup={lineup} "
                            f"step={step_no} reason=step_disaster_recovery env_obs={env_obs}"
                        )
                        return

                    frame_no = env_obs.get("frame_no", frame_no)
                    observation = env_obs.get("observation", {})
                    terminated = env_obs.get("terminated", 0)
                    truncated = env_obs.get("truncated", 0)

                    for camp_idx in range(self.agent_num):
                        obs_i = observation.get(str(camp_idx)) if isinstance(observation, dict) else None
                        hero_id = lineup[camp_idx] if camp_idx < len(lineup) else 0
                        side = side_names[camp_idx] if camp_idx < len(side_names) else str(camp_idx)
                        skills, marks = extract_passive_buff_state(find_own_hero(obs_i, hero_id))
                        current_state = (skills, marks)
                        attack_no = parse_passive_attack_no(tags[camp_idx])

                        if self._is_passive_air_attack_attempt(actions[camp_idx], tags[camp_idx]):
                            self._record_passive_buff_ids(
                                skill_ids_seen[camp_idx], mark_ids_seen[camp_idx], skills, marks
                            )
                            self._log_passive_buff_debug_event(
                                event="after_attack",
                                ep=global_ep,
                                lineup=lineup,
                                step_no=step_no,
                                frame_no=frame_no,
                                side=side,
                                side_idx=camp_idx,
                                hero_id=hero_id,
                                tag=tags[camp_idx],
                                action=actions[camp_idx],
                                attack_no=attack_no,
                                skills=skills,
                                marks=marks,
                            )
                        elif current_state != last_buff_state.get(camp_idx):
                            self._record_passive_buff_ids(
                                skill_ids_seen[camp_idx], mark_ids_seen[camp_idx], skills, marks
                            )
                            self._log_passive_buff_debug_event(
                                event="buff_changed",
                                ep=global_ep,
                                lineup=lineup,
                                step_no=step_no,
                                frame_no=frame_no,
                                side=side,
                                side_idx=camp_idx,
                                hero_id=hero_id,
                                tag=tags[camp_idx],
                                action=actions[camp_idx],
                                attack_no=attack_no,
                                skills=skills,
                                marks=marks,
                            )
                        last_buff_state[camp_idx] = current_state

                    step_no += 1
                    if (
                        terminated
                        or truncated
                        or frame_no >= max_frames
                        or (is_train_test and frame_no >= 1000)
                        or debug_agent.episode_done(step_no, settle_steps=settle_steps)
                    ):
                        break

                for camp_idx in range(self.agent_num):
                    if True:
                        side = side_names[camp_idx] if camp_idx < len(side_names) else str(camp_idx)
                        hero_id = lineup[camp_idx] if camp_idx < len(lineup) else 0
                        self._passive_debug_log(
                            format_passive_summary_line(
                                global_ep,
                                lineup,
                                side,
                                hero_id,
                                attacks_attempted[camp_idx],
                                skill_ids_seen[camp_idx],
                                mark_ids_seen[camp_idx],
                                frame_no,
                                debug_agent.illegal_button_count,
                            )
                        )
                global_ep += 1

    @staticmethod
    def _normalize_passive_debug_lineups(raw_lineups):
        if not isinstance(raw_lineups, list) or not raw_lineups:
            return [[112, 112], [133, 133]]
        lineups = []
        for item in raw_lineups:
            if not isinstance(item, (list, tuple)) or len(item) != 2:
                continue
            try:
                lineups.append([int(item[0]), int(item[1])])
            except (TypeError, ValueError):
                continue
        return lineups or [[112, 112], [133, 133]]

    def run_luban_skill1_passive_buff_debug_episodes(self, debug_conf):
        """Cast Luban skill 1 once, then air-attack to identify sweep buff ids."""
        from agent_ppo.debug import (
            LubanSkill1PassiveBuffDebugAgent,
            extract_passive_buff_state,
            find_own_hero,
            format_passive_summary_line,
            parse_passive_attack_no,
        )

        lineup = [112, 112]
        episodes = int(debug_conf.get("luban_skill1_buff_debug_episodes", 1))
        max_frames = int(debug_conf.get("luban_skill1_buff_debug_max_frames", 500))
        settle_steps = int(debug_conf.get("luban_skill1_buff_debug_settle_steps", 12))
        debug_agent = LubanSkill1PassiveBuffDebugAgent(
            skill_wait_steps=int(debug_conf.get("luban_skill1_buff_debug_skill_wait_steps", 20)),
            skill_to_attack_gap_steps=int(debug_conf.get("luban_skill1_buff_debug_skill_to_attack_gap_steps", 7)),
            attack_interval_steps=int(debug_conf.get("luban_skill1_buff_debug_attack_interval_steps", 7)),
            max_attacks=int(debug_conf.get("luban_skill1_buff_debug_max_attacks", 4)),
        )

        is_train_test = os.environ.get("is_train_test", "False").lower() == "true"
        side_names = ["blue", "red"]
        self._passive_debug_log(
            f"[PASSIVE_BUFF_CONFIG] mode=luban_skill1_then_air_attack lineup={lineup} "
            f"episodes={episodes} max_frames={max_frames} skill_wait_steps={debug_agent.skill_wait_steps} "
            f"skill_to_attack_gap_steps={debug_agent.skill_to_attack_gap_steps} "
            f"attack_interval_steps={debug_agent.attack_interval_steps} max_attacks={debug_agent.max_attacks}"
        )

        for ep in range(max(1, episodes)):
            debug_agent.reset()
            usr_conf, _, _ = self.env_conf_manager.update_config(lineup)
            self._inject_debug_summoner_skills(usr_conf)
            self._passive_debug_log(
                f"[PASSIVE_BUFF_EPISODE_BEGIN] mode=luban_skill1_then_air_attack ep={ep} lineup={lineup} usr_conf={usr_conf}"
            )

            env_obs = self.env.reset(usr_conf=usr_conf)
            if handle_disaster_recovery(env_obs, self.logger):
                self._passive_debug_log(
                    f"[PASSIVE_BUFF_ABORT] mode=luban_skill1_then_air_attack ep={ep} "
                    f"reason=reset_disaster_recovery env_obs={env_obs}"
                )
                return

            observation = env_obs.get("observation", {})
            last_buff_state = {}
            skill_ids_seen = {camp_idx: {} for camp_idx in range(self.agent_num)}
            mark_ids_seen = {camp_idx: {} for camp_idx in range(self.agent_num)}
            attacks_attempted = {camp_idx: 0 for camp_idx in range(self.agent_num)}
            skill_cast_attempted = {camp_idx: 0 for camp_idx in range(self.agent_num)}

            for camp_idx in range(self.agent_num):
                obs_i = observation.get(str(camp_idx)) if isinstance(observation, dict) else None
                last_buff_state[camp_idx] = extract_passive_buff_state(find_own_hero(obs_i, 112))

            self._passive_debug_log(
                f"[PASSIVE_BUFF_START] mode=luban_skill1_then_air_attack ep={ep} lineup={lineup}"
            )

            step_no = 0
            frame_no = env_obs.get("frame_no", 0)
            while True:
                actions = []
                tags = []
                for camp_idx in range(self.agent_num):
                    obs_i = observation.get(str(camp_idx)) if isinstance(observation, dict) else None
                    action, tag = debug_agent.act(obs_i or {}, step_no, agent_id=camp_idx)
                    actions.append(action)
                    tags.append(tag)
                    if self._is_passive_air_attack_attempt(action, tag):
                        attacks_attempted[camp_idx] += 1
                    if isinstance(tag, str) and tag.startswith("skill1_release") and int(action[0]) == 4:
                        skill_cast_attempted[camp_idx] += 1

                env_reward, env_obs = self.env.step(actions)
                if handle_disaster_recovery(env_obs, self.logger):
                    self._passive_debug_log(
                        f"[PASSIVE_BUFF_ABORT] mode=luban_skill1_then_air_attack ep={ep} "
                        f"step={step_no} reason=step_disaster_recovery env_obs={env_obs}"
                    )
                    return

                frame_no = env_obs.get("frame_no", frame_no)
                observation = env_obs.get("observation", {})
                terminated = env_obs.get("terminated", 0)
                truncated = env_obs.get("truncated", 0)

                for camp_idx in range(self.agent_num):
                    obs_i = observation.get(str(camp_idx)) if isinstance(observation, dict) else None
                    side = side_names[camp_idx] if camp_idx < len(side_names) else str(camp_idx)
                    skills, marks = extract_passive_buff_state(find_own_hero(obs_i, 112))
                    current_state = (skills, marks)
                    attack_no = parse_passive_attack_no(tags[camp_idx])
                    if isinstance(tags[camp_idx], str) and tags[camp_idx].startswith("skill1_release"):
                        self._record_passive_buff_ids(
                            skill_ids_seen[camp_idx], mark_ids_seen[camp_idx], skills, marks
                        )
                        self._log_passive_buff_debug_event(
                            event="after_skill1",
                            ep=ep,
                            lineup=lineup,
                            step_no=step_no,
                            frame_no=frame_no,
                            side=side,
                            side_idx=camp_idx,
                            hero_id=112,
                            tag=tags[camp_idx],
                            action=actions[camp_idx],
                            attack_no=attack_no,
                            skills=skills,
                            marks=marks,
                        )
                    elif self._is_passive_air_attack_attempt(actions[camp_idx], tags[camp_idx]):
                        self._record_passive_buff_ids(
                            skill_ids_seen[camp_idx], mark_ids_seen[camp_idx], skills, marks
                        )
                        self._log_passive_buff_debug_event(
                            event="after_attack",
                            ep=ep,
                            lineup=lineup,
                            step_no=step_no,
                            frame_no=frame_no,
                            side=side,
                            side_idx=camp_idx,
                            hero_id=112,
                            tag=tags[camp_idx],
                            action=actions[camp_idx],
                            attack_no=attack_no,
                            skills=skills,
                            marks=marks,
                        )
                    elif current_state != last_buff_state.get(camp_idx):
                        self._record_passive_buff_ids(
                            skill_ids_seen[camp_idx], mark_ids_seen[camp_idx], skills, marks
                        )
                        self._log_passive_buff_debug_event(
                            event="buff_changed",
                            ep=ep,
                            lineup=lineup,
                            step_no=step_no,
                            frame_no=frame_no,
                            side=side,
                            side_idx=camp_idx,
                            hero_id=112,
                            tag=tags[camp_idx],
                            action=actions[camp_idx],
                            attack_no=attack_no,
                            skills=skills,
                            marks=marks,
                        )
                    last_buff_state[camp_idx] = current_state

                step_no += 1
                if (
                    terminated
                    or truncated
                    or frame_no >= max_frames
                    or (is_train_test and frame_no >= 1000)
                    or debug_agent.episode_done(step_no, settle_steps=settle_steps)
                ):
                    break

            for camp_idx in range(self.agent_num):
                side = side_names[camp_idx] if camp_idx < len(side_names) else str(camp_idx)
                self._passive_debug_log(
                    format_passive_summary_line(
                        ep,
                        lineup,
                        side,
                        112,
                        attacks_attempted[camp_idx],
                        skill_ids_seen[camp_idx],
                        mark_ids_seen[camp_idx],
                        frame_no,
                        debug_agent.illegal_button_count,
                    )
                    + f" skill1_cast_attempted={skill_cast_attempted[camp_idx]}"
                )

    def _log_passive_buff_debug_event(self, event, ep, lineup, step_no, frame_no,
                                      side, side_idx, hero_id, tag, action,
                                      attack_no, skills, marks):
        from agent_ppo.debug import format_passive_buff_line

        if True:
            self._passive_debug_log(
                format_passive_buff_line(
                    event,
                    ep,
                    lineup,
                    step_no,
                    frame_no,
                    side,
                    hero_id,
                    tag,
                    action,
                    attack_no,
                    skills,
                    marks,
                )
            )
        self._emit_passive_buff_monitor(
            event=event,
            ep=ep,
            lineup=lineup,
            side_idx=side_idx,
            hero_id=hero_id,
            attack_no=attack_no,
            skills=skills,
            marks=marks,
        )

    def _emit_passive_buff_monitor(self, event, ep, lineup, side_idx, hero_id,
                                   attack_no, skills, marks):
        if not self.monitor:
            return
        skill_slots = list(skills)[:3]
        mark_slots = list(marks)[:2]
        event_code = {"after_attack": 1, "buff_changed": 2, "after_skill1": 3}.get(event, 0)
        monitor_data = {
            "passive_buff_debug_event_code": event_code,
            "passive_buff_debug_episode": int(ep),
            "passive_buff_debug_lineup_code": int(lineup[0]) * 1000 + int(lineup[1]),
            "passive_buff_debug_side": int(side_idx),
            "passive_buff_debug_hero_id": int(hero_id or 0),
            "passive_buff_debug_attack_no": int(attack_no or 0),
            "passive_buff_debug_skill_count": len(skills),
            "passive_buff_debug_mark_count": len(marks),
        }
        for idx in range(3):
            cid, times = skill_slots[idx] if idx < len(skill_slots) else (0, 0)
            monitor_data[f"passive_buff_debug_skill_id_{idx}"] = int(cid or 0)
            monitor_data[f"passive_buff_debug_skill_times_{idx}"] = int(times or 0)
        for idx in range(2):
            cid, layer = mark_slots[idx] if idx < len(mark_slots) else (0, 0)
            monitor_data[f"passive_buff_debug_mark_id_{idx}"] = int(cid or 0)
            monitor_data[f"passive_buff_debug_mark_layer_{idx}"] = int(layer or 0)
        self.monitor.put_data({os.getpid(): monitor_data})

    def _passive_debug_log(self, message):
        _emit_debug_log(self.logger, message)

    @staticmethod
    def _record_passive_buff_ids(skill_ids_seen, mark_ids_seen, skills, marks):
        for cid, times in skills:
            prev = skill_ids_seen.get(cid, 0)
            skill_ids_seen[cid] = max(prev, 0 if times is None else int(times))
        for cid, layer in marks:
            prev = mark_ids_seen.get(cid, 0)
            mark_ids_seen[cid] = max(prev, 0 if layer is None else int(layer))

    @staticmethod
    def _is_passive_air_attack_attempt(action, tag):
        if not isinstance(tag, str) or "air_attack#" not in tag:
            return False
        try:
            return int(action[0]) == 3
        except (TypeError, ValueError, IndexError):
            return False

    # ===== Luban buff debug helpers =====

    def run_luban_buff_debug_episodes(self, debug_conf):
        """Focused platform-log debug for Luban sweep and recovery buff ids.

        This path bypasses model load, reward, sampling, DumpCollector, and
        training. It only drives both camps with the scripted Luban actions and
        emits logger.info lines when self buff_state changes.
        """
        from agent_ppo.debug import (
            LubanBuffDebugAgent,
            extract_buff_state,
            find_luban_hero,
            format_buff_line,
            format_summary_line,
            parse_attack_no,
        )

        max_episodes = int(debug_conf.get("luban_buff_debug_max_episodes", 4))
        max_frames = int(debug_conf.get("luban_buff_debug_max_frames", 1200))
        settle_steps = int(debug_conf.get("luban_buff_debug_settle_steps", 30))
        debug_agent = LubanBuffDebugAgent(
            attack_cooldown_steps=int(debug_conf.get("luban_buff_debug_attack_cooldown_steps", 7)),
            skill_to_attack_gap_steps=int(debug_conf.get("luban_buff_debug_skill_to_attack_gap_steps", 8)),
            min_premove_steps=int(debug_conf.get("luban_buff_debug_min_premove_steps", 45)),
            max_premove_steps=int(debug_conf.get("luban_buff_debug_max_premove_steps", 50)),
        )

        is_train_test = os.environ.get("is_train_test", "False").lower() == "true"
        side_names = ["blue", "red"]
        for episode_idx in range(max_episodes):
            debug_agent.reset_for_episode(episode_idx)
            usr_conf, _, _ = self.env_conf_manager.update_config([112, 112])
            self._inject_debug_summoner_skills(usr_conf)

            env_obs = self.env.reset(usr_conf=usr_conf)
            if handle_disaster_recovery(env_obs, self.logger):
                break

            observation = env_obs.get("observation", {})
            last_buff_state = {}
            skill_ids_seen = {camp_idx: {} for camp_idx in range(self.agent_num)}
            mark_ids_seen = {camp_idx: {} for camp_idx in range(self.agent_num)}
            attacks_attempted = {camp_idx: 0 for camp_idx in range(self.agent_num)}

            for camp_idx in range(self.agent_num):
                obs_i = observation.get(str(camp_idx)) if isinstance(observation, dict) else None
                last_buff_state[camp_idx] = extract_buff_state(find_luban_hero(obs_i))

            step_no = 0
            frame_no = env_obs.get("frame_no", 0)
            while True:
                primary_obs = observation.get("0") if isinstance(observation, dict) else None
                debug_agent.transition_to_engaged_if_ready(
                    primary_obs, step_no, find_luban_hero(primary_obs)
                )

                actions = []
                tags = []
                for camp_idx in range(self.agent_num):
                    obs_i = observation.get(str(camp_idx)) if isinstance(observation, dict) else None
                    action, tag = debug_agent.act(obs_i or {}, step_no, agent_id=camp_idx)
                    actions.append(action)
                    tags.append(tag)
                    if self._is_luban_attack_attempt(action, tag):
                        attacks_attempted[camp_idx] += 1

                env_reward, env_obs = self.env.step(actions)
                if handle_disaster_recovery(env_obs, self.logger):
                    break

                frame_no = env_obs.get("frame_no", frame_no)
                observation = env_obs.get("observation", {})
                terminated = env_obs.get("terminated", 0)
                truncated = env_obs.get("truncated", 0)

                for camp_idx in range(self.agent_num):
                    obs_i = observation.get(str(camp_idx)) if isinstance(observation, dict) else None
                    skills, marks = extract_buff_state(find_luban_hero(obs_i))
                    current_state = (skills, marks)
                    if debug_agent.phase == "engaged" and current_state != last_buff_state.get(camp_idx):
                        self._record_luban_buff_ids(
                            skill_ids_seen[camp_idx], mark_ids_seen[camp_idx], skills, marks
                        )
                        if self.logger is not None:
                            side = side_names[camp_idx] if camp_idx < len(side_names) else str(camp_idx)
                            self.logger.info(
                                format_buff_line(
                                    "[LUBAN_BUFF]",
                                    episode_idx,
                                    step_no,
                                    frame_no,
                                    side,
                                    tags[camp_idx],
                                    actions[camp_idx],
                                    parse_attack_no(tags[camp_idx]),
                                    skills,
                                    marks,
                                )
                            )
                    last_buff_state[camp_idx] = current_state

                step_no += 1
                if (
                    terminated
                    or truncated
                    or frame_no >= max_frames
                    or (is_train_test and frame_no >= 1000)
                    or debug_agent.episode_done(step_no, settle_steps=settle_steps)
                ):
                    break

            for camp_idx in range(self.agent_num):
                if self.logger is not None:
                    side = side_names[camp_idx] if camp_idx < len(side_names) else str(camp_idx)
                    self.logger.info(
                        format_summary_line(
                            episode_idx,
                            side,
                            attacks_attempted[camp_idx],
                            skill_ids_seen[camp_idx],
                            mark_ids_seen[camp_idx],
                            frame_no,
                        )
                    )

    @staticmethod
    def _record_luban_buff_ids(skill_ids_seen, mark_ids_seen, skills, marks):
        for cid, times in skills:
            prev = skill_ids_seen.get(cid, 0)
            skill_ids_seen[cid] = max(prev, 0 if times is None else int(times))
        for cid, layer in marks:
            prev = mark_ids_seen.get(cid, 0)
            mark_ids_seen[cid] = max(prev, 0 if layer is None else int(layer))

    @staticmethod
    def _is_luban_attack_attempt(action, tag):
        if not isinstance(tag, str) or "attack#" not in tag:
            return False
        try:
            return int(action[0]) == 3
        except (TypeError, ValueError, IndexError):
            return False

    # ===== Phase A debug helpers =====

    def run_debug_episodes(self, debug_conf):
        """Drive both agents with a fixed-action DebugAgent and feed every frame
        into a DumpCollector. Bypasses model load, training sample emission, reward
        manager, and agent.predict — agent state is untouched. After
        debug_max_episodes episodes, writes summary.json and returns. Workflow
        caller sys.exit(0)s after this."""
        from agent_ppo.debug import DebugAgent, DumpCollector

        dump_dir = debug_conf.get("debug_dump_dir", "agent_ppo/debug/dumps")
        max_episodes = int(debug_conf.get("debug_max_episodes", 1))
        max_frames = int(debug_conf.get("debug_max_frames", 20000))
        summary_interval_frames = int(debug_conf.get("debug_summary_interval_frames", 2000))
        sample_interval = int(debug_conf.get("debug_sample_frame_interval", 200))
        max_dumps = int(debug_conf.get("debug_max_frame_dumps", 80))
        os.makedirs(dump_dir, exist_ok=True)

        debug_agent = DebugAgent()
        collector = DumpCollector(
            dump_dir=dump_dir,
            sample_frame_interval=sample_interval,
            max_frame_dumps=max_dumps,
            logger=self.logger,
        )

        is_train_test = os.environ.get("is_train_test", "False").lower() == "true"
        summary_path = os.path.join(dump_dir, "summary.json")
        progress_summary_path = os.path.join(dump_dir, "summary_in_progress.json")
        try:
            for episode_idx in range(max_episodes):
                lineup = next(self.lineup_iterator)
                usr_conf, _, _ = self.env_conf_manager.update_config(lineup)
                self._inject_debug_summoner_skills(usr_conf)
                if self.logger is not None:
                    self.logger.info(
                        f"[debug] episode {episode_idx + 1}/{max_episodes}, lineup={lineup}, "
                        f"max_frames={max_frames}, summary_interval_frames={summary_interval_frames}"
                    )

                env_obs = self.env.reset(usr_conf=usr_conf)
                if handle_disaster_recovery(env_obs, self.logger):
                    break

                observation = env_obs.get("observation", {})
                env_id = self._extract_env_id(env_obs, observation)
                collector.on_episode_start()
                for camp_idx in range(self.agent_num):
                    obs_i = observation.get(str(camp_idx)) if isinstance(observation, dict) else None
                    collector.on_frame(obs_i, frame_no=0, env_id=env_id)
                collector.write_summary(progress_summary_path)
                debug_agent.reset()

                step_no = 0
                frame_no = 0
                next_summary_frame = max(1, summary_interval_frames)
                while True:
                    actions = []
                    for camp_idx in range(self.agent_num):
                        obs_i = observation.get(str(camp_idx)) if isinstance(observation, dict) else None
                        action = debug_agent.act(obs_i or {}, step_no, agent_id=camp_idx)
                        actions.append(action)

                    env_reward, env_obs = self.env.step(actions)
                    if handle_disaster_recovery(env_obs, self.logger):
                        break

                    frame_no = env_obs.get("frame_no", frame_no)
                    observation = env_obs.get("observation", {})
                    terminated = env_obs.get("terminated", 0)
                    truncated = env_obs.get("truncated", 0)

                    for camp_idx in range(self.agent_num):
                        obs_i = observation.get(str(camp_idx)) if isinstance(observation, dict) else None
                        collector.on_frame(obs_i, frame_no=frame_no, env_id=env_id)

                    step_no += 1
                    if frame_no >= next_summary_frame:
                        collector.write_summary(progress_summary_path)
                        if self.logger is not None:
                            self.logger.info(
                                f"[debug] progress summary written at frame={frame_no}, "
                                f"path={progress_summary_path}"
                            )
                        next_summary_frame += max(1, summary_interval_frames)

                    if terminated or truncated or frame_no >= max_frames or (is_train_test and frame_no >= 1000):
                        break

                collector.write_summary(progress_summary_path)
                if self.logger is not None:
                    self.logger.info(
                        f"[debug] episode {episode_idx + 1} ended at frame={frame_no}, "
                        f"phase_counter={debug_agent.phase_counter}, "
                        f"illegal_button_count={debug_agent.illegal_button_count}"
                    )

            collector.write_summary(summary_path)
            if self.logger is not None:
                self.logger.info(
                    f"[debug] summary written to {summary_path}, "
                    f"frames_observed={collector.frames_observed}, "
                    f"frame_dumps_written={collector.frame_dumps_written}"
                )
        finally:
            try:
                collector.write_summary(progress_summary_path)
            except Exception as exc:
                if self.logger is not None:
                    self.logger.warning(f"[debug] failed to write final progress summary: {exc}")

    def _inject_debug_summoner_skills(self, usr_conf):
        """Inject 闪现 (80115) for both camps. Phase A bypasses agent.init_config so
        we don't perturb agent state; default summoner skill keeps env happy."""
        blue_hero_ids, red_hero_ids = EnvConfManager.extract_hero_ids_from_usr_conf(usr_conf)
        EnvConfManager.inject_select_skills(usr_conf, "blue_camp", {hid: 80115 for hid in blue_hero_ids})
        EnvConfManager.inject_select_skills(usr_conf, "red_camp", {hid: 80115 for hid in red_hero_ids})

    @staticmethod
    def _extract_env_id(env_obs, observation):
        env_id = None
        if isinstance(env_obs, dict):
            env_id = env_obs.get("env_id")
        if env_id is None and isinstance(observation, dict):
            for camp_obs in observation.values():
                if isinstance(camp_obs, dict) and camp_obs.get("env_id"):
                    env_id = camp_obs["env_id"]
                    break
        return env_id or "unknown"
