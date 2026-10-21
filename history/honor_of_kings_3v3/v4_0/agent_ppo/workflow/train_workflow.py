#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""


import os
import time
import random
import numpy as np
from agent_ppo.feature.definition_v1 import (
    build_frame_v1,
    build_command_sparse_reward_v1,
    FrameCollectorV1,
    REWARD_DETAIL_NAMES,
    accumulate_total_reward_v1,
    lineup_iterator_v1,
    late_monster_reward_offsets_v1,
    split_reward_payload_v1,
)
from agent_ppo.agent import NONE_ACTION, NONE_COL_ACTION
from agent_ppo.conf.conf import FrozenV1Config, GameConfig, V1TrainingConfig
from agent_ppo.feature.feature_process import v1_utils as u
from agent_ppo.workflow.opponent_scheduler_v1 import OpponentMixSchedulerV1
from tools.env_conf_manager import EnvConfManager
from tools.model_pool_utils import get_valid_model_pool
from tools.metrics_utils import get_training_metrics
from common_python.utils.workflow_disaster_recovery import handle_disaster_recovery


REWARD_COMPONENT_NAMES = REWARD_DETAIL_NAMES + ("final",)
ATK_MONSTER_REWARD_INDEX = REWARD_DETAIL_NAMES.index("atk_monster")


def workflow(envs, agents, logger=None, monitor=None, *args, **kwargs):
    # Whether the agent is training, corresponding to do_learns
    # 智能体是否进行训练
    do_learns = [True, True]
    last_save_model_time = time.time()

    # Create environment configuration manager instance
    # 创建对局配置管理器实例
    env_conf_manager = EnvConfManager(
        config_path="agent_ppo/conf/train_env_conf.toml",
        logger=logger,
    )

    # Create EpisodeRunner instance
    # 创建 EpisodeRunner 实例
    episode_runner = EpisodeRunner(
        env=envs[0],
        agents=agents,
        logger=logger,
        monitor=monitor,
        env_conf_manager=env_conf_manager,
    )

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
    def __init__(self, env, agents, logger, monitor, env_conf_manager):
        self.env = env
        self.agents = agents
        self.agent_num = len(agents)
        self.logger = logger
        self.monitor = monitor
        self.env_conf_manager = env_conf_manager
        self.lineup_iterator = lineup_iterator_v1(
            joint_lineup_block=V1TrainingConfig.OFFICIAL_LINEUP_BLOCK
        )
        self.frame_collector = FrameCollectorV1(
            self.agent_num,
            gamma=V1TrainingConfig.GAMMA,
            lamda=V1TrainingConfig.LAMBDA,
        )
        self.episode_cnt = 0
        self.last_monitor_report_time = 0.0
        self.latest_platform_metrics = {}
        self.last_routine_log_time = 0.0
        now = time.time()
        system_random = random.SystemRandom()
        self.next_platform_metrics_fetch_time = now + system_random.uniform(
            0.0,
            GameConfig.PLATFORM_METRICS_FETCH_INTERVAL_SECONDS,
        )
        self.next_platform_metrics_fetch_episode = system_random.randrange(
            GameConfig.PLATFORM_METRICS_FETCH_EPISODE_INTERVAL
        )
        self.opponent_scheduler = self._build_opponent_scheduler()
        self.current_episode_opponent = None

    def _build_opponent_scheduler(self):
        if not V1TrainingConfig.OPPONENT_MIX_ENABLED:
            return None
        model_pool = list(get_valid_model_pool(self.logger) or [])
        model_index = int(V1TrainingConfig.OPPONENT_ANCHOR_MODEL_POOL_INDEX)
        if model_index < 0 or model_index >= len(model_pool):
            raise RuntimeError(
                "9:1 opponent mix requires a fixed anchor model ID; set "
                'kaiwu.json to {"model_pool": [YOUR_STRONGEST_MODEL_ID]}'
            )
        anchor_model_id = int(model_pool[model_index])
        return OpponentMixSchedulerV1(
            anchor_model_id=anchor_model_id,
            selfplay_games_per_block=(
                V1TrainingConfig.OPPONENT_SELFPLAY_GAMES_PER_BLOCK
            ),
            anchor_games_per_block=(
                V1TrainingConfig.OPPONENT_ANCHOR_GAMES_PER_BLOCK
            ),
            seed=V1TrainingConfig.OPPONENT_MIX_SEED + os.getpid(),
        )

    def _monitor_report_due(self, is_train_test=False, now=None):
        now = time.time() if now is None else float(now)
        return bool(
            is_train_test
            or now - self.last_monitor_report_time
            >= GameConfig.MONITOR_REPORT_INTERVAL_SECONDS
        )

    def _routine_log(self, message, is_train_test=False, now=None):
        now = time.time() if now is None else float(now)
        if (
            self.logger
            and (
                is_train_test
                or now - self.last_routine_log_time
                >= GameConfig.ROUTINE_LOG_INTERVAL_SECONDS
            )
        ):
            self.logger.info(message)
            self.last_routine_log_time = now

    def _refresh_platform_metrics(self, is_train_test=False, now=None):
        now = time.time() if now is None else float(now)
        if not is_train_test and (
            self.monitor is None
            or now < self.next_platform_metrics_fetch_time
            or self.episode_cnt < self.next_platform_metrics_fetch_episode
        ):
            return
        training_metrics = get_training_metrics(self.logger)
        basic = (training_metrics or {}).get("basic", {})
        if basic:
            self.latest_platform_metrics = {
                f"platform_{name}": float(value)
                for name, value in basic.items()
                if isinstance(value, (int, float))
            }
        self.next_platform_metrics_fetch_time = (
            now + GameConfig.PLATFORM_METRICS_FETCH_INTERVAL_SECONDS
        )
        self.next_platform_metrics_fetch_episode = (
            self.episode_cnt
            + GameConfig.PLATFORM_METRICS_FETCH_EPISODE_INTERVAL
        )

    def run_episodes(self):
        # Single environment process
        # 单局流程
        is_train_test = os.environ.get("is_train_test", "False").lower() == "true"
        while True:
            # Shared platform metrics are diagnostic only. Stagger the
            # time-based fetch across actor processes instead of making all
            # 512 workflows call the RPC at the same episode-count boundary.
            self._refresh_platform_metrics(is_train_test=is_train_test)

            # Update environment configuration
            # Can use a list of length 2 to pass in the lineup id of the current game
            # 更新对局配置, 可以用长度为2的列表传入当前对局的官方智能体id
            offcial_lineup_ids = next(self.lineup_iterator)
            usr_conf, is_eval, monitor_side = self.env_conf_manager.update_config(offcial_lineup_ids)
            episode_opponent = str(
                usr_conf.get("episode", {}).get("opponent_agent", "selfplay")
            )
            if self.opponent_scheduler is not None and is_eval:
                episode_opponent = (
                    self.opponent_scheduler.next_evaluation_opponent()
                )
                usr_conf["episode"]["opponent_agent"] = episode_opponent
            elif self.opponent_scheduler is not None:
                episode_opponent = self.opponent_scheduler.next_training_opponent()
                usr_conf["episode"]["opponent_agent"] = episode_opponent
            self.current_episode_opponent = episode_opponent

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
            self.reset_agents(
                observation,
                opponent_agent=episode_opponent,
                monitor_side=monitor_side,
            )

            # Reset environment frame collector
            # 重置环境帧收集器
            self.frame_collector.reset(num_agents=self.agent_num)

            # Game variables
            # 对局变量
            self.episode_cnt += 1
            frame_no = 0
            # Each camp needs an independent accumulator. List multiplication
            # aliases the same dict and mixes hero IDs/rewards across camps.
            total_reward_dicts = [{} for _ in range(self.agent_num)]
            reward_component_sums = [
                np.zeros(len(REWARD_COMPONENT_NAMES), dtype=np.float64)
                for _ in range(self.agent_num)
            ]
            hero_monster_reward_sums = [
                {
                    hero_id: 0.0
                    for hero_id in V1TrainingConfig.HERO_MONSTER_REWARD_WEIGHTS
                }
                for _ in range(self.agent_num)
            ]

            self._routine_log(
                f"Episode {self.episode_cnt} start, usr_conf is {usr_conf}",
                is_train_test=is_train_test,
            )

            # Reward initialization
            # 回报初始化
            for i, (do_sample, agent) in enumerate(zip(self.do_samples, self.agents)):
                if do_sample:
                    for hero_info in observation[str(i)]["heroes"]:
                        hero_id = accumulate_total_reward_v1(
                            total_reward_dicts[i], hero_info
                        )
                        components = self._reward_components(hero_info)
                        reward_component_sums[i] += components
                        if hero_id in hero_monster_reward_sums[i]:
                            hero_monster_reward_sums[i][hero_id] += components[
                                ATK_MONSTER_REWARD_INDEX
                            ]

            while True:
                # Flush only at a state boundary, before sampling/executing the
                # next action.  Flushing after prediction but after env.step
                # drops that just-executed transition because its action frame
                # belongs to the collector that is immediately reset.
                if len(self.frame_collector) >= GameConfig.SEND_SAMPLE_NUM:
                    for i, (do_sample, agent) in enumerate(
                        zip(self.do_samples, self.agents)
                    ):
                        if not is_eval and do_sample:
                            rewards, auxiliary_rewards, command_reward = (
                                self._controlled_reward_payload(
                                    agent, observation[str(i)]
                                )
                            )
                            (
                                actor_bootstrap,
                                actor_auxiliary_bootstrap,
                                command_bootstrap,
                            ) = agent.bootstrap_values(observation[str(i)])
                            self.frame_collector.save_last_frame(
                                agent_id=i,
                                rewards=rewards,
                                auxiliary_rewards=auxiliary_rewards,
                                command_reward=command_reward,
                                actor_bootstrap_values=actor_bootstrap,
                                actor_auxiliary_bootstrap_values=(
                                    actor_auxiliary_bootstrap
                                ),
                                command_bootstrap_value=command_bootstrap,
                            )
                    list_agents_samples = self.frame_collector.sample_process()
                    self._routine_log(
                        "collector boundary flush, "
                        f"len(frame_collector) is {len(self.frame_collector)}",
                        is_train_test=is_train_test,
                    )
                    yield list_agents_samples
                    self.frame_collector.reset(num_agents=self.agent_num)
                    # No action has been executed and bootstrap_values does not
                    # advance recurrent state, so prediction can safely restart
                    # from the same observation after the yield.
                    continue

                # Initialize the default actions. If the agent does not make a decision, env.step uses the default action.
                # 初始化默认的actions，如果智能体不进行决策，则env.step使用默认action
                hero_actions = []
                collaboration_actions = []
                for i, (do_predict, do_sample, agent) in enumerate(zip(self.do_predicts, self.do_samples, self.agents)):
                    if do_predict:
                        if not is_eval:
                            hero_action, collaboration_action = agent.predict(observation[str(i)])
                        else:
                            hero_action, collaboration_action = agent.exploit(observation[str(i)])
                        hero_actions.append(hero_action)
                        collaboration_actions.append(collaboration_action)

                        # Only sample when do_sample=True and is_eval=False
                        # 评估对局数据不采样，不是训练中最新模型产生的数据不采样
                        if not is_eval and do_sample:
                            rewards, auxiliary_rewards, command_reward = self._controlled_reward_payload(
                                agent, observation[str(i)]
                            )
                            frame = build_frame_v1(
                                agent,
                                observation[str(i)],
                                rewards,
                                auxiliary_rewards,
                                command_reward,
                            )
                            self.frame_collector.save_frame(frame, agent_id=i)
                        elif is_eval and i == monitor_side:
                            # Evaluation does not create samples, but consuming
                            # the same reward boundary keeps custom episode
                            # diagnostics (structure, KDA, Tyrant, inactivity)
                            # comparable with training episodes.
                            self._controlled_reward_payload(
                                agent, observation[str(i)]
                            )
                    else:
                        hero_actions.append(NONE_ACTION)
                        collaboration_actions.append(NONE_COL_ACTION)

                # Step forward
                # 推进环境到下一帧，得到新的状态
                env_reward, env_obs = self.env.step([hero_actions, collaboration_actions])
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
                        for hero_info in observation[str(i)]["heroes"]:
                            hero_id = accumulate_total_reward_v1(
                                total_reward_dicts[i], hero_info
                            )
                            components = self._reward_components(hero_info)
                            reward_component_sums[i] += components
                            if hero_id in hero_monster_reward_sums[i]:
                                hero_monster_reward_sums[i][hero_id] += components[
                                    ATK_MONSTER_REWARD_INDEX
                                ]

                # Normal end or timeout exit, run train_test will exit early
                # 正常结束或超时退出，运行train_test时会提前退出
                is_gameover = terminated or truncated or (
                    is_train_test and frame_no >= 1000
                )
                if is_gameover:
                    # Consume the terminal observation before building monitor
                    # data.  Previously the custom crystal +/-6 was consumed
                    # only after reporting and then cleared by the next reset,
                    # so its dashboard panel was permanently zero even though
                    # the terminal training sample received the reward.
                    terminal_payloads = {}
                    if is_eval:
                        if self.do_predicts[monitor_side]:
                            terminal_payloads[monitor_side] = (
                                self._controlled_reward_payload(
                                    self.agents[monitor_side],
                                    observation[str(monitor_side)],
                                    episode_terminal=True,
                                )
                            )
                    else:
                        for i, (do_sample, agent) in enumerate(
                            zip(self.do_samples, self.agents)
                        ):
                            if do_sample:
                                terminal_payloads[i] = (
                                    self._training_terminal_payload(
                                        agent, observation[str(i)]
                                    )
                                )
                    if is_gameover:
                        self._routine_log(
                            f"episode_{self.episode_cnt} terminated in "
                            f"fno_{frame_no}, truncated:{truncated}, "
                            f"eval:{is_eval}, "
                            f"reward_sum:{total_reward_dicts[monitor_side]}",
                            is_train_test=is_train_test,
                        )

                        total_reward = sum(total_reward_dicts[monitor_side].values()) / len(
                            total_reward_dicts[monitor_side]
                        )
                        report_now = time.time()
                        if self.monitor and self._monitor_report_due(
                            is_train_test=is_train_test, now=report_now
                        ):
                            component_mean = reward_component_sums[monitor_side] / max(
                                len(total_reward_dicts[monitor_side]), 1
                            )
                            monitor_data = {
                                "reward": total_reward,
                                **{
                                    f"reward_{name}": float(value)
                                    for name, value in zip(
                                        REWARD_COMPONENT_NAMES, component_mean
                                    )
                                    if name not in ("tower", "win_crystal")
                                },
                            }
                            monitor_data.update(
                                {
                                    f"reward_atk_monster_{hero_id}": float(value)
                                    for hero_id, value in hero_monster_reward_sums[
                                        monitor_side
                                    ].items()
                                }
                            )
                            monitored_agent = self.agents[monitor_side]
                            for metric_name, attribute in (
                                ("reward_cake_total", "cake_reward_total"),
                                ("reward_cake_progress", "cake_progress_reward_total"),
                                ("reward_cake_pickup", "cake_pickup_reward_total"),
                                ("reward_cake_clawback", "cake_clawback_total"),
                                ("late_jungle_reward_raw", "late_monster_raw_total"),
                                ("late_jungle_reward_offset", "late_monster_offset_total"),
                                ("late_jungle_reward_effective", "late_monster_effective_total"),
                            ):
                                monitor_data[metric_name] = float(
                                    getattr(monitored_agent, attribute, 0.0)
                                )
                            for hero_id in u.HERO_CONFIG_IDS:
                                for metric_prefix, attribute in (
                                    ("cake_reward", "cake_reward_by_config"),
                                    ("cake_progress", "cake_progress_by_config"),
                                    ("cake_pickup", "cake_pickup_by_config"),
                                    ("cake_clawback", "cake_clawback_by_config"),
                                ):
                                    monitor_data[
                                        f"{metric_prefix}_{hero_id}"
                                    ] = float(
                                        getattr(
                                            monitored_agent,
                                            attribute,
                                            {},
                                        ).get(hero_id, 0.0)
                                    )
                            monitor_data["reward_jungle_route_shaping"] = float(
                                getattr(
                                    self.agents[monitor_side],
                                    "jungle_route_shaping_total",
                                    0.0,
                                )
                            )
                            monitor_data["reward_lane_last_hit"] = float(
                                getattr(
                                    self.agents[monitor_side],
                                    "last_hit_reward_total",
                                    0.0,
                                )
                            )
                            monitor_data["lane_last_hit_count"] = float(
                                getattr(
                                    self.agents[monitor_side],
                                    "last_hit_count",
                                    0,
                                )
                            )
                            monitor_data["reward_no_target_penalty"] = float(
                                getattr(
                                    self.agents[monitor_side],
                                    "no_target_penalty_total",
                                    0.0,
                                )
                            )
                            monitor_data["no_target_penalty_count"] = float(
                                getattr(
                                    self.agents[monitor_side],
                                    "no_target_penalty_count",
                                    0,
                                )
                            )
                            monitor_data["no_target_grace_count"] = float(
                                getattr(
                                    self.agents[monitor_side],
                                    "no_target_grace_count",
                                    0,
                                )
                            )
                            monitor_data["strategic_wait_exemption_count"] = float(
                                getattr(
                                    self.agents[monitor_side],
                                    "strategic_wait_exemption_count",
                                    0,
                                )
                            )
                            monitor_data["reward_lane_attendance"] = float(
                                getattr(
                                    self.agents[monitor_side],
                                    "lane_attendance_shaping_total",
                                    0.0,
                                )
                            )
                            monitor_data["lane_attendance_shaping_count"] = float(
                                getattr(
                                    self.agents[monitor_side],
                                    "lane_attendance_shaping_count",
                                    0,
                                )
                            )
                            monitor_data["reward_custom_kda"] = float(
                                getattr(
                                    self.agents[monitor_side],
                                    "custom_kda_reward_total",
                                    0.0,
                                )
                            )
                            monitor_data["reward_tyrant"] = float(
                                getattr(
                                    self.agents[monitor_side],
                                    "tyrant_reward_total",
                                    0.0,
                                )
                            )
                            monitor_data["reward_kai_wolf"] = float(
                                getattr(
                                    self.agents[monitor_side],
                                    "kai_wolf_reward_total",
                                    0.0,
                                )
                            )
                            monitor_data["kai_wolf_reward_count"] = float(
                                getattr(
                                    self.agents[monitor_side],
                                    "kai_wolf_reward_count",
                                    0,
                                )
                            )
                            monitor_data["reward_treasure_thief"] = float(
                                getattr(
                                    self.agents[monitor_side],
                                    "treasure_thief_reward_total",
                                    0.0,
                                )
                            )
                            monitor_data["treasure_thief_reward_count"] = float(
                                getattr(
                                    self.agents[monitor_side],
                                    "treasure_thief_reward_count",
                                    0,
                                )
                            )
                            monitor_data["reward_speed_buff"] = float(
                                getattr(
                                    self.agents[monitor_side],
                                    "speed_buff_reward_total",
                                    0.0,
                                )
                            )
                            monitor_data["speed_buff_reward_count"] = float(
                                getattr(
                                    self.agents[monitor_side],
                                    "speed_buff_reward_count",
                                    0,
                                )
                            )
                            monitor_data["reward_structure_nonlinear"] = float(
                                getattr(
                                    self.agents[monitor_side],
                                    "structure_reward_total",
                                    0.0,
                                )
                            )
                            monitor_data["reward_post_kill_push"] = float(
                                getattr(
                                    self.agents[monitor_side],
                                    "post_kill_push_reward_total",
                                    0.0,
                                )
                            )
                            monitor_data[
                                "reward_terminal_crystal_custom"
                            ] = float(
                                getattr(
                                    self.agents[monitor_side],
                                    "terminal_crystal_reward_total",
                                    0.0,
                                )
                            )
                            monitor_data["reward_crystal_danger"] = float(
                                getattr(
                                    self.agents[monitor_side],
                                    "crystal_danger_reward_total",
                                    0.0,
                                )
                            )
                            monitor_data["reward_defense_pressure"] = float(
                                getattr(
                                    self.agents[monitor_side],
                                    "defense_pressure_reward_total",
                                    0.0,
                                )
                            )
                            monitor_data["reward_temporal_focus"] = float(
                                getattr(
                                    self.agents[monitor_side],
                                    "temporal_focus_reward_total",
                                    0.0,
                                )
                            )
                            monitor_data[
                                "reward_teammate_abandonment"
                            ] = float(
                                getattr(
                                    self.agents[monitor_side],
                                    "teammate_abandonment_penalty_total",
                                    0.0,
                                )
                            )
                            monitor_data["reward_teammate_support"] = float(
                                getattr(
                                    self.agents[monitor_side],
                                    "teammate_support_reward_total",
                                    0.0,
                                )
                            )
                            monitor_data["reward_group_attack"] = float(
                                getattr(
                                    self.agents[monitor_side],
                                    "group_attack_reward_total",
                                    0.0,
                                )
                            )
                            for metric_name in (
                                "teammate_abandonment_penalty_count",
                                "teammate_support_count",
                                "temporal_focus_count",
                                "defense_pressure_count",
                                "group_attack_count",
                                "command_teammate_abandonment_penalty_count",
                                "command_teammate_support_count",
                            ):
                                monitor_name = metric_name.replace(
                                    "_penalty", ""
                                )
                                monitor_data[monitor_name] = float(
                                    getattr(
                                        self.agents[monitor_side],
                                        metric_name,
                                        0,
                                    )
                                )
                            monitor_data["reward_long_inactivity"] = float(
                                getattr(
                                    self.agents[monitor_side],
                                    "long_inactivity_penalty_total",
                                    0.0,
                                )
                            )
                            for metric_name in (
                                "long_inactivity_penalty_count",
                                "long_inactivity_stationary_count",
                                "long_inactivity_wandering_count",
                                "long_inactivity_no_damage_count",
                                "long_inactivity_tyrant_exemption_count",
                            ):
                                monitor_data[metric_name] = float(
                                    getattr(
                                        self.agents[monitor_side],
                                        metric_name,
                                        0,
                                    )
                                )
                            monitor_data["reward_command_sparse"] = float(
                                getattr(
                                    self.agents[monitor_side],
                                    "command_sparse_reward_total",
                                    0.0,
                                )
                            )
                            monitor_data["reward_command_response"] = float(
                                getattr(
                                    self.agents[monitor_side],
                                    "command_response_reward_total",
                                    0.0,
                                )
                            )
                            monitor_data["reward_command_nonresponse"] = float(
                                getattr(
                                    self.agents[monitor_side],
                                    "command_nonresponse_penalty_total",
                                    0.0,
                                )
                            )
                            monitor_data["reward_command_idle"] = float(
                                getattr(
                                    self.agents[monitor_side],
                                    "command_no_command_idle_penalty_total",
                                    0.0,
                                )
                            )
                            monitor_data["reward_command_farm_camp"] = float(
                                getattr(
                                    self.agents[monitor_side],
                                    "command_farm_camp_reward_total",
                                    0.0,
                                )
                            )
                            monitor_data["reward_tower_risk"] = float(
                                getattr(
                                    self.agents[monitor_side],
                                    "tower_risk_penalty_total",
                                    0.0,
                                )
                            )
                            monitor_data["tower_risk_hit_count"] = float(
                                getattr(
                                    self.agents[monitor_side],
                                    "tower_risk_hit_count",
                                    0,
                                )
                            )
                            monitor_data["reward_defense_wave"] = float(
                                getattr(
                                    self.agents[monitor_side],
                                    "defense_wave_reward_total",
                                    0.0,
                                )
                            )
                            monitor_data["defense_wave_completion_count"] = float(
                                getattr(
                                    self.agents[monitor_side],
                                    "defense_wave_completion_count",
                                    0,
                                )
                            )
                            monitor_data["structure_positive_reward"] = float(
                                getattr(
                                    self.agents[monitor_side],
                                    "structure_positive_reward_total",
                                    0.0,
                                )
                            )
                            monitor_data["structure_negative_reward"] = float(
                                getattr(
                                    self.agents[monitor_side],
                                    "structure_negative_reward_total",
                                    0.0,
                                )
                            )
                            structure_events = int(
                                getattr(
                                    self.agents[monitor_side],
                                    "structure_attribution_event_count",
                                    0,
                                )
                            )
                            monitor_data["structure_attribution_coverage"] = float(
                                getattr(
                                    self.agents[monitor_side],
                                    "structure_attribution_coverage_total",
                                    0.0,
                                )
                                / max(structure_events, 1)
                            )
                            monitor_data["structure_attribution_fallback_count"] = float(
                                getattr(
                                    self.agents[monitor_side],
                                    "structure_attribution_fallback_count",
                                    0,
                                )
                            )
                            monitor_data["structure_attributed_hit_count"] = float(
                                getattr(
                                    self.agents[monitor_side],
                                    "structure_attributed_hit_count",
                                    0,
                                )
                            )
                            actor_structure_rewards = getattr(
                                self.agents[monitor_side],
                                "structure_actor_reward_by_config",
                                {},
                            )
                            actor_structure_hits = getattr(
                                self.agents[monitor_side],
                                "structure_actor_hit_by_config",
                                {},
                            )
                            for config_id in (132, 141, 193):
                                monitor_data[
                                    f"structure_actor_reward_{config_id}"
                                ] = float(
                                    actor_structure_rewards.get(
                                        config_id, 0.0
                                    )
                                )
                                monitor_data[
                                    f"structure_actor_hit_{config_id}"
                                ] = float(
                                    actor_structure_hits.get(
                                        config_id, 0
                                    )
                                )
                            monitor_data["command_nonresponse_count"] = float(
                                getattr(
                                    self.agents[monitor_side],
                                    "command_nonresponse_penalty_count",
                                    0,
                                )
                            )
                            monitor_data["command_idle_penalty_count"] = float(
                                getattr(
                                    self.agents[monitor_side],
                                    "command_no_command_idle_penalty_count",
                                    0,
                                )
                            )
                            monitored_agent = self.agents[monitor_side]
                            command_attempts = int(
                                getattr(
                                    monitored_agent,
                                    "command_legalizer_attempt_count",
                                    0,
                                )
                            )
                            invalid_commands = int(
                                getattr(monitored_agent, "invalid_command_count", 0)
                            )
                            monitor_data["command_legalizer_invalid_count"] = float(
                                invalid_commands
                            )
                            monitor_data["command_legalizer_valid_rate"] = float(
                                1.0 - invalid_commands / max(command_attempts, 1)
                            )
                            platform_train_step = self.latest_platform_metrics.get(
                                "platform_train_global_step"
                            )
                            checkpoint_identity = getattr(
                                monitored_agent,
                                "loaded_checkpoint_identity",
                                None,
                            )
                            if (
                                platform_train_step is not None
                                and isinstance(checkpoint_identity, tuple)
                                and len(checkpoint_identity) >= 2
                            ):
                                monitor_data["actor_policy_lag_updates"] = float(
                                    max(
                                        float(platform_train_step)
                                        - float(checkpoint_identity[1]),
                                        0.0,
                                    )
                                )
                            lane_opening = getattr(
                                self.agents[monitor_side],
                                "lane_opening_legalizer",
                                None,
                            )
                            monitor_data["lane_opening_mask_count"] = float(
                                getattr(lane_opening, "mask_application_count", 0)
                            )
                            monitor_data["lane_first_wave_attack_mask_count"] = float(
                                getattr(lane_opening, "attack_application_count", 0)
                            )
                            monitor_data["lane_opening_release_count"] = float(
                                getattr(lane_opening, "released_count", 0)
                            )
                            jungle_opening = getattr(
                                self.agents[monitor_side],
                                "jungle_opening_legalizer",
                                None,
                            )
                            monitor_data["jungle_opening_mask_count"] = float(
                                getattr(jungle_opening, "mask_application_count", 0)
                            )
                            monitor_data["jungle_opening_attack_mask_count"] = float(
                                getattr(jungle_opening, "attack_application_count", 0)
                            )
                            monitor_data["jungle_camp_exit_mask_count"] = float(
                                getattr(jungle_opening, "exit_application_count", 0)
                            )
                            monitor_data["jungle_opening_interrupt_count"] = float(
                                getattr(jungle_opening, "interrupt_count", 0)
                            )
                            feature_process = getattr(
                                self.agents[monitor_side], "feature_processes", None
                            )
                            if feature_process is not None:
                                monitor_data.update(
                                    feature_process.lane_monitor_metrics()
                                )
                                cake_counters = feature_process.cake_counters
                                for metric_name, counter_name in (
                                    ("cake_opportunity_count", "opportunity_start"),
                                    ("cake_opportunity_assigned_count", "opportunity_assigned"),
                                    ("cake_opportunity_conflict_count", "opportunity_conflict"),
                                    ("cake_progress_positive_total", "progress_positive_total"),
                                    ("cake_progress_cap_count", "progress_cap"),
                                    ("cake_progress_clawback_total", "progress_clawback_total"),
                                    ("cake_progress_clawback_count", "progress_clawback_count"),
                                    ("cake_opportunity_reopen_blocked_count", "opportunity_reopen_blocked"),
                                    ("cake_pickup_success_count", "pickup_success"),
                                    ("cake_pickup_ambiguous_count", "pickup_ambiguous"),
                                    ("cake_pickup_official_count", "pickup_official"),
                                    ("cake_pickup_high_hp_zero_count", "pickup_high_hp_zero"),
                                    ("cake_pickup_reward_total", "pickup_reward_total"),
                                    ("cake_abort_hard_count", "hard_objective_cancel"),
                                    ("cake_abort_external_count", "external_cancel"),
                                    ("cake_abort_controlled_count", "controlled_abort"),
                                    ("cake_abort_timeout_count", "timeout_abort"),
                                    ("cake_low_hp_death_count", "low_hp_death_after_opportunity"),
                                ):
                                    monitor_data[metric_name] = float(
                                        cake_counters.get(counter_name, 0.0)
                                    )
                                v40_counters = feature_process.v40_counters
                                for metric_name, counter_name in (
                                    ("kai_respawn_count", "kai_respawn"),
                                    ("kai_respawn_reentry_started_count", "kai_respawn_reentry_started"),
                                    ("kai_respawn_reentry_success_count", "kai_respawn_reentry_success"),
                                    ("kai_respawn_reentry_abort_count", "kai_respawn_reentry_abort"),
                                    ("kai_respawn_empty_camp_visit_count", "kai_respawn_empty_camp_visit"),
                                    ("kai_respawn_lane_diversion_count", "kai_respawn_lane_diversion"),
                                    ("kai_respawn_replan_count", "kai_respawn_replan"),
                                    ("kai_respawn_reentry_timeout_count", "kai_respawn_reentry_timeout"),
                                    ("kai_respawn_to_jungle_frames", "kai_respawn_to_jungle_frames"),
                                    ("kai_respawn_to_jungle_observed_count", "kai_respawn_to_jungle_observed"),
                                    ("kai_respawn_to_jungle_missing_count", "kai_respawn_to_jungle_missing"),
                                    ("kai_respawn_to_monster_frames", "kai_respawn_to_first_monster_engagement_frames"),
                                    ("kai_respawn_to_monster_observed_count", "kai_respawn_to_first_monster_engagement_observed"),
                                    ("kai_respawn_to_monster_missing_count", "kai_respawn_to_first_monster_engagement_missing"),
                                    ("kai_respawn_empty_wait_frames", "kai_respawn_empty_wait_frames"),
                                    ("kai_respawn_empty_wait_count", "kai_respawn_empty_wait_count"),
                                    ("kai_respawn_empty_wait_success_frames", "kai_respawn_empty_wait_success_frames"),
                                    ("kai_respawn_empty_wait_success_count", "kai_respawn_empty_wait_success_count"),
                                    ("kai_respawn_empty_wait_timeout_frames", "kai_respawn_empty_wait_timeout_frames"),
                                    ("kai_respawn_empty_wait_timeout_count", "kai_respawn_empty_wait_timeout_count"),
                                    ("kai_respawn_empty_wait_cancel_frames", "kai_respawn_empty_wait_cancel_frames"),
                                    ("kai_respawn_empty_wait_cancel_count", "kai_respawn_empty_wait_cancel_count"),
                                    ("kai_respawn_controlled_started_count", "kai_respawn_controlled_started"),
                                    ("kai_respawn_controlled_success_count", "kai_respawn_controlled_success"),
                                    ("kai_respawn_controlled_timeout_count", "kai_respawn_controlled_timeout"),
                                    ("kai_respawn_controlled_cancel_count", "kai_respawn_controlled_cancel"),
                                    ("kai_respawn_controlled_to_jungle_frames", "kai_respawn_controlled_to_jungle_frames"),
                                    ("kai_ctl_to_jungle_observed_count", "kai_respawn_controlled_to_jungle_observed"),
                                    ("kai_ctl_to_jungle_missing_count", "kai_respawn_controlled_to_jungle_missing"),
                                    ("kai_ctl_to_monster_frames", "kai_respawn_controlled_to_first_monster_engagement_frames"),
                                    ("kai_ctl_to_monster_observed_count", "kai_respawn_controlled_to_first_monster_engagement_observed"),
                                    ("kai_ctl_to_monster_missing_count", "kai_respawn_controlled_to_first_monster_engagement_missing"),
                                    ("kai_respawn_controlled_empty_wait_frames", "kai_respawn_controlled_empty_wait_frames"),
                                    ("kai_respawn_controlled_empty_wait_count", "kai_respawn_controlled_empty_wait_count"),
                                    ("kai_ctl_empty_wait_success_frames", "kai_respawn_controlled_empty_wait_success_frames"),
                                    ("kai_ctl_empty_wait_success_count", "kai_respawn_controlled_empty_wait_success_count"),
                                    ("kai_ctl_empty_wait_timeout_frames", "kai_respawn_controlled_empty_wait_timeout_frames"),
                                    ("kai_ctl_empty_wait_timeout_count", "kai_respawn_controlled_empty_wait_timeout_count"),
                                    ("kai_ctl_empty_wait_cancel_frames", "kai_respawn_controlled_empty_wait_cancel_frames"),
                                    ("kai_ctl_empty_wait_cancel_count", "kai_respawn_controlled_empty_wait_cancel_count"),
                                    ("kai_respawn_official_started_count", "kai_respawn_official_started"),
                                    ("kai_respawn_official_success_count", "kai_respawn_official_success"),
                                    ("kai_respawn_official_timeout_count", "kai_respawn_official_timeout"),
                                    ("kai_respawn_official_cancel_count", "kai_respawn_official_cancel"),
                                    ("kai_respawn_official_to_jungle_frames", "kai_respawn_official_to_jungle_frames"),
                                    ("kai_off_to_jungle_observed_count", "kai_respawn_official_to_jungle_observed"),
                                    ("kai_off_to_jungle_missing_count", "kai_respawn_official_to_jungle_missing"),
                                    ("kai_off_to_monster_frames", "kai_respawn_official_to_first_monster_engagement_frames"),
                                    ("kai_off_to_monster_observed_count", "kai_respawn_official_to_first_monster_engagement_observed"),
                                    ("kai_off_to_monster_missing_count", "kai_respawn_official_to_first_monster_engagement_missing"),
                                    ("kai_respawn_official_empty_wait_frames", "kai_respawn_official_empty_wait_frames"),
                                    ("kai_respawn_official_empty_wait_count", "kai_respawn_official_empty_wait_count"),
                                    ("kai_off_empty_wait_success_frames", "kai_respawn_official_empty_wait_success_frames"),
                                    ("kai_off_empty_wait_success_count", "kai_respawn_official_empty_wait_success_count"),
                                    ("kai_off_empty_wait_timeout_frames", "kai_respawn_official_empty_wait_timeout_frames"),
                                    ("kai_off_empty_wait_timeout_count", "kai_respawn_official_empty_wait_timeout_count"),
                                    ("kai_off_empty_wait_cancel_frames", "kai_respawn_official_empty_wait_cancel_frames"),
                                    ("kai_off_empty_wait_cancel_count", "kai_respawn_official_empty_wait_cancel_count"),
                                    ("official_respawn_command_count", "official_respawn_command"),
                                    ("official_respawn_engagement_count", "official_respawn_engagement"),
                                    ("timer_route_candidate_count", "timer_route_candidate"),
                                    ("timer_route_follow_count", "timer_route_follow"),
                                    ("timer_route_progress_positive_total", "timer_route_progress_positive_total"),
                                    ("timer_route_progress_negative_total", "timer_route_progress_negative_total"),
                                    ("timer_route_phi_upper_clamp_count", "timer_route_phi_upper_clamp"),
                                    ("timer_route_phi_lower_clamp_count", "timer_route_phi_lower_clamp"),
                                    ("timer_route_terminal_settlement_count", "timer_route_terminal_settlement_count"),
                                    ("timer_route_terminal_settlement_total", "timer_route_terminal_settlement_total"),
                                    ("timer_route_no_target_exempt_frames", "timer_route_no_target_exempt_frames"),
                                    ("timer_route_inactivity_exempt_frames", "timer_route_inactivity_exempt_frames"),
                                    ("timer_route_stall_count", "timer_route_stall"),
                                    ("timer_route_real_engagement_count", "timer_route_real_engagement"),
                                    ("timer_route_clear_count", "timer_route_clear"),
                                    ("timer_route_abandon_count", "timer_route_abandon"),
                                    ("official_timer_farm_command_count", "official_timer_farm_command"),
                                    ("official_timer_farm_engagement_count", "official_timer_farm_engagement"),
                                    ("official_timer_farm_completion_count", "official_timer_farm_completion"),
                                    ("navigation_generation_advance_count", "navigation_generation_advance"),
                                    ("navigation_stale_lock_block_count", "navigation_stale_lock_block"),
                                    ("respawn_conversion_preempt_count", "respawn_conversion_preempt"),
                                    ("late_conversion_preempt_count", "late_conversion_preempt"),
                                    ("tyrant_three_stack_first_frame", "tyrant_three_stack_first_frame"),
                                    ("post_three_stack_tyrant_count", "post_three_stack_tyrant"),
                                    ("late_jungle_attack_frames", "late_jungle_attack_frames"),
                                    ("late_farm_task_start_count", "late_farm_task_start"),
                                    ("late_farm_task_cancel_count", "late_farm_task_cancel_conversion"),
                                    ("late_conversion_opportunity_count", "late_conversion_opportunity"),
                                    ("late_conversion_follow_count", "late_conversion_follow"),
                                    ("late_wave_escort_count", "late_wave_escort"),
                                    ("late_group_count", "late_group"),
                                    ("post_kill_conversion_count", "post_kill_structure_conversion"),
                                    ("timeout_pressure_push_count", "timeout_pressure_push"),
                                ):
                                    monitor_data[metric_name] = float(
                                        v40_counters.get(counter_name, 0.0)
                                    )
                                lane_counters = feature_process.lane_promotion_counters
                                for metric_name, counter_name in (
                                    ("lane_position_candidate_count", "position_candidate"),
                                    ("lane_stable_target_count", "stable_target"),
                                    ("lane_promotion_sent_count", "promotion_sent"),
                                    ("lane_promotion_confirmed_count", "promotion_confirmed"),
                                    ("lane_cooldown_wait_frames", "cooldown_wait_frames"),
                                    ("lane_target_change_cancel_count", "target_change_cancel"),
                                    ("lane_target_invalid_cancel_count", "target_invalid_cancel"),
                                    ("lane_failure_lock_count", "failure_lock"),
                                    ("lane_target_active_frames", "target_active_frames"),
                                    ("lane_refresh_suppressed_count", "refresh_suppressed"),
                                    ("lane_retreat_handoff_count", "retreat_handoff"),
                                    ("lane_retreat_protection_count", "retreat_protection"),
                                    ("lane_low_hp_protection_count", "low_hp_protection"),
                                    ("lane_outnumbered_protection_count", "outnumbered_protection"),
                                    ("lane_geometry_protection_count", "geometry_protection"),
                                    ("lane_defense_protection_count", "defense_protection"),
                                    ("lane_forced_route_protection_count", "forced_route_protection"),
                                ):
                                    monitor_data[metric_name] = float(
                                        lane_counters.get(counter_name, 0.0)
                                    )
                                monitor_data[
                                    "lane_retreat_protection_count"
                                ] = float(
                                    lane_counters.get(
                                        "retreat_protection", 0.0
                                    )
                                    + lane_counters.get(
                                        "low_hp_protection", 0.0
                                    )
                                    + lane_counters.get(
                                        "outnumbered_protection", 0.0
                                    )
                                    + lane_counters.get(
                                        "geometry_protection", 0.0
                                    )
                                    + lane_counters.get(
                                        "defense_protection", 0.0
                                    )
                                )
                                monitor_data["lane_promotion_confirm_rate"] = float(
                                    lane_counters.get("promotion_confirmed", 0.0)
                                    / max(lane_counters.get("promotion_sent", 0.0), 1.0)
                                )
                                for camp_id in (1, 2):
                                    for hero_id in (132, 141):
                                        for target_kind in (
                                            "hero",
                                            "soldier",
                                            "structure",
                                        ):
                                            for counter_name, short_name in (
                                                ("position_candidate", "cand"),
                                                ("stable_target", "stable"),
                                                ("promotion_sent", "sent"),
                                                ("promotion_confirmed", "confirm"),
                                            ):
                                                monitor_data[
                                                    f"lane_{short_name}_c{camp_id}_h{hero_id}_{target_kind}"
                                                ] = float(
                                                    lane_counters.get(
                                                        f"{counter_name}_camp{camp_id}_config{hero_id}_{target_kind}",
                                                        0.0,
                                                    )
                                                )
                                monitor_data["episode_timeout_count"] = float(
                                    bool(truncated)
                                )
                                own_crystal = feature_process._memo_context["organs"][2]
                                enemy_crystal = feature_process._memo_context["organs"][5]
                                own_actor = own_crystal["actor"] or {}
                                enemy_actor = enemy_crystal["actor"] or {}
                                own_ratio = u.as_float(
                                    u.actor_get(own_actor, "hp", default=0.0),
                                    0.0,
                                ) / max(
                                    1.0,
                                    u.as_float(
                                        u.actor_get(
                                            own_actor,
                                            "max_hp",
                                            "maxHp",
                                            default=1.0,
                                        ),
                                        1.0,
                                    ),
                                )
                                enemy_ratio = u.as_float(
                                    u.actor_get(enemy_actor, "hp", default=0.0),
                                    0.0,
                                ) / max(
                                    1.0,
                                    u.as_float(
                                        u.actor_get(
                                            enemy_actor,
                                            "max_hp",
                                            "maxHp",
                                            default=1.0,
                                        ),
                                        1.0,
                                    ),
                                )
                                monitor_data["episode_timeout_advantage_count"] = float(
                                    bool(truncated and own_ratio > enemy_ratio)
                                )
                                task_counters = (
                                    feature_process.official_task_counters
                                )
                                for metric_name, counter_name in (
                                    ("official_task_start_count", "farm_camp_start"),
                                    ("official_task_navigation_count", "navigation_response"),
                                    ("official_task_arrival_count", "camp_arrival"),
                                    ("official_task_engagement_count", "engagement_response"),
                                    ("official_task_navigation_latency_frames", "last_navigation_latency_frames"),
                                    ("official_task_engagement_latency_frames", "last_arrival_to_engagement_frames"),
                                    ("official_task_completion_count", "task_completion"),
                                    ("official_task_abort_count", "task_abort"),
                                    ("official_task_protocol_rejected_count", "task_end_protocol_rejected"),
                                    ("official_task_navigation_failure_count", "task_end_navigation_nonresponse"),
                                    ("official_task_engagement_failure_count", "task_end_engagement_nonresponse"),
                                    ("official_task_persistence_failure_count", "task_end_persistence_failure"),
                                    ("official_task_lifecycle_lock_count", "failed_lifecycle_lock"),
                                    ("official_deer_command_count", "official_deer_command"),
                                    ("official_deer_renewal_count", "official_deer_renewal"),
                                    ("official_deer_successor_count", "official_deer_successor"),
                                ):
                                    monitor_data[metric_name] = float(
                                        task_counters.get(counter_name, 0.0)
                                    )
                                monitor_data[
                                    "official_task_recovery_count"
                                ] = float(
                                    task_counters.get(
                                        "navigation_recovery", 0.0
                                    )
                                    + task_counters.get(
                                        "engagement_recovery", 0.0
                                    )
                                    + task_counters.get(
                                        "persistence_recovery", 0.0
                                    )
                                )
                                tower_counters = (
                                    feature_process.tower_risk_counters
                                )
                                monitor_data[
                                    "tower_risk_warning_frames"
                                ] = float(
                                    tower_counters.get(
                                        "phase_warning_frames", 0.0
                                    )
                                )
                                monitor_data[
                                    "tower_risk_danger_frames"
                                ] = float(
                                    tower_counters.get(
                                        "phase_danger_frames", 0.0
                                    )
                                )
                                monitor_data[
                                    "tower_risk_critical_frames"
                                ] = float(
                                    tower_counters.get(
                                        "phase_critical_frames", 0.0
                                    )
                                )
                                monitor_data[
                                    "tower_risk_reset_count"
                                ] = float(
                                    sum(
                                        value
                                        for name, value in tower_counters.items()
                                        if name.startswith("reset_")
                                    )
                                )
                                monitor_data[
                                    "tower_risk_warning_count"
                                ] = float(
                                    tower_counters.get(
                                        "event_start", 0.0
                                    )
                                )
                                monitor_data[
                                    "tower_risk_danger_count"
                                ] = float(
                                    tower_counters.get(
                                        "enter_danger", 0.0
                                    )
                                )
                                monitor_data[
                                    "tower_risk_critical_count"
                                ] = float(
                                    tower_counters.get(
                                        "enter_critical", 0.0
                                    )
                                )
                                defense_counters = (
                                    feature_process.defense_wave_counters
                                )
                                for metric_name, counter_name in (
                                    ("defense_wave_under_count", "window_under"),
                                    ("defense_wave_front_count", "window_front"),
                                    ("defense_wave_response_count", "first_response"),
                                    ("defense_wave_progress_total", "progress_reward"),
                                    ("defense_wave_ignore_candidate_count", "ignore_candidate"),
                                    ("defense_wave_ignore_penalty_count", "ignore_penalty"),
                                    ("defense_wave_ignore_exemption_count", "ignore_exemption"),
                                    ("defense_wave_outer_count", "window_structure_outer"),
                                    ("defense_wave_high_count", "window_structure_high"),
                                    ("defense_wave_crystal_count", "window_structure_crystal"),
                                    ("defense_wave_missing_count", "soldier_missing"),
                                    ("defense_wave_missing_peak_frames", "missing_peak_frames"),
                                    ("defense_wave_invalidated_count", "window_invalidated"),
                                    ("defense_wave_runtime_reuse_count", "window_invalidated_runtime_reuse"),
                                    ("defense_wave_teammate_progress_count", "other_progress_suppressed"),
                                ):
                                    monitor_data[metric_name] = float(
                                        defense_counters.get(
                                            counter_name, 0.0
                                        )
                                    )
                            collision = getattr(
                                self.agents[monitor_side], "collision_avoidance", None
                            )
                            monitor_data["collision_escape_count"] = float(
                                getattr(collision, "escape_count", 0)
                            )
                            monitor_data["collision_blocked_direction_mean"] = float(
                                getattr(collision, "masked_direction_total", 0)
                                / max(
                                    getattr(collision, "mask_application_count", 0),
                                    1,
                                )
                            )
                            flash = getattr(
                                self.agents[monitor_side], "flash_legalizer", None
                            )
                            smite = getattr(
                                self.agents[monitor_side],
                                "tyrant_smite_legalizer",
                                None,
                            )
                            for metric_name, attribute in (
                                ("tyrant_smite_lock_count", "lock_count"),
                                ("tyrant_smite_open_count", "open_count"),
                                ("tyrant_smite_force_count", "force_count"),
                                ("tyrant_smite_pending_count", "force_pending_count"),
                            ):
                                monitor_data[metric_name] = float(
                                    getattr(smite, attribute, 0)
                                )
                            for metric_name, attribute in (
                                ("flash_official_legal_count", "official_legal_count"),
                                ("flash_gate_open_count", "gate_open_count"),
                                ("flash_blocked_count", "blocked_count"),
                                ("flash_no_safe_landing_count", "no_safe_landing_count"),
                                ("flash_trigger_low_hp_count", "low_hp_count"),
                                ("flash_trigger_tower_count", "tower_count"),
                                ("flash_trigger_chase_count", "chase_count"),
                                (
                                    "flash_trigger_marco_ultimate_count",
                                    "marco_ultimate_count",
                                ),
                            ):
                                monitor_data[metric_name] = float(
                                    getattr(flash, attribute, 0)
                                )
                            safety = getattr(
                                self.agents[monitor_side],
                                "safety_legalizer",
                                None,
                            )
                            for metric_name, attribute in (
                                ("spawn_boundary_move_block_count", "spawn_move_block_count"),
                                ("spawn_boundary_target_block_count", "spawn_target_block_count"),
                                ("spawn_boundary_displacement_block_count", "spawn_displacement_block_count"),
                                ("spawn_boundary_recovery_count", "spawn_recovery_count"),
                                ("tower_critical_move_block_count", "tower_critical_move_block_count"),
                                ("tower_critical_displacement_block_count", "tower_critical_displacement_block_count"),
                                ("safety_all_illegal_recovery_count", "all_illegal_recovery_count"),
                            ):
                                monitor_data[metric_name] = float(
                                    getattr(safety, attribute, 0)
                                )
                            monitor_data.update(self.latest_platform_metrics)
                            if self.opponent_scheduler is not None:
                                monitor_data.update(
                                    self.opponent_scheduler.monitor_metrics()
                                )
                            monitor_data["opponent_current_is_anchor"] = float(
                                not is_eval
                                and self.current_episode_opponent not in (
                                    None,
                                    "selfplay",
                                )
                            )
                            monitor_data["eval_current_is_anchor"] = float(
                                is_eval
                                and self.current_episode_opponent not in (
                                    None,
                                    "common_ai",
                                    "selfplay",
                                )
                            )
                            self.monitor.put_data({os.getpid(): monitor_data})
                            self.last_monitor_report_time = report_now
                        if is_eval:
                            pass
                        else:
                            # Reward for saving the last state of the environment
                            # 保存环境最后状态的reward
                            for i, (do_sample, agent) in enumerate(zip(self.do_samples, self.agents)):
                                if not is_eval and do_sample:
                                    self._save_training_terminal_frame(
                                        frame_collector=self.frame_collector,
                                        agent_id=i,
                                        agent=agent,
                                        observation=observation[str(i)],
                                        terminal_payload=terminal_payloads[i],
                                        terminated=terminated,
                                    )

                    # Sample process
                    # 进行样本处理，准备训练
                    if len(self.frame_collector) > 0 and not is_eval:
                        list_agents_samples = self.frame_collector.sample_process()
                        self._routine_log(
                            f"is_gameover({is_gameover}), "
                            f"len(frame_collector) is {len(self.frame_collector)}",
                            is_train_test=is_train_test,
                        )
                        yield list_agents_samples
                        self.frame_collector.reset(num_agents=self.agent_num)

                    if is_gameover:
                        break

    def reset_agents(
        self,
        observation,
        opponent_agent=None,
        monitor_side=None,
    ):
        if opponent_agent is None:
            opponent_agent = self.env_conf_manager.get_opponent_agent()
        opponent_agent = str(opponent_agent)
        if monitor_side is None:
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
            if self.do_predicts[i]:
                agent.reset(observation[str(i)])

    @staticmethod
    def _training_terminal_payload(agent, observation):
        return EpisodeRunner._controlled_reward_payload(
            agent, observation, episode_terminal=True
        )

    @staticmethod
    def _save_training_terminal_frame(
        frame_collector,
        agent_id,
        agent,
        observation,
        terminal_payload,
        terminated,
    ):
        (
            terminal_rewards,
            terminal_auxiliary_rewards,
            terminal_command_reward,
        ) = terminal_payload
        if terminated:
            actor_bootstrap = (0.0, 0.0)
            actor_auxiliary_bootstrap = tuple(
                np.zeros(
                    FrozenV1Config.AUXILIARY_VALUE_COUNT,
                    dtype=np.float32,
                )
                for _ in range(FrozenV1Config.CONTROLLED_ACTOR_COUNT)
            )
            command_bootstrap = 0.0
        else:
            (
                actor_bootstrap,
                actor_auxiliary_bootstrap,
                command_bootstrap,
            ) = agent.bootstrap_values(observation)
        frame_collector.save_last_frame(
            agent_id=agent_id,
            rewards=terminal_rewards,
            auxiliary_rewards=terminal_auxiliary_rewards,
            command_reward=terminal_command_reward,
            actor_bootstrap_values=actor_bootstrap,
            actor_auxiliary_bootstrap_values=actor_auxiliary_bootstrap,
            command_bootstrap_value=command_bootstrap,
        )

    @staticmethod
    def _controlled_reward_payload(
        agent, observation, episode_terminal=False
    ):
        hero_infos = agent.feature_processes.controlled_infos(observation)
        rewards = []
        auxiliary_rewards = []
        raw_monster_details = []
        controlled_config_ids = []
        for info in hero_infos:
            final_reward, auxiliary = split_reward_payload_v1(
                info.get("reward", [])
            )
            rewards.append(final_reward)
            auxiliary_rewards.append(auxiliary)
            reward_vector = np.asarray(info.get("reward", []), dtype=np.float32)
            raw_monster_details.append(max(float(reward_vector[8]), 0.0))
            controlled_config_ids.append(int(info.get("config_id", 0)))
        if len(rewards) != 2:
            raise RuntimeError("v1 workflow requires exactly two controlled rewards")
        route_shaping = tuple(
            agent.consume_reward_shaping(
                observation, episode_terminal=episode_terminal
            )
        )
        if len(route_shaping) != 2 or not np.isfinite(route_shaping).all():
            raise RuntimeError("custom reward shaping must contain two finite rewards")
        rewards = [
            float(reward) + float(shaping)
            for reward, shaping in zip(rewards, route_shaping)
        ]
        frame_no = u.frame_number(observation.get("frame_state", {}) or {})
        late_scale = agent.feature_processes.late_monster_reward_scale(
            expected_frame=frame_no
        )
        late_offsets = late_monster_reward_offsets_v1(
            controlled_config_ids,
            raw_monster_details,
            late_scale,
            agent.last_reward_boundary_is_new,
        )
        rewards = [
            float(reward) + float(offset)
            for reward, offset in zip(rewards, late_offsets)
        ]
        if agent.last_reward_boundary_is_new:
            kai_raw = sum(
                raw_detail
                for config_id, raw_detail in zip(
                    controlled_config_ids, raw_monster_details
                )
                if config_id == 193
            )
            agent.late_monster_raw_total += float(kai_raw)
            agent.late_monster_offset_total += float(sum(late_offsets))
            agent.late_monster_effective_total += float(kai_raw * late_scale)
        auxiliary_rewards = np.stack(auxiliary_rewards, axis=0)
        custom_auxiliary = np.asarray(
            agent.last_auxiliary_reward_shaping, dtype=np.float32
        )
        if (
            custom_auxiliary.shape != auxiliary_rewards.shape
            or not np.isfinite(custom_auxiliary).all()
        ):
            raise RuntimeError(
                "custom auxiliary reward shaping must be finite [2,5]"
            )
        command_reward = EpisodeRunner._command_sparse_reward(
            agent, observation, hero_infos
        )
        if getattr(agent, "last_reward_boundary_is_new", False):
            agent.command_sparse_reward_total += float(command_reward)
        return rewards, auxiliary_rewards + custom_auxiliary, command_reward

    @staticmethod
    def _command_sparse_reward(agent, observation, controlled_infos=None):
        """Reward command intent only from sparse team-causal outcomes.

        Dense actor HP/economy/experience/damage, lane shaping, and jungle
        routing are intentionally excluded.  Forced opening commands already
        carry ``command.is_train == 0`` in the frozen command protocol.
        """

        return build_command_sparse_reward_v1(
            agent, observation, controlled_infos=controlled_infos
        )

    @staticmethod
    def _controlled_rewards(agent, observation):
        """Compatibility helper returning only the final scalar rewards."""

        rewards, _, _ = EpisodeRunner._controlled_reward_payload(agent, observation)
        return rewards

    @staticmethod
    def _reward_components(hero_info):
        raw_values = hero_info.get("reward", [])
        split_reward_payload_v1(raw_values)
        return np.asarray(raw_values, dtype=np.float64).reshape(-1)
