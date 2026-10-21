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
    FrameCollectorV1,
    REWARD_DETAIL_NAMES,
    lineup_iterator_v1,
    split_reward_payload_v1,
)
from agent_ppo.agent import NONE_ACTION, NONE_COL_ACTION
from agent_ppo.conf.conf import FrozenV1Config, GameConfig
from tools.env_conf_manager import EnvConfManager
from tools.model_pool_utils import get_valid_model_pool
from tools.metrics_utils import get_training_metrics
from common_python.utils.workflow_disaster_recovery import handle_disaster_recovery


REWARD_COMPONENT_NAMES = REWARD_DETAIL_NAMES + ("final",)


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
        self.lineup_iterator = lineup_iterator_v1([132, 141, 193])
        self.frame_collector = FrameCollectorV1(self.agent_num)
        self.episode_cnt = 0
        self.last_monitor_report_time = 0.0
        self.latest_platform_metrics = {}

    def _monitor_report_due(self, is_train_test=False, now=None):
        now = time.time() if now is None else float(now)
        return bool(
            is_train_test
            or now - self.last_monitor_report_time
            >= GameConfig.MONITOR_REPORT_INTERVAL_SECONDS
        )

    def run_episodes(self):
        # Single environment process
        # 单局流程
        while True:
            # Retrieving training metrics
            # 获取训练中的指标
            if self.episode_cnt % 10 == 0:
                training_metrics = get_training_metrics(self.logger)
                basic = (training_metrics or {}).get("basic", {})
                if basic:
                    self.latest_platform_metrics = {
                        f"platform_{name}": float(value)
                        for name, value in basic.items()
                        if isinstance(value, (int, float))
                    }

            # Update environment configuration
            # Can use a list of length 2 to pass in the lineup id of the current game
            # 更新对局配置, 可以用长度为2的列表传入当前对局的官方智能体id
            offcial_lineup_ids = next(self.lineup_iterator)
            usr_conf, is_eval, monitor_side = self.env_conf_manager.update_config(offcial_lineup_ids)

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

            is_train_test = os.environ.get("is_train_test", "False").lower() == "true"
            self.logger.info(f"Episode {self.episode_cnt} start, usr_conf is {usr_conf}")

            # Reward initialization
            # 回报初始化
            for i, (do_sample, agent) in enumerate(zip(self.do_samples, self.agents)):
                if do_sample:
                    for hero_info in observation[str(i)]["heroes"]:
                        total_reward_dicts[i][hero_info["config_id"]] = hero_info["reward"][-1]
                        reward_component_sums[i] += self._reward_components(hero_info)

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
                            rewards, auxiliary_rewards = (
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
                                actor_bootstrap_values=actor_bootstrap,
                                actor_auxiliary_bootstrap_values=(
                                    actor_auxiliary_bootstrap
                                ),
                                command_bootstrap_value=command_bootstrap,
                            )
                    list_agents_samples = self.frame_collector.sample_process()
                    self.logger.info(
                        "collector boundary flush, "
                        f"len(frame_collector) is {len(self.frame_collector)}"
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
                            rewards, auxiliary_rewards = self._controlled_reward_payload(
                                agent, observation[str(i)]
                            )
                            frame = build_frame_v1(
                                agent,
                                observation[str(i)],
                                rewards,
                                auxiliary_rewards,
                            )
                            self.frame_collector.save_frame(frame, agent_id=i)
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
                            total_reward_dicts[i][hero_info["config_id"]] += hero_info["reward"][-1]
                            reward_component_sums[i] += self._reward_components(hero_info)

                # Normal end or timeout exit, run train_test will exit early
                # 正常结束或超时退出，运行train_test时会提前退出
                is_gameover = terminated or truncated or (is_train_test and frame_no >= 1000)
                if is_gameover:
                    if is_gameover:
                        self.logger.info(
                            f"episode_{self.episode_cnt} terminated in fno_{frame_no}, truncated:{truncated}, eval:{is_eval}, reward_sum:{total_reward_dicts[monitor_side]}"
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
                                },
                            }
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
                            monitor_data.update(self.latest_platform_metrics)
                            self.monitor.put_data({os.getpid(): monitor_data})
                            self.last_monitor_report_time = report_now
                        if is_eval:
                            pass
                        else:
                            # Reward for saving the last state of the environment
                            # 保存环境最后状态的reward
                            for i, (do_sample, agent) in enumerate(zip(self.do_samples, self.agents)):
                                if not is_eval and do_sample:
                                    (
                                        terminal_rewards,
                                        terminal_auxiliary_rewards,
                                    ) = self._controlled_reward_payload(
                                        agent, observation[str(i)]
                                    )
                                    if terminated:
                                        actor_bootstrap = (0.0, 0.0)
                                        actor_auxiliary_bootstrap = tuple(
                                            np.zeros(
                                                FrozenV1Config.AUXILIARY_VALUE_COUNT,
                                                dtype=np.float32,
                                            )
                                            for _ in range(
                                                FrozenV1Config.CONTROLLED_ACTOR_COUNT
                                            )
                                        )
                                        command_bootstrap = 0.0
                                    else:
                                        (
                                            actor_bootstrap,
                                            actor_auxiliary_bootstrap,
                                            command_bootstrap,
                                        ) = agent.bootstrap_values(
                                            observation[str(i)]
                                        )
                                    self.frame_collector.save_last_frame(
                                        agent_id=i,
                                        rewards=terminal_rewards,
                                        auxiliary_rewards=terminal_auxiliary_rewards,
                                        actor_bootstrap_values=actor_bootstrap,
                                        actor_auxiliary_bootstrap_values=actor_auxiliary_bootstrap,
                                        command_bootstrap_value=command_bootstrap,
                                    )

                    # Sample process
                    # 进行样本处理，准备训练
                    if len(self.frame_collector) > 0 and not is_eval:
                        list_agents_samples = self.frame_collector.sample_process()
                        self.logger.info(
                            f"is_gameover({is_gameover}), len(frame_collector) is {len(self.frame_collector)}"
                        )
                        yield list_agents_samples
                        self.frame_collector.reset(num_agents=self.agent_num)

                    if is_gameover:
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
            if self.do_predicts[i]:
                agent.reset(observation[str(i)])

    @staticmethod
    def _controlled_reward_payload(agent, observation):
        hero_infos = agent.feature_processes.controlled_infos(observation)
        rewards = []
        auxiliary_rewards = []
        for info in hero_infos:
            final_reward, auxiliary = split_reward_payload_v1(
                info.get("reward", [])
            )
            rewards.append(final_reward)
            auxiliary_rewards.append(auxiliary)
        if len(rewards) != 2:
            raise RuntimeError("v1 workflow requires exactly two controlled rewards")
        route_shaping = tuple(agent.consume_reward_shaping(observation))
        if len(route_shaping) != 2 or not np.isfinite(route_shaping).all():
            raise RuntimeError("custom reward shaping must contain two finite rewards")
        rewards = [
            float(reward) + float(shaping)
            for reward, shaping in zip(rewards, route_shaping)
        ]
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
        return rewards, auxiliary_rewards + custom_auxiliary

    @staticmethod
    def _controlled_rewards(agent, observation):
        """Compatibility helper returning only the final scalar rewards."""

        rewards, _ = EpisodeRunner._controlled_reward_payload(agent, observation)
        return rewards

    @staticmethod
    def _reward_components(hero_info):
        raw_values = hero_info.get("reward", [])
        split_reward_payload_v1(raw_values)
        return np.asarray(raw_values, dtype=np.float64).reshape(-1)
