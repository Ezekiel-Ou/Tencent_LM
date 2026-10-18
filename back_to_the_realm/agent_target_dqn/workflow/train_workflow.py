#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2025 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""


import time
import os
import math
from kaiwu_agent.utils.common_func import Frame, attached

from tools.train_env_conf_validate import check_usr_conf, read_usr_conf
from agent_target_dqn.feature.definition import (
    reward_shaping,
    sample_process,
)
from agent_target_dqn.feature.preprocessor import Preprocessor
from tools.metrics_utils import get_training_metrics


def _safe_score_value(score, field, default=0):
    if score is None:
        return default
    return getattr(score, field, default)


def _safe_game_info_value(obs, field, default=0):
    game_info = getattr(obs, "game_info", None)
    if game_info is None:
        return default
    return getattr(game_info, field, default)


def _grid_delta_direction(prev_grid_pos, curr_grid_pos):
    if prev_grid_pos is None or curr_grid_pos is None:
        return None
    dx = curr_grid_pos[0] - prev_grid_pos[0]
    dz = curr_grid_pos[1] - prev_grid_pos[1]
    if dx == 0 and dz == 0:
        return None

    theta = math.atan2(dz, dx)
    if theta < 0:
        theta += 2 * math.pi
    return int(round(theta / (math.pi / 4))) % 8


def _direction_delta(direction_a, direction_b):
    if direction_a is None or direction_b is None:
        return None
    diff = abs(int(direction_a) - int(direction_b)) % 8
    return min(diff, 8 - diff)


def _format_direction(direction):
    return "None" if direction is None else str(int(direction))


def _format_delta(delta):
    return "None" if delta is None else str(int(delta))


@attached
def workflow(envs, agents, logger=None, monitor=None):
    env, agent = envs[0], agents[0]
    epoch_num = 100000
    episode_num_every_epoch = 1
    g_data_truncat = 256
    last_save_model_time = 0

    # Read and validate configuration file
    # 配置文件读取和校验
    usr_conf = read_usr_conf("agent_target_dqn/conf/train_env_conf.toml", logger)
    if usr_conf is None:
        logger.error(f"usr_conf is None, please check agent_target_dqn/conf/train_env_conf.toml")
        return

    # check_usr_conf is a tool to check whether the game configuration is correct
    # It is recommended to perform a check before calling reset.env
    # check_usr_conf会检查游戏配置是否正确，建议调用reset.env前先检查一下
    valid = check_usr_conf(usr_conf, logger)
    if not valid:
        logger.error(f"check_usr_conf return False, please check")
        return

    for epoch in range(epoch_num):
        epoch_total_rew = 0
        data_length = 0
        for g_data in run_episodes(episode_num_every_epoch, env, agent, g_data_truncat, usr_conf, logger, monitor):
            data_length += len(g_data)
            total_rew = sum([i.rew for i in g_data])
            epoch_total_rew += total_rew
            agent.learn(g_data)
            g_data.clear()

        avg_step_reward = 0
        if data_length:
            avg_step_reward = f"{(epoch_total_rew/data_length):.2f}"

        # save model file
        # 保存model文件
        now = time.time()
        if now - last_save_model_time >= 5400:
            agent.save_model()
            last_save_model_time = now

        logger.info(f"Avg Step Reward: {avg_step_reward}, Epoch: {epoch}, Data Length: {data_length}")


def run_episodes(n_episode, env, agent, g_data_truncat, usr_conf, logger, monitor):
    for episode in range(n_episode):
        collector = list()
        preprocessor = Preprocessor()

        # Retrieving training metrics
        # 获取训练中的指标
        training_metrics = get_training_metrics()
        if training_metrics:
            logger.info(f"training_metrics is {training_metrics}")

        # Reset the game and get the initial state
        # 重置游戏, 并获取初始状态
        obs, state_env_info = env.reset(usr_conf=usr_conf)

        # Disaster recovery
        # 容灾
        if obs is None:
            continue

        # At the start of each game, support loading the latest model file
        # The call will load the latest model from a remote training node
        # 每次对局开始时, 支持加载最新model文件, 该调用会从远程的训练节点加载最新模型
        agent.load_model(id="latest")

        # Feature processing
        # 特征处理
        obs_data, remain_info = agent.observation_process(obs, preprocessor, state_env_info)
        logger.info(
            "[episode_start] "
            f"episode={episode}, "
            f"stage={remain_info.get('stage')}, "
            f"coverage={float(remain_info.get('view_coverage', 0.0)) * 100.0:.2f}%, "
            f"discovered={remain_info.get('discovered_treasure_count', 0)}, "
            f"collected={remain_info.get('treasure_collected_count', 0)}/{remain_info.get('treasure_count', 0)}"
        )

        done = False
        step = 0
        bump_cnt = 0
        stuck_cnt = 0
        episode_total_reward = 0.0
        diy_1 = 0
        diy_2 = 0
        diy_3 = 0
        diy_4 = 0
        diy_5 = 0
        last_report_monitor_time = 0
        last_progress_log_time = time.time()
        latest_reward_metrics = {}
        direction_check_count = 0
        target_action_align_count = 0
        action_move_align_count = 0
        target_move_align_count = 0
        latest_direction_debug = {}

        while not done:
            # Agent performs inference, gets the predicted action for the next frame
            # Agent 进行推理, 获取下一帧的预测动作
            act_data, model_version = agent.predict(list_obs_data=[obs_data])

            # Unpack ActData into action
            # ActData 解包成动作
            act = agent.action_process(act_data[0])

            # Interact with the environment, execute actions, get the next state
            # 与环境交互, 执行动作, 获取下一步的状态
            frame_no, _obs, score, terminated, truncated, _state_env_info = env.step(act)
            if _obs is None:
                break

            step += 1

            # Feature processing
            # 特征处理
            _obs_data, _remain_info = agent.observation_process(_obs, preprocessor, _state_env_info)
            move_dir = act % 8
            use_talent = act // 8
            target_dir = remain_info.get("target_dir")
            actual_move_dir = _grid_delta_direction(remain_info.get("grid_pos"), _remain_info.get("grid_pos"))
            target_action_delta = _direction_delta(target_dir, move_dir)
            action_move_delta = _direction_delta(move_dir, actual_move_dir)
            target_move_delta = _direction_delta(target_dir, actual_move_dir)
            grid_unchanged = remain_info.get("grid_pos") == _remain_info.get("grid_pos")
            if actual_move_dir is not None:
                direction_check_count += 1
                if target_action_delta is not None and target_action_delta <= 1:
                    target_action_align_count += 1
                if action_move_delta is not None and action_move_delta <= 1:
                    action_move_align_count += 1
                if target_move_delta is not None and target_move_delta <= 1:
                    target_move_align_count += 1
            if grid_unchanged:
                stuck_cnt += 1
            latest_direction_debug = {
                "target_type": remain_info.get("target_type"),
                "target_dir": target_dir,
                "target_dist_feature": remain_info.get("target_dist_feature"),
                "move_dir": move_dir,
                "use_talent": use_talent,
                "actual_move_dir": actual_move_dir,
                "target_action_delta": target_action_delta,
                "action_move_delta": action_move_delta,
                "target_move_delta": target_move_delta,
                "grid_unchanged": grid_unchanged,
                "prev_grid": remain_info.get("grid_pos"),
                "curr_grid": _remain_info.get("grid_pos"),
            }
            # Disaster recovery
            # 容灾
            if truncated and frame_no is None:
                break

            treasures_num = 0

            # Calculate reward
            # 计算 reward
            if _obs is None:
                reward = 0
            else:
                (
                    reward,
                    is_bump,
                    reward_metrics,
                ) = reward_shaping(
                    frame_no,
                    score,
                    terminated,
                    truncated,
                    remain_info,
                    _remain_info,
                    obs,
                    _obs,
                )
                target_follow_reward = 0.0
                stuck_penalty = -0.35 if grid_unchanged and not terminated and not truncated else 0.0
                if remain_info.get("target_type") != "end" and target_dir is not None:
                    if target_move_delta is not None:
                        target_follow_reward += 0.8 if target_move_delta <= 1 else -0.8
                    elif grid_unchanged:
                        target_follow_reward -= 0.4
                    elif target_action_delta is not None and target_action_delta > 2:
                        target_follow_reward -= 0.3
                reward += target_follow_reward + stuck_penalty
                reward_metrics["target_follow"] = target_follow_reward
                reward_metrics["stuck_penalty"] = stuck_penalty
                diy_1 = float(reward_metrics.get("view_coverage", 0.0)) * 100.0
                diy_2 = float(reward_metrics.get("discovered_treasure_count", 0.0))
                diy_3 += float(reward_metrics.get("weighted_exploration", 0.0))
                diy_4 += float(reward_metrics.get("weighted_treasure_dist", 0.0))
                diy_5 = float(reward_metrics.get("treasure_collected_count", 0.0))
                latest_reward_metrics = reward_metrics
                episode_total_reward += float(reward)

                treasures_num = _remain_info.get("treasure_collected_count", 0)

                # Wall bump behavior statistics
                # 撞墙行为统计
                bump_cnt += is_bump

            now = time.time()
            if monitor and now - last_report_monitor_time >= 30:
                monitor_data = {
                    "diy_1": float(diy_1),
                    "diy_2": float(diy_2),
                    "diy_3": float(diy_3),
                    "diy_4": float(diy_4),
                    "diy_5": float(diy_5),
                }
                monitor.put_data({os.getpid(): monitor_data})
                last_report_monitor_time = now

            if logger and now - last_progress_log_time >= 60:
                avg_reward = episode_total_reward / max(1, step)
                target_action_align_rate = 100.0 * target_action_align_count / max(1, direction_check_count)
                action_move_align_rate = 100.0 * action_move_align_count / max(1, direction_check_count)
                target_move_align_rate = 100.0 * target_move_align_count / max(1, direction_check_count)
                logger.info(
                    "[train_progress] "
                    f"episode={episode}, step={step}, frame_no={frame_no}, "
                    f"stage={_remain_info.get('stage')}, "
                    f"coverage={float(_remain_info.get('view_coverage', 0.0)) * 100.0:.2f}%, "
                    f"discovered={_remain_info.get('discovered_treasure_count', 0)}, "
                    f"collected={_remain_info.get('treasure_collected_count', 0)}/{_remain_info.get('treasure_count', 0)}, "
                    f"buff={_safe_game_info_value(_obs, 'buff_count', 0)}, "
                    f"score={_safe_score_value(score, 'total_score', 0)}, "
                    f"treasure_score={_safe_score_value(score, 'treasure_score', 0)}, "
                    f"reward={float(reward):.3f}, avg_reward={avg_reward:.3f}, total_reward={episode_total_reward:.3f}, "
                    f"r_exp={float(latest_reward_metrics.get('weighted_exploration', 0.0)):.3f}, "
                    f"r_treasure_dist={float(latest_reward_metrics.get('weighted_treasure_dist', 0.0)):.3f}, "
                    f"r_treasure={float(latest_reward_metrics.get('weighted_treasure_collect', 0.0)):.3f}, "
                    f"r_end={float(latest_reward_metrics.get('weighted_end', 0.0)):.3f}, "
                    f"r_target={float(latest_reward_metrics.get('target_follow', 0.0)):.3f}, "
                    f"r_stuck={float(latest_reward_metrics.get('stuck_penalty', 0.0)):.3f}, "
                    f"bumps={bump_cnt}, stuck={stuck_cnt}, act={act}, "
                    f"epsilon={float(getattr(agent.algorithm, 'epsilon', 0.0)):.4f}, "
                    f"target_type={latest_direction_debug.get('target_type')}, "
                    f"target_dir={_format_direction(latest_direction_debug.get('target_dir'))}, "
                    f"move_dir={_format_direction(latest_direction_debug.get('move_dir'))}, "
                    f"use_talent={latest_direction_debug.get('use_talent')}, "
                    f"actual_dir={_format_direction(latest_direction_debug.get('actual_move_dir'))}, "
                    f"delta_target_action={_format_delta(latest_direction_debug.get('target_action_delta'))}, "
                    f"delta_action_move={_format_delta(latest_direction_debug.get('action_move_delta'))}, "
                    f"delta_target_move={_format_delta(latest_direction_debug.get('target_move_delta'))}, "
                    f"target_action_align={target_action_align_rate:.1f}%, "
                    f"action_move_align={action_move_align_rate:.1f}%, "
                    f"target_move_align={target_move_align_rate:.1f}%, "
                    f"grid_unchanged={latest_direction_debug.get('grid_unchanged')}, "
                    f"prev_grid={latest_direction_debug.get('prev_grid')}, "
                    f"curr_grid={latest_direction_debug.get('curr_grid')}"
                )
                last_progress_log_time = now

            # Determine game over, and update the number of victories
            # 判断游戏结束, 并更新胜利次数
            if truncated:
                logger.info(
                    f"truncated is True, so this episode {episode} timeout, \
                        collected treasures: {treasures_num}"
                )
            elif terminated:
                logger.info(
                    f"terminated is True, so this episode {episode} reach the end, \
                        collected treasures: {treasures_num}"
                )
            done = terminated or truncated

            # Construct game frames to prepare for sample construction
            # 构造游戏帧，为构造样本做准备
            frame = Frame(
                obs=obs_data.feature,
                _obs=_obs_data.feature,
                obs_legal=obs_data.legal_act,
                _obs_legal=_obs_data.legal_act,
                act=act,
                rew=reward,
                done=done,
                ret=reward,
            )

            collector.append(frame)

            # If the number of game frames reaches the threshold, the sample is processed and sent to training
            # 如果游戏帧数达到阈值，则进行样本处理，将样本送去训练
            if len(collector) % g_data_truncat == 0:
                collector = sample_process(collector)
                yield collector

            # If the game is over, the sample is processed and sent to training
            # 如果游戏结束，则进行样本处理，将样本送去训练
            if done:
                if len(collector) > 0:
                    collector = sample_process(collector)
                    yield collector

                if monitor:
                    monitor_data = {
                        "diy_1": float(diy_1),
                        "diy_2": float(diy_2),
                        "diy_3": float(diy_3),
                        "diy_4": float(diy_4),
                        "diy_5": float(diy_5),
                    }
                    monitor.put_data({os.getpid(): monitor_data})

                done_reason = "terminated" if terminated else "truncated"
                target_action_align_rate = 100.0 * target_action_align_count / max(1, direction_check_count)
                action_move_align_rate = 100.0 * action_move_align_count / max(1, direction_check_count)
                target_move_align_rate = 100.0 * target_move_align_count / max(1, direction_check_count)
                logger.info(
                    "[episode_summary] "
                    f"episode={episode}, reason={done_reason}, step={step}, frame_no={frame_no}, "
                    f"stage={_remain_info.get('stage')}, "
                    f"coverage={float(_remain_info.get('view_coverage', 0.0)) * 100.0:.2f}%, "
                    f"discovered={_remain_info.get('discovered_treasure_count', 0)}, "
                    f"collected={_remain_info.get('treasure_collected_count', 0)}/{_remain_info.get('treasure_count', 0)}, "
                    f"buff={_safe_game_info_value(_obs, 'buff_count', 0)}, "
                    f"bumps={bump_cnt}, stuck={stuck_cnt}, score={_safe_score_value(score, 'total_score', 0)}, "
                    f"treasure_score={_safe_score_value(score, 'treasure_score', 0)}, "
                    f"episode_reward={episode_total_reward:.3f}, "
                    f"diy_1={float(diy_1):.2f}, diy_2={float(diy_2):.2f}, "
                    f"diy_3={float(diy_3):.3f}, diy_4={float(diy_4):.3f}, diy_5={float(diy_5):.2f}, "
                    f"direction_checks={direction_check_count}, "
                    f"target_action_align={target_action_align_rate:.1f}%, "
                    f"action_move_align={action_move_align_rate:.1f}%, "
                    f"target_move_align={target_move_align_rate:.1f}%, "
                    f"last_target_type={latest_direction_debug.get('target_type')}, "
                    f"last_target_dir={_format_direction(latest_direction_debug.get('target_dir'))}, "
                    f"last_move_dir={_format_direction(latest_direction_debug.get('move_dir'))}, "
                    f"last_actual_dir={_format_direction(latest_direction_debug.get('actual_move_dir'))}"
                )

                break

            # Status update
            # 状态更新
            obs_data = _obs_data
            remain_info = _remain_info
            state_env_info = _state_env_info
