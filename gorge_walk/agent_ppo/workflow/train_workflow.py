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

from kaiwu_agent.utils.common_func import Frame, attached
from tools.train_env_conf_validate import check_usr_conf, read_usr_conf

# 补回你最关心的核心引用
from agent_ppo.feature.definition import sample_process, reward_shaping
from agent_ppo.conf.conf import Config


def safe_get(obj, key, default=None):
    """
    针对开悟环境设计的万能取值器：
    1. 优先尝试 getattr (处理 C++ 导出的对象)
    2. 失败后尝试 .get (处理普通字典)
    """
    if obj is None:
        return default
    # 尝试作为对象属性获取
    val = getattr(obj, key, None)
    if val is not None:
        return val
    # 尝试作为字典键值获取
    if isinstance(obj, dict):
        return obj.get(key, default)
    # 兜底返回默认值
    return default


# ----------------- 借鉴优点：鲁棒的得分提取函数 -----------------
# def _extract_score_value(next_extra_info):
#     """
#     不管环境返回的是 score、total_score 还是嵌套在 game_info 里，统统抓出来
#     """
#     try:
#         if isinstance(next_extra_info, dict):
#             # 尝试从 game_info 抓取
#             game_info = next_extra_info.safe_get("game_info", {})
#             for key in ["total_score", "score", "value"]:
#                 if key in game_info:
#                     return float(game_info[key])
#             # 尝试直接从 extra_info 抓取
#             for key in ["total_score", "score"]:
#                 if key in next_extra_info:
#                     return float(next_extra_info[key])
#     except Exception:
#         pass
#     return 0.0

def _extract_score_value(next_extra_info):
    """
    安全提取分数逻辑
    """
    # 1. 尝试从 game_info 内部抓取
    game_info = safe_get(next_extra_info, "game_info")
    if game_info:
        score = safe_get(game_info, "total_score")
        if score is not None: return float(score)
        
    # 2. 尝试直接从主对象抓取
    score = safe_get(next_extra_info, "total_score")
    if score is not None: return float(score)
    
    return 0.0

@attached
def workflow(envs, agents, logger=None, monitor=None):
    """
    PPO 分布式训练工作流
    """
    try:
        env, agent = envs[0], agents[0]

        # 1. 校验配置
        usr_conf = read_usr_conf("agent_ppo/conf/train_env_conf.toml", logger)
        if usr_conf is None or not check_usr_conf(usr_conf, logger):
            raise RuntimeError("Config check failed!")

        # ----------------- 借鉴优点：最优模型追踪与监控变量 -----------------
        best_reward = -1e18
        last_put_data_time = 0
        last_save_time = time.time()
        monitor_data = {"reward": 0.0, "win_rate": 0.0, "steps": 0}

        logger.info("PPO Training Workflow Started...")

        while True:
            # 每次运行一个训练周期（由 run_episodes 内部控制局数）
            for g_data, ep_reward, ep_steps, win_flag in run_episodes(
                1, env, agent, usr_conf, logger
            ):
                # 2. PPO 学习核心
                if g_data and len(g_data) > 0:
                    agent.learn(g_data)
                    g_data.clear()

                # 3. 借鉴优点：保存表现最好的模型 (Best Model)
                # 以“单局奖励”作为评价指标
                if ep_reward > best_reward:
                    best_reward = ep_reward
                    agent.save_model(id="best")
                    logger.info(f"!!! New Best Reward: {best_reward:.4f} !!!")

                # 4. 定期保存最新模型 (Latest)
                if time.time() - last_save_time > 300: # 5分钟存一次
                    agent.save_model(id="latest")
                    last_save_time = time.time()

                # 5. 监控数据更新
                monitor_data["reward"] = float(ep_reward)
                monitor_data["win_rate"] = float(win_flag)
                monitor_data["steps"] = int(ep_steps)

                # 6. 定期上报至开悟 Dashboard
                now = time.time()
                if monitor and now - last_put_data_time >= 60:
                    monitor.put_data({os.getpid(): monitor_data})
                    last_put_data_time = now

    except Exception as e:
        if logger:
            logger.error(f"Workflow Critical Error: {e}")
        raise RuntimeError(e)

def run_episodes(n_episode, env, agent, usr_conf, logger=None):
    """
    执行环境交互并采集 PPO 样本
    """
    try:
        for episode in range(n_episode):
            collector = []
            episode_reward = 0.0
            prev_score = 0.0
            step_count = 0

            # 环境重置
            obs, extra_info = env.reset(usr_conf=usr_conf)
            if safe_get(extra_info, "result_code", 0) < 0: continue
            
            agent.reset()
            # 获取初始 250 维特征
            obs_data, _ = agent.observation_process(obs, extra_info)

            done = False
            max_step_no = int(os.environ.get("max_step_no", "2000"))

            while not done:
                # 1. 采样：获取 action, logprob, value
                act_data_list, _ = agent.predict(list_obs_data=[obs_data])
                ppo_info = act_data_list[0] 

                if isinstance(ppo_info, list):
                    ppo_info = ppo_info[0]
                
                # 2. 执行动作
                act = agent.action_process(ppo_info)
                step_no, next_obs, score_info, terminated, truncated, next_extra_info = env.step(act)
                
                if safe_get(next_extra_info, "result_code", 0) != 0: break
                step_count += 1

                # 3. 处理下一帧特征
                next_obs_data, _ = agent.observation_process(next_obs, next_extra_info)

                # 4. 调用奖励函数：必须传入两个 ObsData 以计算位移/距离差
                score = _extract_score_value(next_extra_info)
                reward = reward_shaping(
                    frame_no=step_no,
                    score=score,
                    prev_score=prev_score,
                    terminated=terminated,
                    truncated=truncated,
                    obs=obs_data,    # 当前状态数据对象
                    _obs=next_obs_data # 下一状态数据对象
                )

                # 5. 构建 PPO 样本
                # 注意：truncated(超时) 不算真正的 done，否则会截断优势函数的传递
                real_done = float(terminated) 

                frame = Frame(
                    state=obs_data.feature,
                    action=ppo_info.action,
                    logprob=ppo_info.logprob,
                    reward=reward,
                    done=real_done,
                    value=ppo_info.value,
                    next_state=next_obs_data.feature,
                )
                collector.append(frame)

                # 6. 状态推进
                episode_reward += reward
                prev_score = score
                obs_data = next_obs_data
                done = terminated or truncated or step_count >= max_step_no

            # 返回结果：样本、奖励、步数、是否成功
            win_flag = 1.0 if (done and not truncated) else 0.0
            if len(collector) > 0:
                yield sample_process(collector), episode_reward, step_count, win_flag

    except Exception as e:
        if logger: logger.error(f"run_episodes error: {e}")
        raise RuntimeError(e)