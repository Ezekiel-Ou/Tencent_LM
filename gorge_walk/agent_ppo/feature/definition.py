#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2025 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""

import numpy as np
from kaiwu_agent.utils.common_func import create_cls, attached
from agent_ppo.conf.conf import Config


# PPO 单步样本结构
# 对应：
# state + action + logprob + reward + done + value + next_state
SampleData = create_cls(
    "SampleData",
    state=None,
    action=None,
    logprob=None,
    reward=None,
    done=None,
    value=None,
    next_state=None,
)


@attached
def sample_process(list_game_data):
    return [SampleData(**i.__dict__) for i in list_game_data]


@attached
def observation_process(raw_obs, env_info=None):
    """
    将 11 维原始观测与 env_info 转换为 250 维特征向量
    raw_obs: [end_dist, treasure_dist_0...9]
    env_info: 即 Agent 中的 extra_info
    """
    # 转换为 numpy 数组并检查
    raw_obs = np.asarray(raw_obs, dtype=np.float32).reshape(-1)
    if raw_obs.shape[0] < 11:
        raise RuntimeError(f"raw obs dim error: got {raw_obs.shape[0]}, expect at least 11")
    
    if env_info is None:
        return np.zeros(Config.DIM_OF_OBSERVATION, dtype=np.float32)

    # 1. 位置特征 (1 + 64 + 64 = 129维)
    pos_x, pos_z = int(env_info.pos_x), int(env_info.pos_z)
    state = np.array([(pos_x * 64 + pos_z) / 4095.0], dtype=np.float32)
    
    pos_row = np.zeros(64, dtype=np.float32)
    pos_row[min(pos_x, 63)] = 1.0
    pos_col = np.zeros(64, dtype=np.float32)
    pos_col[min(pos_z, 63)] = 1.0

    # 2. 距离特征 (1 + 10 = 11维)
    max_dist_norm = 150.0
    end_dist = np.array([min(float(raw_obs[0]), max_dist_norm) / max_dist_norm], dtype=np.float32)
    treasure_dist = np.array([min(float(x), max_dist_norm) / max_dist_norm for x in raw_obs[1:11]], dtype=np.float32)

    # 3. 视野图特征 (25 * 3 = 75维)
    local_view = list(env_info.local_view)
    view_radius = int(env_info.view)
    side = view_radius * 2 + 1
    
    # 确保视野长度正确
    if len(local_view) != side * side:
        local_view = (local_view + [0] * (side * side))[:side * side]

    # 提取：0-障碍物, 4-宝箱, 3-终点
    obstacle_flat = np.array([1.0 if x == 0 else 0.0 for x in local_view], dtype=np.float32)
    treasure_flat = np.array([1.0 if x == 4 else 0.0 for x in local_view], dtype=np.float32)
    end_flat = np.array([1.0 if x == 3 else 0.0 for x in local_view], dtype=np.float32)

    # 4. 记忆图特征 (25维)
    location_memory = np.asarray(env_info.location_memory, dtype=np.float32).reshape(64, 64)
    memory_patch = np.zeros((side, side), dtype=np.float32)
    for i in range(side):
        for j in range(side):
            gx, gz = pos_x - view_radius + i, pos_z - view_radius + j
            if 0 <= gx < 64 and 0 <= gz < 64:
                memory_patch[i, j] = location_memory[gx, gz]
    memory_flat = np.clip(memory_patch.reshape(-1), 0.0, 1.0)

    # 5. 宝箱状态 (10维)
    raw_status = list(env_info.treasure_status)[:10]
    treasure_status = np.array([1.0 if int(x) == 1 else 0.0 for x in raw_status], dtype=np.float32)

    # 6. 拼接
    feature = np.concatenate([
        state, pos_row, pos_col, end_dist, treasure_dist,
        obstacle_flat, treasure_flat, end_flat, memory_flat, treasure_status
    ]).astype(np.float32)

    return feature

# @attached
# def reward_shaping(frame_no, score, prev_score, terminated, truncated, obs, _obs):
#     """
#     PPO 奖励函数，利用 250 维特征进行动态引导
#     obs: 当前状态特征 (ObsData对象，含feature)
#     _obs: 下一状态特征 (ObsData对象，含feature)
#     """
#     reward = 0.0
    
#     # 1. 基础步数惩罚
#     reward -= 0.02

#     # 2. 路径距离引导 (利用 index 129: end_dist)
#     # 环境提供的 end_dist 是避开障碍的最短路径距离，极具引导价值
#     cur_dist = obs.feature[129]
#     next_dist = _obs.feature[129]
    
#     if next_dist < cur_dist:
#         reward += 0.2    # 向终点迈进
#     elif next_dist > cur_dist:
#         reward -= 0.2    # 远离终点

#     # 3. 撞墙/无效动作检测
#     cur_row = np.argmax(obs.feature[1:65])
#     cur_col = np.argmax(obs.feature[65:129])
#     next_row = np.argmax(_obs.feature[1:65])
#     next_col = np.argmax(_obs.feature[65:129])

#     if cur_row == next_row and cur_col == next_col:
#         reward -= 1.0  # 没动弹，重罚
#     else:
#         # 4. 防“鬼打墙”：利用 memory_flat (index 215-239)
#         dr, dc = next_row - cur_row, next_col - cur_col
#         # 计算刚才这一步迈上去的格子在视野中的相对索引
#         visited_idx = 215 + (2 + dr) * 5 + (2 + dc)
#         if 215 <= visited_idx < 240:
#             visit_value = float(obs.feature[visited_idx])
#             if visit_value <= 0.0:
#                 reward += 0.15   # 探索新格子奖励
#             elif visit_value <= 0.2:
#                 reward -= 0.15   # 踩旧格子微惩
#             else:
#                 reward -= 0.45   # 严重重复探索，重惩（遏制横跳）

#     # 5. 宝箱收集奖励 (利用 index 240-249: treasure_status)
#     cur_status = obs.feature[240:250]
#     next_status = _obs.feature[240:250]
#     collected = np.sum(cur_status) - np.sum(next_status)
#     if collected > 0:
#         reward += 25.0 * collected  # 吃到宝箱大奖

#     # 6. 宝箱距离引导 (利用 index 130-139: treasure_dist)
#     active_indices = np.where(cur_status == 1)[0]
#     if len(active_indices) > 0:
#         cur_min_t_dist = np.min(obs.feature[130:140][active_indices])
#         next_min_t_dist = np.min(_obs.feature[130:140][active_indices])
#         if next_min_t_dist < cur_min_t_dist:
#             reward += 0.05
#         elif next_min_t_dist > cur_min_t_dist:
#             reward -= 0.05

#     # 7. 终点奖励
#     if terminated and not truncated:
#         reward += 150.0

#     return float(reward)

# import numpy as np


# @attached
# def reward_shaping(frame_no, score, prev_score, terminated, truncated, obs, _obs):
#     """
#     优化版 PPO / DQN 通用奖励函数
#     改动点：
#     1. 只对“目标宝箱”靠近才宽恕，不对任意宝箱宽恕
#     2. 探索奖励随时间衰减
#     3. 后期显著增强终点导向
#     4. 加入“终点附近被墙阻挡时的绕路宽恕”
#     """

#     reward = 0.0

#     # ==========================================================
#     # 0. 基础状态解析
#     # ==========================================================
#     cur_feature = obs.feature
#     next_feature = _obs.feature

#     # 位置
#     cur_row = int(np.argmax(cur_feature[1:65]))
#     cur_col = int(np.argmax(cur_feature[65:129]))
#     next_row = int(np.argmax(next_feature[1:65]))
#     next_col = int(np.argmax(next_feature[65:129]))

#     moved = not (cur_row == next_row and cur_col == next_col)

#     # 终点距离（环境最短路距离）
#     cur_end_dist = float(cur_feature[129])
#     next_end_dist = float(next_feature[129])

#     # 宝箱距离与状态
#     cur_t_dists = cur_feature[130:140]
#     next_t_dists = next_feature[130:140]
#     cur_t_status = cur_feature[240:250]
#     next_t_status = next_feature[240:250]

#     available_idx = np.where(cur_t_status == 1)[0]

#     # 当前是否吃到宝箱
#     collected_diff = int(np.sum(cur_t_status) - np.sum(next_t_status))
#     just_got_treasure = collected_diff > 0

#     # 剩余步数
#     max_step = 2000
#     remain_steps = max_step - frame_no

#     # ==========================================================
#     # 1. 基础步数惩罚（后期略增强）
#     # ==========================================================
#     if frame_no < 1200:
#         base_penalty = 0.02
#     elif frame_no < 1700:
#         base_penalty = 0.03
#     else:
#         base_penalty = 0.05
#     reward -= base_penalty

#     # ==========================================================
#     # 2. 宝箱即时奖励（保留，但避免过强）
#     # ==========================================================
#     if just_got_treasure:
#         reward += 25.0 * collected_diff

#     # ==========================================================
#     # 3. 目标宝箱选择：只宽恕“目标宝箱”
#     #    不是最近宝箱绝对值，而是选一个更值得追的目标
#     #    这里先用简单启发式：dist_to_treasure + 0.25 * treasure_to_end_proxy
#     #    当前没有 treasure->end 真值，就先退化为最近宝箱
#     # ==========================================================
#     target_t_idx = None
#     cur_target_dist = None
#     next_target_dist = None
#     approaching_target_treasure = False

#     if len(available_idx) > 0:
#         # 简单版本：目标宝箱 = 当前最近未收集宝箱
#         # 后续如果你愿意，可以替换成更复杂的路径价值打分
#         target_t_idx = available_idx[np.argmin(cur_t_dists[available_idx])]
#         cur_target_dist = float(cur_t_dists[target_t_idx])
#         next_target_dist = float(next_t_dists[target_t_idx])

#         if next_target_dist < cur_target_dist:
#             approaching_target_treasure = True

#     # ==========================================================
#     # 4. 动态引导权重：前期偏宝箱，后期偏终点
#     # ==========================================================
#     if frame_no < 1200:
#         w_treasure = 0.35
#         w_end = 0.20
#     elif frame_no < 1600:
#         w_treasure = 0.25
#         w_end = 0.35
#     else:
#         w_treasure = 0.10
#         w_end = 0.60

#     # ==========================================================
#     # 5. 宝箱引导（只针对目标宝箱）
#     # ==========================================================
#     t_guidance = 0.0
#     if target_t_idx is not None:
#         if next_target_dist < cur_target_dist:
#             t_guidance = +1.0
#         elif next_target_dist > cur_target_dist:
#             t_guidance = -1.0
#         reward += w_treasure * t_guidance

#     # ==========================================================
#     # 6. 终点引导
#     #    关键优化：终点附近若被墙挡住，不要因为“暂时没变近”就重罚
#     # ==========================================================
#     end_guidance = 0.0

#     # 是否朝终点取得进展
#     progress_to_end = next_end_dist < cur_end_dist

#     # “终点附近”阈值：可调
#     near_end = cur_end_dist <= 8

#     # 绕墙判定：
#     # - 已经在终点附近
#     # - 位置发生了变化（不是撞墙）
#     # - 但是 end_dist 没下降，甚至略升
#     # 这种情况大概率是在绕障碍
#     near_end_detour = near_end and moved and (next_end_dist >= cur_end_dist)

#     if progress_to_end:
#         end_guidance = +1.0
#     else:
#         if near_end_detour:
#             # 终点附近绕墙宽恕：不给重罚，只给极轻微负分甚至 0
#             end_guidance = -0.05
#         else:
#             # 一般情况下，远离终点要罚；但若当前正在接近目标宝箱，则减轻
#             end_guidance = -0.15 if approaching_target_treasure else -0.35

#     reward += w_end * end_guidance

#     # ==========================================================
#     # 7. 探索奖励（随时间衰减）
#     # ==========================================================
#     # 早期鼓励探索，后期自动减弱，避免快结束时还乱钻新区域
#     explore_decay = max(0.15, 1.0 - frame_no / 1500.0)
#     explore_bonus = 0.20 * explore_decay

#     if moved:
#         dr, dc = next_row - cur_row, next_col - cur_col
#         visited_idx = 215 + (2 + dr) * 5 + (2 + dc)

#         if 215 <= visited_idx < 240:
#             visit_value = float(cur_feature[visited_idx])

#             if visit_value <= 0.0:
#                 reward += explore_bonus

#     # ==========================================================
#     # 8. 防撞墙 / 防横跳
#     #    关键优化：
#     #    - 原地不动重罚
#     #    - 重复踩踏只在“没有正当目的”时重罚
#     #    - 正当目的：刚拿宝箱 / 接近终点 / 接近目标宝箱 / 终点附近绕墙
#     # ==========================================================
#     if not moved:
#         reward -= 0.8
#     else:
#         dr, dc = next_row - cur_row, next_col - cur_col
#         visited_idx = 215 + (2 + dr) * 5 + (2 + dc)

#         if 215 <= visited_idx < 240:
#             visit_value = float(cur_feature[visited_idx])

#             forgiveness = (
#                 just_got_treasure
#                 or progress_to_end
#                 or approaching_target_treasure
#                 or near_end_detour
#             )

#             if visit_value > 0.3:
#                 reward -= (0.08 if forgiveness else 0.45)

#     # ==========================================================
#     # 9. “终点附近但不会绕”的额外修正
#     # ==========================================================
#     # 如果已经离终点非常近，但多步都没有实际推进，说明策略可能卡在墙边。
#     # 单步 reward 里能做的，是：
#     # - 不要重罚“为绕墙而产生的暂时绕远”
#     # - 同时，若在终点附近移动到了新格子，给一点轻微鼓励
#     if near_end and moved and not progress_to_end:
#         # 说明可能在寻找绕路通道
#         reward += 0.05

#     # ==========================================================
#     # 10. 终局奖励
#     # ==========================================================
#     if terminated and not truncated:
#         # 固定大额终点奖励，避免和途中宝箱奖励重复记账过多
#         reward += 150.0

#     # truncated 不额外重罚，保持稳定
#     return float(reward)

# @attached
# def reward_shaping(frame_no, score, prev_score, terminated, truncated, obs, _obs):
#     reward = 0.0

#     cur_feature = obs.feature
#     next_feature = _obs.feature

#     # =========================
#     # 0. 基础解析
#     # =========================
#     cur_row = int(np.argmax(cur_feature[1:65]))
#     cur_col = int(np.argmax(cur_feature[65:129]))
#     next_row = int(np.argmax(next_feature[1:65]))
#     next_col = int(np.argmax(next_feature[65:129]))
#     moved = not (cur_row == next_row and cur_col == next_col)

#     # 注意：这些距离都是归一化后的
#     cur_end_dist = float(cur_feature[129])
#     next_end_dist = float(next_feature[129])

#     cur_t_dists = cur_feature[130:140]
#     next_t_dists = next_feature[130:140]

#     cur_t_status = cur_feature[240:250]
#     next_t_status = next_feature[240:250]
#     available_idx = np.where(cur_t_status == 1)[0]

#     collected_diff = int(np.sum(cur_t_status) - np.sum(next_t_status))
#     just_got_treasure = collected_diff > 0
#     collected_total = int(10 - np.sum(next_t_status))

#     remain_steps = 2000 - frame_no

#     # 归一化阈值
#     NEAR_END_TH = 8.0 / 150.0
#     VERY_NEAR_END_TH = 6.0 / 150.0
#     OPPORTUNITY_TH = 2.0 / 150.0

#     near_end = cur_end_dist <= NEAR_END_TH
#     very_near_end = cur_end_dist <= VERY_NEAR_END_TH

#     progress_to_end = next_end_dist < cur_end_dist

#     # =========================
#     # 1. 基础步数惩罚
#     # =========================
#     if frame_no < 1200:
#         reward -= 0.02
#     elif frame_no < 1700:
#         reward -= 0.03
#     else:
#         reward -= 0.05

#     # =========================
#     # 2. 宝箱即时奖励
#     # =========================
#     if just_got_treasure:
#         reward += 25.0 * collected_diff

#     # =========================
#     # 3. 模式切换：宝箱模式 / 终点模式
#     #    不做真正锁定，但至少不要同一步双目标拉扯
#     # =========================
#     go_end_mode = (
#         frame_no > 1550
#         or collected_total >= 7
#         or near_end
#         or remain_steps <= 200
#     )

#     # =========================
#     # 4. 主引导：二选一
#     # =========================
#     if go_end_mode:
#         # ---------- 终点模式 ----------
#         if progress_to_end:
#             reward += 0.50
#         else:
#             # 终点附近允许轻微绕路，但不允许一直踩老路
#             if near_end and moved:
#                 reward -= 0.03
#             else:
#                 reward -= 0.18

#     else:
#         # ---------- 宝箱模式 ----------
#         if len(available_idx) > 0:
#             target_t_idx = available_idx[np.argmin(cur_t_dists[available_idx])]
#             cur_target_dist = float(cur_t_dists[target_t_idx])
#             next_target_dist = float(next_t_dists[target_t_idx])

#             # 机会宝箱：只有非常近时才允许顺手吃
#             # 这里因为没有跨步状态，只做弱版本
#             if next_target_dist < cur_target_dist:
#                 reward += 0.30
#             elif next_target_dist > cur_target_dist:
#                 reward -= 0.12

#         # 宝箱模式下，终点引导只保留极弱保底
#         if progress_to_end:
#             reward += 0.05

#     # =========================
#     # 5. 探索奖励（时间衰减）
#     # =========================
#     explore_decay = max(0.15, 1.0 - frame_no / 1500.0)
#     explore_bonus = 0.20 * explore_decay

#     visit_value = None
#     if moved:
#         dr, dc = next_row - cur_row, next_col - cur_col
#         visited_idx = 215 + (2 + dr) * 5 + (2 + dc)

#         if 215 <= visited_idx < 240:
#             visit_value = float(cur_feature[visited_idx])

#             if visit_value <= 0.0:
#                 # 新格子探索
#                 reward += explore_bonus

#     # =========================
#     # 6. 防撞墙 / 防横跳 / 防终点附近卡墙
#     # =========================
#     if not moved:
#         reward -= 0.8
#     else:
#         if visit_value is not None and visit_value > 0.3:
#             # 一般旧格子惩罚
#             old_cell_penalty = 0.35

#             # 如果在终点模式且已经很靠近终点，
#             # 走旧格子但又没接近终点，说明很可能在入口附近乱绕
#             if go_end_mode and very_near_end and (not progress_to_end):
#                 old_cell_penalty = 0.75

#             # 如果刚拿到宝箱，给一点宽恕
#             if just_got_treasure:
#                 old_cell_penalty *= 0.4

#             reward -= old_cell_penalty

#     # =========================
#     # 7. 终点附近找入口奖励
#     #    关键：只奖励“新格子上的绕路”，不奖励旧格子乱绕
#     # =========================
#     if go_end_mode and near_end and moved and (not progress_to_end):
#         if visit_value is not None and visit_value <= 0.0:
#             reward += 0.12   # 终点附近走新格子，鼓励找入口
#         elif visit_value is not None and visit_value > 0.3:
#             reward -= 0.15   # 终点附近走旧格子且没推进，额外惩罚

#     # =========================
#     # 8. 终局奖励
#     # =========================
#     if terminated and not truncated:
#         reward += 150.0

#     return float(reward)

# @attached
# def reward_shaping(frame_no, score, prev_score, terminated, truncated, obs, _obs):
#     """
#     融合版 PPO 奖励函数（TSP先验 + 机会宝箱 + 防卡墙）
#     适配当前 definition.py 的 250 维特征结构：
#     - 1:65   -> row one-hot
#     - 65:129 -> col one-hot
#     - 129    -> end_dist (已归一化)
#     - 130:140 -> treasure_dist (已归一化)
#     - 215:240 -> memory_flat
#     - 240:250 -> treasure_status
#     """
#     reward = 0.0

#     cur_feature = obs.feature
#     next_feature = _obs.feature

#     # ==========================================================
#     # 0. 基础解析
#     # ==========================================================
#     cur_row = int(np.argmax(cur_feature[1:65]))
#     cur_col = int(np.argmax(cur_feature[65:129]))
#     next_row = int(np.argmax(next_feature[1:65]))
#     next_col = int(np.argmax(next_feature[65:129]))
#     moved = not (cur_row == next_row and cur_col == next_col)

#     # 注意：这些距离都是归一化后的
#     cur_end_dist = float(cur_feature[129])
#     next_end_dist = float(next_feature[129])

#     cur_t_dists = cur_feature[130:140]
#     next_t_dists = next_feature[130:140]

#     cur_t_status = cur_feature[240:250]
#     next_t_status = next_feature[240:250]

#     collected_diff = int(np.sum(cur_t_status) - np.sum(next_t_status))
#     just_got_treasure = collected_diff > 0
#     collected_total = int(10 - np.sum(cur_t_status))

#     available_idx = np.where(cur_t_status == 1)[0]

#     # 归一化阈值
#     near_end_th = Config.NEAR_END_DIST / Config.MAX_DIST_NORM
#     very_near_end_th = Config.VERY_NEAR_END_DIST / Config.MAX_DIST_NORM
#     opportunity_th = Config.OPPORTUNITY_DIST / Config.MAX_DIST_NORM

#     near_end = cur_end_dist <= near_end_th
#     very_near_end = cur_end_dist <= very_near_end_th
#     progress_to_end = next_end_dist < cur_end_dist

#     remain_steps = 2000 - frame_no

#     # ==========================================================
#     # 1. 基础步数惩罚
#     # ==========================================================
#     if frame_no < 800:
#         reward -= 0.02
#     elif frame_no < 1400:
#         reward -= 0.03
#     else:
#         reward -= 0.05

#     # ==========================================================
#     # 2. 即时得分奖励
#     # ==========================================================
#     # 吃到宝箱 / 环境加分时奖励
#     if score > prev_score:
#         reward += 25.0

#     # 双保险：根据宝箱状态变化再给一次一致奖励
#     if just_got_treasure:
#         reward += 10.0 * collected_diff

#     # ==========================================================
#     # 3. 模式切换：终点模式 or 宝箱模式
#     #    不做硬锁定，但避免“终点/宝箱”双重强拉扯
#     # ==========================================================
#     go_end_mode = (
#         frame_no > 1000
#         or collected_total >= 7
#         # or near_end
#         or remain_steps <= 500
#     )

#     # ==========================================================
#     # 4. 终点主引导（终点模式）
#     # ==========================================================
#     if go_end_mode:
#         if progress_to_end:
#             reward += 0.50
#         else:
#             # 终点附近允许轻微绕路，但不能无限乱绕
#             if near_end and moved:
#                 reward -= 0.03
#             else:
#                 reward -= 0.18

#     # ==========================================================
#     # 5. TSP阶段目标宝箱引导（宝箱模式）
#     # ==========================================================
#     target_idx = None

#     if (not go_end_mode) and (collected_total < len(Config.OPTIMAL_TREASURE_ORDER)):
#         target_idx = Config.OPTIMAL_TREASURE_ORDER[collected_total]

#         # 主目标宝箱仍存在，才引导
#         if cur_t_status[target_idx] == 1:
#             cur_target_dist = float(cur_t_dists[target_idx])
#             next_target_dist = float(next_t_dists[target_idx])

#             if next_target_dist < cur_target_dist:
#                 reward += Config.TSP_TARGET_REWARD
#             elif next_target_dist > cur_target_dist:
#                 reward -= Config.TSP_TARGET_PENALTY

#             # 防“死磕目标宝箱”：
#             # 既没接近主目标，也没接近终点，说明可能在墙边乱绕
#             if (next_target_dist >= cur_target_dist) and (next_end_dist >= cur_end_dist):
#                 reward -= 0.25

#     # # ==========================================================
#     # # 6. 机会宝箱（顺手吃，不推翻TSP主目标）
#     # # ==========================================================
#     # if (not go_end_mode) and (len(available_idx) > 0):
#     #     nearest_idx = available_idx[np.argmin(cur_t_dists[available_idx])]
#     #     nearest_dist = float(cur_t_dists[nearest_idx])
#     #     next_nearest_dist = float(next_t_dists[nearest_idx])

#     #     # 非常近的宝箱才允许“顺手吃”
#     #     if nearest_dist <= opportunity_th:
#     #         # 若它不是当前TSP主目标，给一个较小机会奖励
#     #         if (target_idx is None) or (nearest_idx != target_idx):
#     #             if next_nearest_dist < nearest_dist:
#     #                 reward += 0.12

#     # ==========================================================
#     # 7. 宝箱模式下，保留很弱的终点保底引导
#     # ==========================================================
#     if (not go_end_mode) and progress_to_end:
#         reward += 0.05

#     # ==========================================================
#     # 8. 探索奖励（随时间衰减）
#     # ==========================================================
#     explore_decay = max(0.15, 1.0 - frame_no / 1500.0)
#     explore_bonus = 0.20 * explore_decay

#     visit_value = None
#     if moved:
#         dr, dc = next_row - cur_row, next_col - cur_col
#         visited_idx = 215 + (2 + dr) * 5 + (2 + dc)

#         if 215 <= visited_idx < 240:
#             visit_value = float(cur_feature[visited_idx])

#             # 新格子探索奖励
#             if visit_value <= 0.0:
#                 reward += explore_bonus

#     # ==========================================================
#     # 9. 防撞墙 / 防横跳 / 防终点附近卡墙
#     # ==========================================================
#     if not moved:
#         reward -= 0.8
#     else:
#         if visit_value is not None and visit_value > 0.3:
#             # 默认旧格子惩罚
#             old_cell_penalty = 0.35

#             # 终点非常近时，如果走旧格子且没接近终点，说明很可能在入口附近乱绕
#             if go_end_mode and very_near_end and (not progress_to_end):
#                 old_cell_penalty = 0.75

#             # 刚拿到宝箱，适当宽恕一点
#             if just_got_treasure:
#                 old_cell_penalty *= 0.4

#             reward -= old_cell_penalty

#     # ==========================================================
#     # 10. 终点附近“找入口”奖励
#     #     只奖励新格子的绕路，不奖励旧格子乱绕
#     # ==========================================================
#     if go_end_mode and near_end and moved and (not progress_to_end):
#         if visit_value is not None and visit_value <= 0.0:
#             reward += 0.2   # 终点附近走到新格子，鼓励找入口
#         elif visit_value is not None and visit_value > 0.3:
#             reward -= 0.15   # 终点附近旧格子乱绕，额外惩罚

#     # ==========================================================
#     # 11. 终局奖励
#     # ==========================================================
#     if terminated and not truncated:
#         reward += 150.0

#     if truncated and not terminated:
#         reward -= 100.0

#     return float(reward)


# @attached
# def reward_shaping(frame_no, score, prev_score, terminated, truncated, obs, _obs):
#     """
#     修正版 PPO 奖励函数
#     重点修复：
#     1. 吃到宝箱后原地打转
#     2. 只吃一个宝箱就去终点
#     """
#     reward = 0.0

#     cur_feature = obs.feature
#     next_feature = _obs.feature

#     # ==========================================================
#     # 0. 基础解析
#     # ==========================================================
#     cur_row = int(np.argmax(cur_feature[1:65]))
#     cur_col = int(np.argmax(cur_feature[65:129]))
#     next_row = int(np.argmax(next_feature[1:65]))
#     next_col = int(np.argmax(next_feature[65:129]))
#     moved = not (cur_row == next_row and cur_col == next_col)

#     # 归一化距离
#     cur_end_dist = float(cur_feature[129])
#     next_end_dist = float(next_feature[129])

#     cur_t_dists = cur_feature[130:140]
#     next_t_dists = next_feature[130:140]

#     cur_t_status = cur_feature[240:250]
#     next_t_status = next_feature[240:250]

#     collected_diff = int(np.sum(cur_t_status) - np.sum(next_t_status))
#     just_got_treasure = collected_diff > 0
#     collected_total = int(10 - np.sum(cur_t_status))

#     available_idx = np.where(cur_t_status == 1)[0]

#     near_end_th = Config.NEAR_END_DIST / Config.MAX_DIST_NORM
#     very_near_end_th = Config.VERY_NEAR_END_DIST / Config.MAX_DIST_NORM
#     opportunity_th = Config.OPPORTUNITY_DIST / Config.MAX_DIST_NORM

#     near_end = cur_end_dist <= near_end_th
#     very_near_end = cur_end_dist <= very_near_end_th
#     progress_to_end = next_end_dist < cur_end_dist

#     remain_steps = 2000 - frame_no

#     # ==========================================================
#     # 1. 基础步数惩罚
#     # ==========================================================
#     if frame_no < 800:
#         reward -= 0.02
#     elif frame_no < 1400:
#         reward -= 0.03
#     else:
#         reward -= 0.05

#     # ==========================================================
#     # 2. 即时得分奖励
#     # ==========================================================
#     if score > prev_score:
#         reward += 25.0

#     if just_got_treasure:
#         reward += 10.0 * collected_diff

#     # ==========================================================
#     # 3. 更晚进入终点模式
#     #    解决“只吃一个宝箱就回终点”
#     # ==========================================================
#     go_end_mode = (
#         frame_no > 1400
#         or collected_total >= 7
#         or remain_steps <= 300
#     )

#     # ==========================================================
#     # 4. 终点主引导（只在终点模式）
#     # ==========================================================
#     if go_end_mode:
#         if progress_to_end:
#             reward += 0.50
#         else:
#             if near_end and moved:
#                 reward -= 0.03
#             else:
#                 reward -= 0.18

#     # ==========================================================
#     # 5. TSP阶段目标宝箱引导（加强）
#     # ==========================================================
#     target_idx = None

#     if (not go_end_mode) and (collected_total < len(Config.OPTIMAL_TREASURE_ORDER)):
#         target_idx = Config.OPTIMAL_TREASURE_ORDER[collected_total]

#         if cur_t_status[target_idx] == 1:
#             cur_target_dist = float(cur_t_dists[target_idx])
#             next_target_dist = float(next_t_dists[target_idx])

#             # 比原来更强一点，强化继续吃宝箱的倾向
#             if next_target_dist < cur_target_dist:
#                 reward += 0.50
#             elif next_target_dist > cur_target_dist:
#                 reward -= 0.18

#             # 防死磕目标宝箱
#             if (next_target_dist >= cur_target_dist) and (next_end_dist >= cur_end_dist):
#                 reward -= 0.25

#     # # ==========================================================
#     # # 6. 机会宝箱（恢复）
#     # #    让模型更容易形成“顺手多吃几个”的行为
#     # # ==========================================================
#     # if (not go_end_mode) and (len(available_idx) > 0):
#     #     nearest_idx = available_idx[np.argmin(cur_t_dists[available_idx])]
#     #     nearest_dist = float(cur_t_dists[nearest_idx])
#     #     next_nearest_dist = float(next_t_dists[nearest_idx])

#     #     if nearest_dist <= opportunity_th:
#     #         if (target_idx is None) or (nearest_idx != target_idx):
#     #             if next_nearest_dist < nearest_dist:
#     #                 reward += 0.08

#     # ==========================================================
#     # 7. 探索奖励（随时间衰减）
#     # ==========================================================
#     explore_decay = max(0.15, 1.0 - frame_no / 1500.0)
#     explore_bonus = 0.20 * explore_decay

#     visit_value = None
#     if moved:
#         dr, dc = next_row - cur_row, next_col - cur_col
#         visited_idx = 215 + (2 + dr) * 5 + (2 + dc)

#         if 215 <= visited_idx < 240:
#             visit_value = float(cur_feature[visited_idx])

#             if visit_value <= 0.0:
#                 reward += explore_bonus

#     # ==========================================================
#     # 8. 防撞墙 / 防横跳
#     #    修复“吃到宝箱后原地打转”
#     # ==========================================================
#     if not moved:
#         reward -= 0.8
#         # 吃到宝箱后还不动，额外轻罚
#         if just_got_treasure:
#             reward -= 0.3
#     else:
#         if visit_value is not None and visit_value > 0.3:
#             old_cell_penalty = 0.35

#             if go_end_mode and very_near_end and (not progress_to_end):
#                 old_cell_penalty = 0.75

#             # 原来是 *0.4，太宽松，容易拿完宝箱原地绕
#             if just_got_treasure:
#                 old_cell_penalty *= 0.7

#             reward -= old_cell_penalty

#     # 吃到宝箱后鼓励尽快离开当前位置，减少局部打转
#     if just_got_treasure and moved:
#         if visit_value is not None and visit_value <= 0.3:
#             reward += 0.08

#     # ==========================================================
#     # 9. 终点附近“找入口”奖励
#     # ==========================================================
#     if go_end_mode and near_end and moved and (not progress_to_end):
#         if visit_value is not None and visit_value <= 0.0:
#             reward += 0.20
#         elif visit_value is not None and visit_value > 0.3:
#             reward -= 0.15

#     # ==========================================================
#     # 10. 终局奖励 / 惩罚
#     # ==========================================================
#     if terminated and not truncated:
#         reward += 150.0

#     if truncated and not terminated:
#         reward -= 100.0

#     return float(reward)

@attached
def reward_shaping(frame_no, score, prev_score, terminated, truncated, obs, _obs):
    """
    修正版 PPO 奖励函数
    重点修复：
    1. 吃到宝箱后原地打转
    2. 只吃一个宝箱就去终点
    3. 绕不过墙（新增：障碍区绕行宽恕）
    """
    reward = 0.0

    cur_feature = obs.feature
    next_feature = _obs.feature

    # ==========================================================
    # 0. 基础解析
    # ==========================================================
    cur_row = int(np.argmax(cur_feature[1:65]))
    cur_col = int(np.argmax(cur_feature[65:129]))
    next_row = int(np.argmax(next_feature[1:65]))
    next_col = int(np.argmax(next_feature[65:129]))
    moved = not (cur_row == next_row and cur_col == next_col)

    # 归一化距离
    cur_end_dist = float(cur_feature[129])
    next_end_dist = float(next_feature[129])

    cur_t_dists = cur_feature[130:140]
    next_t_dists = next_feature[130:140]

    cur_t_status = cur_feature[240:250]
    next_t_status = next_feature[240:250]

    collected_diff = int(np.sum(cur_t_status) - np.sum(next_t_status))
    just_got_treasure = collected_diff > 0
    collected_total = int(10 - np.sum(cur_t_status))

    available_idx = np.where(cur_t_status == 1)[0]

    near_end_th = Config.NEAR_END_DIST / Config.MAX_DIST_NORM
    very_near_end_th = Config.VERY_NEAR_END_DIST / Config.MAX_DIST_NORM
    opportunity_th = Config.OPPORTUNITY_DIST / Config.MAX_DIST_NORM

    near_end = cur_end_dist <= near_end_th
    very_near_end = cur_end_dist <= very_near_end_th
    progress_to_end = next_end_dist < cur_end_dist

    remain_steps = 2000 - frame_no

    # ==========================================================
    # 0.5 局部障碍密度（新增）
    # feature[140:165] 是 5x5 obstacle_flat
    # 障碍密度高，说明更可能处于墙边/通道/绕行场景
    # ==========================================================
    obstacle_patch = cur_feature[140:165]
    obstacle_density = float(np.mean(obstacle_patch))
    in_obstacle_area = obstacle_density > 0.25

    # ==========================================================
    # 1. 基础步数惩罚
    # ==========================================================
    if frame_no < 800:
        reward -= 0.02
    elif frame_no < 1400:
        reward -= 0.03
    else:
        reward -= 0.05

    # ==========================================================
    # 2. 即时得分奖励
    # ==========================================================
    if score > prev_score:
        reward += 25.0

    if just_got_treasure:
        reward += 10.0 * collected_diff

    # ==========================================================
    # 3. 更晚进入终点模式
    # ==========================================================
    go_end_mode = (
        frame_no > 1400
        or collected_total >= 7
        or remain_steps <= 300
    )

    # ==========================================================
    # 4. 终点主引导（只在终点模式）
    #    新增：障碍区绕墙宽恕
    # ==========================================================
    if go_end_mode:
        if progress_to_end:
            reward += 0.50
        else:
            # 在障碍密集区移动，允许试探绕墙
            if moved and in_obstacle_area:
                reward -= 0.05
            elif near_end and moved:
                reward -= 0.03
            else:
                reward -= 0.18

    # ==========================================================
    # 5. TSP阶段目标宝箱引导（加强）
    #    新增：障碍区减轻“距离没降”的惩罚
    # ==========================================================
    target_idx = None

    if (not go_end_mode) and (collected_total < len(Config.OPTIMAL_TREASURE_ORDER)):
        target_idx = Config.OPTIMAL_TREASURE_ORDER[collected_total]

        if cur_t_status[target_idx] == 1:
            cur_target_dist = float(cur_t_dists[target_idx])
            next_target_dist = float(next_t_dists[target_idx])

            if next_target_dist < cur_target_dist:
                reward += 0.50
            elif next_target_dist > cur_target_dist:
                # 障碍区允许短期绕远
                reward -= 0.08 if in_obstacle_area else 0.18

            # 防死磕目标宝箱：
            # 若既没接近目标，也没接近终点，在障碍区里减轻惩罚
            if (next_target_dist >= cur_target_dist) and (next_end_dist >= cur_end_dist):
                reward -= 0.10 if in_obstacle_area else 0.25

    # ==========================================================
    # 6. 机会宝箱（恢复）
    # ==========================================================
    if (not go_end_mode) and (len(available_idx) > 0):
        nearest_idx = available_idx[np.argmin(cur_t_dists[available_idx])]
        nearest_dist = float(cur_t_dists[nearest_idx])
        next_nearest_dist = float(next_t_dists[nearest_idx])

        if nearest_dist <= opportunity_th:
            if (target_idx is None) or (nearest_idx != target_idx):
                if next_nearest_dist < nearest_dist:
                    reward += 0.08

    # ==========================================================
    # 7. 探索奖励（随时间衰减）
    # ==========================================================
    explore_decay = max(0.15, 1.0 - frame_no / 1500.0)
    explore_bonus = 0.20 * explore_decay

    visit_value = None
    if moved:
        dr, dc = next_row - cur_row, next_col - cur_col
        visited_idx = 215 + (2 + dr) * 5 + (2 + dc)

        if 215 <= visited_idx < 240:
            visit_value = float(cur_feature[visited_idx])

            if visit_value <= 0.0:
                reward += explore_bonus

    # ==========================================================
    # 8. 防撞墙 / 防横跳
    #    新增：障碍区减轻旧格子惩罚，允许合理回头绕墙
    # ==========================================================
    if not moved:
        reward -= 0.8
        if just_got_treasure:
            reward -= 0.3
    else:
        if visit_value is not None and visit_value > 0.3:
            # 默认旧格子惩罚
            old_cell_penalty = 0.20 if in_obstacle_area else 0.35

            # 终点非常近时，如果走旧格子且没接近终点
            if go_end_mode and very_near_end and (not progress_to_end):
                old_cell_penalty = 0.45 if in_obstacle_area else 0.75

            # 吃完宝箱后收紧宽恕，但仍允许合理离开
            if just_got_treasure:
                old_cell_penalty *= 0.7

            reward -= old_cell_penalty

    # 吃到宝箱后鼓励尽快离开当前位置，减少局部打转
    if just_got_treasure and moved:
        if visit_value is not None and visit_value <= 0.3:
            reward += 0.08

    # ==========================================================
    # 9. 终点附近“找入口”奖励
    # ==========================================================
    if go_end_mode and near_end and moved and (not progress_to_end):
        if visit_value is not None and visit_value <= 0.0:
            reward += 0.20
        elif visit_value is not None and visit_value > 0.3:
            reward -= 0.15

    # ==========================================================
    # 10. 终局奖励 / 惩罚
    # ==========================================================
    if terminated and not truncated:
        reward += 150.0

    if truncated and not terminated:
        reward -= 100.0

    return float(reward)