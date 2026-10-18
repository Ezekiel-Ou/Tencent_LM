#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2025 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""


import numpy as np
from kaiwu_agent.utils.common_func import attached, create_cls
from agent_target_dqn.conf.conf import Config


def bump(a1, b1, a2, b2):
    """
    This function is used to determine whether the game hits a wall.
        - There will be no bump in the first frame.
        - Starting from the second frame, if the moving distance is less than 500, it will be considered as hitting a wall.

    该函数用于判断是否撞墙
        - 第一帧不会bump
        - 第二帧开始, 如果移动距离小于500则视为撞墙
    """
    if a2 == -1 and b2 == -1:
        return False
    if a1 == -1 and b1 == -1:
        return False

    dist = ((a1 - a2) ** 2 + (b1 - b2) ** 2) ** (0.5)

    return dist <= 500


# The create_cls function is used to dynamically create a class. The first parameter of the function is the type name,
# and the remaining parameters are the attributes of the class, which should have a default value of None.
# create_cls函数用于动态创建一个类，函数第一个参数为类型名称，剩余参数为类的属性，属性默认值应设为None
ObsData = create_cls(
    "ObsData",
    feature=None,
    legal_act=None,
)


ActData = create_cls(
    "ActData",
    move_dir=None,
    use_talent=None,
)


SampleData = create_cls(
    "SampleData",
    obs=None,
    _obs=None,
    obs_legal=None,
    _obs_legal=None,
    act=None,
    rew=None,
    ret=None,
    done=None,
)


def convert_pos_to_grid_pos(x, z):
    """
    Convert the position 'pos' into grid-based coordinates
    将pos转换为珊格化后坐标

    Args:
        x (float): x
        z (float): z

    Returns:
        _type_: tuple
    """
    x = (x + 2250) // 500
    z = (z + 5250) // 500

    # This step is necessary in order to be aligned with the order of json files
    # 这一步是为了与json文件的顺序保持一致
    x, z = z, x

    return x, z


def _relative_distance(relative_pos, field="grid_distance"):
    if relative_pos is None:
        return -1
    return float(getattr(relative_pos, field, -1))


def _visible_positions(relative_pos_list):
    visible_positions = []
    for index, relative_pos in enumerate(relative_pos_list or []):
        dist = _relative_distance(relative_pos)
        if dist > 0:
            visible_positions.append((dist, index))
    return visible_positions


def _active_positions(relative_pos_list):
    active_positions = []
    for index, relative_pos in enumerate(relative_pos_list or []):
        if relative_pos is not None and getattr(relative_pos, "direction", 0) != 0:
            active_positions.append((float(getattr(relative_pos, "l2_distance", 1e9)), index))
    return active_positions


def _progress_reward(prev_dist, curr_dist):
    if curr_dist < prev_dist:
        return 1.0
    if curr_dist > prev_dist:
        return -1.0
    return -0.1


def _count_from_game_info(obs_data, field):
    game_info = getattr(obs_data, "game_info", None)
    return int(getattr(game_info, field, 0)) if game_info is not None else 0


def _score_value(score, field, default=0.0):
    if score is None:
        return default
    return float(getattr(score, field, default))


def _treasure_collect_reward(prev_count, curr_count):
    reward = 0.0
    for collected_index in range(prev_count + 1, curr_count + 1):
        reward += 16.0 + 2.0 * collected_index
    return reward


def _treasure_milestone_reward(prev_count, curr_count, treasure_count):
    reward = 0.0
    milestones = [(4, 12.0), (6, 22.0)]
    if treasure_count > 0:
        milestones.append((treasure_count, 42.0))

    used_thresholds = set()
    for threshold, value in milestones:
        if threshold in used_thresholds:
            continue
        used_thresholds.add(threshold)
        if prev_count < threshold <= curr_count:
            reward += value
    return reward


FULL_VIEW_COVERAGE_RATIO = 0.70
MIN_TREASURE_TO_END = 5
FORCE_END_STAGE_FRAME = 700


def _treasure_end_gate(treasure_count):
    if treasure_count is None:
        return MIN_TREASURE_TO_END
    return min(MIN_TREASURE_TO_END, max(0, int(treasure_count)))


def _policy_stage(frame_no, treasure_collected_count, discovered_treasure_count, view_coverage, treasure_count):
    del discovered_treasure_count
    if frame_no is not None and frame_no >= FORCE_END_STAGE_FRAME:
        return 3
    if view_coverage >= FULL_VIEW_COVERAGE_RATIO and treasure_collected_count >= _treasure_end_gate(treasure_count):
        return 3
    if view_coverage >= FULL_VIEW_COVERAGE_RATIO:
        return 2
    return 1


def reward_shaping(
    frame_no,
    score,
    terminated,
    truncated,
    remain_info,
    _remain_info,
    obs_data,
    _obs_data,
):
    pos = _obs_data.frame_state.heroes[0].pos
    curr_pos_x, curr_pos_z = pos.x, pos.z
    prev_pos = obs_data.frame_state.heroes[0].pos
    prev_pos_x, prev_pos_z = prev_pos.x, prev_pos.z

    is_bump = bump(curr_pos_x, curr_pos_z, prev_pos_x, prev_pos_z)

    end_dist = _relative_distance(_remain_info.get("end_pos"), "l2_distance")
    prev_end_dist = _relative_distance(remain_info.get("end_pos"), "l2_distance")
    buff_dist = _relative_distance(_remain_info.get("buff_pos"))
    prev_buff_dist = _relative_distance(remain_info.get("buff_pos"))
    treasure_positions = _remain_info.get("treasure_pos") or []
    prev_treasure_positions = remain_info.get("treasure_pos") or []

    treasure_count = _remain_info.get("treasure_count")
    treasure_collected_count = _remain_info.get("treasure_collected_count") or 0
    prev_treasure_collected_count = remain_info.get("treasure_collected_count") or 0
    discovered_treasure_count = max(
        int(_remain_info.get("discovered_treasure_count") or 0),
        treasure_collected_count,
    )
    prev_discovered_treasure_count = max(
        int(remain_info.get("discovered_treasure_count") or 0),
        prev_treasure_collected_count,
    )
    view_coverage = float(_remain_info.get("view_coverage") or 0.0)
    prev_view_coverage = float(remain_info.get("view_coverage") or 0.0)
    end_gate_count = _treasure_end_gate(treasure_count)
    stage = int(
        _remain_info.get("stage")
        or _policy_stage(frame_no, treasure_collected_count, discovered_treasure_count, view_coverage, treasure_count)
    )
    treasure_goal_count = max(discovered_treasure_count, end_gate_count)
    remaining_treasure_count = max(0, treasure_goal_count - treasure_collected_count)
    has_uncollected_treasure = treasure_collected_count < treasure_goal_count

    visible_treasures = _visible_positions(treasure_positions)
    prev_visible_treasures = _visible_positions(prev_treasure_positions)
    active_treasures = _active_positions(treasure_positions)
    has_visible_treasure = bool(visible_treasures) and has_uncollected_treasure
    has_reachable_buff = buff_dist > 0

    reward_treasure_dist = 0.0
    if has_visible_treasure:
        curr_dist, treasure_index = min(visible_treasures, key=lambda item: item[0])
        prev_dist = _relative_distance(prev_treasure_positions[treasure_index]) if treasure_index < len(prev_treasure_positions) else -1
        if prev_dist > 0:
            reward_treasure_dist = _progress_reward(prev_dist, curr_dist)
        else:
            reward_treasure_dist = 0.2

    reward_treasure_global = 0.0
    if not has_visible_treasure and active_treasures and has_uncollected_treasure:
        _, treasure_index = min(active_treasures, key=lambda item: item[0])
        curr_dist = _relative_distance(treasure_positions[treasure_index], "l2_distance")
        prev_dist = _relative_distance(prev_treasure_positions[treasure_index], "l2_distance") if treasure_index < len(prev_treasure_positions) else -1
        if prev_dist > 0 and curr_dist > 0:
            reward_treasure_global = _progress_reward(prev_dist, curr_dist)

    reward_treasure = _treasure_collect_reward(prev_treasure_collected_count, treasure_collected_count)
    reward_treasure_milestone = _treasure_milestone_reward(
        prev_treasure_collected_count,
        treasure_collected_count,
        treasure_goal_count,
    )

    reward_buff_dist = 0.0
    if has_reachable_buff and prev_buff_dist > 0:
        reward_buff_dist = _progress_reward(prev_buff_dist, buff_dist)

    reward_buff = 0.0
    buff_delta = _count_from_game_info(_obs_data, "buff_count") - _count_from_game_info(obs_data, "buff_count")
    if buff_delta > 0 or (prev_buff_dist > 0 and buff_dist < 0 and prev_buff_dist <= 3):
        reward_buff = 1.0

    reward_end_dist = 0.0
    reward_early_end_dist = 0.0
    if stage == 3 and prev_end_dist > 0 and end_dist > 0:
        reward_end_dist = _progress_reward(prev_end_dist, end_dist)
    elif stage != 3 and prev_end_dist > 0 and end_dist > 0:
        if end_dist < prev_end_dist and end_dist <= 45:
            reward_early_end_dist -= 0.6
        if end_dist <= 25:
            reward_early_end_dist -= 2.0 * (26.0 - min(end_dist, 26.0)) / 26.0

    reward_win = 0.0
    if terminated:
        if stage == 3 and treasure_collected_count >= end_gate_count:
            reward_win = 60.0 + 5.0 * treasure_collected_count + min(60.0, _score_value(score, "total_score") * 0.04)
        else:
            undiscovered_count = max(0, treasure_goal_count - discovered_treasure_count)
            missing_view_ratio = max(0.0, FULL_VIEW_COVERAGE_RATIO - view_coverage)
            missing_gate_count = max(0, end_gate_count - treasure_collected_count)
            reward_win = (
                -420.0
                - 60.0 * missing_gate_count
                - 28.0 * remaining_treasure_count
                - 12.0 * undiscovered_count
                - 220.0 * missing_view_ratio
                + 2.0 * treasure_collected_count
            )
    elif truncated:
        missing_view_ratio = max(0.0, FULL_VIEW_COVERAGE_RATIO - view_coverage)
        missing_gate_count = max(0, end_gate_count - treasure_collected_count)
        reward_win = (
            -300.0
            - 45.0 * missing_gate_count
            - 20.0 * remaining_treasure_count
            - 180.0 * missing_view_ratio
            + 2.0 * treasure_collected_count
        )

    reward_step = -0.0002
    reward_bump = -1.2 if is_bump else 0.0

    memory_map = _remain_info.get("memory_map") or remain_info.get("memory_map") or []
    memory_center = float(memory_map[len(memory_map) // 2]) if memory_map else 0.0
    reward_memory = 0.08 if memory_center < 0.2 else -0.03 * memory_center

    reward_exploration = 0.0
    recent_position_map = _remain_info.get("recent_position_map") or remain_info.get("recent_position_map") or {}
    arrival_position_map = _remain_info.get("arrival_position_map") or remain_info.get("arrival_position_map") or {}
    visible_position_map = _remain_info.get("visible_position_map") or {}
    prev_visible_position_map = remain_info.get("visible_position_map") or {}
    hero_grid_pos = convert_pos_to_grid_pos(curr_pos_x, curr_pos_z)
    hero_grid_key = (hero_grid_pos[0], hero_grid_pos[1])
    recent_pass_times = recent_position_map.get(hero_grid_key, 0)
    lifetime_pass_times = arrival_position_map.get(hero_grid_key, 0)
    if lifetime_pass_times <= 1:
        reward_exploration = 0.50
    elif lifetime_pass_times <= 3:
        reward_exploration = 0.20
    elif lifetime_pass_times <= 8:
        reward_exploration = 0.0
    else:
        reward_exploration = -min(0.015 * (lifetime_pass_times - 8), 0.08)
    if recent_pass_times > 4:
        reward_exploration -= min(0.008 * (recent_pass_times - 4), 0.04)
    visible_delta = max(0, len(visible_position_map) - len(prev_visible_position_map))
    if visible_delta == 0 and (lifetime_pass_times > 8 or recent_pass_times > 4):
        reward_exploration -= 0.05
    reward_vision = min(0.05 * visible_delta, 2.5)
    reward_coverage = max(0.0, view_coverage - prev_view_coverage) * 130.0
    reward_discovery = 28.0 * max(0, discovered_treasure_count - prev_discovered_treasure_count)

    reward_flicker = 0.0
    curr_talent_status = _obs_data.frame_state.heroes[0].talent.status
    prev_talent_status = obs_data.frame_state.heroes[0].talent.status
    if prev_talent_status == 1 and curr_talent_status == 0:
        move_dist = ((curr_pos_x - prev_pos_x) ** 2 + (curr_pos_z - prev_pos_z) ** 2) ** 0.5
        objective_progress = 0.0
        if has_visible_treasure and prev_visible_treasures:
            curr_min_treasure = min([dist for dist, _ in visible_treasures])
            prev_min_treasure = min([dist for dist, _ in prev_visible_treasures])
            objective_progress = prev_min_treasure - curr_min_treasure
        elif reward_treasure_global != 0:
            objective_progress = reward_treasure_global
        elif has_reachable_buff and prev_buff_dist > 0:
            objective_progress = prev_buff_dist - buff_dist
        elif stage == 1 and reward_exploration > 0:
            objective_progress = reward_exploration
        elif remaining_treasure_count <= 0 and prev_end_dist > 0 and end_dist > 0:
            objective_progress = prev_end_dist - end_dist

        if is_bump or move_dist < 1000:
            reward_flicker = -1.2
        elif objective_progress > 0 and move_dist >= 4000:
            reward_flicker = 1.8
        elif objective_progress > 0:
            reward_flicker = 0.8
        else:
            reward_flicker = -2.5

    if stage == 1:
        weighted_treasure_collect = reward_treasure * 1.0 + reward_treasure_milestone * 0.8
        weighted_treasure_dist = reward_treasure_dist * 0.6 + reward_treasure_global * 0.25
        weighted_exploration = reward_exploration * 2.2 + reward_memory + reward_vision + reward_coverage + reward_discovery
        weighted_end = reward_early_end_dist * 3.0
    elif stage == 2:
        weighted_treasure_collect = reward_treasure + reward_treasure_milestone
        weighted_treasure_dist = reward_treasure_dist * 2.2 + reward_treasure_global * 1.2
        weighted_exploration = reward_exploration * 0.25 + reward_memory * 0.5
        weighted_end = reward_early_end_dist * 2.0
    else:
        weighted_treasure_collect = reward_treasure + reward_treasure_milestone
        weighted_treasure_dist = reward_treasure_dist * 0.2 + reward_treasure_global * 0.1
        weighted_exploration = reward_exploration * 0.05 + reward_memory * 0.25
        weighted_end = reward_end_dist * 2.5
    weighted_buff = reward_buff * 12.0
    weighted_buff_dist = reward_buff_dist * 1.8
    weighted_flicker = reward_flicker * 1.2

    reward = [
        weighted_treasure_collect,
        weighted_treasure_dist,
        weighted_buff,
        weighted_buff_dist,
        weighted_end,
        reward_win,
        weighted_flicker,
        reward_step,
        reward_bump,
        weighted_exploration,
    ]

    return (
        sum(reward),
        is_bump,
        {
            "weighted_end": weighted_end,
            "weighted_exploration": weighted_exploration,
            "weighted_treasure_dist": weighted_treasure_dist,
            "weighted_treasure_collect": weighted_treasure_collect,
            "view_coverage": view_coverage,
            "discovered_treasure_count": discovered_treasure_count,
            "treasure_collected_count": treasure_collected_count,
        },
    )


@attached
def sample_process(list_game_data):
    return [SampleData(**i.__dict__) for i in list_game_data]


# SampleData <----> NumpyData
@attached
def SampleData2NumpyData(g_data):
    return np.hstack(
        (
            np.array(g_data.obs, dtype=np.float32),
            np.array(g_data._obs, dtype=np.float32),
            np.array(g_data.obs_legal, dtype=np.float32),
            np.array(g_data._obs_legal, dtype=np.float32),
            np.array(g_data.act, dtype=np.float32),
            np.array(g_data.rew, dtype=np.float32),
            np.array(g_data.ret, dtype=np.float32),
            np.array(g_data.done, dtype=np.float32),
        )
    )


@attached
def NumpyData2SampleData(s_data):
    obs_data_size = (2 * (Config.VIEW_SIZE) + 1) ** 2 * 4 + 404
    return SampleData(
        # Refer to the DESC_OBS_SPLIT configuration in config.py for dimension reference
        # 维度参考config.py 中的 DESC_OBS_SPLIT配置
        obs=s_data[:obs_data_size],
        _obs=s_data[obs_data_size : 2 * obs_data_size],
        obs_legal=s_data[-8:-6],
        _obs_legal=s_data[-6:-4],
        act=s_data[-4],
        rew=s_data[-3],
        ret=s_data[-2],
        done=s_data[-1],
    )
