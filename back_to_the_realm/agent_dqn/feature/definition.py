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
from agent_dqn.conf.conf import Config


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

    return dist <= Config.BUMP_DISTANCE_THRESHOLD


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


def _collect_reward_context(remain_info, next_remain_info, prev_obs_data, curr_obs_data, terminated):
    # Current and previous world positions.
    curr_pos = curr_obs_data.frame_state.heroes[0].pos
    prev_pos = prev_obs_data.frame_state.heroes[0].pos

    curr_pos_x, curr_pos_z = curr_pos.x, curr_pos.z
    prev_pos_x, prev_pos_z = prev_pos.x, prev_pos.z

    # Distances and counters from remain info.
    end_dist = next_remain_info.get("end_pos").l2_distance
    prev_end_dist = remain_info.get("end_pos").l2_distance

    treasure_dists = [pos.grid_distance if pos.grid_distance > 0 else -1 for pos in next_remain_info.get("treasure_pos")]
    prev_treasure_dists = [pos.grid_distance if pos.grid_distance > 0 else -1 for pos in remain_info.get("treasure_pos")]

    treasure_count = next_remain_info.get("treasure_count")
    treasure_collected_count = next_remain_info.get("treasure_collected_count")
    prev_treasure_collected_count = remain_info.get("treasure_collected_count")

    prev_visible_treasure_idx = {i for i, d in enumerate(prev_treasure_dists) if d > 0}
    curr_visible_treasure_idx = {i for i, d in enumerate(treasure_dists) if d > 0}

    prev_topk_seen_treasure_avg_dist = remain_info.get("topk_seen_treasure_avg_dist", -1.0)
    curr_topk_seen_treasure_avg_dist = next_remain_info.get("topk_seen_treasure_avg_dist", -1.0)
    prev_topk_seen_treasure_ids = set(remain_info.get("topk_seen_treasure_ids", []))
    curr_topk_seen_treasure_ids = set(next_remain_info.get("topk_seen_treasure_ids", []))
    curr_active_treasure_ids = set(next_remain_info.get("active_treasure_ids", set()))

    vision_coverage_ratio = float(next_remain_info.get("vision_coverage_ratio", 0.0))
    prev_vision_coverage_ratio = float(remain_info.get("vision_coverage_ratio", 0.0))
    newly_seen_grid_count = int(next_remain_info.get("newly_seen_grid_count", 0))
    frontier_dist_before = float(remain_info.get("frontier_dist_after", -1.0))
    frontier_dist_after = float(next_remain_info.get("frontier_dist_after", -1.0))
    lower_half_coverage_before = float(remain_info.get("lower_half_coverage_ratio", 0.0))
    lower_half_coverage_after = float(next_remain_info.get("lower_half_coverage_ratio", 0.0))
    right_lower_coverage_before = float(remain_info.get("right_lower_coverage_ratio", 0.0))
    right_lower_coverage_after = float(next_remain_info.get("right_lower_coverage_ratio", 0.0))
    distance_10_steps = float(next_remain_info.get("distance_10_steps", -1.0))
    min_treasure_before_finish = int(getattr(Config, "MIN_TREASURE_BEFORE_FINISH", 9))

    phase_threshold = float(Config.VISION_COVERAGE_PHASE1_THRESHOLD)
    final_reward_coverage_threshold = float(Config.FINAL_REWARD_COVERAGE_THRESHOLD)
    lower_half_threshold = float(getattr(Config, "LOWER_HALF_COVERAGE_THRESHOLD", 0.65))
    final_reward_enabled = (
        vision_coverage_ratio > final_reward_coverage_threshold and len(curr_topk_seen_treasure_ids) == 0
        and lower_half_coverage_after >= lower_half_threshold
        and treasure_collected_count >= min_treasure_before_finish
    )

    exploration_phase = int(next_remain_info.get("exploration_phase", 1 if vision_coverage_ratio < phase_threshold else 2))
    is_exploration_phase = exploration_phase == 1

    objective_reward_scale = 1.0
    if is_exploration_phase:
        objective_reward_scale = float(Config.PHASE1_OBJECTIVE_REWARD_SCALE)

    vision_exploration_scale = float(Config.PHASE2_VISION_EXPLORATION_SCALE)
    if is_exploration_phase:
        vision_exploration_scale = float(Config.PHASE1_VISION_EXPLORATION_SCALE)

    reward_win = 0
    if terminated and final_reward_enabled:
        reward_win = Config.FINAL_REWARD_BASE + Config.FINAL_REWARD_TREASURE_BONUS * treasure_collected_count

    return {
        "curr_pos_x": curr_pos_x,
        "curr_pos_z": curr_pos_z,
        "prev_pos_x": prev_pos_x,
        "prev_pos_z": prev_pos_z,
        "end_dist": end_dist,
        "prev_end_dist": prev_end_dist,
        "treasure_dists": treasure_dists,
        "prev_treasure_dists": prev_treasure_dists,
        "treasure_count": treasure_count,
        "treasure_collected_count": treasure_collected_count,
        "prev_treasure_collected_count": prev_treasure_collected_count,
        "prev_visible_treasure_idx": prev_visible_treasure_idx,
        "curr_visible_treasure_idx": curr_visible_treasure_idx,
        "prev_topk_seen_treasure_avg_dist": prev_topk_seen_treasure_avg_dist,
        "curr_topk_seen_treasure_avg_dist": curr_topk_seen_treasure_avg_dist,
        "prev_topk_seen_treasure_ids": prev_topk_seen_treasure_ids,
        "curr_active_treasure_ids": curr_active_treasure_ids,
        "prev_vision_coverage_ratio": prev_vision_coverage_ratio,
        "vision_coverage_ratio": vision_coverage_ratio,
        "newly_seen_grid_count": newly_seen_grid_count,
        "frontier_dist_before": frontier_dist_before,
        "frontier_dist_after": frontier_dist_after,
        "lower_half_coverage_before": lower_half_coverage_before,
        "lower_half_coverage_after": lower_half_coverage_after,
        "right_lower_coverage_before": right_lower_coverage_before,
        "right_lower_coverage_after": right_lower_coverage_after,
        "distance_10_steps": distance_10_steps,
        "min_treasure_before_finish": min_treasure_before_finish,
        "objective_reward_scale": objective_reward_scale,
        "vision_exploration_scale": vision_exploration_scale,
        "reward_win": reward_win,
    }


def _compute_end_distance_reward(prev_end_dist, end_dist):
    if prev_end_dist > 0 and end_dist > 0:
        end_progress = prev_end_dist - end_dist
        return np.clip(
            end_progress / Config.END_DISTANCE_REWARD_SCALE,
            Config.END_DISTANCE_REWARD_MIN,
            Config.END_DISTANCE_REWARD_MAX,
        )
    return 0


def _compute_treasure_rewards(ctx):
    reward_treasure_dist = 0
    reward_treasure = 0
    reward_new_visible_treasure = 0
    reward_topk_seen_treasure = 0

    is_treasures_remain = ctx["treasure_collected_count"] < ctx["treasure_count"]

    if is_treasures_remain:
        visible_dists = [d for d in ctx["treasure_dists"] if d > 0]
        if visible_dists:
            min_dist = min(visible_dists)
            min_index = ctx["treasure_dists"].index(min_dist)
            prev_min_dist = ctx["prev_treasure_dists"][min_index]
            if prev_min_dist > 0:
                treasure_progress = prev_min_dist - min_dist
                reward_treasure_dist = np.clip(
                    treasure_progress / Config.TREASURE_DISTANCE_REWARD_SCALE,
                    Config.TREASURE_DISTANCE_REWARD_MIN,
                    Config.TREASURE_DISTANCE_REWARD_MAX,
                )
            elif min_dist > 0:
                reward_treasure_dist = Config.TREASURE_DISTANCE_DISCOVERY_BONUS

    collected_delta = ctx["treasure_collected_count"] - ctx["prev_treasure_collected_count"]
    if collected_delta > 0:
        reward_treasure = Config.TREASURE_COLLECTED_REWARD_BASE + Config.TREASURE_COLLECTED_REWARD_PER_CHEST * collected_delta

    newly_visible_count = len(ctx["curr_visible_treasure_idx"] - ctx["prev_visible_treasure_idx"])
    if newly_visible_count > 0:
        reward_new_visible_treasure = min(
            Config.NEW_VISIBLE_TREASURE_REWARD_MAX,
            Config.NEW_VISIBLE_TREASURE_REWARD_PER_CHEST * newly_visible_count,
        )

    if ctx["prev_topk_seen_treasure_avg_dist"] > 0 and ctx["curr_topk_seen_treasure_avg_dist"] > 0:
        topk_progress = ctx["prev_topk_seen_treasure_avg_dist"] - ctx["curr_topk_seen_treasure_avg_dist"]
        reward_topk_seen_treasure += np.clip(
            topk_progress / Config.TOPK_TREASURE_PROGRESS_REWARD_SCALE,
            Config.TOPK_TREASURE_PROGRESS_REWARD_MIN,
            Config.TOPK_TREASURE_PROGRESS_REWARD_MAX,
        )

    if collected_delta > 0:
        removed_topk_ids = ctx["prev_topk_seen_treasure_ids"] - ctx["curr_active_treasure_ids"]
        if removed_topk_ids:
            reward_topk_seen_treasure += min(
                Config.TOPK_TREASURE_COLLECTED_REWARD_MAX,
                Config.TOPK_TREASURE_COLLECTED_REWARD_PER_CHEST * len(removed_topk_ids),
            )

    return reward_treasure_dist, reward_treasure, reward_new_visible_treasure, reward_topk_seen_treasure


def _compute_buff_rewards(remain_info, next_remain_info):
    reward_buff_dist = 0
    reward_buff = 0

    buff_dist = next_remain_info.get("buff_pos").grid_distance
    prev_buff_dist = remain_info.get("buff_pos").grid_distance

    if buff_dist > 0 and prev_buff_dist > 0:
        buff_progress = prev_buff_dist - buff_dist
        reward_buff_dist = np.clip(
            buff_progress / Config.BUFF_DISTANCE_REWARD_SCALE,
            Config.BUFF_DISTANCE_REWARD_MIN,
            Config.BUFF_DISTANCE_REWARD_MAX,
        )

    if prev_buff_dist > 0 and buff_dist < 0 and prev_buff_dist <= Config.BUFF_COLLECT_DISTANCE_THRESHOLD:
        reward_buff = Config.BUFF_COLLECT_REWARD

    return reward_buff_dist, reward_buff


def _compute_post_treasure_end_penalty(ctx):
    reward_post_treasure_end = 0
    if ctx["treasure_count"] > 0 and ctx["treasure_collected_count"] >= ctx["treasure_count"]:
        if ctx["prev_end_dist"] > 0 and ctx["end_dist"] > 0:
            end_progress_after_all = ctx["prev_end_dist"] - ctx["end_dist"]
            if end_progress_after_all <= 0:
                reward_post_treasure_end = np.clip(
                    end_progress_after_all / Config.POST_TREASURE_END_PROGRESS_SCALE
                    - Config.POST_TREASURE_END_PROGRESS_OFFSET,
                    Config.POST_TREASURE_END_PENALTY_MIN,
                    Config.POST_TREASURE_END_PENALTY_MAX,
                )
    return reward_post_treasure_end


def _compute_move_style_reward(curr_pos_x, curr_pos_z, prev_pos_x, prev_pos_z):
    reward_move_style = 0

    curr_grid_x, curr_grid_z = convert_pos_to_grid_pos(curr_pos_x, curr_pos_z)
    prev_grid_x, prev_grid_z = convert_pos_to_grid_pos(prev_pos_x, prev_pos_z)

    move_dx = abs(curr_grid_x - prev_grid_x)
    move_dz = abs(curr_grid_z - prev_grid_z)
    if move_dx + move_dz > 0:
        straight_reward = Config.PREFER_STRAIGHT_REWARD
        diagonal_reward = Config.PREFER_DIAGONAL_REWARD
        reward_move_style = straight_reward if (move_dx == 0 or move_dz == 0) else diagonal_reward

    return reward_move_style, curr_grid_x, curr_grid_z


def _compute_small_loop_penalty(remain_info, curr_grid_x, curr_grid_z):
    recent_position_map = remain_info.get("recent_position_map")
    loop_radius = int(Config.SMALL_LOOP_RADIUS)
    loop_threshold = int(Config.SMALL_LOOP_VISIT_THRESHOLD)
    loop_scale = float(Config.SMALL_LOOP_PENALTY_SCALE)

    local_visit_count = 0
    for (x, z), visit in recent_position_map.items():
        if abs(x - curr_grid_x) <= loop_radius and abs(z - curr_grid_z) <= loop_radius:
            local_visit_count += visit

    if local_visit_count > loop_threshold:
        return -min(Config.SMALL_LOOP_MAX_PENALTY, loop_scale * (local_visit_count - loop_threshold))
    return 0


def _compute_flicker_reward(ctx, prev_obs_data, curr_obs_data, is_bump):
    reward_flicker = 0
    curr_talent_cd = curr_obs_data.frame_state.heroes[0].talent.status
    prev_talent_cd = prev_obs_data.frame_state.heroes[0].talent.status

    if prev_talent_cd == 1 and curr_talent_cd == 0:
        move_dist = ((ctx["curr_pos_x"] - ctx["prev_pos_x"]) ** 2 + (ctx["curr_pos_z"] - ctx["prev_pos_z"]) ** 2) ** 0.5
        end_progress = ctx["prev_end_dist"] - ctx["end_dist"] if (ctx["prev_end_dist"] > 0 and ctx["end_dist"] > 0) else 0

        if is_bump or move_dist < Config.FLICKER_BAD_DIST_THRESHOLD:
            reward_flicker = Config.FLICKER_BUMP_PENALTY
        elif move_dist > Config.FLICKER_GOOD_DIST_THRESHOLD and end_progress > 0:
            reward_flicker = Config.FLICKER_LONG_MOVE_REWARD
        elif end_progress > 0:
            reward_flicker = Config.FLICKER_PROGRESS_REWARD
        else:
            reward_flicker = Config.FLICKER_NO_PROGRESS_PENALTY

    return reward_flicker


def _compute_exploration_rewards(ctx, remain_info, curr_pos_x, curr_pos_z, frontier_dist_before, frontier_dist_after):
    recent_position_map = remain_info.get("recent_position_map")
    arrival_position_map = remain_info.get("arrival_position_map", {})
    hero_grid_pos = convert_pos_to_grid_pos(curr_pos_x, curr_pos_z)

    visit_count = arrival_position_map.get((hero_grid_pos[0], hero_grid_pos[1]), 1)
    if visit_count <= 1:
        reward_exploration = Config.GLOBAL_FIRST_VISIT_REWARD
    else:
        recent_pass_times = recent_position_map.get((hero_grid_pos[0], hero_grid_pos[1]), 0)
        reward_exploration = max(
            -Config.GLOBAL_REPEAT_VISIT_PENALTY_SCALE * (visit_count - 1)
            - Config.EXPLORATION_REPEAT_GRID_PENALTY_SCALE * recent_pass_times,
            -(Config.GLOBAL_REPEAT_VISIT_PENALTY_MAX + Config.EXPLORATION_REPEAT_GRID_PENALTY_MAX),
        )

    reward_vision_exploration = 0
    for threshold in getattr(Config, "COVERAGE_MILESTONE_THRESHOLDS", ()):
        if ctx["prev_vision_coverage_ratio"] < threshold <= ctx["vision_coverage_ratio"]:
            reward_vision_exploration += float(Config.COVERAGE_MILESTONE_REWARD)

    reward_frontier_progress = 0
    if frontier_dist_before >= 0 and frontier_dist_after >= 0:
        frontier_progress = frontier_dist_before - frontier_dist_after
        reward_frontier_progress = np.clip(
            frontier_progress * Config.FRONTIER_PROGRESS_REWARD_SCALE,
            Config.FRONTIER_PROGRESS_REWARD_MIN,
            Config.FRONTIER_PROGRESS_REWARD_MAX,
        )

    return reward_exploration, reward_vision_exploration, reward_frontier_progress


def _compute_lower_half_coverage_reward(ctx):
    threshold = float(getattr(Config, "LOWER_HALF_COVERAGE_THRESHOLD", 0.65))
    if ctx["lower_half_coverage_before"] < threshold <= ctx["lower_half_coverage_after"]:
        return float(getattr(Config, "LOWER_HALF_COVERAGE_ONCE_REWARD", 1.2))
    return 0


def _compute_right_lower_coverage_reward(ctx):
    threshold = float(getattr(Config, "RIGHT_LOWER_COVERAGE_TARGET", 0.35))
    if ctx["right_lower_coverage_before"] < threshold <= ctx["right_lower_coverage_after"]:
        return float(getattr(Config, "RIGHT_LOWER_COVERAGE_ONCE_REWARD", 1.0))
    return 0


def _compute_stall_penalty(ctx):
    distance_10_steps = ctx["distance_10_steps"]
    if distance_10_steps < 0:
        return 0

    if distance_10_steps < Config.STALL_DISTANCE_THRESHOLD:
        return -Config.STALL_PENALTY
    return 0


def _compute_early_finish_penalty(ctx, terminated):
    min_treasure_before_finish = ctx["min_treasure_before_finish"]
    treasure_shortage = max(0, min_treasure_before_finish - ctx["treasure_collected_count"])
    if treasure_shortage <= 0:
        return 0

    reward_early_finish = 0
    if ctx["prev_end_dist"] > 0 and ctx["end_dist"] > 0:
        end_progress = ctx["prev_end_dist"] - ctx["end_dist"]
        if end_progress > 0:
            reward_early_finish += np.clip(
                -end_progress / Config.END_APPROACH_WITH_FEW_TREASURES_SCALE,
                Config.END_APPROACH_WITH_FEW_TREASURES_MIN,
                Config.END_APPROACH_WITH_FEW_TREASURES_MAX,
            )

    if terminated:
        reward_early_finish += max(
            Config.EARLY_FINISH_PENALTY_MIN,
            Config.EARLY_FINISH_PENALTY_BASE + Config.EARLY_FINISH_PENALTY_PER_MISSING * treasure_shortage,
        )

    return reward_early_finish


def _build_reward_weights(objective_reward_scale, vision_exploration_scale):
    return {
        "reward_end_dist": Config.REWARD_WEIGHT_END_DIST * objective_reward_scale,
        "reward_win": Config.REWARD_WEIGHT_WIN * objective_reward_scale,
        "reward_buff_dist": Config.REWARD_WEIGHT_BUFF_DIST,
        "reward_buff": Config.REWARD_WEIGHT_BUFF,
        "reward_treasure_dists": Config.REWARD_WEIGHT_TREASURE_DIST,
        "reward_treasure": Config.REWARD_WEIGHT_TREASURE,
        "reward_new_visible_treasure": Config.REWARD_WEIGHT_NEW_VISIBLE_TREASURE,
        "reward_topk_seen_treasure": Config.REWARD_WEIGHT_TOPK_SEEN_TREASURE,
        "reward_flicker": Config.REWARD_WEIGHT_FLICKER,
        "reward_post_treasure_end": Config.REWARD_WEIGHT_POST_TREASURE_END,
        "reward_move_style": Config.REWARD_WEIGHT_MOVE_STYLE,
        "reward_small_loop": Config.REWARD_WEIGHT_SMALL_LOOP,
        "reward_step": Config.REWARD_WEIGHT_STEP,
        "reward_bump": Config.REWARD_WEIGHT_BUMP,
        "reward_memory": Config.REWARD_WEIGHT_MEMORY,
        "reward_exploration": Config.REWARD_WEIGHT_EXPLORATION * vision_exploration_scale,
        "reward_vision_exploration": Config.REWARD_WEIGHT_VISION_EXPLORATION * vision_exploration_scale,
        "reward_frontier_progress": Config.REWARD_WEIGHT_FRONTIER_PROGRESS * vision_exploration_scale,
        "reward_lower_half_coverage": Config.REWARD_WEIGHT_LOWER_HALF_COVERAGE * vision_exploration_scale,
        "reward_right_lower_coverage": Config.REWARD_WEIGHT_RIGHT_LOWER_COVERAGE * vision_exploration_scale,
        "reward_early_finish": Config.REWARD_WEIGHT_EARLY_FINISH,
        "reward_stall": Config.REWARD_WEIGHT_STALL,
    }


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
    # Keep function signature stable for framework compatibility.
    _ = (frame_no, score, truncated)

    ctx = _collect_reward_context(remain_info, _remain_info, obs_data, _obs_data, terminated)

    reward_end_dist = _compute_end_distance_reward(ctx["prev_end_dist"], ctx["end_dist"])
    reward_win = ctx["reward_win"]

    (
        reward_treasure_dist,
        reward_treasure,
        reward_new_visible_treasure,
        reward_topk_seen_treasure,
    ) = _compute_treasure_rewards(ctx)

    reward_buff_dist, reward_buff = _compute_buff_rewards(remain_info, _remain_info)

    reward_step = Config.STEP_REWARD_BASE
    reward_post_treasure_end = _compute_post_treasure_end_penalty(ctx)

    memory_map = remain_info.get("memory_map")
    reward_memory = float(memory_map[len(memory_map) // 2])

    reward_move_style, curr_grid_x, curr_grid_z = _compute_move_style_reward(
        ctx["curr_pos_x"],
        ctx["curr_pos_z"],
        ctx["prev_pos_x"],
        ctx["prev_pos_z"],
    )

    reward_small_loop = _compute_small_loop_penalty(remain_info, curr_grid_x, curr_grid_z)

    is_bump = bump(ctx["curr_pos_x"], ctx["curr_pos_z"], ctx["prev_pos_x"], ctx["prev_pos_z"])
    reward_bump = 1 if is_bump else 0

    reward_flicker = _compute_flicker_reward(ctx, obs_data, _obs_data, is_bump)

    reward_exploration, reward_vision_exploration, reward_frontier_progress = _compute_exploration_rewards(
        ctx,
        remain_info,
        ctx["curr_pos_x"],
        ctx["curr_pos_z"],
        ctx["frontier_dist_before"],
        ctx["frontier_dist_after"],
    )
    reward_lower_half_coverage = _compute_lower_half_coverage_reward(ctx)
    reward_right_lower_coverage = _compute_right_lower_coverage_reward(ctx)
    reward_early_finish = _compute_early_finish_penalty(ctx, terminated)
    reward_stall = _compute_stall_penalty(ctx)

    reward_weight = _build_reward_weights(ctx["objective_reward_scale"], ctx["vision_exploration_scale"])

    reward = [
        reward_end_dist * reward_weight["reward_end_dist"],
        reward_win * reward_weight["reward_win"],
        reward_buff_dist * reward_weight["reward_buff_dist"],
        reward_buff * reward_weight["reward_buff"],
        reward_treasure_dist * reward_weight["reward_treasure_dists"],
        reward_treasure * reward_weight["reward_treasure"],
        reward_new_visible_treasure * reward_weight["reward_new_visible_treasure"],
        reward_topk_seen_treasure * reward_weight["reward_topk_seen_treasure"],
        reward_flicker * reward_weight["reward_flicker"],
        reward_post_treasure_end * reward_weight["reward_post_treasure_end"],
        reward_move_style * reward_weight["reward_move_style"],
        reward_small_loop * reward_weight["reward_small_loop"],
        reward_step * reward_weight["reward_step"],
        reward_bump * reward_weight["reward_bump"],
        reward_memory * reward_weight["reward_memory"],
        reward_exploration * reward_weight["reward_exploration"],
        reward_vision_exploration * reward_weight["reward_vision_exploration"],
        reward_frontier_progress * reward_weight["reward_frontier_progress"],
        reward_lower_half_coverage * reward_weight["reward_lower_half_coverage"],
        reward_right_lower_coverage * reward_weight["reward_right_lower_coverage"],
        reward_early_finish * reward_weight["reward_early_finish"],
        reward_stall * reward_weight["reward_stall"],
    ]

    return (
        sum(reward),
        is_bump,
        reward_end_dist * reward_weight["reward_end_dist"],
        reward_exploration * reward_weight["reward_exploration"]
        + reward_vision_exploration * reward_weight["reward_vision_exploration"]
        + reward_frontier_progress * reward_weight["reward_frontier_progress"]
        + reward_lower_half_coverage * reward_weight["reward_lower_half_coverage"]
        + reward_right_lower_coverage * reward_weight["reward_right_lower_coverage"],
        reward_treasure_dist * reward_weight["reward_treasure_dists"],
        reward_treasure * reward_weight["reward_treasure"] + reward_early_finish * reward_weight["reward_early_finish"],
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
