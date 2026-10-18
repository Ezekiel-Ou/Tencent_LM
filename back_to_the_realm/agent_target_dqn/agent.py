#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2025 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""


import torch

torch.set_num_threads(1)
torch.set_num_interop_threads(1)

from collections import deque
from agent_target_dqn.feature.definition import ObsData
import math
import numpy as np
from kaiwu_agent.agent.base_agent import (
    BaseAgent,
    predict_wrapper,
    exploit_wrapper,
    learn_wrapper,
    save_model_wrapper,
    load_model_wrapper,
)
from kaiwu_agent.utils.common_func import attached
from agent_target_dqn.algorithm.algorithm import Algorithm
from arena_proto.back_to_the_realm.custom_pb2 import (
    RelativePosition,
    RelativeDirection,
)


def one_hot_encoding(grid_pos):
    """
    This function performs one_hot_encoding on the grid position features transmitted by proto and returns a vector of length 256.
        - The first 128 dimensions are the one-hot features of the x-axis
        - The last 128 dimensions are the one-hot features of the z-axis

    此函数将proto传输的网格位置特征进行one_hot_encoding处理, 返回一个长度为256的向量
        - 前128维是x轴的one-hot特征
        - 后128维是z轴的one-hot特征
    """
    one_hot_pos_x, one_hot_pos_z = np.zeros(128).tolist(), np.zeros(128).tolist()
    one_hot_pos_x[grid_pos.x], one_hot_pos_z[grid_pos.z] = 1, 1

    return one_hot_pos_x + one_hot_pos_z


def read_relative_position(rel_pos):
    """
    This function unpacks and processes the relative position features transmitted by proto, and returns a vector of length 9.
        - The first 8 dimensions are one-hot direction features
        - The last dimension is the distance feature

    此函数将proto传输的相对位置特征进行拆包并处理, 返回一个长度为9的向量
        - 前8维是one-hot的方向特征
        - 最后一维是距离特征
    """
    direction = [0] * 8
    if rel_pos.direction != RelativeDirection.RELATIVE_DIRECTION_NONE:
        direction[rel_pos.direction - 1] = 1

    grid_distance = 1 if rel_pos.grid_distance < 0 else rel_pos.grid_distance / (128 * 128)
    feature = direction + [grid_distance]
    return feature


def read_target_direction(target_pos_features):
    if not target_pos_features or max(target_pos_features[:8]) <= 0:
        return None
    return int(np.argmax(target_pos_features[:8]))


def read_grid_target_position(hero_grid_pos, target_grid_pos):
    rel_pos = RelativePosition()
    dx = float(target_grid_pos[0] - hero_grid_pos.x)
    dz = float(target_grid_pos[1] - hero_grid_pos.z)
    if dx == 0 and dz == 0:
        rel_pos.direction = RelativeDirection.RELATIVE_DIRECTION_NONE
    else:
        theta = math.atan2(dz, dx)
        if theta < 0:
            theta += 2 * math.pi
        rel_pos.direction = int(round(theta / (math.pi / 4))) % 8 + 1
    rel_pos.l2_distance = math.hypot(dx, dz)
    rel_pos.path_distance = 0
    rel_pos.grid_distance = 128 * 128
    return read_relative_position(rel_pos)


def exploration_target_features(memory_map, obstacle_map, local_visible_map=None):
    view_width = 51
    view_center = view_width // 2
    memory_grid = np.array(memory_map, dtype=np.float32).reshape(view_width, view_width)
    obstacle_grid = np.array(obstacle_map, dtype=np.float32).reshape(view_width, view_width)
    if local_visible_map is None:
        visible_grid = np.ones((view_width, view_width), dtype=np.float32)
    else:
        visible_grid = np.array(local_visible_map, dtype=np.float32).reshape(view_width, view_width)

    farthest_unseen = None
    sector_scores = []
    for x in range(view_width):
        for z in range(view_width):
            dx, dz = x - view_center, z - view_center
            distance = (dx * dx + dz * dz) ** 0.5
            if distance < 5 or obstacle_grid[x][z] <= 0:
                continue
            theta = np.arctan2(dz, dx)
            if theta < 0:
                theta += 2 * np.pi
            direction = int(round(theta / (np.pi / 4))) % 8
            while len(sector_scores) <= direction:
                sector_scores.append({"passable": 0, "frontier": 0.0, "low_memory": 0.0, "far_frontier": 0.0})

            unseen = 1.0 if visible_grid[x][z] <= 0 else 0.0
            low_memory = max(0.0, 1.0 - float(memory_grid[x][z]))
            far_weight = min(1.0, distance / view_center)
            if unseen > 0 and distance >= 8:
                unseen_score = distance * 2.0 + low_memory * 3.0 + far_weight
                if farthest_unseen is None or unseen_score > farthest_unseen[0]:
                    farthest_unseen = (unseen_score, direction)
            sector_scores[direction]["passable"] += 1
            sector_scores[direction]["frontier"] += unseen
            sector_scores[direction]["low_memory"] += low_memory * far_weight
            if distance >= 16:
                sector_scores[direction]["far_frontier"] += unseen

    if farthest_unseen is not None:
        _, best_direction = farthest_unseen
        direction_features = [0] * 8
        direction_features[best_direction] = 1
        return direction_features + [0.2]

    ranked_scores = []
    for direction, score in enumerate(sector_scores):
        if score["passable"] < 3:
            continue
        total_score = score["frontier"] * 3.0 + score["far_frontier"] * 2.0 + score["low_memory"] * 0.2
        ranked_scores.append((total_score, direction))

    if not ranked_scores:
        return [0] * 8 + [1]

    _, best_direction = max(ranked_scores, key=lambda item: item[0])
    direction_features = [0] * 8
    direction_features[best_direction] = 1
    return direction_features + [0.2]


FULL_VIEW_COVERAGE_RATIO = 0.70
MIN_TREASURE_TO_END = 5
FORCE_END_STAGE_FRAME = 700
STAGE1_BUFF_GRID_THRESHOLD = 12
STAGE2_BUFF_GRID_THRESHOLD = 10
BUFF_INSERT_RATIO = 0.5
DIRECTION_OFFSETS = (
    (1, 0),
    (1, 1),
    (0, 1),
    (-1, 1),
    (-1, 0),
    (-1, -1),
    (0, -1),
    (1, -1),
)


def treasure_end_gate(treasure_count):
    if treasure_count is None:
        return MIN_TREASURE_TO_END
    return min(MIN_TREASURE_TO_END, max(0, int(treasure_count)))


def policy_stage(frame_no, treasure_collected_count, discovered_treasure_count, view_coverage, treasure_count):
    del discovered_treasure_count
    if frame_no is not None and frame_no >= FORCE_END_STAGE_FRAME:
        return 3
    if view_coverage >= FULL_VIEW_COVERAGE_RATIO and treasure_collected_count >= treasure_end_gate(treasure_count):
        return 3
    if view_coverage >= FULL_VIEW_COVERAGE_RATIO:
        return 2
    return 1


def route_target_features(obstacle_map, target_map, fallback_features):
    view_width = 51
    center = view_width // 2
    obstacle_grid = np.array(obstacle_map, dtype=np.float32).reshape(view_width, view_width)
    target_grid = np.array(target_map, dtype=np.float32).reshape(view_width, view_width)

    if obstacle_grid[center, center] <= 0 or target_grid.max() <= 0:
        return fallback_features

    dist = np.full((view_width, view_width), -1, dtype=np.int32)
    queue = deque()
    for x, z in zip(*np.where(target_grid > 0)):
        if obstacle_grid[x, z] <= 0:
            continue
        dist[x, z] = 0
        queue.append((int(x), int(z)))

    while queue:
        x, z = queue.popleft()
        next_dist = dist[x, z] + 1
        for dx, dz in DIRECTION_OFFSETS:
            nx, nz = x + dx, z + dz
            if nx < 0 or nz < 0 or nx >= view_width or nz >= view_width:
                continue
            if dist[nx, nz] >= 0 or obstacle_grid[nx, nz] <= 0:
                continue
            dist[nx, nz] = next_dist
            queue.append((nx, nz))

    current_dist = dist[center, center]
    if current_dist <= 0:
        return fallback_features

    best_direction = None
    best_dist = current_dist
    for direction, (dx, dz) in enumerate(DIRECTION_OFFSETS):
        nx, nz = center + dx, center + dz
        if nx < 0 or nz < 0 or nx >= view_width or nz >= view_width:
            continue
        if dist[nx, nz] >= 0 and dist[nx, nz] < best_dist:
            best_dist = dist[nx, nz]
            best_direction = direction

    if best_direction is None:
        return fallback_features

    route_features = [0] * 8
    route_features[best_direction] = 1
    return route_features + [fallback_features[8]]


@attached
class Agent(BaseAgent):
    def __init__(self, agent_type="player", device=None, logger=None, monitor=None):
        self.agent_type = agent_type
        self.logger = logger
        self.algorithm = Algorithm(device, monitor)

    @predict_wrapper
    def predict(self, list_obs_data):
        return self.algorithm.predict_detail(list_obs_data, exploit_flag=False)

    @exploit_wrapper
    def exploit(self, list_obs_data):
        return self.algorithm.predict_detail(list_obs_data, exploit_flag=True)

    @learn_wrapper
    def learn(self, list_sample_data):
        self.algorithm.learn(list_sample_data)

    @save_model_wrapper
    def save_model(self, path=None, id="1"):
        # To save the model, it can consist of multiple files,
        # and it is important to ensure that each filename includes the "model.ckpt-id" field.
        # 保存模型, 可以是多个文件, 需要确保每个文件名里包括了model.ckpt-id字段
        model_file_path = f"{path}/model.ckpt-{str(id)}.pkl"

        # Copy the model's state dictionary to the CPU
        # 将模型的状态字典拷贝到CPU
        model_state_dict_cpu = {k: v.clone().cpu() for k, v in self.algorithm.model.state_dict().items()}
        torch.save(model_state_dict_cpu, model_file_path)

        self.logger.info(f"save model {model_file_path} successfully")

    @load_model_wrapper
    def load_model(self, path=None, id="1"):
        # When loading the model, you can load multiple files,
        # and it is important to ensure that each filename matches the one used during the save_model process.
        # 加载模型, 可以加载多个文件, 注意每个文件名需要和save_model时保持一致
        model_file_path = f"{path}/model.ckpt-{str(id)}.pkl"
        try:
            self.algorithm.model.load_state_dict(torch.load(model_file_path, map_location=self.algorithm.device))
        except (FileNotFoundError, RuntimeError) as exc:
            if str(id) != "latest":
                raise
            self.logger.info(f"model {model_file_path} can not be loaded, keep current model: {exc}")
            return
        self.logger.info(f"load model {model_file_path} successfully")

    def action_process(self, act_data):
        result = act_data.move_dir
        result += act_data.use_talent * 8
        return result

    def observation_process(self, raw_obs, preprocessor, state_env_info=None):
        """
        This function is an important feature processing function, mainly responsible for:
            - Parsing information in the raw data
            - Parsing preprocessed feature data
            - Processing the features and returning the processed feature vector
            - Concatenation of features
            - Annotation of legal actions
        Function inputs:
            - raw_obs: Preprocessed feature data
            - state_env_info: Environment information returned by the game
        Function outputs:
            - observation: Feature vector
            - legal_action: Annotation of legal actions

        该函数是特征处理的重要函数, 主要负责：
            - 解析原始数据里的信息
            - 解析预处理后的特征数据
            - 对特征进行处理, 并返回处理后的特征向量
            - 特征的拼接
            - 合法动作的标注
        函数的输入：
            - raw_obs: 预处理后的特征数据
            - state_env_info: 游戏返回的环境信息
        函数的输出：
            - observation: 特征向量
            - legal_action: 合法动作的标注
        """
        feature, legal_act = [], []

        # Unpack the preprocessed feature data according to the protocol
        # 对预处理后的特征数据按照协议进行解包
        (
            norm_pos,
            grid_pos,
            start_pos,
            end_pos,
            buff_pos,
            treasure_pos_list,
            obstacle_map,
            memory_map,
            treasure_map,
            end_map,
            recent_position_map,
            arrival_position_map,
            visible_position_map,
            local_visible_map,
            view_coverage,
            discovered_treasures,
            treasure_collected_count,
            treasure_count,
        ) = preprocessor.process(raw_obs)
        frame_no = None
        if raw_obs is not None:
            frame_state = getattr(raw_obs, "frame_state", None)
            frame_no = getattr(frame_state, "frame_no", None)
            if frame_no is None:
                frame_no = getattr(frame_state, "frameNo", None)
        if frame_no is None and state_env_info is not None:
            frame_no = getattr(state_env_info, "frame_no", None)
            if frame_no is None:
                frame_no = getattr(state_env_info, "frameNo", None)
        discovered_treasure_count = max(len(discovered_treasures), treasure_collected_count)
        stage = policy_stage(frame_no, treasure_collected_count, discovered_treasure_count, view_coverage, treasure_count)

        # Feature processing 1: One-hot encoding of the current position
        # 特征处理1：当前位置的one-hot编码
        one_hot_pos = one_hot_encoding(grid_pos)

        # Feature processing 2: Normalized position
        # 特征处理2：归一化位置
        norm_pos = [norm_pos.x, norm_pos.z]

        # Feature processing 3: Information about the current tactical target
        # 特征处理3：当前位置相对当前战术目标点位的信息
        if stage == 3:
            target_pos_features = read_relative_position(end_pos)
            target_pos_features = route_target_features(obstacle_map, end_map, target_pos_features)
            target_type = "end"
        else:
            target_pos_features = [0] * 8 + [1]
            target_type = "none"

        # Feature processing 4: Information about the current position relative to the treasure position
        # 特征处理4: 当前位置相对宝箱位置的信息
        treasure_pos_features = []
        for treasure_pos in treasure_pos_list:
            treasure_pos_features = treasure_pos_features + list(read_relative_position(treasure_pos))

        # Feature processing 5: Whether the buff is collectable
        # 特征处理5：buff是否可收集
        buff_availability = 0
        if raw_obs:
            for organ in raw_obs.frame_state.organs:
                if organ.sub_type == 2:
                    buff_availability = organ.status

        # Feature processing 6: Whether the flash skill can be used
        # 特征处理6：闪现技能是否可使用
        talent_availability = 0
        if raw_obs:
            talent_availability = raw_obs.frame_state.heroes[0].talent.status

        # Feature processing 7: Next tactical target.
        # Stage 1/2 target known treasures first, then nearby buff, then exploration; Stage 3 returns to the end.
        # 特征处理7：下一个战术目标。阶段1/2优先已知宝箱，其次近处buff，再探索；阶段3回终点。
        visible_treasures = [
            (treasure_pos.grid_distance, index)
            for index, treasure_pos in enumerate(treasure_pos_list)
            if treasure_pos.grid_distance > 0
        ]
        active_treasures = [
            (treasure_pos.l2_distance, index)
            for index, treasure_pos in enumerate(treasure_pos_list)
            if treasure_pos.direction != RelativeDirection.RELATIVE_DIRECTION_NONE
        ]
        known_treasures = [
            (
                math.hypot(float(pos[0] - grid_pos.x), float(pos[1] - grid_pos.z)),
                config_id,
                pos,
            )
            for config_id, pos in discovered_treasures.items()
        ]
        buff_dist = buff_pos.grid_distance
        if stage in (1, 2) and visible_treasures:
            nearest_treasure_dist, nearest_treasure_index = min(visible_treasures, key=lambda item: item[0])
            buff_threshold = STAGE1_BUFF_GRID_THRESHOLD if stage == 1 else STAGE2_BUFF_GRID_THRESHOLD
            if (
                buff_dist > 0
                and buff_dist <= buff_threshold
                and buff_dist <= nearest_treasure_dist * BUFF_INSERT_RATIO
            ):
                target_pos_features = read_relative_position(buff_pos)
                target_type = "buff_near_treasure"
            else:
                target_pos_features = read_relative_position(treasure_pos_list[nearest_treasure_index])
                target_pos_features = route_target_features(obstacle_map, treasure_map, target_pos_features)
                target_type = "treasure_visible"
        elif stage in (1, 2) and active_treasures:
            _, nearest_treasure_index = min(active_treasures, key=lambda item: item[0])
            target_pos_features = read_relative_position(treasure_pos_list[nearest_treasure_index])
            target_type = "treasure_known"
        elif stage in (1, 2) and known_treasures:
            _, _, nearest_known_treasure = min(known_treasures, key=lambda item: item[0])
            target_pos_features = read_grid_target_position(grid_pos, nearest_known_treasure)
            target_type = "treasure_remembered"
        elif stage == 1 and 0 < buff_dist <= STAGE1_BUFF_GRID_THRESHOLD:
            target_pos_features = read_relative_position(buff_pos)
            target_type = "buff"
        elif stage == 2 and 0 < buff_dist <= STAGE2_BUFF_GRID_THRESHOLD:
            target_pos_features = read_relative_position(buff_pos)
            target_type = "buff"
        elif stage in (1, 2):
            target_pos_features = exploration_target_features(memory_map, obstacle_map, local_visible_map)
            target_type = "exploration"
        target_dir = read_target_direction(target_pos_features)

        # Feature concatenation:
        # Concatenate all necessary features as vector features (2 + 128*2 + 9  + 9*15 + 2 + 4*51*51 = 10808)
        # 特征拼接：将所有需要的特征进行拼接作为向量特征 (2 + 128*2 + 9  + 9*15 + 2 + 4*51*51 = 10808)
        feature_vec = (
            norm_pos
            + one_hot_pos
            + target_pos_features
            + treasure_pos_features
            + [buff_availability, talent_availability]
        )
        effective_end_map = end_map if stage == 3 else [0] * len(end_map)
        feature_map = obstacle_map + effective_end_map + treasure_map + memory_map
        # Legal actions
        # 合法动作
        legal_act = list(raw_obs.legal_act)

        remain_info = {
            "memory_map": memory_map,
            "end_pos": end_pos,
            "buff_pos": buff_pos,
            "treasure_pos": treasure_pos_list,
            "recent_position_map": recent_position_map,
            "arrival_position_map": arrival_position_map,
            "visible_position_map": visible_position_map,
            "local_visible_map": local_visible_map,
            "view_coverage": view_coverage,
            "discovered_treasure_count": discovered_treasure_count,
            "treasure_collected_count": treasure_collected_count,
            "treasure_count": treasure_count,
            "stage": stage,
            "grid_pos": (grid_pos.x, grid_pos.z),
            "target_type": target_type,
            "target_dir": target_dir,
            "target_dist_feature": float(target_pos_features[8]) if target_pos_features else 1.0,
        }

        return ObsData(feature=feature_vec + feature_map, legal_act=legal_act), remain_info
