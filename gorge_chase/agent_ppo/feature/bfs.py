#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Pure-function geometric / BFS utilities for Gorge Chase preprocessing.

纯函数几何 / BFS 工具集，供 preprocessor 与 rules 调用。

约定（Convention）：
  - map_grid: numpy int array，shape (H, W)，1 = 可通行，0 = 障碍
    （严格对齐 `开发指南/数据协议.md`：map_info `1=可通行, 0=障碍物`）
  - 坐标：map_grid[row, col] 中 row 对应 z（上下），col 对应 x（左右）
    但我们对外使用 (col, row) == (x, z) 保持与 env 一致
  - 8 方向索引严格按动作协议：0=E, 1=NE, 2=N, 3=NW, 4=W, 5=SW, 6=S, 7=SE
  - BFS 返回 "第一步方向 0..7"，可直接作为动作 id 使用
"""

from collections import deque

import numpy as np

from agent_ppo.conf.conf import Config

# 8 方向偏移（col/x, row/z）
_DIR_OFFSETS = Config.DIR_OFFSETS  # 动作协议方向
_FLASH_DIST = Config.FLASH_DISTANCES


def _in_bounds(x, z, w, h):
    return 0 <= x < w and 0 <= z < h


def _walkable(map_grid, x, z):
    """Cell (x, z) is walkable: inside bounds and map_grid[z, x] != 0."""
    h, w = map_grid.shape
    return _in_bounds(x, z, w, h) and map_grid[z, x] != 0


def can_move_to(map_grid, from_x, from_z, dir_idx):
    """Check if a single move from (from_x, from_z) in direction dir_idx is legal.

    严格按 `开发指南/环境详述.md:202-205`：
      - 普通移动：目标格可通行
      - 斜向移动：目标格可通行，且相邻两条直边至少有一条可通行
      - 障碍物处理：速度为 1 时向障碍物方向移动将停留在原地
    """
    dx, dz = _DIR_OFFSETS[dir_idx]
    tx, tz = from_x + dx, from_z + dz
    if not _walkable(map_grid, tx, tz):
        return False
    if dx != 0 and dz != 0:
        # 斜向：相邻两直边之一可通行
        side1 = _walkable(map_grid, from_x + dx, from_z)
        side2 = _walkable(map_grid, from_x, from_z + dz)
        if not (side1 or side2):
            return False
    return True


def compute_movable_mask(map_grid, hero_x, hero_z):
    """Return 8D int mask: 1 = direction is a legal move (not into a wall).

    注意 env 的 legal_action[0:7] 恒为 true，此函数才是真实的撞墙检查。
    """
    return [int(can_move_to(map_grid, hero_x, hero_z, k)) for k in range(8)]


def compute_obstacle_rays(map_grid, hero_x, hero_z, max_range=None):
    """For each of 8 directions, return the distance to the first obstacle.

    返回 8D list[int]：沿方向 k 能连续走多少格（不撞墙，不越界）。
    封顶 max_range（默认 10，等于直线闪现距离）。
    """
    if max_range is None:
        max_range = Config.FLASH_RANGE_STRAIGHT
    rays = [0] * 8
    for k, (dx, dz) in enumerate(_DIR_OFFSETS):
        d = 0
        x, z = hero_x, hero_z
        for _ in range(max_range):
            nx, nz = x + dx, z + dz
            if not _walkable(map_grid, nx, nz):
                break
            # 斜向需要满足斜向移动的相邻约束
            if dx != 0 and dz != 0:
                if not (_walkable(map_grid, x + dx, z) or _walkable(map_grid, x, z + dz)):
                    break
            d += 1
            x, z = nx, nz
        rays[k] = d
    return rays


def compute_flash_landing(map_grid, hero_x, hero_z, treasure_set=None, buff_set=None):
    """Compute 8 flash landings and path-item counts.

    按 `环境详述.md:208-212` 闪现规则：
      - 直线方向闪现 10 格，斜向闪现 8 格
      - 障碍物处理：从最远距离向近处逐格搜索，闪现到范围内最远的可通行格子
      - 路径收集：闪现路径上经过的宝箱和 buff 会被收集

    Args:
      treasure_set: set of (x, z) of treasures (in local coords), or None
      buff_set: set of (x, z) of buffs (in local coords), or None

    Returns:
      list of 8 tuples: (dx, dz, path_item_count)
        dx, dz: 落点相对 hero 的偏移（若无处可去则 (0, 0)）
        path_item_count: 从 hero+1 到落点（含）所经过的宝箱/buff 格数
    """
    if treasure_set is None:
        treasure_set = set()
    if buff_set is None:
        buff_set = set()

    results = []
    for k, (dx, dz) in enumerate(_DIR_OFFSETS):
        max_r = _FLASH_DIST[k]
        # 从最远处往近处搜可通行落点
        land_dx, land_dz = 0, 0
        for r in range(max_r, 0, -1):
            tx = hero_x + dx * r
            tz = hero_z + dz * r
            if _walkable(map_grid, tx, tz):
                land_dx, land_dz = dx * r, dz * r
                break
        # 统计路径上的宝箱/buff（从 hero+1 到落点含端点）
        item_cnt = 0
        if land_dx != 0 or land_dz != 0:
            steps = max(abs(land_dx), abs(land_dz))
            for s in range(1, steps + 1):
                px = hero_x + dx * s
                pz = hero_z + dz * s
                if (px, pz) in treasure_set or (px, pz) in buff_set:
                    item_cnt += 1
        results.append((land_dx, land_dz, item_cnt))
    return results


def compute_flash_useful_mask(flash_landings):
    """8D mask: 1 = flash direction has a non-zero landing offset."""
    return [int(dx != 0 or dz != 0) for (dx, dz, _) in flash_landings]


def local_bfs(map_grid, start_x, start_z):
    """BFS from start, returning (dist_map, first_dir_map).

    dist_map[z, x]     = 最短步数（不可达 = -1）
    first_dir_map[z, x] = 从 start 出发第一步的方向 0..7（不可达 = -1）

    对角线移动算 1 步（与动作协议一致：一步可对角走 1 格）。
    """
    h, w = map_grid.shape
    dist_map = -np.ones((h, w), dtype=np.int32)
    first_dir_map = -np.ones((h, w), dtype=np.int32)
    if not _walkable(map_grid, start_x, start_z):
        return dist_map, first_dir_map
    dist_map[start_z, start_x] = 0
    q = deque()
    q.append((start_x, start_z))
    while q:
        x, z = q.popleft()
        for k, (dx, dz) in enumerate(_DIR_OFFSETS):
            nx, nz = x + dx, z + dz
            if not can_move_to(map_grid, x, z, k):
                continue
            if dist_map[nz, nx] != -1:
                continue
            dist_map[nz, nx] = dist_map[z, x] + 1
            # 第一步方向：从起点出发直接走的方向；已经离起点 > 1 步时继承
            if dist_map[z, x] == 0:
                first_dir_map[nz, nx] = k
            else:
                first_dir_map[nz, nx] = first_dir_map[z, x]
            q.append((nx, nz))
    return dist_map, first_dir_map


def bfs_query(dist_map, first_dir_map, target_x, target_z):
    """Query BFS result for one target.

    Returns:
      (dist, first_dir): dist = -1 if unreachable; first_dir in 0..7 or -1
    """
    h, w = dist_map.shape
    if not _in_bounds(target_x, target_z, w, h):
        return -1, -1
    return int(dist_map[target_z, target_x]), int(first_dir_map[target_z, target_x])


def direction_onehot_cos_sin(dir_idx):
    """Encode direction 0..7 (or -1) as (cos, sin, is_unknown).

    用 cos/sin 而非 one-hot 保持方向的连续性（避免邻近方向在特征空间中正交）。
    """
    if dir_idx < 0 or dir_idx > 7:
        return 0.0, 0.0, 1.0
    # 动作协议：0=E(0°), 1=NE(45°), 2=N(90°), 3=NW(135°), 4=W(180°),
    #           5=SW(225°), 6=S(270°), 7=SE(315°)
    angle = dir_idx * (np.pi / 4.0)
    return float(np.cos(angle)), float(np.sin(angle)), 0.0


def bucket_direction_to_cos_sin(hero_relative_direction):
    """Env 字段 hero_relative_direction（0=无效, 1=东, 2=东北, ..., 8=东南）转 (cos, sin, is_unknown).

    映射表（`数据协议.md:44`）：
      1=东   → 0°   → (1, 0)
      2=东北 → 45°  → (sqrt(2)/2, sqrt(2)/2)
      3=北   → 90°  → (0, 1)
      4=西北 → 135°
      5=西   → 180°
      6=西南 → 225°
      7=南   → 270°
      8=东南 → 315°
    """
    if hero_relative_direction <= 0:
        return 0.0, 0.0, 1.0
    # env 的方位编号与动作协议方向索引偏移 1 并且顺序相同
    dir_idx = hero_relative_direction - 1
    return direction_onehot_cos_sin(dir_idx)
