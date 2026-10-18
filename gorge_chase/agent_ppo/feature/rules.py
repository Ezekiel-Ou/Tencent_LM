#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Rule override layer for Gorge Chase PPO.

规则接管层：硬掩码 + 邻宝箱强拾 + 贴身逃生强闪。

约定：
  - 所有规则在 agent.predict 的最后一步调用 apply_rule_override
  - 规则只替换"最终执行动作"，不改动 RL 策略分布 prob
    （PPO 的 ratio = new_prob / old_prob 必须用 RL 的分布，否则失真）
"""

from agent_ppo.conf.conf import Config


def _angle_diff_8(a, b):
    d = abs(int(a) - int(b)) % 8
    return min(d, 8 - d)


def _closest_monster_dir(rule_hints):
    dists = rule_hints.get("monster_bfs_dists", [])
    dirs = rule_hints.get("monster_dirs", [])
    best_d = None
    best_dist = None
    for dist, d in zip(dists, dirs):
        if d is None or d < 0 or d > 7:
            continue
        if dist is None:
            continue
        dist_i = int(dist)
        if best_dist is None or dist_i < best_dist:
            best_dist = dist_i
            best_d = int(d)
    return best_d


def _blocked_treasure_dir_early(rule_hints):
    step_no = int(rule_hints.get("step_no", 0))
    if step_no > 10:
        return None
    if int(rule_hints.get("treasure_in_view_count", 0)) <= 0:
        return None
    t_dir = int(rule_hints.get("nearest_treasure_dir", -1))
    if t_dir < 0 or t_dir > 7:
        return None
    m_dir = _closest_monster_dir(rule_hints)
    if m_dir is None:
        return None
    if _angle_diff_8(t_dir, m_dir) <= 1:
        return t_dir
    return None


def _pick_safe_move(blocked_dir, movable_mask, monster_dirs,
                    map_grid=None, hero_pos=None, monster_positions=None):
    candidates = [k for k in range(8) if movable_mask[k] and k != blocked_dir]
    if not candidates:
        return None

    if map_grid is not None and hero_pos is not None and monster_positions:
        from agent_ppo.feature import bfs as bfs_mod

        hero_x, hero_z = hero_pos
        best_dir = None
        best_score = float("-inf")
        for k in candidates:
            dx, dz = Config.DIR_OFFSETS[k]
            nx = hero_x + int(dx)
            nz = hero_z + int(dz)
            dist_map, _ = bfs_mod.local_bfs(map_grid, nx, nz)

            min_monster_dist = float("inf")
            for mx, mz in monster_positions:
                h, w = map_grid.shape
                if 0 <= mx < w and 0 <= mz < h:
                    dist = int(dist_map[mz, mx])
                    if dist == -1:
                        min_monster_dist = float("inf")
                        break
                    min_monster_dist = min(min_monster_dist, dist)

            if min_monster_dist > best_score:
                best_score = min_monster_dist
                best_dir = k
        return best_dir

    m_dir = -1
    for d in monster_dirs:
        if d is not None and 0 <= d <= 7:
            m_dir = d
            break
    if m_dir == -1:
        return candidates[0]
    return max(candidates, key=lambda k: _angle_diff_8(k, m_dir))


def _is_any_monster_adjacent(hero_pos, monster_positions):
    if hero_pos is None or not monster_positions:
        return False
    hx, hz = hero_pos
    for mx, mz in monster_positions:
        if max(abs(int(mx) - int(hx)), abs(int(mz) - int(hz))) <= 1:
            return True
    return False


def _dir_from_delta_8(dx, dz):
    sdx = 0 if dx == 0 else (1 if dx > 0 else -1)
    sdz = 0 if dz == 0 else (1 if dz > 0 else -1)
    if sdx == 0 and sdz == 0:
        return None
    for k, (odx, odz) in enumerate(Config.DIR_OFFSETS):
        if int(odx) == sdx and int(odz) == sdz:
            return int(k)
    return None


def _blocked_monster_dir_close_euclid(rule_hints, max_dist=2):
    hero_pos = rule_hints.get("hero_pos")
    monster_positions = rule_hints.get("monster_positions", [])
    if hero_pos is None or not monster_positions:
        return None
    hx, hz = hero_pos
    max_d2 = int(max_dist) * int(max_dist)
    best_dir = None
    best_d2 = None
    for mx, mz in monster_positions:
        dx = int(mx) - int(hx)
        dz = int(mz) - int(hz)
        d2 = dx * dx + dz * dz
        if d2 > max_d2:
            continue
        dir_idx = _dir_from_delta_8(dx, dz)
        if dir_idx is None:
            continue
        if best_d2 is None or d2 < best_d2:
            best_d2 = d2
            best_dir = dir_idx
    return best_dir


def adjacent_treasure_action(rule_hints):
    """若有宝箱在英雄 8 邻格内且对应方向可移动，返回该方向；否则 None."""
    d = rule_hints.get("adjacent_treasure_dir", -1)
    if d is not None and 0 <= d < 8:
        blocked = _blocked_treasure_dir_early(rule_hints)
        if blocked is not None and int(d) == int(blocked):
            return None
        return int(d)
    return None


def _pick_safe_flash(flash_useful_mask, flash_landings, monster_dirs, flash_ready,
                     map_grid=None, hero_pos=None, monster_positions=None):
    """挑一个"远离最近怪物"的有效闪现方向。

    策略：计算每个闪现落点与怪物的BFS距离，选择距离最大的方向。
    如果闪现后穿墙导致BFS不可达（距离=-1），这是最优选择。
    如果没有地图/怪物信息，回退到原策略（选夹角最大或落点距离最大）。
    返回 action index 8..15，失败返回 None.
    """
    if not flash_ready:
        return None
    useful_dirs = [k for k in range(8) if flash_useful_mask[k]]
    if not useful_dirs:
        return None

    # 新策略：如果有地图和怪物位置信息，计算闪现后的BFS距离
    if map_grid is not None and hero_pos is not None and monster_positions:
        from agent_ppo.feature import bfs as bfs_mod

        hero_x, hero_z = hero_pos
        best_dir = None
        best_score = float('-inf')

        for k in useful_dirs:
            # 计算闪现落点
            dx, dz, _ = flash_landings[k]
            landing_x = hero_x + dx
            landing_z = hero_z + dz

            # 从落点计算到所有怪物的BFS距离
            dist_map, _ = bfs_mod.local_bfs(map_grid, landing_x, landing_z)

            # 计算到最近怪物的距离
            min_monster_dist = float('inf')
            for mx, mz in monster_positions:
                h, w = map_grid.shape
                if 0 <= mx < w and 0 <= mz < h:
                    dist = int(dist_map[mz, mx])
                    if dist == -1:  # 不可达（穿墙），这是最优情况
                        min_monster_dist = float('inf')
                        break
                    min_monster_dist = min(min_monster_dist, dist)

            # 选择距离最大的方向（不可达=inf最优）
            if min_monster_dist > best_score:
                best_score = min_monster_dist
                best_dir = k

        if best_dir is not None:
            return 8 + best_dir

    # 回退策略：使用原有的夹角/距离逻辑
    # 找最近怪物的第一步方向（0..7），没有则 -1
    m_dir = -1
    for d in monster_dirs:
        if d is not None and 0 <= d <= 7:
            m_dir = d
            break

    if m_dir == -1:
        # 没怪物信息 -> 选落点距离最大的方向
        best = max(
            useful_dirs,
            key=lambda k: abs(flash_landings[k][0]) + abs(flash_landings[k][1]),
        )
        return 8 + best

    # 选与 m_dir 夹角最大的方向（理想的反方向是 (m_dir + 4) % 8）
    def angle_diff(a, b):
        d = abs(a - b) % 8
        return min(d, 8 - d)  # 0..4

    best = max(
        useful_dirs,
        key=lambda k: (angle_diff(k, m_dir),
                       abs(flash_landings[k][0]) + abs(flash_landings[k][1])),
    )
    return 8 + best


def panic_flash_action(rule_hints):
    """贴身逃生：最近怪物 BFS 距离 <= 阈值时强制闪现到最安全方向。

    阈值前/后期差异：
      前期 PANIC_FLASH_THRESHOLD_EARLY = 1
      后期 PANIC_FLASH_THRESHOLD_LATE  = 2
    """
    if not Config.RULE_PANIC_FLASH:
        return None
    if not rule_hints.get("flash_ready", False):
        return None
    dists = rule_hints.get("monster_bfs_dists", [])
    if not dists:
        return None
    is_late = rule_hints.get("is_late_phase", False)
    thr = (Config.PANIC_FLASH_THRESHOLD_LATE if is_late
           else Config.PANIC_FLASH_THRESHOLD_EARLY)
    is_adjacent = _is_any_monster_adjacent(
        rule_hints.get("hero_pos"),
        rule_hints.get("monster_positions", []),
    )
    if (not is_adjacent) and min(dists) > thr:
        return None
    return _pick_safe_flash(
        rule_hints["flash_useful_mask"],
        rule_hints["flash_landings"],
        rule_hints.get("monster_dirs", []),
        rule_hints["flash_ready"],
        rule_hints.get("map_grid"),
        rule_hints.get("hero_pos"),
        rule_hints.get("monster_positions", []),
    )


def apply_rule_override(rl_action, legal_action_16d, rule_hints):
    """Apply rule priority: legal-mask > panic-flash > close-monster-avoid > adjacent-treasure > rl_action.

    Args:
      rl_action: int in [0, 16), sampled from policy under legal_action_16d mask
      legal_action_16d: 16D int list
      rule_hints: dict from preprocessor

    Returns:
      (final_action, override_flag)
        override_flag: False = RL 原值, True = 规则覆盖
    """
    # 优先级 1：如果 RL 动作越界或撞掩码（理论上应被 softmax 排除，此处兜底）
    if rl_action < 0 or rl_action >= 16 or legal_action_16d[rl_action] == 0:
        # 在合法动作里选第一个
        for k, v in enumerate(legal_action_16d):
            if v == 1:
                rl_action = k
                break

    # 优先级 2：贴身逃生
    a = panic_flash_action(rule_hints)
    if a is not None and legal_action_16d[a] == 1:
        return a, True

    blocked = _blocked_monster_dir_close_euclid(rule_hints, max_dist=2)
    if blocked is not None and 0 <= int(rl_action) <= 7 and int(rl_action) == int(blocked):
        a = _pick_safe_move(
            blocked_dir=int(blocked),
            movable_mask=rule_hints.get("movable_mask", [0] * 8),
            monster_dirs=rule_hints.get("monster_dirs", []),
            map_grid=rule_hints.get("map_grid"),
            hero_pos=rule_hints.get("hero_pos"),
            monster_positions=rule_hints.get("monster_positions", []),
        )
        if a is not None and legal_action_16d[a] == 1:
            return a, True

    blocked = _blocked_treasure_dir_early(rule_hints)
    if blocked is not None and 0 <= int(rl_action) <= 7 and int(rl_action) == int(blocked):
        a = _pick_safe_move(
            blocked_dir=int(blocked),
            movable_mask=rule_hints.get("movable_mask", [0] * 8),
            monster_dirs=rule_hints.get("monster_dirs", []),
            map_grid=rule_hints.get("map_grid"),
            hero_pos=rule_hints.get("hero_pos"),
            monster_positions=rule_hints.get("monster_positions", []),
        )
        if a is not None and legal_action_16d[a] == 1:
            return a, True

    # 优先级 3：邻格宝箱强制拾取（白送分，无风险）
    if Config.RULE_ADJACENT_TREASURE:
        a = adjacent_treasure_action(rule_hints)
        if a is not None and legal_action_16d[a] == 1:
            return a, True

    return rl_action, False
