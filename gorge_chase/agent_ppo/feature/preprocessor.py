#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Feature preprocessor and reward design for Gorge Chase PPO (refactored).

峡谷追猎 PPO 特征预处理与奖励设计（重构版）。

输出：
  - map_tensor (C, H, W): 6 通道空间视图，喂给 CNN 分支
  - sym_feat (SYM_FEATURE_LEN,): 130D 符号特征，喂给 MLP 分支
  - legal_action_16d (16,): 包含撞墙检查 + 闪现冷却 + 闪现有效性的完整 16D 掩码
  - reward: 阶段感知 + 势函数塑形 + 事件奖励
  - rule_hints: 供 rules.py 使用的辅助字典
"""

import numpy as np

from agent_ppo.conf.conf import Config
from agent_ppo.feature import bfs as bfs_mod


def _clip01(v):
    return float(max(0.0, min(1.0, v)))


def _norm(v, v_max, v_min=0.0):
    if v_max - v_min <= 1e-6:
        return 0.0
    return _clip01((float(v) - v_min) / (v_max - v_min))


def _lerp(a, b, t):
    return a * (1.0 - t) + b * t


class Preprocessor:
    """State-ful per-episode feature computer.

    状态按 episode 维护；每局开始调用 reset()。
    """

    def __init__(self):
        self.reset()

    def reset(self):
        # 进度
        self.step_no = 0
        self.max_step = 1000

        # 事件计数（检测"新拾取"事件）
        self.last_treasures_collected = 0
        self.last_collected_buff = 0
        self.last_flash_count = 0

        # 位置记忆（撞墙检测、怪物运动矢量）
        self.last_hero_pos = None  # (x, z)
        self.last_monster_pos = {}  # monster_id -> (x, z)

        # 英雄轨迹（通道 5 使用）
        self.trajectory = []  # list[(x, z)], 最近在末尾

        # 势函数
        self.last_potential = 0.0

        # 最近一帧最近怪物 BFS 距离（检测闪现逃生）
        self.last_min_monster_bfs = Config.BFS_MAX_STEPS

        # 最近一帧 used_flash 前的 flash_count（帧边界标记）
        self.prev_flash_count = 0

        # 地图边长（env 可能给的不是 21，以实际为准）
        self.view_h = Config.MAP_VIEW
        self.view_w = Config.MAP_VIEW

        # 阶段累计（监控用，不参与网络）
        self.wall_bump_count = 0
        self.anti_stall_count = 0
        self.treasure_discover_count = 0

        # 本局已见过的宝箱（按全局坐标去重）
        self.seen_treasures = set()

    # -----------------------------------------------------------------
    # 主入口
    # -----------------------------------------------------------------
    def feature_process(self, env_obs, last_action):
        observation = env_obs["observation"]
        frame_state = observation["frame_state"]
        env_info = observation["env_info"]
        map_info_raw = observation.get("map_info", None)
        legal_act_raw = observation.get("legal_action", None)

        self.step_no = int(observation.get("step_no", env_info.get("step_no", 0)))
        self.max_step = int(env_info.get("max_step", 1000))

        # ---------- 1. 归一化地图为 numpy ----------
        map_grid = self._normalize_map(map_info_raw)
        h, w = map_grid.shape
        self.view_h, self.view_w = h, w
        center_x = w // 2
        center_z = h // 2

        # 英雄全局位置（用于事件检测 / 距离计算）
        hero = frame_state.get("heroes", {}) or {}
        if isinstance(hero, list):
            hero = hero[0] if hero else {}
        hero_pos = hero.get("pos", {}) or {}
        hero_global_x = int(hero_pos.get("x", 0))
        hero_global_z = int(hero_pos.get("z", 0))

        # ---------- 2. 整理 organs（宝箱 / buff）到局部坐标 ----------
        organs = frame_state.get("organs", []) or []
        treasure_local = []  # (local_x, local_z, dist_bucket, dir_bucket, status)
        buff_local = []
        for o in organs:
            sub_type = int(o.get("sub_type", 0))
            status = int(o.get("status", 0))
            bucket = int(o.get("hero_l2_distance", 5))
            dir_e = int(o.get("hero_relative_direction", 0))
            opos = o.get("pos", {}) or {}
            ox, oz = int(opos.get("x", 0)), int(opos.get("z", 0))
            # 局部坐标（以英雄为中心，中心像素 = (center_x, center_z)）
            lx = ox - hero_global_x + center_x
            lz = oz - hero_global_z + center_z
            entry = (lx, lz, bucket, dir_e, status, ox, oz)
            if sub_type == 1 and status == 1:
                treasure_local.append(entry)
            elif sub_type == 2 and status == 1:
                buff_local.append(entry)

        treasure_in_view_set = {(e[0], e[1]) for e in treasure_local
                                if 0 <= e[0] < w and 0 <= e[1] < h}
        buff_in_view_set = {(e[0], e[1]) for e in buff_local
                             if 0 <= e[0] < w and 0 <= e[1] < h}
        visible_treasure_global = {(e[5], e[6]) for e in treasure_local}

        # ---------- 3. 怪物信息 ----------
        monsters_raw = frame_state.get("monsters", []) or []
        monsters = []  # 最多 2 个，按 env 顺序
        for i in range(2):
            if i < len(monsters_raw):
                monsters.append(monsters_raw[i])
            else:
                monsters.append(None)

        # ---------- 4. 几何工具 ----------
        rays = bfs_mod.compute_obstacle_rays(map_grid, center_x, center_z)
        flash_landings = bfs_mod.compute_flash_landing(
            map_grid, center_x, center_z, treasure_in_view_set, buff_in_view_set
        )
        movable_mask = bfs_mod.compute_movable_mask(map_grid, center_x, center_z)
        flash_useful_mask = bfs_mod.compute_flash_useful_mask(flash_landings)

        # 英雄出发的 BFS（一次即可覆盖所有局部目标）
        hero_dist_map, hero_first_dir_map = bfs_mod.local_bfs(map_grid, center_x, center_z)

        # ---------- 5. 构建 map_tensor ----------
        map_tensor = self._build_map_tensor(
            map_grid, treasure_in_view_set, buff_in_view_set,
            monsters, hero_global_x, hero_global_z, center_x, center_z,
        )

        second_monster_spawn_hint = self._compute_second_monster_spawn_hint(
            hero_global_x=hero_global_x,
            hero_global_z=hero_global_z,
            env_info=env_info,
            monsters=monsters,
        )
        anti_stall_hint, anti_stall = self._compute_anti_stall_hint(
            hero_global_x=hero_global_x,
            hero_global_z=hero_global_z,
        )

        # ---------- 6. 构建符号特征 ----------
        sym_feat, dead_end_score = self._build_sym_feat(
            hero=hero,
            env_info=env_info,
            monsters=monsters,
            treasure_local=treasure_local,
            buff_local=buff_local,
            hero_global=(hero_global_x, hero_global_z),
            center=(center_x, center_z),
            map_grid=map_grid,
            hero_dist_map=hero_dist_map,
            hero_first_dir_map=hero_first_dir_map,
            rays=rays,
            flash_landings=flash_landings,
            movable_mask=movable_mask,
            flash_useful_mask=flash_useful_mask,
            second_monster_spawn_hint=second_monster_spawn_hint,
            anti_stall_hint=anti_stall_hint,
            last_action=last_action,
        )

        # ---------- 7. 16D 完整合法动作掩码 ----------
        flash_cd = int(hero.get("flash_cooldown", 0))
        flash_ready = 1 if flash_cd <= 0 else 0
        env_flash_mask = self._parse_env_flash_mask(legal_act_raw)
        legal_action_16d = self._build_legal_16(
            movable_mask, flash_ready, flash_useful_mask, env_flash_mask
        )

        # ---------- 8. 事件检测 & 奖励 ----------
        cur_treasures_collected = int(env_info.get("treasures_collected", 0))
        cur_collected_buff = int(env_info.get("collected_buff", 0))
        cur_flash_count = int(env_info.get("flash_count", 0))
        picked_treasure = cur_treasures_collected > self.last_treasures_collected
        picked_buff = cur_collected_buff > self.last_collected_buff
        used_flash = cur_flash_count > self.last_flash_count
        new_seen_treasures = visible_treasure_global - self.seen_treasures
        discovered_new_treasure = len(new_seen_treasures) > 0
        if discovered_new_treasure:
            self.treasure_discover_count += 1
            self.seen_treasures.update(new_seen_treasures)

        # 撞墙检测：执行了移动动作（0..7）但英雄位置未变
        bumped_wall = False
        if (self.last_hero_pos is not None
                and last_action is not None
                and 0 <= int(last_action) <= 7):
            if (hero_global_x, hero_global_z) == self.last_hero_pos:
                bumped_wall = True
                self.wall_bump_count += 1

        # 怪物 BFS 距离（用于逃生奖励）
        min_monster_bfs = Config.BFS_MAX_STEPS
        monster_bfs_dists = [Config.BFS_MAX_STEPS, Config.BFS_MAX_STEPS]
        for mi, m in enumerate(monsters):
            if m is None:
                continue
            mpos = m.get("pos", {}) or {}
            mx, mz = int(mpos.get("x", 0)), int(mpos.get("z", 0))
            lx = mx - hero_global_x + center_x
            lz = mz - hero_global_z + center_z
            if 0 <= lx < w and 0 <= lz < h:
                d = int(hero_dist_map[lz, lx])
                if d >= 0:
                    monster_bfs_dists[mi] = d
                    if d < min_monster_bfs:
                        min_monster_bfs = d

        # 宝箱 BFS 最近距离（势函数用）
        nearest_treasure_bfs = Config.BFS_MAX_STEPS
        nearest_treasure_dir = -1
        for (lx, lz, _b, _d, _s, _gx, _gz) in treasure_local:
            if 0 <= lx < w and 0 <= lz < h:
                d = int(hero_dist_map[lz, lx])
                if d >= 0 and d < nearest_treasure_bfs:
                    nearest_treasure_bfs = d
                    nearest_treasure_dir = int(hero_first_dir_map[lz, lx])

        if nearest_treasure_dir < 0 and treasure_local:
            best = None
            for (lx, lz, bucket, dir_e, _s, _gx, _gz) in treasure_local:
                if dir_e < 1 or dir_e > 8:
                    continue
                if best is None or bucket < best[0]:
                    best = (bucket, dir_e)
            if best is not None:
                nearest_treasure_dir = int(best[1]) - 1

        # 阶段因子（平滑过渡）
        max_monster_speed = 1
        for m in monsters:
            if m is None:
                continue
            max_monster_speed = max(max_monster_speed, int(m.get("speed", 1)))
        late_factor = _clip01((max_monster_speed - 1.0) / 1.0)  # 1->0, 2->1
        is_late_phase = late_factor > 0.5

        # 闪现路径是否含宝箱/buff（用最近一次 used_flash 对应的方向）
        flash_path_had_item = False
        if used_flash and last_action is not None and 8 <= int(last_action) <= 15:
            dir_idx = int(last_action) - 8
            if flash_landings[dir_idx][2] > 0:
                flash_path_had_item = True

        # 闪现逃生判定（距离增长 >= 3）
        monster_bfs_increased = (min_monster_bfs - self.last_min_monster_bfs) >= 3

        # 死角检测
        dead_end = dead_end_score > Config.DEAD_END_RATIO_THRESHOLD

        reward = self._compute_reward(
            picked_treasure=picked_treasure,
            picked_buff=picked_buff,
            discovered_new_treasure=discovered_new_treasure,
            used_flash=used_flash,
            flash_path_had_item=flash_path_had_item,
            monster_bfs_increased=monster_bfs_increased,
            bumped_wall=bumped_wall,
            dead_end=dead_end,
            anti_stall=anti_stall,
            late_factor=late_factor,
            nearest_treasure_bfs=nearest_treasure_bfs,
            min_monster_bfs=min_monster_bfs,
        )
        if anti_stall:
            self.anti_stall_count += 1

        # ---------- 9. 更新状态 ----------
        self.last_treasures_collected = cur_treasures_collected
        self.last_collected_buff = cur_collected_buff
        self.last_flash_count = cur_flash_count
        self.last_hero_pos = (hero_global_x, hero_global_z)
        self.last_min_monster_bfs = min_monster_bfs
        for mi, m in enumerate(monsters):
            if m is None:
                continue
            mpos = m.get("pos", {}) or {}
            self.last_monster_pos[mi] = (int(mpos.get("x", 0)), int(mpos.get("z", 0)))

        # 轨迹（存局部偏移）
        self.trajectory.append((hero_global_x, hero_global_z))
        if len(self.trajectory) > Config.TRAJECTORY_LEN:
            self.trajectory.pop(0)

        # ---------- 10. 规则提示 ----------
        # 方向数组（对齐 2 个怪物，用于 rules.panic_flash）
        monster_dirs = []
        for mi, m in enumerate(monsters):
            if m is None:
                monster_dirs.append(-1)
                continue
            mpos = m.get("pos", {}) or {}
            mx, mz = int(mpos.get("x", 0)), int(mpos.get("z", 0))
            lx = mx - hero_global_x + center_x
            lz = mz - hero_global_z + center_z
            d, fd = bfs_mod.bfs_query(hero_dist_map, hero_first_dir_map, lx, lz)
            monster_dirs.append(fd)

        # 邻格宝箱判定（给规则层）
        adjacent_treasure_dir = -1
        for (lx, lz, _b, _d, _s, _gx, _gz) in treasure_local:
            dx = lx - center_x
            dz = lz - center_z
            # 仅判断 8 邻格
            if max(abs(dx), abs(dz)) == 1 and (dx != 0 or dz != 0):
                for k, (odx, odz) in enumerate(Config.DIR_OFFSETS):
                    if odx == dx and odz == dz:
                        if movable_mask[k]:
                            adjacent_treasure_dir = k
                            break
                if adjacent_treasure_dir != -1:
                    break

        rule_hints = {
            "step_no": int(self.step_no),
            "movable_mask": movable_mask,
            "flash_useful_mask": flash_useful_mask,
            "flash_ready": bool(flash_ready),
            "adjacent_treasure_dir": adjacent_treasure_dir,
            "treasure_in_view_count": int(len(treasure_in_view_set)),
            "nearest_treasure_dir": int(nearest_treasure_dir),
            "min_monster_bfs": int(min_monster_bfs),
            "monster_bfs_dists": monster_bfs_dists,
            "monster_dirs": monster_dirs,
            "is_late_phase": is_late_phase,
            "flash_landings": flash_landings,
            "map_grid": map_grid,
            "hero_pos": (center_x, center_z),
            "monster_positions": [(mx - hero_global_x + center_x, mz - hero_global_z + center_z)
                                  for m in monsters if m is not None
                                  for mx, mz in [(int(m.get("pos", {}).get("x", 0)),
                                                  int(m.get("pos", {}).get("z", 0)))]],
        }

        return {
            "map_tensor": map_tensor.astype(np.float32),
            "sym_feat": sym_feat.astype(np.float32),
            "legal_action_16d": legal_action_16d,
            "movable_mask_8d": movable_mask,
            "flash_useful_mask_8d": flash_useful_mask,
            "reward": [float(reward)],
            "rule_hints": rule_hints,
        }

    # -----------------------------------------------------------------
    # 地图归一化
    # -----------------------------------------------------------------
    def _normalize_map(self, map_info_raw):
        """Convert raw map_info (list of list / ndarray) to (H, W) int array with 1=walkable, 0=obstacle."""
        if map_info_raw is None:
            return np.ones((Config.MAP_VIEW, Config.MAP_VIEW), dtype=np.int32)
        arr = np.asarray(map_info_raw, dtype=np.int32)
        if arr.ndim != 2:
            return np.ones((Config.MAP_VIEW, Config.MAP_VIEW), dtype=np.int32)
        # env 协议：1=可通行, 0=障碍。但实现有时返回 0/非 0，非 0 视为可通行
        return (arr != 0).astype(np.int32)

    def _build_map_tensor(
        self, map_grid, treasure_set, buff_set, monsters,
        hero_gx, hero_gz, cx, cz,
    ):
        h, w = map_grid.shape
        C = Config.MAP_CHANNELS
        VH, VW = Config.MAP_VIEW, Config.MAP_VIEW
        # 先按实际地图尺寸构造，再 resize/pad 到 MAP_VIEW
        tensor = np.zeros((C, h, w), dtype=np.float32)
        # C0 障碍（1=障碍本身，与动作语义一致的"危险"视角）
        tensor[0] = (map_grid == 0).astype(np.float32)
        # C1 宝箱
        for (lx, lz) in treasure_set:
            if 0 <= lx < w and 0 <= lz < h:
                tensor[1, lz, lx] = 1.0
        # C2 buff
        for (lx, lz) in buff_set:
            if 0 <= lx < w and 0 <= lz < h:
                tensor[2, lz, lx] = 1.0
        # C3 怪物
        for m in monsters:
            if m is None:
                continue
            if int(m.get("is_in_view", 0)) == 0:
                continue
            mpos = m.get("pos", {}) or {}
            mx, mz = int(mpos.get("x", 0)), int(mpos.get("z", 0))
            lx = mx - hero_gx + cx
            lz = mz - hero_gz + cz
            if 0 <= lx < w and 0 <= lz < h:
                tensor[3, lz, lx] = 1.0
        # C4 英雄（中心）
        if 0 <= cx < w and 0 <= cz < h:
            tensor[4, cz, cx] = 1.0
        # C5 英雄轨迹（衰减），转到当前局部坐标
        for i, (gx, gz) in enumerate(reversed(self.trajectory)):
            decay = 1.0 - i / max(Config.TRAJECTORY_LEN, 1)
            lx = gx - hero_gx + cx
            lz = gz - hero_gz + cz
            if 0 <= lx < w and 0 <= lz < h:
                tensor[5, lz, lx] = max(tensor[5, lz, lx], decay)

        # 统一到 MAP_VIEW x MAP_VIEW：中心裁剪或 0-padding
        if h == VH and w == VW:
            return tensor
        out = np.zeros((C, VH, VW), dtype=np.float32)
        # 把 (cx, cz) 作为中心对齐到 (VW//2, VH//2)
        ocx, ocz = VW // 2, VH // 2
        for zz in range(VH):
            for xx in range(VW):
                sx = cx + (xx - ocx)
                sz = cz + (zz - ocz)
                if 0 <= sx < w and 0 <= sz < h:
                    out[:, zz, xx] = tensor[:, sz, sx]
        return out

    # -----------------------------------------------------------------
    # 符号特征
    # -----------------------------------------------------------------
    def _build_sym_feat(
        self, hero, env_info, monsters, treasure_local, buff_local,
        hero_global, center, map_grid,
        hero_dist_map, hero_first_dir_map,
        rays, flash_landings, movable_mask, flash_useful_mask,
        second_monster_spawn_hint,
        anti_stall_hint,
        last_action,
    ):
        hgx, hgz = hero_global
        cx, cz = center
        h, w = map_grid.shape

        # ---- hero (6) ----
        flash_cd = int(hero.get("flash_cooldown", 0))
        flash_ready = 1.0 if flash_cd <= 0 else 0.0
        buff_remain = float(hero.get("buff_remaining_time", 0))
        has_speed_buff = 1.0 if buff_remain > 0 else 0.0
        hero_feat = np.array([
            _norm(hgx, Config.MAP_SIZE),
            _norm(hgz, Config.MAP_SIZE),
            _norm(flash_cd, Config.FLASH_COOLDOWN_DEFAULT),
            flash_ready,
            _norm(buff_remain, Config.MAX_BUFF_DURATION),
            has_speed_buff,
        ], dtype=np.float32)

        # ---- monster (10 × 2) ----
        monster_feats = []
        for mi in range(2):
            m = monsters[mi]
            if m is None:
                monster_feats.append(np.zeros(10, dtype=np.float32))
                continue
            is_in_view = int(m.get("is_in_view", 0))
            mpos = m.get("pos", {}) or {}
            mx, mz = int(mpos.get("x", 0)), int(mpos.get("z", 0))
            speed = int(m.get("speed", 1))
            bucket = int(m.get("hero_l2_distance", 5))
            dir_e = int(m.get("hero_relative_direction", 0))

            pos_x_norm = 0.0
            pos_z_norm = 0.0
            euclid_dist = 1.0
            bfs_or_bucket = bucket / 5.0
            delta_dist = 0.0

            if is_in_view:
                pos_x_norm = _norm(mx, Config.MAP_SIZE)
                pos_z_norm = _norm(mz, Config.MAP_SIZE)
                raw = np.hypot(hgx - mx, hgz - mz)
                euclid_dist = _norm(raw, Config.MAP_SIZE * 1.4143)
                lx = mx - hgx + cx
                lz = mz - hgz + cz
                if 0 <= lx < w and 0 <= lz < h:
                    d = int(hero_dist_map[lz, lx])
                    if d >= 0:
                        bfs_or_bucket = d / float(Config.BFS_MAX_STEPS)
                # 运动矢量
                last_mp = self.last_monster_pos.get(mi)
                if last_mp is not None:
                    delta_dist = (np.hypot(hgx - mx, hgz - mz)
                                  - np.hypot(hgx - last_mp[0], hgz - last_mp[1]))
                    delta_dist = float(np.clip(delta_dist / 4.0, -1.0, 1.0))

            cos_d, sin_d, unk_d = bfs_mod.bucket_direction_to_cos_sin(dir_e)

            monster_feats.append(np.array([
                float(is_in_view),
                pos_x_norm,
                pos_z_norm,
                _norm(speed, Config.MAX_MONSTER_SPEED),
                euclid_dist,
                bfs_or_bucket,
                cos_d,
                sin_d,
                unk_d,
                delta_dist,
            ], dtype=np.float32))

        # ---- treasures top-3 (21 = 3 * 7) ----
        treasure_entries = []
        for e in treasure_local:
            (lx, lz, bucket, dir_e, _st, _gx, _gz) = e
            d = Config.BFS_MAX_STEPS
            first_dir = -1
            bfs_valid = 0.0
            local_blocked = 1.0
            if 0 <= lx < w and 0 <= lz < h:
                dd = int(hero_dist_map[lz, lx])
                if dd >= 0:
                    d = dd
                    first_dir = int(hero_first_dir_map[lz, lx])
                    bfs_valid = 1.0
                    local_blocked = 0.0
            treasure_entries.append((d, bucket, dir_e, first_dir, bfs_valid, local_blocked))
        # 排序：先可达的（BFS 小）；不可达的按桶距离
        treasure_entries.sort(key=lambda t: (t[5], t[0], t[1]))
        treasure_feat = []
        for i in range(Config.TOP_K_TREASURE):
            if i < len(treasure_entries):
                d, bucket, dir_e, first_dir, bfs_valid, local_blocked = treasure_entries[i]
                if d < Config.BFS_MAX_STEPS:
                    dist_norm = d / float(Config.BFS_MAX_STEPS)
                else:
                    dist_norm = bucket / 5.0
                euclid_cos, euclid_sin, _unk = bfs_mod.bucket_direction_to_cos_sin(dir_e)
                bfs_cos, bfs_sin, _unk = bfs_mod.direction_onehot_cos_sin(first_dir)
                treasure_feat.extend([
                    dist_norm,
                    euclid_cos,
                    euclid_sin,
                    bfs_cos,
                    bfs_sin,
                    bfs_valid,
                    local_blocked,
                ])
            else:
                treasure_feat.extend([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        treasure_feat = np.array(treasure_feat, dtype=np.float32)

        # ---- buff top-1 (4) ----
        if buff_local:
            # 选 BFS 最小的
            best = None
            for e in buff_local:
                (lx, lz, bucket, dir_e, status, _gx, _gz) = e
                d = Config.BFS_MAX_STEPS
                if 0 <= lx < w and 0 <= lz < h:
                    dd = int(hero_dist_map[lz, lx])
                    if dd >= 0:
                        d = dd
                if best is None or d < best[0]:
                    best = (d, bucket, dir_e, status)
            d, bucket, dir_e, status = best
            dist_norm = d / float(Config.BFS_MAX_STEPS) if d < Config.BFS_MAX_STEPS else bucket / 5.0
            cos_d, sin_d, _unk = bfs_mod.bucket_direction_to_cos_sin(dir_e)
            buff_feat = np.array([dist_norm, cos_d, sin_d, float(status == 1)], dtype=np.float32)
        else:
            buff_feat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

        # ---- obstacle rays (8) ----
        rays_feat = np.array(
            [r / float(Config.FLASH_RANGE_STRAIGHT) for r in rays],
            dtype=np.float32,
        )

        # ---- flash landing (24 = 8 * 3) ----
        flash_feat = []
        for (dx, dz, cnt) in flash_landings:
            flash_feat.extend([
                dx / float(Config.FLASH_RANGE_STRAIGHT),
                dz / float(Config.FLASH_RANGE_STRAIGHT),
                min(cnt, 5) / 5.0,
            ])
        flash_feat = np.array(flash_feat, dtype=np.float32)

        # ---- last action one-hot (16) ----
        last_oh = np.zeros(16, dtype=np.float32)
        if last_action is not None and 0 <= int(last_action) < 16:
            last_oh[int(last_action)] = 1.0

        # ---- movable mask (8) ----
        movable_feat = np.array(movable_mask, dtype=np.float32)

        # ---- flash useful mask (8) ----
        flash_useful_feat = np.array(flash_useful_mask, dtype=np.float32)

        # ---- flash ready (1) ----
        flash_ready_feat = np.array([flash_ready], dtype=np.float32)

        # ---- phase (4) ----
        max_monster_speed = 1
        for m in monsters:
            if m is None:
                continue
            max_monster_speed = max(max_monster_speed, int(m.get("speed", 1)))
        late_factor = _clip01((max_monster_speed - 1.0) / 1.0)
        steps_to_max = _clip01(1.0 - self.step_no / max(self.max_step, 1))
        phase_feat = np.array([
            max_monster_speed / Config.MAX_MONSTER_SPEED,
            1.0 if late_factor > 0.5 else 0.0,
            flash_ready * (1.0 if late_factor > 0.5 else 0.0),
            steps_to_max,
        ], dtype=np.float32)

        # ---- progress (2) ----
        total_tre = max(int(env_info.get("total_treasure", 1)), 1)
        got_tre = int(env_info.get("treasures_collected", 0))
        progress_feat = np.array([
            _norm(self.step_no, self.max_step),
            got_tre / float(total_tre),
        ], dtype=np.float32)

        # ---- second monster spawn hint (4) ----
        spawn_hint_feat = np.array(second_monster_spawn_hint, dtype=np.float32)

        # ---- anti stall hint (4) ----
        anti_stall_feat = np.array(anti_stall_hint, dtype=np.float32)

        sym_feat = np.concatenate([
            hero_feat,                  # 6
            monster_feats[0],           # 10
            monster_feats[1],           # 10
            treasure_feat,              # 9
            buff_feat,                  # 4
            rays_feat,                  # 8
            flash_feat,                 # 24
            last_oh,                    # 16
            movable_feat,               # 8
            flash_useful_feat,          # 8
            flash_ready_feat,           # 1
            phase_feat,                 # 4
            progress_feat,              # 2
            spawn_hint_feat,            # 4
            anti_stall_feat,            # 4
        ]).astype(np.float32)
        assert sym_feat.shape[0] == Config.SYM_FEATURE_LEN, \
            f"sym_feat dim mismatch: got {sym_feat.shape[0]}, want {Config.SYM_FEATURE_LEN}"

        # 死角分数（超过 DEAD_END_RAY_THRESHOLD 的方向占比）
        blocked = sum(1 for r in rays if r <= Config.DEAD_END_RAY_THRESHOLD)
        dead_end_score = blocked / 8.0

        return sym_feat, dead_end_score

    def _compute_second_monster_spawn_hint(self, hero_global_x, hero_global_z, env_info, monsters):
        """Build a hint for the 2nd monster spawn point.

        先验规则：
          - 第二只怪在 step == monster_interval 时出现
          - 出生点是英雄在前 SECOND_MONSTER_SPAWN_LAG 步的位置

        返回 4 维：
          [is_active, rel_dx_norm, rel_dz_norm, steps_until_spawn_norm]
        """
        monster_interval = int(env_info.get("monster_interval", -1))
        spawn_lag = int(Config.SECOND_MONSTER_SPAWN_LAG)

        if monster_interval <= 0 or spawn_lag <= 0:
            return [0.0, 0.0, 0.0, 0.0]

        if len(monsters) > 1 and monsters[1] is not None:
            return [0.0, 0.0, 0.0, 0.0]

        target_step = monster_interval - spawn_lag
        if self.step_no < target_step or self.step_no >= monster_interval:
            return [0.0, 0.0, 0.0, 0.0]

        delta = self.step_no - target_step
        if delta == 0:
            spawn_x, spawn_z = hero_global_x, hero_global_z
        else:
            if len(self.trajectory) < delta:
                return [0.0, 0.0, 0.0, 0.0]
            spawn_x, spawn_z = self.trajectory[-delta]

        rel_dx = np.clip((spawn_x - hero_global_x) / float(spawn_lag), -1.0, 1.0)
        rel_dz = np.clip((spawn_z - hero_global_z) / float(spawn_lag), -1.0, 1.0)
        steps_until_spawn = monster_interval - self.step_no
        steps_until_spawn_norm = np.clip(steps_until_spawn / float(spawn_lag), 0.0, 1.0)

        return [1.0, float(rel_dx), float(rel_dz), float(steps_until_spawn_norm)]

    def _compute_anti_stall_hint(self, hero_global_x, hero_global_z):
        """Build a hint and trigger flag for anti-stall shaping.

        返回:
          anti_stall_hint: [is_valid, rel_dx_norm, rel_dz_norm, l2_dist_norm]
          anti_stall: 最近 window 步位移过小，触发磨蹭惩罚
        """
        window = int(Config.ANTI_STALL_WINDOW)
        threshold = float(Config.ANTI_STALL_L2_THRESHOLD)
        if window <= 0 or threshold <= 0 or len(self.trajectory) < window:
            return [0.0, 0.0, 0.0, 0.0], False

        past_x, past_z = self.trajectory[-window]
        dx = hero_global_x - past_x
        dz = hero_global_z - past_z
        dist_sq = dx * dx + dz * dz
        anti_stall = dist_sq < (threshold * threshold)

        rel_dx = np.clip(dx / max(threshold, 1.0), -1.0, 1.0)
        rel_dz = np.clip(dz / max(threshold, 1.0), -1.0, 1.0)
        l2_dist_norm = np.clip(np.sqrt(dist_sq) / max(threshold, 1e-6), 0.0, 1.0)
        return [1.0, float(rel_dx), float(rel_dz), float(l2_dist_norm)], anti_stall

    # -----------------------------------------------------------------
    # 合法动作掩码
    # -----------------------------------------------------------------
    def _parse_env_flash_mask(self, legal_act_raw):
        """Return env-provided flash mask (8D bool list) for actions 8..15.

        env 的 legal_action 是 16 维（`环境详述.md:191-197`），后 8 位是闪现。
        """
        if legal_act_raw is None:
            return [1] * 8
        if isinstance(legal_act_raw, np.ndarray):
            legal_act_raw = legal_act_raw.tolist()
        if not isinstance(legal_act_raw, (list, tuple)) or len(legal_act_raw) == 0:
            return [1] * 8
        # bool[16] 或 int[]
        if isinstance(legal_act_raw[0], (bool, np.bool_, int, np.integer)):
            if len(legal_act_raw) >= 16:
                return [int(bool(legal_act_raw[8 + j])) for j in range(8)]
            # 若只有 8 维或其他奇怪形状，回退全 1
            return [1] * 8
        return [1] * 8

    def _build_legal_16(self, movable_mask, flash_ready, flash_useful_mask, env_flash_mask):
        """Build 16D legal action mask.

        前 8 位：真实的 movable_mask（撞墙检查），替代 env 恒 1 的值。
        后 8 位：flash_ready AND env_flash_mask[k] AND flash_useful_mask[k]。
        若全 0（极端死局）则回退保留闪现或全 1。
        """
        mask = [0] * 16
        for k in range(8):
            mask[k] = int(movable_mask[k])
        if flash_ready:
            for k in range(8):
                if env_flash_mask[k] and flash_useful_mask[k]:
                    mask[8 + k] = 1
        if sum(mask) == 0:
            # 第一级回退：允许任何有效闪现
            if flash_ready:
                for k in range(8):
                    if flash_useful_mask[k]:
                        mask[8 + k] = 1
            if sum(mask) == 0:
                # 第二级回退：全 1，免得 softmax NaN
                mask = [1] * 16
        return mask

    # -----------------------------------------------------------------
    # 奖励计算
    # -----------------------------------------------------------------
    def _compute_reward(
        self,
        picked_treasure, picked_buff, discovered_new_treasure, used_flash, flash_path_had_item,
        monster_bfs_increased, bumped_wall, dead_end, anti_stall, late_factor,
        nearest_treasure_bfs, min_monster_bfs,
    ):
        # 事件 & 阶段感知
        r_treasure = (_lerp(Config.R_TREASURE_EARLY, Config.R_TREASURE_LATE, late_factor)
                      if picked_treasure else 0.0)
        r_buff = (_lerp(Config.R_BUFF_EARLY, Config.R_BUFF_LATE, late_factor)
                  if picked_buff else 0.0)
        r_discover = Config.R_TREASURE_DISCOVER if discovered_new_treasure else 0.0
        r_survive = _lerp(Config.R_SURVIVE_EARLY, Config.R_SURVIVE_LATE, late_factor)

        # 闪现相关
        r_flash_path_item = 0.0
        r_flash_escape = 0.0
        r_flash_cost = 0.0
        r_flash_waste = 0.0
        if used_flash:
            # 前期奖励"用闪现吃物件"
            if flash_path_had_item:
                r_flash_path_item = Config.R_FLASH_TO_REACH * (1.0 - late_factor)
            # 后期奖励"用闪现拉开距离"
            if monster_bfs_increased:
                r_flash_escape = Config.R_FLASH_ESCAPE * late_factor
            # 前期轻微使用成本（引导不过度闪现）
            r_flash_cost = Config.R_FLASH_COST_EARLY * (1.0 - late_factor)
            # 后期浪费惩罚（课程渐进式开启）
            if late_factor > 0.5 and not monster_bfs_increased and not flash_path_had_item:
                stage = Config.CURRICULUM_STAGE
                if stage == 3:
                    r_flash_waste = Config.R_FLASH_WASTE_STAGE3
                elif stage >= 4:
                    r_flash_waste = Config.R_FLASH_WASTE_STAGE4

        # 常驻
        r_bump = Config.R_BUMP_WALL if bumped_wall else 0.0
        r_dead = Config.R_DEAD_END if dead_end else 0.0
        r_anti_stall = Config.R_ANTI_STALL if anti_stall else 0.0

        # 势函数塑形：Φ = α·(1 - nearest_treasure_bfs/N) + β·(min_monster_bfs/N)
        N = float(Config.BFS_MAX_STEPS)
        phi = (Config.POT_ALPHA_TREASURE * (1.0 - min(nearest_treasure_bfs / N, 1.0))
               + Config.POT_BETA_MONSTER * min(min_monster_bfs / N, 1.0))
        r_pot = Config.GAMMA * phi - self.last_potential
        self.last_potential = phi

        total = (r_treasure + r_buff + r_discover + r_survive
                 + r_flash_path_item + r_flash_escape + r_flash_cost + r_flash_waste
                 + r_bump + r_dead + r_anti_stall + r_pot)
        return total
