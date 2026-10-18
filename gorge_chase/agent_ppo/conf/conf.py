#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Configuration for Gorge Chase PPO (refactored).

峡谷追猎 PPO 配置（重构版）。
"""

import os


class Config:
    # ===== 视野与地图 / View & Map =====
    MAP_VIEW = 21              # 局部视野边长（官方 21x21）
    MAP_CHANNELS = 6           # 多通道地图：C0 障碍 / C1 宝箱 / C2 buff / C3 怪物 / C4 英雄 / C5 轨迹
    MAP_FLAT_LEN = MAP_CHANNELS * MAP_VIEW * MAP_VIEW  # 2646

    # ===== 符号特征拆分 / Symbolic feature split =====
    #   hero(6), m0(10), m1(10), treasures_top3(21), buff_top1(4),
    #   rays_obstacle(8), flash_landing(24), last_action_onehot(16),
    #   movable_mask(8), flash_useful_mask(8), flash_ready(1),
    #   phase(4), progress(2), second_monster_spawn_hint(4), anti_stall_hint(4)
    SYM_FEATURE_SPLIT = [6, 10, 10, 21, 4, 8, 24, 16, 8, 8, 1, 4, 2, 4, 4]
    SYM_FEATURE_LEN = sum(SYM_FEATURE_SPLIT)  # 130

    DIM_OF_OBSERVATION = SYM_FEATURE_LEN  # 兼容旧命名

    # ===== 动作空间 / Action space =====
    ACTION_NUM = 16            # 8 移动 + 8 闪现
    VALUE_NUM = 1

    # ===== PPO 超参数 / PPO hyperparameters =====
    GAMMA = 0.995
    LAMDA = 0.97               # 0.95 -> 0.97，跨越更长的宝箱轨迹做信用分配
    INIT_LEARNING_RATE_START = 3e-4
    BETA_START = 0.02          # 熵系数起始值（探索）
    BETA_END = 0.005           # 熵系数退火终值（利用）
    BETA_ANNEAL_STEPS = 2_000_000
    CLIP_PARAM = 0.2
    VF_COEF = 1.0
    GRAD_CLIP_RANGE = 0.5

    # ===== 归一化常量 / Normalization constants =====
    MAP_SIZE = 128.0
    MAX_MONSTER_SPEED = 2.0    # 官方加速后最大速度
    FLASH_RANGE_STRAIGHT = 10
    FLASH_RANGE_DIAGONAL = 8
    FLASH_COOLDOWN_DEFAULT = 100   # 用配置默认值做归一化基准
    MAX_BUFF_DURATION = 50.0
    BFS_MAX_STEPS = 42         # 21x21 对角最大理论距离上限，够用

    # ===== 8 方向偏移（对齐动作协议）=====
    # 0=East, 1=NE, 2=North, 3=NW, 4=West, 5=SW, 6=South, 7=SE
    DIR_OFFSETS = [
        (+1, 0),   # 0 East
        (+1, -1),  # 1 NE
        (0, -1),   # 2 North
        (-1, -1),  # 3 NW
        (-1, 0),   # 4 West
        (-1, +1),  # 5 SW
        (0, +1),   # 6 South
        (+1, +1),  # 7 SE
    ]
    FLASH_DISTANCES = [
        FLASH_RANGE_STRAIGHT,  # E
        FLASH_RANGE_DIAGONAL,  # NE
        FLASH_RANGE_STRAIGHT,  # N
        FLASH_RANGE_DIAGONAL,  # NW
        FLASH_RANGE_STRAIGHT,  # W
        FLASH_RANGE_DIAGONAL,  # SW
        FLASH_RANGE_STRAIGHT,  # S
        FLASH_RANGE_DIAGONAL,  # SE
    ]

    # ===== 宝箱 / buff top-K =====
    TOP_K_TREASURE = 3
    TOP_K_BUFF = 1

    # ===== 轨迹通道衰减步数 =====
    TRAJECTORY_LEN = 10
    SECOND_MONSTER_SPAWN_LAG = 10
    ANTI_STALL_WINDOW = 10
    ANTI_STALL_L2_THRESHOLD = 5.0

    # ===== 奖励权重（前期 monster.speed == 1）=====
    R_TREASURE_EARLY = 2.0
    R_BUFF_EARLY = 2.0
    R_FLASH_TO_REACH = 0.3     # 闪现路径含宝箱/buff
    R_FLASH_COST_EARLY = -0.02
    R_SURVIVE_EARLY = 0.005

    # ===== 奖励权重（后期 monster.speed > 1）=====
    R_TREASURE_LATE = 1.0
    R_BUFF_LATE = 3.0
    R_FLASH_ESCAPE = 0.8       # 闪现使怪物 BFS 距离增加 >= 3
    R_FLASH_WASTE_STAGE3 = -0.05
    R_FLASH_WASTE_STAGE4 = -0.25
    R_SURVIVE_LATE = 0.02

    # ===== 奖励（常驻）=====
    R_BUMP_WALL = -0.1
    R_DEAD_END = -0.05
    R_ANTI_STALL = -0.08
    R_TREASURE_DISCOVER = 0.08
    R_WIN = 10.0
    R_FAIL = -10.0
    R_SCORE_SHAPED_COEF = 0.01  # 终局额外 = total_score * 此系数

    # ===== 势函数塑形 =====
    POT_ALPHA_TREASURE = 0.3
    POT_BETA_MONSTER = 0.5

    # ===== 规则接管 =====
    RULE_ADJACENT_TREASURE = True
    RULE_PANIC_FLASH = True
    # 阈值说明：英雄与怪物 BFS 距离 <= 阈值 时触发闪现逃生
    # 文档规则：距离 <= 1 即被抓。所以阈值必须 >= 2 才来得及救。
    #   前期（怪物速度=1）：BFS=2 可触发，即"怪物下一步能抓到" → 立即闪
    #   后期（怪物速度=2）：BFS=3 可触发，因为怪物一步跨两格，等 BFS=2 已不够
    PANIC_FLASH_THRESHOLD_EARLY = 2
    PANIC_FLASH_THRESHOLD_LATE = 3

    # ===== 死角判定 =====
    DEAD_END_RAY_THRESHOLD = 2     # 射线 <= 2 视为受阻方向
    DEAD_END_RATIO_THRESHOLD = 0.6 # 超过 60% 方向受阻则算死角

    # ===== 课程 =====
    # 课程自动切换：按累积 episode 数确定阶段
    #   episode_cnt <  8000                -> stage 1  (基础生存)
    #   episode_cnt <  22000 (8000+14000)  -> stage 2  (拾取训练)
    #   episode_cnt <  52000 (22000+30000) -> stage 3  (博弈训练)
    #   episode_cnt >= 52000               -> stage 4  (最终挑战)
    # 每阶段步数比初版更长，给模型充分收敛时间
    CURRICULUM_AUTO = False
    # 阈值说明（重启续训模式 + 24h 总预算）：
    #   ep_cnt 在重启后从 0 开始（因为旧 session 没保存进度文件）
    #   Stage 1 已在上次训练中完成（WIN率 ~40%，基础技能已学会），模型权重通过 ckpt 续
    #   [0, 2500, 5500] 让重启的第 1 局就进入 Stage 2，省去重新跑 Stage 1
    #     ep 0 → Stage 2 起步（直接利用已训 Stage 1 权重）
    #     ep 2500 → Stage 3（怪物加速 + 10 宝箱）
    #     ep 5500 → Stage 4（完整默认配置）
    STAGE_EPISODE_THRESHOLDS = [0, 2500, 5500]
    # 切换日志节流（避免每局都打印）
    CURRICULUM_LOG_EVERY = 1  # 切换时必打印

    # 手动覆盖：若设置 CURRICULUM_STAGE_OVERRIDE 环境变量为 1~4，则强制使用该阶段，
    # 忽略 episode_cnt 自动推进。值为 0 或未设置时使用自动切换。
    CURRICULUM_STAGE_OVERRIDE = int(os.environ.get("CURRICULUM_STAGE_OVERRIDE", "0"))

    # 当前阶段缓存（由 workflow 动态更新，preprocessor 读取以决定惩罚项等）
    CURRICULUM_STAGE = 1
