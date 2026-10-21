#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""


import numpy as np


class GameConfig:
    HERO_NUM = 3
    # Model save interval configuration, used in workflow
    # 模型保存间隔配置，在workflow中使用
    MODEL_SAVE_INTERVAL = 1800
    SEND_SAMPLE_NUM = 1920
    # Aggregate episode and platform metrics into at most one report per actor
    # process per minute, leaving headroom below the monitor proxy quota.
    MONITOR_REPORT_INTERVAL_SECONDS = 60
    # Routine actor logs and shared platform-metric RPCs are diagnostic only.
    # With 512 environments, rate-limit each process and stagger RPC startup
    # rather than emitting/fetching at the same episode-count boundary.
    ROUTINE_LOG_INTERVAL_SECONDS = 300
    PLATFORM_METRICS_FETCH_INTERVAL_SECONDS = 300
    PLATFORM_METRICS_FETCH_EPISODE_INTERVAL = 10


class FrozenV1Config:
    """Immutable-by-convention constants for the active training contract.

    The class name is retained for platform import compatibility.  This class
    is the authoritative runtime contract. ``LegacyConfig`` below is retained
    only so old checkpoints can be identified and rejected with a useful
    signature error; it is not used by the active agent.
    """

    FEATURE_VERSION = "feature_schema_v21_ppo35_task_safety_semantics"
    NETWORK_VERSION = "network_architecture_3v3_hybrid_multivalue_team_macro"
    SAMPLE_PROTOCOL_VERSION = "sample_protocol_hybrid_multivalue_team_macro_v3"
    LEGALIZER_VERSION = (
        "collaboration_rule_legalizer_v25_ppo35_navigation_watchdog"
    )
    ORDINARY_ACTION_LEGALIZER_VERSION = (
        "ordinary_action_legalizer_v23_ppo35_spawn_tower_safety_flash_button8"
    )

    CONTROLLED_ACTOR_COUNT = 2
    TEAM_HERO_COUNT = 3
    HERO_SLOT_COUNT = 6

    FEATURE_BLOCK_NAMES = (
        "image",
        "heroes",
        "soldiers",
        "organs",
        "monsters",
        "bullets",
        "targets",
        "hero_relations",
        "dynamic_lane_position",
        "tyrant_local_position",
        "tyrant_team_context",
        "lane_state",
        "resource_anchors",
        "collaboration_state",
        "global_state",
        "jungle_plan",
        "lane_plan",
        "main_context",
    )
    FEATURE_BLOCK_SHAPES = (
        (6, 17, 17),
        (6, 393),
        (20, 56),
        (6, 48),
        (20, 60),
        (10, 32),
        (39, 48),
        (6, 24),
        (6, 72),
        (1, 54),
        (1, 24),
        (1, 96),
        (5, 32),
        (1, 64),
        (1, 128),
        (1, 67),
        (1, 64),
        (1, 32),
    )
    IMAGE_SHAPE = (6, 17, 17)
    IMAGE_FEATURE_DIM = 1734
    STRUCTURED_FEATURE_DIM = 8423
    FEATURE_DIM = 10157

    HERO_ACTION_HEADS = (13, 25, 42, 42, 39)
    COMMAND_HEADS = (11, 39, 30)
    RAW_LEGAL_ACTION_DIM = 629
    CONDITIONED_LEGAL_ACTION_DIM = 161
    COMMAND_LEGAL_MASK_DIM = 80

    RNN_TYPE = "GRU"
    RNN_HIDDEN_SIZE = 128
    RNN_TIME_STEPS = 16

    HYBRID_VALUE_COMPONENTS = ("self", "team", "enemy")
    HYBRID_VALUE_FORMULA = "(1-team_lambda)*self+team_lambda*team-enemy"
    HYBRID_VALUE_TEAM_LAMBDA = 0.2
    HYBRID_VALUE_INPUT_DIMS = (
        ("self", 384),
        ("team", 256),
        ("enemy", 448),
    )
    AUXILIARY_VALUE_CATEGORIES = (
        "farming",
        "kda",
        "damage",
        "pushing",
        "win",
    )
    # Indices in the official 11 reward-detail fields.  These values have
    # already been multiplied by train_env_conf.toml weights by the environment.
    AUXILIARY_REWARD_COMPONENT_INDICES = (
        (1, 2),
        (4, 5, 6),
        (0, 7, 8),
        (3, 10),
        (9,),
    )
    AUXILIARY_VALUE_COUNT = 5

    HERO_POLICY_ROUTES = (
        (132, "marco"),
        (141, "diaochan"),
        (193, "kai"),
    )

    ACTOR_ACTION_COUNT = 5
    ACTOR_SUB_ACTION_MASK_DIM = 5
    ACTOR_STEP_DIM = 10498
    COMMAND_ACTION_COUNT = 3
    COMMAND_ACTIVE_HEAD_MASK_DIM = 3
    COMMAND_STEP_DIM = 92
    ACTOR_SEQUENCE_PER_ACTOR_DIM = 168096
    ACTOR_SEQUENCES_DIM = 336192
    COMMAND_SEQUENCE_DIM = 1472
    SAMPLE_DIM = 337664

    @classmethod
    def base_checkpoint_signature(cls):
        """Return a fresh, serialization-safe copy of the frozen signature."""

        return {
            "feature_version": cls.FEATURE_VERSION,
            "network_version": cls.NETWORK_VERSION,
            "sample_protocol_version": cls.SAMPLE_PROTOCOL_VERSION,
            "legalizer_version": cls.LEGALIZER_VERSION,
            "ordinary_action_legalizer_version": cls.ORDINARY_ACTION_LEGALIZER_VERSION,
            "reward_engineering_version": V1TrainingConfig.REWARD_ENGINEERING_VERSION,
            "controlled_actor_count": cls.CONTROLLED_ACTOR_COUNT,
            "feature_per_actor": cls.FEATURE_DIM,
            "structured_per_actor": cls.STRUCTURED_FEATURE_DIM,
            "image_per_actor": cls.IMAGE_FEATURE_DIM,
            "hero_action_heads": cls.HERO_ACTION_HEADS,
            "command_heads": cls.COMMAND_HEADS,
            "rnn_type": cls.RNN_TYPE,
            "rnn_hidden_size": cls.RNN_HIDDEN_SIZE,
            "rnn_time_steps": cls.RNN_TIME_STEPS,
            "sample_dim_v1": cls.SAMPLE_DIM,
            "hero_policy_routes": cls.HERO_POLICY_ROUTES,
            "hybrid_value_components": cls.HYBRID_VALUE_COMPONENTS,
            "hybrid_value_formula": cls.HYBRID_VALUE_FORMULA,
            "hybrid_value_team_lambda": cls.HYBRID_VALUE_TEAM_LAMBDA,
            "hybrid_value_input_dims": cls.HYBRID_VALUE_INPUT_DIMS,
            "auxiliary_value_categories": cls.AUXILIARY_VALUE_CATEGORIES,
            "auxiliary_reward_component_indices": cls.AUXILIARY_REWARD_COMPONENT_INDICES,
            "auxiliary_value_count": cls.AUXILIARY_VALUE_COUNT,
        }

    @classmethod
    def validate(cls):
        """Fail at import time if an active contract dimension drifts."""

        def require(condition, message):
            if not condition:
                raise RuntimeError(f"invalid active training contract: {message}")

        block_dims = [int(np.prod(shape)) for shape in cls.FEATURE_BLOCK_SHAPES]
        require(len(cls.FEATURE_BLOCK_NAMES) == len(block_dims), "feature block name/shape count mismatch")
        require(block_dims[0] == cls.IMAGE_FEATURE_DIM, "image dimension mismatch")
        require(sum(block_dims[1:]) == cls.STRUCTURED_FEATURE_DIM, "structured dimension mismatch")
        require(sum(block_dims) == cls.FEATURE_DIM, "full feature dimension mismatch")

        require(sum(cls.HERO_ACTION_HEADS) == cls.CONDITIONED_LEGAL_ACTION_DIM, "hero action dimension mismatch")
        raw_legal_dim = sum(cls.HERO_ACTION_HEADS[:-1]) + cls.HERO_ACTION_HEADS[0] * cls.HERO_ACTION_HEADS[-1]
        require(raw_legal_dim == cls.RAW_LEGAL_ACTION_DIM, "raw legal-action dimension mismatch")
        require(sum(cls.COMMAND_HEADS) == cls.COMMAND_LEGAL_MASK_DIM, "command legal-mask dimension mismatch")

        actor_step_dim = (
            cls.FEATURE_DIM
            + cls.CONDITIONED_LEGAL_ACTION_DIM
            + 1  # return
            + 1  # advantage
            + cls.ACTOR_ACTION_COUNT
            + cls.CONDITIONED_LEGAL_ACTION_DIM  # old action probabilities
            + 1  # is_train
            + cls.ACTOR_SUB_ACTION_MASK_DIM
            + cls.AUXILIARY_VALUE_COUNT  # five auxiliary value returns
            + 1  # per-step GRU reset mask
        )
        require(actor_step_dim == cls.ACTOR_STEP_DIM, "actor step dimension mismatch")

        command_step_dim = (
            cls.COMMAND_LEGAL_MASK_DIM
            + 1  # return
            + 1  # advantage
            + cls.COMMAND_ACTION_COUNT
            + cls.COMMAND_ACTION_COUNT  # old log probabilities
            + 1  # is_train
            + cls.COMMAND_ACTIVE_HEAD_MASK_DIM
        )
        require(command_step_dim == cls.COMMAND_STEP_DIM, "command step dimension mismatch")

        actor_sequence_per_actor = cls.ACTOR_STEP_DIM * cls.RNN_TIME_STEPS + cls.RNN_HIDDEN_SIZE
        require(
            actor_sequence_per_actor == cls.ACTOR_SEQUENCE_PER_ACTOR_DIM,
            "per-actor sequence dimension mismatch",
        )
        require(
            actor_sequence_per_actor * cls.CONTROLLED_ACTOR_COUNT == cls.ACTOR_SEQUENCES_DIM,
            "all-actor sequence dimension mismatch",
        )
        require(
            cls.COMMAND_STEP_DIM * cls.RNN_TIME_STEPS == cls.COMMAND_SEQUENCE_DIM,
            "command sequence dimension mismatch",
        )
        require(
            cls.ACTOR_SEQUENCES_DIM + cls.COMMAND_SEQUENCE_DIM == cls.SAMPLE_DIM,
            "sample dimension mismatch",
        )
        require(tuple(hero_id for hero_id, _ in cls.HERO_POLICY_ROUTES) == (132, 141, 193), "hero route mismatch")
        require(
            len(cls.HYBRID_VALUE_COMPONENTS) == 3,
            "hybrid value component count mismatch",
        )
        require(
            0.0 <= cls.HYBRID_VALUE_TEAM_LAMBDA <= 1.0,
            "hybrid value team lambda must be in [0,1]",
        )
        require(
            len(cls.AUXILIARY_VALUE_CATEGORIES) == cls.AUXILIARY_VALUE_COUNT,
            "auxiliary value category count mismatch",
        )
        require(
            len(set(cls.AUXILIARY_VALUE_CATEGORIES)) == cls.AUXILIARY_VALUE_COUNT,
            "auxiliary value categories must be unique",
        )
        require(
            len(cls.AUXILIARY_REWARD_COMPONENT_INDICES)
            == cls.AUXILIARY_VALUE_COUNT,
            "auxiliary reward mapping count mismatch",
        )
        require(
            sorted(
                index
                for group in cls.AUXILIARY_REWARD_COMPONENT_INDICES
                for index in group
            )
            == list(range(11)),
            "auxiliary reward mappings must partition the 11 detail fields",
        )


FrozenV1Config.validate()


class V1TrainingConfig:
    """PPO hyperparameters for the active hybrid multi-value path."""

    # One shuffled block contains every ordered blue/red official-hero pairing
    # once, then duplicates only the four cross-side matchups in which one camp
    # has official Kai and the other camp's learned policy controls Kai.  This
    # produces a 13-game block without adding another official-Kai mirror game.
    OFFICIAL_LINEUP_BLOCK = (
        (132, 132),
        (132, 141),
        (132, 193),
        (141, 132),
        (141, 141),
        (141, 193),
        (193, 132),
        (193, 141),
        (193, 193),
        (132, 193),
        (141, 193),
        (193, 132),
        (193, 141),
    )

    # Compatibility bootstrap: platform model 330926 was packaged before the
    # CommandRuntimeContextV1 model-pool reload fix and cannot be repaired in
    # place. Keep custom-opponent scheduling disabled until this source package
    # has produced a new platform model ID, then put that ID in kaiwu.json and
    # re-enable the exact 9:1 scheduler.
    OPPONENT_MIX_ENABLED = False
    OPPONENT_SELFPLAY_GAMES_PER_BLOCK = 9
    OPPONENT_ANCHOR_GAMES_PER_BLOCK = 1
    OPPONENT_ANCHOR_MODEL_POOL_INDEX = 0
    OPPONENT_MIX_SEED = 20260715

    GAMMA = 0.995
    LAMBDA = 0.95
    CLIP_PARAM = 0.2
    # Dual-Clip PPO lower-bounds the surrogate for negative advantages when
    # stale behavior-policy samples produce an excessively large ratio.
    DUAL_CLIP_PARAM = 3.0
    VALUE_COEF = 0.5
    AUXILIARY_VALUE_COEF = 0.1
    HYBRID_COMPONENT_REG_COEF = 1e-4
    ENTROPY_COEF = 0.015
    COMMAND_COEF = 1.0
    COMMAND_VALUE_COEF = 0.5
    COMMAND_ENTROPY_COEF = 0.015
    LEARNING_RATE = 0.0006
    USE_GRAD_CLIP = True
    GRAD_CLIP_RANGE = 0.5
    PROBABILITY_EPSILON = 1e-12
    ADVANTAGE_NORMALIZATION = True
    ADVANTAGE_EPSILON = 1e-8
    # Keep the effective learner batch at 1024 while reducing one update from
    # eight micro-batches to four.  This changes execution granularity only;
    # global normalization and the optimizer-step boundary remain at 1024.
    MICRO_BATCH_SIZE = 256
    AMP_ENABLED = True
    # Formal stability takes priority over the small synchronization saving:
    # validate every learner update until the Reverb failure chain is closed.
    RUNTIME_VALIDATION_STARTUP_UPDATES = 4
    RUNTIME_VALIDATION_INTERVAL = 1
    PACKED_SAMPLE_FULL_VALIDATION_STARTUP_SAMPLES = 8
    PACKED_SAMPLE_FULL_VALIDATION_INTERVAL = 512
    # Small potential-style guidance for a controlled Kai. The crystal reward
    # remains the final objective; official Kai follows the command legalizer.
    JUNGLE_ROUTE_PROGRESS_DISTANCE = 5000.0
    JUNGLE_ROUTE_PROGRESS_REWARD = 0.01
    JUNGLE_ENEMY_SIDE_STEP_PENALTY = 0.002
    JUNGLE_ROUTE_STUCK_PENALTY = 0.001
    JUNGLE_LANE_SOAK_MAX_SECONDS = 20.0

    # Custom shaping is deliberately small relative to the terminal crystal
    # signal and the official economy rewards.  It is applied only to the
    # final scalar PPO return; the five official auxiliary returns keep their
    # existing protocol and interpretation.
    LANE_LAST_HIT_REWARD = 0.05
    LANE_ATTENDANCE_END_FRAME = 1800
    LANE_ATTENDANCE_PROGRESS_DISTANCE = 5000.0
    LANE_ATTENDANCE_PROGRESS_REWARD = 0.002
    LANE_ATTENDANCE_NEAR_DISTANCE = 12000.0
    NO_ATTACKABLE_TARGET_PENALTY = 0.0005
    NO_ATTACKABLE_TARGET_START_FRAME = 450
    NO_ATTACKABLE_TARGET_RECOVERY_HP_RATIO = 0.30
    NO_ATTACKABLE_TARGET_GRACE_FRAMES = 15
    NO_ATTACKABLE_TARGET_REFERENCE_FRAMES = 6
    NO_ATTACKABLE_TARGET_MAX_ELAPSED_FRAMES = 30
    STRATEGIC_WAIT_RADIUS = 8000.0
    STRATEGIC_CAMP_WAIT_SECONDS = 5.0

    # Checkpoint-compatible reward engineering used by the next formal
    # continuation.  These values deliberately live outside FrozenV1Config:
    # no feature, action, sample, recurrent-state, model, or checkpoint shape
    # changes when this reward version is enabled.
    REWARD_ENGINEERING_VERSION = "ppo_v3_5_reward_state_machine_v3"
    HERO_MONSTER_REWARD_WEIGHTS = {
        132: 0.05,   # Marco Polo / ADC
        141: 0.005,  # Diaochan / mid
        193: 0.08,   # Kaizer / jungle
    }
    # The environment can route atk_monster by hero, but cannot distinguish
    # monster species. Initial wolves are already guaranteed by the protected
    # opening, so only later lifecycles receive this team-symmetric result
    # bonus. Any hero may secure the camp.
    KAI_WOLF_MEMBER_REWARDS = {
        44: 0.40,
        45: 0.10,
    }
    TREASURE_THIEF_CONFIG_ID = 59
    TREASURE_THIEF_TEAM_REWARD = 0.50
    SPEED_BUFF_CONFIG_ID = 90201
    SPEED_BUFF_TEAM_REWARD = 0.12
    KILL_REWARD = 0.8
    DEATH_REWARD = -1.0
    ASSIST_REWARD = 0.8
    DEATH_MULTIPLIER_BEFORE_CANNON = 0.7
    DEATH_MULTIPLIER_AFTER_CANNON = 1.3
    DEATH_MULTIPLIER_CANNON_FRAME = 6254
    TYRANT_CONFIG_ID = 122
    TYRANT_TEAM_REWARD = 1.0
    TYRANT_REWARD_MAX_KILLS_PER_CAMP = 3
    TEAMMATE_SUPPORT_WINDOW_FRAMES = 90
    TEAMMATE_SUPPORT_RADIUS = 12000.0
    TEAMMATE_SUPPORT_LOW_HP_EXEMPT_RATIO = 0.25
    TEAMMATE_ABANDONMENT_PENALTY = 0.06
    TEAMMATE_SUPPORT_REWARD = 0.05
    TEMPORAL_FOCUS_WINDOW_FRAMES = 90
    TEMPORAL_FOCUS_LOCAL_RADIUS = 12000.0
    TEMPORAL_FOCUS_MIN_PARTICIPANTS = 2
    TEMPORAL_FOCUS_REWARD = 0.05
    GROUP_ATTACK_MIN_PARTICIPANTS = 3
    GROUP_ATTACK_REWARD = 0.08
    DEFENSE_PRESSURE_EVALUATION_FRAMES = 15
    DEFENSE_PRESSURE_FRONT_MARGIN = 4000.0
    DEFENSE_PRESSURE_FRONT_PENALTY = 0.015
    DEFENSE_PRESSURE_UNDER_TOWER_PENALTY = 0.03
    DEFENSE_PRESSURE_WINDOW_CAP = 0.15
    DEFENSE_PRESSURE_LOW_HP_THRESHOLD = 0.50
    DEFENSE_PRESSURE_CRITICAL_HP_THRESHOLD = 0.25
    DEFENSE_PRESSURE_LOW_HP_MULTIPLIER = 1.25
    DEFENSE_PRESSURE_CRITICAL_HP_MULTIPLIER = 1.50
    CRYSTAL_DANGER_RADIUS = 18000.0
    CRYSTAL_DANGER_HERO_BUDGET = 0.04
    CRYSTAL_DANGER_MINION_BUDGET = 0.02
    CRYSTAL_DANGER_LOW_HP_THRESHOLD = 0.50
    CRYSTAL_DANGER_CRITICAL_HP_THRESHOLD = 0.25
    CRYSTAL_DANGER_LOW_HP_MULTIPLIER = 1.25
    CRYSTAL_DANGER_CRITICAL_HP_MULTIPLIER = 1.50
    COMMAND_CONTROLLED_HERO_WEIGHT = 0.25
    COMMAND_OFFICIAL_HERO_WEIGHT = 0.50
    # A confirmed trainable collaboration command is treated as a bounded
    # temporal option.  Moving the official hero toward the accepted goal is
    # useful shaping, but the complete window may contribute at most 0.03 so
    # movement can never dominate KDA, structures, Tyrant, or crystal results.
    COMMAND_GOAL_PROGRESS_BUDGET = 0.03
    COMMAND_FARM_CAMP_COMPLETION_REWARD = 0.03
    COMMAND_TASK_NAVIGATION_WINDOW_FRAMES = 76
    COMMAND_TASK_ENGAGEMENT_WINDOW_FRAMES = 152
    COMMAND_TASK_PERSISTENCE_WINDOW_FRAMES = 152
    COMMAND_TASK_HARD_DEADLINE_FRAMES = 600
    # Response shaping is attached only to a trainable command after the
    # platform echoes the exact target/position.  Forced opening commands and
    # cooldown-only hold steps never create an attribution window.
    COMMAND_RESPONSE_WINDOW_FRAMES = 76
    COMMAND_RESPONSE_MIN_PROGRESS_DISTANCE = 1500.0
    COMMAND_TARGET_LOCK_REWARD = 0.004
    COMMAND_TARGET_RANGE_REWARD = 0.004
    COMMAND_TARGET_FIRST_HIT_REWARD = 0.012
    COMMAND_NO_RESPONSE_PENALTY = 0.01
    # Repeated trainable no-command choices may be penalized only while there
    # is no active/pending command, cooldown is ready, a meaningful objective
    # exists, and the official hero makes no combat or net-position progress.
    COMMAND_NO_COMMAND_IDLE_WINDOW_FRAMES = 76
    COMMAND_NO_COMMAND_IDLE_DECISION_MAX_AGE_FRAMES = 30
    COMMAND_NO_COMMAND_IDLE_MIN_PROGRESS_DISTANCE = 1500.0
    COMMAND_NO_COMMAND_IDLE_PENALTY = 0.005
    # Ordinary monster SetTarget is only exposed to an official Kai after the
    # robust SetPosition approach has brought it near the current lifecycle.
    # Tyrant keeps its separately confirmed target semantics.
    OFFICIAL_KAI_MONSTER_SET_TARGET_DISTANCE = 5000.0
    TERMINAL_CRYSTAL_REWARD = 8.0
    # Keep the old name as a source-compatibility alias for checkpoint-side
    # tooling that imports it.  The reward is now shared by actor and command
    # returns; it is no longer command-only.
    COMMAND_TERMINAL_CRYSTAL_REWARD = TERMINAL_CRYSTAL_REWARD

    # A complete outer/high tower remains worth 1.5, matching the disabled
    # official tower term.  The convex potential moves more of that fixed
    # budget toward the low-HP, high-conversion part of the tower lifecycle.
    STRUCTURE_TOWER_TOTAL_REWARD = 2.0
    STRUCTURE_POTENTIAL_BETA = 0.25
    STRUCTURE_POTENTIAL_POWER = 2.0
    STRUCTURE_TEAM_FLOOR_RATIO = 0.50
    STRUCTURE_PERSONAL_ATTRIBUTION_RATIO = 0.50

    # PPO 3.5 tower-risk state. Warning and target lock are diagnostic; only
    # attributable tower damage changes the reward.
    TOWER_RISK_CRITICAL_HP_RATIO = 0.45
    TOWER_RISK_EXIT_FRAMES = 12
    TOWER_RISK_HIT_PENALTIES = (0.01, 0.02, 0.03)
    TOWER_RISK_EVENT_PENALTY_CAP = 0.12

    # PPO 3.5 defensive-wave window. Front-zone values are scaled uniformly.
    DEFENSE_WAVE_FRONT_MARGIN = 4000.0
    DEFENSE_WAVE_CLEAR_FRAMES = 15
    DEFENSE_WAVE_IGNORE_AGE_FRAMES = 30
    DEFENSE_WAVE_RESPONSE_REWARD = 0.005
    DEFENSE_WAVE_SOLDIER_PROGRESS_CAP = 0.01
    DEFENSE_WAVE_PROGRESS_CAP = 0.03
    DEFENSE_WAVE_COMPLETION_REWARD = 0.01
    DEFENSE_WAVE_POSITIVE_CAP = 0.06
    DEFENSE_WAVE_IGNORE_PENALTY = 0.02
    DEFENSE_WAVE_FRONT_SCALE = 0.50
    # A missing soldier is not treated as cleared.  The window remains
    # unresolved for one existing command-response horizon and is then
    # neutral-invalidated so an incomplete observation cannot manufacture a
    # completion reward or permanently poison later defense windows.
    DEFENSE_WAVE_MISSING_GRACE_FRAMES = COMMAND_RESPONSE_WINDOW_FRAMES
    DEFENSE_WAVE_LOW_HP_EXEMPT_RATIO = (
        NO_ATTACKABLE_TARGET_RECOVERY_HP_RATIO
    )
    # Shadow/diagnostic switch: counters and attribution keep running at zero.
    DEFENSE_WAVE_REWARD_SCALE = 1.0

    # Each enemy hero death opens one bounded conversion opportunity.  It is
    # paid only when an enemy outer/high tower or crystal actually loses HP.
    POST_KILL_PUSH_WINDOW_FRAMES = 225
    POST_KILL_PUSH_REWARD = 0.2
    POST_KILL_PUSH_MAX_TOKENS = 3

    # The detector starts after the first 23 seconds, then evaluates a trailing
    # five-second trajectory once per second.  Persistent violations receive a
    # smooth nonlinear pulse that saturates at -0.30 per evaluation.
    LONG_INACTIVITY_START_FRAME = 349
    LONG_INACTIVITY_WINDOW_FRAMES = 76
    LONG_INACTIVITY_EVALUATION_INTERVAL_FRAMES = 15
    LONG_INACTIVITY_PENALTY_MIN = 0.03
    LONG_INACTIVITY_PENALTY_MAX = 0.30
    LONG_INACTIVITY_PENALTY_TAU = 4.0
    LONG_INACTIVITY_STATIC_PATH_DISTANCE = 1200.0
    LONG_INACTIVITY_STATIC_MAX_EXCURSION = 800.0
    LONG_INACTIVITY_WANDER_NET_DISTANCE = 3000.0
    LONG_INACTIVITY_WANDER_EFFICIENCY = 0.45
    LONG_INACTIVITY_WANDER_MIN_SEGMENT = 300.0
    LONG_INACTIVITY_WANDER_REVERSE_COSINE = -0.342
    LONG_INACTIVITY_WANDER_MIN_REVERSALS = 2
    LONG_INACTIVITY_WANDER_REVISIT_FRAMES = 30
    LONG_INACTIVITY_WANDER_REVISIT_RADIUS = 1000.0
    LONG_INACTIVITY_OBJECTIVE_PROGRESS_DISTANCE = 1500.0
    LONG_INACTIVITY_ROTATION_NET_DISTANCE = 6000.0
    LONG_INACTIVITY_ROTATION_EFFICIENCY = 0.60
    LONG_INACTIVITY_TYRANT_RADIUS = 7000.0
    LONG_INACTIVITY_TYRANT_SPAWN_SECONDS = 5.0


# Configuration related to model and algorithms used
# 模型和算法使用的相关配置
class LegacyConfig:
    HERO_NUM = GameConfig.HERO_NUM
    feature_img_shape = [6, 17, 17]
    feature_hero_frd_shape = [3, 86]
    feature_hero_emy_shape = [3, 86]
    feature_soldier_frd_shape = [10, 18]
    feature_soldier_emy_shape = [10, 18]
    feature_organ_frd_shape = [3, 21]
    feature_organ_emy_shape = [3, 21]
    feature_monster_shape = [20, 20]
    feature_global_info_shape = [1, 47]
    feature_hero_main_shape = [1, 86]

    HERO_SERI_VEC_SPLIT_SHAPE = [
        feature_img_shape,
        feature_hero_frd_shape,
        feature_hero_emy_shape,
        feature_soldier_frd_shape,
        feature_soldier_emy_shape,
        feature_organ_frd_shape,
        feature_organ_emy_shape,
        feature_monster_shape,
        feature_global_info_shape,
        feature_hero_main_shape,
    ]

    LSTM_TIME_STEPS = 16
    LSTM_UNIT_SIZE = 16

    HERO_LABEL_SIZE_LIST = [13, 25, 42, 42, 39]
    legal_action_shape = HERO_LABEL_SIZE_LIST
    feature_shape = [abs(sum([int(np.prod(x)) for x in HERO_SERI_VEC_SPLIT_SHAPE]))]
    reward_sum_shape = [1]
    advantage_shape = [1]

    action_shape = [1] * len(HERO_LABEL_SIZE_LIST)
    probs_shape = HERO_LABEL_SIZE_LIST
    is_train_shape = [1]

    sub_action_shape = [1] * len(HERO_LABEL_SIZE_LIST)

    HERO_DATA_SPLIT_SHAPE = (
        feature_shape
        + legal_action_shape
        + reward_sum_shape
        + advantage_shape
        + action_shape
        + probs_shape
        + is_train_shape
        + sub_action_shape
    )

    HERO_DATA_SPLIT_SHAPE_LIST = [
        feature_shape,
        legal_action_shape,
        reward_sum_shape,
        advantage_shape,
        action_shape,
        probs_shape,
        is_train_shape,
        sub_action_shape,
    ]

    SAMPLE_DIM_LIST = [[sum(HERO_DATA_SPLIT_SHAPE) * LSTM_TIME_STEPS + LSTM_UNIT_SIZE * 2]] * HERO_NUM
    sample_one_size = np.sum(HERO_DATA_SPLIT_SHAPE)
    DATA_SPLIT_SHAPE = feature_shape + legal_action_shape
    SERI_VEC_SPLIT_SHAPE = [(feature_shape,), (legal_action_shape,)]

    MIN_POLICY = 0.00001
    CLIP_PARAM = 0.2
    BETA_START = 0.008
    INIT_LEARNING_RATE_START = 0.0006
    USE_GRAD_CLIP = True
    GRAD_CLIP_RANGE = 0.5

    GAMMA = 0.995
    LAMDA = 0.95

    # The input dimension of samples on the learner from Reverb varies depending on the algorithm used.
    # learner上reverb样本的输入维度, 注意不同的算法维度不一样
    SAMPLE_DIM = (sum(HERO_DATA_SPLIT_SHAPE) * LSTM_TIME_STEPS + LSTM_UNIT_SIZE * 2) * HERO_NUM


class Config:
    """Platform-facing aliases for the atomically activated training path."""

    HERO_NUM = FrozenV1Config.CONTROLLED_ACTOR_COUNT
    TEAM_HERO_NUM = FrozenV1Config.TEAM_HERO_COUNT
    FEATURE_DIM = FrozenV1Config.FEATURE_DIM
    feature_shape = [FEATURE_DIM]
    HERO_LABEL_SIZE_LIST = list(FrozenV1Config.HERO_ACTION_HEADS)
    COMMAND_LABEL_SIZE_LIST = list(FrozenV1Config.COMMAND_HEADS)
    LSTM_TIME_STEPS = FrozenV1Config.RNN_TIME_STEPS
    LSTM_UNIT_SIZE = FrozenV1Config.RNN_HIDDEN_SIZE
    RNN_TYPE = FrozenV1Config.RNN_TYPE
    SAMPLE_DIM = FrozenV1Config.SAMPLE_DIM
    SAMPLE_DIM_LIST = [[SAMPLE_DIM]]

    MIN_POLICY = 0.00001
    CLIP_PARAM = V1TrainingConfig.CLIP_PARAM
    DUAL_CLIP_PARAM = V1TrainingConfig.DUAL_CLIP_PARAM
    BETA_START = V1TrainingConfig.ENTROPY_COEF
    INIT_LEARNING_RATE_START = V1TrainingConfig.LEARNING_RATE
    USE_GRAD_CLIP = V1TrainingConfig.USE_GRAD_CLIP
    GRAD_CLIP_RANGE = V1TrainingConfig.GRAD_CLIP_RANGE
    GAMMA = V1TrainingConfig.GAMMA
    LAMDA = V1TrainingConfig.LAMBDA
