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


class FrozenV1Config:
    """Immutable-by-convention constants for the active training contract.

    The class name is retained for platform import compatibility.  This class
    is the authoritative runtime contract. ``LegacyConfig`` below is retained
    only so old checkpoints can be identified and rejected with a useful
    signature error; it is not used by the active agent.
    """

    FEATURE_VERSION = "feature_schema_v19_lane_contact_guard"
    NETWORK_VERSION = "network_architecture_3v3_hybrid_multivalue_team_macro"
    SAMPLE_PROTOCOL_VERSION = "sample_protocol_hybrid_multivalue_team_macro_v3"
    LEGALIZER_VERSION = "collaboration_rule_legalizer_v20_lane_contact_guard"
    ORDINARY_ACTION_LEGALIZER_VERSION = "ordinary_action_legalizer_v13_single_departure"

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

    GAMMA = 0.995
    LAMBDA = 0.95
    CLIP_PARAM = 0.2
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
    MICRO_BATCH_SIZE = 32
    AMP_ENABLED = True
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
    BETA_START = V1TrainingConfig.ENTROPY_COEF
    INIT_LEARNING_RATE_START = V1TrainingConfig.LEARNING_RATE
    USE_GRAD_CLIP = V1TrainingConfig.USE_GRAD_CLIP
    GRAD_CLIP_RANGE = V1TrainingConfig.GRAD_CLIP_RANGE
    GAMMA = V1TrainingConfig.GAMMA
    LAMDA = V1TrainingConfig.LAMBDA
