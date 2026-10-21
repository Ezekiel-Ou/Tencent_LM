#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""Active Frame, collector, SampleData, and serialization protocol.

Historical ``V1`` symbols are retained because the platform imports them, but
the checkpoint/sample signatures identify the hybrid multi-value protocol.
"""

from collections import OrderedDict
from dataclasses import dataclass, field
import itertools
import random

import numpy as np

try:
    from common_python.utils.common_func import create_cls
except ModuleNotFoundError:  # Local contract tests run without the platform SDK.
    create_cls = None

from agent_ppo.conf.conf import FrozenV1Config, V1TrainingConfig
from agent_ppo.feature import schema_v1


REWARD_DETAIL_NAMES = (
    "hp_rate_sqrt_sqrt",
    "money",
    "exp",
    "tower",
    "killCnt",
    "deadCnt",
    "assistCnt",
    "total_hurt_to_hero",
    "atk_monster",
    "win_crystal",
    "atk_crystal",
)
OFFICIAL_REWARD_VECTOR_DIM = len(REWARD_DETAIL_NAMES) + 1


if create_cls is not None:
    SampleDataV1 = create_cls("SampleDataV1", sample=FrozenV1Config.SAMPLE_DIM)
else:

    @dataclass
    class SampleDataV1:
        sample: object


def _lineup_iterator_shuffle_cycle(camps):
    while True:
        random.shuffle(camps)
        yield from camps


def lineup_iterator_v1(official_model_lineups=None, joint_lineup_block=None):
    """Return shuffled official-hero blocks while preserving the legacy API."""

    if joint_lineup_block is None:
        camps = [
            list(lineups)
            for lineups in itertools.product(official_model_lineups, repeat=2)
        ]
    else:
        camps = [list(lineups) for lineups in joint_lineup_block]
    valid_heroes = set(schema_v1.HERO_CONFIG_IDS)
    if not camps:
        raise ValueError("official lineup block cannot be empty")
    for lineup in camps:
        if len(lineup) != 2 or any(hero_id not in valid_heroes for hero_id in lineup):
            raise ValueError(
                "each official lineup must contain two frozen hero IDs, "
                f"got {lineup}"
            )
    return _lineup_iterator_shuffle_cycle(camps)


def _array(value, size, name, dtype=np.float32):
    result = np.asarray(value, dtype=dtype).reshape(-1)
    if result.size != size:
        raise ValueError(f"{name} must contain {size} values, got {result.size}")
    if not np.isfinite(result).all():
        raise ValueError(f"{name} contains NaN/Inf")
    return result


def accumulate_total_reward_v1(total_reward_by_hero, hero_info):
    """Accumulate a hero that may be absent from the episode's first frame."""

    hero_id = int(hero_info["config_id"])
    total_reward_by_hero[hero_id] = (
        float(total_reward_by_hero.get(hero_id, 0.0))
        + float(hero_info["reward"][-1])
    )
    return hero_id


def split_reward_payload_v1(reward_values):
    """Split the official 11 detail fields + final reward, fail closed.

    Reward-detail values have already been weighted by the environment.  The
    five auxiliary categories are observable groupings, not an additive
    reconstruction of the final reward (which also applies zero-sum/team/time
    transforms).
    """

    values = np.asarray(reward_values, dtype=np.float32).reshape(-1)
    if values.size != OFFICIAL_REWARD_VECTOR_DIM:
        raise ValueError(
            "official reward vector must contain 11 details plus final, "
            f"got {values.size}"
        )
    if not np.isfinite(values).all():
        raise ValueError("official reward vector contains NaN/Inf")
    auxiliary = np.asarray(
        [
            values[list(indices)].sum(dtype=np.float32)
            for indices in FrozenV1Config.AUXILIARY_REWARD_COMPONENT_INDICES
        ],
        dtype=np.float32,
    )
    return float(values[-1]), auxiliary


def late_monster_reward_offsets_v1(
    controlled_config_ids,
    raw_monster_details,
    scale,
    boundary_is_new,
):
    """Return the V40-M02 residual in controlled-actor order.

    The helper is deliberately independent of the workflow so the exact
    actor scope and duplicate-boundary behavior remain directly testable.
    """

    config_ids = tuple(int(value) for value in controlled_config_ids)
    raw = np.asarray(raw_monster_details, dtype=np.float64).reshape(-1)
    if (
        len(config_ids) != FrozenV1Config.CONTROLLED_ACTOR_COUNT
        or raw.size != FrozenV1Config.CONTROLLED_ACTOR_COUNT
    ):
        raise ValueError(
            "late monster residual requires two controlled actors"
        )
    scale = float(scale)
    if not np.isfinite(raw).all() or not np.isfinite(scale):
        raise ValueError("late monster residual contains NaN/Inf")
    if not 0.0 <= scale <= 1.0:
        raise ValueError("late monster reward scale must be in [0,1]")
    if not boundary_is_new:
        return (0.0,) * FrozenV1Config.CONTROLLED_ACTOR_COUNT
    return tuple(
        -(1.0 - scale) * max(float(raw_detail), 0.0)
        if config_id == 193
        else 0.0
        for config_id, raw_detail in zip(config_ids, raw)
    )


def build_command_sparse_reward_v1(agent, observation, controlled_infos=None):
    """Build the command head's sparse team-causal boundary reward.

    The platform usually exposes reward-detail vectors only for the two heroes
    owned by the learned agent.  Never synthesize an official attack-crystal
    term from those controlled payloads: doing so falsely attributes a
    controlled hero's push to the command.  Official KDA and goal progress are
    admitted only by a trainable, legalizer-executed, platform-confirmed
    attribution window.  Tower HP is supplied once by the custom team
    potential, not by reward detail index 3.
    """

    if controlled_infos is None:
        controlled_infos = agent.feature_processes.controlled_infos(observation)
    controlled_infos = tuple(controlled_infos)
    if len(controlled_infos) != FrozenV1Config.CONTROLLED_ACTOR_COUNT:
        raise RuntimeError("command reward requires exactly two controlled heroes")
    controlled_runtime_ids = {
        int(info.get("runtime_id", 0)) for info in controlled_infos
    }
    controlled_config_ids = {
        int(info.get("config_id", 0)) for info in controlled_infos
    }
    official_config_ids = set(schema_v1.HERO_CONFIG_IDS) - controlled_config_ids
    if (
        len(controlled_runtime_ids) != FrozenV1Config.CONTROLLED_ACTOR_COUNT
        or len(controlled_config_ids) != FrozenV1Config.CONTROLLED_ACTOR_COUNT
        or len(official_config_ids) != 1
    ):
        raise RuntimeError("command reward received an invalid controlled lineup")
    official_config_id = official_config_ids.pop()

    all_infos = list(observation.get("heroes", []) or [])
    official_infos = [
        info
        for info in all_infos
        if int(info.get("runtime_id", 0)) not in controlled_runtime_ids
    ]
    if (
        len(all_infos) not in (
            FrozenV1Config.CONTROLLED_ACTOR_COUNT,
            FrozenV1Config.CONTROLLED_ACTOR_COUNT + 1,
        )
        or len(official_infos)
        != len(all_infos) - FrozenV1Config.CONTROLLED_ACTOR_COUNT
    ):
        raise RuntimeError(
            "command reward observation must contain the two controlled heroes "
            "and at most one official reward payload"
        )

    kda_by_runtime = getattr(agent, "last_kda_reward_by_runtime", {})
    kda_by_config = getattr(agent, "last_kda_reward_by_config", {})
    team_event_reward = float(
        getattr(agent, "last_tyrant_team_reward", 0.0)
    ) + float(
        getattr(agent, "last_structure_team_reward", 0.0)
    ) + float(
        getattr(agent, "last_post_kill_push_reward", 0.0)
    ) + float(
        getattr(agent, "last_crystal_danger_reward", 0.0)
    ) + float(
        getattr(agent, "last_defense_pressure_reward", 0.0)
    ) + float(
        getattr(agent, "last_temporal_focus_team_reward", 0.0)
    ) + float(
        getattr(agent, "last_group_attack_team_reward", 0.0)
    ) + float(getattr(agent, "last_crystal_outcome_reward", 0.0))
    controlled_weight = V1TrainingConfig.COMMAND_CONTROLLED_HERO_WEIGHT
    official_weight = V1TrainingConfig.COMMAND_OFFICIAL_HERO_WEIGHT
    if not np.isclose(2.0 * controlled_weight + official_weight, 1.0):
        raise RuntimeError("command reward hero weights must sum to one")
    command_reward = 0.0
    for info in controlled_infos:
        values = np.asarray(info.get("reward", []), dtype=np.float32).reshape(-1)
        split_reward_payload_v1(values)
        runtime_id = int(info.get("runtime_id", 0))
        config_id = int(info.get("config_id", 0))
        kda_reward = float(
            kda_by_runtime.get(runtime_id, kda_by_config.get(config_id, 0.0))
        )
        pushing_reward = float(values[10])
        # Only attack-crystal and KDA enter the per-hero path. Nonlinear tower,
        # post-kill conversion, Tyrant, defense pressure, exposed-crystal
        # danger, temporal focus fire, confirmed three-person kills, and
        # terminal crystal are added once as team events below. Dense actor
        # HP/economy/experience/damage and opening shaping are excluded.
        command_reward += controlled_weight * (
            kda_reward + pushing_reward
        )

    if official_infos:
        official_info = official_infos[0]
        observed_official_config_id = int(official_info.get("config_id", 0))
        if observed_official_config_id != official_config_id:
            raise RuntimeError("official reward payload conflicts with controlled lineup")
        official_values = np.asarray(
            official_info.get("reward", []), dtype=np.float32
        ).reshape(-1)
        split_reward_payload_v1(official_values)
        official_pushing_reward = (
            float(official_values[10])
            if bool(getattr(agent, "last_command_attribution_active", False))
            and int(getattr(agent, "last_command_attribution_intent", 0))
            in (3, 5)
            else 0.0
        )
    else:
        official_pushing_reward = 0.0

    official_kda_reward = float(
        getattr(agent, "last_command_official_kda_reward", 0.0)
    )
    command_reward += official_weight * (
        official_kda_reward + official_pushing_reward
    )
    command_reward += float(
        getattr(agent, "last_command_goal_progress_reward", 0.0)
    )
    command_reward += float(
        getattr(agent, "last_command_no_command_idle_penalty", 0.0)
    )
    command_reward += float(
        getattr(agent, "last_command_farm_camp_reward", 0.0)
    )
    command_reward += float(
        getattr(
            agent,
            "last_command_teammate_abandonment_penalty",
            0.0,
        )
    )
    command_reward += float(
        getattr(agent, "last_command_teammate_support_reward", 0.0)
    )
    # Structure, post-kill conversion, Tyrant, defense pressure,
    # exposed-crystal danger, temporal focus fire, confirmed three-person
    # kills, and terminal crystal are team events. Adding them once is
    # equivalent to adding them inside all three weighted hero terms because
    # the frozen hero weights sum to one.
    command_reward += team_event_reward
    if not np.isfinite(command_reward):
        raise RuntimeError("command sparse reward must be finite")
    return float(command_reward)


@dataclass
class ActorFrameV1:
    runtime_id: int
    feature: np.ndarray
    conditioned_legal_action: np.ndarray
    action: np.ndarray
    old_action_probabilities: np.ndarray
    sub_action_mask: np.ndarray
    rnn_hidden: np.ndarray
    rnn_reset: float = 0.0
    auxiliary_reward: np.ndarray = field(
        default_factory=lambda: np.zeros(
            FrozenV1Config.AUXILIARY_VALUE_COUNT, dtype=np.float32
        )
    )
    auxiliary_value: np.ndarray = field(
        default_factory=lambda: np.zeros(
            FrozenV1Config.AUXILIARY_VALUE_COUNT, dtype=np.float32
        )
    )
    next_auxiliary_value: np.ndarray = field(
        default_factory=lambda: np.zeros(
            FrozenV1Config.AUXILIARY_VALUE_COUNT, dtype=np.float32
        )
    )
    auxiliary_return: np.ndarray = field(
        default_factory=lambda: np.zeros(
            FrozenV1Config.AUXILIARY_VALUE_COUNT, dtype=np.float32
        )
    )
    reward: float = 0.0
    value: float = 0.0
    next_value: float = 0.0
    return_value: float = 0.0
    advantage: float = 0.0
    is_train: float = 1.0

    def validate(self):
        self.runtime_id = int(self.runtime_id)
        if self.runtime_id <= 0:
            raise ValueError("controlled actor runtime_id must be positive")
        self.feature = _array(
            self.feature, FrozenV1Config.FEATURE_DIM, "actor feature"
        )
        self.conditioned_legal_action = _array(
            self.conditioned_legal_action, 161, "conditioned legal action"
        )
        self.action = _array(self.action, 5, "actor action")
        self.old_action_probabilities = _array(
            self.old_action_probabilities, 161, "old action probabilities"
        )
        self.sub_action_mask = _array(self.sub_action_mask, 5, "sub action mask")
        self.rnn_hidden = _array(self.rnn_hidden, 128, "actor GRU hidden")
        self.rnn_reset = float(self.rnn_reset)
        if self.rnn_reset not in (0.0, 1.0):
            raise ValueError("actor rnn_reset must be exactly 0 or 1")
        self.auxiliary_reward = _array(
            self.auxiliary_reward,
            FrozenV1Config.AUXILIARY_VALUE_COUNT,
            "actor auxiliary reward",
        )
        self.auxiliary_value = _array(
            self.auxiliary_value,
            FrozenV1Config.AUXILIARY_VALUE_COUNT,
            "actor auxiliary value",
        )
        self.next_auxiliary_value = _array(
            self.next_auxiliary_value,
            FrozenV1Config.AUXILIARY_VALUE_COUNT,
            "actor next auxiliary value",
        )
        self.auxiliary_return = _array(
            self.auxiliary_return,
            FrozenV1Config.AUXILIARY_VALUE_COUNT,
            "actor auxiliary return",
        )
        for name in (
            "reward",
            "value",
            "next_value",
            "return_value",
            "advantage",
            "is_train",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value):
                raise ValueError(f"actor {name} is NaN/Inf")
            setattr(self, name, value)
        return self


@dataclass
class CommandFrameV1:
    conditioned_legal_mask: np.ndarray
    action: np.ndarray
    old_log_probabilities: np.ndarray
    active_head_mask: np.ndarray
    reward: float = 0.0
    value: float = 0.0
    next_value: float = 0.0
    return_value: float = 0.0
    advantage: float = 0.0
    is_train: float = 0.0

    def validate(self):
        self.conditioned_legal_mask = _array(
            self.conditioned_legal_mask,
            FrozenV1Config.COMMAND_LEGAL_MASK_DIM,
            "command legal mask",
        )
        self.action = _array(self.action, 3, "command action")
        self.old_log_probabilities = _array(
            self.old_log_probabilities, 3, "command old log probabilities"
        )
        self.active_head_mask = _array(
            self.active_head_mask, 3, "command active head mask"
        )
        for name in (
            "reward",
            "value",
            "next_value",
            "return_value",
            "advantage",
            "is_train",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value):
                raise ValueError(f"command {name} is NaN/Inf")
            setattr(self, name, value)
        return self


@dataclass
class TeamFrameV1:
    frame_no: int
    actors: tuple
    command: CommandFrameV1

    def validate(self):
        self.frame_no = int(self.frame_no)
        if len(self.actors) != FrozenV1Config.CONTROLLED_ACTOR_COUNT:
            raise ValueError("TeamFrameV1 requires exactly two controlled actors")
        self.actors = tuple(actor.validate() for actor in self.actors)
        if len({actor.runtime_id for actor in self.actors}) != 2:
            raise ValueError("controlled actor runtime IDs must be distinct")
        self.command = self.command.validate()
        return self


@dataclass
class ObsDataV1:
    feature: np.ndarray
    raw_legal_action: np.ndarray
    rnn_hidden: np.ndarray
    rnn_reset: np.ndarray
    runtime_ids: tuple
    config_ids: tuple
    hero_infos: tuple
    command_context: object
    command_masks: object
    flash_offset_masks: np.ndarray = None


@dataclass
class ActorDecisionV1:
    action: np.ndarray
    probability_vector: np.ndarray
    conditioned_legal_action: np.ndarray
    value: float
    auxiliary_values: np.ndarray
    initial_hidden: np.ndarray
    next_hidden: np.ndarray


@dataclass
class CommandDecisionV1:
    action: np.ndarray
    old_log_probabilities: np.ndarray
    conditioned_legal_mask: np.ndarray
    active_head_mask: np.ndarray
    is_train: float
    value: float
    protocol_action: tuple


@dataclass
class ActDataV1:
    """One inference mode's decisions; the unused mode fields are ``None``."""

    stochastic_actors: tuple
    deterministic_actors: tuple
    stochastic_command: CommandDecisionV1
    deterministic_command: CommandDecisionV1


def build_frame_v1(
    agent, observation, rewards, auxiliary_rewards, command_reward
):
    """Build the single two-actor + one-command frame used by FrameCollectorV1."""

    if len(rewards) != 2:
        raise ValueError("build_frame_v1 requires exactly two controlled rewards")
    auxiliary_rewards = np.asarray(auxiliary_rewards, dtype=np.float32)
    expected_auxiliary_shape = (
        FrozenV1Config.CONTROLLED_ACTOR_COUNT,
        FrozenV1Config.AUXILIARY_VALUE_COUNT,
    )
    if auxiliary_rewards.shape != expected_auxiliary_shape:
        raise ValueError(
            "build_frame_v1 auxiliary rewards must have shape "
            f"{expected_auxiliary_shape}, got {auxiliary_rewards.shape}"
        )
    if not np.isfinite(auxiliary_rewards).all():
        raise ValueError("build_frame_v1 auxiliary rewards contain NaN/Inf")
    if not np.isfinite(float(command_reward)):
        raise ValueError("build_frame_v1 command reward contains NaN/Inf")
    obs_data = agent.obs_data
    act_data = agent.act_data
    actors = []
    for index in range(2):
        decision = act_data.stochastic_actors[index]
        hero_info = obs_data.hero_infos[index]
        button = int(decision.action[0])
        sub_actions = hero_info.get("sub_action_mask", {}) or {}
        sub_action = sub_actions.get(
            str(button), sub_actions.get(button, [0] * 5)
        )
        button_legal = np.asarray(
            hero_info.get("legal_action", []), dtype=np.float32
        ).reshape(-1)[:13]
        is_train = float(button_legal.size == 13 and np.any(button_legal > 0.5))
        actors.append(
            ActorFrameV1(
                runtime_id=obs_data.runtime_ids[index],
                feature=obs_data.feature[index],
                conditioned_legal_action=decision.conditioned_legal_action,
                action=decision.action,
                old_action_probabilities=decision.probability_vector,
                sub_action_mask=sub_action,
                rnn_hidden=decision.initial_hidden,
                rnn_reset=float(obs_data.rnn_reset[index]),
                auxiliary_reward=auxiliary_rewards[index],
                auxiliary_value=decision.auxiliary_values,
                reward=float(rewards[index]),
                value=float(decision.value),
                is_train=is_train,
            )
        )
    command_decision = act_data.stochastic_command
    command = CommandFrameV1(
        conditioned_legal_mask=command_decision.conditioned_legal_mask,
        action=command_decision.action,
        old_log_probabilities=command_decision.old_log_probabilities,
        active_head_mask=command_decision.active_head_mask,
        reward=float(command_reward),
        value=float(command_decision.value),
        is_train=float(command_decision.is_train),
    )
    frame_no = int(observation.get("frame_state", {}).get("frame_no", 0))
    return TeamFrameV1(frame_no=frame_no, actors=tuple(actors), command=command)


def _write_actor_step_v1(frame, result):
    layout = schema_v1.ACTOR_STEP_LAYOUT
    result[layout["feature"].slice] = frame.feature
    result[layout["conditioned_legal_action"].slice] = frame.conditioned_legal_action
    result[layout["return"].start] = frame.return_value
    result[layout["advantage"].start] = frame.advantage
    result[layout["actions"].slice] = frame.action
    result[layout["old_action_probabilities"].slice] = frame.old_action_probabilities
    result[layout["is_train"].start] = frame.is_train
    result[layout["sub_action_mask"].slice] = frame.sub_action_mask
    result[layout["auxiliary_returns"].slice] = frame.auxiliary_return
    result[layout["rnn_reset"].start] = frame.rnn_reset


def _write_actor_sequence_v1(frames, result):
    """Write one validated recurrent chunk with field-wise NumPy batches."""

    count = len(frames)
    target = result[:count]
    layout = schema_v1.ACTOR_STEP_LAYOUT
    array_fields = (
        ("feature", "feature"),
        ("conditioned_legal_action", "conditioned_legal_action"),
        ("actions", "action"),
        ("old_action_probabilities", "old_action_probabilities"),
        ("sub_action_mask", "sub_action_mask"),
        ("auxiliary_returns", "auxiliary_return"),
    )
    for field_name, attribute_name in array_fields:
        target[:, layout[field_name].slice] = np.stack(
            [getattr(frame, attribute_name) for frame in frames], axis=0
        )
    scalar_fields = (
        ("return", "return_value"),
        ("advantage", "advantage"),
        ("is_train", "is_train"),
        ("rnn_reset", "rnn_reset"),
    )
    for field_name, attribute_name in scalar_fields:
        target[:, layout[field_name].start] = np.fromiter(
            (getattr(frame, attribute_name) for frame in frames),
            dtype=np.float32,
            count=count,
        )


def pack_actor_step_v1(frame):
    frame.validate()
    result = np.zeros(FrozenV1Config.ACTOR_STEP_DIM, dtype=np.float32)
    _write_actor_step_v1(frame, result)
    return result


def _write_command_step_v1(frame, result):
    layout = schema_v1.COMMAND_STEP_LAYOUT
    result[layout["conditioned_legal_masks"].slice] = frame.conditioned_legal_mask
    result[layout["return"].start] = frame.return_value
    result[layout["advantage"].start] = frame.advantage
    result[layout["actions"].slice] = frame.action
    result[layout["old_log_probs"].slice] = frame.old_log_probabilities
    result[layout["is_train"].start] = frame.is_train
    result[layout["active_head_mask"].slice] = frame.active_head_mask


def _write_command_sequence_v1(frames, result):
    """Write one validated command chunk without a per-step Python writer."""

    count = len(frames)
    target = result[:count]
    layout = schema_v1.COMMAND_STEP_LAYOUT
    array_fields = (
        ("conditioned_legal_masks", "conditioned_legal_mask"),
        ("actions", "action"),
        ("old_log_probs", "old_log_probabilities"),
        ("active_head_mask", "active_head_mask"),
    )
    for field_name, attribute_name in array_fields:
        target[:, layout[field_name].slice] = np.stack(
            [getattr(frame, attribute_name) for frame in frames], axis=0
        )
    scalar_fields = (
        ("return", "return_value"),
        ("advantage", "advantage"),
        ("is_train", "is_train"),
    )
    for field_name, attribute_name in scalar_fields:
        target[:, layout[field_name].start] = np.fromiter(
            (getattr(frame, attribute_name) for frame in frames),
            dtype=np.float32,
            count=count,
        )


def pack_command_step_v1(frame):
    frame.validate()
    result = np.zeros(FrozenV1Config.COMMAND_STEP_DIM, dtype=np.float32)
    _write_command_step_v1(frame, result)
    return result


def _validate_packed_sample_v1(sample):
    if not np.isfinite(sample).all():
        raise RuntimeError("invalid packed v1 SampleData")


def _pack_prevalidated_sequence_v1(frames, full_validation=True):
    if not frames or len(frames) > FrozenV1Config.RNN_TIME_STEPS:
        raise ValueError("a v1 sample sequence must contain between 1 and 16 frames")
    runtime_ids = tuple(actor.runtime_id for actor in frames[0].actors)
    if any(tuple(actor.runtime_id for actor in frame.actors) != runtime_ids for frame in frames):
        raise ValueError("runtime change must start a new v1 sequence")

    sample = np.zeros(FrozenV1Config.SAMPLE_DIM, dtype=np.float32)
    sample_layout = schema_v1.SAMPLE_LAYOUT
    for actor_index in range(2):
        steps = sample[
            sample_layout[f"actor_{actor_index}_steps"].slice
        ].reshape(
            FrozenV1Config.RNN_TIME_STEPS,
            FrozenV1Config.ACTOR_STEP_DIM,
        )
        _write_actor_sequence_v1(
            [frame.actors[actor_index] for frame in frames], steps
        )
        sample[sample_layout[f"actor_{actor_index}_initial_hidden"].slice] = frames[
            0
        ].actors[actor_index].rnn_hidden

    command_steps = sample[sample_layout["command_steps"].slice].reshape(
        FrozenV1Config.RNN_TIME_STEPS,
        FrozenV1Config.COMMAND_STEP_DIM,
    )
    _write_command_sequence_v1(
        [frame.command for frame in frames], command_steps
    )
    if sample.shape != (FrozenV1Config.SAMPLE_DIM,):
        raise RuntimeError("invalid packed v1 SampleData shape")
    if full_validation:
        _validate_packed_sample_v1(sample)
    return SampleDataV1(sample=sample)


def pack_sequence_v1(frames, full_validation=True):
    """Validate public inputs fully, then serialize one recurrent sequence."""

    frames = [frame.validate() for frame in frames]
    return _pack_prevalidated_sequence_v1(
        frames,
        full_validation=full_validation,
    )


def unpack_sample_v1(sample):
    """Unpack NumPy or torch sample batches without changing frozen storage."""

    if sample.shape[-1] != FrozenV1Config.SAMPLE_DIM:
        raise ValueError(
            "active sample must end in "
            f"{FrozenV1Config.SAMPLE_DIM}, got {sample.shape[-1]}"
        )
    sample_layout = schema_v1.SAMPLE_LAYOUT
    actors = []
    for actor_index in range(2):
        steps = sample[..., sample_layout[f"actor_{actor_index}_steps"].slice].reshape(
            *sample.shape[:-1],
            FrozenV1Config.RNN_TIME_STEPS,
            FrozenV1Config.ACTOR_STEP_DIM,
        )
        hidden = sample[
            ..., sample_layout[f"actor_{actor_index}_initial_hidden"].slice
        ]
        fields = {
            name: steps[..., spec.slice]
            for name, spec in schema_v1.ACTOR_STEP_LAYOUT.items()
        }
        fields["initial_hidden"] = hidden
        actors.append(fields)
    command_steps = sample[..., sample_layout["command_steps"].slice].reshape(
        *sample.shape[:-1],
        FrozenV1Config.RNN_TIME_STEPS,
        FrozenV1Config.COMMAND_STEP_DIM,
    )
    command = {
        name: command_steps[..., spec.slice]
        for name, spec in schema_v1.COMMAND_STEP_LAYOUT.items()
    }
    return {"actors": actors, "command": command}


class FrameCollectorV1:
    """Collect complete episodes, compute actor/command GAE, then pack 16 steps."""

    def __init__(self, num_agents, gamma=None, lamda=None):
        self.gamma = float(
            V1TrainingConfig.GAMMA if gamma is None else gamma
        )
        self.lamda = float(
            V1TrainingConfig.LAMBDA if lamda is None else lamda
        )
        self.packed_sample_count = 0
        self.reset(num_agents)

    def reset(self, num_agents=None):
        if num_agents is not None:
            self.num_agents = int(num_agents)
        self.frames = [OrderedDict() for _ in range(self.num_agents)]
        self.samples = [[] for _ in range(self.num_agents)]

    def save_frame(self, frame, agent_id):
        frame = frame.validate()
        records = self.frames[agent_id]
        if records and frame.frame_no <= next(reversed(records)):
            raise ValueError("v1 frames must be saved in strictly increasing frame order")
        if records:
            previous = records[next(reversed(records))]
            for actor_index in range(2):
                same_runtime = (
                    previous.actors[actor_index].runtime_id
                    == frame.actors[actor_index].runtime_id
                )
                previous.actors[actor_index].next_value = (
                    frame.actors[actor_index].value
                    if same_runtime and frame.actors[actor_index].rnn_reset < 0.5
                    else 0.0
                )
                previous.actors[actor_index].reward = frame.actors[actor_index].reward
                previous.actors[actor_index].next_auxiliary_value = (
                    frame.actors[actor_index].auxiliary_value.copy()
                    if same_runtime and frame.actors[actor_index].rnn_reset < 0.5
                    else np.zeros(
                        FrozenV1Config.AUXILIARY_VALUE_COUNT,
                        dtype=np.float32,
                    )
                )
                previous.actors[actor_index].auxiliary_reward = (
                    frame.actors[actor_index].auxiliary_reward.copy()
                )
            previous.command.next_value = frame.command.value
            previous.command.reward = float(frame.command.reward)
        for actor in frame.actors:
            actor.reward = 0.0
            actor.auxiliary_reward = np.zeros(
                FrozenV1Config.AUXILIARY_VALUE_COUNT, dtype=np.float32
            )
        frame.command.reward = 0.0
        records[frame.frame_no] = frame

    def save_last_frame(
        self,
        rewards,
        auxiliary_rewards,
        agent_id,
        command_reward,
        actor_bootstrap_values=(0.0, 0.0),
        actor_auxiliary_bootstrap_values=None,
        command_bootstrap_value=0.0,
    ):
        records = self.frames[agent_id]
        if not records:
            return
        if len(rewards) != 2 or len(actor_bootstrap_values) != 2:
            raise ValueError("terminal rewards/bootstrap require exactly two actors")
        if not np.isfinite(float(command_reward)):
            raise ValueError("terminal command reward must be finite")
        auxiliary_rewards = np.asarray(auxiliary_rewards, dtype=np.float32)
        expected_shape = (
            FrozenV1Config.CONTROLLED_ACTOR_COUNT,
            FrozenV1Config.AUXILIARY_VALUE_COUNT,
        )
        if auxiliary_rewards.shape != expected_shape or not np.isfinite(
            auxiliary_rewards
        ).all():
            raise ValueError(
                "terminal auxiliary rewards must be finite with shape "
                f"{expected_shape}, got {auxiliary_rewards.shape}"
            )
        if actor_auxiliary_bootstrap_values is None:
            raise ValueError("terminal auxiliary bootstrap values are required")
        auxiliary_bootstrap = np.asarray(
            actor_auxiliary_bootstrap_values, dtype=np.float32
        )
        if auxiliary_bootstrap.shape != expected_shape or not np.isfinite(
            auxiliary_bootstrap
        ).all():
            raise ValueError(
                "terminal auxiliary bootstrap values must be finite with shape "
                f"{expected_shape}, got {auxiliary_bootstrap.shape}"
            )
        last = records[next(reversed(records))]
        for index, actor in enumerate(last.actors):
            actor.reward = float(rewards[index])
            actor.next_value = float(actor_bootstrap_values[index])
            actor.auxiliary_reward = auxiliary_rewards[index].copy()
            actor.next_auxiliary_value = auxiliary_bootstrap[index].copy()
        last.command.reward = float(command_reward)
        last.command.next_value = float(command_bootstrap_value)

    def _calculate_gae(self, agent_id):
        records = list(self.frames[agent_id].values())
        actor_gae = [0.0, 0.0]
        auxiliary_gae = [
            np.zeros(FrozenV1Config.AUXILIARY_VALUE_COUNT, dtype=np.float32)
            for _ in range(FrozenV1Config.CONTROLLED_ACTOR_COUNT)
        ]
        command_gae = 0.0
        next_runtime_ids = None
        next_resets = None
        for frame in reversed(records):
            for index, actor in enumerate(frame.actors):
                if (
                    next_runtime_ids is not None
                    and (
                        actor.runtime_id != next_runtime_ids[index]
                        or next_resets[index]
                    )
                ):
                    actor_gae[index] = 0.0
                    auxiliary_gae[index].fill(0.0)
                delta = actor.reward + self.gamma * actor.next_value - actor.value
                actor_gae[index] = delta + self.gamma * self.lamda * actor_gae[index]
                actor.advantage = actor_gae[index]
                actor.return_value = actor.value + actor_gae[index]
                auxiliary_delta = (
                    actor.auxiliary_reward
                    + self.gamma * actor.next_auxiliary_value
                    - actor.auxiliary_value
                )
                auxiliary_gae[index] = (
                    auxiliary_delta
                    + self.gamma * self.lamda * auxiliary_gae[index]
                )
                actor.auxiliary_return = (
                    actor.auxiliary_value + auxiliary_gae[index]
                ).astype(np.float32, copy=False)
            command = frame.command
            delta = command.reward + self.gamma * command.next_value - command.value
            command_gae = delta + self.gamma * self.lamda * command_gae
            command.advantage = command_gae
            command.return_value = command.value + command_gae
            if command.is_train > 0.5:
                # A real command choice starts a new temporal option.  Its
                # forced hold frames may carry delayed results back to this
                # decision, but the next independent command decision's TD
                # residual must not leak into an older option.  next_value in
                # ``delta`` still bootstraps across the boundary.
                command_gae = 0.0
            next_runtime_ids = tuple(actor.runtime_id for actor in frame.actors)
            next_resets = tuple(actor.rnn_reset > 0.5 for actor in frame.actors)

    @staticmethod
    def _sequence_chunks(records):
        chunks = []
        current = []
        current_runtime_ids = None
        for frame in records:
            runtime_ids = tuple(actor.runtime_id for actor in frame.actors)
            if current and (
                len(current) == FrozenV1Config.RNN_TIME_STEPS
                or runtime_ids != current_runtime_ids
            ):
                chunks.append(current)
                current = []
            if not current:
                current_runtime_ids = runtime_ids
            current.append(frame)
        if current:
            chunks.append(current)
        return chunks

    def sample_process(self):
        for agent_id in range(self.num_agents):
            self._calculate_gae(agent_id)
            records = list(self.frames[agent_id].values())
            samples = []
            for chunk in self._sequence_chunks(records):
                sample_index = self.packed_sample_count
                full_validation = (
                    sample_index
                    < V1TrainingConfig.PACKED_SAMPLE_FULL_VALIDATION_STARTUP_SAMPLES
                    or sample_index
                    % V1TrainingConfig.PACKED_SAMPLE_FULL_VALIDATION_INTERVAL
                    == 0
                )
                samples.append(
                    pack_sequence_v1(
                        chunk,
                        full_validation=full_validation,
                    )
                )
                self.packed_sample_count += 1
            self.samples[agent_id] = samples
        return self.samples

    def __len__(self):
        return sum(len(records) for records in self.frames)


__all__ = (
    "ActorFrameV1",
    "ActorDecisionV1",
    "ActDataV1",
    "CommandFrameV1",
    "CommandDecisionV1",
    "FrameCollectorV1",
    "ObsDataV1",
    "OFFICIAL_REWARD_VECTOR_DIM",
    "REWARD_DETAIL_NAMES",
    "SampleDataV1",
    "TeamFrameV1",
    "build_frame_v1",
    "build_command_sparse_reward_v1",
    "lineup_iterator_v1",
    "late_monster_reward_offsets_v1",
    "pack_actor_step_v1",
    "pack_command_step_v1",
    "pack_sequence_v1",
    "split_reward_payload_v1",
    "unpack_sample_v1",
)
