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

from agent_ppo.conf.conf import FrozenV1Config
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


def lineup_iterator_v1(official_model_lineups=None):
    camps = [list(lineups) for lineups in itertools.product(official_model_lineups, repeat=2)]
    return _lineup_iterator_shuffle_cycle(camps)


def _array(value, size, name, dtype=np.float32):
    result = np.asarray(value, dtype=dtype).reshape(-1)
    if result.size != size:
        raise ValueError(f"{name} must contain {size} values, got {result.size}")
    if not np.isfinite(result).all():
        raise ValueError(f"{name} contains NaN/Inf")
    return result


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
    stochastic_actors: tuple
    deterministic_actors: tuple
    stochastic_command: CommandDecisionV1
    deterministic_command: CommandDecisionV1


def build_frame_v1(agent, observation, rewards, auxiliary_rewards):
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
        reward=0.5 * (float(rewards[0]) + float(rewards[1])),
        value=float(command_decision.value),
        is_train=float(command_decision.is_train),
    )
    frame_no = int(observation.get("frame_state", {}).get("frame_no", 0))
    return TeamFrameV1(frame_no=frame_no, actors=tuple(actors), command=command)


def pack_actor_step_v1(frame):
    frame.validate()
    result = np.zeros(FrozenV1Config.ACTOR_STEP_DIM, dtype=np.float32)
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
    return result


def pack_command_step_v1(frame):
    frame.validate()
    result = np.zeros(FrozenV1Config.COMMAND_STEP_DIM, dtype=np.float32)
    layout = schema_v1.COMMAND_STEP_LAYOUT
    result[layout["conditioned_legal_masks"].slice] = frame.conditioned_legal_mask
    result[layout["return"].start] = frame.return_value
    result[layout["advantage"].start] = frame.advantage
    result[layout["actions"].slice] = frame.action
    result[layout["old_log_probs"].slice] = frame.old_log_probabilities
    result[layout["is_train"].start] = frame.is_train
    result[layout["active_head_mask"].slice] = frame.active_head_mask
    return result


def pack_sequence_v1(frames):
    if not frames or len(frames) > FrozenV1Config.RNN_TIME_STEPS:
        raise ValueError("a v1 sample sequence must contain between 1 and 16 frames")
    frames = [frame.validate() for frame in frames]
    runtime_ids = tuple(actor.runtime_id for actor in frames[0].actors)
    if any(tuple(actor.runtime_id for actor in frame.actors) != runtime_ids for frame in frames):
        raise ValueError("runtime change must start a new v1 sequence")

    sample = np.zeros(FrozenV1Config.SAMPLE_DIM, dtype=np.float32)
    sample_layout = schema_v1.SAMPLE_LAYOUT
    for actor_index in range(2):
        steps = np.zeros(
            (FrozenV1Config.RNN_TIME_STEPS, FrozenV1Config.ACTOR_STEP_DIM),
            dtype=np.float32,
        )
        for step, frame in enumerate(frames):
            steps[step] = pack_actor_step_v1(frame.actors[actor_index])
        sample[sample_layout[f"actor_{actor_index}_steps"].slice] = steps.reshape(-1)
        sample[sample_layout[f"actor_{actor_index}_initial_hidden"].slice] = frames[
            0
        ].actors[actor_index].rnn_hidden

    command_steps = np.zeros(
        (FrozenV1Config.RNN_TIME_STEPS, FrozenV1Config.COMMAND_STEP_DIM),
        dtype=np.float32,
    )
    for step, frame in enumerate(frames):
        command_steps[step] = pack_command_step_v1(frame.command)
    sample[sample_layout["command_steps"].slice] = command_steps.reshape(-1)
    if sample.shape != (FrozenV1Config.SAMPLE_DIM,) or not np.isfinite(sample).all():
        raise RuntimeError("invalid packed v1 SampleData")
    return SampleDataV1(sample=sample)


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

    def __init__(self, num_agents, gamma=0.995, lamda=0.95):
        self.gamma = float(gamma)
        self.lamda = float(lamda)
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
            previous.command.reward = 0.5 * sum(
                actor.reward for actor in frame.actors
            )
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
        actor_bootstrap_values=(0.0, 0.0),
        actor_auxiliary_bootstrap_values=None,
        command_bootstrap_value=0.0,
    ):
        records = self.frames[agent_id]
        if not records:
            return
        if len(rewards) != 2 or len(actor_bootstrap_values) != 2:
            raise ValueError("terminal rewards/bootstrap require exactly two actors")
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
        last.command.reward = 0.5 * (float(rewards[0]) + float(rewards[1]))
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
            self.samples[agent_id] = [
                pack_sequence_v1(chunk) for chunk in self._sequence_chunks(records)
            ]
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
    "lineup_iterator_v1",
    "pack_actor_step_v1",
    "pack_command_step_v1",
    "pack_sequence_v1",
    "split_reward_payload_v1",
    "unpack_sample_v1",
)
