#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""Active SampleData packing, GAE, padding, and sequence-boundary tests."""

from collections import Counter
import unittest
from unittest.mock import patch

import numpy as np
import torch

from agent_ppo.conf.conf import FrozenV1Config, V1TrainingConfig
from agent_ppo.feature import schema_v1
from agent_ppo.feature.definition_v1 import (
    ActorFrameV1,
    CommandFrameV1,
    FrameCollectorV1,
    TeamFrameV1,
    build_command_sparse_reward_v1,
    lineup_iterator_v1,
    pack_actor_step_v1,
    pack_command_step_v1,
    pack_sequence_v1,
    split_reward_payload_v1,
    unpack_sample_v1,
)


def _actor(
    runtime_id,
    marker,
    reward=0.0,
    value=0.0,
    auxiliary_reward=None,
    auxiliary_value=None,
    rnn_reset=0.0,
):
    feature = np.zeros(FrozenV1Config.FEATURE_DIM, dtype=np.float32)
    feature[0] = marker
    legal = np.ones(161, dtype=np.float32)
    probabilities = np.concatenate(
        [np.full(width, 1 / width, dtype=np.float32) for width in (13, 25, 42, 42, 39)]
    )
    return ActorFrameV1(
        runtime_id=runtime_id,
        feature=feature,
        conditioned_legal_action=legal,
        action=np.array([1, 2, 3, 4, 5], dtype=np.float32),
        old_action_probabilities=probabilities,
        sub_action_mask=np.ones(5, dtype=np.float32),
        rnn_hidden=np.full(128, marker, dtype=np.float32),
        rnn_reset=rnn_reset,
        auxiliary_reward=np.zeros(5, dtype=np.float32)
        if auxiliary_reward is None
        else auxiliary_reward,
        auxiliary_value=np.zeros(5, dtype=np.float32)
        if auxiliary_value is None
        else auxiliary_value,
        reward=reward,
        value=value,
        is_train=1,
    )


def _command(reward=0.0, value=0.0, is_train=1.0):
    legal = np.zeros(80, dtype=np.float32)
    legal[0] = 1
    return CommandFrameV1(
        conditioned_legal_mask=legal,
        action=np.array([0, 0, 0], dtype=np.float32),
        old_log_probabilities=np.zeros(3, dtype=np.float32),
        active_head_mask=np.array([1, 0, 0], dtype=np.float32),
        reward=reward,
        value=value,
        is_train=is_train,
    )


def _team(
    frame_no,
    runtime_ids=(11, 12),
    rewards=(0.0, 0.0),
    marker=1.0,
    auxiliary_rewards=None,
    auxiliary_values=None,
    rnn_resets=(0.0, 0.0),
    command_reward=None,
):
    if auxiliary_rewards is None:
        auxiliary_rewards = (
            np.zeros(5, dtype=np.float32),
            np.zeros(5, dtype=np.float32),
        )
    if auxiliary_values is None:
        auxiliary_values = (
            np.zeros(5, dtype=np.float32),
            np.zeros(5, dtype=np.float32),
        )
    return TeamFrameV1(
        frame_no=frame_no,
        actors=(
            _actor(
                runtime_ids[0],
                marker,
                reward=rewards[0],
                auxiliary_reward=auxiliary_rewards[0],
                auxiliary_value=auxiliary_values[0],
                rnn_reset=rnn_resets[0],
            ),
            _actor(
                runtime_ids[1],
                marker + 1,
                reward=rewards[1],
                auxiliary_reward=auxiliary_rewards[1],
                auxiliary_value=auxiliary_values[1],
                rnn_reset=rnn_resets[1],
            ),
        ),
        command=_command(
            reward=sum(rewards) / 2
            if command_reward is None
            else command_reward
        ),
    )


class SampleProtocolV1Test(unittest.TestCase):
    def test_official_kai_targeted_lineup_block_is_exact_and_symmetric(self):
        iterator = lineup_iterator_v1(
            joint_lineup_block=V1TrainingConfig.OFFICIAL_LINEUP_BLOCK
        )
        first_block = [tuple(next(iterator)) for _ in range(13)]
        second_block = [tuple(next(iterator)) for _ in range(13)]
        expected = Counter(
            {
                (132, 132): 1,
                (132, 141): 1,
                (132, 193): 2,
                (141, 132): 1,
                (141, 141): 1,
                (141, 193): 2,
                (193, 132): 2,
                (193, 141): 2,
                (193, 193): 1,
            }
        )
        self.assertEqual(Counter(first_block), expected)
        self.assertEqual(Counter(second_block), expected)
        self.assertEqual(
            Counter(pair[0] for pair in first_block),
            Counter({132: 4, 141: 4, 193: 5}),
        )
        self.assertEqual(
            Counter(pair[1] for pair in first_block),
            Counter({132: 4, 141: 4, 193: 5}),
        )
        for blue_official, red_official in expected:
            self.assertEqual(
                expected[(blue_official, red_official)],
                expected[(red_official, blue_official)],
            )

    def test_legacy_nine_lineup_iterator_remains_compatible(self):
        iterator = lineup_iterator_v1([132, 141, 193])
        block = Counter(tuple(next(iterator)) for _ in range(9))
        self.assertEqual(
            block,
            Counter(
                (blue, red)
                for blue in (132, 141, 193)
                for red in (132, 141, 193)
            ),
        )

    def test_vectorized_sequence_pack_is_byte_identical_to_step_writers(self):
        for count in (1, 7, FrozenV1Config.RNN_TIME_STEPS):
            frames = [_team(100 + 6 * index, marker=index + 1) for index in range(count)]
            for frame_index, frame in enumerate(frames):
                for actor_index, actor in enumerate(frame.actors):
                    marker = frame_index * 10 + actor_index
                    actor.return_value = marker + 0.25
                    actor.advantage = marker - 0.25
                    actor.auxiliary_return = np.arange(5, dtype=np.float32) + marker
                    actor.rnn_reset = float((frame_index + actor_index) % 2)
                    actor.is_train = float((frame_index + actor_index + 1) % 2)
                frame.command.return_value = frame_index + 0.5
                frame.command.advantage = frame_index - 0.5
                frame.command.is_train = float(frame_index % 2)

            actual = pack_sequence_v1(frames).sample
            expected = np.zeros(FrozenV1Config.SAMPLE_DIM, dtype=np.float32)
            for actor_index in range(FrozenV1Config.CONTROLLED_ACTOR_COUNT):
                step_spec = schema_v1.SAMPLE_LAYOUT[f"actor_{actor_index}_steps"]
                step_view = expected[step_spec.slice].reshape(
                    FrozenV1Config.RNN_TIME_STEPS,
                    FrozenV1Config.ACTOR_STEP_DIM,
                )
                for step, frame in enumerate(frames):
                    step_view[step] = pack_actor_step_v1(frame.actors[actor_index])
                hidden_spec = schema_v1.SAMPLE_LAYOUT[
                    f"actor_{actor_index}_initial_hidden"
                ]
                expected[hidden_spec.slice] = frames[0].actors[actor_index].rnn_hidden
            command_spec = schema_v1.SAMPLE_LAYOUT["command_steps"]
            command_view = expected[command_spec.slice].reshape(
                FrozenV1Config.RNN_TIME_STEPS,
                FrozenV1Config.COMMAND_STEP_DIM,
            )
            for step, frame in enumerate(frames):
                command_view[step] = pack_command_step_v1(frame.command)

            np.testing.assert_array_equal(actual, expected)

    def test_pack_exact_dimension_offsets_and_padding(self):
        frames = [_team(100, marker=2), _team(106, marker=3)]
        frames[0].actors[0].return_value = 7
        frames[0].actors[0].advantage = 5
        frames[0].actors[0].auxiliary_return = np.arange(1, 6, dtype=np.float32)
        sample_data = pack_sequence_v1(frames)
        sample = sample_data.sample
        self.assertEqual(sample.shape, (FrozenV1Config.SAMPLE_DIM,))
        self.assertEqual(sample.dtype, np.float32)
        self.assertEqual(sample.nbytes, FrozenV1Config.SAMPLE_DIM * 4)
        unpacked = unpack_sample_v1(sample)
        actor0 = unpacked["actors"][0]
        self.assertEqual(
            actor0["feature"].shape,
            (16, FrozenV1Config.FEATURE_DIM),
        )
        self.assertEqual(actor0["feature"][0, 0], 2)
        self.assertEqual(actor0["feature"][1, 0], 3)
        self.assertEqual(actor0["return"][0, 0], 7)
        self.assertEqual(actor0["advantage"][0, 0], 5)
        self.assertEqual(actor0["auxiliary_returns"].shape, (16, 5))
        np.testing.assert_array_equal(
            actor0["auxiliary_returns"][0], np.arange(1, 6, dtype=np.float32)
        )
        self.assertEqual(actor0["auxiliary_returns"][2:].sum(), 0)
        self.assertEqual(actor0["rnn_reset"].sum(), 0)
        self.assertEqual(actor0["is_train"][:2].sum(), 2)
        self.assertEqual(actor0["is_train"][2:].sum(), 0)
        np.testing.assert_array_equal(actor0["initial_hidden"], np.full(128, 2))
        self.assertEqual(unpacked["command"]["is_train"][:2].sum(), 2)
        self.assertEqual(unpacked["command"]["is_train"][2:].sum(), 0)

    def test_collector_full_sample_validation_uses_configured_cadence(self):
        collector = FrameCollectorV1(1)
        with patch.object(
            V1TrainingConfig,
            "PACKED_SAMPLE_FULL_VALIDATION_STARTUP_SAMPLES",
            1,
        ), patch.object(
            V1TrainingConfig,
            "PACKED_SAMPLE_FULL_VALIDATION_INTERVAL",
            3,
        ), patch(
            "agent_ppo.feature.definition_v1._validate_packed_sample_v1"
        ) as validate:
            for frame_index in range(4):
                collector.reset(1)
                collector.save_frame(_team(100 + frame_index * 6), 0)
                collector.save_last_frame(
                    (0, 0),
                    np.zeros((2, 5), dtype=np.float32),
                    0,
                    command_reward=0.0,
                    actor_auxiliary_bootstrap_values=np.zeros(
                        (2, 5), dtype=np.float32
                    ),
                )
                collector.sample_process()
        self.assertEqual(validate.call_count, 2)
        self.assertEqual(collector.packed_sample_count, 4)

    def test_collector_revalidates_complete_frame_before_pack(self):
        collector = FrameCollectorV1(1)
        frame = _team(100)
        with patch.object(frame, "validate", wraps=frame.validate) as validate:
            collector.save_frame(frame, 0)
            collector.save_last_frame(
                (0, 0),
                np.zeros((2, 5), dtype=np.float32),
                0,
                command_reward=0.0,
                actor_auxiliary_bootstrap_values=np.zeros(
                    (2, 5), dtype=np.float32
                ),
            )
            collector.sample_process()
        self.assertEqual(validate.call_count, 2)

    def test_collector_post_gae_validation_rejects_mutated_nonfinite_field(self):
        collector = FrameCollectorV1(1)
        collector.save_frame(_team(100), 0)
        original = collector._calculate_gae

        def calculate_then_corrupt(agent_id):
            original(agent_id)
            first = next(iter(collector.frames[agent_id].values()))
            first.actors[0].advantage = float("nan")

        with patch.object(
            collector,
            "_calculate_gae",
            side_effect=calculate_then_corrupt,
        ), self.assertRaisesRegex(ValueError, "actor advantage is NaN/Inf"):
            collector.sample_process()

    def test_unpack_supports_torch_batch_without_copying_protocol(self):
        first = pack_sequence_v1([_team(100)]).sample
        second = pack_sequence_v1([_team(200, marker=4)]).sample
        batch = torch.from_numpy(np.stack((first, second)))
        unpacked = unpack_sample_v1(batch)
        self.assertEqual(
            tuple(unpacked["actors"][0]["feature"].shape),
            (2, 16, FrozenV1Config.FEATURE_DIM),
        )
        self.assertEqual(
            tuple(unpacked["command"]["conditioned_legal_masks"].shape),
            (2, 16, 80),
        )
        self.assertEqual(unpacked["actors"][0]["feature"][1, 0, 0].item(), 4)

    def test_old_scalar_critic_sample_is_rejected(self):
        with self.assertRaisesRegex(ValueError, str(FrozenV1Config.SAMPLE_DIM)):
            unpack_sample_v1(np.zeros(332928, dtype=np.float32))

    def test_collector_actor_and_unique_command_gae(self):
        collector = FrameCollectorV1(1, gamma=1.0, lamda=1.0)
        collector.save_frame(_team(100, rewards=(0, 0)), 0)
        step_auxiliary = (
            np.array([1, 2, 3, 4, 5], dtype=np.float32),
            np.array([5, 4, 3, 2, 1], dtype=np.float32),
        )
        terminal_auxiliary = (
            np.array([10, 20, 30, 40, 50], dtype=np.float32),
            np.array([50, 40, 30, 20, 10], dtype=np.float32),
        )
        collector.save_frame(
            _team(
                106,
                rewards=(1, 3),
                auxiliary_rewards=step_auxiliary,
                command_reward=4.0,
            ),
            0,
        )
        collector.save_last_frame(
            (5, 7),
            terminal_auxiliary,
            0,
            command_reward=6.0,
            actor_auxiliary_bootstrap_values=np.zeros((2, 5), dtype=np.float32),
        )
        samples = collector.sample_process()[0]
        self.assertEqual(len(samples), 1)
        unpacked = unpack_sample_v1(samples[0].sample)
        self.assertEqual(unpacked["actors"][0]["advantage"][0, 0], 6)
        self.assertEqual(unpacked["actors"][0]["advantage"][1, 0], 5)
        self.assertEqual(unpacked["actors"][1]["advantage"][0, 0], 10)
        self.assertEqual(unpacked["actors"][1]["advantage"][1, 0], 7)
        np.testing.assert_array_equal(
            unpacked["actors"][0]["auxiliary_returns"][0],
            step_auxiliary[0] + terminal_auxiliary[0],
        )
        np.testing.assert_array_equal(
            unpacked["actors"][0]["auxiliary_returns"][1],
            terminal_auxiliary[0],
        )
        np.testing.assert_array_equal(
            unpacked["actors"][1]["auxiliary_returns"][0],
            step_auxiliary[1] + terminal_auxiliary[1],
        )
        # Consecutive real command choices are separate temporal options.  The
        # later decision's residual does not leak into the earlier one.
        self.assertEqual(unpacked["command"]["advantage"][0, 0], 4)
        self.assertEqual(unpacked["command"]["advantage"][1, 0], 6)

    def test_command_option_accumulates_delayed_reward_through_forced_hold(self):
        collector = FrameCollectorV1(1, gamma=1.0, lamda=1.0)
        first = _team(100, command_reward=0.0)
        hold = _team(106, command_reward=4.0)
        first.command.is_train = 1.0
        hold.command.is_train = 0.0
        collector.save_frame(first, 0)
        collector.save_frame(hold, 0)
        collector.save_last_frame(
            (0.0, 0.0),
            np.zeros((2, 5), dtype=np.float32),
            0,
            command_reward=6.0,
            actor_auxiliary_bootstrap_values=np.zeros((2, 5), dtype=np.float32),
        )
        unpacked = unpack_sample_v1(collector.sample_process()[0][0].sample)
        self.assertEqual(unpacked["command"]["advantage"][0, 0], 10)
        self.assertEqual(unpacked["command"]["advantage"][1, 0], 6)

    def test_command_sparse_reward_uses_025_025_050_and_excludes_dense_details(self):
        class RewardState:
            last_kda_reward_by_runtime = {11: 0.8, 12: -1.0, 13: 2.0}
            last_kda_reward_by_config = {}
            last_command_official_kda_reward = 2.0
            last_command_goal_progress_reward = 0.0
            last_command_no_command_idle_penalty = -0.005
            last_command_attribution_active = True
            last_command_attribution_intent = 3
            last_tyrant_team_reward = 1.0
            last_structure_team_reward = 0.3
            last_post_kill_push_reward = 0.2
            last_crystal_danger_reward = -0.02
            last_defense_pressure_reward = -0.015
            last_temporal_focus_team_reward = 0.05
            last_group_attack_team_reward = 0.08
            last_crystal_outcome_reward = 0.0
            last_command_teammate_abandonment_penalty = -0.06
            last_command_teammate_support_reward = 0.05

        heroes = []
        for runtime_id, config_id in ((11, 132), (12, 141), (13, 193)):
            reward = np.zeros(12, dtype=np.float32)
            # Dense values must not leak into command reward.
            reward[0:3] = (100, 200, 300)
            reward[7:9] = (400, 500)
            heroes.append(
                {
                    "runtime_id": runtime_id,
                    "config_id": config_id,
                    "reward": reward,
                }
            )
        heroes[0]["reward"][3] = 2.0
        heroes[1]["reward"][10] = 4.0
        heroes[2]["reward"][3] = 1.0
        controlled = heroes[:2]
        result = build_command_sparse_reward_v1(
            RewardState(), {"heroes": heroes}, controlled_infos=controlled
        )
        # reward detail index 3 is the disabled official tower component and
        # must not leak into the custom tower path.
        expected = (
            0.25 * 0.8
            + 0.25 * (-1.0 + 4.0)
            + 0.50 * 2.0
            + 1.0
            + 0.3
            + 0.2
            - 0.02
            - 0.015
            + 0.05
            + 0.08
            - 0.06
            + 0.05
            - 0.005
        )
        self.assertAlmostEqual(result, expected, places=7)

    def test_command_sparse_reward_accepts_platform_two_hero_reward_payload(self):
        class RewardState:
            last_kda_reward_by_runtime = {11: 0.8, 12: -1.0}
            last_kda_reward_by_config = {193: 2.0}
            last_command_official_kda_reward = 2.0
            last_command_goal_progress_reward = 0.0
            last_command_attribution_active = True
            last_command_attribution_intent = 3
            last_tyrant_team_reward = 1.0
            last_structure_team_reward = 0.0
            last_post_kill_push_reward = 0.0
            last_crystal_outcome_reward = 0.0

        controlled = []
        for runtime_id, config_id in ((11, 132), (12, 141)):
            reward = np.zeros(12, dtype=np.float32)
            reward[0:3] = (100, 200, 300)
            reward[7:9] = (400, 500)
            controlled.append(
                {
                    "runtime_id": runtime_id,
                    "config_id": config_id,
                    "reward": reward,
                }
            )
        controlled[0]["reward"][3] = 2.0
        controlled[1]["reward"][10] = 4.0

        result = build_command_sparse_reward_v1(
            RewardState(), {"heroes": controlled}, controlled_infos=controlled
        )
        # The platform has no official reward-detail vector. Exact attributed
        # official KDA remains available, but controlled pushing is never used
        # as a synthetic official attack-crystal proxy.
        expected = (
            0.25 * 0.8
            + 0.25 * (-1.0 + 4.0)
            + 0.50 * 2.0
            + 1.0
        )
        self.assertAlmostEqual(result, expected, places=7)

    def test_runtime_change_starts_new_sequence_and_hidden(self):
        collector = FrameCollectorV1(1)
        collector.save_frame(_team(100, marker=1), 0)
        collector.save_frame(_team(106, marker=2), 0)
        collector.save_frame(
            _team(112, runtime_ids=(31, 32), marker=9), 0
        )
        collector.save_last_frame(
            (0, 0),
            np.zeros((2, 5), dtype=np.float32),
            0,
            command_reward=0.0,
            actor_auxiliary_bootstrap_values=np.zeros((2, 5), dtype=np.float32),
        )
        samples = collector.sample_process()[0]
        self.assertEqual(len(samples), 2)
        first = unpack_sample_v1(samples[0].sample)
        second = unpack_sample_v1(samples[1].sample)
        self.assertEqual(first["actors"][0]["is_train"].sum(), 2)
        self.assertEqual(second["actors"][0]["is_train"].sum(), 1)
        np.testing.assert_array_equal(
            second["actors"][0]["initial_hidden"], np.full(128, 9)
        )

    def test_runtime_change_resets_actor_and_auxiliary_gae(self):
        collector = FrameCollectorV1(1, gamma=1.0, lamda=1.0)
        collector.save_frame(_team(100), 0)
        boundary_auxiliary = (
            np.ones(5, dtype=np.float32),
            np.full(5, 2, dtype=np.float32),
        )
        collector.save_frame(
            _team(
                106,
                runtime_ids=(31, 32),
                rewards=(2, 3),
                auxiliary_rewards=boundary_auxiliary,
            ),
            0,
        )
        collector.save_last_frame(
            rewards=(10, 20),
            auxiliary_rewards=np.full((2, 5), 10, dtype=np.float32),
            agent_id=0,
            command_reward=0.0,
            actor_auxiliary_bootstrap_values=np.zeros((2, 5), dtype=np.float32),
        )
        samples = collector.sample_process()[0]
        first = unpack_sample_v1(samples[0].sample)
        second = unpack_sample_v1(samples[1].sample)
        self.assertEqual(first["actors"][0]["advantage"][0, 0], 2)
        self.assertEqual(second["actors"][0]["advantage"][0, 0], 10)
        np.testing.assert_array_equal(
            first["actors"][0]["auxiliary_returns"][0],
            boundary_auxiliary[0],
        )
        np.testing.assert_array_equal(
            second["actors"][0]["auxiliary_returns"][0],
            np.full(5, 10, dtype=np.float32),
        )

    def test_per_step_rnn_reset_breaks_actor_bootstrap_and_gae(self):
        collector = FrameCollectorV1(1, gamma=1.0, lamda=1.0)
        collector.save_frame(_team(100), 0)
        collector.save_frame(
            _team(106, rewards=(2, 3), rnn_resets=(1.0, 0.0)),
            0,
        )
        collector.save_last_frame(
            rewards=(10, 20),
            auxiliary_rewards=np.zeros((2, 5), dtype=np.float32),
            agent_id=0,
            command_reward=0.0,
            actor_auxiliary_bootstrap_values=np.zeros((2, 5), dtype=np.float32),
        )
        unpacked = unpack_sample_v1(collector.sample_process()[0][0].sample)
        self.assertEqual(unpacked["actors"][0]["rnn_reset"][1, 0], 1)
        self.assertEqual(unpacked["actors"][1]["rnn_reset"][1, 0], 0)
        self.assertEqual(unpacked["actors"][0]["advantage"][0, 0], 2)
        self.assertEqual(unpacked["actors"][0]["advantage"][1, 0], 10)
        self.assertEqual(unpacked["actors"][1]["advantage"][0, 0], 23)

    def test_sixteen_step_boundary_has_no_overlap_or_extra_hidden(self):
        collector = FrameCollectorV1(1)
        for index in range(17):
            collector.save_frame(_team(100 + index * 6, marker=index + 1), 0)
        collector.save_last_frame(
            (0, 0),
            np.zeros((2, 5), dtype=np.float32),
            0,
            command_reward=0.0,
            actor_auxiliary_bootstrap_values=np.zeros((2, 5), dtype=np.float32),
        )
        samples = collector.sample_process()[0]
        self.assertEqual(len(samples), 2)
        first = unpack_sample_v1(samples[0].sample)
        second = unpack_sample_v1(samples[1].sample)
        self.assertEqual(first["actors"][0]["is_train"].sum(), 16)
        self.assertEqual(second["actors"][0]["is_train"].sum(), 1)
        self.assertEqual(first["actors"][0]["feature"][-1, 0], 16)
        self.assertEqual(second["actors"][0]["feature"][0, 0], 17)
        self.assertEqual(
            schema_v1.SAMPLE_LAYOUT["actor_0_initial_hidden"].size, 128
        )
        self.assertEqual(
            schema_v1.SAMPLE_LAYOUT["actor_1_initial_hidden"].size, 128
        )

    def test_pack_rejects_runtime_change_inside_sequence(self):
        with self.assertRaisesRegex(ValueError, "runtime change"):
            pack_sequence_v1(
                [_team(100), _team(106, runtime_ids=(31, 32))]
            )

    def test_pack_rejects_invalid_auxiliary_arrays(self):
        frame = _team(100)
        frame.actors[0].auxiliary_return[2] = np.nan
        with self.assertRaisesRegex(ValueError, "auxiliary return.*NaN/Inf"):
            pack_sequence_v1([frame])

    def test_reward_payload_mapping_and_validation(self):
        final_reward, auxiliary = split_reward_payload_v1(
            np.arange(1, 13, dtype=np.float32)
        )
        self.assertEqual(final_reward, 12.0)
        np.testing.assert_array_equal(
            auxiliary, np.array([5, 18, 18, 15, 10], dtype=np.float32)
        )
        with self.assertRaisesRegex(ValueError, "11 details plus final"):
            split_reward_payload_v1(np.arange(11, dtype=np.float32))
        invalid = np.arange(12, dtype=np.float32)
        invalid[4] = np.nan
        with self.assertRaisesRegex(ValueError, "NaN/Inf"):
            split_reward_payload_v1(invalid)

    def test_auxiliary_truncation_bootstrap_is_used_per_head(self):
        collector = FrameCollectorV1(1, gamma=1.0, lamda=1.0)
        current_values = (
            np.array([1, 2, 3, 4, 5], dtype=np.float32),
            np.array([6, 7, 8, 9, 10], dtype=np.float32),
        )
        bootstrap_values = (
            np.array([11, 12, 13, 14, 15], dtype=np.float32),
            np.array([16, 17, 18, 19, 20], dtype=np.float32),
        )
        collector.save_frame(
            _team(100, auxiliary_values=current_values), agent_id=0
        )
        collector.save_last_frame(
            rewards=(0, 0),
            auxiliary_rewards=np.zeros((2, 5), dtype=np.float32),
            agent_id=0,
            command_reward=0.0,
            actor_auxiliary_bootstrap_values=bootstrap_values,
        )
        unpacked = unpack_sample_v1(collector.sample_process()[0][0].sample)
        np.testing.assert_array_equal(
            unpacked["actors"][0]["auxiliary_returns"][0],
            bootstrap_values[0],
        )
        np.testing.assert_array_equal(
            unpacked["actors"][1]["auxiliary_returns"][0],
            bootstrap_values[1],
        )


if __name__ == "__main__":
    unittest.main()
