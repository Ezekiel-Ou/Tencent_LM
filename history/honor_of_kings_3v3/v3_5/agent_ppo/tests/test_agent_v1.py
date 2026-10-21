#!/usr/bin/env python3
"""Atomic active-Agent integration smoke tests."""

import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from agent_ppo.agent import Agent
from agent_ppo import agent as agent_module
from agent_ppo.conf.conf import Config, FrozenV1Config
from agent_ppo.feature.definition import SampleData
from agent_ppo.tests.test_feature_process_v1 import make_observation


class AgentV1Test(unittest.TestCase):
    @staticmethod
    def _assert_act_data_close(expected, actual):
        expected_actors = expected.deterministic_actors
        actual_actors = actual.deterministic_actors
        assert expected_actors is not None
        assert actual_actors is not None
        for expected_actor, actual_actor in zip(
            expected_actors, actual_actors
        ):
            np.testing.assert_array_equal(
                expected_actor.action, actual_actor.action
            )
            np.testing.assert_allclose(
                expected_actor.probability_vector,
                actual_actor.probability_vector,
                rtol=0,
                atol=1e-5,
            )
            np.testing.assert_array_equal(
                expected_actor.conditioned_legal_action,
                actual_actor.conditioned_legal_action,
            )
            np.testing.assert_allclose(
                expected_actor.next_hidden,
                actual_actor.next_hidden,
                rtol=0,
                atol=1e-5,
            )
            np.testing.assert_allclose(
                expected_actor.auxiliary_values,
                actual_actor.auxiliary_values,
                rtol=0,
                atol=1e-5,
            )
            np.testing.assert_allclose(
                expected_actor.value,
                actual_actor.value,
                rtol=0,
                atol=1e-5,
            )
        expected_command = expected.deterministic_command
        actual_command = actual.deterministic_command
        assert expected_command is not None
        assert actual_command is not None
        np.testing.assert_array_equal(
            expected_command.action, actual_command.action
        )
        np.testing.assert_allclose(
            expected_command.old_log_probabilities,
            actual_command.old_log_probabilities,
            rtol=0,
            atol=1e-5,
        )
        np.testing.assert_array_equal(
            expected_command.conditioned_legal_mask,
            actual_command.conditioned_legal_mask,
        )
        np.testing.assert_array_equal(
            expected_command.active_head_mask,
            actual_command.active_head_mask,
        )
        self_protocol = expected_command.protocol_action
        actual_protocol = actual_command.protocol_action
        if self_protocol != actual_protocol:
            raise AssertionError(
                f"command protocol mismatch: {self_protocol} != {actual_protocol}"
            )

    def test_agent_source_is_safe_for_platform_model_pool_reload(self):
        source_path = Path(__file__).resolve().parents[1] / "agent.py"
        source = source_path.read_text(encoding="utf-8")
        self.assertNotIn("set_num_interop_threads", source)
        self.assertNotIn("_batched_model_inference", source)
        self.assertNotIn("finish_inference", source)
        self.assertIn("@load_opponent_agent_wrapper", source)
        compile(source, str(source_path), "exec")

    def test_invalid_command_is_contained_as_non_trainable_no_command(self):
        class InvalidLegalizer:
            def legalize(self, *args):
                return type(
                    "InvalidCommand",
                    (),
                    {"valid": False, "reason": "synthetic_invalid"},
                )()

        agent = Agent.__new__(Agent)
        agent.legalizer = InvalidLegalizer()
        agent.invalid_command_count = 0
        agent.logger = None
        sample = {
            "actions": torch.tensor([[[5, 0, 5]]], dtype=torch.long),
            "old_log_probabilities": torch.ones((1, 1, 3)),
            "conditioned_legal_mask": torch.ones(
                (1, 1, FrozenV1Config.COMMAND_LEGAL_MASK_DIM)
            ),
            "active_head_mask": torch.ones((1, 1, 3)),
            "command_is_train": torch.ones((1, 1)),
        }
        decision = agent._command_decision(
            sample, torch.tensor([[[1.25]]]), context=None, masks=None
        )
        self.assertEqual(decision.protocol_action, (0, 0, 0, 0))
        self.assertEqual(decision.is_train, 0.0)
        self.assertEqual(decision.action.tolist(), [0.0, 0.0, 0.0])
        self.assertEqual(decision.active_head_mask.tolist(), [1.0, 0.0, 0.0])
        self.assertEqual(decision.conditioned_legal_mask[0], 1.0)
        self.assertEqual(decision.conditioned_legal_mask[1:].sum(), 0.0)
        self.assertEqual(agent.command_legalizer_attempt_count, 1)
        self.assertEqual(agent.invalid_command_count, 1)

    def test_platform_aliases_use_active_protocol(self):
        self.assertEqual(Config.HERO_NUM, 2)
        self.assertEqual(Config.LSTM_UNIT_SIZE, 128)
        self.assertEqual(Config.SAMPLE_DIM, 337664)
        sample = SampleData(sample=np.zeros(337664, dtype=np.float32))
        self.assertEqual(np.asarray(sample.sample).shape, (337664,))

    def test_end_to_end_inference_emits_two_heroes_and_one_command(self):
        observation = make_observation()
        agent = Agent(device="cpu")
        agent.reset(observation)
        obs_data = agent.observation_process(observation)
        self.assertEqual(obs_data.feature.shape, (2, 10157))
        self.assertEqual(obs_data.raw_legal_action.shape, (2, 629))

        act_data = agent._model_inference([obs_data])
        agent.update_status(obs_data, act_data)
        hero_actions, command = agent.action_process(
            observation, act_data, stochastic=True
        )
        self.assertEqual(set(hero_actions), {132, 141})
        self.assertTrue(all(len(action) == 5 for action in hero_actions.values()))
        self.assertEqual(len(command), 4)
        self.assertIn(command[0], (0, 1, 2, 3))
        for decision in act_data.stochastic_actors:
            self.assertEqual(decision.auxiliary_values.shape, (5,))
            self.assertTrue(np.isfinite(decision.auxiliary_values).all())
        self.assertEqual(
            set(agent.rnn_state_by_config), {132, 141}
        )
        self.assertTrue(
            all(
                state["hidden"].shape == (FrozenV1Config.RNN_HIDDEN_SIZE,)
                for state in agent.rnn_state_by_config.values()
            )
        )

        scalar_values, auxiliary_values, command_value = agent.bootstrap_values(
            observation
        )
        self.assertEqual(len(scalar_values), 2)
        self.assertEqual(len(auxiliary_values), 2)
        self.assertTrue(all(value.shape == (5,) for value in auxiliary_values))
        self.assertTrue(np.isfinite(np.asarray(scalar_values)).all())
        self.assertTrue(np.isfinite(np.stack(auxiliary_values)).all())
        self.assertTrue(np.isfinite(command_value))

    def test_observation_process_accepts_missing_friendly_frame_state(self):
        runtime_by_config = {132: 11, 141: 12, 193: 13}
        for missing_config, official_config in ((132, 193), (193, 132)):
            with self.subTest(
                missing_config=missing_config,
                official_config=official_config,
            ):
                observation = make_observation()
                observation["official_model_info"] = {
                    "official": {
                        "runtime_id": runtime_by_config[official_config]
                    }
                }
                observation["frame_state"]["hero_states"] = [
                    hero
                    for hero in observation["frame_state"]["hero_states"]
                    if not (
                        hero["camp"] == 1
                        and hero["config_id"] == missing_config
                    )
                ]

                agent = Agent(device="cpu")
                agent.reset(observation)
                obs_data = agent.observation_process(observation)

                self.assertEqual(obs_data.feature.shape, (2, 10157))
                self.assertEqual(
                    obs_data.command_masks.intent.shape,
                    (FrozenV1Config.COMMAND_HEADS[0],),
                )

    def test_inference_builds_only_the_requested_action_branch(self):
        observation = make_observation()
        agent = Agent(device="cpu")
        agent.reset(observation)
        obs_data = agent.observation_process(observation)
        inference_modes = []
        original_forward = agent.model.forward

        def recorded_forward(*args, **kwargs):
            inference_modes.append(torch.is_inference_mode_enabled())
            return original_forward(*args, **kwargs)

        with patch.object(agent.model, "forward", side_effect=recorded_forward), patch(
            "agent_ppo.agent.sample_actor_actions_v1",
            wraps=agent_module.sample_actor_actions_v1,
        ) as actor_sampler, patch(
            "agent_ppo.agent.sample_command_actions_v1",
            wraps=agent_module.sample_command_actions_v1,
        ) as command_sampler:
            stochastic = agent._model_inference([obs_data], stochastic=True)
            deterministic = agent._model_inference([obs_data], stochastic=False)

        self.assertEqual(actor_sampler.call_count, 2)
        self.assertEqual(command_sampler.call_count, 2)
        self.assertFalse(actor_sampler.call_args_list[0].kwargs["deterministic"])
        self.assertTrue(actor_sampler.call_args_list[1].kwargs["deterministic"])
        self.assertIsNotNone(stochastic.stochastic_actors)
        self.assertIsNone(stochastic.deterministic_actors)
        self.assertIsNotNone(stochastic.stochastic_command)
        self.assertIsNone(stochastic.deterministic_command)
        self.assertIsNone(deterministic.stochastic_actors)
        self.assertIsNotNone(deterministic.deterministic_actors)
        self.assertIsNone(deterministic.stochastic_command)
        self.assertIsNotNone(deterministic.deterministic_command)
        self.assertEqual(inference_modes, [True, True])

    def test_observation_hot_path_reuses_frame_cache_without_public_aliasing(self):
        observation = make_observation()
        agent = Agent(device="cpu")
        agent.reset(observation)
        processor = agent.feature_processes

        with patch.object(
            processor,
            "_controlled_infos",
            wraps=processor._controlled_infos,
        ) as controlled_infos:
            obs_data = agent.observation_process(observation)

        self.assertEqual(controlled_infos.call_count, 1)
        self.assertIs(obs_data.feature, processor._memo_features)
        public_features = processor.process_feature(observation)
        public_features[0][0] += 1.0
        self.assertNotEqual(public_features[0][0], obs_data.feature[0, 0])

    def test_channel_four_mask_is_used_by_sampling_and_saved_probability_mask(self):
        observation = make_observation()
        image = observation["heroes"][0]["feature"].reshape(6, 17, 17)
        image[4, 7:10, 9:12] = 1.0
        agent = Agent(device="cpu")
        agent.reset(observation)
        obs_data = agent.observation_process(observation)
        self.assertEqual(obs_data.raw_legal_action[0, 13 + 13], 0.0)
        act_data = agent._model_inference([obs_data])
        # Conditioned layout is button[13] + move[25] + offsets + target.
        self.assertEqual(
            act_data.stochastic_actors[0].conditioned_legal_action[13 + 13],
            0.0,
        )

    def test_last_hit_shaping_is_added_to_farming_auxiliary_channel(self):
        observation = make_observation(frame=500, include_soldiers=False)
        killer = observation["frame_state"]["hero_states"][0]
        observation["frame_state"]["frame_action"]["dead_action"] = [
            {
                "killer": killer,
                "death": {
                    "runtime_id": 9001,
                    "camp": 2,
                    "actor_type": "ACTOR_MONSTER",
                    "sub_type": "ACTOR_SUB_SOLDIER",
                },
            }
        ]
        agent = Agent(device="cpu")
        agent.reset(observation)
        rewards = agent.consume_reward_shaping(observation)
        self.assertAlmostEqual(rewards[0], 0.05, places=7)
        self.assertAlmostEqual(
            agent.last_auxiliary_reward_shaping[0, 0], 0.05, places=7
        )
        self.assertEqual(agent.last_auxiliary_reward_shaping[0, 1:].sum(), 0)
        self.assertEqual(agent.last_auxiliary_reward_shaping[1].sum(), 0)

    def test_learner_checkpoint_restores_step_and_optimizer(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Agent(agent_type="learner", device="cpu")
            source.algorithm.train_step = 7
            source.save_model(path=directory, id="7")
            restored = Agent(agent_type="learner", device="cpu")
            restored.load_model(path=directory, id="7")
            self.assertEqual(restored.algorithm.train_step, 7)
            self.assertEqual(
                restored.loaded_checkpoint_identity[1:],
                (7, "model.ckpt-7.pkl"),
            )


if __name__ == "__main__":
    unittest.main()
