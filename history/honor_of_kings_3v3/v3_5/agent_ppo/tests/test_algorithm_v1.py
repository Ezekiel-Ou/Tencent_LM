#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""End-to-end PPO, hybrid critic, auxiliary critic, and learner tests."""

import copy
import inspect
import unittest
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from agent_ppo.algorithm.algorithm_v1 import (
    AlgorithmV1,
    PPOLossV1,
    _dual_clip_surrogate,
)
from agent_ppo.conf.conf import FrozenV1Config, V1TrainingConfig
from agent_ppo.feature import schema_v1
from agent_ppo.feature.collaboration_v1 import build_command_legal_masks_v1
from agent_ppo.feature.definition_v1 import (
    ActorFrameV1,
    CommandFrameV1,
    TeamFrameV1,
    pack_sequence_v1,
)
from agent_ppo.feature.feature_process.v1_feature_process import FeatureProcessV1
from agent_ppo.model.action_v1 import sample_actor_actions_v1
from agent_ppo.model.command_v1 import evaluate_command_actions_v1
from agent_ppo.model.network_v1 import NetworkV1
from agent_ppo.tests.test_feature_process_v1 import make_observation


def _build_sample(model, command_intent):
    observation = make_observation()
    processor = FeatureProcessV1(1)
    features = processor.process_feature(observation)
    feature_tensor = (
        torch.from_numpy(np.stack(features)).unsqueeze(0).unsqueeze(2)
    )
    hidden = torch.zeros(1, 2, 128)
    raw_legal = torch.from_numpy(
        np.stack(
            [
                np.asarray(info["legal_action"], dtype=np.float32)
                for info in observation["heroes"][:2]
            ]
        )
    ).unsqueeze(0).unsqueeze(2)
    model.eval()
    with torch.no_grad():
        output = model(feature_tensor, hidden)
        actor_sample = sample_actor_actions_v1(
            output["actor"], raw_legal, deterministic=True
        )

    context = replace(processor.command_runtime_context(), jungle_route_active=False)
    masks = build_command_legal_masks_v1(context)
    if command_intent == 2:
        command_action = np.array([2, 1, 0], dtype=np.float32)
        active = np.array([1, 1, 0], dtype=np.float32)
    elif command_intent == 5:
        command_action = np.array([5, 0, 4], dtype=np.float32)
        active = np.array([1, 0, 1], dtype=np.float32)
    else:
        raise ValueError(command_intent)
    conditioned_command_legal = masks.conditioned(command_intent)
    with torch.no_grad():
        command_evaluation = evaluate_command_actions_v1(
            output["command"],
            torch.from_numpy(conditioned_command_legal).reshape(1, 1, 80),
            torch.from_numpy(command_action).reshape(1, 1, 3),
            torch.from_numpy(active).reshape(1, 1, 3),
        )
    command_old_log = command_evaluation["selected_log_probabilities"][0, 0].numpy()

    actor_frames = []
    for actor_index, info in enumerate(observation["heroes"][:2]):
        button = int(actor_sample["actions"][0, actor_index, 0, 0])
        value = float(output["actor"]["actor_value"][0, actor_index, 0, 0])
        auxiliary_value = output["actor"]["actor_auxiliary_values"][
            0, actor_index, 0
        ].numpy()
        actor_frames.append(
            ActorFrameV1(
                runtime_id=info["runtime_id"],
                feature=features[actor_index],
                conditioned_legal_action=actor_sample[
                    "conditioned_legal_action"
                ][0, actor_index, 0].numpy(),
                action=actor_sample["actions"][0, actor_index, 0].numpy(),
                old_action_probabilities=actor_sample[
                    "probability_vector"
                ][0, actor_index, 0].numpy(),
                sub_action_mask=np.asarray(
                    info["sub_action_mask"][str(button)], dtype=np.float32
                ),
                rnn_hidden=np.zeros(128, dtype=np.float32),
                auxiliary_value=auxiliary_value,
                auxiliary_return=auxiliary_value
                + np.linspace(0.1, 0.5, 5, dtype=np.float32),
                value=value,
                return_value=value + 0.5,
                advantage=1.0,
                is_train=1.0,
            )
        )
    command_value = float(output["command"]["command_value"][0, 0, 0])
    command_frame = CommandFrameV1(
        conditioned_legal_mask=conditioned_command_legal,
        action=command_action,
        old_log_probabilities=command_old_log,
        active_head_mask=active,
        value=command_value,
        return_value=command_value + 0.5,
        advantage=1.0,
        is_train=float(masks.command_is_train),
    )
    return pack_sequence_v1(
        [TeamFrameV1(frame_no=100, actors=tuple(actor_frames), command=command_frame)]
    )


class PPOLossV1Test(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(37)

    def test_dual_clip_only_floors_extreme_negative_advantage_surrogate(self):
        ratio = torch.tensor([10.0, 10.0, 0.5])
        advantage = torch.tensor([-1.0, 1.0, -1.0])
        surrogate, active = _dual_clip_surrogate(
            ratio,
            advantage,
            clip_param=0.2,
            dual_clip_param=3.0,
        )
        torch.testing.assert_close(surrogate, torch.tensor([-3.0, 1.2, -0.8]))
        self.assertEqual(active.tolist(), [True, False, False])

    def test_dual_clip_rejects_invalid_floor(self):
        with self.assertRaisesRegex(ValueError, "greater than 1"):
            _dual_clip_surrogate(
                torch.ones(1),
                -torch.ones(1),
                clip_param=0.2,
                dual_clip_param=1.0,
            )

    def test_actor_and_command_losses_apply_dual_clip(self):
        from agent_ppo.feature import schema_v1

        model = NetworkV1()
        sample = _build_sample(model, 2)
        for actor_index in range(2):
            step_start = schema_v1.SAMPLE_LAYOUT[
                f"actor_{actor_index}_steps"
            ].start
            advantage = schema_v1.ACTOR_STEP_LAYOUT["advantage"]
            probabilities = schema_v1.ACTOR_STEP_LAYOUT[
                "old_action_probabilities"
            ]
            sample.sample[step_start + advantage.start] = -1.0
            sample.sample[
                step_start + probabilities.start : step_start + probabilities.stop
            ] *= 0.1

        command_start = schema_v1.SAMPLE_LAYOUT["command_steps"].start
        command_advantage = schema_v1.COMMAND_STEP_LAYOUT["advantage"]
        old_log_probs = schema_v1.COMMAND_STEP_LAYOUT["old_log_probs"]
        sample.sample[command_start + command_advantage.start] = -1.0
        sample.sample[
            command_start + old_log_probs.start : command_start + old_log_probs.stop
        ] -= np.log(10.0)

        batch = torch.from_numpy(sample.sample).unsqueeze(0)
        _, metrics, _ = PPOLossV1()(model, batch)
        self.assertAlmostEqual(metrics["actor_policy_loss"].item(), 3.0, places=5)
        self.assertAlmostEqual(metrics["command_policy_loss"].item(), 3.0, places=5)
        self.assertAlmostEqual(
            metrics["actor_dual_clip_fraction"].item(), 1.0, places=5
        )
        self.assertAlmostEqual(
            metrics["command_dual_clip_fraction"].item(), 1.0, places=5
        )

    def test_actor_and_both_command_parameter_heads_get_nonzero_gradient(self):
        model = NetworkV1()
        samples = [_build_sample(model, 2), _build_sample(model, 5)]
        batch = torch.from_numpy(np.stack([sample.sample for sample in samples]))
        loss, metrics, _ = PPOLossV1()(model, batch)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        gradients = {
            "actor_button": model.actor.policy_router.routes[
                "marco"
            ].ordinary_heads[0].output.weight.grad,
            "actor_target": model.actor.policy_router.routes[
                "marco"
            ].target_query.linear.weight.grad,
            "actor_value_self": model.actor.value_head.self_output.weight.grad,
            "actor_value_team": model.actor.value_head.team_output.weight.grad,
            "actor_value_enemy": model.actor.value_head.enemy_output.weight.grad,
            "actor_auxiliary_value": model.actor.value_head.auxiliary_output.weight.grad,
            "gru": model.actor.backbone.gru.weight_ih_l0.grad,
            "command_intent": model.command.intent_head.output.weight.grad,
            "command_target": model.command.command_target_query.linear.weight.grad,
            "command_position": model.command.command_position_query.linear.weight.grad,
            "command_value": model.command.value_output.weight.grad,
        }
        for name, gradient in gradients.items():
            self.assertIsNotNone(gradient, name)
            self.assertGreater(gradient.abs().sum().item(), 0, name)
            self.assertTrue(torch.isfinite(gradient).all(), name)
        self.assertEqual(metrics["command_policy_valid_count"].item(), 2)
        self.assertEqual(metrics["command_value_valid_count"].item(), 2)
        self.assertAlmostEqual(metrics["command_approx_kl"].item(), 0, places=5)
        self.assertLess(metrics["actor_hybrid_identity_error"].item(), 1e-7)
        self.assertGreater(metrics["actor_auxiliary_value_loss"].item(), 0)
        auxiliary_gradient = model.actor.value_head.auxiliary_output.weight.grad
        self.assertTrue(torch.all(auxiliary_gradient.abs().sum(dim=1) > 0))

    def test_command_value_trains_when_command_policy_is_disabled(self):
        model = NetworkV1()
        sample = _build_sample(model, 2)
        # Disable only command policy training; the unique command value target remains valid.
        from agent_ppo.feature import schema_v1

        command_start = schema_v1.SAMPLE_LAYOUT["command_steps"].start
        is_train_index = schema_v1.COMMAND_STEP_LAYOUT["is_train"].start
        sample.sample[command_start + is_train_index] = 0
        batch = torch.from_numpy(sample.sample).unsqueeze(0)
        loss, metrics, _ = PPOLossV1()(model, batch)
        loss.backward()
        self.assertEqual(metrics["command_policy_valid_count"].item(), 0)
        self.assertEqual(metrics["command_value_valid_count"].item(), 1)
        self.assertIsNotNone(model.command.value_output.weight.grad)
        self.assertGreater(model.command.value_output.weight.grad.abs().sum(), 0)

    def test_one_dead_actor_does_not_invalidate_unique_command_step(self):
        from agent_ppo.feature import schema_v1

        model = NetworkV1()
        sample = _build_sample(model, 2)
        actor_one_start = schema_v1.SAMPLE_LAYOUT["actor_1_steps"].start
        actor_is_train = schema_v1.ACTOR_STEP_LAYOUT["is_train"].start
        sample.sample[actor_one_start + actor_is_train] = 0.0
        batch = torch.from_numpy(sample.sample).unsqueeze(0)
        loss, metrics, _ = PPOLossV1()(model, batch)
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(metrics["actor_valid_count"].item(), 1)
        self.assertEqual(metrics["command_value_valid_count"].item(), 1)
        self.assertEqual(metrics["command_policy_valid_count"].item(), 1)

    def test_learner_replays_per_step_gru_reset_mask(self):
        from agent_ppo.feature import schema_v1

        class CaptureNetwork(NetworkV1):
            def forward(self, feature, initial_hidden, reset_mask=None, hero_ids=None):
                self.captured_reset_mask = reset_mask
                return super().forward(
                    feature,
                    initial_hidden,
                    reset_mask=reset_mask,
                    hero_ids=hero_ids,
                )

        model = CaptureNetwork()
        sample = _build_sample(model, 2)
        actor_one_start = schema_v1.SAMPLE_LAYOUT["actor_1_steps"].start
        reset_index = schema_v1.ACTOR_STEP_LAYOUT["rnn_reset"].start
        sample.sample[actor_one_start + reset_index] = 1.0
        batch = torch.from_numpy(sample.sample).unsqueeze(0)
        loss, _, _ = PPOLossV1()(model, batch)
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(tuple(model.captured_reset_mask.shape), (1, 2, 16))
        self.assertEqual(model.captured_reset_mask[0, 1, 0].item(), 1.0)
        self.assertEqual(model.captured_reset_mask[0, 0, 0].item(), 0.0)

    def test_algorithm_optimizer_step_and_metrics(self):
        model = NetworkV1()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        algorithm = AlgorithmV1(model, optimizer, device=torch.device("cpu"))
        algorithm.micro_batch_size = 1
        samples = [_build_sample(model, 2), _build_sample(model, 5)]
        before = model.command.intent_head.output.weight.detach().clone()
        with patch.object(
            AlgorithmV1,
            "_scalar_metrics_to_cpu",
            side_effect=AlgorithmV1._scalar_metrics_to_cpu,
        ) as transfer:
            metrics = algorithm.learn(samples)
        self.assertEqual(transfer.call_count, 1)
        after = model.command.intent_head.output.weight.detach()
        self.assertFalse(torch.equal(before, after))
        self.assertEqual(metrics["train_step"], 1)
        for name in (
            "total_loss",
            "actor_policy_loss",
            "actor_dual_clip_fraction",
            "command_policy_loss",
            "command_dual_clip_fraction",
            "command_value_loss",
            "actor_auxiliary_value_loss",
            "actor_aux_farming_loss",
            "actor_aux_win_explained_variance",
            "actor_hybrid_cancellation_ratio",
            "gradient_norm",
        ):
            self.assertTrue(np.isfinite(metrics[name]), name)
        self.assertEqual(metrics["micro_batch_size"], 1)
        self.assertEqual(metrics["amp_enabled"], 0.0)
        self.assertIn("actor_explained_variance", metrics)
        self.assertIn("hero_132_entropy", metrics)

    def test_format_data_builds_one_contiguous_float32_batch(self):
        model = NetworkV1()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        algorithm = AlgorithmV1(model, optimizer, device=torch.device("cpu"))
        first = np.arange(FrozenV1Config.SAMPLE_DIM, dtype=np.float32)
        second = torch.from_numpy(first + 1.0)
        batch = algorithm.format_data(
            [SimpleNamespace(sample=first), SimpleNamespace(sample=second)]
        )
        self.assertEqual(tuple(batch.shape), (2, FrozenV1Config.SAMPLE_DIM))
        self.assertEqual(batch.dtype, torch.float32)
        self.assertTrue(batch.is_contiguous())
        torch.testing.assert_close(batch[0], torch.from_numpy(first))
        torch.testing.assert_close(batch[1], torch.from_numpy(first + 1.0))

    def test_cpu_staging_metadata_covers_gru_resets_and_command_heads(self):
        model = NetworkV1()
        sample = _build_sample(model, 2)
        actor_steps = sample.sample[
            schema_v1.SAMPLE_LAYOUT["actor_0_steps"].slice
        ].reshape(
            FrozenV1Config.RNN_TIME_STEPS,
            FrozenV1Config.ACTOR_STEP_DIM,
        )
        actor_steps[3, schema_v1.ACTOR_STEP_LAYOUT["rnn_reset"].start] = 1.0
        command_steps = sample.sample[
            schema_v1.SAMPLE_LAYOUT["command_steps"].slice
        ].reshape(
            FrozenV1Config.RNN_TIME_STEPS,
            FrozenV1Config.COMMAND_STEP_DIM,
        )
        active_slice = schema_v1.COMMAND_STEP_LAYOUT["active_head_mask"].slice
        command_steps[:, active_slice] = 0.0
        command_steps[0, active_slice] = np.array(
            [1.0, 0.0, 1.0],
            dtype=np.float32,
        )

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        algorithm = AlgorithmV1(model, optimizer, device=torch.device("cpu"))
        batch, metadata = algorithm.format_data(
            [sample],
            return_metadata=True,
        )
        self.assertEqual(batch.shape, (1, FrozenV1Config.SAMPLE_DIM))
        self.assertTrue(metadata["has_internal_reset"])
        self.assertEqual(
            metadata["command_head_activity"],
            (False, True),
        )

    def test_nonvalidation_loss_path_avoids_tensor_any_diagnostics(self):
        model = NetworkV1()
        sample = _build_sample(model, 2)
        batch = torch.from_numpy(np.stack([sample.sample]))
        with patch("torch.any", side_effect=AssertionError("unexpected sync")):
            loss, metrics, _ = PPOLossV1()(
                model,
                batch,
                validate_runtime=False,
                has_internal_reset=False,
                command_head_activity=(True, True),
            )
        self.assertTrue(torch.isfinite(loss))
        self.assertIn("actor_explained_variance", metrics)

    def test_scalar_metrics_use_one_packed_transfer_result(self):
        values = AlgorithmV1._scalar_metrics_to_cpu(
            {
                "first": torch.tensor(1.25),
                "second": torch.tensor(-2.5),
            }
        )
        self.assertEqual(values, {"first": 1.25, "second": -2.5})

    def test_amp_overflow_reaches_grad_scaler_recovery(self):
        learn_source = inspect.getsource(AlgorithmV1.learn)
        self.assertNotIn("update_finite", learn_source)
        unscale_index = learn_source.index("self.grad_scaler.unscale_")
        step_index = learn_source.index("self.grad_scaler.step")
        update_index = learn_source.index("self.grad_scaler.update")
        self.assertLess(unscale_index, step_index)
        self.assertLess(step_index, update_index)
        self.assertNotIn(
            "FloatingPointError",
            learn_source[unscale_index:step_index],
        )
        loss_source = inspect.getsource(PPOLossV1.__call__)
        self.assertIn("v1 PPO total loss is NaN/Inf", loss_source)

    def test_auxiliary_targets_do_not_change_policy_objective(self):
        from agent_ppo.feature import schema_v1

        model = NetworkV1()
        sample = _build_sample(model, 2)
        baseline_batch = torch.from_numpy(sample.sample.copy()).unsqueeze(0)
        _, baseline_metrics, _ = PPOLossV1()(model, baseline_batch)
        for actor_index in range(2):
            step_start = schema_v1.SAMPLE_LAYOUT[
                f"actor_{actor_index}_steps"
            ].start
            auxiliary = schema_v1.ACTOR_STEP_LAYOUT["auxiliary_returns"]
            sample.sample[
                step_start + auxiliary.start : step_start + auxiliary.stop
            ] += np.arange(1, 6, dtype=np.float32)
        changed_batch = torch.from_numpy(sample.sample.copy()).unsqueeze(0)
        _, changed_metrics, _ = PPOLossV1()(model, changed_batch)
        torch.testing.assert_close(
            baseline_metrics["actor_policy_loss"],
            changed_metrics["actor_policy_loss"],
            rtol=0,
            atol=0,
        )
        self.assertNotEqual(
            baseline_metrics["actor_auxiliary_value_loss"].item(),
            changed_metrics["actor_auxiliary_value_loss"].item(),
        )

    def test_padding_does_not_contribute_auxiliary_loss(self):
        from agent_ppo.conf.conf import FrozenV1Config
        from agent_ppo.feature import schema_v1

        model = NetworkV1()
        sample = _build_sample(model, 2)
        baseline = torch.from_numpy(sample.sample.copy()).unsqueeze(0)
        _, baseline_metrics, _ = PPOLossV1()(model, baseline)
        auxiliary = schema_v1.ACTOR_STEP_LAYOUT["auxiliary_returns"]
        for actor_index in range(2):
            actor_start = schema_v1.SAMPLE_LAYOUT[
                f"actor_{actor_index}_steps"
            ].start
            for step in range(1, FrozenV1Config.RNN_TIME_STEPS):
                step_start = actor_start + step * FrozenV1Config.ACTOR_STEP_DIM
                sample.sample[
                    step_start + auxiliary.start : step_start + auxiliary.stop
                ] = 1e6
        changed = torch.from_numpy(sample.sample.copy()).unsqueeze(0)
        _, changed_metrics, _ = PPOLossV1()(model, changed)
        torch.testing.assert_close(
            baseline_metrics["actor_auxiliary_value_loss"],
            changed_metrics["actor_auxiliary_value_loss"],
            rtol=0,
            atol=0,
        )

    def test_global_advantage_statistics_span_micro_batches(self):
        from agent_ppo.feature import schema_v1

        model = NetworkV1()
        samples = [_build_sample(model, 2), _build_sample(model, 5)]
        actor_values = (-1.0, 3.0)
        command_values = (-2.0, 2.0)
        for sample, actor_advantage, command_advantage in zip(
            samples, actor_values, command_values
        ):
            for actor_index in range(2):
                start = schema_v1.SAMPLE_LAYOUT[
                    f"actor_{actor_index}_steps"
                ].start
                start += schema_v1.ACTOR_STEP_LAYOUT["advantage"].start
                sample.sample[start] = actor_advantage
            start = schema_v1.SAMPLE_LAYOUT["command_steps"].start
            start += schema_v1.COMMAND_STEP_LAYOUT["advantage"].start
            sample.sample[start] = command_advantage

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        algorithm = AlgorithmV1(model, optimizer, device=torch.device("cpu"))
        statistics = algorithm._batch_statistics(samples)["normalization"]
        self.assertAlmostEqual(statistics["actor_mean"], 1.0)
        self.assertAlmostEqual(statistics["actor_std"], 2.0)
        self.assertAlmostEqual(statistics["command_mean"], 0.0)
        self.assertAlmostEqual(statistics["command_std"], 2.0)

    def test_statistics_plan_unpacks_each_sample_once(self):
        from agent_ppo.algorithm import algorithm_v1 as algorithm_module

        model = NetworkV1()
        samples = [_build_sample(model, 2), _build_sample(model, 5)]
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        algorithm = AlgorithmV1(model, optimizer, device=torch.device("cpu"))
        algorithm.micro_batch_size = 1
        expected = algorithm._batch_statistics(samples)
        with patch.object(
            algorithm_module,
            "unpack_sample_v1",
            wraps=algorithm_module.unpack_sample_v1,
        ) as unpack:
            actual, local_counts = algorithm._batch_statistics_plan(samples)
        self.assertEqual(unpack.call_count, len(samples))
        self.assertEqual(actual, expected)
        self.assertEqual(len(local_counts), 2)

    def test_runtime_validation_cadence(self):
        model = NetworkV1()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        algorithm = AlgorithmV1(model, optimizer, device=torch.device("cpu"))
        with patch.object(
            V1TrainingConfig,
            "RUNTIME_VALIDATION_STARTUP_UPDATES",
            2,
        ), patch.object(V1TrainingConfig, "RUNTIME_VALIDATION_INTERVAL", 5):
            expected = {0: True, 1: True, 2: False, 4: False, 5: True}
            for train_step, should_validate in expected.items():
                algorithm.train_step = train_step
                self.assertEqual(
                    algorithm._should_validate_runtime(),
                    should_validate,
                )

    def test_full_and_micro_batch_updates_match_with_auxiliary_loss(self):
        source = NetworkV1()
        samples = [_build_sample(source, 2), _build_sample(source, 5)]
        full_model = copy.deepcopy(source)
        micro_model = copy.deepcopy(source)
        full = AlgorithmV1(
            full_model,
            torch.optim.Adam(full_model.parameters(), lr=1e-4),
            device=torch.device("cpu"),
        )
        micro = AlgorithmV1(
            micro_model,
            torch.optim.Adam(micro_model.parameters(), lr=1e-4),
            device=torch.device("cpu"),
        )
        full.micro_batch_size = 2
        micro.micro_batch_size = 1
        full.learn(samples)
        micro.learn(samples)
        for name in (
            "actor.value_head.self_output.weight",
            "actor.value_head.team_output.weight",
            "actor.value_head.enemy_output.weight",
            "actor.value_head.auxiliary_output.weight",
            "actor.policy_router.routes.marco.ordinary_heads.0.output.weight",
        ):
            torch.testing.assert_close(
                full_model.state_dict()[name],
                micro_model.state_dict()[name],
                rtol=1e-5,
                atol=1e-6,
            )


if __name__ == "__main__":
    unittest.main()
