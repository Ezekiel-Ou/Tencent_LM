#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""Shape, pooling, GRU, hybrid value, and command network tests."""

import copy
import unittest
from unittest.mock import patch

import numpy as np
import torch

from agent_ppo.conf.conf import FrozenV1Config
from agent_ppo.feature import schema_v1
from agent_ppo.feature.feature_process.v1_feature_process import FeatureProcessV1
from agent_ppo.model.network_v1 import (
    ActorBackboneV1,
    ActorNetworkV1,
    FeatureEncoderV1,
    HeroPolicyRouterV1,
    NetworkV1,
    ResidualPSCNV1,
    masked_max_mean,
)
from agent_ppo.tests.test_feature_process_v1 import make_observation


def _feature_tensor(time=1):
    features = FeatureProcessV1(1).process_feature(make_observation())
    tensor = torch.from_numpy(np.stack(features, axis=0)).unsqueeze(0).unsqueeze(2)
    return tensor.repeat(1, 1, time, 1)


class FeatureEncoderV1Test(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)

    def test_exact_token_summary_and_input_shapes(self):
        encoder = FeatureEncoderV1()
        output = encoder(_feature_tensor())
        expected = {
            "image_embedding": (1, 2, 1, 128),
            "hero_tokens": (1, 2, 1, 6, 128),
            "relation_dynamic_tokens": (1, 2, 1, 6, 64),
            "soldier_tokens": (1, 2, 1, 20, 32),
            "organ_tokens": (1, 2, 1, 6, 32),
            "monster_tokens": (1, 2, 1, 20, 32),
            "bullet_tokens": (1, 2, 1, 10, 32),
            "target_bank": (1, 2, 1, 39, 64),
            "resource_tokens": (1, 2, 1, 5, 32),
            "team_input": (1, 2, 1, schema_v1.TEAM_INPUT_DIM),
            "private_input": (1, 2, 1, schema_v1.PRIVATE_INPUT_DIM),
        }
        for name, shape in expected.items():
            self.assertEqual(tuple(output[name].shape), shape, name)
            self.assertTrue(torch.isfinite(output[name]).all(), name)
        self.assertEqual(int(output["masks"]["soldiers"].sum()), 8)
        self.assertEqual(int(output["masks"]["targets"].sum()), 62)

    def test_masked_max_mean_empty_and_padding_invariance(self):
        tokens = torch.tensor(
            [
                [[1.0, 2.0], [1000.0, 1000.0], [3.0, 0.0]],
                [[9.0, 9.0], [8.0, 8.0], [7.0, 7.0]],
            ]
        )
        mask = torch.tensor([[1, 0, 1], [0, 0, 0]], dtype=torch.bool)
        pooled = masked_max_mean(tokens, mask)
        torch.testing.assert_close(pooled[0], torch.tensor([3.0, 2.0, 2.0, 1.0]))
        torch.testing.assert_close(pooled[1], torch.zeros(4))

    def test_public_raw_assertion_detects_semantic_drift(self):
        feature = _feature_tensor()
        soldier = schema_v1.FEATURE_BLOCK_BY_NAME["soldiers"]
        field = schema_v1.SOLDIER_FIELD_MAP["hp_ratio"].start
        feature[0, 1, 0, soldier.start + field] += 0.25
        with self.assertRaisesRegex(ValueError, "soldiers"):
            FeatureEncoderV1()(feature)

    def test_runtime_validation_can_be_skipped_after_startup(self):
        feature = _feature_tensor()
        soldier = schema_v1.FEATURE_BLOCK_BY_NAME["soldiers"]
        field = schema_v1.SOLDIER_FIELD_MAP["hp_ratio"].start
        feature[0, 1, 0, soldier.start + field] += 0.25
        output = FeatureEncoderV1()(feature, validate_runtime=False)
        self.assertEqual(output["team_input"].shape[-1], schema_v1.TEAM_INPUT_DIM)

    def test_query_private_exclusions_are_allowed(self):
        feature = _feature_tensor()
        hero = schema_v1.FEATURE_BLOCK_BY_NAME["heroes"]
        resource = schema_v1.FEATURE_BLOCK_BY_NAME["resource_anchors"]
        lane = schema_v1.FEATURE_BLOCK_BY_NAME["lane_state"]
        feature[0, 1, 0, hero.start + schema_v1.HERO_FIELD_MAP["is_query_main"].start] = 0.37
        feature[0, 1, 0, resource.start + schema_v1.RESOURCE_ANCHOR_FIELD_MAP["query_eta_norm"].start] = 0.22
        feature[0, 1, 0, lane.start + schema_v1.LANE_STATE_FIELD_MAP["query_has_minion_cover"].start] = 0.63
        output = FeatureEncoderV1()(feature)
        self.assertEqual(output["team_input"].shape[-1], 1560)


class BackboneV1Test(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(11)

    def test_residual_pscn_shape_and_backward(self):
        module = ResidualPSCNV1(790)
        value = torch.randn(4, 790, requires_grad=True)
        output = module(value)
        self.assertEqual(tuple(output.shape), (4, 256))
        output.square().mean().backward()
        self.assertIsNotNone(value.grad)
        self.assertTrue(torch.isfinite(value.grad).all())

    def test_time_one_and_time_sixteen_contract(self):
        model = ActorBackboneV1()
        for time in (1, FrozenV1Config.RNN_TIME_STEPS):
            feature = _feature_tensor(time=time)
            hidden = torch.zeros(1, 2, FrozenV1Config.RNN_HIDDEN_SIZE)
            output = model(feature, hidden)
            self.assertEqual(tuple(output["team_feature"].shape), (1, 2, time, 256))
            self.assertEqual(tuple(output["private_feature"].shape), (1, 2, time, 256))
            self.assertEqual(tuple(output["gru_output"].shape), (1, 2, time, 128))
            self.assertEqual(tuple(output["policy_feature"].shape), (1, 2, time, 256))
            self.assertEqual(tuple(output["next_hidden"].shape), (1, 2, 128))

    def test_actor_gru_states_are_independent(self):
        model = ActorBackboneV1()
        model.eval()
        feature = _feature_tensor(time=1)
        zero = torch.zeros(1, 2, 128)
        changed = zero.clone()
        changed[:, 0] = 1.0
        with torch.no_grad():
            baseline = model(feature, zero)
            altered = model(feature, changed)
        self.assertFalse(torch.equal(baseline["gru_output"][:, 0], altered["gru_output"][:, 0]))
        torch.testing.assert_close(
            baseline["gru_output"][:, 1], altered["gru_output"][:, 1], rtol=0, atol=0
        )
        torch.testing.assert_close(
            baseline["next_hidden"][:, 1], altered["next_hidden"][:, 1], rtol=0, atol=0
        )

    def test_reset_mask_zeros_hidden_before_step(self):
        model = ActorBackboneV1()
        model.eval()
        feature = _feature_tensor(time=1)
        nonzero = torch.ones(1, 2, 128)
        reset = torch.ones(1, 2, 1, dtype=torch.bool)
        with torch.no_grad():
            reset_output = model(feature, nonzero, reset_mask=reset)
            zero_output = model(feature, torch.zeros_like(nonzero))
        torch.testing.assert_close(reset_output["gru_output"], zero_output["gru_output"])
        torch.testing.assert_close(reset_output["next_hidden"], zero_output["next_hidden"])

    def test_gru_step_zero_reset_uses_single_fused_call(self):
        model = ActorBackboneV1()
        model.eval()
        time = FrozenV1Config.RNN_TIME_STEPS
        gru_input = torch.randn(2, 2, time, 128)
        hidden = torch.randn(2, 2, 128)
        reset = torch.zeros(2, 2, time, dtype=torch.bool)
        reset[0, 1, 0] = True

        sequence = gru_input.reshape(4, time, 128)
        reference_hidden = hidden.reshape(4, 128).unsqueeze(0)
        keep = (~reset.reshape(4, time)[:, 0]).to(sequence.dtype).view(1, 4, 1)
        with torch.no_grad():
            reference_output, reference_next = model.gru(
                sequence, reference_hidden * keep
            )
            with patch.object(model.gru, "forward", wraps=model.gru.forward) as forward:
                output, next_hidden = model._run_gru(gru_input, hidden, reset)
        self.assertEqual(forward.call_count, 1)
        torch.testing.assert_close(
            output, reference_output.reshape(2, 2, time, 128)
        )
        torch.testing.assert_close(
            next_hidden, reference_next.squeeze(0).reshape(2, 2, 128)
        )

    def test_gru_internal_reset_keeps_stepwise_semantics(self):
        model = ActorBackboneV1()
        model.eval()
        time = FrozenV1Config.RNN_TIME_STEPS
        gru_input = torch.randn(1, 2, time, 128)
        hidden = torch.randn(1, 2, 128)
        reset = torch.zeros(1, 2, time, dtype=torch.bool)
        reset[0, 0, 7] = True

        sequence = gru_input.reshape(2, time, 128)
        current_hidden = hidden.reshape(2, 128).unsqueeze(0)
        reference_steps = []
        with torch.no_grad():
            for step in range(time):
                keep = (~reset.reshape(2, time)[:, step]).to(sequence.dtype).view(
                    1, 2, 1
                )
                current_hidden = current_hidden * keep
                step_output, current_hidden = model.gru(
                    sequence[:, step : step + 1], current_hidden
                )
                reference_steps.append(step_output)
            with patch.object(model.gru, "forward", wraps=model.gru.forward) as forward:
                output, next_hidden = model._run_gru(gru_input, hidden, reset)
        self.assertEqual(forward.call_count, time)
        torch.testing.assert_close(
            output, torch.cat(reference_steps, dim=1).reshape(1, 2, time, 128)
        )
        torch.testing.assert_close(
            next_hidden, current_hidden.squeeze(0).reshape(1, 2, 128)
        )

    def test_gru_fused_path_matches_stepwise_gradients(self):
        fast = ActorBackboneV1()
        reference = copy.deepcopy(fast)
        time = FrozenV1Config.RNN_TIME_STEPS
        fast_input = torch.randn(1, 2, time, 128, requires_grad=True)
        reference_input = fast_input.detach().clone().requires_grad_(True)
        fast_hidden = torch.randn(1, 2, 128, requires_grad=True)
        reference_hidden = fast_hidden.detach().clone().requires_grad_(True)
        reset = torch.zeros(1, 2, time, dtype=torch.bool)
        reset[0, 1, 0] = True

        fast_output, fast_next = fast._run_gru(fast_input, fast_hidden, reset)
        sequence = reference_input.reshape(2, time, 128)
        current_hidden = reference_hidden.reshape(2, 128).unsqueeze(0)
        reference_steps = []
        for step in range(time):
            keep = (~reset.reshape(2, time)[:, step]).to(sequence.dtype).view(
                1, 2, 1
            )
            current_hidden = current_hidden * keep
            step_output, current_hidden = reference.gru(
                sequence[:, step : step + 1], current_hidden
            )
            reference_steps.append(step_output)
        reference_output = torch.cat(reference_steps, dim=1).reshape(
            1, 2, time, 128
        )
        reference_next = current_hidden.squeeze(0).reshape(1, 2, 128)

        (fast_output.square().mean() + fast_next.square().mean()).backward()
        (
            reference_output.square().mean() + reference_next.square().mean()
        ).backward()
        torch.testing.assert_close(fast_output, reference_output, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(fast_next, reference_next, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(
            fast_input.grad, reference_input.grad, rtol=1e-5, atol=1e-7
        )
        torch.testing.assert_close(
            fast_hidden.grad, reference_hidden.grad, rtol=1e-5, atol=1e-7
        )
        for fast_parameter, reference_parameter in zip(
            fast.gru.parameters(), reference.gru.parameters()
        ):
            torch.testing.assert_close(
                fast_parameter.grad,
                reference_parameter.grad,
                rtol=1e-5,
                atol=1e-7,
            )

    def test_full_backbone_forward_backward_is_finite(self):
        model = ActorBackboneV1()
        feature = _feature_tensor(time=1)
        hidden = torch.zeros(1, 2, 128)
        output = model(feature, hidden)
        loss = output["policy_feature"].square().mean() + output["team_feature"].abs().mean()
        loss.backward()
        required = (
            model.encoder.image_encoder.conv1.weight.grad,
            model.encoder.hero_encoder.layers[0].linear.weight.grad,
            model.team_pscn.input_projection.linear.weight.grad,
            model.private_pscn.input_projection.linear.weight.grad,
            model.gru.weight_ih_l0.grad,
        )
        for gradient in required:
            self.assertIsNotNone(gradient)
            self.assertTrue(torch.isfinite(gradient).all())


class HeroRoutingV1Test(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(19)

    def test_mixed_hero_batch_shapes_and_deterministic_selection(self):
        router = HeroPolicyRouterV1()
        policy = torch.randn(2, 2, 1, 256)
        targets = torch.randn(2, 2, 1, 39, 64)
        hero_ids = torch.tensor([[[132], [141]], [[193], [132]]])
        output = router(policy, targets, hero_ids)
        for logits, width in zip(output["ordinary_logits"], (13, 25, 42, 42)):
            self.assertEqual(tuple(logits.shape), (2, 2, 1, width))
        self.assertEqual(tuple(output["target_logits"].shape), (2, 2, 1, 13, 39))
        self.assertEqual(tuple(output["target_keys"].shape), (2, 2, 1, 39, 64))
        self.assertEqual(tuple(output["target_values"].shape), (2, 2, 1, 39, 64))

    def test_only_selected_route_receives_policy_gradient(self):
        router = HeroPolicyRouterV1()
        policy = torch.randn(1, 2, 1, 256, requires_grad=True)
        targets = torch.randn(1, 2, 1, 39, 64, requires_grad=True)
        hero_ids = torch.full((1, 2, 1), 132)
        output = router(policy, targets, hero_ids)
        loss = sum(logits.square().mean() for logits in output["ordinary_logits"])
        loss = loss + output["target_logits"].square().mean()
        loss.backward()
        marco_grad = sum(
            parameter.grad.abs().sum().item()
            for parameter in router.routes["marco"].parameters()
            if parameter.grad is not None
        )
        self.assertGreater(marco_grad, 0)
        for route_name in ("diaochan", "kai"):
            route_grad = sum(
                parameter.grad.abs().sum().item()
                for parameter in router.routes[route_name].parameters()
                if parameter.grad is not None
            )
            self.assertEqual(route_grad, 0, route_name)

    def test_invalid_route_and_enemy_slot_fail_closed(self):
        router = HeroPolicyRouterV1()
        with self.assertRaisesRegex(ValueError, "999"):
            router(
                torch.randn(1, 1, 1, 256),
                torch.randn(1, 1, 1, 39, 64),
                torch.tensor([[[999]]]),
            )
        main_context = torch.zeros(1, 1, 1, 32)
        main_context[..., 4] = 1
        with self.assertRaisesRegex(ValueError, "enemy hero slot"):
            router.decode_hero_ids(main_context)

    def test_button_conditioned_target_attention(self):
        router = HeroPolicyRouterV1()
        policy = torch.randn(1, 1, 1, 256)
        targets = torch.randn(1, 1, 1, 39, 64)
        output = router(policy, targets, torch.tensor([[[141]]]))
        logits = output["target_logits"][0, 0, 0]
        self.assertFalse(torch.equal(logits[0], logits[1]))

    def test_actor_network_decodes_main_context_and_value(self):
        model = ActorNetworkV1()
        feature = _feature_tensor(time=1)
        hidden = torch.zeros(1, 2, 128)
        output = model(feature, hidden)
        torch.testing.assert_close(output["hero_ids"], torch.tensor([[[132], [141]]]))
        self.assertEqual(tuple(output["actor_value"].shape), (1, 2, 1, 1))
        self.assertEqual(tuple(output["actor_value_self"].shape), (1, 2, 1, 1))
        self.assertEqual(tuple(output["actor_value_team"].shape), (1, 2, 1, 1))
        self.assertEqual(tuple(output["actor_value_enemy"].shape), (1, 2, 1, 1))
        self.assertEqual(tuple(output["actor_value_components"].shape), (1, 2, 1, 3))
        self.assertEqual(tuple(output["actor_auxiliary_values"].shape), (1, 2, 1, 5))
        expected_value = (
            0.8 * output["actor_value_self"]
            + 0.2 * output["actor_value_team"]
            - output["actor_value_enemy"]
        )
        torch.testing.assert_close(output["actor_value"], expected_value)
        torch.testing.assert_close(
            output["actor_value_team"][:, 0], output["actor_value_team"][:, 1]
        )
        torch.testing.assert_close(
            output["actor_value_enemy"][:, 0], output["actor_value_enemy"][:, 1]
        )
        self.assertEqual(tuple(output["target_logits"].shape), (1, 2, 1, 13, 39))

    def test_hybrid_and_auxiliary_heads_receive_finite_nonzero_gradients(self):
        model = ActorNetworkV1()
        output = model(_feature_tensor(), torch.zeros(1, 2, 128))
        loss = output["actor_value"].sum() + output[
            "actor_auxiliary_values"
        ].sum()
        loss.backward()
        gradients = (
            model.value_head.self_output.weight.grad,
            model.value_head.team_output.weight.grad,
            model.value_head.enemy_output.weight.grad,
            model.value_head.auxiliary_output.weight.grad,
        )
        for gradient in gradients:
            self.assertIsNotNone(gradient)
            self.assertTrue(torch.isfinite(gradient).all())
            self.assertGreater(float(gradient.abs().sum()), 0.0)
        per_channel = model.value_head.auxiliary_output.weight.grad.abs().sum(dim=1)
        self.assertTrue(torch.all(per_channel > 0))

    def test_enemy_branch_uses_enemy_entity_information(self):
        torch.manual_seed(23)
        model = ActorNetworkV1()
        model.eval()
        feature = _feature_tensor()
        changed = feature.clone()
        hero_block = schema_v1.FEATURE_BLOCK_BY_NAME["heroes"]
        hp_index = schema_v1.HERO_FIELD_MAP["hp_ratio"].start
        enemy_slot_offset = hero_block.start + 3 * 393 + hp_index
        changed[:, :, :, enemy_slot_offset] = 0.01
        hidden = torch.zeros(1, 2, 128)
        with torch.no_grad():
            original = model(feature, hidden)
            changed_output = model(changed, hidden)
            original_enemy = original["actor_value_enemy"]
            changed_enemy = changed_output["actor_value_enemy"]
        self.assertFalse(torch.equal(original_enemy, changed_enemy))
        torch.testing.assert_close(changed_enemy[:, 0], changed_enemy[:, 1])
        torch.testing.assert_close(
            original["actor_value_team"],
            changed_output["actor_value_team"],
            rtol=0,
            atol=0,
        )


class CommandNetworkV1Test(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(29)

    def test_complete_command_shapes_fixed_slots_and_position_bank(self):
        model = NetworkV1()
        output = model(_feature_tensor(), torch.zeros(1, 2, 128))
        command = output["command"]
        self.assertEqual(tuple(command["intent_logits"].shape), (1, 1, 11))
        self.assertEqual(tuple(command["target_logits"].shape), (1, 1, 11, 39))
        self.assertEqual(tuple(command["position_logits"].shape), (1, 1, 11, 30))
        self.assertEqual(tuple(command["command_value"].shape), (1, 1, 1))
        self.assertEqual(tuple(command["fixed_hero_context"].shape), (1, 1, 3, 256))
        torch.testing.assert_close(command["fixed_hero_occupancy"], torch.ones(1, 1, 3, dtype=torch.int64))
        torch.testing.assert_close(command["official_ids"], torch.tensor([[193]]))
        bank = command["position_bank"]
        self.assertEqual(tuple(bank["coordinates_norm"].shape), (1, 1, 30, 2))
        self.assertEqual(tuple(bank["valid"].shape), (1, 1, 30))
        self.assertAlmostEqual(bank["coordinates_norm"][0, 0, 1, 0].item(), -18000 / 70000, places=6)
        self.assertAlmostEqual(bank["coordinates_norm"][0, 0, 6, 0].item(), -17876 / 70000, places=6)
        self.assertAlmostEqual(bank["coordinates_norm"][0, 0, 9, 0].item(), -66300 / 70000, places=6)

    def test_command_is_invariant_to_controlled_actor_order(self):
        model = NetworkV1()
        model.eval()
        feature = _feature_tensor()
        hidden = torch.randn(1, 2, 128)
        with torch.no_grad():
            original = model(feature, hidden)["command"]
            swapped = model(feature.flip(1), hidden.flip(1))["command"]
        for name in ("intent_logits", "target_logits", "position_logits", "command_value"):
            torch.testing.assert_close(original[name], swapped[name], rtol=1e-5, atol=1e-6)

    def test_command_heads_and_actor_context_receive_gradient(self):
        model = NetworkV1()
        output = model(_feature_tensor(), torch.zeros(1, 2, 128))
        command = output["command"]
        loss = (
            command["intent_logits"].square().mean()
            + command["target_logits"].square().mean()
            + command["position_logits"].square().mean()
            + command["command_value"].square().mean()
        )
        loss.backward()
        gradients = (
            model.command.intent_head.output.weight.grad,
            model.command.command_target_query.linear.weight.grad,
            model.command.command_target_key.linear.weight.grad,
            model.command.command_position_query.linear.weight.grad,
            model.command.position_bank.key.linear.weight.grad,
            model.command.value_output.weight.grad,
            model.actor.backbone.gru.weight_ih_l0.grad,
        )
        for gradient in gradients:
            self.assertIsNotNone(gradient)
            self.assertTrue(torch.isfinite(gradient).all())

    def test_command_time_sixteen_has_no_extra_recurrent_state(self):
        model = NetworkV1()
        output = model(
            _feature_tensor(time=FrozenV1Config.RNN_TIME_STEPS),
            torch.zeros(1, 2, 128),
        )
        self.assertEqual(tuple(output["command"]["command_feature"].shape), (1, 16, 256))
        self.assertEqual(tuple(output["actor"]["next_hidden"].shape), (1, 2, 128))
        self.assertFalse(any("gru" in name.lower() for name, _ in model.command.named_modules()))


if __name__ == "__main__":
    unittest.main()
