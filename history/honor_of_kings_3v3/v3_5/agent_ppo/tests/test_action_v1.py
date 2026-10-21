#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""Frozen 13x39 button-conditioned ordinary-action tests."""

import unittest

import torch

from agent_ppo.conf.conf import FrozenV1Config
from agent_ppo.model.action_v1 import (
    condition_raw_legal_action_v1,
    evaluate_actor_actions_v1,
    masked_probabilities_v1,
    sample_actor_actions_v1,
    select_button_target_row_v1,
    split_raw_legal_action_v1,
)


def _raw_legal(button=1):
    leading = (1, 2, 1)
    raw = torch.zeros(*leading, FrozenV1Config.RAW_LEGAL_ACTION_DIM)
    offset = 0
    raw[..., offset + button] = 1
    offset += 13
    raw[..., offset + 3] = 1
    offset += 25
    raw[..., offset + 4] = 1
    offset += 42
    raw[..., offset + 5] = 1
    offset += 42
    matrix = raw[..., offset:].reshape(*leading, 13, 39)
    matrix[..., button, 7] = 1
    matrix[..., button, 8] = 1
    matrix[..., (button + 1) % 13, 11] = 1
    return raw


def _network_output(requires_grad=False):
    leading = (1, 2, 1)
    ordinary = [
        torch.randn(*leading, width, requires_grad=requires_grad)
        for width in FrozenV1Config.HERO_ACTION_HEADS[:-1]
    ]
    target = torch.randn(*leading, 13, 39, requires_grad=requires_grad)
    # Illegal high logits must never leak through a mask.
    ordinary[0].data[..., 12] = 1000
    target.data[..., :, 38] = 1000
    return {"ordinary_logits": ordinary, "target_logits": target}


class ActionLegalityV1Test(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(23)

    def test_raw_split_and_conditioning_exact_dimensions(self):
        raw = _raw_legal(button=1)
        parts = split_raw_legal_action_v1(raw)
        self.assertEqual([part.shape[-1] for part in parts[:-1]], [13, 25, 42, 42])
        self.assertEqual(tuple(parts[-1].shape[-2:]), (13, 39))
        button = torch.ones(1, 2, 1, dtype=torch.long)
        conditioned = condition_raw_legal_action_v1(raw, button)
        self.assertEqual(conditioned.shape[-1], 161)
        self.assertEqual(conditioned[..., -39 + 7].sum(), 2)
        self.assertEqual(conditioned[..., -39 + 11].sum(), 0)

    def test_sampling_is_button_conditioned_and_never_uses_illegal_logits(self):
        raw = _raw_legal(button=1)
        sampled = sample_actor_actions_v1(
            _network_output(), raw, deterministic=True
        )
        actions = sampled["actions"]
        self.assertTrue(torch.all(actions[..., 0] == 1))
        self.assertTrue(torch.all(actions[..., 1] == 3))
        self.assertTrue(torch.all(actions[..., 2] == 4))
        self.assertTrue(torch.all(actions[..., 3] == 5))
        self.assertTrue(torch.all((actions[..., 4] == 7) | (actions[..., 4] == 8)))
        self.assertTrue(torch.all(actions[..., 4] != 38))
        self.assertEqual(sampled["probability_vector"].shape[-1], 161)
        self.assertEqual(sampled["conditioned_legal_action"].shape[-1], 161)

    def test_normal_attack_can_select_an_enemy_hero_target(self):
        raw = _raw_legal(button=3)
        matrix = raw[..., 122:].reshape(1, 2, 1, 13, 39)
        matrix[..., 3, :] = 0
        matrix[..., 3, 1] = 1  # Enemy hero slots are 1..3.
        sampled = sample_actor_actions_v1(
            _network_output(), raw, deterministic=True
        )
        self.assertTrue(torch.all(sampled["actions"][..., 0] == 3))
        self.assertTrue(torch.all(sampled["actions"][..., 4] == 1))

    def test_training_evaluation_selects_actual_button_row(self):
        raw = _raw_legal(button=1)
        output = _network_output(requires_grad=True)
        sampled = sample_actor_actions_v1(output, raw, deterministic=True)
        evaluated = evaluate_actor_actions_v1(
            output,
            sampled["conditioned_legal_action"],
            sampled["actions"],
        )
        self.assertEqual(tuple(evaluated["selected_log_probabilities"].shape), (1, 2, 1, 5))
        self.assertEqual(tuple(evaluated["entropy"].shape), (1, 2, 1, 5))
        self.assertTrue(torch.isfinite(evaluated["probability_vector"]).all())

    def test_target_gradient_only_uses_selected_button(self):
        raw = _raw_legal(button=1)
        output = _network_output(requires_grad=True)
        actions = torch.tensor([[[[1, 3, 4, 5, 7]], [[1, 3, 4, 5, 7]]]])
        conditioned = condition_raw_legal_action_v1(raw, actions[..., 0])
        evaluated = evaluate_actor_actions_v1(output, conditioned, actions)
        loss = -evaluated["selected_log_probabilities"][..., 4].sum()
        loss.backward()
        gradient = output["target_logits"].grad
        self.assertGreater(gradient[..., 1, :].abs().sum(), 0)
        self.assertEqual(gradient[..., 0, :].abs().sum(), 0)
        self.assertEqual(gradient[..., 2:, :].abs().sum(), 0)

    def test_all_illegal_rows_fail_closed(self):
        with self.assertRaisesRegex(ValueError, "all-illegal"):
            masked_probabilities_v1(
                torch.zeros(2, 13), torch.zeros(2, 13), "button"
            )

    def test_button_selection_rejects_out_of_range(self):
        value = torch.zeros(1, 13, 39)
        with self.assertRaisesRegex(ValueError, "out of"):
            select_button_target_row_v1(value, torch.tensor([13]))


if __name__ == "__main__":
    unittest.main()
