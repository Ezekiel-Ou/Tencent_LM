#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

import unittest

import numpy as np
import torch

from agent_ppo.conf.conf import FrozenV1Config
from agent_ppo.feature.flash_legalizer_v1 import (
    FLASH_BUTTON,
    FlashLegalizerV1,
)
from agent_ppo.model.action_v1 import (
    evaluate_actor_actions_v1,
    sample_actor_actions_v1,
)


def _raw_legal():
    return np.ones(
        (
            FrozenV1Config.CONTROLLED_ACTOR_COUNT,
            FrozenV1Config.RAW_LEGAL_ACTION_DIM,
        ),
        dtype=np.float32,
    )


class FlashLegalizerV1Test(unittest.TestCase):
    def test_flash_uses_official_talent_button(self):
        self.assertEqual(FLASH_BUTTON, 8)

    def test_default_denies_flash_but_preserves_non_flash_slot(self):
        plans = (
            {
                "runtime_id": 1,
                "flash_gate_active": True,
                "allow": False,
                "reason": "no_threat",
            },
            {
                "runtime_id": 2,
                "flash_gate_active": False,
                "allow": False,
                "reason": "not_flash",
            },
        )
        legalizer = FlashLegalizerV1()
        legal, masks, diagnostics = legalizer.apply(_raw_legal(), plans)
        self.assertEqual(legal[0, FLASH_BUTTON], 0.0)
        self.assertEqual(legal[0, 7], 1.0)
        self.assertEqual(legal[1, FLASH_BUTTON], 1.0)
        self.assertEqual(masks.shape, (2, 2, 42))
        self.assertFalse(diagnostics[0]["allowed"])
        self.assertEqual(legalizer.official_legal_count, 1)
        self.assertEqual(legalizer.blocked_count, 1)

    def test_allowed_flash_gets_one_safe_offset_pair(self):
        obstacle = np.zeros((17, 17), dtype=np.float32)
        plans = (
            {
                "runtime_id": 1,
                "flash_gate_active": True,
                "allow": True,
                "mode": "chase",
                "reason": "low_hp_enemy_chase",
                "direction_dx": 1.0,
                "direction_dz": 0.0,
                "obstacle": obstacle,
            },
            {
                "runtime_id": 2,
                "flash_gate_active": False,
                "allow": False,
                "reason": "not_flash",
            },
        )
        legalizer = FlashLegalizerV1()
        legal, masks, diagnostics = legalizer.apply(_raw_legal(), plans)
        self.assertEqual(legal[0, FLASH_BUTTON], 1.0)
        self.assertEqual(legal[0, 7], 1.0)
        self.assertEqual(masks[0, 0].sum(), 1.0)
        self.assertEqual(masks[0, 1].sum(), 1.0)
        self.assertTrue(diagnostics[0]["allowed"])
        self.assertEqual(legalizer.gate_open_count, 1)
        self.assertEqual(legalizer.chase_count, 1)

    def test_sampler_applies_offsets_only_when_flash_selected(self):
        shape = (1, 2, 1)
        ordinary = [
            torch.full((*shape, width), -10.0)
            for width in FrozenV1Config.HERO_ACTION_HEADS[:-1]
        ]
        ordinary[0][..., FLASH_BUTTON] = 10.0
        output = {
            "ordinary_logits": ordinary,
            "target_logits": torch.zeros((*shape, 13, 39)),
        }
        raw = torch.ones((*shape, FrozenV1Config.RAW_LEGAL_ACTION_DIM))
        masks = torch.zeros((*shape, 2, 42))
        masks[..., 0, 28] = 1.0
        masks[..., 1, 21] = 1.0
        sampled = sample_actor_actions_v1(
            output, raw, deterministic=True, flash_offset_masks=masks
        )
        self.assertTrue(
            torch.all(sampled["actions"][..., 0] == FLASH_BUTTON)
        )
        self.assertTrue(torch.all(sampled["actions"][..., 2] == 28))
        self.assertTrue(torch.all(sampled["actions"][..., 3] == 21))
        evaluated = evaluate_actor_actions_v1(
            output,
            sampled["conditioned_legal_action"],
            sampled["actions"],
        )
        torch.testing.assert_close(
            evaluated["selected_log_probabilities"],
            sampled["selected_log_probabilities"],
        )

    def test_sampler_does_not_apply_flash_offsets_to_skill_four(self):
        shape = (1, 2, 1)
        ordinary = [
            torch.full((*shape, width), -10.0)
            for width in FrozenV1Config.HERO_ACTION_HEADS[:-1]
        ]
        ordinary[0][..., 7] = 10.0
        output = {
            "ordinary_logits": ordinary,
            "target_logits": torch.zeros((*shape, 13, 39)),
        }
        raw = torch.ones((*shape, FrozenV1Config.RAW_LEGAL_ACTION_DIM))
        masks = torch.zeros((*shape, 2, 42))
        masks[..., 0, 28] = 1.0
        masks[..., 1, 21] = 1.0
        sampled = sample_actor_actions_v1(
            output,
            raw,
            deterministic=True,
            flash_offset_masks=masks,
        )
        self.assertTrue(torch.all(sampled["actions"][..., 0] == 7))
        self.assertTrue(torch.all(sampled["actions"][..., 2] == 0))
        self.assertTrue(torch.all(sampled["actions"][..., 3] == 0))


if __name__ == "__main__":
    unittest.main()
