#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""Opening lane deployment mask tests."""

import unittest

import numpy as np

from agent_ppo.conf.conf import FrozenV1Config
from agent_ppo.feature.collision_avoidance_v1 import MOVE_BUTTON, MOVE_HEAD_OFFSET
from agent_ppo.feature.lane_opening_v1 import LaneOpeningLegalizerV1


class LaneOpeningLegalizerV1Test(unittest.TestCase):
    @staticmethod
    def _raw_legal():
        result = np.ones(
            (
                FrozenV1Config.CONTROLLED_ACTOR_COUNT,
                FrozenV1Config.RAW_LEGAL_ACTION_DIM,
            ),
            dtype=np.float32,
        )
        return result

    @staticmethod
    def _plans():
        return (
            {
                "runtime_id": 11,
                "hard_mask_active": True,
                "phase": "base_exit",
                "interrupt": "none",
                "waypoint_dx": 20000.0,
                "waypoint_dz": 0.0,
            },
            {
                "runtime_id": 12,
                "hard_mask_active": True,
                "phase": "corridor",
                "interrupt": "none",
                "waypoint_dx": 15000.0,
                "waypoint_dz": -5000.0,
            },
        )

    def test_opening_is_applied_before_sampling_as_a_real_mask(self):
        legalizer = LaneOpeningLegalizerV1()
        result, diagnostics = legalizer.apply(
            self._raw_legal(),
            (132, 141),
            self._plans(),
            ({"runtime_id": 11}, {"runtime_id": 12}),
        )
        for actor_index in range(2):
            button = result[actor_index, :13]
            move = result[actor_index, MOVE_HEAD_OFFSET : MOVE_HEAD_OFFSET + 25]
            self.assertEqual(button.sum(), 1)
            self.assertEqual(button[MOVE_BUTTON], 1)
            self.assertGreaterEqual(move.sum(), 1)
            self.assertLessEqual(move.sum(), 3)
            self.assertTrue(diagnostics[actor_index]["opening_mask_active"])

    def test_collision_escape_has_precedence_over_lane_route(self):
        raw = self._raw_legal()
        raw[0, :13] = 0
        raw[0, MOVE_BUTTON] = 1
        raw[0, MOVE_HEAD_OFFSET : MOVE_HEAD_OFFSET + 25] = 0
        raw[0, MOVE_HEAD_OFFSET + 7] = 1
        legalizer = LaneOpeningLegalizerV1()
        result, diagnostics = legalizer.apply(
            raw,
            (132, 141),
            self._plans(),
            (
                {"runtime_id": 11, "escape_active": True},
                {"runtime_id": 12, "escape_active": False},
            ),
        )
        np.testing.assert_array_equal(result[0], raw[0])
        self.assertEqual(diagnostics[0]["reason"], "collision_escape")

    def test_first_wave_combat_forces_common_attack_and_locked_target(self):
        raw = self._raw_legal()
        plans = list(self._plans())
        plans[0] = dict(
            plans[0],
            phase="engage_wave",
            target_slot=31,
        )
        legalizer = LaneOpeningLegalizerV1()
        result, diagnostics = legalizer.apply(
            raw,
            (132, 141),
            tuple(plans),
            ({"runtime_id": 11}, {"runtime_id": 12}),
        )
        button = result[0, :13]
        self.assertEqual(button.sum(), 1)
        self.assertEqual(button[3], 1)
        target_row = result[0, 122 + 3 * 39 : 122 + 4 * 39]
        self.assertEqual(target_row.sum(), 1)
        self.assertEqual(target_row[31], 1)
        self.assertEqual(diagnostics[0]["reason"], "first_wave_attack")
        self.assertEqual(legalizer.attack_application_count, 1)

    def test_jungle_hero_is_never_constrained(self):
        result, diagnostics = LaneOpeningLegalizerV1().apply(
            self._raw_legal(),
            (193, 141),
            self._plans(),
            ({"runtime_id": 11}, {"runtime_id": 12}),
        )
        np.testing.assert_array_equal(result[0], self._raw_legal()[0])
        self.assertFalse(diagnostics[0]["opening_mask_active"])


if __name__ == "__main__":
    unittest.main()
