#!/usr/bin/env python3
"""Training workflow boundary regression tests."""

import unittest

from agent_ppo.feature.definition_v1 import accumulate_total_reward_v1


class WorkflowV1Test(unittest.TestCase):
    def test_reward_accumulator_accepts_hero_first_seen_after_initial_frame(self):
        totals = {}

        accumulate_total_reward_v1(
            totals,
            {"config_id": 132, "reward": [0.25]},
        )
        accumulate_total_reward_v1(
            totals,
            {"config_id": 193, "reward": [0.5]},
        )
        accumulate_total_reward_v1(
            totals,
            {"config_id": 132, "reward": [-0.1]},
        )

        self.assertAlmostEqual(totals[132], 0.15)
        self.assertAlmostEqual(totals[193], 0.5)

if __name__ == "__main__":
    unittest.main()
