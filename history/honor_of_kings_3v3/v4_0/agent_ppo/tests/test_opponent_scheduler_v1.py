#!/usr/bin/env python3
"""Tests for the exact 9:1 next-stage opponent curriculum."""

import unittest

from agent_ppo.workflow.opponent_scheduler_v1 import OpponentMixSchedulerV1


class OpponentMixSchedulerV1Test(unittest.TestCase):
    def test_every_block_is_exactly_nine_to_one(self):
        scheduler = OpponentMixSchedulerV1(
            anchor_model_id=123456,
            selfplay_games_per_block=9,
            anchor_games_per_block=1,
            seed=7,
        )
        selections = [scheduler.next_training_opponent() for _ in range(40)]
        for start in range(0, len(selections), 10):
            block = selections[start : start + 10]
            self.assertEqual(block.count("selfplay"), 9)
            self.assertEqual(block.count("123456"), 1)

    def test_monitor_metrics_report_actual_selection_ratio(self):
        scheduler = OpponentMixSchedulerV1(123456, seed=3)
        for _ in range(10):
            scheduler.next_training_opponent()
        metrics = scheduler.monitor_metrics()
        self.assertEqual(metrics["opponent_selfplay_episode_count"], 9.0)
        self.assertEqual(metrics["opponent_anchor_episode_count"], 1.0)
        self.assertAlmostEqual(metrics["opponent_anchor_episode_ratio"], 0.1)

    def test_invalid_anchor_fails_fast(self):
        with self.assertRaisesRegex(ValueError, "positive platform model ID"):
            OpponentMixSchedulerV1(0)

    def test_evaluation_alternates_commonai_and_anchor(self):
        scheduler = OpponentMixSchedulerV1(123456, seed=11)
        opponents = [scheduler.next_evaluation_opponent() for _ in range(6)]
        self.assertEqual(
            opponents,
            ["common_ai", "123456", "common_ai", "123456", "common_ai", "123456"],
        )
        metrics = scheduler.monitor_metrics()
        self.assertEqual(metrics["eval_commonai_episode_count"], 3.0)
        self.assertEqual(metrics["eval_anchor_episode_count"], 3.0)

    def test_evaluation_does_not_consume_training_block(self):
        scheduler = OpponentMixSchedulerV1(123456, seed=19)
        for _ in range(7):
            scheduler.next_evaluation_opponent()
        training = [scheduler.next_training_opponent() for _ in range(10)]
        self.assertEqual(training.count("selfplay"), 9)
        self.assertEqual(training.count("123456"), 1)


if __name__ == "__main__":
    unittest.main()
