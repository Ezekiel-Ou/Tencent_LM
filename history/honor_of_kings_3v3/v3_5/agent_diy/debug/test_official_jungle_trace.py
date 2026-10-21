#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

import unittest

from agent_diy.debug.official_jungle_trace import (
    CommandEchoTracker,
    _compact_collaboration_state,
)


class CommandEchoTrackerTest(unittest.TestCase):
    def test_runtime_command_timing_is_preserved_in_trace(self):
        compact = _compact_collaboration_state(
            {
                "collaboration_state": {
                    "micro_cd": 100,
                    "max_duration": 300,
                    "micro_state": [],
                    "micro_last": [],
                }
            }
        )
        self.assertEqual(compact["micro_cd"], 100)
        self.assertEqual(compact["max_duration"], 300)

    def test_set_position_acceptance_requires_matching_state_and_last_send(self):
        tracker = CommandEchoTracker()
        submitted = tracker.update(
            frame=56,
            command=[2, 0, -25530, 23990],
            collaboration_state={"active": [], "last_send": []},
        )
        self.assertEqual(submitted["status"], "submitted")
        accepted = tracker.update(
            frame=62,
            command=[0, 0, 0, 0],
            collaboration_state={
                "active": [
                    {
                        "target_runtime_id": 0,
                        "target_position": {
                            "x": -25530,
                            "z": 23990,
                        },
                    }
                ],
                "last_send": [{"frame": 56}],
            },
        )
        self.assertEqual(accepted["status"], "accepted")
        self.assertEqual(tracker.summary()["wire_accepted_count"], 1)

    def test_set_target_without_echo_is_rejected_after_grace(self):
        tracker = CommandEchoTracker()
        tracker.update(
            frame=100,
            command=[1, 501, 0, 0],
            collaboration_state={"active": [], "last_send": []},
        )
        pending = tracker.update(
            frame=124,
            command=[0, 0, 0, 0],
            collaboration_state={"active": [], "last_send": []},
        )
        self.assertEqual(pending["status"], "pending")
        rejected = tracker.update(
            frame=130,
            command=[0, 0, 0, 0],
            collaboration_state={"active": [], "last_send": []},
        )
        self.assertEqual(rejected["status"], "rejected")
        self.assertEqual(tracker.summary()["wire_rejected_count"], 1)

    def test_unset_acceptance_requires_previous_command_to_disappear(self):
        tracker = CommandEchoTracker()
        tracker.update(
            frame=56,
            command=[2, 0, -25530, 23990],
            collaboration_state={"active": [], "last_send": []},
        )
        active_state = {
            "active": [
                {
                    "target_runtime_id": 0,
                    "target_position": {
                        "x": -25530,
                        "z": 23990,
                    },
                }
            ],
            "last_send": [{"frame": 56}],
        }
        tracker.update(
            frame=62,
            command=[0, 0, 0, 0],
            collaboration_state=active_state,
        )
        submitted = tracker.update(
            frame=100,
            command=[3, 0, 0, 0],
            collaboration_state=active_state,
        )
        self.assertEqual(submitted["status"], "submitted")
        self.assertEqual(submitted["action_type"], 3)

        accepted = tracker.update(
            frame=106,
            command=[0, 0, 0, 0],
            collaboration_state={"active": [], "last_send": []},
        )
        self.assertEqual(accepted["status"], "accepted")
        summary = tracker.summary()
        self.assertEqual(summary["wire_accepted_count"], 2)
        self.assertEqual(summary["wire_unset_accepted_count"], 1)
        self.assertEqual(summary["wire_unset_rejected_count"], 0)


if __name__ == "__main__":
    unittest.main()
