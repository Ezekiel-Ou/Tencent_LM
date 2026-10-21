#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""Channel-4 collision masks and active unsticking tests."""

import copy
import unittest

import numpy as np

from agent_ppo.conf.conf import FrozenV1Config
from agent_ppo.feature.collision_avoidance_v1 import (
    CollisionAvoidanceV1,
    move_direction_vector,
    obstacle_move_mask,
)
from agent_ppo.tests.test_feature_process_v1 import make_observation


class CollisionDirectionMaskV1Test(unittest.TestCase):
    def test_calibrated_direction_cardinals_match_side_image_axes(self):
        np.testing.assert_allclose(move_direction_vector(1), (-1.0, 0.0), atol=1e-6)
        np.testing.assert_allclose(move_direction_vector(7), (0.0, 1.0), atol=1e-6)
        np.testing.assert_allclose(move_direction_vector(13), (1.0, 0.0), atol=1e-6)
        np.testing.assert_allclose(move_direction_vector(19), (0.0, -1.0), atol=1e-6)

    def test_nearby_channel_four_wall_masks_only_colliding_directions(self):
        obstacle = np.zeros((17, 17), dtype=np.float32)
        obstacle[7:10, 9:12] = 1.0
        official = np.ones(25, dtype=np.float32)
        masked, blocked = obstacle_move_mask(obstacle, official)
        self.assertEqual(masked[13], 0.0)
        self.assertIn(13, blocked)
        self.assertEqual(masked[1], 1.0)
        self.assertEqual(masked[0], 1.0)
        self.assertGreater(masked.sum(), 1.0)

    def test_mask_respects_official_legality_and_never_returns_all_zero(self):
        obstacle = np.ones((17, 17), dtype=np.float32)
        official = np.ones(25, dtype=np.float32)
        official[5] = 0.0
        masked, _blocked = obstacle_move_mask(obstacle, official)
        self.assertEqual(masked[5], 0.0)
        self.assertEqual(masked[0], 1.0)
        self.assertGreater(masked.sum(), 0.0)


class ActiveUnstickV1Test(unittest.TestCase):
    @staticmethod
    def _raw_legal(observation):
        return np.stack(
            [
                np.asarray(info["legal_action"], dtype=np.float32)
                for info in observation["heroes"][:2]
            ]
        )

    @staticmethod
    def _actions(first_button=2, first_move=13):
        return np.asarray(
            [
                [first_button, first_move, 0, 0, 0],
                [0, 0, 0, 0, 0],
            ],
            dtype=np.int64,
        )

    def test_persistent_failed_move_forces_side_escape_before_sampling(self):
        avoidance = CollisionAvoidanceV1(1)
        base = make_observation(frame=100, camp=1)
        runtime_ids = tuple(info["runtime_id"] for info in base["heroes"][:2])

        latest_mask = None
        latest_diagnostics = None
        for frame in (100, 106, 112, 118, 124):
            observation = copy.deepcopy(base)
            observation["frame_state"]["frame_no"] = frame
            latest_mask, latest_diagnostics = avoidance.apply(
                observation,
                observation["heroes"][:2],
                self._raw_legal(observation),
            )
            if frame < 124:
                avoidance.record_actions(
                    frame, runtime_ids, self._actions(first_button=2, first_move=13)
                )

        first = latest_diagnostics[0]
        self.assertTrue(first["escape_activated"])
        self.assertTrue(first["escape_active"])
        self.assertNotEqual(first["escape_move"], 13)
        np.testing.assert_array_equal(
            latest_mask[0, :13],
            np.eye(1, 13, 2, dtype=np.float32)[0],
        )
        move_mask = latest_mask[0, 13:38]
        self.assertEqual(move_mask.sum(), 1.0)
        self.assertEqual(move_mask[first["escape_move"]], 1.0)

    def test_real_progress_cancels_active_escape(self):
        avoidance = CollisionAvoidanceV1(1)
        base = make_observation(frame=100, camp=1)
        runtime_ids = tuple(info["runtime_id"] for info in base["heroes"][:2])
        for frame in (100, 106, 112, 118, 124):
            observation = copy.deepcopy(base)
            observation["frame_state"]["frame_no"] = frame
            avoidance.apply(
                observation,
                observation["heroes"][:2],
                self._raw_legal(observation),
            )
            if frame < 124:
                avoidance.record_actions(frame, runtime_ids, self._actions())

        moved = copy.deepcopy(base)
        moved["frame_state"]["frame_no"] = 130
        moved["frame_state"]["hero_states"][0]["location"]["x"] += 1000
        mask, diagnostics = avoidance.apply(
            moved, moved["heroes"][:2], self._raw_legal(moved)
        )
        self.assertFalse(diagnostics[0]["escape_active"])
        self.assertGreater(mask[0, :13].sum(), 1.0)

    def test_macro_route_stuck_forces_escape_without_counter_delay(self):
        avoidance = CollisionAvoidanceV1(1)
        observation = make_observation(frame=100, camp=1)
        runtime_ids = tuple(info["runtime_id"] for info in observation["heroes"][:2])
        mask, diagnostics = avoidance.apply(
            observation,
            observation["heroes"][:2],
            self._raw_legal(observation),
            force_escape_runtime_ids=(runtime_ids[0],),
        )
        self.assertTrue(diagnostics[0]["macro_stuck_forced"])
        self.assertTrue(diagnostics[0]["escape_activated"])
        self.assertTrue(diagnostics[0]["escape_active"])
        self.assertEqual(diagnostics[0]["escape_source"], "macro")
        self.assertEqual(mask[0, :13].argmax(), 2)

    def test_persistent_macro_signal_does_not_restart_escape_each_step(self):
        avoidance = CollisionAvoidanceV1(1)
        base = make_observation(frame=100, camp=1)
        runtime_ids = tuple(info["runtime_id"] for info in base["heroes"][:2])

        activated_frames = []
        escape_moves = []
        for frame in (100, 106, 112, 118, 124, 130, 136, 142):
            observation = copy.deepcopy(base)
            observation["frame_state"]["frame_no"] = frame
            observation["frame_state"]["hero_states"][0]["location"]["x"] += (
                frame - 100
            ) * 200.0
            _mask, diagnostics = avoidance.apply(
                observation,
                observation["heroes"][:2],
                self._raw_legal(observation),
                force_escape_runtime_ids=(runtime_ids[0],),
            )
            if diagnostics[0]["escape_activated"]:
                activated_frames.append(frame)
            if diagnostics[0]["escape_active"]:
                escape_moves.append(diagnostics[0]["escape_move"])

        self.assertEqual(activated_frames, [100])
        self.assertEqual(len(set(escape_moves)), 1)
        self.assertFalse(diagnostics[0]["macro_stuck_forced"])
        self.assertTrue(diagnostics[0]["macro_stuck_requested"])

        retry = copy.deepcopy(base)
        retry["frame_state"]["frame_no"] = 184
        retry["frame_state"]["hero_states"][0]["location"]["x"] += 16800.0
        _mask, retry_diagnostics = avoidance.apply(
            retry,
            retry["heroes"][:2],
            self._raw_legal(retry),
            force_escape_runtime_ids=(runtime_ids[0],),
        )
        self.assertTrue(retry_diagnostics[0]["macro_stuck_forced"])
        self.assertTrue(retry_diagnostics[0]["escape_activated"])


if __name__ == "__main__":
    unittest.main()
