#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""Focused regression tests for the approved PPO 3.5 behavior contract."""

import copy
from dataclasses import replace
import unittest

import numpy as np

from agent_ppo.conf.conf import FrozenV1Config, V1TrainingConfig
from agent_ppo.feature import schema_v1
from agent_ppo.feature.collaboration_v1 import build_command_legal_masks_v1
from agent_ppo.feature.feature_process import v1_utils as u
from agent_ppo.feature.feature_process.v1_feature_process import FeatureProcessV1
from agent_ppo.feature.safety_legalizer_v1 import SafetyLegalizerV1
from agent_ppo.feature.target_capabilities_v1 import (
    TARGET_CAPABILITIES_V1,
    monster_set_target_allowed,
)
from agent_ppo.tests.test_feature_process_v1 import (
    _block,
    _monster,
    _raw_position,
    _soldier,
    make_observation,
)


def _set_spawn_timer(observation, config_id, relation, timer):
    anchor = u.fixed_monster_anchor(
        config_id, relation, 1
    )
    for item in observation["frame_state"]["monster_spawn_infos"]:
        if (
            int(item["monster_id"]) == config_id
            and abs(float(item["born_pos_x"]) - anchor[0]) < 1.0
            and abs(float(item["born_pos_z"]) - anchor[1]) < 1.0
        ):
            item["spawn_timer"] = timer


class TargetCapabilityPPO35Test(unittest.TestCase):
    def test_ordinary_monster_target_is_single_source_permanently_off(self):
        self.assertFalse(
            TARGET_CAPABILITIES_V1["ordinary_monster"].set_target
        )
        self.assertFalse(monster_set_target_allowed(30))
        self.assertFalse(monster_set_target_allowed(44))
        self.assertFalse(monster_set_target_allowed(59))
        self.assertTrue(monster_set_target_allowed(122))

        observation = make_observation(frame=100)
        for index, (config_id, relation) in enumerate(
            schema_v1.MONSTER_TARGET_CONFIG_AND_SIDE
        ):
            observation["frame_state"]["npc_states"].append(
                _monster(config_id, relation, 1, 500 + index)
            )
            _set_spawn_timer(observation, config_id, relation, 0)
        processor = FeatureProcessV1(1)
        processor.process_feature(observation)
        features = processor.process_feature(observation)
        processor.jungle_route_history["opening_released"] = True
        processor.jungle_route_history["opening_stage_index"] = 3
        released = copy.deepcopy(observation)
        released["frame_state"]["frame_no"] = 106
        processor.process_feature(released)
        masks = build_command_legal_masks_v1(
            processor.command_runtime_context()
        )
        self.assertTrue(
            np.all(masks.target_by_intent[9, 8:28] == 0.0)
        )
        self.assertTrue(
            np.all(masks.target_by_intent[4, 8:27] == 0.0)
        )
        self.assertEqual(masks.target_by_intent[4, 27], 1.0)
        targets = _block(features[0], "targets")
        capability_field = schema_v1.TARGET_FIELD_MAP[
            "collaboration_target_legal"
        ].start
        ordinary_index = schema_v1.MONSTER_TARGET_CONFIG_AND_SIDE.index(
            (30, "own")
        )
        self.assertEqual(
            targets[8 + ordinary_index, capability_field], 0.0
        )
        self.assertEqual(targets[27, capability_field], 1.0)

    def test_positive_ordinary_respawn_timer_never_opens_position(self):
        observation = make_observation(frame=100)
        _set_spawn_timer(observation, 30, "own", 1000)
        processor = FeatureProcessV1(1)
        processor.process_feature(observation)
        masks = build_command_legal_masks_v1(
            processor.command_runtime_context()
        )
        self.assertEqual(masks.position_by_intent[10, 12], 0.0)

    def test_unsafe_active_command_uses_unset_when_cooldown_blocks_recovery(self):
        processor = FeatureProcessV1(1)
        processor.process_feature(make_observation(frame=100))
        context = replace(
            processor.command_runtime_context(),
            state_valid=True,
            official_alive=True,
            cooldown_ready=False,
            current_command_active=True,
            unset_allowed=True,
            unset_reason="unsafe_spawn_boundary",
            jungle_route_active=False,
            lane_route_active=False,
            official_task_active=False,
            spawn_boundary_recovery_active=True,
        )
        masks = build_command_legal_masks_v1(context)
        self.assertEqual(np.flatnonzero(masks.intent).tolist(), [1])
        self.assertFalse(masks.command_is_train)


class SafetyPPO35Test(unittest.TestCase):
    @staticmethod
    def _raw():
        return np.ones(
            (
                FrozenV1Config.CONTROLLED_ACTOR_COUNT,
                FrozenV1Config.RAW_LEGAL_ACTION_DIM,
            ),
            dtype=np.float32,
        )

    def test_spawn_boundary_formula_is_camp_mirrored(self):
        for camp, expected in ((1, 57160.0), (2, 57150.0)):
            observation = make_observation(frame=100, camp=camp)
            processor = FeatureProcessV1(camp)
            processor.process_feature(observation)
            self.assertAlmostEqual(
                processor._enemy_spawn_boundary_x(
                    processor._memo_context
                ),
                expected,
            )

    def test_boundary_blocks_forward_move_target_and_uncertain_displacement(self):
        legalizer = SafetyLegalizerV1()
        plans = (
            {
                "runtime_id": 11,
                "position": (57000.0, 0.0),
                "move_speed": 5000.0,
                "decision_seconds": 0.396,
                "spawn_boundary_x": 57160.0,
                "unsafe_enemy_slots": (1,),
                "tower_critical": False,
            },
            {
                "runtime_id": 12,
                "position": (58000.0, 0.0),
                "move_speed": 5000.0,
                "decision_seconds": 0.396,
                "spawn_boundary_x": 57160.0,
                "unsafe_enemy_slots": (),
                "tower_critical": False,
            },
        )
        result, diagnostics = legalizer.apply(
            self._raw(), (132, 193), plans
        )
        move_start = FrozenV1Config.HERO_ACTION_HEADS[0]
        # Direction 13 is canonical +side_x; direction 1 is -side_x.
        self.assertEqual(result[0, move_start + 13], 0.0)
        self.assertEqual(result[0, move_start + 1], 1.0)
        self.assertEqual(result[1, move_start + 13], 0.0)
        self.assertEqual(result[1, move_start], 1.0)
        self.assertEqual(result[0, 5], 0.0)
        target_start = sum(FrozenV1Config.HERO_ACTION_HEADS[:-1])
        for button in range(13):
            self.assertEqual(
                result[0, target_start + button * 39 + 1], 0.0
            )
        self.assertTrue(all(item["has_legal_move"] for item in diagnostics))

    def test_critical_keeps_stop_lateral_and_retreat(self):
        legalizer = SafetyLegalizerV1()
        plans = (
            {
                "runtime_id": 11,
                "position": (10000.0, 0.0),
                "move_speed": 5000.0,
                "spawn_boundary_x": 57160.0,
                "unsafe_enemy_slots": (),
                "tower_critical": True,
                "critical_tower_position": (18000.0, 0.0),
            },
            {
                "runtime_id": 12,
                "position": (0.0, 0.0),
                "move_speed": 5000.0,
                "spawn_boundary_x": 57160.0,
                "unsafe_enemy_slots": (),
                "tower_critical": False,
            },
        )
        result, _ = legalizer.apply(self._raw(), (141, 193), plans)
        move_start = FrozenV1Config.HERO_ACTION_HEADS[0]
        self.assertEqual(result[0, move_start + 13], 0.0)
        self.assertEqual(result[0, move_start + 1], 1.0)
        self.assertEqual(result[0, move_start], 1.0)
        self.assertGreater(np.count_nonzero(result[0, move_start:move_start + 25]), 1)

    def test_target_removal_cannot_leave_a_sampled_button_all_illegal(self):
        raw = np.zeros(
            (
                FrozenV1Config.CONTROLLED_ACTOR_COUNT,
                FrozenV1Config.RAW_LEGAL_ACTION_DIM,
            ),
            dtype=np.float32,
        )
        target_start = sum(FrozenV1Config.HERO_ACTION_HEADS[:-1])
        for actor_index in range(2):
            raw[actor_index, 13] = 1.0
            raw[actor_index, 38 + 21] = 1.0
            raw[actor_index, 80 + 21] = 1.0
        raw[0, 3] = 1.0
        raw[0, target_start + 3 * 39 + 1] = 1.0
        raw[1, 0] = 1.0
        raw[1, target_start] = 1.0
        plans = (
            {
                "position": (0.0, 0.0),
                "spawn_boundary_x": 57160.0,
                "unsafe_enemy_slots": (1,),
            },
            {
                "position": (0.0, 0.0),
                "spawn_boundary_x": 57160.0,
                "unsafe_enemy_slots": (),
            },
        )
        result, _ = SafetyLegalizerV1().apply(
            raw, (132, 193), plans
        )
        self.assertEqual(result[0, 3], 0.0)
        self.assertEqual(result[0, 0], 1.0)
        self.assertEqual(result[0, target_start], 1.0)


class OfficialTaskPPO35Test(unittest.TestCase):
    def test_official_lane_deer_assist_clears_then_returns_upper(self):
        observation = make_observation(frame=100)
        observation["official_model_info"] = {
            "official": {"runtime_id": 11}
        }
        observation["frame_state"]["npc_states"].extend(
            (
                _monster(42, "own", 1, 542),
                _monster(43, "own", 1, 543),
            )
        )
        _set_spawn_timer(observation, 42, "own", 0)
        _set_spawn_timer(observation, 43, "own", 0)
        processor = FeatureProcessV1(1)
        processor.process_feature(observation)
        plan = processor._memo_context["lane_macro"][132]
        self.assertTrue(plan["deer_assist_active"])
        self.assertEqual(
            plan["command_waypoint"],
            u.side_position(
                u.fixed_monster_anchor(42, "own", 1), 1
            ),
        )
        masks = build_command_legal_masks_v1(
            processor.command_runtime_context()
        )
        self.assertEqual(np.flatnonzero(masks.intent).tolist(), [5])
        self.assertFalse(masks.command_is_train)

        cleared = copy.deepcopy(observation)
        cleared["frame_state"]["frame_no"] = 106
        cleared["frame_state"]["npc_states"] = [
            actor
            for actor in cleared["frame_state"]["npc_states"]
            if actor.get("runtime_id") not in (542, 543)
        ]
        official = next(
            actor
            for actor in cleared["frame_state"]["hero_states"]
            if actor["runtime_id"] == 11
        )
        deer_anchor = u.fixed_monster_anchor(42, "own", 1)
        official["location"] = {
            "x": deer_anchor[0],
            "y": 0,
            "z": deer_anchor[1],
        }
        _set_spawn_timer(cleared, 42, "own", 120000)
        _set_spawn_timer(cleared, 43, "own", 120000)
        processor.process_feature(cleared)
        plan = processor._memo_context["lane_macro"][132]
        self.assertEqual(
            plan["deer_assist_end_reason"], "authoritative_clear"
        )
        self.assertTrue(plan["deer_return_active"])
        self.assertEqual(plan["command_waypoint"], (-30000.0, 3150.0))
        masks = build_command_legal_masks_v1(
            processor.command_runtime_context()
        )
        self.assertEqual(masks.intent[1], 0.0)
        self.assertEqual(np.flatnonzero(masks.intent).tolist(), [5])

    def test_learned_farm_option_is_held_and_rewards_authoritative_completion(self):
        first = make_observation(frame=100)
        first["frame_state"]["npc_states"].append(
            _monster(30, "own", 1, 530)
        )
        _set_spawn_timer(first, 30, "own", 0)
        processor = FeatureProcessV1(1)
        processor.process_feature(first)
        processor.jungle_route_history["opening_released"] = True
        processor.jungle_route_history["opening_stage_index"] = 3

        decision = copy.deepcopy(first)
        decision["frame_state"]["frame_no"] = 106
        processor.process_feature(decision)
        context = processor.command_runtime_context()
        masks = build_command_legal_masks_v1(context)
        self.assertEqual(masks.position_by_intent[10, 12], 1.0)
        raw = context.position_raw_coordinates[12]
        processor.record_collaboration_command(
            106,
            (2, 0, int(raw[0]), int(raw[1])),
            intent=10,
            is_train=True,
            position_slot=12,
        )
        self.assertIsNotNone(processor.official_task_state)

        engaged = copy.deepcopy(decision)
        engaged["frame_state"]["frame_no"] = 112
        official = next(
            actor
            for actor in engaged["frame_state"]["hero_states"]
            if actor["runtime_id"] == 13
        )
        official["attack_target"] = 530
        official["hit_target_info"] = [
            {"hit_target": 530, "slot_type": "SLOT_SKILL_0"}
        ]
        engaged["frame_state"]["collaboration_state"]["micro_last"][0][
            "frame_no"
        ] = 106
        engaged["frame_state"]["collaboration_state"]["micro_state"] = [
            {
                "sender": 11,
                "receiver": 13,
                "target_runtime_id": 0,
                "target_position": {
                    "x": raw[0],
                    "y": 0,
                    "z": raw[1],
                },
                "end_frame": 406,
            }
        ]
        processor.process_feature(engaged)
        self.assertTrue(processor.official_task_state["engaged"])
        held = build_command_legal_masks_v1(
            processor.command_runtime_context()
        )
        self.assertFalse(held.command_is_train)

        completed = copy.deepcopy(engaged)
        completed["frame_state"]["frame_no"] = 118
        completed["frame_state"]["npc_states"] = [
            actor
            for actor in completed["frame_state"]["npc_states"]
            if actor.get("runtime_id") != 530
        ]
        _set_spawn_timer(completed, 30, "own", 60000)
        processor.process_feature(completed)
        behavior = processor.consume_behavior_shaping(118)
        self.assertIsNone(processor.official_task_state)
        self.assertAlmostEqual(
            behavior["command_farm_camp_reward"], 0.03
        )
        self.assertAlmostEqual(
            processor.consume_behavior_shaping(118)[
                "command_farm_camp_reward"
            ],
            0.0,
        )

    def test_farm_task_recovery_then_lifecycle_lock(self):
        first = make_observation(frame=100)
        first["frame_state"]["npc_states"].append(
            _monster(30, "own", 1, 530)
        )
        _set_spawn_timer(first, 30, "own", 0)
        processor = FeatureProcessV1(1)
        processor.process_feature(first)
        processor.jungle_route_history["opening_released"] = True
        processor.jungle_route_history["opening_stage_index"] = 3
        decision = copy.deepcopy(first)
        decision["frame_state"]["frame_no"] = 106
        processor.process_feature(decision)
        self.assertTrue(processor._start_official_farm_task(106, 12))

        recovery = copy.deepcopy(decision)
        recovery["frame_state"]["frame_no"] = (
            106
            + V1TrainingConfig.COMMAND_TASK_NAVIGATION_WINDOW_FRAMES
        )
        processor.process_feature(recovery)
        task = processor.official_task_state
        self.assertIsNotNone(task)
        self.assertTrue(task["force_refresh"])
        self.assertEqual(task["navigation_recovery_count"], 1)
        masks = build_command_legal_masks_v1(
            processor.command_runtime_context()
        )
        self.assertFalse(masks.command_is_train)
        self.assertEqual(masks.position_by_intent[10, 12], 1.0)

        failed = copy.deepcopy(recovery)
        failed["frame_state"]["frame_no"] += (
            V1TrainingConfig.COMMAND_TASK_NAVIGATION_WINDOW_FRAMES
        )
        processor.process_feature(failed)
        self.assertIsNone(processor.official_task_state)
        locked = build_command_legal_masks_v1(
            processor.command_runtime_context()
        )
        self.assertEqual(locked.position_by_intent[10, 12], 0.0)

        anchor = tuple(u.fixed_monster_anchor(30, "own", 1))
        processor.monster_camp_history[anchor]["lifecycle_id"] += 1
        unlocked = build_command_legal_masks_v1(
            processor.command_runtime_context()
        )
        self.assertEqual(unlocked.position_by_intent[10, 12], 1.0)

    def test_farm_task_persistence_gets_one_anchor_recovery(self):
        first = make_observation(frame=100)
        first["frame_state"]["npc_states"].append(
            _monster(30, "own", 1, 530)
        )
        _set_spawn_timer(first, 30, "own", 0)
        processor = FeatureProcessV1(1)
        processor.process_feature(first)
        processor.jungle_route_history["opening_released"] = True
        processor.jungle_route_history["opening_stage_index"] = 3
        self.assertTrue(processor._start_official_farm_task(100, 12))
        task = processor.official_task_state
        task["navigation_response"] = True
        task["engaged"] = True
        task["phase"] = "engaged"
        task["last_consistent_frame"] = 100

        drift = copy.deepcopy(first)
        drift["frame_state"]["frame_no"] = (
            100
            + V1TrainingConfig.COMMAND_TASK_PERSISTENCE_WINDOW_FRAMES
        )
        official = next(
            actor
            for actor in drift["frame_state"]["hero_states"]
            if actor["runtime_id"] == 13
        )
        official["attack_target"] = 401
        processor.process_feature(drift)
        self.assertEqual(
            processor.official_task_state[
                "persistence_recovery_count"
            ],
            1,
        )
        self.assertTrue(processor.official_task_state["force_refresh"])

        failed = copy.deepcopy(drift)
        failed["frame_state"]["frame_no"] += (
            V1TrainingConfig.COMMAND_TASK_PERSISTENCE_WINDOW_FRAMES
        )
        processor.process_feature(failed)
        self.assertIsNone(processor.official_task_state)
        self.assertEqual(
            processor.official_task_counters[
                "task_end_persistence_failure"
            ],
            1.0,
        )

    def test_farm_task_engagement_window_starts_at_camp_arrival(self):
        first = make_observation(frame=100)
        first["frame_state"]["npc_states"].append(
            _monster(30, "own", 1, 530)
        )
        _set_spawn_timer(first, 30, "own", 0)
        processor = FeatureProcessV1(1)
        processor.process_feature(first)
        processor.jungle_route_history["opening_released"] = True
        processor.jungle_route_history["opening_stage_index"] = 3

        decision = copy.deepcopy(first)
        decision["frame_state"]["frame_no"] = 106
        processor.process_feature(decision)
        self.assertTrue(processor._start_official_farm_task(106, 12))
        task = processor.official_task_state
        goal = u.side_position(task["anchor"], 1)
        official = next(
            actor
            for actor in decision["frame_state"]["hero_states"]
            if actor["runtime_id"] == 13
        )
        start = (
            float(official["location"]["x"]),
            float(official["location"]["z"] - u.LANE_CENTER_Z),
        )
        initial_distance = max(1.0, float(task["initial_distance"]))
        progress = 2000.0
        progressed_side = (
            start[0] + (goal[0] - start[0]) * progress / initial_distance,
            start[1] + (goal[1] - start[1]) * progress / initial_distance,
        )

        navigated = copy.deepcopy(decision)
        navigated["frame_state"]["frame_no"] = 180
        navigated_official = next(
            actor
            for actor in navigated["frame_state"]["hero_states"]
            if actor["runtime_id"] == 13
        )
        navigated_official["location"] = _raw_position(
            progressed_side[0], progressed_side[1], 1
        )
        processor.process_feature(navigated)
        task = processor.official_task_state
        self.assertTrue(task["navigation_response"])
        self.assertEqual(task["navigation_response_frame"], 180)
        self.assertIsNone(task["camp_arrival_frame"])
        self.assertEqual(task["phase_frame"], 106)

        arrived = copy.deepcopy(navigated)
        arrived["frame_state"]["frame_no"] = 258
        arrived_official = next(
            actor
            for actor in arrived["frame_state"]["hero_states"]
            if actor["runtime_id"] == 13
        )
        arrived_official["location"] = _raw_position(
            goal[0], goal[1], 1
        )
        processor.process_feature(arrived)
        task = processor.official_task_state
        self.assertEqual(task["camp_arrival_frame"], 258)
        self.assertEqual(task["engagement_recovery_count"], 0)

        before_timeout = copy.deepcopy(arrived)
        before_timeout["frame_state"]["frame_no"] = (
            258
            + V1TrainingConfig.COMMAND_TASK_ENGAGEMENT_WINDOW_FRAMES
            - 1
        )
        processor.process_feature(before_timeout)
        self.assertEqual(
            processor.official_task_state["engagement_recovery_count"],
            0,
        )

        at_timeout = copy.deepcopy(arrived)
        at_timeout["frame_state"]["frame_no"] = (
            258
            + V1TrainingConfig.COMMAND_TASK_ENGAGEMENT_WINDOW_FRAMES
        )
        processor.process_feature(at_timeout)
        task = processor.official_task_state
        self.assertEqual(task["engagement_recovery_count"], 1)
        self.assertIsNone(task["camp_arrival_frame"])
        self.assertEqual(task["recovery_start_frame"], at_timeout["frame_state"]["frame_no"])

        deadline = copy.deepcopy(arrived)
        deadline["frame_state"]["frame_no"] = (
            106 + V1TrainingConfig.COMMAND_TASK_HARD_DEADLINE_FRAMES
        )
        processor.process_feature(deadline)
        self.assertIsNone(processor.official_task_state)
        self.assertEqual(
            processor.official_task_counters[
                "last_task_duration_frames"
            ],
            float(V1TrainingConfig.COMMAND_TASK_HARD_DEADLINE_FRAMES),
        )
        self.assertEqual(
            processor.official_task_counters[
                "task_end_engagement_nonresponse"
            ],
            1.0,
        )
        self.assertEqual(
            processor.official_task_counters[
                "task_end_persistence_failure"
            ],
            0.0,
        )

    def test_farm_task_navigation_stall_after_response_recovers_then_fails(self):
        first = make_observation(frame=100)
        first["frame_state"]["npc_states"].append(
            _monster(30, "own", 1, 530)
        )
        _set_spawn_timer(first, 30, "own", 0)
        processor = FeatureProcessV1(1)
        processor.process_feature(first)
        processor.jungle_route_history["opening_released"] = True
        processor.jungle_route_history["opening_stage_index"] = 3

        decision = copy.deepcopy(first)
        decision["frame_state"]["frame_no"] = 106
        processor.process_feature(decision)
        self.assertTrue(processor._start_official_farm_task(106, 12))
        task = processor.official_task_state
        goal = u.side_position(task["anchor"], 1)
        official = next(
            actor
            for actor in decision["frame_state"]["hero_states"]
            if actor["runtime_id"] == 13
        )
        start = (
            float(official["location"]["x"]),
            float(official["location"]["z"] - u.LANE_CENTER_Z),
        )
        initial_distance = max(1.0, float(task["initial_distance"]))
        progress = 2000.0
        progressed_side = (
            start[0] + (goal[0] - start[0]) * progress / initial_distance,
            start[1] + (goal[1] - start[1]) * progress / initial_distance,
        )

        navigated = copy.deepcopy(decision)
        navigated["frame_state"]["frame_no"] = 180
        navigated_official = next(
            actor
            for actor in navigated["frame_state"]["hero_states"]
            if actor["runtime_id"] == 13
        )
        navigated_official["location"] = _raw_position(
            progressed_side[0], progressed_side[1], 1
        )
        processor.process_feature(navigated)
        task = processor.official_task_state
        self.assertTrue(task["navigation_response"])
        self.assertEqual(task["last_navigation_progress_frame"], 180)
        self.assertIsNone(task["camp_arrival_frame"])

        stalled = copy.deepcopy(navigated)
        stalled["frame_state"]["frame_no"] = (
            180
            + V1TrainingConfig.COMMAND_TASK_NAVIGATION_WINDOW_FRAMES
        )
        processor.process_feature(stalled)
        task = processor.official_task_state
        self.assertIsNotNone(task)
        self.assertEqual(task["navigation_recovery_count"], 1)
        self.assertTrue(task["navigation_response"])
        self.assertEqual(task["navigation_response_frame"], 180)
        self.assertEqual(
            task["last_navigation_progress_frame"],
            stalled["frame_state"]["frame_no"],
        )

        failed = copy.deepcopy(stalled)
        failed["frame_state"]["frame_no"] += (
            V1TrainingConfig.COMMAND_TASK_NAVIGATION_WINDOW_FRAMES
        )
        processor.process_feature(failed)
        self.assertIsNone(processor.official_task_state)
        self.assertEqual(
            processor.official_task_counters[
                "task_end_navigation_nonresponse"
            ],
            1.0,
        )
        self.assertEqual(
            processor.official_task_counters[
                "task_end_persistence_failure"
            ],
            0.0,
        )


class MacroAndPenaltyPPO35Test(unittest.TestCase):
    def test_three_post_opening_resources_are_simultaneous_policy_candidates(self):
        first = make_observation(frame=100)
        first["official_model_info"] = {
            "official": {"runtime_id": 11}
        }
        processor = FeatureProcessV1(1)
        processor.process_feature(first)
        processor.jungle_route_history["opening_released"] = True
        processor.jungle_route_history["opening_stage_index"] = 6
        own_wolf_anchor = tuple(
            u.fixed_monster_anchor(44, "own", 1)
        )
        processor.monster_camp_history[own_wolf_anchor][
            "clear_count"
        ] = 1
        processor.monster_camp_history[own_wolf_anchor][
            "cleared_once"
        ] = True

        candidates = copy.deepcopy(first)
        candidates["frame_state"]["frame_no"] = 106
        candidates["frame_state"]["hero_states"][0]["buff_state"][
            "buff_skills"
        ].append({"config_id": 60007, "start_time": 1, "times": 1})
        candidates["frame_state"]["npc_states"].extend(
            (
                _monster(44, "own", 1, 610),
                _monster(45, "own", 1, 611),
                _monster(44, "enemy", 1, 620),
                _monster(45, "enemy", 1, 621),
                _monster(59, "center", 1, 630),
            )
        )
        for config_id, relation in (
            (44, "own"),
            (45, "own"),
            (44, "enemy"),
            (45, "enemy"),
            (59, "center"),
        ):
            _set_spawn_timer(candidates, config_id, relation, 0)
        processor.process_feature(candidates)
        macro = processor._memo_context["jungle_macro"]
        self.assertEqual(
            set(macro["high_value_candidate_slots"]),
            {14, 15, 18},
        )
        self.assertIsNone(macro["target_slot"])
        self.assertEqual(macro["goal_node"], 10)

        selected = copy.deepcopy(candidates)
        selected["frame_state"]["frame_no"] = 112
        selected["frame_state"]["hero_states"][2][
            "attack_target"
        ] = 620
        processor.process_feature(selected)
        macro = processor._memo_context["jungle_macro"]
        self.assertEqual(macro["target_slot"], 15)
        self.assertEqual(macro["goal_node"], 3)

    def test_long_resource_transit_exemption_requires_real_progress(self):
        observation = make_observation(frame=100)
        observation["official_model_info"] = {
            "official": {"runtime_id": 11}
        }
        observation["frame_state"]["npc_states"].append(
            _monster(30, "own", 1, 530)
        )
        processor = FeatureProcessV1(1)
        processor.process_feature(observation)
        context = processor._memo_context
        kai = next(
            hero
            for hero in context["friend_heroes"]
            if u.actor_config_id(hero["actor"]) == 193
        )
        base_macro = {
            "valid": True,
            "interrupt": "none",
            "target_slot": 2,
            "opening_rotation_active": False,
            "path_distance": 20000.0,
            "route_progress_confirmed": False,
        }
        context["jungle_macro"] = base_macro
        attendance = {"active": False, "near": False}
        self.assertFalse(
            processor._strategic_wait_active(
                context, 193, kai, attendance
            )
        )
        context["jungle_macro"] = dict(
            base_macro, route_progress_confirmed=True
        )
        self.assertTrue(
            processor._strategic_wait_active(
                context, 193, kai, attendance
            )
        )


class FeatureSemanticsPPO35Test(unittest.TestCase):
    def test_minion_cover_requires_living_minion_inside_actual_range(self):
        for camp in (1, 2):
            outside = make_observation(
                frame=100, camp=camp, include_soldiers=False
            )
            probe = FeatureProcessV1(camp)
            probe.process_feature(outside)
            tower = probe._memo_context["organs"][3]
            tower_raw = u.location_of(tower["actor"])
            soldier = _soldier(
                801, camp, 0.0, 19, 0.0
            )
            soldier["location"] = {
                "x": tower_raw[0] + 9500.0,
                "y": 0,
                "z": tower_raw[1],
            }
            outside["frame_state"]["npc_states"].append(soldier)
            probe = FeatureProcessV1(camp)
            probe.process_feature(outside)
            tower = probe._memo_context["organs"][3]
            self.assertFalse(
                probe._tower_has_reliable_minion_cover(
                    probe._memo_context, tower
                )
            )

            inside = copy.deepcopy(outside)
            inside["frame_state"]["frame_no"] = 106
            inside_soldier = next(
                actor
                for actor in inside["frame_state"]["npc_states"]
                if actor.get("runtime_id") == 801
            )
            inside_soldier["location"]["x"] = tower_raw[0] + 8500.0
            probe = FeatureProcessV1(camp)
            probe.process_feature(inside)
            tower = probe._memo_context["organs"][3]
            self.assertTrue(
                probe._tower_has_reliable_minion_cover(
                    probe._memo_context, tower
                )
            )

            tower["actor"]["attack_range"] = 0
            tower["actor"]["attack_target"] = 801
            self.assertFalse(
                probe._tower_has_reliable_minion_cover(
                    probe._memo_context, tower
                )
            )


class RewardPPO35Test(unittest.TestCase):
    @staticmethod
    def _release_openings(processor):
        for history in processor.lane_opening_history.values():
            history["released"] = True
            history["release_reason"] = "test_release"

    @staticmethod
    def _defense_threat(frame, runtime_id=801, config_id=19):
        observation = make_observation(
            frame=frame, include_soldiers=False
        )
        own_tower = next(
            actor
            for actor in observation["frame_state"]["npc_states"]
            if actor["config_id"] == 66
        )
        soldier = _soldier(
            runtime_id,
            2,
            own_tower["location"]["x"] + 500.0,
            config_id,
            own_tower["location"]["z"] - u.LANE_CENTER_Z,
        )
        soldier["location"] = {
            "x": own_tower["location"]["x"] + 500.0,
            "y": 0,
            "z": own_tower["location"]["z"],
        }
        soldier["attack_target"] = own_tower["runtime_id"]
        observation["frame_state"]["npc_states"].append(soldier)
        return observation, own_tower, soldier

    def test_tower_hits_are_progressive_and_critical_is_exposed_to_mask(self):
        processor = FeatureProcessV1(1)
        penalties = []
        for frame, expected in ((100, -0.01), (106, -0.02), (112, -0.03)):
            observation = make_observation(
                frame=frame, include_soldiers=False
            )
            hero = next(
                actor
                for actor in observation["frame_state"]["hero_states"]
                if actor["runtime_id"] == 11
            )
            tower = next(
                actor
                for actor in observation["frame_state"]["npc_states"]
                if actor["config_id"] == 70
            )
            hero["location"] = _raw_position(
                tower["location"]["x"],
                tower["location"]["z"] - u.LANE_CENTER_Z,
                1,
            )
            hero["hp"] = 0.40 * hero["max_hp"]
            tower["attack_target"] = 11
            hero["take_hurt_infos"] = [
                {
                    "atker": tower["runtime_id"],
                    "hurt_value": 100 + frame,
                    "source_type": "tower",
                }
            ]
            processor.process_feature(observation)
            behavior = processor.consume_behavior_shaping(frame)
            penalties.append(behavior["tower_risk_rewards"][0])
            self.assertEqual(
                processor.safety_runtime_plans()[0]["tower_critical"],
                True,
            )
            self.assertAlmostEqual(penalties[-1], expected)
        self.assertEqual(
            processor.tower_risk_counters["event_start"], 1.0
        )
        self.assertEqual(
            processor.tower_risk_counters["enter_critical"], 1.0
        )

    def test_tower_risk_resets_immediately_when_tower_observation_dies(self):
        processor = FeatureProcessV1(1)
        threatened = make_observation(
            frame=100, include_soldiers=False
        )
        hero = next(
            actor
            for actor in threatened["frame_state"]["hero_states"]
            if actor["runtime_id"] == 11
        )
        tower = next(
            actor
            for actor in threatened["frame_state"]["npc_states"]
            if actor["config_id"] == 70
        )
        hero["location"] = copy.deepcopy(tower["location"])
        tower["attack_target"] = 11
        processor.process_feature(threatened)
        processor.consume_behavior_shaping(100)
        self.assertIn(11, processor.tower_risk_state)

        destroyed = copy.deepcopy(threatened)
        destroyed["frame_state"]["frame_no"] = 106
        destroyed["frame_state"]["npc_states"] = [
            actor
            for actor in destroyed["frame_state"]["npc_states"]
            if actor.get("runtime_id") != tower["runtime_id"]
        ]
        processor.process_feature(destroyed)
        processor.consume_behavior_shaping(106)
        self.assertNotIn(11, processor.tower_risk_state)
        self.assertEqual(
            processor.tower_risk_counters["reset_tower_invalid"], 1.0
        )

    def test_structure_positive_reward_uses_50_50_attribution(self):
        processor = FeatureProcessV1(1)
        baseline = make_observation(frame=100)
        processor.process_feature(baseline)
        processor.consume_behavior_shaping(100)

        damaged = make_observation(frame=106)
        tower = next(
            actor
            for actor in damaged["frame_state"]["npc_states"]
            if actor["config_id"] == 70
        )
        tower["hp"] = 0.9 * tower["max_hp"]
        hero = next(
            actor
            for actor in damaged["frame_state"]["hero_states"]
            if actor["runtime_id"] == 11
        )
        hero["hit_target_info"] = [
            {
                "hit_target": tower["runtime_id"],
                "slot_type": "SLOT_SKILL_0",
            }
        ]
        hero["hurt_hero_info"] = [
            {
                "hurt_target": tower["runtime_id"],
                "hurt": 0.1 * tower["max_hp"],
            }
        ]
        processor.process_feature(damaged)
        behavior = processor.consume_behavior_shaping(106)
        reward = behavior["structure_team_reward"]
        actor_rewards = behavior["structure_actor_rewards"]
        self.assertGreater(reward, 0.0)
        self.assertAlmostEqual(actor_rewards[0], 1.5 * reward)
        self.assertAlmostEqual(actor_rewards[1], 0.5 * reward)
        self.assertAlmostEqual(sum(actor_rewards), 2.0 * reward)
        self.assertAlmostEqual(
            behavior["structure_attribution_coverage"], 1.0
        )

    def test_post_kill_push_uses_towers_and_crystal_but_r04_uses_towers_only(self):
        cases = (
            (70, 2, 0.2),
            (71, 2, 0.2),
            (72, 2, 0.2),
            (66, 1, -0.2),
            (67, 1, -0.2),
            (68, 1, -0.2),
        )
        for structure_config, death_camp, expected in cases:
            with self.subTest(structure_config=structure_config):
                processor = FeatureProcessV1(1)
                baseline = make_observation(frame=100)
                processor.process_feature(baseline)
                processor.consume_behavior_shaping(100)

                killed = make_observation(frame=106)
                killed["frame_state"]["frame_action"]["dead_action"] = [
                    {
                        "death": {
                            "runtime_id": 900 + structure_config,
                            "config_id": 132,
                            "camp": death_camp,
                            "actor_type": "ACTOR_HERO",
                            "sub_type": "ACTOR_SUB_NONE",
                        }
                    }
                ]
                processor.process_feature(killed)
                self.assertEqual(
                    processor.consume_behavior_shaping(106)[
                        "post_kill_push_reward"
                    ],
                    0.0,
                )

                damaged = make_observation(frame=112)
                structure = next(
                    actor
                    for actor in damaged["frame_state"]["npc_states"]
                    if actor["config_id"] == structure_config
                )
                structure["hp"] = 0.9 * structure["max_hp"]
                processor.process_feature(damaged)
                behavior = processor.consume_behavior_shaping(112)
                self.assertAlmostEqual(
                    behavior["post_kill_push_reward"], expected
                )
                is_crystal = structure_config in (68, 72)
                tower_damage_key = (
                    "own_tower_damage"
                    if death_camp == 1
                    else "enemy_tower_damage"
                )
                push_damage_key = (
                    "own_push_conversion_damage"
                    if death_camp == 1
                    else "enemy_push_conversion_damage"
                )
                self.assertAlmostEqual(
                    behavior[tower_damage_key],
                    0.0 if is_crystal else 0.1,
                )
                self.assertAlmostEqual(
                    behavior[push_damage_key], 0.1
                )
                if is_crystal:
                    self.assertEqual(
                        behavior["structure_team_reward"], 0.0
                    )
                    self.assertEqual(
                        behavior["structure_actor_rewards"],
                        (0.0, 0.0),
                    )
                elif death_camp == 2:
                    self.assertGreater(
                        behavior["structure_team_reward"], 0.0
                    )
                else:
                    self.assertLess(
                        behavior["structure_team_reward"], 0.0
                    )

    def test_crystal_missing_and_repeated_hp_do_not_manufacture_push(self):
        processor = FeatureProcessV1(1)
        baseline = make_observation(frame=100)
        processor.process_feature(baseline)
        processor.consume_behavior_shaping(100)

        killed = make_observation(frame=106)
        killed["frame_state"]["frame_action"]["dead_action"] = [
            {
                "death": {
                    "runtime_id": 921,
                    "config_id": 132,
                    "camp": 2,
                    "actor_type": "ACTOR_HERO",
                    "sub_type": "ACTOR_SUB_NONE",
                }
            }
        ]
        processor.process_feature(killed)
        processor.consume_behavior_shaping(106)

        missing = make_observation(frame=112)
        missing["frame_state"]["npc_states"] = [
            actor
            for actor in missing["frame_state"]["npc_states"]
            if actor.get("config_id") != 72
        ]
        processor.process_feature(missing)
        self.assertEqual(
            processor.consume_behavior_shaping(112)[
                "post_kill_push_reward"
            ],
            0.0,
        )

        repeated = make_observation(frame=118)
        processor.process_feature(repeated)
        self.assertEqual(
            processor.consume_behavior_shaping(118)[
                "post_kill_push_reward"
            ],
            0.0,
        )

        damaged = make_observation(frame=124)
        crystal = next(
            actor
            for actor in damaged["frame_state"]["npc_states"]
            if actor["config_id"] == 72
        )
        crystal["hp"] -= 100
        processor.process_feature(damaged)
        self.assertEqual(
            processor.consume_behavior_shaping(124)[
                "post_kill_push_reward"
            ],
            V1TrainingConfig.POST_KILL_PUSH_REWARD,
        )

        continued = copy.deepcopy(damaged)
        continued["frame_state"]["frame_no"] = 130
        crystal = next(
            actor
            for actor in continued["frame_state"]["npc_states"]
            if actor["config_id"] == 72
        )
        crystal["hp"] -= 100
        processor.process_feature(continued)
        self.assertEqual(
            processor.consume_behavior_shaping(130)[
                "post_kill_push_reward"
            ],
            0.0,
        )

    def test_crystal_death_pays_terminal_and_one_push_conversion(self):
        processor = FeatureProcessV1(1)
        baseline = make_observation(frame=100)
        processor.process_feature(baseline)
        processor.consume_behavior_shaping(100)

        killed = make_observation(frame=106)
        killed["frame_state"]["frame_action"]["dead_action"] = [
            {
                "death": {
                    "runtime_id": 921,
                    "config_id": 132,
                    "camp": 2,
                    "actor_type": "ACTOR_HERO",
                    "sub_type": "ACTOR_SUB_NONE",
                }
            }
        ]
        processor.process_feature(killed)
        processor.consume_behavior_shaping(106)

        destroyed = make_observation(frame=112)
        crystal = next(
            actor
            for actor in destroyed["frame_state"]["npc_states"]
            if actor["config_id"] == 72
        )
        crystal["hp"] = 0
        destroyed["frame_state"]["frame_action"]["dead_action"] = [
            {"death": copy.deepcopy(crystal)}
        ]
        processor.process_feature(destroyed)
        behavior = processor.consume_behavior_shaping(112)
        self.assertEqual(behavior["crystal_outcome_reward"], 8.0)
        self.assertEqual(
            behavior["post_kill_push_reward"],
            V1TrainingConfig.POST_KILL_PUSH_REWARD,
        )
        self.assertEqual(behavior["structure_team_reward"], 0.0)

        repeated = copy.deepcopy(destroyed)
        repeated["frame_state"]["frame_no"] = 118
        processor.process_feature(repeated)
        behavior = processor.consume_behavior_shaping(118)
        self.assertEqual(behavior["crystal_outcome_reward"], 0.0)
        self.assertEqual(behavior["post_kill_push_reward"], 0.0)

    def test_defense_wave_selects_threatened_high_tower_while_outer_lives(self):
        processor = FeatureProcessV1(1)
        baseline = make_observation(frame=100, include_soldiers=False)
        processor.process_feature(baseline)
        processor.consume_behavior_shaping(100)
        self._release_openings(processor)

        threatened = make_observation(frame=106, include_soldiers=False)
        own_high = next(
            actor
            for actor in threatened["frame_state"]["npc_states"]
            if actor["config_id"] == 67
        )
        soldier = _soldier(
            801,
            2,
            own_high["location"]["x"] + 500.0,
            19,
            own_high["location"]["z"] - u.LANE_CENTER_Z,
        )
        soldier["location"] = {
            "x": own_high["location"]["x"] + 500.0,
            "y": 0,
            "z": own_high["location"]["z"],
        }
        soldier["attack_target"] = own_high["runtime_id"]
        threatened["frame_state"]["npc_states"].append(soldier)
        processor.process_feature(threatened)
        processor.consume_behavior_shaping(106)

        self.assertIsNotNone(processor.defense_wave_state)
        self.assertEqual(
            processor.defense_wave_state["structure_config_id"], 67
        )
        self.assertEqual(
            processor.defense_wave_counters["window_structure_high"],
            1.0,
        )

    def test_defense_wave_response_progress_and_completion_are_once(self):
        processor = FeatureProcessV1(1)
        baseline = make_observation(frame=100, include_soldiers=False)
        processor.process_feature(baseline)
        processor.consume_behavior_shaping(100)
        self._release_openings(processor)

        threatened = make_observation(frame=106, include_soldiers=False)
        own_tower = next(
            actor
            for actor in threatened["frame_state"]["npc_states"]
            if actor["config_id"] == 66
        )
        soldier = _soldier(
            801,
            2,
            own_tower["location"]["x"] + 500.0,
            19,
            own_tower["location"]["z"] - u.LANE_CENTER_Z,
        )
        soldier["location"] = {
            "x": own_tower["location"]["x"] + 500.0,
            "y": 0,
            "z": own_tower["location"]["z"],
        }
        soldier["attack_target"] = own_tower["runtime_id"]
        threatened["frame_state"]["npc_states"].append(soldier)
        processor.process_feature(threatened)
        processor.consume_behavior_shaping(106)

        progress = copy.deepcopy(threatened)
        progress["frame_state"]["frame_no"] = 112
        progress_soldier = next(
            actor
            for actor in progress["frame_state"]["npc_states"]
            if actor.get("runtime_id") == 801
        )
        progress_soldier["hp"] = 1200
        hero = next(
            actor
            for actor in progress["frame_state"]["hero_states"]
            if actor["runtime_id"] == 11
        )
        hero["hit_target_info"] = [
            {"hit_target": 801, "slot_type": "SLOT_SKILL_0"}
        ]
        hero["hurt_hero_info"] = [
            {"hurt_target": 801, "hurt": 300}
        ]
        processor.process_feature(progress)
        behavior = processor.consume_behavior_shaping(112)
        self.assertAlmostEqual(
            behavior["defense_wave_rewards"][0], 0.007
        )
        self.assertEqual(behavior["defense_wave_rewards"][1], 0.0)

        cleared = copy.deepcopy(progress)
        cleared["frame_state"]["frame_no"] = 118
        cleared["frame_state"]["npc_states"] = [
            actor
            for actor in cleared["frame_state"]["npc_states"]
            if actor.get("runtime_id") != 801
        ]
        cleared["frame_state"]["frame_action"]["dead_action"] = [
            {"death": progress_soldier, "killer": hero}
        ]
        processor.process_feature(cleared)
        processor.consume_behavior_shaping(118)

        completed = copy.deepcopy(cleared)
        completed["frame_state"]["frame_no"] = 134
        completed["frame_state"]["frame_action"]["dead_action"] = []
        processor.process_feature(completed)
        behavior = processor.consume_behavior_shaping(134)
        self.assertTrue(behavior["defense_wave_completed"])
        for reward in behavior["defense_wave_rewards"]:
            self.assertAlmostEqual(reward, 0.01)
        self.assertFalse(
            processor.consume_behavior_shaping(134)[
                "defense_wave_completed"
            ]
        )

    def test_defense_wave_short_missing_recovers_then_long_missing_invalidates(self):
        processor = FeatureProcessV1(1)
        baseline = make_observation(
            frame=100, include_soldiers=False
        )
        processor.process_feature(baseline)
        processor.consume_behavior_shaping(100)
        self._release_openings(processor)

        threatened, _tower, _soldier_actor = self._defense_threat(106)
        responder = next(
            hero
            for hero in threatened["frame_state"]["hero_states"]
            if hero["runtime_id"] == 11
        )
        responder["hit_target_info"] = [
            {"hit_target": 801, "slot_type": "SLOT_SKILL_0"}
        ]
        processor.process_feature(threatened)
        behavior = processor.consume_behavior_shaping(106)
        self.assertAlmostEqual(
            behavior["defense_wave_rewards"][0],
            V1TrainingConfig.DEFENSE_WAVE_RESPONSE_REWARD,
        )

        missing = copy.deepcopy(threatened)
        missing["frame_state"]["frame_no"] = 112
        missing["frame_state"]["npc_states"] = [
            actor
            for actor in missing["frame_state"]["npc_states"]
            if actor.get("runtime_id") != 801
        ]
        processor.process_feature(missing)
        behavior = processor.consume_behavior_shaping(112)
        self.assertTrue(behavior["defense_wave_active"])
        ledger = processor.defense_wave_state["soldiers"][801]
        self.assertEqual(ledger["missing_since"], 112)

        reappeared = copy.deepcopy(threatened)
        reappeared["frame_state"]["frame_no"] = 118
        processor.process_feature(reappeared)
        behavior = processor.consume_behavior_shaping(118)
        self.assertTrue(behavior["defense_wave_active"])
        self.assertEqual(behavior["defense_wave_rewards"], (0.0, 0.0))
        ledger = processor.defense_wave_state["soldiers"][801]
        self.assertIsNone(ledger["missing_since"])
        self.assertEqual(
            processor.defense_wave_counters["soldier_missing"], 1.0
        )
        self.assertEqual(
            processor.defense_wave_counters["first_response"], 1.0
        )

        missing_again = copy.deepcopy(missing)
        missing_again["frame_state"]["frame_no"] = 124
        processor.process_feature(missing_again)
        processor.consume_behavior_shaping(124)

        invalidated = copy.deepcopy(missing_again)
        invalidated["frame_state"]["frame_no"] = (
            124 + V1TrainingConfig.DEFENSE_WAVE_MISSING_GRACE_FRAMES
        )
        processor.process_feature(invalidated)
        behavior = processor.consume_behavior_shaping(
            invalidated["frame_state"]["frame_no"]
        )
        self.assertTrue(behavior["defense_wave_invalidated"])
        self.assertEqual(
            behavior["defense_wave_invalidation_reason"],
            "observation_invalidated",
        )
        self.assertFalse(behavior["defense_wave_completed"])
        self.assertEqual(behavior["defense_wave_rewards"], (0.0, 0.0))
        self.assertIsNone(processor.defense_wave_state)

        next_wave, _tower, _soldier_actor = self._defense_threat(
            invalidated["frame_state"]["frame_no"] + 6,
            runtime_id=802,
        )
        processor.process_feature(next_wave)
        behavior = processor.consume_behavior_shaping(
            next_wave["frame_state"]["frame_no"]
        )
        self.assertTrue(behavior["defense_wave_active"])
        self.assertIn(802, processor.defense_wave_state["soldiers"])

        late_processor = FeatureProcessV1(1)
        late_processor.process_feature(baseline)
        late_processor.consume_behavior_shaping(100)
        self._release_openings(late_processor)
        threatened, _tower, _soldier_actor = self._defense_threat(106)
        late_processor.process_feature(threatened)
        late_processor.consume_behavior_shaping(106)
        late_processor.process_feature(missing)
        late_processor.consume_behavior_shaping(112)
        late_reappearance = copy.deepcopy(threatened)
        late_reappearance["frame_state"]["frame_no"] = (
            112 + V1TrainingConfig.DEFENSE_WAVE_MISSING_GRACE_FRAMES
        )
        late_processor.process_feature(late_reappearance)
        behavior = late_processor.consume_behavior_shaping(
            late_reappearance["frame_state"]["frame_no"]
        )
        self.assertTrue(behavior["defense_wave_invalidated"])
        self.assertEqual(
            behavior["defense_wave_invalidation_reason"],
            "observation_invalidated",
        )

    def test_defense_wave_runtime_reuse_neutrally_invalidates_window(self):
        processor = FeatureProcessV1(1)
        baseline = make_observation(
            frame=100, include_soldiers=False
        )
        processor.process_feature(baseline)
        processor.consume_behavior_shaping(100)
        self._release_openings(processor)

        threatened, _tower, _soldier_actor = self._defense_threat(106)
        processor.process_feature(threatened)
        processor.consume_behavior_shaping(106)

        missing = copy.deepcopy(threatened)
        missing["frame_state"]["frame_no"] = 112
        missing["frame_state"]["npc_states"] = [
            actor
            for actor in missing["frame_state"]["npc_states"]
            if actor.get("runtime_id") != 801
        ]
        processor.process_feature(missing)
        processor.consume_behavior_shaping(112)

        reused, _tower, _soldier_actor = self._defense_threat(
            118, runtime_id=801, config_id=21
        )
        processor.process_feature(reused)
        behavior = processor.consume_behavior_shaping(118)
        self.assertTrue(behavior["defense_wave_invalidated"])
        self.assertEqual(
            behavior["defense_wave_invalidation_reason"],
            "runtime_reuse",
        )
        self.assertEqual(behavior["defense_wave_rewards"], (0.0, 0.0))
        self.assertIsNone(processor.defense_wave_state)
        self.assertEqual(
            processor.defense_wave_counters[
                "window_invalidated_runtime_reuse"
            ],
            1.0,
        )

        lifecycle_processor = FeatureProcessV1(1)
        lifecycle_processor.process_feature(baseline)
        lifecycle_processor.consume_behavior_shaping(100)
        self._release_openings(lifecycle_processor)
        threatened, _tower, soldier_actor = self._defense_threat(106)
        lifecycle_processor.process_feature(threatened)
        lifecycle_processor.consume_behavior_shaping(106)
        died = copy.deepcopy(threatened)
        died["frame_state"]["frame_no"] = 112
        died["frame_state"]["npc_states"] = [
            actor
            for actor in died["frame_state"]["npc_states"]
            if actor.get("runtime_id") != 801
        ]
        died["frame_state"]["frame_action"]["dead_action"] = [
            {"death": copy.deepcopy(soldier_actor)}
        ]
        lifecycle_processor.process_feature(died)
        lifecycle_processor.consume_behavior_shaping(112)

        impossible_reappearance, _tower, _soldier_actor = (
            self._defense_threat(118)
        )
        lifecycle_processor.process_feature(impossible_reappearance)
        behavior = lifecycle_processor.consume_behavior_shaping(118)
        self.assertTrue(behavior["defense_wave_invalidated"])
        self.assertEqual(
            behavior["defense_wave_invalidation_reason"],
            "runtime_reuse",
        )

    def test_defense_wave_impossible_position_jump_invalidates_runtime(self):
        processor = FeatureProcessV1(1)
        baseline = make_observation(
            frame=100, include_soldiers=False
        )
        processor.process_feature(baseline)
        processor.consume_behavior_shaping(100)
        self._release_openings(processor)

        threatened, _tower, _soldier_actor = self._defense_threat(106)
        processor.process_feature(threatened)
        processor.consume_behavior_shaping(106)

        missing = copy.deepcopy(threatened)
        missing["frame_state"]["frame_no"] = 112
        missing["frame_state"]["npc_states"] = [
            actor
            for actor in missing["frame_state"]["npc_states"]
            if actor.get("runtime_id") != 801
        ]
        processor.process_feature(missing)
        processor.consume_behavior_shaping(112)

        jumped = copy.deepcopy(threatened)
        jumped["frame_state"]["frame_no"] = 118
        soldier = next(
            actor
            for actor in jumped["frame_state"]["npc_states"]
            if actor.get("runtime_id") == 801
        )
        soldier["location"]["x"] += 10000.0
        processor.process_feature(jumped)
        behavior = processor.consume_behavior_shaping(118)
        self.assertTrue(behavior["defense_wave_invalidated"])
        self.assertEqual(
            behavior["defense_wave_invalidation_reason"],
            "runtime_reuse",
        )
        self.assertEqual(behavior["defense_wave_rewards"], (0.0, 0.0))
        self.assertIsNone(processor.defense_wave_state)

        hp_processor = FeatureProcessV1(1)
        hp_processor.process_feature(baseline)
        hp_processor.consume_behavior_shaping(100)
        self._release_openings(hp_processor)
        low_hp, _tower, low_hp_soldier = self._defense_threat(106)
        low_hp_soldier["hp"] = 800
        hp_processor.process_feature(low_hp)
        hp_processor.consume_behavior_shaping(106)

        hp_missing = copy.deepcopy(low_hp)
        hp_missing["frame_state"]["frame_no"] = 112
        hp_missing["frame_state"]["npc_states"] = [
            actor
            for actor in hp_missing["frame_state"]["npc_states"]
            if actor.get("runtime_id") != 801
        ]
        hp_processor.process_feature(hp_missing)
        hp_processor.consume_behavior_shaping(112)

        hp_reset = copy.deepcopy(low_hp)
        hp_reset["frame_state"]["frame_no"] = 118
        reset_soldier = next(
            actor
            for actor in hp_reset["frame_state"]["npc_states"]
            if actor.get("runtime_id") == 801
        )
        reset_soldier["hp"] = 1500
        hp_processor.process_feature(hp_reset)
        behavior = hp_processor.consume_behavior_shaping(118)
        self.assertTrue(behavior["defense_wave_invalidated"])
        self.assertEqual(
            behavior["defense_wave_invalidation_reason"],
            "runtime_reuse",
        )
        self.assertIsNone(hp_processor.defense_wave_state)

    def test_other_actor_progress_is_hard_ignore_suppression_and_expires(self):
        processor = FeatureProcessV1(1)
        baseline = make_observation(
            frame=100, include_soldiers=False
        )
        processor.process_feature(baseline)
        processor.consume_behavior_shaping(100)
        self._release_openings(processor)

        threatened, own_tower, _soldier_actor = self._defense_threat(106)
        processor.process_feature(threatened)
        processor.consume_behavior_shaping(106)

        progressed = copy.deepcopy(threatened)
        progressed["frame_state"]["frame_no"] = 112
        soldier = next(
            actor
            for actor in progressed["frame_state"]["npc_states"]
            if actor.get("runtime_id") == 801
        )
        soldier["hp"] = 1400
        teammate = next(
            actor
            for actor in progressed["frame_state"]["hero_states"]
            if actor["runtime_id"] == 12
        )
        teammate["hit_target_info"] = [
            {"hit_target": 801, "slot_type": "SLOT_SKILL_0"}
        ]
        teammate["hurt_hero_info"] = [
            {"hurt_target": 801, "hurt": 100}
        ]
        processor.process_feature(progressed)
        processor.consume_behavior_shaping(112)

        suppressed = copy.deepcopy(progressed)
        suppressed["frame_state"]["frame_no"] = 136
        for hero in suppressed["frame_state"]["hero_states"]:
            hero["hit_target_info"] = []
            hero["hurt_hero_info"] = []
        actor = next(
            hero
            for hero in suppressed["frame_state"]["hero_states"]
            if hero["runtime_id"] == 11
        )
        actor["location"] = copy.deepcopy(own_tower["location"])
        actor["hit_target_info"] = [
            {"hit_target": 21, "slot_type": "SLOT_SKILL_0"}
        ]
        processor.process_feature(suppressed)
        behavior = processor.consume_behavior_shaping(136)
        self.assertEqual(behavior["defense_wave_rewards"][0], 0.0)
        self.assertEqual(
            processor.defense_wave_counters[
                "other_progress_suppressed"
            ],
            1.0,
        )
        self.assertEqual(
            processor.defense_wave_counters["ignore_candidate"], 0.0
        )
        self.assertEqual(
            processor.defense_wave_counters["ignore_penalty"], 0.0
        )

        repeated = copy.deepcopy(suppressed)
        repeated["frame_state"]["frame_no"] = 142
        enemy = next(
            hero
            for hero in repeated["frame_state"]["hero_states"]
            if hero["runtime_id"] == 21
        )
        enemy["location"] = copy.deepcopy(own_tower["location"])
        enemy["attack_target"] = own_tower["runtime_id"]
        processor.process_feature(repeated)
        processor.consume_behavior_shaping(142)
        self.assertEqual(
            processor.defense_wave_counters[
                "other_progress_suppressed"
            ],
            1.0,
        )
        self.assertEqual(
            processor.defense_wave_counters[
                "ignore_exempt_division"
            ],
            1.0,
        )
        self.assertEqual(
            processor.defense_wave_counters["ignore_penalty"], 0.0
        )

        expired = copy.deepcopy(suppressed)
        expired["frame_state"]["frame_no"] = 148
        processor.process_feature(expired)
        behavior = processor.consume_behavior_shaping(148)
        self.assertAlmostEqual(
            behavior["defense_wave_rewards"][0], -0.02
        )
        self.assertEqual(
            processor.defense_wave_counters["ignore_candidate"], 1.0
        )
        self.assertEqual(
            processor.defense_wave_counters["ignore_penalty"], 1.0
        )

    def test_defense_progress_uses_only_attributed_hp_drop(self):
        processor = FeatureProcessV1(1)
        baseline = make_observation(frame=100, include_soldiers=False)
        processor.process_feature(baseline)
        processor.consume_behavior_shaping(100)
        self._release_openings(processor)

        threatened = make_observation(frame=106, include_soldiers=False)
        own_tower = next(
            actor
            for actor in threatened["frame_state"]["npc_states"]
            if actor["config_id"] == 66
        )
        soldier = _soldier(
            801,
            2,
            own_tower["location"]["x"] + 500.0,
            19,
            own_tower["location"]["z"] - u.LANE_CENTER_Z,
        )
        soldier["attack_target"] = own_tower["runtime_id"]
        threatened["frame_state"]["npc_states"].append(soldier)
        processor.process_feature(threatened)
        processor.consume_behavior_shaping(106)

        progress = copy.deepcopy(threatened)
        progress["frame_state"]["frame_no"] = 112
        progress_soldier = next(
            actor
            for actor in progress["frame_state"]["npc_states"]
            if actor.get("runtime_id") == 801
        )
        progress_soldier["hp"] = 1100
        hero = next(
            actor
            for actor in progress["frame_state"]["hero_states"]
            if actor["runtime_id"] == 11
        )
        hero["hit_target_info"] = [
            {"hit_target": 801, "slot_type": "SLOT_SKILL_0"}
        ]
        hero["hurt_hero_info"] = [
            {"hurt_target": 801, "hurt": 100}
        ]
        processor.process_feature(progress)
        behavior = processor.consume_behavior_shaping(112)
        expected_progress = 0.01 * 100.0 / 1500.0
        self.assertAlmostEqual(
            behavior["defense_wave_rewards"][0],
            0.005 + expected_progress,
        )
        self.assertAlmostEqual(
            processor.defense_wave_counters["progress_reward"],
            expected_progress,
        )

    def test_prior_response_expires_before_ignore_penalty(self):
        processor = FeatureProcessV1(1)
        baseline = make_observation(frame=100, include_soldiers=False)
        processor.process_feature(baseline)
        processor.consume_behavior_shaping(100)
        self._release_openings(processor)

        threatened = make_observation(frame=106, include_soldiers=False)
        own_tower = next(
            actor
            for actor in threatened["frame_state"]["npc_states"]
            if actor["config_id"] == 66
        )
        soldier = _soldier(
            801,
            2,
            own_tower["location"]["x"] + 500.0,
            19,
            own_tower["location"]["z"] - u.LANE_CENTER_Z,
        )
        soldier["location"] = {
            "x": own_tower["location"]["x"] + 500.0,
            "y": 0,
            "z": own_tower["location"]["z"],
        }
        soldier["attack_target"] = own_tower["runtime_id"]
        threatened["frame_state"]["npc_states"].append(soldier)
        processor.process_feature(threatened)
        processor.consume_behavior_shaping(106)

        response = copy.deepcopy(threatened)
        response["frame_state"]["frame_no"] = 112
        response_soldier = next(
            actor
            for actor in response["frame_state"]["npc_states"]
            if actor.get("runtime_id") == 801
        )
        response_soldier["hp"] = 1400
        hero = next(
            actor
            for actor in response["frame_state"]["hero_states"]
            if actor["runtime_id"] == 11
        )
        hero["hit_target_info"] = [
            {"hit_target": 801, "slot_type": "SLOT_SKILL_0"}
        ]
        hero["hurt_hero_info"] = [
            {"hurt_target": 801, "hurt": 100}
        ]
        processor.process_feature(response)
        processor.consume_behavior_shaping(112)

        ignored = copy.deepcopy(response)
        ignored["frame_state"]["frame_no"] = 143
        ignored_hero = next(
            actor
            for actor in ignored["frame_state"]["hero_states"]
            if actor["runtime_id"] == 11
        )
        ignored_hero["location"] = copy.deepcopy(own_tower["location"])
        ignored_hero["hit_target_info"] = [
            {"hit_target": 21, "slot_type": "SLOT_SKILL_0"}
        ]
        ignored_hero["hurt_hero_info"] = []
        processor.process_feature(ignored)
        behavior = processor.consume_behavior_shaping(143)
        self.assertAlmostEqual(
            behavior["defense_wave_rewards"][0], -0.02
        )
        self.assertEqual(
            processor.defense_wave_counters["ignore_candidate"], 1.0
        )
        self.assertEqual(
            processor.defense_wave_counters["ignore_penalty"], 1.0
        )

    def test_zero_weight_keeps_defense_diagnostics(self):
        previous_scale = V1TrainingConfig.DEFENSE_WAVE_REWARD_SCALE
        V1TrainingConfig.DEFENSE_WAVE_REWARD_SCALE = 0.0
        try:
            processor = FeatureProcessV1(1)
            baseline = make_observation(
                frame=100, include_soldiers=False
            )
            processor.process_feature(baseline)
            processor.consume_behavior_shaping(100)
            self._release_openings(processor)
            threatened = make_observation(
                frame=106, include_soldiers=False
            )
            own_tower = next(
                actor
                for actor in threatened["frame_state"]["npc_states"]
                if actor["config_id"] == 66
            )
            soldier = _soldier(
                801,
                2,
                own_tower["location"]["x"] + 500.0,
                19,
                own_tower["location"]["z"] - u.LANE_CENTER_Z,
            )
            soldier["attack_target"] = own_tower["runtime_id"]
            threatened["frame_state"]["npc_states"].append(soldier)
            processor.process_feature(threatened)
            processor.consume_behavior_shaping(106)

            response = copy.deepcopy(threatened)
            response["frame_state"]["frame_no"] = 112
            hero = next(
                actor
                for actor in response["frame_state"]["hero_states"]
                if actor["runtime_id"] == 11
            )
            hero["hit_target_info"] = [
                {"hit_target": 801, "slot_type": "SLOT_SKILL_0"}
            ]
            processor.process_feature(response)
            behavior = processor.consume_behavior_shaping(112)
            self.assertEqual(
                behavior["defense_wave_rewards"], (0.0, 0.0)
            )
            self.assertEqual(
                processor.defense_wave_counters["first_response"], 1.0
            )
        finally:
            V1TrainingConfig.DEFENSE_WAVE_REWARD_SCALE = previous_scale


if __name__ == "__main__":
    unittest.main()
