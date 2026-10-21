#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""Controlled-Kai opening rule tests."""

import unittest

import numpy as np

from agent_ppo.conf.conf import FrozenV1Config
from agent_ppo.feature.collaboration_v1 import build_command_legal_masks_v1
from agent_ppo.feature.collision_avoidance_v1 import MOVE_BUTTON, MOVE_HEAD_OFFSET
from agent_ppo.feature.jungle_opening_v1 import JungleOpeningLegalizerV1
from agent_ppo.feature.feature_process.v1_feature_process import (
    FeatureProcessV1,
    KAI_OPENING_CAMP_ORDER,
    KAI_OPENING_MAX_FRAME,
)
from agent_ppo.feature.feature_process import v1_utils as u
from agent_ppo.tests.test_feature_process_v1 import (
    _monster,
    _raw_position,
    make_observation,
)


class JungleOpeningLegalizerV1Test(unittest.TestCase):
    @staticmethod
    def _raw_legal(skills=True):
        result = np.ones(
            (FrozenV1Config.CONTROLLED_ACTOR_COUNT, FrozenV1Config.RAW_LEGAL_ACTION_DIM),
            dtype=np.float32,
        )
        if not skills:
            result[:, 4:9] = 0.0
        return result

    @staticmethod
    def _plans(active=True, interrupt="none"):
        return (
            {"runtime_id": 11, "phase": "upper_deer", "interrupt": interrupt,
             "hard_mask_active": active, "waypoint_dx": 20000.0, "waypoint_dz": 8000.0},
            {"runtime_id": 12, "phase": "inactive", "interrupt": "none",
             "hard_mask_active": False, "waypoint_dx": 0.0, "waypoint_dz": 0.0},
        )

    def test_controlled_kai_is_forced_to_move_before_sampling(self):
        raw = self._raw_legal()
        result, diagnostics = JungleOpeningLegalizerV1().apply(
            raw, (193, 141), self._plans(), ({"runtime_id": 11}, {"runtime_id": 12})
        )
        button = result[0, :13]
        move = result[0, MOVE_HEAD_OFFSET : MOVE_HEAD_OFFSET + 25]
        self.assertEqual(button.sum(), 1)
        self.assertEqual(button[MOVE_BUTTON], 1)
        self.assertGreaterEqual(move.sum(), 1)
        self.assertLessEqual(move.sum(), 3)
        self.assertEqual(diagnostics[0]["reason"], "upper_jungle_rule")
        np.testing.assert_array_equal(result[1], raw[1])

    def test_attackable_upper_camp_forces_common_attack_before_sampling(self):
        raw = self._raw_legal()
        plans = list(self._plans())
        plans[0] = dict(plans[0], force_attack=True, target_slot=18)
        legalizer = JungleOpeningLegalizerV1()
        result, diagnostics = legalizer.apply(
            self._raw_legal(skills=False), (193, 141), tuple(plans), ()
        )
        button = result[0, :13]
        target_row = result[0, 122 + 3 * 39 : 122 + 4 * 39]
        self.assertEqual(button.sum(), 1)
        self.assertEqual(button[3], 1)
        self.assertEqual(target_row.sum(), 1)
        self.assertEqual(target_row[18], 1)
        self.assertEqual(diagnostics[0]["reason"], "upper_jungle_attack")
        self.assertEqual(legalizer.attack_application_count, 1)

    def test_protected_first_lower_camp_is_forced_before_sampling(self):
        raw = self._raw_legal()
        plans = list(self._plans())
        plans[0] = dict(
            plans[0],
            phase="lower_clear",
            hard_mask_active=True,
            force_attack=True,
            target_slot=10,
        )
        result, diagnostics = JungleOpeningLegalizerV1().apply(
            self._raw_legal(skills=False), (193, 141), tuple(plans), ()
        )
        button = result[0, :13]
        target_row = result[0, 122 + 3 * 39 : 122 + 4 * 39]
        self.assertEqual(button[3], 1)
        self.assertEqual(target_row[10], 1)
        self.assertEqual(diagnostics[0]["reason"], "lower_jungle_attack")

    def test_attackable_camp_consumes_available_skill_before_common_attack(self):
        raw = self._raw_legal()
        plans = list(self._plans())
        plans[0] = dict(
            plans[0],
            force_attack=True,
            target_slot=18,
            target_dx=8000.0,
            target_dz=1000.0,
        )
        result, diagnostics = JungleOpeningLegalizerV1().apply(
            raw, (193, 141), tuple(plans), ()
        )
        self.assertEqual(np.flatnonzero(result[0, :13]).tolist(), [8])
        target_row = result[0, 122 + 8 * 39 : 122 + 9 * 39]
        self.assertEqual(np.flatnonzero(target_row).tolist(), [18])
        self.assertEqual(diagnostics[0]["reason"], "upper_jungle_skill")

    def test_opening_skill_priority_falls_through_platform_cooldowns(self):
        raw = self._raw_legal()
        raw[0, 8] = 0.0
        raw[0, 5] = 0.0
        plans = list(self._plans())
        plans[0] = dict(
            plans[0],
            force_attack=True,
            target_slot=18,
            target_dx=8000.0,
            target_dz=1000.0,
        )
        result, _diagnostics = JungleOpeningLegalizerV1().apply(
            raw, (193, 141), tuple(plans), ()
        )
        self.assertEqual(np.flatnonzero(result[0, :13]).tolist(), [4])

    def test_directional_opening_skill_uses_center_offsets_and_camp_target(self):
        raw = self._raw_legal()
        raw[0, 8] = 0.0
        plans = list(self._plans())
        plans[0] = dict(
            plans[0],
            force_attack=True,
            target_slot=18,
            target_dx=8000.0,
            target_dz=1000.0,
        )
        result, diagnostics = JungleOpeningLegalizerV1().apply(
            raw, (193, 141), tuple(plans), ()
        )
        offset_x_start = sum(FrozenV1Config.HERO_ACTION_HEADS[:2])
        offset_z_start = (
            offset_x_start + FrozenV1Config.HERO_ACTION_HEADS[2]
        )
        self.assertEqual(np.flatnonzero(result[0, :13]).tolist(), [5])
        self.assertEqual(
            np.flatnonzero(result[0, offset_x_start:offset_x_start + 42]).tolist(),
            [21],
        )
        self.assertEqual(
            np.flatnonzero(result[0, offset_z_start:offset_z_start + 42]).tolist(),
            [21],
        )
        target_start = 122 + 5 * 39
        self.assertEqual(
            np.flatnonzero(result[0, target_start:target_start + 39]).tolist(),
            [18],
        )
        self.assertTrue(diagnostics[0]["skill_target_locked"])
        self.assertEqual(diagnostics[0]["skill_offset"], (21, 21))

    def test_directional_opening_skill_uses_nearest_legal_offsets(self):
        raw = self._raw_legal()
        raw[0, 8] = 0.0
        offset_x_start = sum(FrozenV1Config.HERO_ACTION_HEADS[:2])
        offset_z_start = (
            offset_x_start + FrozenV1Config.HERO_ACTION_HEADS[2]
        )
        raw[0, offset_x_start:offset_x_start + 42] = 0.0
        raw[0, offset_x_start + 20] = 1.0
        raw[0, offset_x_start + 24] = 1.0
        raw[0, offset_z_start:offset_z_start + 42] = 0.0
        raw[0, offset_z_start + 23] = 1.0
        plans = list(self._plans())
        plans[0] = dict(
            plans[0],
            force_attack=True,
            target_slot=18,
            target_dx=8000.0,
            target_dz=1000.0,
        )
        result, diagnostics = JungleOpeningLegalizerV1().apply(
            raw, (193, 141), tuple(plans), ()
        )
        self.assertEqual(
            np.flatnonzero(result[0, offset_x_start:offset_x_start + 42]).tolist(),
            [20],
        )
        self.assertEqual(
            np.flatnonzero(result[0, offset_z_start:offset_z_start + 42]).tolist(),
            [23],
        )
        self.assertEqual(diagnostics[0]["skill_offset"], (20, 23))

    def test_opening_skill_never_uses_an_empty_target_row(self):
        raw = self._raw_legal()
        plans = list(self._plans())
        plans[0] = dict(
            plans[0],
            force_attack=True,
            target_slot=18,
            target_dx=8000.0,
            target_dz=1000.0,
        )
        for button in (8, 5, 4, 6, 7):
            start = 122 + button * 39
            raw[0, start:start + 39] = 0.0
        result, diagnostics = JungleOpeningLegalizerV1().apply(
            raw, (193, 141), tuple(plans), ()
        )
        self.assertEqual(np.flatnonzero(result[0, :13]).tolist(), [3])
        self.assertEqual(diagnostics[0]["reason"], "upper_jungle_attack")

        raw[0, 122 + 3 * 39 + 18] = 0.0
        moved, diagnostics = JungleOpeningLegalizerV1().apply(
            raw, (193, 141), tuple(plans), ()
        )
        self.assertEqual(np.flatnonzero(moved[0, :13]).tolist(), [2])
        self.assertEqual(diagnostics[0]["reason"], "upper_jungle_rule")

    def test_opening_travel_never_spends_second_skill(self):
        result, diagnostics = JungleOpeningLegalizerV1().apply(
            self._raw_legal(),
            (193, 141),
            self._plans(),
            (),
        )
        self.assertEqual(np.flatnonzero(result[0, :13]).tolist(), [2])
        self.assertEqual(result[0, 5], 0.0)
        self.assertEqual(diagnostics[0]["reason"], "upper_jungle_rule")

    def test_frozen_opening_order_follows_outer_lower_perimeter(self):
        self.assertEqual(
            tuple(item[0] for item in KAI_OPENING_CAMP_ORDER),
            (
                "wolf",
                "deer",
                "pioneer",
                "outer_archer",
                "inner_archer",
                "red_statue",
            ),
        )

    def test_deer_to_pioneer_route_is_side_normalized_for_both_camps(self):
        expected_prefix = (
            u.side_position(u.fixed_monster_anchor(42, "own", 1), 1),
            (-25000.0, 15000.0 - u.LANE_CENTER_Z),
            (-25000.0, 0.0),
            (-23000.0, 0.0),
        )
        camp1 = FeatureProcessV1(1)._opening_route_waypoints("pioneer")
        camp2 = FeatureProcessV1(2)._opening_route_waypoints("pioneer")
        self.assertEqual(camp1[:4], expected_prefix)
        self.assertEqual(
            camp2[0],
            u.side_position(u.fixed_monster_anchor(42, "own", 2), 2),
        )
        self.assertEqual(camp2[1:4], expected_prefix[1:])

        self.assertEqual(
            u.raw_position(camp1[1], 1),
            (-25000.0, 15000.0),
        )
        self.assertEqual(
            u.raw_position(camp2[1], 2),
            (25000.0, 15000.0),
        )
        self.assertEqual(
            camp1[-1],
            u.side_position(u.fixed_monster_anchor(30, "own", 1), 1),
        )
        self.assertEqual(
            camp2[-1],
            u.side_position(u.fixed_monster_anchor(30, "own", 2), 2),
        )

    def test_feature_plan_keeps_first_lower_rotation_hard_masked(self):
        processor = FeatureProcessV1(1)
        for config_id in (42, 44):
            anchor = tuple(u.fixed_monster_anchor(config_id, "own", 1))
            processor.monster_camp_history[anchor] = {
                "ever_alive": True,
                "cleared_once": True,
                "live_cycle_seen": False,
                "clear_count": 1,
                "last_clear_frame": 50,
            }
        observation = make_observation(
            frame=100, camp=1, include_soldiers=False
        )
        observation["official_model_info"] = {"official": {"runtime_id": 11}}
        observation["frame_state"]["npc_states"].append(
            _monster(30, "own", 1, 601)
        )
        lower_anchor = u.side_position(
            u.fixed_monster_anchor(30, "own", 1), 1
        )
        kai = next(
            hero
            for hero in observation["frame_state"]["hero_states"]
            if hero["camp"] == 1 and hero["config_id"] == 193
        )
        kai["location"] = _raw_position(
            lower_anchor[0] - 8000.0, lower_anchor[1], 1
        )
        processor.process_feature(observation)
        macro = processor._memo_context["jungle_macro"]
        plan = processor.jungle_opening_runtime_plans()[1]
        self.assertEqual(macro["phase"], "lower_clear")
        self.assertTrue(macro["opening_rotation_active"])
        self.assertFalse(macro["lower_farm_started"])
        self.assertTrue(plan["hard_mask_active"])

    def test_interrupt_precedes_opening_but_collision_does_not(self):
        raw = self._raw_legal()
        interrupted, diagnostics = JungleOpeningLegalizerV1().apply(
            raw, (193, 141), self._plans(interrupt="retreat"), ()
        )
        np.testing.assert_array_equal(interrupted, raw)
        self.assertEqual(diagnostics[0]["reason"], "retreat")
        escaped, diagnostics = JungleOpeningLegalizerV1().apply(
            raw, (193, 141), self._plans(), ({"runtime_id": 11, "escape_active": True},)
        )
        self.assertFalse(np.array_equal(escaped, raw))
        self.assertEqual(diagnostics[0]["reason"], "upper_jungle_rule")

        inactive_collision, diagnostics = JungleOpeningLegalizerV1().apply(
            raw,
            (193, 141),
            self._plans(active=False),
            ({"runtime_id": 11, "escape_active": True},),
        )
        np.testing.assert_array_equal(inactive_collision, raw)
        self.assertEqual(diagnostics[0]["reason"], "collision_escape")

    def test_route_uses_each_camps_asymmetric_debug_anchors(self):
        camp1 = FeatureProcessV1(1)
        camp1.process_feature(make_observation(camp=1))
        camp1_positions = camp1._route_node_positions(camp1._memo_context)
        self.assertEqual(camp1_positions[1], (-25530.0, 17875.0))
        self.assertEqual(camp1_positions[2], (-14010.0, 21555.0))

        camp2 = FeatureProcessV1(2)
        camp2.process_feature(make_observation(camp=2))
        camp2_positions = camp2._route_node_positions(camp2._memo_context)
        self.assertEqual(camp2_positions[1], (-25330.0, 17715.0))
        self.assertEqual(camp2_positions[2], (-13980.0, 20735.0))

    def test_feature_macro_activates_the_controlled_kai_rule(self):
        processor = FeatureProcessV1(1)
        observation = make_observation(camp=1)
        observation["official_model_info"] = {"official": {"runtime_id": 11}}
        processor.process_feature(observation)
        plans = processor.jungle_opening_runtime_plans()
        kai_plan = next(
            plan
            for info, plan in zip(processor._memo_context["controlled_infos"], plans)
            if int(info["config_id"]) == 193
        )
        self.assertEqual(kai_plan["phase"], "upper_wolf")
        self.assertTrue(kai_plan["hard_mask_active"])

    def test_monster_anchor_leash_recovery_uses_monster_not_kai_distance(self):
        processor = FeatureProcessV1(1)
        observation = make_observation(
            frame=100, camp=1, include_soldiers=False
        )
        observation["official_model_info"] = {
            "official": {"runtime_id": 11}
        }
        wolf = _monster(44, "own", 1, 601)
        wolf_anchor_raw = u.fixed_monster_anchor(44, "own", 1)
        wolf["location"] = {
            "x": wolf_anchor_raw[0] + 5001.0,
            "y": 0,
            "z": wolf_anchor_raw[1],
        }
        observation["frame_state"]["npc_states"].append(wolf)
        wolf_anchor_side = u.side_position(wolf_anchor_raw, 1)
        kai = next(
            hero
            for hero in observation["frame_state"]["hero_states"]
            if hero["camp"] == 1 and hero["config_id"] == 193
        )
        kai["location"] = _raw_position(*wolf_anchor_side, 1)
        processor.process_feature(observation)
        macro = processor._memo_context["jungle_macro"]
        plan = next(
            item
            for info, item in zip(
                processor._memo_context["controlled_infos"],
                processor.jungle_opening_runtime_plans(),
            )
            if int(info["config_id"]) == 193
        )
        self.assertTrue(macro["opening_leash_recover"])
        self.assertGreater(macro["opening_monster_anchor_distance"], 4500.0)
        self.assertEqual(macro["next_waypoint"], wolf_anchor_side)
        self.assertFalse(macro["opening_target_attackable"])
        self.assertTrue(plan["leash_recover"])

    def test_kai_distance_alone_never_triggers_4500_leash_recovery(self):
        processor = FeatureProcessV1(1)
        observation = make_observation(
            frame=100, camp=1, include_soldiers=False
        )
        observation["official_model_info"] = {
            "official": {"runtime_id": 11}
        }
        wolf = _monster(44, "own", 1, 601)
        observation["frame_state"]["npc_states"].append(wolf)
        wolf_anchor_side = u.side_position(
            u.fixed_monster_anchor(44, "own", 1), 1
        )
        kai = next(
            hero
            for hero in observation["frame_state"]["hero_states"]
            if hero["camp"] == 1 and hero["config_id"] == 193
        )
        kai["location"] = _raw_position(
            wolf_anchor_side[0] - 10000.0,
            wolf_anchor_side[1],
            1,
        )
        processor.process_feature(observation)
        macro = processor._memo_context["jungle_macro"]
        self.assertFalse(macro["opening_leash_recover"])
        self.assertEqual(macro["opening_monster_anchor_distance"], 0.0)

    def test_upper_entry_commitment_does_not_route_kai_back_to_lane(self):
        processor = FeatureProcessV1(1)
        processor.process_feature(make_observation(camp=1))
        positions = processor._route_node_positions(processor._memo_context)
        for position in ((-25000.0, 3000.0), (-25000.0, 7000.0)):
            _path, next_node, _distance = processor._shortest_route(
                positions, position, 1
            )
            self.assertEqual(next_node, 1)

    def test_route_stuck_is_a_recovery_pulse_not_a_dense_level(self):
        processor = FeatureProcessV1(1)
        route_stuck_frames = []
        route_rewards = []
        for frame in (100, 130, 160, 166):
            observation = make_observation(frame=frame, camp=1)
            observation["official_model_info"] = {"official": {"runtime_id": 11}}
            kai = next(
                hero
                for hero in observation["frame_state"]["hero_states"]
                if hero["camp"] == 1 and hero["config_id"] == 193
            )
            kai["location"] = _raw_position(-35000.0, 8000.0, 1)
            processor.process_feature(observation)
            macro = processor._memo_context["jungle_macro"]
            if macro["route_stuck"]:
                route_stuck_frames.append(frame)
            route_rewards.append(macro["route_shaping_reward"])

        self.assertEqual(route_stuck_frames, [160])
        self.assertEqual(route_rewards, [0.0, 0.0, 0.0, 0.0])

    def test_continuous_wrong_way_movement_triggers_route_recovery(self):
        processor = FeatureProcessV1(1)
        route_stuck_frames = []
        for frame, side_x in ((100, -35000.0), (112, -30000.0), (124, -25000.0)):
            observation = make_observation(frame=frame, camp=1)
            observation["official_model_info"] = {
                "official": {"runtime_id": 11}
            }
            kai = next(
                hero
                for hero in observation["frame_state"]["hero_states"]
                if hero["camp"] == 1 and hero["config_id"] == 193
            )
            kai["location"] = _raw_position(side_x, 8000.0, 1)
            processor.process_feature(observation)
            if processor._memo_context["jungle_macro"]["route_stuck"]:
                route_stuck_frames.append(frame)
        self.assertEqual(route_stuck_frames, [124])

    def test_opening_waypoints_cannot_skip_a_required_intermediate_point(self):
        processor = FeatureProcessV1(1)
        processor.jungle_route_history["opening_route_key"] = "wolf"
        processor.jungle_route_history["opening_waypoint_index"] = 1
        observation = make_observation(frame=100, camp=1)
        observation["official_model_info"] = {
            "official": {"runtime_id": 11}
        }
        kai = next(
            hero
            for hero in observation["frame_state"]["hero_states"]
            if hero["camp"] == 1 and hero["config_id"] == 193
        )
        later_waypoint = FeatureProcessV1(1)._opening_route_waypoints("wolf")[3]
        kai["location"] = _raw_position(*later_waypoint, 1)
        processor.process_feature(observation)
        self.assertEqual(
            processor.jungle_route_history["opening_waypoint_index"], 1
        )

    def test_official_kai_trace_upper_positions_switch_directly_to_wolf(self):
        trace_positions = {
            1: (-30344.0, 12800.0),
            2: (-26032.0, 11917.0),
        }
        for camp, position in trace_positions.items():
            with self.subTest(camp=camp):
                processor = FeatureProcessV1(camp)
                observation = make_observation(frame=284, camp=camp)
                kai = next(
                    hero
                    for hero in observation["frame_state"]["hero_states"]
                    if hero["camp"] == camp and hero["config_id"] == 193
                )
                kai["location"] = _raw_position(*position, camp)
                processor.process_feature(observation)
                macro = processor._memo_context["jungle_macro"]
                self.assertEqual(macro["official_upper_route_stage"], 4)
                self.assertEqual(
                    macro["next_waypoint"],
                    u.side_position(
                        u.fixed_monster_anchor(44, "own", camp), camp
                    ),
                )

    def test_official_kai_pioneer_route_descends_directly_through_lower_entry(self):
        processor = FeatureProcessV1(1)
        for config_id in (44, 42):
            anchor = tuple(u.fixed_monster_anchor(config_id, "own", 1))
            processor.monster_camp_history[anchor] = {
                "ever_alive": True,
                "cleared_once": True,
                "live_cycle_seen": False,
                "clear_count": 1,
                "last_clear_frame": 50,
            }

        positions = ((-11214.0, 21990.0), (-20000.0, 3000.0))
        expected_indices = (0, 1)
        expected_waypoints = (
            (-23000.0, 0.0),
            u.side_position(u.fixed_monster_anchor(30, "own", 1), 1),
        )
        for offset, (position, expected_index) in enumerate(
            zip(positions, expected_indices)
        ):
            observation = make_observation(
                frame=100 + offset * 6,
                camp=1,
                include_soldiers=False,
            )
            kai = next(
                hero
                for hero in observation["frame_state"]["hero_states"]
                if hero["camp"] == 1 and hero["config_id"] == 193
            )
            kai["location"] = _raw_position(*position, 1)
            processor.process_feature(observation)
            self.assertEqual(
                processor.jungle_route_history["opening_waypoint_index"],
                expected_index,
            )
            self.assertEqual(
                processor._memo_context["jungle_macro"]["next_waypoint"],
                expected_waypoints[offset],
            )

    def test_official_kai_trace_base_exit_uses_five_thousand_tolerance(self):
        processor = FeatureProcessV1(1)
        observation = make_observation(
            frame=164, camp=1, include_soldiers=False
        )
        kai = next(
            hero
            for hero in observation["frame_state"]["hero_states"]
            if hero["camp"] == 1 and hero["config_id"] == 193
        )
        kai["location"] = _raw_position(-52139.0, 4882.0, 1)

        processor.process_feature(observation)
        macro = processor._memo_context["jungle_macro"]

        self.assertEqual(macro["official_upper_route_stage"], 1)
        self.assertEqual(macro["next_waypoint"], (-30000.0, 3500.0))

    def test_official_kai_trace_upper_band_accepts_seven_thousand_five_hundred(self):
        processor = FeatureProcessV1(1)
        observation = make_observation(
            frame=1784, camp=1, include_soldiers=False
        )
        kai = next(
            hero
            for hero in observation["frame_state"]["hero_states"]
            if hero["camp"] == 1 and hero["config_id"] == 193
        )
        kai["location"] = _raw_position(-31365.0, 7776.0, 1)

        processor.process_feature(observation)
        macro = processor._memo_context["jungle_macro"]

        self.assertEqual(macro["official_upper_route_stage"], 4)
        self.assertEqual(
            macro["next_waypoint"],
            u.side_position(u.fixed_monster_anchor(44, "own", 1), 1),
        )

    def test_opening_stage_is_frozen_after_timeout_release(self):
        processor = FeatureProcessV1(1)
        timed_out = make_observation(
            frame=KAI_OPENING_MAX_FRAME,
            camp=1,
            include_soldiers=False,
        )
        processor.process_feature(timed_out)
        self.assertEqual(
            processor._memo_context["jungle_macro"]["opening_stage_index"],
            0,
        )

        after = make_observation(
            frame=KAI_OPENING_MAX_FRAME + 6,
            camp=1,
            include_soldiers=False,
        )
        after["frame_state"]["npc_states"].extend(
            [
                _monster(42, "own", 1, 501),
                _monster(43, "own", 1, 502),
                _monster(44, "own", 1, 503),
                _monster(45, "own", 1, 504),
            ]
        )
        kai = next(
            hero
            for hero in after["frame_state"]["hero_states"]
            if hero["camp"] == 1 and hero["config_id"] == 193
        )
        kai["location"] = _raw_position(-25530.0, 12000.0, 1)
        processor.process_feature(after)
        macro = processor._memo_context["jungle_macro"]

        self.assertFalse(macro["opening_rotation_active"])
        self.assertEqual(macro["opening_release_reason"], "opening_timeout")
        self.assertEqual(macro["opening_stage_index"], 0)

    def test_enemy_side_contact_caused_by_bad_path_does_not_release_opening(self):
        processor = FeatureProcessV1(1)
        observation = make_observation(frame=100, camp=1)
        observation["official_model_info"] = {
            "official": {"runtime_id": 11}
        }
        kai = next(
            hero
            for hero in observation["frame_state"]["hero_states"]
            if hero["camp"] == 1 and hero["config_id"] == 193
        )
        enemy = next(
            hero
            for hero in observation["frame_state"]["hero_states"]
            if hero["camp"] == 2
        )
        kai["location"] = _raw_position(10000.0, 10000.0, 1)
        enemy["location"] = _raw_position(10500.0, 10000.0, 1)
        processor.process_feature(observation)
        macro = processor._memo_context["jungle_macro"]
        self.assertEqual(macro["interrupt"], "none")
        self.assertTrue(macro["opening_rotation_active"])

    def test_nearby_enemy_inside_own_jungle_still_releases_opening(self):
        processor = FeatureProcessV1(1)
        observation = make_observation(frame=100, camp=1)
        observation["official_model_info"] = {
            "official": {"runtime_id": 11}
        }
        kai = next(
            hero
            for hero in observation["frame_state"]["hero_states"]
            if hero["camp"] == 1 and hero["config_id"] == 193
        )
        enemy = next(
            hero
            for hero in observation["frame_state"]["hero_states"]
            if hero["camp"] == 2
        )
        kai["location"] = _raw_position(-14000.0, 15000.0, 1)
        enemy["location"] = _raw_position(-14500.0, 15000.0, 1)
        processor.process_feature(observation)
        macro = processor._memo_context["jungle_macro"]
        self.assertIn(macro["interrupt"], ("contest_enemy", "retreat"))
        self.assertFalse(macro["opening_rotation_active"])

    def test_lower_route_cursor_never_reverses_to_wolf_after_deer(self):
        processor = FeatureProcessV1(1)
        processor.process_feature(make_observation(camp=1))
        positions = processor._route_node_positions(processor._memo_context)
        signature = ("lower_clear", "farm", "none", 5)

        path, next_node, _distance = processor._committed_shortest_route(
            positions, positions[2], 5, signature
        )
        self.assertEqual(path, (1, 0, 4, 5))
        self.assertEqual(next_node, 1)

        path, next_node, _distance = processor._committed_shortest_route(
            positions, positions[1], 5, signature
        )
        self.assertEqual(path, (0, 4, 5))
        self.assertEqual(next_node, 0)

        # Even if movement or collision pushes Kai back toward the wolf, the
        # same route cannot reattach to node 2 or reverse to node 1.
        path, next_node, _distance = processor._committed_shortest_route(
            positions, positions[2], 5, signature
        )
        self.assertEqual(path, (0, 4, 5))
        self.assertEqual(next_node, 0)

    def test_deer_main_death_switches_to_living_lesser_monster(self):
        processor = FeatureProcessV1(1)
        wolf_anchor = tuple(u.fixed_monster_anchor(44, "own", 1))
        processor.monster_camp_history[wolf_anchor] = {
            "ever_alive": True,
            "cleared_once": True,
            "live_cycle_seen": False,
            "cycle_seen_runtimes": {501, 502},
            "expected_members": 2,
            "clear_count": 1,
            "last_clear_frame": 80,
        }
        observation = make_observation(frame=100, camp=1)
        observation["official_model_info"] = {"official": {"runtime_id": 11}}
        observation["frame_state"]["npc_states"].append(
            _monster(43, "own", 1, 502)
        )
        for spawn in observation["frame_state"]["monster_spawn_infos"]:
            spawn["spawn_timer"] = 0
        kai = next(
            hero
            for hero in observation["frame_state"]["hero_states"]
            if hero["camp"] == 1 and hero["config_id"] == 193
        )
        kai["location"] = _raw_position(-25530, 17875, 1)
        processor.process_feature(observation)
        macro = processor._memo_context["jungle_macro"]
        plans = processor.jungle_opening_runtime_plans()
        kai_plan = plans[1]
        self.assertEqual(macro["phase"], "upper_deer")
        self.assertEqual(macro["target_slot"], 12)
        self.assertTrue(macro["opening_target_attackable"])
        self.assertEqual(kai_plan["target_slot"], 20)

        result, diagnostics = JungleOpeningLegalizerV1().apply(
            self._raw_legal(), (141, 193), plans, ()
        )
        self.assertEqual(diagnostics[1]["reason"], "upper_jungle_skill")
        target_row = result[1, 122 + 8 * 39 : 122 + 9 * 39]
        self.assertEqual(np.flatnonzero(target_row).tolist(), [20])

    def test_wolf_main_death_switches_to_living_lesser_monster(self):
        processor = FeatureProcessV1(1)
        alive = make_observation(frame=90, camp=1)
        alive["official_model_info"] = {"official": {"runtime_id": 11}}
        alive["frame_state"]["npc_states"].append(
            _monster(42, "own", 1, 501)
        )
        for spawn in alive["frame_state"]["monster_spawn_infos"]:
            spawn["spawn_timer"] = 0
        processor.process_feature(alive)

        observation = make_observation(frame=100, camp=1)
        observation["official_model_info"] = {"official": {"runtime_id": 11}}
        observation["frame_state"]["npc_states"].append(
            _monster(45, "own", 1, 504)
        )
        for spawn in observation["frame_state"]["monster_spawn_infos"]:
            spawn["spawn_timer"] = (
                60000 if spawn["monster_id"] in (42, 43) else 0
            )
        kai = next(
            hero
            for hero in observation["frame_state"]["hero_states"]
            if hero["camp"] == 1 and hero["config_id"] == 193
        )
        kai["location"] = _raw_position(-14010, 21555, 1)
        processor.process_feature(observation)
        macro = processor._memo_context["jungle_macro"]
        self.assertEqual(macro["phase"], "upper_wolf")
        self.assertEqual(macro["target_slot"], 16)
        self.assertTrue(macro["opening_target_attackable"])


    def test_wolf_buffs_do_not_skip_a_living_lesser_wolf(self):
        processor = FeatureProcessV1(1)
        alive = make_observation(frame=90, camp=1)
        alive["official_model_info"] = {"official": {"runtime_id": 11}}
        alive["frame_state"]["npc_states"].extend(
            [_monster(44, "own", 1, 503), _monster(45, "own", 1, 504)]
        )
        for spawn in alive["frame_state"]["monster_spawn_infos"]:
            spawn["spawn_timer"] = 0
        processor.process_feature(alive)

        companion_only = make_observation(frame=96, camp=1)
        companion_only["official_model_info"] = {
            "official": {"runtime_id": 11}
        }
        companion_only["frame_state"]["npc_states"].append(
            _monster(45, "own", 1, 504)
        )
        for spawn in companion_only["frame_state"]["monster_spawn_infos"]:
            spawn["spawn_timer"] = 0
        kai = next(
            hero
            for hero in companion_only["frame_state"]["hero_states"]
            if hero["camp"] == 1 and hero["config_id"] == 193
        )
        kai["location"] = _raw_position(-14010, 21555, 1)
        kai["buff_state"]["buff_skills"] = [
            {"config_id": 90204, "start_time": 90, "times": 1},
            {"config_id": 902070, "start_time": 90, "times": 1},
        ]
        processor.process_feature(companion_only)
        macro = processor._memo_context["jungle_macro"]
        self.assertEqual(macro["opening_camp_name"], "wolf")
        self.assertEqual(
            u.actor_runtime_id(macro["opening_target_actor"]), 504
        )
        self.assertFalse(
            processor.monster_camp_history[
                tuple(u.fixed_monster_anchor(44, "own", 1))
            ]["cleared_once"]
        )

    def test_initial_positive_spawn_countdown_does_not_clear_unseen_wolf(self):
        processor = FeatureProcessV1(1)
        observation = make_observation(frame=100, camp=1)
        observation["official_model_info"] = {
            "official": {"runtime_id": 11}
        }
        processor.process_feature(observation)
        macro = processor._memo_context["jungle_macro"]
        wolf_state = processor.monster_camp_history[
            tuple(u.fixed_monster_anchor(44, "own", 1))
        ]
        self.assertEqual(macro["opening_camp_name"], "wolf")
        self.assertFalse(wolf_state["spawn_timer_zero_seen"])
        self.assertFalse(wolf_state["timer_death_transition"])
        self.assertFalse(wolf_state["cleared_once"])

    def test_official_kai_accepts_living_deer_first_after_upper_entry(self):
        observation = make_observation(
            frame=400, camp=1, include_soldiers=False
        )
        observation["frame_state"]["npc_states"].extend(
            [
                _monster(42, "own", 1, 501),
                _monster(43, "own", 1, 502),
                _monster(44, "own", 1, 503),
                _monster(45, "own", 1, 504),
            ]
        )
        for spawn in observation["frame_state"]["monster_spawn_infos"]:
            spawn["spawn_timer"] = 0
        kai = next(
            hero
            for hero in observation["frame_state"]["hero_states"]
            if hero["camp"] == 1 and hero["config_id"] == 193
        )
        kai["location"] = _raw_position(-25530, 12000, 1)

        processor = FeatureProcessV1(1)
        processor.process_feature(observation)
        macro = processor._memo_context["jungle_macro"]

        self.assertTrue(macro["kai_is_official"])
        self.assertEqual(macro["opening_camp_name"], "deer")
        self.assertEqual(macro["opening_stage_index"], 1)
        self.assertEqual(
            macro["next_waypoint"],
            u.side_position(u.fixed_monster_anchor(42, "own", 1), 1),
        )

    def test_controlled_kai_keeps_strict_wolf_first_order(self):
        observation = make_observation(
            frame=400, camp=1, include_soldiers=False
        )
        observation["official_model_info"] = {
            "official": {"runtime_id": 11}
        }
        observation["frame_state"]["npc_states"].extend(
            [
                _monster(42, "own", 1, 501),
                _monster(43, "own", 1, 502),
                _monster(44, "own", 1, 503),
                _monster(45, "own", 1, 504),
            ]
        )
        for spawn in observation["frame_state"]["monster_spawn_infos"]:
            spawn["spawn_timer"] = 0
        kai = next(
            hero
            for hero in observation["frame_state"]["hero_states"]
            if hero["camp"] == 1 and hero["config_id"] == 193
        )
        kai["location"] = _raw_position(-25530, 12000, 1)

        processor = FeatureProcessV1(1)
        processor.process_feature(observation)
        macro = processor._memo_context["jungle_macro"]

        self.assertFalse(macro["kai_is_official"])
        self.assertEqual(macro["opening_camp_name"], "wolf")
        self.assertEqual(macro["opening_stage_index"], 0)

    def test_official_releases_after_first_lower_camp(self):
        processor = FeatureProcessV1(1)
        for config_id in (44, 42, 30):
            anchor = tuple(u.fixed_monster_anchor(config_id, "own", 1))
            processor.monster_camp_history[anchor] = {
                "ever_alive": True,
                "cleared_once": True,
                "live_cycle_seen": False,
                "cycle_seen_runtimes": {config_id * 10},
                "expected_members": 2,
                "clear_count": 1,
                "last_clear_frame": 50,
            }
        observation = make_observation(
            frame=1000, camp=1, include_soldiers=False
        )
        observation["frame_state"]["npc_states"].extend(
            [
                _monster(32, "own", 1, 701),
                _monster(33, "own", 1, 702),
            ]
        )
        for spawn in observation["frame_state"]["monster_spawn_infos"]:
            spawn["spawn_timer"] = 0
        kai = next(
            hero
            for hero in observation["frame_state"]["hero_states"]
            if hero["camp"] == 1 and hero["config_id"] == 193
        )
        kai["location"] = _raw_position(-21848, 5739, 1)

        processor.process_feature(observation)
        macro = processor._memo_context["jungle_macro"]

        self.assertFalse(macro["opening_rotation_active"])
        self.assertEqual(macro["opening_camp_name"], "")
        self.assertEqual(
            macro["opening_release_reason"],
            "protected_first_lower_cleared",
        )

    def test_official_post_pioneer_clear_enters_learned_macro(self):
        processor = FeatureProcessV1(1)
        for config_id in (44, 42, 30):
            anchor = tuple(u.fixed_monster_anchor(config_id, "own", 1))
            processor.monster_camp_history[anchor] = {
                "ever_alive": True,
                "cleared_once": True,
                "live_cycle_seen": False,
                "cycle_seen_runtimes": {config_id * 10},
                "expected_members": 2,
                "clear_count": 1,
                "last_clear_frame": 1364,
            }
        observation = make_observation(
            frame=1364, camp=1, include_soldiers=False
        )
        observation["frame_state"]["npc_states"].extend(
            [
                _monster(32, "own", 1, 701),
                _monster(33, "own", 1, 702),
            ]
        )
        for spawn in observation["frame_state"]["monster_spawn_infos"]:
            spawn["spawn_timer"] = 0
        kai = next(
            hero
            for hero in observation["frame_state"]["hero_states"]
            if hero["camp"] == 1 and hero["config_id"] == 193
        )
        # Returned trace position was 2,520 units from pioneer: only 20
        # outside the legacy radius. The confirmed pioneer clear must advance
        # directly to the next topology node.
        kai["location"] = _raw_position(-24205.0, -14954.0, 1)

        processor.process_feature(observation)
        macro = processor._memo_context["jungle_macro"]

        self.assertFalse(macro["opening_rotation_active"])
        self.assertEqual(macro["opening_camp_name"], "")
        self.assertEqual(
            macro["opening_release_reason"],
            "protected_first_lower_cleared",
        )

    def test_opening_releases_at_maximum_raw_frame(self):
        observation = make_observation(
            frame=KAI_OPENING_MAX_FRAME,
            camp=1,
            include_soldiers=False,
        )
        processor = FeatureProcessV1(1)
        processor.process_feature(observation)
        macro = processor._memo_context["jungle_macro"]

        self.assertFalse(macro["opening_rotation_active"])
        self.assertEqual(macro["opening_release_reason"], "opening_timeout")
        self.assertTrue(macro["lower_farm_started"])

    def test_unseen_external_wolf_clear_uses_timer_zero_to_positive(self):
        processor = FeatureProcessV1(1)
        spawned = make_observation(frame=100, camp=1)
        spawned["official_model_info"] = {
            "official": {"runtime_id": 11}
        }
        for spawn in spawned["frame_state"]["monster_spawn_infos"]:
            spawn["spawn_timer"] = 0
        processor.process_feature(spawned)

        externally_cleared = make_observation(frame=106, camp=1)
        externally_cleared["official_model_info"] = {
            "official": {"runtime_id": 11}
        }
        for spawn in externally_cleared["frame_state"][
            "monster_spawn_infos"
        ]:
            spawn["spawn_timer"] = (
                120000 if spawn["monster_id"] in (44, 45) else 0
            )
        processor.process_feature(externally_cleared)
        macro = processor._memo_context["jungle_macro"]
        wolf_state = processor.monster_camp_history[
            tuple(u.fixed_monster_anchor(44, "own", 1))
        ]
        self.assertEqual(wolf_state["cycle_seen_runtimes"], set())
        self.assertFalse(wolf_state["observer_near"])
        self.assertTrue(wolf_state["timer_death_transition"])
        self.assertTrue(wolf_state["cleared_once"])
        self.assertEqual(
            wolf_state["last_clear_evidence"],
            "spawn_timer_zero_to_positive",
        )
        self.assertEqual(macro["opening_camp_name"], "deer")

    def test_opening_advances_to_outer_archer_only_after_both_pioneer_members(self):
        processor = FeatureProcessV1(1)
        for config_id in (44, 42):
            anchor = tuple(u.fixed_monster_anchor(config_id, "own", 1))
            processor.monster_camp_history[anchor] = {
                "ever_alive": True,
                "cleared_once": True,
                "live_cycle_seen": False,
                "cycle_seen_runtimes": {config_id * 10, config_id * 10 + 1},
                "expected_members": 2,
                "clear_count": 1,
                "last_clear_frame": 50,
            }

        alive = make_observation(frame=100, camp=1, include_soldiers=False)
        alive["official_model_info"] = {"official": {"runtime_id": 11}}
        alive["frame_state"]["npc_states"].extend(
            [_monster(30, "own", 1, 601), _monster(31, "own", 1, 602)]
        )
        pioneer = u.side_position(u.fixed_monster_anchor(30, "own", 1), 1)
        kai = next(
            hero for hero in alive["frame_state"]["hero_states"]
            if hero["camp"] == 1 and hero["config_id"] == 193
        )
        kai["location"] = _raw_position(*pioneer, 1)
        processor.process_feature(alive)

        companion_only = make_observation(
            frame=106, camp=1, include_soldiers=False
        )
        companion_only["official_model_info"] = {
            "official": {"runtime_id": 11}
        }
        companion_only["frame_state"]["npc_states"].append(
            _monster(31, "own", 1, 602)
        )
        companion_kai = next(
            hero for hero in companion_only["frame_state"]["hero_states"]
            if hero["camp"] == 1 and hero["config_id"] == 193
        )
        companion_kai["location"] = _raw_position(*pioneer, 1)
        processor.process_feature(companion_only)
        macro = processor._memo_context["jungle_macro"]
        self.assertEqual(macro["opening_camp_name"], "pioneer")
        self.assertEqual(
            u.actor_runtime_id(macro["opening_target_actor"]), 602
        )

        transient_omission = make_observation(
            frame=112, camp=1, include_soldiers=False
        )
        transient_omission["official_model_info"] = {
            "official": {"runtime_id": 11}
        }
        for spawn in transient_omission["frame_state"]["monster_spawn_infos"]:
            spawn["spawn_timer"] = 0
        omitted_kai = next(
            hero
            for hero in transient_omission["frame_state"]["hero_states"]
            if hero["camp"] == 1 and hero["config_id"] == 193
        )
        omitted_kai["location"] = _raw_position(*pioneer, 1)
        processor.process_feature(transient_omission)
        macro = processor._memo_context["jungle_macro"]
        self.assertEqual(macro["opening_camp_name"], "pioneer")
        pioneer_state = processor.monster_camp_history[
            tuple(u.fixed_monster_anchor(30, "own", 1))
        ]
        self.assertFalse(pioneer_state["cleared_once"])
        self.assertFalse(pioneer_state["spawn_timer_positive"])

        cleared = make_observation(frame=118, camp=1, include_soldiers=False)
        cleared["official_model_info"] = {"official": {"runtime_id": 11}}
        for spawn in cleared["frame_state"]["monster_spawn_infos"]:
            spawn["spawn_timer"] = (
                60000 if spawn["monster_id"] in (30, 31) else 0
            )
        cleared_kai = next(
            hero for hero in cleared["frame_state"]["hero_states"]
            if hero["camp"] == 1 and hero["config_id"] == 193
        )
        cleared_kai["location"] = _raw_position(*pioneer, 1)
        processor.process_feature(cleared)
        macro = processor._memo_context["jungle_macro"]
        self.assertEqual(macro["opening_camp_name"], "outer_archer")
        self.assertTrue(macro["opening_rotation_active"])
        self.assertFalse(macro["camp_exit_active"])
        self.assertEqual(
            pioneer_state["last_clear_evidence"],
            "spawn_timer_zero_to_positive",
        )

    def test_inner_and_outer_archers_share_cfgs_but_not_physical_camp(self):
        observation = make_observation(
            frame=100, camp=1, include_soldiers=False
        )
        inner_major = _monster(32, "own", 1, 701)
        inner_minor = _monster(33, "own", 1, 702)
        inner_minor["location"] = {"x": -18460, "y": 0, "z": -27091}
        outer_major = _monster(32, "own", 1, 703)
        outer_major["location"] = {"x": -29860, "y": 0, "z": -24480}
        outer_minor = _monster(33, "own", 1, 704)
        observation["frame_state"]["npc_states"].extend(
            [inner_major, inner_minor, outer_major, outer_minor]
        )
        processor = FeatureProcessV1(1)
        processor.process_feature(observation)
        context = processor._memo_context
        inner = processor._opening_camp_status(
            context, "inner_archer"
        )
        outer = processor._opening_camp_status(
            context, "outer_archer"
        )
        self.assertEqual(
            {u.actor_config_id(actor) for actor in inner["actors"]},
            {32, 33},
        )
        self.assertEqual(
            {u.actor_config_id(actor) for actor in outer["actors"]},
            {32, 33},
        )
        self.assertEqual(
            {u.actor_runtime_id(actor) for actor in inner["actors"]},
            {701, 702},
        )
        self.assertEqual(
            {u.actor_runtime_id(actor) for actor in outer["actors"]},
            {703, 704},
        )

    def test_post_opening_archer_slots_follow_each_physical_camp(self):
        def archer(config_id, relation, runtime_id, camp_name):
            actor = _monster(config_id, relation, 1, runtime_id)
            anchor_config = 32 if camp_name == "inner" else 33
            anchor = u.fixed_monster_anchor(anchor_config, relation, 1)
            actor["location"] = {
                "x": anchor[0],
                "y": 0,
                "z": anchor[1],
            }
            return actor

        processor = FeatureProcessV1(1)
        processor.jungle_route_history["opening_released"] = True
        first = make_observation(
            frame=100, camp=1, include_soldiers=False
        )
        first["frame_state"]["npc_states"].extend(
            [
                archer(32, "own", 701, "inner"),
                archer(33, "own", 702, "inner"),
                archer(32, "own", 703, "outer"),
                archer(33, "own", 704, "outer"),
                archer(32, "enemy", 711, "inner"),
                archer(33, "enemy", 712, "inner"),
                archer(32, "enemy", 713, "outer"),
                archer(33, "enemy", 714, "outer"),
            ]
        )
        processor.process_feature(first)
        monsters = processor._memo_context["monsters"]
        self.assertEqual(u.actor_runtime_id(monsters[6]["actor"]), 701)
        self.assertEqual(u.actor_runtime_id(monsters[8]["actor"]), 703)
        self.assertEqual(u.actor_runtime_id(monsters[7]["actor"]), 711)
        self.assertEqual(u.actor_runtime_id(monsters[9]["actor"]), 713)

        # After the inner major disappears, its physical slot must advance to
        # the living inner companion instead of borrowing the outer cfg32.
        second = make_observation(
            frame=106, camp=1, include_soldiers=False
        )
        second["frame_state"]["npc_states"].extend(
            [
                archer(33, "own", 702, "inner"),
                archer(32, "own", 703, "outer"),
                archer(33, "own", 704, "outer"),
            ]
        )
        processor.process_feature(second)
        monsters = processor._memo_context["monsters"]
        self.assertEqual(monsters[6]["config_id"], 32)
        self.assertEqual(u.actor_runtime_id(monsters[6]["actor"]), 702)
        self.assertEqual(u.actor_runtime_id(monsters[8]["actor"]), 703)

        # An empty inner camp must remain empty even while the same cfg is
        # alive in the outer camp.
        third = make_observation(
            frame=112, camp=1, include_soldiers=False
        )
        third["frame_state"]["npc_states"].append(
            archer(32, "own", 703, "outer")
        )
        processor.process_feature(third)
        monsters = processor._memo_context["monsters"]
        self.assertIsNone(monsters[6]["actor"])
        self.assertEqual(u.actor_runtime_id(monsters[8]["actor"]), 703)

    def test_post_opening_official_kai_uses_position_for_inner_companion(self):
        processor = FeatureProcessV1(1)
        processor.jungle_route_history["opening_released"] = True
        observation = make_observation(
            frame=100, camp=1, include_soldiers=False
        )
        inner_anchor = u.fixed_monster_anchor(32, "own", 1)
        companion = _monster(33, "own", 1, 702)
        companion["location"] = {
            "x": inner_anchor[0],
            "y": 0,
            "z": inner_anchor[1],
        }
        observation["frame_state"]["npc_states"].append(companion)
        kai = next(
            hero
            for hero in observation["frame_state"]["hero_states"]
            if hero["camp"] == 1 and hero["config_id"] == 193
        )
        inner_side = u.side_position(inner_anchor, 1)
        kai["location"] = _raw_position(*inner_side, 1)

        processor.process_feature(observation)
        runtime_context = processor.command_runtime_context()
        masks = build_command_legal_masks_v1(runtime_context)
        self.assertFalse(
            processor._memo_context["jungle_macro"][
                "opening_rotation_active"
            ]
        )
        self.assertEqual(runtime_context.jungle_route_target_slot, 14)
        self.assertEqual(runtime_context.target_runtime_ids[14], 702)
        self.assertEqual(masks.target_by_intent[9, 14], 0.0)
        self.assertEqual(masks.position_by_intent[10, 16], 1.0)

    def test_post_opening_archer_resolution_is_camp_symmetric(self):
        for camp in (1, 2):
            with self.subTest(camp=camp):
                processor = FeatureProcessV1(camp)
                processor.jungle_route_history["opening_released"] = True
                observation = make_observation(
                    frame=100, camp=camp, include_soldiers=False
                )
                inner_anchor = u.fixed_monster_anchor(32, "own", camp)
                outer_anchor = u.fixed_monster_anchor(33, "own", camp)
                inner_companion = _monster(
                    33, "own", camp, 700 + camp
                )
                inner_companion["location"] = {
                    "x": inner_anchor[0],
                    "y": 0,
                    "z": inner_anchor[1],
                }
                outer_major = _monster(32, "own", camp, 710 + camp)
                outer_major["location"] = {
                    "x": outer_anchor[0],
                    "y": 0,
                    "z": outer_anchor[1],
                }
                observation["frame_state"]["npc_states"].extend(
                    [inner_companion, outer_major]
                )
                processor.process_feature(observation)
                monsters = processor._memo_context["monsters"]
                self.assertEqual(
                    u.actor_runtime_id(monsters[6]["actor"]), 700 + camp
                )
                self.assertEqual(
                    u.actor_runtime_id(monsters[8]["actor"]), 710 + camp
                )

    def test_kai_holds_wolf_anchor_until_first_spawn(self):
        observation = make_observation(
            frame=100, camp=1, include_soldiers=False
        )
        observation["official_model_info"] = {"official": {"runtime_id": 11}}
        wolf = u.side_position(u.fixed_monster_anchor(44, "own", 1), 1)
        kai = next(
            hero for hero in observation["frame_state"]["hero_states"]
            if hero["camp"] == 1 and hero["config_id"] == 193
        )
        kai["location"] = _raw_position(*wolf, 1)
        processor = FeatureProcessV1(1)
        processor.process_feature(observation)
        macro = processor._memo_context["jungle_macro"]
        plan = processor.jungle_opening_runtime_plans()[1]
        self.assertEqual(macro["opening_camp_name"], "wolf")
        self.assertEqual(macro["next_waypoint"], wolf)
        self.assertTrue(plan["wait_for_spawn"])

        result, diagnostics = JungleOpeningLegalizerV1().apply(
            self._raw_legal(), (141, 193),
            processor.jungle_opening_runtime_plans(), (),
        )
        self.assertEqual(np.flatnonzero(result[1, :13]).tolist(), [0])
        self.assertEqual(diagnostics[1]["reason"], "opening_spawn_wait")

    def test_opening_releases_only_after_red_statue_is_cleared(self):
        processor = FeatureProcessV1(1)
        for index, (name, _phase, _node, config_id, _members, expected) in enumerate(
            (
                ("wolf", "upper_wolf", 2, 44, (44, 45), 2),
                ("deer", "upper_deer", 1, 42, (42, 43), 2),
                ("pioneer", "lower_clear", 5, 30, (30, 31), 2),
                ("outer_archer", "lower_clear", 7, 33, (32, 33), 2),
                ("inner_archer", "lower_clear", 8, 32, (32, 33), 2),
            )
        ):
            anchor = tuple(u.fixed_monster_anchor(config_id, "own", 1))
            processor.monster_camp_history[anchor] = {
                "ever_alive": True,
                "cleared_once": True,
                "live_cycle_seen": False,
                "cycle_seen_runtimes": {
                    800 + index * 2,
                    801 + index * 2,
                },
                "expected_members": expected,
                "clear_count": 1,
                "last_clear_frame": 50,
            }
        red = make_observation(frame=100, camp=1, include_soldiers=False)
        red["official_model_info"] = {"official": {"runtime_id": 11}}
        red["frame_state"]["npc_states"].append(
            _monster(49, "own", 1, 901)
        )
        for spawn in red["frame_state"]["monster_spawn_infos"]:
            spawn["spawn_timer"] = 0
        red_anchor = u.side_position(u.fixed_monster_anchor(49, "own", 1), 1)
        red_kai = next(
            hero for hero in red["frame_state"]["hero_states"]
            if hero["camp"] == 1 and hero["config_id"] == 193
        )
        red_kai["location"] = _raw_position(*red_anchor, 1)
        processor.process_feature(red)
        macro = processor._memo_context["jungle_macro"]
        self.assertEqual(macro["opening_camp_name"], "red_statue")
        self.assertTrue(macro["opening_rotation_active"])

        gone = make_observation(frame=106, camp=1, include_soldiers=False)
        gone["official_model_info"] = {"official": {"runtime_id": 11}}
        for spawn in gone["frame_state"]["monster_spawn_infos"]:
            spawn["spawn_timer"] = (
                60000 if spawn["monster_id"] == 49 else 0
            )
        gone_kai = next(
            hero for hero in gone["frame_state"]["hero_states"]
            if hero["camp"] == 1 and hero["config_id"] == 193
        )
        gone_kai["location"] = _raw_position(*red_anchor, 1)
        processor.process_feature(gone)
        macro = processor._memo_context["jungle_macro"]
        self.assertFalse(macro["opening_rotation_active"])
        self.assertEqual(macro["opening_release_reason"], "six_camps_cleared")


if __name__ == "__main__":
    unittest.main()
