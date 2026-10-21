#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""Exact PPO 4.0 legality, state-latch and reward-boundary tests."""

import copy
from collections import deque
from dataclasses import replace
import math
import unittest

import numpy as np

from agent_ppo.conf.conf import FrozenV1Config, V1TrainingConfig
from agent_ppo.feature.collaboration_v1 import (
    CommandRuleLegalizerV1,
    build_command_legal_masks_v1,
)
from agent_ppo.feature.feature_process import v1_utils as u
from agent_ppo.feature.definition_v1 import late_monster_reward_offsets_v1
from agent_ppo.feature.feature_process.v1_feature_process import (
    FeatureProcessV1,
    JUNGLE_CAMP_CLEAR_NEAR_DISTANCE,
    KAI_OPENING_CAMP_ORDER,
    TIMER_CAMP_WAIT_RAW_FRAMES,
)
from agent_ppo.feature.v40_legalizers import (
    LateConversionLegalizerV1,
    RespawnReentryLegalizerV1,
    TyrantSmiteLegalizerV1,
    TARGET_MATRIX_OFFSET,
)
from agent_ppo.tests.test_collaboration_v1 import _context
from agent_ppo.tests.test_feature_process_v1 import _monster, make_observation


def _raw_legal():
    return np.ones(
        (
            FrozenV1Config.CONTROLLED_ACTOR_COUNT,
            FrozenV1Config.RAW_LEGAL_ACTION_DIM,
        ),
        dtype=np.float32,
    )


def _targets(mask, actor_index=0):
    return mask[actor_index, TARGET_MATRIX_OFFSET:].reshape(13, 39)


def _set_side_position(actor, side_x, side_z, camp):
    raw_x = side_x if camp == 1 else -side_x
    actor["location"] = {
        "x": raw_x,
        "y": 0,
        "z": side_z + u.LANE_CENTER_Z,
    }


def _set_hp(actor, ratio):
    actor["hp"] = float(actor["max_hp"]) * float(ratio)


class V40OrdinaryLegalizerTest(unittest.TestCase):
    def test_tyrant_smite_lock_open_force_and_pending_are_fail_safe(self):
        legalizer = TyrantSmiteLegalizerV1()
        base = _raw_legal()
        plans = (
            {"runtime_id": 13, "active": True, "lock": True},
            {},
        )
        locked, diagnostics = legalizer.apply(base, (193, 132), plans)
        self.assertEqual(_targets(locked)[8, 27], 0.0)
        self.assertEqual(diagnostics[0]["state"], "lock")
        self.assertTrue(np.any(locked[0, :13] > 0.5))

        opened, diagnostics = legalizer.apply(
            base,
            (193, 132),
            ({"active": True, "lock": False}, {}),
        )
        self.assertEqual(_targets(opened)[8, 27], 1.0)
        self.assertEqual(diagnostics[0]["state"], "open")

        forced, diagnostics = legalizer.apply(
            base,
            (193, 132),
            ({"active": True, "force": True}, {}),
        )
        np.testing.assert_array_equal(
            forced[0, :13],
            np.eye(1, 13, 8, dtype=np.float32).reshape(-1),
        )
        self.assertEqual(np.count_nonzero(_targets(forced)[8]), 1)
        self.assertEqual(_targets(forced)[8, 27], 1.0)
        self.assertEqual(diagnostics[0]["state"], "force")

        unavailable = base.copy()
        _targets(unavailable)[8, 27] = 0.0
        pending, diagnostics = legalizer.apply(
            unavailable,
            (193, 132),
            ({"active": True, "force": True}, {}),
        )
        np.testing.assert_array_equal(pending, unavailable)
        self.assertEqual(diagnostics[0]["state"], "force_pending")
        self.assertTrue(np.any(pending[0, :13] > 0.5))

    def test_tyrant_smite_runtime_plan_honors_exact_hp_and_enemy_radius_boundaries_on_both_camps(self):
        for camp in (1, 2):
            with self.subTest(camp=camp):
                processor = FeatureProcessV1(camp)
                processor.jungle_route_history["opening_released"] = True
                processor.jungle_route_history["opening_release_reason"] = (
                    "six_camps_cleared"
                )
                observation = make_observation(frame=5000, camp=camp)
                observation["official_model_info"] = {
                    "official": {"runtime_id": 11}
                }
                kai = next(
                    hero
                    for hero in observation["frame_state"]["hero_states"]
                    if hero["camp"] == camp and hero["config_id"] == 193
                )
                kai["skill_state"]["slot_states"][5]["config_id"] = 80104
                _set_hp(kai, 0.099)
                tyrant = _monster(122, "center", camp, 501)
                tyrant["hp"] = 3000.0
                tyrant["max_hp"] = 12000.0
                observation["frame_state"]["npc_states"].append(tyrant)
                tyrant_side = u.side_position(
                    u.fixed_monster_anchor(122, "center", camp), camp
                )
                enemy_kai = next(
                    hero
                    for hero in observation["frame_state"]["hero_states"]
                    if hero["camp"] != camp and hero["config_id"] == 193
                )
                _set_side_position(
                    enemy_kai,
                    tyrant_side[0] + 10000.0,
                    tyrant_side[1],
                    camp,
                )
                processor.process_feature(observation)
                plan = next(
                    item
                    for item in processor.tyrant_smite_runtime_plans()
                    if item["runtime_id"] == kai["runtime_id"]
                )
                self.assertTrue(plan["active"])
                self.assertTrue(plan["lock"])
                self.assertFalse(plan["force"])

                outside = copy.deepcopy(observation)
                outside["frame_state"]["frame_no"] = 5006
                outside_enemy_kai = next(
                    hero
                    for hero in outside["frame_state"]["hero_states"]
                    if hero["camp"] != camp
                    and hero["config_id"] == 193
                )
                _set_side_position(
                    outside_enemy_kai,
                    tyrant_side[0] + 10000.001,
                    tyrant_side[1],
                    camp,
                )
                processor.process_feature(outside)
                plan = next(
                    item
                    for item in processor.tyrant_smite_runtime_plans()
                    if item["runtime_id"] == kai["runtime_id"]
                )
                self.assertFalse(plan["lock"])

                lethal = copy.deepcopy(outside)
                lethal["frame_state"]["frame_no"] = 5012
                lethal_tyrant = next(
                    actor
                    for actor in lethal["frame_state"]["npc_states"]
                    if actor.get("config_id") == 122
                )
                lethal_tyrant["hp"] = 2000.0
                processor.process_feature(lethal)
                plan = next(
                    item
                    for item in processor.tyrant_smite_runtime_plans()
                    if item["runtime_id"] == kai["runtime_id"]
                )
                self.assertTrue(plan["force"])
                self.assertFalse(plan["lock"])

    def test_tyrant_smite_unknown_and_nonfinite_hp_fail_open(self):
        processor = FeatureProcessV1(1)
        processor.jungle_route_history["opening_released"] = True
        processor.jungle_route_history["opening_release_reason"] = (
            "six_camps_cleared"
        )
        observation = make_observation(frame=5000, camp=1)
        observation["official_model_info"] = {"official": {"runtime_id": 11}}
        kai = next(
            hero
            for hero in observation["frame_state"]["hero_states"]
            if hero["camp"] == 1 and hero["config_id"] == 193
        )
        kai["skill_state"]["slot_states"][5]["config_id"] = 80104
        tyrant = _monster(122, "center", 1, 501)
        tyrant["hp"] = 3000.0
        tyrant["max_hp"] = 12000.0
        observation["frame_state"]["npc_states"].append(tyrant)
        processor.process_feature(observation)
        cached_tyrant = processor._memo_context["tyrant"]["actor"]

        cases = (
            ("missing", None),
            ("none", None),
            ("nan", float("nan")),
            ("positive_inf", float("inf")),
            ("negative", -0.001),
        )
        for name, raw_hp in cases:
            with self.subTest(name=name):
                if name == "missing":
                    cached_tyrant.pop("hp", None)
                else:
                    cached_tyrant["hp"] = raw_hp
                plan = next(
                    item
                    for item in processor.tyrant_smite_runtime_plans()
                    if item["runtime_id"] == kai["runtime_id"]
                )
                self.assertFalse(plan["tyrant_hp_known"])
                self.assertFalse(plan["active"])
                self.assertFalse(plan["lock"])
                self.assertFalse(plan["force"])

        for raw_hp, expected_force in ((2000.0, True), (2000.0 + 1e-6, False)):
            with self.subTest(raw_hp=raw_hp):
                cached_tyrant["hp"] = raw_hp
                plan = next(
                    item
                    for item in processor.tyrant_smite_runtime_plans()
                    if item["runtime_id"] == kai["runtime_id"]
                )
                self.assertTrue(plan["tyrant_hp_known"])
                self.assertTrue(plan["active"])
                self.assertEqual(plan["force"], expected_force)

                legal = _raw_legal()
                if expected_force:
                    result, diagnostics = TyrantSmiteLegalizerV1().apply(
                        legal,
                        (193, 132),
                        (plan, {}),
                    )
                    self.assertEqual(diagnostics[0]["state"], "force")
                    self.assertEqual(_targets(result)[8, 27], 1.0)

                    _targets(legal)[8, 27] = 0.0
                    result, diagnostics = TyrantSmiteLegalizerV1().apply(
                        legal,
                        (193, 132),
                        (plan, {}),
                    )
                    self.assertEqual(diagnostics[0]["state"], "force_pending")
                    np.testing.assert_array_equal(result, legal)

    def test_respawn_reentry_uses_semantic_monster_slot_and_never_zeroes(self):
        processor = FeatureProcessV1(1)
        processor._memo_context = {
            "controlled_infos": (
                {"config_id": 193, "runtime_id": 13},
                {"config_id": 132, "runtime_id": 11},
            ),
            "jungle_macro": {
                "target_slot": 2,
                "respawn_reentry_active": True,
                "kai_is_official": False,
                "next_waypoint_dx": 1000.0,
                "next_waypoint_dz": 0.0,
                "timer_wait_active": False,
            },
        }
        plans = processor.respawn_reentry_runtime_plans()
        self.assertEqual(plans[0]["target_slot"], 10)

        result, diagnostics = RespawnReentryLegalizerV1().apply(
            _raw_legal(), (193, 132), plans
        )
        self.assertEqual(diagnostics[0]["reason"], "engage_camp")
        self.assertEqual(_targets(result)[3, 10], 1.0)
        self.assertEqual(np.count_nonzero(_targets(result)[3]), 1)
        self.assertTrue(np.any(result[0, :13] > 0.5))

    def test_late_conversion_blocks_only_farming_and_preserves_self_defense(self):
        base = _raw_legal()
        result, _diagnostics = LateConversionLegalizerV1().apply(
            base,
            (193, 132),
            (
                {
                    "active": True,
                    "self_defense_slots": (10,),
                    "enemy_attacking_tyrant": False,
                    "destination_valid": False,
                },
                {},
            ),
        )
        target = _targets(result)
        self.assertTrue(np.all(target[:, 8:10] == 0.0))
        self.assertTrue(np.all(target[:, 11:28] == 0.0))
        self.assertTrue(np.all(target[:, 10] == 1.0))
        self.assertTrue(np.all(target[:, 1:4] == 1.0))
        self.assertTrue(np.all(target[:, 28:39] == 1.0))
        self.assertTrue(np.any(result[0, :13] > 0.5))

    def test_late_destination_8000_boundary_and_three_direction_fan(self):
        plan = {
            "active": True,
            "self_defense_slots": (),
            "enemy_attacking_tyrant": False,
            "destination_valid": True,
            "destination_distance": 8000.0,
            "destination_dx": 1000.0,
            "destination_dz": 0.0,
            "combat_target_available": False,
        }
        exact, diagnostics = LateConversionLegalizerV1().apply(
            _raw_legal(), (193, 132), (plan, {})
        )
        self.assertEqual(diagnostics[0]["selected_directions"], ())
        self.assertGreater(np.count_nonzero(exact[0, :13]), 1)

        beyond = dict(plan, destination_distance=8000.001)
        moved, diagnostics = LateConversionLegalizerV1().apply(
            _raw_legal(), (193, 132), (beyond, {})
        )
        self.assertEqual(np.flatnonzero(moved[0, :13]).tolist(), [2])
        self.assertLessEqual(len(diagnostics[0]["selected_directions"]), 3)
        self.assertGreater(len(diagnostics[0]["selected_directions"]), 0)


class V40RuntimeStateTest(unittest.TestCase):
    @staticmethod
    def _official_timer_context(spawn_timer_ms=2000):
        processor = FeatureProcessV1(1)
        processor.process_feature(make_observation(frame=100, camp=1))
        context = processor._memo_context
        monster = context["monsters"][0]
        anchor = tuple(float(value) for value in monster["anchor"])
        camp_state = processor.monster_camp_history[anchor]
        camp_state.update(
            cleared_once=True,
            clear_count=1,
            lifecycle_id=1,
            spawn_timer_ms=float(spawn_timer_ms),
            spawn_timer_positive=spawn_timer_ms > 0,
            alive_runtime_ids=(),
        )
        context["monster_spawn"][0]["spawn_timer"] = spawn_timer_ms
        return processor, context, anchor

    def test_official_timer_task_accepts_positive_timer_and_waits_exactly_three_seconds(self):
        processor, context, anchor = self._official_timer_context()
        self.assertTrue(
            processor._start_official_farm_task(
                100, 10, context=context
            )
        )
        task = processor.official_task_state
        self.assertTrue(task["timer_navigation"])
        self.assertTrue(task["start_timer_was_positive"])
        self.assertEqual(task["start_clear_count"], 1)
        self.assertEqual(task["start_navigation_generation_id"], 1)
        self.assertEqual(task["start_spawn_seconds"], 2.0)

        goal = u.side_position(anchor, 1)
        official = context["official_hero"]
        official["side_position"] = goal
        _set_side_position(official["actor"], *goal, 1)
        context["frame_no"] = 106
        processor._update_official_task_state(context)
        self.assertIsNotNone(processor.official_task_state)
        self.assertEqual(
            processor.official_task_state["phase"], "timer_wait"
        )

        context["frame_no"] = 151
        processor._update_official_task_state(context)
        self.assertIsNotNone(processor.official_task_state)

        context["frame_no"] = 152
        processor._update_official_task_state(context)
        self.assertIsNone(processor.official_task_state)
        self.assertEqual(
            processor.official_task_counters[
                "task_end_timer_wait_timeout"
            ],
            1.0,
        )
        self.assertEqual(
            processor.official_task_counters["failed_lifecycle_lock"],
            1.0,
        )

    def test_official_timer_task_rejects_excess_wait_and_invalidates_new_clear_or_upward_jump(self):
        processor, context, _anchor = self._official_timer_context(
            spawn_timer_ms=120000
        )
        self.assertFalse(
            processor._start_official_farm_task(
                100, 10, context=context
            )
        )

        processor, context, anchor = self._official_timer_context()
        self.assertTrue(
            processor._start_official_farm_task(
                100, 10, context=context
            )
        )
        processor.monster_camp_history[anchor]["clear_count"] = 2
        context["frame_no"] = 106
        processor._update_official_task_state(context)
        self.assertIsNone(processor.official_task_state)

    def test_official_timer_task_rewards_authoritative_clear_only_after_real_engagement(self):
        processor, context, anchor = self._official_timer_context()
        self.assertTrue(
            processor._start_official_farm_task(
                100, 10, context=context
            )
        )
        processor.official_task_state["engaged"] = True
        processor.official_task_state["first_engagement_frame"] = 104
        processor.monster_camp_history[anchor]["clear_count"] = 2
        processor.monster_camp_history[anchor]["last_clear_frame"] = 106
        context["frame_no"] = 106
        processor._update_official_task_state(context)
        self.assertIsNone(processor.official_task_state)
        self.assertGreater(processor.official_task_reward_pending, 0.0)
        self.assertEqual(
            processor.v40_counters["official_timer_farm_completion"],
            1.0,
        )

    @staticmethod
    def _controlled_timer_observation(frame, kai_position=None):
        observation = make_observation(frame=frame, camp=1)
        observation["official_model_info"] = {
            "official": {"runtime_id": 11}
        }
        for spawn in observation["frame_state"]["monster_spawn_infos"]:
            spawn["spawn_timer"] = 300000
        observation["frame_state"]["monster_spawn_infos"][14][
            "spawn_timer"
        ] = 0
        observation["frame_state"]["monster_spawn_infos"][16][
            "spawn_timer"
        ] = 0
        if kai_position is not None:
            kai = next(
                actor
                for actor in observation["frame_state"]["hero_states"]
                if actor["camp"] == 1 and actor["config_id"] == 193
            )
            _set_side_position(kai, *kai_position, 1)
        for info in observation["heroes"]:
            legal = np.zeros(
                FrozenV1Config.RAW_LEGAL_ACTION_DIM, dtype=np.float32
            )
            legal[:122] = 1.0
            legal[122:].reshape(13, 39)[:, 0] = 1.0
            info["legal_action"] = legal
        return observation

    def test_controlled_timer_route_exposes_feature_gradient_discount_and_progress_exemption(self):
        processor = FeatureProcessV1(1)
        warm = self._controlled_timer_observation(1000)
        processor.process_feature(warm)
        history = processor.jungle_route_history
        history["opening_released"] = True
        history["opening_release_reason"] = "six_camps_cleared"
        for monster in processor._memo_context["monsters"]:
            if monster["relation"] != "own":
                continue
            anchor = tuple(float(value) for value in monster["anchor"])
            processor.monster_camp_history[anchor].update(
                cleared_once=True,
                clear_count=1,
            )

        first = self._controlled_timer_observation(1006)
        processor.process_feature(first)
        first_macro = processor._memo_context["jungle_macro"]
        self.assertEqual(first_macro["phase"], "timer_navigation")
        self.assertTrue(first_macro["timer_navigation_active"])
        self.assertEqual(first_macro["route_shaping_reward"], 0.0)
        jungle_fields = processor._build_jungle_plan(
            processor._memo_context,
            next(
                hero
                for hero in processor._memo_context["friend_heroes"]
                if u.actor_config_id(hero["actor"]) == 193
            ),
        )[0]
        from agent_ppo.feature import schema_v1

        control_index = schema_v1.JUNGLE_PLAN_FIELD_MAP[
            "phase_control_cycle"
        ].start
        self.assertEqual(jungle_fields[control_index], 1.0)

        kai_position = next(
            hero["side_position"]
            for hero in processor._memo_context["friend_heroes"]
            if u.actor_config_id(hero["actor"]) == 193
        )
        waypoint = first_macro["next_waypoint"]
        dx = waypoint[0] - kai_position[0]
        dz = waypoint[1] - kai_position[1]
        norm = math.hypot(dx, dz)
        closer_position = (
            kai_position[0] + dx / norm * 1000.0,
            kai_position[1] + dz / norm * 1000.0,
        )
        closer = self._controlled_timer_observation(
            1012, closer_position
        )
        processor.process_feature(closer)
        closer_macro = processor._memo_context["jungle_macro"]
        self.assertGreater(closer_macro["route_shaping_reward"], 0.0)
        self.assertTrue(closer_macro["route_progress_confirmed"])
        behavior = processor.consume_behavior_shaping(1012)
        kai_index = tuple(
            int(info["config_id"])
            for info in processor._memo_context["controlled_infos"]
        ).index(193)
        self.assertEqual(behavior["no_target_counts"][kai_index], 0)
        self.assertEqual(behavior["strategic_wait_counts"][kai_index], 1)

        phi = float(history["timer_route"]["phi"])
        stationary = self._controlled_timer_observation(
            1018, closer_position
        )
        processor.process_feature(stationary)
        stationary_reward = processor._memo_context["jungle_macro"][
            "route_shaping_reward"
        ]
        self.assertAlmostEqual(
            stationary_reward,
            (V1TrainingConfig.GAMMA - 1.0) * phi,
            places=7,
        )

        processor, context, anchor = self._official_timer_context()
        self.assertTrue(
            processor._start_official_farm_task(
                100, 10, context=context
            )
        )
        processor.monster_camp_history[anchor]["spawn_timer_ms"] = 3100.0
        context["frame_no"] = 106
        processor._update_official_task_state(context)
        self.assertIsNone(processor.official_task_state)

    def _released_controlled_timer_processor(self, frame=1000):
        processor = FeatureProcessV1(1)
        processor.process_feature(self._controlled_timer_observation(frame))
        history = processor.jungle_route_history
        history["opening_released"] = True
        history["opening_release_reason"] = "six_camps_cleared"
        for monster in processor._memo_context["monsters"]:
            if monster["relation"] != "own":
                continue
            anchor = tuple(float(value) for value in monster["anchor"])
            processor.monster_camp_history[anchor].update(
                cleared_once=True,
                clear_count=1,
            )
        return processor

    def test_timer_candidates_use_live_or_timer_union_with_exact_wait_boundary(self):
        processor = FeatureProcessV1(1)
        processor.process_feature(make_observation(frame=1000, camp=1))
        context = processor._memo_context
        kai = next(
            hero
            for hero in context["friend_heroes"]
            if u.actor_config_id(hero["actor"]) == 193
        )
        opening = {
            name: processor._opening_camp_status(
                context, name, kai["side_position"]
            )
            for name, *_rest in KAI_OPENING_CAMP_ORDER
        }
        camp_name = next(
            name for name, status in opening.items() if status["target_slot"] is not None
        )
        status = opening[camp_name]
        slot = int(status["target_slot"])
        anchor = tuple(float(value) for value in status["anchor"])
        camp_state = processor.monster_camp_history[anchor]
        member_slots = processor._camp_member_slots(context, slot)

        camp_state.update(
            cleared_once=False,
            clear_count=0,
            alive_runtime_ids=(9001,),
        )
        status["alive_actors"] = ({"runtime_id": 9001},)
        for member_slot in member_slots:
            context["monster_spawn"][member_slot]["spawn_timer"] = float("nan")
        candidates = processor._v40_timer_candidates(context, kai, opening)
        live = next(item for item in candidates if item["camp_name"] == camp_name)
        self.assertTrue(live["alive"])
        self.assertEqual(live["spawn_seconds"], 0.0)

        camp_state["alive_runtime_ids"] = ()
        status["alive_actors"] = ()
        camp_state["cleared_once"] = True
        for member_slot in member_slots:
            context["monster_spawn"][member_slot]["spawn_timer"] = 0.0
        candidates = processor._v40_timer_candidates(context, kai, opening)
        timer = next(item for item in candidates if item["camp_name"] == camp_name)
        self.assertFalse(timer["alive"])

        exact_ms = (timer["arrival_eta"] + 3.0) * 1000.0
        for member_slot in member_slots:
            context["monster_spawn"][member_slot]["spawn_timer"] = exact_ms
        candidates = processor._v40_timer_candidates(context, kai, opening)
        self.assertTrue(any(item["camp_name"] == camp_name for item in candidates))
        for member_slot in member_slots:
            context["monster_spawn"][member_slot]["spawn_timer"] = exact_ms + 0.001
        candidates = processor._v40_timer_candidates(context, kai, opening)
        self.assertFalse(any(item["camp_name"] == camp_name for item in candidates))

        camp_state["cleared_once"] = False
        for member_slot in member_slots:
            context["monster_spawn"][member_slot]["spawn_timer"] = 1.0
        candidates = processor._v40_timer_candidates(context, kai, opening)
        self.assertFalse(any(item["camp_name"] == camp_name for item in candidates))

    def test_controlled_timer_wait_has_absolute_deadline_and_route_stuck_lock(self):
        processor = self._released_controlled_timer_processor()
        processor.process_feature(self._controlled_timer_observation(1006))
        first_macro = processor._memo_context["jungle_macro"]
        slot = int(first_macro["target_slot"])
        anchor = u.side_position(
            processor._memo_context["monsters"][slot]["anchor"], 1
        )

        processor.process_feature(
            self._controlled_timer_observation(1012, anchor)
        )
        route = processor.jungle_route_history["timer_route"]
        identity = tuple(route["identity"])
        self.assertTrue(processor._memo_context["jungle_macro"]["timer_wait_active"])
        self.assertEqual(route["wait_deadline_frame"], 1012 + TIMER_CAMP_WAIT_RAW_FRAMES)

        before = 1012 + TIMER_CAMP_WAIT_RAW_FRAMES - 1
        processor.jungle_route_history["frame"] = before - 6
        processor.process_feature(
            self._controlled_timer_observation(before, anchor)
        )
        self.assertTrue(processor._memo_context["jungle_macro"]["timer_wait_active"])

        deadline = 1012 + TIMER_CAMP_WAIT_RAW_FRAMES
        processor.process_feature(
            self._controlled_timer_observation(deadline, anchor)
        )
        self.assertFalse(processor._memo_context["jungle_macro"]["timer_wait_active"])
        self.assertIsNone(processor.jungle_route_history["timer_route"])
        self.assertIn(
            identity,
            processor.jungle_route_history["timer_route_failed_generations"],
        )

        stuck_processor = self._released_controlled_timer_processor()
        stationary = self._controlled_timer_observation(1006)
        stuck_processor.process_feature(stationary)
        stuck_route = stuck_processor.jungle_route_history["timer_route"]
        stuck_identity = tuple(stuck_route["identity"])
        stuck_processor.jungle_route_history["stuck_frames"] = 10_000
        stuck_processor.jungle_route_history["no_progress_frames"] = 10_000
        stuck_processor.process_feature(
            self._controlled_timer_observation(1012)
        )
        self.assertTrue(stuck_processor._memo_context["jungle_macro"]["route_stuck"])
        self.assertIsNone(stuck_processor.jungle_route_history["timer_route"])
        self.assertIn(
            stuck_identity,
            stuck_processor.jungle_route_history[
                "timer_route_failed_generations"
            ],
        )

    def test_live_camp_atomically_takes_over_an_active_empty_wait(self):
        processor = self._released_controlled_timer_processor()
        processor.process_feature(self._controlled_timer_observation(1006))
        slot = int(processor._memo_context["jungle_macro"]["target_slot"])
        target = processor._memo_context["monsters"][slot]
        anchor = u.side_position(target["anchor"], 1)
        processor.process_feature(
            self._controlled_timer_observation(1012, anchor)
        )
        self.assertIsNotNone(processor.jungle_route_history["timer_route"])

        live = self._controlled_timer_observation(1018, anchor)
        live["frame_state"]["npc_states"].append(
            _monster(target["config_id"], target["relation"], 1, 9001)
        )
        processor.process_feature(live)
        macro = processor._memo_context["jungle_macro"]
        self.assertFalse(macro["timer_wait_active"])
        self.assertFalse(macro["timer_navigation_active"])
        self.assertIsNone(processor.jungle_route_history["timer_route"])
        self.assertEqual(processor.v40_counters["timer_route_live_takeover"], 1.0)

    def test_official_respawn_task_uses_outer_deadline_and_generation_lock(self):
        processor, context, anchor = self._official_timer_context()
        self.assertTrue(
            processor._start_official_farm_task(
                390,
                10,
                context=context,
                forced=True,
                respawn_epoch=7,
                absolute_deadline_frame=400,
            )
        )
        context["frame_no"] = 399
        processor._update_official_task_state(context)
        self.assertIsNotNone(processor.official_task_state)
        context["frame_no"] = 400
        processor._update_official_task_state(context)
        self.assertIsNone(processor.official_task_state)
        self.assertEqual(
            processor.official_task_counters["task_end_respawn_deadline"],
            1.0,
        )
        self.assertFalse(
            processor._start_official_farm_task(
                401,
                10,
                context=context,
                forced=True,
                respawn_epoch=7,
                absolute_deadline_frame=500,
            )
        )

        processor.monster_camp_history[anchor]["clear_count"] = 2
        self.assertTrue(
            processor._start_official_farm_task(
                402,
                10,
                context=context,
                forced=True,
                respawn_epoch=8,
                absolute_deadline_frame=500,
            )
        )
        context["official_hero"]["alive"] = False
        context["frame_no"] = 403
        processor._update_official_task_state(context)
        self.assertIsNone(processor.official_task_state)

    def test_timer_potential_uses_one_gamma_per_stored_transition(self):
        rewards = []
        phis = []
        for gap in (6, 12, 18, 7):
            processor = self._released_controlled_timer_processor()
            processor.process_feature(self._controlled_timer_observation(1006))
            kai_position = next(
                hero["side_position"]
                for hero in processor._memo_context["friend_heroes"]
                if u.actor_config_id(hero["actor"]) == 193
            )
            waypoint = processor._memo_context["jungle_macro"]["next_waypoint"]
            dx = waypoint[0] - kai_position[0]
            dz = waypoint[1] - kai_position[1]
            norm = math.hypot(dx, dz)
            closer = (
                kai_position[0] + dx / norm * 1000.0,
                kai_position[1] + dz / norm * 1000.0,
            )
            processor.process_feature(
                self._controlled_timer_observation(1006 + gap, closer)
            )
            macro = processor._memo_context["jungle_macro"]
            phi = float(processor.jungle_route_history["timer_route"]["phi"])
            rewards.append(float(macro["route_shaping_reward"]))
            phis.append(phi)
            self.assertAlmostEqual(
                macro["route_shaping_reward"],
                V1TrainingConfig.GAMMA * phi,
                places=7,
            )
        for reward in rewards[1:]:
            self.assertAlmostEqual(reward, rewards[0], places=7)
        for phi in phis[1:]:
            self.assertAlmostEqual(phi, phis[0], places=7)

    def test_timer_potential_ignores_repeated_consumption_and_backward_frames(self):
        processor = self._released_controlled_timer_processor()
        processor.process_feature(self._controlled_timer_observation(1006))
        kai_position = next(
            hero["side_position"]
            for hero in processor._memo_context["friend_heroes"]
            if u.actor_config_id(hero["actor"]) == 193
        )
        waypoint = processor._memo_context["jungle_macro"]["next_waypoint"]
        dx = waypoint[0] - kai_position[0]
        dz = waypoint[1] - kai_position[1]
        norm = math.hypot(dx, dz)
        closer = (
            kai_position[0] + dx / norm * 1000.0,
            kai_position[1] + dz / norm * 1000.0,
        )
        processor.process_feature(self._controlled_timer_observation(1012, closer))
        route = processor.jungle_route_history["timer_route"]
        self.assertGreater(float(route["phi"]), 0.0)
        self.assertGreater(
            max(processor.consume_jungle_route_shaping(1012)),
            0.0,
        )
        self.assertEqual(processor.consume_jungle_route_shaping(1012), (0.0, 0.0))

        backward_closer = (
            kai_position[0] + dx / norm * 1500.0,
            kai_position[1] + dz / norm * 1500.0,
        )
        processor.process_feature(
            self._controlled_timer_observation(1011, backward_closer)
        )
        route = processor.jungle_route_history["timer_route"]
        self.assertEqual(
            processor._memo_context["jungle_macro"]["route_shaping_reward"],
            0.0,
        )
        self.assertIsNone(route)
        self.assertEqual(processor.consume_jungle_route_shaping(1011), (0.0, 0.0))

    def test_real_engagement_requires_actor_attributable_current_camp_evidence(self):
        processor = FeatureProcessV1(1)
        observation = make_observation(frame=1000, camp=1)
        monster = _monster(49, "own", 1, 9001)
        observation["frame_state"]["npc_states"].append(monster)
        processor.process_feature(observation)
        context = processor._memo_context
        kai = next(
            hero
            for hero in context["friend_heroes"]
            if u.actor_config_id(hero["actor"]) == 193
        )
        anchor = tuple(u.fixed_monster_anchor(49, "own", 1))
        camp_state = processor.monster_camp_history[anchor]
        alive = {9001}
        kai["actor"]["attack_target"] = 0
        camp_state["member_hp_drop_by_runtime"] = {}
        self.assertFalse(
            processor._camp_real_engagement(context, kai, alive, camp_state, 1000)
        )

        kai["actor"]["attack_target"] = 9001
        self.assertTrue(
            processor._camp_real_engagement(context, kai, alive, camp_state, 1000)
        )
        kai["actor"]["attack_target"] = 0

        key = ("friend", 193)
        processor.recent_events[key].append(
            {"frame": 1000, "type": "hit_monster", "other_runtime": 9001}
        )
        self.assertTrue(
            processor._camp_real_engagement(context, kai, alive, camp_state, 1000)
        )
        processor.recent_events[key].clear()

        camp_state["member_hp_drop_by_runtime"] = {9001: 100.0}
        context["actor_by_runtime"][9001]["single_hurt_list"] = [
            {"runtime_id": 11, "frame_no": 1000}
        ]
        self.assertFalse(
            processor._camp_real_engagement(context, kai, alive, camp_state, 1000)
        )
        context["actor_by_runtime"][9001]["single_hurt_list"] = [
            {
                "runtime_id": u.actor_runtime_id(kai["actor"]),
                "frame_no": 1000,
            }
        ]
        self.assertTrue(
            processor._camp_real_engagement(context, kai, alive, camp_state, 1000)
        )
        self.assertFalse(
            processor._camp_real_engagement(
                context, kai, {9002}, camp_state, 1000
            )
        )

    def test_late_teamfight_accepts_incoming_or_outgoing_evidence_only_in_window(self):
        for camp in (1, 2):
            for event_type in ("hurt_by_hero", "hit_hero"):
                with self.subTest(camp=camp, event_type=event_type):
                    processor = FeatureProcessV1(camp)
                    processor.process_feature(
                        make_observation(frame=15000, camp=camp)
                    )
                    context = processor._memo_context
                    kai = next(
                        hero
                        for hero in context["friend_heroes"]
                        if u.actor_config_id(hero["actor"]) == 193
                    )
                    enemy = context["enemy_heroes"][0]
                    enemy_runtime = u.actor_runtime_id(enemy["actor"])
                    for index, ally in enumerate(context["friend_heroes"]):
                        ally["side_position"] = (
                            enemy["side_position"][0] - 1000.0 - index,
                            enemy["side_position"][1],
                        )
                    friend = context["friend_heroes"][0]
                    key = ("friend", u.actor_config_id(friend["actor"]))
                    processor.recent_events[key].append(
                        {
                            "frame": 15000,
                            "type": event_type,
                            "other_runtime": enemy_runtime,
                        }
                    )
                    result = processor._v40_late_conversion(
                        context, kai, True, "none"
                    )
                    self.assertTrue(result["recent_teamfight"])

                    processor.recent_events[key].clear()
                    processor.recent_events[key].append(
                        {
                            "frame": 14969,
                            "type": event_type,
                            "other_runtime": enemy_runtime,
                        }
                    )
                    self.assertFalse(
                        processor._v40_late_conversion(
                            context, kai, True, "none"
                        )["recent_teamfight"]
                    )
                    enemy["visible"] = True
                    context["friend_heroes"][1]["side_position"] = (-100000.0, 0.0)
                    context["friend_heroes"][2]["side_position"] = (-100000.0, 0.0)
                    self.assertFalse(
                        processor._v40_late_conversion(
                            context, kai, True, "none"
                        )["recent_teamfight"]
                    )
                    processor.recent_events[key].clear()
                    processor.recent_events[key].append(
                        {
                            "frame": 15000,
                            "type": event_type,
                            "other_runtime": enemy_runtime + 999,
                        }
                    )
                    self.assertFalse(
                        processor._v40_late_conversion(
                            context, kai, True, "none"
                        )["recent_teamfight"]
                    )
                    processor.recent_events[key][-1]["other_runtime"] = enemy_runtime
                    enemy["visible"] = False
                    self.assertFalse(
                        processor._v40_late_conversion(
                            context, kai, True, "none"
                        )["recent_teamfight"]
                    )

    def test_inactivity_timer_route_exemption_freezes_instead_of_resetting(self):
        processor = FeatureProcessV1(1)
        processor.process_feature(make_observation(frame=500, camp=1))
        context = processor._memo_context
        kai = next(
            hero
            for hero in context["friend_heroes"]
            if u.actor_config_id(hero["actor"]) == 193
        )
        frame = 500
        position = kai["side_position"]
        runtime = u.actor_runtime_id(kai["actor"])
        kai["actor"]["total_hurt"] = 0.0
        state = {
            "history": deque(
                (
                    {
                        "frame": frame - V1TrainingConfig.LONG_INACTIVITY_WINDOW_FRAMES,
                        "position": position,
                        "damage": 0.0,
                        "objective_signature": ("jungle", "timer_navigation", 0),
                        "objective_distance": 1000.0,
                    },
                    {
                        "frame": frame,
                        "position": position,
                        "damage": 0.0,
                        "objective_signature": ("jungle", "timer_navigation", 0),
                        "objective_distance": 1000.0,
                    },
                )
            ),
            "next_evaluation_frame": frame,
            "abnormal_steps": 3,
            "last_damage": 0.0,
            "timer_route_exempt_start_frame": None,
        }
        processor.long_inactivity_history[runtime] = state
        context["jungle_macro"] = {
            "valid": True,
            "phase": "timer_navigation",
            "target_slot": 0,
            "next_waypoint": position,
            "route_progress_confirmed": True,
            "timer_navigation_active": True,
            "timer_wait_active": False,
            "respawn_reentry_active": False,
            "interrupt": "none",
            "route_stuck": False,
        }
        penalty, applied, _tyrant, reasons = (
            processor._consume_long_inactivity_penalty(context, 193, kai)
        )
        self.assertEqual(penalty, 0.0)
        self.assertFalse(applied)
        self.assertEqual(reasons, ("timer_route_exempt",))
        self.assertEqual(state["abnormal_steps"], 3)

        context["frame_no"] = 506
        context["jungle_macro"]["timer_navigation_active"] = False
        context["jungle_macro"]["valid"] = False
        penalty, applied, _tyrant, _reasons = (
            processor._consume_long_inactivity_penalty(context, 193, kai)
        )
        self.assertLess(penalty, 0.0)
        self.assertTrue(applied)
        self.assertEqual(state["abnormal_steps"], 4)
        self.assertEqual(
            state["history"][0]["frame"],
            frame - V1TrainingConfig.LONG_INACTIVITY_WINDOW_FRAMES + 6,
        )

        penalty, applied, _tyrant, _reasons = (
            processor._consume_long_inactivity_penalty(context, 193, kai)
        )
        self.assertEqual(penalty, 0.0)
        self.assertFalse(applied)

    def test_respawn_timing_metrics_keep_role_denominators_and_wait_outcomes(self):
        processor = FeatureProcessV1(1)
        controlled = {
            "alive_frame": 100,
            "metric_role": "controlled",
            "metric_to_jungle_recorded": False,
            "metric_to_first_monster_engagement_recorded": False,
            "metric_empty_wait_start_frame": 110,
            "metric_finished": False,
        }
        processor._record_respawn_elapsed_metric(
            controlled, 120, "to_jungle"
        )
        processor._record_respawn_elapsed_metric(
            controlled, 130, "to_first_monster_engagement"
        )
        processor._finish_respawn_metrics(controlled, 130, "success")
        self.assertEqual(processor.v40_counters["kai_respawn_to_jungle_frames"], 20.0)
        self.assertEqual(
            processor.v40_counters[
                "kai_respawn_controlled_to_first_monster_engagement_frames"
            ],
            30.0,
        )
        self.assertEqual(processor.v40_counters["kai_respawn_empty_wait_frames"], 20.0)
        self.assertEqual(
            processor.v40_counters[
                "kai_respawn_controlled_empty_wait_success_count"
            ],
            1.0,
        )

        official = {
            "alive_frame": 200,
            "metric_role": "official",
            "metric_to_jungle_recorded": False,
            "metric_to_first_monster_engagement_recorded": False,
            "metric_empty_wait_start_frame": 210,
            "metric_finished": False,
        }
        processor._finish_respawn_metrics(official, 230, "timeout")
        self.assertEqual(
            processor.v40_counters["kai_respawn_official_timeout"], 1.0
        )
        self.assertEqual(
            processor.v40_counters[
                "kai_respawn_official_to_jungle_missing"
            ],
            1.0,
        )
        self.assertEqual(
            processor.v40_counters[
                "kai_respawn_official_empty_wait_timeout_frames"
            ],
            20.0,
        )

    def test_enemy_kai_death_latch_ignores_missing_and_clears_on_visible_respawn(self):
        processor = FeatureProcessV1(1)
        initial = make_observation(frame=100, camp=1)
        processor.process_feature(initial)

        killed = make_observation(frame=106, camp=1)
        enemy_kai = killed["frame_state"]["hero_states"][5]
        killed["frame_state"]["frame_action"]["dead_action"] = [
            {
                "killer": killed["frame_state"]["hero_states"][2],
                "death": copy.deepcopy(enemy_kai),
            }
        ]
        enemy_kai["hp"] = 0
        processor.process_feature(killed)
        self.assertTrue(processor.enemy_kai_confirmed_dead)

        missing = make_observation(frame=112, camp=1)
        missing["frame_state"]["hero_states"] = [
            hero
            for hero in missing["frame_state"]["hero_states"]
            if hero["runtime_id"] != 23
        ]
        processor.process_feature(missing)
        self.assertTrue(processor.enemy_kai_confirmed_dead)

        respawned = make_observation(frame=118, camp=1)
        respawned_kai = respawned["frame_state"]["hero_states"][5]
        respawned_kai["runtime_id"] = 24
        processor.process_feature(respawned)
        self.assertFalse(processor.enemy_kai_confirmed_dead)

    def test_respawn_reentry_creation_has_no_hp_entry_threshold(self):
        processor = FeatureProcessV1(1)
        initial = make_observation(frame=2000, camp=1)
        processor.process_feature(initial)
        processor.jungle_route_history["opening_released"] = True
        processor.jungle_route_history["opening_release_reason"] = (
            "six_camps_cleared"
        )

        dead = make_observation(frame=2006, camp=1)
        dead["frame_state"]["hero_states"][2]["hp"] = 0
        processor.process_feature(dead)
        self.assertTrue(
            processor.jungle_route_history["respawn_death_pending"]
        )

        revived = make_observation(frame=2012, camp=1)
        _set_hp(revived["frame_state"]["hero_states"][2], 0.01)
        processor.process_feature(revived)
        reentry = processor.jungle_route_history["respawn_reentry"]
        self.assertIsNotNone(reentry)
        self.assertEqual(reentry["alive_frame"], 2012)
        self.assertEqual(reentry["deadline_frame"], 2312)

    def test_missing_controlled_kai_frame_cannot_fake_death_or_respawn(self):
        processor = FeatureProcessV1(1)
        initial = make_observation(frame=2000, camp=1)
        processor.process_feature(initial)
        processor.jungle_route_history["opening_released"] = True
        processor.jungle_route_history["opening_release_reason"] = (
            "six_camps_cleared"
        )

        missing = make_observation(frame=2006, camp=1)
        missing["frame_state"]["hero_states"] = [
            actor
            for actor in missing["frame_state"]["hero_states"]
            if actor["runtime_id"] != 13
        ]
        processor.process_feature(missing)
        self.assertFalse(
            processor.jungle_route_history["respawn_death_pending"]
        )
        self.assertIsNone(
            processor.jungle_route_history["respawn_reentry"]
        )

        returned = make_observation(frame=2012, camp=1)
        processor.process_feature(returned)
        self.assertFalse(
            processor.jungle_route_history["respawn_death_pending"]
        )
        self.assertIsNone(
            processor.jungle_route_history["respawn_reentry"]
        )

    def test_observed_enemy_crystal_attack_preempts_respawn_reentry_before_late_game(self):
        processor = FeatureProcessV1(1)
        processor.process_feature(make_observation(frame=4994, camp=1))
        history = processor.jungle_route_history
        history["opening_released"] = True
        history["opening_release_reason"] = (
            "protected_first_lower_cleared"
        )
        history["respawn_was_alive"] = True
        history["respawn_reentry"] = {
            "epoch": 1,
            "alive_frame": 4994,
            "deadline_frame": 5294,
            "identity": None,
            "camp_name": "",
            "first_engagement_frame": None,
            "empty_visit_identities": set(),
            "last_interrupt_frame": None,
        }

        observation = make_observation(frame=5000, camp=1)
        enemy_crystal = next(
            actor
            for actor in observation["frame_state"]["npc_states"]
            if actor.get("config_id") == 72
        )
        observation["frame_state"]["hero_states"][0][
            "attack_target"
        ] = enemy_crystal["runtime_id"]
        processor.process_feature(observation)
        macro = processor._memo_context["jungle_macro"]
        self.assertFalse(macro["late_conversion"]["active"])
        self.assertFalse(macro["respawn_reentry_active"])
        self.assertEqual(macro["mode"], "push")
        self.assertIsNotNone(history["respawn_reentry"])
        self.assertEqual(history["respawn_reentry"]["identity"], None)

    def test_late_monster_scale_uses_exact_piecewise_formula_and_latch(self):
        processor = FeatureProcessV1(1)
        expected_without_stack = {
            11999: 1.0,
            12000: 1.0,
            14999: 1.0,
            15000: 1.0,
            16500: 0.5,
            18000: 0.0,
        }
        for frame, expected in expected_without_stack.items():
            processor._memo_context = {"frame_no": frame}
            self.assertAlmostEqual(
                processor.late_monster_reward_scale(frame), expected, places=7
            )
        processor.tyrant_three_stack_reached = True
        processor._memo_context = {"frame_no": 15000}
        self.assertAlmostEqual(
            processor.late_monster_reward_scale(15000), 0.5, places=7
        )
        processor._memo_context = {"frame_no": 18000}
        self.assertEqual(processor.late_monster_reward_scale(18000), 0.0)

    def test_late_scale_and_offset_match_all_eight_frozen_boundaries(self):
        processor = FeatureProcessV1(1)
        for reached, expected in (
            (False, (1.0, 1.0, 0.5, 0.0)),
            (True, (1.0, 0.5, 0.25, 0.0)),
        ):
            processor.tyrant_three_stack_reached = reached
            for frame, scale, expected_offset in zip(
                (12000, 15000, 16500, 18000),
                expected,
                (
                    (0.0, 0.0, -0.04, -0.08)
                    if not reached
                    else (0.0, -0.04, -0.06, -0.08)
                ),
            ):
                processor._memo_context = {"frame_no": frame}
                actual_scale = processor.late_monster_reward_scale(frame)
                self.assertAlmostEqual(actual_scale, scale, places=7)
                offset = late_monster_reward_offsets_v1(
                    (193, 132), (0.08, 0.08), actual_scale, True
                )
                self.assertAlmostEqual(
                    offset[0], expected_offset, places=7
                )
                self.assertEqual(offset[1], 0.0)

    @staticmethod
    def _late_state(frame, opportunity=False, interrupt="none"):
        processor = FeatureProcessV1(1)
        processor.process_feature(make_observation(frame=frame, camp=1))
        context = processor._memo_context
        kai = next(
            hero
            for hero in context["friend_heroes"]
            if u.actor_config_id(hero["actor"]) == 193
        )
        if opportunity:
            structure = context["organs"][3]
            context["friend_heroes"][0]["actor"]["attack_target"] = (
                u.actor_runtime_id(structure["actor"])
            )
        return processor._v40_late_conversion(
            context, kai, True, interrupt
        )

    def test_late_hard_gate_boundaries_require_exact_opportunity(self):
        self.assertFalse(self._late_state(14999, opportunity=True)["active"])
        self.assertTrue(self._late_state(15000, opportunity=True)["active"])
        self.assertFalse(self._late_state(15000, opportunity=False)["active"])
        self.assertTrue(self._late_state(17999, opportunity=True)["active"])
        self.assertTrue(self._late_state(18000, opportunity=False)["active"])
        self.assertFalse(
            self._late_state(
                18000,
                opportunity=True,
                interrupt="tyrant_emergency",
            )["active"]
        )

    def test_late_conversion_without_destination_keeps_frozen_macro_mode(self):
        processor = FeatureProcessV1(1)
        processor.process_feature(
            make_observation(frame=100, camp=1, include_soldiers=False)
        )
        processor.jungle_route_history["opening_released"] = True
        processor.jungle_route_history["opening_release_reason"] = (
            "six_camps_cleared"
        )

        terminal = make_observation(
            frame=V1TrainingConfig.LATE_TERMINAL_FRAME,
            camp=1,
            include_soldiers=False,
        )
        # Leave Kai alive, but remove every possible late-conversion command
        # destination: no teammate, soldier, visible enemy, or attack event.
        for hero in terminal["frame_state"]["hero_states"]:
            if hero["camp"] == 1 and hero["config_id"] != 193:
                hero["hp"] = 0
            elif hero["camp"] == 2:
                hero["hp"] = 0

        features = processor.process_feature(terminal)
        macro = processor._memo_context["jungle_macro"]
        late = macro["late_conversion"]

        self.assertTrue(late["active"])
        self.assertIsNone(late["destination"])
        self.assertEqual(late["destination_mode"], "none")
        self.assertEqual(macro["mode"], "farm")
        self.assertEqual(len(features), FrozenV1Config.CONTROLLED_ACTOR_COUNT)
        self.assertTrue(
            all(
                feature.shape == (FrozenV1Config.FEATURE_DIM,)
                for feature in features
            )
        )

    def test_late_conversion_settles_timer_phi_without_failure_lock_same_boundary(self):
        processor = FeatureProcessV1(1)
        warm = make_observation(frame=14994, camp=1)
        warm["official_model_info"] = {"official": {"runtime_id": 11}}
        processor.process_feature(warm)
        history = processor.jungle_route_history
        history["opening_released"] = True
        history["opening_release_reason"] = "six_camps_cleared"
        history["timer_route"] = {
            "identity": ((-1.0, -1.0), 1),
            "start_frame": 14994,
            "deadline_frame": 15294,
            "last_frame": 14994,
            "start_distance": 10000.0,
            "best_distance": 10000.0,
            "last_progress_frame": 14994,
            "phi": 0.03,
        }

        converted = make_observation(frame=15000, camp=1)
        converted["official_model_info"] = {
            "official": {"runtime_id": 11}
        }
        enemy_outer = next(
            actor
            for actor in converted["frame_state"]["npc_states"]
            if actor.get("config_id") == 70
        )
        converted["frame_state"]["hero_states"][1][
            "attack_target"
        ] = enemy_outer["runtime_id"]
        processor.process_feature(converted)
        macro = processor._memo_context["jungle_macro"]
        self.assertTrue(macro["late_conversion"]["active"])
        self.assertAlmostEqual(
            macro["route_shaping_reward"], -0.03, places=7
        )
        self.assertIsNone(history["timer_route"])
        self.assertEqual(history["timer_route_failed_generations"], set())
        self.assertEqual(
            processor.v40_counters["late_conversion_preempt"], 1.0
        )

    def test_late_reward_offset_is_kai_final_only_and_idempotent(self):
        offsets = late_monster_reward_offsets_v1(
            (193, 132), (0.08, 0.08), 0.5, True
        )
        self.assertAlmostEqual(offsets[0], -0.04, places=7)
        self.assertEqual(offsets[1], 0.0)
        self.assertEqual(
            late_monster_reward_offsets_v1(
                (193, 132), (0.08, 0.08), 0.5, False
            ),
            (0.0, 0.0),
        )
        self.assertEqual(
            late_monster_reward_offsets_v1(
                (132, 193), (-0.5, -0.5), 0.0, True
            ),
            (0.0, 0.0),
        )


class V40CakeRewardTest(unittest.TestCase):
    @staticmethod
    def _released_processor():
        processor = FeatureProcessV1(1)
        warmup = V40CakeRewardTest._observation(
            1994, 1.0, (-18000.0, -3000.0)
        )
        processor.process_feature(warmup)
        processor.consume_behavior_shaping(expected_frame=1994)
        for history in processor.lane_opening_history.values():
            history["released"] = True
            history["release_reason"] = "actual_enemy_soldier_contact"
            history["release_frame"] = 1994
            history["contact_reached"] = True
        return processor

    @staticmethod
    def _observation(frame, hp_ratio, side_position=None, cake_present=True):
        observation = make_observation(frame=frame, camp=1)
        hero = observation["frame_state"]["hero_states"][0]
        _set_hp(hero, hp_ratio)
        if side_position is not None:
            _set_side_position(hero, *side_position, 1)
        observation["frame_state"]["cakes"] = [
            cake
            for cake in observation["frame_state"]["cakes"]
            if cake["config_id"] != 104
            and (cake_present or cake["config_id"] != 103)
        ]
        return observation

    def test_progress_is_monotonic_and_controlled_abort_claws_back_once(self):
        processor = self._released_processor()
        start = self._observation(2000, 0.40, (-18000.0, -3000.0))
        processor.process_feature(start)
        processor.consume_behavior_shaping(expected_frame=2000)
        self.assertIsNotNone(processor.cake_opportunities[0])

        closer = self._observation(2006, 0.40, (-21050.0, 0.0))
        processor.process_feature(closer)
        reward = processor.consume_behavior_shaping(expected_frame=2006)
        progress = reward["cake_progress_rewards"][0]
        self.assertGreater(progress, 0.0)

        farther = self._observation(2012, 0.40, (-18000.0, -3000.0))
        processor.process_feature(farther)
        reward = processor.consume_behavior_shaping(expected_frame=2012)
        self.assertEqual(reward["cake_progress_rewards"][0], 0.0)

        abort = self._observation(2018, 0.70, (-18000.0, -3000.0))
        processor.process_feature(abort)
        reward = processor.consume_behavior_shaping(expected_frame=2018)
        self.assertAlmostEqual(reward["cake_clawbacks"][0], -progress, places=7)
        self.assertIsNone(processor.cake_opportunities[0])

        duplicate = processor.consume_behavior_shaping(expected_frame=2018)
        self.assertEqual(duplicate["cake_clawbacks"][0], 0.0)

    def test_conservative_progress_budget_and_short_timeout_are_exact(self):
        processor = self._released_processor()
        processor.process_feature(
            self._observation(2000, 0.40, (-18000.0, -3000.0))
        )
        processor.consume_behavior_shaping(expected_frame=2000)
        processor.process_feature(
            self._observation(2006, 0.40, (-21050.0, 0.0))
        )
        progress = processor.consume_behavior_shaping(
            expected_frame=2006
        )["cake_progress_rewards"][0]
        self.assertGreater(progress, 0.0)
        self.assertLessEqual(progress, 0.015)
        self.assertEqual(processor.cake_opportunities[0]["deadline_frame"], 2120)

        before = self._observation(2119, 0.40, (-21050.0, 0.0))
        processor.process_feature(before)
        self.assertEqual(
            processor.consume_behavior_shaping(2119)["cake_clawbacks"][0],
            0.0,
        )
        boundary = self._observation(2120, 0.40, (-21050.0, 0.0))
        processor.process_feature(boundary)
        self.assertAlmostEqual(
            processor.consume_behavior_shaping(2120)["cake_clawbacks"][0],
            -progress,
            places=7,
        )

    def test_unique_buff_and_proximity_evidence_attributes_pickup(self):
        processor = self._released_processor()
        start = self._observation(2000, 0.40, (-21050.0, 0.0))
        processor.process_feature(start)
        processor.consume_behavior_shaping(expected_frame=2000)

        pickup = self._observation(
            2006, 0.55, (-21050.0, 0.0), cake_present=False
        )
        hero = pickup["frame_state"]["hero_states"][0]
        hero["buff_state"]["buff_skills"].append(
            {"config_id": 90014, "start_time": 2006, "times": 1}
        )
        processor.process_feature(pickup)
        reward = processor.consume_behavior_shaping(expected_frame=2006)
        self.assertEqual(reward["cake_pickup_rewards"][0], 0.0)
        self.assertEqual(
            processor.cake_pending_pickups[103]["winner_key"],
            ("friend", 132),
        )

        healed = self._observation(
            2030, 0.60, (-21050.0, 0.0), cake_present=False
        )
        healed["frame_state"]["hero_states"][0]["buff_state"][
            "buff_skills"
        ].append({"config_id": 90014, "start_time": 2006, "times": 1})
        processor.process_feature(healed)
        reward = processor.consume_behavior_shaping(expected_frame=2030)
        self.assertGreaterEqual(reward["cake_pickup_rewards"][0], 0.04)
        self.assertEqual(reward["cake_pickup_rewards"][1], 0.0)
        self.assertEqual(processor.cake_counters["pickup_success"], 1.0)

    def test_pending_freezes_all_abort_paths_until_delayed_winner_settles(self):
        processor = self._released_processor()
        start = self._observation(2000, 0.40, (-21050.0, 0.0))
        start["frame_state"]["cakes"] = [
            cake
            for cake in start["frame_state"]["cakes"]
            if cake["config_id"] == 103
        ]
        processor.process_feature(start)
        processor.consume_behavior_shaping(expected_frame=2000)

        disappeared = self._observation(
            2006, 0.40, (-21050.0, 0.0), cake_present=False
        )
        processor.process_feature(disappeared)
        processor.consume_behavior_shaping(expected_frame=2006)
        self.assertEqual(processor.cake_opportunities[0]["phase"], "pickup_pending")

        interrupted = self._observation(
            2012, 0.90, (-21050.0, 0.0), cake_present=False
        )
        hero = interrupted["frame_state"]["hero_states"][0]
        hero["hp"] = 0.0
        hero["location"] = None
        interrupted["frame_state"]["frame_action"]["dead_action"] = [
            hero["runtime_id"]
        ]
        enemy_crystal = next(
            actor
            for actor in interrupted["frame_state"]["npc_states"]
            if actor.get("config_id") == 72
        )
        interrupted["frame_state"]["hero_states"][1]["attack_target"] = (
            enemy_crystal["runtime_id"]
        )
        processor.process_feature(interrupted)
        reward = processor.consume_behavior_shaping(expected_frame=2012)
        self.assertEqual(reward["cake_clawbacks"][0], 0.0)
        self.assertIsNotNone(processor.cake_opportunities[0])
        self.assertEqual(processor.cake_counters["hard_objective_cancel"], 0.0)
        self.assertEqual(processor.cake_counters["controlled_abort"], 0.0)
        self.assertEqual(processor.cake_counters["external_cancel"], 0.0)

        winner = self._observation(
            2018, 0.55, (-21050.0, 0.0), cake_present=False
        )
        winner["frame_state"]["hero_states"][0]["buff_state"][
            "buff_skills"
        ].append({"config_id": 90014, "start_time": 2018, "times": 1})
        processor.process_feature(winner)
        reward = processor.consume_behavior_shaping(expected_frame=2018)
        self.assertEqual(reward["cake_pickup_rewards"][0], 0.0)
        self.assertIsNotNone(processor.cake_opportunities[0])

        completed = copy.deepcopy(winner)
        completed["frame_state"]["frame_no"] = 2030
        _set_hp(completed["frame_state"]["hero_states"][0], 0.65)
        processor.process_feature(completed)
        reward = processor.consume_behavior_shaping(expected_frame=2030)
        self.assertGreater(reward["cake_pickup_rewards"][0], 0.04)
        self.assertIsNone(processor.cake_opportunities[0])
        self.assertEqual(processor.cake_counters["pickup_success"], 1.0)

    def test_cake_lifecycle_is_exclusive_across_frames_and_reopens_next_generation(self):
        processor = self._released_processor()
        first = self._observation(2000, 0.40, (-21050.0, 0.0))
        first["frame_state"]["cakes"] = [
            cake
            for cake in first["frame_state"]["cakes"]
            if cake["config_id"] == 103
        ]
        second_hero = first["frame_state"]["hero_states"][1]
        _set_hp(second_hero, 1.0)
        _set_side_position(second_hero, -21050.0, 0.0, 1)
        processor.process_feature(first)
        processor.consume_behavior_shaping(expected_frame=2000)
        self.assertEqual(processor.cake_opportunities[0]["cake_config_id"], 103)
        self.assertIsNone(processor.cake_opportunities[1])

        staggered = copy.deepcopy(first)
        staggered["frame_state"]["frame_no"] = 2006
        _set_hp(staggered["frame_state"]["hero_states"][1], 0.40)
        processor.process_feature(staggered)
        processor.consume_behavior_shaping(expected_frame=2006)
        self.assertIsNone(processor.cake_opportunities[1])

        hard = copy.deepcopy(staggered)
        hard["frame_state"]["frame_no"] = 2012
        enemy_crystal = next(
            actor
            for actor in hard["frame_state"]["npc_states"]
            if actor.get("config_id") == 72
        )
        hard["frame_state"]["hero_states"][1]["attack_target"] = (
            enemy_crystal["runtime_id"]
        )
        processor.process_feature(hard)
        processor.consume_behavior_shaping(expected_frame=2012)
        self.assertIsNone(processor.cake_opportunities[0])
        self.assertIsNone(processor.cake_opportunities[1])

        same_lifecycle = copy.deepcopy(first)
        same_lifecycle["frame_state"]["frame_no"] = 2018
        _set_hp(same_lifecycle["frame_state"]["hero_states"][0], 1.0)
        _set_hp(same_lifecycle["frame_state"]["hero_states"][1], 0.40)
        processor.process_feature(same_lifecycle)
        processor.consume_behavior_shaping(expected_frame=2018)
        self.assertIsNone(processor.cake_opportunities[1])

        absent = copy.deepcopy(same_lifecycle)
        absent["frame_state"]["frame_no"] = 2024
        absent["frame_state"]["cakes"] = []
        processor.process_feature(absent)
        processor.consume_behavior_shaping(expected_frame=2024)

        next_lifecycle = copy.deepcopy(same_lifecycle)
        next_lifecycle["frame_state"]["frame_no"] = 2030
        processor.process_feature(next_lifecycle)
        processor.consume_behavior_shaping(expected_frame=2030)
        self.assertIsNotNone(processor.cake_opportunities[1])
        self.assertEqual(processor.cake_opportunities[1]["lifecycle_id"], 2)

    def test_terminal_settles_locked_pickup_without_waiting_for_full_window(self):
        processor = self._released_processor()
        processor.process_feature(
            self._observation(2000, 0.40, (-21050.0, 0.0))
        )
        processor.consume_behavior_shaping(expected_frame=2000)
        pickup = self._observation(
            2006, 0.55, (-21050.0, 0.0), cake_present=False
        )
        pickup["frame_state"]["hero_states"][0]["buff_state"][
            "buff_skills"
        ].append({"config_id": 90014, "start_time": 2006, "times": 1})
        processor.process_feature(pickup)
        processor.consume_behavior_shaping(expected_frame=2006)

        terminal = processor.consume_episode_terminal_shaping(2006)
        self.assertGreaterEqual(terminal["cake_pickup_rewards"][0], 0.04)
        self.assertEqual(terminal["cake_clawbacks"][0], 0.0)
        self.assertIsNone(processor.cake_opportunities[0])

    def test_cake_lifecycle_advances_only_after_real_reappearance_and_freezes_after_tower_death(self):
        processor = FeatureProcessV1(1)
        processor.process_feature(make_observation(frame=100, camp=1))
        self.assertEqual(
            processor.resource_history[103]["lifecycle_id"], 1
        )

        absent = make_observation(frame=106, camp=1)
        absent["frame_state"]["cakes"] = [
            cake
            for cake in absent["frame_state"]["cakes"]
            if cake["config_id"] != 103
        ]
        processor.process_feature(absent)
        self.assertEqual(
            processor.resource_history[103]["transition"],
            "present_to_absent",
        )

        returned = make_observation(frame=112, camp=1)
        processor.process_feature(returned)
        self.assertEqual(
            processor.resource_history[103]["transition"],
            "absent_to_present",
        )
        self.assertEqual(
            processor.resource_history[103]["lifecycle_id"], 2
        )

        disabled = make_observation(frame=118, camp=1)
        own_outer = next(
            actor
            for actor in disabled["frame_state"]["npc_states"]
            if actor.get("config_id") == 66
        )
        own_outer["hp"] = 0
        disabled["frame_state"]["cakes"] = [
            cake
            for cake in disabled["frame_state"]["cakes"]
            if cake["config_id"] != 103
        ]
        processor.process_feature(disabled)
        self.assertEqual(
            processor.resource_history[103]["lifecycle_id"], 2
        )
        self.assertEqual(
            processor.resource_history[103]["transition"], "none"
        )

        jitter = make_observation(frame=124, camp=1)
        jitter_outer = next(
            actor
            for actor in jitter["frame_state"]["npc_states"]
            if actor.get("config_id") == 66
        )
        jitter_outer["hp"] = 0
        processor.process_feature(jitter)
        self.assertEqual(
            processor.resource_history[103]["lifecycle_id"], 2
        )
        self.assertEqual(
            processor.resource_history[103]["transition"], "none"
        )

    def test_hard_objective_wins_same_boundary_over_hp_abort_without_clawback(self):
        processor = self._released_processor()
        processor.process_feature(
            self._observation(2000, 0.40, (-18000.0, -3000.0))
        )
        processor.consume_behavior_shaping(expected_frame=2000)
        processor.process_feature(
            self._observation(2006, 0.40, (-21050.0, 0.0))
        )
        progress = processor.consume_behavior_shaping(
            expected_frame=2006
        )["cake_progress_rewards"][0]
        self.assertGreater(progress, 0.0)

        hard = self._observation(2012, 0.70, (-21050.0, 0.0))
        enemy_crystal = next(
            actor
            for actor in hard["frame_state"]["npc_states"]
            if actor.get("config_id") == 72
        )
        hard["frame_state"]["hero_states"][1]["attack_target"] = (
            enemy_crystal["runtime_id"]
        )
        processor.process_feature(hard)
        reward = processor.consume_behavior_shaping(expected_frame=2012)
        self.assertEqual(reward["cake_clawbacks"][0], 0.0)
        self.assertEqual(
            processor.cake_counters["hard_objective_cancel"], 1.0
        )
        self.assertEqual(processor.cake_counters["controlled_abort"], 0.0)

    def test_temporary_controlled_actor_input_loss_is_external_not_clawback(self):
        processor = self._released_processor()
        processor.process_feature(
            self._observation(2000, 0.40, (-18000.0, -3000.0))
        )
        processor.consume_behavior_shaping(expected_frame=2000)
        processor.process_feature(
            self._observation(2006, 0.40, (-21050.0, 0.0))
        )
        progress = processor.consume_behavior_shaping(
            expected_frame=2006
        )["cake_progress_rewards"][0]
        self.assertGreater(progress, 0.0)

        missing = self._observation(2012, 0.40, (-21050.0, 0.0))
        controlled_runtime = missing["heroes"][0]["runtime_id"]
        missing["frame_state"]["hero_states"] = [
            actor
            for actor in missing["frame_state"]["hero_states"]
            if actor["runtime_id"] != controlled_runtime
        ]
        processor.process_feature(missing)
        reward = processor.consume_behavior_shaping(expected_frame=2012)
        self.assertEqual(reward["cake_clawbacks"][0], 0.0)
        self.assertEqual(processor.cake_counters["external_cancel"], 1.0)
        self.assertEqual(processor.cake_counters["controlled_abort"], 0.0)

    def test_confirmed_death_without_position_is_controlled_abort(self):
        processor = self._released_processor()
        processor.process_feature(
            self._observation(2000, 0.40, (-18000.0, -3000.0))
        )
        processor.consume_behavior_shaping(expected_frame=2000)
        processor.process_feature(
            self._observation(2006, 0.40, (-21050.0, 0.0))
        )
        progress = processor.consume_behavior_shaping(
            expected_frame=2006
        )["cake_progress_rewards"][0]
        self.assertGreater(progress, 0.0)

        dead = self._observation(2012, 0.0, (-21050.0, 0.0))
        dead_hero = dead["frame_state"]["hero_states"][0]
        dead_hero["location"] = None
        dead["frame_state"]["frame_action"]["dead_action"] = [
            dead_hero["runtime_id"]
        ]
        processor.process_feature(dead)
        reward = processor.consume_behavior_shaping(expected_frame=2012)
        self.assertAlmostEqual(
            reward["cake_clawbacks"][0], -progress, places=7
        )
        self.assertEqual(processor.cake_counters["controlled_abort"], 1.0)
        self.assertEqual(processor.cake_counters["external_cancel"], 0.0)

    def test_episode_terminal_settles_cake_and_timer_ledgers_once(self):
        processor = self._released_processor()
        processor.process_feature(
            self._observation(2000, 0.40, (-18000.0, -3000.0))
        )
        processor.consume_behavior_shaping(expected_frame=2000)
        processor.process_feature(
            self._observation(2006, 0.40, (-21050.0, 0.0))
        )
        progress = processor.consume_behavior_shaping(
            expected_frame=2006
        )["cake_progress_rewards"][0]
        processor._memo_context["controlled_infos"][1]["config_id"] = 193
        processor.jungle_route_history["timer_route"] = {"phi": 0.03}

        terminal = processor.consume_episode_terminal_shaping(2006)
        self.assertAlmostEqual(
            terminal["cake_clawbacks"][0], -progress, places=7
        )
        self.assertEqual(terminal["cake_clawbacks"][1], 0.0)
        self.assertEqual(terminal["route_settlements"][0], 0.0)
        self.assertAlmostEqual(
            terminal["route_settlements"][1], -0.03, places=7
        )
        self.assertIsNone(processor.jungle_route_history["timer_route"])
        self.assertIsNone(processor.cake_opportunities[0])

        duplicate = processor.consume_episode_terminal_shaping(2006)
        self.assertEqual(duplicate["cake_clawbacks"], (0.0, 0.0))
        self.assertEqual(duplicate["route_settlements"], (0.0, 0.0))


class V40CommandPromotionTest(unittest.TestCase):
    @staticmethod
    def _promotion_inputs(
        frame=100,
        hp_ratio=0.60,
        target_kind="soldier",
        target_position=(1000.0, 0.0),
        command_position=(1000.0, 0.0),
        cooldown_ready=True,
        source=True,
        local_enemy_count=1,
        lane_interrupt="none",
    ):
        official_actor = {
            "runtime_id": 11,
            "config_id": 132,
            "camp": 1,
            "hp": 3200.0 * hp_ratio,
            "max_hp": 3200.0,
            "attack_range": 6000.0,
            "attack_target": 401,
        }
        target_actor = {
            "runtime_id": 401,
            "config_id": (
                21
                if target_kind == "enemy_hero"
                else 70
                if target_kind in ("outer_tower", "high_tower", "crystal")
                else 19
            ),
            "camp": 2,
            "hp": 1000.0,
            "max_hp": 1000.0,
        }
        official = {
            "actor": official_actor,
            "relation": "friend",
            "alive": True,
            "visible": True,
            "current_valid": True,
            "side_position": (0.0, 0.0),
        }
        enemy_heroes = [
            {
                "actor": {
                    "runtime_id": 21 + index,
                    "config_id": (132, 141, 193)[index % 3],
                    "camp": 2,
                    "hp": 3000.0,
                    "max_hp": 3000.0,
                },
                "relation": "enemy",
                "alive": True,
                "visible": True,
                "side_position": (1000.0 + index, 0.0),
            }
            for index in range(local_enemy_count)
        ]
        context = {
            "frame_no": frame,
            "official_hero": official,
            "collaboration": {
                "active": True,
                "type": "set_position",
                "source": (
                    {
                        "action_type": 2,
                        "intent": 5,
                        "position_slot": 5,
                        "is_train": True,
                        "accepted_frame": 94,
                        "raw_position": command_position,
                    }
                    if source
                    else None
                ),
                "cooldown_ready": cooldown_ready,
                "command_position": command_position,
                "target_runtime": 0,
            },
            "lane_macro": {
                132: {
                    "command_active": False,
                    "interrupt": lane_interrupt,
                    "structure_defense": (
                        lane_interrupt == "structure_defense"
                    ),
                }
            },
            "actor_by_runtime": {401: target_actor},
            "dead_runtime_ids": set(),
            "friend_heroes": [official],
            "enemy_heroes": enemy_heroes,
            "organs": [
                {"alive": True, "side_position": (-60000.0, 0.0)},
                {"alive": True, "side_position": (-72000.0, 0.0)},
                {"alive": True, "side_position": (-80000.0, 0.0)},
            ],
        }
        descriptors = [
            {"kind": "none", "actor": None, "snapshot": None}
            for _ in range(39)
        ]
        descriptors[28] = {
            "kind": target_kind,
            "actor": target_actor,
            "snapshot": {"side_position": target_position},
        }
        runtime_ids = [0] * 39
        runtime_ids[28] = 401
        target_legal = [False] * 39
        target_legal[28] = True
        return context, descriptors, runtime_ids, target_legal

    @staticmethod
    def _advance_promotion(processor, inputs, **updates):
        context, descriptors, runtime_ids, target_legal = inputs
        context.update(updates)
        return processor._update_lane_target_promotion(
            context,
            descriptors,
            runtime_ids,
            target_legal,
            False,
            False,
        )

    def test_soldier_promotion_is_singleton_nontrain_settarget(self):
        base = _context()
        target_runtime_ids = list(base.target_runtime_ids)
        target_runtime_ids[28] = 401
        target_legal = list(base.collaboration_target_legal)
        target_legal[28] = True
        context = replace(
            base,
            official_runtime_id=11,
            official_config_id=132,
            jungle_route_active=False,
            lane_route_active=False,
            cooldown_ready=True,
            lane_promotion_active=True,
            lane_promotion_intent=5,
            lane_promotion_target_slot=28,
            lane_promotion_hold=False,
            target_runtime_ids=tuple(target_runtime_ids),
            collaboration_target_legal=tuple(target_legal),
        )
        masks = build_command_legal_masks_v1(context)
        self.assertEqual(np.flatnonzero(masks.intent).tolist(), [5])
        self.assertEqual(
            np.argwhere(masks.target_by_intent > 0.5).tolist(), [[5, 28]]
        )
        self.assertFalse(masks.command_is_train)
        action = CommandRuleLegalizerV1().legalize(
            5, 28, 0, context, masks
        )
        self.assertTrue(action.valid)
        self.assertEqual(action.action, (1, 401, 0, 0))

    def test_lane_retreat_handoff_prefers_safe_position_and_reuses_only_hard_safety_unset(self):
        base = _context()
        ready = replace(
            base,
            official_runtime_id=11,
            official_config_id=132,
            current_command_active=True,
            cooldown_ready=True,
            jungle_route_active=False,
            lane_route_active=False,
            lane_retreat_handoff_active=True,
            lane_retreat_position_slot=9,
        )
        masks = build_command_legal_masks_v1(ready)
        self.assertEqual(np.flatnonzero(masks.intent).tolist(), [8])
        self.assertEqual(
            np.flatnonzero(masks.position_by_intent[8]).tolist(), [9]
        )
        self.assertFalse(masks.command_is_train)

        cooldown = replace(
            ready,
            cooldown_ready=False,
            unset_allowed=False,
        )
        held = build_command_legal_masks_v1(cooldown)
        self.assertEqual(np.flatnonzero(held.intent).tolist(), [0])
        self.assertFalse(held.command_is_train)

        hard_safety = replace(
            cooldown,
            unset_allowed=True,
            unset_reason="low_hp_active_command",
        )
        unset = build_command_legal_masks_v1(hard_safety)
        self.assertEqual(np.flatnonzero(unset.intent).tolist(), [1])
        self.assertFalse(unset.command_is_train)

    def test_late_conversion_has_priority_over_stale_lane_promotion(self):
        base = _context()
        context = replace(
            base,
            official_runtime_id=11,
            official_config_id=132,
            jungle_route_active=False,
            lane_route_active=False,
            cooldown_ready=True,
            late_conversion_active=True,
            late_conversion_intent=5,
            late_conversion_position_slot=4,
            late_conversion_hold=False,
            lane_promotion_active=True,
            lane_promotion_intent=5,
            lane_promotion_target_slot=28,
            lane_promotion_hold=False,
        )
        masks = build_command_legal_masks_v1(context)
        self.assertEqual(np.flatnonzero(masks.intent).tolist(), [5])
        self.assertEqual(masks.position_by_intent[5, 4], 1.0)
        self.assertEqual(np.count_nonzero(masks.target_by_intent), 0)
        self.assertFalse(masks.command_is_train)

    def test_late_command_handoff_holds_cooldown_then_forces_successor_nontrain(self):
        base = _context()
        cooldown = replace(
            base,
            official_runtime_id=13,
            official_config_id=193,
            current_command_active=True,
            cooldown_ready=False,
            jungle_route_active=False,
            lane_route_active=False,
            late_conversion_active=True,
            late_conversion_intent=5,
            late_conversion_position_slot=4,
            late_conversion_hold=False,
        )
        held = build_command_legal_masks_v1(cooldown)
        self.assertEqual(np.flatnonzero(held.intent).tolist(), [0])
        self.assertFalse(held.command_is_train)

        ready = replace(cooldown, cooldown_ready=True)
        successor = build_command_legal_masks_v1(ready)
        self.assertEqual(np.flatnonzero(successor.intent).tolist(), [5])
        self.assertEqual(
            np.flatnonzero(successor.position_by_intent[5]).tolist(),
            [4],
        )
        self.assertFalse(successor.command_is_train)
        action = CommandRuleLegalizerV1().legalize(
            5, 0, 4, ready, successor
        )
        self.assertTrue(action.valid)
        self.assertEqual(action.action[0], 2)

    def test_terminal_late_command_without_destination_is_safe_no_command(self):
        base = _context()
        context = replace(
            base,
            official_runtime_id=13,
            official_config_id=193,
            current_command_active=False,
            cooldown_ready=True,
            jungle_route_active=False,
            lane_route_active=False,
            late_conversion_active=True,
            late_conversion_intent=0,
            late_conversion_target_slot=-1,
            late_conversion_position_slot=-1,
            late_conversion_hold=False,
        )
        masks = build_command_legal_masks_v1(context)
        self.assertEqual(np.flatnonzero(masks.intent).tolist(), [0])
        self.assertFalse(masks.command_is_train)

    def test_late_conversion_without_old_farm_command_keeps_multiple_options_trainable(self):
        base = _context()
        context = replace(
            base,
            official_runtime_id=13,
            official_config_id=193,
            current_command_active=False,
            cooldown_ready=True,
            jungle_route_active=False,
            lane_route_active=False,
            late_conversion_active=True,
            late_conversion_intent=3,
            late_conversion_target_slot=38,
            late_conversion_position_slot=-1,
            late_conversion_hold=False,
            late_conversion_target_options=((3, 38),),
            late_conversion_position_options=((5, 4), (6, 1)),
            late_conversion_forced_handoff=False,
        )
        masks = build_command_legal_masks_v1(context)
        self.assertEqual(
            np.flatnonzero(masks.intent).tolist(), [0, 3, 5, 6]
        )
        self.assertTrue(masks.command_is_train)

    def test_late_conversion_old_farm_handoff_forces_only_first_safe_successor(self):
        base = _context()
        context = replace(
            base,
            official_runtime_id=13,
            official_config_id=193,
            current_command_active=True,
            cooldown_ready=True,
            jungle_route_active=False,
            lane_route_active=False,
            late_conversion_active=True,
            late_conversion_intent=3,
            late_conversion_target_slot=38,
            late_conversion_hold=False,
            late_conversion_target_options=((3, 38),),
            late_conversion_position_options=((5, 4), (6, 1)),
            late_conversion_forced_handoff=True,
        )
        masks = build_command_legal_masks_v1(context)
        self.assertEqual(np.flatnonzero(masks.intent).tolist(), [3])
        self.assertFalse(masks.command_is_train)

    def test_promotion_stability_uses_fresh_observations_and_exact_eighteen_frame_boundary(self):
        processor = FeatureProcessV1(1)
        inputs = self._promotion_inputs(frame=100)
        first = self._advance_promotion(processor, inputs)
        self.assertTrue(first["hold"])
        self.assertEqual(
            processor.lane_combat_target_promotion[
                "stable_observation_count"
            ],
            1,
        )

        duplicate = self._advance_promotion(processor, inputs)
        self.assertTrue(duplicate["hold"])
        self.assertEqual(
            processor.lane_combat_target_promotion[
                "stable_observation_count"
            ],
            1,
        )

        inputs[0]["frame_no"] = 118
        stable = self._advance_promotion(processor, inputs)
        self.assertFalse(stable["hold"])
        self.assertEqual(stable["intent"], 5)
        self.assertEqual(stable["target_slot"], 28)

        processor = FeatureProcessV1(1)
        inputs = self._promotion_inputs(frame=100)
        self._advance_promotion(processor, inputs)
        inputs[0]["frame_no"] = 119
        reset = self._advance_promotion(processor, inputs)
        self.assertTrue(reset["hold"])
        self.assertEqual(
            processor.lane_combat_target_promotion[
                "stable_observation_count"
            ],
            1,
        )

    def test_promotion_hp_outnumber_defense_and_forced_route_protections_are_fail_closed(self):
        cases = (
            ({"hp_ratio": 0.35}, "low_hp_protected"),
            (
                {"hp_ratio": 0.55, "local_enemy_count": 2},
                "outnumbered_protected",
            ),
            (
                {"lane_interrupt": "structure_defense"},
                "defense_protected",
            ),
        )
        for kwargs, expected in cases:
            with self.subTest(expected=expected):
                processor = FeatureProcessV1(1)
                result = self._advance_promotion(
                    processor, self._promotion_inputs(**kwargs)
                )
                self.assertFalse(result["active"])
                self.assertEqual(
                    processor.lane_combat_target_promotion[
                        "blocked_reason"
                    ],
                    expected,
                )

        processor = FeatureProcessV1(1)
        inputs = self._promotion_inputs(hp_ratio=0.350001)
        self.assertTrue(self._advance_promotion(processor, inputs)["active"])

        processor = FeatureProcessV1(1)
        inputs = self._promotion_inputs(
            hp_ratio=0.550001, local_enemy_count=2
        )
        self.assertTrue(self._advance_promotion(processor, inputs)["active"])

        processor = FeatureProcessV1(1)
        inputs = self._promotion_inputs()
        result = processor._update_lane_target_promotion(
            *inputs, True, False
        )
        self.assertFalse(result["active"])

    def test_promotion_retreat_geometry_honors_cosine_and_distance_boundaries(self):
        target = (1000.0, 0.0)
        exact_goal = (-250.0, math.sqrt(1000.0 ** 2 - 250.0 ** 2))
        processor = FeatureProcessV1(1)
        inputs = self._promotion_inputs(
            target_position=target, command_position=exact_goal
        )
        result = self._advance_promotion(processor, inputs)
        self.assertFalse(result["active"])
        self.assertEqual(
            processor.lane_combat_target_promotion["blocked_reason"],
            "geometry_protected",
        )

        epsilon_goal = (
            -249.999,
            math.sqrt(1000.0 ** 2 - 249.999 ** 2),
        )
        processor = FeatureProcessV1(1)
        inputs = self._promotion_inputs(
            target_position=target, command_position=epsilon_goal
        )
        self.assertTrue(self._advance_promotion(processor, inputs)["active"])

        processor = FeatureProcessV1(1)
        inputs = self._promotion_inputs(
            target_position=(0.0, 0.0),
            command_position=(1000.0, 0.0),
        )
        result = self._advance_promotion(processor, inputs)
        self.assertFalse(result["active"])
        self.assertEqual(
            processor.lane_combat_target_promotion["blocked_reason"],
            "geometry_unverified",
        )

        exact_distance_goal = (-500.0, 0.0)
        processor = FeatureProcessV1(1)
        inputs = self._promotion_inputs(
            target_position=target,
            command_position=exact_distance_goal,
        )
        self.assertFalse(self._advance_promotion(processor, inputs)["active"])

        below_distance_goal = (-499.999, 0.0)
        processor = FeatureProcessV1(1)
        inputs = self._promotion_inputs(
            target_position=target,
            command_position=below_distance_goal,
        )
        self.assertTrue(self._advance_promotion(processor, inputs)["active"])

    def test_promotion_maps_target_kinds_and_rejects_unknown_source_or_monster(self):
        for kind, expected_intent in (
            ("enemy_hero", 2),
            ("soldier", 5),
            ("high_tower", 3),
        ):
            with self.subTest(kind=kind):
                processor = FeatureProcessV1(1)
                inputs = self._promotion_inputs(target_kind=kind)
                self._advance_promotion(processor, inputs)
                inputs[0]["frame_no"] = 106
                result = self._advance_promotion(processor, inputs)
                self.assertEqual(result["intent"], expected_intent)

        processor = FeatureProcessV1(1)
        self.assertFalse(
            self._advance_promotion(
                processor, self._promotion_inputs(source=False)
            )["active"]
        )
        processor = FeatureProcessV1(1)
        self.assertFalse(
            self._advance_promotion(
                processor,
                self._promotion_inputs(target_kind="monster"),
            )["active"]
        )

    def test_active_target_missing_grace_holds_but_never_resends_stale_runtime(self):
        processor = FeatureProcessV1(1)
        inputs = self._promotion_inputs(frame=100)
        self._advance_promotion(processor, inputs)
        inputs[0]["frame_no"] = 106
        ready = self._advance_promotion(processor, inputs)
        self.assertFalse(ready["hold"])

        inputs[0]["frame_no"] = 112
        inputs[0]["collaboration"].update(
            type="set_target", target_runtime=401
        )
        active = self._advance_promotion(processor, inputs)
        self.assertTrue(active["hold"])

        inputs[0]["frame_no"] = 124
        inputs[2][28] = 0
        inputs[3][28] = False
        inputs[0]["actor_by_runtime"] = {}
        grace = self._advance_promotion(processor, inputs)
        self.assertTrue(grace["hold"])
        self.assertTrue(grace["grace_active"])

        inputs[0]["frame_no"] = 131
        expired = self._advance_promotion(processor, inputs)
        self.assertFalse(expired["active"])
        self.assertEqual(
            processor.lane_combat_target_promotion["blocked_reason"],
            "target_grace_expired",
        )

    def test_active_target_low_hp_enters_nontrain_retreat_handoff_same_boundary(self):
        processor = FeatureProcessV1(1)
        inputs = self._promotion_inputs(frame=100)
        self._advance_promotion(processor, inputs)
        inputs[0]["frame_no"] = 106
        self._advance_promotion(processor, inputs)
        inputs[0]["frame_no"] = 112
        inputs[0]["collaboration"].update(
            type="set_target", target_runtime=401
        )
        self._advance_promotion(processor, inputs)

        inputs[0]["frame_no"] = 118
        _set_hp(inputs[0]["official_hero"]["actor"], 0.35)
        result = self._advance_promotion(processor, inputs)
        self.assertTrue(result["retreat_handoff_active"])
        self.assertIn(result["retreat_position_slot"], (6, 7, 9))
        self.assertEqual(
            processor.lane_combat_target_promotion["phase"],
            "retreat_handoff",
        )


class V40CommandLifecycleIntegrationTest(unittest.TestCase):
    @staticmethod
    def _active_target_observation(frame, sender, receiver, target_runtime, sent_frame):
        observation = make_observation(frame=frame, camp=1)
        state = observation["frame_state"]["collaboration_state"]
        for item in state["micro_last"]:
            if item["sender"] == sender:
                item["frame_no"] = sent_frame
        state["micro_state"] = [
            {
                "sender": sender,
                "receiver": receiver,
                "target_runtime_id": target_runtime,
                "target_position": {
                    "x": 999999,
                    "y": 999999,
                    "z": 999999,
                },
                "end_frame": frame + 300,
            }
        ]
        return observation

    @staticmethod
    def _active_position_observation(
        frame, sender, receiver, raw_x, raw_z, sent_frame
    ):
        observation = make_observation(frame=frame, camp=1)
        state = observation["frame_state"]["collaboration_state"]
        for item in state["micro_last"]:
            if item["sender"] == sender:
                item["frame_no"] = sent_frame
        state["micro_state"] = [
            {
                "sender": sender,
                "receiver": receiver,
                "target_runtime_id": 0,
                "target_position": {
                    "x": raw_x,
                    "y": 0,
                    "z": raw_z,
                },
                "end_frame": frame + 600,
            }
        ]
        return observation

    def test_active_target_validity_is_independent_from_reissue_mask(self):
        processor = FeatureProcessV1(1)
        processor.process_feature(make_observation(frame=3000, camp=1))
        processor.jungle_route_history["opening_released"] = True
        processor.jungle_route_history["opening_release_reason"] = (
            "six_camps_cleared"
        )
        initial = make_observation(frame=3006, camp=1)
        processor.process_feature(initial)
        context = processor.command_runtime_context()
        target_slot = context.target_runtime_ids.index(21)
        masks = build_command_legal_masks_v1(context)
        command = CommandRuleLegalizerV1().legalize(
            2, target_slot, 0, context, masks
        )
        self.assertTrue(command.valid)
        processor.record_collaboration_command(
            3006,
            command.action,
            intent=2,
            is_train=True,
            target_slot=target_slot,
        )

        echoed = self._active_target_observation(
            frame=3012,
            sender=11,
            receiver=13,
            target_runtime=21,
            sent_frame=3006,
        )
        processor.process_feature(echoed)
        active = processor.command_runtime_context()
        active_slot = active.target_runtime_ids.index(21)

        self.assertTrue(
            active.collaboration_target_current_valid[active_slot]
        )
        self.assertFalse(active.collaboration_target_legal[active_slot])
        self.assertNotEqual(active.unset_reason, "target_invalid")
        active_masks = build_command_legal_masks_v1(active)
        self.assertEqual(active_masks.target_by_intent[2, active_slot], 0.0)

    def test_c01_full_position_to_target_echo_grace_and_low_hp_unset_loop(self):
        processor = FeatureProcessV1(1)
        initial = make_observation(frame=3000, camp=1)
        initial["official_model_info"] = {"official": {"runtime_id": 11}}
        processor.process_feature(initial)
        for history in processor.lane_opening_history.values():
            history.update(
                released=True,
                release_reason="actual_enemy_soldier_contact",
                release_frame=3000,
                contact_reached=True,
                first_wave_cleared=True,
            )

        source_observation = make_observation(frame=3006, camp=1)
        source_observation["official_model_info"] = {
            "official": {"runtime_id": 11}
        }
        processor.process_feature(source_observation)
        source_context = processor.command_runtime_context()
        source_masks = build_command_legal_masks_v1(source_context)
        self.assertTrue(source_masks.command_is_train)
        self.assertGreater(source_masks.position_by_intent[5, 5], 0.5)
        source_command = CommandRuleLegalizerV1().legalize(
            5, 0, 5, source_context, source_masks
        )
        self.assertTrue(source_command.valid)
        self.assertEqual(source_command.action[0], 2)
        processor.record_collaboration_command(
            3006,
            source_command.action,
            intent=5,
            is_train=True,
            position_slot=5,
        )
        self.assertIsNotNone(processor.pending_collaboration_command)

        def position_echo(frame):
            observation = self._active_position_observation(
                frame,
                sender=12,
                receiver=11,
                raw_x=source_command.action[2],
                raw_z=source_command.action[3],
                sent_frame=3006,
            )
            observation["official_model_info"] = {
                "official": {"runtime_id": 11}
            }
            official = next(
                hero
                for hero in observation["frame_state"]["hero_states"]
                if hero["runtime_id"] == 11
            )
            _set_side_position(official, -6000.0, 0.0, 1)
            enemy_soldier = next(
                actor
                for actor in observation["frame_state"]["npc_states"]
                if actor.get("camp") == 2
                and u.normalize_sub_type(actor.get("sub_type"))
                == "ACTOR_SUB_SOLDIER"
            )
            official["attack_target"] = enemy_soldier["runtime_id"]
            return observation, enemy_soldier["runtime_id"]

        echoed, target_runtime = position_echo(3012)
        processor.process_feature(echoed)
        first_context = processor.command_runtime_context()
        self.assertIsNone(processor.pending_collaboration_command)
        self.assertEqual(
            processor.lane_combat_target_promotion["phase"], "arming"
        )
        self.assertTrue(first_context.lane_promotion_hold)

        stable, stable_target = position_echo(3030)
        self.assertEqual(stable_target, target_runtime)
        processor.process_feature(stable)
        stable_context = processor.command_runtime_context()
        self.assertTrue(stable_context.lane_promotion_active)
        self.assertTrue(stable_context.lane_promotion_hold)

        for frame in (3054, 3078, 3102):
            waiting, waiting_target = position_echo(frame)
            self.assertEqual(waiting_target, target_runtime)
            processor.process_feature(waiting)
            self.assertTrue(
                processor.command_runtime_context().lane_promotion_hold
            )

        ready, ready_target = position_echo(3107)
        self.assertEqual(ready_target, target_runtime)
        processor.process_feature(ready)
        ready_context = processor.command_runtime_context()
        ready_masks = build_command_legal_masks_v1(ready_context)
        self.assertTrue(ready_context.lane_promotion_active)
        self.assertFalse(ready_context.lane_promotion_hold)
        self.assertFalse(ready_masks.command_is_train)
        target_slot = int(ready_context.lane_promotion_target_slot)
        self.assertEqual(ready_context.target_runtime_ids[target_slot], target_runtime)
        forced = CommandRuleLegalizerV1().legalize(
            5, target_slot, 0, ready_context, ready_masks
        )
        self.assertTrue(forced.valid)
        self.assertEqual(forced.action, (1, target_runtime, 0, 0))
        processor.record_collaboration_command(
            3107,
            forced.action,
            intent=5,
            is_train=False,
            target_slot=target_slot,
        )

        active = self._active_target_observation(
            3113, 12, 11, target_runtime, 3107
        )
        active["official_model_info"] = {"official": {"runtime_id": 11}}
        _set_side_position(
            next(
                hero
                for hero in active["frame_state"]["hero_states"]
                if hero["runtime_id"] == 11
            ),
            -6000.0,
            0.0,
            1,
        )
        processor.process_feature(active)
        active_context = processor.command_runtime_context()
        self.assertEqual(
            processor.lane_combat_target_promotion["phase"], "target_active"
        )
        self.assertEqual(active_context.unset_reason, "none")

        missing = self._active_target_observation(
            3119, 12, 11, target_runtime, 3107
        )
        missing["official_model_info"] = {"official": {"runtime_id": 11}}
        _set_side_position(
            next(
                hero
                for hero in missing["frame_state"]["hero_states"]
                if hero["runtime_id"] == 11
            ),
            -6000.0,
            0.0,
            1,
        )
        missing["frame_state"]["npc_states"] = [
            actor
            for actor in missing["frame_state"]["npc_states"]
            if actor.get("runtime_id") != target_runtime
        ]
        processor.process_feature(missing)
        grace_context = processor.command_runtime_context()
        self.assertTrue(grace_context.lane_promotion_hold)
        self.assertNotEqual(grace_context.unset_reason, "target_invalid")

        retreat = self._active_target_observation(
            3125, 12, 11, target_runtime, 3107
        )
        retreat["official_model_info"] = {"official": {"runtime_id": 11}}
        official = next(
            hero
            for hero in retreat["frame_state"]["hero_states"]
            if hero["runtime_id"] == 11
        )
        _set_side_position(official, -6000.0, 0.0, 1)
        _set_hp(official, 0.28)
        processor.process_feature(retreat)
        retreat_context = processor.command_runtime_context()
        retreat_masks = build_command_legal_masks_v1(retreat_context)
        self.assertTrue(retreat_context.lane_retreat_handoff_active)
        self.assertEqual(retreat_context.unset_reason, "low_hp_active_command")
        self.assertEqual(np.flatnonzero(retreat_masks.intent > 0.5).tolist(), [1])
        unset = CommandRuleLegalizerV1().legalize(
            1, 0, 0, retreat_context, retreat_masks
        )
        self.assertEqual(unset.action, (3, 0, 0, 0))
        processor.record_collaboration_command(
            3125, unset.action, intent=1, is_train=False
        )

        acknowledged = make_observation(frame=3131, camp=1)
        acknowledged["official_model_info"] = {
            "official": {"runtime_id": 11}
        }
        for item in acknowledged["frame_state"]["collaboration_state"][
            "micro_last"
        ]:
            if item["sender"] == 12:
                item["frame_no"] = 3125
        acknowledged["frame_state"]["collaboration_state"][
            "micro_state"
        ] = []
        processor.process_feature(acknowledged)
        final_context = processor.command_runtime_context()
        self.assertFalse(final_context.lane_retreat_handoff_active)
        self.assertEqual(
            processor.lane_combat_target_promotion["phase"], "inactive"
        )

    def test_generic_target_failure_lock_resets_on_entity_lifecycle_change(self):
        processor = FeatureProcessV1(1)
        official = {
            "runtime_id": 11,
            "config_id": 132,
            "camp": 1,
            "hp": 3000,
            "max_hp": 3000,
            "actor_type": "ACTOR_HERO",
            "sub_type": "ACTOR_SUB_NONE",
        }
        target = {
            "runtime_id": 500,
            "config_id": 19,
            "camp": 2,
            "hp": 1000,
            "max_hp": 1000,
            "actor_type": "ACTOR_NPC",
            "sub_type": "ACTOR_SUB_SOLDIER",
        }

        def refresh(current_official, actors, dead=()):
            processor._refresh_command_target_identity_lifecycle(
                {
                    "official_hero": {"actor": current_official},
                    "actor_by_runtime": actors,
                    "dead_runtime_ids": set(dead),
                }
            )

        refresh(official, {11: official, 500: target})
        first = processor._generic_target_goal_signature(
            11, 500, "soldier"
        )
        processor._mark_official_goal_failure(first, 2, 100, "test")
        self.assertTrue(processor._official_goal_failed(first))

        # An unchanged identity remains locked.
        refresh(official, {11: official, 500: target})
        self.assertTrue(processor._official_goal_failed(first))

        # A confirmed death retires the old identity even while the corpse
        # remains in the authoritative actor map.
        dead_target = dict(target, hp=0)
        refresh(
            official,
            {11: official, 500: dead_target},
            dead=(500,),
        )
        self.assertFalse(processor._official_goal_failed(first))

        # Reappearance with the same runtime/config is a new generation.
        refresh(official, {11: official, 500: target})
        second = processor._generic_target_goal_signature(
            11, 500, "soldier"
        )
        self.assertNotEqual(first, second)
        self.assertFalse(processor._official_goal_failed(second))

        processor._mark_official_goal_failure(second, 2, 118, "test")
        reused = dict(target, config_id=70, actor_type="ACTOR_ORGAN")
        refresh(official, {11: official, 500: reused})
        third = processor._generic_target_goal_signature(
            11, 500, "outer_tower"
        )
        self.assertNotEqual(second, third)
        self.assertFalse(processor._official_goal_failed(third))

        # Official respawn/runtime replacement clears generic retry state too.
        processor._mark_official_goal_failure(third, 2, 124, "test")
        new_official = dict(official, runtime_id=12)
        refresh(new_official, {12: new_official, 500: reused})
        fourth = processor._generic_target_goal_signature(
            12, 500, "outer_tower"
        )
        self.assertFalse(processor._official_goal_failed(fourth))

    def test_retreat_handoff_requires_matching_safe_echo_or_confirmed_unset(self):
        processor = FeatureProcessV1(1)
        inputs = V40CommandPromotionTest._promotion_inputs(frame=118)
        context = inputs[0]
        identity = (11, 118, 401, 9)
        processor.lane_combat_target_promotion.update(
            phase="retreat_handoff",
            retreat_position_slot=9,
            handoff_identity=identity,
            handoff_enter_frame=118,
            handoff_official_runtime=11,
            handoff_original_signature=("set_target", 401, None, 700),
            safety_unset_sent_frame=None,
        )

        # Natural expiry or a missing active command is not proof that the
        # official hero received a safe successor.
        context["collaboration"].update(
            active=False,
            type="none",
            source=None,
            state_valid=True,
            platform_last_send_frame=0,
        )
        waiting = V40CommandPromotionTest._advance_promotion(
            processor, inputs
        )
        self.assertTrue(waiting["retreat_handoff_active"])

        # A source from another handoff must not release this latch.
        context["collaboration"].update(
            active=True,
            type="set_position",
            source={
                "action_type": 2,
                "intent": 8,
                "is_train": False,
                "position_slot": 9,
                "handoff_identity": (11, 100, 401, 9),
            },
        )
        mismatch = V40CommandPromotionTest._advance_promotion(
            processor, inputs
        )
        self.assertTrue(mismatch["retreat_handoff_active"])

        context["collaboration"]["source"]["handoff_identity"] = identity
        completed = V40CommandPromotionTest._advance_promotion(
            processor, inputs
        )
        self.assertFalse(completed["active"])
        self.assertEqual(
            processor.lane_combat_target_promotion["phase"], "inactive"
        )

        # The Unset path is separately tied to the command recorded by this
        # handoff and the platform's micro_last acknowledgement.
        processor.lane_combat_target_promotion.update(
            phase="retreat_handoff",
            retreat_position_slot=9,
            handoff_identity=identity,
            handoff_enter_frame=118,
            handoff_official_runtime=11,
            handoff_original_signature=("set_target", 401, None, 700),
            safety_unset_sent_frame=None,
        )
        processor.record_collaboration_command(
            124, (3, 0, 0, 0), intent=1, is_train=False
        )
        context["collaboration"].update(
            active=False,
            type="none",
            source=None,
            state_valid=True,
            platform_last_send_frame=124,
        )
        unset_completed = V40CommandPromotionTest._advance_promotion(
            processor, inputs
        )
        self.assertFalse(unset_completed["active"])
        self.assertEqual(
            processor.lane_combat_target_promotion["phase"], "inactive"
        )


if __name__ == "__main__":
    unittest.main()
