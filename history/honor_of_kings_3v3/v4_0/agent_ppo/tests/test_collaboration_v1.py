#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""Command masks, sampling, and official protocol legalizer tests."""

from dataclasses import replace
import unittest

import numpy as np
import torch

from agent_ppo.feature.collaboration_v1 import (
    CommandRuleLegalizerV1,
    build_command_legal_masks_v1,
)
from agent_ppo.feature.feature_process.v1_feature_process import FeatureProcessV1
from agent_ppo.feature.feature_process import v1_utils as u
from agent_ppo.conf.conf import FrozenV1Config
from agent_ppo.model.command_v1 import (
    _optional_head,
    evaluate_command_actions_v1,
    run_command_guard_self_test_v1,
    sample_command_actions_v1,
    select_intent_row_v1,
    split_conditioned_command_legal_v1,
)
from agent_ppo.tests.test_feature_process_v1 import _raw_position, make_observation


def _context(camp=1, active=False, route=False):
    observation = make_observation(camp=camp)
    if active:
        observation["frame_state"]["collaboration_state"]["micro_last"][0][
            "frame_no"
        ] = 90
        observation["frame_state"]["collaboration_state"]["micro_state"] = [
            {
                "sender": 11,
                "receiver": 13,
                "target_runtime_id": 170,
                "target_position": {"x": 999999, "y": 999999, "z": 999999},
                "end_frame": 700,
            }
        ]
    processor = FeatureProcessV1(camp)
    processor.process_feature(observation)
    context = processor.command_runtime_context()
    return context if route else replace(context, jungle_route_active=False)


def _fake_command_output(intent=2, target=1, position=4, requires_grad=False):
    intent_logits = torch.full((1, 1, 11), -5.0, requires_grad=requires_grad)
    target_logits = torch.full((1, 1, 11, 39), -5.0, requires_grad=requires_grad)
    position_logits = torch.full((1, 1, 11, 30), -5.0, requires_grad=requires_grad)
    intent_logits.data[..., intent] = 10.0
    target_logits.data[..., intent, target] = 10.0
    position_logits.data[..., intent, position] = 10.0
    return {
        "intent_logits": intent_logits,
        "target_logits": target_logits,
        "position_logits": position_logits,
    }


def _torch_masks(masks):
    return (
        torch.from_numpy(masks.intent).reshape(1, 1, 11),
        torch.from_numpy(masks.target_by_intent).reshape(1, 1, 11, 39),
        torch.from_numpy(masks.position_by_intent).reshape(1, 1, 11, 30),
    )


class CommandLegalizerV1Test(unittest.TestCase):
    def test_dimension_errors_follow_frozen_command_contract(self):
        intent_width = FrozenV1Config.COMMAND_HEADS[0]
        with self.assertRaisesRegex(
            ValueError,
            rf"must end in \[{intent_width},3\]",
        ):
            select_intent_row_v1(
                torch.zeros(1, intent_width - 1, 3),
                torch.zeros(1, dtype=torch.long),
                3,
            )
        with self.assertRaisesRegex(
            ValueError,
            rf"\[0,{intent_width}\) range",
        ):
            select_intent_row_v1(
                torch.zeros(1, intent_width, 3),
                torch.full((1,), intent_width, dtype=torch.long),
                3,
            )
        with self.assertRaisesRegex(
            ValueError,
            rf"must end in {FrozenV1Config.COMMAND_LEGAL_MASK_DIM}",
        ):
            split_conditioned_command_legal_v1(
                torch.zeros(1, FrozenV1Config.COMMAND_LEGAL_MASK_DIM - 1)
            )

    @staticmethod
    def _monster(config_id, relation, camp, runtime_id):
        anchor = __import__(
            "agent_ppo.feature.feature_process.v1_utils", fromlist=["fixed_monster_anchor"]
        ).fixed_monster_anchor(config_id, relation, camp)
        monster_camp = camp if relation == "own" else 3 - camp if relation == "enemy" else 0
        return {
            "runtime_id": runtime_id,
            "config_id": config_id,
            "camp": monster_camp,
            "actor_type": "ACTOR_MONSTER",
            "sub_type": "ACTOR_SUB_NONE",
            "location": {"x": anchor[0], "y": 0, "z": anchor[1]},
            "camp_visible": [1, 1],
            "hp": 3000,
            "max_hp": 3000,
            "abilities": [0] * 23,
        }

    def test_intent_target_and_position_masks(self):
        masks = build_command_legal_masks_v1(_context())
        self.assertEqual(masks.intent.shape, (11,))
        self.assertEqual(masks.target_by_intent.shape, (11, 39))
        self.assertEqual(masks.position_by_intent.shape, (11, 30))
        self.assertEqual(masks.intent[0], 1)
        self.assertEqual(masks.intent[1], 0)
        self.assertEqual(masks.intent[2], 1)
        self.assertTrue(np.all(masks.target_by_intent[2, 1:4] == 1))
        self.assertEqual(masks.target_by_intent[3, 38], 1)
        self.assertTrue(np.all(masks.position_by_intent[5, [4, 5]] == 1))
        self.assertTrue(np.all(masks.position_by_intent[8, [6, 7, 9]] == 1))
        self.assertTrue(masks.command_is_train)
        self.assertEqual(masks.conditioned(2).shape, (80,))

    def test_platform_reloaded_context_class_is_rebuilt_by_field_contract(self):
        context = _context()
        foreign_type = type(
            "CommandRuntimeContextV1",
            (),
            {"__module__": "model_pool_snapshot.agent_ppo.feature.collaboration_v1"},
        )
        foreign_context = foreign_type()
        for name, value in vars(context).items():
            setattr(foreign_context, name, value)

        expected = build_command_legal_masks_v1(context)
        actual = build_command_legal_masks_v1(foreign_context)
        np.testing.assert_array_equal(actual.intent, expected.intent)
        np.testing.assert_array_equal(
            actual.target_by_intent, expected.target_by_intent
        )
        np.testing.assert_array_equal(
            actual.position_by_intent, expected.position_by_intent
        )
        self.assertEqual(actual.command_is_train, expected.command_is_train)
        original_action = CommandRuleLegalizerV1().legalize(
            2, 1, 0, context, expected
        )
        reloaded_action = CommandRuleLegalizerV1().legalize(
            2, 1, 0, foreign_context, actual
        )
        self.assertEqual(reloaded_action, original_action)

    def test_incompatible_reloaded_context_still_fails_closed(self):
        foreign_type = type(
            "CommandRuntimeContextV1",
            (),
            {"__module__": "model_pool_snapshot.agent_ppo.feature.collaboration_v1"},
        )
        with self.assertRaisesRegex(TypeError, "missing"):
            build_command_legal_masks_v1(foreign_type())

    def test_protocol_mapping_without_policy_rescoring(self):
        context = _context()
        masks = build_command_legal_masks_v1(context)
        legalizer = CommandRuleLegalizerV1()
        self.assertEqual(legalizer.legalize(0, 0, 0, context, masks).action, (0, 0, 0, 0))
        focus = legalizer.legalize(2, 1, 0, context, masks)
        self.assertTrue(focus.valid)
        self.assertEqual(focus.action, (1, 21, 0, 0))
        structure = legalizer.legalize(3, 38, 0, context, masks)
        self.assertEqual(structure.action, (1, 170, 0, 0))
        push = legalizer.legalize(5, 0, 4, context, masks)
        self.assertTrue(push.valid)
        self.assertEqual(push.action[0], 2)
        invalid = legalizer.legalize(2, 38, 0, context, masks)
        self.assertFalse(invalid.valid)
        self.assertEqual(invalid.action, (0, 0, 0, 0))

    def test_unset_ignores_ordinary_cooldown_and_receiver_death(self):
        context = _context(active=True)
        context = replace(context, official_alive=False, cooldown_ready=False)
        masks = build_command_legal_masks_v1(context)
        self.assertEqual(masks.intent.tolist(), [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        result = CommandRuleLegalizerV1().legalize(1, 0, 0, context, masks)
        self.assertTrue(result.valid)
        self.assertEqual(result.action, (3, 0, 0, 0))

    def test_live_safe_active_command_is_kept_without_unset(self):
        context = _context(active=True)
        masks = build_command_legal_masks_v1(context)
        self.assertEqual(masks.intent[0], 1)
        self.assertEqual(masks.intent[1], 0)
        keep = CommandRuleLegalizerV1().legalize(0, 0, 0, context, masks)
        self.assertEqual(keep.action, (0, 0, 0, 0))
        self.assertFalse(masks.command_is_train)

    def test_live_unsafe_active_command_can_be_unset(self):
        context = replace(
            _context(active=True),
            unset_allowed=True,
            unset_reason="unsafe_enemy_side_command",
        )
        masks = build_command_legal_masks_v1(context)
        self.assertEqual(masks.intent[1], 1)
        unset = CommandRuleLegalizerV1().legalize(1, 0, 0, context, masks)
        self.assertEqual(unset.action, (3, 0, 0, 0))

    def test_missing_protocol_state_disables_new_commands(self):
        context = replace(_context(), state_valid=False)
        masks = build_command_legal_masks_v1(context)
        self.assertEqual(masks.intent.tolist(), [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        self.assertFalse(masks.command_is_train)

    def test_camp_two_position_is_converted_back_to_raw_coordinates(self):
        context = _context(camp=2)
        masks = build_command_legal_masks_v1(context)
        result = CommandRuleLegalizerV1().legalize(6, 0, 1, context, masks)
        self.assertTrue(result.valid)
        self.assertEqual(result.action[0], 2)
        self.assertEqual(result.action[2], 18000)

    def test_enemy_camp_position_does_not_depend_on_enemy_outer(self):
        observation = make_observation(camp=1)
        for spawn in observation["frame_state"]["monster_spawn_infos"]:
            spawn["spawn_timer"] = 0
        observation["frame_state"]["npc_states"].extend(
            [
                self._monster(30, "own", 1, 501),
                self._monster(30, "enemy", 1, 502),
            ]
        )

        def masks_for(value):
            processor = FeatureProcessV1(1)
            processor.process_feature(value)
            context = replace(processor.command_runtime_context(), jungle_route_active=False)
            return build_command_legal_masks_v1(context)

        locked = masks_for(observation)
        own_index = 8 + list(__import__(
            "agent_ppo.feature.schema_v1", fromlist=["MONSTER_TARGET_CONFIG_AND_SIDE"]
        ).MONSTER_TARGET_CONFIG_AND_SIDE).index((30, "own"))
        enemy_index = 8 + list(__import__(
            "agent_ppo.feature.schema_v1", fromlist=["MONSTER_TARGET_CONFIG_AND_SIDE"]
        ).MONSTER_TARGET_CONFIG_AND_SIDE).index((30, "enemy"))
        # Official Kai is still far, so the robust first stage is a camp
        # position rather than a monster runtime target.
        self.assertEqual(locked.target_by_intent[9, own_index], 0)
        self.assertEqual(locked.target_by_intent[9, enemy_index], 0)
        self.assertEqual(locked.position_by_intent[10, 10 + own_index - 8], 1)
        self.assertEqual(locked.position_by_intent[10, 10 + enemy_index - 8], 1)

        unlocked_observation = make_observation(camp=1)
        for spawn in unlocked_observation["frame_state"]["monster_spawn_infos"]:
            spawn["spawn_timer"] = 0
        unlocked_observation["frame_state"]["npc_states"] = [
            npc
            for npc in unlocked_observation["frame_state"]["npc_states"]
            if npc.get("config_id") != 70
        ]
        unlocked_observation["frame_state"]["npc_states"].append(
            self._monster(30, "enemy", 1, 502)
        )
        unlocked = masks_for(unlocked_observation)
        self.assertEqual(unlocked.target_by_intent[9, enemy_index], 0)
        self.assertEqual(unlocked.position_by_intent[10, 10 + enemy_index - 8], 1)

    def test_official_kai_ordinary_monster_target_stays_masked_when_near(self):
        observation = make_observation(camp=1)
        for spawn in observation["frame_state"]["monster_spawn_infos"]:
            spawn["spawn_timer"] = 0
        monster = self._monster(30, "own", 1, 501)
        observation["frame_state"]["npc_states"].append(monster)
        official = next(
            hero
            for hero in observation["frame_state"]["hero_states"]
            if hero["runtime_id"] == 13
        )
        official["location"] = dict(monster["location"])
        processor = FeatureProcessV1(1)
        processor.process_feature(observation)
        context = replace(
            processor.command_runtime_context(), jungle_route_active=False
        )
        masks = build_command_legal_masks_v1(context)
        target_index = 8 + list(__import__(
            "agent_ppo.feature.schema_v1",
            fromlist=["MONSTER_TARGET_CONFIG_AND_SIDE"],
        ).MONSTER_TARGET_CONFIG_AND_SIDE).index((30, "own"))
        self.assertEqual(masks.target_by_intent[9, target_index], 0)

    def test_positive_respawn_countdown_never_opens_camp_position(self):
        observation = make_observation(camp=1)
        own_index = list(__import__(
            "agent_ppo.feature.schema_v1", fromlist=["MONSTER_TARGET_CONFIG_AND_SIDE"]
        ).MONSTER_TARGET_CONFIG_AND_SIDE).index((30, "own"))
        observation["frame_state"]["monster_spawn_infos"][own_index]["spawn_timer"] = 30000
        processor = FeatureProcessV1(1)
        processor.process_feature(observation)
        far = build_command_legal_masks_v1(
            replace(processor.command_runtime_context(), jungle_route_active=False)
        )
        self.assertEqual(far.position_by_intent[10, 10 + own_index], 0)

        observation["frame_state"]["monster_spawn_infos"][own_index]["spawn_timer"] = 5000
        processor = FeatureProcessV1(1)
        processor.process_feature(observation)
        soon = build_command_legal_masks_v1(
            replace(processor.command_runtime_context(), jungle_route_active=False)
        )
        self.assertEqual(soon.position_by_intent[10, 10 + own_index], 0)

    def test_idle_official_hero_is_forced_to_refresh_a_meaningful_command(self):
        kai_observation = make_observation(camp=1)
        for hero in kai_observation["frame_state"]["hero_states"]:
            if hero["camp"] == 2:
                hero["location"]["x"] = 50000
        processor = FeatureProcessV1(1)
        processor.process_feature(kai_observation)
        kai = build_command_legal_masks_v1(processor.command_runtime_context())
        self.assertEqual(kai.intent[0], 0)
        self.assertEqual(kai.intent[10], 1)
        self.assertEqual(kai.intent[5], 0)

    def test_official_kai_opening_route_is_constrained_to_upper_waypoint(self):
        context = _context(route=True)
        masks = build_command_legal_masks_v1(context)
        self.assertTrue(context.jungle_route_active)
        self.assertEqual(context.jungle_route_phase, "upper_wolf")
        self.assertEqual(np.flatnonzero(masks.intent).tolist(), [10])
        self.assertEqual(
            np.flatnonzero(masks.position_by_intent[10]).tolist(),
            [context.jungle_route_position_slot],
        )

        camp_two = _context(camp=2, route=True)
        raw_waypoint = camp_two.position_raw_coordinates[
            camp_two.jungle_route_position_slot
        ]
        self.assertGreater(raw_waypoint[0], 0)

    def test_official_kai_near_opening_monster_stays_on_position(self):
        observation = make_observation(camp=1, include_soldiers=False)
        observation["official_model_info"] = {"official": {"runtime_id": 13}}
        observation["frame_state"]["npc_states"].extend(
            [self._monster(44, "own", 1, 501), self._monster(45, "own", 1, 502)]
        )
        wolf = u.side_position(u.fixed_monster_anchor(44, "own", 1), 1)
        kai = next(
            hero for hero in observation["frame_state"]["hero_states"]
            if hero["config_id"] == 193 and hero["camp"] == 1
        )
        kai["location"] = _raw_position(*wolf, 1)
        for spawn in observation["frame_state"]["monster_spawn_infos"]:
            if spawn["monster_id"] in (44, 45):
                spawn["spawn_timer"] = 0
        processor = FeatureProcessV1(1)
        processor.process_feature(observation)
        context = processor.command_runtime_context()
        masks = build_command_legal_masks_v1(context)
        self.assertEqual(context.jungle_route_phase, "upper_wolf")
        self.assertEqual(np.flatnonzero(masks.intent).tolist(), [10])
        self.assertEqual(
            np.flatnonzero(masks.target_by_intent[9]).tolist(), []
        )
        result = CommandRuleLegalizerV1().legalize(
            10, 0, context.jungle_route_position_slot, context, masks
        )
        self.assertTrue(result.valid)
        self.assertEqual(result.action[0], 2)

    def test_official_kai_opening_never_exposes_ordinary_monster_target(self):
        def intent_at_offset(offset):
            observation = make_observation(camp=1, include_soldiers=False)
            observation["frame_state"]["npc_states"].extend(
                [
                    self._monster(44, "own", 1, 511),
                    self._monster(45, "own", 1, 512),
                ]
            )
            wolf = u.side_position(u.fixed_monster_anchor(44, "own", 1), 1)
            kai = next(
                hero for hero in observation["frame_state"]["hero_states"]
                if hero["config_id"] == 193 and hero["camp"] == 1
            )
            kai["location"] = _raw_position(wolf[0] + offset, wolf[1], 1)
            processor = FeatureProcessV1(1)
            processor.process_feature(observation)
            return np.flatnonzero(
                build_command_legal_masks_v1(
                    processor.command_runtime_context()
                ).intent
            ).tolist()

        self.assertEqual(intent_at_offset(4999.0), [10])
        self.assertEqual(intent_at_offset(5001.0), [10])

    def test_inner_archer_companion_runtime_overrides_duplicate_cfg_slot(self):
        processor = FeatureProcessV1(1)
        for config_id in (44, 42, 30, 33):
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
        first = make_observation(frame=100, camp=1, include_soldiers=False)
        inner_major = self._monster(32, "own", 1, 701)
        inner_minor = self._monster(33, "own", 1, 702)
        inner_minor["location"] = {"x": -18460, "y": 0, "z": -27091}
        first["frame_state"]["npc_states"].extend(
            [inner_major, inner_minor]
        )
        inner = u.side_position(u.fixed_monster_anchor(32, "own", 1), 1)
        first_kai = next(
            hero for hero in first["frame_state"]["hero_states"]
            if hero["config_id"] == 193 and hero["camp"] == 1
        )
        first_kai["location"] = _raw_position(*inner, 1)
        processor.process_feature(first)

        second = make_observation(frame=106, camp=1, include_soldiers=False)
        second_minor = self._monster(33, "own", 1, 702)
        second_minor["location"] = {"x": -18460, "y": 0, "z": -27091}
        second["frame_state"]["npc_states"].append(second_minor)
        second_kai = next(
            hero for hero in second["frame_state"]["hero_states"]
            if hero["config_id"] == 193 and hero["camp"] == 1
        )
        second_kai["location"] = _raw_position(*inner, 1)
        processor.process_feature(second)
        context = processor.command_runtime_context()
        masks = build_command_legal_masks_v1(context)
        self.assertFalse(context.jungle_route_active)
        self.assertEqual(
            context.target_runtime_ids[context.jungle_route_target_slot],
            702,
        )
        self.assertEqual(masks.target_by_intent[9].sum(), 0.0)
        self.assertEqual(masks.position_by_intent[10, 16], 1.0)
        self.assertIn(10, np.flatnonzero(masks.intent).tolist())

    def test_rejected_target_releases_local_hold_and_blocks_lifecycle(self):
        processor = FeatureProcessV1(1)
        processor.process_feature(make_observation(frame=100, camp=1))
        processor.record_collaboration_command(100, [1, 401, 0, 0])

        processor.process_feature(make_observation(frame=106, camp=1))
        pending = processor._memo_context["collaboration"]
        self.assertEqual(pending["pending_command_frame"], 100)
        self.assertFalse(pending["cooldown_ready"])

        processor.process_feature(make_observation(frame=118, camp=1))
        rejected = processor._memo_context["collaboration"]
        self.assertEqual(rejected["pending_command_frame"], 0)
        self.assertEqual(rejected["rejected_target_runtime_count"], 1)
        self.assertTrue(rejected["cooldown_ready"])
        context = processor.command_runtime_context()
        slot = context.target_runtime_ids.index(401)
        self.assertFalse(context.collaboration_target_legal[slot])

    def test_active_opening_command_is_held_then_refreshed(self):
        context = replace(
            _context(route=True),
            current_command_active=True,
            cooldown_ready=False,
        )
        held = build_command_legal_masks_v1(context)
        self.assertEqual(np.flatnonzero(held.intent).tolist(), [0])
        self.assertFalse(held.command_is_train)

        refreshed = build_command_legal_masks_v1(
            replace(context, cooldown_ready=True)
        )
        self.assertEqual(np.flatnonzero(refreshed.intent).tolist(), [10])
        self.assertFalse(refreshed.command_is_train)

    def test_same_active_opening_waypoint_is_not_refreshed(self):
        processor = FeatureProcessV1(1)
        first = make_observation(frame=100, camp=1)
        processor.process_feature(first)
        waypoint = processor._memo_context["jungle_macro"]["next_waypoint"]
        raw_waypoint = u.raw_position(waypoint, 1)

        active = make_observation(frame=206, camp=1)
        official_kai = next(
            hero
            for hero in active["frame_state"]["hero_states"]
            if hero["camp"] == 1 and hero["config_id"] == 193
        )
        official_kai["location"] = _raw_position(-20000.0, -1000.0, 1)
        collaboration = active["frame_state"]["collaboration_state"]
        collaboration["micro_last"][0]["frame_no"] = 100
        collaboration["micro_state"] = [
            {
                "sender": 11,
                "receiver": 13,
                "target_runtime_id": 0,
                "target_position": {
                    "x": raw_waypoint[0],
                    "y": 0,
                    "z": raw_waypoint[1],
                },
                "end_frame": 700,
            }
        ]
        processor.process_feature(active)
        context = processor.command_runtime_context()
        self.assertTrue(context.cooldown_ready)
        masks = build_command_legal_masks_v1(context)
        self.assertEqual(np.flatnonzero(masks.intent).tolist(), [0])

    def test_official_lane_opening_command_cannot_be_unset(self):
        context = replace(
            _context(route=True),
            official_config_id=141,
            jungle_route_active=False,
            lane_route_active=True,
            lane_route_phase="corridor",
            lane_route_interrupt="none",
            current_command_active=True,
            cooldown_ready=False,
        )
        held = build_command_legal_masks_v1(context)
        self.assertEqual(np.flatnonzero(held.intent).tolist(), [0])
        refreshed = build_command_legal_masks_v1(
            replace(context, cooldown_ready=True)
        )
        self.assertEqual(np.flatnonzero(refreshed.intent).tolist(), [5])

    def test_official_lane_enemy_contact_forces_target_and_never_unset(self):
        context = replace(
            _context(route=True),
            official_config_id=141,
            jungle_route_active=False,
            lane_route_active=True,
            lane_route_phase="engage_wave",
            lane_route_interrupt="enemy_hero",
            lane_route_intruder_runtime=21,
            current_command_active=True,
            unset_allowed=True,
            unset_reason="unsafe_enemy_side_command",
        )
        held = build_command_legal_masks_v1(
            replace(context, cooldown_ready=False)
        )
        self.assertEqual(np.flatnonzero(held.intent).tolist(), [0])

        refreshed = build_command_legal_masks_v1(
            replace(context, cooldown_ready=True)
        )
        self.assertEqual(np.flatnonzero(refreshed.intent).tolist(), [2])
        self.assertEqual(refreshed.target_by_intent[2, 1], 1)

    def test_official_lane_structure_defense_is_held_and_never_unset(self):
        context = replace(
            _context(route=True),
            official_config_id=132,
            jungle_route_active=False,
            lane_route_active=True,
            lane_route_phase="corridor",
            lane_route_interrupt="structure_defense",
            current_command_active=True,
            unset_allowed=True,
            unset_reason="target_invalid",
        )
        held = build_command_legal_masks_v1(
            replace(context, cooldown_ready=False)
        )
        self.assertEqual(np.flatnonzero(held.intent).tolist(), [0])

        refreshed = build_command_legal_masks_v1(
            replace(context, cooldown_ready=True)
        )
        self.assertEqual(np.flatnonzero(refreshed.intent).tolist(), [7])

    def test_lane_role_cannot_be_sent_to_jungle_before_enemy_outer_falls(self):
        context = replace(
            _context(route=True),
            official_config_id=141,
            jungle_route_active=False,
            lane_route_active=False,
            lane_jungle_unlocked=False,
            jungle_target_legal=(1.0,) * 20,
            jungle_position_legal=(1.0,) * 20,
            cooldown_ready=True,
        )
        locked = build_command_legal_masks_v1(context)
        self.assertEqual(locked.intent[9], 0)
        self.assertEqual(locked.intent[10], 0)
        self.assertEqual(locked.target_by_intent[9].sum(), 0)
        self.assertEqual(locked.position_by_intent[10].sum(), 0)

        unlocked = build_command_legal_masks_v1(
            replace(context, lane_jungle_unlocked=True)
        )
        self.assertEqual(unlocked.intent[9], 0)
        self.assertEqual(unlocked.intent[10], 1)

    def test_official_kai_camp_exit_forces_next_position_refresh(self):
        context = replace(
            _context(route=True),
            jungle_route_phase="lower_clear",
            jungle_camp_exit_active=True,
            current_command_active=True,
            cooldown_ready=False,
        )
        held = build_command_legal_masks_v1(context)
        self.assertEqual(np.flatnonzero(held.intent).tolist(), [0])

        refreshed = build_command_legal_masks_v1(
            replace(context, cooldown_ready=True)
        )
        self.assertEqual(np.flatnonzero(refreshed.intent).tolist(), [10])
        self.assertEqual(refreshed.target_by_intent.sum(), 0)

    def test_official_kai_first_lower_rotation_cannot_be_unset(self):
        context = replace(
            _context(route=True),
            jungle_route_phase="lower_clear",
            jungle_opening_rotation_active=True,
            current_command_active=True,
            unset_allowed=True,
            unset_reason="unsafe_enemy_side_command",
        )
        held = build_command_legal_masks_v1(
            replace(context, cooldown_ready=False)
        )
        self.assertEqual(np.flatnonzero(held.intent).tolist(), [0])
        refreshed = build_command_legal_masks_v1(
            replace(context, cooldown_ready=True)
        )
        self.assertNotIn(1, np.flatnonzero(refreshed.intent).tolist())
        self.assertEqual(np.count_nonzero(refreshed.intent), 1)

    def test_official_kai_team_macro_exposes_push_group_and_outer_defense(self):
        context = _context(route=True)
        push = replace(
            context,
            jungle_route_phase="control_cycle",
            jungle_route_mode="push",
            jungle_route_interrupt="none",
            jungle_route_target_slot=-1,
            jungle_route_position_slot=-1,
        )
        push_masks = build_command_legal_masks_v1(push)
        self.assertEqual(push_masks.intent[3], 1)
        self.assertEqual(push_masks.target_by_intent[3, 38], 1)

        group = replace(
            push,
            jungle_route_mode="group",
            jungle_group_position_slot=1,
        )
        group_masks = build_command_legal_masks_v1(group)
        self.assertEqual(group_masks.intent[6], 1)
        self.assertEqual(group_masks.position_by_intent[6, 1], 1)

        defense = replace(
            push,
            jungle_route_mode="farm",
            jungle_route_interrupt="structure_defense",
            jungle_defense_position_slot=6,
        )
        defense_masks = build_command_legal_masks_v1(defense)
        self.assertEqual(defense_masks.intent[7], 1)
        self.assertEqual(defense_masks.position_by_intent[7, 6], 1)

    def test_official_kai_opening_does_not_divert_to_lane_soak(self):
        processor = FeatureProcessV1(1)
        alive = make_observation(frame=100, camp=1)
        alive["frame_state"]["npc_states"].extend(
            [self._monster(44, "own", 1, 501), self._monster(45, "own", 1, 502)]
        )
        for spawn in alive["frame_state"]["monster_spawn_infos"]:
            spawn["spawn_timer"] = 0
        processor.process_feature(alive)

        cleared = make_observation(frame=106, camp=1)
        for spawn in cleared["frame_state"]["monster_spawn_infos"]:
            spawn["spawn_timer"] = (
                60000 if spawn["monster_id"] in (44, 45) else 0
            )
        kai = next(
            hero for hero in cleared["frame_state"]["hero_states"]
            if hero["config_id"] == 193 and hero["camp"] == 1
        )
        wolf_anchor = u.side_position(
            u.fixed_monster_anchor(44, "own", 1), 1
        )
        kai["location"] = _raw_position(*wolf_anchor, 1)
        processor.process_feature(cleared)

        soaking = make_observation(frame=112, camp=1)
        for spawn in soaking["frame_state"]["monster_spawn_infos"]:
            spawn["spawn_timer"] = (
                60000 if spawn["monster_id"] in (44, 45) else 0
            )
        kai = next(
            hero for hero in soaking["frame_state"]["hero_states"]
            if hero["config_id"] == 193 and hero["camp"] == 1
        )
        kai["buff_state"]["buff_skills"] = [
            {"config_id": 90204}, {"config_id": 902070}
        ]
        enemy_soldier = next(
            npc for npc in soaking["frame_state"]["npc_states"]
            if npc.get("sub_type") == "ACTOR_SUB_SOLDIER" and npc["camp"] == 2
        )
        kai["location"] = dict(enemy_soldier["location"])
        processor.process_feature(soaking)
        context = processor.command_runtime_context()
        self.assertEqual(context.jungle_route_phase, "upper_deer")
        self.assertEqual(context.jungle_lane_target_slot, -1)
        masks = build_command_legal_masks_v1(context)
        self.assertEqual(np.flatnonzero(masks.intent).tolist(), [10])

    def test_official_kai_focuses_a_favorable_nearby_enemy_encounter(self):
        observation = make_observation(camp=1)
        enemy = observation["frame_state"]["hero_states"][3]
        enemy["location"] = {
            "x": -18000,
            "y": 0,
            "z": 24000,
        }
        kai = next(
            hero for hero in observation["frame_state"]["hero_states"]
            if hero["config_id"] == 193 and hero["camp"] == 1
        )
        kai["location"] = {"x": -19000, "y": 0, "z": 24000}
        lane_hero = next(
            hero for hero in observation["frame_state"]["hero_states"]
            if hero["config_id"] == 132 and hero["camp"] == 1
        )
        lane_hero["hp"] = 0
        processor = FeatureProcessV1(1)
        processor.process_feature(observation)
        context = processor.command_runtime_context()
        masks = build_command_legal_masks_v1(context)
        self.assertEqual(context.jungle_route_interrupt, "contest_enemy")
        self.assertEqual(np.flatnonzero(masks.intent).tolist(), [2])
        result = CommandRuleLegalizerV1().legalize(2, 1, 0, context, masks)
        self.assertTrue(result.valid)
        self.assertEqual(result.action, (1, 21, 0, 0))

        observation = make_observation(camp=1, include_soldiers=False)
        observation["official_model_info"] = {"official": {"runtime_id": 11}}
        for hero in observation["frame_state"]["hero_states"]:
            if hero["camp"] == 2:
                hero["location"]["x"] = 50000
        processor = FeatureProcessV1(1)
        processor.process_feature(observation)
        lane = build_command_legal_masks_v1(processor.command_runtime_context())
        self.assertEqual(lane.intent[0], 0)
        self.assertEqual(lane.intent[5], 1)


class CommandSamplingV1Test(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(31)

    def test_target_intent_activates_only_target_head(self):
        masks = build_command_legal_masks_v1(_context())
        intent, target, position = _torch_masks(masks)
        sampled = sample_command_actions_v1(
            _fake_command_output(intent=2, target=1),
            intent,
            target,
            position,
            deterministic=True,
        )
        torch.testing.assert_close(sampled["actions"], torch.tensor([[[2, 1, 0]]]))
        torch.testing.assert_close(sampled["active_head_mask"], torch.tensor([[[1.0, 1.0, 0.0]]]))
        self.assertEqual(sampled["old_log_probabilities"][..., 2].item(), 0)
        self.assertEqual(sampled["conditioned_legal_mask"].shape[-1], 80)

    def test_position_intent_activates_only_position_head(self):
        masks = build_command_legal_masks_v1(_context())
        intent, target, position = _torch_masks(masks)
        sampled = sample_command_actions_v1(
            _fake_command_output(intent=5, position=4),
            intent,
            target,
            position,
            deterministic=True,
        )
        torch.testing.assert_close(sampled["actions"], torch.tensor([[[5, 0, 4]]]))
        torch.testing.assert_close(sampled["active_head_mask"], torch.tensor([[[1.0, 0.0, 1.0]]]))
        self.assertEqual(sampled["old_log_probabilities"][..., 1].item(), 0)

    def test_no_command_has_zero_inactive_actions_and_logs(self):
        masks = build_command_legal_masks_v1(_context())
        intent, target, position = _torch_masks(masks)
        intent.zero_()
        intent[..., 0] = 1
        sampled = sample_command_actions_v1(
            _fake_command_output(intent=0),
            intent,
            target,
            position,
            deterministic=True,
        )
        torch.testing.assert_close(sampled["actions"], torch.tensor([[[0, 0, 0]]]))
        self.assertEqual(sampled["old_log_probabilities"][..., 1:].abs().sum(), 0)

    def test_inconsistent_position_intent_falls_back_to_no_command(self):
        masks = build_command_legal_masks_v1(_context())
        intent, target, position = _torch_masks(masks)
        intent.zero_()
        intent[..., 5] = 1
        position[..., 5, :].zero_()
        sampled = sample_command_actions_v1(
            _fake_command_output(intent=5, position=4),
            intent,
            target,
            position,
            command_is_train=torch.ones(1, 1),
            deterministic=True,
        )
        torch.testing.assert_close(sampled["actions"], torch.tensor([[[0, 0, 0]]]))
        self.assertEqual(sampled["conditioned_legal_mask"][..., 0].item(), 1)
        self.assertEqual(sampled["conditioned_legal_mask"][..., 5].item(), 0)
        self.assertEqual(sampled["command_is_train"].item(), 0)

    def test_nonfinite_no_command_mask_is_repaired(self):
        masks = build_command_legal_masks_v1(_context())
        intent, target, position = _torch_masks(masks)
        intent.zero_()
        intent[..., 0] = float("nan")
        sampled = sample_command_actions_v1(
            _fake_command_output(intent=0),
            intent,
            target,
            position,
            deterministic=True,
        )
        torch.testing.assert_close(sampled["actions"], torch.tensor([[[0, 0, 0]]]))
        self.assertTrue(torch.isfinite(sampled["intent_probabilities"]).all())

    def test_optional_active_all_illegal_is_contained(self):
        result = _optional_head(
            torch.zeros(1, 10),
            torch.zeros(1, 10),
            torch.ones(1, dtype=torch.bool),
            "command position",
            deterministic=True,
        )
        torch.testing.assert_close(result["action"], torch.zeros(1, dtype=torch.long))
        torch.testing.assert_close(
            result["invalid_active"], torch.ones(1, dtype=torch.bool)
        )
        self.assertEqual(result["log_probability"].item(), 0)

    def test_deployment_guard_self_test(self):
        status = run_command_guard_self_test_v1()
        self.assertEqual(
            status,
            "COMMAND_GUARD_SELF_TEST_OK contract=(11, 39, 30) sampler=guarded_v2",
        )

    def test_evaluation_preserves_active_head_gradient(self):
        masks = build_command_legal_masks_v1(_context())
        intent, target, position = _torch_masks(masks)
        output = _fake_command_output(intent=2, target=1, requires_grad=True)
        sampled = sample_command_actions_v1(
            output, intent, target, position, deterministic=True
        )
        evaluated = evaluate_command_actions_v1(
            output,
            sampled["conditioned_legal_mask"],
            sampled["actions"],
            sampled["active_head_mask"],
        )
        loss = -(
            evaluated["selected_log_probabilities"]
            * sampled["active_head_mask"]
        ).sum()
        loss.backward()
        self.assertGreater(output["intent_logits"].grad.abs().sum(), 0)
        self.assertGreater(output["target_logits"].grad[..., 2, :].abs().sum(), 0)
        self.assertEqual(output["position_logits"].grad, None)


if __name__ == "__main__":
    unittest.main()
