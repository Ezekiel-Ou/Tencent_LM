#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""Synthetic protocol tests for the frozen v1 feature builder."""

import copy
import unittest
from dataclasses import replace

import numpy as np

from agent_ppo.conf.conf import FrozenV1Config, V1TrainingConfig
from agent_ppo.feature import schema_v1
from agent_ppo.feature.collaboration_v1 import (
    CommandRuleLegalizerV1,
    build_command_legal_masks_v1,
)
from agent_ppo.feature.feature_process import v1_utils as u
from agent_ppo.feature.feature_process.v1_feature_process import (
    FeatureProcessV1,
    KAI_OPENING_CAMP_ORDER,
    LANE_OFFICIAL_TARGET_COMMAND_DISTANCE,
)


def _raw_position(side_x, side_z, camp):
    raw_x = side_x if camp == 1 else -side_x
    return {"x": raw_x, "y": 0, "z": side_z + u.LANE_CENTER_Z}


def _skills(config_id):
    return {
        "slot_states": [
            {
                "slot_type": f"SLOT_SKILL_{index}",
                "config_id": config_id * 100 + index,
                "level": 1,
                "usable": index in (0, 1, 4, 5),
                "cooldown": 0,
                "cooldown_max": 1000 + index * 1000,
                "succ_used_in_frame": 0,
                "used_times": index,
                "hit_hero_times": 0,
            }
            for index in range(6)
        ]
    }


def _hero(config_id, runtime_id, camp, side_x, side_z, visible_to_camp, official=False):
    max_hp = {132: 3200, 141: 3100, 193: 3600}[config_id]
    return {
        "runtime_id": runtime_id,
        "config_id": config_id,
        "camp": camp,
        "actor_type": "ACTOR_HERO",
        "sub_type": "ACTOR_SUB_NONE",
        "location": _raw_position(side_x, side_z, camp),
        "forward": {"x": 1 if camp == 1 else -1, "y": 0, "z": 0},
        "camp_visible": [int(visible_to_camp == 1), int(visible_to_camp == 2)],
        "hp": max_hp,
        "max_hp": max_hp,
        "ep": 100,
        "max_ep": 100,
        "level": 5,
        "exp": 40,
        "next_level_exp": 100,
        "money": 500,
        "money_cnt": 3000,
        "kill_cnt": 1,
        "dead_cnt": 0,
        "assist_cnt": 2,
        "total_hurt": 1000,
        "revive_time": 0,
        "behav_mode": 2,
        "phy_atk": 300,
        "phy_def": 250,
        "mgc_atk": 200,
        "mgc_def": 150,
        "mov_spd": 5000,
        "atk_spd": 3000,
        "attack_range": 6000,
        "sight_area": 15000,
        "attack_target": 0,
        "abilities": [0] * 23,
        "skill_state": _skills(config_id),
        "buff_state": {
            "buff_skills": [{"config_id": 90201, "start_time": 1000, "times": 1}]
            if official
            else [],
            "buff_marks": [],
        },
        "hit_target_info": [],
        "hurt_hero_info": [],
        "take_hurt_infos": [],
    }


def _organ(config_id):
    camp = 1 if config_id in (66, 67, 68) else 2
    sub_type = {
        66: "ACTOR_SUB_TOWER",
        70: "ACTOR_SUB_TOWER",
        67: "ACTOR_SUB_TOWER_HIGH",
        71: "ACTOR_SUB_TOWER_HIGH",
        68: "ACTOR_SUB_CRYSTAL",
        72: "ACTOR_SUB_CRYSTAL",
    }[config_id]
    max_hp = 8000 if sub_type == "ACTOR_SUB_CRYSTAL" else 10000 if "HIGH" in sub_type else 6000
    return {
        "runtime_id": 100 + config_id,
        "config_id": config_id,
        "camp": camp,
        "actor_type": "ACTOR_ORGAN",
        "sub_type": sub_type,
        "location": {"x": u.ORGAN_ANCHORS[config_id][0], "y": 0, "z": u.ORGAN_ANCHORS[config_id][1]},
        "camp_visible": [1, 1],
        "hp": max_hp,
        "max_hp": max_hp,
        "attack_range": 9000,
        "phy_atk": 600,
        "attack_target": 0,
        "kill_income": 100,
        "abilities": [0] * 23,
    }


def _soldier(runtime_id, camp, side_x, config_id=19, side_z=0):
    return {
        "runtime_id": runtime_id,
        "config_id": config_id,
        "camp": camp,
        "actor_type": "ACTOR_MONSTER",
        "sub_type": "ACTOR_SUB_SOLDIER",
        "location": _raw_position(side_x, side_z, camp),
        "camp_visible": [1, 1],
        "hp": 1500,
        "max_hp": 1800,
        "attack_range": 3000,
        "phy_atk": 100,
        "kill_income": 50,
        "attack_target": 0,
    }


def _monster(config_id, relation, camp, runtime_id):
    anchor = u.fixed_monster_anchor(config_id, relation, camp)
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


def _hero_info(hero):
    return {
        "runtime_id": hero["runtime_id"],
        "config_id": hero["config_id"],
        "feature": np.zeros(FrozenV1Config.IMAGE_FEATURE_DIM, dtype=np.float32),
        "legal_action": np.ones(FrozenV1Config.RAW_LEGAL_ACTION_DIM, dtype=np.float32),
        "sub_action_mask": {str(index): [1, 1, 1, 1, 1] for index in range(13)},
    }


def make_observation(frame=100, camp=1, include_soldiers=True):
    enemy_camp = 3 - camp
    friends = [
        _hero(132, 11, camp, -18000, 0, camp),
        _hero(141, 12, camp, -15000, 1000, camp),
        _hero(193, 13, camp, -12000, -1000, camp, official=True),
    ]
    enemies = [
        _hero(132, 21, enemy_camp, -12000, 0, camp),
        _hero(141, 22, enemy_camp, -15000, 1000, camp),
        _hero(193, 23, enemy_camp, -18000, -1000, camp),
    ]
    npcs = [_organ(config_id) for config_id in (66, 67, 68, 70, 71, 72)]
    if include_soldiers:
        npcs.extend(
            [
                _soldier(301, camp, -2000, 19),
                _soldier(302, camp, -3000, 20),
                _soldier(401, enemy_camp, 2000, 19),
                _soldier(402, enemy_camp, 3000, 21),
            ]
        )
    spawn_infos = []
    for config_id, relation in schema_v1.MONSTER_TARGET_CONFIG_AND_SIDE:
        anchor = u.fixed_monster_anchor(config_id, relation, camp)
        spawn_infos.append(
            {
                "monster_id": config_id,
                "born_pos_x": anchor[0],
                "born_pos_z": anchor[1],
                "spawn_timer": 30000,
            }
        )
    cakes = [
        {
            "config_id": config_id,
            "collider": {
                "location": {
                    "x": u.CAKE_ANCHORS[config_id][0],
                    "y": 0,
                    "z": u.CAKE_ANCHORS[config_id][1],
                }
            },
        }
        for config_id in (102, 103, 104, 105, 106)
    ]
    return {
        "heroes": [_hero_info(hero) for hero in friends],
        "official_model_info": {"official": {"runtime_id": 13}},
        "frame_state": {
            "frame_no": frame,
            "hero_states": friends + enemies,
            "npc_states": npcs,
            "bullets": [],
            "cakes": cakes,
            "monster_spawn_infos": spawn_infos,
            "collaboration_state": {
                "micro_cd": 100,
                "max_duration": 600,
                "micro_last": [
                    {"sender": 11, "frame_no": 0},
                    {"sender": 12, "frame_no": 0},
                    {"sender": 13, "frame_no": 0},
                ],
                "micro_state": [],
            },
            "frame_action": {"dead_action": []},
        },
    }


def _block(feature, name):
    block = schema_v1.FEATURE_BLOCK_BY_NAME[name]
    return feature[block.start : block.stop].reshape(block.shape)


class FeatureProcessV1Test(unittest.TestCase):
    def test_scalar_clip_matches_numpy_reference(self):
        cases = (
            (-float("inf"), -1.0, 1.0),
            (-2.0, -1.0, 1.0),
            (-0.0, 0.0, 1.0),
            (0.25, 0.0, 1.0),
            (2.0, 0.0, 1.0),
            (float("inf"), 0.0, 1.0),
            (float("nan"), 0.0, 1.0),
        )
        for value, low, high in cases:
            expected = float(np.clip(u.as_float(value), low, high))
            actual = u.clip(value, low, high)
            if np.isnan(expected):
                self.assertTrue(np.isnan(actual))
            else:
                self.assertEqual(actual, expected)
                self.assertEqual(np.signbit(actual), np.signbit(expected))

    def test_actor_get_preserves_alias_and_nested_fallback_semantics(self):
        self.assertEqual(u.actor_get({"hp": 7}, "hp", default=0), 7)
        self.assertEqual(
            u.actor_get({"maxHp": 9}, "max_hp", "maxHp", default=0), 9
        )
        self.assertEqual(
            u.actor_get(
                {"hp": None, "actor_state": {"hp": 11}},
                "hp",
                default=0,
            ),
            11,
        )
        self.assertEqual(
            u.actor_get({"actorState": {"hp": 13}}, "hp", default=0),
            13,
        )
        self.assertEqual(u.actor_get(None, "hp", default=17), 17)

    @staticmethod
    def _camp_two_perspective(observation):
        result = copy.deepcopy(observation)
        friends = [
            hero
            for hero in result["frame_state"]["hero_states"]
            if hero["camp"] == 2
        ]
        result["heroes"] = [_hero_info(hero) for hero in friends]
        result["official_model_info"] = {
            "official": {"runtime_id": friends[2]["runtime_id"]}
        }
        return result

    def test_missing_friendly_frame_state_uses_role_preserving_placeholder(self):
        runtime_by_config = {132: 11, 141: 12, 193: 13}
        cases = (
            (132, 193),
            (141, 193),
            (193, 132),
        )

        for missing_config, official_config in cases:
            with self.subTest(
                missing_config=missing_config,
                official_config=official_config,
            ):
                observation = make_observation(frame=100, camp=1)
                observation["official_model_info"] = {
                    "official": {
                        "runtime_id": runtime_by_config[official_config]
                    }
                }
                observation["frame_state"]["hero_states"] = [
                    hero
                    for hero in observation["frame_state"]["hero_states"]
                    if not (
                        hero["camp"] == 1
                        and hero["config_id"] == missing_config
                    )
                ]

                processor = FeatureProcessV1(1)
                features = processor.process_feature(observation)

                self.assertEqual(
                    len(features),
                    FrozenV1Config.CONTROLLED_ACTOR_COUNT,
                )
                missing = next(
                    hero
                    for hero in processor._memo_context["friend_heroes"]
                    if u.actor_config_id(hero["actor"]) == missing_config
                )
                self.assertFalse(missing["current_valid"])
                self.assertFalse(missing["alive"])
                self.assertIsNone(missing["side_position"])
                self.assertEqual(
                    missing["official"],
                    missing_config == official_config,
                )
                self.assertEqual(
                    missing["controlled"],
                    missing_config != official_config,
                )

                returned = make_observation(frame=106, camp=1)
                returned["official_model_info"] = {
                    "official": {
                        "runtime_id": runtime_by_config[official_config]
                    }
                }
                returned_features = processor.process_feature(returned)
                self.assertEqual(
                    len(returned_features),
                    FrozenV1Config.CONTROLLED_ACTOR_COUNT,
                )
                recovered = next(
                    hero
                    for hero in processor._memo_context["friend_heroes"]
                    if u.actor_config_id(hero["actor"]) == missing_config
                )
                self.assertTrue(recovered["current_valid"])
                self.assertTrue(recovered["alive"])
                self.assertIsNotNone(recovered["side_position"])

    def test_custom_kda_is_zero_sum_and_uses_early_death_multiplier(self):
        initial = make_observation(frame=6200, camp=1)
        camp_one = FeatureProcessV1(1)
        camp_two = FeatureProcessV1(2)
        camp_one.process_feature(initial)
        camp_one.consume_behavior_shaping(expected_frame=6200)
        camp_two_initial = self._camp_two_perspective(initial)
        camp_two.process_feature(camp_two_initial)
        camp_two.consume_behavior_shaping(expected_frame=6200)

        current = make_observation(frame=6206, camp=1)
        heroes = current["frame_state"]["hero_states"]
        next(hero for hero in heroes if hero["camp"] == 1 and hero["config_id"] == 132)["kill_cnt"] += 1
        next(hero for hero in heroes if hero["camp"] == 1 and hero["config_id"] == 141)["assist_cnt"] += 1
        next(hero for hero in heroes if hero["camp"] == 2 and hero["config_id"] == 141)["dead_cnt"] += 1
        camp_one.process_feature(current)
        reward_one = camp_one.consume_behavior_shaping(expected_frame=6206)
        camp_two.process_feature(self._camp_two_perspective(current))
        reward_two = camp_two.consume_behavior_shaping(expected_frame=6206)

        expected_controlled = 0.8 + 0.7 / 3.0
        self.assertAlmostEqual(reward_one["kda_auxiliary_rewards"][0], expected_controlled, places=6)
        self.assertAlmostEqual(reward_one["kda_auxiliary_rewards"][1], expected_controlled, places=6)
        mean_one = sum(reward_one["kda_rewards_by_config"].values()) / 3.0
        mean_two = sum(reward_two["kda_rewards_by_config"].values()) / 3.0
        self.assertAlmostEqual(mean_one, -mean_two, places=7)

    def test_custom_kda_late_death_penalty_is_1_3(self):
        processor = FeatureProcessV1(1)
        initial = make_observation(frame=6253, camp=1)
        processor.process_feature(initial)
        processor.consume_behavior_shaping(expected_frame=6253)
        current = make_observation(frame=6254, camp=1)
        current["frame_state"]["hero_states"][0]["dead_cnt"] += 1
        processor.process_feature(current)
        reward = processor.consume_behavior_shaping(expected_frame=6254)
        self.assertAlmostEqual(reward["kda_auxiliary_rewards"][0], -1.3, places=7)

    def test_tyrant_reward_is_team_zero_sum_and_once_per_lifecycle(self):
        processor = FeatureProcessV1(1)
        initial = make_observation(frame=100, camp=1)
        processor.process_feature(initial)
        processor.consume_behavior_shaping(expected_frame=100)

        killed = make_observation(frame=106, camp=1)
        killed["frame_state"]["frame_action"]["dead_action"] = [
            {
                "killer": killed["frame_state"]["hero_states"][2],
                "death": {
                    "runtime_id": 7001,
                    "config_id": 122,
                    "camp": 0,
                    "actor_type": "ACTOR_MONSTER",
                    "sub_type": "ACTOR_SUB_NONE",
                },
            }
        ]
        processor.process_feature(killed)
        reward = processor.consume_behavior_shaping(expected_frame=106)
        self.assertEqual(reward["tyrant_team_reward"], 1.0)
        self.assertEqual(reward["rewards"], (1.0, 1.0))
        self.assertEqual(reward["pushing_auxiliary_rewards"], (1.0, 1.0))

        duplicate = copy.deepcopy(killed)
        duplicate["frame_state"]["frame_no"] = 112
        processor.process_feature(duplicate)
        duplicate_reward = processor.consume_behavior_shaping(expected_frame=112)
        self.assertEqual(duplicate_reward["tyrant_team_reward"], 0.0)

        respawned = make_observation(frame=118, camp=1)
        respawned["frame_state"]["npc_states"].append(
            _monster(122, "center", 1, 7002)
        )
        processor.process_feature(respawned)
        respawn_reward = processor.consume_behavior_shaping(expected_frame=118)
        self.assertEqual(respawn_reward["tyrant_team_reward"], 0.0)
        second_kill = make_observation(frame=124, camp=1)
        second_kill["frame_state"]["frame_action"]["dead_action"] = [
            {
                "killer": second_kill["frame_state"]["hero_states"][2],
                "death": {
                    "runtime_id": 7002,
                    "config_id": 122,
                    "camp": 0,
                    "actor_type": "ACTOR_MONSTER",
                    "sub_type": "ACTOR_SUB_NONE",
                },
            }
        ]
        processor.process_feature(second_kill)
        second_reward = processor.consume_behavior_shaping(expected_frame=124)
        self.assertEqual(second_reward["tyrant_team_reward"], 1.0)

        enemy_processor = FeatureProcessV1(1)
        enemy_processor.process_feature(initial)
        enemy_processor.consume_behavior_shaping(expected_frame=100)
        enemy_kill = copy.deepcopy(killed)
        enemy_kill["frame_state"]["frame_action"]["dead_action"][0]["killer"] = enemy_kill["frame_state"]["hero_states"][3]
        enemy_processor.process_feature(enemy_kill)
        enemy_reward = enemy_processor.consume_behavior_shaping(expected_frame=106)
        self.assertEqual(enemy_reward["tyrant_team_reward"], -1.0)

    def test_tyrant_reward_stops_after_three_kills_for_each_camp(self):
        def lifecycle_rewards(killer_hero_index):
            processor = FeatureProcessV1(1)
            initial = make_observation(frame=100, camp=1)
            processor.process_feature(initial)
            processor.consume_behavior_shaping(expected_frame=100)
            rewards = []

            for kill_index in range(4):
                runtime_id = 7100 + kill_index
                if kill_index:
                    respawn_frame = 100 + kill_index * 12
                    respawned = make_observation(frame=respawn_frame, camp=1)
                    respawned["frame_state"]["npc_states"].append(
                        _monster(122, "center", 1, runtime_id)
                    )
                    processor.process_feature(respawned)
                    respawn_reward = processor.consume_behavior_shaping(
                        expected_frame=respawn_frame
                    )
                    self.assertEqual(respawn_reward["tyrant_team_reward"], 0.0)

                kill_frame = 106 + kill_index * 12
                killed = make_observation(frame=kill_frame, camp=1)
                killed["frame_state"]["frame_action"]["dead_action"] = [
                    {
                        "killer": killed["frame_state"]["hero_states"][
                            killer_hero_index
                        ],
                        "death": {
                            "runtime_id": runtime_id,
                            "config_id": 122,
                            "camp": 0,
                            "actor_type": "ACTOR_MONSTER",
                            "sub_type": "ACTOR_SUB_NONE",
                        },
                    }
                ]
                processor.process_feature(killed)
                reward = processor.consume_behavior_shaping(
                    expected_frame=kill_frame
                )
                rewards.append(reward["tyrant_team_reward"])

            return rewards

        self.assertEqual(lifecycle_rewards(2), [1.0, 1.0, 1.0, 0.0])
        self.assertEqual(lifecycle_rewards(3), [-1.0, -1.0, -1.0, 0.0])

    def test_wolf_initial_lifecycle_is_free_then_respawns_are_zero_sum(self):
        camp_one = FeatureProcessV1(1)
        camp_two = FeatureProcessV1(2)
        initial = make_observation(frame=100, camp=1)
        initial["frame_state"]["npc_states"].extend(
            [
                _monster(44, "own", 1, 7401),
                _monster(45, "own", 1, 7402),
            ]
        )
        camp_one.process_feature(initial)
        camp_one.consume_behavior_shaping(expected_frame=100)
        camp_two.process_feature(self._camp_two_perspective(initial))
        camp_two.consume_behavior_shaping(expected_frame=100)

        major_kill = copy.deepcopy(initial)
        major_kill["frame_state"]["frame_no"] = 106
        major_kill["frame_state"]["npc_states"] = [
            actor
            for actor in major_kill["frame_state"]["npc_states"]
            if actor.get("runtime_id") != 7401
        ]
        major_kill["frame_state"]["frame_action"]["dead_action"] = [
            {
                "killer": major_kill["frame_state"]["hero_states"][2],
                "death": {
                    "runtime_id": 7401,
                    "config_id": 44,
                    "camp": 1,
                    "actor_type": "ACTOR_MONSTER",
                    "sub_type": "ACTOR_SUB_NONE",
                },
            }
        ]
        camp_one.process_feature(major_kill)
        major_one = camp_one.consume_behavior_shaping(expected_frame=106)
        camp_two.process_feature(self._camp_two_perspective(major_kill))
        major_two = camp_two.consume_behavior_shaping(expected_frame=106)
        self.assertEqual(major_one["kai_wolf_team_reward"], 0.0)
        self.assertEqual(major_two["kai_wolf_team_reward"], 0.0)
        self.assertEqual(major_one["kai_wolf_reward_count"], 0)

        duplicate = copy.deepcopy(major_kill)
        duplicate["frame_state"]["frame_no"] = 112
        camp_one.process_feature(duplicate)
        duplicate_reward = camp_one.consume_behavior_shaping(
            expected_frame=112
        )
        self.assertEqual(duplicate_reward["kai_wolf_team_reward"], 0.0)
        self.assertEqual(duplicate_reward["kai_wolf_reward_count"], 0)

        companion_kill = copy.deepcopy(major_kill)
        companion_kill["frame_state"]["frame_no"] = 118
        companion_kill["frame_state"]["npc_states"] = [
            actor
            for actor in companion_kill["frame_state"]["npc_states"]
            if actor.get("runtime_id") != 7402
        ]
        companion_kill["frame_state"]["frame_action"]["dead_action"][0][
            "death"
        ].update(runtime_id=7402, config_id=45)
        camp_one.process_feature(companion_kill)
        companion_reward = camp_one.consume_behavior_shaping(
            expected_frame=118
        )
        self.assertEqual(companion_reward["kai_wolf_team_reward"], 0.0)

        wolf_anchor = tuple(u.fixed_monster_anchor(44, "own", 1))
        camp_one.monster_camp_history[wolf_anchor]["clear_count"] = 1
        camp_two_wolf_anchor = tuple(
            u.fixed_monster_anchor(44, "enemy", 2)
        )
        camp_two.monster_camp_history[camp_two_wolf_anchor][
            "clear_count"
        ] = 1
        respawned = make_observation(frame=124, camp=1)
        respawned["frame_state"]["npc_states"].extend(
            [
                _monster(44, "own", 1, 7501),
                _monster(45, "own", 1, 7502),
            ]
        )
        camp_one.process_feature(respawned)
        camp_one.consume_behavior_shaping(expected_frame=124)
        camp_two.process_feature(self._camp_two_perspective(respawned))
        camp_two.consume_behavior_shaping(expected_frame=124)

        respawn_major_kill = copy.deepcopy(respawned)
        respawn_major_kill["frame_state"]["frame_no"] = 130
        respawn_major_kill["frame_state"]["npc_states"] = [
            actor
            for actor in respawn_major_kill["frame_state"]["npc_states"]
            if actor.get("runtime_id") != 7501
        ]
        respawn_major_kill["frame_state"]["frame_action"]["dead_action"] = [
            {
                # Marco may secure the objective; Kai participation is not
                # required after the protected opening.
                "killer": respawn_major_kill["frame_state"]["hero_states"][0],
                "death": {
                    "runtime_id": 7501,
                    "config_id": 44,
                    "camp": 1,
                    "actor_type": "ACTOR_MONSTER",
                    "sub_type": "ACTOR_SUB_NONE",
                },
            }
        ]
        camp_one.process_feature(respawn_major_kill)
        respawn_major_one = camp_one.consume_behavior_shaping(
            expected_frame=130
        )
        camp_two.process_feature(
            self._camp_two_perspective(respawn_major_kill)
        )
        respawn_major_two = camp_two.consume_behavior_shaping(
            expected_frame=130
        )
        self.assertAlmostEqual(
            respawn_major_one["kai_wolf_team_reward"], 0.40
        )
        self.assertAlmostEqual(
            respawn_major_two["kai_wolf_team_reward"], -0.40
        )
        np.testing.assert_allclose(
            respawn_major_one["farming_auxiliary_rewards"], (0.40, 0.40)
        )

        respawn_companion_kill = copy.deepcopy(respawn_major_kill)
        respawn_companion_kill["frame_state"]["frame_no"] = 136
        respawn_companion_kill["frame_state"]["npc_states"] = [
            actor
            for actor in respawn_companion_kill["frame_state"]["npc_states"]
            if actor.get("runtime_id") != 7502
        ]
        respawn_companion_kill["frame_state"]["frame_action"]["dead_action"][0][
            "death"
        ].update(runtime_id=7502, config_id=45)
        camp_one.process_feature(respawn_companion_kill)
        respawn_companion_reward = camp_one.consume_behavior_shaping(
            expected_frame=136
        )
        self.assertAlmostEqual(
            respawn_companion_reward["kai_wolf_team_reward"], 0.10
        )
        self.assertAlmostEqual(
            respawn_major_one["kai_wolf_team_reward"]
            + respawn_companion_reward["kai_wolf_team_reward"],
            0.50,
        )

    def test_treasure_thief_reward_is_team_zero_sum_and_deduplicated(self):
        processor = FeatureProcessV1(1)
        initial = make_observation(frame=100, camp=1)
        processor.process_feature(initial)
        processor.consume_behavior_shaping(expected_frame=100)

        killed = make_observation(frame=106, camp=1)
        killed["frame_state"]["frame_action"]["dead_action"] = [
            {
                "killer": killed["frame_state"]["hero_states"][0],
                "death": {
                    "runtime_id": 7601,
                    "config_id": 59,
                    "camp": 0,
                    "actor_type": "ACTOR_MONSTER",
                    "sub_type": "ACTOR_SUB_NONE",
                },
            }
        ]
        processor.process_feature(killed)
        reward = processor.consume_behavior_shaping(expected_frame=106)
        self.assertEqual(reward["treasure_thief_team_reward"], 0.50)
        self.assertEqual(reward["treasure_thief_reward_count"], 1)
        self.assertEqual(reward["rewards"], (0.50, 0.50))
        self.assertEqual(reward["farming_auxiliary_rewards"], (0.50, 0.50))

        duplicate = copy.deepcopy(killed)
        duplicate["frame_state"]["frame_no"] = 112
        processor.process_feature(duplicate)
        duplicate_reward = processor.consume_behavior_shaping(
            expected_frame=112
        )
        self.assertEqual(duplicate_reward["treasure_thief_team_reward"], 0.0)

    def test_speed_buff_acquisition_is_team_zero_sum_and_edge_triggered(self):
        processor = FeatureProcessV1(1)
        initial = make_observation(frame=100, camp=1)
        processor.process_feature(initial)
        processor.consume_behavior_shaping(expected_frame=100)

        picked = make_observation(frame=106, camp=1)
        picked["frame_state"]["hero_states"][0]["buff_state"][
            "buff_skills"
        ] = [
            {"config_id": 90201, "start_time": 106, "times": 1}
        ]
        processor.process_feature(picked)
        reward = processor.consume_behavior_shaping(expected_frame=106)
        self.assertEqual(reward["speed_buff_team_reward"], 0.12)
        self.assertEqual(reward["speed_buff_reward_count"], 1)
        np.testing.assert_allclose(reward["rewards"], (0.12, 0.12))
        np.testing.assert_allclose(
            reward["farming_auxiliary_rewards"], (0.12, 0.12)
        )

        held = copy.deepcopy(picked)
        held["frame_state"]["frame_no"] = 112
        processor.process_feature(held)
        held_reward = processor.consume_behavior_shaping(expected_frame=112)
        self.assertEqual(held_reward["speed_buff_team_reward"], 0.0)

    def test_official_kai_post_opening_selects_respawned_wolf(self):
        processor = FeatureProcessV1(1)
        initial = make_observation(frame=1800, camp=1)
        processor.process_feature(initial)
        processor.consume_behavior_shaping(expected_frame=1800)
        wolf_anchor = tuple(u.fixed_monster_anchor(44, "own", 1))
        processor.monster_camp_history[wolf_anchor]["cleared_once"] = True
        processor.monster_camp_history[wolf_anchor]["clear_count"] = 1

        respawned = make_observation(frame=1806, camp=1)
        respawned["frame_state"]["hero_states"][0]["buff_state"][
            "buff_skills"
        ].append({"config_id": 60007, "start_time": 100, "times": 1})
        respawned["frame_state"]["npc_states"].extend(
            [
                _monster(44, "own", 1, 7701),
                _monster(45, "own", 1, 7702),
            ]
        )
        processor.process_feature(respawned)
        macro = processor._memo_context["jungle_macro"]
        self.assertEqual(macro["phase"], "control_cycle")
        self.assertIn(14, macro["high_value_candidate_slots"])
        self.assertEqual(macro["goal_node"], 10)
        self.assertIsNone(macro["target_slot"])

        selected = copy.deepcopy(respawned)
        selected["frame_state"]["frame_no"] = 1812
        selected["frame_state"]["hero_states"][2][
            "attack_target"
        ] = 7701
        processor.process_feature(selected)
        macro = processor._memo_context["jungle_macro"]
        self.assertEqual(macro["goal_node"], 2)
        self.assertEqual(macro["target_slot"], 14)

    def test_official_kai_post_opening_selects_living_treasure_thief(self):
        processor = FeatureProcessV1(1)
        initial = make_observation(frame=1800, camp=1)
        processor.process_feature(initial)
        processor.consume_behavior_shaping(expected_frame=1800)

        thief_alive = make_observation(frame=1806, camp=1)
        thief_alive["frame_state"]["hero_states"][0]["buff_state"][
            "buff_skills"
        ].append({"config_id": 60007, "start_time": 100, "times": 1})
        thief_alive["frame_state"]["npc_states"].append(
            _monster(59, "center", 1, 7801)
        )
        processor.process_feature(thief_alive)
        macro = processor._memo_context["jungle_macro"]
        self.assertEqual(macro["phase"], "control_cycle")
        self.assertIn(18, macro["high_value_candidate_slots"])
        self.assertEqual(macro["goal_node"], 10)
        self.assertIsNone(macro["target_slot"])

        selected = copy.deepcopy(thief_alive)
        selected["frame_state"]["frame_no"] = 1812
        selected["frame_state"]["hero_states"][2][
            "attack_target"
        ] = 7801
        processor.process_feature(selected)
        macro = processor._memo_context["jungle_macro"]
        self.assertEqual(macro["goal_node"], 3)
        self.assertEqual(macro["target_slot"], 18)

    def test_nearby_teammate_death_penalizes_only_noncontributors(self):
        def teammate_death(observation):
            victim = observation["frame_state"]["hero_states"][2]
            killer = observation["frame_state"]["hero_states"][3]
            victim["hp"] = 0
            victim["dead_cnt"] += 1
            observation["frame_state"]["frame_action"]["dead_action"] = [
                {
                    "killer": killer,
                    "death": {
                        **copy.deepcopy(victim),
                        "single_hurt_list": [
                            {"runtime_id": killer["runtime_id"]}
                        ],
                    },
                }
            ]

        no_help = FeatureProcessV1(1)
        initial = make_observation(frame=100, camp=1)
        no_help.process_feature(initial)
        no_help.consume_behavior_shaping(expected_frame=100)
        death = make_observation(frame=106, camp=1)
        teammate_death(death)
        no_help.process_feature(death)
        penalty = no_help.consume_behavior_shaping(expected_frame=106)
        self.assertEqual(
            penalty["teammate_abandonment_counts"], (1, 1)
        )
        self.assertTrue(
            np.allclose(
                penalty["teammate_abandonment_rewards"],
                (-0.06, -0.06),
            )
        )

        one_helped = FeatureProcessV1(1)
        one_helped.process_feature(initial)
        one_helped.consume_behavior_shaping(expected_frame=100)
        engaged = make_observation(frame=103, camp=1)
        engaged["frame_state"]["hero_states"][0][
            "hurt_hero_info"
        ] = [{"hurt_target": 21, "hurt": 50}]
        one_helped.process_feature(engaged)
        one_helped.consume_behavior_shaping(expected_frame=103)
        helped_death = make_observation(frame=106, camp=1)
        teammate_death(helped_death)
        one_helped.process_feature(helped_death)
        helped_reward = one_helped.consume_behavior_shaping(
            expected_frame=106
        )
        self.assertEqual(
            helped_reward["teammate_abandonment_counts"], (0, 1)
        )
        self.assertTrue(
            np.allclose(
                helped_reward["teammate_abandonment_rewards"],
                (0.0, -0.06),
            )
        )
        self.assertEqual(
            helped_reward["teammate_support_counts"], (1, 0)
        )
        self.assertTrue(
            np.allclose(
                helped_reward["teammate_support_rewards"],
                (0.05, 0.0),
            )
        )

    def test_temporal_focus_rewards_second_real_participant_once(self):
        camp_one = FeatureProcessV1(1)
        camp_two = FeatureProcessV1(2)
        initial = make_observation(frame=100, camp=1)
        initial["frame_state"]["hero_states"][3]["location"] = {
            "x": -10000,
            "y": 0,
            "z": u.LANE_CENTER_Z,
        }
        camp_one.process_feature(initial)
        camp_one.consume_behavior_shaping(expected_frame=100)
        camp_two.process_feature(self._camp_two_perspective(initial))
        camp_two.consume_behavior_shaping(expected_frame=100)

        first = copy.deepcopy(initial)
        first["frame_state"]["frame_no"] = 103
        first["frame_state"]["hero_states"][0]["hurt_hero_info"] = [
            {"hurt_target": 21, "hurt": 50}
        ]
        camp_one.process_feature(first)
        first_one = camp_one.consume_behavior_shaping(
            expected_frame=103
        )
        camp_two.process_feature(self._camp_two_perspective(first))
        first_two = camp_two.consume_behavior_shaping(
            expected_frame=103
        )
        self.assertEqual(first_one["temporal_focus_team_reward"], 0.0)
        self.assertEqual(first_two["temporal_focus_team_reward"], 0.0)

        second = copy.deepcopy(initial)
        second["frame_state"]["frame_no"] = 106
        second["frame_state"]["hero_states"][1]["hurt_hero_info"] = [
            {"hurt_target": 21, "hurt": 50}
        ]
        camp_one.process_feature(second)
        second_one = camp_one.consume_behavior_shaping(
            expected_frame=106
        )
        camp_two.process_feature(self._camp_two_perspective(second))
        second_two = camp_two.consume_behavior_shaping(
            expected_frame=106
        )
        self.assertAlmostEqual(
            second_one["temporal_focus_team_reward"], 0.05
        )
        self.assertAlmostEqual(
            second_two["temporal_focus_team_reward"], -0.05
        )
        self.assertEqual(second_one["temporal_focus_count"], 1)

        third = copy.deepcopy(initial)
        third["frame_state"]["frame_no"] = 109
        third["frame_state"]["hero_states"][2]["hurt_hero_info"] = [
            {"hurt_target": 21, "hurt": 50}
        ]
        camp_one.process_feature(third)
        third_reward = camp_one.consume_behavior_shaping(
            expected_frame=109
        )
        self.assertEqual(
            third_reward["temporal_focus_team_reward"], 0.0
        )

    def test_three_person_kill_reward_is_zero_sum_and_requires_three(self):
        initial = make_observation(frame=100, camp=1)
        camp_one = FeatureProcessV1(1)
        camp_two = FeatureProcessV1(2)
        camp_one.process_feature(initial)
        camp_one.consume_behavior_shaping(expected_frame=100)
        camp_two.process_feature(self._camp_two_perspective(initial))
        camp_two.consume_behavior_shaping(expected_frame=100)

        engaged = make_observation(frame=103, camp=1)
        for hero in engaged["frame_state"]["hero_states"][:3]:
            hero["hurt_hero_info"] = [
                {"hurt_target": 21, "hurt": 50}
            ]
        camp_one.process_feature(engaged)
        camp_one.consume_behavior_shaping(expected_frame=103)
        camp_two.process_feature(self._camp_two_perspective(engaged))
        camp_two.consume_behavior_shaping(expected_frame=103)

        killed = make_observation(frame=106, camp=1)
        heroes = killed["frame_state"]["hero_states"]
        victim = heroes[3]
        victim["hp"] = 0
        victim["dead_cnt"] += 1
        heroes[0]["kill_cnt"] += 1
        heroes[1]["assist_cnt"] += 1
        heroes[2]["assist_cnt"] += 1
        killed["frame_state"]["frame_action"]["dead_action"] = [
            {
                "killer": heroes[0],
                "assist_set": [heroes[1], heroes[2]],
                "death": copy.deepcopy(victim),
            }
        ]
        camp_one.process_feature(killed)
        reward_one = camp_one.consume_behavior_shaping(
            expected_frame=106
        )
        camp_two.process_feature(self._camp_two_perspective(killed))
        reward_two = camp_two.consume_behavior_shaping(
            expected_frame=106
        )
        self.assertAlmostEqual(
            reward_one["group_attack_team_reward"], 0.08
        )
        self.assertAlmostEqual(
            reward_two["group_attack_team_reward"], -0.08
        )
        self.assertEqual(reward_one["group_attack_count"], 1)
        self.assertEqual(reward_two["group_attack_count"], 1)

        duplicate = copy.deepcopy(killed)
        duplicate["frame_state"]["frame_no"] = 112
        camp_one.process_feature(duplicate)
        duplicate_reward = camp_one.consume_behavior_shaping(
            expected_frame=112
        )
        self.assertEqual(
            duplicate_reward["group_attack_team_reward"], 0.0
        )
        self.assertEqual(
            duplicate_reward["teammate_abandonment_counts"], (0, 0)
        )

        two_person = FeatureProcessV1(1)
        two_person.process_feature(initial)
        two_person.consume_behavior_shaping(expected_frame=100)
        two_person_kill = copy.deepcopy(killed)
        two_person_kill["frame_state"]["frame_no"] = 112
        two_person_kill["frame_state"]["frame_action"]["dead_action"][0][
            "assist_set"
        ] = [two_person_kill["frame_state"]["hero_states"][1]]
        two_person.process_feature(two_person_kill)
        two_person_reward = two_person.consume_behavior_shaping(
            expected_frame=112
        )
        self.assertEqual(
            two_person_reward["group_attack_team_reward"], 0.0
        )

    def test_exposed_crystal_distance_pressure_is_zero_sum_potential(self):
        camp_one = FeatureProcessV1(1)
        camp_two = FeatureProcessV1(2)
        initial = make_observation(frame=100, camp=1)
        camp_one.process_feature(initial)
        initial_one = camp_one.consume_behavior_shaping(
            expected_frame=100
        )
        camp_two.process_feature(self._camp_two_perspective(initial))
        initial_two = camp_two.consume_behavior_shaping(
            expected_frame=100
        )
        self.assertEqual(initial_one["crystal_danger_reward"], 0.0)
        self.assertEqual(initial_two["crystal_danger_reward"], 0.0)

        exposed = make_observation(frame=106, camp=1)
        for organ in exposed["frame_state"]["npc_states"]:
            if organ["config_id"] in (66, 67):
                organ["hp"] = 0
        camp_one.process_feature(exposed)
        camp_one.consume_behavior_shaping(expected_frame=106)
        camp_two.process_feature(self._camp_two_perspective(exposed))
        camp_two.consume_behavior_shaping(expected_frame=106)

        approached = copy.deepcopy(exposed)
        approached["frame_state"]["frame_no"] = 112
        approached["frame_state"]["hero_states"][3]["location"] = {
            "x": u.ORGAN_ANCHORS[68][0] + 9000,
            "y": 0,
            "z": u.ORGAN_ANCHORS[68][1],
        }
        camp_one.process_feature(approached)
        approach_one = camp_one.consume_behavior_shaping(
            expected_frame=112
        )
        camp_two.process_feature(
            self._camp_two_perspective(approached)
        )
        approach_two = camp_two.consume_behavior_shaping(
            expected_frame=112
        )
        self.assertLess(approach_one["crystal_danger_reward"], 0.0)
        self.assertAlmostEqual(
            approach_one["crystal_danger_reward"],
            -approach_two["crystal_danger_reward"],
        )

        closer = copy.deepcopy(approached)
        closer["frame_state"]["frame_no"] = 118
        closer["frame_state"]["hero_states"][3]["location"]["x"] = (
            u.ORGAN_ANCHORS[68][0] + 3000
        )
        camp_one.process_feature(closer)
        closer_one = camp_one.consume_behavior_shaping(
            expected_frame=118
        )
        camp_two.process_feature(self._camp_two_perspective(closer))
        closer_two = camp_two.consume_behavior_shaping(
            expected_frame=118
        )
        self.assertLess(
            closer_one["crystal_danger_reward"],
            approach_one["crystal_danger_reward"],
        )
        self.assertAlmostEqual(
            closer_one["crystal_danger_reward"],
            -closer_two["crystal_danger_reward"],
        )

    def test_exposed_crystal_enemy_minion_also_creates_pressure(self):
        processor = FeatureProcessV1(1)
        initial = make_observation(frame=100, camp=1)
        processor.process_feature(initial)
        processor.consume_behavior_shaping(expected_frame=100)

        exposed = make_observation(frame=106, camp=1)
        for organ in exposed["frame_state"]["npc_states"]:
            if organ["config_id"] in (66, 67):
                organ["hp"] = 0
        processor.process_feature(exposed)
        processor.consume_behavior_shaping(expected_frame=106)

        pressured = copy.deepcopy(exposed)
        pressured["frame_state"]["frame_no"] = 112
        enemy_minion = next(
            actor
            for actor in pressured["frame_state"]["npc_states"]
            if actor.get("sub_type") == "ACTOR_SUB_SOLDIER"
            and actor["camp"] == 2
        )
        enemy_minion["location"] = {
            "x": u.ORGAN_ANCHORS[68][0] + 3000,
            "y": 0,
            "z": u.ORGAN_ANCHORS[68][1],
        }
        processor.process_feature(pressured)
        reward = processor.consume_behavior_shaping(expected_frame=112)
        self.assertLess(reward["crystal_danger_reward"], 0.0)

    def test_outer_tower_minion_pressure_is_zero_sum_and_windowed(self):
        camp_one = FeatureProcessV1(1)
        camp_two = FeatureProcessV1(2)
        initial = make_observation(frame=100, camp=1)
        camp_one.process_feature(initial)
        camp_one.consume_behavior_shaping(expected_frame=100)
        camp_two.process_feature(self._camp_two_perspective(initial))
        camp_two.consume_behavior_shaping(expected_frame=100)

        pressured = copy.deepcopy(initial)
        pressured["frame_state"]["frame_no"] = 115
        enemy_minion = next(
            actor
            for actor in pressured["frame_state"]["npc_states"]
            if actor.get("sub_type") == "ACTOR_SUB_SOLDIER"
            and actor["camp"] == 2
        )
        enemy_minion["location"] = {
            "x": u.ORGAN_ANCHORS[66][0] + 12000,
            "y": 0,
            "z": u.ORGAN_ANCHORS[66][1],
        }
        camp_one.process_feature(pressured)
        pressure_one = camp_one.consume_behavior_shaping(
            expected_frame=115
        )
        camp_two.process_feature(self._camp_two_perspective(pressured))
        pressure_two = camp_two.consume_behavior_shaping(
            expected_frame=115
        )
        self.assertAlmostEqual(
            pressure_one["defense_pressure_reward"], -0.015
        )
        self.assertAlmostEqual(
            pressure_one["defense_pressure_reward"],
            -pressure_two["defense_pressure_reward"],
        )

        under_tower = copy.deepcopy(pressured)
        under_tower["frame_state"]["frame_no"] = 130
        enemy_minion = next(
            actor
            for actor in under_tower["frame_state"]["npc_states"]
            if actor.get("sub_type") == "ACTOR_SUB_SOLDIER"
            and actor["camp"] == 2
        )
        enemy_minion["location"]["x"] = (
            u.ORGAN_ANCHORS[66][0] + 5000
        )
        camp_one.process_feature(under_tower)
        under_reward = camp_one.consume_behavior_shaping(
            expected_frame=130
        )
        self.assertAlmostEqual(
            under_reward["defense_pressure_reward"], -0.03
        )

        cleared = copy.deepcopy(under_tower)
        cleared["frame_state"]["frame_no"] = 145
        enemy_minion = next(
            actor
            for actor in cleared["frame_state"]["npc_states"]
            if actor.get("sub_type") == "ACTOR_SUB_SOLDIER"
            and actor["camp"] == 2
        )
        enemy_minion["location"]["x"] = 0
        camp_one.process_feature(cleared)
        cleared_reward = camp_one.consume_behavior_shaping(
            expected_frame=145
        )
        self.assertEqual(
            cleared_reward["defense_pressure_reward"], 0.0
        )

    def test_terminal_crystal_outcome_reaches_actor_and_win_aux_once(self):
        processor = FeatureProcessV1(1)
        initial = make_observation(frame=100, camp=1)
        processor.process_feature(initial)
        processor.consume_behavior_shaping(expected_frame=100)
        won = make_observation(frame=106, camp=1)
        won["frame_state"]["frame_action"]["dead_action"] = [
            {
                "death": {
                    "runtime_id": 172,
                    "config_id": 72,
                    "camp": 2,
                    "actor_type": "ACTOR_ORGAN",
                    "sub_type": "ACTOR_SUB_CRYSTAL",
                }
            }
        ]
        processor.process_feature(won)
        reward = processor.consume_behavior_shaping(expected_frame=106)
        self.assertEqual(reward["crystal_outcome_reward"], 8.0)
        self.assertEqual(reward["rewards"], (8.0, 8.0))
        self.assertEqual(reward["win_auxiliary_rewards"], (8.0, 8.0))

        repeated = copy.deepcopy(won)
        repeated["frame_state"]["frame_no"] = 112
        processor.process_feature(repeated)
        repeated_reward = processor.consume_behavior_shaping(expected_frame=112)
        self.assertEqual(repeated_reward["crystal_outcome_reward"], 0.0)

        losing_processor = FeatureProcessV1(1)
        losing_processor.process_feature(initial)
        losing_processor.consume_behavior_shaping(expected_frame=100)
        lost = copy.deepcopy(won)
        lost["frame_state"]["frame_action"]["dead_action"][0]["death"].update(
            {"runtime_id": 168, "config_id": 68, "camp": 1}
        )
        losing_processor.process_feature(lost)
        lost_reward = losing_processor.consume_behavior_shaping(expected_frame=106)
        self.assertEqual(lost_reward["crystal_outcome_reward"], -8.0)
        self.assertEqual(lost_reward["rewards"], (-8.0, -8.0))

    def test_nonlinear_tower_potential_is_zero_sum_and_back_loaded(self):
        initial = make_observation(frame=100, camp=1)
        camp_one = FeatureProcessV1(1)
        camp_two = FeatureProcessV1(2)
        camp_one.process_feature(initial)
        camp_one.consume_behavior_shaping(expected_frame=100)
        camp_two.process_feature(self._camp_two_perspective(initial))
        camp_two.consume_behavior_shaping(expected_frame=100)

        half = make_observation(frame=106, camp=1)
        enemy_outer = next(
            npc
            for npc in half["frame_state"]["npc_states"]
            if npc["config_id"] == 70
        )
        enemy_outer["hp"] = enemy_outer["max_hp"] / 2
        camp_one.process_feature(half)
        half_one = camp_one.consume_behavior_shaping(expected_frame=106)
        camp_two.process_feature(self._camp_two_perspective(half))
        half_two = camp_two.consume_behavior_shaping(expected_frame=106)

        self.assertAlmostEqual(half_one["structure_team_reward"], 0.9)
        self.assertAlmostEqual(
            half_one["structure_team_reward"],
            -half_two["structure_team_reward"],
        )
        for actor_reward in half_one["pushing_auxiliary_rewards"]:
            self.assertAlmostEqual(
                actor_reward,
                half_one["structure_team_reward"],
            )

        destroyed = copy.deepcopy(half)
        destroyed["frame_state"]["frame_no"] = 112
        next(
            npc
            for npc in destroyed["frame_state"]["npc_states"]
            if npc["config_id"] == 70
        )["hp"] = 0
        camp_one.process_feature(destroyed)
        last_half = camp_one.consume_behavior_shaping(expected_frame=112)
        self.assertAlmostEqual(last_half["structure_team_reward"], 1.1)
        self.assertGreater(
            last_half["structure_team_reward"],
            half_one["structure_team_reward"],
        )
        self.assertAlmostEqual(
            half_one["structure_team_reward"]
            + last_half["structure_team_reward"],
            V1TrainingConfig.STRUCTURE_TOWER_TOTAL_REWARD,
        )

    def test_kill_opens_bounded_push_window_until_structure_damage(self):
        processor = FeatureProcessV1(1)
        initial = make_observation(frame=100, camp=1)
        processor.process_feature(initial)
        processor.consume_behavior_shaping(expected_frame=100)

        killed = make_observation(frame=106, camp=1)
        killed["frame_state"]["frame_action"]["dead_action"] = [
            {
                "death": {
                    "runtime_id": 21,
                    "config_id": 132,
                    "camp": 2,
                    "actor_type": "ACTOR_HERO",
                    "sub_type": "ACTOR_SUB_NONE",
                }
            }
        ]
        processor.process_feature(killed)
        kill_reward = processor.consume_behavior_shaping(expected_frame=106)
        self.assertEqual(kill_reward["post_kill_push_reward"], 0.0)

        pushed = make_observation(frame=112, camp=1)
        enemy_outer = next(
            npc
            for npc in pushed["frame_state"]["npc_states"]
            if npc["config_id"] == 70
        )
        enemy_outer["hp"] -= 100
        processor.process_feature(pushed)
        push_reward = processor.consume_behavior_shaping(expected_frame=112)
        self.assertEqual(
            push_reward["post_kill_push_reward"],
            V1TrainingConfig.POST_KILL_PUSH_REWARD,
        )
        self.assertGreater(push_reward["structure_team_reward"], 0.0)

        continued = copy.deepcopy(pushed)
        continued["frame_state"]["frame_no"] = 118
        enemy_outer = next(
            npc
            for npc in continued["frame_state"]["npc_states"]
            if npc["config_id"] == 70
        )
        enemy_outer["hp"] -= 100
        processor.process_feature(continued)
        continued_reward = processor.consume_behavior_shaping(expected_frame=118)
        self.assertEqual(continued_reward["post_kill_push_reward"], 0.0)

    def test_inactivity_starts_after_23_seconds_then_escalates_each_second(self):
        processor = FeatureProcessV1(1)
        applied = []
        for frame in range(0, 601, 6):
            observation = make_observation(frame=frame, camp=1)
            processor.process_feature(observation)
            reward = processor.consume_behavior_shaping(expected_frame=frame)
            if reward["long_inactivity_counts"][0]:
                applied.append(
                    (
                        frame,
                        reward["long_inactivity_penalties"][0],
                        reward["long_inactivity_stationary_counts"][0],
                        reward["long_inactivity_no_damage_counts"][0],
                    )
                )
        self.assertGreaterEqual(len(applied), 2)
        self.assertGreaterEqual(applied[0][0], 425)
        self.assertAlmostEqual(applied[0][1], -0.03)
        self.assertEqual(applied[0][2:], (1, 1))
        expected_second = -(
            0.03 + 0.27 * (1.0 - np.exp(-1.0 / 4.0))
        )
        self.assertAlmostEqual(applied[1][1], expected_second, places=6)
        self.assertEqual(applied[1][2:], (1, 1))

    def test_long_no_damage_rotation_is_not_mislabeled_as_idle(self):
        processor = FeatureProcessV1(1)
        penalties = []
        frames = list(range(2000, 2201, 6))
        for index, frame in enumerate(frames):
            observation = make_observation(frame=frame, camp=1)
            side_x = -30000.0 + 25000.0 * index / (len(frames) - 1)
            observation["frame_state"]["hero_states"][0]["location"] = (
                _raw_position(side_x, 0.0, 1)
            )
            processor.process_feature(observation)
            reward = processor.consume_behavior_shaping(expected_frame=frame)
            penalties.append(reward["long_inactivity_penalties"][0])
        self.assertTrue(all(value == 0.0 for value in penalties))

    def test_five_second_loop_is_detected_as_wandering(self):
        processor = FeatureProcessV1(1)
        wandering_hits = 0
        for index, frame in enumerate(range(2000, 2101, 6)):
            observation = make_observation(frame=frame, camp=1)
            angle = 2.0 * np.pi * (index % 13) / 13.0
            observation["frame_state"]["hero_states"][0]["location"] = (
                _raw_position(
                    -18000.0 + 2000.0 * np.cos(angle),
                    2000.0 * np.sin(angle),
                    1,
                )
            )
            processor.process_feature(observation)
            reward = processor.consume_behavior_shaping(expected_frame=frame)
            wandering_hits += reward["long_inactivity_wandering_counts"][0]
        self.assertGreater(wandering_hits, 0)

    def test_tyrant_five_second_setup_exempts_long_inactivity(self):
        processor = FeatureProcessV1(1)
        tyrant_side = u.side_position(
            u.fixed_monster_anchor(122, "center", 1), 1
        )
        exemptions = 0
        penalties = []
        for frame in range(2000, 2361, 6):
            observation = make_observation(frame=frame, camp=1)
            for hero in observation["frame_state"]["hero_states"][:2]:
                hero["location"] = _raw_position(
                    tyrant_side[0], tyrant_side[1], 1
                )
            for spawn in observation["frame_state"]["monster_spawn_infos"]:
                if spawn["monster_id"] == 122:
                    spawn["spawn_timer"] = 5000
            processor.process_feature(observation)
            reward = processor.consume_behavior_shaping(expected_frame=frame)
            exemptions += reward["long_inactivity_tyrant_exemptions"][0]
            penalties.append(reward["long_inactivity_penalties"][0])
        self.assertGreater(exemptions, 0)
        self.assertTrue(all(value == 0.0 for value in penalties))

    def test_tyrant_wait_exemption_radius_is_seven_thousand(self):
        tyrant_side = u.side_position(
            u.fixed_monster_anchor(122, "center", 1), 1
        )

        def exemption_at(distance):
            processor = FeatureProcessV1(1)
            observation = make_observation(frame=2000, camp=1)
            observation["frame_state"]["hero_states"][0]["location"] = (
                _raw_position(
                    tyrant_side[0] + distance,
                    tyrant_side[1],
                    1,
                )
            )
            for spawn in observation["frame_state"]["monster_spawn_infos"]:
                if spawn["monster_id"] == 122:
                    spawn["spawn_timer"] = 5000
            processor.process_feature(observation)
            return processor.consume_behavior_shaping(
                expected_frame=2000
            )["long_inactivity_tyrant_exemptions"][0]

        self.assertEqual(exemption_at(6999.0), 1)
        self.assertEqual(exemption_at(7001.0), 0)

    def test_enemy_hero_common_attack_legality_uses_normal_attack_button(self):
        observation = make_observation()
        for hero_info in observation["heroes"]:
            legal = np.zeros(FrozenV1Config.RAW_LEGAL_ACTION_DIM, dtype=np.float32)
            legal[:13] = 1
            legal[13:38] = 1
            legal[38:80] = 1
            legal[80:122] = 1
            target_matrix = legal[122:].reshape(13, 39)
            target_matrix[:, 0] = 1
            target_matrix[3, 1] = 1
            hero_info["legal_action"] = legal

        targets = _block(
            FeatureProcessV1(1).process_feature(observation)[0], "targets"
        )
        common_attack = schema_v1.TARGET_FIELD_MAP[
            "common_attack_target_legal"
        ].start
        chosen_skill = schema_v1.TARGET_FIELD_MAP[
            "chosen_skill_target_legal"
        ].start
        self.assertEqual(targets[1, common_attack], 1)
        self.assertEqual(targets[1, chosen_skill], 0)

    def test_all_fourteen_measured_physical_monster_camps_are_represented(self):
        self.assertEqual(len(u.PHYSICAL_MONSTER_CAMP_ANCHORS), 14)
        self.assertEqual(len(set(u.PHYSICAL_MONSTER_CAMP_ANCHORS)), 14)
        represented = {
            u.fixed_monster_anchor(config_id, relation, 1)
            for config_id, relation in schema_v1.MONSTER_TARGET_CONFIG_AND_SIDE
        }
        self.assertEqual(represented, set(u.PHYSICAL_MONSTER_CAMP_ANCHORS))
        self.assertEqual(
            u.fixed_monster_anchor(33, "own", 1), (-29860.0, -24480.0)
        )
        self.assertEqual(
            u.fixed_monster_anchor(33, "enemy", 1), (30070.0, -24610.0)
        )
        self.assertEqual(u.monster_spawn_config_id(33), 32)
        self.assertEqual(u.monster_spawn_config_id(45), 44)

    def test_controlled_kai_receives_opening_upper_jungle_plan(self):
        observation = make_observation(frame=100, camp=1)
        observation["official_model_info"] = {"official": {"runtime_id": 11}}
        for spawn in observation["frame_state"]["monster_spawn_infos"]:
            spawn["spawn_timer"] = 0
        features = FeatureProcessV1(1).process_feature(observation)
        kai_plan = _block(features[1], "jungle_plan")[0]
        fields = schema_v1.JUNGLE_PLAN_FIELD_MAP
        self.assertEqual(kai_plan[fields["query_is_kai"].start], 1)
        self.assertEqual(kai_plan[fields["phase_upper_wolf"].start], 1)
        goal = kai_plan[fields["goal_node_0"].start : fields["goal_node_11"].stop]
        self.assertEqual(int(np.argmax(goal)), 2)
        self.assertEqual(goal.sum(), 1)

    def test_physical_wolf_clear_advances_to_deer_without_lane_soak(self):
        processor = FeatureProcessV1(1)
        alive = make_observation(frame=100, camp=1)
        alive["official_model_info"] = {"official": {"runtime_id": 11}}
        alive["frame_state"]["npc_states"].extend(
            [_monster(44, "own", 1, 501), _monster(45, "own", 1, 502)]
        )
        for spawn in alive["frame_state"]["monster_spawn_infos"]:
            spawn["spawn_timer"] = 0
        processor.process_feature(alive)

        cleared = make_observation(frame=106, camp=1)
        cleared["official_model_info"] = {"official": {"runtime_id": 11}}
        cleared_kai = next(
            hero
            for hero in cleared["frame_state"]["hero_states"]
            if hero["config_id"] == 193 and hero["camp"] == 1
        )
        wolf_anchor = u.side_position(
            u.fixed_monster_anchor(44, "own", 1), 1
        )
        cleared_kai["location"] = _raw_position(*wolf_anchor, 1)
        for spawn in cleared["frame_state"]["monster_spawn_infos"]:
            if spawn["monster_id"] in (44, 45):
                spawn["spawn_timer"] = 60000
            else:
                spawn["spawn_timer"] = 0
        processor.process_feature(cleared)
        fields = schema_v1.JUNGLE_PLAN_FIELD_MAP

        buffed = make_observation(frame=112, camp=1)
        buffed["official_model_info"] = {"official": {"runtime_id": 11}}
        for spawn in buffed["frame_state"]["monster_spawn_infos"]:
            spawn["spawn_timer"] = (
                60000 if spawn["monster_id"] in (44, 45) else 0
            )
        kai = next(hero for hero in buffed["frame_state"]["hero_states"] if hero["config_id"] == 193 and hero["camp"] == 1)
        kai["level"] = 2
        kai["buff_state"]["buff_skills"] = [
            {"config_id": 90204},
            {"config_id": 902070},
        ]
        plan = _block(processor.process_feature(buffed)[1], "jungle_plan")[0]
        self.assertEqual(plan[fields["upper_wolf_cleared_once"].start], 1)
        self.assertEqual(plan[fields["phase_upper_deer"].start], 1)
        self.assertEqual(plan[fields["phase_lane_soak_wave"].start], 0)

        finished = make_observation(frame=118, camp=1, include_soldiers=False)
        finished["official_model_info"] = {"official": {"runtime_id": 11}}
        for spawn in finished["frame_state"]["monster_spawn_infos"]:
            spawn["spawn_timer"] = (
                60000 if spawn["monster_id"] in (44, 45) else 0
            )
        finished_kai = next(
            hero for hero in finished["frame_state"]["hero_states"]
            if hero["config_id"] == 193 and hero["camp"] == 1
        )
        finished_kai["level"] = 2
        finished_kai["buff_state"]["buff_skills"] = [
            {"config_id": 90204},
            {"config_id": 902070},
        ]
        plan = _block(processor.process_feature(finished)[1], "jungle_plan")[0]
        self.assertEqual(plan[fields["phase_lane_soak_wave"].start], 0)

    def test_upper_clear_without_enemy_wave_skips_lane_soak(self):
        processor = FeatureProcessV1(1)
        wolf_alive = make_observation(
            frame=100, camp=1, include_soldiers=False
        )
        wolf_alive["official_model_info"] = {
            "official": {"runtime_id": 11}
        }
        wolf_alive["frame_state"]["npc_states"].extend(
            [_monster(44, "own", 1, 501), _monster(45, "own", 1, 502)]
        )
        for spawn in wolf_alive["frame_state"]["monster_spawn_infos"]:
            spawn["spawn_timer"] = 0
        processor.process_feature(wolf_alive)

        deer_alive = make_observation(
            frame=106, camp=1, include_soldiers=False
        )
        deer_alive["official_model_info"] = {
            "official": {"runtime_id": 11}
        }
        deer_alive["frame_state"]["npc_states"].extend(
            [_monster(42, "own", 1, 503), _monster(43, "own", 1, 504)]
        )
        deer_alive_kai = next(
            hero
            for hero in deer_alive["frame_state"]["hero_states"]
            if hero["config_id"] == 193 and hero["camp"] == 1
        )
        wolf_anchor = u.side_position(
            u.fixed_monster_anchor(44, "own", 1), 1
        )
        deer_alive_kai["location"] = _raw_position(*wolf_anchor, 1)
        for spawn in deer_alive["frame_state"]["monster_spawn_infos"]:
            spawn["spawn_timer"] = (
                60000 if spawn["monster_id"] in (44, 45) else 0
            )
        processor.process_feature(deer_alive)

        cleared = make_observation(
            frame=112, camp=1, include_soldiers=False
        )
        cleared["official_model_info"] = {"official": {"runtime_id": 11}}
        kai = next(
            hero for hero in cleared["frame_state"]["hero_states"]
            if hero["config_id"] == 193 and hero["camp"] == 1
        )
        deer_anchor = u.side_position(
            u.fixed_monster_anchor(42, "own", 1), 1
        )
        kai["location"] = _raw_position(*deer_anchor, 1)
        kai["buff_state"]["buff_skills"] = [
            {"config_id": 90204},
            {"config_id": 902070},
        ]
        for spawn in cleared["frame_state"]["monster_spawn_infos"]:
            if spawn["monster_id"] in (42, 43, 44, 45):
                spawn["spawn_timer"] = 60000
        plan = _block(processor.process_feature(cleared)[1], "jungle_plan")[0]
        fields = schema_v1.JUNGLE_PLAN_FIELD_MAP
        self.assertEqual(plan[fields["phase_lane_soak_wave"].start], 0)

    def test_removed_lane_soak_does_not_retry_after_death(self):
        processor = FeatureProcessV1(1)
        alive = make_observation(frame=100, camp=1)
        alive["official_model_info"] = {"official": {"runtime_id": 11}}
        alive["frame_state"]["npc_states"].extend(
            [_monster(44, "own", 1, 501), _monster(45, "own", 1, 502)]
        )
        for spawn in alive["frame_state"]["monster_spawn_infos"]:
            spawn["spawn_timer"] = 0
        processor.process_feature(alive)

        soaking = make_observation(frame=106, camp=1)
        soaking["official_model_info"] = {"official": {"runtime_id": 11}}
        for spawn in soaking["frame_state"]["monster_spawn_infos"]:
            spawn["spawn_timer"] = (
                60000 if spawn["monster_id"] in (44, 45) else 0
            )
        kai = next(
            hero for hero in soaking["frame_state"]["hero_states"]
            if hero["config_id"] == 193 and hero["camp"] == 1
        )
        wolf_anchor = u.side_position(
            u.fixed_monster_anchor(44, "own", 1), 1
        )
        kai["location"] = _raw_position(*wolf_anchor, 1)
        kai["buff_state"]["buff_skills"] = [
            {"config_id": 90204}, {"config_id": 902070}
        ]
        fields = schema_v1.JUNGLE_PLAN_FIELD_MAP
        plan = _block(processor.process_feature(soaking)[1], "jungle_plan")[0]
        self.assertEqual(plan[fields["phase_lane_soak_wave"].start], 0)
        self.assertEqual(plan[fields["phase_upper_deer"].start], 1)

        dead = make_observation(frame=112, camp=1)
        dead["official_model_info"] = {"official": {"runtime_id": 11}}
        dead_kai = next(
            hero for hero in dead["frame_state"]["hero_states"]
            if hero["config_id"] == 193 and hero["camp"] == 1
        )
        dead_kai["hp"] = 0
        dead_kai["buff_state"]["buff_skills"] = [
            {"config_id": 90204}, {"config_id": 902070}
        ]
        processor.process_feature(dead)

        revived = make_observation(frame=118, camp=1)
        revived["official_model_info"] = {"official": {"runtime_id": 11}}
        revived_kai = next(
            hero for hero in revived["frame_state"]["hero_states"]
            if hero["config_id"] == 193 and hero["camp"] == 1
        )
        revived_kai["level"] = 2
        revived_kai["buff_state"]["buff_skills"] = [
            {"config_id": 90204}, {"config_id": 902070}
        ]
        plan = _block(processor.process_feature(revived)[1], "jungle_plan")[0]
        self.assertEqual(plan[fields["phase_lane_soak_wave"].start], 0)

    def test_lane_to_tyrant_path_uses_lower_entry_not_central_wall(self):
        processor = FeatureProcessV1(1)
        observation = make_observation()
        processor.process_feature(observation)
        positions = processor._route_node_positions(processor._memo_context)
        _path, next_node, _distance = processor._shortest_route(positions, (0.0, 0.0), 9)
        self.assertEqual(next_node, 4)

    def test_enemy_jungle_invasion_releases_protected_opening(self):
        observation = make_observation(frame=100, camp=1)
        enemy = observation["frame_state"]["hero_states"][3]
        kai = next(
            hero
            for hero in observation["frame_state"]["hero_states"]
            if hero["camp"] == 1 and hero["config_id"] == 193
        )
        kai["location"] = _raw_position(-14000, 15000, 1)
        enemy["location"] = _raw_position(-18000, 15000, 1)
        features = FeatureProcessV1(1).process_feature(observation)
        plan = _block(features[0], "jungle_plan")[0]
        fields = schema_v1.JUNGLE_PLAN_FIELD_MAP
        self.assertEqual(plan[fields["own_jungle_invaded"].start], 1)
        interrupts = plan[
            fields["interrupt_none"].start : fields["interrupt_structure_defense"].stop
        ]
        self.assertEqual(interrupts.sum(), 1)
        self.assertEqual(plan[fields["interrupt_none"].start], 0)

        lane_hero = next(
            hero
            for hero in observation["frame_state"]["hero_states"]
            if hero["camp"] == 1 and hero["config_id"] == 132
        )
        lane_hero["hp"] = 0
        interrupted = _block(
            FeatureProcessV1(1).process_feature(observation)[0],
            "jungle_plan",
        )[0]
        self.assertEqual(interrupted[fields["interrupt_none"].start], 0)

    def test_nearby_enemy_opening_interrupt_uses_ten_thousand_boundary(self):
        fields = schema_v1.JUNGLE_PLAN_FIELD_MAP

        inside = make_observation(frame=100, camp=1)
        for enemy in inside["frame_state"]["hero_states"][3:]:
            enemy["location"] = _raw_position(50000, 0, 1)
        inside_kai = next(
            hero
            for hero in inside["frame_state"]["hero_states"]
            if hero["camp"] == 1 and hero["config_id"] == 193
        )
        inside_kai["location"] = _raw_position(-12000, 15000, 1)
        inside["frame_state"]["hero_states"][3]["location"] = _raw_position(
            -2000, 15000, 1
        )
        inside_plan = _block(
            FeatureProcessV1(1).process_feature(inside)[0], "jungle_plan"
        )[0]
        self.assertEqual(inside_plan[fields["interrupt_none"].start], 0)

        outside = make_observation(frame=100, camp=1)
        for enemy in outside["frame_state"]["hero_states"][3:]:
            enemy["location"] = _raw_position(50000, 0, 1)
        outside_kai = next(
            hero
            for hero in outside["frame_state"]["hero_states"]
            if hero["camp"] == 1 and hero["config_id"] == 193
        )
        outside_kai["location"] = _raw_position(-12000, 15000, 1)
        outside["frame_state"]["hero_states"][3]["location"] = _raw_position(
            -1999, 15000, 1
        )
        outside_plan = _block(
            FeatureProcessV1(1).process_feature(outside)[0], "jungle_plan"
        )[0]
        self.assertEqual(outside_plan[fields["interrupt_none"].start], 1)

    def test_outer_proximity_does_not_release_opening_after_lane_hero_death(self):
        observation = make_observation(frame=100, camp=1)
        enemy = observation["frame_state"]["hero_states"][3]
        enemy["location"] = _raw_position(-17876, 0, 1)
        fields = schema_v1.JUNGLE_PLAN_FIELD_MAP
        alive_plan = _block(
            FeatureProcessV1(1).process_feature(observation)[0],
            "jungle_plan",
        )[0]
        self.assertEqual(alive_plan[fields["interrupt_structure_defense"].start], 0)

        lane_hero = next(
            hero
            for hero in observation["frame_state"]["hero_states"]
            if hero["camp"] == 1 and hero["config_id"] == 132
        )
        lane_hero["hp"] = 0
        dead_plan = _block(
            FeatureProcessV1(1).process_feature(observation)[0],
            "jungle_plan",
        )[0]
        self.assertEqual(dead_plan[fields["interrupt_structure_defense"].start], 0)

    def test_outer_targeting_does_not_release_opening_but_base_targeting_does(self):
        observation = make_observation(frame=100, camp=1)
        enemy = observation["frame_state"]["hero_states"][3]
        enemy["attack_target"] = 166
        plan = _block(
            FeatureProcessV1(1).process_feature(observation)[0], "jungle_plan"
        )[0]
        fields = schema_v1.JUNGLE_PLAN_FIELD_MAP
        self.assertEqual(plan[fields["interrupt_structure_defense"].start], 0)

        lane_hero = next(
            hero
            for hero in observation["frame_state"]["hero_states"]
            if hero["camp"] == 1 and hero["config_id"] == 132
        )
        lane_hero["hp"] = 0
        plan = _block(
            FeatureProcessV1(1).process_feature(observation)[0], "jungle_plan"
        )[0]
        self.assertEqual(plan[fields["interrupt_structure_defense"].start], 0)

        base = make_observation(frame=100, camp=1)
        own_outer = next(
            npc for npc in base["frame_state"]["npc_states"]
            if npc.get("runtime_id") == 166
        )
        own_outer["hp"] = 0
        base["frame_state"]["hero_states"][3]["attack_target"] = 167
        base_plan = _block(
            FeatureProcessV1(1).process_feature(base)[0], "jungle_plan"
        )[0]
        self.assertEqual(
            base_plan[fields["interrupt_structure_defense"].start], 1
        )

    def test_no_time_based_group_fallback(self):
        processor = FeatureProcessV1(1)
        alive = make_observation(frame=100, camp=1)
        alive["frame_state"]["npc_states"].extend(
            [_monster(42, "own", 1, 501), _monster(43, "own", 1, 502)]
        )
        processor.process_feature(alive)

        late = make_observation(frame=3600, camp=1)
        kai = next(
            hero for hero in late["frame_state"]["hero_states"]
            if hero["camp"] == 1 and hero["config_id"] == 193
        )
        kai["buff_state"]["buff_skills"] = [
            {"config_id": 90204}, {"config_id": 902070}
        ]
        for hero in late["frame_state"]["hero_states"][3:]:
            hero["camp_visible"] = [0, 0]
        for soldier in late["frame_state"]["npc_states"]:
            if soldier.get("sub_type") == "ACTOR_SUB_SOLDIER" and soldier["camp"] == 1:
                soldier["location"] = _raw_position(0, 0, 1)
        processor.process_feature(late)
        self.assertNotEqual(processor._memo_context["jungle_macro"]["mode"], "group")

    def test_push_window_preempts_optional_jungle_and_tyrant_cycle(self):
        processor = FeatureProcessV1(1)
        alive = make_observation(frame=894, camp=1)
        alive["frame_state"]["npc_states"].extend(
            [_monster(42, "own", 1, 501), _monster(43, "own", 1, 502)]
        )
        processor.process_feature(alive)
        processor.jungle_route_history["opening_released"] = True
        processor.jungle_route_history["lower_farm_started"] = True

        pushing = make_observation(frame=900, camp=1)
        kai = next(
            hero for hero in pushing["frame_state"]["hero_states"]
            if hero["config_id"] == 193 and hero["camp"] == 1
        )
        kai["buff_state"]["buff_skills"] = [
            {"config_id": 90204}, {"config_id": 902070}
        ]
        for hero in pushing["frame_state"]["hero_states"]:
            if hero["camp"] == 1:
                hero["location"] = _raw_position(10000, 0, 1)
            else:
                hero["location"] = _raw_position(50000, 0, 1)
        friend_soldier = next(
            npc for npc in pushing["frame_state"]["npc_states"]
            if npc.get("sub_type") == "ACTOR_SUB_SOLDIER" and npc["camp"] == 1
        )
        friend_soldier["location"] = _raw_position(17000, 0, 1)
        plan = _block(processor.process_feature(pushing)[1], "jungle_plan")[0]
        fields = schema_v1.JUNGLE_PLAN_FIELD_MAP
        self.assertEqual(plan[fields["mode_push"].start], 1)
        self.assertEqual(plan[fields["phase_control_cycle"].start], 1)
        self.assertEqual(plan[fields["phase_lane_soak_wave"].start], 0)

    def test_group_opportunity_cannot_preempt_protected_opening_rotation(self):
        processor = FeatureProcessV1(1)
        opening = make_observation(frame=100, camp=1)
        # The synthetic default has a locally favorable visible fight. It
        # must not replace the wolf opening before the protected route ends.
        processor.process_feature(opening)
        macro = processor._memo_context["jungle_macro"]
        self.assertTrue(macro["opening_rotation_active"])
        self.assertEqual(macro["phase"], "upper_wolf")
        self.assertEqual(macro["mode"], "farm")

    def test_controlled_kai_gets_one_shot_route_progress_shaping(self):
        processor = FeatureProcessV1(1)
        first = make_observation(frame=100, camp=1)
        first["official_model_info"] = {"official": {"runtime_id": 11}}
        first["frame_state"]["npc_states"].extend(
            (_monster(44, "own", 1, 544), _monster(45, "own", 1, 545))
        )
        for spawn in first["frame_state"]["monster_spawn_infos"]:
            if spawn["monster_id"] in (44, 45):
                spawn["spawn_timer"] = 0
        processor.process_feature(first)
        self.assertEqual(processor.consume_jungle_route_shaping(100), (0.0, 0.0))

        second = make_observation(frame=106, camp=1)
        second["official_model_info"] = {"official": {"runtime_id": 11}}
        second["frame_state"]["npc_states"].extend(
            (_monster(44, "own", 1, 544), _monster(45, "own", 1, 545))
        )
        for spawn in second["frame_state"]["monster_spawn_infos"]:
            if spawn["monster_id"] in (44, 45):
                spawn["spawn_timer"] = 0
        kai = next(
            hero
            for hero in second["frame_state"]["hero_states"]
            if hero["config_id"] == 193 and hero["camp"] == 1
        )
        kai["location"] = _raw_position(-22000, 0, 1)
        processor.process_feature(second)
        rewards = processor.consume_jungle_route_shaping(106)
        self.assertEqual(rewards[0], 0.0)
        self.assertGreater(rewards[1], 0.0)
        self.assertEqual(processor.consume_jungle_route_shaping(106), (0.0, 0.0))

    def test_controlled_kai_enemy_side_incursion_is_diagnostic_only(self):
        observation = make_observation(frame=100, camp=1)
        observation["official_model_info"] = {"official": {"runtime_id": 11}}
        kai = next(
            hero
            for hero in observation["frame_state"]["hero_states"]
            if hero["config_id"] == 193 and hero["camp"] == 1
        )
        kai["location"] = _raw_position(12000, 15000, 1)
        processor = FeatureProcessV1(1)
        processor.process_feature(observation)
        rewards = processor.consume_jungle_route_shaping(100)
        self.assertEqual(rewards[1], 0.0)
        self.assertTrue(
            processor._memo_context["jungle_macro"][
                "enemy_side_incursion"
            ]
        )

    def test_two_actor_shape_dtype_finite_and_masks(self):
        features = FeatureProcessV1(1).process_feature(make_observation())
        self.assertEqual(len(features), 2)
        for feature in features:
            self.assertEqual(feature.shape, (FrozenV1Config.FEATURE_DIM,))
            self.assertEqual(feature.dtype, np.float32)
            self.assertTrue(np.isfinite(feature).all())
            np.testing.assert_array_equal(
                _block(feature, "heroes")[:, schema_v1.HERO_FIELD_MAP["slot_exists"].start],
                np.ones(6),
            )
            np.testing.assert_array_equal(
                _block(feature, "organs")[:, schema_v1.ORGAN_FIELD_MAP["slot_exists"].start],
                np.ones(6),
            )
            np.testing.assert_array_equal(
                _block(feature, "monsters")[:, schema_v1.MONSTER_FIELD_MAP["slot_exists"].start],
                np.ones(20),
            )
            self.assertEqual(
                int(_block(feature, "soldiers")[:, schema_v1.SOLDIER_FIELD_MAP["slot_exists"].start].sum()),
                4,
            )

    def test_public_blocks_identical_between_actor_queries(self):
        first, second = FeatureProcessV1(1).process_feature(make_observation())
        for name in (
            "soldiers",
            "monsters",
            "dynamic_lane_position",
            "tyrant_team_context",
            "collaboration_state",
            "global_state",
        ):
            np.testing.assert_array_equal(_block(first, name), _block(second, name), err_msg=name)
        self.assertFalse(np.array_equal(_block(first, "main_context"), _block(second, "main_context")))

    def test_target_order_and_target38(self):
        feature = FeatureProcessV1(1).process_feature(make_observation())[0]
        targets = _block(feature, "targets")
        type_enemy = schema_v1.TARGET_FIELD_MAP["type_enemy_hero"].start
        type_monster = schema_v1.TARGET_FIELD_MAP["type_monster"].start
        type_outer = schema_v1.TARGET_FIELD_MAP["type_outer_tower"].start
        self.assertTrue(np.all(targets[1:4, type_enemy] == 1))
        self.assertTrue(np.all(targets[8:28, type_monster] == 1))
        self.assertEqual(targets[38, type_outer], 1)

    def test_last_seen_never_reads_hidden_current_position(self):
        processor = FeatureProcessV1(1)
        visible = processor.process_feature(make_observation(frame=100))[0]
        old_side_x = _block(visible, "heroes")[3, schema_v1.HERO_FIELD_MAP["side_x_norm"].start]
        hidden_observation = make_observation(frame=106)
        enemy = hidden_observation["frame_state"]["hero_states"][3]
        enemy["camp_visible"] = [0, 1]
        enemy["location"] = {"x": 60000, "y": 0, "z": 30000}
        hidden = processor.process_feature(hidden_observation)[0]
        hero = _block(hidden, "heroes")[3]
        self.assertEqual(hero[schema_v1.HERO_FIELD_MAP["visible_current"].start], 0)
        self.assertEqual(hero[schema_v1.HERO_FIELD_MAP["last_seen_used"].start], 1)
        self.assertAlmostEqual(hero[schema_v1.HERO_FIELD_MAP["side_x_norm"].start], old_side_x)

        rollback = make_observation(frame=20)
        rollback_enemy = rollback["frame_state"]["hero_states"][3]
        rollback_enemy["camp_visible"] = [0, 1]
        rollback_enemy["location"] = {"x": 60000, "y": 0, "z": 30000}
        reset_feature = processor.process_feature(rollback)[0]
        reset_hero = _block(reset_feature, "heroes")[3]
        self.assertEqual(reset_hero[schema_v1.HERO_FIELD_MAP["last_seen_used"].start], 0)
        self.assertEqual(reset_hero[schema_v1.HERO_FIELD_MAP["location_valid"].start], 0)

    def test_camp_mirror_keeps_side_coordinates(self):
        camp1 = FeatureProcessV1(1).process_feature(make_observation(camp=1))[0]
        camp2 = FeatureProcessV1(2).process_feature(make_observation(camp=2))[0]
        hero_fields = schema_v1.HERO_FIELD_MAP
        for field_name in ("side_x_norm", "side_z_norm"):
            index = hero_fields[field_name].start
            self.assertAlmostEqual(_block(camp1, "heroes")[0, index], _block(camp2, "heroes")[0, index])

    def test_same_frame_is_memoized_without_history_drift(self):
        processor = FeatureProcessV1(1)
        observation = make_observation(frame=100)
        first = processor.process_feature(observation)
        second = processor.process_feature(copy.deepcopy(observation))
        np.testing.assert_array_equal(first[0], second[0])
        np.testing.assert_array_equal(first[1], second[1])

    def test_dynamic_lane_boundary_switch_preserves_61_buckets(self):
        processor = FeatureProcessV1(1)
        first = processor.process_feature(make_observation(frame=100))[0]
        dynamic = _block(first, "dynamic_lane_position")
        fields = schema_v1.DYNAMIC_LANE_POSITION_FIELD_MAP
        self.assertEqual(dynamic[0, fields["dynamic_lane_valid"].start], 1)
        self.assertEqual(dynamic[0, fields["own_active_boundary_outer"].start], 1)
        self.assertEqual(
            dynamic[0, fields["dynamic_lane_bucket_0"].start : fields["dynamic_lane_bucket_60"].stop].sum(),
            1,
        )

        switched_observation = make_observation(frame=106)
        own_outer = next(
            actor
            for actor in switched_observation["frame_state"]["npc_states"]
            if actor["config_id"] == 66
        )
        own_outer["hp"] = 0
        switched = processor.process_feature(switched_observation)[0]
        switched_dynamic = _block(switched, "dynamic_lane_position")
        self.assertEqual(switched_dynamic[0, fields["own_active_boundary_high"].start], 1)
        self.assertEqual(
            switched_dynamic[0, fields["dynamic_lane_bucket_0"].start : fields["dynamic_lane_bucket_60"].stop].sum(),
            1,
        )

    def test_tyrant_window_and_alive_actor(self):
        outside = make_observation(frame=100)
        outside["frame_state"]["monster_spawn_infos"][-1]["spawn_timer"] = 31000
        feature = FeatureProcessV1(1).process_feature(outside)[0]
        local = _block(feature, "tyrant_local_position")
        self.assertEqual(local.sum(), 0)

        alive = make_observation(frame=100)
        anchor = u.MONSTER_ANCHORS[(122, "center")]
        alive["frame_state"]["npc_states"].append(
            {
                "runtime_id": 8122,
                "config_id": 122,
                "camp": 0,
                "actor_type": "ACTOR_MONSTER",
                "sub_type": "ACTOR_SUB_BAOJUN",
                "location": {"x": anchor[0], "y": 0, "z": anchor[1]},
                "camp_visible": [1, 1],
                "hp": 18000,
                "max_hp": 18000,
                "attack_target": 0,
            }
        )
        alive["frame_state"]["monster_spawn_infos"][-1]["spawn_timer"] = 0
        feature = FeatureProcessV1(1).process_feature(alive)[0]
        local = _block(feature, "tyrant_local_position")[0]
        fields = schema_v1.TYRANT_LOCAL_POSITION_FIELD_MAP
        self.assertEqual(local[fields["tyrant_fine_valid"].start], 1)
        self.assertEqual(local[fields["tyrant_alive"].start], 1)
        self.assertEqual(
            local[fields["local_x_bucket_0"].start : fields["local_x_bucket_20"].stop].sum(),
            1,
        )

    def test_buff_mark_and_unknown_hash(self):
        observation = make_observation(frame=100)
        hero = observation["frame_state"]["hero_states"][0]
        unknown = 777777
        hero["buff_state"] = {
            "buff_skills": [
                {"config_id": 90000, "start_time": 1000},
                {"config_id": 900009, "start_time": 1000},
                {"config_id": 60006, "start_time": 1000},
                {"config_id": unknown, "start_time": 1000},
            ],
            "buff_marks": [
                {"config_id": 13200, "origin_actorId": 11, "layer": 4},
                {"config_id": 14190, "origin_actor_id": 21, "layer": 2},
            ],
        }
        feature = FeatureProcessV1(1).process_feature(observation)[0]
        hero_vector = _block(feature, "heroes")[0]
        fields = schema_v1.HERO_FIELD_MAP
        self.assertEqual(hero_vector[fields["has_recover_combo"].start], 1)
        self.assertEqual(hero_vector[fields["tyrant_stage_1"].start], 1)
        self.assertEqual(hero_vector[fields["tyrant_stage_2"].start], 1)
        self.assertEqual(hero_vector[fields["tyrant_stage_3"].start], 0)
        bucket = schema_v1.stable_unknown_buff_bucket(unknown)
        self.assertEqual(hero_vector[fields[f"buff_unknown_hash_{bucket}_present"].start], 1)
        self.assertEqual(hero_vector[fields["marco_layer_norm"].start], 4 / 9)
        self.assertEqual(hero_vector[fields["diaochan_layer_norm"].start], 2 / 3)
        self.assertEqual(hero_vector[fields["mark_origin_is_self"].start], 1)
        self.assertEqual(hero_vector[fields["mark_origin_is_enemy"].start], 1)

    def test_bullet_velocity_and_relative_motion_window(self):
        processor = FeatureProcessV1(1)
        first = make_observation(frame=100)
        first["frame_state"]["bullets"] = [
            {
                "runtime_id": 900,
                "camp": 2,
                "source_actor": 21,
                "slot_type": "SLOT_SKILL_1",
                "skill_id": 13210,
                "location": _raw_position(10000, 0, 1),
            }
        ]
        first_feature = processor.process_feature(first)[0]
        bullet_fields = schema_v1.BULLET_FIELD_MAP
        first_bullet = _block(first_feature, "bullets")[0]
        self.assertEqual(first_bullet[bullet_fields["slot_exists"].start], 1)
        self.assertEqual(first_bullet[bullet_fields["velocity_valid"].start], 0)

        second = make_observation(frame=106)
        second["frame_state"]["bullets"] = [
            {
                "runtime_id": 900,
                "camp": 2,
                "source_actor": 21,
                "slot_type": "SLOT_SKILL_1",
                "skill_id": 13210,
                "location": _raw_position(0, 0, 1),
            }
        ]
        second_feature = processor.process_feature(second)[0]
        second_bullet = _block(second_feature, "bullets")[0]
        self.assertEqual(second_bullet[bullet_fields["velocity_valid"].start], 1)
        self.assertEqual(second_bullet[bullet_fields["relative_motion_valid"].start], 1)
        self.assertGreater(second_bullet[bullet_fields["radial_closing_speed_norm"].start], 0)
        self.assertEqual(second_bullet[bullet_fields["collision_course"].start], 1)

    def test_resource_timer_and_fixed_structure_lifecycle(self):
        processor = FeatureProcessV1(1)
        processor.process_feature(make_observation(frame=100))
        picked = make_observation(frame=106)
        picked["frame_state"]["cakes"] = [
            cake for cake in picked["frame_state"]["cakes"] if cake["config_id"] != 102
        ]
        enemy_outer = next(
            actor for actor in picked["frame_state"]["npc_states"] if actor["config_id"] == 70
        )
        enemy_outer["hp"] = 0
        feature = processor.process_feature(picked)[0]
        resources = _block(feature, "resource_anchors")
        resource_fields = schema_v1.RESOURCE_ANCHOR_FIELD_MAP
        self.assertEqual(resources[0, resource_fields["available"].start], 0)
        self.assertEqual(resources[0, resource_fields["respawn_timer_valid"].start], 1)
        self.assertGreater(resources[0, resource_fields["respawn_timer_norm"].start], 0)
        organs = _block(feature, "organs")
        organ_fields = schema_v1.ORGAN_FIELD_MAP
        self.assertEqual(organs[3, organ_fields["slot_exists"].start], 1)
        self.assertEqual(organs[3, organ_fields["alive"].start], 0)
        targets = _block(feature, "targets")
        self.assertEqual(targets[38, schema_v1.TARGET_FIELD_MAP["type_high_tower"].start], 1)

    def test_active_collaboration_protocol_mapping(self):
        observation = make_observation(frame=100)
        observation["frame_state"]["collaboration_state"]["micro_last"][0]["frame_no"] = 90
        observation["frame_state"]["collaboration_state"]["micro_state"] = [
            {
                "sender": 11,
                "receiver": 13,
                "target_runtime_id": 170,
                "target_position": {"x": 999999, "y": 999999, "z": 999999},
                "end_frame": 700,
            }
        ]
        feature = FeatureProcessV1(1).process_feature(observation)[0]
        state = _block(feature, "collaboration_state")[0]
        fields = schema_v1.COLLABORATION_STATE_FIELD_MAP
        self.assertEqual(state[fields["current_type_set_target"].start], 1)
        self.assertEqual(state[fields["current_command_active"].start], 1)
        self.assertEqual(state[fields["target_runtime_maps_to_slot"].start], 1)
        self.assertEqual(state[fields["receiver_is_official_hero"].start], 1)
        self.assertEqual(state[fields["cooldown_ready"].start], 0)

    def test_local_command_emission_throttles_rejected_protocol_command(self):
        processor = FeatureProcessV1(1)
        processor.process_feature(make_observation(frame=100))
        processor.record_collaboration_command(100, [1, 501, 0, 0])

        processor.process_feature(make_observation(frame=106))
        self.assertFalse(processor.command_runtime_context().cooldown_ready)

        processor.process_feature(make_observation(frame=200))
        self.assertTrue(processor.command_runtime_context().cooldown_ready)

    def test_only_confirmed_trainable_command_opens_result_attribution(self):
        processor = FeatureProcessV1(1)
        processor.process_feature(make_observation(frame=100))
        processor.record_collaboration_command(
            100, [1, 21, 0, 0], intent=2, is_train=True
        )

        accepted = make_observation(frame=106)
        accepted["frame_state"]["collaboration_state"]["micro_last"][0][
            "frame_no"
        ] = 106
        accepted["frame_state"]["collaboration_state"]["micro_state"] = [
            {
                "sender": 11,
                "receiver": 13,
                "target_runtime_id": 21,
                "target_position": {
                    "x": 999999,
                    "y": 999999,
                    "z": 999999,
                },
                "end_frame": 700,
            }
        ]
        official = next(
            hero
            for hero in accepted["frame_state"]["hero_states"]
            if hero["runtime_id"] == 13
        )
        target = next(
            hero
            for hero in accepted["frame_state"]["hero_states"]
            if hero["runtime_id"] == 21
        )
        accepted["frame_state"]["frame_action"]["dead_action"] = [
            {"death": copy.deepcopy(target), "killer": copy.deepcopy(official)}
        ]
        processor.process_feature(accepted)
        behavior = processor.consume_behavior_shaping(expected_frame=106)
        self.assertTrue(behavior["command_attribution_active"])
        self.assertEqual(behavior["command_attribution_intent"], 2)
        self.assertAlmostEqual(
            behavior["command_official_kda_reward"],
            V1TrainingConfig.KILL_REWARD,
        )

    def test_confirmed_forced_command_does_not_open_attribution(self):
        processor = FeatureProcessV1(1)
        processor.process_feature(make_observation(frame=100))
        processor.record_collaboration_command(
            100, [1, 21, 0, 0], intent=2, is_train=False
        )
        accepted = make_observation(frame=106)
        accepted["frame_state"]["collaboration_state"]["micro_last"][0][
            "frame_no"
        ] = 106
        accepted["frame_state"]["collaboration_state"]["micro_state"] = [
            {
                "sender": 11,
                "receiver": 13,
                "target_runtime_id": 21,
                "target_position": {
                    "x": 999999,
                    "y": 999999,
                    "z": 999999,
                },
                "end_frame": 700,
            }
        ]
        processor.process_feature(accepted)
        behavior = processor.consume_behavior_shaping(expected_frame=106)
        self.assertFalse(behavior["command_attribution_active"])
        self.assertEqual(behavior["command_official_kda_reward"], 0.0)

    def test_confirmed_position_command_gets_bounded_progress_difference(self):
        processor = FeatureProcessV1(1)
        processor.process_feature(make_observation(frame=100))
        raw_goal = (0, int(u.LANE_CENTER_Z))
        processor.record_collaboration_command(
            100,
            [2, 0, raw_goal[0], raw_goal[1]],
            intent=6,
            is_train=True,
        )

        def command_observation(frame, official_side_x):
            observation = make_observation(frame=frame)
            official = next(
                hero
                for hero in observation["frame_state"]["hero_states"]
                if hero["runtime_id"] == 13
            )
            official["location"] = _raw_position(
                official_side_x, -1000, 1
            )
            collaboration = observation["frame_state"]["collaboration_state"]
            collaboration["micro_last"][0]["frame_no"] = 106
            collaboration["micro_state"] = [
                {
                    "sender": 11,
                    "receiver": 13,
                    "target_runtime_id": 0,
                    "target_position": {
                        "x": raw_goal[0],
                        "y": 0,
                        "z": raw_goal[1],
                    },
                    "end_frame": 700,
                }
            ]
            return observation

        processor.process_feature(command_observation(106, -12000))
        first = processor.consume_behavior_shaping(expected_frame=106)
        self.assertEqual(first["command_goal_progress_reward"], 0.0)

        processor.process_feature(command_observation(112, -6000))
        second = processor.consume_behavior_shaping(expected_frame=112)
        self.assertGreater(second["command_goal_progress_reward"], 0.0)
        self.assertLessEqual(
            second["command_goal_progress_reward"],
            V1TrainingConfig.COMMAND_GOAL_PROGRESS_BUDGET,
        )

    def test_confirmed_target_command_rewards_lock_range_and_first_hit(self):
        processor = FeatureProcessV1(1)
        processor.process_feature(make_observation(frame=100))
        processor.record_collaboration_command(
            100, [1, 21, 0, 0], intent=2, is_train=True
        )
        accepted = make_observation(frame=106)
        collaboration = accepted["frame_state"]["collaboration_state"]
        collaboration["micro_last"][0]["frame_no"] = 106
        collaboration["micro_state"] = [
            {
                "sender": 11,
                "receiver": 13,
                "target_runtime_id": 21,
                "target_position": {
                    "x": 999999, "y": 999999, "z": 999999
                },
                "end_frame": 700,
            }
        ]
        official = next(
            hero for hero in accepted["frame_state"]["hero_states"]
            if hero["runtime_id"] == 13
        )
        target = next(
            hero for hero in accepted["frame_state"]["hero_states"]
            if hero["runtime_id"] == 21
        )
        target["location"] = dict(official["location"])
        official["attack_target"] = 21
        official["hurt_hero_info"] = [{"hurt_target": 21, "hurt": 100}]
        processor.process_feature(accepted)
        behavior = processor.consume_behavior_shaping(expected_frame=106)
        expected = (
            V1TrainingConfig.COMMAND_TARGET_LOCK_REWARD
            + V1TrainingConfig.COMMAND_TARGET_RANGE_REWARD
            + V1TrainingConfig.COMMAND_TARGET_FIRST_HIT_REWARD
        )
        self.assertAlmostEqual(
            behavior["command_goal_progress_reward"], expected
        )
        self.assertEqual(behavior["command_nonresponse_penalty"], 0.0)

    def test_confirmed_target_command_gets_one_bounded_nonresponse_penalty(self):
        processor = FeatureProcessV1(1)
        processor.process_feature(make_observation(frame=100))
        processor.record_collaboration_command(
            100, [1, 21, 0, 0], intent=2, is_train=True
        )

        def active_target_observation(frame):
            observation = make_observation(frame=frame)
            collaboration = observation["frame_state"]["collaboration_state"]
            collaboration["micro_last"][0]["frame_no"] = 106
            collaboration["micro_state"] = [
                {
                    "sender": 11,
                    "receiver": 13,
                    "target_runtime_id": 21,
                    "target_position": {
                        "x": 999999, "y": 999999, "z": 999999
                    },
                    "end_frame": 700,
                }
            ]
            official = next(
                hero
                for hero in observation["frame_state"]["hero_states"]
                if hero["runtime_id"] == 13
            )
            # Locking without approach or damage must still time out; this is
            # the observed "circles a target forever" failure mode.
            official["attack_target"] = 21
            return observation

        processor.process_feature(active_target_observation(106))
        first = processor.consume_behavior_shaping(expected_frame=106)
        self.assertEqual(first["command_nonresponse_penalty"], 0.0)

        processor.process_feature(active_target_observation(182))
        failed = processor.consume_behavior_shaping(expected_frame=182)
        self.assertAlmostEqual(
            failed["command_nonresponse_penalty"],
            -V1TrainingConfig.COMMAND_NO_RESPONSE_PENALTY,
        )
        self.assertTrue(failed["command_nonresponse_applied"])
        self.assertAlmostEqual(
            failed["command_goal_progress_reward"],
            -V1TrainingConfig.COMMAND_NO_RESPONSE_PENALTY,
        )

        processor.process_feature(active_target_observation(188))
        repeated = processor.consume_behavior_shaping(expected_frame=188)
        self.assertEqual(repeated["command_nonresponse_penalty"], 0.0)
        self.assertFalse(repeated["command_nonresponse_applied"])

    def test_saturated_response_budget_does_not_report_zero_penalty_as_applied(self):
        processor = FeatureProcessV1(1)
        processor.process_feature(make_observation(frame=100))
        processor.record_collaboration_command(
            100, [1, 21, 0, 0], intent=2, is_train=True
        )

        def active_target_observation(frame):
            observation = make_observation(frame=frame)
            collaboration = observation["frame_state"]["collaboration_state"]
            collaboration["micro_last"][0]["frame_no"] = 106
            collaboration["micro_state"] = [
                {
                    "sender": 11,
                    "receiver": 13,
                    "target_runtime_id": 21,
                    "target_position": {
                        "x": 999999, "y": 999999, "z": 999999
                    },
                    "end_frame": 700,
                }
            ]
            official = next(
                hero
                for hero in observation["frame_state"]["hero_states"]
                if hero["runtime_id"] == 13
            )
            official["attack_target"] = 21
            return observation

        processor.process_feature(active_target_observation(106))
        processor.consume_behavior_shaping(expected_frame=106)
        processor.active_command_attribution["progress_total"] = (
            -V1TrainingConfig.COMMAND_GOAL_PROGRESS_BUDGET
        )

        processor.process_feature(active_target_observation(182))
        saturated = processor.consume_behavior_shaping(expected_frame=182)
        self.assertEqual(saturated["command_nonresponse_penalty"], 0.0)
        self.assertFalse(saturated["command_nonresponse_applied"])

    def test_repeated_trainable_no_command_penalizes_idle_official_hero(self):
        processor = FeatureProcessV1(1)
        processor.jungle_route_history["opening_released"] = True
        processor.process_feature(make_observation(frame=100))
        processor.record_collaboration_command(
            100, [0, 0, 0, 0], intent=0, is_train=True
        )
        observed_penalties = []
        for frame in range(106, 185, 6):
            processor.process_feature(make_observation(frame=frame))
            behavior = processor.consume_behavior_shaping(
                expected_frame=frame
            )
            observed_penalties.append(
                behavior["command_no_command_idle_penalty"]
            )
            processor.record_collaboration_command(
                frame, [0, 0, 0, 0], intent=0, is_train=True
            )
        self.assertEqual(
            sum(value < 0.0 for value in observed_penalties), 1
        )
        self.assertAlmostEqual(
            min(observed_penalties),
            -V1TrainingConfig.COMMAND_NO_COMMAND_IDLE_PENALTY,
        )

    def test_forced_no_command_never_penalizes_official_idleness(self):
        processor = FeatureProcessV1(1)
        processor.jungle_route_history["opening_released"] = True
        processor.process_feature(make_observation(frame=100))
        processor.record_collaboration_command(
            100, [0, 0, 0, 0], intent=0, is_train=False
        )
        for frame in range(106, 203, 6):
            processor.process_feature(make_observation(frame=frame))
            behavior = processor.consume_behavior_shaping(
                expected_frame=frame
            )
            self.assertEqual(
                behavior["command_no_command_idle_penalty"], 0.0
            )
            processor.record_collaboration_command(
                frame, [0, 0, 0, 0], intent=0, is_train=False
            )

    def test_lane_plan_assigns_distinct_rows_and_measured_opening_route(self):
        observation = make_observation(frame=100, include_soldiers=False)
        for hero in observation["frame_state"]["hero_states"][:2]:
            hero["location"] = _raw_position(-60000, 12000, 1)
        processor = FeatureProcessV1(1)
        features = processor.process_feature(observation)
        fields = schema_v1.LANE_PLAN_FIELD_MAP
        marco = _block(features[0], "lane_plan")[0]
        diaochan = _block(features[1], "lane_plan")[0]
        self.assertEqual(marco[fields["assigned_upper"].start], 1)
        self.assertEqual(diaochan[fields["assigned_lower"].start], 1)
        self.assertEqual(marco[fields["phase_base_exit"].start], 1)
        self.assertEqual(diaochan[fields["phase_base_exit"].start], 1)
        self.assertEqual(marco[fields["hard_mask_active"].start], 1)
        self.assertGreater(marco[fields["next_waypoint_dx_norm"].start], 0)

    def test_marco_overshoot_still_targets_fixed_positive_row_entry(self):
        observation = make_observation(frame=100, include_soldiers=False)
        marco = next(
            hero
            for hero in observation["frame_state"]["hero_states"]
            if hero["camp"] == 1 and hero["config_id"] == 132
        )
        marco["location"] = _raw_position(-17000.0, 12000.0, 1)
        processor = FeatureProcessV1(1)
        processor.process_feature(observation)
        plan = processor._memo_context["lane_macro"][132]
        self.assertEqual(plan["phase"], "stage_outer")
        self.assertEqual(plan["next_waypoint"], (-18000.0, 3150.0))
        self.assertFalse(processor.lane_opening_history[132]["row_entry_reached"])

    def test_lane_waypoint_arrival_advances_before_stationary_deadlock(self):
        for config_id, side_x, side_z, expected_phase in (
            (132, -50050, 8000, "corridor"),
            (141, -50050, 8000, "corridor"),
            (132, -30050, 3150, "stage_outer"),
            (141, -30050, -3150, "stage_outer"),
        ):
            observation = make_observation(frame=100, include_soldiers=False)
            for enemy in observation["frame_state"]["hero_states"][3:]:
                enemy["camp_visible"] = [0, 0]
            hero = next(
                item
                for item in observation["frame_state"]["hero_states"][:2]
                if item["config_id"] == config_id
            )
            hero["location"] = _raw_position(side_x, side_z, 1)
            processor = FeatureProcessV1(1)
            processor.process_feature(observation)
            self.assertEqual(
                processor._memo_context["lane_macro"][config_id]["phase"],
                expected_phase,
            )

    def test_first_wave_contact_releases_controlled_micro_to_ppo(self):
        observation = make_observation(frame=100, include_soldiers=False)
        observation["frame_state"]["npc_states"].extend(
            [
                _soldier(301, 1, -2000, 19, side_z=3150),
                _soldier(401, 2, -2000, 19, side_z=3150),
                _soldier(302, 1, -2000, 20, side_z=-3150),
                _soldier(402, 2, -2000, 20, side_z=-3150),
            ]
        )
        for hero, side_z in zip(
            observation["frame_state"]["hero_states"][:2], (3150, -3150)
        ):
            hero["location"] = _raw_position(-4000, side_z, 1)
        processor = FeatureProcessV1(1)
        first = processor.process_feature(observation)
        fields = schema_v1.LANE_PLAN_FIELD_MAP
        first_plan = _block(first[0], "lane_plan")[0]
        self.assertEqual(first_plan[fields["phase_released"].start], 1)
        self.assertEqual(first_plan[fields["hard_mask_active"].start], 0)
        self.assertEqual(first_plan[fields["first_wave_captured"].start], 1)
        later = copy.deepcopy(observation)
        later["frame_state"]["frame_no"] = 136
        later["frame_state"]["npc_states"] = [
            npc
            for npc in later["frame_state"]["npc_states"]
            if npc.get("sub_type") != "ACTOR_SUB_SOLDIER"
        ]
        later_plan = _block(processor.process_feature(later)[0], "lane_plan")[0]
        still_active = copy.deepcopy(later)
        still_active["frame_state"]["frame_no"] = 166
        later_plan = _block(
            processor.process_feature(still_active)[0], "lane_plan"
        )[0]
        self.assertEqual(later_plan[fields["phase_released"].start], 1)
        self.assertEqual(later_plan[fields["hard_mask_active"].start], 0)
        self.assertEqual(later_plan[fields["released_by_first_wave_clear"].start], 0)
        self.assertTrue(processor.lane_opening_history[132]["first_wave_cleared"])

    def test_controlled_upper_opening_uses_staged_wolf_route(self):
        processor = FeatureProcessV1(1)
        observation = make_observation(frame=100, camp=1)
        observation["official_model_info"] = {"official": {"runtime_id": 11}}
        kai = next(
            hero
            for hero in observation["frame_state"]["hero_states"]
            if hero["camp"] == 1 and hero["config_id"] == 193
        )
        kai["location"] = _raw_position(-53000, 2600, 1)
        processor.process_feature(observation)
        macro = processor._memo_context["jungle_macro"]
        self.assertEqual(macro["phase"], "upper_wolf")
        self.assertEqual(macro["next_node"], 2)
        self.assertEqual(macro["next_waypoint"], (-50000.0, 8000.0))

    def test_official_kai_uses_debug_upper_inner_waypoints_before_wolf(self):
        processor = FeatureProcessV1(1)
        waiting = make_observation(frame=100, camp=1)
        kai = next(
            hero
            for hero in waiting["frame_state"]["hero_states"]
            if hero["camp"] == 1 and hero["config_id"] == 193
        )
        kai["location"] = _raw_position(-53000, 2600, 1)
        processor.process_feature(waiting)
        macro = processor._memo_context["jungle_macro"]
        route = (
            (-50000.0, 8000.0),
            (-30000.0, 3500.0),
            (-25000.0, 0.0),
            (-25000.0, 15000.0 - u.LANE_CENTER_Z),
            u.side_position(u.fixed_monster_anchor(44, "own", 1), 1),
        )
        self.assertTrue(macro["kai_is_official"])
        self.assertEqual(macro["phase"], "upper_wolf")
        self.assertEqual(macro["next_node"], 2)
        # The official-only monotonic base-exit gate accepts x >= -54000
        # instead of pulling a hero already past that frontier backward.
        self.assertEqual(macro["next_waypoint"], route[1])

        current = waiting
        for index, waypoint in enumerate(route):
            current = copy.deepcopy(current)
            current["frame_state"]["frame_no"] = 106 + index * 6
            next(
                hero
                for hero in current["frame_state"]["hero_states"]
                if hero["camp"] == 1 and hero["config_id"] == 193
            )["location"] = _raw_position(waypoint[0], waypoint[1], 1)
            processor.process_feature(current)
            if index + 1 < len(route):
                self.assertEqual(
                    processor._memo_context["jungle_macro"]["next_waypoint"],
                    route[index + 1],
                )

        held = processor._memo_context["jungle_macro"]
        self.assertTrue(
            processor.jungle_route_history["official_upper_entry_reached"]
        )
        self.assertTrue(
            processor.jungle_route_history["official_upper_inner_reached"]
        )
        self.assertEqual(held["next_node"], 2)
        self.assertEqual(held["next_waypoint"], route[-1])
        self.assertEqual(
            held["official_upper_route_stage_name"], "upper_route_complete"
        )

    def test_official_kai_waypoint_cursor_does_not_regress_on_displacement(self):
        processor = FeatureProcessV1(1)
        entered = make_observation(frame=100, camp=1)
        route = (
            (-50000.0, 8000.0),
            (-30000.0, 3500.0),
            (-25000.0, 0.0),
        )
        for index, waypoint in enumerate(route):
            entered["frame_state"]["frame_no"] = 100 + index * 6
            next(
                hero
                for hero in entered["frame_state"]["hero_states"]
                if hero["camp"] == 1 and hero["config_id"] == 193
            )["location"] = _raw_position(waypoint[0], waypoint[1], 1)
            processor.process_feature(entered)
        self.assertTrue(
            processor.jungle_route_history["official_upper_entry_reached"]
        )

        displaced = copy.deepcopy(entered)
        displaced["frame_state"]["frame_no"] += 6
        next(
            hero
            for hero in displaced["frame_state"]["hero_states"]
            if hero["camp"] == 1 and hero["config_id"] == 193
        )["location"] = _raw_position(-33000, -8500, 1)
        processor.process_feature(displaced)
        macro = processor._memo_context["jungle_macro"]
        self.assertTrue(
            processor.jungle_route_history["official_upper_entry_reached"]
        )
        self.assertEqual(macro["next_node"], 2)
        self.assertEqual(
            macro["next_waypoint"],
            (-25000.0, 15000.0 - u.LANE_CENTER_Z),
        )

    def test_official_kai_command_echo_cannot_regress_physical_cursor(self):
        processor = FeatureProcessV1(1)
        observation = make_observation(frame=100, camp=1)
        route = (
            (-50000.0, 8000.0),
            (-30000.0, 3500.0),
            (-25000.0, 0.0),
            (-25000.0, 15000.0 - u.LANE_CENTER_Z),
        )
        for index, waypoint in enumerate(route):
            observation["frame_state"]["frame_no"] = 100 + index * 6
            next(
                hero
                for hero in observation["frame_state"]["hero_states"]
                if hero["camp"] == 1 and hero["config_id"] == 193
            )["location"] = _raw_position(waypoint[0], waypoint[1], 1)
            processor.process_feature(observation)
        self.assertTrue(
            processor.jungle_route_history["official_upper_inner_reached"]
        )
        processor.jungle_route_history["official_upper_route_interrupted"] = True

        returned = copy.deepcopy(observation)
        returned["frame_state"]["frame_no"] += 6
        next(
            hero
            for hero in returned["frame_state"]["hero_states"]
            if hero["camp"] == 1 and hero["config_id"] == 193
        )["location"] = _raw_position(-17875.0, 0.0, 1)
        processor.process_feature(returned)
        macro = processor._memo_context["jungle_macro"]
        self.assertTrue(macro["official_upper_entry_reached"])
        self.assertTrue(macro["official_upper_inner_reached"])
        self.assertEqual(
            macro["next_waypoint"],
            u.side_position(u.fixed_monster_anchor(44, "own", 1), 1),
        )

    def test_official_kai_confirmed_upper_region_releases_stale_entry(self):
        processor = FeatureProcessV1(1)
        observation = make_observation(frame=100, camp=1)
        route_prefix = ((-50000.0, 8000.0), (-30000.0, 3500.0))
        for index, waypoint in enumerate(route_prefix):
            observation["frame_state"]["frame_no"] = 100 + index * 6
            next(
                hero
                for hero in observation["frame_state"]["hero_states"]
                if hero["camp"] == 1 and hero["config_id"] == 193
            )["location"] = _raw_position(waypoint[0], waypoint[1], 1)
            processor.process_feature(observation)

        overshot = copy.deepcopy(observation)
        overshot["frame_state"]["frame_no"] += 6
        next(
            hero
            for hero in overshot["frame_state"]["hero_states"]
            if hero["camp"] == 1 and hero["config_id"] == 193
        )["location"] = _raw_position(
            -25000.0, 15000.0 - u.LANE_CENTER_Z, 1
        )
        processor.process_feature(overshot)
        macro = processor._memo_context["jungle_macro"]
        self.assertTrue(macro["official_upper_entry_reached"])
        self.assertTrue(macro["official_upper_inner_reached"])
        self.assertEqual(macro["official_upper_route_stage"], 4)
        self.assertEqual(
            macro["next_waypoint"],
            u.side_position(u.fixed_monster_anchor(44, "own", 1), 1),
        )

    def test_controlled_kai_reacquires_fixed_wolf_route_from_lane(self):
        processor = FeatureProcessV1(1)
        deer_anchor = tuple(
            float(value) for value in u.fixed_monster_anchor(42, "own", 1)
        )
        processor.monster_camp_history[deer_anchor] = {
            "ever_alive": True,
            "cleared_once": True,
            "live_cycle_seen": False,
            "clear_count": 1,
            "last_clear_frame": 90,
        }
        observation = make_observation(frame=100, camp=1)
        observation["official_model_info"] = {"official": {"runtime_id": 11}}
        kai = next(
            hero
            for hero in observation["frame_state"]["hero_states"]
            if hero["camp"] == 1 and hero["config_id"] == 193
        )
        kai["location"] = _raw_position(-18000.0, 0.0, 1)
        processor.process_feature(observation)
        macro = processor._memo_context["jungle_macro"]
        self.assertFalse(macro["kai_is_official"])
        self.assertEqual(macro["phase"], "upper_wolf")
        self.assertEqual(macro["next_node"], 2)
        self.assertEqual(macro["next_waypoint"], (-50000.0, 8000.0))

    def test_official_kai_wolf_resume_keeps_physical_waypoint_cursor(self):
        processor = FeatureProcessV1(1)
        deer_anchor = tuple(
            float(value) for value in u.fixed_monster_anchor(42, "own", 1)
        )
        processor.monster_camp_history[deer_anchor] = {
            "ever_alive": True,
            "cleared_once": True,
            "live_cycle_seen": False,
            "clear_count": 1,
            "last_clear_frame": 90,
        }
        processor.jungle_route_history.update(
            opening_route_key="wolf",
            opening_waypoint_index=3,
            official_upper_route_stage=3,
            official_upper_entry_reached=True,
            official_upper_inner_reached=False,
            official_upper_route_interrupted=True,
        )
        observation = make_observation(frame=100, camp=1)
        kai = next(
            hero
            for hero in observation["frame_state"]["hero_states"]
            if hero["camp"] == 1 and hero["config_id"] == 193
        )
        kai["location"] = _raw_position(-17875.0, 0.0, 1)
        processor.process_feature(observation)
        reacquire = processor._memo_context["jungle_macro"]
        self.assertEqual(reacquire["phase"], "upper_wolf")
        self.assertEqual(reacquire["official_upper_route_stage"], 3)
        self.assertEqual(
            reacquire["next_waypoint"],
            (-25000.0, 15000.0 - u.LANE_CENTER_Z),
        )

        inner = copy.deepcopy(observation)
        inner["frame_state"]["frame_no"] = 106
        kai = next(
            hero
            for hero in inner["frame_state"]["hero_states"]
            if hero["camp"] == 1 and hero["config_id"] == 193
        )
        kai["location"] = _raw_position(
            -25000.0, 15000.0 - u.LANE_CENTER_Z, 1
        )
        processor.process_feature(inner)
        toward_wolf = processor._memo_context["jungle_macro"]
        self.assertEqual(toward_wolf["official_upper_route_stage"], 4)
        self.assertEqual(
            toward_wolf["next_waypoint"],
            u.side_position(u.fixed_monster_anchor(44, "own", 1), 1),
        )

        at_wolf = copy.deepcopy(inner)
        at_wolf["frame_state"]["frame_no"] = 112
        kai = next(
            hero
            for hero in at_wolf["frame_state"]["hero_states"]
            if hero["camp"] == 1 and hero["config_id"] == 193
        )
        wolf_side = u.side_position(
            u.fixed_monster_anchor(44, "own", 1), 1
        )
        kai["location"] = _raw_position(wolf_side[0], wolf_side[1], 1)
        processor.process_feature(at_wolf)
        wolf = processor._memo_context["jungle_macro"]
        self.assertEqual(wolf["official_upper_route_stage"], 4)
        self.assertEqual(wolf["next_node"], 2)
        self.assertEqual(
            wolf["next_waypoint"],
            u.side_position(u.fixed_monster_anchor(44, "own", 1), 1),
        )

    def test_lane_target_lock_prevents_low_hp_target_oscillation(self):
        observation = make_observation(frame=100, include_soldiers=False)
        observation["frame_state"]["npc_states"].extend(
            [
                _soldier(301, 1, -3000, 19, side_z=3150),
                _soldier(401, 2, -1000, 19, side_z=3150),
                _soldier(402, 2, 1000, 20, side_z=3150),
            ]
        )
        observation["frame_state"]["npc_states"][-2]["hp"] = 100
        observation["frame_state"]["npc_states"][-1]["hp"] = 900
        for enemy in observation["frame_state"]["hero_states"][3:]:
            enemy["camp_visible"] = [0, 0]
        processor = FeatureProcessV1(1)
        processor.process_feature(observation)
        first_target = processor._memo_context["lane_macro"][132]["target_soldier"]
        self.assertEqual(first_target["runtime_id"], 401)

        changed = copy.deepcopy(observation)
        changed["frame_state"]["frame_no"] = 106
        enemy_soldiers = [
            soldier
            for soldier in changed["frame_state"]["npc_states"]
            if soldier.get("sub_type") == "ACTOR_SUB_SOLDIER"
            and soldier["camp"] == 2
        ]
        next(item for item in enemy_soldiers if item["runtime_id"] == 401)["hp"] = 900
        next(item for item in enemy_soldiers if item["runtime_id"] == 402)["hp"] = 50
        processor.process_feature(changed)
        locked_target = processor._memo_context["lane_macro"][132]["target_soldier"]
        self.assertEqual(locked_target["runtime_id"], 401)

    def test_lane_waypoint_cannot_jump_back_more_than_three_thousand(self):
        processor = FeatureProcessV1(1)
        first = make_observation(frame=100, include_soldiers=False)
        for enemy in first["frame_state"]["hero_states"][3:]:
            enemy["camp_visible"] = [0, 0]
        first["frame_state"]["hero_states"][0]["location"] = _raw_position(
            -18000, 3150, 1
        )
        processor.process_feature(first)

        advanced = copy.deepcopy(first)
        advanced["frame_state"]["frame_no"] = 106
        advanced["frame_state"]["hero_states"][0]["location"] = _raw_position(
            -5000, 3150, 1
        )
        processor.process_feature(advanced)
        plan = processor._memo_context["lane_macro"][132]
        self.assertEqual(plan["next_waypoint"][0], -8000.0)
        self.assertEqual(plan["waypoint_dx"], -3000.0)

    def test_lane_rule_has_no_fixed_frame_release(self):
        observation = make_observation(frame=3000, include_soldiers=False)
        for enemy in observation["frame_state"]["hero_states"][3:]:
            enemy["camp_visible"] = [0, 0]
        for hero, side_z in zip(
            observation["frame_state"]["hero_states"][:2], (3150, -3150)
        ):
            hero["location"] = _raw_position(-12000, side_z, 1)
        features = FeatureProcessV1(1).process_feature(observation)
        fields = schema_v1.LANE_PLAN_FIELD_MAP
        for feature in features:
            plan = _block(feature, "lane_plan")[0]
            self.assertEqual(plan[fields["opening_active"].start], 1)
            self.assertEqual(plan[fields["hard_mask_active"].start], 1)
            self.assertEqual(plan[fields["phase_released"].start], 0)

    def test_last_hit_bonus_and_farming_auxiliary_are_consumed_once(self):
        observation = make_observation(frame=500, include_soldiers=False)
        for info in observation["heroes"][:2]:
            legal = np.zeros(FrozenV1Config.RAW_LEGAL_ACTION_DIM, dtype=np.float32)
            legal[:122] = 1.0
            legal[122:].reshape(13, 39)[:, 0] = 1.0
            info["legal_action"] = legal
        killer = observation["frame_state"]["hero_states"][0]
        observation["frame_state"]["frame_action"]["dead_action"] = [
            {
                "killer": killer,
                "death": {
                    "runtime_id": 9001,
                    "config_id": 19,
                    "camp": 2,
                    "actor_type": "ACTOR_MONSTER",
                    "sub_type": "ACTOR_SUB_SOLDIER",
                },
            }
        ]
        processor = FeatureProcessV1(1)
        processor.process_feature(observation)
        shaping = processor.consume_behavior_shaping(500)
        self.assertEqual(shaping["last_hit_counts"], (1, 0))
        np.testing.assert_allclose(
            shaping["farming_auxiliary_rewards"], (0.05, 0.0), atol=1e-7
        )
        self.assertEqual(shaping["no_target_counts"], (0, 0))
        self.assertEqual(shaping["no_target_grace_counts"], (1, 1))
        self.assertAlmostEqual(
            shaping["rewards"][0],
            V1TrainingConfig.LANE_LAST_HIT_REWARD,
            places=7,
        )
        self.assertEqual(
            processor.consume_behavior_shaping(500)["rewards"],
            (0.0, 0.0),
        )

    def test_no_target_penalty_has_grace_and_decision_time_normalization(self):
        processor = FeatureProcessV1(1)

        def no_target_observation(frame):
            observation = make_observation(frame=frame, include_soldiers=False)
            for info in observation["heroes"][:2]:
                legal = np.zeros(
                    FrozenV1Config.RAW_LEGAL_ACTION_DIM, dtype=np.float32
                )
                legal[:122] = 1.0
                legal[122:].reshape(13, 39)[:, 0] = 1.0
                info["legal_action"] = legal
            return observation

        for frame in (500, 506, 512):
            processor.process_feature(no_target_observation(frame))
            shaping = processor.consume_behavior_shaping(frame)
            self.assertEqual(shaping["no_target_counts"], (0, 0))
            self.assertEqual(shaping["no_target_grace_counts"], (1, 1))
        processor.process_feature(no_target_observation(518))
        shaping = processor.consume_behavior_shaping(518)
        self.assertEqual(shaping["no_target_counts"], (1, 1))
        self.assertAlmostEqual(shaping["no_target_penalties"][0], -0.0005)
        processor.process_feature(no_target_observation(521))
        shaping = processor.consume_behavior_shaping(521)
        self.assertAlmostEqual(shaping["no_target_penalties"][0], -0.00025)

    def test_strategic_grass_wait_suppresses_no_target_penalty(self):
        processor = FeatureProcessV1(1)
        for frame in (500, 506, 512, 518):
            observation = make_observation(frame=frame, include_soldiers=False)
            for hero in observation["frame_state"]["hero_states"][:2]:
                hero["is_in_grass"] = True
            for info in observation["heroes"][:2]:
                legal = np.zeros(
                    FrozenV1Config.RAW_LEGAL_ACTION_DIM, dtype=np.float32
                )
                legal[:122] = 1.0
                legal[122:].reshape(13, 39)[:, 0] = 1.0
                info["legal_action"] = legal
            processor.process_feature(observation)
            shaping = processor.consume_behavior_shaping(frame)
            self.assertEqual(shaping["no_target_counts"], (0, 0))
            self.assertEqual(shaping["strategic_wait_counts"], (1, 1))

    def test_tyrant_gather_wait_suppresses_kai_no_target_penalty(self):
        processor = FeatureProcessV1(1)
        tyrant_anchor = u.fixed_monster_anchor(122, "center", 1)
        for frame in (500, 506, 512, 518):
            observation = make_observation(frame=frame, include_soldiers=False)
            observation["official_model_info"] = {"official": {"runtime_id": 11}}
            kai = next(
                hero
                for hero in observation["frame_state"]["hero_states"]
                if hero["config_id"] == 193 and hero["camp"] == 1
            )
            kai["location"] = {
                "x": tyrant_anchor[0],
                "y": 0,
                "z": tyrant_anchor[1],
            }
            for spawn in observation["frame_state"]["monster_spawn_infos"]:
                if spawn["monster_id"] == 122:
                    spawn["spawn_timer"] = 20000
            for info in observation["heroes"]:
                legal = np.zeros(
                    FrozenV1Config.RAW_LEGAL_ACTION_DIM, dtype=np.float32
                )
                legal[:122] = 1.0
                legal[122:].reshape(13, 39)[:, 0] = 1.0
                info["legal_action"] = legal
            processor.process_feature(observation)
            shaping = processor.consume_behavior_shaping(frame)
            self.assertEqual(shaping["no_target_counts"][1], 0)
            self.assertEqual(shaping["strategic_wait_counts"][1], 1)

    def test_released_lane_uses_signed_progress_without_standing_reward(self):
        processor = FeatureProcessV1(1)

        def lane_observation(frame, marco_x):
            observation = make_observation(frame=frame, include_soldiers=False)
            observation["frame_state"]["hero_states"][0]["location"] = (
                _raw_position(marco_x, 3150, 1)
            )
            for spawn in observation["frame_state"]["monster_spawn_infos"]:
                if spawn["monster_id"] == 122:
                    spawn["spawn_timer"] = 60000
            return observation

        processor.process_feature(lane_observation(500, -30000))
        first = processor.consume_behavior_shaping(500)
        self.assertEqual(first["lane_attendance_counts"], (0, 0))
        for history in processor.lane_opening_history.values():
            history["released"] = True
            history["release_reason"] = "first_wave_cleared"
        processor.process_feature(lane_observation(503, -30000))
        processor.consume_behavior_shaping(503)
        processor.process_feature(lane_observation(509, -25000))
        forward = processor.consume_behavior_shaping(509)
        self.assertAlmostEqual(
            forward["lane_attendance_rewards"][0],
            V1TrainingConfig.LANE_ATTENDANCE_PROGRESS_REWARD,
        )
        processor.process_feature(lane_observation(515, -30000))
        backward = processor.consume_behavior_shaping(515)
        self.assertAlmostEqual(
            backward["lane_attendance_rewards"][0],
            -V1TrainingConfig.LANE_ATTENDANCE_PROGRESS_REWARD,
        )

    def test_lane_monitor_distinguishes_lane_wave_and_eating_signals(self):
        processor = FeatureProcessV1(1)
        observation = make_observation(frame=500, include_soldiers=True)
        processor.process_feature(observation)
        metrics = processor.lane_monitor_metrics()
        for role in ("controlled", "official"):
            for suffix in (
                "on_lane_rate",
                "near_wave_rate",
                "eating_signal_rate",
                "opening_progress_mean",
            ):
                value = metrics[f"lane_{role}_{suffix}"]
                self.assertTrue(np.isfinite(value))
                self.assertGreaterEqual(value, 0.0)
                self.assertLessEqual(value, 1.0)
        self.assertIn("lane_hero_132_on_lane_rate", metrics)
        self.assertIn("lane_hero_141_near_wave_rate", metrics)
        self.assertIn("lane_first_wave_clear_count", metrics)
        self.assertIn("lane_opening_emergency_release_count", metrics)

    def test_official_lane_hero_uses_position_then_soldier_target(self):
        far = make_observation(frame=100, include_soldiers=False)
        far["official_model_info"] = {"official": {"runtime_id": 11}}
        official = far["frame_state"]["hero_states"][0]
        official["location"] = _raw_position(-60000, 12000, 1)
        processor = FeatureProcessV1(1)
        processor.process_feature(far)
        context = processor.command_runtime_context()
        masks = build_command_legal_masks_v1(context)
        self.assertEqual(context.official_config_id, 132)
        self.assertEqual(context.lane_route_phase, "base_exit")
        self.assertEqual(masks.intent[5], 1)
        far_command = CommandRuleLegalizerV1().legalize(5, 0, 5, context, masks)
        self.assertTrue(far_command.valid)
        self.assertEqual(far_command.action[0], 2)

        near = make_observation(frame=106, include_soldiers=False)
        near["official_model_info"] = {"official": {"runtime_id": 11}}
        official = near["frame_state"]["hero_states"][0]
        official["location"] = _raw_position(-4000, 3150, 1)
        near["frame_state"]["npc_states"].append(
            _soldier(401, 2, -2000, 19, side_z=3150)
        )
        processor = FeatureProcessV1(1)
        processor.process_feature(near)
        context = processor.command_runtime_context()
        masks = build_command_legal_masks_v1(context)
        self.assertEqual(context.lane_route_phase, "engage_wave")
        self.assertTrue(context.lane_route_active)
        self.assertEqual(np.flatnonzero(masks.intent).tolist(), [5])
        target_command = CommandRuleLegalizerV1().legalize(
            5, 0, 5, context, masks
        )
        self.assertTrue(target_command.valid)
        self.assertEqual(target_command.reason, "push_lane_set_target")
        self.assertEqual(target_command.action, (1, 401, 0, 0))

        # A soldier may disappear after the route chose it. The same intent
        # remains executable as SetPosition and must not fail predict().
        stale_target_legal = list(context.collaboration_target_legal)
        stale_target_legal[context.lane_route_target_slot] = 0.0
        stale_context = replace(
            context, collaboration_target_legal=tuple(stale_target_legal)
        )
        stale_masks = build_command_legal_masks_v1(stale_context)
        fallback_command = CommandRuleLegalizerV1().legalize(
            5, 0, 5, stale_context, stale_masks
        )
        self.assertTrue(fallback_command.valid)
        self.assertEqual(fallback_command.reason, "set_position")
        self.assertEqual(fallback_command.action[0], 2)

    def test_official_lane_keeps_fixed_corridor_before_row_entry(self):
        observation = make_observation(frame=272, include_soldiers=True)
        observation["official_model_info"] = {"official": {"runtime_id": 11}}
        official = observation["frame_state"]["hero_states"][0]
        official["location"] = _raw_position(-33000, 7000, 1)
        processor = FeatureProcessV1(1)
        processor.process_feature(observation)
        plan = processor._memo_context["lane_macro"][132]
        self.assertEqual(plan["phase"], "corridor")
        self.assertIsNotNone(plan["own_front"])
        self.assertFalse(plan["row_entry_reached"])
        self.assertEqual(plan["command_waypoint"], plan["next_waypoint"])
        self.assertEqual(plan["command_waypoint"], (-30000.0, 3150.0))
        context = processor.command_runtime_context()
        self.assertEqual(
            context.position_raw_coordinates[5],
            u.raw_position(plan["command_waypoint"], 1),
        )

    def test_official_lane_retries_once_then_locks_and_confirms_successor(self):
        def lane_observation(frame):
            observation = make_observation(
                frame=frame, camp=1, include_soldiers=False
            )
            observation["official_model_info"] = {
                "official": {"runtime_id": 11}
            }
            official = observation["frame_state"]["hero_states"][0]
            official["location"] = _raw_position(-60000, 12000, 1)
            return observation

        processor = FeatureProcessV1(1)
        processor.process_feature(lane_observation(100))
        first_context = processor.command_runtime_context()
        first_masks = build_command_legal_masks_v1(first_context)
        first_command = CommandRuleLegalizerV1().legalize(
            5, 0, 5, first_context, first_masks
        )
        self.assertEqual(first_command.action[0], 2)
        processor.record_collaboration_command(
            100, first_command.action, intent=5, is_train=False
        )

        processor.process_feature(lane_observation(118))
        retry_context = processor.command_runtime_context()
        retry_masks = build_command_legal_masks_v1(retry_context)
        self.assertTrue(
            processor._memo_context["collaboration"][
                "guard_retry_pending"
            ]
        )
        self.assertEqual(np.flatnonzero(retry_masks.intent).tolist(), [5])
        retry_command = CommandRuleLegalizerV1().legalize(
            5, 0, 5, retry_context, retry_masks
        )
        self.assertEqual(retry_command.action, first_command.action)
        processor.record_collaboration_command(
            118, retry_command.action, intent=5, is_train=False
        )

        processor.process_feature(lane_observation(136))
        fallback_context = processor.command_runtime_context()
        fallback_masks = build_command_legal_masks_v1(fallback_context)
        collaboration = processor._memo_context["collaboration"]
        self.assertEqual(collaboration["guard_failed_goal_count"], 1)
        self.assertTrue(collaboration["guard_fallback_pending"])
        self.assertEqual(np.flatnonzero(fallback_masks.intent).tolist(), [5])
        fallback_command = CommandRuleLegalizerV1().legalize(
            5, 0, 5, fallback_context, fallback_masks
        )
        self.assertEqual(fallback_command.action[0], 2)
        self.assertNotEqual(fallback_command.action, first_command.action)
        self.assertEqual(
            processor._memo_context["lane_macro"][132][
                "official_route_cursor"
            ],
            1,
        )
        processor.record_collaboration_command(
            136, fallback_command.action, intent=5, is_train=False
        )

        accepted = lane_observation(142)
        collaboration_state = accepted["frame_state"][
            "collaboration_state"
        ]
        collaboration_state["micro_last"][1]["frame_no"] = 142
        collaboration_state["micro_state"] = [
            {
                "sender": 12,
                "receiver": 11,
                "target_runtime_id": 0,
                "target_position": {
                    "x": fallback_command.action[2],
                    "y": 0,
                    "z": fallback_command.action[3],
                },
                "end_frame": 742,
            }
        ]
        processor.process_feature(accepted)
        collaboration = processor._memo_context["collaboration"]
        self.assertFalse(collaboration["guard_fallback_pending"])
        self.assertTrue(collaboration["guard_fallback_confirmed"])
        self.assertEqual(
            processor._memo_context["lane_macro"][132][
                "official_route_cursor"
            ],
            1,
        )
        held = build_command_legal_masks_v1(
            processor.command_runtime_context()
        )
        self.assertEqual(np.flatnonzero(held.intent).tolist(), [0])

    def test_accepted_but_unfollowed_lane_goal_retries_once_without_unset(self):
        def lane_observation(frame, raw_goal=None, sent_frame=0):
            observation = make_observation(
                frame=frame, camp=1, include_soldiers=False
            )
            observation["official_model_info"] = {
                "official": {"runtime_id": 11}
            }
            observation["frame_state"]["hero_states"][0][
                "location"
            ] = _raw_position(-60000, 12000, 1)
            if raw_goal is not None:
                collaboration = observation["frame_state"][
                    "collaboration_state"
                ]
                collaboration["micro_last"][1]["frame_no"] = sent_frame
                collaboration["micro_state"] = [
                    {
                        "sender": 12,
                        "receiver": 11,
                        "target_runtime_id": 0,
                        "target_position": {
                            "x": raw_goal[0],
                            "y": 0,
                            "z": raw_goal[1],
                        },
                        "end_frame": 900,
                    }
                ]
            return observation

        processor = FeatureProcessV1(1)
        processor.process_feature(lane_observation(100))
        context = processor.command_runtime_context()
        masks = build_command_legal_masks_v1(context)
        command = CommandRuleLegalizerV1().legalize(
            5, 0, 5, context, masks
        )
        raw_goal = command.action[2:4]
        processor.record_collaboration_command(
            100, command.action, intent=5, is_train=False
        )
        processor.process_feature(
            lane_observation(106, raw_goal, sent_frame=106)
        )
        held = build_command_legal_masks_v1(
            processor.command_runtime_context()
        )
        self.assertEqual(np.flatnonzero(held.intent).tolist(), [0])

        processor.process_feature(
            lane_observation(182, raw_goal, sent_frame=106)
        )
        self.assertTrue(
            processor._memo_context["collaboration"][
                "guard_retry_pending"
            ]
        )
        processor.process_feature(
            lane_observation(206, raw_goal, sent_frame=106)
        )
        retry_context = processor.command_runtime_context()
        retry_masks = build_command_legal_masks_v1(retry_context)
        self.assertEqual(np.flatnonzero(retry_masks.intent).tolist(), [5])
        retry = CommandRuleLegalizerV1().legalize(
            5, 0, 5, retry_context, retry_masks
        )
        self.assertEqual(retry.action, command.action)
        processor.record_collaboration_command(
            206, retry.action, intent=5, is_train=False
        )
        processor.process_feature(
            lane_observation(212, raw_goal, sent_frame=212)
        )
        processor.process_feature(
            lane_observation(288, raw_goal, sent_frame=212)
        )
        collaboration = processor._memo_context["collaboration"]
        self.assertEqual(collaboration["guard_failed_goal_count"], 1)
        self.assertTrue(collaboration["guard_fallback_pending"])
        cooldown_hold = build_command_legal_masks_v1(
            processor.command_runtime_context()
        )
        self.assertEqual(np.flatnonzero(cooldown_hold.intent).tolist(), [0])
        self.assertFalse(cooldown_hold.command_is_train)

    def test_official_kai_fallback_must_be_accepted_before_opening_release(self):
        processor = FeatureProcessV1(1)
        first = make_observation(
            frame=100, camp=1, include_soldiers=False
        )
        processor.process_feature(first)
        macro = processor._memo_context["jungle_macro"]
        official_runtime = processor._memo_context["official_hero"][
            "actor"
        ]["runtime_id"]
        failed_signature = processor._jungle_position_goal_signature(
            official_runtime,
            macro["opening_camp_name"],
            macro["opening_camp_lifecycle_id"],
            macro["next_waypoint"],
        )
        processor.official_command_guard["failed_signatures"].add(
            failed_signature
        )
        processor.official_command_guard["fallback_pending"] = True
        processor.official_command_guard[
            "fallback_source_signature"
        ] = failed_signature

        waiting = make_observation(
            frame=106, camp=1, include_soldiers=False
        )
        processor.process_feature(waiting)
        macro = processor._memo_context["jungle_macro"]
        self.assertTrue(macro["opening_fallback_active"])
        self.assertNotEqual(macro["opening_release_reason"], "official_noncompliant")
        context = processor.command_runtime_context()
        masks = build_command_legal_masks_v1(context)
        fallback_command = CommandRuleLegalizerV1().legalize(
            10,
            0,
            context.jungle_route_position_slot,
            context,
            masks,
        )
        self.assertTrue(fallback_command.valid)
        self.assertEqual(fallback_command.action[0], 2)
        processor.record_collaboration_command(
            106, fallback_command.action, intent=10, is_train=False
        )

        accepted = make_observation(
            frame=112, camp=1, include_soldiers=False
        )
        collaboration_state = accepted["frame_state"][
            "collaboration_state"
        ]
        collaboration_state["micro_last"][0]["frame_no"] = 112
        collaboration_state["micro_state"] = [
            {
                "sender": 11,
                "receiver": 13,
                "target_runtime_id": 0,
                "target_position": {
                    "x": fallback_command.action[2],
                    "y": 0,
                    "z": fallback_command.action[3],
                },
                "end_frame": 712,
            }
        ]
        processor.process_feature(accepted)
        released = processor._memo_context["jungle_macro"]
        self.assertFalse(released["opening_fallback_active"])
        self.assertEqual(
            released["opening_release_reason"], "official_noncompliant"
        )
        self.assertTrue(released["official_noncompliant"])

    def test_official_kai_uses_safe_command_if_all_camp_fallbacks_failed(self):
        processor = FeatureProcessV1(1)
        first = make_observation(
            frame=100, camp=1, include_soldiers=False
        )
        processor.process_feature(first)
        context = processor._memo_context
        macro = context["jungle_macro"]
        official_runtime = context["official_hero"]["actor"][
            "runtime_id"
        ]
        current_signature = processor._jungle_position_goal_signature(
            official_runtime,
            macro["opening_camp_name"],
            macro["opening_camp_lifecycle_id"],
            macro["next_waypoint"],
        )
        processor.official_command_guard["failed_signatures"].add(
            current_signature
        )
        kai_position = context["official_hero"]["side_position"]
        for camp_name, *_rest in KAI_OPENING_CAMP_ORDER:
            if camp_name == macro["opening_camp_name"]:
                continue
            status = processor._opening_camp_status(
                context, camp_name, kai_position
            )
            processor.official_command_guard[
                "failed_signatures"
            ].add(
                processor._jungle_position_goal_signature(
                    official_runtime,
                    camp_name,
                    status["lifecycle_id"],
                    status["side_anchor"],
                    fallback=True,
                )
            )

        processor.process_feature(
            make_observation(
                frame=106, camp=1, include_soldiers=False
            )
        )
        fallback = processor._memo_context["jungle_macro"]
        self.assertTrue(fallback["opening_fallback_active"])
        self.assertTrue(
            fallback["opening_camp_name"].startswith("safe_handoff_")
        )
        command_context = processor.command_runtime_context()
        masks = build_command_legal_masks_v1(command_context)
        self.assertEqual(np.flatnonzero(masks.intent).tolist(), [10])
        command = CommandRuleLegalizerV1().legalize(
            10,
            0,
            command_context.jungle_route_position_slot,
            command_context,
            masks,
        )
        self.assertTrue(command.valid)
        self.assertEqual(command.action[0], 2)
        processor.process_feature(
            make_observation(
                frame=112, camp=1, include_soldiers=False
            )
        )
        persisted = processor._memo_context["jungle_macro"]
        self.assertEqual(
            persisted["opening_camp_name"],
            fallback["opening_camp_name"],
        )
        self.assertEqual(
            persisted["next_waypoint"], fallback["next_waypoint"]
        )

    def test_failed_jungle_goal_is_scoped_to_camp_lifecycle(self):
        processor = FeatureProcessV1(1)
        position = (-30000.0, 8000.0)
        old_goal = processor._jungle_position_goal_signature(
            13, "wolf", 1, position
        )
        respawned_goal = processor._jungle_position_goal_signature(
            13, "wolf", 2, position
        )
        processor.official_command_guard["failed_signatures"].add(old_goal)
        self.assertTrue(processor._official_goal_failed(old_goal))
        self.assertFalse(processor._official_goal_failed(respawned_goal))

    def test_learned_command_retries_once_then_forces_safe_successor(self):
        def released_lane_observation(frame):
            observation = make_observation(
                frame=frame, camp=1, include_soldiers=False
            )
            observation["official_model_info"] = {
                "official": {"runtime_id": 11}
            }
            return observation

        processor = FeatureProcessV1(1)
        processor.process_feature(released_lane_observation(100))
        processor.lane_opening_history[132].update(
            released=True,
            release_reason="first_wave_cleared",
            release_frame=100,
        )
        processor.process_feature(released_lane_observation(106))
        context = processor.command_runtime_context()
        enemy_slot = context.target_runtime_ids.index(21)
        masks = build_command_legal_masks_v1(context)
        first = CommandRuleLegalizerV1().legalize(
            2, enemy_slot, 0, context, masks
        )
        self.assertTrue(first.valid)
        self.assertEqual(first.action, (1, 21, 0, 0))
        processor.record_collaboration_command(
            106, first.action, intent=2, is_train=True
        )

        processor.process_feature(released_lane_observation(124))
        retry_context = processor.command_runtime_context()
        retry_masks = build_command_legal_masks_v1(retry_context)
        retry = CommandRuleLegalizerV1().legalize(
            2, enemy_slot, 0, retry_context, retry_masks
        )
        self.assertTrue(retry.valid)
        self.assertEqual(retry.action, first.action)
        processor.record_collaboration_command(
            124, retry.action, intent=2, is_train=True
        )

        processor.process_feature(released_lane_observation(142))
        fallback_context = processor.command_runtime_context()
        fallback_masks = build_command_legal_masks_v1(fallback_context)
        self.assertTrue(fallback_context.generic_command_fallback_pending)
        self.assertFalse(
            fallback_context.collaboration_target_legal[enemy_slot]
        )
        self.assertEqual(np.flatnonzero(fallback_masks.intent).tolist(), [6])
        fallback_slot = int(
            np.flatnonzero(fallback_masks.position_by_intent[6])[0]
        )
        fallback = CommandRuleLegalizerV1().legalize(
            6, 0, fallback_slot, fallback_context, fallback_masks
        )
        self.assertTrue(fallback.valid)
        self.assertEqual(fallback.action[0], 2)
        processor.record_collaboration_command(
            142, fallback.action, intent=6, is_train=False
        )

        accepted = released_lane_observation(148)
        collaboration_state = accepted["frame_state"][
            "collaboration_state"
        ]
        collaboration_state["micro_last"][1]["frame_no"] = 148
        collaboration_state["micro_state"] = [
            {
                "sender": 12,
                "receiver": 11,
                "target_runtime_id": 0,
                "target_position": {
                    "x": fallback.action[2],
                    "y": 0,
                    "z": fallback.action[3],
                },
                "end_frame": 748,
            }
        ]
        processor.process_feature(accepted)
        accepted_context = processor.command_runtime_context()
        self.assertFalse(
            accepted_context.generic_command_fallback_pending
        )

    def test_official_lane_cannot_target_assigned_wave_before_row_entry(self):
        observation = make_observation(
            frame=500, camp=1, include_soldiers=False
        )
        observation["official_model_info"] = {"official": {"runtime_id": 11}}
        official = observation["frame_state"]["hero_states"][0]
        official["location"] = _raw_position(-18000, 10000, 1)
        observation["frame_state"]["npc_states"].append(
            _soldier(401, 2, 5000, 19, side_z=3150)
        )
        processor = FeatureProcessV1(1)
        processor.process_feature(observation)
        plan = processor._memo_context["lane_macro"][132]
        self.assertEqual(plan["phase"], "corridor")
        self.assertFalse(plan["row_entry_reached"])
        self.assertLessEqual(
            plan["target_distance"], LANE_OFFICIAL_TARGET_COMMAND_DISTANCE
        )
        context = processor.command_runtime_context()
        self.assertEqual(context.lane_route_target_slot, -1)
        masks = build_command_legal_masks_v1(context)
        command = CommandRuleLegalizerV1().legalize(5, 0, 5, context, masks)
        self.assertEqual(command.action[0], 2)

    def test_official_lane_target_gate_includes_twenty_five_thousand(self):
        observation = make_observation(
            frame=500, camp=1, include_soldiers=False
        )
        observation["official_model_info"] = {"official": {"runtime_id": 11}}
        official = observation["frame_state"]["hero_states"][0]
        official["location"] = _raw_position(-18000, 3150, 1)
        # Enemy camp coordinates mirror to our side x=7000: exactly 25k away.
        observation["frame_state"]["npc_states"].extend(
            (
                _soldier(301, 1, 7000, 19, side_z=3150),
                _soldier(401, 2, -7000, 19, side_z=3150),
            )
        )
        processor = FeatureProcessV1(1)
        processor.process_feature(observation)
        plan = processor._memo_context["lane_macro"][132]
        self.assertAlmostEqual(plan["target_distance"], 25000.0)
        self.assertEqual(LANE_OFFICIAL_TARGET_COMMAND_DISTANCE, 26000.0)
        context = processor.command_runtime_context()
        self.assertGreaterEqual(context.lane_route_target_slot, 28)
        masks = build_command_legal_masks_v1(context)
        command = CommandRuleLegalizerV1().legalize(5, 0, 5, context, masks)
        self.assertEqual(command.action, (1, 401, 0, 0))

    def test_official_lane_reacquires_row_after_upper_jungle_detour(self):
        entered = make_observation(frame=300, include_soldiers=False)
        entered["official_model_info"] = {"official": {"runtime_id": 11}}
        official = entered["frame_state"]["hero_states"][0]
        official["location"] = _raw_position(-18000, 3150, 1)
        processor = FeatureProcessV1(1)
        processor.process_feature(entered)
        self.assertTrue(
            processor._memo_context["lane_macro"][132]["row_entry_reached"]
        )

        detoured = make_observation(frame=306, include_soldiers=False)
        detoured["official_model_info"] = {"official": {"runtime_id": 11}}
        detoured_official = detoured["frame_state"]["hero_states"][0]
        detoured_official["location"] = _raw_position(-12000, 12000, 1)
        processor.process_feature(detoured)
        plan = processor._memo_context["lane_macro"][132]
        self.assertTrue(plan["row_entry_reached"])
        self.assertEqual(plan["phase"], "stage_outer")
        self.assertEqual(plan["command_waypoint"], (-18000.0, 3150.0))

    def test_wave_contact_requires_current_assigned_row_proximity(self):
        detoured = make_observation(
            frame=500, camp=1, include_soldiers=False
        )
        detoured["official_model_info"] = {"official": {"runtime_id": 11}}
        official = detoured["frame_state"]["hero_states"][0]
        official["location"] = _raw_position(-4000, -4569, 1)
        detoured["frame_state"]["npc_states"].append(
            _soldier(401, 2, 2000, 19, side_z=3150)
        )
        processor = FeatureProcessV1(1)
        processor.process_feature(detoured)
        plan = processor._memo_context["lane_macro"][132]
        self.assertLess(plan["target_distance"], 9000.0)
        self.assertFalse(plan["contact_reached"])
        self.assertTrue(plan["opening_active"])
        self.assertEqual(plan["release_reason"], "none")

        on_row = copy.deepcopy(detoured)
        on_row["frame_state"]["frame_no"] = 506
        on_row["frame_state"]["hero_states"][0]["location"] = _raw_position(
            -4000, 3150, 1
        )
        processor.process_feature(on_row)
        plan = processor._memo_context["lane_macro"][132]
        self.assertTrue(plan["contact_reached"])
        self.assertEqual(plan["release_reason"], "none")
        self.assertTrue(plan["opening_active"])

        missing = copy.deepcopy(on_row)
        missing["frame_state"]["frame_no"] = 536
        missing["frame_state"]["npc_states"] = [
            actor
            for actor in missing["frame_state"]["npc_states"]
            if actor.get("sub_type") != "ACTOR_SUB_SOLDIER"
        ]
        processor.process_feature(missing)
        cleared = copy.deepcopy(missing)
        cleared["frame_state"]["frame_no"] = 566
        processor.process_feature(cleared)
        plan = processor._memo_context["lane_macro"][132]
        self.assertTrue(plan["first_wave_cleared"])
        self.assertFalse(plan["opening_active"])
        self.assertEqual(plan["release_reason"], "first_wave_cleared")

    def test_official_lane_wave_clear_and_defense_do_not_release_before_contact(self):
        captured = make_observation(
            frame=100, camp=1, include_soldiers=False
        )
        captured["official_model_info"] = {"official": {"runtime_id": 11}}
        captured["frame_state"]["hero_states"][0]["location"] = _raw_position(
            -30000, 3150, 1
        )
        captured["frame_state"]["npc_states"].append(
            _soldier(401, 2, -2000, 19, side_z=3150)
        )
        processor = FeatureProcessV1(1)
        processor.process_feature(captured)
        self.assertTrue(
            processor._memo_context["lane_macro"][132]["first_wave_captured"]
        )

        missing = make_observation(
            frame=136, camp=1, include_soldiers=False
        )
        missing["official_model_info"] = {"official": {"runtime_id": 11}}
        missing["frame_state"]["hero_states"][0]["location"] = _raw_position(
            -30000, 3150, 1
        )
        processor.process_feature(missing)

        pressured = make_observation(
            frame=166, camp=1, include_soldiers=False
        )
        pressured["official_model_info"] = {"official": {"runtime_id": 11}}
        pressured["frame_state"]["hero_states"][0]["location"] = _raw_position(
            -30000, 3150, 1
        )
        pressured["frame_state"]["hero_states"][3]["location"] = _raw_position(
            -60000, 3150, 1
        )
        processor.process_feature(pressured)
        plan = processor._memo_context["lane_macro"][132]
        self.assertTrue(plan["first_wave_cleared"])
        self.assertFalse(plan["contact_reached"])
        self.assertEqual(plan["interrupt"], "structure_defense")
        self.assertTrue(plan["opening_active"])
        self.assertEqual(plan["release_reason"], "none")

        resumed = make_observation(
            frame=172, camp=1, include_soldiers=False
        )
        resumed["official_model_info"] = {"official": {"runtime_id": 11}}
        resumed["frame_state"]["hero_states"][0]["location"] = _raw_position(
            -30000, 3150, 1
        )
        for enemy in resumed["frame_state"]["hero_states"][3:]:
            enemy["location"] = _raw_position(50000, 0, 1)
        processor.process_feature(resumed)
        plan = processor._memo_context["lane_macro"][132]
        self.assertEqual(plan["interrupt"], "none")
        self.assertTrue(plan["opening_active"])
        self.assertEqual(plan["release_reason"], "none")

    def test_official_lane_enemy_contact_is_transient_and_resumes_route(self):
        threatened = make_observation(frame=100, include_soldiers=False)
        threatened["official_model_info"] = {"official": {"runtime_id": 11}}
        official = threatened["frame_state"]["hero_states"][0]
        official["location"] = _raw_position(-30000, 3150, 1)
        enemy = threatened["frame_state"]["hero_states"][3]
        enemy["location"] = _raw_position(-22000, 3150, 1)

        processor = FeatureProcessV1(1)
        processor.process_feature(threatened)
        plan = processor._memo_context["lane_macro"][132]
        self.assertEqual(plan["interrupt"], "enemy_hero")
        self.assertTrue(plan["transient_official_combat"])
        self.assertTrue(plan["opening_active"])
        self.assertEqual(plan["release_reason"], "none")
        masks = build_command_legal_masks_v1(processor.command_runtime_context())
        self.assertEqual(np.flatnonzero(masks.intent).tolist(), [2])

        resumed = make_observation(frame=106, include_soldiers=False)
        resumed["official_model_info"] = {"official": {"runtime_id": 11}}
        resumed_official = resumed["frame_state"]["hero_states"][0]
        resumed_official["location"] = _raw_position(-30000, 3150, 1)
        for resumed_enemy in resumed["frame_state"]["hero_states"][3:]:
            resumed_enemy["location"] = _raw_position(50000, 0, 1)
        processor.process_feature(resumed)
        resumed_plan = processor._memo_context["lane_macro"][132]
        self.assertEqual(resumed_plan["interrupt"], "none")
        self.assertTrue(resumed_plan["opening_active"])
        self.assertEqual(resumed_plan["release_reason"], "none")
        resumed_masks = build_command_legal_masks_v1(
            processor.command_runtime_context()
        )
        self.assertEqual(np.flatnonzero(resumed_masks.intent).tolist(), [5])

    def test_official_lane_structure_defense_is_transient_before_wave_clear(self):
        pressured = make_observation(frame=100, include_soldiers=False)
        pressured["official_model_info"] = {"official": {"runtime_id": 11}}
        official = pressured["frame_state"]["hero_states"][0]
        official["location"] = _raw_position(-30000, 3150, 1)
        enemy = pressured["frame_state"]["hero_states"][3]
        enemy["location"] = _raw_position(-60000, 3150, 1)

        processor = FeatureProcessV1(1)
        processor.process_feature(pressured)
        plan = processor._memo_context["lane_macro"][132]
        self.assertEqual(plan["interrupt"], "structure_defense")
        self.assertTrue(plan["opening_active"])
        self.assertEqual(plan["release_reason"], "none")
        masks = build_command_legal_masks_v1(processor.command_runtime_context())
        self.assertEqual(np.flatnonzero(masks.intent).tolist(), [7])

        resumed = make_observation(frame=106, include_soldiers=False)
        resumed["official_model_info"] = {"official": {"runtime_id": 11}}
        resumed_official = resumed["frame_state"]["hero_states"][0]
        resumed_official["location"] = _raw_position(-30000, 3150, 1)
        for resumed_enemy in resumed["frame_state"]["hero_states"][3:]:
            resumed_enemy["location"] = _raw_position(50000, 0, 1)
        processor.process_feature(resumed)
        resumed_plan = processor._memo_context["lane_macro"][132]
        self.assertEqual(resumed_plan["interrupt"], "none")
        self.assertTrue(resumed_plan["opening_active"])
        self.assertEqual(resumed_plan["release_reason"], "none")


if __name__ == "__main__":
    unittest.main()
