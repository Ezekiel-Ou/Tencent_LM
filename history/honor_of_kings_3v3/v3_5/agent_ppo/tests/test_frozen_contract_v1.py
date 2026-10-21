#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

import unittest

import numpy as np

from agent_ppo.conf.conf import FrozenV1Config
from agent_ppo.feature import schema_v1


class FrozenConfigV1Test(unittest.TestCase):
    def test_frozen_signature(self):
        signature = FrozenV1Config.base_checkpoint_signature()

        self.assertEqual(
            signature["feature_version"],
            "feature_schema_v21_ppo35_task_safety_semantics",
        )
        self.assertEqual(
            signature["network_version"],
            "network_architecture_3v3_hybrid_multivalue_team_macro",
        )
        self.assertEqual(
            signature["sample_protocol_version"],
            "sample_protocol_hybrid_multivalue_team_macro_v3",
        )
        self.assertEqual(
            signature["ordinary_action_legalizer_version"],
            "ordinary_action_legalizer_v23_ppo35_spawn_tower_safety_flash_button8",
        )
        self.assertEqual(
            signature["reward_engineering_version"],
            "ppo_v3_5_reward_state_machine_v3",
        )
        self.assertEqual(signature["controlled_actor_count"], 2)
        self.assertEqual(signature["feature_per_actor"], 10157)
        self.assertEqual(signature["structured_per_actor"], 8423)
        self.assertEqual(signature["image_per_actor"], 1734)
        self.assertEqual(signature["hero_action_heads"], (13, 25, 42, 42, 39))
        self.assertEqual(signature["command_heads"], (11, 39, 30))
        self.assertEqual(signature["rnn_type"], "GRU")
        self.assertEqual(signature["rnn_hidden_size"], 128)
        self.assertEqual(signature["rnn_time_steps"], 16)
        self.assertEqual(signature["sample_dim_v1"], 337664)
        self.assertEqual(signature["hero_policy_routes"], ((132, "marco"), (141, "diaochan"), (193, "kai")))
        self.assertEqual(signature["hybrid_value_components"], ("self", "team", "enemy"))
        self.assertEqual(signature["hybrid_value_team_lambda"], 0.2)
        self.assertEqual(
            signature["auxiliary_value_categories"],
            ("farming", "kda", "damage", "pushing", "win"),
        )

    def test_feature_dimension_arithmetic(self):
        block_dims = [int(np.prod(shape)) for shape in FrozenV1Config.FEATURE_BLOCK_SHAPES]

        self.assertEqual(block_dims[0], 1734)
        self.assertEqual(sum(block_dims[1:]), 8423)
        self.assertEqual(sum(block_dims), 10157)
        self.assertEqual(FrozenV1Config.RAW_LEGAL_ACTION_DIM, 13 + 25 + 42 + 42 + 13 * 39)
        self.assertEqual(FrozenV1Config.CONDITIONED_LEGAL_ACTION_DIM, 13 + 25 + 42 + 42 + 39)

    def test_sample_dimension_arithmetic(self):
        actor_step = 10157 + 161 + 1 + 1 + 5 + 161 + 1 + 5 + 5 + 1
        command_step = 80 + 1 + 1 + 3 + 3 + 1 + 3
        actor_sequences = 2 * (actor_step * 16 + 128)
        command_sequence = command_step * 16

        self.assertEqual(actor_step, FrozenV1Config.ACTOR_STEP_DIM)
        self.assertEqual(command_step, FrozenV1Config.COMMAND_STEP_DIM)
        self.assertEqual(actor_sequences, FrozenV1Config.ACTOR_SEQUENCES_DIM)
        self.assertEqual(command_sequence, FrozenV1Config.COMMAND_SEQUENCE_DIM)
        self.assertEqual(actor_sequences + command_sequence, FrozenV1Config.SAMPLE_DIM)


class FeatureSchemaV1Test(unittest.TestCase):
    def test_schema_self_validation(self):
        self.assertTrue(schema_v1.validate_schema_v1())

    def test_top_level_block_order_shapes_and_offsets(self):
        expected_names = (
            "image",
            "heroes",
            "soldiers",
            "organs",
            "monsters",
            "bullets",
            "targets",
            "hero_relations",
            "dynamic_lane_position",
            "tyrant_local_position",
            "tyrant_team_context",
            "lane_state",
            "resource_anchors",
            "collaboration_state",
            "global_state",
            "jungle_plan",
            "lane_plan",
            "main_context",
        )
        expected_shapes = (
            (6, 17, 17),
            (6, 393),
            (20, 56),
            (6, 48),
            (20, 60),
            (10, 32),
            (39, 48),
            (6, 24),
            (6, 72),
            (1, 54),
            (1, 24),
            (1, 96),
            (5, 32),
            (1, 64),
            (1, 128),
            (1, 67),
            (1, 64),
            (1, 32),
        )

        self.assertEqual(tuple(block.name for block in schema_v1.FEATURE_BLOCKS), expected_names)
        self.assertEqual(tuple(block.shape for block in schema_v1.FEATURE_BLOCKS), expected_shapes)
        self.assertEqual(schema_v1.FEATURE_BLOCKS[0].start, 0)
        self.assertEqual(schema_v1.FEATURE_BLOCKS[-1].stop, 10157)
        for left, right in zip(schema_v1.FEATURE_BLOCKS, schema_v1.FEATURE_BLOCKS[1:]):
            self.assertEqual(left.stop, right.start)

    def test_exact_field_widths(self):
        expected_widths = {
            "heroes": 393,
            "soldiers": 56,
            "organs": 48,
            "monsters": 60,
            "bullets": 32,
            "targets": 48,
            "hero_relations": 24,
            "dynamic_lane_position": 72,
            "tyrant_local_position": 54,
            "tyrant_team_context": 24,
            "lane_state": 96,
            "resource_anchors": 32,
            "collaboration_state": 64,
            "global_state": 128,
            "jungle_plan": 67,
            "lane_plan": 64,
            "main_context": 32,
        }

        for name, width in expected_widths.items():
            self.assertEqual(len(schema_v1.FIELD_LAYOUTS[name]), width, name)
            self.assertEqual(schema_v1.FIELD_LAYOUTS[name][-1].stop, width, name)

    def test_critical_field_indices(self):
        self.assertEqual(schema_v1.HERO_FIELD_MAP["slot_exists"].start, 0)
        self.assertEqual(schema_v1.HERO_FIELD_MAP["abilities_valid"].start, 143)
        self.assertEqual(schema_v1.HERO_FIELD_MAP["buff_raw_10000_present"].start, 144)
        self.assertEqual(schema_v1.HERO_FIELD_MAP["mark_known"].start, 304)
        self.assertEqual(schema_v1.HERO_FIELD_MAP["movement_valid"].start, 353)
        self.assertEqual(schema_v1.HERO_FIELD_MAP["recent_hit_hero"].start, 365)
        self.assertEqual(schema_v1.HERO_FIELD_MAP["is_in_grass"].start, 385)

        self.assertEqual(schema_v1.SOLDIER_FIELD_MAP["slot_exists"].start, 31)
        self.assertEqual(schema_v1.SOLDIER_FIELD_MAP["is_enemy"].start, 36)
        self.assertEqual(schema_v1.ORGAN_FIELD_MAP["slot_exists"].start, 23)
        self.assertEqual(schema_v1.MONSTER_FIELD_MAP["slot_exists"].start, 36)
        self.assertEqual(schema_v1.TARGET_FIELD_MAP["collaboration_target_legal"].start, 42)
        self.assertEqual(schema_v1.MAIN_CONTEXT_FIELD_MAP["legal_button_0"].start, 16)
        self.assertEqual(schema_v1.MAIN_CONTEXT_FIELD_MAP["structure_target_available"].start, 31)

    def test_mask_reconstruction_indices(self):
        self.assertEqual(
            dict(schema_v1.MASK_RECONSTRUCTION_INDICES),
            {
                "hero_valid_mask": 0,
                "soldier_valid_mask": 31,
                "organ_valid_mask": 23,
                "monster_valid_mask": 36,
                "bullet_valid_mask": 0,
                "target_valid_mask": 0,
                "collaboration_target_legal": 42,
                "resource_anchor_valid_mask": 0,
            },
        )
        self.assertEqual(schema_v1.ENEMY_HERO_SLOT_INDICES, (3, 4, 5))
        self.assertEqual(schema_v1.ENEMY_CURRENT_VISIBLE_FIELD_INDEX, 2)

    def test_fixed_slots_and_target_alignment(self):
        self.assertEqual(schema_v1.HERO_SLOTS[:3], ("friend_132", "friend_141", "friend_193"))
        self.assertEqual(schema_v1.ORGAN_SLOTS[:3], ("own_outer_tower", "own_high_tower", "own_crystal"))
        self.assertEqual(len(schema_v1.MONSTER_TARGET_SLOTS), 20)
        self.assertEqual(len(schema_v1.TARGET_SLOTS), 39)
        self.assertEqual(schema_v1.TARGET_SLOTS[0], "none")
        self.assertEqual(schema_v1.TARGET_SLOTS[1:4], ("enemy_hero_132", "enemy_hero_141", "enemy_hero_193"))
        self.assertEqual(schema_v1.TARGET_SLOTS[8:28], schema_v1.MONSTER_TARGET_SLOTS)
        self.assertEqual(schema_v1.TARGET_SLOTS[28:38], tuple(f"enemy_soldier_{i}" for i in range(10)))
        self.assertEqual(schema_v1.TARGET_SLOTS[38], "current_enemy_structure")

    def test_buff_whitelist_and_stable_hash(self):
        self.assertEqual(len(schema_v1.BUFF_RAW_WHITELIST), 115)
        self.assertEqual(len(set(schema_v1.BUFF_RAW_WHITELIST)), 115)
        self.assertEqual(tuple(sorted(schema_v1.BUFF_RAW_WHITELIST)), schema_v1.BUFF_RAW_WHITELIST)
        self.assertEqual(schema_v1.BUFF_RAW_WHITELIST[:3], (10000, 11001, 11003))
        self.assertEqual(schema_v1.BUFF_RAW_WHITELIST[-1], 1329999)
        for config_id in (1, 90208, 1234567):
            expected = ((config_id * 2654435761) % (2**32)) % 8
            self.assertEqual(schema_v1.stable_unknown_buff_bucket(config_id), expected)

    def test_command_enums_and_position_masks(self):
        self.assertEqual(len(schema_v1.COMMAND_INTENTS), 11)
        self.assertEqual(schema_v1.COMMAND_INTENTS[0:2], ("no_command", "unset"))
        self.assertEqual(schema_v1.TARGET_ACTIVE_COMMAND_INTENTS, (2, 3, 4, 9))
        self.assertEqual(schema_v1.POSITION_ACTIVE_COMMAND_INTENTS, (5, 6, 7, 8, 10))
        self.assertEqual(len(schema_v1.COMMAND_POSITION_BANK), 30)
        self.assertEqual(schema_v1.COMMAND_POSITION_LEGAL_BY_INTENT[5], (4, 5))
        self.assertEqual(schema_v1.COMMAND_POSITION_LEGAL_BY_INTENT[6], (1, 2, 3, 8))
        self.assertEqual(schema_v1.COMMAND_POSITION_LEGAL_BY_INTENT[7], (5, 6, 7))
        self.assertEqual(schema_v1.COMMAND_POSITION_LEGAL_BY_INTENT[8], (6, 7, 9))

    def test_team_private_input_dimensions(self):
        self.assertEqual(sum(schema_v1.TEAM_INPUT_COMPONENT_DIMS.values()), 1560)
        self.assertEqual(sum(schema_v1.PRIVATE_INPUT_COMPONENT_DIMS.values()), 921)


class SampleLayoutV1Test(unittest.TestCase):
    def test_per_step_layouts(self):
        self.assertEqual(schema_v1.ACTOR_STEP_LAYOUT["feature"].size, 10157)
        self.assertEqual(schema_v1.ACTOR_STEP_LAYOUT["conditioned_legal_action"].size, 161)
        self.assertEqual(schema_v1.ACTOR_STEP_LAYOUT["actions"].size, 5)
        self.assertEqual(schema_v1.ACTOR_STEP_LAYOUT["old_action_probabilities"].size, 161)
        self.assertEqual(schema_v1.ACTOR_STEP_LAYOUT["sub_action_mask"].size, 5)
        self.assertEqual(schema_v1.ACTOR_STEP_LAYOUT["sub_action_mask"].stop, 10492)
        self.assertEqual(
            (schema_v1.ACTOR_STEP_LAYOUT["auxiliary_returns"].start,
             schema_v1.ACTOR_STEP_LAYOUT["auxiliary_returns"].stop),
            (10492, 10497),
        )
        self.assertEqual(
            (schema_v1.ACTOR_STEP_LAYOUT["rnn_reset"].start,
             schema_v1.ACTOR_STEP_LAYOUT["rnn_reset"].stop),
            (10497, 10498),
        )
        self.assertEqual(tuple(schema_v1.ACTOR_STEP_LAYOUT.values())[-1].stop, 10498)

        self.assertEqual(schema_v1.COMMAND_STEP_LAYOUT["conditioned_legal_masks"].size, 80)
        self.assertEqual(schema_v1.COMMAND_STEP_LAYOUT["actions"].size, 3)
        self.assertEqual(schema_v1.COMMAND_STEP_LAYOUT["old_log_probs"].size, 3)
        self.assertEqual(schema_v1.COMMAND_STEP_LAYOUT["active_head_mask"].size, 3)
        self.assertEqual(tuple(schema_v1.COMMAND_STEP_LAYOUT.values())[-1].stop, 92)

    def test_sample_round_trip_slices(self):
        sample = np.arange(FrozenV1Config.SAMPLE_DIM, dtype=np.float32)
        parts = [sample[spec.slice] for spec in schema_v1.SAMPLE_LAYOUT.values()]
        rebuilt = np.concatenate(parts)

        np.testing.assert_array_equal(rebuilt, sample)
        self.assertEqual(parts[0].size, 10498 * 16)
        self.assertEqual(parts[1].size, 128)
        self.assertEqual(parts[2].size, 10498 * 16)
        self.assertEqual(parts[3].size, 128)
        self.assertEqual(parts[4].size, 92 * 16)

    def test_full_checkpoint_signature_contains_command_contract(self):
        signature = schema_v1.checkpoint_signature_v1()

        self.assertEqual(signature["sample_dim_v1"], 337664)
        self.assertEqual(signature["hero_policy_routes"], ((132, "marco"), (141, "diaochan"), (193, "kai")))
        self.assertEqual(signature["command_intents"], schema_v1.COMMAND_INTENTS)
        self.assertEqual(signature["command_position_bank"], schema_v1.COMMAND_POSITION_BANK)
        self.assertEqual(
            signature["legalizer_version"],
            "collaboration_rule_legalizer_v25_ppo35_navigation_watchdog",
        )
        self.assertEqual(
            signature["feature_version"],
            "feature_schema_v21_ppo35_task_safety_semantics",
        )
        self.assertEqual(
            signature["ordinary_action_legalizer_version"],
            "ordinary_action_legalizer_v23_ppo35_spawn_tower_safety_flash_button8",
        )
        self.assertEqual(
            signature["reward_engineering_version"],
            "ppo_v3_5_reward_state_machine_v3",
        )
        self.assertEqual(
            dict(signature["lane_opening_legalizer_signature"])["ordering"],
            "after_channel4_collision_before_sampling",
        )
        self.assertEqual(
            dict(signature["jungle_opening_legalizer_signature"])["combat_hold"],
            "until_six_camps_cleared_opening_timeout_or_emergency",
        )
        self.assertEqual(
            dict(signature["jungle_opening_legalizer_signature"])["spawn_wait"],
            "hold_position_until_living_actor_observed",
        )
        self.assertEqual(
            dict(signature["jungle_opening_legalizer_signature"])["cleared_camp_wait"],
            "excluded_until_living_actor_observed",
        )
        self.assertEqual(
            dict(signature["jungle_opening_legalizer_signature"])["topology_cursor"],
            "monotonic_official_descent_clear_skip_lower_graph",
        )
        self.assertEqual(
            dict(signature["jungle_opening_legalizer_signature"])[
                "opening_max_raw_frame"
            ],
            1800,
        )
        self.assertEqual(
            dict(signature["jungle_opening_legalizer_signature"])[
                "official_pioneer_route"
            ],
            "direct_own_lower_entry_then_pioneer",
        )
        self.assertEqual(
            dict(signature["jungle_opening_legalizer_signature"])[
                "post_release_stage"
            ],
            "terminal_freeze",
        )
        self.assertEqual(
            dict(signature["jungle_opening_legalizer_signature"])[
                "official_lower_route"
            ],
            "committed_topology_5000_and_confirmed_clear_skip",
        )
        self.assertEqual(
            dict(signature["lane_opening_legalizer_signature"])[
                "assigned_row_z"
            ],
            ((132, 3150.0), (141, -3150.0)),
        )
        self.assertEqual(
            dict(signature["jungle_opening_legalizer_signature"])[
                "leash_recovery"
            ],
            "monster_own_anchor_distance_above_4500_only",
        )
        self.assertEqual(
            dict(signature["hybrid_value_signature"])["formula"],
            "(1-team_lambda)*self+team_lambda*team-enemy",
        )
        self.assertEqual(
            dict(signature["hybrid_value_signature"])["auxiliary_categories"],
            ("farming", "kda", "damage", "pushing", "win"),
        )

        signature["sample_dim_v1"] = -1
        self.assertEqual(schema_v1.checkpoint_signature_v1()["sample_dim_v1"], 337664)


if __name__ == "__main__":
    unittest.main()
