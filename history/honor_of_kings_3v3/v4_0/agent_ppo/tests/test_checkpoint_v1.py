#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""Strict checkpoint signature and state-dict compatibility tests."""

import copy
import os
import tempfile
import unittest

import torch

from agent_ppo.feature.schema_v1 import checkpoint_signature_v1
from agent_ppo.model.checkpoint_v1 import (
    build_checkpoint_payload_v1,
    load_checkpoint_v1,
    save_checkpoint_v1,
    signature_fingerprint_v1,
    validate_checkpoint_signature_v1,
)
from agent_ppo.model.network_v1 import NetworkV1


PRE_SIX_CAMP_JUNGLE_OPENING_SIGNATURE = (
    ("kai_config_id", 193),
    ("phases", ("upper_deer", "upper_wolf", "lower_clear")),
    ("ordering", "after_channel4_collision_before_lane_opening_and_sampling"),
    ("combat_hold", "until_first_lower_camp_engaged_or_emergency"),
    ("combat_action", "forced_legal_common_attack"),
    ("camp_exit_distance", 6000.0),
    ("cleared_camp_wait", "excluded_until_living_actor_observed"),
    ("topology_cursor", "monotonic_until_goal_or_interrupt_changes"),
)
PRE_SIX_CAMP_FINGERPRINT = (
    "d70ba859f676b77973895e7784382ef68d1097bc5e0149dd74f5185b546ea86a"
)
PRE_SKILL_SIX_CAMP_JUNGLE_OPENING_SIGNATURE = (
    ("kai_config_id", 193),
    ("phases", ("upper_deer", "upper_wolf", "lower_clear")),
    ("ordering", "after_channel4_collision_before_lane_opening_and_sampling"),
    ("combat_hold", "until_six_camps_cleared_or_emergency"),
    ("spawn_wait", "hold_position_until_living_actor_observed"),
    ("combat_action", "forced_legal_common_attack"),
    ("camp_exit_distance", 6000.0),
    ("cleared_camp_wait", "excluded_until_living_actor_observed"),
    ("topology_cursor", "monotonic_until_goal_or_interrupt_changes"),
)
PRE_SKILL_SIX_CAMP_FINGERPRINT = (
    "2814385704a4d02163f20c08b914c516855b0e8cf74a6e8df75d136b180e21c0"
)
PRE_REGION_JUNGLE_OPENING_SIGNATURE = (
    ("kai_config_id", 193),
    ("phases", ("upper_deer", "upper_wolf", "lower_clear")),
    ("ordering", "after_channel4_collision_before_lane_opening_and_sampling"),
    ("combat_hold", "until_six_camps_cleared_or_emergency"),
    ("spawn_wait", "hold_position_until_living_actor_observed"),
    ("combat_action", "forced_available_skill_then_legal_common_attack"),
    ("skill_priority", (8, 5, 4, 6, 7)),
    ("camp_exit_distance", 6000.0),
    ("cleared_camp_wait", "excluded_until_living_actor_observed"),
    ("topology_cursor", "monotonic_until_goal_or_interrupt_changes"),
)
PRE_REGION_FINGERPRINT = (
    "94190f4d925caea5b1538053603ced8982eadb30f90daaf5d7aae14e306fd848"
)
PRE_ADAPTIVE_TIMEOUT_JUNGLE_OPENING_SIGNATURE = (
    ("kai_config_id", 193),
    ("phases", ("upper_deer", "upper_wolf", "lower_clear")),
    ("ordering", "after_channel4_collision_before_lane_opening_and_sampling"),
    ("combat_hold", "until_six_camps_cleared_or_emergency"),
    ("spawn_wait", "hold_position_until_living_actor_observed"),
    ("combat_action", "forced_available_skill_then_legal_common_attack"),
    ("skill_priority", (8, 5, 4, 6, 7)),
    ("camp_exit_distance", 6000.0),
    ("cleared_camp_wait", "excluded_until_living_actor_observed"),
    (
        "topology_cursor",
        "monotonic_with_official_confirmed_region_pass_gates",
    ),
)
PRE_ADAPTIVE_TIMEOUT_FINGERPRINT = (
    "07034eea528d6060edaaed2bc7052145b753d3b338ecc0ef583e478e482a2745"
)
PRE_MONOTONIC_DESCENT_JUNGLE_OPENING_SIGNATURE = (
    ("kai_config_id", 193),
    ("phases", ("upper_deer", "upper_wolf", "lower_clear")),
    ("ordering", "after_channel4_collision_before_lane_opening_and_sampling"),
    (
        "combat_hold",
        "until_six_camps_cleared_opening_timeout_or_emergency",
    ),
    ("opening_max_raw_frame", 1800),
    ("official_upper_order", "adaptive_wolf_or_deer_then_require_both"),
    (
        "official_lower_route",
        "committed_existing_topology_after_pioneer",
    ),
    ("spawn_wait", "hold_position_until_living_actor_observed"),
    ("combat_action", "forced_available_skill_then_legal_common_attack"),
    ("skill_priority", (8, 5, 4, 6, 7)),
    ("camp_exit_distance", 6000.0),
    ("cleared_camp_wait", "excluded_until_living_actor_observed"),
    (
        "topology_cursor",
        "monotonic_with_official_region_gates_and_lower_graph",
    ),
)
PRE_MONOTONIC_DESCENT_FINGERPRINT = (
    "31fb2f82761c1e8164795861d5ccbb3ef8af8a0625cc46e37498d1fe19856e95"
)
PRE_CONFIRMED_CLEAR_SKIP_JUNGLE_OPENING_SIGNATURE = (
    ("kai_config_id", 193),
    ("phases", ("upper_deer", "upper_wolf", "lower_clear")),
    ("ordering", "after_channel4_collision_before_lane_opening_and_sampling"),
    (
        "combat_hold",
        "until_six_camps_cleared_opening_timeout_or_emergency",
    ),
    ("opening_max_raw_frame", 1800),
    ("official_upper_order", "adaptive_wolf_or_deer_then_require_both"),
    (
        "official_upper_route",
        "five_thousand_arrival_monotonic_pass_band_7500",
    ),
    (
        "official_pioneer_route",
        "direct_own_lower_entry_then_pioneer",
    ),
    (
        "official_lower_route",
        "committed_existing_topology_after_pioneer",
    ),
    ("post_release_stage", "terminal_freeze"),
    ("spawn_wait", "hold_position_until_living_actor_observed"),
    ("combat_action", "forced_available_skill_then_legal_common_attack"),
    ("skill_priority", (8, 5, 4, 6, 7)),
    ("camp_exit_distance", 6000.0),
    ("cleared_camp_wait", "excluded_until_living_actor_observed"),
    ("topology_cursor", "monotonic_official_descent_and_lower_graph"),
)
PRE_CONFIRMED_CLEAR_SKIP_FINGERPRINT = (
    "23fe8203be80173ee8536274de1f2bafc30c1518496cf1b987e85edffa3d3b5c"
)
PRE_RESPONSE_GUARD_LANE_OPENING_SIGNATURE = (
    ("lane_hero_config_ids", (132, 141)),
    ("opening_direction_fan", 3),
    ("waypoint_arrival_distance", 1800.0),
    ("release_condition", "actual_enemy_soldier_contact"),
    ("post_contact_control", "ppo_for_controlled_official_wave_guard"),
    ("ordering", "after_channel4_collision_before_sampling"),
)
PRE_KAI_COMBAT_LOCK_JUNGLE_OPENING_SIGNATURE = (
    ("kai_config_id", 193),
    ("phases", ("upper_deer", "upper_wolf", "lower_clear")),
    (
        "ordering",
        "after_channel4_collision_before_lane_opening_and_sampling",
    ),
    (
        "combat_hold",
        "until_six_camps_cleared_opening_timeout_or_emergency",
    ),
    ("opening_max_raw_frame", 1800),
    ("official_upper_order", "adaptive_wolf_or_deer_then_require_both"),
    (
        "official_upper_route",
        "five_thousand_arrival_monotonic_pass_band_7500",
    ),
    (
        "official_pioneer_route",
        "direct_own_lower_entry_then_pioneer",
    ),
    (
        "official_lower_route",
        "committed_topology_5000_and_confirmed_clear_skip",
    ),
    ("post_release_stage", "terminal_freeze"),
    ("spawn_wait", "hold_position_until_living_actor_observed"),
    (
        "combat_action",
        "forced_available_skill_then_legal_common_attack",
    ),
    ("skill_priority", (8, 5, 4, 6, 7)),
    ("camp_exit_distance", 6000.0),
    ("cleared_camp_wait", "excluded_until_living_actor_observed"),
    (
        "topology_cursor",
        "monotonic_official_descent_clear_skip_lower_graph",
    ),
)
PRE_RESPONSE_GUARD_FINGERPRINT = (
    "1d0c0e4cc45a4fe5e5b1f4c1c4bf16a3fcd767b05a64e7a46ad958e09f4ae2e6"
)
PRE_FLASH_BUTTON_FIX_FINGERPRINT = (
    "b7206313dcd6828501500ba7a694cf12038f2d8eb412cb8cea341c4a0c76f0b2"
)
PPO30_FLASH_BUTTON8_FINGERPRINT = (
    "87c82985450ce4dbeba09a28c24812566b32e58cfd344d8e1fb6acc4030d27f1"
)
PPO35_FINGERPRINT = (
    "99fe8fa1a65c1ed97ef0028585d6a77fc535058ab19e214f63cd6c4b480c30aa"
)
PRE_REVIEW_PPO40_FINGERPRINT = (
    "5fee496e006c425edb0bcad5ba93803d41f7548d69ea5623e4224bd87df89da3"
)
CURRENT_PPO40_FINGERPRINT = (
    "7608fa7b301e9077d83347d4581cd84a665cc740b7cd7b66063b9a279351960a"
)
PPO35_REWARD_STATE_MACHINE_V2_FINGERPRINT = (
    "5a8fb0615cf776ac289c52bf4ac16c2d8b060e73aeccf92817a9060c6104df1a"
)
PPO35_REWARD_STATE_MACHINE_V1_FINGERPRINT = (
    "3d2f9d6cf3f698a5e2fa94d2993ef83325563d62cf167e8691c521c4bd4bd374"
)


def _ppo30_flash_button8_signature():
    signature = checkpoint_signature_v1()
    signature.pop("reward_engineering_version")
    signature["feature_version"] = (
        "feature_schema_v20_lane_role_response_guard"
    )
    signature["legalizer_version"] = (
        "collaboration_rule_legalizer_v22_official_response_guard"
    )
    signature["ordinary_action_legalizer_version"] = (
        "ordinary_action_legalizer_v22_kai_camp_combat_lock_flash_button8"
    )
    return signature


def _ppo35_reward_state_machine_v1_signature():
    signature = checkpoint_signature_v1()
    signature["feature_version"] = (
        "feature_schema_v21_ppo35_task_safety_semantics"
    )
    signature["legalizer_version"] = (
        "collaboration_rule_legalizer_v23_ppo35_task_capability_boundary"
    )
    signature["ordinary_action_legalizer_version"] = (
        "ordinary_action_legalizer_v23_ppo35_spawn_tower_safety_flash_button8"
    )
    signature["reward_engineering_version"] = (
        "ppo_v3_5_task_structure_defense_v1"
    )
    return signature


def _ppo35_reward_state_machine_v2_signature():
    signature = checkpoint_signature_v1()
    signature["feature_version"] = (
        "feature_schema_v21_ppo35_task_safety_semantics"
    )
    signature["legalizer_version"] = (
        "collaboration_rule_legalizer_v24_ppo35_task_capability_phase_clocks"
    )
    signature["ordinary_action_legalizer_version"] = (
        "ordinary_action_legalizer_v23_ppo35_spawn_tower_safety_flash_button8"
    )
    signature["reward_engineering_version"] = (
        "ppo_v3_5_reward_state_machine_v2"
    )
    return signature


def _ppo35_signature():
    signature = checkpoint_signature_v1()
    signature["feature_version"] = (
        "feature_schema_v21_ppo35_task_safety_semantics"
    )
    signature["legalizer_version"] = (
        "collaboration_rule_legalizer_v25_ppo35_navigation_watchdog"
    )
    signature["ordinary_action_legalizer_version"] = (
        "ordinary_action_legalizer_v23_ppo35_spawn_tower_safety_flash_button8"
    )
    signature["reward_engineering_version"] = (
        "ppo_v3_5_reward_state_machine_v3"
    )
    return signature


def _pre_review_ppo40_signature():
    signature = checkpoint_signature_v1()
    signature["feature_version"] = (
        "feature_schema_v22_ppo40_conversion_reentry_semantics"
    )
    signature["legalizer_version"] = (
        "collaboration_rule_legalizer_v26_ppo40_reentry_conversion_target_promotion"
    )
    signature["ordinary_action_legalizer_version"] = (
        "ordinary_action_legalizer_v24_ppo40_reentry_smite_late_conversion"
    )
    signature["reward_engineering_version"] = (
        "ppo_v4_0_positive_conversion_state_machine_v1"
    )
    return signature


def _pre_response_guard_signature():
    signature = _ppo30_flash_button8_signature()
    signature["feature_version"] = "feature_schema_v19_lane_contact_guard"
    signature["legalizer_version"] = (
        "collaboration_rule_legalizer_v21_kai_monster_position"
    )
    signature["ordinary_action_legalizer_version"] = (
        "ordinary_action_legalizer_v20_flash_gate"
    )
    signature["lane_opening_legalizer_signature"] = (
        PRE_RESPONSE_GUARD_LANE_OPENING_SIGNATURE
    )
    signature["jungle_opening_legalizer_signature"] = (
        PRE_KAI_COMBAT_LOCK_JUNGLE_OPENING_SIGNATURE
    )
    return signature


class CheckpointV1Test(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(41)

    def test_signature_fingerprint_is_stable_and_complete(self):
        signature = checkpoint_signature_v1()
        first = signature_fingerprint_v1(signature)
        second = signature_fingerprint_v1(copy.deepcopy(signature))
        self.assertEqual(first, second)
        self.assertEqual(first, CURRENT_PPO40_FINGERPRINT)
        self.assertEqual(len(first), 64)
        for key in (
            "feature_block_shapes",
            "network_encoder_dims",
            "pscn_signature",
            "hybrid_value_signature",
            "sample_protocol_version",
            "auxiliary_value_categories",
            "auxiliary_reward_component_indices",
            "hero_action_heads",
            "command_heads",
            "rnn_type",
            "hero_policy_routes",
            "command_position_bank",
            "ordinary_action_legalizer_version",
            "collision_avoidance_signature",
            "reward_engineering_version",
        ):
            self.assertIn(key, signature)

    def test_signature_mismatch_names_changed_contract(self):
        signature = checkpoint_signature_v1()
        signature["rnn_hidden_size"] = 127
        with self.assertRaisesRegex(ValueError, "rnn_hidden_size"):
            validate_checkpoint_signature_v1(signature)

    def test_pre_six_camp_signature_has_exact_one_way_migration(self):
        signature = _pre_response_guard_signature()
        signature["jungle_opening_legalizer_signature"] = (
            PRE_SIX_CAMP_JUNGLE_OPENING_SIGNATURE
        )
        signature["ordinary_action_legalizer_version"] = (
            "ordinary_action_legalizer_v13_single_departure"
        )
        signature["legalizer_version"] = (
            "collaboration_rule_legalizer_v20_lane_contact_guard"
        )
        self.assertEqual(
            signature_fingerprint_v1(signature),
            PRE_SIX_CAMP_FINGERPRINT,
        )
        self.assertEqual(
            validate_checkpoint_signature_v1(signature),
            PRE_SIX_CAMP_FINGERPRINT,
        )

        unknown_opening = checkpoint_signature_v1()
        unknown_opening["jungle_opening_legalizer_signature"] = (
            ("unrecognized_opening_contract", True),
        )
        with self.assertRaisesRegex(
            ValueError,
            "jungle_opening_legalizer_signature",
        ):
            validate_checkpoint_signature_v1(unknown_opening)

        signature["rnn_hidden_size"] = 127
        with self.assertRaisesRegex(ValueError, "rnn_hidden_size"):
            validate_checkpoint_signature_v1(signature)

    def test_pre_skill_six_camp_signature_has_exact_one_way_migration(self):
        signature = _pre_response_guard_signature()
        signature["jungle_opening_legalizer_signature"] = (
            PRE_SKILL_SIX_CAMP_JUNGLE_OPENING_SIGNATURE
        )
        signature["ordinary_action_legalizer_version"] = (
            "ordinary_action_legalizer_v13_single_departure"
        )
        signature["legalizer_version"] = (
            "collaboration_rule_legalizer_v20_lane_contact_guard"
        )
        self.assertEqual(
            signature_fingerprint_v1(signature),
            PRE_SKILL_SIX_CAMP_FINGERPRINT,
        )
        self.assertEqual(
            validate_checkpoint_signature_v1(signature),
            PRE_SKILL_SIX_CAMP_FINGERPRINT,
        )

    def test_pre_region_route_signature_has_exact_one_way_migration(self):
        signature = _pre_response_guard_signature()
        signature["jungle_opening_legalizer_signature"] = (
            PRE_REGION_JUNGLE_OPENING_SIGNATURE
        )
        signature["ordinary_action_legalizer_version"] = (
            "ordinary_action_legalizer_v14_opening_skill_route"
        )
        self.assertEqual(
            signature_fingerprint_v1(signature),
            PRE_REGION_FINGERPRINT,
        )
        self.assertEqual(
            validate_checkpoint_signature_v1(signature),
            PRE_REGION_FINGERPRINT,
        )

    def test_pre_adaptive_timeout_signature_has_exact_one_way_migration(self):
        signature = _pre_response_guard_signature()
        signature["jungle_opening_legalizer_signature"] = (
            PRE_ADAPTIVE_TIMEOUT_JUNGLE_OPENING_SIGNATURE
        )
        signature["ordinary_action_legalizer_version"] = (
            "ordinary_action_legalizer_v15_official_region_route"
        )
        self.assertEqual(
            signature_fingerprint_v1(signature),
            PRE_ADAPTIVE_TIMEOUT_FINGERPRINT,
        )
        self.assertEqual(
            validate_checkpoint_signature_v1(signature),
            PRE_ADAPTIVE_TIMEOUT_FINGERPRINT,
        )

    def test_pre_monotonic_descent_signature_has_exact_one_way_migration(self):
        signature = _pre_response_guard_signature()
        signature["jungle_opening_legalizer_signature"] = (
            PRE_MONOTONIC_DESCENT_JUNGLE_OPENING_SIGNATURE
        )
        signature["ordinary_action_legalizer_version"] = (
            "ordinary_action_legalizer_v16_official_adaptive_timeout"
        )
        self.assertEqual(
            signature_fingerprint_v1(signature),
            PRE_MONOTONIC_DESCENT_FINGERPRINT,
        )
        self.assertEqual(
            validate_checkpoint_signature_v1(signature),
            PRE_MONOTONIC_DESCENT_FINGERPRINT,
        )

    def test_pre_confirmed_clear_skip_signature_has_exact_one_way_migration(self):
        signature = _pre_response_guard_signature()
        signature["jungle_opening_legalizer_signature"] = (
            PRE_CONFIRMED_CLEAR_SKIP_JUNGLE_OPENING_SIGNATURE
        )
        signature["ordinary_action_legalizer_version"] = (
            "ordinary_action_legalizer_v17_official_monotonic_descent"
        )
        self.assertEqual(
            signature_fingerprint_v1(signature),
            PRE_CONFIRMED_CLEAR_SKIP_FINGERPRINT,
        )
        self.assertEqual(
            validate_checkpoint_signature_v1(signature),
            PRE_CONFIRMED_CLEAR_SKIP_FINGERPRINT,
        )

    def test_pre_response_guard_signature_has_exact_one_way_migration(self):
        signature = _pre_response_guard_signature()
        self.assertEqual(
            signature_fingerprint_v1(signature),
            PRE_RESPONSE_GUARD_FINGERPRINT,
        )
        self.assertEqual(
            validate_checkpoint_signature_v1(signature),
            PRE_RESPONSE_GUARD_FINGERPRINT,
        )

    def test_pre_flash_button_fix_signature_has_exact_one_way_migration(self):
        signature = _ppo30_flash_button8_signature()
        signature["ordinary_action_legalizer_version"] = (
            "ordinary_action_legalizer_v21_kai_camp_combat_lock"
        )
        self.assertEqual(
            signature_fingerprint_v1(signature),
            PRE_FLASH_BUTTON_FIX_FINGERPRINT,
        )
        self.assertEqual(
            validate_checkpoint_signature_v1(signature),
            PRE_FLASH_BUTTON_FIX_FINGERPRINT,
        )

        signature["rnn_hidden_size"] = 127
        with self.assertRaisesRegex(ValueError, "rnn_hidden_size"):
            validate_checkpoint_signature_v1(signature)

    def test_pre_flash_button_fix_checkpoint_loads_with_current_weights(self):
        model = NetworkV1()
        payload = build_checkpoint_payload_v1(model, train_step=175862)
        payload["signature"] = _ppo30_flash_button8_signature()
        payload["signature"]["ordinary_action_legalizer_version"] = (
            "ordinary_action_legalizer_v21_kai_camp_combat_lock"
        )
        payload["signature_fingerprint"] = signature_fingerprint_v1(
            payload["signature"]
        )
        self.assertEqual(
            payload["signature_fingerprint"],
            PRE_FLASH_BUTTON_FIX_FINGERPRINT,
        )

        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "model.ckpt-175862.pkl")
            torch.save(payload, path)
            restored = NetworkV1()
            result = load_checkpoint_v1(path, restored, map_location="cpu")

        self.assertEqual(result["train_step"], 175862)
        self.assertEqual(
            result["compatibility_migration"],
            "flash_button7_to_official_button8_v1",
        )
        for name, value in model.state_dict().items():
            torch.testing.assert_close(value, restored.state_dict()[name])

    def test_ppo30_flash_button8_checkpoint_migrates_to_ppo35(self):
        model = NetworkV1()
        payload = build_checkpoint_payload_v1(model, train_step=175862)
        payload["signature"] = _ppo30_flash_button8_signature()
        payload["signature_fingerprint"] = signature_fingerprint_v1(
            payload["signature"]
        )
        self.assertEqual(
            payload["signature_fingerprint"],
            PPO30_FLASH_BUTTON8_FINGERPRINT,
        )

        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "model.ckpt-175862.pkl")
            torch.save(payload, path)
            restored = NetworkV1()
            result = load_checkpoint_v1(path, restored, map_location="cpu")

        self.assertEqual(
            result["compatibility_migration"],
            "ppo30_flash8_to_ppo35_task_safety_reward_v1",
        )
        for name, value in model.state_dict().items():
            torch.testing.assert_close(value, restored.state_dict()[name])

    def test_ppo35_reward_state_machine_v1_checkpoint_migrates_to_v3(self):
        model = NetworkV1()
        payload = build_checkpoint_payload_v1(model, train_step=175862)
        payload["signature"] = (
            _ppo35_reward_state_machine_v1_signature()
        )
        payload["signature_fingerprint"] = signature_fingerprint_v1(
            payload["signature"]
        )
        self.assertEqual(
            payload["signature_fingerprint"],
            PPO35_REWARD_STATE_MACHINE_V1_FINGERPRINT,
        )

        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "model.ckpt-175862.pkl")
            torch.save(payload, path)
            restored = NetworkV1()
            result = load_checkpoint_v1(
                path, restored, map_location="cpu"
            )

        self.assertEqual(
            result["compatibility_migration"],
            "ppo35_reward_state_machine_v1_to_v3",
        )
        for name, value in model.state_dict().items():
            torch.testing.assert_close(
                value, restored.state_dict()[name]
            )

    def test_ppo35_reward_state_machine_v2_checkpoint_migrates_to_v3(self):
        model = NetworkV1()
        payload = build_checkpoint_payload_v1(model, train_step=175862)
        payload["signature"] = (
            _ppo35_reward_state_machine_v2_signature()
        )
        payload["signature_fingerprint"] = signature_fingerprint_v1(
            payload["signature"]
        )
        self.assertEqual(
            payload["signature_fingerprint"],
            PPO35_REWARD_STATE_MACHINE_V2_FINGERPRINT,
        )

        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "model.ckpt-175862.pkl")
            torch.save(payload, path)
            restored = NetworkV1()
            result = load_checkpoint_v1(
                path, restored, map_location="cpu"
            )

        self.assertEqual(
            result["compatibility_migration"],
            "ppo35_reward_state_machine_v2_to_v3",
        )
        for name, value in model.state_dict().items():
            torch.testing.assert_close(
                value, restored.state_dict()[name]
            )

    def test_ppo35_checkpoint_migrates_strictly_to_ppo40(self):
        model = NetworkV1()
        payload = build_checkpoint_payload_v1(model, train_step=175862)
        payload["signature"] = _ppo35_signature()
        payload["signature_fingerprint"] = signature_fingerprint_v1(
            payload["signature"]
        )
        self.assertEqual(
            payload["signature_fingerprint"],
            PPO35_FINGERPRINT,
        )

        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "model.ckpt-175862.pkl")
            torch.save(payload, path)
            restored = NetworkV1()
            result = load_checkpoint_v1(
                path, restored, map_location="cpu"
            )

        self.assertEqual(
            result["compatibility_migration"],
            "ppo35_to_ppo40_positive_conversion_v1",
        )
        for name, value in model.state_dict().items():
            torch.testing.assert_close(
                value, restored.state_dict()[name]
            )

        drifted = _ppo35_signature()
        drifted["rnn_hidden_size"] = 127
        with self.assertRaisesRegex(ValueError, "rnn_hidden_size"):
            validate_checkpoint_signature_v1(drifted)

    def test_pre_review_ppo40_checkpoint_migrates_to_review_closed_v2(self):
        model = NetworkV1()
        payload = build_checkpoint_payload_v1(model, train_step=175863)
        payload["signature"] = _pre_review_ppo40_signature()
        payload["signature_fingerprint"] = signature_fingerprint_v1(
            payload["signature"]
        )
        self.assertEqual(
            payload["signature_fingerprint"],
            PRE_REVIEW_PPO40_FINGERPRINT,
        )

        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "model.ckpt-175863.pkl")
            torch.save(payload, path)
            restored = NetworkV1()
            result = load_checkpoint_v1(
                path, restored, map_location="cpu"
            )

        self.assertEqual(
            result["compatibility_migration"],
            "ppo40_pre_review_to_review_closed_v2",
        )
        for name, value in model.state_dict().items():
            torch.testing.assert_close(
                value, restored.state_dict()[name]
            )

    def test_pre_six_camp_checkpoint_loads_strictly_with_current_weights(self):
        model = NetworkV1()
        payload = build_checkpoint_payload_v1(model, train_step=175862)
        payload["signature"] = _pre_response_guard_signature()
        payload["signature"]["jungle_opening_legalizer_signature"] = (
            PRE_SIX_CAMP_JUNGLE_OPENING_SIGNATURE
        )
        payload["signature"]["ordinary_action_legalizer_version"] = (
            "ordinary_action_legalizer_v13_single_departure"
        )
        payload["signature"]["legalizer_version"] = (
            "collaboration_rule_legalizer_v20_lane_contact_guard"
        )
        payload["signature_fingerprint"] = signature_fingerprint_v1(
            payload["signature"]
        )
        self.assertEqual(
            payload["signature_fingerprint"],
            PRE_SIX_CAMP_FINGERPRINT,
        )

        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "model.ckpt-175862.pkl")
            torch.save(payload, path)
            restored = NetworkV1()
            result = load_checkpoint_v1(path, restored, map_location="cpu")

            self.assertEqual(result["train_step"], 175862)
            self.assertEqual(
                result["compatibility_migration"],
                "pre_six_camp_opening_to_six_camp_opening_v1",
            )
            self.assertEqual(
                result["signature_fingerprint"],
                PRE_SIX_CAMP_FINGERPRINT,
            )
            for name, value in model.state_dict().items():
                torch.testing.assert_close(value, restored.state_dict()[name])

            payload["signature_fingerprint"] = "0" * 64
            corrupt_path = os.path.join(directory, "corrupt-legacy.pkl")
            torch.save(payload, corrupt_path)
            with self.assertRaisesRegex(ValueError, "fingerprint is corrupt"):
                load_checkpoint_v1(
                    corrupt_path,
                    NetworkV1(),
                    map_location="cpu",
                )

    def test_save_and_strict_load_round_trip_with_optimizer(self):
        model = NetworkV1()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "model.ckpt-hybrid.pkl")
            fingerprint = save_checkpoint_v1(
                path,
                model,
                optimizer=optimizer,
                train_step=17,
                metadata={"feature_version": "feature_schema_v1"},
            )
            restored = NetworkV1()
            restored_optimizer = torch.optim.Adam(restored.parameters(), lr=1e-4)
            result = load_checkpoint_v1(
                path,
                restored,
                optimizer=restored_optimizer,
                map_location="cpu",
                load_optimizer=True,
            )
            self.assertEqual(result["train_step"], 17)
            self.assertEqual(result["signature_fingerprint"], fingerprint)
            for name, value in model.state_dict().items():
                torch.testing.assert_close(value, restored.state_dict()[name])

    def test_legacy_payload_and_missing_parameter_fail_closed(self):
        model = NetworkV1()
        with tempfile.TemporaryDirectory() as directory:
            legacy_path = os.path.join(directory, "legacy.pkl")
            torch.save(model.state_dict(), legacy_path)
            with self.assertRaisesRegex(ValueError, "legacy or unknown"):
                load_checkpoint_v1(legacy_path, NetworkV1(), map_location="cpu")

            old_format_payload = build_checkpoint_payload_v1(model)
            old_format_payload["checkpoint_format"] = "agent_ppo_frozen_v1"
            old_format_payload["checkpoint_format_version"] = 1
            old_format_path = os.path.join(directory, "old-format.pkl")
            torch.save(old_format_payload, old_format_path)
            with self.assertRaisesRegex(ValueError, "legacy or unknown"):
                load_checkpoint_v1(
                    old_format_path, NetworkV1(), map_location="cpu"
                )

            payload = build_checkpoint_payload_v1(model)
            removed = next(iter(payload["model_state_dict"]))
            del payload["model_state_dict"][removed]
            broken_path = os.path.join(directory, "broken.pkl")
            torch.save(payload, broken_path)
            with self.assertRaises(RuntimeError):
                load_checkpoint_v1(broken_path, NetworkV1(), map_location="cpu")


if __name__ == "__main__":
    unittest.main()
