#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""Executable constants for the frozen feature/network v1 contract.

This module is deliberately data-only.  It does not activate the v1 runtime
pipeline; it gives subsequent feature, model, sample, and checkpoint work one
authoritative set of names, indices, fixed slots, enums, and layout checks.
"""

from collections import OrderedDict
from dataclasses import dataclass
from math import prod
from types import MappingProxyType

from agent_ppo.conf.conf import FrozenV1Config
from agent_ppo.feature.collision_avoidance_v1 import COLLISION_AVOIDANCE_SIGNATURE
from agent_ppo.feature.lane_opening_v1 import LANE_OPENING_LEGALIZER_SIGNATURE
from agent_ppo.feature.jungle_opening_v1 import JUNGLE_OPENING_LEGALIZER_SIGNATURE


@dataclass(frozen=True)
class FieldSpec:
    name: str
    start: int
    stop: int

    @property
    def size(self):
        return self.stop - self.start

    @property
    def slice(self):
        return slice(self.start, self.stop)


@dataclass(frozen=True)
class FeatureBlockSpec:
    name: str
    shape: tuple
    start: int
    stop: int

    @property
    def size(self):
        return self.stop - self.start

    @property
    def slice(self):
        return slice(self.start, self.stop)


def _make_scalar_layout(names):
    names = tuple(names)
    if len(names) != len(set(names)):
        raise RuntimeError("frozen v1 field names must be unique")
    return tuple(FieldSpec(name=name, start=index, stop=index + 1) for index, name in enumerate(names))


def _make_field_map(layout):
    return MappingProxyType(OrderedDict((field.name, field) for field in layout))


def _make_group_map(entries):
    return MappingProxyType(OrderedDict((name, FieldSpec(name, start, stop)) for name, start, stop in entries))


def _make_feature_blocks():
    blocks = []
    offset = 0
    for name, shape in zip(FrozenV1Config.FEATURE_BLOCK_NAMES, FrozenV1Config.FEATURE_BLOCK_SHAPES):
        size = prod(shape)
        blocks.append(FeatureBlockSpec(name=name, shape=shape, start=offset, stop=offset + size))
        offset += size
    return tuple(blocks)


FEATURE_BLOCKS = _make_feature_blocks()
FEATURE_BLOCK_BY_NAME = MappingProxyType(OrderedDict((block.name, block) for block in FEATURE_BLOCKS))
STRUCTURED_BLOCK_NAMES = FrozenV1Config.FEATURE_BLOCK_NAMES[1:]


HERO_CONFIG_IDS = (132, 141, 193)
HERO_SLOTS = (
    "friend_132",
    "friend_141",
    "friend_193",
    "enemy_132",
    "enemy_141",
    "enemy_193",
)
CONTROLLED_LINEUPS = (
    (132, 141),
    (132, 193),
    (141, 193),
)
HERO_POLICY_ROUTES = MappingProxyType(OrderedDict(FrozenV1Config.HERO_POLICY_ROUTES))

ORGAN_SLOTS = (
    "own_outer_tower",
    "own_high_tower",
    "own_crystal",
    "enemy_outer_tower",
    "enemy_high_tower",
    "enemy_crystal",
)

MONSTER_TARGET_SLOTS = (
    "own_49_red_statue",
    "enemy_49_red_statue",
    "own_30_greater_demon_pioneer",
    "enemy_30_greater_demon_pioneer",
    "own_31_lesser_demon_pioneer",
    "enemy_31_lesser_demon_pioneer",
    "own_32_greater_demon_archer",
    "enemy_32_greater_demon_archer",
    "own_33_lesser_demon_archer",
    "enemy_33_lesser_demon_archer",
    "own_42_greater_whitetail_deer",
    "enemy_42_greater_whitetail_deer",
    "own_43_lesser_whitetail_deer",
    "enemy_43_lesser_whitetail_deer",
    "own_44_greater_dark_wolf",
    "enemy_44_greater_dark_wolf",
    "own_45_lesser_dark_wolf",
    "enemy_45_lesser_dark_wolf",
    "center_59_treasure_thief",
    "center_122_tyrant",
)
MONSTER_TARGET_CONFIG_AND_SIDE = (
    (49, "own"),
    (49, "enemy"),
    (30, "own"),
    (30, "enemy"),
    (31, "own"),
    (31, "enemy"),
    (32, "own"),
    (32, "enemy"),
    (33, "own"),
    (33, "enemy"),
    (42, "own"),
    (42, "enemy"),
    (43, "own"),
    (43, "enemy"),
    (44, "own"),
    (44, "enemy"),
    (45, "own"),
    (45, "enemy"),
    (59, "center"),
    (122, "center"),
)

TARGET_SLOTS = (
    "none",
    "enemy_hero_132",
    "enemy_hero_141",
    "enemy_hero_193",
    "friend_hero_132",
    "friend_hero_141",
    "friend_hero_193",
    "query_main_hero",
    *MONSTER_TARGET_SLOTS,
    *(f"enemy_soldier_{index}" for index in range(10)),
    "current_enemy_structure",
)

RESOURCE_ANCHOR_SLOTS = (
    "central_speed_cake_102",
    "own_outer_tower_health_pack",
    "own_high_tower_health_pack",
    "enemy_outer_tower_health_pack",
    "enemy_high_tower_health_pack",
)

COMMAND_ACTION_TYPES = (
    "none",
    "set_target",
    "set_position",
    "unset",
)
COMMAND_INTENTS = (
    "no_command",
    "unset",
    "focus_enemy_hero",
    "attack_structure",
    "take_key_objective",
    "push_lane",
    "gather_team",
    "defend_structure",
    "retreat_safe",
    "attack_jungle_monster",
    "move_or_wait_jungle_camp",
)
COMMAND_POSITION_BANK = (
    "none",
    "hero132_current",
    "hero141_current",
    "hero193_current",
    "friendly_wave_front",
    "lane_clash_or_enemy_wave_front",
    "own_outer_anchor",
    "own_high_or_crystal_anchor",
    "tyrant_gather_anchor",
    "own_spring_anchor",
    *MONSTER_TARGET_SLOTS,
)
TARGET_ACTIVE_COMMAND_INTENTS = (2, 3, 4, 9)
POSITION_ACTIVE_COMMAND_INTENTS = (5, 6, 7, 8, 10)
COMMAND_POSITION_LEGAL_BY_INTENT = MappingProxyType(
    {
        5: (4, 5),
        6: (1, 2, 3, 8),
        7: (5, 6, 7),
        8: (6, 7, 9),
        10: tuple(range(10, 30)),
    }
)


BUFF_RAW_WHITELIST = (
    10000,
    11001,
    11003,
    20080,
    20090,
    20140,
    20160,
    20230,
    20231,
    20240,
    20250,
    20251,
    21250,
    50000,
    50009,
    60005,
    60006,
    60007,
    90000,
    90005,
    90014,
    90015,
    90201,
    90202,
    90204,
    90206,
    132007,
    132050,
    132051,
    132060,
    132150,
    132160,
    132371,
    132940,
    132941,
    132961,
    132962,
    132971,
    132972,
    141001,
    141250,
    141251,
    141340,
    141350,
    141950,
    141999,
    193150,
    193160,
    193230,
    193231,
    193350,
    193351,
    193361,
    193362,
    193371,
    193372,
    193950,
    193990,
    193999,
    801049,
    900009,
    900052,
    902070,
    911220,
    911260,
    911290,
    911360,
    911361,
    912220,
    912250,
    912370,
    912371,
    913225,
    914110,
    914220,
    914221,
    914230,
    914232,
    914240,
    914245,
    914250,
    914260,
    914261,
    915110,
    915120,
    915122,
    915127,
    915128,
    915130,
    915136,
    915146,
    915147,
    915148,
    915150,
    915160,
    915190,
    915225,
    915234,
    915236,
    915247,
    915422,
    915423,
    915453,
    915600,
    915720,
    915731,
    915734,
    915735,
    915740,
    915900,
    915950,
    917470,
    917671,
    919900,
    1329999,
)
BUFF_STRONG_SEMANTIC_FLAGS = (
    "has_90000_recover",
    "has_900009_recover_effect",
    "has_recover_combo",
    "has_90201_speed",
    "has_90014_health_pack_window",
    "has_90204_902070_dark_wolf_long",
    "has_90206_dark_wolf_short",
    "has_90202_monster49_settlement",
    "has_801049_smite_target_effect",
    "tyrant_stage_1",
    "tyrant_stage_2",
    "tyrant_stage_3",
)
BUFF_FAMILIES = ("132", "141", "193", "60", "80", "90", "91", "other")
BUFF_UNKNOWN_HASH_BUCKET_COUNT = 8


def stable_unknown_buff_bucket(config_id):
    return ((int(config_id) * 2654435761) % (2**32)) % BUFF_UNKNOWN_HASH_BUCKET_COUNT


SKILL_SLOT_NAMES = (
    "common_attack",
    "skill_1",
    "skill_2",
    "skill_3",
    "recover_extension",
    "summoner",
)
SKILL_FIELD_NAMES = (
    "exists",
    "level_norm",
    "usable",
    "cooldown_ratio",
    "cooldown_max_norm",
    "succ_used_in_frame",
    "used_times_norm",
    "hit_hero_times_norm",
    "has_next_config",
    "combo_effect_time_norm",
)
ABILITY_NAMES = (
    "no_control",
    "no_move",
    "no_skill",
    "immune_negative",
    "immune_control",
    "no_move_rotate",
    "immune_crit",
    "blindness",
    "move_protect",
    "no_recover_energy",
    "freeze",
    "dead_control",
    "no_collision_detection",
    "no_joint_skill",
    "abort_move",
    "forbid_select",
    "renewal",
    "sprint",
    "no_move_but_can_ratate",
    "forbid_select_by_skill_org",
    "immune_positive_and_persist_from_other_originators",
    "repressed",
    "immune_de_move_speed",
)

HERO_FIELD_NAMES = (
    "slot_exists",
    "alive",
    "visible_current",
    "current_observation_valid",
    "last_seen_used",
    "last_seen_age_norm",
    "is_friend",
    "is_enemy",
    "controlled_by_us",
    "official_teammate",
    "opponent_controlled",
    "is_query_main",
    "hero_id_132",
    "hero_id_141",
    "hero_id_193",
    "respawning",
    "hp_ratio",
    "max_hp_norm",
    "hp_known",
    "low_hp",
    "revive_time_norm",
    "ep_ratio",
    "max_ep_norm",
    "ep_known",
    "uses_ep",
    "hp_recover_norm",
    "ep_recover_norm",
    "behavior_dead",
    "behavior_moving",
    "behavior_attacking",
    "behavior_casting",
    "behavior_other",
    "phy_atk_norm",
    "phy_def_norm",
    "mgc_atk_norm",
    "mgc_def_norm",
    "mov_spd_norm",
    "atk_spd_norm",
    "phy_armor_hurt_norm",
    "mgc_armor_hurt_norm",
    "crit_rate_norm",
    "crit_effect_norm",
    "phy_vamp_norm",
    "mgc_vamp_norm",
    "cd_reduce_norm",
    "ctrl_reduce_norm",
    "attack_range_norm",
    "sight_area_norm",
    *(f"{slot}_{field}" for slot in SKILL_SLOT_NAMES for field in SKILL_FIELD_NAMES),
    "level_norm",
    "exp_progress_norm",
    "money_current_norm",
    "money_total_norm",
    "money_delta_norm",
    "equipment_value_norm",
    "equipment_slot_count_norm",
    "kill_cnt_norm",
    "dead_cnt_norm",
    "assist_cnt_norm",
    "total_hurt_to_hero_log_norm",
    "total_be_hurt_by_hero_log_norm",
    *(f"ability_{name}" for name in ABILITY_NAMES),
    "abilities_valid",
    *(f"buff_raw_{config_id}_present" for config_id in BUFF_RAW_WHITELIST),
    *BUFF_STRONG_SEMANTIC_FLAGS,
    *(f"{name}_age_norm" for name in BUFF_STRONG_SEMANTIC_FLAGS),
    *(f"buff_family_{family}_present" for family in BUFF_FAMILIES),
    *(f"buff_unknown_hash_{index}_present" for index in range(BUFF_UNKNOWN_HASH_BUCKET_COUNT)),
    "buff_count_norm",
    "new_buff_count_norm",
    "lost_buff_count_norm",
    "unknown_buff_count_norm",
    "has_no_buff",
    "mark_known",
    "marco_13200_present",
    "marco_layer_norm",
    "diaochan_mark_present",
    "diaochan_layer_norm",
    "mark_origin_is_self",
    "mark_origin_is_friend",
    "mark_origin_is_enemy",
    "mark_origin_unknown",
    "other_mark_present",
    "location_valid",
    "movement_cache_valid",
    "side_x_norm",
    "side_z_norm",
    *(f"side_x_coarse_bucket_{index}" for index in range(11)),
    *(f"side_z_coarse_bucket_{index}" for index in range(9)),
    *(
        f"region_{name}"
        for name in (
            "own_base",
            "own_tower_zone",
            "own_lane",
            "mid_lane",
            "enemy_lane",
            "enemy_tower_zone",
            "enemy_base",
            "upper_jungle",
            "lower_jungle",
            "tyrant_zone",
            "speed_buff_zone",
            "unknown",
        )
    ),
    "distance_to_lane_center_norm",
    "distance_to_own_base_norm",
    "distance_to_enemy_base_norm",
    "movement_valid",
    "delta_side_x_per_frame_norm",
    "delta_side_z_per_frame_norm",
    "speed_per_frame_norm",
    "toward_enemy_crystal",
    "toward_own_crystal",
    "toward_query_hero",
    "toward_nearest_enemy",
    "toward_nearest_friend",
    "toward_tyrant",
    "toward_lane_center",
    "away_from_enemy_tower",
    "recent_hit_hero",
    "recent_hit_soldier",
    "recent_hit_monster",
    "recent_hit_organ",
    "recent_hurt_by_hero",
    "recent_hurt_by_soldier",
    "recent_hurt_by_monster",
    "recent_hurt_by_tower",
    "recent_skill_used",
    *(f"last_skill_slot_{index}" for index in range(6)),
    "recent_damage_dealt_norm",
    "recent_damage_taken_norm",
    "recent_kill",
    "recent_assist",
    "recent_death",
    "is_in_grass",
    "in_enemy_tower_range",
    "targeted_by_enemy_tower",
    "in_own_tower_range",
    "enemy_tower_has_minion_target",
    "own_minion_cover",
    "nearest_enemy_tower_distance_norm",
    "nearest_own_tower_distance_norm",
)
HERO_FIELDS = _make_scalar_layout(HERO_FIELD_NAMES)
HERO_FIELD_MAP = _make_field_map(HERO_FIELDS)
HERO_GROUPS = _make_group_map(
    (
        ("identity_validity", 0, 16),
        ("hp_ep_behavior", 16, 32),
        ("combat_attributes", 32, 48),
        ("skills", 48, 108),
        ("economy_statistics", 108, 120),
        ("abilities", 120, 144),
        ("buff", 144, 304),
        ("mark", 304, 314),
        ("static_position", 314, 353),
        ("movement_trend", 353, 365),
        ("recent_combat", 365, 385),
        ("grass_tower_tactical", 385, 393),
    )
)


SOLDIER_FIELD_NAMES = (
    "location_valid",
    "position_current_valid",
    "side_x_norm",
    "side_z_norm",
    *(f"side_x_coarse_bucket_{index}" for index in range(11)),
    *(f"side_z_coarse_bucket_{index}" for index in range(9)),
    *(
        f"lane_zone_{name}"
        for name in (
            "own_base",
            "own_high_to_outer",
            "own_outer_to_mid",
            "mid",
            "enemy_outer_to_mid",
            "enemy_high_to_outer",
            "enemy_base",
        )
    ),
    "slot_exists",
    "alive",
    "visible_current",
    "belongs_to_official_target_order",
    "is_friend",
    "is_enemy",
    "type_melee",
    "type_ranged",
    "type_cannon",
    "type_super",
    "hp_ratio",
    "max_hp_norm",
    "hp_known",
    "low_hp",
    "attack_range_norm",
    "phy_atk_norm",
    "kill_income_norm",
    "attack_target_none",
    "attack_target_hero132",
    "attack_target_hero141",
    "attack_target_hero193",
    "attack_target_soldier",
    "attack_target_organ",
    "attack_target_monster",
    "attack_target_other",
)
SOLDIER_FIELDS = _make_scalar_layout(SOLDIER_FIELD_NAMES)
SOLDIER_FIELD_MAP = _make_field_map(SOLDIER_FIELDS)
SOLDIER_GROUPS = _make_group_map(
    (
        ("lane_position", 0, 31),
        ("validity", 31, 35),
        ("relation", 35, 37),
        ("type", 37, 41),
        ("hp", 41, 45),
        ("combat", 45, 48),
        ("attack_target", 48, 56),
    )
)


ORGAN_FIELD_NAMES = (
    "anchor_location_valid",
    "side_x_norm",
    "side_z_norm",
    *(f"side_x_coarse_bucket_{index}" for index in range(11)),
    *(f"side_z_coarse_bucket_{index}" for index in range(9)),
    "slot_exists",
    "alive",
    "visible_current",
    "current_actor_position_valid",
    "is_friend",
    "is_enemy",
    "type_outer",
    "type_high",
    "type_crystal",
    "hp_ratio",
    "max_hp_norm",
    "hp_known",
    "low_hp",
    "attack_range_norm",
    "attack_norm",
    "kill_income_norm",
    "attack_target_none",
    "attack_target_friend_hero",
    "attack_target_enemy_hero",
    "attack_target_soldier",
    "attack_target_other",
    "friend_soldier_in_range",
    "enemy_soldier_in_range",
    "tower_hit_streak_norm",
    "current_target_has_minion_cover",
)
ORGAN_FIELDS = _make_scalar_layout(ORGAN_FIELD_NAMES)
ORGAN_FIELD_MAP = _make_field_map(ORGAN_FIELDS)
ORGAN_GROUPS = _make_group_map(
    (
        ("anchor_position", 0, 23),
        ("validity", 23, 27),
        ("relation", 27, 29),
        ("type", 29, 32),
        ("hp", 32, 36),
        ("combat", 36, 39),
        ("attack_target", 39, 44),
        ("soldier_range", 44, 46),
        ("tower_context", 46, 48),
    )
)


MONSTER_FIELD_NAMES = (
    "location_valid",
    "position_current_valid",
    "side_x_norm",
    "side_z_norm",
    *(f"side_x_coarse_bucket_{index}" for index in range(11)),
    *(f"side_z_coarse_bucket_{index}" for index in range(9)),
    *(
        f"region_{name}"
        for name in (
            "own_base",
            "own_tower_zone",
            "own_lane",
            "mid_lane",
            "enemy_lane",
            "enemy_tower_zone",
            "enemy_base",
            "upper_jungle",
            "lower_jungle",
            "tyrant_zone",
            "speed_buff_zone",
            "unknown",
        )
    ),
    "slot_exists",
    "alive",
    "visible_current",
    "position_source_is_current_actor",
    "side_relation_own",
    "side_relation_enemy",
    "side_relation_center",
    *(f"monster_type_{config_id}" for config_id in (49, 30, 31, 32, 33, 42, 43, 44, 45, 59, 122)),
    "hp_ratio",
    "max_hp_norm",
    "hp_known",
    "spawn_timer_valid",
    "spawn_timer_norm",
    "team_attacking_monster",
)
MONSTER_FIELDS = _make_scalar_layout(MONSTER_FIELD_NAMES)
MONSTER_FIELD_MAP = _make_field_map(MONSTER_FIELDS)
MONSTER_GROUPS = _make_group_map(
    (
        ("mid_position", 0, 36),
        ("validity", 36, 40),
        ("side_relation", 40, 43),
        ("type", 43, 54),
        ("hp", 54, 57),
        ("spawn_attack", 57, 60),
    )
)


BULLET_FIELD_NAMES = (
    "slot_exists",
    "source_valid",
    "position_valid",
    "velocity_valid",
    "source_type_hero132",
    "source_type_hero141",
    "source_type_hero193",
    "source_type_tower",
    "source_slot_common",
    "source_slot_skill1",
    "source_slot_skill2",
    "source_slot_skill3",
    "source_slot_other",
    "source_slot_unknown",
    "relative_dx_norm",
    "relative_dz_norm",
    "distance_norm",
    "sin_bearing",
    "cos_bearing",
    "velocity_side_x_norm",
    "velocity_side_z_norm",
    "speed_norm",
    "relative_motion_valid",
    "relative_velocity_dx_norm",
    "relative_velocity_dz_norm",
    "radial_closing_speed_norm",
    "tangential_speed_norm",
    "time_to_closest_approach_norm",
    "closest_approach_distance_norm",
    "collision_course",
    "targets_query_hero",
    "unknown_skill",
)
BULLET_FIELDS = _make_scalar_layout(BULLET_FIELD_NAMES)
BULLET_FIELD_MAP = _make_field_map(BULLET_FIELDS)


TARGET_FIELD_NAMES = (
    "slot_valid",
    "entity_exists",
    "alive",
    "visible_current",
    "position_valid",
    "selectable",
    "current_observation_valid",
    "last_seen_used",
    "type_none",
    "type_enemy_hero",
    "type_friend_hero",
    "type_self",
    "type_monster",
    "type_soldier",
    "type_outer_tower",
    "type_high_tower",
    "type_crystal",
    "type_other",
    "relative_dx_norm",
    "relative_dz_norm",
    "distance_norm",
    "sin_bearing",
    "cos_bearing",
    "target_side_x_norm",
    "target_side_z_norm",
    "in_local_range",
    "same_semantic_region",
    "topology_reachable",
    "hp_ratio",
    "hp_known",
    "low_hp",
    "threat_to_query",
    "attacking_friend",
    "marco_mark_layer_norm",
    "diaochan_mark_layer_norm",
    "resource_or_structure_value_norm",
    "any_button_target_legal",
    "common_attack_target_legal",
    "chosen_skill_target_legal",
    "is_query_current_attack_target",
    "has_minion_cover_if_structure",
    "attack_range_margin_norm",
    "collaboration_target_legal",
    "official_hero_alive",
    "official_hero_distance_norm",
    "official_hero_in_effective_range",
    "is_official_current_command_target",
    "official_to_target_eta_norm",
)
TARGET_FIELDS = _make_scalar_layout(TARGET_FIELD_NAMES)
TARGET_FIELD_MAP = _make_field_map(TARGET_FIELDS)
TARGET_GROUPS = _make_group_map(
    (
        ("validity", 0, 8),
        ("identity_type", 8, 18),
        ("relative_geometry", 18, 28),
        ("combat_resource", 28, 36),
        ("action_context", 36, 42),
        ("collaboration_context", 42, 48),
    )
)


HERO_RELATION_FIELD_NAMES = (
    "relation_valid",
    "other_observation_confidence",
    "is_query_self",
    "is_friend",
    "is_enemy",
    "relative_dx_norm",
    "relative_dz_norm",
    "distance_norm",
    "sin_bearing",
    "cos_bearing",
    "hp_ratio_diff",
    "level_diff_norm",
    "gold_diff_norm",
    "query_can_attack_other",
    "other_can_attack_query",
    "query_attacking_other",
    "other_attacking_query",
    "query_recently_damaged_other",
    "other_recently_damaged_query",
    "other_moving_toward_query",
    "query_moving_toward_other",
    "query_facing_other",
    "other_facing_query",
    "same_semantic_region",
)
HERO_RELATION_FIELDS = _make_scalar_layout(HERO_RELATION_FIELD_NAMES)
HERO_RELATION_FIELD_MAP = _make_field_map(HERO_RELATION_FIELDS)


DYNAMIC_LANE_POSITION_FIELD_NAMES = (
    "dynamic_lane_valid",
    "own_active_boundary_outer",
    "own_active_boundary_high",
    "own_active_boundary_crystal",
    "enemy_active_boundary_outer",
    "enemy_active_boundary_high",
    "enemy_active_boundary_crystal",
    "own_active_boundary_side_x_norm",
    "enemy_active_boundary_side_x_norm",
    "dynamic_lane_progress",
    *(f"dynamic_lane_bucket_{index}" for index in range(61)),
    "bucket_residual",
)
DYNAMIC_LANE_POSITION_FIELDS = _make_scalar_layout(DYNAMIC_LANE_POSITION_FIELD_NAMES)
DYNAMIC_LANE_POSITION_FIELD_MAP = _make_field_map(DYNAMIC_LANE_POSITION_FIELDS)


TYRANT_LOCAL_POSITION_FIELD_NAMES = (
    "tyrant_fine_valid",
    "tyrant_alive",
    "tyrant_spawn_within_30s",
    "team_attacking_tyrant",
    "enemy_attacking_tyrant",
    "relative_dx_norm",
    "relative_dz_norm",
    "distance_norm",
    *(f"local_x_bucket_{index}" for index in range(21)),
    *(f"local_z_bucket_{index}" for index in range(21)),
    "topology_outside",
    "topology_own_entrance",
    "topology_enemy_entrance",
    "topology_pit",
)
TYRANT_LOCAL_POSITION_FIELDS = _make_scalar_layout(TYRANT_LOCAL_POSITION_FIELD_NAMES)
TYRANT_LOCAL_POSITION_FIELD_MAP = _make_field_map(TYRANT_LOCAL_POSITION_FIELDS)


TYRANT_TEAM_CONTEXT_FIELD_NAMES = (
    "controlled_1_distance_to_tyrant",
    "controlled_2_distance_to_tyrant",
    "official_distance_to_tyrant",
    "controlled_1_eta_to_tyrant",
    "controlled_2_eta_to_tyrant",
    "official_eta_to_tyrant",
    "controlled_1_near_tyrant",
    "controlled_2_near_tyrant",
    "official_near_tyrant",
    "controlled_1_moving_toward_tyrant",
    "controlled_2_moving_toward_tyrant",
    "official_moving_toward_tyrant",
    "controlled_1_combat_ready",
    "controlled_2_combat_ready",
    "official_combat_ready",
    "friend_count_near_tyrant_norm",
    "enemy_count_near_tyrant_norm",
    "tyrant_eta_advantage",
    "tyrant_local_number_advantage",
    "friend_smite_ready",
    "enemy_smite_ready_confidence",
    "all_three_can_reach_before_window",
    "official_command_available",
    "official_command_goal_is_tyrant",
)
TYRANT_TEAM_CONTEXT_FIELDS = _make_scalar_layout(TYRANT_TEAM_CONTEXT_FIELD_NAMES)
TYRANT_TEAM_CONTEXT_FIELD_MAP = _make_field_map(TYRANT_TEAM_CONTEXT_FIELDS)


LANE_STATE_FIELD_NAMES = (
    "friend_alive_count_norm",
    "enemy_alive_count_norm",
    "friend_melee_count_norm",
    "enemy_melee_count_norm",
    "friend_ranged_count_norm",
    "enemy_ranged_count_norm",
    "friend_cannon_count_norm",
    "enemy_cannon_count_norm",
    "friend_super_count_norm",
    "enemy_super_count_norm",
    "friend_total_hp_norm",
    "enemy_total_hp_norm",
    "friend_total_attack_norm",
    "enemy_total_attack_norm",
    "friend_cannon_present",
    "enemy_cannon_present",
    "friend_wave_present",
    "enemy_wave_present",
    "friend_wave_value_norm",
    "enemy_wave_value_norm",
    "own_front_valid",
    "own_front_side_x_norm",
    "own_front_side_z_norm",
    "own_front_dynamic_progress",
    "own_front_velocity_norm",
    "own_front_to_enemy_boundary_norm",
    "own_front_to_own_boundary_norm",
    "own_front_hp_ratio",
    "own_front_is_cannon",
    "own_front_under_enemy_tower",
    "enemy_front_observed",
    "enemy_front_inferred",
    "enemy_front_confidence",
    "enemy_front_side_x_norm",
    "enemy_front_side_z_norm",
    "enemy_front_dynamic_progress",
    "enemy_front_velocity_norm",
    "enemy_front_to_own_boundary_norm",
    "enemy_front_to_enemy_boundary_norm",
    "enemy_front_hp_ratio",
    "enemy_front_is_cannon",
    "enemy_front_under_own_tower",
    "front_gap_norm",
    "lane_front_advantage",
    "friend_soldiers_near_clash_norm",
    "enemy_soldiers_near_clash_norm",
    "friend_heroes_near_clash_norm",
    "enemy_heroes_near_clash_norm",
    "waves_engaged",
    "clash_position_valid",
    "clash_side_x_norm",
    "local_number_advantage",
    "friend_wave_under_enemy_outer",
    "friend_wave_under_enemy_high",
    "friend_wave_near_enemy_crystal",
    "enemy_wave_under_own_outer",
    "enemy_wave_under_own_high",
    "enemy_wave_near_own_crystal",
    "enemy_tower_targeting_friend_soldier",
    "own_tower_targeting_enemy_soldier",
    "friend_wave_has_tower_aggro",
    "enemy_wave_has_tower_aggro",
    "query_has_minion_cover",
    "enemy_has_minion_cover",
    "both_outer_alive",
    "own_outer_down",
    "enemy_outer_down",
    "both_outer_down",
    "own_high_exposed",
    "enemy_high_exposed",
    "own_crystal_exposed",
    "enemy_crystal_exposed",
    "can_pressure_enemy_outer",
    "can_pressure_enemy_high",
    "can_pressure_enemy_crystal",
    "enemy_pressure_own_outer",
    "enemy_pressure_own_high",
    "enemy_pressure_own_crystal",
    "wave_supports_push",
    "wave_requires_defense",
    "own_front_delta_norm",
    "enemy_front_delta_norm",
    "front_gap_delta_norm",
    "friend_wave_hp_delta_norm",
    "enemy_wave_hp_delta_norm",
    "friend_clear_speed_norm",
    "enemy_clear_speed_norm",
    "enemy_outer_recent_damage_norm",
    "enemy_high_recent_damage_norm",
    "enemy_crystal_recent_damage_norm",
    "own_outer_recent_damage_norm",
    "own_high_recent_damage_norm",
    "own_crystal_recent_damage_norm",
    "friend_wave_recently_spawned",
    "enemy_wave_recently_spawned",
    "lane_trend_valid",
)
LANE_STATE_FIELDS = _make_scalar_layout(LANE_STATE_FIELD_NAMES)
LANE_STATE_FIELD_MAP = _make_field_map(LANE_STATE_FIELDS)
LANE_STATE_GROUPS = _make_group_map(
    (
        ("wave_composition", 0, 20),
        ("lane_fronts", 20, 44),
        ("pressure_combat", 44, 64),
        ("push_defense_phase", 64, 80),
        ("recent_trend", 80, 96),
    )
)


RESOURCE_ANCHOR_FIELD_NAMES = (
    "slot_exists",
    "visible_current",
    "available",
    "permanently_disabled",
    "is_speed_buff",
    "is_health_pack",
    "is_own_side",
    "is_enemy_side",
    "is_neutral",
    "anchor_side_x_norm",
    "anchor_side_z_norm",
    "respawn_timer_valid",
    "respawn_timer_norm",
    "controlled_hero_1_distance_norm",
    "controlled_hero_2_distance_norm",
    "official_hero_distance_norm",
    "nearest_friend_distance_norm",
    "nearest_enemy_distance_norm",
    "friend_count_near_norm",
    "enemy_count_near_norm",
    "low_hp_friend_near",
    "low_hp_enemy_near",
    "effect_held_by_friend",
    "effect_held_by_enemy",
    "friend_eta_min_norm",
    "enemy_eta_min_norm",
    "eta_advantage",
    "contested",
    "safe_access",
    "eta_comparison_valid",
    "associated_tower_alive",
    "query_eta_norm",
)
RESOURCE_ANCHOR_FIELDS = _make_scalar_layout(RESOURCE_ANCHOR_FIELD_NAMES)
RESOURCE_ANCHOR_FIELD_MAP = _make_field_map(RESOURCE_ANCHOR_FIELDS)


COLLABORATION_STATE_FIELD_NAMES = (
    "current_type_none",
    "current_type_set_target",
    "current_type_set_position",
    "current_type_unset",
    "current_command_active",
    "sender_is_controlled_hero_1",
    "sender_is_controlled_hero_2",
    "sender_unknown",
    "receiver_is_official_hero",
    "receiver_alive",
    "target_runtime_maps_to_slot",
    "target_alive",
    "target_visible",
    "target_position_valid",
    "target_is_last_seen",
    "command_end_frame_valid",
    "remaining_duration_norm",
    "micro_cd_norm",
    "cooldown_ready",
    "max_duration_norm",
    "command_age_norm",
    "target_changed_or_invalid",
    "command_persists_through_receiver_death",
    "current_state_valid",
    "official_slot_valid",
    "official_alive",
    "official_respawning",
    "official_position_valid",
    "official_to_target_distance_norm",
    "official_to_position_distance_norm",
    "official_approaching_command_goal",
    "official_moving_away_from_goal",
    "official_attacking_command_target",
    "official_recent_hit_command_target",
    "official_recent_damage_to_target_norm",
    "official_recent_kill_on_command_target",
    "official_recent_assist_on_command_target",
    "official_real_cmd_matches_command",
    "official_reached_effective_range",
    "official_response_valid",
    "last_send_valid",
    "frames_since_last_send_norm",
    "last_send_was_valid",
    "last_send_was_invalid",
    "last_command_overwritten",
    "last_command_unset",
    "last_target_died",
    "repeated_same_target",
    "can_send_set_target",
    "can_send_set_position",
    "can_send_unset",
    "current_command_still_legal",
    "current_command_goal_completed",
    "official_far_from_team",
    "official_grouped_with_team",
    "official_in_danger",
    "enemy_hero_target_available",
    "structure_target_available",
    "soldier_target_available",
    "objective_target_available",
    "gather_position_available",
    "defend_position_available",
    "push_position_available",
    "objective_position_available",
)
COLLABORATION_STATE_FIELDS = _make_scalar_layout(COLLABORATION_STATE_FIELD_NAMES)
COLLABORATION_STATE_FIELD_MAP = _make_field_map(COLLABORATION_STATE_FIELDS)
COLLABORATION_STATE_GROUPS = _make_group_map(
    (
        ("protocol_current_command", 0, 24),
        ("official_hero_response", 24, 40),
        ("command_history", 40, 48),
        ("macro_readiness", 48, 64),
    )
)


GLOBAL_STATE_FIELD_NAMES = (
    "game_time_norm",
    "early_game",
    "mid_game",
    "late_game",
    "super_late_game",
    "stage_progress",
    "match_clock_valid",
    "timeout_pressure",
    "wave_cycle_sin",
    "wave_cycle_cos",
    "tyrant_countdown_norm",
    "tyrant_countdown_valid",
    "speed_buff_countdown_norm",
    "speed_buff_countdown_valid",
    "nearest_health_pack_countdown_norm",
    "nearest_health_pack_countdown_valid",
    "friend_gold_sum_norm",
    "enemy_gold_sum_norm",
    "gold_advantage_norm",
    "friend_gold_mean_norm",
    "enemy_gold_mean_norm",
    "controlled_gold_sum_norm",
    "official_gold_norm",
    "richest_friend_gold_norm",
    "richest_enemy_gold_norm",
    "friend_level_sum_norm",
    "enemy_level_sum_norm",
    "level_advantage_norm",
    "friend_exp_progress_mean",
    "enemy_exp_progress_mean",
    "friend_hp_power_norm",
    "enemy_hp_power_norm",
    "friend_attribute_power_norm",
    "enemy_attribute_power_norm",
    "power_advantage_norm",
    "friend_skill_ready_count_norm",
    "enemy_skill_ready_count_norm",
    "friend_ultimate_ready_count_norm",
    "enemy_ultimate_ready_count_norm",
    "key_buff_advantage",
    "friend_alive_count_norm",
    "enemy_alive_count_norm",
    "alive_advantage",
    "controlled_alive_count_norm",
    "official_hero_alive",
    "friend_dead_count_norm",
    "enemy_dead_count_norm",
    "max_friend_revive_time_norm",
    "min_friend_revive_time_norm",
    "max_enemy_revive_time_norm",
    "min_enemy_revive_time_norm",
    "friend_low_hp_count_norm",
    "enemy_low_hp_count_norm",
    "local_number_advantage_query",
    "force_window_by_alive_advantage",
    "danger_window_by_alive_disadvantage",
    "own_outer_alive",
    "enemy_outer_alive",
    "own_high_alive",
    "enemy_high_alive",
    "own_crystal_alive",
    "enemy_crystal_alive",
    "own_outer_hp_ratio",
    "enemy_outer_hp_ratio",
    "own_high_hp_ratio",
    "enemy_high_hp_ratio",
    "own_crystal_hp_ratio",
    "enemy_crystal_hp_ratio",
    "outer_tower_advantage",
    "high_tower_advantage",
    "crystal_hp_advantage",
    "own_crystal_exposed",
    "enemy_crystal_exposed",
    "own_lane_front_norm",
    "enemy_lane_front_norm",
    "lane_front_advantage",
    "wave_supports_push",
    "wave_requires_defense",
    "structure_pressure_advantage",
    "crystal_race_active",
    "tyrant_alive",
    "tyrant_spawn_within_30s",
    "tyrant_team_attacking",
    "tyrant_enemy_attacking",
    "friend_tyrant_stage_1",
    "friend_tyrant_stage_2",
    "friend_tyrant_stage_3",
    "enemy_tyrant_stage_1",
    "enemy_tyrant_stage_2",
    "enemy_tyrant_stage_3",
    "dark_wolf_own_available_or_soon",
    "dark_wolf_enemy_available_or_soon",
    "friend_has_dark_wolf_buff",
    "enemy_has_dark_wolf_buff",
    "speed_buff_available_or_soon",
    "friend_has_speed_buff",
    "enemy_has_speed_buff",
    "own_outer_health_pack_available",
    "own_high_health_pack_available",
    "enemy_outer_health_pack_available",
    "enemy_high_health_pack_available",
    "objective_local_number_advantage",
    "objective_contest_risk",
    "objective_control_advantage",
    "friend_grouped",
    "enemy_grouped",
    "friend_split",
    "enemy_split",
    "friend_avg_hp",
    "enemy_avg_hp",
    "lowest_friend_hp",
    "lowest_enemy_hp",
    "friend_damage_ready_count_norm",
    "enemy_damage_ready_count_norm",
    "friend_escape_ready_count_norm",
    "enemy_escape_ready_count_norm",
    "fight_readiness_advantage",
    "disengage_pressure",
    "isolated_enemy_exists",
    "endangered_friend_exists",
    "official_available_for_command",
    "command_cooldown_ready",
    "current_command_active",
    "current_command_legal",
    "official_grouped_with_team",
    "official_far_from_team",
    "valid_collaboration_target_exists",
    "valid_collaboration_position_exists",
)
GLOBAL_STATE_FIELDS = _make_scalar_layout(GLOBAL_STATE_FIELD_NAMES)
GLOBAL_STATE_FIELD_MAP = _make_field_map(GLOBAL_STATE_FIELDS)
GLOBAL_STATE_GROUPS = _make_group_map(
    (
        ("clock_stage", 0, 16),
        ("economy_level_power", 16, 40),
        ("alive_revive", 40, 56),
        ("structure_lane", 56, 80),
        ("objectives_resources", 80, 104),
        ("teamfight_collaboration", 104, 128),
    )
)

JUNGLE_PLAN_FIELD_NAMES = (
    "plan_valid",
    "kai_is_official",
    "team_emergency_active",
    "query_is_kai",
    "phase_upper_deer",
    "phase_upper_wolf",
    "phase_lane_soak_wave",
    "phase_lower_clear",
    "phase_tyrant_prepare",
    "phase_control_cycle",
    "mode_farm",
    "mode_group",
    "mode_push",
    "interrupt_none",
    "interrupt_contest_enemy",
    "interrupt_retreat",
    "interrupt_tyrant_emergency",
    "interrupt_recover",
    "interrupt_structure_defense",
    *(f"tyrant_buff_tier_{index}" for index in range(4)),
    "kai_level_norm",
    "kai_level_at_least_2",
    "kai_level_at_least_3",
    "upper_deer_cleared_once",
    "upper_wolf_cleared_once",
    "lower_remaining_norm",
    "tyrant_alive",
    "tyrant_spawn_within_arrival",
    "enemy_attacking_tyrant",
    "enemy_jungle_locked",
    "own_jungle_invaded",
    "contest_favorable",
    "ally_eta_norm",
    "enemy_eta_norm",
    "next_waypoint_dx_norm",
    "next_waypoint_dz_norm",
    "next_waypoint_distance_norm",
    "path_distance_norm",
    "route_progress",
    "route_stuck",
    *(f"goal_node_{index}" for index in range(12)),
    *(f"next_node_{index}" for index in range(12)),
)
JUNGLE_PLAN_FIELDS = _make_scalar_layout(JUNGLE_PLAN_FIELD_NAMES)
JUNGLE_PLAN_FIELD_MAP = _make_field_map(JUNGLE_PLAN_FIELDS)


LANE_PLAN_FIELD_NAMES = (
    "plan_valid",
    "query_is_lane_hero",
    "query_is_official",
    "opening_active",
    "hard_mask_active",
    "contact_reached",
    "assigned_upper",
    "assigned_lower",
    "own_wave_present",
    "enemy_wave_present",
    "phase_inactive",
    "phase_base_exit",
    "phase_corridor",
    "phase_stage_outer",
    "phase_follow_wave",
    "phase_engage_wave",
    "phase_released",
    "interrupt_none",
    "interrupt_enemy_hero",
    "interrupt_recent_hit",
    "interrupt_low_hp",
    "interrupt_structure_defense",
    "interrupt_dead",
    "waypoint_valid",
    "next_waypoint_dx_norm",
    "next_waypoint_dz_norm",
    "next_waypoint_distance_norm",
    "route_progress",
    "query_side_x_norm",
    "query_side_z_norm",
    "assigned_row_z_norm",
    "own_row_front_valid",
    "own_row_front_side_x_norm",
    "own_row_front_side_z_norm",
    "enemy_row_front_valid",
    "enemy_row_front_side_x_norm",
    "enemy_row_front_side_z_norm",
    "row_clash_valid",
    "row_clash_side_x_norm",
    "row_clash_side_z_norm",
    "distance_to_own_wave_norm",
    "distance_to_enemy_wave_norm",
    "nearest_enemy_hero_distance_norm",
    "query_hp_ratio",
    "enemy_hero_near",
    "recent_enemy_hero_hit",
    "structure_defense_required",
    "own_outer_alive",
    "first_wave_captured",
    "waiting_for_wave",
    "following_wave",
    "official_can_target_soldier",
    "target_soldier_slot_norm",
    "target_soldier_hp_ratio",
    "target_soldier_distance_norm",
    "target_soldier_in_attack_range",
    "stable_clash_valid",
    "stable_clash_side_x_norm",
    "first_wave_contact_soon",
    "released_by_first_wave_clear",
    "target_lock_active",
    "route_node_base_exit",
    "route_node_corridor",
    "route_node_stage_or_wave",
)
LANE_PLAN_FIELDS = _make_scalar_layout(LANE_PLAN_FIELD_NAMES)
LANE_PLAN_FIELD_MAP = _make_field_map(LANE_PLAN_FIELDS)


MAIN_CONTEXT_FIELD_NAMES = (
    "query_hero_slot_friend132",
    "query_hero_slot_friend141",
    "query_hero_slot_friend193",
    "query_hero_slot_enemy132",
    "query_hero_slot_enemy141",
    "query_hero_slot_enemy193",
    "controlled_lineup_132_141",
    "controlled_lineup_132_193",
    "controlled_lineup_141_193",
    "official_hero_id_132",
    "official_hero_id_141",
    "official_hero_id_193",
    "query_slot_valid",
    "query_alive",
    "query_visible",
    "query_is_controlled_by_us",
    *(f"legal_button_{index}" for index in range(13)),
    "any_enemy_hero_target_available",
    "any_nonhero_unit_target_available",
    "structure_target_available",
)
MAIN_CONTEXT_FIELDS = _make_scalar_layout(MAIN_CONTEXT_FIELD_NAMES)
MAIN_CONTEXT_FIELD_MAP = _make_field_map(MAIN_CONTEXT_FIELDS)


FIELD_LAYOUTS = MappingProxyType(
    OrderedDict(
        (
            ("heroes", HERO_FIELDS),
            ("soldiers", SOLDIER_FIELDS),
            ("organs", ORGAN_FIELDS),
            ("monsters", MONSTER_FIELDS),
            ("bullets", BULLET_FIELDS),
            ("targets", TARGET_FIELDS),
            ("hero_relations", HERO_RELATION_FIELDS),
            ("dynamic_lane_position", DYNAMIC_LANE_POSITION_FIELDS),
            ("tyrant_local_position", TYRANT_LOCAL_POSITION_FIELDS),
            ("tyrant_team_context", TYRANT_TEAM_CONTEXT_FIELDS),
            ("lane_state", LANE_STATE_FIELDS),
            ("resource_anchors", RESOURCE_ANCHOR_FIELDS),
            ("collaboration_state", COLLABORATION_STATE_FIELDS),
            ("global_state", GLOBAL_STATE_FIELDS),
            ("jungle_plan", JUNGLE_PLAN_FIELDS),
            ("lane_plan", LANE_PLAN_FIELDS),
            ("main_context", MAIN_CONTEXT_FIELDS),
        )
    )
)
FIELD_MAPS = MappingProxyType(OrderedDict((name, _make_field_map(layout)) for name, layout in FIELD_LAYOUTS.items()))


MASK_RECONSTRUCTION_FIELDS = MappingProxyType(
    OrderedDict(
        (
            ("hero_valid_mask", ("heroes", "slot_exists")),
            ("soldier_valid_mask", ("soldiers", "slot_exists")),
            ("organ_valid_mask", ("organs", "slot_exists")),
            ("monster_valid_mask", ("monsters", "slot_exists")),
            ("bullet_valid_mask", ("bullets", "slot_exists")),
            ("target_valid_mask", ("targets", "slot_valid")),
            ("collaboration_target_legal", ("targets", "collaboration_target_legal")),
            ("resource_anchor_valid_mask", ("resource_anchors", "slot_exists")),
        )
    )
)
MASK_RECONSTRUCTION_INDICES = MappingProxyType(
    OrderedDict(
        (mask_name, FIELD_MAPS[block_name][field_name].start)
        for mask_name, (block_name, field_name) in MASK_RECONSTRUCTION_FIELDS.items()
    )
)
ENEMY_HERO_SLOT_INDICES = (3, 4, 5)
ENEMY_CURRENT_VISIBLE_FIELD_INDEX = HERO_FIELD_MAP["visible_current"].start


def _contiguous_layout(sizes):
    offset = 0
    entries = []
    for name, size in sizes:
        entries.append((name, offset, offset + size))
        offset += size
    return _make_group_map(entries)


ACTOR_STEP_LAYOUT = _contiguous_layout(
    (
        ("feature", FrozenV1Config.FEATURE_DIM),
        ("conditioned_legal_action", FrozenV1Config.CONDITIONED_LEGAL_ACTION_DIM),
        ("return", 1),
        ("advantage", 1),
        ("actions", FrozenV1Config.ACTOR_ACTION_COUNT),
        ("old_action_probabilities", FrozenV1Config.CONDITIONED_LEGAL_ACTION_DIM),
        ("is_train", 1),
        ("sub_action_mask", FrozenV1Config.ACTOR_SUB_ACTION_MASK_DIM),
        ("auxiliary_returns", FrozenV1Config.AUXILIARY_VALUE_COUNT),
        ("rnn_reset", 1),
    )
)
COMMAND_STEP_LAYOUT = _make_group_map(
    (
        ("conditioned_legal_masks", 0, 80),
        ("return", 80, 81),
        ("advantage", 81, 82),
        ("actions", 82, 85),
        ("old_log_probs", 85, 88),
        ("is_train", 88, 89),
        ("active_head_mask", 89, 92),
    )
)
SAMPLE_LAYOUT = _contiguous_layout(
    (
        ("actor_0_steps", FrozenV1Config.ACTOR_STEP_DIM * FrozenV1Config.RNN_TIME_STEPS),
        ("actor_0_initial_hidden", FrozenV1Config.RNN_HIDDEN_SIZE),
        ("actor_1_steps", FrozenV1Config.ACTOR_STEP_DIM * FrozenV1Config.RNN_TIME_STEPS),
        ("actor_1_initial_hidden", FrozenV1Config.RNN_HIDDEN_SIZE),
        ("command_steps", FrozenV1Config.COMMAND_SEQUENCE_DIM),
    )
)


TEAM_INPUT_COMPONENT_DIMS = MappingProxyType(
    OrderedDict(
        (
            ("hero_tokens", 6 * 128),
            ("soldier_masked_max_mean", 64),
            ("organ_tokens", 6 * 32),
            ("monster_masked_max_mean", 64),
            ("resource_tokens", 5 * 32),
            ("lane_state", 96),
            ("tyrant_team_context", 24),
            ("collaboration_state", 64),
            ("global_state", 128),
        )
    )
)
PRIVATE_INPUT_COMPONENT_DIMS = MappingProxyType(
    OrderedDict(
        (
            ("image_embedding", 128),
            ("relation_dynamic_tokens", 6 * 64),
            ("bullet_masked_max_mean", 64),
            ("target_masked_max_mean", 128),
            ("tyrant_local_position", 54),
            ("jungle_plan", 67),
            ("lane_plan", 64),
            ("main_context", 32),
        )
    )
)
TEAM_INPUT_DIM = 1560
PRIVATE_INPUT_DIM = 921
NETWORK_ENCODER_DIMS = MappingProxyType(
    OrderedDict(
        (
            ("image", (6, 18, 12, 768, 128)),
            ("hero", (393, 256, 128)),
            ("relation_dynamic", (96, 128, 64)),
            ("soldier", (56, 64, 32)),
            ("organ", (48, 64, 32)),
            ("monster", (60, 64, 32)),
            ("bullet", (32, 64, 32)),
            ("target", (48, 128, 64)),
            ("resource", (32, 64, 32)),
        )
    )
)
PSCN_SIGNATURE = (
    ("team", TEAM_INPUT_DIM, 256, 128, 64, 64, 256),
    ("private", PRIVATE_INPUT_DIM, 256, 128, 64, 64, 256),
)
HYBRID_VALUE_SIGNATURE = (
    ("components", FrozenV1Config.HYBRID_VALUE_COMPONENTS),
    ("formula", FrozenV1Config.HYBRID_VALUE_FORMULA),
    ("team_lambda", FrozenV1Config.HYBRID_VALUE_TEAM_LAMBDA),
    ("input_dims", FrozenV1Config.HYBRID_VALUE_INPUT_DIMS),
    ("component_hidden_dim", 128),
    ("auxiliary_hidden_dim", 128),
    ("auxiliary_categories", FrozenV1Config.AUXILIARY_VALUE_CATEGORIES),
)


def checkpoint_signature_v1():
    signature = FrozenV1Config.base_checkpoint_signature()
    signature.update(
        {
            "feature_block_shapes": tuple(
                (block.name, block.shape, block.start, block.stop) for block in FEATURE_BLOCKS
            ),
            "hero_slots": HERO_SLOTS,
            "target_slots": TARGET_SLOTS,
            "command_action_types": COMMAND_ACTION_TYPES,
            "command_intents": COMMAND_INTENTS,
            "command_position_bank": COMMAND_POSITION_BANK,
            "collision_avoidance_signature": COLLISION_AVOIDANCE_SIGNATURE,
            "lane_opening_legalizer_signature": LANE_OPENING_LEGALIZER_SIGNATURE,
            "jungle_opening_legalizer_signature": JUNGLE_OPENING_LEGALIZER_SIGNATURE,
            "network_encoder_dims": tuple(NETWORK_ENCODER_DIMS.items()),
            "pscn_signature": PSCN_SIGNATURE,
            "hybrid_value_signature": HYBRID_VALUE_SIGNATURE,
            "team_input_dim": TEAM_INPUT_DIM,
            "private_input_dim": PRIVATE_INPUT_DIM,
            "actor_step_layout": tuple(
                (name, spec.start, spec.stop)
                for name, spec in ACTOR_STEP_LAYOUT.items()
            ),
            "sample_layout": tuple(
                (name, spec.start, spec.stop)
                for name, spec in SAMPLE_LAYOUT.items()
            ),
        }
    )
    return signature


def _validate_contiguous(specs, expected_stop, label):
    offset = 0
    for spec in specs:
        if spec.start != offset or spec.stop <= spec.start:
            raise RuntimeError(f"invalid frozen v1 {label} layout at {spec.name}")
        offset = spec.stop
    if offset != expected_stop:
        raise RuntimeError(f"invalid frozen v1 {label} size: expected {expected_stop}, got {offset}")


def validate_schema_v1():
    """Validate every frozen field, slot, mask, and serialized slice."""

    FrozenV1Config.validate()
    _validate_contiguous(FEATURE_BLOCKS, FrozenV1Config.FEATURE_DIM, "feature block")

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
    for block_name, expected_width in expected_widths.items():
        _validate_contiguous(FIELD_LAYOUTS[block_name], expected_width, f"{block_name} field")

    if len(HERO_SLOTS) != 6 or len(ORGAN_SLOTS) != 6 or len(RESOURCE_ANCHOR_SLOTS) != 5:
        raise RuntimeError("invalid frozen v1 fixed entity slots")
    if len(MONSTER_TARGET_SLOTS) != 20 or len(TARGET_SLOTS) != 39:
        raise RuntimeError("invalid frozen v1 target slots")
    if tuple(TARGET_SLOTS[8:28]) != MONSTER_TARGET_SLOTS:
        raise RuntimeError("monster and target slot ordering drifted")
    if len(COMMAND_INTENTS) != 11 or len(COMMAND_POSITION_BANK) != 30:
        raise RuntimeError("invalid frozen v1 command enums")
    if len(BUFF_RAW_WHITELIST) != 115 or len(set(BUFF_RAW_WHITELIST)) != 115:
        raise RuntimeError("invalid frozen v1 Buff whitelist")
    if tuple(sorted(BUFF_RAW_WHITELIST)) != BUFF_RAW_WHITELIST:
        raise RuntimeError("frozen v1 Buff whitelist must remain numerically sorted")

    expected_mask_indices = {
        "hero_valid_mask": 0,
        "soldier_valid_mask": 31,
        "organ_valid_mask": 23,
        "monster_valid_mask": 36,
        "bullet_valid_mask": 0,
        "target_valid_mask": 0,
        "collaboration_target_legal": 42,
        "resource_anchor_valid_mask": 0,
    }
    if dict(MASK_RECONSTRUCTION_INDICES) != expected_mask_indices:
        raise RuntimeError("invalid frozen v1 mask reconstruction indices")
    if ENEMY_CURRENT_VISIBLE_FIELD_INDEX != 2:
        raise RuntimeError("invalid frozen v1 enemy visibility index")

    _validate_contiguous(tuple(ACTOR_STEP_LAYOUT.values()), FrozenV1Config.ACTOR_STEP_DIM, "actor step")
    _validate_contiguous(tuple(COMMAND_STEP_LAYOUT.values()), FrozenV1Config.COMMAND_STEP_DIM, "command step")
    _validate_contiguous(tuple(SAMPLE_LAYOUT.values()), FrozenV1Config.SAMPLE_DIM, "sample")
    if ACTOR_STEP_LAYOUT["auxiliary_returns"].size != FrozenV1Config.AUXILIARY_VALUE_COUNT:
        raise RuntimeError("invalid auxiliary-return width in actor step")
    if ACTOR_STEP_LAYOUT["sub_action_mask"].stop != ACTOR_STEP_LAYOUT["auxiliary_returns"].start:
        raise RuntimeError("auxiliary returns must be appended after all legacy actor fields")
    if ACTOR_STEP_LAYOUT["rnn_reset"].size != 1:
        raise RuntimeError("actor GRU reset mask must be one scalar per step")
    if sum(TEAM_INPUT_COMPONENT_DIMS.values()) != TEAM_INPUT_DIM:
        raise RuntimeError("invalid frozen v1 team input dimension")
    if sum(PRIVATE_INPUT_COMPONENT_DIMS.values()) != PRIVATE_INPUT_DIM:
        raise RuntimeError("invalid frozen v1 private input dimension")

    if HERO_FIELD_MAP["abilities_valid"].start != 143:
        raise RuntimeError("hero ability layout drifted")
    if HERO_FIELD_MAP["mark_known"].start != 304:
        raise RuntimeError("hero Mark layout drifted")
    if HERO_FIELD_MAP["movement_valid"].start != 353:
        raise RuntimeError("hero movement layout drifted")
    if TARGET_FIELD_MAP["collaboration_target_legal"].start != 42:
        raise RuntimeError("target collaboration layout drifted")
    if MAIN_CONTEXT_FIELD_MAP["legal_button_0"].start != 16:
        raise RuntimeError("main-context legal button layout drifted")
    return True


validate_schema_v1()
