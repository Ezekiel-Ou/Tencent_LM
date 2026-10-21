#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright (C) 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Monitor panel configuration for agent_diy.
"""

from kaiwudrl.common.monitor.monitor_config_builder import MonitorConfigBuilder


def _add_metric(panel, metric_name, precision):
    return panel.add_metric(
        metrics_name=metric_name,
        expr=f"round(avg({metric_name}{{}}), {precision})",
    )


def build_monitor():
    monitor = MonitorConfigBuilder()

    config = monitor.title("hok1v1-diy").add_group(
        group_name="diagnostics",
        group_name_en="diagnostics",
    )

    config = config.add_panel(name="ppo_health", name_en="ppo_health", type="line")
    for metric_name, precision in (
        ("total_loss", "0.01"),
        ("value_loss", "0.01"),
        ("policy_loss", "0.01"),
        ("entropy_loss", "0.01"),
        ("mean_ratio", "0.0001"),
        ("approx_kl", "0.000001"),
        ("clip_fraction", "0.0001"),
    ):
        config = _add_metric(config, metric_name, precision)
    config = config.end_panel()

    config = config.add_panel(name="match_eval", name_en="match_eval", type="line")
    for metric_name, precision in (
        ("reward", "0.01"),
    ):
        config = _add_metric(config, metric_name, precision)
    config = config.end_panel()

    config = config.add_panel(name="match_result", name_en="match_result", type="line")
    for metric_name, precision in (
        ("reward_win", "0.0001"),
    ):
        config = _add_metric(config, metric_name, precision)
    config = config.end_panel()

    config = config.add_panel(name="objective_economy", name_en="objective_economy", type="line")
    for metric_name in (
        "reward_tower_hp_point",
        "reward_money",
        "reward_exp",
        "reward_last_hit",
        "reward_safe_last_hit",
        "reward_minion_tower_push",
        "reward_enemy_dead_enemy_cake",
        "reward_enemy_minion_tower_front",
        "reward_enemy_minion_under_own_tower",
        "reward_river_crab_pressure",
        "reward_early_own_half_hero_offense",
        "reward_opening_trade_pressure",
        "reward_forward",
    ):
        config = _add_metric(config, metric_name, "0.0001")
    config = config.end_panel()

    config = config.add_panel(name="defense_pressure", name_en="defense_pressure", type="line")
    for metric_name, precision in (
        ("enemy_minion_tower_front_count", "0.01"),
        ("enemy_minion_under_own_tower_count", "0.01"),
        ("enemy_minion_defense_event_count", "0.01"),
        ("enemy_minion_defense_multiplier", "0.01"),
        ("river_crab_pressure_count", "0.01"),
        ("early_own_half_offense_intent_count", "0.01"),
        ("early_own_half_offense_damage_count", "0.01"),
        ("early_own_half_offense_damage_value", "0.01"),
        ("opening_trade_start_count", "0.01"),
        ("opening_trade_visible_count", "0.01"),
        ("opening_trade_intent_count", "0.01"),
        ("opening_trade_damage_count", "0.01"),
        ("opening_trade_80110_good_count", "0.01"),
        ("opening_trade_good_count", "0.01"),
        ("opening_trade_strong_count", "0.01"),
        ("opening_trade_bad_count", "0.01"),
        ("opening_trade_wasted_80110_count", "0.01"),
        ("opening_trade_reward_value", "0.01"),
    ):
        config = _add_metric(config, metric_name, precision)
    config = config.end_panel()

    config = config.add_panel(name="combat_reward", name_en="combat_reward", type="line")
    for metric_name in (
        "reward_hp_point",
        "reward_ep_rate",
        "reward_kill",
        "reward_death",
        "reward_cleanse_success",
        "reward_duel_summoner_timing",
        "reward_no_op_streak_penalty",
    ):
        config = _add_metric(config, metric_name, "0.0001")
    config = config.end_panel()

    config = config.add_panel(name="skill_hit_reward", name_en="skill_hit_reward", type="line")
    for metric_name in (
        "reward_luban_skill1_hit_enemy_soldier",
        "reward_luban_skill1_hit_enemy_hero",
        "reward_luban_skill2_hit_enemy_hero",
        "reward_luban_skill3_hit_enemy_hero",
        "reward_direnjie_skill1_hit_enemy_hero",
        "reward_direnjie_skill3_hit_enemy_hero",
        "reward_direnjie_skill3_followup_damage",
        "reward_direnjie_skill3_miss",
        "direnjie_skill3_followup_count",
        "direnjie_skill3_miss_count",
    ):
        config = _add_metric(config, metric_name, "0.0001")
    config = config.end_panel()

    config = config.add_panel(name="action_buttons", name_en="action_buttons", type="line")
    for metric_name in (
        "action_noop_count",
        "action_move_count",
        "action_attack_count",
        "action_skill_count",
        "action_recover_count",
        "action_summoner_count",
    ):
        config = _add_metric(config, metric_name, "0.01")
    config = config.end_panel()

    config = config.add_panel(name="skill_usage", name_en="skill_usage", type="line")
    for metric_name in (
        "skill_1_used_count",
        "skill_2_used_count",
        "skill_3_used_count",
        "summoner_skill_used_count",
    ):
        config = _add_metric(config, metric_name, "0.01")
    config = config.end_panel()

    config = config.add_panel(name="target_selection", name_en="target_selection", type="line")
    for metric_name in (
        "target_none_count",
        "target_enemy_hero_count",
        "target_self_hero_count",
        "target_enemy_soldier_count",
        "target_enemy_tower_count",
        "target_monster_count",
        "attack_target_enemy_hero_count",
        "attack_target_enemy_soldier_count",
        "skill_target_enemy_hero_count",
    ):
        config = _add_metric(config, metric_name, "0.01")
    config = config.end_panel()

    config = config.add_panel(name="rule_overview", name_en="rule_overview", type="line")
    for metric_name in (
        "rule_override_count",
    ):
        config = _add_metric(config, metric_name, "0.01")
    config = config.end_panel()

    config = config.add_panel(name="rule_force_home", name_en="rule_force_home", type="line")
    for metric_name in (
        "force_home_trigger_count",
        "force_home_override_count",
        "force_home_start_count",
        "force_home_retreat_count",
        "force_home_return_count",
        "force_home_no_enemy_minion_cnt",
        "force_home_own_wave_cnt",
    ):
        config = _add_metric(config, metric_name, "0.01")
    config = config.end_panel()

    config = config.add_panel(name="rule_opening", name_en="rule_opening", type="line")
    for metric_name in (
        "opening_unstuck_count",
        "opening_wave_guard_exit_enemy_atk_cnt",
        "opening_air_attack_count",
        "opening_air_attack_skip_count",
        "opening_air_hold_count",
        "opening_berserk_trigger_count",
        "opening_berserk_skip_count",
        "opening_enemy_hero_attack_count",
        "opening_enemy_hero_attack_skip_count",
        "opening_direnjie_skill1_count",
        "opening_direnjie_skill1_skip_count",
        "opening_luban_sweep_attack_count",
        "opening_luban_sweep_hold_count",
    ):
        config = _add_metric(config, metric_name, "0.01")
    config = config.end_panel()

    config = config.add_panel(name="rule_direnjie_skill2", name_en="rule_direnjie_skill2", type="line")
    for metric_name in (
        "cleanse_override_count",
        "skill2_blocked_count",
        "skill2_total_cast_count",
        "skill2_cleanse_rate",
    ):
        config = _add_metric(config, metric_name, "0.01")
    config = config.end_panel()

    config = config.add_panel(name="rule_luban_skill1", name_en="rule_luban_skill1", type="line")
    for metric_name in (
        "luban_skill1_aim_assist_count",
    ):
        config = _add_metric(config, metric_name, "0.01")
    config = config.end_panel()

    config = config.add_panel(name="summoner_duel", name_en="summoner_duel", type="line")
    for metric_name, precision in (
        ("duel_summoner_cast_count", "0.01"),
        ("duel_summoner_good_count", "0.01"),
        ("duel_summoner_mid_count", "0.01"),
        ("duel_summoner_touch_count", "0.01"),
        ("duel_summoner_poke_count", "0.01"),
        ("duel_summoner_wasted_count", "0.01"),
        ("duel_summoner_80110_cast_count", "0.01"),
        ("duel_summoner_80110_good_count", "0.01"),
    ):
        config = _add_metric(config, metric_name, precision)
    config = config.end_panel()

    config = config.add_panel(name="recover_cake", name_en="recover_cake", type="line")
    for metric_name, precision in (
        ("reward_cake_pickup", "0.0001"),
        ("reward_recover_skill_low_hp", "0.0001"),
        ("cake_success_count", "0.01"),
        ("cake_interrupted_count", "0.01"),
        ("cake_wasted_count", "0.01"),
        ("cake_high_hp_penalty_count", "0.01"),
        ("recover_attempt_count", "0.01"),
        ("recover_success_count", "0.01"),
        ("recover_interrupted_count", "0.01"),
        ("recover_high_hp_penalty_count", "0.01"),
    ):
        config = _add_metric(config, metric_name, precision)
    config = config.end_panel()

    return config.end_group().build()
