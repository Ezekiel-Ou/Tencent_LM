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
        group_name="algorithm",
        group_name_en="algorithm",
    )

    config = (
        config.add_panel(name="reward", name_en="reward", type="line")
        .add_metric(metrics_name="reward", expr="round(avg(reward{}), 0.01)")
        .end_panel()
    )

    config = (
        config.add_panel(name="total_loss", name_en="total_loss", type="line")
        .add_metric(metrics_name="total_loss", expr="round(avg(total_loss{}), 0.01)")
        .end_panel()
    )

    config = (
        config.add_panel(name="value_loss", name_en="value_loss", type="line")
        .add_metric(metrics_name="value_loss", expr="round(avg(value_loss{}), 0.01)")
        .end_panel()
    )

    config = (
        config.add_panel(name="ppo_kl", name_en="ppo_kl", type="line")
        .add_metric(metrics_name="approx_kl", expr="round(avg(approx_kl{}), 0.000001)")
        .end_panel()
    )

    config = (
        config.add_panel(name="ppo_clip", name_en="ppo_clip", type="line")
        .add_metric(metrics_name="clip_fraction", expr="round(avg(clip_fraction{}), 0.0001)")
        .end_panel()
    )

    config = config.add_panel(name="objective_reward", name_en="objective_reward", type="line")
    for metric_name in ("reward_tower_hp_point", "reward_money"):
        config = _add_metric(config, metric_name, "0.0001")
    config = config.end_panel()

    config = config.add_panel(name="target_focus", name_en="target_focus", type="line")
    for metric_name in (
        "attack_target_enemy_hero_count",
        "attack_target_enemy_soldier_count",
        "skill_target_enemy_hero_count",
        "target_enemy_tower_count",
        "target_monster_count",
    ):
        config = _add_metric(config, metric_name, "0.01")
    config = config.end_panel()

    config = config.add_panel(name="rule_intervention", name_en="rule_intervention", type="line")
    for metric_name in (
        "rule_override_count",
        "force_home_trigger_count",
        "force_home_override_count",
        "force_home_start_count",
        "force_home_retreat_count",
        "force_home_return_count",
        "opening_unstuck_count",
        "cleanse_override_count",
        "skill2_blocked_count",
        "skill2_total_cast_count",
        "skill2_cast_outside_window_count",
        "skill2_cleanse_rate",
        "luban_skill1_aim_assist_count",
    ):
        config = _add_metric(config, metric_name, "0.01")
    config = config.end_panel()

    config = config.add_panel(name="new_reward_signals", name_en="new_reward_signals", type="line")
    for metric_name in (
        "reward_cleanse_success",
        "reward_cake_pickup",
        "reward_recover_skill_low_hp",
        "reward_safe_last_hit",
        "reward_minion_tower_push",
        "reward_enemy_dead_enemy_cake",
        "reward_win",
        "reward_berserk_timing",
        "reward_berserk_no_damage_penalty",
        "reward_no_op_streak_penalty",
    ):
        config = _add_metric(config, metric_name, "0.0001")
    config = config.end_panel()

    config = config.add_panel(name="cleanse_berserk", name_en="cleanse_berserk", type="line")
    for metric_name in (
        "enemy_cleansed_us_count",
        "berserk_total_cast_count",
        "berserk_no_damage_count",
    ):
        config = _add_metric(config, metric_name, "0.01")
    config = config.end_panel()

    config = config.add_panel(name="recover_debug", name_en="recover_debug", type="line")
    for metric_name in (
        "cake_high_hp_penalty_count",
        "recover_attempt_count",
        "recover_success_count",
        "recover_interrupted_count",
        "recover_high_hp_penalty_count",
    ):
        config = _add_metric(config, metric_name, "0.01")
    config = config.end_panel()

    config = config.add_panel(name="skill_hit_detail", name_en="skill_hit_detail", type="line")
    for metric_name in (
        "reward_luban_skill1_hit_enemy_soldier",
        "reward_luban_skill1_hit_enemy_hero",
        "reward_luban_skill2_hit_enemy_hero",
        "reward_luban_skill3_hit_enemy_hero",
        "reward_direnjie_skill1_hit_enemy_hero",
        "reward_direnjie_skill3_hit_enemy_hero",
        "reward_direnjie_skill3_followup_damage",
        "direnjie_skill3_followup_count",
    ):
        config = _add_metric(config, metric_name, "0.0001")
    config = config.end_panel()

    return config.end_group().build()
