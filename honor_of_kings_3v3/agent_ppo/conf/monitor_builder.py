#!/usr/bin/env python3
"""Formal PPO training monitor panels."""

from kaiwudrl.common.monitor.monitor_config_builder import MonitorConfigBuilder

from agent_ppo.conf.conf import FrozenV1Config


CAKE_HERO_METRICS = tuple(
    f"cake_{reward_kind}_{hero_id}"
    for hero_id in (132, 141, 193)
    for reward_kind in ("reward", "progress", "pickup", "clawback")
)

LANE_PROMOTION_BUCKET_PANELS = tuple(
    (
        f"线上{kind_label}分桶",
        f"lane_{kind_name}_buckets",
        tuple(
            f"lane_{event_name}_c{camp}_h{hero_id}_{kind_name}"
            for camp in (1, 2)
            for hero_id in (132, 141)
            for event_name in ("cand", "stable", "sent", "confirm")
        ),
    )
    for kind_name, kind_label in (
        ("hero", "英雄目标"),
        ("soldier", "士兵目标"),
        ("structure", "结构目标"),
    )
)


def build_monitor():
    core_metrics = (
        ("reward", "回报"),
        ("actor_policy_loss", "角色策略损失"),
        ("actor_value_loss", "角色价值损失"),
        ("actor_entropy", "角色熵"),
        ("actor_approx_kl", "角色近似 KL"),
        ("actor_clip_fraction", "角色裁剪比例"),
        ("actor_dual_clip_fraction", "角色 Dual-Clip 触发比例"),
        ("actor_explained_variance", "角色价值解释方差"),
        ("command_policy_loss", "指令策略损失"),
        ("command_value_loss", "指令价值损失"),
        ("command_entropy", "指令熵"),
        ("command_approx_kl", "指令近似 KL"),
        ("command_clip_fraction", "指令裁剪比例"),
        ("command_dual_clip_fraction", "指令 Dual-Clip 触发比例"),
        ("command_explained_variance", "指令价值解释方差"),
        ("command_policy_valid_count", "有效指令策略步数"),
        ("command_unset_rate", "Unset 比例"),
        ("command_no_command_rate", "无指令比例"),
        ("command_legalizer_invalid_count", "协作指令非法次数"),
        ("command_legalizer_valid_rate", "协作指令合法率"),
        ("actor_policy_lag_updates", "角色策略滞后步数"),
        ("gradient_norm", "裁剪前梯度范数"),
        ("reward_hp_rate_sqrt_sqrt", "血量奖励"),
        ("reward_money", "经济奖励"),
        ("reward_exp", "经验奖励"),
        ("reward_total_hurt_to_hero", "英雄伤害奖励"),
        ("reward_atk_monster_132", "马可野怪奖励"),
        ("reward_atk_monster_141", "貂蝉野怪奖励"),
        ("reward_atk_monster_193", "铠野怪奖励"),
        ("reward_cake_total", "血包学习奖励"),
        ("reward_cake_progress", "血包接近奖励"),
        ("reward_cake_pickup", "血包拾取奖励"),
        ("reward_cake_clawback", "血包回收奖励"),
        ("late_jungle_reward_raw", "后期野怪原始奖励"),
        ("late_jungle_reward_offset", "后期野怪抵消奖励"),
        ("late_jungle_reward_effective", "后期野怪有效奖励"),
        ("reward_atk_crystal", "攻击水晶奖励"),
        ("reward_jungle_route_shaping", "打野路线奖励"),
        ("reward_lane_last_hit", "线上补刀奖励"),
        ("reward_no_target_penalty", "无目标惩罚"),
        ("reward_lane_attendance", "回线进度奖励"),
        ("reward_custom_kda", "自定义 KDA 净奖励"),
        ("reward_tyrant", "暴君奖励"),
        ("reward_structure_nonlinear", "自定义非线性防御塔奖励"),
        ("reward_post_kill_push", "击杀后推进窗口奖励"),
        ("reward_terminal_crystal_custom", "自定义终局水晶奖励"),
        ("reward_crystal_danger", "水晶危险奖励"),
        ("reward_defense_pressure", "防守压力奖励"),
        ("reward_temporal_focus", "时序集火奖励"),
        ("reward_teammate_abandonment", "队友未支援惩罚"),
        ("reward_teammate_support", "队友支援奖励"),
        ("reward_group_attack", "三人集火奖励"),
        ("reward_long_inactivity", "长期无贡献惩罚"),
        ("long_inactivity_penalty_count", "长期无贡献惩罚次数"),
        ("reward_command_sparse", "指令稀疏回报"),
        ("reward_command_response", "指令响应回报"),
        ("reward_command_nonresponse", "指令无响应惩罚"),
        ("reward_command_idle", "空指令无效惩罚"),
        ("reward_command_farm_camp", "刷野任务奖励"),
        ("reward_tower_risk", "越塔风险奖励"),
        ("reward_defense_wave", "防守兵线奖励"),
        ("structure_positive_reward", "敌塔推进奖励"),
        ("structure_negative_reward", "己塔损失奖励"),
        ("structure_actor_reward_132", "马可推塔归因奖励"),
        ("structure_actor_reward_141", "貂蝉推塔归因奖励"),
        ("structure_actor_reward_193", "铠推塔归因奖励"),
        ("opponent_anchor_episode_ratio", "固定锚点训练局比例"),
    )
    core_grouped_panels = (
        (
            "资源目标奖励",
            "resource_objective_rewards",
            (
                "reward_kai_wolf",
                "reward_treasure_thief",
                "reward_speed_buff",
            ),
        ),
        (
            "辅助价值损失",
            "auxiliary_value_losses",
            (
                "actor_auxiliary_value_loss",
                *(
                    f"actor_aux_{category}_loss"
                    for category in FrozenV1Config.AUXILIARY_VALUE_CATEGORIES
                ),
            ),
        ),
        (
            "辅助价值解释方差",
            "auxiliary_value_explained_variance",
            tuple(
                f"actor_aux_{category}_explained_variance"
                for category in FrozenV1Config.AUXILIARY_VALUE_CATEGORIES
            ),
        ),
    )

    diagnostic_metrics = (
        ("no_target_grace_count", "无目标宽限次数"),
        ("strategic_wait_exemption_count", "战略等待豁免次数"),
        ("long_inactivity_stationary_count", "长期不动命中次数"),
        ("long_inactivity_wandering_count", "长期徘徊命中次数"),
        ("long_inactivity_no_damage_count", "长期无伤害命中次数"),
        ("long_inactivity_tyrant_exemption_count", "暴君刷新等待豁免次数"),
        ("command_nonresponse_count", "指令无响应次数"),
        ("command_idle_penalty_count", "空指令无效次数"),
        ("teammate_abandonment_count", "队友未支援次数"),
        ("teammate_support_count", "队友支援次数"),
        ("temporal_focus_count", "时序集火次数"),
        ("defense_pressure_count", "防守压力次数"),
        ("group_attack_count", "三人集火次数"),
        ("command_teammate_abandonment_count", "指令未支援次数"),
        ("command_teammate_support_count", "指令支援次数"),
        ("collision_escape_count", "碰撞脱困次数"),
        ("collision_blocked_direction_mean", "碰撞阻塞方向"),
        ("actor_hybrid_cancellation_ratio", "混合价值抵消比"),
        ("tower_risk_hit_count", "确认塔伤次数"),
        ("tower_risk_warning_frames", "越塔警告帧"),
        ("tower_risk_danger_frames", "越塔危险帧"),
        ("tower_risk_critical_frames", "越塔危急帧"),
        ("tower_risk_reset_count", "越塔状态重置"),
        ("defense_wave_completion_count", "防守清线完成"),
        ("defense_wave_under_count", "塔下威胁窗口"),
        ("defense_wave_front_count", "塔前威胁窗口"),
        ("defense_wave_response_count", "清线首次响应"),
        ("defense_wave_progress_total", "清线归因进度"),
        ("defense_wave_ignore_candidate_count", "忽视清线候选"),
        ("defense_wave_ignore_penalty_count", "忽视清线惩罚"),
        ("defense_wave_ignore_exemption_count", "忽视清线豁免"),
        ("structure_attribution_coverage", "塔伤归因覆盖率"),
        ("structure_attribution_fallback_count", "塔伤回退次数"),
        ("structure_attributed_hit_count", "确认推塔命中"),
        ("structure_actor_hit_132", "马可推塔命中"),
        ("structure_actor_hit_141", "貂蝉推塔命中"),
        ("structure_actor_hit_193", "铠推塔命中"),
        ("spawn_boundary_move_block_count", "出生区移动拦截"),
        ("spawn_boundary_target_block_count", "出生区目标拦截"),
        ("spawn_boundary_displacement_block_count", "出生区位移拦截"),
        ("spawn_boundary_recovery_count", "出生区恢复次数"),
        ("tower_critical_move_block_count", "危急越塔拦截"),
        ("tower_critical_displacement_block_count", "危急位移拦截"),
        ("safety_all_illegal_recovery_count", "安全动作恢复"),
        ("official_task_start_count", "官方任务开始"),
        ("official_task_navigation_count", "官方导航响应"),
        ("official_task_engagement_count", "官方接战响应"),
        ("official_task_completion_count", "官方任务完成"),
        ("official_task_abort_count", "官方任务中止"),
        ("official_task_recovery_count", "官方任务恢复"),
        ("official_task_protocol_rejected_count", "官方协议拒绝"),
        ("official_task_navigation_failure_count", "官方导航失败"),
        ("official_task_engagement_failure_count", "官方接战失败"),
        ("official_task_persistence_failure_count", "官方持续失败"),
        ("official_task_lifecycle_lock_count", "官方生命周期锁"),
        ("official_deer_command_count", "官方鹿区指令"),
        ("official_deer_renewal_count", "官方鹿区续发"),
        ("official_deer_successor_count", "官方回线后继"),
    )
    diagnostic_grouped_panels = (
        (
            "血包学习状态",
            "cake_learning_state",
            (
                "cake_opportunity_count",
                "cake_opportunity_assigned_count",
                "cake_opportunity_conflict_count",
                "cake_progress_positive_total",
                "cake_progress_cap_count",
                "cake_progress_clawback_total",
                "cake_progress_clawback_count",
                "cake_opportunity_reopen_blocked_count",
                "cake_pickup_success_count",
                "cake_pickup_ambiguous_count",
                "cake_pickup_official_count",
                "cake_pickup_high_hp_zero_count",
                "cake_pickup_reward_total",
                "cake_abort_hard_count",
                "cake_abort_external_count",
                "cake_abort_controlled_count",
                "cake_abort_timeout_count",
                "cake_low_hp_death_count",
            ),
        ),
        (
            "分英雄血包奖励",
            "cake_rewards_by_hero",
            CAKE_HERO_METRICS,
        ),
        (
            "复活再入状态",
            "respawn_reentry_state",
            (
                "kai_respawn_count",
                "kai_respawn_reentry_started_count",
                "kai_respawn_reentry_success_count",
                "kai_respawn_reentry_abort_count",
                "kai_respawn_empty_camp_visit_count",
                "kai_respawn_lane_diversion_count",
                "kai_respawn_replan_count",
                "kai_respawn_reentry_timeout_count",
                "official_respawn_command_count",
                "official_respawn_engagement_count",
            ),
        ),
        (
            "复活再入耗时",
            "respawn_reentry_timing",
            (
                "kai_respawn_to_jungle_frames",
                "kai_respawn_to_jungle_observed_count",
                "kai_respawn_to_jungle_missing_count",
                "kai_respawn_to_monster_frames",
                "kai_respawn_to_monster_observed_count",
                "kai_respawn_to_monster_missing_count",
                "kai_respawn_empty_wait_frames",
                "kai_respawn_empty_wait_count",
                "kai_respawn_empty_wait_success_frames",
                "kai_respawn_empty_wait_success_count",
                "kai_respawn_empty_wait_timeout_frames",
                "kai_respawn_empty_wait_timeout_count",
                "kai_respawn_empty_wait_cancel_frames",
                "kai_respawn_empty_wait_cancel_count",
            ),
        ),
        (
            "受控铠复活再入",
            "controlled_respawn_reentry",
            (
                "kai_respawn_controlled_started_count",
                "kai_respawn_controlled_success_count",
                "kai_respawn_controlled_timeout_count",
                "kai_respawn_controlled_cancel_count",
                "kai_respawn_controlled_to_jungle_frames",
                "kai_ctl_to_jungle_observed_count",
                "kai_ctl_to_jungle_missing_count",
                "kai_ctl_to_monster_frames",
                "kai_ctl_to_monster_observed_count",
                "kai_ctl_to_monster_missing_count",
                "kai_respawn_controlled_empty_wait_frames",
                "kai_respawn_controlled_empty_wait_count",
                "kai_ctl_empty_wait_success_frames",
                "kai_ctl_empty_wait_success_count",
                "kai_ctl_empty_wait_timeout_frames",
                "kai_ctl_empty_wait_timeout_count",
                "kai_ctl_empty_wait_cancel_frames",
                "kai_ctl_empty_wait_cancel_count",
            ),
        ),
        (
            "官方铠复活再入",
            "official_respawn_reentry",
            (
                "kai_respawn_official_started_count",
                "kai_respawn_official_success_count",
                "kai_respawn_official_timeout_count",
                "kai_respawn_official_cancel_count",
                "kai_respawn_official_to_jungle_frames",
                "kai_off_to_jungle_observed_count",
                "kai_off_to_jungle_missing_count",
                "kai_off_to_monster_frames",
                "kai_off_to_monster_observed_count",
                "kai_off_to_monster_missing_count",
                "kai_respawn_official_empty_wait_frames",
                "kai_respawn_official_empty_wait_count",
                "kai_off_empty_wait_success_frames",
                "kai_off_empty_wait_success_count",
                "kai_off_empty_wait_timeout_frames",
                "kai_off_empty_wait_timeout_count",
                "kai_off_empty_wait_cancel_frames",
                "kai_off_empty_wait_cancel_count",
            ),
        ),
        (
            "计时野区轮转",
            "timer_jungle_rotation",
            (
                "timer_route_candidate_count",
                "timer_route_follow_count",
                "timer_route_progress_positive_total",
                "timer_route_progress_negative_total",
                "timer_route_phi_upper_clamp_count",
                "timer_route_phi_lower_clamp_count",
                "timer_route_terminal_settlement_count",
                "timer_route_terminal_settlement_total",
                "timer_route_no_target_exempt_frames",
                "timer_route_inactivity_exempt_frames",
                "timer_route_stall_count",
                "timer_route_real_engagement_count",
                "timer_route_clear_count",
                "timer_route_abandon_count",
                "official_timer_farm_command_count",
                "official_timer_farm_engagement_count",
                "official_timer_farm_completion_count",
                "navigation_generation_advance_count",
                "navigation_stale_lock_block_count",
                "respawn_conversion_preempt_count",
            ),
        ),
        (
            "后期转化状态",
            "late_conversion_state",
            (
                "tyrant_three_stack_first_frame",
                "post_three_stack_tyrant_count",
                "late_jungle_attack_frames",
                "late_farm_task_start_count",
                "late_farm_task_cancel_count",
                "late_conversion_opportunity_count",
                "late_conversion_follow_count",
                "late_conversion_preempt_count",
                "late_wave_escort_count",
                "late_group_count",
                "post_kill_conversion_count",
                "timeout_pressure_push_count",
                "episode_timeout_count",
                "episode_timeout_advantage_count",
            ),
        ),
        (
            "惩击门控状态",
            "smite_gate_state",
            (
                "tyrant_smite_lock_count",
                "tyrant_smite_open_count",
                "tyrant_smite_force_count",
                "tyrant_smite_pending_count",
            ),
        ),
        (
            "线上目标化次数",
            "lane_target_promotion",
            (
                "lane_position_candidate_count",
                "lane_stable_target_count",
                "lane_promotion_sent_count",
                "lane_promotion_confirmed_count",
                "lane_promotion_confirm_rate",
                "lane_cooldown_wait_frames",
                "lane_target_change_cancel_count",
                "lane_target_invalid_cancel_count",
                "lane_failure_lock_count",
                "lane_target_active_frames",
                "lane_refresh_suppressed_count",
                "lane_retreat_handoff_count",
            ),
        ),
        (
            "线上撤退保护",
            "lane_retreat_protection",
            (
                "lane_retreat_protection_count",
                "lane_low_hp_protection_count",
                "lane_outnumbered_protection_count",
                "lane_geometry_protection_count",
                "lane_defense_protection_count",
                "lane_forced_route_protection_count",
            ),
        ),
        *LANE_PROMOTION_BUCKET_PANELS,
        (
            "越塔阶段进入",
            "tower_risk_phase_entries",
            (
                "tower_risk_warning_count",
                "tower_risk_danger_count",
                "tower_risk_critical_count",
            ),
        ),
        (
            "防守结构窗口",
            "defense_wave_structure_counts",
            (
                "defense_wave_outer_count",
                "defense_wave_high_count",
                "defense_wave_crystal_count",
            ),
        ),
        (
            "防守账本状态",
            "defense_wave_ledger_state",
            (
                "defense_wave_missing_count",
                "defense_wave_missing_peak_frames",
                "defense_wave_invalidated_count",
                "defense_wave_runtime_reuse_count",
                "defense_wave_teammate_progress_count",
            ),
        ),
        (
            "官方任务阶段",
            "official_task_phase_timing",
            (
                "official_task_arrival_count",
                "official_task_navigation_latency_frames",
                "official_task_engagement_latency_frames",
            ),
        ),
        (
            "资源目标次数",
            "resource_objective_counts",
            (
                "kai_wolf_reward_count",
                "treasure_thief_reward_count",
                "speed_buff_reward_count",
            ),
        ),
        (
            "分英雄策略熵",
            "hero_policy_entropy",
            (
                "hero_132_entropy",
                "hero_141_entropy",
                "hero_193_entropy",
            ),
        ),
        (
            "分英雄价值损失",
            "hero_value_losses",
            (
                "hero_132_value_loss",
                "hero_141_value_loss",
                "hero_193_value_loss",
            ),
        ),
        (
            "闪现门控状态",
            "flash_gate_state",
            (
                "flash_official_legal_count",
                "flash_gate_open_count",
                "flash_blocked_count",
                "flash_no_safe_landing_count",
            ),
        ),
        (
            "闪现触发原因",
            "flash_trigger_reasons",
            (
                "flash_trigger_low_hp_count",
                "flash_trigger_tower_count",
                "flash_trigger_chase_count",
                "flash_trigger_marco_ultimate_count",
            ),
        ),
        (
            "兵线开局规则",
            "lane_opening_rules",
            (
                "lane_opening_mask_count",
                "lane_opening_release_count",
                "lane_first_wave_attack_mask_count",
                "lane_first_wave_clear_count",
                "lane_opening_emergency_release_count",
            ),
        ),
        (
            "打野开局规则",
            "jungle_opening_rules",
            (
                "jungle_opening_attack_mask_count",
                "jungle_camp_exit_mask_count",
            ),
        ),
        (
            "混合价值分量",
            "hybrid_value_components",
            (
                "actor_value_self_mean",
                "actor_value_team_mean",
                "actor_value_enemy_mean",
            ),
        ),
    )

    builder = MonitorConfigBuilder().title("3v3 Hybrid Multi-Value PPO")
    for group_name, group_name_en, metrics, grouped_panels in (
        ("核心指标", "core", core_metrics, core_grouped_panels),
        (
            "诊断指标",
            "diagnostic",
            diagnostic_metrics,
            diagnostic_grouped_panels,
        ),
    ):
        builder = builder.add_group(
            group_name=group_name,
            group_name_en=group_name_en,
        )
        for metric, title in metrics:
            builder = (
                builder.add_panel(name=title, name_en=metric, type="line")
                .add_metric(metrics_name=metric, expr=f"avg({metric}{{}})")
                .end_panel()
            )
        for panel_name, panel_name_en, panel_metrics in grouped_panels:
            builder = builder.add_panel(
                name=panel_name,
                name_en=panel_name_en,
                type="line",
            )
            for metric in panel_metrics:
                builder = builder.add_metric(
                    metrics_name=metric,
                    expr=f"avg({metric}{{}})",
                )
            builder = builder.end_panel()
        builder = builder.end_group()
    return builder.build()


__all__ = ("build_monitor",)
