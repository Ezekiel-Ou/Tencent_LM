#!/usr/bin/env python3
"""Formal PPO training monitor panels."""

from kaiwudrl.common.monitor.monitor_config_builder import MonitorConfigBuilder

from agent_ppo.conf.conf import FrozenV1Config


def build_monitor():
    metrics = (
        ("reward", "回报"),
        ("total_loss", "总损失"),
        ("actor_policy_loss", "角色策略损失"),
        ("actor_value_loss", "角色价值损失"),
        ("actor_entropy", "角色熵"),
        ("actor_approx_kl", "角色近似 KL"),
        ("actor_clip_fraction", "角色裁剪比例"),
        ("actor_explained_variance", "角色价值解释方差"),
        ("command_policy_loss", "指令策略损失"),
        ("command_value_loss", "指令价值损失"),
        ("command_entropy", "指令熵"),
        ("command_approx_kl", "指令近似 KL"),
        ("command_clip_fraction", "指令裁剪比例"),
        ("command_explained_variance", "指令价值解释方差"),
        ("command_policy_valid_count", "有效指令策略步数"),
        ("command_unset_rate", "Unset 比例"),
        ("command_no_command_rate", "无指令比例"),
        ("gradient_norm", "裁剪前梯度范数"),
        ("hero_132_entropy", "马可波罗策略熵"),
        ("hero_141_entropy", "貂蝉策略熵"),
        ("hero_193_entropy", "凯策略熵"),
        ("hero_132_value_loss", "马可波罗价值损失"),
        ("hero_141_value_loss", "貂蝉价值损失"),
        ("hero_193_value_loss", "凯价值损失"),
        ("reward_hp_rate_sqrt_sqrt", "血量奖励"),
        ("reward_money", "经济奖励"),
        ("reward_exp", "经验奖励"),
        ("reward_tower", "防御塔奖励"),
        ("reward_killCnt", "击杀奖励"),
        ("reward_deadCnt", "死亡奖励"),
        ("reward_assistCnt", "助攻奖励"),
        ("reward_total_hurt_to_hero", "英雄伤害奖励"),
        ("reward_atk_monster", "野怪奖励"),
        ("reward_win_crystal", "摧毁水晶奖励"),
        ("reward_atk_crystal", "攻击水晶奖励"),
        ("platform_sample_production_and_consumption_ratio", "样本产消比"),
        ("platform_train_global_step", "平台训练步数"),
        ("platform_load_model_succ_cnt", "模型加载成功数"),
    )
    metrics += (("reward_jungle_route_shaping", "打野路线奖励"),)
    metrics += (
        ("reward_lane_last_hit", "线上补刀奖励"),
        ("lane_last_hit_count", "线上补刀次数"),
        ("reward_no_target_penalty", "无目标惩罚"),
        ("no_target_penalty_count", "无目标惩罚次数"),
        ("no_target_grace_count", "无目标宽限次数"),
        ("strategic_wait_exemption_count", "战略等待豁免次数"),
        ("reward_lane_attendance", "回线进度奖励"),
        ("lane_attendance_shaping_count", "回线进度计数"),
        ("lane_opening_mask_count", "开局上线约束次数"),
        ("lane_opening_release_count", "开局上线释放次数"),
        ("lane_controlled_alive_steps", "受控线上英雄存活观测数"),
        ("lane_controlled_on_lane_rate", "受控线上英雄在线率"),
        ("lane_controlled_near_wave_rate", "受控线上英雄近兵线率"),
        ("lane_controlled_eating_signal_rate", "受控线上英雄吃线信号率"),
        ("lane_controlled_opening_progress_mean", "受控线上英雄开局推进度"),
        ("lane_official_alive_steps", "官方线上英雄存活观测数"),
        ("lane_official_on_lane_rate", "官方线上英雄在线率"),
        ("lane_official_near_wave_rate", "官方线上英雄近兵线率"),
        ("lane_official_eating_signal_rate", "官方线上英雄吃线信号率"),
        ("lane_official_opening_progress_mean", "官方线上英雄开局推进度"),
        ("lane_hero_132_on_lane_rate", "马可波罗在线率"),
        ("lane_hero_132_near_wave_rate", "马可波罗近兵线率"),
        ("lane_hero_132_eating_signal_rate", "马可波罗吃线信号率"),
        ("lane_hero_141_on_lane_rate", "貂蝉在线率"),
        ("lane_hero_141_near_wave_rate", "貂蝉近兵线率"),
        ("lane_hero_141_eating_signal_rate", "貂蝉吃线信号率"),
    )
    metrics += (
        ("collision_escape_count", "碰撞脱困次数"),
        ("collision_blocked_direction_mean", "碰撞阻塞方向"),
    )
    metrics += (
        ("lane_first_wave_attack_mask_count", "首波强制攻击"),
        ("lane_first_wave_clear_count", "首波清完释放"),
        ("lane_opening_emergency_release_count", "上线紧急释放"),
        ("jungle_opening_attack_mask_count", "上野强制攻击"),
        ("jungle_camp_exit_mask_count", "清野离场约束"),
    )
    auxiliary_titles = {
        "farming": "发育",
        "kda": "KDA",
        "damage": "伤害",
        "pushing": "推进",
        "win": "胜负",
    }
    metrics += (
        ("actor_auxiliary_value_loss", "辅助价值总损失"),
        ("actor_hybrid_component_reg_loss", "混合价值正则"),
        ("actor_hybrid_identity_error", "混合价值恒等误差"),
        ("actor_hybrid_cancellation_ratio", "混合价值抵消比"),
        ("actor_value_self_mean", "个人价值均值"),
        ("actor_value_team_mean", "团队价值均值"),
        ("actor_value_enemy_mean", "敌方价值均值"),
    )
    for category in FrozenV1Config.AUXILIARY_VALUE_CATEGORIES:
        title = auxiliary_titles[category]
        metrics += (
            (f"actor_aux_{category}_loss", f"{title}价值损失"),
            (
                f"actor_aux_{category}_explained_variance",
                f"{title}价值解释方差",
            ),
            (f"actor_aux_{category}_target_mean", f"{title}目标均值"),
            (f"actor_aux_{category}_target_std", f"{title}目标标准差"),
        )
    builder = (
        MonitorConfigBuilder()
        .title("3v3 Hybrid Multi-Value PPO")
        .add_group(group_name="训练指标", group_name_en="training")
    )
    for metric, title in metrics:
        builder = (
            builder.add_panel(name=title, name_en=metric, type="line")
            .add_metric(metrics_name=metric, expr=f"avg({metric}{{}})")
            .end_panel()
        )
    return builder.end_group().build()


__all__ = ("build_monitor",)
