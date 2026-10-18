#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Monitor panel config builder for Gorge Chase PPO (refactored).

峡谷追猎 PPO 监控面板配置构建器（重构版）：
  - 算法指标组：loss 系列 + approx_kl + clip_frac + beta
  - 得分指标组：total_score / treasures_collected / treasure_discover_count /
                flash_count / collected_buff / wall_bump_count / anti_stall_count /
                rule_override_count / curriculum_stage
  - 回合指标组：reward / episode_steps / episode_cnt
"""

from kaiwudrl.common.monitor.monitor_config_builder import MonitorConfigBuilder


def _add_line_panel(mb, name_cn, name_en, metric_name):
    return (
        mb.add_panel(name=name_cn, name_en=name_en, type="line")
        .add_metric(metrics_name=metric_name, expr=f"avg({metric_name}{{}})")
        .end_panel()
    )


def build_monitor():
    """Create the monitoring panel config for the Tencent KaiWu platform.

    创建监控面板配置（供腾讯开悟平台使用）。
    """
    mb = MonitorConfigBuilder()

    # ===== 算法指标 =====
    mb = mb.title("峡谷追猎").add_group(group_name="算法指标", group_name_en="algorithm")
    mb = _add_line_panel(mb, "累积回报", "reward", "reward")
    mb = _add_line_panel(mb, "总损失", "total_loss", "total_loss")
    mb = _add_line_panel(mb, "价值损失", "value_loss", "value_loss")
    mb = _add_line_panel(mb, "策略损失", "policy_loss", "policy_loss")
    mb = _add_line_panel(mb, "熵损失", "entropy_loss", "entropy_loss")
    mb = _add_line_panel(mb, "近似KL", "approx_kl", "approx_kl")
    mb = _add_line_panel(mb, "裁剪占比", "clip_frac", "clip_frac")
    mb = _add_line_panel(mb, "熵系数", "beta", "beta")
    mb = mb.end_group()

    # ===== 得分与行为指标 =====
    mb = mb.add_group(group_name="得分与行为", group_name_en="score_behavior")
    mb = _add_line_panel(mb, "总得分", "total_score", "total_score")
    mb = _add_line_panel(mb, "已拾宝箱", "treasures_collected", "treasures_collected")
    mb = _add_line_panel(mb, "发现宝箱次数", "treasure_discover_count", "treasure_discover_count")
    mb = _add_line_panel(mb, "闪现使用次数", "flash_count", "flash_count")
    mb = _add_line_panel(mb, "已拾buff", "collected_buff", "collected_buff")
    mb = _add_line_panel(mb, "撞墙步数", "wall_bump_count", "wall_bump_count")
    mb = _add_line_panel(mb, "磨蹭惩罚次数", "anti_stall_count", "anti_stall_count")
    mb = _add_line_panel(mb, "规则接管次数", "rule_override_count", "rule_override_count")
    mb = _add_line_panel(mb, "课程阶段", "curriculum_stage", "curriculum_stage")
    mb = mb.end_group()

    # ===== 回合指标 =====
    mb = mb.add_group(group_name="回合", group_name_en="episode")
    mb = _add_line_panel(mb, "单局步数", "episode_steps", "episode_steps")
    mb = _add_line_panel(mb, "对局数", "episode_cnt", "episode_cnt")
    mb = mb.end_group()

    return mb.build()
