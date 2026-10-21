#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""


from kaiwudrl.common.monitor.monitor_config_builder import MonitorConfigBuilder


def build_monitor():
    """
    # This function is used to create monitoring panel configurations for custom indicators.
    # 该函数用于创建自定义指标的监控面板配置。
    """
    monitor = MonitorConfigBuilder()

    config_dict = (
        monitor.title("智能决策1v1")
        .add_group(
            group_name="算法指标",
            group_name_en="algorithm",
        )
        .add_panel(
            name="累积回报",
            name_en="reward",
            type="line",
        )
        .add_metric(
            metrics_name="reward",
            expr="round(avg(reward{}), 0.01)",
        )
        .end_panel()
        .add_panel(
            name="总损失",
            name_en="total_loss",
            type="line",
        )
        .add_metric(
            metrics_name="total_loss",
            expr="round(avg(total_loss{}), 0.01)",
        )
        .end_panel()
        .add_panel(
            name="价值损失",
            name_en="value_loss",
            type="line",
        )
        .add_metric(
            metrics_name="value_loss",
            expr="round(avg(value_loss{}), 0.01)",
        )
        .end_panel()
        .add_panel(
            name="策略损失",
            name_en="policy_loss",
            type="line",
        )
        .add_metric(
            metrics_name="policy_loss",
            expr="round(avg(policy_loss{}), 0.01)",
        )
        .end_panel()
        .add_panel(
            name="熵损失",
            name_en="entropy_loss",
            type="line",
        )
        .add_metric(
            metrics_name="entropy_loss",
            expr="round(avg(entropy_loss{}), 0.01)",
        )
        .end_panel()
        .add_panel(
            name="passive_buff_debug",
            name_en="passive_buff_debug",
            type="line",
        )
        .add_metric(
            metrics_name="passive_buff_debug_hero_id",
            expr="round(avg(passive_buff_debug_hero_id{}), 1)",
        )
        .add_metric(
            metrics_name="passive_buff_debug_lineup_code",
            expr="round(avg(passive_buff_debug_lineup_code{}), 1)",
        )
        .add_metric(
            metrics_name="passive_buff_debug_side",
            expr="round(avg(passive_buff_debug_side{}), 1)",
        )
        .add_metric(
            metrics_name="passive_buff_debug_event_code",
            expr="round(avg(passive_buff_debug_event_code{}), 1)",
        )
        .add_metric(
            metrics_name="passive_buff_debug_attack_no",
            expr="round(avg(passive_buff_debug_attack_no{}), 1)",
        )
        .add_metric(
            metrics_name="passive_buff_debug_skill_count",
            expr="round(avg(passive_buff_debug_skill_count{}), 1)",
        )
        .add_metric(
            metrics_name="passive_buff_debug_mark_count",
            expr="round(avg(passive_buff_debug_mark_count{}), 1)",
        )
        .add_metric(
            metrics_name="passive_buff_debug_skill_id_0",
            expr="round(avg(passive_buff_debug_skill_id_0{}), 1)",
        )
        .add_metric(
            metrics_name="passive_buff_debug_skill_times_0",
            expr="round(avg(passive_buff_debug_skill_times_0{}), 1)",
        )
        .add_metric(
            metrics_name="passive_buff_debug_skill_id_1",
            expr="round(avg(passive_buff_debug_skill_id_1{}), 1)",
        )
        .add_metric(
            metrics_name="passive_buff_debug_skill_id_2",
            expr="round(avg(passive_buff_debug_skill_id_2{}), 1)",
        )
        .add_metric(
            metrics_name="passive_buff_debug_mark_id_0",
            expr="round(avg(passive_buff_debug_mark_id_0{}), 1)",
        )
        .add_metric(
            metrics_name="passive_buff_debug_mark_layer_0",
            expr="round(avg(passive_buff_debug_mark_layer_0{}), 1)",
        )
        .add_metric(
            metrics_name="passive_buff_debug_mark_id_1",
            expr="round(avg(passive_buff_debug_mark_id_1{}), 1)",
        )
        .end_panel()
        .end_group()
        .build()
    )
    return config_dict
