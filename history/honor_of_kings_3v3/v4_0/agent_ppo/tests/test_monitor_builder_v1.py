#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""Static checks for monitor labels without requiring the platform package."""

import ast
import importlib.util
from pathlib import Path
import re
import sys
import types
import unittest
from unittest import mock


class MonitorBuilderV1Test(unittest.TestCase):
    @staticmethod
    def _build_with_fake_platform():
        class FakeMonitorConfigBuilder:
            def __init__(self):
                self.groups = []
                self.current_group = None
                self.current_panel = None

            def title(self, _title):
                return self

            def add_group(self, group_name, group_name_en):
                self.current_group = {
                    "name": group_name,
                    "name_en": group_name_en,
                    "panels": [],
                }
                self.groups.append(self.current_group)
                return self

            def add_panel(self, name, name_en, type):
                self.current_panel = {
                    "name": name,
                    "name_en": name_en,
                    "type": type,
                    "metrics": [],
                }
                self.current_group["panels"].append(self.current_panel)
                return self

            def add_metric(self, metrics_name, expr):
                self.current_panel["metrics"].append(
                    {"name": metrics_name, "expr": expr}
                )
                return self

            def end_panel(self):
                self.current_panel = None
                return self

            def end_group(self):
                self.current_group = None
                return self

            def build(self):
                return self.groups

        builder_module = types.ModuleType(
            "kaiwudrl.common.monitor.monitor_config_builder"
        )
        builder_module.MonitorConfigBuilder = FakeMonitorConfigBuilder
        fake_modules = {
            "kaiwudrl": types.ModuleType("kaiwudrl"),
            "kaiwudrl.common": types.ModuleType("kaiwudrl.common"),
            "kaiwudrl.common.monitor": types.ModuleType("kaiwudrl.common.monitor"),
            "kaiwudrl.common.monitor.monitor_config_builder": builder_module,
        }
        path = (
            Path(__file__).resolve().parents[1] / "conf" / "monitor_builder.py"
        )
        spec = importlib.util.spec_from_file_location(
            "monitor_builder_under_test", path
        )
        module = importlib.util.module_from_spec(spec)
        with mock.patch.dict(sys.modules, fake_modules):
            spec.loader.exec_module(module)
            return module.build_monitor()

    def test_only_custom_kda_net_reward_panel_is_registered(self):
        source = (
            Path(__file__).resolve().parents[1] / "conf" / "monitor_builder.py"
        ).read_text(encoding="utf-8")

        for disabled_metric in (
            "reward_killCnt",
            "reward_deadCnt",
            "reward_assistCnt",
        ):
            self.assertNotIn(f'("{disabled_metric}",', source)
        self.assertIn(
            '("reward_custom_kda", "自定义 KDA 净奖励"),',
            source,
        )

    def test_disabled_official_structure_panels_are_replaced_by_custom_ones(self):
        source = (
            Path(__file__).resolve().parents[1] / "conf" / "monitor_builder.py"
        ).read_text(encoding="utf-8")
        self.assertNotIn('("reward_tower",', source)
        self.assertNotIn('("reward_win_crystal",', source)
        self.assertIn('("reward_structure_nonlinear",', source)
        self.assertIn('("reward_terminal_crystal_custom",', source)

    def test_terminal_custom_reward_is_consumed_before_monitor_reporting(self):
        source = (
            Path(__file__).resolve().parents[1]
            / "workflow"
            / "train_workflow.py"
        ).read_text(encoding="utf-8")
        consume_index = source.index("terminal_payloads = {}")
        report_index = source.index("monitor_data = {", consume_index)
        self.assertLess(consume_index, report_index)
        self.assertIn(
            "terminal_payload=terminal_payloads[i]", source[report_index:]
        )
        self.assertIn(
            'if name not in ("tower", "win_crystal")',
            source[report_index:],
        )

    def test_new_lane_metric_labels_follow_platform_contract(self):
        source = (
            Path(__file__).resolve().parents[1] / "conf" / "monitor_builder.py"
        ).read_text(encoding="utf-8")

        for key in (
            "lane_first_wave_attack_mask_count",
            "lane_first_wave_clear_count",
            "lane_opening_emergency_release_count",
            "jungle_opening_attack_mask_count",
            "jungle_camp_exit_mask_count",
            "reward_atk_monster_132",
            "reward_atk_monster_141",
            "reward_atk_monster_193",
            "reward_kai_wolf",
            "kai_wolf_reward_count",
            "reward_treasure_thief",
            "treasure_thief_reward_count",
            "reward_speed_buff",
            "speed_buff_reward_count",
            "tower_risk_warning_count",
            "tower_risk_danger_count",
            "tower_risk_critical_count",
            "defense_wave_outer_count",
            "defense_wave_high_count",
            "defense_wave_crystal_count",
            "tower_critical_displacement_block_count",
            "defense_wave_missing_count",
            "defense_wave_missing_peak_frames",
            "defense_wave_invalidated_count",
            "defense_wave_runtime_reuse_count",
            "defense_wave_teammate_progress_count",
            "official_task_arrival_count",
            "official_task_navigation_latency_frames",
            "official_task_engagement_latency_frames",
        ):
            self.assertIn(f'"{key}"', source)
        workflow_source = (
            Path(__file__).resolve().parents[1]
            / "workflow"
            / "train_workflow.py"
        ).read_text(encoding="utf-8")
        for key in (
            "tower_risk_warning_count",
            "tower_risk_danger_count",
            "tower_risk_critical_count",
            "defense_wave_outer_count",
            "defense_wave_high_count",
            "defense_wave_crystal_count",
            "tower_critical_displacement_block_count",
            "defense_wave_missing_count",
            "defense_wave_missing_peak_frames",
            "defense_wave_invalidated_count",
            "defense_wave_runtime_reuse_count",
            "defense_wave_teammate_progress_count",
            "official_task_arrival_count",
            "official_task_navigation_latency_frames",
            "official_task_engagement_latency_frames",
        ):
            self.assertIn(f'"{key}"', workflow_source)

    def test_every_literal_panel_label_follows_platform_contract(self):
        source = (
            Path(__file__).resolve().parents[1] / "conf" / "monitor_builder.py"
        ).read_text(encoding="utf-8")
        tree = ast.parse(source)
        allowed = re.compile(r"^[\u4e00-\u9fffA-Za-z0-9* -]{1,20}$")
        labels = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Tuple):
                continue
            if len(node.elts) == 2:
                metric, label = node.elts
                if (
                    isinstance(metric, ast.Constant)
                    and isinstance(metric.value, str)
                    and isinstance(label, ast.Constant)
                    and isinstance(label.value, str)
                    and "_" not in label.value
                ):
                    labels.append((metric.value, label.value))
            elif (
                len(node.elts) == 3
                and isinstance(node.elts[0], ast.Constant)
                and isinstance(node.elts[0].value, str)
                and isinstance(node.elts[1], ast.Constant)
                and isinstance(node.elts[1].value, str)
                and isinstance(node.elts[2], ast.Tuple)
            ):
                labels.append((node.elts[1].value, node.elts[0].value))
        self.assertTrue(labels)
        for metric, label in labels:
            self.assertRegex(label, allowed, msg=f"invalid panel label for {metric}")

    def test_temporary_performance_timing_panels_are_removed(self):
        source = (
            Path(__file__).resolve().parents[1] / "conf" / "monitor_builder.py"
        ).read_text(encoding="utf-8")
        for key in (
            "learn_total_ms",
            "batch_statistics_ms",
            "format_h2d_ms",
            "forward_backward_ms",
            "metric_sync_ms",
            "optimizer_ms",
            "framework_gap_ms",
            "save_model_ms",
            "predict_total_ms",
            "observation_process_ms",
            "inference_prepare_ms",
            "inference_forward_ms",
            "inference_decode_ms",
            "action_process_ms",
            "bootstrap_values_ms",
            "load_model_ms",
            "env_reset_ms",
            "env_step_ms",
            "collector_save_frame_ms",
            "sample_pack_ms",
            "sample_send_ms",
            "samples_per_flush",
        ):
            self.assertNotIn(f'(\"{key}\",', source)
        self.assertNotIn('metrics_name=f"{metric}_p95"', source)
        self.assertNotIn('group_name="性能诊断"', source)

    def test_monitor_is_split_into_core_and_diagnostic_groups(self):
        source = (
            Path(__file__).resolve().parents[1] / "conf" / "monitor_builder.py"
        ).read_text(encoding="utf-8")
        self.assertIn(
            '("核心指标", "core", core_metrics, core_grouped_panels)',
            source,
        )
        self.assertIn('"诊断指标",', source)
        self.assertIn("diagnostic_grouped_panels", source)

    def test_disabled_training_evaluation_panels_are_removed(self):
        source = (
            Path(__file__).resolve().parents[1] / "conf" / "monitor_builder.py"
        ).read_text(encoding="utf-8")
        self.assertNotIn('"eval_commonai_episode_count"', source)
        self.assertNotIn('"eval_anchor_episode_count"', source)
        self.assertIn(
            '("opponent_anchor_episode_ratio", "固定锚点训练局比例"),',
            source,
        )

    def test_related_diagnostics_are_grouped_without_dropping_metrics(self):
        source = (
            Path(__file__).resolve().parents[1] / "conf" / "monitor_builder.py"
        ).read_text(encoding="utf-8")
        for panel_name in (
            "资源目标奖励",
            "辅助价值损失",
            "辅助价值解释方差",
            "资源目标次数",
            "分英雄策略熵",
            "分英雄价值损失",
            "闪现门控状态",
            "闪现触发原因",
            "兵线开局规则",
            "打野开局规则",
            "混合价值分量",
        ):
            self.assertIn(f'"{panel_name}"', source)
        for metric in (
            "reward_kai_wolf",
            "reward_treasure_thief",
            "reward_speed_buff",
            "flash_official_legal_count",
            "flash_gate_open_count",
            "flash_blocked_count",
            "flash_no_safe_landing_count",
            "flash_trigger_low_hp_count",
            "flash_trigger_tower_count",
            "flash_trigger_chase_count",
            "flash_trigger_marco_ultimate_count",
        ):
            self.assertIn(f'"{metric}"', source)

    def test_current_panel_and_metric_counts(self):
        groups = self._build_with_fake_platform()
        self.assertEqual(
            [(group["name_en"], len(group["panels"])) for group in groups],
            [("core", 70), ("diagnostic", 84)],
        )
        panels = [
            panel
            for group in groups
            for panel in group["panels"]
        ]
        self.assertEqual(len(panels), 154)
        allowed = re.compile(r"^[\u4e00-\u9fffA-Za-z0-9* -]{1,20}$")
        for panel in panels:
            self.assertRegex(
                panel["name"],
                allowed,
                msg=f'invalid panel label for {panel["name_en"]}',
            )
        metrics = [
            metric["name"]
            for panel in panels
            for metric in panel["metrics"]
        ]
        self.assertEqual(len(metrics), 374)
        self.assertEqual(len(metrics), len(set(metrics)))
        self.assertTrue(all(1 <= len(metric) <= 40 for metric in metrics))
        self.assertLessEqual(
            max(len(panel["metrics"]) for panel in panels),
            20,
        )

    def test_command_legality_and_policy_lag_panels_are_registered(self):
        source = (
            Path(__file__).resolve().parents[1] / "conf" / "monitor_builder.py"
        ).read_text(encoding="utf-8")
        for metric in (
            "command_legalizer_invalid_count",
            "command_legalizer_valid_rate",
            "actor_policy_lag_updates",
            "reward_command_response",
            "reward_command_nonresponse",
            "reward_command_idle",
            "command_nonresponse_count",
            "command_idle_penalty_count",
            "reward_crystal_danger",
            "reward_defense_pressure",
            "reward_temporal_focus",
            "reward_teammate_abandonment",
            "reward_teammate_support",
            "reward_group_attack",
            "teammate_abandonment_count",
            "teammate_support_count",
            "temporal_focus_count",
            "defense_pressure_count",
            "group_attack_count",
            "command_teammate_abandonment_count",
            "command_teammate_support_count",
        ):
            self.assertIn(f'("{metric}",', source)

        agent_source = (
            Path(__file__).resolve().parents[1] / "agent.py"
        ).read_text(encoding="utf-8")
        self.assertIn(
            "self.last_command_goal_progress_reward\n"
            "            - self.last_command_nonresponse_penalty",
            agent_source,
        )

        workflow_source = (
            Path(__file__).resolve().parents[1]
            / "workflow"
            / "train_workflow.py"
        ).read_text(encoding="utf-8")
        self.assertIn('"command_legalizer_attempt_count"', workflow_source)
        self.assertIn('"invalid_command_count"', workflow_source)
        self.assertIn('"platform_train_global_step"', workflow_source)
        self.assertIn('monitor_data["actor_policy_lag_updates"]', workflow_source)

    def test_archived_lane_v3_and_redundant_panels_are_removed(self):
        source = (
            Path(__file__).resolve().parents[1] / "conf" / "monitor_builder.py"
        ).read_text(encoding="utf-8")
        removed_metrics = (
            "total_loss",
            "reward_atk_monster",
            "platform_sample_production_and_consumption_ratio",
            "platform_train_global_step",
            "platform_load_model_succ_cnt",
            "opponent_selfplay_episode_count",
            "opponent_anchor_episode_count",
            "opponent_current_is_anchor",
            "eval_current_is_anchor",
            "joint_inference_attempt_count",
            "joint_inference_forward_count",
            "joint_inference_side_count",
            "joint_inference_fallback_count",
            "joint_inference_coverage",
            "lane_last_hit_count",
            "no_target_penalty_count",
            "lane_attendance_shaping_count",
            "actor_hybrid_identity_error",
            "actor_hybrid_component_reg_loss",
            "lane_controlled_alive_steps",
            "lane_controlled_on_lane_rate",
            "lane_controlled_near_wave_rate",
            "lane_controlled_eating_signal_rate",
            "lane_controlled_opening_progress_mean",
            "lane_official_alive_steps",
            "lane_official_on_lane_rate",
            "lane_official_near_wave_rate",
            "lane_official_eating_signal_rate",
            "lane_official_opening_progress_mean",
            "lane_hero_132_on_lane_rate",
            "lane_hero_132_near_wave_rate",
            "lane_hero_132_eating_signal_rate",
            "lane_hero_141_on_lane_rate",
            "lane_hero_141_near_wave_rate",
            "lane_hero_141_eating_signal_rate",
        )
        for metric in removed_metrics:
            self.assertNotIn(f'("{metric}",', source)
        for category in ("farming", "kda", "damage", "pushing", "win"):
            self.assertNotIn(f"actor_aux_{category}_target_mean", source)
            self.assertNotIn(f"actor_aux_{category}_target_std", source)


if __name__ == "__main__":
    unittest.main()
