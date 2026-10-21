#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

from pathlib import Path
import tomllib
import unittest

from agent_diy.debug.official_jungle_probe import (
    OfficialJungleProbe,
    SCENARIO_BY_NAME,
    selected_scenarios,
)


def _actor(
    config_id,
    runtime_id,
    x,
    z,
    hp=5000.0,
    max_hp=5000.0,
    attack_target=0,
    actor_type="monster",
):
    return {
        "config_id": config_id,
        "runtime_id": runtime_id,
        "camp": 1 if config_id in (132, 141, 193) else 0,
        "hp": hp,
        "max_hp": max_hp,
        "attack_target": attack_target,
        "actor_type": actor_type,
        "camp_visible": [True, True],
        "collider": {"location": {"x": x, "z": z}},
    }


class _FeatureProcess:
    def __init__(
        self,
        official_config_id=132,
        frame=56,
        official_side=(-60000.0, 9885.0),
        collaboration=None,
        monster_position=(-25530.0, 23990.0),
        monster_hp=5000.0,
        official_attack_target=0,
    ):
        official_raw = (
            official_side[0],
            official_side[1] + 6115.0,
        )
        official = _actor(
            official_config_id,
            10,
            official_raw[0],
            official_raw[1],
            attack_target=official_attack_target,
            actor_type="hero",
        )
        monster = _actor(
            42,
            501,
            monster_position[0],
            monster_position[1],
            hp=monster_hp,
        )
        self.monster = monster
        self.monster_visible = True
        self.authoritative_cleared = False
        deer_anchor = (-25530.0, 23990.0)
        self.monster_camp_history = {
            deer_anchor: {
                "observed_runtime_ids": (501,),
                "alive_runtime_ids": (501,),
                "observer_near": False,
                "spawn_timer_ms": 0.0,
                "spawn_timer_positive": False,
                "timer_death_transition": False,
                "last_clear_evidence": "none",
                "last_clear_frame": None,
                "clear_count": 0,
            }
        }
        self._memo_context = {
            "frame_no": frame,
            "official_hero": {
                "actor": official,
                "side_position": official_side,
                "alive": True,
            },
            "collaboration": collaboration
            or {
                "active": False,
                "type": "none",
                "target_runtime": 0,
                "command_position": None,
                "cooldown_ready": True,
                "micro_cd": 100,
                "max_duration": 300,
                "platform_last_send_frame": 0,
            },
            "actor_by_runtime": {10: official, 501: monster},
            "lane_macro": {
                official_config_id: {
                    "command_active": False,
                }
            },
            "organs": [
                {"alive": True},
                {"alive": True},
                {"alive": True},
                {"alive": True},
                {"alive": True},
                {"alive": True},
            ],
        }

    def _opening_camp_status(self, context, camp_name, kai_position=None):
        if camp_name == "wolf":
            anchor = (-15000.0, 27000.0)
            side_anchor = (-15000.0, 20885.0)
        else:
            anchor = (-25530.0, 23990.0)
            side_anchor = (-25530.0, 17875.0)
        return {
            "name": camp_name,
            "anchor": anchor,
            "side_anchor": side_anchor,
            "alive_actors": (
                (self.monster,)
                if self.monster_visible and self.monster["hp"] > 0
                else ()
            ),
            "target_actor": (self.monster if self.monster["hp"] > 0 else None),
            "cleared": self.authoritative_cleared,
            "lifecycle_id": 1,
            "seen_count": 1,
            "expected_members": 1,
        }


def _observation(frame=56):
    return {
        "heroes": [
            {"config_id": 193, "runtime_id": 29},
            {"config_id": 141, "runtime_id": 30},
        ],
        "frame_state": {"frame_no": frame},
    }


class OfficialJungleProbeTest(unittest.TestCase):
    def test_default_matrix_and_explicit_selection(self):
        self.assertEqual(
            [item.name for item in selected_scenarios("")],
            [
                "deer_team_refresh_until_clear_marco",
                "deer_team_refresh_until_clear_diaochan",
                "deer_team_renew_on_expiry_until_clear_marco",
                "deer_team_renew_on_expiry_until_clear_diaochan",
            ],
        )
        for scenario in selected_scenarios(""):
            self.assertTrue(scenario.run_controlled_kai_only)
            self.assertFalse(scenario.freeze_controlled_heroes)
        result = selected_scenarios("baseline_marco,deer_set_target_diaochan")
        self.assertEqual(
            [item.name for item in result],
            ["baseline_marco", "deer_set_target_diaochan"],
        )
        with self.assertRaises(ValueError):
            selected_scenarios("missing")

    def test_diy_and_ppo_use_formal_command_timing(self):
        repository = Path(__file__).resolve().parents[2]
        for agent_name in ("agent_diy", "agent_ppo"):
            with (repository / agent_name / "conf" / "train_env_conf.toml").open(
                "rb"
            ) as handle:
                config = tomllib.load(handle)["collaboration_command"]
            self.assertEqual(config["command_cooldown"], 100)
            self.assertEqual(config["command_duration"], 300)

    def test_far_once_never_refreshes_after_command_expiry(self):
        feature = _FeatureProcess()
        probe = OfficialJungleProbe()
        probe.reset(SCENARIO_BY_NAME["deer_far_once_marco"])
        first, _ = probe.apply(
            [0, 0, 0, 0],
            _observation(56),
            1,
            (193, 141),
            feature,
            {"intent": 0},
        )
        self.assertEqual(first, [2, 0, -25530, 23990])

        feature._memo_context["frame_no"] = 362
        feature._memo_context["collaboration"].update(
            active=False,
            type="none",
            command_position=None,
            cooldown_ready=True,
            platform_last_send_frame=56,
        )
        second, diagnostic = probe.apply(
            [0, 0, 0, 0],
            _observation(362),
            1,
            (193, 141),
            feature,
            {"intent": 0},
        )
        self.assertEqual(second, [0, 0, 0, 0])
        self.assertEqual(
            diagnostic["reason"],
            "far_position_already_sent_once",
        )
        self.assertEqual(probe.summary()["position_command_count"], 1)

    def test_far_refresh_resends_same_goal_on_cooldown(self):
        side_anchor = (-25530.0, 17875.0)
        feature = _FeatureProcess()
        probe = OfficialJungleProbe()
        probe.reset(SCENARIO_BY_NAME["deer_far_refresh_marco"])
        first, _ = probe.apply(
            [0, 0, 0, 0],
            _observation(56),
            1,
            (193, 141),
            feature,
            {"intent": 0},
        )
        self.assertEqual(first, [2, 0, -25530, 23990])

        feature._memo_context["frame_no"] = 164
        feature._memo_context["collaboration"].update(
            active=True,
            type="set_position",
            command_position=side_anchor,
            cooldown_ready=True,
            platform_last_send_frame=56,
        )
        second, diagnostic = probe.apply(
            [0, 0, 0, 0],
            _observation(164),
            1,
            (193, 141),
            feature,
            {"intent": 0},
        )
        self.assertEqual(second, first)
        self.assertEqual(
            diagnostic["reason"],
            "refresh_same_far_position",
        )
        summary = probe.summary()
        self.assertEqual(summary["position_command_count"], 2)
        self.assertEqual(summary["same_position_refresh_count"], 1)
        self.assertTrue(summary["formal_command_config_match"])

    def test_far_refresh_until_contact_stops_at_arrival(self):
        side_anchor = (-25530.0, 17875.0)
        feature = _FeatureProcess()
        probe = OfficialJungleProbe()
        probe.reset(SCENARIO_BY_NAME["deer_far_refresh_until_contact_marco"])
        first, _ = probe.apply(
            [0, 0, 0, 0],
            _observation(56),
            1,
            (193, 141),
            feature,
            {"intent": 0},
        )
        self.assertEqual(first, [2, 0, -25530, 23990])

        feature._memo_context["frame_no"] = 164
        feature._memo_context["official_hero"]["side_position"] = side_anchor
        feature._memo_context["collaboration"].update(
            active=True,
            type="set_position",
            command_position=side_anchor,
            cooldown_ready=True,
            platform_last_send_frame=56,
        )
        stopped, diagnostic = probe.apply(
            [0, 0, 0, 0],
            _observation(164),
            1,
            (193, 141),
            feature,
            {"intent": 0},
        )
        self.assertEqual(stopped, [0, 0, 0, 0])
        self.assertEqual(
            diagnostic["reason"],
            "stop_position_after_camp_arrival_radius",
        )

        feature._memo_context["frame_no"] = 470
        feature._memo_context["official_hero"]["side_position"] = (
            -66300.0,
            13465.0,
        )
        feature._memo_context["collaboration"].update(
            active=False,
            type="none",
            command_position=None,
            cooldown_ready=True,
        )
        probe.apply(
            [0, 0, 0, 0],
            _observation(470),
            1,
            (193, 141),
            feature,
            {"intent": 0},
        )
        summary = probe.summary()
        self.assertEqual(summary["position_command_count"], 1)
        self.assertEqual(summary["position_commands_after_stop"], 0)
        self.assertEqual(summary["position_control_stop_frame"], 164)
        self.assertEqual(
            summary["position_control_stop_reason"],
            "camp_arrival_radius",
        )
        self.assertEqual(summary["first_inactive_after_stop_frame"], 470)
        self.assertEqual(
            summary["first_spring_return_after_stop_frame"],
            470,
        )

    def test_far_refresh_until_contact_stops_on_attack_before_arrival(self):
        side_anchor = (-25530.0, 17875.0)
        feature = _FeatureProcess()
        probe = OfficialJungleProbe()
        probe.reset(SCENARIO_BY_NAME["deer_far_refresh_until_contact_marco"])
        probe.apply(
            [0, 0, 0, 0],
            _observation(56),
            1,
            (193, 141),
            feature,
            {"intent": 0},
        )
        feature._memo_context["frame_no"] = 164
        feature._memo_context["official_hero"]["actor"]["attack_target"] = 501
        feature._memo_context["collaboration"].update(
            active=True,
            type="set_position",
            command_position=side_anchor,
            cooldown_ready=True,
            platform_last_send_frame=56,
        )
        stopped, diagnostic = probe.apply(
            [0, 0, 0, 0],
            _observation(164),
            1,
            (193, 141),
            feature,
            {"intent": 0},
        )
        self.assertEqual(stopped, [0, 0, 0, 0])
        self.assertEqual(
            diagnostic["reason"],
            "stop_position_after_camp_engaged",
        )
        feature.monster["hp"] = 0
        feature.authoritative_cleared = True
        feature.monster_camp_history[(-25530.0, 23990.0)].update(
            spawn_timer_ms=30000.0,
            spawn_timer_positive=True,
            timer_death_transition=True,
            last_clear_evidence="spawn_timer_zero_to_positive",
            last_clear_frame=200,
            clear_count=1,
            alive_runtime_ids=(),
        )
        feature._memo_context["frame_no"] = 200
        after_stop, _ = probe.apply(
            [0, 0, 0, 0],
            _observation(200),
            1,
            (193, 141),
            feature,
            {"intent": 0},
        )
        self.assertEqual(after_stop, [0, 0, 0, 0])
        summary = probe.summary()
        self.assertEqual(summary["first_camp_engage_frame"], 164)
        self.assertEqual(summary["first_camp_clear_frame"], 200)
        self.assertEqual(summary["stop_to_clear_frames"], 36)
        self.assertEqual(summary["position_commands_after_stop"], 0)
        self.assertEqual(
            summary["position_control_stop_reason"],
            "camp_engaged",
        )
        self.assertGreater(
            summary["position_control_stop_distance"],
            5000.0,
        )

    def test_far_unset_on_contact_clears_active_position_immediately(self):
        side_anchor = (-25530.0, 17875.0)
        feature = _FeatureProcess()
        probe = OfficialJungleProbe()
        probe.reset(SCENARIO_BY_NAME["deer_far_unset_on_contact_marco"])
        probe.apply(
            [0, 0, 0, 0],
            _observation(56),
            1,
            (193, 141),
            feature,
            {"intent": 0},
        )

        feature._memo_context["frame_no"] = 164
        feature._memo_context["official_hero"]["side_position"] = side_anchor
        feature._memo_context["collaboration"].update(
            active=True,
            type="set_position",
            command_position=side_anchor,
            cooldown_ready=True,
            platform_last_send_frame=56,
            remaining_frames=192,
        )
        unset, diagnostic = probe.apply(
            [0, 0, 0, 0],
            _observation(164),
            1,
            (193, 141),
            feature,
            {"intent": 0},
        )
        self.assertEqual(unset, [3, 0, 0, 0])
        self.assertEqual(
            diagnostic["reason"],
            "unset_position_after_camp_arrival_radius",
        )

        feature._memo_context["frame_no"] = 170
        feature._memo_context["collaboration"].update(
            active=False,
            type="none",
            command_position=None,
            remaining_frames=0,
        )
        after_unset, _ = probe.apply(
            [0, 0, 0, 0],
            _observation(170),
            1,
            (193, 141),
            feature,
            {"intent": 0},
        )
        self.assertEqual(after_unset, [0, 0, 0, 0])
        summary = probe.summary()
        self.assertEqual(summary["unset_command_count"], 1)
        self.assertEqual(summary["unset_command_frames"], (164,))
        self.assertTrue(summary["position_control_stop_active"])
        self.assertEqual(
            summary["position_control_stop_remaining_frames"],
            192,
        )
        self.assertEqual(summary["first_inactive_after_stop_frame"], 170)
        self.assertEqual(summary["unset_to_inactive_frames"], 6)

    def test_refresh_until_clear_uses_authoritative_lifecycle_only(self):
        side_anchor = (-25530.0, 17875.0)
        feature = _FeatureProcess()
        probe = OfficialJungleProbe()
        probe.reset(SCENARIO_BY_NAME["deer_far_refresh_until_clear_marco"])
        probe.apply(
            [0, 0, 0, 0],
            _observation(56),
            1,
            (193, 141),
            feature,
            {"intent": 0},
        )

        feature._memo_context["frame_no"] = 164
        feature._memo_context["official_hero"]["actor"]["attack_target"] = 501
        feature._memo_context["collaboration"].update(
            active=True,
            type="set_position",
            command_position=side_anchor,
            cooldown_ready=True,
            platform_last_send_frame=56,
            remaining_frames=192,
        )
        refreshed, _ = probe.apply(
            [0, 0, 0, 0],
            _observation(164),
            1,
            (193, 141),
            feature,
            {"intent": 0},
        )
        self.assertEqual(refreshed, [2, 0, -25530, 23990])
        self.assertEqual(probe.summary()["position_control_stop_frame"], 0)

        feature.monster_visible = False
        feature._memo_context["frame_no"] = 170
        feature._memo_context["official_hero"]["actor"]["attack_target"] = 0
        feature._memo_context["collaboration"]["cooldown_ready"] = False
        probe.apply(
            [0, 0, 0, 0],
            _observation(170),
            1,
            (193, 141),
            feature,
            {"intent": 0},
        )
        summary = probe.summary()
        self.assertFalse(summary["camp_cleared"])
        self.assertEqual(
            summary["first_unobserved_after_engage_frame"],
            170,
        )

        feature.monster_visible = True
        feature._memo_context["frame_no"] = 176
        probe.apply(
            [0, 0, 0, 0],
            _observation(176),
            1,
            (193, 141),
            feature,
            {"intent": 0},
        )
        self.assertEqual(
            probe.summary()["reappearance_after_unobserved_count"],
            1,
        )

        feature.monster["hp"] = 0
        feature.monster_visible = False
        feature.authoritative_cleared = True
        feature.monster_camp_history[(-25530.0, 23990.0)].update(
            spawn_timer_ms=30000.0,
            spawn_timer_positive=True,
            timer_death_transition=True,
            last_clear_evidence="spawn_timer_zero_to_positive",
            last_clear_frame=200,
            clear_count=1,
            alive_runtime_ids=(),
        )
        feature._memo_context["frame_no"] = 200
        cleared, diagnostic = probe.apply(
            [0, 0, 0, 0],
            _observation(200),
            1,
            (193, 141),
            feature,
            {"intent": 0},
        )
        self.assertEqual(cleared, [3, 0, 0, 0])
        self.assertEqual(
            diagnostic["reason"],
            "unset_position_after_authoritative_camp_clear",
        )
        summary = probe.summary()
        self.assertTrue(summary["camp_clear_is_authoritative"])
        self.assertEqual(summary["first_camp_clear_frame"], 200)
        self.assertEqual(
            summary["position_control_stop_reason"],
            "authoritative_camp_clear",
        )
        self.assertEqual(summary["unset_command_count"], 1)

    def test_staged_refresh_reasserts_matching_waypoint_on_cooldown(self):
        feature = _FeatureProcess()
        probe = OfficialJungleProbe()
        probe.reset(SCENARIO_BY_NAME["deer_staged_refresh_until_contact_marco"])
        first, _ = probe.apply(
            [0, 0, 0, 0],
            _observation(56),
            1,
            (193, 141),
            feature,
            {"intent": 0},
        )
        feature._memo_context["frame_no"] = 164
        feature._memo_context["collaboration"].update(
            active=True,
            type="set_position",
            command_position=(-50000.0, 8000.0),
            cooldown_ready=True,
            platform_last_send_frame=56,
        )
        second, diagnostic = probe.apply(
            [0, 0, 0, 0],
            _observation(164),
            1,
            (193, 141),
            feature,
            {"intent": 0},
        )
        self.assertEqual(second, first)
        self.assertEqual(
            diagnostic["reason"],
            "refresh_current_staged_waypoint",
        )

    def test_staged_route_advances_only_from_physical_arrival(self):
        feature = _FeatureProcess()
        probe = OfficialJungleProbe()
        probe.reset(SCENARIO_BY_NAME["deer_staged_waypoints_marco"])
        first, _ = probe.apply(
            [0, 0, 0, 0],
            _observation(56),
            1,
            (193, 141),
            feature,
            {"intent": 0},
        )
        self.assertEqual(first, [2, 0, -50000, 14115])

        feature._memo_context["frame_no"] = 164
        feature._memo_context["official_hero"]["side_position"] = (
            -50000.0,
            8000.0,
        )
        feature._memo_context["collaboration"].update(
            active=True,
            type="set_position",
            command_position=(-50000.0, 8000.0),
            cooldown_ready=True,
            platform_last_send_frame=56,
        )
        second, diagnostic = probe.apply(
            [0, 0, 0, 0],
            _observation(164),
            1,
            (193, 141),
            feature,
            {"intent": 0},
        )
        self.assertEqual(second, [2, 0, -30000, 9615])
        self.assertEqual(
            diagnostic["reason"],
            "advance_to_next_staged_waypoint",
        )
        self.assertEqual(
            diagnostic["position_control"]["waypoint_name"],
            "upper_lane_corridor",
        )

    def test_staged_route_mirrors_to_red_side(self):
        feature = _FeatureProcess()
        probe = OfficialJungleProbe()
        probe.reset(SCENARIO_BY_NAME["deer_staged_waypoints_marco"])
        result, _ = probe.apply(
            [0, 0, 0, 0],
            _observation(56),
            2,
            (193, 141),
            feature,
            {"intent": 0},
        )
        self.assertEqual(result, [2, 0, 50000, 14115])

    def test_anchor_position_bypasses_outer_tower_gate(self):
        probe = OfficialJungleProbe()
        probe.reset(SCENARIO_BY_NAME["deer_position_anchor_marco"])
        result, diagnostic = probe.apply(
            proposed_command=[2, 0, -50000, 14115],
            observation=_observation(),
            camp=1,
            config_ids=(193, 141),
            feature_process=_FeatureProcess(),
            policy_decision={
                "intent": 5,
                "intent_name": "push_lane",
            },
        )
        self.assertEqual(result, [2, 0, -25530, 23990])
        self.assertTrue(diagnostic["enemy_outer_alive"])
        self.assertTrue(diagnostic["override_applied"])

    def test_matching_position_is_held_without_resend(self):
        side_anchor = (-25530.0, 17875.0)
        feature = _FeatureProcess(
            collaboration={
                "active": True,
                "type": "set_position",
                "target_runtime": 0,
                "command_position": side_anchor,
                "cooldown_ready": False,
                "platform_last_send_frame": 56,
            }
        )
        probe = OfficialJungleProbe()
        probe.reset(SCENARIO_BY_NAME["deer_position_anchor_marco"])
        result, diagnostic = probe.apply(
            [0, 0, 0, 0],
            _observation(),
            1,
            (193, 141),
            feature,
            {"intent": 0},
        )
        self.assertEqual(result, [0, 0, 0, 0])
        self.assertEqual(diagnostic["reason"], "hold_matching_set_position")

    def test_near_visible_monster_emits_raw_set_target(self):
        feature = _FeatureProcess(official_side=(-25530.0, 17875.0))
        probe = OfficialJungleProbe(target_attempt_limit=2)
        probe.reset(SCENARIO_BY_NAME["deer_set_target_marco"])
        result, diagnostic = probe.apply(
            [0, 0, 0, 0],
            _observation(),
            1,
            (193, 141),
            feature,
            {"intent": 0},
        )
        self.assertEqual(result, [1, 501, 0, 0])
        self.assertEqual(diagnostic["reason"], "send_raw_monster_set_target")

    def test_target_without_platform_echo_is_recorded_rejected(self):
        feature = _FeatureProcess(
            frame=56,
            official_side=(-25530.0, 17875.0),
        )
        probe = OfficialJungleProbe(target_attempt_limit=2)
        probe.reset(SCENARIO_BY_NAME["deer_set_target_marco"])
        result, _ = probe.apply(
            [0, 0, 0, 0],
            _observation(56),
            1,
            (193, 141),
            feature,
            {"intent": 0},
        )
        self.assertEqual(result[0], 1)
        feature._memo_context["frame_no"] = 86
        result, diagnostic = probe.apply(
            [0, 0, 0, 0],
            _observation(86),
            1,
            (193, 141),
            feature,
            {"intent": 0},
        )
        self.assertEqual(diagnostic["target_attempts"][0]["status"], "rejected")
        self.assertEqual(result[0], 1)

    def test_matching_platform_target_echo_is_recorded_accepted(self):
        feature = _FeatureProcess(
            frame=56,
            official_side=(-25530.0, 17875.0),
        )
        probe = OfficialJungleProbe(target_attempt_limit=2)
        probe.reset(SCENARIO_BY_NAME["deer_set_target_marco"])
        probe.apply(
            [0, 0, 0, 0],
            _observation(56),
            1,
            (193, 141),
            feature,
            {"intent": 0},
        )
        feature._memo_context["frame_no"] = 62
        feature._memo_context["collaboration"].update(
            active=True,
            type="set_target",
            target_runtime=501,
            cooldown_ready=False,
            platform_last_send_frame=56,
        )
        result, diagnostic = probe.apply(
            [0, 0, 0, 0],
            _observation(62),
            1,
            (193, 141),
            feature,
            {"intent": 0},
        )
        self.assertEqual(result, [0, 0, 0, 0])
        self.assertEqual(diagnostic["target_attempts"][0]["status"], "accepted")

    def test_baseline_gather_to_jungle_is_classified(self):
        feature = _FeatureProcess(
            official_side=(-25530.0, 17875.0),
            official_attack_target=501,
        )
        probe = OfficialJungleProbe()
        probe.reset(SCENARIO_BY_NAME["baseline_marco"])
        proposed = [2, 0, -25530, 23990]
        result, diagnostic = probe.apply(
            proposed,
            _observation(),
            1,
            (193, 141),
            feature,
            {
                "intent": 6,
                "intent_name": "gather_team",
                "selected_position_region": "upper_jungle",
            },
        )
        self.assertEqual(result, proposed)
        self.assertEqual(
            diagnostic["jungle_cause"],
            "policy_gather_team_to_jungle",
        )

    def test_monster_displacement_marks_leash_failure(self):
        feature = _FeatureProcess(
            official_side=(-25530.0, 17875.0),
            monster_position=(-20530.0, 23990.0),
            monster_hp=4000.0,
            official_attack_target=501,
        )
        probe = OfficialJungleProbe()
        probe.reset(SCENARIO_BY_NAME["deer_position_anchor_marco"])
        probe.apply(
            [0, 0, 0, 0],
            _observation(),
            1,
            (193, 141),
            feature,
            {"intent": 0},
        )
        summary = probe.summary()
        self.assertTrue(summary["camp_engaged"])
        self.assertTrue(summary["monster_leash_failure"])
        self.assertGreaterEqual(summary["max_monster_anchor_distance"], 4500.0)
        self.assertEqual(
            summary["monster_pull_threshold_steps"]["4500"],
            1,
        )
        self.assertEqual(
            summary["monster_pull_threshold_steps"]["6000"],
            0,
        )

    def test_team_expiry_renewal_holds_active_goal_then_reissues(self):
        side_anchor = (-25530.0, 17875.0)
        feature = _FeatureProcess()
        probe = OfficialJungleProbe()
        probe.reset(
            SCENARIO_BY_NAME[
                "deer_team_renew_on_expiry_until_clear_marco"
            ]
        )
        first, _ = probe.apply(
            [0, 0, 0, 0],
            _observation(56),
            1,
            (193, 141),
            feature,
            {"intent": 0},
        )
        self.assertEqual(first, [2, 0, -25530, 23990])

        feature._memo_context["frame_no"] = 164
        feature._memo_context["collaboration"].update(
            active=True,
            type="set_position",
            command_position=side_anchor,
            cooldown_ready=True,
            platform_last_send_frame=56,
            remaining_frames=192,
        )
        held, diagnostic = probe.apply(
            [0, 0, 0, 0],
            _observation(164),
            1,
            (193, 141),
            feature,
            {"intent": 0},
        )
        self.assertEqual(held, [0, 0, 0, 0])
        self.assertEqual(
            diagnostic["reason"],
            "hold_matching_position_until_expiry",
        )

        feature._memo_context["frame_no"] = 362
        feature._memo_context["collaboration"].update(
            active=False,
            type="none",
            command_position=None,
            cooldown_ready=True,
            platform_last_send_frame=56,
            remaining_frames=0,
        )
        renewed, diagnostic = probe.apply(
            [0, 0, 0, 0],
            _observation(362),
            1,
            (193, 141),
            feature,
            {"intent": 0},
        )
        self.assertEqual(renewed, [2, 0, -25530, 23990])
        self.assertEqual(diagnostic["reason"], "reissue_expired_position")
        summary = probe.summary()
        self.assertEqual(summary["position_command_count"], 2)
        self.assertEqual(summary["expired_position_reissue_count"], 1)
        self.assertEqual(summary["matching_position_hold_steps"], 1)
        self.assertEqual(summary["position_command_intervals"], (306,))

    def test_team_clear_holds_old_command_then_uses_lane_successor(self):
        side_anchor = (-25530.0, 17875.0)
        feature = _FeatureProcess()
        feature.authoritative_cleared = True
        feature.monster_visible = False
        feature.monster_camp_history[(-25530.0, 23990.0)].update(
            spawn_timer_ms=30000.0,
            spawn_timer_positive=True,
            timer_death_transition=True,
            last_clear_evidence="spawn_timer_zero_to_positive",
            last_clear_frame=200,
            clear_count=1,
            alive_runtime_ids=(),
        )
        feature._memo_context["frame_no"] = 200
        feature._memo_context["collaboration"].update(
            active=True,
            type="set_position",
            command_position=side_anchor,
            cooldown_ready=True,
            platform_last_send_frame=164,
            remaining_frames=264,
        )
        probe = OfficialJungleProbe()
        probe.reset(
            SCENARIO_BY_NAME["deer_team_refresh_until_clear_marco"]
        )
        held, diagnostic = probe.apply(
            [0, 0, 0, 0],
            _observation(200),
            1,
            (193, 141),
            feature,
            {"intent": 0},
        )
        self.assertEqual(held, [0, 0, 0, 0])
        self.assertEqual(
            diagnostic["reason"],
            "hold_old_deer_command_until_successor_cooldown",
        )

        feature._memo_context["frame_no"] = 206
        feature._memo_context["collaboration"].update(
            active=True,
            type="set_position",
            command_position=side_anchor,
            cooldown_ready=False,
            remaining_frames=258,
        )
        handed_off, diagnostic = probe.apply(
            [0, 0, 0, 0],
            _observation(206),
            1,
            (193, 141),
            feature,
            {"intent": 5},
        )
        self.assertEqual(handed_off, [0, 0, 0, 0])
        self.assertEqual(
            diagnostic["reason"],
            "hold_old_deer_command_until_successor_cooldown",
        )
        successor = [2, 0, -50000, 14115]
        feature._memo_context["frame_no"] = 266
        feature._memo_context["collaboration"].update(
            cooldown_ready=True,
            remaining_frames=198,
        )
        handed_off, diagnostic = probe.apply(
            successor,
            _observation(266),
            1,
            (193, 141),
            feature,
            {"intent": 5},
        )
        self.assertEqual(handed_off, successor)
        self.assertEqual(
            diagnostic["reason"],
            "direct_successor_command_after_clear",
        )
        summary = probe.summary()
        self.assertEqual(summary["policy_handoff_frame"], 200)
        self.assertEqual(summary["first_successor_command_frame"], 266)
        self.assertEqual(summary["first_successor_command"], tuple(successor))
        self.assertEqual(summary["unset_command_count"], 0)


if __name__ == "__main__":
    unittest.main()
