#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

import unittest

from agent_diy.debug.official_skill_direction import (
    OfficialSkillDirectionTracker,
)


def _actor(config_id, runtime_id, x, z, **extra):
    result = {
        "config_id": config_id,
        "runtime_id": runtime_id,
        "camp": 1,
        "hp": 5000,
        "max_hp": 5000,
        "location": {"x": x, "z": z},
        "actor_type": "hero" if config_id in (132, 141, 193) else "monster",
    }
    result.update(extra)
    return result


class OfficialSkillDirectionTrackerTest(unittest.TestCase):
    def test_directional_skill_keeps_actual_degree_target_and_success_evidence(self):
        target = _actor(42, 501, -20000, 6115)
        official = _actor(
            132,
            10,
            -30000,
            6115,
            forward={"x": 1000, "z": 0},
            attack_target=501,
            real_cmd=[
                {
                    "command_type": 8,
                    # The real platform may serialize inactive union members
                    # as default dictionaries. command_type must win.
                    "obj_skill": {
                        "skillID": 0,
                        "actorID": 0,
                        "slotType": 0,
                    },
                    "dir_skill": {
                        "skillID": 13210,
                        "actorID": 501,
                        "slotType": "SLOT_SKILL_0",
                        "degree": 90,
                    },
                }
            ],
            skill_state={
                "slot_states": [
                    {
                        "slot_type": "SLOT_SKILL_0",
                        "configId": 13210,
                        "usable": True,
                        "cooldown": 0,
                        "usedTimes": 3,
                        "succUsedInFrame": 1,
                        "hitHeroTimes": 0,
                    }
                ]
            },
            hit_target_info=[
                {
                    "hit_target": 501,
                    "skill_id": 13210,
                    "slot_type": "SLOT_SKILL_0",
                    "conti_hit_count": 1,
                }
            ],
        )
        tracker = OfficialSkillDirectionTracker()
        evidence = tracker.observe(
            frame_state={"frame_no": 100, "bullets": []},
            official_actor=official,
            camp=1,
            by_runtime={10: official, 501: target},
            collaboration={
                "active": True,
                "type": "set_target",
                "target_runtime": 501,
                "target_actor": target,
            },
        )

        command = evidence["skill_commands"][0]
        self.assertEqual(command["kind"], "dir_skill")
        self.assertEqual(command["payload_selected_by"], "command_type")
        self.assertEqual(command["degree_raw"], 90.0)
        self.assertEqual(command["actor"]["runtime_id"], 501)
        self.assertEqual(command["slot_index"], 0)
        self.assertEqual(len(evidence["successful_skill_slots"]), 1)
        self.assertEqual(evidence["hits"][0]["target"]["runtime_id"], 501)
        geometry = evidence["references"]["attack_target"]["geometry"]
        self.assertEqual(geometry["bearing_side_degrees"], 0.0)
        self.assertFalse(geometry["target_is_behind_forward"])

    def test_position_skill_and_bullet_velocity_are_mirrored_to_camp_two(self):
        official = _actor(
            141,
            20,
            30000,
            6115,
            forward={"x": -1000, "z": 0},
            realCmd=[
                {
                    "commandType": 9,
                    "objSkill": {
                        "skillId": 0,
                        "actorId": 0,
                        "slotType": 0,
                    },
                    "posSkill": {
                        "skillId": 14120,
                        "slotType": 1,
                        "destPos": {"x": 25000, "z": 7115},
                    },
                }
            ],
        )
        tracker = OfficialSkillDirectionTracker()
        first = {
            "frame_no": 100,
            "bullets": [
                {
                    "runtime_id": 900,
                    "source_actor": 20,
                    "skill_id": 14110,
                    "slot_type": 0,
                    "location": {"x": 29000, "z": 6115},
                }
            ],
        }
        tracker.observe(first, official, 2, {20: official}, {})
        second = {
            "frame_no": 106,
            "bullets": [
                {
                    "runtime_id": 900,
                    "source_actor": 20,
                    "skill_id": 14110,
                    "slot_type": 0,
                    "location": {"x": 23000, "z": 6115},
                }
            ],
        }
        evidence = tracker.observe(second, official, 2, {20: official}, {})

        destination = evidence["skill_commands"][0]["destination"]
        self.assertEqual(destination["raw"], [25000.0, 7115.0])
        self.assertEqual(destination["side"], [-25000.0, 1000.0])
        velocity = evidence["bullets"][0]["velocity_per_raw_frame"]
        self.assertEqual(velocity["raw"], [-1000.0, 0.0])
        self.assertEqual(velocity["side"], [1000.0, 0.0])

    def test_success_without_real_command_is_counted_as_protocol_gap(self):
        official = _actor(
            193,
            30,
            -30000,
            6115,
            skill_state={
                "slot_states": [
                    {
                        "slot_type": 1,
                        "configId": 19320,
                        "succUsedInFrame": 1,
                    }
                ]
            },
        )
        tracker = OfficialSkillDirectionTracker()
        evidence = tracker.observe(
            {"frame_no": 200, "bullets": []}, official, 1, {30: official}, {}
        )

        self.assertFalse(evidence["real_cmd_field_present"])
        self.assertEqual(evidence["skill_commands"], [])
        self.assertEqual(
            tracker.summary()["successful_skill_without_real_cmd"], 1
        )

    def test_projectile_moving_away_from_explicit_actor_is_only_a_candidate(self):
        target = _actor(141, 51, 10000, 0)
        target["camp"] = 2
        official = _actor(
            132,
            50,
            0,
            0,
            real_cmd=[
                {
                    "command_type": 8,
                    "dir_skill": {
                        "skillID": 13210,
                        "actorID": 51,
                        "slotType": 1,
                        "degree": 180,
                    },
                }
            ],
        )
        tracker = OfficialSkillDirectionTracker()
        tracker.observe(
            {
                "frame_no": 100,
                "bullets": [
                    {
                        "runtime_id": 900,
                        "source_actor": 50,
                        "skill_id": 13210,
                        "location": {"x": 1000, "z": 0},
                    }
                ],
            },
            official,
            1,
            {50: official, 51: target},
            {},
        )
        evidence = tracker.observe(
            {
                "frame_no": 106,
                "bullets": [
                    {
                        "runtime_id": 900,
                        "source_actor": 50,
                        "skill_id": 13210,
                        "location": {"x": -5000, "z": 0},
                    }
                ],
            },
            official,
            1,
            {50: official, 51: target},
            {},
        )

        assessment = evidence["aim_assessment"]
        self.assertEqual(len(assessment["geometric_reverse_candidates"]), 1)
        self.assertFalse(assessment["reverse_candidate_is_conclusion"])
        link = assessment["projectile_aims"][0]
        self.assertEqual(link["degree_execution_error"], 0.0)
        self.assertEqual(
            link["primary_target_alignment"]["source"],
            "explicit_skill_actor",
        )
        self.assertTrue(
            link["primary_target_alignment"]["is_reverse_candidate"]
        )

    def test_platform_command_type_seven_selects_object_skill(self):
        target = _actor(141, 61, 10000, 0)
        target["camp"] = 2
        official = _actor(
            132,
            60,
            0,
            0,
            real_cmd=[
                {
                    "command_type": 7,
                    "obj_skill": {
                        "skillID": 13230,
                        "actorID": 61,
                        "slotType": 3,
                    },
                    "dir_skill": {
                        "skillID": 0,
                        "actorID": 0,
                        "slotType": 0,
                        "degree": 0,
                    },
                }
            ],
        )
        evidence = OfficialSkillDirectionTracker().observe(
            {"frame_no": 100, "bullets": []},
            official,
            1,
            {60: official, 61: target},
            {},
        )

        command = evidence["skill_commands"][0]
        self.assertEqual(command["kind"], "obj_skill")
        self.assertEqual(command["skill_id"], 13230)
        self.assertEqual(command["actor"]["runtime_id"], 61)


if __name__ == "__main__":
    unittest.main()
