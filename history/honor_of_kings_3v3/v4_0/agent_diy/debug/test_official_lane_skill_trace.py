#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

import unittest

from agent_diy.debug.official_lane_skill_trace import selected_scenarios


class LaneSkillScenarioTest(unittest.TestCase):
    def test_default_matrix_contains_only_lane_scenarios(self):
        self.assertEqual(
            [scenario.name for scenario in selected_scenarios("")],
            ["lane_marco", "lane_diaochan", "lane_kai"],
        )

    def test_old_baseline_names_remain_accepted_as_aliases(self):
        scenarios = selected_scenarios("baseline_diaochan,baseline_kai")
        self.assertEqual(
            [scenario.name for scenario in scenarios],
            ["lane_diaochan", "lane_kai"],
        )

    def test_jungle_scenario_is_rejected(self):
        with self.assertRaises(ValueError):
            selected_scenarios("deer_team_refresh_until_clear_marco")


if __name__ == "__main__":
    unittest.main()
