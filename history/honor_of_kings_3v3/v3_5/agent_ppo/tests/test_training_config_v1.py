#!/usr/bin/env python3
"""First-stage formal training configuration tests."""

from pathlib import Path
import json
import tomllib
import unittest

from agent_ppo.conf.conf import GameConfig, FrozenV1Config, V1TrainingConfig


ROOT = Path(__file__).resolve().parents[2]


class TrainingConfigV1Test(unittest.TestCase):
    def test_next_stage_reward_and_opponent(self):
        with (ROOT / "agent_ppo/conf/train_env_conf.toml").open("rb") as file:
            config = tomllib.load(file)
        self.assertEqual(config["episode"]["opponent_agent"], "selfplay")
        # Opponent mixing is intentionally user-managed compatibility state.
        self.assertFalse(V1TrainingConfig.OPPONENT_MIX_ENABLED)
        self.assertEqual(
            V1TrainingConfig.OPPONENT_SELFPLAY_GAMES_PER_BLOCK,
            9,
        )
        self.assertEqual(
            V1TrainingConfig.OPPONENT_ANCHOR_GAMES_PER_BLOCK,
            1,
        )
        with (ROOT / "kaiwu.json").open("r", encoding="utf-8") as file:
            model_pool = json.load(file)["model_pool"]
        self.assertIsInstance(model_pool, list)
        self.assertEqual(V1TrainingConfig.OPPONENT_ANCHOR_MODEL_POOL_INDEX, 0)
        reward = config["reward_policies"]["default"]
        expected = {
            "hp_rate_sqrt_sqrt": 2.0,
            "money": 0.004,
            "exp": 0.002,
            "tower": 0.0,
            "killCnt": 0.0,
            "deadCnt": 0.0,
            "assistCnt": 0.0,
            "total_hurt_to_hero": 0.2,
            "atk_monster": 0.05,
            "win_crystal": 0,
            "atk_crystal": 4,
        }
        self.assertEqual(reward, expected)
        customers = config["reward_policies"]["customer"]
        self.assertEqual(len(customers), 3)
        customers_by_hero = {}
        for customer in customers:
            self.assertEqual(len(customer["hero_ids"]), 1)
            hero_id = customer["hero_ids"][0]
            self.assertNotIn(hero_id, customers_by_hero)
            customers_by_hero[hero_id] = customer
            self.assertEqual(customer["killCnt"], 0.0)
            self.assertEqual(customer["deadCnt"], 0.0)
            self.assertEqual(customer["assistCnt"], 0.0)
            for name, value in reward.items():
                if name != "atk_monster":
                    self.assertEqual(customer[name], value)
        self.assertEqual(
            {
                hero_id: customer["atk_monster"]
                for hero_id, customer in customers_by_hero.items()
            },
            V1TrainingConfig.HERO_MONSTER_REWARD_WEIGHTS,
        )
        self.assertTrue(config["reward_policies"]["whether_use_zero_sum_reward"])
        self.assertEqual(config["reward_policies"]["team_spirit"], 0.3)

    def test_effective_and_micro_batch_settings(self):
        with (ROOT / "conf/configure_app.toml").open("rb") as file:
            config = tomllib.load(file)
        # The 3v3 local_aisrv_workflow rejects the generic framework's
        # on-policy switch; train_test must use the project's supported keys.
        self.assertNotIn("algorithm_on_policy_or_off_policy", config["app"])
        self.assertEqual(config["app"]["dump_model_freq"], 8)
        self.assertEqual(config["app"]["train_batch_size"], 1024)
        self.assertEqual(V1TrainingConfig.MICRO_BATCH_SIZE, 256)
        self.assertEqual(1024 % V1TrainingConfig.MICRO_BATCH_SIZE, 0)

        with (ROOT / "agent_ppo/conf/train_env_conf.toml").open("rb") as file:
            env_config = tomllib.load(file)
        # Formal in-training evaluation is intentionally disabled.
        self.assertEqual(env_config["episode"]["eval_interval"], 10000000000)
        self.assertTrue(V1TrainingConfig.ADVANTAGE_NORMALIZATION)
        self.assertTrue(V1TrainingConfig.AMP_ENABLED)
        self.assertGreater(V1TrainingConfig.RUNTIME_VALIDATION_STARTUP_UPDATES, 0)
        self.assertEqual(V1TrainingConfig.RUNTIME_VALIDATION_INTERVAL, 1)
        self.assertGreater(
            V1TrainingConfig.PACKED_SAMPLE_FULL_VALIDATION_STARTUP_SAMPLES,
            0,
        )
        self.assertGreater(
            V1TrainingConfig.PACKED_SAMPLE_FULL_VALIDATION_INTERVAL,
            0,
        )
        self.assertEqual(GameConfig.ROUTINE_LOG_INTERVAL_SECONDS, 300)
        self.assertEqual(
            GameConfig.PLATFORM_METRICS_FETCH_INTERVAL_SECONDS,
            300,
        )
        self.assertEqual(
            GameConfig.PLATFORM_METRICS_FETCH_EPISODE_INTERVAL,
            10,
        )
        self.assertEqual(V1TrainingConfig.VALUE_COEF, 0.5)
        self.assertEqual(V1TrainingConfig.AUXILIARY_VALUE_COEF, 0.1)
        self.assertEqual(V1TrainingConfig.HYBRID_COMPONENT_REG_COEF, 1e-4)
        self.assertEqual(V1TrainingConfig.COMMAND_VALUE_COEF, 0.5)
        self.assertEqual(V1TrainingConfig.ENTROPY_COEF, 0.015)
        self.assertEqual(V1TrainingConfig.COMMAND_ENTROPY_COEF, 0.015)
        self.assertEqual(V1TrainingConfig.JUNGLE_ROUTE_PROGRESS_DISTANCE, 5000.0)
        self.assertEqual(V1TrainingConfig.JUNGLE_ROUTE_PROGRESS_REWARD, 0.01)
        self.assertEqual(V1TrainingConfig.JUNGLE_ENEMY_SIDE_STEP_PENALTY, 0.002)
        self.assertEqual(V1TrainingConfig.JUNGLE_ROUTE_STUCK_PENALTY, 0.001)
        self.assertEqual(V1TrainingConfig.JUNGLE_LANE_SOAK_MAX_SECONDS, 20.0)
        self.assertEqual(V1TrainingConfig.LANE_LAST_HIT_REWARD, 0.05)
        self.assertEqual(V1TrainingConfig.LANE_ATTENDANCE_END_FRAME, 1800)
        self.assertEqual(
            V1TrainingConfig.LANE_ATTENDANCE_PROGRESS_DISTANCE, 5000.0
        )
        self.assertEqual(V1TrainingConfig.LANE_ATTENDANCE_PROGRESS_REWARD, 0.002)
        self.assertEqual(V1TrainingConfig.LANE_ATTENDANCE_NEAR_DISTANCE, 12000.0)
        self.assertEqual(V1TrainingConfig.NO_ATTACKABLE_TARGET_PENALTY, 0.0005)
        self.assertEqual(V1TrainingConfig.NO_ATTACKABLE_TARGET_START_FRAME, 450)
        self.assertEqual(
            V1TrainingConfig.NO_ATTACKABLE_TARGET_RECOVERY_HP_RATIO,
            0.30,
        )
        self.assertEqual(V1TrainingConfig.NO_ATTACKABLE_TARGET_GRACE_FRAMES, 15)
        self.assertEqual(
            V1TrainingConfig.NO_ATTACKABLE_TARGET_REFERENCE_FRAMES, 6
        )
        self.assertEqual(
            V1TrainingConfig.NO_ATTACKABLE_TARGET_MAX_ELAPSED_FRAMES, 30
        )

        workflow_source = (
            ROOT / "agent_ppo/workflow/train_workflow.py"
        ).read_text(encoding="utf-8")
        self.assertNotIn("self.episode_cnt % 10", workflow_source)
        self.assertIn("system_random = random.SystemRandom()", workflow_source)
        self.assertIn("def _routine_log(", workflow_source)
        self.assertIn("def _refresh_platform_metrics(", workflow_source)
        self.assertEqual(V1TrainingConfig.STRATEGIC_WAIT_RADIUS, 8000.0)
        self.assertEqual(V1TrainingConfig.STRATEGIC_CAMP_WAIT_SECONDS, 5.0)
        self.assertEqual(V1TrainingConfig.KILL_REWARD, 0.8)
        self.assertEqual(V1TrainingConfig.DEATH_REWARD, -1.0)
        self.assertEqual(V1TrainingConfig.ASSIST_REWARD, 0.8)
        self.assertEqual(
            V1TrainingConfig.DEATH_MULTIPLIER_BEFORE_CANNON, 0.7
        )
        self.assertEqual(
            V1TrainingConfig.DEATH_MULTIPLIER_AFTER_CANNON, 1.3
        )
        self.assertEqual(V1TrainingConfig.DEATH_MULTIPLIER_CANNON_FRAME, 6254)
        self.assertEqual(V1TrainingConfig.TYRANT_TEAM_REWARD, 1.0)
        self.assertEqual(
            V1TrainingConfig.TYRANT_REWARD_MAX_KILLS_PER_CAMP, 3
        )
        self.assertEqual(
            V1TrainingConfig.REWARD_ENGINEERING_VERSION,
            "ppo_v3_5_reward_state_machine_v3",
        )
        self.assertEqual(
            V1TrainingConfig.TEAMMATE_SUPPORT_WINDOW_FRAMES, 90
        )
        self.assertEqual(
            V1TrainingConfig.TEAMMATE_SUPPORT_RADIUS, 12000.0
        )
        self.assertEqual(
            V1TrainingConfig.TEAMMATE_ABANDONMENT_PENALTY, 0.06
        )
        self.assertEqual(
            V1TrainingConfig.TEAMMATE_SUPPORT_REWARD, 0.05
        )
        self.assertEqual(
            V1TrainingConfig.TEMPORAL_FOCUS_REWARD, 0.05
        )
        self.assertEqual(
            V1TrainingConfig.GROUP_ATTACK_MIN_PARTICIPANTS, 3
        )
        self.assertEqual(V1TrainingConfig.GROUP_ATTACK_REWARD, 0.08)
        self.assertEqual(
            V1TrainingConfig.CRYSTAL_DANGER_RADIUS, 18000.0
        )
        self.assertEqual(
            V1TrainingConfig.DEFENSE_PRESSURE_WINDOW_CAP, 0.15
        )
        self.assertEqual(
            V1TrainingConfig.HERO_MONSTER_REWARD_WEIGHTS,
            {132: 0.05, 141: 0.005, 193: 0.08},
        )
        self.assertEqual(
            V1TrainingConfig.KAI_WOLF_MEMBER_REWARDS,
            {44: 0.40, 45: 0.10},
        )
        self.assertEqual(V1TrainingConfig.TREASURE_THIEF_CONFIG_ID, 59)
        self.assertEqual(V1TrainingConfig.TREASURE_THIEF_TEAM_REWARD, 0.50)
        self.assertEqual(V1TrainingConfig.SPEED_BUFF_CONFIG_ID, 90201)
        self.assertEqual(V1TrainingConfig.SPEED_BUFF_TEAM_REWARD, 0.12)
        self.assertEqual(V1TrainingConfig.COMMAND_CONTROLLED_HERO_WEIGHT, 0.25)
        self.assertEqual(V1TrainingConfig.COMMAND_OFFICIAL_HERO_WEIGHT, 0.50)
        self.assertEqual(V1TrainingConfig.COMMAND_GOAL_PROGRESS_BUDGET, 0.03)
        self.assertEqual(V1TrainingConfig.COMMAND_RESPONSE_WINDOW_FRAMES, 76)
        self.assertEqual(
            V1TrainingConfig.COMMAND_RESPONSE_MIN_PROGRESS_DISTANCE,
            1500.0,
        )
        self.assertEqual(
            V1TrainingConfig.COMMAND_NO_RESPONSE_PENALTY, 0.01
        )
        self.assertEqual(
            V1TrainingConfig.COMMAND_NO_COMMAND_IDLE_PENALTY, 0.005
        )
        self.assertEqual(
            V1TrainingConfig.OFFICIAL_KAI_MONSTER_SET_TARGET_DISTANCE,
            5000.0,
        )
        self.assertEqual(V1TrainingConfig.TERMINAL_CRYSTAL_REWARD, 8.0)
        self.assertEqual(V1TrainingConfig.STRUCTURE_TOWER_TOTAL_REWARD, 2.0)
        self.assertEqual(V1TrainingConfig.STRUCTURE_POTENTIAL_BETA, 0.25)
        self.assertEqual(V1TrainingConfig.STRUCTURE_POTENTIAL_POWER, 2.0)
        self.assertEqual(
            V1TrainingConfig.COMMAND_FARM_CAMP_COMPLETION_REWARD,
            0.03,
        )
        self.assertEqual(V1TrainingConfig.TOWER_RISK_CRITICAL_HP_RATIO, 0.45)
        self.assertEqual(
            V1TrainingConfig.TOWER_RISK_HIT_PENALTIES,
            (0.01, 0.02, 0.03),
        )
        self.assertEqual(V1TrainingConfig.DEFENSE_WAVE_POSITIVE_CAP, 0.06)
        self.assertEqual(
            V1TrainingConfig.DEFENSE_WAVE_MISSING_GRACE_FRAMES,
            V1TrainingConfig.COMMAND_RESPONSE_WINDOW_FRAMES,
        )
        self.assertEqual(V1TrainingConfig.POST_KILL_PUSH_WINDOW_FRAMES, 225)
        self.assertEqual(V1TrainingConfig.POST_KILL_PUSH_REWARD, 0.2)
        self.assertEqual(V1TrainingConfig.LONG_INACTIVITY_START_FRAME, 349)
        self.assertEqual(V1TrainingConfig.LONG_INACTIVITY_WINDOW_FRAMES, 76)
        self.assertEqual(
            V1TrainingConfig.LONG_INACTIVITY_EVALUATION_INTERVAL_FRAMES, 15
        )
        self.assertEqual(V1TrainingConfig.LONG_INACTIVITY_PENALTY_MIN, 0.03)
        self.assertEqual(V1TrainingConfig.LONG_INACTIVITY_PENALTY_MAX, 0.30)
        self.assertEqual(V1TrainingConfig.LONG_INACTIVITY_TYRANT_RADIUS, 7000.0)
        self.assertEqual(
            V1TrainingConfig.LONG_INACTIVITY_TYRANT_SPAWN_SECONDS, 5.0
        )
        self.assertEqual(FrozenV1Config.HYBRID_VALUE_TEAM_LAMBDA, 0.2)
        self.assertEqual(
            FrozenV1Config.AUXILIARY_VALUE_CATEGORIES,
            ("farming", "kda", "damage", "pushing", "win"),
        )
        self.assertEqual(FrozenV1Config.SAMPLE_DIM, 337664)


if __name__ == "__main__":
    unittest.main()
