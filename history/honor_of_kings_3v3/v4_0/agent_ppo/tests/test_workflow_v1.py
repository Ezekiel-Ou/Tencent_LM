#!/usr/bin/env python3
"""Training workflow boundary regression tests."""

import unittest
from unittest.mock import patch
import sys
import types

import numpy as np

from agent_ppo.conf.conf import FrozenV1Config, V1TrainingConfig
from agent_ppo.feature.definition_v1 import (
    FrameCollectorV1,
    accumulate_total_reward_v1,
)


def _install_workflow_dependency_stub(module_name, **attributes):
    if module_name in sys.modules:
        return
    module = types.ModuleType(module_name)
    for name, value in attributes.items():
        setattr(module, name, value)
    sys.modules[module_name] = module


_install_workflow_dependency_stub(
    "tools.env_conf_manager", EnvConfManager=object
)
_install_workflow_dependency_stub(
    "tools.model_pool_utils", get_valid_model_pool=lambda _logger: []
)
_install_workflow_dependency_stub(
    "tools.metrics_utils", get_training_metrics=lambda *_args, **_kwargs: {}
)
_install_workflow_dependency_stub(
    "common_python.utils.workflow_disaster_recovery",
    handle_disaster_recovery=lambda function: function,
)

from agent_ppo.workflow.train_workflow import EpisodeRunner


class _TerminalFeatureProcess:
    def controlled_infos(self, _observation):
        reward_0 = [0.0] * 11 + [0.25]
        reward_1 = [0.0] * 11 + [-0.1]
        return (
            {"config_id": 193, "reward": reward_0},
            {"config_id": 132, "reward": reward_1},
        )

    def late_monster_reward_scale(self, expected_frame):
        assert expected_frame == 20000
        return 1.0


class _TerminalAgent:
    def __init__(self):
        self.feature_processes = _TerminalFeatureProcess()
        self.last_auxiliary_reward_shaping = np.zeros(
            (
                FrozenV1Config.CONTROLLED_ACTOR_COUNT,
                FrozenV1Config.AUXILIARY_VALUE_COUNT,
            ),
            dtype=np.float32,
        )
        self.last_reward_boundary_is_new = False
        self.command_sparse_reward_total = 0.0
        self.bootstrap_calls = []
        self.terminal_flags = []

    def consume_reward_shaping(self, _observation, episode_terminal=False):
        self.terminal_flags.append(bool(episode_terminal))
        return (0.04, -0.02)

    def bootstrap_values(self, observation):
        self.bootstrap_calls.append(observation)
        return (
            (0.5, 0.25),
            tuple(
                np.full(
                    FrozenV1Config.AUXILIARY_VALUE_COUNT,
                    0.125,
                    dtype=np.float32,
                )
                for _ in range(FrozenV1Config.CONTROLLED_ACTOR_COUNT)
            ),
            0.75,
        )


class _TerminalCollector:
    def __init__(self):
        self.saved = []

    def save_last_frame(self, **kwargs):
        self.saved.append(kwargs)


class WorkflowV1Test(unittest.TestCase):
    def test_default_collector_uses_active_training_horizon(self):
        collector = FrameCollectorV1(num_agents=1)

        self.assertEqual(collector.gamma, V1TrainingConfig.GAMMA)
        self.assertEqual(collector.lamda, V1TrainingConfig.LAMBDA)

    def test_reward_accumulator_accepts_hero_first_seen_after_initial_frame(self):
        totals = {}

        accumulate_total_reward_v1(
            totals,
            {"config_id": 132, "reward": [0.25]},
        )
        accumulate_total_reward_v1(
            totals,
            {"config_id": 193, "reward": [0.5]},
        )
        accumulate_total_reward_v1(
            totals,
            {"config_id": 132, "reward": [-0.1]},
        )

        self.assertAlmostEqual(totals[132], 0.15)
        self.assertAlmostEqual(totals[193], 0.5)

    def _terminal_payload(self, agent, observation):
        with patch.object(EpisodeRunner, "_command_sparse_reward", return_value=0.0):
            return EpisodeRunner._training_terminal_payload(agent, observation)

    def test_truncated_terminal_settles_shaping_and_bootstraps_once(self):
        observation = {"frame_state": {"frame_no": 20000}}
        agent = _TerminalAgent()
        collector = _TerminalCollector()
        payload = self._terminal_payload(agent, observation)

        EpisodeRunner._save_training_terminal_frame(
            collector,
            agent_id=0,
            agent=agent,
            observation=observation,
            terminal_payload=payload,
            terminated=False,
        )

        self.assertEqual(agent.terminal_flags, [True])
        self.assertEqual(agent.bootstrap_calls, [observation])
        self.assertEqual(len(collector.saved), 1)
        saved = collector.saved[0]
        self.assertAlmostEqual(saved["rewards"][0], 0.29, places=7)
        self.assertAlmostEqual(saved["rewards"][1], -0.12, places=7)
        self.assertEqual(saved["actor_bootstrap_values"], (0.5, 0.25))
        self.assertEqual(saved["command_bootstrap_value"], 0.75)

    def test_terminated_terminal_uses_zero_bootstrap_without_model_call(self):
        observation = {"frame_state": {"frame_no": 20000}}
        agent = _TerminalAgent()
        collector = _TerminalCollector()
        payload = self._terminal_payload(agent, observation)

        EpisodeRunner._save_training_terminal_frame(
            collector,
            agent_id=0,
            agent=agent,
            observation=observation,
            terminal_payload=payload,
            terminated=True,
        )

        self.assertEqual(agent.terminal_flags, [True])
        self.assertEqual(agent.bootstrap_calls, [])
        saved = collector.saved[0]
        self.assertEqual(saved["actor_bootstrap_values"], (0.0, 0.0))
        self.assertEqual(saved["command_bootstrap_value"], 0.0)
        for values in saved["actor_auxiliary_bootstrap_values"]:
            np.testing.assert_array_equal(
                values,
                np.zeros(FrozenV1Config.AUXILIARY_VALUE_COUNT),
            )

if __name__ == "__main__":
    unittest.main()
