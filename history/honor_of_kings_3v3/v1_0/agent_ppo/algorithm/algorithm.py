#!/usr/bin/env python3
"""Platform compatibility surface for the active PPO learner."""

from agent_ppo.algorithm.algorithm_v1 import AlgorithmV1, PPOLossV1

Algorithm = AlgorithmV1

__all__ = ("Algorithm", "AlgorithmV1", "PPOLossV1")
