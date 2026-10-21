#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""Checkpoint-compatible episode scheduler for mixed training opponents."""

import random


class OpponentMixSchedulerV1:
    """Emit an exact per-block self-play/fixed-anchor opponent mix."""

    def __init__(
        self,
        anchor_model_id,
        selfplay_games_per_block=9,
        anchor_games_per_block=1,
        seed=0,
    ):
        self.anchor_model_id = str(anchor_model_id)
        if not self.anchor_model_id.isdigit() or int(self.anchor_model_id) <= 0:
            raise ValueError("anchor_model_id must be a positive platform model ID")

        self.selfplay_games_per_block = int(selfplay_games_per_block)
        self.anchor_games_per_block = int(anchor_games_per_block)
        if self.selfplay_games_per_block <= 0:
            raise ValueError("selfplay_games_per_block must be positive")
        if self.anchor_games_per_block <= 0:
            raise ValueError("anchor_games_per_block must be positive")

        self._rng = random.Random(int(seed))
        self._block = []
        self._next_eval_is_anchor = False
        self.selfplay_episode_count = 0
        self.anchor_episode_count = 0
        self.eval_commonai_episode_count = 0
        self.eval_anchor_episode_count = 0

    @property
    def block_size(self):
        return self.selfplay_games_per_block + self.anchor_games_per_block

    def _new_block(self):
        block = ["selfplay"] * self.selfplay_games_per_block
        block.extend([self.anchor_model_id] * self.anchor_games_per_block)
        self._rng.shuffle(block)
        return block

    def next_training_opponent(self):
        """Select one non-evaluation opponent and advance the block."""
        if not self._block:
            self._block = self._new_block()
        opponent = self._block.pop()
        if opponent == "selfplay":
            self.selfplay_episode_count += 1
        else:
            self.anchor_episode_count += 1
        return opponent

    def next_evaluation_opponent(self):
        """Alternate CommonAI and the immutable previous-stage anchor."""
        if self._next_eval_is_anchor:
            opponent = self.anchor_model_id
            self.eval_anchor_episode_count += 1
        else:
            opponent = "common_ai"
            self.eval_commonai_episode_count += 1
        self._next_eval_is_anchor = not self._next_eval_is_anchor
        return opponent

    def monitor_metrics(self):
        total = self.selfplay_episode_count + self.anchor_episode_count
        return {
            "opponent_selfplay_episode_count": float(
                self.selfplay_episode_count
            ),
            "opponent_anchor_episode_count": float(self.anchor_episode_count),
            "opponent_anchor_episode_ratio": float(
                self.anchor_episode_count / max(total, 1)
            ),
            "eval_commonai_episode_count": float(
                self.eval_commonai_episode_count
            ),
            "eval_anchor_episode_count": float(
                self.eval_anchor_episode_count
            ),
        }


__all__ = ("OpponentMixSchedulerV1",)
