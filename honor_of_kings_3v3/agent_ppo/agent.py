#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""Active agent: two controlled actors and one learned team command."""

import os
import numpy as np
import torch

# Platform actor processes expose many CPU cores.  Small time=1 inference is
# severely slowed by per-op thread-pool fan-out, so keep the official baseline
# intra-op setting.  Do not configure inter-op threads here: the platform
# model-pool loader removes that call from saved opponent packages, which can
# leave a surrounding try/except block syntactically empty during reload.
# This changes execution only, never model/sample semantics.
torch.set_num_threads(1)

try:
    from kaiwudrl.interface.agent import BaseAgent
except ModuleNotFoundError:  # Local contract/smoke tests do not ship the SDK.
    class BaseAgent:
        def __init__(self, agent_type="player", device=None, logger=None, monitor=None):
            self.agent_type = agent_type


try:
    # The platform wrapper resolves a model ID from kaiwu.json/model management
    # and delegates inference to that immutable opponent package.
    from kaiwu_agent.agent.base_agent import load_opponent_agent_wrapper
except (ImportError, ModuleNotFoundError):
    # Local unit tests do not ship the platform model-pool SDK.
    def load_opponent_agent_wrapper(func):
        return func


from agent_ppo.algorithm.algorithm_v1 import AlgorithmV1
from agent_ppo.conf.conf import FrozenV1Config, V1TrainingConfig
from agent_ppo.feature.collaboration_v1 import (
    CommandRuleLegalizerV1,
    build_command_legal_masks_v1,
)
from agent_ppo.feature.collision_avoidance_v1 import CollisionAvoidanceV1
from agent_ppo.feature.lane_opening_v1 import LaneOpeningLegalizerV1
from agent_ppo.feature.jungle_opening_v1 import JungleOpeningLegalizerV1
from agent_ppo.feature.flash_legalizer_v1 import FlashLegalizerV1
from agent_ppo.feature.safety_legalizer_v1 import SafetyLegalizerV1
from agent_ppo.feature.v40_legalizers import (
    LateConversionLegalizerV1,
    RespawnReentryLegalizerV1,
    TyrantSmiteLegalizerV1,
)
from agent_ppo.feature.definition_v1 import (
    ActDataV1,
    ActorDecisionV1,
    CommandDecisionV1,
    ObsDataV1,
)
from agent_ppo.feature.feature_process.v1_feature_process import FeatureProcessV1
from agent_ppo.feature.feature_process import v1_utils as u
from agent_ppo.model.action_v1 import sample_actor_actions_v1
from agent_ppo.model.checkpoint_v1 import load_checkpoint_v1, save_checkpoint_v1
from agent_ppo.model.command_v1 import (
    run_command_guard_self_test_v1,
    sample_command_actions_v1,
)
from agent_ppo.model.network_v1 import NetworkV1


NONE_ACTION = {
    132: [0, 0, 0, 0, 0],
    141: [0, 0, 0, 0, 0],
    193: [0, 0, 0, 0, 0],
}
NONE_COL_ACTION = [0, 0, 0, 0]


class Agent(BaseAgent):
    def __init__(self, agent_type="player", device=None, logger=None, monitor=None):
        super().__init__(agent_type, device, logger, monitor)
        self.runtime_agent_type = str(agent_type)
        self.device = torch.device(device or "cpu")
        self.logger = logger
        self.monitor = monitor
        command_guard_status = run_command_guard_self_test_v1()
        if self.logger:
            self.logger.info(command_guard_status)
        self.model = NetworkV1().to(self.device)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=V1TrainingConfig.LEARNING_RATE,
            betas=(0.9, 0.999),
            eps=1e-8,
        )
        self.algorithm = AlgorithmV1(
            self.model,
            self.optimizer,
            device=self.device,
            logger=logger,
            monitor=monitor,
        )
        self.legalizer = CommandRuleLegalizerV1()
        self.feature_processes = None
        self.collision_avoidance = None
        self.lane_opening_legalizer = None
        self.jungle_opening_legalizer = None
        self.flash_legalizer = None
        self.safety_legalizer = None
        self.respawn_reentry_legalizer = None
        self.tyrant_smite_legalizer = None
        self.late_conversion_legalizer = None
        self.hero_camp = None
        self.rnn_state_by_config = {}
        self.cur_model_name = ""
        self.loaded_checkpoint_identity = None
        self.obs_data = None
        self.act_data = None
        self.command_legalizer_attempt_count = 0
        self.invalid_command_count = 0
        self.jungle_route_shaping_total = 0.0
        self.last_hit_reward_total = 0.0
        self.last_hit_count = 0
        self.no_target_penalty_total = 0.0
        self.no_target_penalty_count = 0
        self.no_target_grace_count = 0
        self.strategic_wait_exemption_count = 0
        self.lane_attendance_shaping_total = 0.0
        self.lane_attendance_shaping_count = 0
        self.custom_kda_reward_total = 0.0
        self.tyrant_reward_total = 0.0
        self.kai_wolf_reward_total = 0.0
        self.kai_wolf_reward_count = 0
        self.treasure_thief_reward_total = 0.0
        self.treasure_thief_reward_count = 0
        self.speed_buff_reward_total = 0.0
        self.speed_buff_reward_count = 0
        self.structure_reward_total = 0.0
        self.structure_positive_reward_total = 0.0
        self.structure_negative_reward_total = 0.0
        self.structure_attribution_coverage_total = 0.0
        self.structure_attribution_event_count = 0
        self.structure_attribution_fallback_count = 0
        self.structure_attributed_hit_count = 0
        self.structure_actor_reward_by_config = {
            config_id: 0.0 for config_id in u.HERO_CONFIG_IDS
        }
        self.structure_actor_hit_by_config = {
            config_id: 0 for config_id in u.HERO_CONFIG_IDS
        }
        self.post_kill_push_reward_total = 0.0
        self.terminal_crystal_reward_total = 0.0
        self.crystal_danger_reward_total = 0.0
        self.defense_pressure_reward_total = 0.0
        self.defense_pressure_count = 0
        self.temporal_focus_reward_total = 0.0
        self.temporal_focus_count = 0
        self.teammate_abandonment_penalty_total = 0.0
        self.teammate_abandonment_penalty_count = 0
        self.teammate_support_reward_total = 0.0
        self.teammate_support_count = 0
        self.group_attack_reward_total = 0.0
        self.group_attack_count = 0
        self.long_inactivity_penalty_total = 0.0
        self.long_inactivity_penalty_count = 0
        self.long_inactivity_tyrant_exemption_count = 0
        self.long_inactivity_stationary_count = 0
        self.long_inactivity_wandering_count = 0
        self.long_inactivity_no_damage_count = 0
        self.command_sparse_reward_total = 0.0
        self.late_monster_raw_total = 0.0
        self.late_monster_offset_total = 0.0
        self.late_monster_effective_total = 0.0
        self.cake_reward_total = 0.0
        self.cake_progress_reward_total = 0.0
        self.cake_pickup_reward_total = 0.0
        self.cake_clawback_total = 0.0
        self.cake_reward_by_config = {
            config_id: 0.0 for config_id in u.HERO_CONFIG_IDS
        }
        self.cake_progress_by_config = {
            config_id: 0.0 for config_id in u.HERO_CONFIG_IDS
        }
        self.cake_pickup_by_config = {
            config_id: 0.0 for config_id in u.HERO_CONFIG_IDS
        }
        self.cake_clawback_by_config = {
            config_id: 0.0 for config_id in u.HERO_CONFIG_IDS
        }
        self.last_kda_reward_by_runtime = {}
        self.last_kda_reward_by_config = {}
        self.last_tyrant_team_reward = 0.0
        self.last_structure_team_reward = 0.0
        self.last_post_kill_push_reward = 0.0
        self.last_crystal_outcome_reward = 0.0
        self.last_crystal_danger_reward = 0.0
        self.last_defense_pressure_reward = 0.0
        self.last_temporal_focus_team_reward = 0.0
        self.last_group_attack_team_reward = 0.0
        self.last_command_teammate_abandonment_penalty = 0.0
        self.last_command_teammate_support_reward = 0.0
        self.last_command_official_kda_reward = 0.0
        self.last_command_goal_progress_reward = 0.0
        self.last_command_nonresponse_penalty = 0.0
        self.last_command_no_command_idle_penalty = 0.0
        self.last_command_farm_camp_reward = 0.0
        self.command_farm_camp_reward_total = 0.0
        self.tower_risk_penalty_total = 0.0
        self.tower_risk_hit_count = 0
        self.defense_wave_reward_total = 0.0
        self.defense_wave_completion_count = 0
        self.command_response_reward_total = 0.0
        self.command_nonresponse_penalty_total = 0.0
        self.command_nonresponse_penalty_count = 0
        self.command_no_command_idle_penalty_total = 0.0
        self.command_no_command_idle_penalty_count = 0
        self.command_teammate_abandonment_penalty_total = 0.0
        self.command_teammate_abandonment_penalty_count = 0
        self.command_teammate_support_reward_total = 0.0
        self.command_teammate_support_count = 0
        self.last_command_attribution_active = False
        self.last_command_attribution_intent = 0
        self.last_reward_boundary_frame = None
        self.last_reward_boundary_is_new = False
        self.last_auxiliary_reward_shaping = np.zeros(
            (
                FrozenV1Config.CONTROLLED_ACTOR_COUNT,
                FrozenV1Config.AUXILIARY_VALUE_COUNT,
            ),
            dtype=np.float32,
        )

    def observation_process(self, observation):
        if self.feature_processes is None:
            raise RuntimeError("Agent.reset must be called before observation_process")
        feature = self.feature_processes.process_feature(observation, copy=False)
        hero_infos = self.feature_processes.controlled_infos(observation)
        runtime_ids = tuple(
            u.as_int(u.safe_get(info, "runtime_id", default=0), 0)
            for info in hero_infos
        )
        config_ids = tuple(
            u.as_int(u.safe_get(info, "config_id", default=0), 0)
            for info in hero_infos
        )
        raw_legal = np.stack(
            [
                np.asarray(u.safe_get(info, "legal_action", default=[]), dtype=np.float32)
                for info in hero_infos
            ],
            axis=0,
        )
        if raw_legal.shape != (2, FrozenV1Config.RAW_LEGAL_ACTION_DIM):
            raise ValueError(f"invalid v1 raw legal action shape: {raw_legal.shape}")
        if self.collision_avoidance is None:
            raise RuntimeError("Agent.reset must initialize collision avoidance")
        jungle_runtime_plans = self.feature_processes.jungle_opening_runtime_plans()
        force_escape_runtime_ids = tuple(
            int(plan.get("runtime_id", 0))
            for plan in jungle_runtime_plans
            if plan.get("route_stuck", False)
            and plan.get("hard_mask_active", False)
            and int(plan.get("runtime_id", 0)) > 0
        )
        raw_legal, self.collision_diagnostics = self.collision_avoidance.apply(
            observation,
            hero_infos,
            raw_legal,
            force_escape_runtime_ids=force_escape_runtime_ids,
        )
        if self.jungle_opening_legalizer is None:
            raise RuntimeError("Agent.reset must initialize jungle opening legalizer")
        raw_legal, self.jungle_opening_diagnostics = self.jungle_opening_legalizer.apply(
            raw_legal,
            config_ids,
            jungle_runtime_plans,
            self.collision_diagnostics,
        )
        if self.lane_opening_legalizer is None:
            raise RuntimeError("Agent.reset must initialize lane opening legalizer")
        raw_legal, self.lane_opening_diagnostics = self.lane_opening_legalizer.apply(
            raw_legal,
            config_ids,
            self.feature_processes.lane_opening_runtime_plans(),
            self.collision_diagnostics,
        )
        if self.respawn_reentry_legalizer is None:
            raise RuntimeError("Agent.reset must initialize respawn reentry legalizer")
        raw_legal, self.respawn_reentry_diagnostics = (
            self.respawn_reentry_legalizer.apply(
                raw_legal,
                config_ids,
                self.feature_processes.respawn_reentry_runtime_plans(),
            )
        )
        if self.tyrant_smite_legalizer is None:
            raise RuntimeError("Agent.reset must initialize Tyrant Smite legalizer")
        raw_legal, self.tyrant_smite_diagnostics = (
            self.tyrant_smite_legalizer.apply(
                raw_legal,
                config_ids,
                self.feature_processes.tyrant_smite_runtime_plans(),
            )
        )
        if self.late_conversion_legalizer is None:
            raise RuntimeError("Agent.reset must initialize late conversion legalizer")
        raw_legal, self.late_conversion_diagnostics = (
            self.late_conversion_legalizer.apply(
                raw_legal,
                config_ids,
                self.feature_processes.late_conversion_runtime_plans(),
            )
        )
        if self.safety_legalizer is None:
            raise RuntimeError("Agent.reset must initialize safety legalizer")
        raw_legal, self.safety_diagnostics = self.safety_legalizer.apply(
            raw_legal,
            config_ids,
            self.feature_processes.safety_runtime_plans(),
        )
        if self.flash_legalizer is None:
            raise RuntimeError("Agent.reset must initialize flash legalizer")
        raw_legal, flash_offset_masks, self.flash_diagnostics = (
            self.flash_legalizer.apply(
                raw_legal,
                self.feature_processes.flash_runtime_plans(),
            )
        )
        hidden = np.zeros((2, FrozenV1Config.RNN_HIDDEN_SIZE), dtype=np.float32)
        rnn_reset = np.ones(2, dtype=np.float32)
        frame_heroes = {
            u.actor_runtime_id(actor): actor
            for actor in observation.get("frame_state", {}).get("hero_states", [])
        }
        for index, (config_id, runtime_id) in enumerate(zip(config_ids, runtime_ids)):
            cached = self.rnn_state_by_config.get(config_id)
            actor = frame_heroes.get(runtime_id)
            alive = bool(actor) and u.is_alive(actor)
            if cached is not None and cached["runtime_id"] == runtime_id and alive:
                hidden[index] = cached["hidden"]
                rnn_reset[index] = 0.0
            else:
                self.rnn_state_by_config[config_id] = {
                    "runtime_id": runtime_id,
                    "hidden": np.zeros(FrozenV1Config.RNN_HIDDEN_SIZE, dtype=np.float32),
                }
        command_context = self.feature_processes.command_runtime_context()
        command_masks = build_command_legal_masks_v1(command_context)
        return ObsDataV1(
            feature=feature,
            raw_legal_action=raw_legal,
            rnn_hidden=hidden,
            rnn_reset=rnn_reset,
            runtime_ids=runtime_ids,
            config_ids=config_ids,
            hero_infos=hero_infos,
            command_context=command_context,
            command_masks=command_masks,
            flash_offset_masks=flash_offset_masks,
        )

    @staticmethod
    def _actor_decisions(
        sample, value, auxiliary_values, initial_hidden, next_hidden
    ):
        decisions = []
        for index in range(2):
            decisions.append(
                ActorDecisionV1(
                    action=sample["actions"][0, index, 0]
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(np.float32),
                    probability_vector=sample["probability_vector"][0, index, 0]
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(np.float32),
                    conditioned_legal_action=sample[
                        "conditioned_legal_action"
                    ][0, index, 0]
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(np.float32),
                    value=float(value[0, index, 0, 0].detach().cpu()),
                    auxiliary_values=auxiliary_values[0, index, 0]
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(np.float32),
                    initial_hidden=initial_hidden[index].copy(),
                    next_hidden=next_hidden[0, index]
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(np.float32),
                )
            )
        return tuple(decisions)

    def _command_decision(self, sample, command_value, context, masks):
        action = sample["actions"][0, 0].detach().cpu().numpy().astype(np.float32)
        self.command_legalizer_attempt_count = int(
            getattr(self, "command_legalizer_attempt_count", 0)
        ) + 1
        legalized = self.legalizer.legalize(
            int(action[0]), int(action[1]), int(action[2]), context, masks
        )
        if not legalized.valid:
            self.invalid_command_count += 1
            if self.logger:
                self.logger.info(
                    "sampled command failed the shared v1 legalizer; "
                    f"falling back to no-command: {legalized.reason}"
                )
            safe_legal_mask = np.zeros(
                FrozenV1Config.COMMAND_LEGAL_MASK_DIM, dtype=np.float32
            )
            safe_legal_mask[0] = 1.0
            return CommandDecisionV1(
                action=np.zeros(3, dtype=np.float32),
                old_log_probabilities=np.zeros(3, dtype=np.float32),
                conditioned_legal_mask=safe_legal_mask,
                active_head_mask=np.asarray((1.0, 0.0, 0.0), dtype=np.float32),
                # The sampled action was not executed, so this containment
                # record must never enter the command PPO objective.
                is_train=0.0,
                value=float(command_value[0, 0, 0].detach().cpu()),
                protocol_action=(0, 0, 0, 0),
            )
        return CommandDecisionV1(
            action=action,
            old_log_probabilities=sample["old_log_probabilities"][0, 0]
            .detach()
            .cpu()
            .numpy()
            .astype(np.float32),
            conditioned_legal_mask=sample["conditioned_legal_mask"][0, 0]
            .detach()
            .cpu()
            .numpy()
            .astype(np.float32),
            active_head_mask=sample["active_head_mask"][0, 0]
            .detach()
            .cpu()
            .numpy()
            .astype(np.float32),
            is_train=float(sample["command_is_train"][0, 0].detach().cpu()),
            value=float(command_value[0, 0, 0].detach().cpu()),
            protocol_action=legalized.action,
        )

    def _model_inference(self, list_obs_data, stochastic=True):
        if len(list_obs_data) != 1:
            raise ValueError("Agent v1 inference expects one environment observation")
        obs_data = list_obs_data[0]
        feature = (
            torch.from_numpy(obs_data.feature)
            .to(self.device)
            .unsqueeze(0)
            .unsqueeze(2)
        )
        raw_legal = (
            torch.from_numpy(obs_data.raw_legal_action)
            .to(self.device)
            .unsqueeze(0)
            .unsqueeze(2)
        )
        hidden = (
            torch.from_numpy(obs_data.rnn_hidden)
            .to(self.device)
            .unsqueeze(0)
        )
        intent_mask = (
            torch.from_numpy(obs_data.command_masks.intent)
            .to(self.device)
            .reshape(1, 1, FrozenV1Config.COMMAND_HEADS[0])
        )
        target_masks = (
            torch.from_numpy(obs_data.command_masks.target_by_intent)
            .to(self.device)
            .reshape(
                1,
                1,
                FrozenV1Config.COMMAND_HEADS[0],
                FrozenV1Config.COMMAND_HEADS[1],
            )
        )
        position_masks = (
            torch.from_numpy(obs_data.command_masks.position_by_intent)
            .to(self.device)
            .reshape(
                1,
                1,
                FrozenV1Config.COMMAND_HEADS[0],
                FrozenV1Config.COMMAND_HEADS[2],
            )
        )
        command_is_train = torch.tensor(
            [[float(obs_data.command_masks.command_is_train)]],
            device=self.device,
        )
        self.model.eval()
        deterministic = not bool(stochastic)
        with torch.inference_mode():
            output = self.model(feature, hidden)
            actor_sample = sample_actor_actions_v1(
                output["actor"],
                raw_legal,
                deterministic=deterministic,
                flash_offset_masks=(
                    torch.from_numpy(obs_data.flash_offset_masks)
                    .to(self.device)
                    .unsqueeze(0)
                    .unsqueeze(2)
                ),
            )
            command_sample = sample_command_actions_v1(
                output["command"],
                intent_mask,
                target_masks,
                position_masks,
                command_is_train=command_is_train,
                deterministic=deterministic,
            )
        next_hidden = output["actor"]["next_hidden"]
        value = output["actor"]["actor_value"]
        auxiliary_values = output["actor"]["actor_auxiliary_values"]
        actor_decisions = self._actor_decisions(
            actor_sample,
            value,
            auxiliary_values,
            obs_data.rnn_hidden,
            next_hidden,
        )
        command_decision = self._command_decision(
            command_sample,
            output["command"]["command_value"],
            obs_data.command_context,
            obs_data.command_masks,
        )
        return ActDataV1(
            stochastic_actors=actor_decisions if stochastic else None,
            deterministic_actors=None if stochastic else actor_decisions,
            stochastic_command=command_decision if stochastic else None,
            deterministic_command=None if stochastic else command_decision,
        )

    def predict(self, observation):
        obs_data = self.observation_process(observation)
        act_data = self._model_inference([obs_data], stochastic=True)
        self.update_status(obs_data, act_data, stochastic=True)
        return self.action_process(observation, act_data, stochastic=True)

    def exploit(self, observation):
        obs_data = self.observation_process(observation)
        act_data = self._model_inference([obs_data], stochastic=False)
        self.update_status(obs_data, act_data, stochastic=False)
        return self.action_process(observation, act_data, stochastic=False)

    def bootstrap_values(self, observation):
        """Evaluate a truncation state without emitting or recording an action."""

        obs_data = self.observation_process(observation)
        feature = torch.from_numpy(obs_data.feature).to(self.device).unsqueeze(0).unsqueeze(2)
        hidden = torch.from_numpy(obs_data.rnn_hidden).to(self.device).unsqueeze(0)
        self.model.eval()
        with torch.inference_mode():
            output = self.model(feature, hidden)
        actor_values = tuple(
            float(output["actor"]["actor_value"][0, index, 0, 0].cpu())
            for index in range(2)
        )
        actor_auxiliary_values = tuple(
            output["actor"]["actor_auxiliary_values"][0, index, 0]
            .cpu()
            .numpy()
            .astype(np.float32)
            for index in range(2)
        )
        command_value = float(output["command"]["command_value"][0, 0, 0].cpu())
        return actor_values, actor_auxiliary_values, command_value

    def consume_jungle_route_shaping(self, observation):
        """Process boundary states and consume shaping exactly once per frame."""

        if self.feature_processes is None:
            self.last_auxiliary_reward_shaping.fill(0.0)
            return (0.0,) * FrozenV1Config.CONTROLLED_ACTOR_COUNT
        self.feature_processes.process_feature(observation, copy=False)
        frame_state = observation.get("frame_state", {}) or {}
        rewards = self.feature_processes.consume_jungle_route_shaping(
            expected_frame=u.frame_number(frame_state)
        )
        self.jungle_route_shaping_total += float(sum(rewards))
        return rewards

    def consume_reward_shaping(self, observation, episode_terminal=False):
        """Consume all custom scalar shaping exactly once at this boundary."""

        if self.feature_processes is None:
            self.last_auxiliary_reward_shaping.fill(0.0)
            self.last_kda_reward_by_runtime = {}
            self.last_kda_reward_by_config = {}
            self.last_tyrant_team_reward = 0.0
            self.last_structure_team_reward = 0.0
            self.last_post_kill_push_reward = 0.0
            self.last_crystal_outcome_reward = 0.0
            self.last_crystal_danger_reward = 0.0
            self.last_defense_pressure_reward = 0.0
            self.last_temporal_focus_team_reward = 0.0
            self.last_group_attack_team_reward = 0.0
            self.last_command_teammate_abandonment_penalty = 0.0
            self.last_command_teammate_support_reward = 0.0
            self.last_command_official_kda_reward = 0.0
            self.last_command_goal_progress_reward = 0.0
            self.last_command_nonresponse_penalty = 0.0
            self.last_command_no_command_idle_penalty = 0.0
            self.last_command_farm_camp_reward = 0.0
            self.last_command_attribution_active = False
            self.last_command_attribution_intent = 0
            self.last_reward_boundary_is_new = False
            return (0.0,) * FrozenV1Config.CONTROLLED_ACTOR_COUNT
        self.feature_processes.process_feature(observation, copy=False)
        frame_state = observation.get("frame_state", {}) or {}
        frame_no = u.frame_number(frame_state)
        self.last_reward_boundary_is_new = frame_no != self.last_reward_boundary_frame
        self.last_reward_boundary_frame = frame_no
        route = self.consume_jungle_route_shaping(observation)
        behavior = self.feature_processes.consume_behavior_shaping(
            expected_frame=frame_no
        )
        behavior_rewards = np.asarray(
            behavior["rewards"], dtype=np.float32
        )
        self.last_hit_count += int(sum(behavior["last_hit_counts"]))
        self.last_hit_reward_total += float(sum(behavior["last_hit_rewards"]))
        self.cake_reward_total += float(sum(behavior["cake_rewards"]))
        self.cake_progress_reward_total += float(
            sum(behavior["cake_progress_rewards"])
        )
        self.cake_pickup_reward_total += float(
            sum(behavior["cake_pickup_rewards"])
        )
        self.cake_clawback_total += float(
            sum(behavior["cake_clawbacks"])
        )
        for actor_index, info in enumerate(
            self.feature_processes._memo_context["controlled_infos"]
        ):
            config_id = u.as_int(
                u.safe_get(info, "config_id", default=0), 0
            )
            if config_id not in self.cake_reward_by_config:
                continue
            self.cake_reward_by_config[config_id] += float(
                behavior["cake_rewards"][actor_index]
            )
            self.cake_progress_by_config[config_id] += float(
                behavior["cake_progress_rewards"][actor_index]
            )
            self.cake_pickup_by_config[config_id] += float(
                behavior["cake_pickup_rewards"][actor_index]
            )
            self.cake_clawback_by_config[config_id] += float(
                behavior["cake_clawbacks"][actor_index]
            )
        if episode_terminal:
            terminal = (
                self.feature_processes.consume_episode_terminal_shaping(
                    expected_frame=frame_no
                )
            )
            terminal_route = np.asarray(
                terminal["route_settlements"], dtype=np.float32
            )
            terminal_cake_pickup = np.asarray(
                terminal["cake_pickup_rewards"], dtype=np.float32
            )
            terminal_cake = np.asarray(
                terminal["cake_clawbacks"], dtype=np.float32
            )
            if (
                terminal_route.shape
                != (FrozenV1Config.CONTROLLED_ACTOR_COUNT,)
                or terminal_cake_pickup.shape
                != (FrozenV1Config.CONTROLLED_ACTOR_COUNT,)
                or terminal_cake.shape
                != (FrozenV1Config.CONTROLLED_ACTOR_COUNT,)
                or not np.isfinite(terminal_route).all()
                or not np.isfinite(terminal_cake_pickup).all()
                or not np.isfinite(terminal_cake).all()
            ):
                raise RuntimeError(
                    "episode terminal shaping must contain two finite rewards"
                )
            route = tuple(
                float(value) + float(terminal_value)
                for value, terminal_value in zip(route, terminal_route)
            )
            behavior_rewards += terminal_cake_pickup + terminal_cake
            self.jungle_route_shaping_total += float(
                terminal_route.sum()
            )
            self.cake_reward_total += float(
                terminal_cake_pickup.sum() + terminal_cake.sum()
            )
            self.cake_pickup_reward_total += float(
                terminal_cake_pickup.sum()
            )
            self.cake_clawback_total += float(terminal_cake.sum())
            for actor_index, info in enumerate(
                self.feature_processes._memo_context[
                    "controlled_infos"
                ]
            ):
                config_id = u.as_int(
                    u.safe_get(info, "config_id", default=0), 0
                )
                if config_id not in self.cake_reward_by_config:
                    continue
                terminal_pickup = float(terminal_cake_pickup[actor_index])
                terminal_value = terminal_pickup + float(
                    terminal_cake[actor_index]
                )
                self.cake_reward_by_config[config_id] += terminal_value
                self.cake_pickup_by_config[config_id] += terminal_pickup
                self.cake_clawback_by_config[config_id] += float(
                    terminal_cake[actor_index]
                )
        self.no_target_penalty_count += int(sum(behavior["no_target_counts"]))
        self.no_target_penalty_total += float(
            sum(behavior["no_target_penalties"])
        )
        self.no_target_grace_count += int(
            sum(behavior["no_target_grace_counts"])
        )
        self.strategic_wait_exemption_count += int(
            sum(behavior["strategic_wait_counts"])
        )
        self.long_inactivity_penalty_count += int(
            sum(behavior["long_inactivity_counts"])
        )
        self.long_inactivity_penalty_total += float(
            sum(behavior["long_inactivity_penalties"])
        )
        self.long_inactivity_tyrant_exemption_count += int(
            sum(behavior["long_inactivity_tyrant_exemptions"])
        )
        self.long_inactivity_stationary_count += int(
            sum(behavior["long_inactivity_stationary_counts"])
        )
        self.long_inactivity_wandering_count += int(
            sum(behavior["long_inactivity_wandering_counts"])
        )
        self.long_inactivity_no_damage_count += int(
            sum(behavior["long_inactivity_no_damage_counts"])
        )
        self.lane_attendance_shaping_total += float(
            sum(behavior["lane_attendance_rewards"])
        )
        self.lane_attendance_shaping_count += int(
            sum(behavior["lane_attendance_counts"])
        )
        self.last_kda_reward_by_runtime = dict(
            behavior["kda_rewards_by_runtime"]
        )
        self.last_kda_reward_by_config = dict(
            behavior["kda_rewards_by_config"]
        )
        self.last_tyrant_team_reward = float(behavior["tyrant_team_reward"])
        self.last_structure_team_reward = float(
            behavior["structure_team_reward"]
        )
        self.structure_positive_reward_total += float(
            behavior["structure_positive_team_reward"]
        )
        self.structure_negative_reward_total += float(
            behavior["structure_negative_team_reward"]
        )
        if behavior["structure_positive_team_reward"] > 0.0:
            self.structure_attribution_coverage_total += float(
                behavior["structure_attribution_coverage"]
            )
            self.structure_attribution_event_count += 1
        self.structure_attribution_fallback_count += int(
            behavior["structure_attribution_fallback"]
        )
        self.structure_attributed_hit_count += int(
            sum(behavior["structure_attributed_hit_counts"])
        )
        for config_id, reward_value, hit_count in zip(
            behavior.get("controlled_config_ids", ()),
            behavior["structure_actor_rewards"],
            behavior["structure_attributed_hit_counts"],
        ):
            config_id = int(config_id)
            if config_id in self.structure_actor_reward_by_config:
                self.structure_actor_reward_by_config[
                    config_id
                ] += float(reward_value)
                self.structure_actor_hit_by_config[
                    config_id
                ] += int(hit_count)
        self.last_post_kill_push_reward = float(
            behavior["post_kill_push_reward"]
        )
        self.last_crystal_outcome_reward = float(
            behavior["crystal_outcome_reward"]
        )
        self.last_crystal_danger_reward = float(
            behavior["crystal_danger_reward"]
        )
        self.last_defense_pressure_reward = float(
            behavior["defense_pressure_reward"]
        )
        self.last_temporal_focus_team_reward = float(
            behavior["temporal_focus_team_reward"]
        )
        self.last_group_attack_team_reward = float(
            behavior["group_attack_team_reward"]
        )
        self.last_command_teammate_abandonment_penalty = float(
            behavior["command_teammate_abandonment_penalty"]
        )
        self.last_command_teammate_support_reward = float(
            behavior["command_teammate_support_reward"]
        )
        self.last_command_official_kda_reward = float(
            behavior["command_official_kda_reward"]
        )
        self.last_command_goal_progress_reward = float(
            behavior["command_goal_progress_reward"]
        )
        self.last_command_nonresponse_penalty = float(
            behavior["command_nonresponse_penalty"]
        )
        self.last_command_no_command_idle_penalty = float(
            behavior["command_no_command_idle_penalty"]
        )
        self.last_command_farm_camp_reward = float(
            behavior["command_farm_camp_reward"]
        )
        self.command_farm_camp_reward_total += (
            self.last_command_farm_camp_reward
        )
        self.tower_risk_penalty_total += float(
            sum(behavior["tower_risk_rewards"])
        )
        self.tower_risk_hit_count += int(
            sum(behavior["tower_risk_hit_counts"])
        )
        self.defense_wave_reward_total += float(
            sum(behavior["defense_wave_rewards"])
        )
        self.defense_wave_completion_count += int(
            behavior["defense_wave_completed"]
        )
        # The training-side goal-progress reward intentionally includes the
        # bounded nonresponse penalty.  Keep the monitor-side response subtotal
        # disjoint so dashboards can add it to the separately reported
        # nonresponse subtotal without counting the penalty twice.
        self.command_response_reward_total += (
            self.last_command_goal_progress_reward
            - self.last_command_nonresponse_penalty
        )
        self.command_nonresponse_penalty_total += (
            self.last_command_nonresponse_penalty
        )
        self.command_nonresponse_penalty_count += int(
            behavior["command_nonresponse_applied"]
        )
        self.command_no_command_idle_penalty_total += (
            self.last_command_no_command_idle_penalty
        )
        self.command_no_command_idle_penalty_count += int(
            behavior["command_no_command_idle_applied"]
        )
        self.teammate_abandonment_penalty_total += float(
            sum(behavior["teammate_abandonment_rewards"])
        )
        self.teammate_abandonment_penalty_count += int(
            sum(behavior["teammate_abandonment_counts"])
        )
        self.teammate_support_reward_total += float(
            sum(behavior["teammate_support_rewards"])
        )
        self.teammate_support_count += int(
            sum(behavior["teammate_support_counts"])
        )
        self.temporal_focus_reward_total += (
            self.last_temporal_focus_team_reward
        )
        self.temporal_focus_count += int(
            behavior["temporal_focus_count"]
        )
        self.defense_pressure_reward_total += (
            self.last_defense_pressure_reward
        )
        self.defense_pressure_count += int(
            behavior["defense_pressure_count"]
        )
        self.group_attack_reward_total += (
            self.last_group_attack_team_reward
        )
        self.group_attack_count += int(behavior["group_attack_count"])
        self.crystal_danger_reward_total += (
            self.last_crystal_danger_reward
        )
        self.command_teammate_abandonment_penalty_total += (
            self.last_command_teammate_abandonment_penalty
        )
        self.command_teammate_abandonment_penalty_count += int(
            behavior["command_teammate_abandonment_count"]
        )
        self.command_teammate_support_reward_total += (
            self.last_command_teammate_support_reward
        )
        self.command_teammate_support_count += int(
            behavior["command_teammate_support_count"]
        )
        self.last_command_attribution_active = bool(
            behavior["command_attribution_active"]
        )
        self.last_command_attribution_intent = int(
            behavior["command_attribution_intent"]
        )
        self.custom_kda_reward_total += float(
            sum(behavior["kda_auxiliary_rewards"])
        )
        self.tyrant_reward_total += self.last_tyrant_team_reward
        self.kai_wolf_reward_total += float(
            behavior["kai_wolf_team_reward"]
        )
        self.kai_wolf_reward_count += int(
            behavior["kai_wolf_reward_count"]
        )
        self.treasure_thief_reward_total += float(
            behavior["treasure_thief_team_reward"]
        )
        self.treasure_thief_reward_count += int(
            behavior["treasure_thief_reward_count"]
        )
        self.speed_buff_reward_total += float(
            behavior["speed_buff_team_reward"]
        )
        self.speed_buff_reward_count += int(
            behavior["speed_buff_reward_count"]
        )
        self.structure_reward_total += self.last_structure_team_reward
        self.post_kill_push_reward_total += self.last_post_kill_push_reward
        self.terminal_crystal_reward_total += self.last_crystal_outcome_reward
        self.last_auxiliary_reward_shaping.fill(0.0)
        self.last_auxiliary_reward_shaping[:, 0] = np.asarray(
            behavior["farming_auxiliary_rewards"], dtype=np.float32
        )
        self.last_auxiliary_reward_shaping[:, 1] = np.asarray(
            behavior["kda_auxiliary_rewards"], dtype=np.float32
        )
        self.last_auxiliary_reward_shaping[:, 3] = np.asarray(
            behavior["pushing_auxiliary_rewards"], dtype=np.float32
        )
        self.last_auxiliary_reward_shaping[:, 4] = np.asarray(
            behavior["win_auxiliary_rewards"], dtype=np.float32
        )
        rewards = tuple(
            float(route_reward) + float(behavior_reward)
            for route_reward, behavior_reward in zip(route, behavior_rewards)
        )
        if len(rewards) != FrozenV1Config.CONTROLLED_ACTOR_COUNT:
            raise RuntimeError("custom reward shaping actor count drifted")
        return rewards

    def action_process(self, observation, act_data, stochastic):
        actor_decisions = (
            act_data.stochastic_actors
            if stochastic
            else act_data.deterministic_actors
        )
        command_decision = (
            act_data.stochastic_command
            if stochastic
            else act_data.deterministic_command
        )
        if actor_decisions is None or command_decision is None:
            mode = "stochastic" if stochastic else "deterministic"
            raise RuntimeError(f"ActDataV1 does not contain {mode} decisions")
        if self.collision_avoidance is not None:
            self.collision_avoidance.record_actions(
                u.frame_number(observation.get("frame_state", {}) or {}),
                self.obs_data.runtime_ids,
                np.stack([decision.action for decision in actor_decisions]),
            )
        hero_actions = {
            config_id: [int(value) for value in decision.action]
            for config_id, decision in zip(
                self.obs_data.config_ids, actor_decisions
            )
        }
        protocol_action = list(command_decision.protocol_action)
        if self.feature_processes is not None:
            self.feature_processes.record_collaboration_command(
                u.frame_number(observation.get("frame_state", {}) or {}),
                protocol_action,
                intent=int(command_decision.action[0]),
                is_train=bool(command_decision.is_train > 0.5),
                target_slot=int(command_decision.action[1]),
                position_slot=int(command_decision.action[2]),
            )
        return hero_actions, protocol_action

    def update_status(self, obs_data, act_data, stochastic=True):
        self.obs_data = obs_data
        self.act_data = act_data
        actor_decisions = (
            act_data.stochastic_actors
            if stochastic
            else act_data.deterministic_actors
        )
        if actor_decisions is None:
            mode = "stochastic" if stochastic else "deterministic"
            raise RuntimeError(f"ActDataV1 does not contain {mode} actors")
        for index, (config_id, runtime_id) in enumerate(
            zip(obs_data.config_ids, obs_data.runtime_ids)
        ):
            hero_info = obs_data.hero_infos[index]
            # A zero button mask is the protocol-side dead/non-actionable state.
            button_legal = np.asarray(
                hero_info.get("legal_action", []), dtype=np.float32
            ).reshape(-1)[:13]
            next_hidden = actor_decisions[index].next_hidden
            if button_legal.size != 13 or not np.any(button_legal > 0.5):
                next_hidden = np.zeros(128, dtype=np.float32)
            self.rnn_state_by_config[config_id] = {
                "runtime_id": runtime_id,
                "hidden": next_hidden.copy(),
            }

    def reset(self, observation):
        hero_infos = list(observation.get("heroes", []) or [])
        if not hero_infos:
            raise ValueError("Agent v1 reset requires hero observations")
        camp_value = next(
            (
                info.get("camp")
                for info in hero_infos
                if info.get("camp") is not None
                and u.normalize_camp(info.get("camp")) in (1, 2)
            ),
            None,
        )
        if camp_value is None:
            frame_actors = {
                u.actor_runtime_id(actor): actor
                for actor in observation.get("frame_state", {}).get(
                    "hero_states", []
                )
                if u.actor_runtime_id(actor)
            }
            frame_actor = next(
                (
                    frame_actors.get(
                        u.as_int(info.get("runtime_id", 0), 0)
                    )
                    for info in hero_infos
                    if frame_actors.get(
                        u.as_int(info.get("runtime_id", 0), 0)
                    )
                    is not None
                ),
                None,
            )
            camp_value = u.actor_camp(frame_actor) if frame_actor else 0
        self.hero_camp = u.normalize_camp(camp_value)
        if self.hero_camp not in (1, 2):
            raise ValueError(f"invalid reset camp: {self.hero_camp}")
        self.feature_processes = FeatureProcessV1(self.hero_camp)
        self.collision_avoidance = CollisionAvoidanceV1(self.hero_camp)
        self.jungle_opening_legalizer = JungleOpeningLegalizerV1()
        self.lane_opening_legalizer = LaneOpeningLegalizerV1()
        self.flash_legalizer = FlashLegalizerV1()
        self.safety_legalizer = SafetyLegalizerV1()
        self.respawn_reentry_legalizer = RespawnReentryLegalizerV1()
        self.tyrant_smite_legalizer = TyrantSmiteLegalizerV1()
        self.late_conversion_legalizer = LateConversionLegalizerV1()
        self.collision_diagnostics = ()
        self.jungle_opening_diagnostics = ()
        self.lane_opening_diagnostics = ()
        self.flash_diagnostics = ()
        self.safety_diagnostics = ()
        self.respawn_reentry_diagnostics = ()
        self.tyrant_smite_diagnostics = ()
        self.late_conversion_diagnostics = ()
        self.rnn_state_by_config.clear()
        self.obs_data = None
        self.act_data = None
        self.jungle_route_shaping_total = 0.0
        self.last_hit_reward_total = 0.0
        self.last_hit_count = 0
        self.no_target_penalty_total = 0.0
        self.no_target_penalty_count = 0
        self.no_target_grace_count = 0
        self.strategic_wait_exemption_count = 0
        self.lane_attendance_shaping_total = 0.0
        self.lane_attendance_shaping_count = 0
        self.custom_kda_reward_total = 0.0
        self.tyrant_reward_total = 0.0
        self.kai_wolf_reward_total = 0.0
        self.kai_wolf_reward_count = 0
        self.treasure_thief_reward_total = 0.0
        self.treasure_thief_reward_count = 0
        self.speed_buff_reward_total = 0.0
        self.speed_buff_reward_count = 0
        self.structure_reward_total = 0.0
        self.structure_positive_reward_total = 0.0
        self.structure_negative_reward_total = 0.0
        self.structure_attribution_coverage_total = 0.0
        self.structure_attribution_event_count = 0
        self.structure_attribution_fallback_count = 0
        self.structure_attributed_hit_count = 0
        self.structure_actor_reward_by_config = {
            config_id: 0.0 for config_id in u.HERO_CONFIG_IDS
        }
        self.structure_actor_hit_by_config = {
            config_id: 0 for config_id in u.HERO_CONFIG_IDS
        }
        self.post_kill_push_reward_total = 0.0
        self.terminal_crystal_reward_total = 0.0
        self.crystal_danger_reward_total = 0.0
        self.defense_pressure_reward_total = 0.0
        self.defense_pressure_count = 0
        self.temporal_focus_reward_total = 0.0
        self.temporal_focus_count = 0
        self.teammate_abandonment_penalty_total = 0.0
        self.teammate_abandonment_penalty_count = 0
        self.teammate_support_reward_total = 0.0
        self.teammate_support_count = 0
        self.group_attack_reward_total = 0.0
        self.group_attack_count = 0
        self.long_inactivity_penalty_total = 0.0
        self.long_inactivity_penalty_count = 0
        self.long_inactivity_tyrant_exemption_count = 0
        self.long_inactivity_stationary_count = 0
        self.long_inactivity_wandering_count = 0
        self.long_inactivity_no_damage_count = 0
        self.command_sparse_reward_total = 0.0
        self.late_monster_raw_total = 0.0
        self.late_monster_offset_total = 0.0
        self.late_monster_effective_total = 0.0
        self.cake_reward_total = 0.0
        self.cake_progress_reward_total = 0.0
        self.cake_pickup_reward_total = 0.0
        self.cake_clawback_total = 0.0
        self.cake_reward_by_config = {
            config_id: 0.0 for config_id in u.HERO_CONFIG_IDS
        }
        self.cake_progress_by_config = {
            config_id: 0.0 for config_id in u.HERO_CONFIG_IDS
        }
        self.cake_pickup_by_config = {
            config_id: 0.0 for config_id in u.HERO_CONFIG_IDS
        }
        self.cake_clawback_by_config = {
            config_id: 0.0 for config_id in u.HERO_CONFIG_IDS
        }
        self.last_kda_reward_by_runtime = {}
        self.last_kda_reward_by_config = {}
        self.last_tyrant_team_reward = 0.0
        self.last_structure_team_reward = 0.0
        self.last_post_kill_push_reward = 0.0
        self.last_crystal_outcome_reward = 0.0
        self.last_crystal_danger_reward = 0.0
        self.last_defense_pressure_reward = 0.0
        self.last_temporal_focus_team_reward = 0.0
        self.last_group_attack_team_reward = 0.0
        self.last_command_teammate_abandonment_penalty = 0.0
        self.last_command_teammate_support_reward = 0.0
        self.last_command_official_kda_reward = 0.0
        self.last_command_goal_progress_reward = 0.0
        self.last_command_nonresponse_penalty = 0.0
        self.last_command_no_command_idle_penalty = 0.0
        self.last_command_farm_camp_reward = 0.0
        self.command_farm_camp_reward_total = 0.0
        self.tower_risk_penalty_total = 0.0
        self.tower_risk_hit_count = 0
        self.defense_wave_reward_total = 0.0
        self.defense_wave_completion_count = 0
        self.command_response_reward_total = 0.0
        self.command_nonresponse_penalty_total = 0.0
        self.command_nonresponse_penalty_count = 0
        self.command_no_command_idle_penalty_total = 0.0
        self.command_no_command_idle_penalty_count = 0
        self.command_teammate_abandonment_penalty_total = 0.0
        self.command_teammate_abandonment_penalty_count = 0
        self.command_teammate_support_reward_total = 0.0
        self.command_teammate_support_count = 0
        self.last_command_attribution_active = False
        self.last_command_attribution_intent = 0
        self.last_reward_boundary_frame = None
        self.last_reward_boundary_is_new = False
        self.last_auxiliary_reward_shaping.fill(0.0)

    def learn(self, list_sample_data):
        return self.algorithm.learn(list_sample_data)

    def _apply_configured_learning_rate(self):
        """Keep a resumed learner on the active training-stage learning rate."""

        learning_rate = float(V1TrainingConfig.LEARNING_RATE)
        self.optimizer.defaults["lr"] = learning_rate
        for group in self.optimizer.param_groups:
            group["lr"] = learning_rate
            if "initial_lr" in group:
                group["initial_lr"] = learning_rate

    def save_model(self, path=None, id="1"):
        if path is None:
            raise ValueError("save_model requires a checkpoint directory")
        model_file_path = os.path.join(path, f"model.ckpt-{id}.pkl")
        save_checkpoint_v1(
            model_file_path,
            self.model,
            optimizer=self.optimizer,
            train_step=self.algorithm.train_step,
            metadata={"camp": self.hero_camp},
        )
        if self.logger:
            self.logger.info(f"save hybrid multi-value model {model_file_path} successfully")

    def load_model(self, path=None, id="1"):
        if path is None:
            raise ValueError("load_model requires a checkpoint directory")
        model_file_path = os.path.join(path, f"model.ckpt-{id}.pkl")
        if self.cur_model_name == model_file_path:
            if self.logger:
                self.logger.info(f"current model is {model_file_path}, skip reload")
            return
        restore_optimizer = self.runtime_agent_type.lower() == "learner"
        result = load_checkpoint_v1(
            model_file_path,
            self.model,
            optimizer=self.optimizer if restore_optimizer else None,
            map_location=self.device,
            load_optimizer=restore_optimizer,
        )
        if restore_optimizer:
            # Preserve Adam moments and train_step, but do not let the source
            # checkpoint's param-group LR override the active PPO 4.0 stage.
            self._apply_configured_learning_rate()
        self.algorithm.train_step = result["train_step"]
        self.cur_model_name = model_file_path
        self.loaded_checkpoint_identity = (
            result["signature_fingerprint"],
            int(result["train_step"]),
            os.path.basename(model_file_path),
        )
        if self.logger:
            if result.get("compatibility_migration"):
                self.logger.info(
                    "loaded checkpoint through compatibility migration "
                    f"{result['compatibility_migration']}; current six-camp "
                    "opening behavior remains active"
                )
            self.logger.info(f"load hybrid multi-value model {model_file_path} successfully")

    @load_opponent_agent_wrapper
    def load_opponent_agent(self, id="1"):
        pass


__all__ = ("Agent", "NONE_ACTION", "NONE_COL_ACTION")
