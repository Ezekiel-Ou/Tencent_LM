#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""Active agent: two controlled actors and one learned team command."""

import os
import numpy as np
import torch

# Platform actor processes expose many CPU cores.  Small time=1 inference is
# severely slowed by per-op thread-pool fan-out, so keep the official baseline
# actor setting.  This changes execution only, never model/sample semantics.
torch.set_num_threads(1)
try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    # PyTorch permits setting inter-op threads only before parallel work starts.
    pass

try:
    from kaiwudrl.interface.agent import BaseAgent
except ModuleNotFoundError:  # Local contract/smoke tests do not ship the SDK.
    class BaseAgent:
        def __init__(self, agent_type="player", device=None, logger=None, monitor=None):
            self.agent_type = agent_type


from agent_ppo.algorithm.algorithm_v1 import AlgorithmV1
from agent_ppo.conf.conf import FrozenV1Config, V1TrainingConfig
from agent_ppo.feature.collaboration_v1 import (
    CommandRuleLegalizerV1,
    build_command_legal_masks_v1,
)
from agent_ppo.feature.collision_avoidance_v1 import CollisionAvoidanceV1
from agent_ppo.feature.lane_opening_v1 import LaneOpeningLegalizerV1
from agent_ppo.feature.jungle_opening_v1 import JungleOpeningLegalizerV1
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
        self.hero_camp = None
        self.rnn_state_by_config = {}
        self.cur_model_name = ""
        self.obs_data = None
        self.act_data = None
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
        feature = np.stack(
            self.feature_processes.process_feature(observation), axis=0
        ).astype(np.float32, copy=False)
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

    def _model_inference(self, list_obs_data):
        if len(list_obs_data) != 1:
            raise ValueError("Agent v1 inference expects one environment observation")
        obs_data = list_obs_data[0]
        feature = torch.from_numpy(obs_data.feature).to(self.device).unsqueeze(0).unsqueeze(2)
        raw_legal = (
            torch.from_numpy(obs_data.raw_legal_action)
            .to(self.device)
            .unsqueeze(0)
            .unsqueeze(2)
        )
        hidden = torch.from_numpy(obs_data.rnn_hidden).to(self.device).unsqueeze(0)
        intent_mask = torch.from_numpy(obs_data.command_masks.intent).to(self.device).reshape(
            1, 1, FrozenV1Config.COMMAND_HEADS[0]
        )
        target_masks = (
            torch.from_numpy(obs_data.command_masks.target_by_intent)
            .to(self.device)
            .reshape(1, 1, FrozenV1Config.COMMAND_HEADS[0], FrozenV1Config.COMMAND_HEADS[1])
        )
        position_masks = (
            torch.from_numpy(obs_data.command_masks.position_by_intent)
            .to(self.device)
            .reshape(1, 1, FrozenV1Config.COMMAND_HEADS[0], FrozenV1Config.COMMAND_HEADS[2])
        )
        command_is_train = torch.tensor(
            [[float(obs_data.command_masks.command_is_train)]],
            device=self.device,
        )
        self.model.eval()
        with torch.no_grad():
            output = self.model(feature, hidden)
            stochastic_actor = sample_actor_actions_v1(
                output["actor"], raw_legal, deterministic=False
            )
            deterministic_actor = sample_actor_actions_v1(
                output["actor"], raw_legal, deterministic=True
            )
            stochastic_command = sample_command_actions_v1(
                output["command"],
                intent_mask,
                target_masks,
                position_masks,
                command_is_train=command_is_train,
                deterministic=False,
            )
            deterministic_command = sample_command_actions_v1(
                output["command"],
                intent_mask,
                target_masks,
                position_masks,
                command_is_train=command_is_train,
                deterministic=True,
            )
        next_hidden = output["actor"]["next_hidden"]
        value = output["actor"]["actor_value"]
        auxiliary_values = output["actor"]["actor_auxiliary_values"]
        stochastic_actors = self._actor_decisions(
            stochastic_actor,
            value,
            auxiliary_values,
            obs_data.rnn_hidden,
            next_hidden,
        )
        deterministic_actors = self._actor_decisions(
            deterministic_actor,
            value,
            auxiliary_values,
            obs_data.rnn_hidden,
            next_hidden,
        )
        return ActDataV1(
            stochastic_actors=stochastic_actors,
            deterministic_actors=deterministic_actors,
            stochastic_command=self._command_decision(
                stochastic_command,
                output["command"]["command_value"],
                obs_data.command_context,
                obs_data.command_masks,
            ),
            deterministic_command=self._command_decision(
                deterministic_command,
                output["command"]["command_value"],
                obs_data.command_context,
                obs_data.command_masks,
            ),
        )

    def predict(self, observation):
        obs_data = self.observation_process(observation)
        act_data = self._model_inference([obs_data])
        self.update_status(obs_data, act_data)
        actions = self.action_process(observation, act_data, stochastic=True)
        return actions

    def exploit(self, observation):
        obs_data = self.observation_process(observation)
        act_data = self._model_inference([obs_data])
        self.update_status(obs_data, act_data)
        actions = self.action_process(observation, act_data, stochastic=False)
        return actions

    def bootstrap_values(self, observation):
        """Evaluate a truncation state without emitting or recording an action."""

        obs_data = self.observation_process(observation)
        feature = torch.from_numpy(obs_data.feature).to(self.device).unsqueeze(0).unsqueeze(2)
        hidden = torch.from_numpy(obs_data.rnn_hidden).to(self.device).unsqueeze(0)
        self.model.eval()
        with torch.no_grad():
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
        self.feature_processes.process_feature(observation)
        frame_state = observation.get("frame_state", {}) or {}
        rewards = self.feature_processes.consume_jungle_route_shaping(
            expected_frame=u.frame_number(frame_state)
        )
        self.jungle_route_shaping_total += float(sum(rewards))
        return rewards

    def consume_reward_shaping(self, observation):
        """Consume all custom scalar shaping exactly once at this boundary."""

        if self.feature_processes is None:
            self.last_auxiliary_reward_shaping.fill(0.0)
            return (0.0,) * FrozenV1Config.CONTROLLED_ACTOR_COUNT
        self.feature_processes.process_feature(observation)
        frame_state = observation.get("frame_state", {}) or {}
        frame_no = u.frame_number(frame_state)
        route = self.consume_jungle_route_shaping(observation)
        behavior = self.feature_processes.consume_behavior_shaping(
            expected_frame=frame_no
        )
        behavior_rewards = tuple(behavior["rewards"])
        self.last_hit_count += int(sum(behavior["last_hit_counts"]))
        self.last_hit_reward_total += float(sum(behavior["last_hit_rewards"]))
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
        self.lane_attendance_shaping_total += float(
            sum(behavior["lane_attendance_rewards"])
        )
        self.lane_attendance_shaping_count += int(
            sum(behavior["lane_attendance_counts"])
        )
        self.last_auxiliary_reward_shaping.fill(0.0)
        self.last_auxiliary_reward_shaping[:, 0] = np.asarray(
            behavior["farming_auxiliary_rewards"], dtype=np.float32
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
            )
        return hero_actions, protocol_action

    def update_status(self, obs_data, act_data):
        self.obs_data = obs_data
        self.act_data = act_data
        for index, (config_id, runtime_id) in enumerate(
            zip(obs_data.config_ids, obs_data.runtime_ids)
        ):
            hero_info = obs_data.hero_infos[index]
            # A zero button mask is the protocol-side dead/non-actionable state.
            button_legal = np.asarray(
                hero_info.get("legal_action", []), dtype=np.float32
            ).reshape(-1)[:13]
            next_hidden = act_data.stochastic_actors[index].next_hidden
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
        camp_value = hero_infos[0].get("camp")
        if camp_value is None:
            first_runtime = u.as_int(hero_infos[0].get("runtime_id", 0), 0)
            frame_actor = next(
                (
                    actor
                    for actor in observation.get("frame_state", {}).get(
                        "hero_states", []
                    )
                    if u.actor_runtime_id(actor) == first_runtime
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
        self.collision_diagnostics = ()
        self.jungle_opening_diagnostics = ()
        self.lane_opening_diagnostics = ()
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
        self.last_auxiliary_reward_shaping.fill(0.0)

    def learn(self, list_sample_data):
        return self.algorithm.learn(list_sample_data)

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
        self.algorithm.train_step = result["train_step"]
        self.cur_model_name = model_file_path
        if self.logger:
            self.logger.info(f"load hybrid multi-value model {model_file_path} successfully")

    def load_opponent_agent(self, id="1"):
        pass


__all__ = ("Agent", "NONE_ACTION", "NONE_COL_ACTION")
