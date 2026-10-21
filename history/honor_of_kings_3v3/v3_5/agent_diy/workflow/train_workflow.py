#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""Official-platform runner for official-model jungle diagnostics."""

import os

from agent_ppo.agent import NONE_ACTION, NONE_COL_ACTION
from agent_ppo.feature.feature_process import v1_utils as u

from agent_diy.debug.official_jungle_probe import (
    OfficialJungleProbe,
    selected_scenarios,
)
from agent_diy.debug.official_jungle_trace import (
    OfficialJungleTraceRecorder,
    policy_command_snapshot,
)
from common_python.utils.workflow_disaster_recovery import (
    handle_disaster_recovery,
)
from tools.env_conf_manager import EnvConfManager


def workflow(envs, agents, logger=None, monitor=None, *args, **kwargs):
    runner = OfficialJungleDebugRunner(
        env=envs[0],
        agents=agents,
        logger=logger,
        env_conf_manager=EnvConfManager(
            config_path="agent_diy/conf/train_env_conf.toml",
            logger=logger,
        ),
    )
    runner.run()


class OfficialJungleDebugRunner:
    """Run an attribution-safe platform experiment matrix."""

    def __init__(self, env, agents, logger, env_conf_manager):
        self.env = env
        self.agents = agents
        self.logger = logger
        self.env_conf_manager = env_conf_manager
        self.max_frame = int(os.environ.get("DEBUG_MAX_ENV_FRAME", "2200"))
        self.repeats = max(1, int(os.environ.get("DEBUG_REPEATS", "1")))
        self.policy_mode = (
            os.environ.get("DEBUG_POLICY_MODE", "stochastic").strip().lower()
        )
        if self.policy_mode not in ("stochastic", "deterministic"):
            raise ValueError("DEBUG_POLICY_MODE must be stochastic or deterministic")
        self.scenarios = selected_scenarios()
        target_attempts = int(os.environ.get("DEBUG_TARGET_ATTEMPTS", "4"))
        self.probes = [
            OfficialJungleProbe(target_attempt_limit=target_attempts)
            for _agent in agents
        ]
        self.recorders = [
            OfficialJungleTraceRecorder(
                getattr(agent, "runtime_agent_type", f"agent_{index}"),
                logger=logger,
            )
            for index, agent in enumerate(agents)
        ]

    def run(self):
        try:
            for repeat_index in range(self.repeats):
                for scenario_index, scenario in enumerate(self.scenarios):
                    if not self._run_episode(
                        scenario=scenario,
                        repeat_index=repeat_index,
                        scenario_index=scenario_index,
                    ):
                        return
        finally:
            for recorder in self.recorders:
                recorder.close()

    def _run_episode(self, scenario, repeat_index, scenario_index):
        lineup = [
            scenario.official_config_id,
            scenario.official_config_id,
        ]
        usr_conf, _, monitor_side = self.env_conf_manager.update_config(lineup)
        env_obs = self.env.reset(usr_conf=usr_conf)
        if handle_disaster_recovery(env_obs, self.logger):
            return False

        observation = env_obs["observation"]
        ready = self._reset_agents(observation, scenario)
        if self.logger:
            self.logger.info(
                "DIY official jungle debug started; "
                f"scenario={scenario.name}, repeat={repeat_index + 1}/"
                f"{self.repeats}, scenario_index={scenario_index + 1}/"
                f"{len(self.scenarios)}, official blue/red={lineup}, "
                f"policy_mode={self.policy_mode}"
            )

        final_frame = int(env_obs.get("frame_no", 0))
        while True:
            hero_actions = []
            collaboration_actions = []
            for agent_index, agent in enumerate(self.agents):
                obs = observation.get(str(agent_index))
                if not self._valid_observation(obs):
                    hero_actions.append(NONE_ACTION)
                    collaboration_actions.append(NONE_COL_ACTION)
                    continue
                if not ready[agent_index]:
                    agent.reset(obs)
                    self.probes[agent_index].reset(scenario)
                    self.recorders[agent_index].start_episode(
                        obs,
                        agent.hero_camp,
                        scenario,
                    )
                    ready[agent_index] = True

                inference = (
                    agent.predict if self.policy_mode == "stochastic" else agent.exploit
                )
                hero_action, proposed_command = inference(obs)
                policy_decision = policy_command_snapshot(
                    agent,
                    proposed_command,
                )
                actual_command, diagnostic = self.probes[agent_index].apply(
                    proposed_command=proposed_command,
                    observation=obs,
                    camp=agent.hero_camp,
                    config_ids=agent.obs_data.config_ids,
                    feature_process=agent.feature_processes,
                    policy_decision=policy_decision,
                )
                if scenario.freeze_controlled_heroes:
                    hero_action = {
                        int(config_id): [0, 0, 0, 0, 0]
                        for config_id in agent.obs_data.config_ids
                    }
                elif scenario.run_controlled_kai_only:
                    hero_action = {
                        int(config_id): (
                            list(hero_action[int(config_id)])
                            if int(config_id) == 193
                            else [0, 0, 0, 0, 0]
                        )
                        for config_id in agent.obs_data.config_ids
                    }
                self._record_actual_collaboration_override(
                    agent=agent,
                    observation=obs,
                    proposed=proposed_command,
                    actual=actual_command,
                )
                self.recorders[agent_index].record_step(
                    observation=obs,
                    camp=agent.hero_camp,
                    scenario=scenario,
                    controlled_config_ids=agent.obs_data.config_ids,
                    proposed_command=proposed_command,
                    actual_command=actual_command,
                    controlled_actions=hero_action,
                    policy_decision=policy_decision,
                    feature_process=agent.feature_processes,
                    diagnostic=diagnostic,
                )
                hero_actions.append(hero_action)
                collaboration_actions.append(actual_command)

            _, env_obs = self.env.step([hero_actions, collaboration_actions])
            if handle_disaster_recovery(env_obs, self.logger):
                return False
            observation = env_obs["observation"]
            final_frame = int(env_obs["frame_no"])
            if (
                env_obs["terminated"]
                or env_obs["truncated"]
                or final_frame >= self.max_frame
            ):
                for agent_index, recorder in enumerate(self.recorders):
                    if ready[agent_index]:
                        recorder.end_episode(
                            final_frame,
                            self.probes[agent_index].summary(),
                        )
                if self.logger:
                    self.logger.info(
                        "DIY official jungle debug ended; "
                        f"scenario={scenario.name}, frame={final_frame}, "
                        f"monitor_side={monitor_side}"
                    )
                return True

    def _reset_agents(self, observation, scenario):
        ready = [False] * len(self.agents)
        for index, agent in enumerate(self.agents):
            self.probes[index].reset(scenario)
            obs = observation.get(str(index))
            if self._valid_observation(obs):
                agent.reset(obs)
                self.recorders[index].start_episode(
                    obs,
                    agent.hero_camp,
                    scenario,
                )
                ready[index] = True
        return ready

    @staticmethod
    def _record_actual_collaboration_override(
        agent,
        observation,
        proposed,
        actual,
    ):
        """Keep PPO's local echo throttle aligned with the actual DIY wire."""

        proposed = [int(value) for value in proposed]
        actual = [int(value) for value in actual]
        if proposed == actual:
            return
        feature_process = agent.feature_processes
        frame = u.frame_number(observation.get("frame_state", {}) or {})
        if actual[0] in (1, 2, 3):
            feature_process.record_collaboration_command(
                frame,
                actual,
                intent=0,
                is_train=False,
            )
            return

        pending = getattr(
            feature_process,
            "pending_collaboration_command",
            None,
        )
        if pending is not None and int(pending.get("frame", -1)) == frame:
            feature_process.pending_collaboration_command = None
            collaboration = (getattr(feature_process, "_memo_context", None) or {}).get(
                "collaboration", {}
            ) or {}
            feature_process.local_last_command_frame = int(
                collaboration.get("platform_last_send_frame", 0)
            )
        feature_process.last_command_decision = {
            "frame": frame,
            "action_type": 0,
            "intent": 0,
            "is_train": False,
        }

    @staticmethod
    def _valid_observation(observation):
        return (
            isinstance(observation, dict)
            and bool(observation.get("heroes"))
            and bool(observation.get("frame_state"))
        )


__all__ = ("OfficialJungleDebugRunner", "workflow")
