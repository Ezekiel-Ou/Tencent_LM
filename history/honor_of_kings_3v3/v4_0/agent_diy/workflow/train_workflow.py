#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""Official-platform workflow for official-model lane skill diagnostics."""

import os

from agent_ppo.agent import NONE_ACTION, NONE_COL_ACTION
from common_python.utils.workflow_disaster_recovery import (
    handle_disaster_recovery,
)
from tools.env_conf_manager import EnvConfManager

from agent_diy.debug.official_lane_skill_trace import (
    OfficialLaneSkillTraceRecorder,
    policy_command_snapshot,
    selected_scenarios,
)


def workflow(envs, agents, logger=None, monitor=None, *args, **kwargs):
    runner = OfficialLaneSkillDebugRunner(
        env=envs[0],
        agents=agents,
        logger=logger,
        env_conf_manager=EnvConfManager(
            config_path="agent_diy/conf/train_env_conf.toml",
            logger=logger,
        ),
    )
    runner.run()


class OfficialLaneSkillDebugRunner:
    """Run normal lane play while recording official skill-direction factors."""

    def __init__(self, env, agents, logger, env_conf_manager):
        self.env = env
        self.agents = agents
        self.logger = logger
        self.env_conf_manager = env_conf_manager
        self.max_frame = int(os.environ.get("DEBUG_MAX_ENV_FRAME", "4500"))
        self.repeats = max(1, int(os.environ.get("DEBUG_REPEATS", "3")))
        self.policy_mode = os.environ.get(
            "DEBUG_POLICY_MODE", "stochastic"
        ).strip().lower()
        if self.policy_mode not in ("stochastic", "deterministic"):
            raise ValueError("DEBUG_POLICY_MODE must be stochastic or deterministic")
        self.scenarios = selected_scenarios()
        self.recorders = [
            OfficialLaneSkillTraceRecorder(
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
        lineup = [scenario.official_config_id, scenario.official_config_id]
        usr_conf, _, monitor_side = self.env_conf_manager.update_config(lineup)
        env_obs = self.env.reset(usr_conf=usr_conf)
        if handle_disaster_recovery(env_obs, self.logger):
            return False

        observation = env_obs["observation"]
        ready = self._reset_agents(observation, scenario)
        if self.logger:
            self.logger.info(
                "DIY official lane skill debug started; "
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
                    self.recorders[agent_index].start_episode(
                        obs, agent.hero_camp, scenario
                    )
                    ready[agent_index] = True

                inference = (
                    agent.predict
                    if self.policy_mode == "stochastic"
                    else agent.exploit
                )
                hero_action, command = inference(obs)
                policy_decision = policy_command_snapshot(agent, command)
                self.recorders[agent_index].record_step(
                    observation=obs,
                    camp=agent.hero_camp,
                    scenario=scenario,
                    proposed_command=command,
                    actual_command=command,
                    controlled_actions=hero_action,
                    policy_decision=policy_decision,
                    feature_process=agent.feature_processes,
                )
                hero_actions.append(hero_action)
                collaboration_actions.append(command)

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
                        recorder.end_episode(final_frame)
                if self.logger:
                    self.logger.info(
                        "DIY official lane skill debug ended; "
                        f"scenario={scenario.name}, frame={final_frame}, "
                        f"monitor_side={monitor_side}"
                    )
                return True

    def _reset_agents(self, observation, scenario):
        ready = [False] * len(self.agents)
        for index, agent in enumerate(self.agents):
            obs = observation.get(str(index))
            if self._valid_observation(obs):
                agent.reset(obs)
                self.recorders[index].start_episode(
                    obs, agent.hero_camp, scenario
                )
                ready[index] = True
        return ready

    @staticmethod
    def _valid_observation(observation):
        return (
            isinstance(observation, dict)
            and bool(observation.get("heroes"))
            and bool(observation.get("frame_state"))
        )


__all__ = ("OfficialLaneSkillDebugRunner", "workflow")
