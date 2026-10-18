#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Training workflow for Gorge Chase PPO (auto-curriculum).

峡谷追猎 PPO 训练工作流（自动课程切换）。

课程推进（默认自动）：
  STAGE_EPISODE_THRESHOLDS = [8000, 22000, 52000]
    episode_cnt <  8000  -> stage 1
    episode_cnt <  22000 -> stage 2
    episode_cnt <  52000 -> stage 3
    episode_cnt >= 52000 -> stage 4

手动覆盖：
  设置环境变量 CURRICULUM_STAGE_OVERRIDE=1..4 则强制固定阶段，适合调试或指定阶段深度训练。

监控说明：
  episode_cnt 是 EpisodeRunner 本地 session 计数。如果训练任务中途重启，计数会回零，
  阶段会重新从 1 开始。若希望跨重启保持阶段，可通过 CURRICULUM_STAGE_OVERRIDE 强制。
"""

import os
import time

import numpy as np

from agent_ppo.conf.conf import Config
from agent_ppo.feature.definition import SampleData, sample_process
from tools.metrics_utils import get_training_metrics
from tools.train_env_conf_validate import read_usr_conf
from common_python.utils.workflow_disaster_recovery import handle_disaster_recovery


def _determine_stage(episode_cnt):
    """Return stage (1..4) based on config mode, override, or episode count."""
    if not Config.CURRICULUM_AUTO:
        return 1
    override = Config.CURRICULUM_STAGE_OVERRIDE
    if isinstance(override, int) and 1 <= override <= 4:
        return override
    thresholds = Config.STAGE_EPISODE_THRESHOLDS
    for i, thr in enumerate(thresholds):
        if episode_cnt < thr:
            return i + 1
    return 4


def workflow(envs, agents, logger=None, monitor=None, *args, **kwargs):
    last_save_model_time = time.time()
    env = envs[0]
    agent = agents[0]

    runner = EpisodeRunner(
        env=env,
        agent=agent,
        logger=logger,
        monitor=monitor,
    )

    while True:
        for g_data in runner.run_episodes():
            agent.send_sample_data(g_data)
            g_data.clear()

            now = time.time()
            if now - last_save_model_time >= 1800:
                agent.save_model()
                last_save_model_time = now


class EpisodeRunner:
    def __init__(self, env, agent, logger, monitor):
        self.env = env
        self.agent = agent
        self.logger = logger
        self.monitor = monitor

        self.episode_cnt = 0
        self.current_stage = None
        self._stage_conf_cache = {}  # stage -> usr_conf dict
        self.last_report_monitor_time = 0
        self.last_get_training_metrics_time = 0
        self._verify_switch_on_next_episode = True  # 首局也做一次自检

        # 进度持久化：跨训练重启保持 episode_cnt，避免阶段退回
        # 文件放在 agent_ppo 目录内，训练容器内一般可写
        self._progress_file = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "..", ".curriculum_progress",
        )
        self._progress_save_every = 20  # 每 20 局写一次（降 I/O）
        self._load_progress()

        if logger:
            if not Config.CURRICULUM_AUTO:
                mode = "manual_default"
            else:
                mode = ("override=" + str(Config.CURRICULUM_STAGE_OVERRIDE)
                        if Config.CURRICULUM_STAGE_OVERRIDE else "auto")
            logger.info(
                f"[curriculum] mode={mode} thresholds={Config.STAGE_EPISODE_THRESHOLDS} "
                f"resumed_episode_cnt={self.episode_cnt}"
            )

    # ----------------------------------------------------------------
    # Progress persistence
    # ----------------------------------------------------------------
    def _load_progress(self):
        try:
            if os.path.exists(self._progress_file):
                with open(self._progress_file, "r") as f:
                    self.episode_cnt = int(f.read().strip() or "0")
        except Exception as e:
            self.episode_cnt = 0
            if self.logger:
                self.logger.warning(f"[curriculum] load progress failed: {e}")

    def _save_progress(self):
        try:
            tmp = self._progress_file + ".tmp"
            with open(tmp, "w") as f:
                f.write(str(self.episode_cnt))
            os.replace(tmp, self._progress_file)
        except Exception as e:
            if self.logger:
                self.logger.warning(f"[curriculum] save progress failed: {e}")

    # ----------------------------------------------------------------
    # Stage config management
    # ----------------------------------------------------------------
    def _get_stage_conf(self, stage):
        """Load usr_conf for a stage, or the default config when curriculum is disabled."""
        if stage in self._stage_conf_cache:
            return self._stage_conf_cache[stage]

        if not Config.CURRICULUM_AUTO:
            conf = read_usr_conf("agent_ppo/conf/train_env_conf.toml", self.logger)
            if conf is None and self.logger:
                self.logger.error(
                    "[curriculum] CURRICULUM_AUTO=False but default conf load failed"
                )
            self._stage_conf_cache[stage] = conf
            return conf

        primary = f"agent_ppo/conf/train_env_conf_stage{stage}.toml"
        conf = read_usr_conf(primary, self.logger)
        if conf is None and self.logger:
            self.logger.warning(
                f"[curriculum] failed to read {primary}, fallback to stage4 then default"
            )
        if conf is None:
            conf = read_usr_conf("agent_ppo/conf/train_env_conf_stage4.toml", self.logger)
        if conf is None:
            conf = read_usr_conf("agent_ppo/conf/train_env_conf.toml", self.logger)
        if conf is None and self.logger:
            self.logger.error(
                "[curriculum] all conf fallbacks failed; env.reset will likely crash"
            )

        self._stage_conf_cache[stage] = conf
        return conf

    def _maybe_switch_stage(self):
        """Check whether stage should change; log and update Config.CURRICULUM_STAGE if so."""
        target = _determine_stage(self.episode_cnt)
        if target != self.current_stage:
            prev = self.current_stage
            self.current_stage = target
            Config.CURRICULUM_STAGE = target
            # 标记"下一局需要做切换自检"
            self._verify_switch_on_next_episode = True
            if self.logger:
                self.logger.info(
                    f"[curriculum] stage switch {prev} -> {target} "
                    f"at episode_cnt={self.episode_cnt}"
                )
            # 监控上报课程切换（即时）
            if self.monitor:
                self.monitor.put_data({os.getpid(): {
                    "curriculum_stage": int(target),
                    "episode_cnt": int(self.episode_cnt),
                }})
        return self.current_stage

    def _verify_stage_switch(self, usr_conf, env_obs, stage):
        """Verify env actually picked up the new usr_conf (vs silently caching the old one).

        阶段切换后首个 episode 的自检：把我们传的 usr_conf 与 env 反馈的 env_info
        做字段对比。若不一致，说明 env.reset 未真正应用新配置。

        结果用 WARNING 级别写日志（避免 INFO 被框架限流丢失），
        并同时追加到 .curriculum_verify.log 独立文件，确保可追溯。
        """
        try:
            env_info = env_obs.get("observation", {}).get("env_info", {}) or {}
            conf = (usr_conf or {}).get("env_conf", {}) if isinstance(usr_conf, dict) else {}

            want_max_step = int(conf.get("max_step", -1))
            got_max_step = int(env_info.get("max_step", -1))
            want_total_tre = int(conf.get("treasure_count", -1))
            got_total_tre = int(env_info.get("total_treasure", -1))
            want_mon_int = int(conf.get("monster_interval", -1))
            got_mon_int = int(env_info.get("monster_interval", -1))

            matches = []
            mismatches = []
            for name, w, g in [
                ("max_step", want_max_step, got_max_step),
                ("total_treasure", want_total_tre, got_total_tre),
                ("monster_interval", want_mon_int, got_mon_int),
            ]:
                if w >= 0 and g >= 0:
                    if w == g:
                        matches.append(f"{name}={g}")
                    else:
                        mismatches.append(f"{name}: expected={w} actual={g}")

            if not mismatches:
                msg = (f"[curriculum-verify] stage {stage} SWITCH CONFIRMED: "
                       f"{', '.join(matches)}")
            else:
                msg = (f"[curriculum-verify] stage {stage} FAKE SWITCH "
                       f"(env ignored new usr_conf!) "
                       f"mismatches: {'; '.join(mismatches)} | "
                       f"matches: {', '.join(matches) or 'none'}")

            # 写 WARNING 级别日志（大概率不被框架限流）
            if self.logger:
                if mismatches:
                    self.logger.error(msg)
                else:
                    self.logger.warning(msg)

            # 同时追加到独立文件，避免日志系统限流
            try:
                verify_file = os.path.join(
                    os.path.dirname(os.path.abspath(__file__)),
                    "..", ".curriculum_verify.log",
                )
                with open(verify_file, "a") as f:
                    import datetime
                    ts = datetime.datetime.now().isoformat(timespec="seconds")
                    f.write(f"{ts} pid={os.getpid()} {msg}\n")
            except Exception as e:
                if self.logger:
                    self.logger.warning(f"[curriculum-verify] file write failed: {e}")

            if self.monitor:
                self.monitor.put_data({os.getpid(): {
                    "stage_switch_real": int(len(mismatches) == 0),
                }})
        except Exception as e:
            if self.logger:
                self.logger.warning(f"[curriculum-verify] check failed: {e}")

    # ----------------------------------------------------------------
    # Main loop
    # ----------------------------------------------------------------
    def run_episodes(self):
        while True:
            now = time.time()
            if now - self.last_get_training_metrics_time >= 60:
                training_metrics = get_training_metrics()
                self.last_get_training_metrics_time = now
                if training_metrics is not None and self.logger:
                    self.logger.info(f"training_metrics={training_metrics}")

            # 确定当前阶段并取配置
            stage = self._maybe_switch_stage()
            usr_conf = self._get_stage_conf(stage)
            if usr_conf is None:
                if self.logger:
                    self.logger.error("[curriculum] usr_conf None, sleeping 5s")
                time.sleep(5)
                continue

            env_obs = self.env.reset(usr_conf)
            if handle_disaster_recovery(env_obs, self.logger):
                continue

            # 切换阶段后的首局做一次自检（对比 usr_conf 与 env_info）
            if self._verify_switch_on_next_episode:
                self._verify_stage_switch(usr_conf, env_obs, stage)
                self._verify_switch_on_next_episode = False

            self.agent.reset(env_obs)
            self.agent.load_model(id="latest")

            obs_data, remain_info = self.agent.observation_process(env_obs)

            collector = []
            self.episode_cnt += 1
            done = False
            step = 0
            total_reward = 0.0

            if self.logger:
                self.logger.info(
                    f"Episode {self.episode_cnt} start (stage={stage})"
                )

            final_total_score = 0.0
            final_treasures = 0
            final_flash_count = 0
            final_collected_buff = 0
            final_terminated = False

            while not done:
                act_data = self.agent.predict(list_obs_data=[obs_data])[0]
                act = self.agent.action_process(act_data)

                env_reward, env_obs = self.env.step(act)

                if handle_disaster_recovery(env_obs, self.logger):
                    break

                terminated = env_obs["terminated"]
                truncated = env_obs["truncated"]
                step += 1
                done = terminated or truncated
                final_terminated = bool(terminated)

                _obs_data, _remain_info = self.agent.observation_process(env_obs)

                reward = np.array(
                    _remain_info.get("reward", [0.0]), dtype=np.float32
                )
                total_reward += float(reward[0])

                # 终局奖励：±10 + total_score * R_SCORE_SHAPED_COEF
                final_reward = np.zeros(1, dtype=np.float32)
                if done:
                    env_info = env_obs["observation"]["env_info"]
                    final_total_score = float(env_info.get("total_score", 0.0))
                    final_treasures = int(env_info.get("treasures_collected", 0))
                    final_flash_count = int(env_info.get("flash_count", 0))
                    final_collected_buff = int(env_info.get("collected_buff", 0))

                    base = Config.R_FAIL if terminated else Config.R_WIN
                    final_reward[0] = (
                        base + Config.R_SCORE_SHAPED_COEF * final_total_score
                    )

                    if self.logger:
                        result_str = "FAIL" if terminated else "WIN"
                        self.logger.info(
                            f"[GAMEOVER] ep:{self.episode_cnt} stage:{stage} "
                            f"steps:{step} result:{result_str} "
                            f"score:{final_total_score:.1f} tre:{final_treasures} "
                            f"flash:{final_flash_count} buff:{final_collected_buff} "
                            f"discover:{getattr(self.agent.preprocessor, 'treasure_discover_count', 0)} "
                            f"bumps:{getattr(self.agent.preprocessor, 'wall_bump_count', 0)} "
                            f"anti_stall:{getattr(self.agent.preprocessor, 'anti_stall_count', 0)} "
                            f"overrides:{getattr(self.agent, 'rule_override_count', 0)} "
                            f"reward_total:{total_reward + float(final_reward[0]):.3f}"
                        )

                # 样本帧（map_tensor 扁平化以配合 SampleData 1D 存储）
                map_flat = np.asarray(obs_data.map_tensor, dtype=np.float32).reshape(-1)
                sym_flat = np.asarray(obs_data.sym_feat, dtype=np.float32)

                frame = SampleData(
                    map_tensor=map_flat,
                    sym_feat=sym_flat,
                    legal_action=np.array(obs_data.legal_action, dtype=np.float32),
                    act=np.array([act_data.action[0]], dtype=np.float32),
                    reward=reward,
                    done=np.array([float(done)], dtype=np.float32),
                    reward_sum=np.zeros(1, dtype=np.float32),
                    value=np.array(act_data.value, dtype=np.float32).flatten()[:1],
                    next_value=np.zeros(1, dtype=np.float32),
                    advantage=np.zeros(1, dtype=np.float32),
                    prob=np.array(act_data.prob, dtype=np.float32),
                )
                collector.append(frame)

                if done:
                    if collector:
                        collector[-1].reward = collector[-1].reward + final_reward
                        if truncated and not terminated:
                            collector[-1].next_value = self.agent.evaluate_value(_obs_data)

                    now = time.time()
                    if now - self.last_report_monitor_time >= 60 and self.monitor:
                        monitor_data = {
                            "reward": round(total_reward + float(final_reward[0]), 4),
                            "episode_steps": step,
                            "episode_cnt": self.episode_cnt,
                            "total_score": round(final_total_score, 2),
                            "treasures_collected": final_treasures,
                            "flash_count": final_flash_count,
                            "collected_buff": final_collected_buff,
                            "treasure_discover_count": int(getattr(
                                self.agent.preprocessor, "treasure_discover_count", 0)),
                            "wall_bump_count": int(getattr(
                                self.agent.preprocessor, "wall_bump_count", 0)),
                            "anti_stall_count": int(getattr(
                                self.agent.preprocessor, "anti_stall_count", 0)),
                            "rule_override_count": int(getattr(
                                self.agent, "rule_override_count", 0)),
                            "curriculum_stage": int(stage),
                            "result_fail": int(final_terminated),
                        }
                        self.monitor.put_data({os.getpid(): monitor_data})
                        self.last_report_monitor_time = now

                    if collector:
                        collector = sample_process(collector)
                        yield collector

                    # 进度持久化（节流）
                    if self.episode_cnt % self._progress_save_every == 0:
                        self._save_progress()
                    break

                obs_data = _obs_data
                remain_info = _remain_info
