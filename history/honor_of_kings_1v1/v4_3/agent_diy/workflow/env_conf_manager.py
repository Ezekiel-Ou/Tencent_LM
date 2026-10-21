#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
Environment configuration manager for agent_diy training.

This mirrors the hok_semi eval-opponent rotation while keeping the current
hok1v1 112/133 lineup and summoner-skill protocol.
"""

import copy
import random

try:
    from tools.train_env_conf_validate import read_usr_conf as _read_usr_conf
except ModuleNotFoundError:
    try:
        import tomllib
    except ModuleNotFoundError:
        import tomli as tomllib

    def _read_usr_conf(config_path, logger=None):
        with open(config_path, "rb") as file:
            return tomllib.load(file)


class EnvConfManager:
    def __init__(self, config_path, logger=None):
        self.config_path = config_path
        self.logger = logger
        self.usr_conf = None
        self.episode_cnt = 0
        self.eval_interval = 0
        self.random_eval_start = 0
        self.default_opponent_agent = None
        self.current_opponent_agent = None
        self.train_opponent_types = []
        self.train_opponent_weights = []
        self.auto_switch_monitor_side = False
        self.monitor_side = 0
        self.initialize()

    def initialize(self):
        self.usr_conf = _read_usr_conf(self.config_path, self.logger)
        if self.usr_conf is None:
            raise ValueError(f"usr_conf is None, please check {self.config_path}")

        episode_conf = self.usr_conf.get("episode", {})
        monitor_conf = self.usr_conf.get("monitor", {})

        raw_eval_interval = int(episode_conf.get("eval_interval", 0))
        self.eval_interval = raw_eval_interval if raw_eval_interval > 0 else 0
        self.default_opponent_agent = episode_conf.get("opponent_agent", "selfplay")
        self.train_opponent_types = list(episode_conf.get("opponent_agent_types", []))
        self.train_opponent_weights = list(episode_conf.get("opponent_agent_weights", []))
        self.auto_switch_monitor_side = bool(monitor_conf.get("auto_switch_monitor_side", False))
        self.monitor_side = int(monitor_conf.get("monitor_side", 0))

        if self.eval_interval > 0:
            self.random_eval_start = random.randint(0, self.eval_interval - 1)

    def get_current_config(self):
        return self.usr_conf

    def get_monitor_side(self):
        return self.monitor_side

    def get_opponent_agent(self):
        return self.current_opponent_agent or self.usr_conf["episode"]["opponent_agent"]

    def update_config(self, lineup=None):
        if lineup:
            self._update_lineup(lineup)

        if self.auto_switch_monitor_side:
            self.monitor_side = 1 - self.monitor_side
        self.usr_conf["monitor"]["monitor_side"] = self.monitor_side

        is_eval = self._is_eval_episode()
        if is_eval:
            self.usr_conf["episode"]["eval_opponent_type"] = self._select_eval_opponent()

        opponent_agent = self._select_train_opponent() if not is_eval else self.usr_conf["episode"]["eval_opponent_type"]
        self.current_opponent_agent = opponent_agent
        self.usr_conf["episode"]["opponent_agent"] = opponent_agent
        self.episode_cnt += 1

        return self.get_current_config(), is_eval, self.get_monitor_side()

    def _update_lineup(self, lineup):
        if len(lineup) != 2 or not all(isinstance(hero_id, int) for hero_id in lineup):
            raise ValueError("Invalid lineup format, expected list of 2 integers")
        self.usr_conf["lineups"]["blue_camp"][0]["hero_id"] = lineup[0]
        self.usr_conf["lineups"]["red_camp"][0]["hero_id"] = lineup[1]

    def _is_eval_episode(self):
        if self.eval_interval <= 0:
            return False
        return (self.episode_cnt + self.random_eval_start) % self.eval_interval == 0

    def _select_eval_opponent(self):
        episode_conf = self.usr_conf.get("episode", {})
        eval_opponents = episode_conf.get("eval_opponent_types")
        if not eval_opponents:
            eval_opponents = [episode_conf.get("eval_opponent_type", "common_ai")]
        return random.choice(eval_opponents)

    def _select_train_opponent(self):
        if self.default_opponent_agent != "mixed":
            return self.default_opponent_agent

        opponents = [str(item) for item in self.train_opponent_types if str(item)]
        if not opponents:
            return "selfplay"

        weights = self._normalized_weights(self.train_opponent_weights, len(opponents))
        return random.choices(opponents, weights=weights, k=1)[0]

    def _normalized_weights(self, weights, expected_len):
        if len(weights) != expected_len:
            return [1.0] * expected_len
        normalized = []
        for weight in weights:
            try:
                normalized.append(max(float(weight), 0.0))
            except (TypeError, ValueError):
                normalized.append(0.0)
        return normalized if sum(normalized) > 0 else [1.0] * expected_len

    def _current_lineup(self):
        blue_hero_ids, red_hero_ids = self.extract_hero_ids_from_usr_conf(self.usr_conf)
        return blue_hero_ids + red_hero_ids

    @staticmethod
    def extract_hero_ids_from_usr_conf(usr_conf):
        lineups = usr_conf.get("lineups", {})
        blue_hero_ids = [item["hero_id"] for item in lineups.get("blue_camp", [])]
        red_hero_ids = [item["hero_id"] for item in lineups.get("red_camp", [])]
        return blue_hero_ids, red_hero_ids

    @staticmethod
    def inject_select_skills(usr_conf, camp_key, select_skills):
        if not isinstance(select_skills, dict):
            hero_ids = [
                item["hero_id"]
                for item in usr_conf.get("lineups", {}).get(camp_key, [])
            ]
            select_skills = {hero_id: int(select_skills) for hero_id in hero_ids}

        for hero_conf in usr_conf.get("lineups", {}).get(camp_key, []):
            hero_id = hero_conf.get("hero_id")
            skill_id = select_skills.get(hero_id, select_skills.get(str(hero_id)))
            if skill_id is not None:
                skill_id = int(skill_id)
                hero_conf["select_skill"] = skill_id
                hero_conf["summoner_skill_id"] = skill_id

        camp_conf = usr_conf.get(camp_key)
        if isinstance(camp_conf, dict):
            hero_ids = [item["hero_id"] for item in usr_conf.get("lineups", {}).get(camp_key, [])]
            skill_id = None
            for hero_id in hero_ids:
                skill_id = select_skills.get(hero_id, select_skills.get(str(hero_id)))
                if skill_id is not None:
                    break
            if skill_id is not None:
                camp_conf["select_skill"] = int(skill_id)
                camp_conf["summoner_skill_id"] = int(skill_id)

    def snapshot(self):
        return copy.deepcopy(self.usr_conf)
