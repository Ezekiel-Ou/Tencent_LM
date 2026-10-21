#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""

from collections import OrderedDict
from agent_ppo.feature.feature_process.hero_process import HeroProcess
from agent_ppo.feature.feature_process.organ_process import OrganProcess
from agent_ppo.feature.feature_process.soldier_process import SoldierProcess
from agent_ppo.feature.feature_process.monster_process import MonsterProcess
from agent_ppo.feature.feature_process.global_process import GlobalProcess


ACTOR_TYPE_NAMES = {
    0: "ACTOR_HERO",
    1: "ACTOR_MONSTER",
    2: "ACTOR_ORGAN",
    3: "ACTOR_EYE",
    4: "ACTOR_CALL",
    5: "ACTOR_BULLET",
    6: "ACTOR_BLOCK",
    7: "ACTOR_INTERACTITEM",
    8: "ACTOR_SHENFU",
    9: "ACTOR_VEHICLE",
    10: "ACTOR_ALL",
    11: "ACTOR_INVALID",
}

ACTOR_SUB_TYPE_NAMES = {
    0: "ACTOR_SUB_NONE",
    11: "ACTOR_SUB_SOLDIER",
    12: "ACTOR_SUB_BAOJUN",
    13: "ACTOR_SUB_BARON",
    14: "ACTOR_SUB_BLUEBA",
    15: "ACTOR_SUB_REDBA",
    16: "ACTOR_SUB_BIGDRAGON",
    17: "ACTOR_SUB_DARK_BIGDRAGON",
    21: "ACTOR_SUB_TOWER",
    22: "ACTOR_SUB_TOWER_HIGH",
    23: "ACTOR_SUB_TOWER_SPRING",
    24: "ACTOR_SUB_CRYSTAL",
}


class FeatureProcess:
    def __init__(self, camp):
        self.reset(camp)
        self.init_monster_targets()

    def reset(self, camp):
        self.camp = camp
        self.hero_process = HeroProcess(camp)
        self.organ_process = OrganProcess(camp)
        self.soldier_process = SoldierProcess(camp)
        self.monster_process = MonsterProcess(camp)
        self.global_process = GlobalProcess(camp)

        self.main_camp_hero_dict = OrderedDict()
        self.enemy_camp_hero_dict = OrderedDict()
        self.main_camp_organ_dict = {}
        self.enemy_camp_organ_dict = {}
        self.main_camp_soldier_dict = OrderedDict()
        self.enemy_camp_soldier_dict = OrderedDict()
        self.monster_dict = {}
        self.data_dicts = {
            "hero": (self.main_camp_hero_dict, self.enemy_camp_hero_dict),
            "organ": (self.main_camp_organ_dict, self.enemy_camp_organ_dict),
            "soldier": (self.main_camp_soldier_dict, self.enemy_camp_soldier_dict),
            "monster": (self.monster_dict,),
        }

    def _camp_name(self, camp):
        if isinstance(camp, int):
            return f"PLAYERCAMP_{camp}"
        return camp

    def _actor_type_name(self, actor_type):
        if isinstance(actor_type, int):
            return ACTOR_TYPE_NAMES.get(actor_type, actor_type)
        return actor_type

    def _actor_sub_type_name(self, sub_type):
        if isinstance(sub_type, int):
            return ACTOR_SUB_TYPE_NAMES.get(sub_type, sub_type)
        return sub_type

    def _normalize_hero(self, hero):
        normalized = dict(hero)
        normalized["camp"] = self._camp_name(normalized.get("camp"))
        return normalized

    def _normalize_npc(self, npc):
        normalized = dict(npc)
        normalized["camp"] = self._camp_name(normalized.get("camp"))
        normalized["actor_type"] = self._actor_type_name(normalized.get("actor_type"))
        normalized["sub_type"] = self._actor_sub_type_name(normalized.get("sub_type"))
        return normalized

    def process_feature(self, observation):
        # Process frame state
        # 处理帧状态
        frame_state = observation["frame_state"]
        self.generate_hero_info_dict(frame_state)
        self.generate_all_npc_info_dict(frame_state)
        self.data_dicts = {
            "frame_no": frame_state["frame_no"],
            "hero": (self.main_camp_hero_dict, self.enemy_camp_hero_dict),
            "organ": (self.main_camp_organ_dict, self.enemy_camp_organ_dict),
            "soldier": (self.main_camp_soldier_dict, self.enemy_camp_soldier_dict),
            "monster": (self.monster_dict,),
        }

        # Process feature
        # 处理特征
        (
            main_camp_hero_feature,
            enemy_camp_hero_feature,
            main_camp_hero_feature_dict,
        ) = self.hero_process.process_vec_hero(self.data_dicts)
        soldier_feature = self.soldier_process.process_vec_soldier(self.data_dicts)
        organ_feature = self.organ_process.process_vec_organ(self.data_dicts)
        monster_feature = self.monster_process.process_vec_monster(self.data_dicts)
        global_feature = self.global_process.process_vec_global(self.data_dicts)

        feature = (
            main_camp_hero_feature
            + enemy_camp_hero_feature
            + soldier_feature
            + organ_feature
            + monster_feature
            + global_feature
        )

        return feature, main_camp_hero_feature_dict

    def generate_hero_info_dict(self, frame_state):
        self.main_camp_hero_dict.clear()
        self.enemy_camp_hero_dict.clear()
        for hero in frame_state["hero_states"]:
            hero = self._normalize_hero(hero)
            if hero["camp"] == self.camp:
                self.main_camp_hero_dict[hero["config_id"]] = hero
            else:
                self.enemy_camp_hero_dict[hero["config_id"]] = hero
        self.main_camp_hero_dict = OrderedDict(sorted(self.main_camp_hero_dict.items()))
        self.enemy_camp_hero_dict = OrderedDict(sorted(self.enemy_camp_hero_dict.items()))

    def generate_all_npc_info_dict(self, frame_state):
        self.main_camp_organ_dict.clear()
        self.enemy_camp_organ_dict.clear()
        self.main_camp_soldier_dict.clear()
        self.enemy_camp_soldier_dict.clear()
        self.monster_dict.clear()
        monster_lookup = {}

        for npc in frame_state["npc_states"]:
            npc = self._normalize_npc(npc)
            config_id = npc.get("config_id")
            npc_type = npc.get("actor_type")
            npc_subtype = npc.get("sub_type")
            npc_camp = npc.get("camp")
            npc_hp = npc.get("hp", 0)

            # Process soldier
            # 处理小兵
            if npc_subtype == "ACTOR_SUB_SOLDIER" and npc_hp > 0:
                if npc_camp == self.camp:
                    self.main_camp_soldier_dict[npc["runtime_id"]] = npc
                else:
                    self.enemy_camp_soldier_dict[npc["runtime_id"]] = npc

            # Process organ
            # 处理防御塔
            if npc_type == "ACTOR_ORGAN":
                organ_dict = self.main_camp_organ_dict if npc_camp == self.camp else self.enemy_camp_organ_dict
                if npc_subtype == "ACTOR_SUB_TOWER" and config_id in [66, 67, 70, 71]:
                    organ_dict[f"tower_{config_id}"] = npc
                elif npc_subtype == "ACTOR_SUB_CRYSTAL":
                    organ_dict["crystal"] = npc

            # Process monster
            # 处理野怪
            if npc_type == "ACTOR_MONSTER" and npc_subtype == "ACTOR_SUB_NONE":
                is_enemy = (
                    False
                    if config_id in [59, 122]
                    else (
                        npc.get("location", {}).get("x", 0) < 0
                        and self.camp == "PLAYERCAMP_2"
                        or npc.get("location", {}).get("x", 0) >= 0
                        and self.camp == "PLAYERCAMP_1"
                    )
                )
                key = (config_id, is_enemy)
                if key not in monster_lookup:
                    monster_lookup[key] = npc

        for _name, is_enemy_expected, cfg_id in self.monster_targets:
            self.monster_dict[_name] = monster_lookup.get((cfg_id, is_enemy_expected), None)

        # Sort soldier dict
        # 排序小兵字典
        self.main_camp_soldier_dict = OrderedDict(sorted(self.main_camp_soldier_dict.items()))
        self.enemy_camp_soldier_dict = OrderedDict(sorted(self.enemy_camp_soldier_dict.items()))

    def init_monster_targets(self):
        self.monster_targets = []
        monster_defs = [
            ("Red_Statue", 49, True),
            ("Greater_Demon_Pioneer", 30, True),
            ("Lesser_Demon_Pioneer", 31, True),
            ("Greater_Demon_Archer", 32, True),
            ("Lesser_Demon_Archer", 33, True),
            ("Greater_Whitetail_Deer", 42, True),
            ("Lesser_Whitetail_Deer", 43, True),
            ("Greater_Shadow_Wolf", 44, True),
            ("Lesser_Shadow_Wolf", 45, True),
            ("Treasure_Thief", 59, False),
            ("Tyrant", 122, False),
        ]

        for name, config_id, has_enemy in monster_defs:
            self.monster_targets.append((name, False, config_id))

            if has_enemy:
                enemy_name = f"{name}_Enemy"
                self.monster_targets.append((enemy_name, True, config_id))
