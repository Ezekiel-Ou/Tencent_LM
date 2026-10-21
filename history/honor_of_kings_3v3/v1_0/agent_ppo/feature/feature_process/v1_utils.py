#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""Protocol-safe helpers and measured map constants for Feature Schema v1."""

import math
from collections import OrderedDict

import numpy as np


RAW_FRAME_SECONDS = 0.066
LANE_CENTER_Z = 6115.0
SIDE_X_SCALE = 70000.0
SIDE_Z_SCALE = 35000.0
GLOBAL_DISTANCE_SCALE = 80000.0
LOCAL_DISTANCE_SCALE = 15000.0
TOWER_DISTANCE_SCALE = 13000.0

HERO_CONFIG_IDS = (132, 141, 193)

ACTOR_TYPE_NAMES = {
    0: "ACTOR_HERO",
    1: "ACTOR_MONSTER",
    2: "ACTOR_ORGAN",
    5: "ACTOR_BULLET",
    8: "ACTOR_SHENFU",
}
ACTOR_TYPE_ALIASES = {
    "ACTOR_TYPE_HERO": "ACTOR_HERO",
    "ACTOR_TYPE_MONSTER": "ACTOR_MONSTER",
    "ACTOR_TYPE_ORGAN": "ACTOR_ORGAN",
    "ACTOR_TYPE_BULLET": "ACTOR_BULLET",
    "ACTOR_TYPE_SHENFU": "ACTOR_SHENFU",
}
ACTOR_SUB_TYPE_NAMES = {
    0: "ACTOR_SUB_NONE",
    11: "ACTOR_SUB_SOLDIER",
    12: "ACTOR_SUB_BAOJUN",
    21: "ACTOR_SUB_TOWER",
    22: "ACTOR_SUB_TOWER_HIGH",
    23: "ACTOR_SUB_TOWER_SPRING",
    24: "ACTOR_SUB_CRYSTAL",
}

ORGAN_ANCHORS = {
    42: (-66300.0, 19580.0),
    43: (66300.0, 19580.0),
    66: (-17876.0, 6145.0),
    67: (-32486.0, 6170.0),
    68: (-48000.0, 6200.0),
    70: (17875.0, 6145.0),
    71: (32446.0, 6018.0),
    72: (48020.0, 6020.0),
}
CAKE_ANCHORS = {
    102: (0.0, -5826.0),
    103: (-21050.0, 6115.0),
    104: (-35950.0, 6115.0),
    105: (21050.0, 6115.0),
    106: (35950.0, 6115.0),
}
# Exact physical-camp anchors measured from
# ``monster_spawn_infos.born_pos_x/born_pos_z``.  The official target protocol
# exposes twenty monster actor slots, but the map contains fourteen physical
# camps: twelve side camps plus the central thief and Tyrant.  Companion slots
# share their camp's exact spawn anchor.  The two archer camps on each side are
# represented separately: config 32 selects the inner camp and config 33 the
# outer camp, so no measured physical camp disappears behind a config/side
# collision.
MONSTER_ANCHORS = {
    (49, "own"): (-20860.0, -17570.0),
    (49, "enemy"): (20710.0, -17420.0),
    (30, "own"): (-23470.0, -11250.0),
    (30, "enemy"): (23160.0, -10960.0),
    (31, "own"): (-23470.0, -11250.0),
    (31, "enemy"): (23160.0, -10960.0),
    (32, "own"): (-19840.0, -26940.0),
    (32, "enemy"): (20060.0, -26790.0),
    (33, "own"): (-29860.0, -24480.0),
    (33, "enemy"): (30070.0, -24610.0),
    (42, "own"): (-25530.0, 23990.0),
    (42, "enemy"): (25330.0, 23830.0),
    (43, "own"): (-25530.0, 23990.0),
    (43, "enemy"): (25330.0, 23830.0),
    (44, "own"): (-14010.0, 27670.0),
    (44, "enemy"): (13980.0, 26850.0),
    (45, "own"): (-14010.0, 27670.0),
    (45, "enemy"): (13980.0, 26850.0),
    (59, "center"): (-10.0, 21900.0),
    (122, "center"): (20.0, -20990.0),
}

PHYSICAL_MONSTER_CAMP_ANCHORS = (
    (-20860.0, -17570.0),
    (20710.0, -17420.0),
    (-23470.0, -11250.0),
    (23160.0, -10960.0),
    (-19840.0, -26940.0),
    (-29860.0, -24480.0),
    (20060.0, -26790.0),
    (30070.0, -24610.0),
    (-25530.0, 23990.0),
    (25330.0, 23830.0),
    (-14010.0, 27670.0),
    (13980.0, 26850.0),
    (-10.0, 21900.0),
    (20.0, -20990.0),
)

# Navigation backbone measured in the topology debug rounds. Coordinates use
# camp-1 raw x as the canonical side-coordinate x; z is lane-centred at runtime
# and command output mirrors x back for camp 2. Edges deliberately use observed
# jungle
# entrances; this prevents an Euclidean planner from asking a hero to cross the
# central wall on the way from lane to the Tyrant pit.
JUNGLE_ROUTE_NODE_NAMES = (
    "upper_lane_entry",
    "whitetail_deer",
    "dark_wolf",
    "upper_mid_exit",
    "lower_lane_entry",
    "demon_pioneer",
    "red_statue",
    "outer_archer",
    "inner_archer",
    "tyrant",
    "lane_clash",
    "safe_tower",
)
JUNGLE_ROUTE_STATIC_RAW_ANCHORS = (
    (-25000.0, 6115.0),
    (-25530.0, 23990.0),
    (-14010.0, 27670.0),
    (-10.0, 21900.0),
    (-23000.0, 6115.0),
    (-23470.0, -11250.0),
    (-20860.0, -17570.0),
    (-29860.0, -24480.0),
    (-19840.0, -26940.0),
    (20.0, -20990.0),
)
JUNGLE_ROUTE_EDGES = (
    (0, 1),
    (1, 2),
    (2, 3),
    (0, 3),
    (0, 4),
    (4, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    (6, 8),
    (8, 9),
    (6, 9),
)

REGION_NAMES = (
    "own_base",
    "own_tower_zone",
    "own_lane",
    "mid_lane",
    "enemy_lane",
    "enemy_tower_zone",
    "enemy_base",
    "upper_jungle",
    "lower_jungle",
    "tyrant_zone",
    "speed_buff_zone",
    "unknown",
)
LANE_ZONE_NAMES = (
    "own_base",
    "own_high_to_outer",
    "own_outer_to_mid",
    "mid",
    "enemy_outer_to_mid",
    "enemy_high_to_outer",
    "enemy_base",
)


def safe_get(obj, *names, default=None):
    for name in names:
        if isinstance(obj, dict) and name in obj:
            return obj[name]
    return default


def actor_get(actor, *names, default=None):
    value = safe_get(actor, *names, default=None)
    if value is not None:
        return value
    nested = safe_get(actor, "actor_state", "actorState", default={}) or {}
    return safe_get(nested, *names, default=default)


def as_list(value):
    return list(value) if isinstance(value, (list, tuple)) else []


def as_float(value, default=0.0):
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float(default)
    return result if math.isfinite(result) else float(default)


def as_int(value, default=0):
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def normalize_camp(value):
    if isinstance(value, str):
        if value.startswith("PLAYERCAMP_"):
            return as_int(value.rsplit("_", 1)[-1], 0)
        return as_int(value, 0)
    return as_int(value, 0)


def normalize_actor_type(value):
    value = ACTOR_TYPE_NAMES.get(value, value)
    return ACTOR_TYPE_ALIASES.get(value, value)


def normalize_sub_type(value):
    return ACTOR_SUB_TYPE_NAMES.get(value, value)


def frame_number(frame_state):
    return as_int(safe_get(frame_state, "frame_no", "frameNo", default=0), 0)


def location_of(obj):
    if not isinstance(obj, dict):
        return None
    collider = safe_get(obj, "collider", default={}) or {}
    location = safe_get(collider, "location", default=None)
    if not isinstance(location, dict):
        location = actor_get(obj, "location", default=None)
    if not isinstance(location, dict):
        nested = safe_get(obj, "actor_state", "actorState", default={}) or {}
        nested_collider = safe_get(nested, "collider", default={}) or {}
        location = safe_get(nested_collider, "location", default=None)
    if not isinstance(location, dict):
        return None
    x = safe_get(location, "x", default=None)
    z = safe_get(location, "z", default=None)
    if x is None or z is None:
        return None
    x, z = as_float(x, float("nan")), as_float(z, float("nan"))
    if not math.isfinite(x) or not math.isfinite(z) or abs(x) >= 90000 or abs(z) >= 90000:
        return None
    return (x, z)


def visible_to_camp(actor, own_camp, is_friend=False):
    camp_visible = as_list(actor_get(actor, "camp_visible", "campVisible", default=[]))
    if len(camp_visible) >= own_camp:
        return bool(camp_visible[own_camp - 1])
    explicit = actor_get(actor, "visible", "is_visible", "isVisible", default=None)
    if explicit is not None:
        return bool(explicit)
    return bool(is_friend)


def is_alive(actor):
    return as_float(actor_get(actor, "hp", default=0), 0) > 0 and as_float(
        actor_get(actor, "max_hp", "maxHp", default=0), 0
    ) > 0


def clip(value, low=0.0, high=1.0):
    return float(np.clip(as_float(value), low, high))


def signed_clip(value):
    return clip(value, -1.0, 1.0)


def ratio(numerator, denominator):
    denominator = as_float(denominator)
    return 0.0 if denominator <= 0 else clip(as_float(numerator) / denominator)


def min_max(value, low, high):
    if high <= low:
        raise ValueError("high must be greater than low")
    return clip((as_float(value) - low) / (high - low))


def log_norm(value, maximum):
    return clip(math.log1p(max(0.0, as_float(value))) / math.log1p(maximum))


def side_position(raw_position, own_camp):
    if raw_position is None:
        return None
    raw_x, raw_z = raw_position
    side_x = raw_x if own_camp == 1 else -raw_x
    return (side_x, raw_z - LANE_CENTER_Z)


def raw_position(side_position_value, own_camp):
    if side_position_value is None:
        return None
    side_x, side_z = side_position_value
    return (side_x if own_camp == 1 else -side_x, side_z + LANE_CENTER_Z)


def normalized_side_position(side_position_value):
    if side_position_value is None:
        return (0.0, 0.0)
    return (
        signed_clip(side_position_value[0] / SIDE_X_SCALE),
        signed_clip(side_position_value[1] / SIDE_Z_SCALE),
    )


def distance(first, second):
    if first is None or second is None:
        return None
    return math.hypot(first[0] - second[0], first[1] - second[1])


def relative_geometry(origin, target, scale=LOCAL_DISTANCE_SCALE):
    if origin is None or target is None:
        return (0.0, 0.0, 0.0, 0.0, 0.0, False)
    dx = target[0] - origin[0]
    dz = target[1] - origin[1]
    dist = math.hypot(dx, dz)
    if dist > 1e-6:
        sin_bearing, cos_bearing = dz / dist, dx / dist
    else:
        sin_bearing, cos_bearing = 0.0, 1.0
    return (
        signed_clip(dx / scale),
        signed_clip(dz / scale),
        clip(dist / scale),
        signed_clip(sin_bearing),
        signed_clip(cos_bearing),
        True,
    )


def one_hot(index, size):
    result = np.zeros(size, dtype=np.float32)
    if 0 <= int(index) < size:
        result[int(index)] = 1.0
    return result


def uniform_bucket(value, low, high, count):
    if count <= 0:
        raise ValueError("bucket count must be positive")
    value = as_float(value)
    if value <= low:
        return 0
    if value >= high:
        return count - 1
    return min(count - 1, int((value - low) / (high - low) * count))


def coarse_position_buckets(side_position_value):
    if side_position_value is None:
        return (np.zeros(11, dtype=np.float32), np.zeros(9, dtype=np.float32))
    return (
        one_hot(uniform_bucket(side_position_value[0], -SIDE_X_SCALE, SIDE_X_SCALE, 11), 11),
        one_hot(uniform_bucket(side_position_value[1], -SIDE_Z_SCALE, SIDE_Z_SCALE, 9), 9),
    )


def semantic_region(side_position_value):
    """Conservative deterministic region partition over measured anchors."""

    if side_position_value is None:
        return REGION_NAMES.index("unknown")
    side_x, side_z = side_position_value
    if abs(side_x) > SIDE_X_SCALE or abs(side_z) > SIDE_Z_SCALE:
        return REGION_NAMES.index("unknown")

    tyrant_side = side_position((20.0, -20990.0), 1)
    speed_side = side_position((0.0, -5826.0), 1)
    if distance(side_position_value, tyrant_side) <= 7500:
        return REGION_NAMES.index("tyrant_zone")
    if distance(side_position_value, speed_side) <= 5000:
        return REGION_NAMES.index("speed_buff_zone")

    if abs(side_z) <= 6500:
        if side_x <= -43000:
            name = "own_base"
        elif side_x <= -26000:
            name = "own_tower_zone"
        elif side_x <= -6500:
            name = "own_lane"
        elif side_x < 6500:
            name = "mid_lane"
        elif side_x < 26000:
            name = "enemy_lane"
        elif side_x < 43000:
            name = "enemy_tower_zone"
        else:
            name = "enemy_base"
        return REGION_NAMES.index(name)
    return REGION_NAMES.index("upper_jungle" if side_z > 0 else "lower_jungle")


def region_one_hot(side_position_value):
    return one_hot(semantic_region(side_position_value), len(REGION_NAMES))


def lane_zone(side_x, own_anchors, enemy_anchors):
    own_outer, own_high, own_crystal = own_anchors
    enemy_outer, enemy_high, enemy_crystal = enemy_anchors
    if side_x <= own_high:
        index = 0
    elif side_x <= own_outer:
        index = 1
    elif side_x < -3000:
        index = 2
    elif side_x <= 3000:
        index = 3
    elif side_x < enemy_outer:
        index = 4
    elif side_x < enemy_high:
        index = 5
    else:
        index = 6
    return one_hot(index, 7)


def side_relation_for_monster(actor, own_camp):
    config_id = as_int(actor_get(actor, "config_id", "configId", default=0), 0)
    if config_id in (59, 122):
        return "center"
    position = location_of(actor)
    if position is None:
        return "unknown"
    raw_x = position[0]
    is_own = raw_x < 0 if own_camp == 1 else raw_x >= 0
    return "own" if is_own else "enemy"


def fixed_organ_config_ids(own_camp):
    if own_camp == 1:
        return (66, 67, 68, 70, 71, 72)
    return (70, 71, 72, 66, 67, 68)


def fixed_organ_side_x(own_camp):
    return tuple(side_position(ORGAN_ANCHORS[config_id], own_camp)[0] for config_id in fixed_organ_config_ids(own_camp))


def fixed_resource_config_ids(own_camp):
    return (102, 103, 104, 105, 106) if own_camp == 1 else (102, 105, 106, 103, 104)


def fixed_monster_anchor(config_id, relation, own_camp):
    relation_key = "center" if relation == "center" else relation
    if own_camp == 1:
        raw = MONSTER_ANCHORS.get((config_id, relation_key))
    else:
        mirrored = {"own": "enemy", "enemy": "own", "center": "center"}[relation_key]
        raw = MONSTER_ANCHORS.get((config_id, mirrored))
    return raw


def monster_spawn_config_id(config_id):
    """Return the major-monster id owning a physical camp spawn timer.

    Companion actors do not have independent ``monster_spawn_infos`` entries.
    Their protocol slots therefore join to the major actor's spawn record at
    the same exact camp anchor.
    """

    return {31: 30, 33: 32, 43: 42, 45: 44}.get(int(config_id), int(config_id))


def normalized_skill_slot(value):
    if isinstance(value, int):
        return value if 0 <= value <= 5 else None
    text = str(value or "").upper()
    for index in range(6):
        if text in (str(index), f"SLOT_SKILL_{index}"):
            return index
    return None


def skill_slots(actor):
    result = [None] * 6
    skill_state = actor_get(actor, "skill_state", "skillState", default={}) or {}
    raw_slots = safe_get(skill_state, "slot_states", "slotStates", default=[]) or []
    for slot in raw_slots:
        index = normalized_skill_slot(safe_get(slot, "slot_type", "slotType", default=None))
        if index is not None:
            result[index] = slot
    return result


def buff_items(actor):
    buff_state = actor_get(actor, "buff_state", "buffState", default={}) or {}
    return as_list(safe_get(buff_state, "buff_skills", "buffSkills", default=[]))


def mark_items(actor):
    buff_state = actor_get(actor, "buff_state", "buffState", default={}) or {}
    return as_list(safe_get(buff_state, "buff_marks", "buffMarks", default=[]))


def actor_runtime_id(actor):
    return as_int(actor_get(actor, "runtime_id", "runtimeId", default=0), 0)


def actor_config_id(actor):
    return as_int(actor_get(actor, "config_id", "configId", default=0), 0)


def actor_camp(actor):
    return normalize_camp(actor_get(actor, "camp", default=0))


def actor_index_by_runtime(heroes, npcs):
    result = {}
    for actor in list(heroes) + list(npcs):
        runtime_id = actor_runtime_id(actor)
        if runtime_id:
            result[runtime_id] = actor
    return result


def target_type(actor):
    if not actor:
        return "none"
    actor_type = normalize_actor_type(actor_get(actor, "actor_type", "actorType", default=None))
    sub_type = normalize_sub_type(actor_get(actor, "sub_type", "subType", default=None))
    if actor_type == "ACTOR_HERO":
        return "hero"
    if sub_type == "ACTOR_SUB_SOLDIER":
        return "soldier"
    if actor_type == "ACTOR_MONSTER":
        return "monster"
    if actor_type == "ACTOR_ORGAN":
        if sub_type == "ACTOR_SUB_CRYSTAL":
            return "crystal"
        if sub_type == "ACTOR_SUB_TOWER_HIGH":
            return "high_tower"
        if sub_type == "ACTOR_SUB_TOWER":
            return "outer_tower"
        return "organ"
    return "other"


def ordered_by_runtime(actors):
    return sorted(actors, key=actor_runtime_id)


def ordered_by_config(actors):
    return sorted(actors, key=lambda actor: (actor_config_id(actor), actor_runtime_id(actor)))


def map_by_config(actors):
    result = OrderedDict()
    for actor in ordered_by_config(actors):
        result[actor_config_id(actor)] = actor
    return result
