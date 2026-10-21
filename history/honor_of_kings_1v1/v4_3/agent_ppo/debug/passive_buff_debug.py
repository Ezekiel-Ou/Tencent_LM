"""
Passive buff-id debug for Luban (112) and Di Renjie (133).

This workflow drives both camps with normal-attack-air only:
    [button=3, move_x=8, move_z=8, skill_x=8, skill_z=8, target=0]

It is intentionally isolated from model loading, sampling, and learning. The
goal is to identify which buff ids appear after repeated passive-triggering
normal attacks.
"""

from typing import Any, Dict, List, Optional, Sequence, Tuple


NO_OP: List[int] = [1, 8, 8, 8, 8, 0]
AIR_ATTACK: List[int] = [3, 8, 8, 8, 8, 0]
LUBAN_SKILL1: List[int] = [4, 8, 8, 8, 8, 2]


def _coerce_int(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _normalize_camp(value):
    if value is None:
        return None
    if isinstance(value, str):
        if value.endswith("_1") or value == "1":
            return 1
        if value.endswith("_2") or value == "2":
            return 2
    return _coerce_int(value)


def _get_field(d: dict, key: str, alt_keys: Optional[Sequence[str]] = None, default=None):
    if not isinstance(d, dict):
        return default
    for k in (key, *(alt_keys or ())):
        if k in d:
            return d[k]
    actor_state = d.get("actor_state") or d.get("actorState")
    if isinstance(actor_state, dict):
        for k in (key, *(alt_keys or ())):
            if k in actor_state:
                return actor_state[k]
    return default


def find_own_hero(observation: Optional[dict], hero_id: Optional[int] = None) -> Optional[dict]:
    """Return this observation's own hero, optionally constrained by config_id."""
    if not isinstance(observation, dict):
        return None
    fs = observation.get("frame_state")
    if not isinstance(fs, dict):
        return None
    heroes = fs.get("hero_states") or fs.get("heroes") or []
    own_camp = _normalize_camp(
        observation.get("player_camp") or observation.get("camp_id") or observation.get("camp")
    )
    own_player_id = _coerce_int(observation.get("player_id") or observation.get("runtime_id"))
    hero_id = _coerce_int(hero_id)

    fallback = None
    for hero in heroes:
        if not isinstance(hero, dict):
            continue
        cid = _coerce_int(_get_field(hero, "config_id", ["configId"]))
        if hero_id is not None and cid != hero_id:
            continue
        if fallback is None:
            fallback = hero
        runtime_id = _coerce_int(_get_field(hero, "runtime_id", ["runtimeId", "player_id", "playerId"]))
        if own_player_id is not None and runtime_id == own_player_id:
            return hero
        camp = _normalize_camp(_get_field(hero, "camp"))
        if own_camp is not None and camp is not None and camp == own_camp:
            return hero
    return fallback


def extract_buff_state(hero: Optional[dict]) -> Tuple[Tuple[Tuple[int, Optional[int]], ...],
                                                       Tuple[Tuple[int, Optional[int]], ...]]:
    """Return (buff_skills, buff_marks) as sorted tuples of (config_id, secondary)."""
    if not isinstance(hero, dict):
        return tuple(), tuple()
    bs = _get_field(hero, "buff_state", ["buffState"])
    if not isinstance(bs, dict):
        return tuple(), tuple()

    skills = []
    for skill in bs.get("buff_skills", []) or []:
        if not isinstance(skill, dict):
            continue
        cid = _coerce_int(skill.get("configId") or skill.get("config_id") or skill.get("id"))
        if cid is None:
            continue
        times = _coerce_int(skill.get("times"))
        skills.append((cid, times))

    marks = []
    for container_key in ("buff_marks", "buffMarks", "marks", "mark_states", "markStates"):
        seq = bs.get(container_key)
        if not isinstance(seq, list):
            continue
        for mark in seq:
            if not isinstance(mark, dict):
                continue
            cid = _coerce_int(
                mark.get("configId")
                or mark.get("config_id")
                or mark.get("buff_mark_id")
                or mark.get("buffMarkId")
                or mark.get("mark_id")
                or mark.get("markId")
                or mark.get("id")
            )
            if cid is None:
                continue
            layer = _coerce_int(mark.get("layer") or mark.get("layers") or mark.get("level") or mark.get("count"))
            marks.append((cid, layer))
        break

    return tuple(sorted(skills)), tuple(sorted(marks))


class PassiveBuffDebugAgent:
    """Normal-attack-air scripted agent with per-attack tags."""

    def __init__(self, attack_interval_steps: int = 7, max_attacks: int = 12):
        self.attack_interval_steps = max(1, int(attack_interval_steps))
        self.max_attacks = max(1, int(max_attacks))
        self.illegal_button_count = 0

    def reset(self) -> None:
        self.illegal_button_count = 0

    def act(self, observation: Optional[dict], step_no: int, agent_id: int = 0) -> Tuple[List[int], str]:
        attack_idx = step_no // self.attack_interval_steps
        within = step_no % self.attack_interval_steps
        if attack_idx >= self.max_attacks:
            return list(NO_OP), "post_attack_idle"
        if within != 0:
            return list(NO_OP), f"cooldown_after_air_attack#{attack_idx + 1}"

        action = list(AIR_ATTACK)
        legal = observation.get("legal_action") if isinstance(observation, dict) else None
        if legal is not None and not self._is_button_legal(action[0], legal):
            self.illegal_button_count += 1
            return list(NO_OP), f"air_attack#{attack_idx + 1}->masked_noop"
        return action, f"air_attack#{attack_idx + 1}"

    def episode_done(self, step_no: int, settle_steps: int = 10) -> bool:
        return step_no >= self.attack_interval_steps * self.max_attacks + max(1, int(settle_steps))

    @staticmethod
    def _is_button_legal(button_idx: int, legal_action) -> bool:
        try:
            return bool(legal_action[button_idx])
        except (TypeError, IndexError):
            return True


class LubanSkill1PassiveBuffDebugAgent:
    """Cast Luban skill 1 once, then normal-attack-air several times."""

    def __init__(self, skill_wait_steps: int = 20, skill_to_attack_gap_steps: int = 7,
                 attack_interval_steps: int = 7, max_attacks: int = 4):
        self.skill_wait_steps = max(1, int(skill_wait_steps))
        self.skill_to_attack_gap_steps = max(1, int(skill_to_attack_gap_steps))
        self.attack_interval_steps = max(1, int(attack_interval_steps))
        self.max_attacks = max(1, int(max_attacks))
        self.illegal_button_count = 0
        self.skill_cast_step: Optional[int] = None
        self.skill_cast_count = 0

    def reset(self) -> None:
        self.illegal_button_count = 0
        self.skill_cast_step = None
        self.skill_cast_count = 0

    def act(self, observation: Optional[dict], step_no: int, agent_id: int = 0) -> Tuple[List[int], str]:
        legal = observation.get("legal_action") if isinstance(observation, dict) else None

        if self.skill_cast_step is None:
            if self._is_button_legal(4, legal) or step_no >= self.skill_wait_steps:
                action = self._skill1_action(legal)
                if legal is not None and not self._is_button_legal(action[0], legal):
                    self.illegal_button_count += 1
                    return list(NO_OP), "skill1_release->masked_noop"
                self.skill_cast_step = step_no
                self.skill_cast_count += 1
                return action, "skill1_release"
            return list(NO_OP), "wait_skill1_legal"

        local = step_no - self.skill_cast_step
        if local < self.skill_to_attack_gap_steps:
            return list(NO_OP), f"post_skill_wait@{local}"

        attack_local = local - self.skill_to_attack_gap_steps
        attack_idx = attack_local // self.attack_interval_steps
        within = attack_local % self.attack_interval_steps
        if attack_idx >= self.max_attacks:
            return list(NO_OP), "post_skill_attack_idle"
        if within != 0:
            return list(NO_OP), f"cooldown_after_post_skill_air_attack#{attack_idx + 1}"

        action = list(AIR_ATTACK)
        if legal is not None and not self._is_button_legal(action[0], legal):
            self.illegal_button_count += 1
            return list(NO_OP), f"post_skill_air_attack#{attack_idx + 1}->masked_noop"
        return action, f"post_skill_air_attack#{attack_idx + 1}"

    def episode_done(self, step_no: int, settle_steps: int = 10) -> bool:
        if self.skill_cast_step is None:
            return step_no >= self.skill_wait_steps + settle_steps
        local = step_no - self.skill_cast_step
        return local >= (
            self.skill_to_attack_gap_steps
            + self.attack_interval_steps * self.max_attacks
            + max(1, int(settle_steps))
        )

    @staticmethod
    def _is_button_legal(button_idx: int, legal_action) -> bool:
        try:
            return bool(legal_action[button_idx])
        except (TypeError, IndexError):
            return True

    @staticmethod
    def _skill1_action(legal_action) -> List[int]:
        action = list(LUBAN_SKILL1)
        target_mask = _target_mask_for_button(legal_action, button_idx=4)
        if target_mask and not _is_target_legal(action[-1], target_mask):
            action[-1] = _first_legal_target(target_mask, fallback=2)
        return action


def _target_mask_for_button(legal_action, button_idx: int) -> Optional[List[int]]:
    try:
        values = list(legal_action)
    except TypeError:
        return None
    target_size = 9
    top_size = 12
    if len(values) < target_size * top_size:
        return None
    start = len(values) - target_size * top_size + int(button_idx) * target_size
    end = start + target_size
    if start < 0 or end > len(values):
        return None
    return values[start:end]


def _is_target_legal(target_idx: int, target_mask: Sequence[int]) -> bool:
    try:
        return bool(target_mask[int(target_idx)])
    except (TypeError, ValueError, IndexError):
        return False


def _first_legal_target(target_mask: Sequence[int], fallback: int = 0) -> int:
    for idx, value in enumerate(target_mask):
        if value:
            return idx
    return int(fallback)


def parse_attack_no(tag: str) -> Optional[int]:
    if not isinstance(tag, str) or "air_attack#" not in tag:
        return None
    try:
        tail = tag.split("air_attack#", 1)[1]
        return int(tail.split("->", 1)[0].split(" ", 1)[0])
    except (IndexError, TypeError, ValueError):
        return None


def _format_pairs(pairs, value_name: str) -> str:
    if not pairs:
        return "[]"
    return "[" + ",".join(f"({cid},{value_name}={value})" for cid, value in pairs) + "]"


def format_passive_buff_line(event: str, ep: int, lineup: Sequence[int], step: int,
                             frame_no: Any, side: str, hero_id: int, tag: str,
                             action: Sequence[int], attack_no: Optional[int],
                             skills, marks) -> str:
    attack_repr = "-" if attack_no is None else str(attack_no)
    return (
        f"[PASSIVE_BUFF] event={event} ep={ep} lineup={list(lineup)} side={side} "
        f"hero={hero_id} step={step} fno={frame_no} action={tag} "
        f"attack#={attack_repr} act={list(action)} "
        f"skills={_format_pairs(skills, 'times')} marks={_format_pairs(marks, 'layer')}"
    )


def format_passive_summary_line(ep: int, lineup: Sequence[int], side: str, hero_id: int,
                                attacks_attempted: int, skill_ids_seen: Dict[int, int],
                                mark_ids_seen: Dict[int, int], final_frame: Any,
                                illegal_button_count: int) -> str:
    sk = ",".join(f"{cid}:max_times={times}" for cid, times in sorted(skill_ids_seen.items()))
    mk = ",".join(f"{cid}:max_layer={layer}" for cid, layer in sorted(mark_ids_seen.items()))
    return (
        f"[PASSIVE_BUFF_SUMMARY] ep={ep} lineup={list(lineup)} side={side} hero={hero_id} "
        f"final_fno={final_frame} attacks_attempted={attacks_attempted} "
        f"illegal_button_count={illegal_button_count} skill_ids=[{sk}] mark_ids=[{mk}]"
    )


__all__ = [
    "AIR_ATTACK",
    "LUBAN_SKILL1",
    "LubanSkill1PassiveBuffDebugAgent",
    "NO_OP",
    "PassiveBuffDebugAgent",
    "extract_buff_state",
    "find_own_hero",
    "format_passive_buff_line",
    "format_passive_summary_line",
    "parse_attack_no",
]
