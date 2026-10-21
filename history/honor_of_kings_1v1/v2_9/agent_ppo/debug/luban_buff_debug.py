"""
Focused debug for Luban (config_id=112) sweep / recovery buff identification.

Runs a fixed 4-episode script with lineup forced to [Luban, Luban] on both
camps. Output is restricted to logger.info lines prefixed [LUBAN_BUFF] /
[LUBAN_SUMMARY] so the platform log viewer screenshot is enough to interpret the
result. No JSON dumps, no DumpCollector.

Episode plan (after a pre-move engagement phase ends in each episode):
    ep=0: 4 normal attacks against the nearest enemy soldier
    ep=1: 5 normal attacks against the nearest enemy soldier
    ep=2: skill 1 (Q, button=4), then 1 normal attack
    ep=3: recovery skill (button=7), waited, recovery skill again

Pre-move: walk diagonally toward enemy base for a fixed warmup window, then
enter the strict script once an enemy soldier is visible or a hard cap is hit.
The pre-move phase is silent.

Logging strategy:
- 1 line whenever Luban's buff_state changes between consecutive frames
- 1 [LUBAN_SUMMARY] line at episode end with the union of buff ids observed
"""

from typing import Any, Dict, List, Optional, Sequence, Tuple


NO_OP: List[int] = [1, 8, 8, 8, 8, 0]
INVALID: List[int] = [0, 15, 15, 15, 15, 0]

# Action templates (see 开发指南/环境详述.md for action schema).
MOVE_FORWARD: List[int] = [2, 15, 15, 8, 8, 0]     # walk +x +z toward enemy base
ATTACK_SOLDIER: List[int] = [3, 8, 8, 8, 8, 3]     # normal attack on nearest soldier
SKILL_1: List[int] = [4, 8, 8, 12, 8, 0]           # Luban Q (jumping bombs), aim +x
RECOVER: List[int] = [7, 8, 8, 8, 8, 2]            # recovery, target self


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
    """Read a field that may live at the flat root or nested under actor_state."""
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


def find_luban_hero(observation: Optional[dict]) -> Optional[dict]:
    """Return the Luban (config_id=112) hero dict from this agent's own frame_state.

    The agent's observation is camp-local; the hero list contains both camps'
    heroes but with this agent's camp marked. We return the first hero with
    config_id == 112 — under lineup [112, 112] the agent-perspective owner
    Luban appears first or is uniquely identifiable via camp field, but for
    buff inspection any Luban whose camp matches the observer suffices.
    """
    if not isinstance(observation, dict):
        return None
    fs = observation.get("frame_state")
    if not isinstance(fs, dict):
        return None
    heroes = fs.get("hero_states") or fs.get("heroes") or []
    own_camp = _normalize_camp(observation.get("camp_id") or observation.get("camp"))
    own_player_id = _coerce_int(observation.get("player_id") or observation.get("runtime_id"))
    fallback = None
    for h in heroes:
        if not isinstance(h, dict):
            continue
        cid = _coerce_int(_get_field(h, "config_id", ["configId"]))
        if cid != 112:
            continue
        runtime_id = _coerce_int(_get_field(h, "runtime_id", ["runtimeId", "player_id", "playerId"]))
        if own_player_id is not None and runtime_id == own_player_id:
            return h
        camp = _get_field(h, "camp")
        camp_int = _normalize_camp(camp)
        if fallback is None:
            fallback = h
        if own_camp is not None and camp_int is not None and camp_int == own_camp:
            return h
    return fallback


def extract_buff_state(hero: Optional[dict]) -> Tuple[Tuple[Tuple[int, Optional[int]], ...],
                                                       Tuple[Tuple[int, Optional[int]], ...]]:
    """Return (buff_skills, buff_marks) as sorted tuples of (config_id, secondary).
    Secondary is 'times' for skills and 'layer' for marks. Empty tuples when absent.
    """
    if not isinstance(hero, dict):
        return tuple(), tuple()
    bs = _get_field(hero, "buff_state", ["buffState"])
    if not isinstance(bs, dict):
        return tuple(), tuple()
    skills = []
    for s in bs.get("buff_skills", []) or []:
        if not isinstance(s, dict):
            continue
        cid = _coerce_int(s.get("configId") or s.get("config_id"))
        if cid is None:
            continue
        times = _coerce_int(s.get("times"))
        skills.append((cid, times))
    marks = []
    for container_key in ("buff_marks", "buffMarks", "marks", "mark_states", "markStates"):
        seq = bs.get(container_key)
        if not isinstance(seq, list):
            continue
        for m in seq:
            if not isinstance(m, dict):
                continue
            cid = _coerce_int(
                m.get("configId") or m.get("config_id") or m.get("buff_mark_id")
                or m.get("buffMarkId") or m.get("mark_id") or m.get("markId") or m.get("id")
            )
            if cid is None:
                continue
            layer = _coerce_int(m.get("layer") or m.get("layers") or m.get("level") or m.get("count"))
            marks.append((cid, layer))
        break
    return tuple(sorted(skills)), tuple(sorted(marks))


def enemy_soldier_in_range(observation: Optional[dict], luban: Optional[dict],
                           attack_range: float = 9000.0) -> bool:
    """Return True if any NPC of sub_type soldier on a different camp from Luban
    is within `attack_range` units. Distance is in HOK world units (x,z plane).

    Soldier identification: actor_type=1 && sub_type=11 (matches dump_collector
    constants). Defaults to a generous 9000-unit radius so we engage as soon as a
    soldier wave is *visible*, not strictly within auto-attack range; Luban will
    auto-walk into actual attack range once button=3 is issued.
    """
    if not isinstance(observation, dict) or not isinstance(luban, dict):
        return False
    fs = observation.get("frame_state")
    if not isinstance(fs, dict):
        return False
    npcs = fs.get("npc_states") or []
    luban_loc = _get_field(luban, "location")
    if not isinstance(luban_loc, dict):
        return False
    lx = _coerce_int(luban_loc.get("x"))
    lz = _coerce_int(luban_loc.get("z"))
    if lx is None or lz is None:
        return False
    luban_camp = _normalize_camp(_get_field(luban, "camp"))
    for npc in npcs:
        if not isinstance(npc, dict):
            continue
        actor_type = _coerce_int(_get_field(npc, "actor_type", ["actorType"]))
        sub_type = _coerce_int(_get_field(npc, "sub_type", ["subType"]))
        if actor_type != 1 or sub_type != 11:
            continue
        camp = _normalize_camp(_get_field(npc, "camp"))
        if camp is not None and luban_camp is not None and camp == luban_camp:
            continue
        loc = _get_field(npc, "location")
        if not isinstance(loc, dict):
            continue
        nx = _coerce_int(loc.get("x"))
        nz = _coerce_int(loc.get("z"))
        if nx is None or nz is None:
            continue
        dx = nx - lx
        dz = nz - lz
        if (dx * dx + dz * dz) <= (attack_range * attack_range):
            return True
    return False


class LubanBuffDebugAgent:
    """4-episode scripted agent for identifying Luban sweep & recovery buff ids.

    One shared instance drives both camps. State is per-episode (`episode`,
    `phase`, `engagement_step`) and pure across the two camp calls within a
    step — no per-camp counters. The workflow caller is responsible for
    counting attempts (the action tag returned from `act()` carries
    "attack#N/M" so the caller can parse it for logging).

    Both camps walk forward during pre-move; `agent_id=1` mirrors the
    directional bins so red Luban also walks toward enemy base.
    """

    def __init__(self, attack_cooldown_steps: int = 7, skill_to_attack_gap_steps: int = 8,
                 min_premove_steps: int = 45, max_premove_steps: int = 50):
        self.attack_cooldown_steps = max(1, int(attack_cooldown_steps))
        self.skill_to_attack_gap_steps = max(1, int(skill_to_attack_gap_steps))
        self.min_premove_steps = max(0, int(min_premove_steps))
        self.max_premove_steps = max(1, int(max_premove_steps))
        if self.max_premove_steps < self.min_premove_steps:
            self.max_premove_steps = self.min_premove_steps
        self.episode = 0
        self.phase = "premove"
        self.engagement_step: Optional[int] = None
        self.illegal_button_count = 0
        self.last_action_tag = "init"

    def reset_for_episode(self, episode_idx: int) -> None:
        self.episode = int(episode_idx)
        self.phase = "premove"
        self.engagement_step = None
        self.illegal_button_count = 0
        self.last_action_tag = "init"

    # ---------- core dispatch ----------

    def act(self, observation: Optional[dict], step_no: int, agent_id: int = 0
            ) -> Tuple[List[int], str]:
        """Return (action_list, tag). Pure given (episode, phase, step_no, agent_id)."""
        if self.phase == "premove":
            action, tag = self._premove_act(observation, step_no)
        else:
            action, tag = self._engaged_act(step_no)
        if agent_id == 1:
            action = self._mirror_action(action)
        # legal-action mask: if button is disallowed, fall back to no-op.
        legal = observation.get("legal_action") if isinstance(observation, dict) else None
        if legal is not None and not self._is_button_legal(action[0], legal):
            self.illegal_button_count += 1
            self.last_action_tag = f"{tag}->masked_noop"
            return list(NO_OP), self.last_action_tag
        self.last_action_tag = tag
        return action, tag

    def transition_to_engaged_if_ready(self, observation: Optional[dict], step_no: int,
                                       luban_hero: Optional[dict]) -> bool:
        """Called once per env step (before act()). Returns True if a transition
        just happened so the workflow can emit a phase-transition log line.
        """
        if self.phase != "premove":
            return False
        if step_no < self.min_premove_steps:
            return False
        cond_soldier = enemy_soldier_in_range(observation, luban_hero)
        cond_timeout = step_no >= self.max_premove_steps
        if cond_soldier or cond_timeout:
            self.phase = "engaged"
            self.engagement_step = step_no
            return True
        return False

    # ---------- premove logic ----------

    def _premove_act(self, observation: Optional[dict], step_no: int) -> Tuple[List[int], str]:
        return list(MOVE_FORWARD), "premove"

    # ---------- engaged scripts ----------

    def _engaged_act(self, step_no: int) -> Tuple[List[int], str]:
        if self.engagement_step is None:
            return list(NO_OP), "engaged_no_start"
        local = step_no - self.engagement_step
        if self.episode == 0:
            return self._attack_sequence(local, total=4)
        if self.episode == 1:
            return self._attack_sequence(local, total=5)
        if self.episode == 2:
            return self._skill_then_attack(local)
        if self.episode == 3:
            return self._recover_sequence(local)
        return list(NO_OP), f"unknown_ep{self.episode}"

    def _attack_sequence(self, local_step: int, total: int) -> Tuple[List[int], str]:
        """Issue button=3 every attack_cooldown_steps until `total` attempts."""
        attempt_idx = local_step // self.attack_cooldown_steps
        within = local_step % self.attack_cooldown_steps
        if attempt_idx >= total:
            return list(NO_OP), f"post_attack_idle@{local_step}"
        if within == 0:
            return list(ATTACK_SOLDIER), f"attack#{attempt_idx + 1}/{total}"
        return list(NO_OP), f"cooldown_after_attack#{attempt_idx + 1}"

    def _skill_then_attack(self, local_step: int) -> Tuple[List[int], str]:
        """ep=2: step 0 = skill1; step 1..gap-1 = noop; step gap = single attack."""
        if local_step == 0:
            return list(SKILL_1), "skill1_release"
        if local_step == self.skill_to_attack_gap_steps:
            return list(ATTACK_SOLDIER), "post_skill_attack#1"
        return list(NO_OP), f"ep2_idle@{local_step}"

    def _recover_sequence(self, local_step: int) -> Tuple[List[int], str]:
        """ep=3: button=7 twice (start and ~30 env steps later), no attacks at all."""
        if local_step == 0:
            return list(RECOVER), "recover#1"
        if local_step == 30:
            return list(RECOVER), "recover#2"
        return list(NO_OP), f"ep3_idle@{local_step}"

    def episode_done(self, step_no: int, settle_steps: int = 30) -> bool:
        """Return True once the strict script has had time to expose buff changes."""
        if self.engagement_step is None:
            return False
        local = step_no - self.engagement_step
        settle_steps = max(1, int(settle_steps))
        if self.episode == 0:
            return local >= self.attack_cooldown_steps * 4 + settle_steps
        if self.episode == 1:
            return local >= self.attack_cooldown_steps * 5 + settle_steps
        if self.episode == 2:
            return local >= self.skill_to_attack_gap_steps + settle_steps
        if self.episode == 3:
            return local >= 30 + settle_steps
        return local >= settle_steps

    # ---------- helpers ----------

    @staticmethod
    def _is_button_legal(button_idx: int, legal_action) -> bool:
        try:
            return bool(legal_action[button_idx])
        except (TypeError, IndexError):
            return True

    @staticmethod
    def _mirror_action(action: Sequence[int]) -> List[int]:
        mirrored = list(action)
        for idx in (1, 2, 3, 4):
            try:
                mirrored[idx] = 15 - int(mirrored[idx])
            except (TypeError, ValueError, IndexError):
                pass
        return mirrored


def parse_attack_no(tag: str) -> Optional[int]:
    if not isinstance(tag, str) or "attack#" not in tag:
        return None
    try:
        tail = tag.split("attack#", 1)[1]
        return int(tail.split("/", 1)[0].split(" ", 1)[0])
    except (IndexError, TypeError, ValueError):
        return None


def format_buff_line(prefix: str, ep: int, step: int, frame_no: Any, side: str, tag: str,
                     action: Sequence[int], attack_no: Optional[int], skills, marks,
                     extra: str = "") -> str:
    skills_repr = "[]" if not skills else "[" + ",".join(f"({c},{t})" for c, t in skills) + "]"
    marks_repr = "[]" if not marks else "[" + ",".join(f"({c},{l})" for c, l in marks) + "]"
    attack_repr = "-" if attack_no is None else str(attack_no)
    base = (
        f"{prefix} ep={ep} side={side} step={step} fno={frame_no} "
        f"action={tag} attack#={attack_repr} act={list(action)} "
        f"skills={skills_repr} marks={marks_repr}"
    )
    if extra:
        base += f" {extra}"
    return base


def format_summary_line(ep: int, side: str, attacks_attempted: int,
                        skill_ids_seen: Dict[int, int],
                        mark_ids_seen: Dict[int, int],
                        final_frame: Any) -> str:
    sk = ",".join(f"{cid}:max_times={t}" for cid, t in sorted(skill_ids_seen.items()))
    mk = ",".join(f"{cid}:max_layer={l}" for cid, l in sorted(mark_ids_seen.items()))
    return (
        f"[LUBAN_SUMMARY] ep={ep} side={side} final_fno={final_frame} "
        f"attacks_attempted={attacks_attempted} skill_ids=[{sk}] mark_ids=[{mk}]"
    )


__all__ = [
    "LubanBuffDebugAgent",
    "find_luban_hero",
    "extract_buff_state",
    "enemy_soldier_in_range",
    "format_buff_line",
    "format_summary_line",
    "parse_attack_no",
    "NO_OP",
    "MOVE_FORWARD",
    "ATTACK_SOLDIER",
    "SKILL_1",
    "RECOVER",
]
