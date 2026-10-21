"""
Fixed-action debug agent for agent_ppo feature-field calibration.

The DebugAgent emits a deterministic action sequence so the DumpCollector
can observe value ranges (positions, HP, EP, attack/combat stats,
attack_range, sight_area, cooldowns, money deltas, behave strings,
buff/mark IDs, sub_type encoding, NPC composition, command/event fields).
Phase B feature engineering will be designed against the resulting
agent_ppo/debug/dumps/summary.json.

Action format (length 6, see 开发指南/环境详述.md):
    [button, move_x, move_z, skill_x, skill_z, target]
    button   : 0..11   (0=invalid, 1=no-op, 2=move, 3=attack, 4-6=skills,
                        7=recover, 8=summoner, 9=recall, 10=skill4, 11=equip)
    move_x   : 0..15   (8=center, 0=-x, 15=+x)
    move_z   : 0..15   (8=center, 0=-z, 15=+z)
    skill_x  : 0..15
    skill_z  : 0..15
    target   : 0..8    (0=None, 1=enemy hero, 2=self, 3-6=4 nearest soldiers,
                        7=enemy tower, 8=monster)

The platform debug log may only retain the first ~2000 frames. The env
consumes 6 frames per step, so this script keeps the important probes
inside the first ~330 steps: four-direction coordinate sweeps, lane push,
tower target, soldier/hero/monster targets, and all active button slots.
"""

from typing import List, Optional, Sequence


NO_OP_ACTION: List[int] = [1, 8, 8, 8, 8, 0]  # button=1 no-op, sub-actions ignored
INVALID_ACTION: List[int] = [0, 15, 15, 15, 15, 0]  # framework "skip" sentinel


class _Phase:
    __slots__ = ("start", "end", "action", "tag")

    def __init__(self, start: int, end: int, action: Sequence[int], tag: str):
        self.start = start
        self.end = end
        self.action = list(action)
        self.tag = tag

    def __repr__(self) -> str:
        return f"_Phase({self.tag}, steps={self.start}..{self.end}, action={self.action})"


# Script: each phase is (step_start, step_end_exclusive, action, tag).
# Keep the first full pass within ~2000 frames (333 env steps). If the episode
# continues, a second pass repeats the high-value interactions.
SCRIPT: List[_Phase] = [
    _Phase(   0,   10, [1, 8, 8, 8, 8, 0], "stand_still"),
    _Phase(  10,   28, [2, 15, 8, 8, 8, 0], "probe_plus_x"),
    _Phase(  28,   46, [2, 0, 8, 8, 8, 0], "probe_minus_x"),
    _Phase(  46,   64, [2, 8, 15, 8, 8, 0], "probe_plus_z"),
    _Phase(  64,   82, [2, 8, 0, 8, 8, 0], "probe_minus_z"),
    _Phase(  82,  145, [2, 15, 15, 8, 8, 0], "push_lane_forward"),
    _Phase( 145,  175, [3, 8, 8, 8, 8, 3], "attack_nearest_soldier"),
    _Phase( 175,  205, [3, 8, 8, 8, 8, 7], "attack_enemy_tower"),
    _Phase( 205,  230, [3, 8, 8, 8, 8, 1], "attack_enemy_hero"),
    _Phase( 230,  252, [4, 8, 8, 12, 8, 1], "skill1_enemy_hero"),
    _Phase( 252,  274, [5, 8, 8, 12, 8, 1], "skill2_enemy_hero"),
    _Phase( 274,  296, [6, 8, 8, 12, 8, 1], "skill3_enemy_hero"),
    _Phase( 296,  306, [7, 8, 8, 8, 8, 2], "recover_skill"),
    _Phase( 306,  318, [8, 8, 8, 12, 8, 1], "summoner_skill"),
    _Phase( 318,  334, [3, 8, 8, 8, 8, 8], "attack_monster_probe"),
    _Phase( 334,  420, [2, 0, 0, 8, 8, 0], "return_lane_reverse"),
    _Phase( 420,  520, [2, 15, 15, 8, 8, 0], "second_push_lane_forward"),
    _Phase( 520,  620, [3, 8, 8, 8, 8, 7], "second_tower_pressure"),
    _Phase( 620,  760, [3, 8, 8, 8, 8, 1], "second_hero_pressure"),
    _Phase( 760,  860, [3, 8, 8, 8, 8, 1], "pure_normal_attack_test"),
    _Phase( 860,  950, [2, 15, 8, 8, 8, 0], "walk_into_tower_range"),
    _Phase( 950, 1050, [1, 8, 8, 8, 8, 0], "tower_aggro_stand"),
    _Phase(1050, 1200, [2, 8, 8, 8, 8, 0], "soldier_bullet_test_walk"),
    _Phase(1200, 1350, [1, 8, 8, 8, 8, 0], "soldier_bullet_test_stand"),
    _Phase(1350, 1500, [3, 8, 8, 8, 8, 1], "third_hero_attack_burst"),
]


class DebugAgent:
    """Stateless scripted agent. One instance per training process; fed via workflow."""

    def __init__(self, script: Optional[List[_Phase]] = None):
        self.script = script if script is not None else SCRIPT
        self._max_step = max(p.end for p in self.script) if self.script else 0
        self.illegal_button_count = 0  # bumped when scripted button is masked illegal
        self.calls = 0
        self.phase_counter = {}  # tag -> int

    def reset(self) -> None:
        self.calls = 0
        self.illegal_button_count = 0
        self.phase_counter = {}

    def act(self, observation: dict, step_no: int, agent_id: int = 0) -> List[int]:
        """Return a 6-int action for this step.

        Both agents run the same probe, but agent 1 mirrors directional bins.
        This gives both camps a chance to expose red/blue-specific visibility,
        tower, soldier, bullet, and skill fields in the same debug run.
        """
        self.calls += 1
        action = self._lookup(step_no)
        if agent_id == 1:
            action = self._mirror_action(action)
        legal = observation.get("legal_action") if isinstance(observation, dict) else None
        if legal is not None and not self._is_button_legal(action[0], legal):
            self.illegal_button_count += 1
            return list(NO_OP_ACTION)
        return list(action)

    # ----- internals -----

    def _lookup(self, step_no: int) -> List[int]:
        for phase in self.script:
            if phase.start <= step_no < phase.end:
                self.phase_counter[phase.tag] = self.phase_counter.get(phase.tag, 0) + 1
                return phase.action
        # past the script's end; remain idle
        self.phase_counter["script_exhausted"] = self.phase_counter.get("script_exhausted", 0) + 1
        return list(NO_OP_ACTION)

    @staticmethod
    def _is_button_legal(button_idx: int, legal_action) -> bool:
        # legal_action is the concatenated mask; first 12 entries are the button mask.
        try:
            return bool(legal_action[button_idx])
        except (TypeError, IndexError):
            return True  # be permissive if mask unreadable

    @staticmethod
    def _mirror_action(action: Sequence[int]) -> List[int]:
        mirrored = list(action)
        for idx in (1, 2, 3, 4):
            try:
                mirrored[idx] = 15 - int(mirrored[idx])
            except (TypeError, ValueError, IndexError):
                pass
        return mirrored
