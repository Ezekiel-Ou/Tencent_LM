"""
agent_ppo debug package.

Debug-only utilities used to instrument the env with a fixed-action
script and dump observation statistics. Not used in the training path.
Activated via [debug] section in agent_ppo/conf/train_env_conf.toml.
"""

from agent_ppo.debug.debug_agent import DebugAgent
from agent_ppo.debug.dump_collector import DumpCollector
from agent_ppo.debug.luban_buff_debug import (
    LubanBuffDebugAgent,
    extract_buff_state,
    find_luban_hero,
    format_buff_line,
    format_summary_line,
    parse_attack_no,
)
from agent_ppo.debug.passive_buff_debug import (
    LubanSkill1PassiveBuffDebugAgent,
    PassiveBuffDebugAgent,
    extract_buff_state as extract_passive_buff_state,
    find_own_hero,
    format_passive_buff_line,
    format_passive_summary_line,
    parse_attack_no as parse_passive_attack_no,
)

__all__ = [
    "DebugAgent",
    "DumpCollector",
    "LubanBuffDebugAgent",
    "LubanSkill1PassiveBuffDebugAgent",
    "PassiveBuffDebugAgent",
    "extract_buff_state",
    "extract_passive_buff_state",
    "find_luban_hero",
    "find_own_hero",
    "format_buff_line",
    "format_passive_buff_line",
    "format_passive_summary_line",
    "format_summary_line",
    "parse_attack_no",
    "parse_passive_attack_no",
]
