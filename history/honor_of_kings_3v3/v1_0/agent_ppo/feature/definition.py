#!/usr/bin/env python3
"""Platform compatibility surface for the active sample protocol."""

from agent_ppo.feature.definition_v1 import (
    ActDataV1,
    ActorDecisionV1,
    ActorFrameV1,
    CommandDecisionV1,
    CommandFrameV1,
    FrameCollectorV1,
    ObsDataV1,
    SampleDataV1,
    TeamFrameV1,
    build_frame_v1,
    lineup_iterator_v1,
    pack_sequence_v1,
    split_reward_payload_v1,
    unpack_sample_v1,
)

SampleData = SampleDataV1
ObsData = ObsDataV1
ActData = ActDataV1
FrameCollector = FrameCollectorV1
build_frame = build_frame_v1
lineup_iterator = lineup_iterator_v1
sample_process = lambda collector: collector.sample_process()

__all__ = (
    "ActData", "ActDataV1", "ActorDecisionV1", "ActorFrameV1",
    "CommandDecisionV1", "CommandFrameV1", "FrameCollector",
    "FrameCollectorV1", "ObsData", "ObsDataV1", "SampleData",
    "SampleDataV1", "TeamFrameV1", "build_frame", "build_frame_v1",
    "lineup_iterator", "lineup_iterator_v1", "pack_sequence_v1",
    "split_reward_payload_v1",
    "sample_process", "unpack_sample_v1",
)
