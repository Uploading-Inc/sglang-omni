# SPDX-License-Identifier: Apache-2.0
"""The systems the runbook can serve and record, one per interaction strategy."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, replace
from typing import Literal

from benchmarks.duplex.profiles import PROFILES, ProfileName
from benchmarks.duplex.reference_capture import PROTOCOL_FORMAT, TraceFormat
from sglang_omni.scheduling.session import NODE_STAGE_MAX_OPEN_SESSIONS

EngineName = Literal["minicpmo", "qwen3-omni", "cascade"]


@dataclass(frozen=True, kw_only=True)
class Engine:

    name: EngineName
    model_id: str
    model_revision: str | None
    record_profile: ProfileName
    launch_config: str | None
    hf_model_ids: tuple[str, ...]
    server_gpu_count: int
    max_sessions: int
    sets_max_sessions: bool
    needs_aligner: bool

    @property
    def trace_format(self) -> TraceFormat:
        return PROTOCOL_FORMAT[PROFILES[self.record_profile].protocol]

    @property
    def turn_detection(self) -> str | None:
        """Native systems take no turn_detection, turn-based ones detect turns
        by server VAD."""
        if PROFILES[self.record_profile].protocol == "native":
            return None
        else:
            return turn_detection_override() or json.dumps(
                {"type": "server_vad", "silence_duration_ms": 500}
            )

    @property
    def ready_text(self) -> str:
        """What /v1/realtime/capabilities answers once the server takes sessions."""
        if PROFILES[self.record_profile].protocol == "native":
            return '"native_full_duplex":true'
        else:
            return '"interaction":"turn_based"'


ENGINES: dict[EngineName, Engine] = {
    "minicpmo": Engine(
        name="minicpmo",
        model_id="openbmb/MiniCPM-o-4_5",
        model_revision="503e754207c94da6bb26850b4469f367c9ea3582",
        record_profile="minicpmo-native-pr2377",
        launch_config=None,
        hf_model_ids=(),
        server_gpu_count=1,
        max_sessions=8,
        sets_max_sessions=True,
        needs_aligner=False,
    ),
    "qwen3-omni": Engine(
        name="qwen3-omni",
        model_id="Qwen/Qwen3-Omni-30B-A3B-Instruct",
        model_revision=None,
        record_profile="turn-based",
        launch_config="examples/qwen3_omni/session_h100.yaml",
        hf_model_ids=("Qwen/Qwen3-Omni-30B-A3B-Instruct",),
        server_gpu_count=2,
        # note (jeffro): the launch config owns the capacity and explains it;
        # this copy only sizes --num-shards, and a unit test keeps them equal.
        max_sessions=4,
        sets_max_sessions=False,
        needs_aligner=False,
    ),
    "cascade": Engine(
        name="cascade",
        model_id="Qwen/Qwen3.8-27B",
        model_revision=None,
        record_profile="turn-based",
        launch_config=(
            "examples/streaming_cascade/profiles/"
            "qwen3-asr-qwen3.8-27b-vivian/launch.yaml"
        ),
        hf_model_ids=(
            "Qwen/Qwen3.8-27B",
            "Qwen/Qwen3-ASR-1.7B",
            "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
            "Qwen/Qwen3-ForcedAligner-0.6B",
        ),
        server_gpu_count=2,
        max_sessions=NODE_STAGE_MAX_OPEN_SESSIONS,
        sets_max_sessions=False,
        needs_aligner=True,
    ),
}


# Uploading-Inc fork: the turn-based baselines differ only in how a turn ends. FDB_TURN_DETECTION (a JSON
# turn_detection object) replaces the recorder's server_vad 500 ms request, and FDB_LAUNCH_CONFIG (a path relative
# to the repository) replaces the engine's launch config, e.g. examples/qwen3_omni/session_a100_semantic.yaml with
# {"type": "semantic_vad", "eagerness": "medium"}. Both are read once, at import, and recorded in the run manifest
# through the recorder's --turn-detection and the server command.
TURN_DETECTION_ENV = "FDB_TURN_DETECTION"
LAUNCH_CONFIG_ENV = "FDB_LAUNCH_CONFIG"
TURN_DETECTION_TYPES = ("server_vad", "semantic_vad")


def turn_detection_override() -> str | None:
    raw = os.environ.get(TURN_DETECTION_ENV)
    if not raw:
        return None
    else:
        pass
    value = json.loads(raw)
    if not isinstance(value, dict) or value.get("type") not in TURN_DETECTION_TYPES:
        raise ValueError(f"{TURN_DETECTION_ENV} must be a JSON object with type in {TURN_DETECTION_TYPES}")
    else:
        pass
    return json.dumps(value, sort_keys=True)


if os.environ.get(LAUNCH_CONFIG_ENV):
    ENGINES = {
        name: (
            replace(engine, launch_config=os.environ[LAUNCH_CONFIG_ENV])
            if engine.launch_config is not None
            else engine
        )
        for name, engine in ENGINES.items()
    }
else:
    pass
