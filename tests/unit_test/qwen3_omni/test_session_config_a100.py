# SPDX-License-Identifier: Apache-2.0
"""Uploading-Inc fork: the A100 Qwen3-Omni Session profiles (server VAD and Smart Turn) assemble."""

from pathlib import Path

from sglang_omni.config.manager import ConfigManager
from sglang_omni.proto.streams import SESSION_CONTROL

EXAMPLES = Path(__file__).resolve().parents[3] / "examples/qwen3_omni"


def load(name: str):
    return ConfigManager.from_file(str(EXAMPLES / name)).config


def stage(config, name: str):
    return next(s for s in config.stages if s.name == name)


def test_a100_profile_is_the_h100_graph_with_a_triton_talker() -> None:
    a100, h100 = load("session_a100.yaml"), load("session_h100.yaml")
    assert a100.session_graph() == h100.session_graph()
    assert stage(a100, "talker_ar").engine.model_extra["moe_runner_backend"] == "triton"


def test_semantic_profile_routes_vad_through_turn_detection() -> None:
    config = load("session_a100_semantic.yaml")
    graph = config.session_graph()
    assert graph.reading_nodes("vad", SESSION_CONTROL) == ("turn_detection",)
    assert graph.reading_nodes("turn_detection", SESSION_CONTROL) == ("dialog",)
    assert "turn_detection" in graph.publish and "vad" not in graph.publish
    assert stage(config, "vad").factory.silence_duration_ms == 160
    assert stage(config, "talker_ar").engine.model_extra["moe_runner_backend"] == "triton"
