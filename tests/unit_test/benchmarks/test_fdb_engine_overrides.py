# SPDX-License-Identifier: Apache-2.0
"""Uploading-Inc fork: FDB_TURN_DETECTION and FDB_LAUNCH_CONFIG select a turn-based baseline."""

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
PROBE = (
    "from benchmarks.duplex.fdb.engines import ENGINES\n"
    "q = ENGINES['qwen3-omni']\n"
    "print(q.launch_config, q.turn_detection, ENGINES['minicpmo'].launch_config, ENGINES['minicpmo'].turn_detection)\n"
)


def probe(**env: str) -> str:
    return subprocess.run([sys.executable, "-c", PROBE], cwd=ROOT, capture_output=True, text=True,
                          env={**os.environ, "PYTHONPATH": str(ROOT), **env}).stdout + ""


def test_defaults_are_unchanged() -> None:
    out = probe()
    assert out.split()[0] == "examples/qwen3_omni/session_h100.yaml"
    assert '"silence_duration_ms": 500' in out and out.rstrip().endswith("None None")


def test_overrides_apply_to_turn_based_engines_only() -> None:
    td = json.dumps({"type": "semantic_vad", "eagerness": "medium"})
    out = probe(FDB_TURN_DETECTION=td, FDB_LAUNCH_CONFIG="examples/qwen3_omni/session_a100_semantic.yaml")
    assert out.split()[0] == "examples/qwen3_omni/session_a100_semantic.yaml"
    assert '"type": "semantic_vad"' in out and out.rstrip().endswith("None None")


def test_a_bad_turn_detection_refuses() -> None:
    r = subprocess.run([sys.executable, "-c", PROBE], cwd=ROOT, capture_output=True, text=True,
                       env={**os.environ, "PYTHONPATH": str(ROOT), "FDB_TURN_DETECTION": '{"type": "magic"}'})
    assert r.returncode != 0 and "FDB_TURN_DETECTION" in r.stderr
