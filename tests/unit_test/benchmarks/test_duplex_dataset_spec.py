# SPDX-License-Identifier: Apache-2.0
"""Uploading-Inc fork: FDB_DATASET_SPEC registers other datasets in the FDB formats."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from benchmarks.duplex import dataset_spec

ROOT = Path(__file__).resolve().parents[3]


def test_default_is_full_duplex_bench() -> None:
    spec = dataset_spec.load_spec(None)
    assert spec["v10"]["candor_turn_taking"] == ("turn_taking", 119)
    assert set(spec["v15"]) == {"user_interruption", "user_backchannel", "talking_to_other", "background_speech"}


def test_spec_validation(tmp_path: Path) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"name": "x", "v10": {"a": {"task": "chat", "declared": 1}}}))
    with pytest.raises(ValueError):
        dataset_spec.load_spec(str(bad))
    bad.write_text(json.dumps({"name": "x", "v15": {"made_up": 3}}))
    with pytest.raises(ValueError):
        dataset_spec.load_spec(str(bad))
    bad.write_text(json.dumps({"name": "x"}))
    with pytest.raises(ValueError):
        dataset_spec.load_spec(str(bad))


def test_a_spec_replaces_the_registry_in_every_module(tmp_path: Path) -> None:
    spec = tmp_path / "fdb-zh.json"
    spec.write_text(json.dumps({
        "name": "fdb-zh", "language": "zh",
        "v10": {"zh_pause_handling": {"task": "pause_handling", "declared": 239},
                "zh_turn_taking": {"task": "turn_taking", "declared": 155}},
        "v15": {"user_backchannel": 199}}))
    probe = (
        "from benchmarks.duplex import v10_dataset, v15_dataset\n"
        "from benchmarks.duplex.fdb import aggregate\n"
        "print(v10_dataset.SUBSET_TASKS, v10_dataset.DECLARED_COUNTS, v15_dataset.SUBSETS, aggregate.CATEGORIES)\n"
    )
    out = subprocess.run([sys.executable, "-c", probe], cwd=ROOT, capture_output=True, text=True, check=True,
                         env={**os.environ, "PYTHONPATH": str(ROOT), dataset_spec.SPEC_ENV: str(spec)}).stdout
    assert "'zh_pause_handling': 'pause_handling'" in out and "'zh_turn_taking': 155" in out
    assert "('user_backchannel',) ('all', 'user_backchannel')" in out
