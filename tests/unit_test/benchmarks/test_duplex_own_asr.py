# SPDX-License-Identifier: Apache-2.0
"""own_asr writes the transcripts and the receipt that unified-score reads, cropping interruptions at R."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import soundfile

from benchmarks.duplex import own_asr, unified_v10
from benchmarks.duplex.reference_asr import validate_transcript
from benchmarks.duplex.reference_v10 import MANIFEST_KIND

SAMPLE_RATE = 16000
WINDOW_S = 12.0
INTERRUPT = [{"context": "tell me about tea", "interrupt": "and coffee", "timestamp": [3.0, 5.0]}]
PAUSES = [{"text": "[PAUSE]", "timestamp": [1.0, 2.0]}]


def build_export(root: Path) -> Path:
    tree = root / "tree"
    rows = []
    for sample_id, annotation_name, annotation in (
        ("synthetic_pause_handling/1", "pause.json", PAUSES),
        ("synthetic_user_interruption/1", "interrupt.json", INTERRUPT),
    ):
        sample_dir = tree / sample_id
        sample_dir.mkdir(parents=True)
        soundfile.write(sample_dir / "output.wav", np.zeros(int(WINDOW_S * SAMPLE_RATE), dtype=np.int16), SAMPLE_RATE)
        (sample_dir / annotation_name).write_text(json.dumps(annotation))
        rows.append({
            "sample_id": sample_id, "subset": sample_id.split("/")[0], "eligible": True, "reasons": [],
            "window": {"T_s": WINDOW_S},
            "files": {name: hashlib.sha256((sample_dir / name).read_bytes()).hexdigest()
                      for name in ("output.wav", annotation_name)},
        })
    (tree / "manifest.json").write_text(json.dumps({"kind": MANIFEST_KIND, "samples": rows}))
    return tree


def test_own_transcripts_score_under_the_unified_rules(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    tree = build_export(tmp_path)
    heard = []

    def recognise(audio_path: Path) -> list[own_asr.TimedWord]:
        duration_s = soundfile.info(str(audio_path)).duration
        heard.append(round(duration_s, 3))
        return [("hello", 0.5, 0.75), ("there", 2.5, 2.75)]

    counts = own_asr.transcribe_tree(tree, recognise, {"backend": "fake"})
    assert counts == {"synthetic_pause_handling": 1, "synthetic_user_interruption": 1}
    assert sorted(heard) == [7.0, 12.0]                                   # the interruption is cropped at R = 5.0
    interrupted = json.loads((tree / "synthetic_user_interruption/1/output.json").read_text())
    assert [chunk["timestamp"] for chunk in interrupted["chunks"]] == [[5.5, 5.75], [7.5, 7.75]]
    assert validate_transcript(interrupted, WINDOW_S) == []
    receipt = json.loads((tree / "asr.json").read_text())
    assert receipt["config"] == {"backend": "fake"}
    assert receipt["subsets"]["synthetic_user_interruption"]["task"] == "user_interruption"
    monkeypatch.setattr(unified_v10, "load_silero_vad", lambda: None)
    monkeypatch.setattr(unified_v10, "speech_segments", lambda path, window_s, vad: [(2.0, 4.0), (5.5, 6.0)])
    scored = unified_v10.score_tree(tree)["subsets"]
    assert scored["synthetic_pause_handling"]["metrics"]["pause_intrusion_rate"]["count"] == 1
    assert scored["synthetic_user_interruption"]["counts"]["eligible"] == 1
    with pytest.raises(SystemExit):
        own_asr.transcribe_tree(tree, recognise, {"backend": "fake"})       # never overwrites a receipt
