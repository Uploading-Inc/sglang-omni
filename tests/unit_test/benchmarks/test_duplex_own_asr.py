# SPDX-License-Identifier: Apache-2.0
"""own_asr writes the transcripts and the receipt that unified-score reads, cropping interruptions at R."""

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pytest
import soundfile

from benchmarks.duplex import own_asr, unified_v10
from benchmarks.duplex.reference_asr import validate_transcript
from benchmarks.duplex.own_asr_qwen import ALIGNER_FRAME_S, on_segment
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


# Phase 1 (ashy_ocean_72t46qrk7x): zh units whose aligner word started on the reply's last 80 ms frame end, past T.
# The 16 kHz output.wav length in samples, the interruption end R (None for turn-taking), and unified-score's error.
PAST_THE_END_UNITS = [
    ("zh_turn_taking/74", 298154, None, "word start 18.64 is outside [0, 18.634625]"),
    ("zh_turn_taking/128", 307063, None, "word start 19.2 is outside [0, 19.191437]"),
    ("zh_turn_taking/153", 296132, None, "word start 18.56 is outside [0, 18.50825]"),
    ("zh_user_interruption/161", 344044, 6.475, "word start 21.515 is outside [0, 21.50275]"),
]


def last_frame_end(audio_path: Path) -> list[own_asr.TimedWord]:
    """Where Qwen3-ForcedAligner started those words: the end of the segment's last 80 ms frame, to the ms."""
    info = soundfile.info(str(audio_path))
    start_s = round(math.ceil(info.frames / info.samplerate / ALIGNER_FRAME_S) * ALIGNER_FRAME_S, 3)
    return [("嗯", start_s, start_s)]


def unified_problems(tmp_path: Path, unit: str, frames: int, end_r: float | None,
                     recognise: own_asr.Recogniser) -> list[str]:
    sample_dir = tmp_path / unit
    sample_dir.mkdir(parents=True, exist_ok=True)
    soundfile.write(sample_dir / "output.wav", np.zeros(frames, dtype=np.int16), SAMPLE_RATE)
    task, reference = ("turn_taking", 5.0) if end_r is None else ("user_interruption", (end_r - 1.0, end_r))
    if end_r is not None:
        (sample_dir / "interrupt.json").write_text(json.dumps([{"timestamp": [end_r - 1.0, end_r]}]))
    document = json.loads(own_asr.transcribe_sample(sample_dir, task, recognise))
    spans = unified_v10.word_spans(document["chunks"])
    return unified_v10.timestamp_problems(task, spans, reference, unified_v10.seconds(frames / SAMPLE_RATE))


@pytest.mark.parametrize(("unit", "frames", "end_r", "error"), PAST_THE_END_UNITS)
def test_an_aligner_start_on_the_last_frame_end_scores_once_on_the_segment(
        tmp_path: Path, unit: str, frames: int, end_r: float | None, error: str) -> None:
    assert unified_problems(tmp_path / "raw", unit, frames, end_r, last_frame_end) == [error]
    assert unified_problems(tmp_path / "fixed", unit, frames, end_r, on_segment(last_frame_end)) == []


def test_an_aligner_start_beyond_one_frame_still_fails_unified_score(tmp_path: Path) -> None:
    def past(audio_path: Path) -> list[own_asr.TimedWord]:
        return [("嗯", start_s + ALIGNER_FRAME_S + 0.001, start_s + ALIGNER_FRAME_S + 0.001)
                for _, start_s, _ in last_frame_end(audio_path)]

    problems = unified_problems(tmp_path, "zh_turn_taking/153", 296132, None, on_segment(past))
    assert "word start 18.641 is outside [0, 18.50825]" in problems
