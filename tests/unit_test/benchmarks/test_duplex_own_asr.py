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


# Phase 1 zh units whose aligner word started on the reply's last 80 ms frame end, past T, and ended one frame later.
# ashy_ocean_72t46qrk7x failed on the start (prereg §16); with the start moved, helpful_hamster_pvjn44n8pj failed on
# the end (prereg §17). The 16 kHz output.wav length in samples, the interruption end R (None for turn-taking), and
# unified-score's errors without on_segment; the last of those is helpful_hamster's error string where it is one of its
# four units, because the end is not touched by the start move.
PAST_THE_END_UNITS = [
    ("zh_turn_taking/74", 298154, None,
     ["word start 18.64 is outside [0, 18.634625]", "word end 18.72 is outside [0, 18.714625]"]),
    ("zh_turn_taking/128", 307063, None,
     ["word start 19.2 is outside [0, 19.191437]", "word end 19.28 is outside [0, 19.271437]"]),
    ("zh_turn_taking/153", 296132, None,
     ["word start 18.56 is outside [0, 18.50825]", "word end 18.64 is outside [0, 18.58825]"]),
    ("zh_user_interruption/161", 344044, 6.475,
     ["word start 21.515 is outside [0, 21.50275]", "word end 21.595 is outside [0, 21.58275]"]),
    ("zh_turn_taking/124", 291684, None,
     ["word start 18.24 is outside [0, 18.23025]", "word end 18.32 is outside [0, 18.31025]"]),
    ("zh_turn_taking/79", 295094, None,
     ["word start 18.48 is outside [0, 18.443375]", "word end 18.56 is outside [0, 18.523375]"]),
    ("zh_turn_taking/53", 318304, None,
     ["word start 19.92 is outside [0, 19.894]", "word end 20.0 is outside [0, 19.974]"]),
    ("zh_turn_taking/19", 315912, None,
     ["word start 19.76 is outside [0, 19.7445]", "word end 19.84 is outside [0, 19.8245]"]),
]


def last_frame_end(audio_path: Path) -> list[own_asr.TimedWord]:
    """What Qwen3-ForcedAligner gave those words: a start on the end of the segment's last 80 ms frame and an end one
    frame later, each rounded to the ms like qwen-asr."""
    info = soundfile.info(str(audio_path))
    frame = math.ceil(info.frames / info.samplerate / ALIGNER_FRAME_S)
    return [("嗯", round(frame * ALIGNER_FRAME_S, 3), round((frame + 1) * ALIGNER_FRAME_S, 3))]


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


@pytest.mark.parametrize(("unit", "frames", "end_r", "errors"), PAST_THE_END_UNITS)
def test_an_aligner_word_on_the_last_frame_end_scores_once_on_the_segment(
        tmp_path: Path, unit: str, frames: int, end_r: float | None, errors: list[str]) -> None:
    assert unified_problems(tmp_path / "raw", unit, frames, end_r, last_frame_end) == errors
    assert unified_problems(tmp_path / "fixed", unit, frames, end_r, on_segment(last_frame_end)) == []


# funny_tooth_v3lq4t58n7 (prereg §19): several trailing characters one to three frames past the last frame end. The
# 16 kHz length and the reply's last words: the past-the-end times are the Phase 1 errors', the first word is a stand-in.
PILED_UP_UNITS = [
    ("zh_turn_taking/106", 293311, [(18.0, 18.3), (18.56, 18.56), (18.56, 18.56), (18.48, 18.56), (18.48, 18.56)]),
    ("zh_turn_taking/124", 291684, [(18.0, 18.2), (18.32, 18.32), (18.32, 18.4), (18.24, 18.32)]),
    ("zh_turn_taking/79", 295094, [(18.0, 18.4)] + [(18.56, 18.56)] * 6),
]


@pytest.mark.parametrize(("unit", "frames", "words"), PILED_UP_UNITS)
def test_trailing_words_piled_past_the_end_score_on_the_segment(
        tmp_path: Path, unit: str, frames: int, words: list[tuple[float, float]]) -> None:
    def piled(audio_path: Path) -> list[own_asr.TimedWord]:
        return [("嗯", start_s, end_s) for start_s, end_s in words]

    assert any("is outside" in p for p in unified_problems(tmp_path / "raw", unit, frames, None, piled))
    assert unified_problems(tmp_path / "fixed", unit, frames, None, on_segment(piled)) == []


def test_any_start_past_the_end_moves_and_an_end_within_one_frame_stays(tmp_path: Path) -> None:
    def times(*pairs: tuple[float, float]) -> own_asr.Recogniser:
        def recognise(audio_path: Path) -> list[own_asr.TimedWord]:
            info = soundfile.info(str(audio_path))
            d = info.frames / info.samplerate
            return [("嗯", round(d + a, 3), round(d + b, 3)) for a, b in pairs]
        return recognise

    path = tmp_path / "reply.wav"
    soundfile.write(path, np.zeros(296132, dtype=np.int16), SAMPLE_RATE)       # d = 18.50825 s
    assert on_segment(times((-0.5, 0.05), (-0.5, 0.17), (0.3, 0.4), (1.0, 1.0)))(path) == [
        ("嗯", 18.008, 18.558),                                                  # end within one frame: the scorer takes it
        ("嗯", 18.008, 18.50825), ("嗯", 18.50825, 18.50825), ("嗯", 18.50825, 18.50825)]
    assert on_segment(times((-1.0, -0.5)))(path) == [("嗯", 17.508, 18.008)]  # inside: untouched
