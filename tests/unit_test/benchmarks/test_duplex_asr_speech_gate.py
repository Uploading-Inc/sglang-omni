# SPDX-License-Identifier: Apache-2.0
"""Uploading-Inc fork: the zh recogniser hears nothing in silence or noise, and still hears a short voiced sound.

A word the aligner starts on its last frame, past the reply's end, is moved onto that end (own_asr_qwen.on_segment).

No GPU and no Qwen3-ASR: the model is a stand-in that, like Qwen3-ASR-1.7B on the Phase 1 smoke, answers "嗯。" to
anything. The VAD is the real pinned Silero model, so run this where silero-vad 6.2.1 is installed (the aligner venv).
"""

import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile

from benchmarks.duplex import asr_speech_gate, own_asr
from benchmarks.duplex.own_asr_qwen import ALIGNER_FRAME_S, check_aligner_frame, qwen_recogniser

RATE = 16000
WINDOW_S = 10.0
VOWEL_AT_S = 3.0
# Phase 1 units whose aligner word started past the reply's end (ashy_ocean_72t46qrk7x): 16 kHz output.wav length in
# samples, and the interruption end R that own_asr crops at, or None for turn-taking.
# Phase 1 units: ashy_ocean_72t46qrk7x (prereg §16), then helpful_hamster_pvjn44n8pj (prereg §17)
PAST_THE_END_UNITS = [("zh_turn_taking/74", 298154, None), ("zh_turn_taking/128", 307063, None),
                      ("zh_turn_taking/153", 296132, None), ("zh_user_interruption/161", 344044, 6.475),
                      ("zh_turn_taking/124", 291684, None), ("zh_turn_taking/79", 295094, None),
                      ("zh_turn_taking/53", 318304, None), ("zh_turn_taking/19", 315912, None)]


class HallucinatingQwen:
    """Answers "嗯。" with one aligner token wherever it is asked, as Qwen3-ASR does on silence and noise."""

    def __init__(self) -> None:
        self.heard: list[str] = []

    def transcribe(self, audio: list[str], language: list[str], return_time_stamps: bool) -> list[SimpleNamespace]:
        assert language == ["Chinese"] and return_time_stamps
        self.heard.extend(audio)
        return [SimpleNamespace(text="嗯。", time_stamps=[SimpleNamespace(text="嗯", start_time=3.04, end_time=3.36)])]


class FrameGridQwen:
    """Answers "嗯。" with one token starting at start_s, or by default where Qwen3-ForcedAligner put the eight Phase 1
    words: a start at the end of the segment's last 80 ms frame, ceil(d / 80 ms) * 80 ms, and an end one frame later,
    each rounded to the ms like qwen-asr. A given start_s ends where it starts."""

    def __init__(self, start_s: float | None = None) -> None:
        self.start_s = start_s
        self.segment_s: list[float] = []
        self.starts: list[float] = []
        self.ends: list[float] = []

    def transcribe(self, audio: list[str], language: list[str], return_time_stamps: bool) -> list[SimpleNamespace]:
        info = soundfile.info(audio[0])
        self.segment_s.append(info.frames / info.samplerate)
        frame = math.ceil(info.frames / info.samplerate / ALIGNER_FRAME_S)
        start_s = round(frame * ALIGNER_FRAME_S, 3) if self.start_s is None else self.start_s
        end_s = round((frame + 1) * ALIGNER_FRAME_S, 3) if self.start_s is None else self.start_s
        self.starts.append(start_s)
        self.ends.append(end_s)
        return [SimpleNamespace(text="嗯。", time_stamps=[SimpleNamespace(text="嗯", start_time=start_s,
                                                                         end_time=end_s)])]


def write(path: Path, samples: np.ndarray, rate: int = RATE) -> Path:
    soundfile.write(str(path), np.clip(np.round(samples * 32768), -32768, 32767).astype(np.int16), rate,
                    subtype="PCM_16")
    return path


def resonate(x: np.ndarray, centre_hz: float, bandwidth_hz: float) -> np.ndarray:
    r, theta = np.exp(-np.pi * bandwidth_hz / RATE), 2 * np.pi * centre_hz / RATE
    a1, a2, out = 2 * r * np.cos(theta), -r * r, np.zeros_like(x)
    for n in range(len(x)):
        out[n] = (1 - r) * x[n] + a1 * (out[n - 1] if n > 0 else 0.0) + a2 * (out[n - 2] if n > 1 else 0.0)
    return out


def vowel(duration_s: float) -> np.ndarray:
    """A voiced vowel: a glottal pulse train falling from 130 to 110 Hz through three formants, peak -6 dBFS."""
    n = int(duration_s * RATE)
    phase = np.cumsum(np.linspace(130, 110, n) / RATE)
    pulses = np.diff(np.floor(phase), prepend=0.0)
    source = np.zeros(n)
    for i in range(n):
        source[i] = pulses[i] + 0.97 * (source[i - 1] if i else 0.0)
    source += 0.002 * np.random.default_rng(0).standard_normal(n)
    for centre_hz, bandwidth_hz in ((700, 110), (1220, 120), (2600, 160)):
        source = resonate(source, centre_hz, bandwidth_hz)
    envelope = np.minimum(1, np.minimum(np.arange(n), n - np.arange(n)) / (0.04 * RATE))
    shaped = source * envelope
    return shaped / np.abs(shaped).max() * 10 ** (-6 / 20)


def window_with(clip: np.ndarray) -> np.ndarray:
    samples = np.zeros(int(WINDOW_S * RATE))
    start = int(VOWEL_AT_S * RATE)
    samples[start:start + len(clip)] = clip
    return samples


@pytest.fixture(scope="module")
def vad():
    return asr_speech_gate.load_gate_vad()


def no_speech_signals() -> dict[str, np.ndarray]:
    n = int(WINDOW_S * RATE)
    signals = {"silence": np.zeros(n)}
    for seed in range(3):
        for level_db in (-50, -30):
            signals[f"white{level_db}dBFS-seed{seed}"] = (np.random.default_rng(seed).standard_normal(n)
                                                         * 10 ** (level_db / 20))
    return signals


def test_the_gate_is_the_preregistered_one() -> None:
    assert asr_speech_gate.GATE_CONFIG == {"threshold": 0.3, "min_speech_duration_ms": 100,
                                           "min_silence_duration_ms": 100, "speech_pad_ms": 0}
    assert asr_speech_gate.gate_receipt()["model_sha256"] == asr_speech_gate.SILERO_VAD_MODEL_SHA256


@pytest.mark.parametrize("name", sorted(no_speech_signals()))
def test_silence_and_noise_never_reach_qwen(tmp_path: Path, vad, name: str) -> None:
    path = write(tmp_path / f"{name}.wav", no_speech_signals()[name])
    model = HallucinatingQwen()
    assert asr_speech_gate.speech_regions(path, vad) == []
    assert qwen_recogniser(model, "Chinese", vad)(path) == []
    assert model.heard == []


@pytest.mark.parametrize("duration_s", [0.3, 0.4])
def test_a_short_voiced_sound_is_transcribed_on_its_own_timeline(tmp_path: Path, vad, duration_s: float) -> None:
    path = write(tmp_path / "vowel.wav", window_with(vowel(duration_s)))
    regions = asr_speech_gate.speech_regions(path, vad)
    assert len(regions) == 1
    assert VOWEL_AT_S - 0.05 <= regions[0][0] and regions[0][1] <= VOWEL_AT_S + duration_s + 0.05
    model = HallucinatingQwen()
    assert qwen_recogniser(model, "Chinese", vad)(path) == [("嗯", 3.04, 3.36)]   # the whole file, times untouched
    assert model.heard == [str(path)]


def test_a_silent_pause_sample_gets_an_empty_transcript(tmp_path: Path, vad) -> None:
    sample_dir = tmp_path / "zh_pause_handling" / "1"
    sample_dir.mkdir(parents=True)
    write(sample_dir / "output.wav", np.zeros(int(WINDOW_S * RATE)))
    document = json.loads(own_asr.transcribe_sample(sample_dir, "pause_handling",
                                                    qwen_recogniser(HallucinatingQwen(), "Chinese", vad)))
    assert document == {"text": "", "chunks": []}


def test_an_empty_segment_has_no_speech(tmp_path: Path, vad) -> None:
    path = write(tmp_path / "empty.wav", np.zeros(0))
    model = HallucinatingQwen()
    assert qwen_recogniser(model, "Chinese", vad)(path) == [] and model.heard == []


def test_other_rates_are_refused(tmp_path: Path, vad) -> None:
    with pytest.raises(SystemExit):
        asr_speech_gate.speech_regions(write(tmp_path / "24k.wav", np.zeros(24000), 24000), vad)


def test_the_loader_refuses_another_silero(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(asr_speech_gate.importlib.metadata, "version", lambda name: "6.2.0")
    with pytest.raises(SystemExit, match="pinned to 6.2.1"):
        asr_speech_gate.load_gate_vad()
    monkeypatch.undo()
    monkeypatch.setattr(asr_speech_gate, "file_sha256", lambda path: "0" * 64)
    with pytest.raises(SystemExit, match="SHA-256"):
        asr_speech_gate.load_gate_vad()


@pytest.mark.parametrize(("unit", "frames", "end_r"), PAST_THE_END_UNITS)
def test_a_word_on_the_aligners_last_frame_end_is_moved_onto_the_reply_end(
        tmp_path: Path, vad, unit: str, frames: int, end_r: float | None) -> None:
    sample_dir = tmp_path / unit
    sample_dir.mkdir(parents=True)
    samples = np.zeros(frames)
    clip = vowel(0.4)
    samples[10 * RATE:10 * RATE + len(clip)] = clip                       # after R, so the crop keeps it
    write(sample_dir / "output.wav", samples)
    task = "turn_taking" if end_r is None else "user_interruption"
    if end_r is not None:
        (sample_dir / "interrupt.json").write_text(json.dumps([{"timestamp": [end_r - 1.0, end_r]}]))
    model = FrameGridQwen()
    document = json.loads(own_asr.transcribe_sample(sample_dir, task, qwen_recogniser(model, "Chinese", vad)))
    (segment_s,), (aligner_start_s,) = model.segment_s, model.starts
    assert 0 < aligner_start_s - segment_s < ALIGNER_FRAME_S               # the Phase 1 failure, reproduced
    window_s = round(frames / RATE, 6)                                     # T as unified-score reads it
    ((start_s, end_s),) = [chunk["timestamp"] for chunk in document["chunks"]]
    assert 0 <= start_s <= window_s and window_s - start_s < 2e-6          # the scorer's [0, T], at T
    (aligner_end_s,) = model.ends
    assert ALIGNER_FRAME_S < aligner_end_s - segment_s <= 2 * ALIGNER_FRAME_S   # helpful_hamster's failure
    assert start_s <= end_s <= window_s and window_s - end_s < 2e-6        # the end is moved onto T as well


def test_a_start_any_distance_past_the_end_is_moved_onto_it(tmp_path: Path, vad) -> None:
    path = write(tmp_path / "reply.wav", window_with(vowel(0.4)))
    for start_s in (WINDOW_S + ALIGNER_FRAME_S + 0.001, WINDOW_S + 1.0):
        assert qwen_recogniser(FrameGridQwen(start_s), "Chinese", vad)(path) == [("嗯", WINDOW_S, WINDOW_S)]


def test_a_start_on_the_segment_end_is_untouched(tmp_path: Path, vad) -> None:
    path = write(tmp_path / "reply.wav", window_with(vowel(0.4)))
    assert qwen_recogniser(FrameGridQwen(WINDOW_S), "Chinese", vad)(path) == [("嗯", WINDOW_S, WINDOW_S)]


def test_the_aligner_frame_is_checked() -> None:
    def aligner(frame_ms: float) -> SimpleNamespace:
        return SimpleNamespace(forced_aligner=SimpleNamespace(timestamp_segment_time=frame_ms))

    model = aligner(80)
    assert check_aligner_frame(model) is model
    with pytest.raises(SystemExit, match="40 ms frames"):
        check_aligner_frame(aligner(40))
