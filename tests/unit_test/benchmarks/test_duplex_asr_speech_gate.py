# SPDX-License-Identifier: Apache-2.0
"""Uploading-Inc fork: the zh recogniser hears nothing in silence or noise, and still hears a short voiced sound.

No GPU and no Qwen3-ASR: the model is a stand-in that, like Qwen3-ASR-1.7B on the Phase 1 smoke, answers "嗯。" to
anything. The VAD is the real pinned Silero model, so run this where silero-vad 6.2.1 is installed (the aligner venv).
"""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile

from benchmarks.duplex import asr_speech_gate, own_asr
from benchmarks.duplex.own_asr_qwen import qwen_recogniser

RATE = 16000
WINDOW_S = 10.0
VOWEL_AT_S = 3.0


class HallucinatingQwen:
    """Answers "嗯。" with one aligner token wherever it is asked, as Qwen3-ASR does on silence and noise."""

    def __init__(self) -> None:
        self.heard: list[str] = []

    def transcribe(self, audio: list[str], language: list[str], return_time_stamps: bool) -> list[SimpleNamespace]:
        assert language == ["Chinese"] and return_time_stamps
        self.heard.extend(audio)
        return [SimpleNamespace(text="嗯。", time_stamps=[SimpleNamespace(text="嗯", start_time=3.04, end_time=3.36)])]


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
