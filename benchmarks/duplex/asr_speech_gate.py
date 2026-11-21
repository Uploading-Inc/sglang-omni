# SPDX-License-Identifier: Apache-2.0
"""Uploading-Inc fork: a Silero VAD gate in front of an own-ASR recogniser. A segment with no speech region gets no word.

Qwen3-ASR-1.7B writes "嗯。" for 10 s of digital silence and for -50 dBFS white noise (Phase 1 smoke,
placid_wire_tq79y41yyk), so a model that stays silent would be scored as answering with a backchannel. Filtering filler
words would delete real backchannels, which the duplex endpoints measure, so the gate asks the acoustic question instead:
is there any speech in the segment the recogniser is given? If none, the recogniser is not run. If there is, it runs on
the whole segment as before, so its timestamps stay on the segment's own timeline.

Parameters (omni-pretrain docs/prereg/duplex_controller.md §15), fixed before any Phase 1 result:
- threshold 0.3: the highest Silero frame probability over 57 silence / stationary-noise signals (zeros; white, pink and
  brown noise at -70 to -20 dBFS, 3 seeds each; 50 Hz hum; one click) is 0.127, so 0.3 is a 2.4x margin, and the
  span-end threshold Silero derives from it (0.3 - 0.15 = 0.15) is also above every noise frame: noise can neither
  start a span nor keep one going. Lower is what keeps quiet real backchannels: with Silero's defaults (0.5, 250 ms),
  5 of the 199 FDB-zh user_backchannel clips (each set alone 3 s into 10 s of silence) have no speech region; with
  0.3 / 100 ms, 2 do (a "哦哦哦" and a "哎"), and all 31 clips that are only "嗯" pass;
- min_speech_duration_ms 100: about three Silero windows, under the shortest "嗯" in FDB-zh (0.40 s); 250 ms (Silero's
  default and unified-score's) loses 3 more clips;
- min_silence_duration_ms 100 and speech_pad_ms 0, as unified-score's; neither can change whether a region exists.
"""

from __future__ import annotations

import importlib.metadata
import importlib.resources
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import soundfile
import torch
from pydantic import JsonValue

from benchmarks.duplex.run_artifacts import file_sha256

SILERO_VAD_VERSION = "6.2.1"
SILERO_VAD_MODEL = "silero_vad.jit"
SILERO_VAD_MODEL_SHA256 = "e1122837f4154c511485fe0b9c64455f7b929c96fbb8d79fbdb336383ebd3720"
GATE_RATE = 16000
GATE_CONFIG = {"threshold": 0.3, "min_speech_duration_ms": 100, "min_silence_duration_ms": 100, "speech_pad_ms": 0}

Span = tuple[float, float]


def silero_model_path() -> Path:
    return Path(str(importlib.resources.files("silero_vad.data") / SILERO_VAD_MODEL))


def load_gate_vad() -> Any:
    """The pinned Silero model; any other package version or model file stops the run."""
    version = importlib.metadata.version("silero-vad")
    if version != SILERO_VAD_VERSION:
        raise SystemExit(f"silero-vad {version} installed; the speech gate is pinned to {SILERO_VAD_VERSION}")
    elif file_sha256(silero_model_path()) != SILERO_VAD_MODEL_SHA256:
        raise SystemExit(f"{silero_model_path()} does not match the speech gate's pinned SHA-256")
    else:
        from silero_vad import load_silero_vad

        return load_silero_vad(onnx=False)


def gate_receipt() -> dict[str, JsonValue]:
    return {"package": "silero-vad", "version": SILERO_VAD_VERSION, "model": SILERO_VAD_MODEL,
            "model_sha256": SILERO_VAD_MODEL_SHA256, "sample_rate": GATE_RATE, "config": dict(GATE_CONFIG)}


def speech_regions(audio_path: Path, vad: Any) -> list[Span]:
    """Silero speech regions [start, end) in seconds of a 16 kHz mono segment; [] for an empty one."""
    from silero_vad import get_speech_timestamps

    info = soundfile.info(str(audio_path))
    if (info.samplerate, info.channels) != (GATE_RATE, 1):
        raise SystemExit(f"{audio_path} is {info.samplerate} Hz, {info.channels} channels; the gate reads 16 kHz mono")
    else:
        pass
    audio, _ = soundfile.read(str(audio_path), dtype="float32")
    if audio.size == 0:
        return []
    else:
        timestamps = get_speech_timestamps(torch.from_numpy(np.ascontiguousarray(audio)), vad,
                                           sampling_rate=GATE_RATE, **GATE_CONFIG)
        return [(span["start"] / GATE_RATE, span["end"] / GATE_RATE) for span in timestamps]


def gated(recognise: Callable[[Path], list[tuple[str, float, float]]],
          vad: Any) -> Callable[[Path], list[tuple[str, float, float]]]:
    """recognise, run only on a segment that holds a speech region."""

    def recognise_speech(audio_path: Path) -> list[tuple[str, float, float]]:
        if speech_regions(audio_path, vad):
            return recognise(audio_path)
        else:
            return []

    return recognise_speech
