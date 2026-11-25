# SPDX-License-Identifier: Apache-2.0
"""Transcribe a v1.0 reference tree with Qwen3-ASR and Qwen3-ForcedAligner, for zh datasets.

The artifacts and the interruption crop are own_asr's. Each aligner token is one word; for zh a token is a
character. A segment with no Silero speech region gets no word and is never given to Qwen3-ASR, which writes "嗯。"
for silence and for noise (asr_speech_gate). A word start at most one aligner frame past the segment's end is
moved onto that end, and so is that word's end when it lies more than one and at most two frames past it
(on_segment). Run it from the aligner venv, which holds qwen-asr, its pinned transformers and silero-vad.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import soundfile
import torch

from benchmarks.duplex.asr_speech_gate import gate_receipt, gated, load_gate_vad
from benchmarks.duplex.own_asr import Recogniser, TimedWord, transcribe_tree

ASR_MODEL_ID = "Qwen/Qwen3-ASR-1.7B"
ALIGNER_MODEL_ID = "Qwen/Qwen3-ForcedAligner-0.6B"
MAX_NEW_TOKENS = 512
# Qwen3-ForcedAligner-0.6B's timestamp_segment_time (80 ms): a time is a frame index times this. load_model checks it.
ALIGNER_FRAME_S = 0.08
TIME_DECIMALS = 6


def on_segment(recognise: Recogniser) -> Recogniser:
    """recognise, with a word start no more than one aligner frame past the segment's end d moved onto d.

    The aligner gives a time as an 80 ms frame index. When a segment's length d is not a whole number of frames, its
    last frame ends at ceil(d / 80 ms) * 80 ms, up to one frame past d. In Phase 1 (ashy_ocean_72t46qrk7x), four zh
    replies got a word starting exactly there, 5 to 52 ms past d, which unified-score rejects. Such a start lies in the
    segment's last partial frame, so it becomes d, floored to the microsecond: d + R then still rounds inside [0, T]
    at unified-score's microsecond resolution, even for a 16 kHz segment whose odd length puts T on a rounding
    boundary. A start more than one frame past d is not a frame-grid artefact; it is left as it is, so unified-score
    still fails it.

    Word ends. The aligner ends a word one frame after a start on that last frame end, so such a word ends at
    ceil(d / 80 ms) * 80 ms + 80 ms, more than one and at most two frames past d. unified-score allows an end one frame
    past T (ASR_END_TOLERANCE_S) and rejects this one: in Phase 1 (helpful_hamster_pvjn44n8pj) four zh replies failed
    so, 10 to 37 ms over that allowance, each a word that starts on the last frame end. An end more than one and at
    most two frames past d is moved onto d, like the start, when the word's start (after the move above) is not past d.
    An end at most one frame past d is left as it is (the scorer takes it); an end more than two frames past d, and
    every time of a word whose start stays past d, are left for the scorer to fail. Both moves touch only
    times unified-score rejects, so every transcript that could be scored before is unchanged.
    """

    def recognise_on_segment(audio_path: Path) -> list[TimedWord]:
        info = soundfile.info(str(audio_path))
        duration_s = info.frames / info.samplerate
        end_s = info.frames * 10 ** TIME_DECIMALS // info.samplerate / 10 ** TIME_DECIMALS

        def clamp(start_s: float) -> float:
            return end_s if end_s < start_s <= duration_s + ALIGNER_FRAME_S + 1e-9 else start_s

        def clamp_end(start_s: float, end_time_s: float) -> float:
            past_s = end_time_s - duration_s
            moved = start_s <= end_s and ALIGNER_FRAME_S + 1e-9 < past_s <= 2 * ALIGNER_FRAME_S + 1e-9
            return end_s if moved else end_time_s

        return [(text, clamp(start_s), clamp_end(clamp(start_s), end_time_s))
                for text, start_s, end_time_s in recognise(audio_path)]

    return recognise_on_segment


def check_aligner_frame(model: Any) -> Any:
    """model, if its forced aligner times on ALIGNER_FRAME_S frames; on_segment's one-frame bound depends on it."""
    frame_ms = float(model.forced_aligner.timestamp_segment_time)
    if frame_ms != ALIGNER_FRAME_S * 1000:
        raise SystemExit(f"the forced aligner times on {frame_ms:g} ms frames; on_segment assumes "
                         f"{ALIGNER_FRAME_S * 1000:g} ms")
    else:
        return model


def load_model(asr_model: str = ASR_MODEL_ID, aligner_model: str = ALIGNER_MODEL_ID, device: str = "cuda:0") -> Any:
    from qwen_asr import Qwen3ASRModel

    return check_aligner_frame(Qwen3ASRModel.from_pretrained(
        asr_model, dtype=torch.bfloat16, device_map=device, max_inference_batch_size=1,
        max_new_tokens=MAX_NEW_TOKENS, forced_aligner=aligner_model,
        forced_aligner_kwargs={"dtype": torch.bfloat16, "device_map": device},
    ))


def qwen_recogniser(model: Any, language: str, vad: Any) -> Recogniser:
    """The recogniser own_asr scores with: Qwen3-ASR's aligner tokens, on the segment, behind the speech gate."""

    def recognise(audio_path: Path) -> list[TimedWord]:
        result = model.transcribe(audio=[str(audio_path)], language=[language], return_time_stamps=True)[0]
        if result.text.strip():
            return [(token.text, float(token.start_time), float(token.end_time)) for token in result.time_stamps]
        else:
            return []

    return gated(on_segment(recognise), vad)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tree", type=Path, required=True)
    parser.add_argument("--language", default="Chinese")
    parser.add_argument("--asr-model", default=ASR_MODEL_ID)
    parser.add_argument("--aligner-model", default=ALIGNER_MODEL_ID)
    parser.add_argument("--device", default="cuda:0")
    arguments = parser.parse_args()
    vad = load_gate_vad()
    model = load_model(arguments.asr_model, arguments.aligner_model, arguments.device)
    config = {"backend": "qwen3-asr", "model_id": arguments.asr_model, "aligner_id": arguments.aligner_model,
              "language": arguments.language, "device_type": arguments.device, "speech_gate": gate_receipt(),
              "start_clamp_s": ALIGNER_FRAME_S,
              "end_clamp_s": [ALIGNER_FRAME_S, 2 * ALIGNER_FRAME_S]}
    print(json.dumps(transcribe_tree(arguments.tree, qwen_recogniser(model, arguments.language, vad), config), indent=2))


if __name__ == "__main__":
    main()
