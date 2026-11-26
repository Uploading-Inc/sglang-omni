# SPDX-License-Identifier: Apache-2.0
"""Transcribe a v1.0 reference tree with Qwen3-ASR and Qwen3-ForcedAligner, for zh datasets.

The artifacts and the interruption crop are own_asr's. Each aligner token is one word; for zh a token is a
character. A segment with no Silero speech region gets no word and is never given to Qwen3-ASR, which writes "嗯。"
for silence and for noise (asr_speech_gate). A word time past the segment's end is moved onto that end (a word end
only when it lies more than one aligner frame past it; on_segment). Run it from the aligner venv, which holds qwen-asr, its pinned transformers and silero-vad.
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
    """recognise, with every word time past the segment's end d moved onto d (an end only past one aligner frame).

    The segment has no audio after d, so no word can start there. Qwen3-ForcedAligner can still place one: a time is an
    80 ms frame index chosen by a classifier that is not bounded by the audio's length, and qwen-asr's
    `fix_timestamp` repairs out-of-order indices by copying or interpolating their neighbours. Phase 1 saw all of it:
    - ashy_ocean_72t46qrk7x: four zh replies with one word starting on the last frame end, ceil(d / 80 ms) * 80 ms,
      5 to 52 ms past d (prereg §16);
    - helpful_hamster_pvjn44n8pj: the same words ending one frame later, beyond unified-score's one-frame end allowance
      (§17);
    - funny_tooth_v3lq4t58n7: up to five trailing characters per reply starting and ending one to three frames past
      the last frame end, up to 228 ms past d (§19).

    So a bound on how far past d an artefact can lie does not hold; the rule is the audio's own bound:
    - a word start past d is moved onto d, floored to the microsecond. d + R then still rounds inside [0, T] at
      unified-score's microsecond resolution, also for a 16 kHz segment whose odd length puts T on a rounding boundary;
    - a word end more than one aligner frame past d is moved onto d. An end at most one frame past d is left as it is:
      unified-score accepts it (ASR_END_TOLERANCE_S), for the same 80 ms reason.

    Only times unified-score rejected are moved, so every transcript that could be scored before is unchanged, byte for
    byte. The text is never changed, and no word is dropped.
    """

    def recognise_on_segment(audio_path: Path) -> list[TimedWord]:
        info = soundfile.info(str(audio_path))
        duration_s = info.frames / info.samplerate
        end_s = info.frames * 10 ** TIME_DECIMALS // info.samplerate / 10 ** TIME_DECIMALS

        def clamp(start_s: float) -> float:
            return end_s if start_s > end_s else start_s

        def clamp_end(end_time_s: float) -> float:
            return end_s if end_time_s > duration_s + ALIGNER_FRAME_S + 1e-9 else end_time_s

        return [(text, clamp(start_s), clamp_end(end_time_s)) for text, start_s, end_time_s in recognise(audio_path)]

    return recognise_on_segment


def check_aligner_frame(model: Any) -> Any:
    """model, if its forced aligner times on ALIGNER_FRAME_S frames; on_segment's one-frame end allowance assumes it."""
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
              "past_end_clamp": {"start_past_s": 0.0, "end_past_s": ALIGNER_FRAME_S}}
    print(json.dumps(transcribe_tree(arguments.tree, qwen_recogniser(model, arguments.language, vad), config), indent=2))


if __name__ == "__main__":
    main()
