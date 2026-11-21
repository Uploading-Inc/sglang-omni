# SPDX-License-Identifier: Apache-2.0
"""Transcribe a v1.0 reference tree with Qwen3-ASR and Qwen3-ForcedAligner, for zh datasets.

The artifacts and the interruption crop are own_asr's. Each aligner token is one word; for zh a token is a
character. A segment with no Silero speech region gets no word and is never given to Qwen3-ASR, which writes "嗯。"
for silence and for noise (asr_speech_gate). Run it from the aligner venv, which holds qwen-asr, its pinned
transformers and silero-vad.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from benchmarks.duplex.asr_speech_gate import gate_receipt, gated, load_gate_vad
from benchmarks.duplex.own_asr import Recogniser, TimedWord, transcribe_tree

ASR_MODEL_ID = "Qwen/Qwen3-ASR-1.7B"
ALIGNER_MODEL_ID = "Qwen/Qwen3-ForcedAligner-0.6B"
MAX_NEW_TOKENS = 512


def load_model(asr_model: str = ASR_MODEL_ID, aligner_model: str = ALIGNER_MODEL_ID, device: str = "cuda:0") -> Any:
    from qwen_asr import Qwen3ASRModel

    return Qwen3ASRModel.from_pretrained(
        asr_model, dtype=torch.bfloat16, device_map=device, max_inference_batch_size=1,
        max_new_tokens=MAX_NEW_TOKENS, forced_aligner=aligner_model,
        forced_aligner_kwargs={"dtype": torch.bfloat16, "device_map": device},
    )


def qwen_recogniser(model: Any, language: str, vad: Any) -> Recogniser:
    """The recogniser own_asr scores with: aligner tokens of Qwen3-ASR's text, behind the speech gate."""

    def recognise(audio_path: Path) -> list[TimedWord]:
        result = model.transcribe(audio=[str(audio_path)], language=[language], return_time_stamps=True)[0]
        if result.text.strip():
            return [(token.text, float(token.start_time), float(token.end_time)) for token in result.time_stamps]
        else:
            return []

    return gated(recognise, vad)


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
              "language": arguments.language, "device_type": arguments.device, "speech_gate": gate_receipt()}
    print(json.dumps(transcribe_tree(arguments.tree, qwen_recogniser(model, arguments.language, vad), config), indent=2))


if __name__ == "__main__":
    main()
