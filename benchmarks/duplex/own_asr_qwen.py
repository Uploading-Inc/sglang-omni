# SPDX-License-Identifier: Apache-2.0
"""Transcribe a v1.0 reference tree with Qwen3-ASR and Qwen3-ForcedAligner, for zh datasets.

The artifacts and the interruption crop are own_asr's. Each aligner token is one word; for zh a token is a
character. Run it from the aligner venv, which holds qwen-asr and its pinned transformers.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from qwen_asr import Qwen3ASRModel

from benchmarks.duplex.own_asr import TimedWord, transcribe_tree

ASR_MODEL_ID = "Qwen/Qwen3-ASR-1.7B"
ALIGNER_MODEL_ID = "Qwen/Qwen3-ForcedAligner-0.6B"
MAX_NEW_TOKENS = 512


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tree", type=Path, required=True)
    parser.add_argument("--language", default="Chinese")
    parser.add_argument("--asr-model", default=ASR_MODEL_ID)
    parser.add_argument("--aligner-model", default=ALIGNER_MODEL_ID)
    parser.add_argument("--device", default="cuda:0")
    arguments = parser.parse_args()
    model = Qwen3ASRModel.from_pretrained(
        arguments.asr_model, dtype=torch.bfloat16, device_map=arguments.device, max_inference_batch_size=1,
        max_new_tokens=MAX_NEW_TOKENS, forced_aligner=arguments.aligner_model,
        forced_aligner_kwargs={"dtype": torch.bfloat16, "device_map": arguments.device},
    )

    def recognise(audio_path: Path) -> list[TimedWord]:
        result = model.transcribe(audio=[str(audio_path)], language=[arguments.language], return_time_stamps=True)[0]
        if result.text.strip():
            return [(token.text, float(token.start_time), float(token.end_time)) for token in result.time_stamps]
        else:
            return []

    config = {"backend": "qwen3-asr", "model_id": arguments.asr_model, "aligner_id": arguments.aligner_model,
              "language": arguments.language, "device_type": arguments.device}
    print(json.dumps(transcribe_tree(arguments.tree, recognise, config), indent=2))


if __name__ == "__main__":
    main()
