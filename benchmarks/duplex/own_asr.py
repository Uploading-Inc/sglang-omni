# SPDX-License-Identifier: Apache-2.0
"""Transcribe a v1.0 reference tree with code of this repository, for datasets registered by FDB_DATASET_SPEC.

It writes what unified-score reads, without the Full-Duplex-Bench source:
- per eligible sample, output.json = {"text": the words joined by spaces, "chunks": [{"text", "timestamp": [s, e]}]};
- asr.json = {"created_utc", "config", "subsets": {subset: {"task", "transcripts": {name: sha256}}}}.

User-interruption audio is cropped at the end R of the interruption (interrupt.json[0]["timestamp"][1]) before
recognition, and R is added back to every word, as reference-asr does. This module's recogniser is the harness's
pinned Parakeet (en). The Qwen3-ASR recogniser (zh) lives in own_asr_qwen, which runs in the aligner venv.

Only the zh recogniser sits behind asr_speech_gate. Parakeet returns no word on silence and noise (Phase 1 smoke), this
recogniser is the harness's reference-asr one, and a gate can only remove words, so on en it could only add misses.
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from typing import Protocol

import soundfile
from pydantic import JsonValue

from benchmarks.duplex.reference_asr import load_nemo_model
from benchmarks.duplex.reference_core import ASR_MODEL_ID, atomic_write_json, utc_now
from benchmarks.duplex.run_artifacts import file_sha256
from benchmarks.duplex.v10_dataset import SUBSET_TASKS, SUBSETS

TimedWord = tuple[str, float, float]


class Recogniser(Protocol):
    def __call__(self, audio_path: Path) -> list[TimedWord]: ...


def transcript(words: list[TimedWord], offset_s: float) -> dict[str, JsonValue]:
    chunks = [
        {"text": text, "timestamp": [round(start_s + offset_s, 6), round(end_s + offset_s, 6)]}
        for text, start_s, end_s in words
        if text.strip()
    ]
    return {"text": " ".join(chunk["text"] for chunk in chunks).strip(), "chunks": chunks}


def transcribe_sample(sample_dir: Path, task: str, recognise: Recogniser) -> bytes:
    audio_path = sample_dir / "output.wav"
    if task == "user_interruption":
        end_s = float(json.loads((sample_dir / "interrupt.json").read_text("utf-8"))[0]["timestamp"][1])
        samples, sample_rate = soundfile.read(str(audio_path), dtype="int16")
        with tempfile.TemporaryDirectory() as stage_dir:
            cropped_path = Path(stage_dir) / "cropped.wav"
            soundfile.write(str(cropped_path), samples[round(end_s * sample_rate):], sample_rate, subtype="PCM_16")
            document = transcript(recognise(cropped_path), end_s)
    else:
        document = transcript(recognise(audio_path), 0.0)
    return json.dumps(document, ensure_ascii=False).encode("utf-8")


def transcribe_tree(tree: Path, recognise: Recogniser, config: dict[str, JsonValue]) -> dict[str, int]:
    receipt_path = tree / "asr.json"
    if receipt_path.exists():
        raise SystemExit(f"{receipt_path} exists; transcribe a new export instead")
    else:
        pass
    rows = json.loads((tree / "manifest.json").read_text("utf-8"))["samples"]
    subsets = {}
    for subset in SUBSETS:
        names = sorted(row["sample_id"].split("/")[1] for row in rows if row["subset"] == subset and row["eligible"])
        if names:
            transcripts = {}
            for name in names:
                sample_dir = tree / subset / name
                (sample_dir / "output.json").write_bytes(transcribe_sample(sample_dir, SUBSET_TASKS[subset], recognise))
                transcripts[name] = file_sha256(sample_dir / "output.json")
            task = "user_interruption" if SUBSET_TASKS[subset] == "user_interruption" else "default"
            subsets[subset] = {"task": task, "transcripts": transcripts}
        else:
            pass
    atomic_write_json(receipt_path, {"created_utc": utc_now(), "config": config, "subsets": subsets})
    return {subset: len(entry["transcripts"]) for subset, entry in subsets.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tree", type=Path, required=True)
    parser.add_argument("--nemo", type=Path, required=True)
    parser.add_argument("--nemo-sha256", required=True)
    parser.add_argument("--device", default="cuda")
    arguments = parser.parse_args()
    checkpoint_sha256 = file_sha256(arguments.nemo)
    if checkpoint_sha256 != arguments.nemo_sha256:
        raise SystemExit(f"{arguments.nemo} sha256 {checkpoint_sha256} != --nemo-sha256")
    else:
        pass
    model = load_nemo_model(arguments.nemo, arguments.device)

    def recognise(audio_path: Path) -> list[TimedWord]:
        hypothesis = model.transcribe([str(audio_path)], timestamps=True)[0]
        return [(word["word"], float(word["start"]), float(word["end"])) for word in hypothesis.timestamp["word"]]

    config = {"backend": "parakeet", "model_id": ASR_MODEL_ID, "checkpoint_sha256": checkpoint_sha256,
              "device_type": arguments.device}
    print(json.dumps(transcribe_tree(arguments.tree, recognise, config), indent=2))


if __name__ == "__main__":
    main()
