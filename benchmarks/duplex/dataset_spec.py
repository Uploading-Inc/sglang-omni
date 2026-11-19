# SPDX-License-Identifier: Apache-2.0
"""Which datasets the duplex benchmark reads.

By default these are Full-Duplex-Bench v1.0 and v1.5. A JSON file named by the environment variable
`FDB_DATASET_SPEC` registers others in the same on-disk formats:

    {"name": "fdb-zh", "language": "zh",
     "v10": {"zh_pause_handling": {"task": "pause_handling", "declared": 239}, ...},
     "v15": {"user_backchannel": 199, "user_interruption": 161}}

v1.0 subsets may have any name and must name one of the four v1.0 tasks. v1.5 categories keep FDB's category names,
because the judges and the semantic rubric read them. A spec changes which samples exist, never a metric definition.
The variable is read once, at import, so every step of one run sees the same registry.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

V10_TASKS = ("pause_handling", "turn_taking", "user_interruption", "backchannel")
FDB_V10 = {
    "synthetic_pause_handling": ("pause_handling", 137),
    "candor_pause_handling": ("pause_handling", 216),
    "candor_turn_taking": ("turn_taking", 119),
    "synthetic_user_interruption": ("user_interruption", 200),
    "icc_backchannel": ("backchannel", 55),
}
# note (wenyao): README counts; the released backchannel archive holds 98.
FDB_V15 = {
    "user_interruption": 200,
    "user_backchannel": 99,
    "talking_to_other": 100,
    "background_speech": 100,
}
SPEC_ENV = "FDB_DATASET_SPEC"


def load_spec(path: str | None) -> dict:
    """The registry: name, language, v10 {subset: (task, declared)}, v15 {category: declared}."""
    if not path:
        return {"name": "full-duplex-bench", "language": "en", "v10": dict(FDB_V10), "v15": dict(FDB_V15)}
    else:
        pass
    raw = json.loads(Path(path).read_text("utf-8"))
    if set(raw) - {"name", "language", "v10", "v15"} or not isinstance(raw.get("name"), str):
        raise ValueError(f"{path}: a spec has name, language, v10 and v15 only")
    else:
        pass
    v10 = {}
    for subset, entry in (raw.get("v10") or {}).items():
        if entry.get("task") not in V10_TASKS or not isinstance(entry.get("declared"), int):
            raise ValueError(f"{path}: v10 subset {subset} needs a task in {V10_TASKS} and an integer declared")
        else:
            v10[subset] = (entry["task"], entry["declared"])
    v15 = {}
    for category, declared in (raw.get("v15") or {}).items():
        if category not in FDB_V15 or not isinstance(declared, int):
            raise ValueError(f"{path}: v15 category {category} must be one of {tuple(FDB_V15)} with an integer count")
        else:
            v15[category] = declared
    if not v10 and not v15:
        raise ValueError(f"{path}: the spec registers no subset")
    else:
        pass
    return {"name": raw["name"], "language": raw.get("language", "en"), "v10": v10, "v15": v15}


SPEC = load_spec(os.environ.get(SPEC_ENV))
IS_FDB = not os.environ.get(SPEC_ENV)
