from __future__ import annotations

import json
import pickle
from csv import DictWriter
from pathlib import Path
from typing import Any


def load_json_if_exists(path: Path):
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def save_json(path: Path, obj: Any):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def save_pickle(path: Path, obj: Any):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def load_pickle_if_exists(path: Path):
    if path.exists():
        with open(path, "rb") as f:
            return pickle.load(f)
    return None


def save_manifest(path: Path, manifest: dict):
    save_json(path, manifest)


def append_rows_to_csv(path: Path, rows: list[dict]):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    write_header = not path.exists()
    with open(path, "a", encoding="utf-8", newline="") as f:
        writer = DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)
