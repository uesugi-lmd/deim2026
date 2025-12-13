# src/deim2026/image_db/io.py
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

def load_annotations(path_json: Path) -> List[Dict[str, Any]]:
    with open(path_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict):
        if "annotations" in data and isinstance(data["annotations"], list):
            return normalize_list_annotations(data["annotations"])
        return [{"id": k, "relpath": v} for k, v in data.items() if isinstance(v, str)]

    if isinstance(data, list):
        return normalize_list_annotations(data)

    raise ValueError("Unsupported JSON format")

def normalize_list_annotations(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for ann in items:
        if not isinstance(ann, dict):
            continue

        entry = {}
        for k in ["id", "image_id", "img_id"]:
            if k in ann:
                entry["id"] = str(ann[k])
        for k in ["relpath", "path", "image_path", "img_path"]:
            if k in ann:
                entry["relpath"] = str(ann[k])
        out.append(entry)
    return out

def resolve_image_path(ann: Dict[str, Any], root_dir: Path) -> Tuple[str, Path]:
    _id = ann.get("id")
    rel = ann.get("relpath")

    if rel is None:
        raise FileNotFoundError(f"No relpath field: keys={list(ann.keys())}")

    if rel.startswith("./"):
        rel = rel[2:]

    abs_path = (root_dir / rel).resolve()

    if _id is None:
        _id = abs_path.stem

    return _id, abs_path
