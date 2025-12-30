#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from tqdm import tqdm

from ir import Retriever


def load_any_json(path: Path) -> Any:
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        # jsonl fallback
        items = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if not s:
                    continue
                if s.endswith(","):
                    s = s[:-1]
                items.append(json.loads(s))
        return items


def normalize_list_format(data: Any) -> List[Dict[str, Any]]:
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in ["data", "items", "annotations"]:
            if key in data and isinstance(data[key], list):
                return data[key]
    raise ValueError("Unexpected LasCo json format．list/dict(data/items/annotations) を想定しています．")


def collect_unique_paths(root: Path, json_paths: List[Path], which: str) -> List[str]:
    """
    which: "query" / "target" / "both"
    """
    assert which in ["query", "target", "both"]
    paths: List[str] = []
    for jp in json_paths:
        data = normalize_list_format(load_any_json(jp))
        for ex in data:
            if which in ["query", "both"]:
                paths.append(str(root / ex["query-image"][1]))
            if which in ["target", "both"]:
                paths.append(str(root / ex["target-image"][1]))

    seen = set()
    uniq = []
    for p in paths:
        if p not in seen:
            uniq.append(p)
            seen.add(p)
    return uniq


@torch.no_grad()
def encode_dense(vdr: Retriever, image_paths: List[str], batch_size: int) -> np.ndarray:
    enc = vdr.encoder_p
    device = next(vdr.parameters()).device

    outs: List[np.ndarray] = []
    for i in tqdm(range(0, len(image_paths), batch_size), desc="encode_dense"):
        batch = image_paths[i:i + batch_size]
        imgs = [enc.load_image_file(p) for p in batch]       # each [1,3,224,224]
        x = torch.cat(imgs, dim=0).to(device).type(enc.dtype)
        h = enc(x)                                           # [B,L,D]
        z = h.mean(dim=1).float()                            # [B,D]
        z = z / (z.norm(dim=-1, keepdim=True) + 1e-6)
        outs.append(z.cpu().numpy().astype(np.float32))
    return np.concatenate(outs, axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default="/home/uesugi/research/dataset/raw/lasco")
    ap.add_argument("--train_json", type=str, default="lasco_train.json")
    ap.add_argument("--val_json", type=str, default="lasco_val.json")
    ap.add_argument("--which", type=str, default="both", choices=["query", "target", "both"])
    ap.add_argument("--ckpt", type=str, default="vsearch/vdr-cross-modal")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--out_dir", type=str, default="./cache/lasco/dense")
    args = ap.parse_args()

    root = Path(args.root)
    json_paths = [root / args.train_json, root / args.val_json]

    paths = collect_unique_paths(root, json_paths, which=args.which)
    print(f"Collected unique images: {len(paths)}  (which={args.which})")

    vdr = Retriever.from_pretrained(args.ckpt).to(args.device).eval()
    embs = encode_dense(vdr, paths, batch_size=args.batch_size)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths_path = out_dir / "paths.json"
    emb_path = out_dir / "emb.npy"

    with paths_path.open("w", encoding="utf-8") as f:
        json.dump(paths, f, ensure_ascii=False, indent=2)
    np.save(emb_path, embs)

    print(f"Saved: {paths_path}")
    print(f"Saved: {emb_path}  shape={embs.shape} dtype={embs.dtype}")


if __name__ == "__main__":
    main()
