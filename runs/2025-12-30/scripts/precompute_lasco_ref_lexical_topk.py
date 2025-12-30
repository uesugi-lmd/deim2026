#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
from tqdm import tqdm

from ir import Retriever


def load_any_json(path: Path) -> Any:
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
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


def normalize_list_format(data: Any) -> List[Dict]:
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in ["data", "items", "annotations"]:
            if key in data and isinstance(data[key], list):
                return data[key]
    raise ValueError("Unexpected LasCo json format．")


def collect_unique_ref_paths(root: Path, json_paths: List[Path]) -> List[str]:
    paths: List[str] = []
    for jp in json_paths:
        data = normalize_list_format(load_any_json(jp))
        for ex in data:
            paths.append(str(root / ex["query-image"][1]))

    seen = set()
    uniq = []
    for p in paths:
        if p not in seen:
            uniq.append(p)
            seen.add(p)
    return uniq


@torch.no_grad()
def encode_ref_lexical_topk(vdr: Retriever, ref_paths: List[str], topk: int, batch_size: int):
    enc = vdr.encoder_p
    device = next(vdr.parameters()).device
    W = enc.proj                                             # [V,D]
    V = W.shape[0]

    all_idx = []
    all_val = []

    for i in tqdm(range(0, len(ref_paths), batch_size), desc="encode_ref_lexical_topk"):
        batch = ref_paths[i:i + batch_size]
        imgs = [enc.load_image_file(p) for p in batch]
        x = torch.cat(imgs, dim=0).to(device).type(enc.dtype) # [B,3,224,224]
        h = enc(x)                                             # [B,L,D]
        lv = (h @ W.t()).float()                               # [B,L,V]
        sr = lv.max(dim=1)[0]                                  # [B,V]
        sr = torch.clamp(sr, min=0.0)

        k = min(topk, V)
        vals, idx = torch.topk(sr, k=k, dim=-1)
        all_idx.append(idx.cpu().numpy().astype(np.int32))
        all_val.append(vals.cpu().numpy().astype(np.float16))

    idx_np = np.concatenate(all_idx, axis=0)
    val_np = np.concatenate(all_val, axis=0)
    return idx_np, val_np, V


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default="/home/uesugi/research/dataset/raw/lasco")
    ap.add_argument("--train_json", type=str, default="lasco_train.json")
    ap.add_argument("--val_json", type=str, default="lasco_val.json")
    ap.add_argument("--ckpt", type=str, default="vsearch/vdr-cross-modal")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--topk", type=int, default=768)
    ap.add_argument("--out_dir", type=str, default="./cache/lasco/ref_lexical_topk")
    args = ap.parse_args()

    root = Path(args.root)
    json_paths = [root / args.train_json, root / args.val_json]

    ref_paths = collect_unique_ref_paths(root, json_paths)
    print(f"Collected unique ref (query-image): {len(ref_paths)}")

    vdr = Retriever.from_pretrained(args.ckpt).to(args.device).eval()
    idx_np, val_np, V = encode_ref_lexical_topk(vdr, ref_paths, topk=args.topk, batch_size=args.batch_size)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    (out_dir / "meta.json").write_text(
        json.dumps({"V": int(V), "topk": int(args.topk)}, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )
    (out_dir / "paths.json").write_text(json.dumps(ref_paths, ensure_ascii=False, indent=2), encoding="utf-8")
    np.save(out_dir / "idx.npy", idx_np)
    np.save(out_dir / "val.npy", val_np)

    print(f"Saved: {out_dir / 'meta.json'}")
    print(f"Saved: {out_dir / 'paths.json'}")
    print(f"Saved: {out_dir / 'idx.npy'}  shape={idx_np.shape} dtype={idx_np.dtype}")
    print(f"Saved: {out_dir / 'val.npy'}  shape={val_np.shape} dtype={val_np.dtype}")


if __name__ == "__main__":
    main()
