#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

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


def normalize_list_format(data: Any) -> List[Dict[str, Any]]:
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in ["data", "items", "annotations"]:
            if key in data and isinstance(data[key], list):
                return data[key]
    raise ValueError("Unexpected LasCo json format．")


def safe_l2norm(x: torch.Tensor, dim: int = -1, eps: float = 1e-6) -> torch.Tensor:
    return x / (x.norm(dim=dim, keepdim=True) + eps)


class DenseCache:
    def __init__(self, paths_json: str, emb_npy: str):
        self.paths: List[str] = json.loads(Path(paths_json).read_text(encoding="utf-8"))
        self.emb: np.ndarray = np.load(emb_npy)
        assert self.emb.shape[0] == len(self.paths)
        self.path2i = {p: i for i, p in enumerate(self.paths)}
        self.D = int(self.emb.shape[1])

    def get(self, paths: List[str]) -> torch.Tensor:
        idx = [self.path2i[p] for p in paths]
        return torch.from_numpy(self.emb[idx])


class LexicalTopKCache:
    def __init__(self, meta_json: str, paths_json: str, idx_npy: str, val_npy: str):
        meta = json.loads(Path(meta_json).read_text(encoding="utf-8"))
        self.V = int(meta["V"])
        self.topk = int(meta["topk"])
        self.paths: List[str] = json.loads(Path(paths_json).read_text(encoding="utf-8"))
        self.idx: np.ndarray = np.load(idx_npy)
        self.val: np.ndarray = np.load(val_npy)
        assert self.idx.shape == self.val.shape
        assert self.idx.shape[0] == len(self.paths)
        self.path2i = {p: i for i, p in enumerate(self.paths)}

    def densify(self, paths: List[str], device: torch.device, dtype=torch.float32) -> torch.Tensor:
        B = len(paths)
        sr = torch.zeros((B, self.V), device=device, dtype=dtype)
        rows = [self.path2i[p] for p in paths]
        idx = torch.from_numpy(self.idx[rows]).to(device).long()
        val = torch.from_numpy(self.val[rows].astype(np.float32)).to(device).to(dtype)
        sr.scatter_(dim=1, index=idx, src=val)
        return sr


class LasCoDataset(Dataset):
    def __init__(self, root: str, json_file: str):
        self.root = Path(root)
        self.items = normalize_list_format(load_any_json(self.root / json_file))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int) -> Dict[str, Any]:
        ex = self.items[i]
        return {
            "ref": str(self.root / ex["query-image"][1]),
            "tgt": str(self.root / ex["target-image"][1]),
            "text": ex["query-text"],
        }


def collate(batch: List[Dict[str, Any]]) -> Tuple[List[str], List[str], List[str]]:
    refs = [b["ref"] for b in batch]
    texts = [b["text"] for b in batch]
    tgts = [b["tgt"] for b in batch]
    return refs, texts, tgts


@torch.no_grad()
def r_at_k_full_gallery(zq: torch.Tensor, gallery: torch.Tensor, pos_index: torch.Tensor, ks=(1, 5, 10)) -> Dict[str, float]:
    zq = safe_l2norm(zq.float())
    gallery = safe_l2norm(gallery.float())
    scores = zq @ gallery.t()
    ranks = scores.argsort(dim=1, descending=True)
    out = {}
    for k in ks:
        hit = (ranks[:, :k] == pos_index.unsqueeze(1)).any(dim=1).float().mean().item()
        out[f"R@{k}"] = float(hit)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lasco_root", type=str, default="/home/uesugi/research/dataset/raw/lasco")
    ap.add_argument("--val_json", type=str, default="lasco_val.json")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--eval_batches", type=int, default=50)

    ap.add_argument("--dense_paths_json", type=str, required=True)
    ap.add_argument("--dense_emb_npy", type=str, required=True)

    ap.add_argument("--reflex_meta_json", type=str, required=True)
    ap.add_argument("--reflex_paths_json", type=str, required=True)
    ap.add_argument("--reflex_idx_npy", type=str, required=True)
    ap.add_argument("--reflex_val_npy", type=str, required=True)

    ap.add_argument("--ckpt", type=str, default="vsearch/vdr-cross-modal")
    ap.add_argument("--load_model_ckpt", type=str, default="", help="trainスクリプトの ckpt_best.pt を指定すると compose を評価できる")

    args = ap.parse_args()
    device = torch.device(args.device)

    dense_cache = DenseCache(args.dense_paths_json, args.dense_emb_npy)
    ref_cache = LexicalTopKCache(args.reflex_meta_json, args.reflex_paths_json, args.reflex_idx_npy, args.reflex_val_npy)

    val_ds = LasCoDataset(args.lasco_root, args.val_json)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate)

    # val gallery = unique val targets
    val_data = normalize_list_format(load_any_json(Path(args.lasco_root) / args.val_json))
    val_tgt_paths_all = [str(Path(args.lasco_root) / ex["target-image"][1]) for ex in val_data]
    seen = set()
    val_gallery_paths = []
    for p in val_tgt_paths_all:
        if p not in seen:
            val_gallery_paths.append(p)
            seen.add(p)
    val_tgt2g = {p: i for i, p in enumerate(val_gallery_paths)}
    gallery_np = dense_cache.emb[[dense_cache.path2i[p] for p in val_gallery_paths]]
    gallery = safe_l2norm(torch.from_numpy(gallery_np).to(device))

    print(f"[VAL] items={len(val_ds)} gallery={len(val_gallery_paths)}")

    # --- baselines
    oracle_scores = []
    refonly_scores = []
    model_scores = []

    # optional: load your trained model parts (delta_gen + decoder) for compose baseline
    vdr = Retriever.from_pretrained(args.ckpt).to(device).eval()

    delta_gen = None
    decoder = None
    V_img, D_img = vdr.encoder_p.proj.shape

    if args.load_model_ckpt:
        ck = torch.load(args.load_model_ckpt, map_location="cpu")
        # NOTE: must match train script module names
        # Build minimal modules inline
        # Decoder
        decoder = torch.nn.Linear(ref_cache.V, dense_cache.D, bias=False).to(device)
        decoder.weight.data.copy_(vdr.encoder_p.proj.t().float())
        decoder.load_state_dict(ck["decoder"], strict=True)
        decoder.eval()

        # DeltaGen (bottleneck は ckpt args に入っているはずだが，無い場合 512 を仮定)
        bneck = int(ck.get("args", {}).get("bottleneck", 512))
        k_delta = int(ck.get("args", {}).get("k_delta", 64))
        class _DG(torch.nn.Module):
            def __init__(self, V, k, b):
                super().__init__()
                self.V = V
                self.k = k
                self.f_plus = torch.nn.Sequential(torch.nn.Linear(V, b), torch.nn.GELU(), torch.nn.Linear(b, V))
                self.f_minus = torch.nn.Sequential(torch.nn.Linear(V, b), torch.nn.GELU(), torch.nn.Linear(b, V))
            def forward(self, x):
                up = torch.nn.functional.softplus(self.f_plus(x))
                um = torch.nn.functional.softplus(self.f_minus(x))
                k = min(self.k, self.V)
                vp, ip = torch.topk(up, k=k, dim=-1)
                vm, im = torch.topk(um, k=k, dim=-1)
                dp = torch.zeros_like(up); dm = torch.zeros_like(um)
                dp.scatter_(1, ip, vp); dm.scatter_(1, im, vm)
                return dp, dm
        delta_gen = _DG(ref_cache.V, k_delta, bneck).to(device)
        delta_gen.load_state_dict(ck["delta_gen"], strict=True)
        delta_gen.eval()

        print(f"[LOAD] model_ckpt={args.load_model_ckpt}")

    for bi, (refs, texts, tgts) in enumerate(val_dl):
        if bi >= args.eval_batches:
            break

        zr = safe_l2norm(dense_cache.get(refs).to(device).float())
        zt = safe_l2norm(dense_cache.get(tgts).to(device).float())
        pos = torch.tensor([val_tgt2g[p] for p in tgts], device=device)

        # Oracle: zq = zt
        m_or = r_at_k_full_gallery(zt, gallery, pos)
        oracle_scores.append(m_or)

        # Ref-only: zq = zr
        m_rf = r_at_k_full_gallery(zr, gallery, pos)
        refonly_scores.append(m_rf)

        # Your model compose (if ckpt provided)
        if delta_gen is not None and decoder is not None:
            sr = ref_cache.densify(refs, device=device)
            ht = vdr.encoder_q.embed(texts, training=False, topk=ref_cache.V).to(device)
            dsp, dsm = delta_gen(ht)
            sq = torch.clamp(sr + dsp, min=0.0) - torch.clamp(dsm, min=0.0)
            zq = safe_l2norm(decoder(sq).float())
            m_md = r_at_k_full_gallery(zq, gallery, pos)
            model_scores.append(m_md)

    def avg(scores):
        keys = scores[0].keys()
        return {k: float(sum(s[k] for s in scores) / len(scores)) for k in keys}

    print("[Oracle zq=zt]", avg(oracle_scores))
    print("[Ref-only zq=zr]", avg(refonly_scores))
    if model_scores:
        print("[Model compose]", avg(model_scores))
    else:
        print("[Model compose] skipped (set --load_model_ckpt ckpt_best.pt)")

if __name__ == "__main__":
    main()
