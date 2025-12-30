#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from ir import Retriever


# -------------------------
# Common utils
# -------------------------
def load_any_json(path: Path) -> Any:
    # supports normal json or json-lines-ish
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
        for k in ["data", "items", "annotations", "queries"]:
            if k in data and isinstance(data[k], list):
                return data[k]
    raise ValueError("JSON format not recognized (expected list or dict with common list fields).")


def safe_l2norm(x: torch.Tensor, dim: int = -1, eps: float = 1e-6) -> torch.Tensor:
    return x / (x.norm(dim=dim, keepdim=True) + eps)


@torch.no_grad()
def recall_at_k(scores: torch.Tensor, gt_index: torch.Tensor, ks=(1, 5, 10, 50)) -> Dict[str, float]:
    """
    scores: [B, N]
    gt_index: [B] (index in gallery)
    """
    ranks = scores.argsort(dim=1, descending=True)
    out = {}
    for k in ks:
        hit = (ranks[:, :k] == gt_index.unsqueeze(1)).any(dim=1).float().mean().item()
        out[f"R@{k}"] = float(hit)
    return out


# -------------------------
# Model (load trained parts)
# -------------------------
class DeltaLexicalGenerator(nn.Module):
    def __init__(self, din: int, V: int, k_delta: int, bottleneck: int = 512):
        super().__init__()
        self.V = V
        self.k_delta = k_delta
        self.f_plus = nn.Sequential(
            nn.Linear(din, bottleneck),
            nn.GELU(),
            nn.Linear(bottleneck, V),
        )
        self.f_minus = nn.Sequential(
            nn.Linear(din, bottleneck),
            nn.GELU(),
            nn.Linear(bottleneck, V),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        up = F.softplus(self.f_plus(x))
        um = F.softplus(self.f_minus(x))
        k = min(self.k_delta, self.V)
        vp, ip = torch.topk(up, k=k, dim=-1)
        vm, im = torch.topk(um, k=k, dim=-1)
        dp = torch.zeros_like(up); dm = torch.zeros_like(um)
        dp.scatter_(dim=1, index=ip, src=vp)
        dm.scatter_(dim=1, index=im, src=vm)
        return dp, dm


class DenseDecoder(nn.Module):
    def __init__(self, V: int, D: int):
        super().__init__()
        self.linear = nn.Linear(V, D, bias=False)

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        z = self.linear(s.float())
        return safe_l2norm(z)


@dataclass
class LoadedLexicalCIR:
    delta_gen: DeltaLexicalGenerator
    decoder: DenseDecoder
    V: int
    D: int
    k_delta: int
    text_topk: int


def load_trained_parts(ckpt_path: str, device: torch.device, override_k_delta: Optional[int] = None,
                       override_text_topk: Optional[int] = None) -> LoadedLexicalCIR:
    ckpt = torch.load(ckpt_path, map_location=device)

    V = int(ckpt.get("V", ckpt["args"].get("V", 27623)))
    D = int(ckpt.get("D", ckpt["args"].get("D", 768)))

    # try to recover k_delta/text_topk from args if present
    args = ckpt.get("args", {})
    k_delta = int(override_k_delta if override_k_delta is not None else args.get("k_delta", 64))
    text_topk = int(override_text_topk if override_text_topk is not None else args.get("text_topk", 768))

    # build modules with same dims
    delta_gen = DeltaLexicalGenerator(din=V, V=V, k_delta=k_delta, bottleneck=int(args.get("bottleneck", 512))).to(device)
    decoder = DenseDecoder(V=V, D=D).to(device)

    # load weights
    if "delta_gen" in ckpt:
        delta_gen.load_state_dict(ckpt["delta_gen"], strict=True)
    else:
        raise ValueError("Checkpoint missing 'delta_gen' state_dict.")

    if "decoder" in ckpt:
        decoder.load_state_dict(ckpt["decoder"], strict=True)
    else:
        raise ValueError("Checkpoint missing 'decoder' state_dict.")

    delta_gen.eval()
    decoder.eval()

    return LoadedLexicalCIR(delta_gen=delta_gen, decoder=decoder, V=V, D=D, k_delta=k_delta, text_topk=text_topk)


# -------------------------
# VDR encoders (no DB needed)
# -------------------------
@torch.no_grad()
def encode_image_dense_vdr(
    vdr: Retriever,
    image_paths: List[str],
    device: torch.device,
    batch_size: int = 64,
    use_fp16: bool = True,
) -> torch.Tensor:
    """
    Extract dense 768-dim embeddings from VDR image backbone (token mean pool).
    No need for precomputed DB.
    """
    enc = vdr.encoder_p
    # dtype = torch.float16 if (use_fp16 and device.type == "cuda") else torch.float32
    dtype = torch.float32

    outs = []
    for i in range(0, len(image_paths), batch_size):
        batch = image_paths[i:i+batch_size]
        imgs = [enc.load_image_file(p) for p in batch]  # each [1,3,224,224]
        x = torch.cat(imgs, dim=0).to(device=device, dtype=dtype)

        # IMPORTANT: enc(images) returns [N, L, D] before proj (as seen in your debug)
        h = enc(x)  # [N,49,768]
        h = h.float()
        z = h.mean(dim=1)  # [N,768]
        z = safe_l2norm(z)
        outs.append(z)

    return torch.cat(outs, dim=0)  # [N,768]


@torch.no_grad()
def encode_image_lexical_vdr(
    vdr: Retriever,
    image_paths: List[str],
    device: torch.device,
    topk: int = 768,
) -> torch.Tensor:
    """
    VDR image lexical embedding: [B,V] (topk sparse)
    """
    # encoder_p.embed accepts (paths) and returns [B,V]
    sr = vdr.encoder_p.embed(image_paths, topk=topk)  # on vdr.device
    return sr.to(device)


@torch.no_grad()
def encode_text_lexical_vdr(
    vdr: Retriever,
    texts: List[str],
    device: torch.device,
    topk: int = 768,
) -> torch.Tensor:
    """
    VDR text lexical embedding: [B,V] (topk sparse)
    """
    ht = vdr.encoder_q.embed(texts, topk=topk)
    return ht.to(device)


# -------------------------
# Dataset adapters
# -------------------------
@dataclass
class CIRExample:
    ref_path: str
    tgt_path: str
    text: str


def _try_get(d: Dict[str, Any], keys: List[str]) -> Optional[Any]:
    for k in keys:
        if k in d:
            return d[k]
    return None


def parse_generic_triplet_json(root: str, ann_json: str) -> List[CIRExample]:
    """
    Robust-ish parser:
    - expects list of dicts
    - tries common key variants for ref/tgt/text

    Supported patterns:
    - LasCo: {"query-image":[id,"train2014/...jpg"], "target-image":[id,"train2014/...jpg"], "query-text":"..."}
    - Others: ("reference"/"ref"/"query-image") etc.
    """
    rootp = Path(root)
    items = normalize_list_format(load_any_json(rootp / ann_json))
    exs: List[CIRExample] = []

    for it in items:
        # ref
        qimg = _try_get(it, ["query-image", "query_image", "reference", "ref", "reference_image", "source_image", "source"])
        # tgt
        timg = _try_get(it, ["target-image", "target_image", "target", "target_image_path", "target_image", "tgt"])
        # text
        txt = _try_get(it, ["query-text", "query_text", "text", "caption", "modification", "relative_caption", "query"])

        if qimg is None or timg is None or txt is None:
            raise KeyError(
                "Annotation keys not found. Need ref/tgt/text.\n"
                f"Available keys example: {list(it.keys())}"
            )

        # LasCo-style [id, "path.jpg"]
        if isinstance(qimg, list) and len(qimg) >= 2:
            ref_rel = qimg[1]
        else:
            ref_rel = qimg

        if isinstance(timg, list) and len(timg) >= 2:
            tgt_rel = timg[1]
        else:
            tgt_rel = timg

        ref_path = str(rootp / ref_rel) if not str(ref_rel).startswith("/") else str(ref_rel)
        tgt_path = str(rootp / tgt_rel) if not str(tgt_rel).startswith("/") else str(tgt_rel)

        exs.append(CIRExample(ref_path=ref_path, tgt_path=tgt_path, text=str(txt)))

    return exs


def build_gallery_from_targets(exs: List[CIRExample]) -> Tuple[List[str], Dict[str, int]]:
    seen = {}
    gallery = []
    for e in exs:
        if e.tgt_path not in seen:
            seen[e.tgt_path] = len(gallery)
            gallery.append(e.tgt_path)
    return gallery, seen


# -------------------------
# Compose + validate
# -------------------------
@torch.no_grad()
def compose_zq(
    parts: LoadedLexicalCIR,
    vdr: Retriever,
    sr: torch.Tensor,      # [B,V]
    texts: List[str],
    device: torch.device,
    mask_mode: str = "lex_support",   # "none" | "lex_support"
) -> torch.Tensor:
    """
    zq = decoder( (sr + Δ+) - (Δ-) )
    mask_mode:
      - none: no mask
      - lex_support: mask based on ht support (recommended; avoids tokenizer mismatch)
    """
    ht = encode_text_lexical_vdr(vdr, texts, device=device, topk=parts.text_topk)  # [B,V]

    if mask_mode == "lex_support":
        m = (ht != 0).float()
        ht = ht * m
    else:
        m = None

    dsp, dsm = parts.delta_gen(ht)  # [B,V]
    if m is not None:
        dsp = dsp * m
        dsm = dsm * m

    sqp = torch.clamp(sr + dsp, min=0.0)
    sqm = torch.clamp(dsm, min=0.0)
    sq = sqp - sqm
    zq = parts.decoder(sq)  # [B,D] normalized
    return zq


@torch.no_grad()
def validate_dataset(
    dataset_name: str,
    exs: List[CIRExample],
    vdr: Retriever,
    parts: LoadedLexicalCIR,
    device: torch.device,
    batch_size: int,
    gallery_cache_npy: Optional[str] = None,
    force_rebuild_gallery: bool = False,
    mask_mode: str = "lex_support",
) -> Dict[str, float]:
    """
    - builds (or loads) gallery dense embeddings
    - iterates over queries, composes zq, scores against gallery, computes recalls
    """
    gallery_paths, tgt2g = build_gallery_from_targets(exs)

    # gallery dense
    if gallery_cache_npy is not None:
        cache_path = Path(gallery_cache_npy)
        if cache_path.exists() and not force_rebuild_gallery:
            gal = torch.from_numpy(np.load(str(cache_path))).to(device)
            gal = safe_l2norm(gal.float())
        else:
            gal = encode_image_dense_vdr(vdr, gallery_paths, device=device, batch_size=batch_size)
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(str(cache_path), gal.detach().cpu().numpy().astype(np.float32))
    else:
        gal = encode_image_dense_vdr(vdr, gallery_paths, device=device, batch_size=batch_size)

    N = gal.size(0)

    r1, r5, r10, r50 = [], [], [], []
    pbar = tqdm(range(0, len(exs), batch_size), desc=f"[VAL {dataset_name}] queries", dynamic_ncols=True)
    for i in pbar:
        batch = exs[i:i+batch_size]
        refs = [b.ref_path for b in batch]
        texts = [b.text for b in batch]
        tgts = [b.tgt_path for b in batch]

        # ref lexical (no DB needed)
        sr = encode_image_lexical_vdr(vdr, refs, device=device, topk=768)  # [B,V]

        # compose
        zq = compose_zq(parts, vdr, sr, texts, device=device, mask_mode=mask_mode)  # [B,D]

        # scores
        scores = zq @ gal.t()  # [B,N]
        gt = torch.tensor([tgt2g[t] for t in tgts], device=device)

        m = recall_at_k(scores, gt, ks=(1, 5, 10, 50))
        r1.append(m["R@1"]); r5.append(m["R@5"]); r10.append(m["R@10"]); r50.append(m["R@50"])

        pbar.set_postfix(R1=f"{np.mean(r1):.4f}", R5=f"{np.mean(r5):.4f}", R10=f"{np.mean(r10):.4f}", N=N)

    out = {
        "R@1": float(np.mean(r1)),
        "R@5": float(np.mean(r5)),
        "R@10": float(np.mean(r10)),
        "R@50": float(np.mean(r50)),
        "gallery_size": int(N),
        "num_queries": int(len(exs)),
    }
    return out


# -------------------------
# Main
# -------------------------
def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--ckpt_model", type=str, required=True, help="trained ckpt_best.pt or ckpt_latest.pt")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--vdr_ckpt", type=str, default="vsearch/vdr-cross-modal")

    ap.add_argument("--dataset", type=str, choices=["lasco", "cirr", "circo"], required=True)
    ap.add_argument("--root", type=str, required=True, help="dataset root directory")
    ap.add_argument("--ann_json", type=str, required=True, help="annotation json filename in root")

    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--mask_mode", type=str, default="lex_support", choices=["lex_support", "none"])

    ap.add_argument("--gallery_cache_npy", type=str, default=None, help="optional .npy cache for gallery dense embeddings")
    ap.add_argument("--force_rebuild_gallery", action="store_true")

    ap.add_argument("--override_k_delta", type=int, default=None)
    ap.add_argument("--override_text_topk", type=int, default=None)

    args = ap.parse_args()

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    device = torch.device(args.device)

    # load VDR
    vdr = Retriever.from_pretrained(args.vdr_ckpt).to(device).eval()

    # load trained parts
    parts = load_trained_parts(
        args.ckpt_model,
        device=device,
        override_k_delta=args.override_k_delta,
        override_text_topk=args.override_text_topk,
    )

    # parse dataset
    exs = parse_generic_triplet_json(args.root, args.ann_json)

    print(f"[DATA] dataset={args.dataset} queries={len(exs)} root={args.root} ann={args.ann_json}")
    print(f"[MODEL] ckpt={args.ckpt_model}")
    print(f"[VDR]   ckpt={args.vdr_ckpt}")
    print(f"[CFG]   V={parts.V} D={parts.D} k_delta={parts.k_delta} text_topk={parts.text_topk} mask_mode={args.mask_mode}")

    out = validate_dataset(
        dataset_name=args.dataset,
        exs=exs,
        vdr=vdr,
        parts=parts,
        device=device,
        batch_size=args.batch_size,
        gallery_cache_npy=args.gallery_cache_npy,
        force_rebuild_gallery=args.force_rebuild_gallery,
        mask_mode=args.mask_mode,
    )

    print("\n=== Validation Result ===")
    for k, v in out.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
