#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import datetime
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from ir import Retriever


# ============================================================
# Utils
# ============================================================
def l2norm(x: torch.Tensor) -> torch.Tensor:
    return x / (x.norm(dim=-1, keepdim=True) + 1e-6)


def json_load(p: Path) -> Any:
    return json.loads(p.read_text(encoding="utf-8"))


def json_dump(obj: Any, p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, ensure_ascii=False), encoding="utf-8")


# ============================================================
# CIRR rc2 loader (val)
# ============================================================
@dataclass
class CIRRQuery:
    pairid: int
    ref_name: str
    tgt_hard: str
    tgt_soft: Dict[str, float]
    caption: str
    members: List[str]  # subset candidates (img_set members)


def load_cirr_rc2_val(cirr_root: Path) -> List[CIRRQuery]:
    """
    expects:
      {cirr_root}/cirr/captions/cap.rc2.val.json
      {cirr_root}/images/dev/*.jpg (or png)  (CIRR official structure)
    """
    ann_path = cirr_root / "cirr" / "captions" / "cap.rc2.val.json"
    anns = json_load(ann_path)
    out: List[CIRRQuery] = []
    for it in anns:
        out.append(
            CIRRQuery(
                pairid=int(it["pairid"]),
                ref_name=str(it["reference"]),
                tgt_hard=str(it["target_hard"]),
                tgt_soft={k: float(v) for k, v in it["target_soft"].items()},
                caption=str(it["caption"]),
                members=list(it["img_set"]["members"]),
            )
        )
    return out


def build_name_to_path_cirr_images(cirr_root: Path) -> Dict[str, str]:
    """
    CIRR image naming:
      'dev-903-0-img0' -> file under {cirr_root}/images/dev/dev-903-0-img0.jpg (typically)
    We'll search under images/{dev,train,test1} for robustness.
    """
    img_root = cirr_root / "images"
    splits = ["dev", "train", "test1"]
    name2path: Dict[str, str] = {}

    for sp in splits:
        sp_dir = img_root / sp
        if not sp_dir.exists():
            continue
        for p in sp_dir.rglob("*"):
            if not p.is_file():
                continue
            if p.suffix.lower() not in [".jpg", ".jpeg", ".png", ".webp"]:
                continue
            name = p.stem  # dev-xxx-...
            # keep first occurrence
            if name not in name2path:
                name2path[name] = str(p)

    if len(name2path) == 0:
        raise RuntimeError(f"[CIRR] No images found under {img_root}.")
    return name2path


# ============================================================
# Model pieces (same checkpoint convention as your training script)
# ============================================================
class DeltaLexicalGenerator(nn.Module):
    def __init__(self, din: int, V: int, k_delta: int, bottleneck: int = 512):
        super().__init__()
        self.V = V
        self.k_delta = k_delta
        self.f_plus = nn.Sequential(nn.Linear(din, bottleneck), nn.GELU(), nn.Linear(bottleneck, V))
        self.f_minus = nn.Sequential(nn.Linear(din, bottleneck), nn.GELU(), nn.Linear(bottleneck, V))

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
        return l2norm(z)


@dataclass
class LoadedParts:
    delta_gen: DeltaLexicalGenerator
    decoder: DenseDecoder
    V: int
    D: int
    k_delta: int
    text_topk: int


def load_parts_from_ckpt(ckpt_path: Path, device: torch.device,
                         override_k_delta: Optional[int] = None,
                         override_text_topk: Optional[int] = None) -> LoadedParts:
    ckpt = torch.load(str(ckpt_path), map_location=device)
    args = ckpt.get("args", {})

    V = int(ckpt.get("V", args.get("V", 27623)))
    D = int(ckpt.get("D", args.get("D", 768)))
    k_delta = int(override_k_delta if override_k_delta is not None else args.get("k_delta", 64))
    text_topk = int(override_text_topk if override_text_topk is not None else args.get("text_topk", 768))
    bottleneck = int(args.get("bottleneck", 512))

    delta_gen = DeltaLexicalGenerator(din=V, V=V, k_delta=k_delta, bottleneck=bottleneck).to(device)
    decoder = DenseDecoder(V=V, D=D).to(device)

    # keys must match your training save
    delta_gen.load_state_dict(ckpt["delta_gen"], strict=True)
    decoder.load_state_dict(ckpt["decoder"], strict=True)

    delta_gen.eval(); decoder.eval()
    return LoadedParts(delta_gen=delta_gen, decoder=decoder, V=V, D=D, k_delta=k_delta, text_topk=text_topk)


# ============================================================
# VDR encoders (dtype-safe)
# ============================================================
@torch.no_grad()
def encode_image_dense_vdr(vdr: Retriever, image_paths: List[str], device: torch.device, batch_size: int) -> torch.Tensor:
    """
    Dense embedding for image: mean over tokens from VDR image encoder forward output.
    Force float32 to avoid HalfTensor/FloatTensor mismatch.
    """
    enc = vdr.encoder_p
    enc = enc.float()

    outs: List[torch.Tensor] = []
    for i in range(0, len(image_paths), batch_size):
        batch = image_paths[i:i+batch_size]
        imgs = [enc.load_image_file(p) for p in batch]
        x = torch.cat(imgs, dim=0).to(device=device, dtype=torch.float32)
        h = enc(x)  # [B,49,768]
        z = l2norm(h.mean(dim=1).float())  # [B,768]
        outs.append(z.cpu())
    return torch.cat(outs, dim=0).to(device)


@torch.no_grad()
def encode_ref_lexical(vdr: Retriever, ref_paths: List[str], device: torch.device, topk: int) -> torch.Tensor:
    vdr.encoder_p = vdr.encoder_p.float()
    sr = vdr.encoder_p.embed(ref_paths, topk=topk)  # [B,V]
    return sr.to(device)


@torch.no_grad()
def encode_text_lexical(vdr: Retriever, texts: List[str], device: torch.device, topk: int) -> torch.Tensor:
    ht = vdr.encoder_q.embed(texts, topk=topk)      # [B,V]
    return ht.to(device)


@torch.no_grad()
def compose_query(parts: LoadedParts, vdr: Retriever, sr: torch.Tensor, texts: List[str],
                  device: torch.device, mask_mode: str) -> torch.Tensor:
    ht = encode_text_lexical(vdr, texts, device=device, topk=parts.text_topk)  # [B,V]

    m = None
    if mask_mode == "lex_support":
        m = (ht != 0).float()
        ht = ht * m

    dsp, dsm = parts.delta_gen(ht)

    if m is not None:
        dsp = dsp * m
        dsm = dsm * m

    sq = torch.clamp(sr + dsp, min=0.0) - torch.clamp(dsm, min=0.0)
    zq = parts.decoder(sq)  # [B,768]
    return zq


# ============================================================
# CIRR rc2 evaluation
# ============================================================
def recall_at_k(hit: bool) -> float:
    return 1.0 if hit else 0.0


@torch.no_grad()
def validate_cirr_rc2(
    cirr_root: Path,
    ckpt_model: Path,
    vdr_ckpt: str,
    device: torch.device,
    batch_size: int,
    pred_k: int,
    mask_mode: str,
    gallery_cache_npy: Optional[Path],
    force_rebuild_gallery: bool,
    save_recalls_json: Optional[Path],
    save_recalls_subset_json: Optional[Path],
    override_k_delta: Optional[int],
    override_text_topk: Optional[int],
) -> Dict[str, Any]:
    start = time.time()

    # load annotations + images mapping
    queries = load_cirr_rc2_val(cirr_root)
    name2path = build_name_to_path_cirr_images(cirr_root)

    # build gallery ids = all unique images appearing in dev split in annotations:
    # ValCirr uses data_loader.dataset.id2embpth (precomputed).
    # Here we simply take all names present in name2path, but you can restrict to dev-* if desired.
    # To be safe, we use all mapped images; evaluation will work.
    gallery_names = sorted(list(name2path.keys()))
    gallery_paths = [name2path[n] for n in gallery_names]
    name2gidx = {n: i for i, n in enumerate(gallery_names)}

    print(f"[DATA] cirr_root={cirr_root}")
    print(f"[VAL]  queries={len(queries)} gallery={len(gallery_names)} pred_k={pred_k}")
    print(f"[MODEL] ckpt={ckpt_model}")
    print(f"[VDR]   ckpt={vdr_ckpt}")
    print(f"[CFG]   mask={mask_mode}")

    # load model pieces
    vdr = Retriever.from_pretrained(vdr_ckpt).to(device).eval()
    vdr.encoder_p = vdr.encoder_p.float()
    parts = load_parts_from_ckpt(
        ckpt_model, device=device,
        override_k_delta=override_k_delta,
        override_text_topk=override_text_topk
    )
    print(f"[CFG]   V={parts.V} D={parts.D} k_delta={parts.k_delta} text_topk={parts.text_topk}")

    # encode gallery (cache)
    if gallery_cache_npy is not None and gallery_cache_npy.exists() and not force_rebuild_gallery:
        print(f"[CACHE] load gallery dense from {gallery_cache_npy}")
        gal = torch.from_numpy(np.load(str(gallery_cache_npy))).to(device).float()
        gal = l2norm(gal)
    else:
        print("[ENC] encoding gallery with VDR (dense)... (this can be slow)")
        gal = encode_image_dense_vdr(vdr, gallery_paths, device=device, batch_size=batch_size)
        if gallery_cache_npy is not None:
            gallery_cache_npy.parent.mkdir(parents=True, exist_ok=True)
            np.save(str(gallery_cache_npy), gal.detach().cpu().numpy().astype(np.float32))
            print(f"[CACHE] saved gallery dense to {gallery_cache_npy}")

    # build helper dicts for evaluation like ValCirr
    pairid2ref = {q.pairid: q.ref_name for q in queries}
    pairid2members = {q.pairid: q.members for q in queries}
    pairid2target_hard = {q.pairid: q.tgt_hard for q in queries}
    pairid2target_soft = {q.pairid: q.tgt_soft for q in queries}

    # compute query embeddings + rankings
    recalls: Dict[str, Any] = {"version": "rc2", "metric": "recall"}
    recalls_subset: Dict[str, Any] = {"version": "rc2", "metric": "recall_subset"}

    # batched compute
    for i in tqdm(range(0, len(queries), batch_size), desc="[VAL queries]", dynamic_ncols=True):
        batch = queries[i:i+batch_size]
        ref_names = [q.ref_name for q in batch]
        captions = [q.caption for q in batch]
        pairids = [q.pairid for q in batch]

        # map ref -> path
        ref_paths = [name2path[n] for n in ref_names]

        # sr lexical (ref image)
        sr = encode_ref_lexical(vdr, ref_paths, device=device, topk=768)
        zq = compose_query(parts, vdr, sr, captions, device=device, mask_mode=mask_mode)  # [B,768]

        sims = zq @ gal.t()  # [B,G]

        # exclude reference image itself (official behavior)
        for bi, pid in enumerate(pairids):
            ref_name = pairid2ref[pid]
            j = name2gidx.get(ref_name, None)
            if j is not None:
                sims[bi, j] = -100.0

        # rank
        topk_idx = sims.topk(k=pred_k, dim=1, largest=True).indices  # [B,pred_k]

        for bi, pid in enumerate(pairids):
            idxs = topk_idx[bi].tolist()
            pred_names = [gallery_names[j] for j in idxs]

            recalls[str(pid)] = pred_names[:50]

            members = set(pairid2members[pid])
            subset = [n for n in pred_names if n in members][:3]
            recalls_subset[str(pid)] = subset

    # compute Recall@K (hard)
    r_at = {}
    for k in [1, 5, 10, 50]:
        hit = 0.0
        for pid_str, pred_list in recalls.items():
            if pid_str in ["version", "metric"]:
                continue
            pid = int(pid_str)
            tgt = pairid2target_hard[pid]
            hit += recall_at_k(tgt in pred_list[:k])
        r = hit / (len(queries) + 1e-9)
        r_at[k] = r

    # compute Recall_subset@K (soft)
    rsub_at = {}
    for k in [1, 2, 3]:
        acc = 0.0
        for pid_str, pred_list in recalls_subset.items():
            if pid_str in ["version", "metric"]:
                continue
            pid = int(pid_str)
            tgt_soft = pairid2target_soft[pid]  # dict name->score
            best = 0.0
            for name, score in tgt_soft.items():
                if name in pred_list[:k]:
                    best = max(best, float(score))
            acc += best
        rsub = acc / (len(queries) + 1e-9)
        rsub_at[k] = rsub

    # save
    if save_recalls_json is not None:
        json_dump(recalls, save_recalls_json)
        print(f"[SAVE] recalls -> {save_recalls_json}")
    if save_recalls_subset_json is not None:
        json_dump(recalls_subset, save_recalls_subset_json)
        print(f"[SAVE] recalls_subset -> {save_recalls_subset_json}")

    total = time.time() - start
    print("Evaluation time {}".format(str(datetime.timedelta(seconds=int(total)))))

    return {
        "Recall@K": r_at,
        "Recall_subset@K": rsub_at,
        "num_queries": len(queries),
        "gallery": len(gallery_names),
        "pred_k": pred_k,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cirr_root", type=str, required=True)
    ap.add_argument("--ckpt_model", type=str, required=True)
    ap.add_argument("--vdr_ckpt", type=str, default="vsearch/vdr-cross-modal")
    ap.add_argument("--device", type=str, default="cuda")

    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--pred_k", type=int, default=50, help="ranking length, >=50 recommended")
    ap.add_argument("--mask_mode", type=str, default="lex_support", choices=["lex_support", "none"])

    ap.add_argument("--gallery_cache_npy", type=str, default=None)
    ap.add_argument("--force_rebuild_gallery", action="store_true")

    ap.add_argument("--save_recalls_json", type=str, default=None)
    ap.add_argument("--save_recalls_subset_json", type=str, default=None)

    ap.add_argument("--override_k_delta", type=int, default=None)
    ap.add_argument("--override_text_topk", type=int, default=None)

    args = ap.parse_args()
    device = torch.device(args.device)

    out = validate_cirr_rc2(
        cirr_root=Path(args.cirr_root),
        ckpt_model=Path(args.ckpt_model),
        vdr_ckpt=args.vdr_ckpt,
        device=device,
        batch_size=args.batch_size,
        pred_k=args.pred_k,
        mask_mode=args.mask_mode,
        gallery_cache_npy=Path(args.gallery_cache_npy) if args.gallery_cache_npy else None,
        force_rebuild_gallery=args.force_rebuild_gallery,
        save_recalls_json=Path(args.save_recalls_json) if args.save_recalls_json else None,
        save_recalls_subset_json=Path(args.save_recalls_subset_json) if args.save_recalls_subset_json else None,
        override_k_delta=args.override_k_delta,
        override_text_topk=args.override_text_topk,
    )

    print("\n=== CIRR rc2 Validation ===")
    for k, v in out["Recall@K"].items():
        print(f"Recall@{k}: {v*100:.2f}")
    for k, v in out["Recall_subset@K"].items():
        print(f"Recall_subset@{k}: {v*100:.2f}")


if __name__ == "__main__":
    main()
