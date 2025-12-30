#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Set

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from ir import Retriever


# =========================
# Utils
# =========================
def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_json_maybe_list_or_jsonlines(path: Path) -> List[Dict[str, Any]]:
    """
    CIRR val json in your example is a JSON list.
    Some files can be jsonlines-like; support both.
    """
    txt = path.read_text(encoding="utf-8").strip()
    if not txt:
        return []
    if txt[0] == "[":
        data = json.loads(txt)
        assert isinstance(data, list)
        return data
    # jsonlines-ish
    items = []
    for line in txt.splitlines():
        s = line.strip()
        if not s:
            continue
        if s.endswith(","):
            s = s[:-1]
        items.append(json.loads(s))
    return items


def safe_l2norm(x: torch.Tensor, dim: int = -1, eps: float = 1e-6) -> torch.Tensor:
    return x / (x.norm(dim=dim, keepdim=True) + eps)


@torch.no_grad()
def recall_hit_any_gt(scores: torch.Tensor, gt_indices_list: List[List[int]], ks=(1, 5, 10, 50)) -> Dict[str, float]:
    """
    scores: [B, N]
    gt_indices_list: list of list of gt indices in [0, N)
    hit if any gt index appears in top-k.
    """
    ranks = scores.argsort(dim=1, descending=True)
    out = {}
    for k in ks:
        hits = []
        topk = ranks[:, :k]
        for i in range(scores.size(0)):
            gts = set(gt_indices_list[i])
            hits.append(1.0 if any(int(x) in gts for x in topk[i].tolist()) else 0.0)
        out[f"R@{k}"] = float(np.mean(hits))
    return out


# =========================
# Load trained parts
# =========================
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
        return safe_l2norm(self.linear(s.float()))


@dataclass
class LoadedLexicalCIR:
    delta_gen: DeltaLexicalGenerator
    decoder: DenseDecoder
    V: int
    D: int
    k_delta: int
    text_topk: int


def load_trained_parts(ckpt_path: str, device: torch.device,
                       override_k_delta: Optional[int] = None,
                       override_text_topk: Optional[int] = None) -> LoadedLexicalCIR:
    ckpt = torch.load(ckpt_path, map_location=device)
    args = ckpt.get("args", {})

    V = int(ckpt.get("V", 27623))
    D = int(ckpt.get("D", 768))

    k_delta = int(override_k_delta if override_k_delta is not None else args.get("k_delta", 64))
    text_topk = int(override_text_topk if override_text_topk is not None else args.get("text_topk", 768))
    bottleneck = int(args.get("bottleneck", 512))

    delta_gen = DeltaLexicalGenerator(din=V, V=V, k_delta=k_delta, bottleneck=bottleneck).to(device)
    decoder = DenseDecoder(V=V, D=D).to(device)

    delta_gen.load_state_dict(ckpt["delta_gen"], strict=True)
    decoder.load_state_dict(ckpt["decoder"], strict=True)
    delta_gen.eval(); decoder.eval()

    return LoadedLexicalCIR(delta_gen=delta_gen, decoder=decoder, V=V, D=D, k_delta=k_delta, text_topk=text_topk)


# =========================
# VDR encode (no DB)
# =========================
@torch.no_grad()
def encode_gallery_dense_vdr(
    vdr: Retriever,
    image_paths: List[str],
    device: torch.device,
    batch_size: int = 64,
) -> torch.Tensor:
    """
    Dense: mean pool over tokens from VDR image encoder forward() output [N,49,768]
    Use float32 to avoid fp16 mismatch.
    """
    enc = vdr.encoder_p
    enc = enc.float()  # make sure weights are float32

    outs = []
    for i in range(0, len(image_paths), batch_size):
        batch = image_paths[i:i+batch_size]
        imgs = [enc.load_image_file(p) for p in batch]
        x = torch.cat(imgs, dim=0).to(device=device, dtype=torch.float32)

        h = enc(x)          # [N,49,768]
        z = h.mean(dim=1)   # [N,768]
        z = safe_l2norm(z.float())
        outs.append(z)

    return torch.cat(outs, dim=0)


@torch.no_grad()
def encode_ref_lexical_vdr(vdr: Retriever, ref_paths: List[str], device: torch.device, topk: int = 768) -> torch.Tensor:
    vdr.encoder_p = vdr.encoder_p.float()
    sr = vdr.encoder_p.embed(ref_paths, topk=topk)   # [B,V]
    return sr.to(device)


@torch.no_grad()
def encode_text_lexical_vdr(vdr: Retriever, texts: List[str], device: torch.device, topk: int = 768) -> torch.Tensor:
    ht = vdr.encoder_q.embed(texts, topk=topk)       # [B,V]
    return ht.to(device)


@torch.no_grad()
def compose_zq(parts: LoadedLexicalCIR, vdr: Retriever, sr: torch.Tensor, texts: List[str],
               device: torch.device, mask_mode: str = "lex_support") -> torch.Tensor:
    ht = encode_text_lexical_vdr(vdr, texts, device=device, topk=parts.text_topk)
    if mask_mode == "lex_support":
        m = (ht != 0).float()
        ht = ht * m
    else:
        m = None
    dsp, dsm = parts.delta_gen(ht)
    if m is not None:
        dsp = dsp * m
        dsm = dsm * m
    sq = torch.clamp(sr + dsp, min=0.0) - torch.clamp(dsm, min=0.0)
    zq = parts.decoder(sq)
    return zq


# =========================
# CIRR helpers
# =========================
def cirr_id_to_relpath(img_id: str) -> str:
    """
    CIRR image id example: 'dev-903-0-img0'
    We need a mapping to actual file path.

    Common CIRR layout (varies by release):
      root/images/dev/<img_id>.jpg   OR  <img_id>.png
      root/images/train/<...>

    We'll try jpg then png.
    """
    return img_id  # just return id; extension resolved later


def resolve_cirr_image_path(cirr_root: Path, img_id: str) -> str:
    # split name is prefix before first '-': dev/test1/train
    split = img_id.split("-")[0]
    base = cirr_root / "images" / split / img_id
    for ext in [".jpg", ".png", ".jpeg", ".webp"]:
        p = Path(str(base) + ext)
        if p.exists():
            return str(p)
    # if already has ext
    if base.exists():
        return str(base)
    raise FileNotFoundError(f"[CIRR] image file not found for id={img_id} under {cirr_root}/images/{split}/")


@dataclass
class CIRRQuery:
    ref_id: str
    caption: str
    members: List[str]           # candidate set
    gt_hard: str                 # target_hard


def load_cirr_val_queries(cirr_root: str, ann_path: str) -> List[CIRRQuery]:
    path = Path(ann_path)
    if not path.is_absolute():
        path = Path(cirr_root) / path
    items = load_json_maybe_list_or_jsonlines(path)

    qs = []
    for it in items:
        ref = it["reference"]
        cap = it["caption"]
        gt = it["target_hard"]
        members = it["img_set"]["members"]
        qs.append(CIRRQuery(ref_id=ref, caption=cap, members=members, gt_hard=gt))
    return qs


# =========================
# CIRCO helpers
# =========================
def build_coco_id_to_filename(coco_images_dir: Path) -> Dict[int, str]:
    """
    CIRCO uses COCO-style img ids. We need id -> filename.
    We scan the directory once and parse 12-digit COCO id from name.

    Expected filenames: COCO_val2014_000000271520.jpg (or train2014)
    """
    id2path: Dict[int, str] = {}
    for p in coco_images_dir.rglob("*"):
        if not p.is_file():
            continue
        name = p.name
        if "COCO_" in name and name.endswith((".jpg", ".jpeg", ".png")):
            # take last 12 digits before extension
            stem = p.stem
            digits = stem.split("_")[-1]
            if digits.isdigit():
                img_id = int(digits)
                id2path[img_id] = str(p)
    return id2path


@dataclass
class CIRCOQuery:
    ref_img_id: int
    relative_caption: str
    gt_img_ids: List[int]   # multiple GTs


def load_circo_val_queries(circo_root: str, ann_json: str) -> List[CIRCOQuery]:
    path = Path(circo_root) / ann_json
    data = load_json(path)
    assert isinstance(data, list), "CIRCO val.json should be a list."
    qs = []
    for it in data:
        qs.append(
            CIRCOQuery(
                ref_img_id=int(it["reference_img_id"]),
                relative_caption=str(it["relative_caption"]),
                gt_img_ids=[int(x) for x in it["gt_img_ids"]],
            )
        )
    return qs


# =========================
# Validation routines
# =========================
@torch.no_grad()
def validate_cirr(
    cirr_root: str,
    ann_json: str,
    vdr: Retriever,
    parts: LoadedLexicalCIR,
    device: torch.device,
    batch_size: int = 64,
    mask_mode: str = "lex_support",
    subset_gallery: bool = True,
    gallery_cache_npy: Optional[str] = None,
    force_rebuild_gallery: bool = False,
) -> Dict[str, float]:
    """
    CIRR:
      subset_gallery=True: per-query gallery = img_set.members (standard CIRR setting)
      subset_gallery=False: global gallery = all images in split (not recommended unless you want it)
    """
    root = Path(cirr_root)
    qs = load_cirr_val_queries(cirr_root, ann_json)

    # Precompute global gallery if requested
    global_gallery_paths: Optional[List[str]] = None
    global_gallery_emb: Optional[torch.Tensor] = None
    global_id2idx: Optional[Dict[str, int]] = None

    if not subset_gallery:
        # build set from all members across queries (cheap, no directory scan)
        ids: List[str] = []
        seen: Set[str] = set()
        for q in qs:
            for m in q.members:
                if m not in seen:
                    seen.add(m)
                    ids.append(m)

        global_gallery_paths = [resolve_cirr_image_path(root, mid) for mid in ids]
        global_id2idx = {mid: i for i, mid in enumerate(ids)}

        if gallery_cache_npy is not None:
            cp = Path(gallery_cache_npy)
            if cp.exists() and not force_rebuild_gallery:
                global_gallery_emb = safe_l2norm(torch.from_numpy(np.load(str(cp))).to(device).float())
            else:
                global_gallery_emb = encode_gallery_dense_vdr(vdr, global_gallery_paths, device=device, batch_size=batch_size)
                cp.parent.mkdir(parents=True, exist_ok=True)
                np.save(str(cp), global_gallery_emb.detach().cpu().numpy().astype(np.float32))
        else:
            global_gallery_emb = encode_gallery_dense_vdr(vdr, global_gallery_paths, device=device, batch_size=batch_size)

    r1, r5, r10, r50 = [], [], [], []
    pbar = tqdm(range(0, len(qs), batch_size), desc="[VAL CIRR]", dynamic_ncols=True)

    for i in pbar:
        batch = qs[i:i+batch_size]
        refs = [resolve_cirr_image_path(root, q.ref_id) for q in batch]
        texts = [q.caption for q in batch]

        # ref lexical
        sr = encode_ref_lexical_vdr(vdr, refs, device=device, topk=768)
        zq = compose_zq(parts, vdr, sr, texts, device=device, mask_mode=mask_mode)

        if subset_gallery:
            # Each query has different gallery; we evaluate one-by-one (still batched in zq)
            for bi, q in enumerate(batch):
                member_ids = q.members
                member_paths = [resolve_cirr_image_path(root, mid) for mid in member_ids]
                gal = encode_gallery_dense_vdr(vdr, member_paths, device=device, batch_size=batch_size)  # [M,768]
                scores = (zq[bi:bi+1] @ gal.t())  # [1,M]
                gt_idx = member_ids.index(q.gt_hard)
                m = recall_hit_any_gt(scores, [[gt_idx]], ks=(1, 5, 10, 50))
                r1.append(m["R@1"]); r5.append(m["R@5"]); r10.append(m["R@10"]); r50.append(m["R@50"])
        else:
            assert global_gallery_emb is not None and global_id2idx is not None
            scores = zq @ global_gallery_emb.t()  # [B,N]
            gt_idx = torch.tensor([global_id2idx[q.gt_hard] for q in batch], device=device)
            # single gt each
            ranks = scores.argsort(dim=1, descending=True)
            for k, arr in [(1, r1), (5, r5), (10, r10), (50, r50)]:
                hit = (ranks[:, :k] == gt_idx.unsqueeze(1)).any(dim=1).float().mean().item()
                arr.append(hit)

        pbar.set_postfix(R1=f"{np.mean(r1):.4f}", R5=f"{np.mean(r5):.4f}", R10=f"{np.mean(r10):.4f}")

    return {"R@1": float(np.mean(r1)), "R@5": float(np.mean(r5)), "R@10": float(np.mean(r10)), "R@50": float(np.mean(r50)), "num_queries": len(qs)}


@torch.no_grad()
def validate_circo(
    circo_root: str,
    ann_json: str,
    coco_images_dir: str,
    vdr: Retriever,
    parts: LoadedLexicalCIR,
    device: torch.device,
    batch_size: int = 64,
    mask_mode: str = "lex_support",
    gallery_cache_npy: Optional[str] = None,
    force_rebuild_gallery: bool = False,
) -> Dict[str, float]:
    """
    CIRCO:
      - Global gallery typically = all COCO images in the split you use.
      - Here we build gallery as union of all gt_img_ids and all reference_img_id / target_img_id to keep it manageable.
        If you want full COCO gallery, you can scan coco_images_dir and embed all, but that's heavy.
    """
    root = Path(circo_root)
    qs = load_circo_val_queries(circo_root, ann_json)

    coco_dir = Path(coco_images_dir)
    id2path = build_coco_id_to_filename(coco_dir)
    if len(id2path) == 0:
        raise RuntimeError(f"[CIRCO] could not build id->path from {coco_dir}. Please point to actual COCO images dir.")

    # Build gallery ids (union; lighter but not the official full-gallery setting)
    gal_ids: List[int] = []
    seen: Set[int] = set()
    for q in qs:
        for gid in q.gt_img_ids:
            if gid not in seen:
                seen.add(gid); gal_ids.append(gid)
        if q.ref_img_id not in seen:
            seen.add(q.ref_img_id); gal_ids.append(q.ref_img_id)

    gal_paths = [id2path[i] for i in gal_ids if i in id2path]
    id2idx = {img_id: i for i, img_id in enumerate(gal_ids) if img_id in id2path}

    # Embed gallery (cacheable)
    if gallery_cache_npy is not None:
        cp = Path(gallery_cache_npy)
        if cp.exists() and not force_rebuild_gallery:
            gal = safe_l2norm(torch.from_numpy(np.load(str(cp))).to(device).float())
        else:
            gal = encode_gallery_dense_vdr(vdr, gal_paths, device=device, batch_size=batch_size)
            cp.parent.mkdir(parents=True, exist_ok=True)
            np.save(str(cp), gal.detach().cpu().numpy().astype(np.float32))
    else:
        gal = encode_gallery_dense_vdr(vdr, gal_paths, device=device, batch_size=batch_size)

    # Evaluate
    r1, r5, r10, r50 = [], [], [], []
    pbar = tqdm(range(0, len(qs), batch_size), desc="[VAL CIRCO]", dynamic_ncols=True)
    for i in pbar:
        batch = qs[i:i+batch_size]
        refs = [id2path[q.ref_img_id] for q in batch]
        texts = [q.relative_caption for q in batch]

        sr = encode_ref_lexical_vdr(vdr, refs, device=device, topk=768)
        zq = compose_zq(parts, vdr, sr, texts, device=device, mask_mode=mask_mode)

        scores = zq @ gal.t()  # [B,N]
        gt_list = [[id2idx[g] for g in q.gt_img_ids if g in id2idx] for q in batch]
        m = recall_hit_any_gt(scores, gt_list, ks=(1, 5, 10, 50))
        r1.append(m["R@1"]); r5.append(m["R@5"]); r10.append(m["R@10"]); r50.append(m["R@50"])

        pbar.set_postfix(R1=f"{np.mean(r1):.4f}", R5=f"{np.mean(r5):.4f}", R10=f"{np.mean(r10):.4f}", G=len(gal_paths))

    return {"R@1": float(np.mean(r1)), "R@5": float(np.mean(r5)), "R@10": float(np.mean(r10)), "R@50": float(np.mean(r50)), "num_queries": len(qs), "gallery_size": len(gal_paths)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", type=str, choices=["cirr", "circo"], required=True)

    ap.add_argument("--ckpt_model", type=str, required=True)
    ap.add_argument("--vdr_ckpt", type=str, default="vsearch/vdr-cross-modal")
    ap.add_argument("--device", type=str, default="cuda")

    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--mask_mode", type=str, default="lex_support", choices=["lex_support", "none"])

    # CIRR
    ap.add_argument("--cirr_root", type=str, default="/home/uesugi/research/dataset/raw/cirr")
    ap.add_argument("--cirr_ann", type=str, default="cirr/captions/cap.rc2.val.json")
    ap.add_argument("--cirr_subset_gallery", action="store_true", help="use per-query img_set.members (standard)")
    ap.add_argument("--cirr_gallery_cache_npy", type=str, default=None)

    # CIRCO
    ap.add_argument("--circo_root", type=str, default="/home/uesugi/research/dataset/raw/circo")
    ap.add_argument("--circo_ann", type=str, default="annotations/val.json")
    ap.add_argument("--circo_coco_images_dir", type=str, default=None,
                    help="directory that contains COCO images (val2014/train2014 etc). required for circo")
    ap.add_argument("--circo_gallery_cache_npy", type=str, default=None)

    ap.add_argument("--force_rebuild_gallery", action="store_true")
    ap.add_argument("--override_k_delta", type=int, default=None)
    ap.add_argument("--override_text_topk", type=int, default=None)

    args = ap.parse_args()
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    device = torch.device(args.device)

    vdr = Retriever.from_pretrained(args.vdr_ckpt).to(device).eval()
    # keep image encoder in float32 to avoid fp16 mismatch
    vdr.encoder_p = vdr.encoder_p.float()

    parts = load_trained_parts(
        args.ckpt_model,
        device=device,
        override_k_delta=args.override_k_delta,
        override_text_topk=args.override_text_topk,
    )

    print(f"[MODEL] ckpt={args.ckpt_model}")
    print(f"[VDR]   ckpt={args.vdr_ckpt}")
    print(f"[CFG]   V={parts.V} D={parts.D} k_delta={parts.k_delta} text_topk={parts.text_topk} mask={args.mask_mode}")

    if args.task == "cirr":
        out = validate_cirr(
            cirr_root=args.cirr_root,
            ann_json=args.cirr_ann,
            vdr=vdr,
            parts=parts,
            device=device,
            batch_size=args.batch_size,
            mask_mode=args.mask_mode,
            subset_gallery=args.cirr_subset_gallery,
            gallery_cache_npy=args.cirr_gallery_cache_npy,
            force_rebuild_gallery=args.force_rebuild_gallery,
        )
        print("\n=== CIRR Validation ===")
        for k, v in out.items():
            print(f"{k}: {v}")

    elif args.task == "circo":
        if args.circo_coco_images_dir is None:
            raise ValueError("--circo_coco_images_dir is required for circo (directory containing COCO images).")
        out = validate_circo(
            circo_root=args.circo_root,
            ann_json=args.circo_ann,
            coco_images_dir=args.circo_coco_images_dir,
            vdr=vdr,
            parts=parts,
            device=device,
            batch_size=args.batch_size,
            mask_mode=args.mask_mode,
            gallery_cache_npy=args.circo_gallery_cache_npy,
            force_rebuild_gallery=args.force_rebuild_gallery,
        )
        print("\n=== CIRCO Validation ===")
        for k, v in out.items():
            print(f"{k}: {v}")


if __name__ == "__main__":
    main()
