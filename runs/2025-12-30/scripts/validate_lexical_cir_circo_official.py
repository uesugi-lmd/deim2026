#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from ir import Retriever


# =========================
# CIRCO metrics (official-like)
# =========================
def compute_metrics_circo_official(
    data_path: Path,
    predictions_dict: Dict[int, List[int]],
    ranks: List[int],
) -> Tuple[Dict[int, float], Dict[int, float], Dict[str, float]]:
    """
    Same logic as the code you pasted.
    mAP@k: uses all gt_img_ids
    Recall@k: uses only target_img_id
    semantic mAP@10: grouped by semantic_aspects
    """
    ann_path = data_path / "annotations" / "val.json"
    anns: List[dict] = json.loads(ann_path.read_text(encoding="utf-8"))

    semantic_aspects_list = [
        "cardinality", "addition", "negation", "direct_addressing", "compare_change",
        "comparative_statement", "statement_with_conjunction",
        "spatial_relations_background", "viewpoint"
    ]

    aps_atk = defaultdict(list)
    recalls_atk = defaultdict(list)
    semantic_aps_at10 = defaultdict(list)

    for query_id, predictions in predictions_dict.items():
        qid = int(query_id)
        target_img_id = int(anns[qid]["target_img_id"])
        gt_img_ids = np.array(anns[qid]["gt_img_ids"], dtype=int)
        semantic_aspects = anns[qid].get("semantic_aspects", [])

        if len(set(predictions)) != len(predictions):
            raise ValueError(f"Query {qid} has duplicate predictions.")

        predictions = np.array(predictions, dtype=int)

        ap_labels = np.isin(predictions, gt_img_ids)
        precisions = np.cumsum(ap_labels, axis=0) * ap_labels
        precisions = precisions / np.arange(1, ap_labels.shape[0] + 1)

        for rank in ranks:
            aps_atk[rank].append(float(np.sum(precisions[:rank]) / min(len(gt_img_ids), rank)))

        recall_labels = (predictions == target_img_id)
        for rank in ranks:
            recalls_atk[rank].append(float(np.sum(recall_labels[:rank])))

        for aspect in semantic_aspects:
            semantic_aps_at10[aspect].append(float(np.sum(precisions[:10]) / min(len(gt_img_ids), 10)))

    map_atk = {rank: float(np.mean(aps_atk[rank])) for rank in ranks}
    recall_atk = {rank: float(np.mean(recalls_atk[rank])) for rank in ranks}

    semantic_map_at10 = {}
    for aspect in semantic_aspects_list:
        semantic_map_at10[aspect] = float(np.mean(semantic_aps_at10[aspect])) if len(semantic_aps_at10[aspect]) > 0 else float("nan")

    return map_atk, recall_atk, semantic_map_at10


# =========================
# CIRCO dataset loading (no torch Dataset needed)
# =========================
@dataclass
class CIRCOValQuery:
    qid: int
    ref_img_id: int
    tgt_img_id: int
    relative_caption: str
    gt_img_ids: List[int]


def load_circo_val_queries(data_path: Path) -> List[CIRCOValQuery]:
    anns: List[dict] = json.loads((data_path / "annotations" / "val.json").read_text(encoding="utf-8"))
    out: List[CIRCOValQuery] = []
    for it in anns:
        out.append(
            CIRCOValQuery(
                qid=int(it["id"]),
                ref_img_id=int(it["reference_img_id"]),
                tgt_img_id=int(it["target_img_id"]),
                relative_caption=str(it["relative_caption"]),
                gt_img_ids=[int(x) for x in it["gt_img_ids"]],
            )
        )
    # sanity: qids consecutive 0..N-1
    out_sorted = sorted(out, key=lambda x: x.qid)
    assert [q.qid for q in out_sorted] == list(range(len(out_sorted))), "CIRCO query ids must be consecutive from 0."
    return out_sorted


def load_coco2017_unlabeled_gallery(data_path: Path) -> Tuple[List[int], List[str], Dict[int, int]]:
    """
    Uses COCO2017_unlabeled/annotations/image_info_unlabeled2017.json
    gallery img paths: COCO2017_unlabeled/unlabeled2017/<file_name>
    """
    info_path = data_path / "COCO2017_unlabeled" / "annotations" / "image_info_unlabeled2017.json"
    info = json.loads(info_path.read_text(encoding="utf-8"))
    images = info["images"]

    img_ids: List[int] = [int(im["id"]) for im in images]
    img_paths: List[str] = [str(data_path / "COCO2017_unlabeled" / "unlabeled2017" / im["file_name"]) for im in images]
    id2idx: Dict[int, int] = {img_id: i for i, img_id in enumerate(img_ids)}
    return img_ids, img_paths, id2idx


# =========================
# Model components (match your training ckpt format)
# =========================
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
        s = s.float()
        z = self.linear(s)
        return z / (z.norm(dim=-1, keepdim=True) + 1e-6)


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

    # expected keys from your training script
    delta_gen.load_state_dict(ckpt["delta_gen"], strict=True)
    decoder.load_state_dict(ckpt["decoder"], strict=True)

    delta_gen.eval(); decoder.eval()
    return LoadedParts(delta_gen=delta_gen, decoder=decoder, V=V, D=D, k_delta=k_delta, text_topk=text_topk)


# =========================
# VDR encoding (safe dtype)
# =========================
def safe_l2norm(x: torch.Tensor) -> torch.Tensor:
    return x / (x.norm(dim=-1, keepdim=True) + 1e-6)


@torch.no_grad()
def encode_gallery_dense_vdr(vdr: Retriever, image_paths: List[str], device: torch.device, batch_size: int) -> torch.Tensor:
    """
    Dense gallery feature = mean over tokens from VDR image encoder forward output [N,49,768].
    Use float32 to avoid fp16 mismatch.
    """
    enc = vdr.encoder_p
    enc = enc.float()

    outs = []
    for i in tqdm(range(0, len(image_paths), batch_size), desc="[ENC gallery]", dynamic_ncols=True):
        batch = image_paths[i:i+batch_size]
        imgs = [enc.load_image_file(p) for p in batch]
        x = torch.cat(imgs, dim=0).to(device=device, dtype=torch.float32)
        h = enc(x)           # [B,49,768]
        z = safe_l2norm(h.mean(dim=1).float())  # [B,768]
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
def compose_query(
    parts: LoadedParts,
    vdr: Retriever,
    sr: torch.Tensor,
    texts: List[str],
    device: torch.device,
    mask_mode: str,
) -> torch.Tensor:
    ht = encode_text_lexical(vdr, texts, device=device, topk=parts.text_topk)  # [B,V]

    m = None
    if mask_mode == "lex_support":
        m = (ht != 0).float()
        ht = ht * m

    dsp, dsm = parts.delta_gen(ht)  # [B,V]

    if m is not None:
        dsp = dsp * m
        dsm = dsm * m

    sq = torch.clamp(sr + dsp, min=0.0) - torch.clamp(dsm, min=0.0)
    zq = parts.decoder(sq)  # [B,D]
    return zq


# =========================
# Validation main
# =========================
@torch.no_grad()
def validate_circo(
    data_path: Path,
    ckpt_model: Path,
    vdr_ckpt: str,
    device: torch.device,
    batch_size: int,
    pred_k: int,
    ranks: List[int],
    mask_mode: str,
    gallery_cache_npy: Optional[Path],
    force_rebuild_gallery: bool,
    save_predictions_json: Optional[Path],
    override_k_delta: Optional[int],
    override_text_topk: Optional[int],
) -> Dict[str, Any]:
    # Load queries and gallery
    queries = load_circo_val_queries(data_path)
    gal_img_ids, gal_paths, id2idx = load_coco2017_unlabeled_gallery(data_path)

    print(f"[DATA] queries={len(queries)} gallery={len(gal_img_ids)} data_path={data_path}")
    print(f"[MODEL] ckpt={ckpt_model}")
    print(f"[VDR]   ckpt={vdr_ckpt}")

    # Load model parts + VDR
    vdr = Retriever.from_pretrained(vdr_ckpt).to(device).eval()
    vdr.encoder_p = vdr.encoder_p.float()  # important

    parts = load_parts_from_ckpt(ckpt_model, device=device, override_k_delta=override_k_delta, override_text_topk=override_text_topk)
    print(f"[CFG]   V={parts.V} D={parts.D} k_delta={parts.k_delta} text_topk={parts.text_topk} mask={mask_mode}")

    # Encode gallery (cache)
    if gallery_cache_npy is not None and gallery_cache_npy.exists() and not force_rebuild_gallery:
        print(f"[CACHE] load gallery dense from {gallery_cache_npy}")
        gal = torch.from_numpy(np.load(str(gallery_cache_npy))).to(device).float()
        gal = safe_l2norm(gal)
    else:
        gal = encode_gallery_dense_vdr(vdr, gal_paths, device=device, batch_size=batch_size)
        if gallery_cache_npy is not None:
            gallery_cache_npy.parent.mkdir(parents=True, exist_ok=True)
            np.save(str(gallery_cache_npy), gal.detach().cpu().numpy().astype(np.float32))
            print(f"[CACHE] saved gallery dense to {gallery_cache_npy}")

    # Prepare predictions dict
    predictions_dict: Dict[int, List[int]] = {}

    # batched loop over queries
    for i in tqdm(range(0, len(queries), batch_size), desc="[VAL queries]", dynamic_ncols=True):
        batch = queries[i:i+batch_size]
        ref_ids = [q.ref_img_id for q in batch]
        texts = [q.relative_caption for q in batch]

        # map ref id -> path
        ref_paths = [gal_paths[id2idx[rid]] for rid in ref_ids]

        sr = encode_ref_lexical(vdr, ref_paths, device=device, topk=768)
        zq = compose_query(parts, vdr, sr, texts, device=device, mask_mode=mask_mode)  # [B,768]

        scores = zq @ gal.t()  # [B, G]
        topk_idx = scores.topk(k=pred_k, dim=1, largest=True).indices  # [B, pred_k]

        for bi, q in enumerate(batch):
            idxs = topk_idx[bi].tolist()
            pred_img_ids = [int(gal_img_ids[j]) for j in idxs]
            # ensure unique (topk gives unique indices anyway)
            predictions_dict[int(q.qid)] = pred_img_ids

    # Compute metrics with official-like function
    map_atk, recall_atk, semantic_map_at10 = compute_metrics_circo_official(
        data_path=data_path,
        predictions_dict=predictions_dict,
        ranks=ranks,
    )

    # Save predictions if needed
    if save_predictions_json is not None:
        save_predictions_json.parent.mkdir(parents=True, exist_ok=True)
        with open(save_predictions_json, "w") as f:
            json.dump({str(k): v for k, v in predictions_dict.items()}, f)
        print(f"[SAVE] predictions -> {save_predictions_json}")

    return {
        "mAP@k": map_atk,
        "Recall@k": recall_atk,
        "semantic_mAP@10": semantic_map_at10,
        "pred_k": pred_k,
        "num_queries": len(queries),
        "gallery_size": len(gal_img_ids),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_path", type=str, required=True, help="CIRCO root directory (contains COCO2017_unlabeled/ and annotations/)")
    ap.add_argument("--ckpt_model", type=str, required=True, help="trained lexical-cir checkpoint (ckpt_best.pt)")
    ap.add_argument("--vdr_ckpt", type=str, default="vsearch/vdr-cross-modal")
    ap.add_argument("--device", type=str, default="cuda")

    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--pred_k", type=int, default=50, help="length of ranking list per query (must be >= max(ranks))")
    ap.add_argument("--ranks", type=int, nargs="+", default=[5, 10, 25, 50])
    ap.add_argument("--mask_mode", type=str, default="lex_support", choices=["lex_support", "none"])

    ap.add_argument("--gallery_cache_npy", type=str, default=None)
    ap.add_argument("--force_rebuild_gallery", action="store_true")

    ap.add_argument("--save_predictions_json", type=str, default=None)

    ap.add_argument("--override_k_delta", type=int, default=None)
    ap.add_argument("--override_text_topk", type=int, default=None)

    args = ap.parse_args()
    device = torch.device(args.device)

    assert args.pred_k >= max(args.ranks), "--pred_k must be >= max(--ranks)"

    out = validate_circo(
        data_path=Path(args.data_path),
        ckpt_model=Path(args.ckpt_model),
        vdr_ckpt=args.vdr_ckpt,
        device=device,
        batch_size=args.batch_size,
        pred_k=args.pred_k,
        ranks=args.ranks,
        mask_mode=args.mask_mode,
        gallery_cache_npy=Path(args.gallery_cache_npy) if args.gallery_cache_npy else None,
        force_rebuild_gallery=args.force_rebuild_gallery,
        save_predictions_json=Path(args.save_predictions_json) if args.save_predictions_json else None,
        override_k_delta=args.override_k_delta,
        override_text_topk=args.override_text_topk,
    )

    print("\nWe remind that the mAP@k metrics are computed considering all the ground truth images for each query,")
    print("the Recall@k metrics are computed considering only the target image for each query.\n")

    print("mAP@k metrics")
    for rank in args.ranks:
        print(f"mAP@{rank}: {out['mAP@k'][rank] * 100:.2f}")

    print("\nRecall@k metrics")
    for rank in args.ranks:
        print(f"Recall@{rank}: {out['Recall@k'][rank] * 100:.2f}")

    print("\nSemantic mAP@10 metrics")
    for aspect, val in out["semantic_mAP@10"].items():
        if np.isnan(val):
            print(f"Semantic mAP@10 for aspect '{aspect}': NaN (no samples)")
        else:
            print(f"Semantic mAP@10 for aspect '{aspect}': {val * 100:.2f}")


if __name__ == "__main__":
    main()
