#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
from pathlib import Path
from typing import List, Dict, Any, Tuple

import torch
import torch.nn.functional as F

from ir import Retriever


def gather_images(paths: List[str]) -> List[str]:
    exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
    out: List[str] = []
    for p in paths:
        pp = Path(p)
        if pp.is_dir():
            for f in sorted(pp.rglob("*")):
                if f.suffix.lower() in exts:
                    out.append(str(f))
        else:
            # glob 対応
            if any(ch in p for ch in ["*", "?", "["]):
                for f in sorted(Path().glob(p)):
                    if f.is_file() and f.suffix.lower() in exts:
                        out.append(str(f))
            else:
                if pp.suffix.lower() in exts:
                    out.append(str(pp))
    # 重複除去（順序維持）
    seen = set()
    uniq = []
    for x in out:
        if x not in seen:
            uniq.append(x)
            seen.add(x)
    return uniq


@torch.no_grad()
def compute_metrics_for_image(vdr: Retriever, image_path: str, topk: int) -> Dict[str, Any]:
    enc = vdr.encoder_p
    device = next(vdr.parameters()).device

    # --- 1) Dense token features: h = [1,49,768]
    x = enc.load_image_file(image_path)          # [1,3,224,224] 想定
    x = x.to(device).type(enc.dtype)

    h = enc(x)                                   # [1,L,D]
    assert h.dim() == 3, f"expected h as [N,L,D], got {tuple(h.shape)}"
    N, L, D = h.shape

    # --- 2) Lexical projection matrix W: [V,D]
    W = enc.proj
    assert W.dim() == 2 and W.shape[1] == D, f"expected W as [V,D]=[*,{D}], got {tuple(W.shape)}"
    V = W.shape[0]

    # --- 3) lv = h @ W^T : [1,L,V]
    lv = h @ W.t()

    # --- 4) Token-level “self reconstruction”: h_hat_tok = lv @ W : [1,L,D]
    #      (This ignores max/topk, purely tests projection geometry.)
    h_hat_tok = lv @ W
    cos_tok = F.cosine_similarity(h_hat_tok.flatten(1).float(), h.flatten(1).float()).item()
    rmse_tok = (h_hat_tok.float() - h.float()).pow(2).mean().sqrt().item()

    # --- 5) Compare v->h_hat against pooled dense targets
    h_mean = h.mean(dim=1)                        # [1,D]
    h_max = h.max(dim=1)[0]                       # [1,D]

    # (a) topk as given
    v_topk = enc.embed([image_path], topk=topk)   # [1,V]
    assert v_topk.shape == (1, V), f"expected v as [1,{V}], got {tuple(v_topk.shape)}"
    h_hat_topk = v_topk @ W                       # [1,D]

    cos_mean_topk = F.cosine_similarity(h_hat_topk.float(), h_mean.float()).item()
    rmse_mean_topk = (h_hat_topk.float() - h_mean.float()).pow(2).mean().sqrt().item()

    cos_max_topk = F.cosine_similarity(h_hat_topk.float(), h_max.float()).item()
    rmse_max_topk = (h_hat_topk.float() - h_max.float()).pow(2).mean().sqrt().item()

    nnz_topk = int((v_topk != 0).sum().item())

    # (b) “no topk” by setting topk=V (attempt)
    v_all = enc.embed([image_path], topk=V)
    h_hat_all = v_all @ W

    cos_mean_all = F.cosine_similarity(h_hat_all.float(), h_mean.float()).item()
    rmse_mean_all = (h_hat_all.float() - h_mean.float()).pow(2).mean().sqrt().item()

    cos_max_all = F.cosine_similarity(h_hat_all.float(), h_max.float()).item()
    rmse_max_all = (h_hat_all.float() - h_max.float()).pow(2).mean().sqrt().item()

    nnz_all = int((v_all != 0).sum().item())

    return {
        "image": image_path,
        "L": L, "D": D, "V": V,
        "token_recon": {"cos": cos_tok, "rmse": rmse_tok},
        "v2h_topk": {
            "topk": topk,
            "nnz": nnz_topk,
            "cos_to_h_mean": cos_mean_topk,
            "rmse_to_h_mean": rmse_mean_topk,
            "cos_to_h_max": cos_max_topk,
            "rmse_to_h_max": rmse_max_topk,
        },
        "v2h_all": {
            "topk": V,
            "nnz": nnz_all,
            "cos_to_h_mean": cos_mean_all,
            "rmse_to_h_mean": rmse_mean_all,
            "cos_to_h_max": cos_max_all,
            "rmse_to_h_max": rmse_max_all,
        },
    }


def summarize(rows: List[Dict[str, Any]]) -> None:
    def mean(xs):
        return sum(xs) / max(len(xs), 1)

    cos_tok = [r["token_recon"]["cos"] for r in rows]
    rmse_tok = [r["token_recon"]["rmse"] for r in rows]

    cos_mean_topk = [r["v2h_topk"]["cos_to_h_mean"] for r in rows]
    rmse_mean_topk = [r["v2h_topk"]["rmse_to_h_mean"] for r in rows]
    cos_max_topk = [r["v2h_topk"]["cos_to_h_max"] for r in rows]
    rmse_max_topk = [r["v2h_topk"]["rmse_to_h_max"] for r in rows]

    cos_mean_all = [r["v2h_all"]["cos_to_h_mean"] for r in rows]
    rmse_mean_all = [r["v2h_all"]["rmse_to_h_mean"] for r in rows]
    cos_max_all = [r["v2h_all"]["cos_to_h_max"] for r in rows]
    rmse_max_all = [r["v2h_all"]["rmse_to_h_max"] for r in rows]

    print("\n=== Summary (mean over images) ===")
    print(f"[token recon]    cos={mean(cos_tok): .4f}   rmse={mean(rmse_tok): .4f}")
    print(f"[v->h_mean topk] cos={mean(cos_mean_topk): .4f}   rmse={mean(rmse_mean_topk): .4f}")
    print(f"[v->h_max  topk] cos={mean(cos_max_topk): .4f}   rmse={mean(rmse_max_topk): .4f}")
    print(f"[v->h_mean all ] cos={mean(cos_mean_all): .4f}   rmse={mean(rmse_mean_all): .4f}")
    print(f"[v->h_max  all ] cos={mean(cos_max_all): .4f}   rmse={mean(rmse_max_all): .4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, default="vsearch/vdr-cross-modal")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--topk", type=int, default=None, help="use this topk for embed(); default: encoder config topk")
    ap.add_argument("paths", nargs="+", help="image files / directories / glob patterns")
    args = ap.parse_args()

    vdr = Retriever.from_pretrained(args.ckpt).to(args.device).eval()

    # Determine default topk from config if not given
    default_topk = getattr(vdr.encoder_p.config, "topk", None)
    topk = args.topk if args.topk is not None else (default_topk if default_topk is not None else 0)

    images = gather_images(args.paths)
    if len(images) == 0:
        raise SystemExit("No images found. Provide image paths, a directory, or a glob like 'imgs/*.png'.")

    print(f"Loaded {len(images)} images")
    print(f"Using ckpt={args.ckpt}, device={args.device}, topk={topk}")

    rows = []
    for i, img in enumerate(images):
        r = compute_metrics_for_image(vdr, img, topk=topk)
        rows.append(r)

        print(f"\n[{i+1}/{len(images)}] {img}")
        print(f"  dims: L={r['L']} D={r['D']} V={r['V']}")
        print(f"  token_recon: cos={r['token_recon']['cos']:.4f} rmse={r['token_recon']['rmse']:.4f}")
        print(f"  v2h_topk(topk={r['v2h_topk']['topk']}, nnz={r['v2h_topk']['nnz']}): "
              f"cos(mean)={r['v2h_topk']['cos_to_h_mean']:.4f} rmse(mean)={r['v2h_topk']['rmse_to_h_mean']:.4f} | "
              f"cos(max)={r['v2h_topk']['cos_to_h_max']:.4f} rmse(max)={r['v2h_topk']['rmse_to_h_max']:.4f}")
        print(f"  v2h_all (topk={r['v2h_all']['topk']}, nnz={r['v2h_all']['nnz']}): "
              f"cos(mean)={r['v2h_all']['cos_to_h_mean']:.4f} rmse(mean)={r['v2h_all']['rmse_to_h_mean']:.4f} | "
              f"cos(max)={r['v2h_all']['cos_to_h_max']:.4f} rmse(max)={r['v2h_all']['rmse_to_h_max']:.4f}")

    summarize(rows)


if __name__ == "__main__":
    main()
