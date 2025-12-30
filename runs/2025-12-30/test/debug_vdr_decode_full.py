#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
from typing import List, Dict, Any

import torch
import torch.nn.functional as F

from ir import Retriever


# -----------------------------
# Utilities
# -----------------------------
def gather_images(paths: List[str]) -> List[str]:
    exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
    out: List[str] = []
    for p in paths:
        pp = Path(p)
        if pp.is_dir():
            for f in sorted(pp.rglob("*")):
                if f.is_file() and f.suffix.lower() in exts:
                    out.append(str(f))
        else:
            # glob 対応
            if any(ch in p for ch in ["*", "?", "["]):
                for f in sorted(Path().glob(p)):
                    if f.is_file() and f.suffix.lower() in exts:
                        out.append(str(f))
            else:
                if pp.is_file() and pp.suffix.lower() in exts:
                    out.append(str(pp))

    # 重複除去（順序維持）
    seen = set()
    uniq = []
    for x in out:
        if x not in seen:
            uniq.append(x)
            seen.add(x)
    return uniq


def safe_mean(xs: List[float]) -> float:
    return float(sum(xs) / max(len(xs), 1))


# -----------------------------
# Core: reconstructions
# -----------------------------
@torch.no_grad()
def token_recon_ridge(h: torch.Tensor, W: torch.Tensor, ridge: float) -> Dict[str, float]:
    """
    h: [1,L,D] (float/half ok)
    W: [V,D]    (proj matrix)
    Reconstruct h from lv=hW^T using ridge-regularized least squares:
      h_hat = lv W (W^T W + λI)^(-1)
    """
    h_f = h.float()
    W_f = W.float()

    lv = h_f @ W_f.t()                      # [1,L,V]
    G = W_f.t() @ W_f                       # [D,D]
    D = G.shape[0]
    G_reg = G + ridge * torch.eye(D, device=G.device, dtype=G.dtype)

    # inv でも solve でもよい（solveの方が安定）
    # h_hat = (lv @ W_f) @ torch.linalg.inv(G_reg)
    h_hat = torch.linalg.solve(G_reg, (lv @ W_f).transpose(-1, -2)).transpose(-1, -2)  # [1,L,D]

    # token-wise cosine (平均)
    cos_tok = F.cosine_similarity(
        F.normalize(h_hat, dim=-1),
        F.normalize(h_f, dim=-1),
        dim=-1
    ).mean().item()

    rmse = (h_hat - h_f).pow(2).mean().sqrt().item()

    return {"cos": float(cos_tok), "rmse": float(rmse)}


@torch.no_grad()
def v2h_linear(v: torch.Tensor, W: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
    """
    v: [1,V]
    W: [V,D]
    target: [1,D]
    naive linear decode: h_hat = v @ W
    """
    v_f = v.float()
    W_f = W.float()
    tgt = target.float()
    h_hat = v_f @ W_f                        # [1,D]

    cos = F.cosine_similarity(h_hat, tgt).item()
    rmse = (h_hat - tgt).pow(2).mean().sqrt().item()
    return {"cos": float(cos), "rmse": float(rmse)}


@torch.no_grad()
def compute_metrics_for_image(vdr: Retriever, image_path: str, topk: int, ridge: float) -> Dict[str, Any]:
    enc = vdr.encoder_p
    device = next(vdr.parameters()).device

    # --- Dense token features
    x = enc.load_image_file(image_path).to(device).type(enc.dtype)  # [1,3,224,224]
    h = enc(x)                                                      # [1,L,D]
    assert h.dim() == 3, f"expected h [1,L,D], got {tuple(h.shape)}"
    _, L, D = h.shape

    # --- Projection
    W = enc.proj                                                    # [V,D]
    assert W.dim() == 2 and W.shape[1] == D, f"W expected [V,{D}], got {tuple(W.shape)}"
    V = W.shape[0]

    # pooled dense targets (比較用)
    h_mean = h.mean(dim=1)                                          # [1,D]
    h_max  = h.max(dim=1)[0]                                         # [1,D]

    # --- Token-level ridge reconstruction (tests invertibility of W before max/topk)
    tok_rec = token_recon_ridge(h, W, ridge=ridge)

    # --- v from embed (topk)
    v_topk = enc.embed([image_path], topk=topk)                      # [1,V]
    nnz_topk = int((v_topk != 0).sum().item())
    v2h_topk_mean = v2h_linear(v_topk, W, h_mean)
    v2h_topk_max  = v2h_linear(v_topk, W, h_max)

    # --- pseudo "no topk": set topk=V
    v_all = enc.embed([image_path], topk=V)
    nnz_all = int((v_all != 0).sum().item())
    v2h_all_mean = v2h_linear(v_all, W, h_mean)
    v2h_all_max  = v2h_linear(v_all, W, h_max)

    return {
        "image": image_path,
        "dims": {"L": int(L), "D": int(D), "V": int(V)},
        "token_recon_ridge": {"ridge": float(ridge), **tok_rec},
        "v2h_topk": {
            "topk": int(topk),
            "nnz": int(nnz_topk),
            "to_h_mean": v2h_topk_mean,
            "to_h_max": v2h_topk_max,
        },
        "v2h_all": {
            "topk": int(V),
            "nnz": int(nnz_all),
            "to_h_mean": v2h_all_mean,
            "to_h_max": v2h_all_max,
        },
    }


def summarize(rows: List[Dict[str, Any]]) -> None:
    tok_cos = [r["token_recon_ridge"]["cos"] for r in rows]
    tok_rmse = [r["token_recon_ridge"]["rmse"] for r in rows]

    topk_cos_mean = [r["v2h_topk"]["to_h_mean"]["cos"] for r in rows]
    topk_rmse_mean = [r["v2h_topk"]["to_h_mean"]["rmse"] for r in rows]
    topk_cos_max = [r["v2h_topk"]["to_h_max"]["cos"] for r in rows]
    topk_rmse_max = [r["v2h_topk"]["to_h_max"]["rmse"] for r in rows]

    all_cos_mean = [r["v2h_all"]["to_h_mean"]["cos"] for r in rows]
    all_rmse_mean = [r["v2h_all"]["to_h_mean"]["rmse"] for r in rows]
    all_cos_max = [r["v2h_all"]["to_h_max"]["cos"] for r in rows]
    all_rmse_max = [r["v2h_all"]["to_h_max"]["rmse"] for r in rows]

    print("\n=== Summary (mean over images) ===")
    print(f"[token recon ridge]    cos={safe_mean(tok_cos): .4f}   rmse={safe_mean(tok_rmse): .4f}")
    print(f"[v->h_mean topk]       cos={safe_mean(topk_cos_mean): .4f}   rmse={safe_mean(topk_rmse_mean): .4f}")
    print(f"[v->h_max  topk]       cos={safe_mean(topk_cos_max): .4f}   rmse={safe_mean(topk_rmse_max): .4f}")
    print(f"[v->h_mean all ]       cos={safe_mean(all_cos_mean): .4f}   rmse={safe_mean(all_rmse_mean): .4f}")
    print(f"[v->h_max  all ]       cos={safe_mean(all_cos_max): .4f}   rmse={safe_mean(all_rmse_max): .4f}")


# -----------------------------
# Entry
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, default="vsearch/vdr-cross-modal")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--topk", type=int, default=None, help="embed topk; default uses encoder config topk (if exists)")
    ap.add_argument("--ridge", type=float, default=1e-2, help="ridge λ for token-level reconstruction")
    ap.add_argument("paths", nargs="+", help="image files / directories / glob patterns")
    args = ap.parse_args()

    vdr = Retriever.from_pretrained(args.ckpt).to(args.device).eval()

    default_topk = getattr(vdr.encoder_p.config, "topk", None)
    topk = args.topk if args.topk is not None else (default_topk if default_topk is not None else 768)

    images = gather_images(args.paths)
    if len(images) == 0:
        raise SystemExit("No images found. Provide image paths, a directory, or a glob like 'imgs/*.png'.")

    print(f"Loaded {len(images)} images")
    print(f"Using ckpt={args.ckpt}, device={args.device}, topk={topk}, ridge={args.ridge}")

    rows: List[Dict[str, Any]] = []
    for i, img in enumerate(images):
        r = compute_metrics_for_image(vdr, img, topk=topk, ridge=args.ridge)
        rows.append(r)

        L = r["dims"]["L"]; D = r["dims"]["D"]; V = r["dims"]["V"]
        print(f"\n[{i+1}/{len(images)}] {img}")
        print(f"  dims: L={L} D={D} V={V}")

        tr = r["token_recon_ridge"]
        print(f"  token_recon_ridge(ridge={tr['ridge']}): cos={tr['cos']:.4f} rmse={tr['rmse']:.4f}")

        t = r["v2h_topk"]
        print(f"  v2h_topk(topk={t['topk']}, nnz={t['nnz']}): "
              f"cos(mean)={t['to_h_mean']['cos']:.4f} rmse(mean)={t['to_h_mean']['rmse']:.4f} | "
              f"cos(max)={t['to_h_max']['cos']:.4f} rmse(max)={t['to_h_max']['rmse']:.4f}")

        a = r["v2h_all"]
        print(f"  v2h_all (topk={a['topk']}, nnz={a['nnz']}): "
              f"cos(mean)={a['to_h_mean']['cos']:.4f} rmse(mean)={a['to_h_mean']['rmse']:.4f} | "
              f"cos(max)={a['to_h_max']['cos']:.4f} rmse(max)={a['to_h_max']['rmse']:.4f}")

    summarize(rows)


if __name__ == "__main__":
    main()
