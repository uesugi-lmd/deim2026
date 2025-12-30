#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
VDR (vsearch/vdr-cross-modal) の画像エンコーダについて，
可逆な線形部分だけ（h <-> lv）を切り出して検証するスクリプト．

- h:  encoder_p(x)            -> [N, L, D]  (dense token features)
- W:  encoder_p.proj          -> [V, D]     (lexical projection / dictionary)
- lv: h @ W^T                 -> [N, L, V]  (token-wise lexical scores)  <-- 可逆対象
- h_hat: lv @ W @ (W^T W + λI)^(-1) -> [N, L, D]  (ridge-regularized inverse)

※ embed() は使わない（max/elu/normalize/topk などの不可逆処理を含むため）
"""

import argparse
from pathlib import Path
from typing import List

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

    # unique (keep order)
    seen = set()
    uniq = []
    for x in out:
        if x not in seen:
            uniq.append(x)
            seen.add(x)
    return uniq


@torch.no_grad()
def encode_linear(h: torch.Tensor, W: torch.Tensor) -> torch.Tensor:
    """
    h: [N,L,D], W: [V,D]
    lv = h @ W^T -> [N,L,V]
    """
    return h @ W.t()


@torch.no_grad()
def decode_linear(lv: torch.Tensor, W: torch.Tensor, ridge: float = 1e-2) -> torch.Tensor:
    """
    lv: [N,L,V], W: [V,D]
    h_hat = lv @ W @ (W^T W + λI)^(-1) -> [N,L,D]
    solve-based implementation:
      (W^T W + λI) * X = (lv @ W)^T
      h_hat = X^T
    """
    lvf = lv.float()
    Wf = W.float()

    G = Wf.t() @ Wf                                # [D,D]
    D = G.shape[0]
    G_reg = G + ridge * torch.eye(D, device=G.device, dtype=G.dtype)

    rhs = (lvf @ Wf).transpose(-1, -2)              # [N,D,L]
    sol = torch.linalg.solve(G_reg, rhs)            # [N,D,L]
    h_hat = sol.transpose(-1, -2)                   # [N,L,D]
    return h_hat


@torch.no_grad()
def metrics(h: torch.Tensor, h_hat: torch.Tensor) -> dict:
    # token-wise cosine mean
    cos_tok = F.cosine_similarity(
        F.normalize(h_hat.float(), dim=-1),
        F.normalize(h.float(), dim=-1),
        dim=-1
    ).mean().item()

    # rmse
    rmse = (h_hat.float() - h.float()).pow(2).mean().sqrt().item()

    # max abs diff (sanity)
    mad = (h_hat.float() - h.float()).abs().max().item()

    return {"cos_tok_mean": float(cos_tok), "rmse": float(rmse), "max_abs_diff": float(mad)}


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, default="vsearch/vdr-cross-modal")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--ridge", type=float, default=1e-2)
    ap.add_argument("--print_dims", action="store_true", help="print L,D,V and W stats")
    ap.add_argument("paths", nargs="+", help="image files / dirs / glob")
    args = ap.parse_args()

    vdr = Retriever.from_pretrained(args.ckpt).to(args.device).eval()
    enc = vdr.encoder_p

    images = gather_images(args.paths)
    if len(images) == 0:
        raise SystemExit("No images found.")

    # W: [V,D]
    W = enc.proj
    V, D = W.shape

    if args.print_dims:
        print(f"W shape: {tuple(W.shape)} (V={V}, D={D}) dtype={W.dtype} device={W.device}")
        # rough conditioning info (optional)
        # computing full svd on D=768 is OK; on some GPUs it can be heavy, so keep it light
        G = (W.float().t() @ W.float()).detach()
        # trace/norm as cheap proxies
        print(f"WtW trace={G.trace().item():.4e}, fro_norm={torch.linalg.norm(G).item():.4e}")

    print(f"Loaded {len(images)} images | ckpt={args.ckpt} | device={args.device} | ridge={args.ridge}")

    all_cos, all_rmse, all_mad = [], [], []

    for i, img_path in enumerate(images):
        # load -> x: [1,3,224,224]
        x = enc.load_image_file(img_path).to(args.device).type(enc.dtype)

        # h: [1,L,D]
        h = enc(x)
        N, L, D2 = h.shape
        assert D2 == D, f"D mismatch: h has {D2}, W has {D}"

        # lv: [1,L,V]
        lv = encode_linear(h, W)

        # h_hat: [1,L,D]
        h_hat = decode_linear(lv, W, ridge=args.ridge)

        m = metrics(h, h_hat)
        all_cos.append(m["cos_tok_mean"])
        all_rmse.append(m["rmse"])
        all_mad.append(m["max_abs_diff"])

        print(f"\n[{i+1}/{len(images)}] {img_path}")
        print(f"  dims: h={tuple(h.shape)} lv={tuple(lv.shape)} h_hat={tuple(h_hat.shape)}")
        print(f"  token_cos_mean={m['cos_tok_mean']:.6f}  rmse={m['rmse']:.6f}  max_abs_diff={m['max_abs_diff']:.6f}")

    print("\n=== Summary (mean over images) ===")
    print(f"token_cos_mean={sum(all_cos)/len(all_cos):.6f}")
    print(f"rmse={sum(all_rmse)/len(all_rmse):.6f}")
    print(f"max_abs_diff={sum(all_mad)/len(all_mad):.6f}")


if __name__ == "__main__":
    main()
