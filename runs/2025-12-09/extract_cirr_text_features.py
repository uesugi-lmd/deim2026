#!/usr/bin/env python
# build_cirr_text_features.py

import json
from pathlib import Path
from typing import List

import numpy as np
import torch
from tqdm import tqdm
from ir import Retriever


# ============================================================
# Helper: compute embedding
# ============================================================
def compute_embedding(texts: List[str], retriever, device="cuda"):
    """
    retriever.encoder_q.embed() をバッチ化して呼ぶヘルパー
    """
    all_embs = []

    BATCH = 64
    with torch.no_grad():
        for i in range(0, len(texts), BATCH):
            batch = texts[i:i+BATCH]
            v = retriever.encoder_q.embed(batch)   # (B, D)
            all_embs.append(v.detach().cpu().numpy())

    return np.vstack(all_embs)   # (N, D)


# ============================================================
# Main
# ============================================================
def main():
    cirr_root = Path("/home/uesugi/research/dataset/raw/cirr")
    split = "train"

    out_path = Path(f"./cirr_features/cirr_{split}_text_embs.npz")
    out_path.parent.mkdir(exist_ok=True)

    # -------------------------------
    # Load retriever (VDR cross-modal)
    # -------------------------------
    device = "cuda"
    retriever = Retriever.from_pretrained("vsearch/vdr-cross-modal").to(device)
    retriever.eval()
    for p in retriever.parameters():
        p.requires_grad = False

    # -------------------------------
    # Load CIRR captions
    # -------------------------------
    with open(cirr_root / "cirr" / "captions" / f"cap.rc2.{split}.json") as f:
        triplets = json.load(f)

    captions = sorted({t["caption"].strip() for t in triplets})
    print(f"[CIRR] Num unique captions = {len(captions)}")

    # CIRR は single caption ----
    # ただし FashionIQ の multi-caption 互換のために
    # joined_caption も生成しておく（今回は同じ値）
    joined_captions = captions.copy()

    # -------------------------------
    # Compute embeddings
    # -------------------------------
    print("Computing embeddings for original captions...")
    embs_single = compute_embedding(captions, retriever, device=device)   # (N, D)

    print("Computing embeddings for joined captions...")
    embs_joined = compute_embedding(joined_captions, retriever, device=device)

    assert embs_single.shape == embs_joined.shape

    # -------------------------------
    # Save NPZ
    # -------------------------------
    np.savez_compressed(
        out_path,
        captions=np.array(captions, dtype=object),
        embs_single=embs_single.astype(np.float32),
        joined_captions=np.array(joined_captions, dtype=object),
        embs_joined=embs_joined.astype(np.float32),
    )

    print(f"[DONE] Saved CIRR text features → {out_path}")
    print(" shapes:")
    print("  captions         :", len(captions))
    print("  embs_single      :", embs_single.shape)
    print("  joined_captions  :", len(joined_captions))
    print("  embs_joined      :", embs_joined.shape)


if __name__ == "__main__":
    main()
