#!/usr/bin/env python
# extract_fashioniq_text_features.py

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import json
from pathlib import Path

import numpy as np
from tqdm import tqdm

import torch
from ir import Retriever


# ============================================================
# Config
# ============================================================

DEVICE = "cuda"

FASHIONIQ_ROOT = Path("/home/uesugi/research/dataset/raw/fashioniq")

# "train", "val", "test" を切り替えて実行
SPLIT = "train"   # ★変更して使う

OUT_DIR = Path("./fashioniq_features")
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_NPZ = OUT_DIR / f"fashioniq_{SPLIT}_text_embs.npz"

BATCH_SIZE = 64

CATEGORIES = ["dress", "shirt", "toptee"]


# ============================================================
# Utils
# ============================================================

def load_all_captions_from_fashioniq(split: str):
    """
    FashionIQ の split（train/val/test）に応じてキャプションを読み込み、
    すべての caption を「1文として」抽出し、重複なしで返す。
    
    例： ["is solid black", "with straps"] → "is solid black with straps"
    """

    caption_root = FASHIONIQ_ROOT / "captions"
    captions = set()

    if split == "train":
        # cap.all.train.json が統合版
        json_path = caption_root / "cap.all.train.json"
        print(f"[LOAD] {json_path}")

        with open(json_path, "r") as f:
            anns = json.load(f)

        for ann in anns:
            merged_text = " ".join([c.strip() for c in ann["captions"]])
            captions.add(merged_text)

    else:
        # val/test → カテゴリ別JSONをマージ
        for cat in CATEGORIES:
            json_path = caption_root / f"cap.{cat}.{split}.json"

            if not json_path.exists():
                print(f"[WARN] Missing: {json_path}")
                continue

            print(f"[LOAD] {json_path}")
            with open(json_path, "r") as f:
                anns = json.load(f)

            for ann in anns:
                merged_text = " ".join([c.strip() for c in ann["captions"]])
                captions.add(merged_text)

    captions = sorted(list(captions))
    print(f"[INFO] Found {len(captions)} unique merged captions for split={split}")
    return captions


# ============================================================
# Main
# ============================================================

def main():

    # 1) caption のユニークリスト
    captions = load_all_captions_from_fashioniq(SPLIT)
    if len(captions) == 0:
        print("[ERROR] No captions found.")
        return

    # 2) Retriever (lexical encoder) ロード
    ret = Retriever.from_pretrained("vsearch/vdr-cross-modal").to(DEVICE)
    ret.eval()
    for p in ret.parameters():
        p.requires_grad = False

    all_embs = []

    # 3) バッチで lexical embeddings を計算
    print(f"Extracting caption embeddings for split={SPLIT} ...")
    with torch.no_grad():
        for i in tqdm(range(0, len(captions), BATCH_SIZE)):
            batch = captions[i:i + BATCH_SIZE]
            V = ret.encoder_q.embed(batch)       # (B, 27623)
            all_embs.append(V.cpu().numpy())

    embs = np.concatenate(all_embs, axis=0)
    cap_arr = np.array(captions, dtype=object)

    # 4) 保存
    np.savez_compressed(OUT_NPZ, captions=cap_arr, embs=embs)

    print(f"[DONE] Saved caption features → {OUT_NPZ}")
    print("captions.shape:", cap_arr.shape, "embs.shape:", embs.shape)


if __name__ == "__main__":
    main()
