#!/usr/bin/env python
# extract_fashioniq_image_features.py

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import json
from pathlib import Path

import numpy as np
from tqdm import tqdm
from PIL import Image

import torch
from torchvision import transforms

from ir import Retriever


# ============================================================
# Config
# ============================================================

DEVICE = "cuda"

FASHIONIQ_ROOT = Path("/home/uesugi/research/dataset/raw/fashioniq")

# "train", "val", "test" を切り替えて実行
SPLIT = "val"     # ★ 必要に応じて変更（train / val / test）

# 出力先
OUT_DIR = Path("./fashioniq_features")
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_NPZ = OUT_DIR / f"fashioniq_{SPLIT}_image_features.npz"


# FashionIQ categories
CATEGORIES = ["dress", "shirt", "toptee"]


# ============================================================
# Util: load image IDs from FashionIQ captions
# ============================================================

def load_image_ids_from_fashioniq(split: str):
    """
    FashionIQ の split (train/val/test) に対応する caption JSON を読み込み，
    candidate / target に登場するすべての image ID (ASIN) を集める。
    
    train のみ特別に cap.all.train.json があるが，
    val/test はカテゴリ別 JSON のみ。
    """

    caption_root = FASHIONIQ_ROOT / "captions"
    image_ids = set()

    if split == "train":
        json_path = caption_root / "cap.all.train.json"
        print(f"[LOAD] {json_path}")
        with open(json_path, "r") as f:
            anns = json.load(f)

        for ann in anns:
            image_ids.add(ann["candidate"])
            image_ids.add(ann["target"])

    else:
        # val/test の場合は 3カテゴリをまとめる
        for cat in CATEGORIES:
            json_path = caption_root / f"cap.{cat}.{split}.json"
            print(f"[LOAD] {json_path}")
            if not json_path.exists():
                print(f"[WARN] {json_path} not found. Skip.")
                continue

            with open(json_path, "r") as f:
                anns = json.load(f)

            for ann in anns:
                image_ids.add(ann["candidate"])
                if split != "test":  # test split は target なし
                    if "target" in ann:
                        image_ids.add(ann["target"])

    image_ids = sorted(list(image_ids))
    print(f"[INFO] Found {len(image_ids)} unique image IDs for split={split}")
    return image_ids


# ============================================================
# Preprocess / Loader
# ============================================================

def build_preprocess():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5),
                             std=(0.5, 0.5, 0.5)),
    ])


def load_image_tensor(img_path: Path, preprocess):
    img = Image.open(img_path).convert("RGB")
    return preprocess(img).unsqueeze(0)   # (1,3,224,224)


# ============================================================
# Main
# ============================================================

def main():
    # 1) 対象 split の画像IDを取得
    image_ids = load_image_ids_from_fashioniq(SPLIT)
    if len(image_ids) == 0:
        print("[ERROR] No image IDs found.")
        return

    # 2) Retriever ロード
    ret = Retriever.from_pretrained("vsearch/vdr-cross-modal").to(DEVICE)
    ret.eval()
    for p in ret.parameters():
        p.requires_grad = False

    preprocess = build_preprocess()

    names = []
    embs = []

    # 3) 各画像IDに対して特徴量抽出
    print(f"Extracting image features for split={SPLIT} ...")
    with torch.no_grad():
        for img_id in tqdm(image_ids):
            # FashionIQ は .jpg の場合も .png の場合もある → 両方チェック
            for ext in ["jpg", "png", "jpeg"]:
                img_file = FASHIONIQ_ROOT / "images" / f"{img_id}.{ext}"
                if img_file.exists():
                    break

            if not img_file.exists():
                print(f"[WARN] Image not found: {img_id}")
                continue

            img_tensor = load_image_tensor(img_file, preprocess).to(DEVICE)
            V = ret.encoder_p.embed(img_tensor)[0].cpu().numpy()

            names.append(img_id)
            embs.append(V)

    # 4) 保存
    names = np.array(names, dtype=object)
    embs = np.stack(embs, axis=0)  # (N, 27623)

    np.savez_compressed(OUT_NPZ, names=names, embs=embs)

    print(f"[DONE] Saved {len(names)} image features → {OUT_NPZ}")
    print("names.shape:", names.shape, "embs.shape:", embs.shape)


if __name__ == "__main__":
    main()
