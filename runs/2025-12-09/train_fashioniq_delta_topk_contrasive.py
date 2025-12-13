#!/usr/bin/env python
# FashionIQ: Delta-TopK (|Vq-Vr|) + GatedComposer + Bi-directional InfoNCE
# + caption cache auto-fill + full gallery validation

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import time
import json
import random
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from ir import Retriever
from data.my_datasets import FashionIQFeatureDataset


# ============================================================
# 0. Config
# ============================================================

DEVICE = "cuda"

FASHIONIQ_ROOT = "/home/uesugi/research/dataset/raw/fashioniq"
FEATURE_DIR = "./fashioniq_features"

IMG_FEAT_TRAIN = f"{FEATURE_DIR}/fashioniq_train_image_features.npz"
TXT_FEAT_TRAIN = f"{FEATURE_DIR}/fashioniq_train_text_embs.npz"
IMG_FEAT_VAL   = f"{FEATURE_DIR}/fashioniq_val_image_features.npz"
TXT_FEAT_VAL   = f"{FEATURE_DIR}/fashioniq_val_text_embs.npz"

DRESS_TYPES = ["dress", "shirt", "toptee"]

BATCH_SIZE = 32
LR         = 1e-4
EPOCHS     = 40
TEMP       = 0.07
HIDDEN_DIM = 256
TOPK       = 768        # ★ 差分ベースで残す次元数

CAPTION_BATCH_SIZE = 64

EXP_DIR = "./checkpoints/exp_fashioniq_delta_topk"
Path(EXP_DIR).mkdir(parents=True, exist_ok=True)


# ============================================================
# 1. Utilities
# ============================================================

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def save_config():
    cfg = {
        "TOPK": TOPK,
        "TEMP": TEMP,
        "LR": LR,
        "HIDDEN_DIM": HIDDEN_DIM,
        "BATCH_SIZE": BATCH_SIZE,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(Path(EXP_DIR) / "config.json", "w") as f:
        json.dump(cfg, f, indent=2)


def join_captions(caps: List[str]) -> str:
    """
    FashionIQ captions の結合ルールを統一
    """
    cleaned = [c.strip().rstrip(".") for c in caps]
    return ". ".join(cleaned)


# ============================================================
# 2. Caption cache utilities
# ============================================================

def load_caption_cache(npz_path: str) -> Dict[str, np.ndarray]:
    if not Path(npz_path).exists():
        return {}
    data = np.load(npz_path, allow_pickle=True)
    return {str(c): data["embs"][i] for i, c in enumerate(data["captions"])}


def save_caption_cache(npz_path: str, cache: Dict[str, np.ndarray]):
    captions = np.array(list(cache.keys()), dtype=object)
    embs = np.stack([cache[c] for c in captions]).astype(np.float32)
    np.savez_compressed(npz_path, captions=captions, embs=embs)
    print(f"[SAVE] Updated caption embeddings → {npz_path}")


def load_triplets(split: str) -> List[dict]:
    root = Path(FASHIONIQ_ROOT) / "captions"
    if split == "train":
        p = root / "cap.all.train.json"
        with open(p) as f:
            return json.load(f)

    triplets = []
    for dt in DRESS_TYPES:
        p = root / f"cap.{dt}.{split}.json"
        if p.exists():
            with open(p) as f:
                triplets.extend(json.load(f))
    return triplets


@torch.no_grad()
def ensure_caption_features(npz_path: str, split: str):
    triplets = load_triplets(split)
    needed = {join_captions(t["captions"]) for t in triplets}

    cache = load_caption_cache(npz_path)
    missing = sorted(list(needed - set(cache.keys())))

    print(f"[CaptionCache] split={split} missing={len(missing)}")
    if not missing:
        return

    ret = Retriever.from_pretrained("vsearch/vdr-cross-modal").to(DEVICE)
    ret.eval()
    for p in ret.parameters():
        p.requires_grad = False

    for i in tqdm(range(0, len(missing), CAPTION_BATCH_SIZE)):
        batch = missing[i:i + CAPTION_BATCH_SIZE]
        V = ret.encoder_q.embed(batch).cpu().numpy()
        for j, cap in enumerate(batch):
            cache[cap] = V[j]

    save_caption_cache(npz_path, cache)


# ============================================================
# 3. Delta Top-K Projector
# ============================================================

class LexicalDeltaTopKProjector:
    """
    |V_q - V_r| が大きい次元を基準に top-K を残す
    """

    def __init__(self, k: int):
        self.k = k

    def __call__(self, V_q: torch.Tensor, V_r: torch.Tensor) -> torch.Tensor:
        delta = torch.abs(V_q - V_r)
        _, idx = torch.topk(delta, k=self.k, dim=-1)
        mask = torch.zeros_like(V_q)
        mask.scatter_(1, idx, 1.0)
        return V_q * mask


# ============================================================
# 4. Gated Composer（差分注入型）
# ============================================================

class GatedComposer(nn.Module):
    def __init__(self, dim: int = 27623, hidden: int = HIDDEN_DIM):
        super().__init__()
        self.down = nn.Sequential(
            nn.Linear(dim * 2, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.up = nn.Linear(hidden, dim)
        self.sig = nn.Sigmoid()
        with torch.no_grad():
            self.up.bias.fill_(-2.0)

    def forward(self, V_r, V_m):
        h = self.down(torch.cat([V_r, V_m], dim=-1))
        g = self.sig(self.up(h))
        V_q = V_r + g * V_m          # ★ 差分注入
        return F.normalize(V_q + 1e-6, dim=-1)


# ============================================================
# 5. Dataset loader
# ============================================================

def load_dataset(split: str) -> FashionIQFeatureDataset:
    img = IMG_FEAT_TRAIN if split == "train" else IMG_FEAT_VAL
    txt = TXT_FEAT_TRAIN if split == "train" else TXT_FEAT_VAL

    return FashionIQFeatureDataset(
        dataset_root=FASHIONIQ_ROOT,
        split=split,
        dress_types=DRESS_TYPES,
        img_feat_npz=img,
        txt_feat_npz=txt,
        caption_cache=load_caption_cache(txt),
        join_captions=True,
    )


# ============================================================
# 6. Full gallery validation
# ============================================================

@torch.no_grad()
def build_gallery(npz_path: str):
    data = np.load(npz_path, allow_pickle=True)
    names = [str(n) for n in data["names"]]
    embs = torch.tensor(data["embs"], device=DEVICE)
    return F.normalize(embs, dim=-1), names


@torch.no_grad()
def compute_recall_full_gallery(
    model, projector, loader, gallery_embs, gallery_names, k_values=(1,5,10)
):
    name2idx = {n: i for i, n in enumerate(gallery_names)}
    hit = {k: 0 for k in k_values}
    total = 0

    for batch in tqdm(loader, desc="[Val-FullGallery]"):
        V_r = batch["V_r"].to(DEVICE)
        V_m = batch["V_m"].to(DEVICE)
        tgt_names = batch["tgt_name"]

        V_q = model(V_r, V_m)
        V_q = projector(V_q, V_r)
        V_q = F.normalize(V_q, dim=-1)

        sims = V_q @ gallery_embs.T
        topk = sims.topk(max(k_values), dim=-1).indices

        for i, tname in enumerate(tgt_names):
            total += 1
            if tname not in name2idx:
                continue
            gt = name2idx[tname]
            for k in k_values:
                if (topk[i, :k] == gt).any():
                    hit[k] += 1

    return {f"R@{k}": hit[k] / total for k in k_values}


# ============================================================
# 7. Training loop
# ============================================================

def train():
    set_seed()
    save_config()

    ensure_caption_features(TXT_FEAT_TRAIN, "train")
    ensure_caption_features(TXT_FEAT_VAL, "val")

    train_ds = load_dataset("train")
    val_ds   = load_dataset("val")

    train_loader = DataLoader(train_ds, BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader   = DataLoader(val_ds,   BATCH_SIZE, shuffle=False, num_workers=4)

    model = GatedComposer().to(DEVICE)
    projector = LexicalDeltaTopKProjector(TOPK)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    gallery_embs, gallery_names = build_gallery(IMG_FEAT_VAL)

    best_r10 = -1.0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0

        for batch in train_loader:
            V_r = batch["V_r"].to(DEVICE)
            V_m = batch["V_m"].to(DEVICE)
            V_t = batch["V_t"].to(DEVICE)

            V_q = model(V_r, V_m)
            V_q = projector(V_q, V_r)

            V_q = F.normalize(V_q, dim=-1)
            V_t = F.normalize(V_t, dim=-1)

            logits = (V_q @ V_t.T) / TEMP
            labels = torch.arange(len(V_r), device=DEVICE)

            loss = 0.5 * (
                F.cross_entropy(logits, labels) +
                F.cross_entropy(logits.T, labels)
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"[Epoch {epoch:03d}] Train Loss = {total_loss/len(train_loader):.4f}")

        recalls = compute_recall_full_gallery(
            model, projector, val_loader, gallery_embs, gallery_names
        )
        print(f"[Epoch {epoch:03d}] Validation: {recalls}")

        if recalls["R@10"] > best_r10:
            best_r10 = recalls["R@10"]
            torch.save(model.state_dict(), Path(EXP_DIR) / "best_model.pth")
            print(f"*** New BEST R@10 = {best_r10:.4f} ***")


if __name__ == "__main__":
    train()
