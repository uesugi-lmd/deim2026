#!/usr/bin/env python
# train_fashioniq_topk_contrastive.py
# FashionIQ + lexical Top-K + GatedComposer + InfoNCE + LazyCaptionEmbeddingCache

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import time
import json
import random
from pathlib import Path
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

# dataset
from data.my_datasets import FashionIQFeatureDataset

# retriever
from ir import Retriever

# lazy caption cache（新ファイル）
from data.lazy_caption_cache import LazyCaptionEmbeddingCache



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
TOPK       = 768

EXP_DIR = "./checkpoints/exp_fashioniq_topk"
Path(EXP_DIR).mkdir(parents=True, exist_ok=True)



# ============================================================
# 1. Utilities
# ============================================================

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def save_config():
    cfg = {
        "root": FASHIONIQ_ROOT,
        "dress_types": DRESS_TYPES,
        "img_train": IMG_FEAT_TRAIN,
        "txt_train": TXT_FEAT_TRAIN,
        "img_val": IMG_FEAT_VAL,
        "txt_val": TXT_FEAT_VAL,
        "batch": BATCH_SIZE,
        "lr": LR,
        "epochs": EPOCHS,
        "temp": TEMP,
        "hidden": HIDDEN_DIM,
        "topk": TOPK,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(f"{EXP_DIR}/config.json", "w") as f:
        json.dump(cfg, f, indent=4)



# ============================================================
# 2. Lexical Top-K Projector
# ============================================================

class LexicalTopKProjector:
    def __init__(self, k: int = 768):
        self.k = k

    def __call__(self, V: torch.Tensor, captions=None) -> torch.Tensor:
        vals, idx = torch.topk(V, k=self.k, dim=-1)
        V_out = torch.zeros_like(V)
        V_out.scatter_(1, idx, vals)
        return V_out



# ============================================================
# 3. Gated Composer
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
        V_q = (1 - g) * V_r + g * V_m
        return F.normalize(V_q + 1e-6, dim=-1)



# ============================================================
# 4. Dataset Loader
# ============================================================

def load_dataset(split: str, caption_cache) -> FashionIQFeatureDataset:
    if split == "train":
        img, txt = IMG_FEAT_TRAIN, TXT_FEAT_TRAIN
    elif split == "val":
        img, txt = IMG_FEAT_VAL, TXT_FEAT_VAL
    else:
        raise ValueError("split must be 'train' or 'val'")

    return FashionIQFeatureDataset(
        dataset_root=FASHIONIQ_ROOT,
        split=split,
        dress_types=DRESS_TYPES,
        img_feat_npz=img,
        txt_feat_npz=txt,
        caption_cache=caption_cache,   # ★ new
        join_captions=True,
    )



# ============================================================
# 5. Validation Recall@K
# ============================================================

@torch.no_grad()
def compute_recall(model, projector, val_loader, k_values=(1,5,10)):
    model.eval()

    Q_list, T_list = [], []

    for batch in tqdm(val_loader, desc="[Val]"):
        V_r = batch["V_r"].to(DEVICE)
        V_m = batch["V_m"].to(DEVICE)
        V_t = batch["V_t"].to(DEVICE)

        V_q = model(V_r, V_m)

        V_q = projector(V_q)
        V_t = projector(V_t)

        Q_list.append(V_q)
        T_list.append(V_t)

    Q = torch.cat(Q_list)
    T = torch.cat(T_list)

    sims = Q @ T.T
    N = sims.size(0)
    correct_idx = torch.arange(N, device=DEVICE)

    recalls = {}
    for k in k_values:
        topk_idx = sims.topk(k, dim=-1).indices
        if k == 1:
            ok = (topk_idx[:, 0] == correct_idx)
        else:
            ok = (topk_idx == correct_idx.unsqueeze(1)).any(dim=1)
        recalls[f"R@{k}"] = ok.float().mean().item()

    return recalls



# ============================================================
# 6. Training Loop
# ============================================================

def train():
    set_seed(42)
    save_config()

    # ------------------------------
    # Load retriever (lexical encoder)
    # ------------------------------
    retriever = Retriever.from_pretrained("vsearch/vdr-cross-modal").to(DEVICE)
    retriever.eval()
    for p in retriever.parameters():
        p.requires_grad = False

    # ------------------------------
    # Lazy caption cache (train/val)
    # ------------------------------
    caption_cache_train = LazyCaptionEmbeddingCache(TXT_FEAT_TRAIN, retriever)
    caption_cache_val   = LazyCaptionEmbeddingCache(TXT_FEAT_VAL, retriever)

    # ------------------------------
    # Datasets
    # ------------------------------
    train_dataset = load_dataset("train", caption_cache_train)
    val_dataset   = load_dataset("val", caption_cache_val)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                              shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE,
                            shuffle=False, num_workers=4, pin_memory=True)

    # ------------------------------
    # Models
    # ------------------------------
    model = GatedComposer().to(DEVICE)
    projector = LexicalTopKProjector(TOPK)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    best_r10 = -1
    best_epoch = None
    loss_log = []

    # ------------------------------
    # Training starts
    # ------------------------------
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0

        for batch in train_loader:

            V_r = batch["V_r"].to(DEVICE)
            V_m = batch["V_m"].to(DEVICE)
            V_t = batch["V_t"].to(DEVICE)

            V_q = model(V_r, V_m)

            V_q = projector(V_q)
            V_t = projector(V_t)

            logits = (V_q @ V_t.T) / TEMP
            labels = torch.arange(len(V_r), device=DEVICE)

            loss = F.cross_entropy(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            loss_log.append(loss.item())

        mean_loss = total_loss / len(train_loader)
        print(f"[Epoch {epoch}] Train Loss = {mean_loss:.4f}")

        # ------------------------------
        # Validation
        # ------------------------------
        recalls = compute_recall(model, projector, val_loader)
        print(f"[Epoch {epoch}] Validation = {recalls}")

        # ------------------------------
        # Best checkpoint
        # ------------------------------
        if recalls["R@10"] > best_r10:
            best_r10 = recalls["R@10"]
            best_epoch = epoch

            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "recalls": recalls,
                    "train_loss": mean_loss,
                },
                Path(EXP_DIR) / "best_ckpt.pth"
            )

            print(f"*** New BEST! epoch={epoch}, R@10={best_r10:.4f} ***")

    # ------------------------------
    # Save caption caches (updated npz)
    # ------------------------------
    caption_cache_train.save()
    caption_cache_val.save()

    np.save(Path(EXP_DIR) / "loss.npy", np.array(loss_log))
    print(f"[Finished] BEST epoch = {best_epoch}, BEST R@10 = {best_r10:.4f}")


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    train()
