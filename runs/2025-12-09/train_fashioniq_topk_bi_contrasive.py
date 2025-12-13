#!/usr/bin/env python
# train_fashioniq_topk_contrastive.py
# FashionIQ: lexical Top-K + GatedComposer + Bi-directional InfoNCE
# + (安全な) caption cache 自動追記 + (正しい) 全ギャラリー検索で validation

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
TOPK       = 768

EXP_DIR = "./checkpoints/exp_fashioniq_topk"
Path(EXP_DIR).mkdir(parents=True, exist_ok=True)

CAPTION_BATCH_SIZE = 64  # caption feature 追記計算用


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
    with open(Path(EXP_DIR) / "config.json", "w") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)


def join_captions(caps: List[str]) -> str:
    """
    FashionIQ captions(2文) の結合ルールを **ここに固定**。
    KeyError の原因になりやすいので、抽出/学習/評価で統一する。
    例: ["is ...", "is ..."] -> "is .... is ...."
    """
    # 末尾のピリオド揺れを吸収しつつ ". " で結合
    cleaned = [c.strip().rstrip(".") for c in caps]
    return ". ".join(cleaned)


def load_triplets(split: str, dress_types: List[str]) -> List[dict]:
    """
    FashionIQ の triplets(json) をロード。
    train は cap.all.train.json があるが、ここでは dress_types を統一的に使う。
    """
    root = Path(FASHIONIQ_ROOT) / "captions"
    triplets: List[dict] = []

    if split == "train":
        # train は all があるならそれを優先（あなたの環境に合わせる）
        all_path = root / "cap.all.train.json"
        if all_path.exists():
            with open(all_path, "r") as f:
                triplets = json.load(f)
            return triplets

    # val/test/train のカテゴリ別
    for dt in dress_types:
        p = root / f"cap.{dt}.{split}.json"
        if not p.exists():
            continue
        with open(p, "r") as f:
            triplets.extend(json.load(f))
    return triplets


def load_caption_cache(npz_path: str) -> Dict[str, np.ndarray]:
    """
    npz: captions (M,), embs (M,D)
    -> dict[caption] = emb
    """
    p = Path(npz_path)
    if not p.exists():
        return {}
    data = np.load(p, allow_pickle=True)
    captions = data["captions"]
    embs = data["embs"]
    return {str(c): embs[i] for i, c in enumerate(captions)}


def save_caption_cache(npz_path: str, cache: Dict[str, np.ndarray]):
    """
    dict -> npz（上書き保存）
    """
    captions = np.array(list(cache.keys()), dtype=object)
    embs = np.stack([cache[c] for c in captions], axis=0).astype(np.float32)
    np.savez_compressed(npz_path, captions=captions, embs=embs)
    print(f"[SAVE] Updated caption embeddings → {npz_path}")


@torch.no_grad()
def ensure_caption_features(npz_path: str, split: str, dress_types: List[str]):
    """
    split に必要な caption（結合版）を全列挙し、
    cache に無いものだけ Retriever で計算して npz に追記する。

    ★ DataLoader worker でやらない（ファイル競合を避ける）
    """
    triplets = load_triplets(split, dress_types)

    needed = set()
    for ann in triplets:
        # JSON は "captions": [.., ..]
        caps = ann["captions"]
        needed.add(join_captions(caps))

    cache = load_caption_cache(npz_path)
    missing = sorted(list(needed - set(cache.keys())))
    print(f"[CaptionCache] split={split} needed={len(needed)} cached={len(cache)} missing={len(missing)}")

    if len(missing) == 0:
        return

    # retriever
    ret = Retriever.from_pretrained("vsearch/vdr-cross-modal").to(DEVICE)
    ret.eval()
    for p in ret.parameters():
        p.requires_grad = False

    # infer dim
    dummy = ret.encoder_q.embed(["dummy"])
    dim = int(dummy.shape[-1])

    # compute missing in batches
    for i in tqdm(range(0, len(missing), CAPTION_BATCH_SIZE), desc=f"[BuildCaption] {split}"):
        batch = missing[i:i + CAPTION_BATCH_SIZE]
        V = ret.encoder_q.embed(batch)  # (B, D)
        V = V.detach().cpu().numpy().astype(np.float32)
        for j, cap in enumerate(batch):
            cache[cap] = V[j]

    save_caption_cache(npz_path, cache)


# ============================================================
# 2. Lexical Top-K Projector
# ============================================================

class LexicalTopKProjector:
    """
    27623-d lexical vector -> top-K だけ残す。
    """
    def __init__(self, k: int = 768):
        self.k = k

    def __call__(self, V: torch.Tensor) -> torch.Tensor:
        # V: (B,D)
        vals, idx = torch.topk(V, k=self.k, dim=-1)
        out = torch.zeros_like(V)
        out.scatter_(1, idx, vals)
        return out


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

    def forward(self, V_r: torch.Tensor, V_m: torch.Tensor) -> torch.Tensor:
        h = self.down(torch.cat([V_r, V_m], dim=-1))
        g = self.sig(self.up(h))
        V_q = (1.0 - g) * V_r + g * V_m
        return F.normalize(V_q + 1e-6, dim=-1)


# ============================================================
# 4. Dataset Loader
# ============================================================

def load_dataset(split: str) -> FashionIQFeatureDataset:
    if split == "train":
        img, txt = IMG_FEAT_TRAIN, TXT_FEAT_TRAIN
    elif split == "val":
        img, txt = IMG_FEAT_VAL, TXT_FEAT_VAL
    else:
        raise ValueError("split must be 'train' or 'val'")

    caption_cache = load_caption_cache(txt)

    return FashionIQFeatureDataset(
        dataset_root=FASHIONIQ_ROOT,
        split=split,
        dress_types=DRESS_TYPES,
        img_feat_npz=img,
        txt_feat_npz=txt,
        caption_cache=caption_cache,   # ★ 必須（あなたの最新 Dataset 仕様）
        join_captions=True,            # ★ join_captions のルールと揃える前提
    )


# ============================================================
# 5. Validation: 全ギャラリー検索で Recall@K
# ============================================================

@torch.no_grad()
def build_gallery(img_feat_npz: str) -> Tuple[torch.Tensor, List[str]]:
    """
    gallery = split 内の全画像特徴（N,D）
    """
    data = np.load(img_feat_npz, allow_pickle=True)
    names = [str(x) for x in data["names"]]
    embs = torch.tensor(data["embs"], dtype=torch.float32, device=DEVICE)
    embs = F.normalize(embs, dim=-1)
    return embs, names


@torch.no_grad()
def compute_recall_full_gallery(
    model: nn.Module,
    projector: LexicalTopKProjector,
    val_loader: DataLoader,
    gallery_embs: torch.Tensor,
    gallery_names: List[str],
    k_values=(1, 5, 10),
) -> Dict[str, float]:
    """
    各 query について、gallery 全体から検索し、
    target_name が top-k に入るかで Recall@K を計算
    """
    model.eval()

    # name -> index
    name2idx = {n: i for i, n in enumerate(gallery_names)}

    hit = {k: 0 for k in k_values}
    total = 0

    for batch in tqdm(val_loader, desc="[Val-FullGallery]"):
        V_r = batch["V_r"].to(DEVICE)  # (B,D)
        V_m = batch["V_m"].to(DEVICE)  # (B,D)
        tgt_names = batch["tgt_name"]  # list[str] を想定

        V_q = model(V_r, V_m)
        V_q = projector(V_q)
        V_q = F.normalize(V_q + 1e-6, dim=-1)

        # gallery 側も topk を適用したいならここで適用（今回は query のみでも可）
        # ただし比較は同一空間が理想なので、両方に適用する
        G = projector(gallery_embs)
        G = F.normalize(G + 1e-6, dim=-1)

        sims = V_q @ G.T  # (B, N)

        max_k = max(k_values)
        topk_idx = sims.topk(max_k, dim=-1).indices  # (B, max_k)

        for i in range(len(tgt_names)):
            total += 1
            tname = str(tgt_names[i])
            if tname not in name2idx:
                continue
            gt = name2idx[tname]
            row = topk_idx[i]

            for k in k_values:
                if (row[:k] == gt).any().item():
                    hit[k] += 1

    recalls = {f"R@{k}": (hit[k] / max(total, 1)) for k in k_values}
    return recalls


# ============================================================
# 6. Training Loop（双方向 InfoNCE）
# ============================================================

def train():
    set_seed(42)
    save_config()

    # --- 重要: 学習前に caption cache を不足分だけ追記して整合を取る ---
    ensure_caption_features(TXT_FEAT_TRAIN, split="train", dress_types=DRESS_TYPES)
    ensure_caption_features(TXT_FEAT_VAL,   split="val",   dress_types=DRESS_TYPES)

    train_dataset = load_dataset("train")
    val_dataset   = load_dataset("val")

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    model = GatedComposer().to(DEVICE)
    projector = LexicalTopKProjector(k=TOPK)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    # gallery は val の全画像
    gallery_embs, gallery_names = build_gallery(IMG_FEAT_VAL)

    best_r10 = -1.0
    best_epoch = None
    loss_log = []

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0

        for batch in train_loader:
            V_r = batch["V_r"].to(DEVICE)
            V_m = batch["V_m"].to(DEVICE)
            V_t = batch["V_t"].to(DEVICE)

            V_q = model(V_r, V_m)

            V_q = projector(V_q)
            V_t = projector(V_t)

            V_q = F.normalize(V_q + 1e-6, dim=-1)
            V_t = F.normalize(V_t + 1e-6, dim=-1)

            logits = (V_q @ V_t.T) / TEMP
            labels = torch.arange(len(V_r), device=DEVICE)

            loss_i2t = F.cross_entropy(logits, labels)
            loss_t2i = F.cross_entropy(logits.T, labels)
            loss = 0.5 * (loss_i2t + loss_t2i)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            loss_log.append(loss.item())

        mean_loss = total_loss / max(len(train_loader), 1)
        print(f"[Epoch {epoch:03d}] Train Loss = {mean_loss:.4f}")

        # ---- Validation: 全ギャラリー検索で Recall@K ----
        recalls = compute_recall_full_gallery(
            model=model,
            projector=projector,
            val_loader=val_loader,
            gallery_embs=gallery_embs,
            gallery_names=gallery_names,
            k_values=(1, 5, 10),
        )
        print(f"[Epoch {epoch:03d}] Validation (Full Gallery): {recalls}")

        if recalls["R@10"] > best_r10:
            best_r10 = recalls["R@10"]
            best_epoch = epoch

            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "recalls": recalls,
                    "train_loss": mean_loss,
                    "config": {
                        "TOPK": TOPK,
                        "TEMP": TEMP,
                        "HIDDEN_DIM": HIDDEN_DIM,
                        "LR": LR,
                        "BATCH_SIZE": BATCH_SIZE,
                    }
                },
                Path(EXP_DIR) / "best_ckpt.pth",
            )
            print(f"*** New BEST at epoch {epoch}! R@10 = {best_r10:.4f} ***")

    np.save(Path(EXP_DIR) / "loss.npy", np.array(loss_log, dtype=np.float32))
    print(f"[Finished] BEST epoch = {best_epoch}, BEST R@10 = {best_r10:.4f}")


if __name__ == "__main__":
    train()
