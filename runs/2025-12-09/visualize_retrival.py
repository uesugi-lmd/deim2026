# visualize_cirr_retrieval.py

import os
import json
from pathlib import Path
from typing import List, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image

from data.my_datasets import CIRRFeatureDataset


# ============================================================
# Global Settings
# ============================================================

DEVICE = "cuda"
DIM_LEXICAL = 27623
HIDDEN_DIM = 256

CIRR_ROOT = "/home/uesugi/research/dataset/raw/cirr"
SPLIT = "train"   # or "val"

if SPLIT == "train":
    IMG_FEAT = "./cirr_features/cirr_train_image_features.npz"
    TXT_FEAT = "./cirr_features/cirr_train_text_embs.npz"
else:
    IMG_FEAT = "./cirr_features/cirr_val_image_features.npz"
    TXT_FEAT = "./cirr_features/cirr_val_text_embs.npz"

CKPT_PATH = "./checkpoints/exp001/epoch3.pth"
SAVE_DIR = Path("./retrieval_vis")
SAVE_DIR.mkdir(exist_ok=True)

SAMPLE_IDS = [0, 10, 100, 500, 1000]


# ============================================================
# Model Definition
# ============================================================

class GatedComposer(nn.Module):
    """Learned gating between reference V_r and modification V_m."""

    def __init__(self, dim: int = DIM_LEXICAL, hidden: int = HIDDEN_DIM):
        super().__init__()

        self.down = nn.Sequential(
            nn.Linear(dim * 2, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.up = nn.Linear(hidden, dim)
        self.sig = nn.Sigmoid()

        # bias initialized small → model starts close to V_r
        with torch.no_grad():
            self.up.bias.fill_(-2.0)

    def forward(self, V_r: torch.Tensor, V_m: torch.Tensor) -> torch.Tensor:
        h = self.down(torch.cat([V_r, V_m], dim=-1))
        g = self.sig(self.up(h))
        V_q = (1.0 - g) * V_r + g * V_m
        return F.normalize(V_q + 1e-6, dim=-1)


def load_model(path: str) -> nn.Module:
    model = GatedComposer().to(DEVICE)
    state = torch.load(path, map_location=DEVICE)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


# ============================================================
# Image Utility Functions
# ============================================================

def load_image_paths(cirr_root: str, split: str) -> Dict[str, str]:
    """Load mapping from image name → relative file path."""
    f = Path(cirr_root) / "cirr" / "image_splits" / f"split.rc2.{split}.json"
    return json.load(open(f))


def open_image(path: Path, size=(224, 224)) -> Image.Image:
    """Load image with center crop → resize."""
    img = Image.open(path).convert("RGB")
    w, h = img.size
    side = min(w, h)
    left = (w - side) // 2
    top = (h - side) // 2
    img = img.crop((left, top, left + side, top + side))
    img = img.resize(size, Image.BILINEAR)
    return img


# ============================================================
# Retrieval
# ============================================================

def compute_topk(
    V_q: torch.Tensor,
    img_feat: torch.Tensor,
    img_names: List[str],
    topk: int = 5
) -> List[str]:
    """Cosine similarity based top-k retrieval."""
    V_q = F.normalize(V_q, dim=-1)
    scores = img_feat @ V_q.unsqueeze(1)     # (N, 1)
    idx = scores.squeeze(1).topk(topk).indices.tolist()
    return [img_names[i] for i in idx]


# ============================================================
# Visualization
# ============================================================

def visualize_query(
    ref_img: Path,
    caption: str,
    tgt_img: Path,
    retrieved: List[Path],
    save_path: Path
):
    """2×4 layout visualization for CIRR retrieval result."""

    fig = plt.figure(figsize=(16, 8))

    # --- Row 1 ---
    ax = fig.add_subplot(2, 4, 1)
    ax.imshow(open_image(ref_img))
    ax.set_title("Reference")
    ax.axis("off")

    ax = fig.add_subplot(2, 4, 2)
    ax.text(0.05, 0.5, caption, fontsize=12, wrap=True)
    ax.set_title("Caption")
    ax.axis("off")

    ax = fig.add_subplot(2, 4, 3)
    ax.imshow(open_image(tgt_img))
    ax.set_title("GT Target")
    ax.axis("off")

    ax = fig.add_subplot(2, 4, 4)
    ax.imshow(open_image(retrieved[0]))
    ax.set_title("Rank 1")
    ax.axis("off")

    # --- Row 2: rank 2-5 ---
    for i in range(4):
        ax = fig.add_subplot(2, 4, 5 + i)
        ax.imshow(open_image(retrieved[i+1]))
        ax.set_title(f"Rank {i+2}")
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close(fig)


# ============================================================
# Main Retrieval Pipeline
# ============================================================

def main():

    # 1) Load model
    model = load_model(CKPT_PATH)

    # 2) Load feature dataset
    dataset = CIRRFeatureDataset(
        cirr_root=CIRR_ROOT,
        split=SPLIT,
        img_feat_npz=IMG_FEAT,
        txt_feat_npz=TXT_FEAT,
    )

    # 3) Prepare image feature matrix
    img_names = dataset.img_names.tolist()
    img_feat = torch.tensor(dataset.img_embs, dtype=torch.float32).to(DEVICE)
    img_feat = F.normalize(img_feat, dim=-1)

    # 4) Name → path mapping
    img_path_map = load_image_paths(CIRR_ROOT, split=SPLIT)

    # 5) Visualize selected samples
    for idx in SAMPLE_IDS:

        if idx >= len(dataset):
            print(f"[Skip] idx={idx} out of range.")
            continue

        item = dataset[idx]

        V_r = item["V_r"].unsqueeze(0).to(DEVICE)
        V_m = item["V_m"].unsqueeze(0).to(DEVICE)
        V_q = model(V_r, V_m).squeeze(0)

        ref_name = item["ref_name"]
        tgt_name = item["tgt_name"]
        caption = item["caption"]

        top5 = compute_topk(V_q, img_feat, img_names, topk=5)

        ref_path = Path(CIRR_ROOT) / "images" / img_path_map[ref_name]
        tgt_path = Path(CIRR_ROOT) / "images" /img_path_map[tgt_name]
        retrieved_paths = [Path(CIRR_ROOT) / "images" / img_path_map[n] for n in top5]

        print(f"\n=== Query idx={idx} ===")
        print("Caption:", caption)
        print("Reference:", ref_name)
        print("GT Target:", tgt_name)
        print("Top-5:", top5)

        save_path = SAVE_DIR / f"{SPLIT}_query_{idx}.png"
        visualize_query(
            ref_img=ref_path,
            caption=caption,
            tgt_img=tgt_path,
            retrieved=retrieved_paths,
            save_path=save_path,
        )


if __name__ == "__main__":
    main()
