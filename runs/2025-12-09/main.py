import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import random, json, time, subprocess
from pathlib import Path

from data.my_datasets import CIRRFeatureDataset


# ============================================================
# 0. Utility
# ============================================================

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def get_git_hash():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
    except:
        return "unknown"


# ============================================================
# 1. Config
# ============================================================

DEVICE = "cuda"
IMAGE_FEAT_PATH = "./cirr_features/cirr_train_image_features.npz"
TEXT_FEAT_PATH  = "./cirr_features/cirr_train_text_embs.npz"
CIRR_ROOT       = "/home/uesugi/research/dataset/raw/cirr"

BATCH_SIZE      = 32
LR              = 1e-4
EPOCHS          = 100
TEMP            = 0.03
HIDDEN_DIM      = 256

EXP_DIR = "./checkpoints/exp001"


# ============================================================
# 2. Save config
# ============================================================

def save_config():
    Path(EXP_DIR).mkdir(parents=True, exist_ok=True)
    config = {
        "DEVICE": DEVICE,
        "BATCH_SIZE": BATCH_SIZE,
        "LR": LR,
        "EPOCHS": EPOCHS,
        "TEMP": TEMP,
        "HIDDEN_DIM": HIDDEN_DIM,
        "IMAGE_FEAT_PATH": IMAGE_FEAT_PATH,
        "TEXT_FEAT_PATH": TEXT_FEAT_PATH,
        "git_hash": get_git_hash(),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(f"{EXP_DIR}/config.json", "w") as f:
        json.dump(config, f, indent=4)


# ============================================================
# 3. Model
# ============================================================

class GatedComposer(nn.Module):
    def __init__(self, dim=27623, hidden=HIDDEN_DIM):
        super().__init__()

        # ---- 2-layer MLP ----
        self.down = nn.Sequential(
            nn.Linear(dim * 2, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )

        # ---- Gate projection ----
        self.up = nn.Linear(hidden, dim)
        self.sig = nn.Sigmoid()

        # ---- Gate bias trick: initial g ≈ 0.12 ----
        self.up.bias.data.fill_(-2.0)

    def forward(self, V_r, V_m):
        # (B, 55246) → hidden
        h = self.down(torch.cat([V_r, V_m], dim=-1))

        # Gate (B, dim)
        g = self.sig(self.up(h))

        # Blending
        V_q = (1 - g) * V_r + g * V_m

        # Normalize for contrastive training
        return F.normalize(V_q + 1e-6, dim=-1)



# ============================================================
# 4. Dataset loader
# ============================================================

def load_dataset():
    return CIRRFeatureDataset(
        cirr_root=CIRR_ROOT,
        split="train",
        img_feat_npz=IMAGE_FEAT_PATH,
        txt_feat_npz=TEXT_FEAT_PATH
    )


# ============================================================
# 5. Training Loop
# ============================================================

def train():
    set_seed(42)
    save_config()

    dataset = load_dataset()
    loader = DataLoader(dataset, batch_size=BATCH_SIZE,
                        shuffle=True, num_workers=4, pin_memory=True)

    model = GatedComposer().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    loss_log = []

    for epoch in range(EPOCHS):
        total_loss = 0

        for step, batch in enumerate(loader):

            V_r = batch["V_r"].to(DEVICE)
            V_m = batch["V_m"].to(DEVICE)
            V_t = batch["V_t"].to(DEVICE)

            V_q = model(V_r, V_m)

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
            loss_log.append(loss.item())

            if (step + 1) % 100 == 0:
                print(f"[Epoch {epoch+1} Step {step+1}] Loss = {total_loss/(step+1):.4f}")

        mean_loss = total_loss / len(loader)
        print(f"=== Epoch {epoch+1} Completed | Mean Loss = {mean_loss:.4f} ===")

        # ============================================================
        #  保存（A）state_dict（最軽量）
        # ============================================================
        torch.save(model.state_dict(), f"{EXP_DIR}/epoch{epoch+1}.pth")

        # ============================================================
        #  保存（B）full model（アーキテクチャを含む完全保存）
        # ============================================================
        torch.save(model, f"{EXP_DIR}/epoch{epoch+1}_full.pth")

        # ============================================================
        #  保存（C）checkpoint（再現性が最高）
        # ============================================================
        ckpt = {
            "epoch": epoch + 1,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "config": {
                "DEVICE": DEVICE,
                "BATCH_SIZE": BATCH_SIZE,
                "LR": LR,
                "EPOCHS": EPOCHS,
                "TEMP": TEMP,
                "HIDDEN_DIM": HIDDEN_DIM,
                "IMAGE_FEAT_PATH": IMAGE_FEAT_PATH,
                "TEXT_FEAT_PATH": TEXT_FEAT_PATH,
            },
            "git_hash": get_git_hash(),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "mean_loss": mean_loss,
        }
        torch.save(ckpt, f"{EXP_DIR}/epoch{epoch+1}_ckpt.pth")

    np.save(f"{EXP_DIR}/loss.npy", np.array(loss_log))
    return model



# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    model = train()
