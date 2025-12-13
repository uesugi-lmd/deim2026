import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import time
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.my_datasets import CIRRFeatureDataset


# ============================================================
# 0. Config
# ============================================================

DEVICE = "cuda"

CIRR_ROOT = "/home/uesugi/research/dataset/raw/cirr"

IMAGE_FEAT_TRAIN = "./cirr_features/cirr_train_image_features.npz"
TEXT_FEAT_TRAIN  = "./cirr_features/cirr_train_text_embs.npz"

IMAGE_FEAT_VAL   = "./cirr_features/cirr_val_image_features.npz"
TEXT_FEAT_VAL    = "./cirr_features/cirr_val_text_embs.npz"

BATCH_SIZE = 32
LR         = 1e-4
EPOCHS     = 50
TEMP       = 0.07         # InfoNCE temperature
HIDDEN_DIM = 256          # GatedComposer hidden size
TOPK       = 768          # lexical の非ゼロ次元数を 768 に制限

EXP_DIR    = "./checkpoints/exp_topk_contrastive"
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
        "DEVICE": DEVICE,
        "CIRR_ROOT": CIRR_ROOT,
        "IMAGE_FEAT_TRAIN": IMAGE_FEAT_TRAIN,
        "TEXT_FEAT_TRAIN": TEXT_FEAT_TRAIN,
        "IMAGE_FEAT_VAL": IMAGE_FEAT_VAL,
        "TEXT_FEAT_VAL": TEXT_FEAT_VAL,
        "BATCH_SIZE": BATCH_SIZE,
        "LR": LR,
        "EPOCHS": EPOCHS,
        "TEMP": TEMP,
        "HIDDEN_DIM": HIDDEN_DIM,
        "TOPK": TOPK,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(Path(EXP_DIR) / "config.json", "w") as f:
        json.dump(cfg, f, indent=4)


# ============================================================
# 2. LexicalTopKProjector
#    （A案：まずはシンプル＋後から拡張しやすい形）
# ============================================================

class LexicalTopKProjector:
    """
    27623 次元の lexical ベクトルから上位 TOPK 成分だけを残すプロジェクタ。

    - いまは「値が大きい順」に top-k を取るだけの素直な実装
    - 将来的に:
        * stopwords 除外
        * POS による重み付け
        * caption 語彙の強制残し
      を入れるフックとして使えるようにしておく
    """

    def __init__(self, topk: int = 768):
        self.topk = topk

        # 将来的に vocab / pos_tags / stopwords などをここに持たせる
        self.vocab = None      # {token_id: word}
        self.pos_tags = None   # {word: POS}
        self.stopwords = None  # set([...])

    def _apply_basic_topk(self, V: torch.Tensor) -> torch.Tensor:
        """
        V: (B, D) lexical vector
        戻り値: (B, D) 上位 topk 以外は 0 にしたベクトル
        """
        # values, indices: (B, topk)
        vals, idx = torch.topk(V, k=self.topk, dim=-1)
        V_topk = torch.zeros_like(V)
        V_topk.scatter_(1, idx, vals)
        return V_topk

    def __call__(self, V: torch.Tensor, captions=None) -> torch.Tensor:
        """
        captions は将来的に
          - caption 語彙を必ず残す
          - 品詞情報を使う
        ために使う想定。
        いまは未使用で、単純な top-k のみ適用する。
        """
        # ★今のところは単純 top-k だけ
        return self._apply_basic_topk(V)


# ============================================================
# 3. Model: GatedComposer
# ============================================================

class GatedComposer(nn.Module):
    """
    V_r, V_m （27623 次元の lexical vector）から
    composed query V_q を生成するゲート付きコンポーザ。
    """

    def __init__(self, dim: int = 27623, hidden: int = HIDDEN_DIM):
        super().__init__()

        # 2-layer MLP で V_r, V_m を混ぜる
        self.down = nn.Sequential(
            nn.Linear(dim * 2, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )

        self.up = nn.Linear(hidden, dim)
        self.sig = nn.Sigmoid()

        # ゲート初期値を小さめにして V_r 寄りからスタート
        with torch.no_grad():
            self.up.bias.fill_(-2.0)

    def forward(self, V_r: torch.Tensor, V_m: torch.Tensor) -> torch.Tensor:
        """
        V_r, V_m: (B, 27623)
        戻り値: 正規化済み query ベクトル V_q （B, 27623）
        """
        h = self.down(torch.cat([V_r, V_m], dim=-1))  # (B, H)
        g = self.sig(self.up(h))                      # (B, D)
        V_q = (1.0 - g) * V_r + g * V_m               # (B, D)
        return F.normalize(V_q + 1e-6, dim=-1)


# ============================================================
# 4. Dataset loader
# ============================================================

def load_dataset(split: str):
    """
    CIRRFeatureDataset は
    - "V_r": reference image lexical
    - "V_m": modification text lexical
    - "V_t": target image lexical
    - もしあれば "caption"（文字列）
    を返す想定。
    """
    if split == "train":
        img_npz = IMAGE_FEAT_TRAIN
        txt_npz = TEXT_FEAT_TRAIN
    else:
        img_npz = IMAGE_FEAT_VAL
        txt_npz = TEXT_FEAT_VAL

    return CIRRFeatureDataset(
        cirr_root=CIRR_ROOT,
        split=split,
        img_feat_npz=img_npz,
        txt_feat_npz=txt_npz
    )


# ============================================================
# 5. Validation: R@1,5,10（top-k projector 付き）
# ============================================================

@torch.no_grad()
def compute_recall_at_k(model: nn.Module,
                        projector: LexicalTopKProjector,
                        val_loader: DataLoader,
                        k_list=(1, 5, 10)):

    model.eval()
    all_Q = []
    all_T = []

    for batch in tqdm(val_loader, desc="[Val]"):
        V_r = batch["V_r"].to(DEVICE)
        V_m = batch["V_m"].to(DEVICE)
        V_t = batch["V_t"].to(DEVICE)

        # composed query
        V_q = model(V_r, V_m)              # (B, D)

        # top-k マスクを適用
        captions = batch.get("caption", None)
        if captions is None:
            V_q_topk = projector(V_q)
        else:
            V_q_topk = projector(V_q, captions=captions)

        V_t_topk = projector(V_t)

        all_Q.append(V_q_topk)
        all_T.append(V_t_topk)

    Q = torch.cat(all_Q, dim=0)  # (N, D)
    T = torch.cat(all_T, dim=0)  # (N, D)

    sims = Q @ T.T               # (N, N)
    N = sims.size(0)
    indices = torch.arange(N, device=DEVICE)

    recalls = {}
    for k in k_list:
        topk_idx = sims.topk(k, dim=-1).indices  # (N, k)
        if k == 1:
            correct = (topk_idx[:, 0] == indices)
        else:
            correct = (topk_idx == indices.unsqueeze(1)).any(dim=1)
        recalls[f"R@{k}"] = correct.float().mean().item()

    return recalls


# ============================================================
# 6. Training loop（対象損失 / best epoch 保存）
# ============================================================

def train():
    set_seed(42)
    save_config()

    # Dataset / Loader
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

    # Model & Projector
    model = GatedComposer().to(DEVICE)
    projector = LexicalTopKProjector(topk=TOPK)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    best_r10 = -1.0
    best_epoch = None
    loss_log = []

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0

        for step, batch in enumerate(train_loader):
            V_r = batch["V_r"].to(DEVICE)
            V_m = batch["V_m"].to(DEVICE)
            V_t = batch["V_t"].to(DEVICE)

            # 1) composed query in lexical space
            V_q = model(V_r, V_m)          # (B, D)

            # 2) top-k mask
            captions = batch.get("caption", None)
            if captions is None:
                V_q_topk = projector(V_q)
            else:
                V_q_topk = projector(V_q, captions=captions)

            V_t_topk = projector(V_t)

            # 3) 対象損失（片方向 InfoNCE）
            #    正例: (q_i, t_i)
            #    負例: (q_i, t_j≠i) すべて
            logits = (V_q_topk @ V_t_topk.T) / TEMP   # (B, B)
            labels = torch.arange(logits.size(0), device=DEVICE)
            loss = F.cross_entropy(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            loss_log.append(loss.item())

            if (step + 1) % 100 == 0:
                print(f"[Epoch {epoch:03d} Step {step+1:05d}] "
                      f"Train Loss = {total_loss/(step+1):.4f}")

        mean_loss = total_loss / len(train_loader)
        print(f"=== Epoch {epoch:03d}/{EPOCHS} finished | "
              f"Train Loss = {mean_loss:.4f} ===")

        # ---- Validation ----
        recalls = compute_recall_at_k(model, projector, val_loader)
        print(f"[Epoch {epoch:03d}] Validation: {recalls}")

        # ---- Best checkpoint (R@10 ベース) ----
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
                Path(EXP_DIR) / "best_ckpt.pth",
            )
            print(f"*** New BEST model saved at epoch {epoch}! "
                  f"(R@10={best_r10:.4f}) ***")

    np.save(Path(EXP_DIR) / "loss.npy", np.array(loss_log))
    print(f"[Done] Training finished. BEST epoch = {best_epoch}, "
          f"BEST R@10 = {best_r10:.4f}")


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    train()
