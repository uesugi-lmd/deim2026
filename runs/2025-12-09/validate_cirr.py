import torch
import numpy as np
from tqdm import tqdm
from pathlib import Path
import re

from main import GatedComposer
from data.my_datasets import CIRRFeatureDataset

DEVICE = "cuda"


# ============================================================
# Utility
# ============================================================

def load_ckpt(path, hidden_dim):
    """Load checkpoint that contains model_state only."""
    ckpt = torch.load(path, map_location=DEVICE)

    model = GatedComposer(hidden=hidden_dim).to(DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


def recall_at_k(ranks, target_idx, k):
    return 1.0 if target_idx in ranks[:k] else 0.0


def group_recall_at_k(ranks, group_idx, k):
    return 1.0 if any(g in ranks[:k] for g in group_idx) else 0.0


# ============================================================
# Evaluate a single checkpoint
# ============================================================

def evaluate_ckpt(
    ckpt_path,
    cirr_root,
    img_feat_npz,
    txt_feat_npz,
    hidden_dim=256,
    split="val"
):
    print(f"\n=== Evaluating {ckpt_path} ===")
    model = load_ckpt(ckpt_path, hidden_dim)

    dataset = CIRRFeatureDataset(
        cirr_root=cirr_root,
        split=split,
        img_feat_npz=img_feat_npz,
        txt_feat_npz=txt_feat_npz,
    )

    gallery = torch.tensor(dataset.img_embs).float().to(DEVICE)

    R1, R5, R10, R50 = [], [], [], []
    GR1, GR5, GR10 = [], [], []

    for idx in tqdm(range(len(dataset))):
        sample = dataset[idx]

        V_r = sample["V_r"].unsqueeze(0).to(DEVICE)
        V_m = sample["V_m"].unsqueeze(0).to(DEVICE)
        tgt = sample["tgt_name"]
        group = sample.get("group_members", [])

        V_q = model(V_r, V_m)
        scores = (V_q @ gallery.T).squeeze(0)
        ranks = torch.argsort(scores, descending=True).cpu().tolist()

        tgt_idx = dataset.name2idx[tgt]
        group_idx = [dataset.name2idx[g] for g in group if g in dataset.name2idx]

        # Hard target
        R1.append(recall_at_k(ranks, tgt_idx, 1))
        R5.append(recall_at_k(ranks, tgt_idx, 5))
        R10.append(recall_at_k(ranks, tgt_idx, 10))
        R50.append(recall_at_k(ranks, tgt_idx, 50))

        # Group recall
        if group_idx:
            GR1.append(group_recall_at_k(ranks, group_idx, 1))
            GR5.append(group_recall_at_k(ranks, group_idx, 5))
            GR10.append(group_recall_at_k(ranks, group_idx, 10))

    results = {
        "R1": np.mean(R1),
        "R5": np.mean(R5),
        "R10": np.mean(R10),
        "R50": np.mean(R50),
        "GR1": np.mean(GR1) if GR1 else None,
        "GR5": np.mean(GR5) if GR5 else None,
        "GR10": np.mean(GR10) if GR10 else None,
    }

    print("-- Results --")
    for k, v in results.items():
        print(f"{k}: {v:.4f}" if v is not None else f"{k}: N/A")

    return results


# ============================================================
# Evaluate ckpts at intervals (e.g., every 10 epochs)
# ============================================================

def validate_ckpt_range(
    ckpt_dir="./checkpoints/exp001",
    cirr_root="/home/uesugi/research/dataset/raw/cirr",
    img_feat_npz="./cirr_features/cirr_val_image_features.npz",
    txt_feat_npz="./cirr_features/cirr_val_text_embs.npz",
    hidden_dim=256,
    epoch_step=10,          # ← ここが刻み幅設定
    split="val",
    score_key="R1"          # ← 最良 ckpt を選ぶ指標
):
    ckpt_dir = Path(ckpt_dir)

    # 例：epoch10_ckpt.pth から epoch100_ckpt.pth を拾う
    pattern = re.compile(r"epoch(\d+)_ckpt\.pth")
    ckpt_files = []

    for f in ckpt_dir.glob("epoch*_ckpt.pth"):
        m = pattern.match(f.name)
        if m:
            epoch = int(m.group(1))
            if epoch % epoch_step == 0:  # ← 間引き条件
                ckpt_files.append((epoch, f))

    ckpt_files = sorted(ckpt_files)

    if not ckpt_files:
        print("No checkpoint files found.")
        return None

    best_epoch = None
    best_score = -1
    best_ckpt_path = None
    all_results = {}

    for epoch, path in ckpt_files:
        result = evaluate_ckpt(
            path, cirr_root, img_feat_npz, txt_feat_npz,
            hidden_dim, split
        )
        all_results[epoch] = result
        score = result[score_key]

        if score > best_score:
            best_score = score
            best_epoch = epoch
            best_ckpt_path = path

    print("\n==============================")
    print(" BEST CHECKPOINT (by", score_key, ")")
    print("  Epoch:", best_epoch)
    print("  Path:", best_ckpt_path)
    print("  Score:", best_score)
    print("==============================")

    return best_ckpt_path, all_results


if __name__ == "__main__":
    validate_ckpt_range(epoch_step=10)
