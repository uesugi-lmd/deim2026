import torch
import numpy as np
import json
from pathlib import Path
from data.my_datasets import CIRRFeatureDataset
from main import GatedComposer   # 学習時と同じクラス

DEVICE = "cuda"

# ==========================
# Utility
# ==========================

def load_composer(model_path, hidden_dim=256):
    """Load trained GatedComposer from checkpoint or state_dict."""
    model = GatedComposer(hidden=hidden_dim).to(DEVICE)

    ckpt = torch.load(model_path, map_location=DEVICE)

    if "model_state" in ckpt:        # checkpoint 形式
        model.load_state_dict(ckpt["model_state"])
    else:                            # state_dict 形式
        model.load_state_dict(ckpt)

    model.eval()
    return model


def recall_at_k(ranks, target_index, k):
    return 1.0 if target_index in ranks[:k] else 0.0


# ==========================
# Validation main
# ==========================

def validate(
        composer_path="./checkpoints/exp001/epoch5.pth",
        img_feat_npz="./cirr_features/cirr_val_image_features.npz",
        txt_feat_npz="./cirr_features/cirr_val_text_embs.npz",
        cirr_root="/home/uesugi/research/dataset/raw/cirr",
        split="val",
        hidden_dim=256
):
    # ---------------------------------------
    # 1. Load model
    # ---------------------------------------
    model = load_composer(composer_path, hidden_dim)

    # ---------------------------------------
    # 2. Load dataset (feature-only)
    # ---------------------------------------
    dataset = CIRRFeatureDataset(
        cirr_root=cirr_root,
        split=split,
        img_feat_npz=img_feat_npz,
        txt_feat_npz=txt_feat_npz
    )

    # gallery images (N, D)
    img_embs = dataset.img_embs      # np.array
    img_embs = torch.tensor(img_embs).float().to(DEVICE)

    recalls1 = []
    recalls5 = []
    recalls10 = []
    recalls50 = []

    # ---------------------------------------
    # 3. Loop over validation queries
    # ---------------------------------------
    for i in range(len(dataset)):
        sample = dataset[i]

        V_r = sample["V_r"].unsqueeze(0).to(DEVICE)
        V_m = sample["V_m"].unsqueeze(0).to(DEVICE)
        target_name = sample["tgt_name"]

        # ---- Compose ----
        V_q = model(V_r, V_m)            # (1, D)

        # ---- Score all gallery images ----
        scores = (V_q @ img_embs.T).squeeze(0)   # (N,)
        ranks = torch.argsort(scores, descending=True).cpu().tolist()

        # ---- ground truth index ----
        gt_idx = dataset.name2idx[target_name]

        # ---- Recall@K ----
        recalls1.append(recall_at_k(ranks, gt_idx, 1))
        recalls5.append(recall_at_k(ranks, gt_idx, 5))
        recalls10.append(recall_at_k(ranks, gt_idx, 10))
        recalls50.append(recall_at_k(ranks, gt_idx, 50))

    print("=== Validation Result ===")
    print("Recall@1 =", np.mean(recalls1))
    print("Recall@5 =", np.mean(recalls5))
    print("Recall@10 =", np.mean(recalls10))
    print("Recall@10 =", np.mean(recalls50))

    return {
        "R1": np.mean(recalls1),
        "R5": np.mean(recalls5),
        "R10": np.mean(recalls10),
        "R50": np.mean(recalls50)
    }


if __name__ == "__main__":
    validate()
