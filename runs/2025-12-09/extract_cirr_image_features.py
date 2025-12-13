import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

from ir import Retriever


def main():
    cirr_root = Path("/home/uesugi/research/dataset/raw/cirr")
    split = "train"
    out_path = Path(f"./cirr_features/cirr_{split}_image_features.npz")
    out_path.parent.mkdir(exist_ok=True)

    # --- CIRR の image_splits から image_name を列挙 ---
    with open(cirr_root / "cirr" / "image_splits" / f"split.rc2.{split}.json") as f:
        name_to_relpath = json.load(f)

    image_names = sorted(name_to_relpath.keys())
    print("num images:", len(image_names))

    # --- retriever & preprocess ---
    device = "cuda"
    retriever = Retriever.from_pretrained("vsearch/vdr-cross-modal").to(device)
    retriever.eval()

    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5),
                             std=(0.5, 0.5, 0.5)),
    ])

    # VDR の次元数（既に分かっていれば直書きでもOK）
    with torch.no_grad():
        dummy = torch.randn(1, 3, 224, 224).to(device)
        dim = retriever.encoder_p.embed(dummy).shape[-1]

    embs = np.zeros((len(image_names), dim), dtype=np.float32)

    # --- 埋め込み計算 ---
    for i, name in enumerate(tqdm(image_names)):
        rel = name_to_relpath[name]
        img_path = cirr_root / "images" / rel

        img = Image.open(img_path).convert("RGB")
        img = preprocess(img).unsqueeze(0).to(device)

        with torch.no_grad():
            v = retriever.encoder_p.embed(img)  # (1, dim)

        embs[i] = v.squeeze(0).cpu().numpy()

    # names と embs を1つの npz にまとめる
    np.savez_compressed(out_path, names=np.array(image_names), embs=embs)
    print("saved to", out_path)


if __name__ == "__main__":
    main()