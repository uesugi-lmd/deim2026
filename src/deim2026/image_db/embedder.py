# src/deim2026/image_db/embedder.py
from pathlib import Path
from typing import Any, Dict, List
import numpy as np
import torch

from .io import resolve_image_path

def embed_images(retriever, items, root_dir: Path):
    embeddings = []
    ids = []
    meta = []

    with torch.inference_mode():
        for ann in items:
            _id, img_path = resolve_image_path(ann, root_dir)
            emb = retriever.encoder_p.embed(str(img_path))

            if isinstance(emb, torch.Tensor):
                emb = emb.detach().cpu().float().numpy()

            embeddings.append(emb.reshape(-1))
            ids.append(_id)
            meta.append({"id": _id, "path": str(img_path)})

    return {
        "embeddings": np.array(embeddings, dtype=np.float32),
        "ids": np.array(ids),
        "meta": meta,
    }
