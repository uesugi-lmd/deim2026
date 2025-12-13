import hashlib
from pathlib import Path
from typing import Any, Dict, List, Tuple
import json

import numpy as np
import torch

try:
    import faiss
    HAS_FAISS = True
except Exception:
    HAS_FAISS = False


# ============================
# Path handling & annotation load
# ============================

def md5_short(text: str, n: int = 16) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()[:n]


def resolve_image_path(ann: Dict[str, Any], root_dir: Path) -> Tuple[str, Path]:
    _id = None
    for key in ["id", "image_id", "img_id"]:
        if key in ann and ann[key]:
            _id = str(ann[key]).strip()
            break

    rel = None
    for key in ["relpath", "rel_path", "path", "image_path", "img_path"]:
        if key in ann and ann[key]:
            rel = str(ann[key]).strip()
            break

    if rel and rel.startswith("./"):
        rel = rel[2:]

    if rel:
        parts = Path(rel).parts
        if parts and parts[0] == root_dir.name:
            abs_path = (root_dir.parent / rel).resolve()
        else:
            abs_path = (root_dir / rel).resolve()
    else:
        raise FileNotFoundError(f"No path field in annotation: {ann.keys()}")

    if not _id:
        _id = md5_short(str(abs_path))

    return _id, abs_path


# ============================
# Embedding
# ============================

def embed_images(
    retriever: "Retriever",
    items: List[Dict[str, Any]],
    root_dir: Path,
    batch_size: int = 1,
) -> Dict[str, Any]:

    vecs = []
    ids = []
    meta = []
    seen = set()

    with torch.inference_mode():
        for idx, ann in enumerate(items):
            try:
                _id, img_path = resolve_image_path(ann, root_dir)
            except Exception as e:
                print(f"[WARN] skip annotation: {e}")
                continue

            if _id in seen:
                continue
            seen.add(_id)

            emb = retriever.encoder_p.embed(str(img_path))
            if isinstance(emb, torch.Tensor):
                emb = emb.detach().cpu().float().numpy()
            emb = np.asarray(emb, dtype=np.float32).reshape(-1)

            vecs.append(emb)
            ids.append(_id)
            meta.append({
                "id": _id,
                "path": str(img_path),
                "index": len(ids) - 1
            })

            if (idx + 1) % 100 == 0:
                print(f"[{idx+1}/{len(items)}] embedded")

    if not vecs:
        embeddings = np.empty((0, 0), dtype=np.float32)
        ids_np = np.empty((0,), dtype="U16")
    else:
        embeddings = np.stack(vecs, axis=0).astype(np.float32)
        ids_np = np.array(ids, dtype="U16")

    return {"embeddings": embeddings, "ids": ids_np, "meta": meta}


# ============================
# Saving
# ============================

def save_npz_and_meta(out_dir: Path, embeddings: np.ndarray, ids: np.ndarray, meta: List[Dict[str, Any]]):
    out_dir.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(out_dir / "vectors.npz", embeddings=embeddings, ids=ids)

    with open(out_dir / "meta.jsonl", "w", encoding="utf-8") as f:
        for m in meta:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")

    stats = {
        "num_vectors": int(embeddings.shape[0]),
        "dim": int(embeddings.shape[1]) if embeddings.size else 0,
    }

    with open(out_dir / "stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    print(f"[OK] Saved vectors.npz, meta.jsonl, stats.json → {out_dir}")


# ============================
# FAISS
# ============================

def build_faiss(out_dir: Path, embeddings: np.ndarray):
    if not HAS_FAISS:
        print("[INFO] FAISS not available. skip.")
        return
    if embeddings.size == 0:
        print("[INFO] No embeddings. skip.")
        return

    dim = embeddings.shape[1]
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-12
    normed = embeddings / norms

    index = faiss.IndexFlatIP(dim)
    index.add(normed.astype(np.float32))

    faiss.write_index(index, str(out_dir / "faiss.index"))
    print(f"[OK] Saved faiss.index (IndexFlatIP) dim={dim}, n={index.ntotal}")
