# src/deim2026/image_db/cli.py

import argparse
from pathlib import Path
import sys

from .io import load_annotations
from .builder import embed_images, save_npz_and_meta, build_faiss
from ir import Retriever  # あなたのリポジトリ構成に合わせてこの import は調整してください


def run(
    image_json: Path,
    root_dir: Path,
    out_dir: Path,
    model_name: str = "vsearch/vdr-cross-modal",
    device: str = "cuda",
    batch_size: int = 1,
):
    """ビルド処理のメイン関数。pytest などからも直接呼び出せる。"""

    # Retriever ロード
    print(f"[INFO] Loading model: {model_name}")
    retriever = Retriever.from_pretrained(model_name)
    retriever = retriever.to(device)

    # アノテーション読み込み
    print(f"[INFO] Loading annotations from {image_json}")
    items = load_annotations(image_json)

    # 埋め込み
    print(f"[INFO] Embedding images (batch={batch_size}) ...")
    result = embed_images(retriever, items, root_dir, batch_size=batch_size)

    embeddings = result["embeddings"]
    ids = result["ids"]
    meta = result["meta"]

    # 保存
    print(f"[INFO] Saving vectors to {out_dir}")
    save_npz_and_meta(out_dir, embeddings, ids, meta)

    # FAISS インデックス作成（失敗しても継続）
    try:
        print("[INFO] Building FAISS index...")
        build_faiss(out_dir, embeddings)
    except Exception as e:
        print(f"[WARN] Could not build FAISS: {e}")

    print("[OK] Database build completed.")
    return {
        "embeddings": embeddings,
        "ids": ids,
        "meta": meta,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description="Build image embedding database")

    parser.add_argument("--image-json", type=Path, required=True,
                        help="image annotations JSON (list or dict format).")
    parser.add_argument("--root-dir", type=Path, required=True,
                        help="Images root directory (split/dev etc).")
    parser.add_argument("--out-dir", type=Path, required=True,
                        help="Output directory for vectors.npz/meta.jsonl/stats.json")
    parser.add_argument("--model", type=str, default="vsearch/vdr-cross-modal",
                        help="Retriever model name (default: vdr-cross-modal)")
    parser.add_argument("--device", type=str, default="cuda",
                        help="cuda / cpu")
    parser.add_argument("--batch-size", type=int, default=1,
                        help="Batch size (future use)")

    args = parser.parse_args(argv)

    return run(
        image_json=args.image_json,
        root_dir=args.root_dir,
        out_dir=args.out_dir,
        model_name=args.model,
        device=args.device,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
