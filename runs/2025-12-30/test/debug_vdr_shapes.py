#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from collections import OrderedDict
from typing import Any, Dict, Tuple, List

import torch
from ir import Retriever


def _shape(x: Any):
    if isinstance(x, torch.Tensor):
        return tuple(x.shape)
    if isinstance(x, (list, tuple)):
        # 先頭だけ代表表示（長いと見づらいので）
        return [_shape(x[0])] if len(x) > 0 else []
    if isinstance(x, dict):
        # input_ids などが入ることが多いので主要キーだけ
        keys = list(x.keys())
        out = {k: _shape(x[k]) for k in keys[:6]}
        if len(keys) > 6:
            out["..."] = f"+{len(keys)-6} keys"
        return out
    return type(x).__name__


def register_shape_hooks(model: torch.nn.Module, only_leaf: bool = True):
    """
    model内のmoduleにforward hookを貼ってshapeを収集する．
    only_leaf=True だと子を持たないleaf moduleのみに貼る（ログが過剰になりにくい）．
    """
    records: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()
    handles = []

    for name, m in model.named_modules():
        if name == "":
            continue
        if only_leaf and any(True for _ in m.children()):
            continue

        def _make_hook(nm: str):
            def hook(mod, inputs, outputs):
                # inputsはタプルで来る
                rec = {
                    "module": mod.__class__.__name__,
                    "in": _shape(inputs),
                    "out": _shape(outputs),
                    "device": None,
                    "dtype": None,
                }
                # 出力tensorがあるなら dtype/device も拾う
                if isinstance(outputs, torch.Tensor):
                    rec["device"] = str(outputs.device)
                    rec["dtype"] = str(outputs.dtype)
                elif isinstance(outputs, (list, tuple)) and len(outputs) > 0 and isinstance(outputs[0], torch.Tensor):
                    rec["device"] = str(outputs[0].device)
                    rec["dtype"] = str(outputs[0].dtype)
                records[nm] = rec
            return hook

        handles.append(m.register_forward_hook(_make_hook(name)))

    return records, handles


def print_records(records: Dict[str, Dict[str, Any]], only_changes: bool = True):
    """
    only_changes=True: shapeが変化した点だけを出す（読みやすい）
    """
    prev_out = None
    for k, v in records.items():
        out_shape = v["out"]
        if only_changes:
            if out_shape == prev_out:
                continue
        prev_out = out_shape
        print(f"[{k}] {v['module']}")
        print(f"  in : {v['in']}")
        print(f"  out: {v['out']}  ({v['dtype']} @ {v['device']})")


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, default="vsearch/vdr-cross-modal")
    ap.add_argument("--image", type=str, required=True)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--topk", type=int, default=None, help="encoder_p.embed に topk を渡す（実装が対応していれば）")
    ap.add_argument("--only_leaf", action="store_true")
    ap.add_argument("--only_changes", action="store_true")
    args = ap.parse_args()

    vdr = Retriever.from_pretrained(args.ckpt).to(args.device).eval()
    print(vdr.encoder_p.embed)

    # 画像側 encoder_p に hook
    records, handles = register_shape_hooks(vdr.encoder_p, only_leaf=args.only_leaf)

    # 画像を通す（embed は内部で encode->forward を呼ぶ想定）
    if args.topk is None:
        emb = vdr.encoder_p.embed([args.image])
    else:
        # 実装によっては topk を受けないことがあるので try
        try:
            emb = vdr.encoder_p.embed([args.image], topk=args.topk)
        except TypeError:
            print("[WARN] encoder_p.embed は topk を受け取れませんでした．topkなしで実行します．")
            emb = vdr.encoder_p.embed([args.image])

    print("\n=== Final embedding ===")
    print("shape:", tuple(emb.shape), "dtype:", emb.dtype, "device:", emb.device)

    print("\n=== Intermediate shapes (encoder_p) ===")
    print_records(records, only_changes=args.only_changes)

    # hook解除
    for h in handles:
        h.remove()


if __name__ == "__main__":
    main()
