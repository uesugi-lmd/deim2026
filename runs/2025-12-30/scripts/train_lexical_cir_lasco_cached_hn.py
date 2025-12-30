#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from ir import Retriever


# =========================
# Utils
# =========================
def load_any_json(path: Path) -> Any:
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        items = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if not s:
                    continue
                if s.endswith(","):
                    s = s[:-1]
                items.append(json.loads(s))
        return items


def normalize_list_format(data: Any) -> List[Dict[str, Any]]:
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in ["data", "items", "annotations"]:
            if key in data and isinstance(data[key], list):
                return data[key]
    raise ValueError("Unexpected LasCo json format．")


def safe_l2norm(x: torch.Tensor, dim: int = -1, eps: float = 1e-6) -> torch.Tensor:
    return x / (x.norm(dim=dim, keepdim=True) + eps)


# =========================
# Caches
# =========================
class DenseCache:
    """
    paths.json + emb.npy
    emb: [N,D] float32
    """
    def __init__(self, paths_json: str, emb_npy: str):
        self.paths: List[str] = json.loads(Path(paths_json).read_text(encoding="utf-8"))
        self.emb: np.ndarray = np.load(emb_npy)  # [N,D]
        assert self.emb.shape[0] == len(self.paths)
        self.path2i = {p: i for i, p in enumerate(self.paths)}
        self.D = int(self.emb.shape[1])

    def get(self, paths: List[str]) -> torch.Tensor:
        idx = [self.path2i[p] for p in paths]
        return torch.from_numpy(self.emb[idx])  # [B,D] on CPU


class LexicalTopKCache:
    """
    meta.json + paths.json + idx.npy + val.npy
    idx: [N,K] int32, val: [N,K] float16
    """
    def __init__(self, meta_json: str, paths_json: str, idx_npy: str, val_npy: str):
        meta = json.loads(Path(meta_json).read_text(encoding="utf-8"))
        self.V = int(meta["V"])
        self.topk = int(meta["topk"])
        self.paths: List[str] = json.loads(Path(paths_json).read_text(encoding="utf-8"))
        self.idx: np.ndarray = np.load(idx_npy)
        self.val: np.ndarray = np.load(val_npy)
        assert self.idx.shape == self.val.shape
        assert self.idx.shape[0] == len(self.paths)
        self.path2i = {p: i for i, p in enumerate(self.paths)}

    def densify(self, paths: List[str], device: torch.device, dtype=torch.float32) -> torch.Tensor:
        B = len(paths)
        sr = torch.zeros((B, self.V), device=device, dtype=dtype)
        rows = [self.path2i[p] for p in paths]
        idx = torch.from_numpy(self.idx[rows]).to(device).long()  # [B,K]
        val = torch.from_numpy(self.val[rows].astype(np.float32)).to(device).to(dtype)
        sr.scatter_(dim=1, index=idx, src=val)
        return sr


# =========================
# Dataset
# =========================
class LasCoDataset(Dataset):
    def __init__(self, root: str, json_file: str):
        self.root = Path(root)
        self.items = normalize_list_format(load_any_json(self.root / json_file))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int) -> Dict[str, Any]:
        ex = self.items[i]
        ref = str(self.root / ex["query-image"][1])
        tgt = str(self.root / ex["target-image"][1])
        text = ex["query-text"]
        return {"ref": ref, "tgt": tgt, "text": text}


def collate(batch: List[Dict[str, Any]]) -> Tuple[List[str], List[str], List[str]]:
    refs = [b["ref"] for b in batch]
    texts = [b["text"] for b in batch]
    tgts = [b["tgt"] for b in batch]
    return refs, texts, tgts


# =========================
# Model blocks
# =========================
class DeltaLexicalGenerator(nn.Module):
    """
    入力 x は語彙表現 [B,V] を想定（VDR encoder_q.embed の出力）．
    bottleneck 付きで V->V を近似し，Δ は topk でスパース化．
    """
    def __init__(self, din: int, V: int, k_delta: int, bottleneck: int = 512):
        super().__init__()
        self.V = V
        self.k_delta = k_delta

        self.f_plus = nn.Sequential(
            nn.Linear(din, bottleneck),
            nn.GELU(),
            nn.Linear(bottleneck, V),
        )
        self.f_minus = nn.Sequential(
            nn.Linear(din, bottleneck),
            nn.GELU(),
            nn.Linear(bottleneck, V),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        up = F.softplus(self.f_plus(x))    # [B,V]
        um = F.softplus(self.f_minus(x))   # [B,V]

        k = min(self.k_delta, self.V)
        vp, ip = torch.topk(up, k=k, dim=-1)
        vm, im = torch.topk(um, k=k, dim=-1)

        dp = torch.zeros_like(up)
        dm = torch.zeros_like(um)
        dp.scatter_(dim=-1, index=ip, src=vp)
        dm.scatter_(dim=-1, index=im, src=vm)
        return dp, dm


class DenseDecoder(nn.Module):
    """
    Linear(V->D) bias=False
    初期値は VDR encoder_p.proj^T（tied init）
    """
    def __init__(self, W_img: torch.Tensor):
        super().__init__()
        V, D = W_img.shape
        self.V = V
        self.D = D
        self.linear = nn.Linear(V, D, bias=False)
        with torch.no_grad():
            self.linear.weight.copy_(W_img.t().float())

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        z = self.linear(s.float())
        return safe_l2norm(z)


class LexicalCIRCached(nn.Module):
    """
    - VDRは frozen
    - text は encoder_q.embed で語彙 [B,V]
    - ref lexical sr は cache
    - dense decoder は学習対象に入れる（重要）
    """
    def __init__(self, vdr: Retriever, V: int, D: int, k_delta: int, temp: float, bottleneck: int):
        super().__init__()
        self.vdr = vdr
        self.V = V
        self.D = D
        self.temp = temp

        for p in self.vdr.parameters():
            p.requires_grad_(False)

        self.delta_gen = DeltaLexicalGenerator(din=V, V=V, k_delta=k_delta, bottleneck=bottleneck)
        self.decoder = DenseDecoder(self.vdr.encoder_p.proj)

    @torch.no_grad()
    def text_lexical(self, texts: List[str], device: torch.device) -> torch.Tensor:
        enc = self.vdr.encoder_q
        ht = enc.embed(texts, training=False, topk=self.V)  # [B,V]
        return ht.to(device)

    def compose(self, sr: torch.Tensor, texts: List[str]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        return zq, dsp, dsm
        """
        device = sr.device
        ht = self.text_lexical(texts, device=device)         # [B,V]
        dsp, dsm = self.delta_gen(ht)                        # [B,V]
        sqp = torch.clamp(sr + dsp, min=0.0)
        sqm = torch.clamp(dsm, min=0.0)
        sq = sqp - sqm
        zq = self.decoder(sq)                                # [B,D]
        return zq, dsp, dsm

    def forward_train(
        self,
        sr: torch.Tensor,
        texts: List[str],
        z_pos: torch.Tensor,
        z_neg: torch.Tensor,
        lambda_rec: float,
        lambda_sparse: float,
        zr: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        z_pos: [B,D] 正例（target）
        z_neg: [M,D] 負例ギャラリサンプル（正例を除外した方がよいが，多少混ざっても動く）
        """
        device = sr.device
        zq, dsp, dsm = self.compose(sr, texts)

        zq = safe_l2norm(zq.float())
        z_pos = safe_l2norm(z_pos.to(device).float())
        z_neg = safe_l2norm(z_neg.to(device).float())

        # logits over (pos + neg)
        # [B,1+M]
        pos_score = (zq * z_pos).sum(dim=-1, keepdim=True)              # [B,1]
        neg_score = zq @ z_neg.t()                                      # [B,M]
        logits = torch.cat([pos_score, neg_score], dim=1) / self.temp
        labels = torch.zeros(logits.size(0), dtype=torch.long, device=device)
        loss_retr = F.cross_entropy(logits, labels)

        # reconstruction (ref dense)
        loss_rec = torch.tensor(0.0, device=device)
        if zr is not None:
            zr = safe_l2norm(zr.to(device).float())
            zhat_r = self.decoder(sr)
            loss_rec = F.mse_loss(zhat_r, zr)

        # sparse regularizer
        loss_sparse = dsp.abs().sum(dim=-1).mean() + dsm.abs().sum(dim=-1).mean()

        loss = loss_retr + lambda_rec * loss_rec + lambda_sparse * loss_sparse

        return {
            "loss": loss,
            "loss_retr": loss_retr.detach(),
            "loss_rec": loss_rec.detach(),
            "loss_sparse": loss_sparse.detach(),
            "inbatch_like_acc": (logits.argmax(dim=1) == 0).float().mean().detach(),
            "sr_nnz": (sr != 0).sum(dim=-1).float().mean().detach(),
            "ds_nnz": (dsp != 0).sum(dim=-1).float().mean().detach(),
            "zq": zq.detach(),
        }


@torch.no_grad()
def r_at_k_full_gallery(zq: torch.Tensor, gallery: torch.Tensor, pos_index: torch.Tensor, ks=(1, 5, 10)) -> Dict[str, float]:
    zq = safe_l2norm(zq.float())
    gallery = safe_l2norm(gallery.float())
    scores = zq @ gallery.t()
    ranks = scores.argsort(dim=1, descending=True)
    out = {}
    for k in ks:
        hit = (ranks[:, :k] == pos_index.unsqueeze(1)).any(dim=1).float().mean().item()
        out[f"R@{k}"] = float(hit)
    return out


def save_ckpt(path: Path, model: LexicalCIRCached, optim: torch.optim.Optimizer, step: int, best_r1: float, args: argparse.Namespace):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "step": step,
            "best_r1": best_r1,
            "args": vars(args),
            "delta_gen": model.delta_gen.state_dict(),
            "decoder": model.decoder.state_dict(),
            "optim": optim.state_dict(),
            "V": model.V,
            "D": model.D,
        },
        str(path)
    )


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--lasco_root", type=str, default="/home/uesugi/research/dataset/raw/lasco")
    ap.add_argument("--train_json", type=str, default="lasco_train.json")
    ap.add_argument("--val_json", type=str, default="lasco_val.json")

    ap.add_argument("--dense_paths_json", type=str, required=True)
    ap.add_argument("--dense_emb_npy", type=str, required=True)

    ap.add_argument("--reflex_meta_json", type=str, required=True)
    ap.add_argument("--reflex_paths_json", type=str, required=True)
    ap.add_argument("--reflex_idx_npy", type=str, required=True)
    ap.add_argument("--reflex_val_npy", type=str, required=True)

    ap.add_argument("--ckpt", type=str, default="vsearch/vdr-cross-modal")
    ap.add_argument("--device", type=str, default="cuda")

    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--num_workers", type=int, default=4)

    ap.add_argument("--k_delta", type=int, default=64)
    ap.add_argument("--bottleneck", type=int, default=512)
    ap.add_argument("--temp", type=float, default=0.07)

    ap.add_argument("--lambda_rec", type=float, default=1.0)
    ap.add_argument("--lambda_sparse", type=float, default=0.0)

    ap.add_argument("--log_every", type=int, default=50)
    ap.add_argument("--eval_every", type=int, default=500)
    ap.add_argument("--eval_batches", type=int, default=50)

    # hard negatives
    ap.add_argument("--neg_pool", type=int, default=4096, help="gallery からサンプルする負例数 M")
    ap.add_argument("--neg_exclude_pos", action="store_true", help="pos と同一画像を負例から除外する（少し遅くなる）")

    ap.add_argument("--out_dir", type=str, default="./runs/lexical_cir_lasco")
    ap.add_argument("--run_name", type=str, default="lasco_cached_hn_v1")
    args = ap.parse_args()

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    device = torch.device(args.device)

    dense_cache = DenseCache(args.dense_paths_json, args.dense_emb_npy)
    ref_cache = LexicalTopKCache(args.reflex_meta_json, args.reflex_paths_json, args.reflex_idx_npy, args.reflex_val_npy)

    train_ds = LasCoDataset(args.lasco_root, args.train_json)
    val_ds = LasCoDataset(args.lasco_root, args.val_json)

    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                          num_workers=args.num_workers, pin_memory=True, collate_fn=collate)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True, collate_fn=collate)

    # val gallery = unique val targets
    val_data = normalize_list_format(load_any_json(Path(args.lasco_root) / args.val_json))
    val_tgt_paths_all = [str(Path(args.lasco_root) / ex["target-image"][1]) for ex in val_data]
    seen = set()
    val_gallery_paths = []
    for p in val_tgt_paths_all:
        if p not in seen:
            val_gallery_paths.append(p)
            seen.add(p)
    val_tgt2g = {p: i for i, p in enumerate(val_gallery_paths)}

    gallery_np = dense_cache.emb[[dense_cache.path2i[p] for p in val_gallery_paths]]  # [N,D]
    gallery = safe_l2norm(torch.from_numpy(gallery_np).to(device))                    # [N,D]
    gallery_N = gallery.size(0)

    # VDR
    vdr = Retriever.from_pretrained(args.ckpt).to(device).eval()
    V_img, D_img = vdr.encoder_p.proj.shape
    assert V_img == ref_cache.V, f"V mismatch: VDR={V_img} cache={ref_cache.V}"
    assert D_img == dense_cache.D, f"D mismatch: VDR D={D_img} cache D={dense_cache.D}"

    model = LexicalCIRCached(vdr=vdr, V=ref_cache.V, D=dense_cache.D,
                             k_delta=args.k_delta, temp=args.temp, bottleneck=args.bottleneck).to(device)

    # IMPORTANT: decoder も学習
    optim = torch.optim.AdamW(
        list(model.delta_gen.parameters()) + list(model.decoder.parameters()),
        lr=args.lr
    )

    out_dir = Path(args.out_dir) / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_latest = out_dir / "ckpt_latest.pt"
    ckpt_best = out_dir / "ckpt_best.pt"

    print(f"[DATA] train={len(train_ds)} val={len(val_ds)}")
    print(f"[CACHE] dense N={len(dense_cache.paths)} D={dense_cache.D}")
    print(f"[CACHE] ref_lexical N={len(ref_cache.paths)} V={ref_cache.V} topk={ref_cache.topk}")
    print(f"[GALLERY] val targets unique N={len(val_gallery_paths)}")
    print(f"[OUT] {out_dir}")

    # initial eval
    best_r1 = -1.0
    model.eval()
    r1s = []
    for bi, (refs, texts, tgts) in enumerate(val_dl):
        if bi >= args.eval_batches:
            break
        sr = ref_cache.densify(refs, device=device)
        zt = dense_cache.get(tgts).to(device)
        zq, _, _ = model.compose(sr, texts)
        pos = torch.tensor([val_tgt2g[p] for p in tgts], device=device)
        m = r_at_k_full_gallery(zq, gallery, pos, ks=(1, 5, 10))
        r1s.append(m["R@1"])
    best_r1 = float(sum(r1s) / max(1, len(r1s)))
    print(f"[VAL init] R@1={best_r1:.4f} (avg over {len(r1s)} batches)")

    save_ckpt(ckpt_best, model, optim, step=0, best_r1=best_r1, args=args)
    save_ckpt(ckpt_latest, model, optim, step=0, best_r1=best_r1, args=args)

    # train
    step = 0
    model.train()

    g = torch.Generator(device=device)
    g.manual_seed(0)

    for ep in range(args.epochs):
        for refs, texts, tgts in train_dl:
            sr = ref_cache.densify(refs, device=device)
            z_pos = dense_cache.get(tgts).to(device)
            zr = dense_cache.get(refs).to(device)

            # sample negatives from val gallery (fast + stable)
            M = min(args.neg_pool, gallery_N)
            neg_idx = torch.randint(0, gallery_N, (M,), generator=g, device=device)
            if args.neg_exclude_pos:
                # remove positives if present (slowish)
                pos_idx = torch.tensor([val_tgt2g.get(p, -1) for p in tgts], device=device)
                mask = torch.ones_like(neg_idx, dtype=torch.bool)
                for pi in pos_idx:
                    if pi >= 0:
                        mask &= (neg_idx != pi)
                neg_idx = neg_idx[mask]
                if neg_idx.numel() < 256:
                    # fallback
                    neg_idx = torch.randint(0, gallery_N, (M,), generator=g, device=device)

            z_neg = gallery[neg_idx]  # [M,D]

            out = model.forward_train(
                sr=sr,
                texts=texts,
                z_pos=z_pos,
                z_neg=z_neg,
                lambda_rec=args.lambda_rec,
                lambda_sparse=args.lambda_sparse,
                zr=zr,
            )

            loss = out["loss"]
            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()

            if step % args.log_every == 0:
                print(
                    f"[train ep={ep} step={step}] "
                    f"loss={loss.item():.4f} retr={out['loss_retr'].item():.4f} rec={out['loss_rec'].item():.4f} "
                    f"sparse={out['loss_sparse'].item():.4f} "
                    f"acc(pos-vs-negs)={out['inbatch_like_acc'].item():.3f} "
                    f"sr_nnz={out['sr_nnz'].item():.1f} ds_nnz={out['ds_nnz'].item():.1f}"
                )

            if step > 0 and step % args.eval_every == 0:
                model.eval()
                r1s, r5s, r10s = [], [], []
                for bi, (vrefs, vtexts, vtgts) in enumerate(val_dl):
                    if bi >= args.eval_batches:
                        break
                    vsr = ref_cache.densify(vrefs, device=device)
                    vzq, _, _ = model.compose(vsr, vtexts)
                    pos = torch.tensor([val_tgt2g[p] for p in vtgts], device=device)
                    m = r_at_k_full_gallery(vzq, gallery, pos, ks=(1, 5, 10))
                    r1s.append(m["R@1"])
                    r5s.append(m["R@5"])
                    r10s.append(m["R@10"])

                val_r1 = float(sum(r1s) / max(1, len(r1s)))
                val_r5 = float(sum(r5s) / max(1, len(r5s)))
                val_r10 = float(sum(r10s) / max(1, len(r10s)))
                print(f"[VAL step={step}] R@1={val_r1:.4f} R@5={val_r5:.4f} R@10={val_r10:.4f}  (avg over {len(r1s)} batches)")

                save_ckpt(ckpt_latest, model, optim, step=step, best_r1=best_r1, args=args)
                if val_r1 > best_r1:
                    best_r1 = val_r1
                    save_ckpt(ckpt_best, model, optim, step=step, best_r1=best_r1, args=args)
                    print(f"[SAVE best] step={step} best_R@1={best_r1:.4f}")
                model.train()

            step += 1

    save_ckpt(ckpt_latest, model, optim, step=step, best_r1=best_r1, args=args)
    print(f"[DONE] saved latest: {ckpt_latest}")
    print(f"[DONE] saved best  : {ckpt_best}")
    print(f"[DONE] best_R@1={best_r1:.4f}")


if __name__ == "__main__":
    main()
