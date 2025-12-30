# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any

import torch
import torch.nn as nn
import torch.nn.functional as F


# -------------------------
# Utilities
# -------------------------
def topk_mask(x: torch.Tensor, k: int) -> torch.Tensor:
    """
    x: [B, V] (assumed non-negative)
    return mask: [B, V] with exactly k ones per row (unless k<=0 -> all ones)
    """
    if k is None or k <= 0 or k >= x.size(-1):
        return torch.ones_like(x)
    vals, idx = torch.topk(x, k=k, dim=-1)
    mask = torch.zeros_like(x)
    mask.scatter_(dim=-1, index=idx, value=1.0)
    return mask


def safe_l2norm(x: torch.Tensor, dim: int = -1, eps: float = 1e-6) -> torch.Tensor:
    return x / (x.norm(dim=dim, keepdim=True) + eps)


def info_nce(query: torch.Tensor, key: torch.Tensor, temperature: float = 0.07) -> torch.Tensor:
    """
    query: [B, D]
    key:   [B, D] (positive pairs are aligned by index)
    """
    q = safe_l2norm(query.float())
    k = safe_l2norm(key.float())
    logits = (q @ k.t()) / temperature  # [B,B]
    labels = torch.arange(logits.size(0), device=logits.device)
    return F.cross_entropy(logits, labels)


# -------------------------
# VDR adapters
# -------------------------
@dataclass
class VDRHandles:
    """
    VDR の学習済みモデル（Retriever）から必要なものだけ抜き出すラッパ．
    """
    vdr: Any  # ir.Retriever


class VDRLexicalProjector:
    """
    VDR の内部（encoder_q / encoder_p）から，"線形可逆な部分" までの語彙表現を取り出す．
    - dense token: h = encoder(x)  -> [B,L,D]
    - dictionary: W = encoder.proj -> [V,D]
    - linear lexical: lv = h @ W^T -> [B,L,V]
    """
    def __init__(self, handles: VDRHandles):
        self.vdr = handles.vdr

    @torch.no_grad()
    def image_h_lv(self, images: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        enc = self.vdr.encoder_p
        device = next(self.vdr.parameters()).device
        x = images.to(device).type(enc.dtype)
        h = enc(x)                    # [B,L,D]
        W = enc.proj                  # [V,D]
        lv = h @ W.t()                # [B,L,V]
        return h, lv, W

    @torch.no_grad()
    def text_h_lv(self, encoding: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        encoding: encoder_q.encode(texts) の出力を想定（input_ids 等）
        VDR 実装によって forward の引数形が違う可能性があるので try で吸収する．
        """
        enc = self.vdr.encoder_q
        W = enc.proj  # [V,D] であることを期待（画像と同様）
        device = next(self.vdr.parameters()).device

        # できるだけ柔軟に forward を呼ぶ
        enc_in = {k: v.to(device) for k, v in encoding.items() if isinstance(v, torch.Tensor)}
        try:
            h = enc(**enc_in)  # もし token 出力なら [B,L,D] になる
        except TypeError:
            # 旧 API: (ids, segments, mask) 形式の可能性
            keys = list(enc_in.keys())
            if len(keys) >= 3:
                h = enc(enc_in[keys[0]], enc_in[keys[1]], enc_in[keys[2]])
            else:
                raise

        # もし [B,D] しか返らない実装なら，L=1 token とみなして整形
        if h.dim() == 2:
            h = h.unsqueeze(1)  # [B,1,D]

        lv = h @ W.t()          # [B,L,V]
        return h, lv, W

    @torch.no_grad()
    def text_encode(self, texts: List[str]) -> Dict[str, torch.Tensor]:
        """
        encoder_q.encode があればそれを使う．なければ embed にフォールバック．
        """
        enc = self.vdr.encoder_q
        if hasattr(enc, "encode"):
            return enc.encode(texts)
        # encode が無い場合：最低限 embed できるならそれでも学習は回るが，
        # Delta Generator の入力が取りづらいのでここではエラーにしておく
        raise RuntimeError("encoder_q.encode(texts) が見つかりません．VDR の text encoder API を確認してください．")

    @torch.no_grad()
    def image_embed_dense(self, images: torch.Tensor) -> torch.Tensor:
        """
        画像の dense ベクトル（検索空間の D 次元）を取る．
        VDR の encoder_p が token を返すので，平均で pooled にする．
        """
        enc = self.vdr.encoder_p
        device = next(self.vdr.parameters()).device
        h = enc(images.to(device).type(enc.dtype))  # [B,L,D]
        z = h.mean(dim=1)                           # [B,D] ひとまず mean pooling
        return safe_l2norm(z)

    @torch.no_grad()
    def image_embed_dense_from_pathlist(self, image_paths: List[str]) -> torch.Tensor:
        enc = self.vdr.encoder_p
        imgs = [enc.load_image_file(p) for p in image_paths]  # each [1,3,224,224]
        x = torch.cat(imgs, dim=0)
        return self.image_embed_dense(x)


# -------------------------
# Modules
# -------------------------
class DeltaLexicalGenerator(nn.Module):
    """
    修正文 T -> Δs+ , Δs- を生成する．
    - 入力：text encoder の pooled 表現 h_t ∈ R^D
    - 出力：Δs+ , Δs- ∈ R^V (non-negative, TopK sparse)
    - 重要：BoW マスク m(T) を掛けて語彙逸脱を抑える
    """
    def __init__(self, d: int, V: int, k_delta: int = 256, use_bias: bool = True):
        super().__init__()
        self.V = V
        self.k_delta = k_delta
        self.f_plus = nn.Linear(d, V, bias=use_bias)
        self.f_minus = nn.Linear(d, V, bias=use_bias)

    def forward(self, h_t: torch.Tensor, bow_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        h_t: [B,D]
        bow_mask: [B,V] in {0,1} (optional)
        """
        u_plus = F.softplus(self.f_plus(h_t))     # [B,V] non-negative
        u_minus = F.softplus(self.f_minus(h_t))   # [B,V] non-negative

        if bow_mask is not None:
            u_plus = u_plus * bow_mask
            u_minus = u_minus * bow_mask

        m_plus = topk_mask(u_plus, self.k_delta)
        m_minus = topk_mask(u_minus, self.k_delta)

        ds_plus = u_plus * m_plus
        ds_minus = u_minus * m_minus
        return ds_plus, ds_minus


class LexicalArithmetic(nn.Module):
    """
    s_q^+ = clip(s_r^+ + Δs^+, 0, inf)
    s_q^- = clip(s_r^- + Δs^-, 0, inf)
    s_q   = s_q^+ - s_q^-
    """
    def __init__(self):
        super().__init__()

    def forward(self, sr_plus: torch.Tensor, sr_minus: torch.Tensor,
                ds_plus: torch.Tensor, ds_minus: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        sq_plus = torch.clamp(sr_plus + ds_plus, min=0.0)
        sq_minus = torch.clamp(sr_minus + ds_minus, min=0.0)
        sq = sq_plus - sq_minus
        return sq_plus, sq_minus, sq


class DenseDecoder(nn.Module):
    """
    Sparse (V) -> Dense (D)
    - まずは tied weights: decoder.weight = W (proj) を使うのが安定
    - biasなし
    """
    def __init__(self, W: torch.Tensor):
        super().__init__()
        # W: [V,D] なので decoder は Linear(V->D) で weight = W^T
        V, D = W.shape
        self.V = V
        self.D = D
        self.linear = nn.Linear(V, D, bias=False)
        # tied init
        with torch.no_grad():
            self.linear.weight.copy_(W.t().float())  # [D,V]

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        z = self.linear(s.float())
        return safe_l2norm(z)


class LexicalCIRModel(nn.Module):
    """
    全体：
      - VDR (frozen) で
          * 参照画像 -> sr (語彙) を作る（ここは linear part でも embed でも選べる）
          * 画像 -> z (dense) を作る（ターゲット用）
          * 修正文 -> pooled h_t を作る
      - G(T): Δs+ Δs-
      - Arithmetic: s_q
      - Decoder: z_hat_q
      - Loss: InfoNCE + rec + sparse + delta(optional)
    """
    def __init__(
        self,
        vdr_handles: VDRHandles,
        V: int = 27623,
        D: int = 768,
        k_ref: int = 768,       # 参照語彙の TopK（VDR の思想に寄せる）
        k_delta: int = 256,
        temperature: float = 0.07,
    ):
        super().__init__()
        self.vdr = vdr_handles.vdr
        self.proj = VDRLexicalProjector(vdr_handles)
        self.V = V
        self.D = D
        self.k_ref = k_ref
        self.temperature = temperature

        # VDR の辞書（画像側）を decoder 初期化に使う（まずはこれが一番安定）
        W_img = self.vdr.encoder_p.proj
        assert W_img.shape[0] == V and W_img.shape[1] == D, f"W_img shape mismatch: {tuple(W_img.shape)}"
        self.decoder = DenseDecoder(W_img)

        self.delta_gen = DeltaLexicalGenerator(d=D, V=V, k_delta=k_delta)
        self.arith = LexicalArithmetic()

        # VDR は固定（最短運用）
        for p in self.vdr.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def build_bow_mask(self, texts: List[str], V: int, device: torch.device) -> Optional[torch.Tensor]:
        """
        修正文BoWマスク（+類義語拡張）は repo 依存なので，
        まずは「VDR の dst があればそれを使う」→無ければ all-ones にフォールバック．
        """
        enc = self.vdr.encoder_q
        if hasattr(enc, "dst"):
            masks = []
            for t in texts:
                # dst: dict(vocab_id or vocab_str -> weight) を期待
                d = enc.dst(t, topk=V)  # heavy なので本当は topk 小さめ推奨
                m = torch.zeros(V, device=device)
                # key が int の場合 / str の場合に対応
                for k in d.keys():
                    if isinstance(k, int) and 0 <= k < V:
                        m[k] = 1.0
                # str の場合は変換表が必要だが repo 不明のためスキップ
                if m.sum() == 0:
                    m[:] = 1.0
                masks.append(m)
            return torch.stack(masks, dim=0)  # [B,V]
        return torch.ones((len(texts), V), device=device)

    @torch.no_grad()
    def ref_image_lexical(self, ref_image_paths: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        参照画像 -> sr_plus (TopK) , sr_minus (zeros)
        - ここでは "線形 lv -> maxpool" を自前でやって，embed() の elu/normalize を避ける
          （提案の核の足し引きを汚さないため）
        """
        enc = self.vdr.encoder_p
        imgs = [enc.load_image_file(p) for p in ref_image_paths]
        x = torch.cat(imgs, dim=0)  # [B,3,224,224]
        h, lv, W = self.proj.image_h_lv(x)    # lv: [B,L,V]
        sr = lv.max(dim=1)[0]                 # [B,V] まずは maxpool のみ
        sr = torch.clamp(sr, min=0.0)         # 非負化（安全）
        sr = sr * topk_mask(sr, self.k_ref)   # TopK
        sr_minus = torch.zeros_like(sr)
        return sr, sr_minus

    @torch.no_grad()
    def target_dense(self, tgt_image_paths: List[str]) -> torch.Tensor:
        return self.proj.image_embed_dense_from_pathlist(tgt_image_paths)  # [B,D]

    @torch.no_grad()
    def text_pooled(self, texts: List[str]) -> torch.Tensor:
        encoding = self.proj.text_encode(texts)
        h, lv, W = self.proj.text_h_lv(encoding)  # h: [B,L,D] or [B,1,D]
        ht = h.mean(dim=1)                        # [B,D]
        return ht

    def forward(
        self,
        ref_image_paths: List[str],
        texts: List[str],
        tgt_image_paths: List[str],
        lambdas: Dict[str, float] | None = None,
    ) -> Dict[str, torch.Tensor]:

        device = next(self.parameters()).device
        lambdas = lambdas or {"rec": 1.0, "sparse": 0.0, "delta": 0.0}

        # --- VDR(frozen) features
        sr_plus, sr_minus = self.ref_image_lexical(ref_image_paths)  # [B,V], [B,V]
        z_t = self.target_dense(tgt_image_paths)                     # [B,D]
        h_t = self.text_pooled(texts).to(device)                     # [B,D]

        # --- BoW mask
        bow = self.build_bow_mask(texts, self.V, device=device)      # [B,V]

        # --- Delta lexical
        ds_plus, ds_minus = self.delta_gen(h_t, bow_mask=bow)        # [B,V], [B,V]

        # --- Arithmetic
        sq_plus, sq_minus, sq = self.arith(sr_plus, sr_minus, ds_plus, ds_minus)  # [B,V]...

        # --- Dense decode
        z_hat_q = self.decoder(sq)                                   # [B,D]

        # --- Losses
        loss_retr = info_nce(z_hat_q, z_t, temperature=self.temperature)

        # reconstruction: D(sr_plus - sr_minus) ≈ z_r (dense of ref)
        # 参照 dense は mean pooling を使う
        with torch.no_grad():
            z_r = self.proj.image_embed_dense_from_pathlist(ref_image_paths)  # [B,D]
        z_hat_r = self.decoder(sr_plus - sr_minus)
        loss_rec = F.mse_loss(z_hat_r, z_r)

        loss_sparse = (ds_plus.abs().sum(dim=-1).mean() + ds_minus.abs().sum(dim=-1).mean())

        # delta alignment (optional): (sr + (ds_plus-ds_minus)) ≈ st (lexical)
        loss_delta = torch.tensor(0.0, device=device)
        if lambdas.get("delta", 0.0) > 0:
            with torch.no_grad():
                # target lexical (same linear pipeline)
                enc = self.vdr.encoder_p
                imgs = [enc.load_image_file(p) for p in tgt_image_paths]
                x = torch.cat(imgs, dim=0)
                _, lv_t, _ = self.proj.image_h_lv(x)
                st = torch.clamp(lv_t.max(dim=1)[0], min=0.0)
                st = st * topk_mask(st, self.k_ref)
            pred = (sr_plus - sr_minus) + (ds_plus - ds_minus)
            loss_delta = F.mse_loss(pred, st)

        loss = (
            loss_retr
            + lambdas.get("rec", 1.0) * loss_rec
            + lambdas.get("sparse", 0.0) * loss_sparse
            + lambdas.get("delta", 0.0) * loss_delta
        )

        return {
            "loss": loss,
            "loss_retr": loss_retr.detach(),
            "loss_rec": loss_rec.detach(),
            "loss_sparse": loss_sparse.detach(),
            "loss_delta": loss_delta.detach(),
            "z_hat_q": z_hat_q.detach(),
            "z_t": z_t.detach(),
            "sr_nnz": (sr_plus != 0).sum(dim=-1).float().mean().detach(),
            "ds_nnz": (ds_plus != 0).sum(dim=-1).float().mean().detach(),
        }
