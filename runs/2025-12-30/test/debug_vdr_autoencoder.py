import torch
import torch.nn.functional as F
from ir import Retriever
from typing import Tuple, List, Union
import numpy as np

elu1p = lambda x: F.elu(x) + 1

def build_topk_mask(embs: Union[torch.Tensor, np.ndarray], k: int = 768, dim: int = -1):        
    if isinstance(embs, np.ndarray):
        embs = torch.from_numpy(embs)
    values, indices = torch.topk(embs, k, dim=dim)
    topk_mask = torch.zeros_like(embs, dtype=torch.bool)
    topk_mask.scatter_(dim=-1, index=indices, value=True)
    return topk_mask

@torch.no_grad()
def debug_image_embed(vdr, image_path, topk=None):
    enc = vdr.encoder_p
    topk = topk or enc.config.topk

    x = enc.load_image_file(image_path)         # [1, 3, 224, 224] など
    x = x.to(enc.device).type(enc.dtype)

    h = enc(x)                                  # [N, L, D] = [1,49,768]
    print("h (tokens,dense):", h.shape, h.dtype, h.device)

    W = enc.proj                                # [V, D] = [27623,768]
    print("proj W:", W.shape, W.dtype, W.device)

    lv = h @ W.t()                              # [1,49,27623]
    print("lv (tokens,lex):", lv.shape)

    v_max = lv.max(1)[0]                        # [1,27623]
    print("v_max:", v_max.shape)

    v_act = elu1p(v_max)
    v_norm = F.normalize(v_act)
    mask = build_topk_mask(v_norm, k=topk)
    v_topk = v_norm * mask

    nnz = (v_topk != 0).sum().item()
    print(f"v_topk: {v_topk.shape}  nnz={nnz}  topk={topk}")

    return {
        "h": h, "lv": lv, "v_max": v_max, "v_topk": v_topk
    }

vdr = Retriever.from_pretrained("vsearch/vdr-cross-modal").to("cuda").eval()
out = debug_image_embed(vdr, "/home/uesugi/research/experiments/VDR/examples/images/mars.png")
