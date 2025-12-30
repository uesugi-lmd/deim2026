import torch
import torch.nn.functional as F
from ir import Retriever

@torch.no_grad()
def decode_approx(vdr, image_path, use_topk=True):
    enc = vdr.encoder_p
    x = enc.load_image_file(image_path).to(enc.device).type(enc.dtype)

    h = enc(x)                      # [1,49,768]
    h_mean = h.mean(dim=1)          # [1,768]  ←「元特徴量」の代表として

    v = enc.embed([image_path])     # [1,27623]（topk込み）
    if not use_topk:
        # topk無しを試したい場合：topk=27623 など全通し（config依存）
        v = enc.embed([image_path], topk=27623)

    W = enc.proj                    # [27623,768]

    h_hat = v @ W                   # [1,768]  ← 近似復元
    # cosineでどれくらい戻ってるか
    cos = F.cosine_similarity(h_hat.float(), h_mean.float()).item()
    l2 = (h_hat.float() - h_mean.float()).pow(2).mean().sqrt().item()

    print("h_mean:", h_mean.shape, "h_hat:", h_hat.shape)
    print("cos(h_hat, h_mean) =", cos)
    print("rmse =", l2)

vdr = Retriever.from_pretrained("vsearch/vdr-cross-modal").to("cuda").eval()
decode_approx(vdr, "/home/uesugi/research/experiments/VDR/examples/images/mars.png", use_topk=True)
decode_approx(vdr, "/home/uesugi/research/experiments/VDR/examples/images/mars.png", use_topk=False)