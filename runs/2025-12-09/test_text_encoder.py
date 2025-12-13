import torch
from ir import Retriever

ret = Retriever.from_pretrained("vsearch/vdr-cross-modal").to("cuda")
ret.eval()


with torch.no_grad():
    txt_emb = ret.encoder_q.embed(["a red dress"])

print("text emb shape:", txt_emb.shape)
print("min/max:", txt_emb.min().item(), txt_emb.max().item())
print("norm:", txt_emb.norm(dim=-1))


dummy_img = torch.randn(1, 3, 224, 224).cuda()
V_r = ret.encoder_p.embed(dummy_img)
V_m = ret.encoder_q.embed(["a red shoe"])

print("V_r shape:", V_r.shape)
print("V_m shape:", V_m.shape)
print("ranges:", float(V_r.min()), float(V_r.max()))
print("norms:", float(V_r.norm()))


