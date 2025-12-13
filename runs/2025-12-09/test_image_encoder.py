import torch
from ir import Retriever

ret = Retriever.from_pretrained("vsearch/vdr-cross-modal").to("cuda")
ret.eval()

dummy = torch.randn(1, 3, 224, 224).cuda()
with torch.no_grad():
    img_emb = ret.encoder_p.embed(dummy)

print("image emb shape:", img_emb.shape)
print("min/max:", img_emb.min().item(), img_emb.max().item())
print("norm:", img_emb.norm(dim=-1))
