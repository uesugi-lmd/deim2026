
import torch
import torch.nn as nn
import torch.nn.functional as F
from ir import Retriever

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

ret = Retriever.from_pretrained("vsearch/vdr-cross-modal").to("cuda")
ret.eval()


with torch.no_grad():
    txt_emb = ret.encoder_q.embed(["a red dress"])

# print("text emb shape:", txt_emb.shape)
# print("min/max:", txt_emb.min().item(), txt_emb.max().item())
# print("norm:", txt_emb.norm(dim=-1))


dummy_img = torch.randn(1, 3, 224, 224).cuda()
V_r = ret.encoder_p.embed(dummy_img)
V_m = ret.encoder_q.embed(["a red shoe"])

# print("V_r shape:", V_r.shape)
# print("V_m shape:", V_m.shape)
# print("ranges:", float(V_r.min()), float(V_r.max()))
# print("norms:", float(V_r.norm()))


class CIRComposer(nn.Module):
    def __init__(self, init_alpha=0.5):
        super().__init__()
        # 学習可能なスカラー
        self.alpha = nn.Parameter(torch.tensor(init_alpha))

    def forward(self, V_r, V_m):
        """
        V_r: (B, 27623) reference image lexical vector
        V_m: (B, 27623) modification text lexical vector
        """

        # 合成
        V_q = V_r + self.alpha * V_m

        # retriever と同じ仕様に揃える
        V_q = F.normalize(V_q, dim=-1)

        return V_q
    
composer = CIRComposer().cuda()
V_q = composer(V_r, V_m)
# print("V_q shape:", V_q.shape)
# print("ranges:", float(V_q.min()), float(V_q.max()))
# print("norm:", float(V_q.norm()))

delta = V_q - V_r
# print("delta max idx:", delta.topk(5).indices)
# print("delta min idx:", (-delta).topk(5).indices)


# retriever はすでにロード済みとする
for p in ret.parameters():
    p.requires_grad = False

composer = CIRComposer().cuda()
optimizer = torch.optim.AdamW([composer.alpha], lr=1e-3)
temperature = 0.07


from data.datasets import CIRRDataset
from torchvision import transforms
from torch.utils.data import DataLoader

preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
])

dataset = CIRRDataset(
    dataset_path="/home/uesugi/research/dataset/raw/cirr",
    split="train",
    mode="relative",
    preprocess=preprocess,
    no_duplicates=False
)

loader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=4)

composer = CIRComposer().cuda()
optimizer = torch.optim.AdamW([composer.alpha], lr=1e-3)
temperature = 0.07

epochs = 5
temperature = 0.07

composer = CIRComposer().cuda()
optimizer = torch.optim.AdamW([composer.alpha], lr=1e-3)

for epoch in range(epochs):
    total_loss = 0.0
    composer.train()

    for step, batch in enumerate(loader):

        ref_img = batch["reference_image"].cuda()
        tgt_img = batch["target_image"].cuda()
        captions = batch["relative_caption"]

        # --- 1. lexical embedding ---
        with torch.no_grad():
            V_r = ret.encoder_p.embed(ref_img)
            V_m = ret.encoder_q.embed(captions)
            V_t = ret.encoder_p.embed(tgt_img)

        # --- 2. compose query ---
        V_q = composer(V_r, V_m)

        # --- 3. SCE loss ---
        logits = (V_q @ V_t.t()) / temperature
        labels = torch.arange(logits.size(0)).cuda()

        loss = 0.5 * (
            F.cross_entropy(logits, labels) +
            F.cross_entropy(logits.t(), labels)
        )

        # --- 4. backward & update ---
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # --- 5. 途中ログ ---
        if (step + 1) % 100 == 0:
            avg = total_loss / (step + 1)
            print(f"Epoch {epoch+1} Step {step+1} | Loss={avg:.4f}  alpha={composer.alpha.item():.4f}")

    print(f"[Epoch {epoch+1}] Mean Loss = {total_loss / len(loader):.4f}")
    print(f"alpha at epoch end: {composer.alpha.item():.4f}")