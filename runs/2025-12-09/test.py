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
