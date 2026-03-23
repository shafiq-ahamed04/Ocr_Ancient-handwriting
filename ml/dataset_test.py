from torchvision import datasets, transforms
from torch.utils.data import DataLoader

train_dir = "data/train"
val_dir = "data/val"

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])

train_dataset = datasets.ImageFolder(train_dir, transform=transform)
val_dataset = datasets.ImageFolder(val_dir, transform=transform)

print("Train samples:", len(train_dataset))
print("Val samples:", len(val_dataset))
print("Classes:", train_dataset.classes)
