import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models


def main() -> None:
  # Paths must match how you trained and stored your data/model
  model_path = "tamil_model.pth"
  val_dir = "dataset/val"

  # Same preprocessing as training
  transform = transforms.Compose(
      [
          transforms.Grayscale(num_output_channels=3),
          transforms.Resize((224, 224)),
          transforms.ToTensor(),
      ]
  )

  val_dataset = datasets.ImageFolder(val_dir, transform=transform)
  val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

  num_classes = len(val_dataset.classes)

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  model = models.resnet18(pretrained=False)
  model.fc = nn.Linear(model.fc.in_features, num_classes)
  state = torch.load(model_path, map_location=device)
  model.load_state_dict(state)
  model.to(device)
  model.eval()

  correct_top1 = 0
  correct_top3 = 0
  total = 0

  with torch.no_grad():
    for images, labels in val_loader:
      images = images.to(device)
      labels = labels.to(device)

      logits = model(images)
      probs = torch.softmax(logits, dim=1)

      # Top‑1
      _, pred_top1 = torch.max(probs, dim=1)
      correct_top1 += (pred_top1 == labels).sum().item()

      # Top‑3
      top3 = torch.topk(probs, k=min(3, probs.shape[1]), dim=1).indices
      # For each sample, check if true label is in its top‑3
      for i in range(labels.size(0)):
        if labels[i] in top3[i]:
          correct_top3 += 1

      total += labels.size(0)

  top1_acc = correct_top1 / total * 100 if total else 0.0
  top3_acc = correct_top3 / total * 100 if total else 0.0

  print(f"Validation samples: {total}")
  print(f"Top‑1 accuracy: {top1_acc:.2f}%")
  print(f"Top‑3 accuracy: {top3_acc:.2f}%")


if __name__ == "__main__":
  main()

