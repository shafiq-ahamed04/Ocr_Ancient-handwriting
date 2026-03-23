"""
Train the CRNN model on the synthetic Tamil dataset
====================================================
Usage:
    cd ml
    python train_crnn.py --data dataset/synthetic --epochs 30 --batch 16

Saves the best model as  tamil_crnn.pth
"""

import argparse
import os
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from crnn_model import CRNN


# ---------------------------------------------------------------------------
# Character encoding
# ---------------------------------------------------------------------------

class LabelCodec:
    """
    Maps Tamil characters to integer indices and back.
    Index 0 is reserved for CTC blank.
    """

    def __init__(self, vocab_path: str | Path | None = None, vocab_chars: str = ""):
        if vocab_path and Path(vocab_path).exists():
            chars = Path(vocab_path).read_text(encoding="utf-8").strip().split("\n")
            chars = [c.strip() for c in chars if c.strip()]
        elif vocab_chars:
            chars = sorted(set(vocab_chars))
        else:
            # Default Tamil set (comprehensive)
            chars = list(sorted(set(
                "அஆஇஈஉஊஎஏஐஒஓஔஃ"
                "கஙசஞடணதநபமயரலவழளறன"
                "ாிீுூெேைொோௌ்"
                "ஜஷஸஹ"
                " .,;:!?()-–—\"'0123456789"
            )))

        self.blank = 0  # CTC blank
        self.char2idx = {c: i + 1 for i, c in enumerate(chars)}
        self.idx2char = {i + 1: c for i, c in enumerate(chars)}
        self.num_classes = len(chars) + 1  # +1 for blank

    def encode(self, text: str) -> list[int]:
        return [self.char2idx.get(c, 0) for c in text]

    def decode(self, indices: list[int]) -> str:
        """CTC greedy decode: collapse repeated chars and remove blanks."""
        chars = []
        prev = -1
        for idx in indices:
            if idx != self.blank and idx != prev:
                ch = self.idx2char.get(idx, "")
                chars.append(ch)
            prev = idx
        return "".join(chars)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class TamilLineDataset(Dataset):
    """Loads line images + ground truth labels from the synthetic dataset."""

    def __init__(self, root_dir: str, codec: LabelCodec, img_h: int = 64, max_w: int = 800):
        self.root = Path(root_dir)
        self.codec = codec
        self.img_h = img_h
        self.max_w = max_w

        # Collect all image/label pairs
        self.samples = []
        for img_path in sorted(self.root.glob("*.png")):
            lbl_path = img_path.with_suffix(".txt")
            if lbl_path.exists():
                self.samples.append((img_path, lbl_path))

        print(f"  Loaded {len(self.samples)} samples from {root_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, lbl_path = self.samples[idx]

        # Load image as grayscale
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            img = np.zeros((self.img_h, 100), dtype=np.uint8)

        # Resize height, keep aspect ratio
        h, w = img.shape
        new_w = max(int(w * self.img_h / h), 32)
        new_w = min(new_w, self.max_w)
        img = cv2.resize(img, (new_w, self.img_h))

        # Normalize to [0, 1]
        img = img.astype(np.float32) / 255.0

        # To tensor: (1, H, W)
        tensor = torch.from_numpy(img).unsqueeze(0)

        # Label
        text = lbl_path.read_text(encoding="utf-8").strip()
        encoded = self.codec.encode(text)

        return tensor, encoded, len(encoded), new_w

    @staticmethod
    def collate_fn(batch):
        """Custom collate: pad images to max width in batch."""
        images, labels, label_lengths, widths = zip(*batch)

        max_w = max(t.shape[2] for t in images)
        img_h = images[0].shape[1]

        padded = torch.zeros(len(images), 1, img_h, max_w)
        for i, img in enumerate(images):
            padded[i, :, :, : img.shape[2]] = img

        # Flatten labels
        flat_labels = []
        for lbl in labels:
            flat_labels.extend(lbl)
        flat_labels = torch.IntTensor(flat_labels)

        label_lengths = torch.IntTensor(label_lengths)
        input_lengths = torch.IntTensor([max_w // 8 for _ in widths])  # approx after CNN pooling

        return padded, flat_labels, input_lengths, label_lengths


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    data_dir = Path(args.data)
    train_dir = data_dir / "train"
    val_dir = data_dir / "val"
    vocab_path = data_dir / "vocab.txt"

    # Codec
    codec = LabelCodec(vocab_path=vocab_path)
    print(f"Vocabulary: {codec.num_classes} classes (incl. blank)")

    # Datasets
    train_ds = TamilLineDataset(str(train_dir), codec, img_h=64)
    val_ds = TamilLineDataset(str(val_dir), codec, img_h=64)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch,
        shuffle=True,
        num_workers=0,
        collate_fn=TamilLineDataset.collate_fn,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch,
        shuffle=False,
        num_workers=0,
        collate_fn=TamilLineDataset.collate_fn,
    )

    # Model
    model = CRNN(img_h=64, nc=1, num_classes=codec.num_classes, rnn_hidden=256)
    model.to(device)
    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")

    # Loss & optimizer
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    best_val_loss = float("inf")
    save_path = Path(args.save)

    for epoch in range(1, args.epochs + 1):
        # ---- Train ----
        model.train()
        total_loss = 0.0
        num_batches = 0

        for images, labels, input_lengths, label_lengths in train_loader:
            images = images.to(device)
            labels = labels.to(device).long()
            input_lengths = input_lengths.to(device).long()
            label_lengths = label_lengths.to(device).long()

            preds = model(images)  # (T, batch, num_classes)

            loss = criterion(preds, labels, input_lengths, label_lengths)
            if torch.isinf(loss) or torch.isnan(loss):
                continue

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1
            
            if num_batches % 50 == 0:
                print(f"  Batch {num_batches}/{len(train_loader)} | Loss: {loss.item():.4f}", flush=True)

        avg_train_loss = total_loss / max(num_batches, 1)

        # ---- Validate ----
        model.eval()
        val_loss = 0.0
        val_batches = 0
        correct_chars = 0
        total_chars = 0

        with torch.no_grad():
            for images, labels, input_lengths, label_lengths in val_loader:
                images = images.to(device)
                labels = labels.to(device).long()
                input_lengths = input_lengths.to(device).long()
                label_lengths = label_lengths.to(device).long()

                preds = model(images)
                loss = criterion(preds, labels, input_lengths, label_lengths)

                if not (torch.isinf(loss) or torch.isnan(loss)):
                    val_loss += loss.item()
                    val_batches += 1

        avg_val_loss = val_loss / max(val_batches, 1)

        lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch:3d}/{args.epochs} | "
            f"Train loss: {avg_train_loss:.4f} | "
            f"Val loss: {avg_val_loss:.4f} | "
            f"LR: {lr:.6f}"
        )

        # Save best
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                "model_state_dict": model.state_dict(),
                "codec_vocab": list(codec.char2idx.keys()),
                "num_classes": codec.num_classes,
                "epoch": epoch,
                "val_loss": avg_val_loss,
            }, str(save_path))
            print(f"  -> Saved best model (val_loss={avg_val_loss:.4f})")

        scheduler.step()

    print(f"\nTraining complete. Best model saved to: {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Tamil CRNN")
    parser.add_argument("--data", type=str, default="dataset/synthetic", help="Dataset root")
    parser.add_argument("--epochs", type=int, default=30, help="Training epochs")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--save", type=str, default="tamil_crnn.pth", help="Model save path")
    args = parser.parse_args()

    train(args)
