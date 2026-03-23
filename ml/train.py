import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np

# Load our model architecture
from crnn_model import CRNN

class TamilCharDataset(Dataset):
    def __init__(self, images, labels, vocab2idx):
        # images: (N, 64, 256)
        # We need to add a channel dimension: (N, 1, 64, 256) and normalize to 0-1
        self.images = torch.tensor(images, dtype=torch.float32).unsqueeze(1) / 255.0
        self.labels = labels
        self.vocab2idx = vocab2idx

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = self.images[idx]
        label_str = self.labels[idx]
        
        # CTC expects a sequence of indices for the label
        # Since these are single characters/classes, the sequence length is 1
        encoded = [self.vocab2idx[label_str]]
        target = torch.tensor(encoded, dtype=torch.long)
        
        return img, target

def collate_fn(batch):
    images, targets = zip(*batch)
    images = torch.stack(images, 0)
    
    # Concatenate targets into 1D tensor
    targets = torch.cat(targets, 0)
    
    # Input lengths for CTC (W size after pooling). 
    # For input 256 wide, CRNN with 4 poolings of stride 2 in width (or whatever the model uses)
    # Let's dynamically calculate width by passing a dummy through the model or knowing the architecture:
    # 256 -> 128 -> 64 -> 32 -> 31 -> 31 (depending on padding/stride).
    # Let's assume the CNN outputs width W_out. We can compute it in the training loop.
    # But CTC loss expects input_lengths for each item in the batch.
    
    # Target lengths (all 1 because these are single labels)
    target_lengths = torch.ones(len(batch), dtype=torch.long)
    
    return images, targets, target_lengths


def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 1. Read dataset_labels.txt and build vocabulary
    with open("dataset_labels.txt", "r", encoding="utf-8") as f:
        labels = [line.strip() for line in f if line.strip()]
        
    unique_chars = sorted(list(set(labels)))
    # 0 is reserved for CTC blank label
    vocab2idx = {char: idx + 1 for idx, char in enumerate(unique_chars)}
    idx2vocab = {idx + 1: char for idx, char in enumerate(unique_chars)}
    
    num_classes = len(unique_chars) + 1
    print(f"Unique characters found: {len(unique_chars)}")
    print(f"Calculated num_classes (incl. blank): {num_classes}")
    
    # Save vocabulary
    with open("vocab.json", "w", encoding="utf-8") as f:
        json.dump(idx2vocab, f, ensure_ascii=False, indent=2)
    print("Saved vocab.json")
    
    # Load images
    print("Loading dataset_images.npy...")
    images = np.load("dataset_images.npy")
    print(f"Loaded total images: {images.shape}")
    
    # Split 80/20
    indices = np.random.permutation(len(labels))
    split = int(0.8 * len(labels))
    train_idx, val_idx = indices[:split], indices[split:]
    
    train_ds = TamilCharDataset(images[train_idx], [labels[i] for i in train_idx], vocab2idx)
    val_ds = TamilCharDataset(images[val_idx], [labels[i] for i in val_idx], vocab2idx)
    
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, collate_fn=collate_fn)
    
    # Initialize Model
    # img_h=64, nc=1, num_classes=num_classes, rnn_hidden=256
    model = CRNN(img_h=64, nc=1, num_classes=num_classes, rnn_hidden=256)
    model.to(device)
    
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer = optim.Adam(model.parameters(), lr=0.0005)
    
    best_loss = float('inf')
    epochs = 50
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        
        for batch_imgs, batch_targets, target_lengths in train_loader:
            batch_imgs = batch_imgs.to(device)
            batch_targets = batch_targets.to(device)
            target_lengths = target_lengths.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass: out shape => (W, B, num_classes)
            out = model(batch_imgs)
            
            # CTC input lengths: W size is out.size(0)
            W_out = out.size(0)
            B = out.size(1)
            input_lengths = torch.full(size=(B,), fill_value=W_out, dtype=torch.long).to(device)
            
            loss = criterion(out, batch_targets, input_lengths, target_lengths)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
        train_loss /= len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_imgs, batch_targets, target_lengths in val_loader:
                batch_imgs = batch_imgs.to(device)
                batch_targets = batch_targets.to(device)
                target_lengths = target_lengths.to(device)
                
                out = model(batch_imgs)
                input_lengths = torch.full(size=(out.size(1),), fill_value=out.size(0), dtype=torch.long).to(device)
                loss = criterion(out, batch_targets, input_lengths, target_lengths)
                val_loss += loss.item()
                
        val_loss /= len(val_loader)
        
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        
        if val_loss < best_loss:
            best_loss = val_loss
            # Save best model
            torch.save({
                'model_state_dict': model.state_dict(),
                'codec_vocab': unique_chars,
                'num_classes': num_classes
            }, 'tamil_crnn_v2.pth')

if __name__ == '__main__':
    train()