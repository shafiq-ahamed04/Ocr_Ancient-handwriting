import torch
import cv2
import numpy as np
import os
from pathlib import Path
from train_crnn import LabelCodec
from crnn_model import CRNN

class SimpleCodec:
    def __init__(self, vocab):
        self.idx2char = {i + 1: c for i, c in enumerate(vocab)}
        self.blank = 0
        
    def decode(self, indices):
        chars = []
        prev = -1
        for idx in indices:
            if idx != self.blank and idx != prev:
                ch = self.idx2char.get(idx, "")
                chars.append(ch)
            prev = idx
        return "".join(chars)

def verify():
    device = torch.device('cpu')
    ckpt_path = 'tamil_crnn.pth'
    if not os.path.exists(ckpt_path):
        print(f"Error: {ckpt_path} not found")
        return

    print(f"Loading {ckpt_path}...")
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    
    vocab = checkpoint.get("codec_vocab", [])
    num_classes = checkpoint.get("num_classes", len(vocab) + 1)
    codec = SimpleCodec(vocab)
    
    model = CRNN(img_h=64, nc=1, num_classes=num_classes, rnn_hidden=256)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    
    val_dir = Path(r"dataset\ancient_tamil\val")
    if not val_dir.exists():
        print(f"Error: {val_dir} not found")
        return

    samples = sorted(list(val_dir.glob("*.png")))[:5]
    print(f"Verifying {len(samples)} samples from {val_dir}...")
    
    for img_path in samples:
        lbl_path = img_path.with_suffix(".txt")
        gt = lbl_path.read_text(encoding="utf-8").strip()
        
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None: continue
        
        h, w = img.shape
        nw = max(int(w * 64 / h), 32)
        img = cv2.resize(img, (nw, 64)).astype(np.float32) / 255.0
        
        tensor = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).to(device)
        
        with torch.no_grad():
            out = model(tensor)
            # out shape: (W, B, C)
            preds = out.argmax(2).squeeze(1) # (W,)
            pred_text = codec.decode(preds.tolist())
            
        print(f"File: {img_path.name}")
        print(f"  GT:   {gt}")
        print(f"  Pred: {pred_text}")
        print("-" * 20)

if __name__ == "__main__":
    verify()
