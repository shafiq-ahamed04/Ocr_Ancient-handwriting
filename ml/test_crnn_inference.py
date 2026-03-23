import torch
import cv2
import numpy as np
from pathlib import Path
from crnn_model import CRNN

class LabelCodec:
    def __init__(self, vocab_chars):
        self.blank = 0
        self.char2idx = {c: i + 1 for i, c in enumerate(vocab_chars)}
        self.idx2char = {i + 1: c for i, c in enumerate(vocab_chars)}
        self.num_classes = len(vocab_chars) + 1

    def decode(self, indices):
        chars = []
        prev = -1
        for idx in indices:
            if idx != self.blank and idx != prev:
                ch = self.idx2char.get(idx, "")
                chars.append(ch)
            prev = idx
        return "".join(chars)

def run_test():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = "tamil_crnn.pth"
    
    if not Path(model_path).exists():
        print(f"Error: {model_path} not found.")
        return

    checkpoint = torch.load(model_path, map_location=device)
    vocab = checkpoint["codec_vocab"]
    codec = LabelCodec(vocab)
    
    model = CRNN(img_h=64, nc=1, num_classes=codec.num_classes)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    
    val_dir = Path("dataset/synthetic/val")
    test_files = ["000017.png", "000035.png", "000158.png"]
    
    print("-" * 50)
    print(f"{'File':<15} | {'Ground Truth':<20} | {'Prediction'}")
    print("-" * 50)
    
    for fname in test_files:
        img_path = val_dir / fname
        lbl_path = img_path.with_suffix(".txt")
        
        if not img_path.exists():
            continue
            
        # Ground Truth
        gt = lbl_path.read_text(encoding="utf-8").strip() if lbl_path.exists() else "???"
        
        # Load and preprocess
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        h, w = img.shape
        new_w = max(int(w * 64 / h), 32)
        img = cv2.resize(img, (new_w, 64))
        img = img.astype(np.float32) / 255.0
        tensor = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).to(device)
        
        # Inference
        with torch.no_grad():
            preds = model(tensor) # (T, 1, num_classes)
            
        # CTC Greedy Decode
        preds = preds.argmax(2) # (T, 1)
        preds = preds.squeeze(1).tolist()
        res = codec.decode(preds)
        
        print(f"{fname:<15} | {gt:<20} | {res}")
    print("-" * 50)

if __name__ == "__main__":
    run_test()
