"""
CRNN (Convolutional Recurrent Neural Network) for Tamil Line-Level OCR
======================================================================
Architecture:
  CNN feature extractor  →  Bidirectional LSTM  →  Linear projection  →  CTC loss

Input:  grayscale line image resized to (1, 64, W)
Output: sequence of Tamil character indices
"""

import torch
import torch.nn as nn


class BidirectionalLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.rnn = nn.LSTM(input_size, hidden_size, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(hidden_size * 2, output_size)

    def forward(self, x):
        out, _ = self.rnn(x)
        out = self.fc(out)
        return out


class CRNN(nn.Module):
    """
    CRNN for sequence recognition.

    Input height must be 64. Width can be variable.
    Height reduction path: 64 → 32 → 16 → 8 → 4 → 2 → 1 (via 6 pooling stages)
    """

    def __init__(self, img_h=64, nc=1, num_classes=150, rnn_hidden=256):
        super().__init__()

        # CNN backbone — downsample height from 64 → 1
        self.cnn = nn.Sequential(
            # Block 1: (nc, 64, W) → (64, 32, W)
            nn.Conv2d(nc, 64, 3, 1, 1),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),                         # H: 64→32, W: W→W/2

            # Block 2: (64, 32, W/2) → (128, 16, W/2)
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),                         # H: 32→16, W: W/2→W/4

            # Block 3: (128, 16, W/4) → (256, 8, W/4)
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),                         # H: 16→8, W: W/4→W/8

            # Block 4: (256, 8, W/8) → (256, 4, W/8)
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.ReLU(True),
            nn.MaxPool2d((2, 2), (2, 1), (0, 1)),       # H: 8→4, W: ~W/8

            # Block 5: (256, 4, W/8) → (512, 2, W/8)
            nn.Conv2d(256, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.MaxPool2d((2, 2), (2, 1), (0, 1)),       # H: 4→2, W: ~W/8

            # Block 6: (512, 2, W/8) → (512, 1, W/8)
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.ReLU(True),
            nn.AdaptiveAvgPool2d((1, None)),             # Collapse H→1, keep W
        )

        # RNN: 2-layer bidirectional LSTM
        self.rnn = nn.Sequential(
            BidirectionalLSTM(512, rnn_hidden, rnn_hidden),
            BidirectionalLSTM(rnn_hidden, rnn_hidden, num_classes),
        )

    def forward(self, x):
        """
        Args:
            x: (batch, 1, 64, W) grayscale images

        Returns:
            (T, batch, num_classes) log-probabilities for CTC
        """
        conv = self.cnn(x)            # (batch, 512, 1, T)
        b, c, h, w = conv.size()
        assert h == 1, f"CNN output height must be 1, got {h}"

        conv = conv.squeeze(2)        # (batch, 512, T)
        conv = conv.permute(0, 2, 1)  # (batch, T, 512)

        out = self.rnn(conv)          # (batch, T, num_classes)
        out = out.permute(1, 0, 2)    # (T, batch, num_classes)
        out = nn.functional.log_softmax(out, dim=2)

        return out


if __name__ == "__main__":
    model = CRNN(img_h=64, nc=1, num_classes=150)
    dummy = torch.randn(2, 1, 64, 256)
    out = model(dummy)
    print(f"Input:  {dummy.shape}")
    print(f"Output: {out.shape}")
    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")
