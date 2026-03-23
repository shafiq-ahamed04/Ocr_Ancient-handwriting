"""
Ancient Tamil Characters Recognition System — Backend
======================================================
FastAPI server providing:
  POST /ocr       — image → preprocessed → Tesseract OCR → Tamil text + bounding boxes
  POST /classify  — single-char image → ResNet-18 top-3 predictions
  POST /export/pdf — text → downloadable PDF
  GET  /health    — liveness probe
"""

import base64
import io
import os
import textwrap
from pathlib import Path

import cv2
import numpy as np
import pytesseract
import torch
import torch.nn as nn
from PIL import Image, ImageDraw, ImageFont
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fpdf import FPDF
from torchvision import models, transforms

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Tamil OCR API",
    description="AI-powered Tamil character recognition using Tesseract + OpenCV + ResNet-18",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------
ML_DIR = Path(__file__).resolve().parent.parent / "ml"
MODEL_PATH = ML_DIR / "tamil_model.pth"
DATASET_DIR = ML_DIR / "dataset" / "train"

# Tesseract config — Tamil + English
TESS_CONFIG = r"--oem 3 --psm 6 -l tam+eng"

# ---------------------------------------------------------------------------
# ResNet-18 classifier (lazy-loaded)
# ---------------------------------------------------------------------------
_classifier = None
_class_names: list[str] = []
_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_classify_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])


def _load_classifier():
    """Load the trained ResNet-18 model once on first use."""
    global _classifier, _class_names

    if _classifier is not None:
        return

    # Discover class names from training dataset folder names
    if DATASET_DIR.exists():
        _class_names = sorted(
            [d.name for d in DATASET_DIR.iterdir() if d.is_dir()],
            key=lambda x: int(x) if x.isdigit() else x,
        )
    else:
        _class_names = [str(i) for i in range(40)]

    num_classes = len(_class_names) if _class_names else 40

    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    if MODEL_PATH.exists():
        state = torch.load(str(MODEL_PATH), map_location=_device, weights_only=True)
        model.load_state_dict(state)

    model.to(_device)
    model.eval()
    _classifier = model


# ---------------------------------------------------------------------------
# Image preprocessing helpers (OpenCV)
# ---------------------------------------------------------------------------


def _bytes_to_cv(data: bytes) -> np.ndarray:
    """Decode raw bytes into an OpenCV BGR image."""
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode the uploaded image.")
    return img


def preprocess(img: np.ndarray) -> np.ndarray:
    """
    Apply a sequence of OpenCV operations to make Tamil text sharper
    for Tesseract:
      1. Convert to grayscale
      2. Apply CLAHE (contrast-limited adaptive histogram equalisation)
      3. Bilateral filter (edge-preserving noise removal)
      4. Adaptive threshold (Gaussian)
      5. Morphological close to fill small gaps
    Returns a binary (black text on white) image.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # Noise removal while keeping edges
    denoised = cv2.bilateralFilter(enhanced, d=9, sigmaColor=75, sigmaSpace=75)

    # Adaptive threshold → binary
    binary = cv2.adaptiveThreshold(
        denoised, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=31,
        C=11,
    )

    # Close small gaps in characters
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    return cleaned


def _cv_to_base64(img: np.ndarray, fmt: str = ".png") -> str:
    """Encode an OpenCV image to a base64 data-URI."""
    _, buf = cv2.imencode(fmt, img)
    b64 = base64.b64encode(buf.tobytes()).decode()
    return f"data:image/png;base64,{b64}"


# ---------------------------------------------------------------------------
# Bounding-box helpers
# ---------------------------------------------------------------------------


def _get_bounding_boxes(preprocessed: np.ndarray):
    """
    Run Tesseract `image_to_data` and return a list of word-level bounding
    boxes with their detected text.
    """
    data = pytesseract.image_to_data(
        preprocessed, config=TESS_CONFIG, output_type=pytesseract.Output.DICT
    )

    boxes = []
    n = len(data["text"])
    for i in range(n):
        txt = (data["text"][i] or "").strip()
        conf = int(data["conf"][i]) if data["conf"][i] != "-1" else -1
        if not txt or conf < 0:
            continue
        boxes.append({
            "text": txt,
            "confidence": conf,
            "x": data["left"][i],
            "y": data["top"][i],
            "w": data["width"][i],
            "h": data["height"][i],
        })
    return boxes


def _draw_boxes(original: np.ndarray, boxes: list) -> np.ndarray:
    """Draw bounding rectangles + text labels on a copy of the original image."""
    annotated = original.copy()
    for b in boxes:
        x, y, w, h = b["x"], b["y"], b["w"], b["h"]
        cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 220, 120), 2)
        # Put confidence near the box
        label = f'{b["confidence"]}%'
        cv2.putText(
            annotated, label,
            (x, max(y - 6, 14)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 220, 120), 1, cv2.LINE_AA,
        )
    return annotated


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/health")
def health():
    tesseract_ok = False
    try:
        ver = pytesseract.get_tesseract_version()
        tesseract_ok = True
    except Exception:
        ver = None

    return {
        "status": "ok",
        "tesseract_version": str(ver) if ver else "not found",
        "tesseract_ok": tesseract_ok,
        "model_loaded": _classifier is not None,
        "model_path_exists": MODEL_PATH.exists(),
    }


@app.post("/ocr")
async def ocr(file: UploadFile = File(...)):
    """
    Full OCR pipeline:
      1. Read & decode the uploaded image
      2. Preprocess with OpenCV
      3. Run Tesseract OCR for Tamil+English text
      4. Extract bounding boxes
      5. Hybrid logic: if Tesseract confidence is low, switch to CRNN
      6. Return JSON with extracted text, boxes, and annotated image (base64).
    """
    data = await file.read()
    original = _bytes_to_cv(data)

    # Preprocess for Tesseract
    preprocessed = preprocess(original)

    # OCR — extract full text
    full_text = pytesseract.image_to_string(preprocessed, config=TESS_CONFIG).strip()

    # Bounding boxes
    boxes = _get_bounding_boxes(preprocessed)
    
    # Calculate average confidence for auto-switching
    avg_conf = 0
    if boxes:
        avg_conf = sum(b["confidence"] for b in boxes) / len(boxes)
    
    # DECISION: If Tesseract is unsure or finds nothing, use CRNN Manuscript engine
    engine = "Tesseract (Standard Mode)"
    if not full_text or avg_conf < 50:
        try:
            _load_crnn()
            # Feed the grayscale/CLAHE patch (as fixed recently)
            gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            gray_enhanced = clahe.apply(gray)
            
            # Use the improved helper (robust thresholding)
            lines_data = _segment_lines(gray_enhanced)
            
            line_results = []
            debug_dir = ML_DIR / "images" / "debug_patches"
            debug_dir.mkdir(parents=True, exist_ok=True)
            
            for i, (patch, y1, y2) in enumerate(lines_data):
                # Save patch for visual inspection
                cv2.imwrite(str(debug_dir / f"ocr_switch_line_{i}.png"), patch)
                
                line_txt = _crnn_predict_line(patch)
                line_results.append({
                    "line_number": i + 1, 
                    "text": line_txt,
                    "y1": y1,
                    "y2": y2
                })
            
            crnn_text = "\n".join(r["text"] for r in line_results if r["text"].strip())
            if crnn_text:
                full_text = crnn_text
                engine = "CRNN (Manuscript AI)"
                # Prepare boxes for the UI based on our segmentation
                boxes = [{"x": 0, "y": r["y1"], "w": original.shape[1], "h": r["y2"]-r["y1"], "confidence": 99, "text": r["text"]} for r in line_results if r["text"].strip()]
        except Exception as e:
            # Fallback to Tesseract if CRNN fails
            pass

    # Draw boxes on original
    annotated = _draw_boxes(original, boxes)
        
    annotated_b64 = _cv_to_base64(annotated)
    preprocessed_b64 = _cv_to_base64(preprocessed)

    # Classification (ResNet) — only for small, single-character-ish images
    classification = None
    h, w = original.shape[:2]
    if max(h, w) < 512 and MODEL_PATH.exists():
        try:
            _load_classifier()
            pil_img = Image.fromarray(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
            tensor = _classify_transform(pil_img).unsqueeze(0).to(_device)
            with torch.no_grad():
                logits = _classifier(tensor)
                probs = torch.softmax(logits, dim=1)[0]
            topk = torch.topk(probs, k=min(3, len(probs)))
            candidates = []
            for idx, prob in zip(topk.indices.tolist(), topk.values.tolist()):
                label = _class_names[idx] if idx < len(_class_names) else str(idx)
                candidates.append({
                    "index": idx,
                    "label": label,
                    "probability": round(prob * 100, 2),
                })
            classification = {
                "predicted_label": candidates[0]["label"] if candidates else "?",
                "candidates": candidates,
            }
        except Exception as exc:
            classification = {"error": str(exc)}

    return {
        "text": full_text,
        "boxes": boxes,
        "annotated_image": annotated_b64,
        "preprocessed_image": preprocessed_b64,
        "classification": classification,
        "word_count": len(full_text.split()) if full_text else 0,
        "char_count": len(full_text),
        "engine": engine,
    }


@app.post("/classify")
async def classify(file: UploadFile = File(...)):
    """Classify a single Tamil character image using the trained ResNet-18."""
    _load_classifier()

    data = await file.read()
    pil_img = Image.open(io.BytesIO(data)).convert("RGB")
    tensor = _classify_transform(pil_img).unsqueeze(0).to(_device)

    with torch.no_grad():
        logits = _classifier(tensor)
        probs = torch.softmax(logits, dim=1)[0]

    topk = torch.topk(probs, k=min(5, len(probs)))
    candidates = []
    for idx, prob in zip(topk.indices.tolist(), topk.values.tolist()):
        label = _class_names[idx] if idx < len(_class_names) else str(idx)
        candidates.append({
            "index": idx,
            "label": label,
            "probability": round(prob * 100, 2),
        })

    return {
        "predicted_label": candidates[0]["label"] if candidates else "?",
        "candidates": candidates,
    }


@app.post("/export/pdf")
async def export_pdf(text: str = Form(...)):
    """Generate a downloadable PDF from extracted Tamil text."""
    pdf = FPDF()
    pdf.add_page()

    # Try to use a Tamil-compatible font if available, otherwise use built-in
    tamil_font_path = Path(__file__).parent / "fonts" / "NotoSansTamil-Regular.ttf"
    if tamil_font_path.exists():
        pdf.add_font("NotoTamil", "", str(tamil_font_path), uni=True)
        pdf.set_font("NotoTamil", size=14)
    else:
        # Fallback — built-in font (limited Unicode support)
        pdf.add_font("DejaVu", "", "", uni=True)  # will use built-in
        pdf.set_font("Helvetica", size=14)

    pdf.cell(0, 10, "Extracted Tamil Text", ln=True, align="C")
    pdf.ln(6)
    pdf.set_font_size(12)

    # Wrap long text
    for line in text.split("\n"):
        if line.strip():
            pdf.multi_cell(0, 7, line)
        else:
            pdf.ln(4)

    buf = io.BytesIO()
    pdf.output(buf)
    buf.seek(0)

    return StreamingResponse(
        buf,
        media_type="application/pdf",
        headers={"Content-Disposition": 'attachment; filename="extracted-tamil-text.pdf"'},
    )


# ---------------------------------------------------------------------------
# CRNN-based Manuscript OCR (Deep Learning pipeline)
# ---------------------------------------------------------------------------
import sys
sys.path.insert(0, str(ML_DIR))

CRNN_MODEL_PATH = ML_DIR / "tamil_crnn_v2.pth"
_crnn_model = None
_crnn_codec = None


class _CRNNCodec:
    """Minimal label codec for CRNN inference."""

    def __init__(self, vocab: list[str]):
        self.blank = 0
        self.idx2char = {i + 1: c for i, c in enumerate(vocab)}
        self.num_classes = len(vocab) + 1

    def decode(self, indices: list[int]) -> str:
        chars = []
        prev = -1
        for idx in indices:
            if idx != self.blank and idx != prev:
                ch = self.idx2char.get(idx, "")
                chars.append(ch)
            prev = idx
        return "".join(chars)


def _load_crnn():
    """Lazy-load the CRNN model from tamil_crnn.pth."""
    global _crnn_model, _crnn_codec

    if _crnn_model is not None:
        return

    if not CRNN_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"CRNN model not found at {CRNN_MODEL_PATH}. "
            "Run: cd ml && python train_crnn.py"
        )

    checkpoint = torch.load(str(CRNN_MODEL_PATH), map_location=_device, weights_only=False)

    vocab = checkpoint.get("codec_vocab", [])
    if not vocab:
        import json
        vocab_path = ML_DIR / "vocab.json"
        if vocab_path.exists():
            with open(vocab_path, "r", encoding="utf-8") as f:
                vocab_dict = json.load(f)
                vocab = [vocab_dict[str(i)] for i in range(1, len(vocab_dict) + 1)]
    
    print(f"Loaded {len(vocab)} vocab items from model/fallback")
    
    num_classes = checkpoint.get("num_classes", len(vocab) + 1)
    _crnn_codec = _CRNNCodec(vocab)

    from crnn_model import CRNN as CRNNModel
    model = CRNNModel(img_h=64, nc=1, num_classes=num_classes, rnn_hidden=256)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(_device)
    model.eval()
    _crnn_model = model


def _segment_lines(gray: np.ndarray, min_line_height: int = 15) -> list[tuple[np.ndarray, int, int]]:
    """
    Segment a grayscale paragraph image into individual text lines.
    Returns: list of (line_patch, start_y, end_y)
    """
    # 1. Binarize
    if len(gray.shape) == 3:
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
        
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # 2. Horizontal projection
    h_proj = np.sum(binary, axis=1)
    
    # 3. Robust Thresholding
    sorted_proj = np.sort(h_proj)
    p95 = sorted_proj[int(len(sorted_proj) * 0.95)]
    threshold = p95 * 0.2
    
    # 4. Find line boundaries
    in_line = False
    start = 0
    results = []

    for y, val in enumerate(h_proj):
        if not in_line and val > threshold:
            in_line = True
            start = y
        elif in_line and val <= threshold:
            in_line = False
            if y - start >= min_line_height:
                pad_top = max(0, start - 8)
                pad_bot = min(gray.shape[0], y + 8)
                patch = gray[pad_top:pad_bot, :]
                results.append((patch, pad_top, pad_bot))

    if in_line and gray.shape[0] - start >= min_line_height:
        results.append((gray[start:, :], start, gray.shape[0]))

    if not results:
        results = [(gray, 0, gray.shape[0])]

    return results


def _crnn_predict_line(line_img: np.ndarray) -> str:
    """Run CRNN inference on a single line image."""
    # Resize to height 64, keep aspect
    h, w = line_img.shape
    new_w = max(int(w * 64 / max(h, 1)), 32)
    resized = cv2.resize(line_img, (new_w, 64))

    # Normalize
    # We use simple 0-1 normalization to match training
    tensor = torch.from_numpy(resized.astype(np.float32) / 255.0)
    tensor = tensor.unsqueeze(0).unsqueeze(0)  # (1, 1, 64, W)
    tensor = tensor.to(_device)

    with torch.no_grad():
        log_probs = _crnn_model(tensor)  # (T, 1, num_classes)
        preds = log_probs.argmax(dim=2).squeeze(1)  # (T,)

    decoded = _crnn_codec.decode(preds.tolist())
    
    # Save to debug log
    with open("ocr_debug.txt", "a", encoding="utf-8") as f:
        f.write(f"[CRNN DEBUG] Predicted text: '{decoded}'\n")
    print(f"[CRNN DEBUG] Predicted text: '{decoded}' (Length: {len(decoded)})", flush=True)
    
    return decoded


@app.post("/ocr/manuscript")
async def ocr_manuscript(file: UploadFile = File(...)):
    """
    Manuscript OCR pipeline using CRNN.
    Standardized to use refined segmentation and coordinate mapping.
    """
    _load_crnn()

    data = await file.read()
    original = _bytes_to_cv(data)

    # Preprocess
    gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray_enhanced = clahe.apply(gray)
    
    # Use centralized segmentation helper
    lines_data = _segment_lines(gray_enhanced)
    
    line_results = []
    debug_dir = ML_DIR / "images" / "debug_patches"
    debug_dir.mkdir(parents=True, exist_ok=True)
    
    for i, (patch, y1, y2) in enumerate(lines_data):
        # Save patch for visual inspection
        cv2.imwrite(str(debug_dir / f"line_manuscript_{i}.png"), patch)
        
        text = _crnn_predict_line(patch)
        line_results.append({
            "line_number": i + 1,
            "text": text,
            "y1": y1,
            "y2": y2
        })

    # Combine into full paragraph
    full_text = "\n".join(r["text"] for r in line_results if r["text"].strip())

    # Create boxes for response
    boxes = [{"x": 0, "y": r["y1"], "w": original.shape[1], "h": r["y2"]-r["y1"], "confidence": 99, "text": r["text"]} for r in line_results if r["text"].strip()]

    # Draw line segmentation boxes 
    annotated = _draw_boxes(original, boxes)
    
    annotated_b64 = _cv_to_base64(annotated)
    preprocessed_b64 = _cv_to_base64(preprocess(original))

    return {
        "text": full_text,
        "boxes": boxes,
        "num_lines": len(line_results),
        "annotated_image": annotated_b64,
        "preprocessed_image": preprocessed_b64,
        "word_count": len(full_text.split()) if full_text else 0,
        "char_count": len(full_text),
        "engine": "CRNN (Manuscript Native)",
    }
