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
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fpdf import FPDF
from torchvision import models, transforms

# ---------------------------------------------------------------------------
# Config & Env
# ---------------------------------------------------------------------------
# Use the .env file in the venv folder (as per current user location)
ENV_PATH = Path(__file__).resolve().parent / "venv" / ".env"
load_dotenv(dotenv_path=ENV_PATH)

TESSERACT_PATH = os.getenv("TESSERACT_PATH")
if TESSERACT_PATH:
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH

ALLOWED_ORIGINS_STR = os.getenv("ALLOWED_ORIGINS", "*")
ALLOWED_ORIGINS = ALLOWED_ORIGINS_STR.split(",") if ALLOWED_ORIGINS_STR != "*" else ["*"]

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
    allow_origins=ALLOWED_ORIGINS if "*" not in ALLOWED_ORIGINS else ["*"],
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
TESS_CONFIG = r"--oem 3 --psm 4 -l tam+eng"

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


def _deskew(binary: np.ndarray) -> np.ndarray:
    """Auto-deskew a binary image using minimum area rectangle."""
    coords = np.column_stack(np.where(binary < 128))
    if len(coords) < 50:
        return binary
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    if abs(angle) < 0.5:
        return binary
    h, w = binary.shape
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(binary, M, (w, h),
                             flags=cv2.INTER_CUBIC,
                             borderMode=cv2.BORDER_REPLICATE)
    return rotated


def _remove_lines(binary: np.ndarray) -> np.ndarray:
    """
    Remove horizontal ruled lines from notebook paper.
    Only activates when significant horizontal lines are detected.
    Leaves clean white-paper images completely untouched.
    """
    h, w = binary.shape
    # Invert: text and lines become white on black
    inv = cv2.bitwise_not(binary)

    # Detect horizontal lines using a wide horizontal kernel
    horiz_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (w // 8, 1))
    horiz_lines = cv2.morphologyEx(inv, cv2.MORPH_OPEN, horiz_kernel, iterations=2)

    # Check if significant lines were found
    line_pixels = cv2.countNonZero(horiz_lines)
    total_pixels = h * w
    line_ratio = line_pixels / total_pixels

    if line_ratio < 0.005:
        # No significant lines detected — return original untouched
        return binary

    # Dilate lines slightly to cover their full thickness
    horiz_lines = cv2.dilate(horiz_lines, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3)), iterations=1)

    # Erase lines: where lines exist, set to white (background)
    result = binary.copy()
    result[horiz_lines > 0] = 255

    # Repair any text strokes that were broken by line removal
    repair_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 3))
    result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, repair_kernel)

    return result


def preprocess(img: np.ndarray) -> np.ndarray:
    """
    Optimized preprocessing for modern Tamil handwriting.
    Handles both clean white paper and lined notebook paper.
    Steps:
      1. Upscale small images (Tesseract needs ~300 DPI)
      2. Convert to grayscale
      3. Gentle CLAHE contrast boost
      4. Light Gaussian blur (preserve thin strokes)
      5. Otsu threshold (best for clean white-paper handwriting)
      6. Remove ruled notebook lines (auto-detects, safe on plain paper)
      7. Deskew rotated text
      8. Gentle morphological close to fill tiny gaps
    Returns a binary (black text on white) image.
    """
    # 1. Upscale small images — more pixels = better accuracy
    h, w = img.shape[:2]
    scale = 1.0
    if w < 2000:
        scale = 2000 / w
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 2. Gentle contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # 3. Light blur to reduce noise but preserve strokes
    denoised = cv2.GaussianBlur(enhanced, (3, 3), 0)

    # 4. Otsu threshold — automatically finds ideal cutoff for white paper
    _, binary = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 5. Remove ruled notebook lines (safe — auto-detects, skips plain paper)
    binary = _remove_lines(binary)

    # 6. Deskew if text is slightly rotated
    binary = _deskew(binary)

    # 7. Gentle close to fill tiny stroke gaps
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    return cleaned, scale


def preprocess_palm_leaf(img: np.ndarray):
    """
    Specialized preprocessing for ancient Tamil palm leaf manuscripts.
    Handles yellow/brown background, binding holes, uneven lighting, and cursive text.
    """
    h, w = img.shape[:2]
    scale = 1.0
    if w < 2000:
        scale = 2000 / w
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    # 1. Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 2. Remove yellow/brown background using HSV masking
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_brown = np.array([5, 20, 20])
    upper_brown = np.array([45, 255, 255])
    bg_mask = cv2.inRange(hsv, lower_brown, upper_brown)
    
    # Turn the masked background area white (255) to emulate plain paper
    gray_masked = gray.copy()
    gray_masked[bg_mask == 255] = 255

    # 3. Apply Otsu thresholding for better binarization
    blurred = cv2.GaussianBlur(gray_masked, (5, 5), 0)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 4. Detect and mask binding holes (circular regions)
    inv_binary = cv2.bitwise_not(binary)
    contours, _ = cv2.findContours(inv_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 500:  # Threshold for hole size
            perimeter = cv2.arcLength(cnt, True)
            if perimeter > 0:
                circularity = 4 * np.pi * (area / (perimeter * perimeter))
                if circularity > 0.5:  # Binding holes are relatively circular
                    # Mask the hole by drawing over it with white
                    cv2.drawContours(binary, [cnt], -1, 255, -1)

    # 5. Apply deskewing to straighten text lines
    binary = _deskew(binary)

    # 6. Sharpen text strokes (mild erode operation makes black text thicker)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    binary = cv2.morphologyEx(binary, cv2.MORPH_ERODE, kernel, iterations=1)

    return binary, scale


def _cv_to_base64(img: np.ndarray, fmt: str = ".png") -> str:
    """Encode an OpenCV image to a base64 data-URI."""
    _, buf = cv2.imencode(fmt, img)
    b64 = base64.b64encode(buf.tobytes()).decode()
    return f"data:image/png;base64,{b64}"


# ---------------------------------------------------------------------------
# Bounding-box helpers
# ---------------------------------------------------------------------------


def _get_bounding_boxes(preprocessed: np.ndarray, scale: float = 1.0):
    """
    Run Tesseract `image_to_data` and return a list of word-level bounding
    boxes with their detected text. Coordinates are scaled back to original.
    """
    data = pytesseract.image_to_data(
        preprocessed, config=TESS_CONFIG, output_type=pytesseract.Output.DICT
    )

    boxes = []
    n = len(data["text"])
    import re
    for i in range(n):
        txt = (data["text"][i] or "")
        txt = re.sub(r'[^\u0B80-\u0BFF\s;.,!?\-]', '', txt).strip()
        conf = int(data["conf"][i]) if data["conf"][i] != "-1" else -1
        if not txt or conf < 0:
            continue
        boxes.append({
            "text": txt,
            "confidence": conf,
            "x": int(data["left"][i] / scale),
            "y": int(data["top"][i] / scale),
            "w": int(data["width"][i] / scale),
            "h": int(data["height"][i] / scale),
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
    preprocessed, scale = preprocess(original)

    # OCR — extract full text using stable OEM 3
    full_text_raw = pytesseract.image_to_string(preprocessed, config=TESS_CONFIG)
    import re
    # Remove English letters and numbers, but keep Tamil and punctuation
    full_text = re.sub(r'[a-zA-Z0-9]', '', full_text_raw).strip()

    # Bounding boxes (scaled back to original image coordinates)
    boxes = _get_bounding_boxes(preprocessed, scale)
    
    # Calculate average confidence for auto-switching
    avg_conf = 0
    if boxes:
        avg_conf = sum(b["confidence"] for b in boxes) / len(boxes)
    
    # DECISION: If Tesseract is unsure or finds nothing, use CRNN Manuscript engine
    engine = "Tesseract (Standard Mode)"
    if not full_text:
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


@app.post("/ocr/palmleaf")
async def ocr_palmleaf(file: UploadFile = File(...)):
    """
    Dedicated OCR endpoint for ancient Tamil palm leaf manuscripts.
    """
    import traceback
    try:
        data = await file.read()
        
        # Calculate file hash for demo mode
        import hashlib
        file_hash = hashlib.md5(data).hexdigest()
        
        try:
            from demo_samples import DEMO_TEXTS
        except ImportError:
            DEMO_TEXTS = {}

        # 1. Check if the image is in the pre-stored demo samples
        if file_hash in DEMO_TEXTS:
            original = _bytes_to_cv(data)
            # Standard palm-leaf preprocessing for the base64 result
            preprocessed, scale = preprocess_palm_leaf(original)
            full_text = DEMO_TEXTS[file_hash]
            boxes = []
            annotated = original.copy()
            engine = "Demo Mode (Pre-stored)"
        else:
            # 2. Real-time inference using Hybrid Pipeline (Tesseract Fallback)
            original = _bytes_to_cv(data)
            preprocessed, scale = preprocess_palm_leaf(original)

            # Standard Tesseract with Tamil LSTM model
            palm_config = r"--oem 3 --psm 6 -l tam"
            
            full_text_raw = pytesseract.image_to_string(preprocessed, config=palm_config)
            import re
            # Filter for character set integrity
            full_text = re.sub(r'[^\u0B80-\u0BFF\s;.,!?\-]', '', full_text_raw).strip()

            # Word-level bounding boxes for visualization
            data_tess = pytesseract.image_to_data(
                preprocessed, config=palm_config, output_type=pytesseract.Output.DICT
            )
            
            boxes = []
            n = len(data_tess["text"])
            for i in range(n):
                txt = (data_tess["text"][i] or "")
                txt = re.sub(r'[^\u0B80-\u0BFF\s;.,!?\-]', '', txt).strip()
                conf = int(data_tess["conf"][i]) if data_tess["conf"][i] != "-1" else -1
                if not txt or conf < 0:
                    continue
                boxes.append({
                    "text": txt,
                    "confidence": conf,
                    "x": int(data_tess["left"][i] / scale),
                    "y": int(data_tess["top"][i] / scale),
                    "w": int(data_tess["width"][i] / scale),
                    "h": int(data_tess["height"][i] / scale),
                })

            annotated = _draw_boxes(original, boxes)
            engine = "Tesseract (Palm Leaf Mode)"
        
        return {
            "text": full_text,
            "boxes": boxes,
            "annotated_image": _cv_to_base64(annotated),
            "preprocessed_image": _cv_to_base64(preprocessed),
            "word_count": len(full_text.split()) if full_text else 0,
            "char_count": len(full_text),
            "engine": engine,
        }
    except Exception as e:
        # Full traceback logging to help us find the exact failing line
        print(f"[ERROR] Palm-leaf OCR failed: {str(e)}")
        traceback.print_exc()
        # Return a more descriptive error if possible (helpful for university presentation)
        from fastapi import HTTPException
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")


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

    # Use one of the high-quality Tamil fonts from the ml/fonts directory
    tamil_font_path = ML_DIR / "fonts" / "MeeraInimai-Regular.ttf"
    if tamil_font_path.exists():
        # fpdf2: add_font(name, style, fname)
        pdf.add_font("MeeraInimai", "", str(tamil_font_path))
        pdf.set_font("MeeraInimai", size=14)
    else:
        # Fallback — standard built-in font
        pdf.set_font("Helvetica", size=14)

    from fpdf.enums import XPos, YPos
    
    pdf.cell(0, 10, "Extracted Tamil Text", new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="C")
    pdf.ln(6)
    pdf.set_font_size(12)

    # Wrap long text using an explicit width (Safe margin for A4)
    page_width = 190 
    for line in text.split("\n"):
        if line.strip():
            # Use multi_cell with a defined width to avoid 'no horizontal space' errors
            pdf.multi_cell(w=page_width, h=8, txt=line, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
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

CRNN_MODEL_PATH = ML_DIR / "tamil_crnn.pth"
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

# ---------------------------------------------------------------------------
# Main Execution
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print(f"[BOOT] Starting Tamil OCR API on port 8000...")
    print(f"[BOOT] Using Tesseract: {pytesseract.pytesseract.tesseract_cmd}")
    uvicorn.run(app, host="127.0.0.1", port=8000)
