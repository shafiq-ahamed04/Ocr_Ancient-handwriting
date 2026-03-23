"""
Synthetic Ancient Tamil Text Dataset Generator
===============================================
Generates line-level images of Tamil text with realistic palm-leaf-like
backgrounds and various augmentations, paired with ground-truth label files.

Usage:
    cd ml
    python generate_dataset.py --count 5000 --out dataset/synthetic
"""

import argparse
import math
import os
import random
import textwrap
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance

# ---------------------------------------------------------------------------
# Tamil corpus — classical Tamil sentences/phrases for rendering
# ---------------------------------------------------------------------------
TAMIL_CORPUS = [
    "அகர முதல எழுத்தெல்லாம் ஆதி பகவன் முதற்றே உலகு",
    "கற்றதனால் ஆய பயனென்கொல் வாலறிவன் நற்றாள் தொழாஅர் எனின்",
    "மலர்மிசை ஏகினான் மாணடி சேர்ந்தார் நிலமிசை நீடுவாழ்வார்",
    "வேண்டுதல் வேண்டாமை இலானடி சேர்ந்தார்க்கு யாண்டும் இடும்பை இல",
    "இருள்சேர் இருவினையும் சேரா இறைவன் பொருள்சேர் புகழ்புரிந்தார் மாட்டு",
    "பொறிவாயில் ஐந்தவித்தான் பொய்தீர் ஒழுக்க நெறிநின்றார் நீடுவாழ்வார்",
    "தனக்குவமை இல்லாதான் தாள்சேர்ந்தார்க் கல்லால் மனக்கவலை மாற்றல் அரிது",
    "அறவாழி அந்தணன் தாள்சேர்ந்தார்க் கல்லால் பிறவாழி நீந்தல் அரிது",
    "கோளில் பொறியிற் குணமிலவே எண்குணத்தான் தாளை வணங்காத் தலை",
    "பிறவிப் பெருங்கடல் நீந்துவர் நீந்தார் இறைவன் அடிசேரா தார்",
    "வான்நின்று உலகம் வழங்கி வருதலால் தான்அமிழ்தம் என்றுணரற் பாற்று",
    "துப்பார்க்குத் துப்பாய துப்பாக்கித் துப்பார்க்குத் துப்பாய தூஉம் மழை",
    "நீர்இன்று அமையாது உலகெனின் யார்யார்க்கும் வான்இன்று அமையாது ஒழுக்கு",
    "ஒழுக்கம் விழுப்பம் தரலான் ஒழுக்கம் உயிரினும் ஓம்பப் படும்",
    "பரிந்தோம்பிக் காக்க ஒழுக்கம் தெரிந்தோம்பித் தேரினும் அஃதே துணை",
    "அன்பும் அறனும் உடைத்தாயின் இல்வாழ்க்கை பண்பும் பயனும் அது",
    "அறத்தாற்றின் இல்வாழ்க்கை ஆற்றின் புறத்தாற்றிற் போஒய்ப் பெறுவது எவன்",
    "இல்வாழ்வான் என்பான் இயல்புடைய மூவர்க்கும் நல்லாற்றின் நின்ற துணை",
    "அறனெனப் பட்டதே இல்வாழ்க்கை அஃதும் பிறன்பழிப்பது இல்லாயின் நன்று",
    "வையத்துள் வாழ்வாங்கு வாழ்பவன் வானுறையும் தெய்வத்துள் வைக்கப் படும்",
    "செய்யாமல் செய்த உதவிக்கு வையகமும் வானகமும் ஆற்றல் அரிது",
    "காலத்தி னால்செய்த நன்றி சிறிதெனினும் ஞாலத்தின் மாணப் பெரிது",
    "பயன்தூக்கார் செய்த உதவி நயன்தூக்கின் நன்மை கடலின் பெரிது",
    "தினைத்துணை நன்றி செயினும் பனைத்துணையாக் கொள்வர் பயன்தெரி வார்",
    "உத்தமனாய்க் கற்றுணர்ந்து ஒண்பொருள் செய்து ஈவான் எத்துணையும் இல்லாத இல்",
    "காணி நிலம் போதுமென மனம் கொள்ளாரோ யாருமிவ்வையகத்தில்",
    "தமிழுக்கும் அமுதென்று பேர் அந்தத் தமிழ் இன்பத் தமிழ்எங்கள் உயிருக்கு நேர்",
    "யாமறிந்த மொழிகளிலே தமிழ்மொழி போல் இனிதாவது எங்கும் காணோம்",
    "செந்தமிழ் நாடெனும் போதினிலே இன்பத் தேன்வந்து பாயுது காதினிலே",
    "நிலாவை ரசிக்கும் போது உன் நினைவு வருகிறது",
    "இயற்கை மூலிகைகளில் இருந்து எடுத்துக்கொள்ளப்பட்ட மருந்துகள் நோய்களை குணப்படுத்தும்",
    "பழந்தமிழ் இலக்கியங்கள் நமது பண்பாட்டின் அடையாளம்",
    "கல்வி கற்றவர் எந்த நாட்டிலும் மதிக்கப்படுவர்",
    "அறிவியல் வளர்ச்சி மனித வாழ்வை மேம்படுத்தியுள்ளது",
    "தமிழ் இலக்கணம் மிகவும் செழுமையானது",
    "சங்க இலக்கியம் தமிழின் பழமையை உணர்த்துகிறது",
    "திருக்குறள் உலகப் பொதுமறை ஆகும்",
    "வள்ளுவர் வாக்கு அனைத்து காலத்திற்கும் பொருந்தும்",
    "கடல் கடந்தும் தமிழர் புகழ் பரவியுள்ளது",
    "தமிழ் மொழி இரண்டாயிரம் ஆண்டுகளுக்கு மேல் பழமையானது",
    "பண்டைய தமிழகத்தில் கல்வி சிறப்பாக இருந்தது",
    "ஓலைச்சுவடிகள் பண்டைய அறிவின் சான்றுகள்",
    "தமிழர் கலாச்சாரம் உலகின் பழமையான கலாச்சாரங்களில் ஒன்று",
    "பண்டைய தமிழ் நாகரிகம் சிந்துவெளி நாகரிகத்துடன் தொடர்புடையது",
    "தொல்காப்பியம் தமிழின் முதல் இலக்கண நூல்",
    "சிலப்பதிகாரம் தமிழின் சிறந்த காவியங்களில் ஒன்று",
    "மணிமேகலை ஒரு பௌத்த காவியம் ஆகும்",
    "கம்பராமாயணம் தமிழின் பெருங்காவியம்",
    "பெரிய புராணம் சைவ சமய வரலாற்று நூல்",
    "நாலாயிர திவ்ய பிரபந்தம் வைணவ இலக்கியம்",
    "தேவாரம் சைவ சமய பக்தி இலக்கியம்",
    "திருவாசகம் மாணிக்கவாசகர் அருளிய நூல்",
    "குறிஞ்சிப்பாட்டு சங்க கால இலக்கியம்",
    "புறநானூறு வீரம் பற்றிய சங்க இலக்கியம்",
    "அகநானூறு காதல் பற்றிய சங்க இலக்கியம்",
    "நற்றிணை ஐந்திணை பற்றிய பாடல்கள்",
    "குறுந்தொகை சிறிய அகப்பாடல்கள்",
    "ஐங்குறுநூறு ஐந்திணைத் தொகுப்பு",
    "பதிற்றுப்பத்து சேர மன்னர்களைப் பாடியது",
    "கலித்தொகை கலிப்பா வகை இலக்கியம்",
    "பரிபாடல் பரிபாடல் வகை இலக்கியம்",
    "செல்வதற்கு இதுவே சரியான நேரம்",
    "இங்கே செல்வதற்கு வழி என்ன?",
    "அவன் ஊருக்குச் செல்வதற்குத் தயாரானான்",
    "படிப்பதற்குப் புத்தகங்கள் தேவை",
    "அனைவருக்கும் வணக்கம்",
    "தமிழ் மொழி இனிமையானது",
    "வாழ்க தமிழ் மக்கள்",
    "யாதும் ஊரே யாவரும் கேளிர்",
    "தீதும் நன்றும் பிறர்தர வாரா",
    "பெரியோரை வியத்தலும் இலமே",
    "சிறியோரை இகழ்தல் அதனினும் இலமே",
    "யான்மகன் அல்லேன் நீயென் மகனே",
    "அறம் எனப்படுவது யாதெனக் கேட்பின்",
    "மறவாதே இதுவே தமிழர் பண்பு",
    "உண்டி கொடுத்தோர் உயிர் கொடுத்தோரே",
    "மன்னனும் மாசறக் கற்றோனும் சீர்தூக்கின்",
    "கல்வி கரையில கற்பவர் நாள்சில",
    "யாதானும் நாடாமல் ஊராமால் என்னொருவன்",
    "சாந்துயும் கல்லாத வாறு",
    "பொன்னும் மெய்ப்பொருளும் தருபவன்",
    "இன்பமும் துன்பமும் இல்லா நிலை",
    "அன்பே சிவம் என அறிவிலார் கூறுவர்",
    "மந்திரம் ஆவது நீறு வானவர் மேலது நீறு",
    "தோடுடைய செவியன் விடையேறியோர் தூவெண் மதிசூடி",
    "நெஞ்சைக் அள்ளும் சிலப்பதிகாரம் என்றோர்",
    "மணியாரம் படைத்த தமிழ்",
    "தமிழுக்கும் அமுதென்று பேர்",
    "கண்டு கேட்டு உண்டு உயிர்த்து உற்றறியும்",
    "ஐம்புலனும் ஒண்டொடி கண்ணே உள",
    "உடுக்கை இழந்தவன் கைபோல ஆங்கே",
    "இடுக்கண் களைவதாம் நட்பு",
    "பல்லார் முனியப் பயிறல் அதனினும்",
    "நல்லார் எனப்படுவார் யார்",
    "உளவரை தூக்காதார் வாழ்க்கை",
    "வளர்வது போலக் கெடும்",
]

# Full Tamil Unicode character set for the CRNN vocabulary
TAMIL_CHARS = (
    "அஆஇஈஉஊஎஏஐஒஓஔஃ"
    "கஙசஞடணதநபமயரலவழளறன"
    "ாிீுூெேைொோௌ்"
    "ஜஷஸஹ"
    "க்ஷஸ்ரீ"
)


# ---------------------------------------------------------------------------
# Background textures
# ---------------------------------------------------------------------------

def _palm_leaf_bg(w: int, h: int) -> np.ndarray:
    """Generate a synthetic palm-leaf-like background."""
    # Start with a brownish base colour
    base_r = random.randint(180, 220)
    base_g = random.randint(160, 195)
    base_b = random.randint(100, 140)
    bg = np.full((h, w, 3), [base_b, base_g, base_r], dtype=np.uint8)

    # Add horizontal grain lines (palm leaf veins)
    num_lines = random.randint(3, 8)
    for _ in range(num_lines):
        y = random.randint(0, h - 1)
        thickness = random.randint(1, 2)
        darkness = random.randint(15, 40)
        cv2.line(
            bg,
            (0, y),
            (w, y + random.randint(-2, 2)),
            (base_b - darkness, base_g - darkness, base_r - darkness),
            thickness,
        )

    # Gaussian noise for texture
    noise = np.random.normal(0, 8, bg.shape).astype(np.int16)
    bg = np.clip(bg.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    # Slight blur to blend
    bg = cv2.GaussianBlur(bg, (3, 3), 0)

    return bg


def _paper_bg(w: int, h: int) -> np.ndarray:
    """Generate a aged paper background."""
    base = random.randint(220, 245)
    bg = np.full((h, w, 3), [base - 10, base - 5, base], dtype=np.uint8)
    noise = np.random.normal(0, 5, bg.shape).astype(np.int16)
    bg = np.clip(bg.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return bg


def _stone_bg(w: int, h: int) -> np.ndarray:
    """Generate a stone inscription background."""
    base = random.randint(140, 180)
    bg = np.full((h, w, 3), [base, base + 5, base + 10], dtype=np.uint8)
    noise = np.random.normal(0, 12, bg.shape).astype(np.int16)
    bg = np.clip(bg.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    bg = cv2.GaussianBlur(bg, (5, 5), 0)
    return bg


import glob

REAL_BGS = glob.glob(r"C:\final year Project\ml\dataset\THPLMD\THIRIKADUGAM_ORIG\THIRIKADUGAM ORIGINAL\*.jpg")

def _real_palm_leaf_bg(w: int, h: int) -> np.ndarray:
    """Extract a random patch from a real palm leaf dataset."""
    if not REAL_BGS:
        return _palm_leaf_bg(w, h)
        
    bg_path = random.choice(REAL_BGS)
    try:
        img = cv2.imread(bg_path)
        if img is None: return _palm_leaf_bg(w, h)
        
        # Real images are huge (e.g., 4000x3000). A line of text in real life might be 100px high in that image.
        # Let's scale the real image down so it roughly matches our 64px height scale.
        scale = random.uniform(0.1, 0.3)
        img = cv2.resize(img, (0,0), fx=scale, fy=scale)
        
        ih, iw = img.shape[:2]
        
        # If resized image is smaller than needed, just resize it to fit
        if ih < h or iw < w:
            img = cv2.resize(img, (max(iw, w), max(ih, h)))
            ih, iw = img.shape[:2]
            
        # Random crop
        y = random.randint(0, ih - h)
        x = random.randint(0, iw - w)
        
        crop = img[y:y+h, x:x+w]
        
        # Add slight variation
        if random.random() > 0.5:
            # Random brightness/contrast
            alpha = random.uniform(0.8, 1.2)
            beta = random.randint(-20, 20)
            crop = cv2.convertScaleAbs(crop, alpha=alpha, beta=beta)
            
        return crop
    except Exception:
        return _palm_leaf_bg(w, h)

def get_random_background(w: int, h: int) -> np.ndarray:
    """Return a random background texture, heavily favoring real ones."""
    # 80% chance for real palm leaf, 20% for synthetic
    if random.random() < 0.8 and REAL_BGS:
        return _real_palm_leaf_bg(w, h)
    
    fn = random.choice([_palm_leaf_bg, _paper_bg, _stone_bg])
    return fn(w, h)


# ---------------------------------------------------------------------------
# Advanced Handwriting Augmentations
# ---------------------------------------------------------------------------

def apply_elastic_transform(img_np: np.ndarray, alpha: float = 80, sigma: float = 4) -> np.ndarray:
    """
    Apply elastic deformation to an image to simulate handwriting instability.
    alpha: Magnitude of distortion
    sigma: Smoothness of distortion
    """
    from scipy.ndimage import gaussian_filter, map_coordinates
    
    shape = img_np.shape
    dx = gaussian_filter((np.random.rand(*shape[:2]) * 2 - 1), sigma) * alpha
    dy = gaussian_filter((np.random.rand(*shape[:2]) * 2 - 1), sigma) * alpha
    
    x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1))
    
    result = np.zeros_like(img_np)
    for i in range(shape[2]):
        result[:, :, i] = map_coordinates(img_np[:, :, i], indices, order=1).reshape(shape[:2])
    
    return result


def apply_slant(img_np: np.ndarray, max_slant: float = 0.25) -> np.ndarray:
    """Apply horizontal shear to simulate slanted handwriting."""
    slant = random.uniform(-max_slant, max_slant)
    h, w = img_np.shape[:2]
    M = np.float32([[1, slant, 0], [0, 1, 0]])
    new_w = int(w + abs(slant * h))
    # Adjust matrix to keep centered
    if slant > 0:
        M[0, 2] = 0
    else:
        M[0, 2] = abs(slant * h)
        
    return cv2.warpAffine(img_np, M, (new_w, h), borderMode=cv2.BORDER_REPLICATE)


def apply_ink_bleed(img_np: np.ndarray) -> np.ndarray:
    """Simulate ink spreading/smudging into paper textures."""
    # Convert to gray for processing
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    
    # Create an ink mask (darker pixels)
    _, mask = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)
    
    # Randomly smudge the mask
    kernel_size = random.choice([3, 5])
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    
    # Dilation makes the ink "bleed" outward
    bleeded_mask = cv2.dilate(mask, kernel, iterations=1)
    
    # Blend the bleed back into the original using a noisy factor
    noise = np.random.randint(0, 50, gray.shape, dtype=np.uint8)
    bleed_effect = cv2.bitwise_and(bleeded_mask, noise)
    
    # Overlay the bleed effect
    res = img_np.copy()
    for i in range(3):
        res[:, :, i] = cv2.subtract(res[:, :, i], bleed_effect // 2)
        
    return res


# ---------------------------------------------------------------------------
# Augmentation
# ---------------------------------------------------------------------------

def augment_image(pil_img: Image.Image) -> Image.Image:
    """Apply random degradation effects to simulate ancient documents."""
    # Convert to NumPy for advanced CV2 operations
    arr = np.array(pil_img)

    # 1. Random Slant (Shear) - HIGH priority for handwriting simulation
    if random.random() < 0.6:
        arr = apply_slant(arr, max_slant=0.3)

    # 2. Elastic Transform - Simulates shaky hands
    if random.random() < 0.4:
        try:
            arr = apply_elastic_transform(arr, alpha=random.uniform(30, 60), sigma=random.uniform(3, 5))
        except ImportError:
            # Skip if scipy isn't available
            pass

    # 3. Ink Bleed / Smudge
    if random.random() < 0.4:
        arr = apply_ink_bleed(arr)

    # Convert back to PIL for standard effects
    img = Image.fromarray(arr)

    # Random contrast
    if random.random() < 0.5:
        factor = random.uniform(0.7, 1.4)
        img = ImageEnhance.Contrast(img).enhance(factor)

    # Random blur (simulate focus issues or worn text)
    if random.random() < 0.3:
        radius = random.uniform(0.5, 1.5)
        img = img.filter(ImageFilter.GaussianBlur(radius=radius))

    # Random slight rotation
    if random.random() < 0.5:
        angle = random.uniform(-3, 3)
        # Use a more natural fill color for manuscripts
        img = img.rotate(angle, fillcolor=(180, 160, 120), expand=False)

    # Random noise overlay
    if random.random() < 0.4:
        arr = np.array(img)
        noise = np.random.normal(0, random.randint(4, 12), arr.shape).astype(np.int16)
        arr = np.clip(arr.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        img = Image.fromarray(arr)

    return img


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def render_tamil_line(
    text: str,
    font: ImageFont.FreeTypeFont,
    img_height: int = 64,
    padding: int = 10,
) -> tuple[Image.Image, str]:
    """
    Render a single line of Tamil text on a random background.
    Returns (PIL image, ground truth text).
    """
    # Measure text size
    dummy = Image.new("RGB", (1, 1))
    draw = ImageDraw.Draw(dummy)
    bbox = draw.textbbox((0, 0), text, font=font)
    text_w = bbox[2] - bbox[0] + padding * 2
    text_h = bbox[3] - bbox[1] + padding * 2

    # Scale to target height
    scale = img_height / max(text_h, 1)
    render_h = max(text_h, img_height)
    render_w = max(text_w, 100)

    # Create background
    bg_np = get_random_background(render_w, render_h)
    img = Image.fromarray(cv2.cvtColor(bg_np, cv2.COLOR_BGR2RGB))

    # Draw text
    draw = ImageDraw.Draw(img)

    # Random ink colour (dark brown to black)
    ink_r = random.randint(10, 70)
    ink_g = random.randint(5, 50)
    ink_b = random.randint(0, 40)
    ink_colour = (ink_r, ink_g, ink_b)

    # Random alpha for faded text effect
    text_y = (render_h - (bbox[3] - bbox[1])) // 2 - bbox[1]
    draw.text((padding, text_y), text, font=font, fill=ink_colour)

    # Resize to standard height
    final_w = int(render_w * (img_height / render_h))
    final_w = max(final_w, 32)
    img = img.resize((final_w, img_height), Image.LANCZOS)

    # Apply augmentation
    img = augment_image(img)

    return img, text


# ---------------------------------------------------------------------------
# Dataset generation
# ---------------------------------------------------------------------------

def load_fonts(fonts_dir: Path, sizes: list[int] | None = None) -> list[ImageFont.FreeTypeFont]:
    """Load all .ttf fonts from the fonts directory at various sizes."""
    if sizes is None:
        sizes = [28, 32, 36, 40, 44]

    fonts = []
    font_files = list(fonts_dir.glob("*.ttf"))

    if not font_files:
        print(f"[WARNING] No .ttf fonts found in {fonts_dir}")
        print("  Run: python download_fonts.py")
        # Fallback: try system Nirmala font
        nirmala = Path("C:/Windows/Fonts/Nirmala.ttc")
        if nirmala.exists():
            print(f"  Using fallback: {nirmala}")
            for size in sizes:
                try:
                    fonts.append(ImageFont.truetype(str(nirmala), size))
                except Exception:
                    pass
        return fonts

    for fp in font_files:
        for size in sizes:
            try:
                f = ImageFont.truetype(str(fp), size)
                # Test that it can actually render Tamil
                f.getbbox("அ")
                fonts.append(f)
            except Exception:
                pass

    print(f"Loaded {len(fonts)} font variants from {len(font_files)} font files")
    return fonts


def generate_dataset(
    output_dir: str = "dataset/synthetic",
    count: int = 5000,
    img_height: int = 64,
    train_ratio: float = 0.8,
):
    """Generate the full synthetic dataset."""
    out = Path(output_dir)
    train_dir = out / "train"
    val_dir = out / "val"
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)

    fonts_dir = Path(__file__).parent / "fonts"
    fonts = load_fonts(fonts_dir)

    if not fonts:
        print("ERROR: No fonts available. Cannot generate dataset.")
        print("Run: python download_fonts.py")
        return

    corpus = TAMIL_CORPUS.copy()

    # Also generate sub-phrases for variety
    expanded_corpus = []
    for sentence in corpus:
        expanded_corpus.append(sentence)
        words = sentence.split()
        # Create 2-5 word fragments
        if len(words) >= 3:
            for _ in range(2):
                start = random.randint(0, max(0, len(words) - 3))
                length = random.randint(2, min(5, len(words) - start))
                expanded_corpus.append(" ".join(words[start : start + length]))

    print(f"Corpus size: {len(expanded_corpus)} unique text segments")
    print(f"Generating {count} images at height={img_height}px ...")

    for i in range(count):
        text = random.choice(expanded_corpus)
        font = random.choice(fonts)

        try:
            img, gt = render_tamil_line(text, font, img_height=img_height)
        except Exception as e:
            print(f"  [skip] #{i}: {e}")
            continue

        # Train/val split
        if random.random() < train_ratio:
            dest = train_dir
        else:
            dest = val_dir

        img_path = dest / f"{i:06d}.png"
        lbl_path = dest / f"{i:06d}.txt"

        img.save(str(img_path))
        lbl_path.write_text(gt, encoding="utf-8")

        if (i + 1) % 500 == 0:
            print(f"  Generated {i + 1}/{count}")

    train_count = len(list(train_dir.glob("*.png")))
    val_count = len(list(val_dir.glob("*.png")))
    print(f"\nDone! Train: {train_count}, Val: {val_count}")
    print(f"  Train dir: {train_dir}")
    print(f"  Val dir:   {val_dir}")

    # Write character vocabulary file
    all_chars = set()
    for text in expanded_corpus:
        all_chars.update(text)
    all_chars.discard(" ")  # space handled separately
    vocab = sorted(all_chars)
    vocab_path = out / "vocab.txt"
    vocab_path.write_text("\n".join(vocab), encoding="utf-8")
    print(f"  Vocab: {len(vocab)} chars -> {vocab_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic Tamil OCR dataset")
    parser.add_argument("--count", type=int, default=5000, help="Number of samples")
    parser.add_argument("--out", type=str, default="dataset/synthetic", help="Output dir")
    parser.add_argument("--height", type=int, default=64, help="Image height in px")
    args = parser.parse_args()

    generate_dataset(output_dir=args.out, count=args.count, img_height=args.height)
