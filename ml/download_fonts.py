"""
Download free Tamil-capable TTF fonts from Google Fonts.
Saves them into ml/fonts/ for synthetic dataset generation.
"""

import os
import urllib.request
import zipfile
import io
from pathlib import Path

FONTS_DIR = Path(__file__).parent / "fonts"

# Google Fonts API download links (direct TTF zip URLs)
GOOGLE_FONTS = {
    "NotoSansTamil": "https://github.com/google/fonts/raw/main/ofl/notosanstamil/NotoSansTamil%5Bwdth%2Cwght%5D.ttf",
    "NotoSerifTamil": "https://github.com/google/fonts/raw/main/ofl/notoseriftamil/NotoSerifTamil%5Bwdth%2Cwght%5D.ttf",
    "Catamaran": "https://github.com/google/fonts/raw/main/ofl/catamaran/Catamaran%5Bwght%5D.ttf",
    "HindMadurai": "https://github.com/google/fonts/raw/main/ofl/hindmadurai/HindMadurai-Regular.ttf",
    "MuktaMalar": "https://github.com/google/fonts/raw/main/ofl/muktamalar/MuktaMalar-Regular.ttf",
    "Ponnala": "https://github.com/google/fonts/raw/main/ofl/ponnala/Ponnala-Regular.ttf",
    "MeeraInimai": "https://github.com/google/fonts/raw/main/ofl/meerainimai/MeeraInimai-Regular.ttf",
    "Anek_Tamil": "https://github.com/google/fonts/raw/main/ofl/anektamil/AnekTamil%5Bwdth%2Cwght%5D.ttf",
    "NotoSansBrahmi": "https://github.com/google/fonts/raw/main/ofl/notosansbrahmi/NotoSansBrahmi-Regular.ttf",
    "NotoSansGrantha": "https://github.com/google/fonts/raw/main/ofl/notosansgrantha/NotoSansGrantha-Regular.ttf",
}


def download_fonts():
    FONTS_DIR.mkdir(parents=True, exist_ok=True)

    downloaded = 0
    for name, url in GOOGLE_FONTS.items():
        filename = url.split("/")[-1]
        # URL decode %5B -> [ and %5D -> ] and %2C -> ,
        filename = filename.replace("%5B", "[").replace("%5D", "]").replace("%2C", ",")
        dest = FONTS_DIR / filename

        if dest.exists():
            print(f"  [skip] {filename} (already exists)")
            continue

        print(f"  [downloading] {name} -> {filename}")
        try:
            urllib.request.urlretrieve(url, str(dest))
            downloaded += 1
            print(f"  [ok] {filename} ({dest.stat().st_size / 1024:.0f} KB)")
        except Exception as e:
            print(f"  [FAIL] {name}: {e}")

    print(f"\nDone. {downloaded} fonts downloaded to {FONTS_DIR}")
    print(f"Total fonts available: {len(list(FONTS_DIR.glob('*.ttf')))}")


if __name__ == "__main__":
    download_fonts()
