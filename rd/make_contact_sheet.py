import os
import math
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
REFINED_DIR = ROOT / "outputs" / "sweeps" / "refined_sweep_outputs"
INPUT_DIR = str(REFINED_DIR)
OUTPUT_FILE = REFINED_DIR / "refined_contact_sheet.png"

image_files = [
    f for f in os.listdir(INPUT_DIR)
    if f.endswith(".png") and f != "contact_sheet.png"
]
image_files.sort()

if not image_files:
    raise ValueError(f"No PNG files found in {INPUT_DIR}")

thumb_w, thumb_h = 220, 220
label_h = 30
cols = 4
rows = math.ceil(len(image_files) / cols)

sheet_w = cols * thumb_w
sheet_h = rows * (thumb_h + label_h)

sheet = Image.new("RGB", (sheet_w, sheet_h), "white")
draw = ImageDraw.Draw(sheet)

for idx, filename in enumerate(image_files):
    path = os.path.join(INPUT_DIR, filename)
    img = Image.open(path).convert("RGB")
    img = img.resize((thumb_w, thumb_h))

    x = (idx % cols) * thumb_w
    y = (idx // cols) * (thumb_h + label_h)

    sheet.paste(img, (x, y))
    draw.text((x + 5, y + thumb_h + 5), filename.replace(".png", ""), fill="black")

sheet.save(OUTPUT_FILE)
print(f"Saved contact sheet to {OUTPUT_FILE}")