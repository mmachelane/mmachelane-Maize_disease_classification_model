"""
Merge PlantDoc corn images into data/plantdoc_maize/ using our class naming.
PlantDoc mapping:
  Corn Gray leaf spot  → Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot
  Corn leaf blight     → Corn_(maize)___Northern_Leaf_Blight
  Corn rust leaf       → Corn_(maize)___Common_rust_
  (no healthy class in PlantDoc corn subset)

Run: python prepare_plantdoc.py
"""

import os, shutil, json
from pathlib import Path

REPO_DIR    = Path(os.path.dirname(os.path.abspath(__file__)))
PLANTDOC    = REPO_DIR / 'data' / 'plantdoc' / 'train'
OUT_DIR     = REPO_DIR / 'data' / 'plantdoc_maize'
NAMES_PATH  = REPO_DIR / 'models' / 'class_names.json'

MAPPING = {
    'Corn Gray leaf spot': 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
    'Corn leaf blight':    'Corn_(maize)___Northern_Leaf_Blight',
    'Corn rust leaf':      'Corn_(maize)___Common_rust_',
}

with open(NAMES_PATH) as f:
    class_names = json.load(f)

# Create output dirs for all classes (even ones with no PlantDoc data)
for cls in class_names:
    (OUT_DIR / cls).mkdir(parents=True, exist_ok=True)

total = 0
for src_name, dst_name in MAPPING.items():
    src = PLANTDOC / src_name
    dst = OUT_DIR / dst_name
    if not src.is_dir():
        print(f"WARNING: {src} not found — skipping")
        continue
    imgs = list(src.glob('*'))
    imgs = [p for p in imgs if p.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp'}]
    for p in imgs:
        shutil.copy2(p, dst / p.name)
    print(f"  {src_name} → {dst_name}: {len(imgs)} images")
    total += len(imgs)

print(f"\nTotal PlantDoc corn images merged: {total}")
print(f"Output: {OUT_DIR}")
print("\nClass counts:")
for cls in class_names:
    n = len(list((OUT_DIR / cls).glob('*')))
    print(f"  {cls}: {n}")
