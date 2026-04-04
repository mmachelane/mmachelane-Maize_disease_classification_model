"""
Quick evaluation: 100 images (25 per class), mixed PV + field samples.
Run with: python quick_eval.py
"""
import os, json, random
import numpy as np
from PIL import Image

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

REPO_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(REPO_DIR, 'models', 'best_model_joint.keras')
NAMES_PATH = os.path.join(REPO_DIR, 'models', 'class_names.json')
PV_DIR     = os.path.join(REPO_DIR, 'data', 'maize_only')
FIELD_DIR  = os.path.join(REPO_DIR, 'data', 'maize_in_field')

IMG_SIZE   = (224, 224)
PER_CLASS  = 25
SEED       = 99

random.seed(SEED)

with open(NAMES_PATH) as f:
    class_names = json.load(f)

SHORT = {
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot': 'Cercospora',
    'Corn_(maize)___Common_rust_':                        'Common Rust',
    'Corn_(maize)___healthy':                             'Healthy',
    'Corn_(maize)___Northern_Leaf_Blight':                'N. Leaf Blight',
}

print(f"Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)
print(f"Model loaded.\n")

# Collect images: mix PV (15) + field (10) per class where available
samples = []  # (path, true_label_idx)

for idx, cls in enumerate(class_names):
    pv_cls    = os.path.join(PV_DIR, cls)
    field_cls = os.path.join(FIELD_DIR, cls)

    pv_imgs    = [os.path.join(pv_cls, f)    for f in os.listdir(pv_cls)    if f.lower().endswith(('.jpg','.jpeg','.png'))] if os.path.isdir(pv_cls)    else []
    field_imgs = [os.path.join(field_cls, f) for f in os.listdir(field_cls) if f.lower().endswith(('.jpg','.jpeg','.png'))] if os.path.isdir(field_cls) else []

    n_field = min(10, len(field_imgs))
    n_pv    = PER_CLASS - n_field

    picked = random.sample(pv_imgs, min(n_pv, len(pv_imgs))) + random.sample(field_imgs, n_field)
    random.shuffle(picked)
    samples.extend([(p, idx) for p in picked])
    print(f"  {SHORT[cls]:18s}: {len(pv_imgs):4d} PV  +  {len(field_imgs):3d} field  →  sampled {len(picked)} ({n_pv} PV + {n_field} field)")

print(f"\nTotal samples: {len(samples)}\n")

# Run inference
correct = 0
per_class_correct = {i: 0 for i in range(len(class_names))}
per_class_total   = {i: 0 for i in range(len(class_names))}
confusion = np.zeros((len(class_names), len(class_names)), dtype=int)

for path, true_idx in samples:
    img  = Image.open(path).convert('RGB').resize(IMG_SIZE)
    arr  = np.array(img, dtype=np.float32)
    arr  = np.expand_dims(arr, 0)
    pred = model.predict(arr, verbose=0)[0]
    pred_idx = int(np.argmax(pred))

    confusion[true_idx][pred_idx] += 1
    per_class_total[true_idx] += 1
    if pred_idx == true_idx:
        correct += 1
        per_class_correct[true_idx] += 1

# Results
print("=" * 50)
print(f"  OVERALL ACCURACY: {correct}/{len(samples)} = {correct/len(samples)*100:.1f}%")
print("=" * 50)

print("\nPer-class accuracy:")
for i, cls in enumerate(class_names):
    t = per_class_total[i]
    c = per_class_correct[i]
    bar = '█' * c + '░' * (t - c)
    print(f"  {SHORT[cls]:18s}: {c:2d}/{t}  {bar}  {c/t*100:.0f}%")

print("\nConfusion matrix (rows=actual, cols=predicted):")
labels = [SHORT[c][:10] for c in class_names]
header = f"{'':18s}" + "".join(f"{l:>12s}" for l in labels)
print(header)
for i, cls in enumerate(class_names):
    row = f"  {SHORT[cls]:16s}" + "".join(f"{confusion[i][j]:>12d}" for j in range(len(class_names)))
    print(row)
