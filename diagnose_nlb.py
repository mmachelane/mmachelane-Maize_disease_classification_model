"""
Diagnose Cercospora vs NLB confusion.
Runs ALL field images for both classes, reports confidence + per-source breakdown.
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

CLASSES_OF_INTEREST = [
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
    'Corn_(maize)___Northern_Leaf_Blight',
]
SHORT = {
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot': 'Cercospora',
    'Corn_(maize)___Common_rust_':                        'Common Rust',
    'Corn_(maize)___healthy':                             'Healthy',
    'Corn_(maize)___Northern_Leaf_Blight':                'N. Leaf Blight',
}

with open(NAMES_PATH) as f:
    class_names = json.load(f)

print("Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)
print("Loaded.\n")

def eval_folder(folder, true_cls):
    true_idx = class_names.index(true_cls)
    imgs = [os.path.join(folder, f) for f in os.listdir(folder)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    correct, total = 0, 0
    conf_correct, conf_wrong = [], []
    pred_counts = {SHORT[c]: 0 for c in class_names}

    for path in imgs:
        try:
            img  = Image.open(path).convert('RGB').resize(IMG_SIZE)
            arr  = np.expand_dims(np.array(img, dtype=np.float32), 0)
            pred = model.predict(arr, verbose=0)[0]
            pred_idx  = int(np.argmax(pred))
            confidence = float(pred[pred_idx]) * 100
            pred_counts[SHORT[class_names[pred_idx]]] += 1
            total += 1
            if pred_idx == true_idx:
                correct += 1
                conf_correct.append(confidence)
            else:
                conf_wrong.append(confidence)
        except Exception:
            pass

    return {
        'correct': correct, 'total': total,
        'avg_conf_correct': np.mean(conf_correct) if conf_correct else 0,
        'avg_conf_wrong':   np.mean(conf_wrong)   if conf_wrong   else 0,
        'pred_counts': pred_counts,
    }

for cls in CLASSES_OF_INTEREST:
    print(f"{'='*55}")
    print(f"TRUE CLASS: {SHORT[cls]}")
    print(f"{'='*55}")

    for source_name, base_dir in [('PlantVillage', PV_DIR), ('Field', FIELD_DIR)]:
        folder = os.path.join(base_dir, cls)
        if not os.path.isdir(folder):
            continue
        r = eval_folder(folder, cls)
        acc = r['correct'] / r['total'] * 100 if r['total'] else 0
        print(f"\n  [{source_name}]  {r['correct']}/{r['total']} correct ({acc:.1f}%)")
        print(f"  Avg confidence when correct: {r['avg_conf_correct']:.1f}%")
        print(f"  Avg confidence when wrong:   {r['avg_conf_wrong']:.1f}%")
        print(f"  Predicted as:")
        for label, count in sorted(r['pred_counts'].items(), key=lambda x: -x[1]):
            if count > 0:
                bar = '█' * int(count / r['total'] * 30)
                print(f"    {label:18s}: {count:4d}  {bar}")
    print()

# Overall class imbalance summary
print(f"{'='*55}")
print("FIELD DATA IMBALANCE")
print(f"{'='*55}")
for cls in class_names:
    folder = os.path.join(FIELD_DIR, cls)
    n = len([f for f in os.listdir(folder) if f.lower().endswith(('.jpg','.jpeg','.png'))]) if os.path.isdir(folder) else 0
    bar = '█' * (n // 20)
    print(f"  {SHORT[cls]:18s}: {n:4d}  {bar}")
