"""
Temperature scaling calibration.
Finds optimal temperature T on the validation set so model confidence
actually reflects accuracy. T > 1 = model was overconfident (reduce confidence).

Run AFTER training: python calibrate.py
Saves: models/temperature.json
"""

import os, json
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.applications.efficientnet import preprocess_input as eff_prep
from scipy.optimize import minimize_scalar

REPO_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_WEIGHTS = os.path.join(REPO_DIR, 'models', 'best_model_joint_phaseB.weights.h5')
NAMES_PATH = os.path.join(REPO_DIR, 'models', 'class_names.json')
MAIZE_DIR  = os.path.join(REPO_DIR, 'data', 'maize_only')
FIELD_DIR  = os.path.join(REPO_DIR, 'data', 'maize_in_field')
TEMP_PATH  = os.path.join(REPO_DIR, 'models', 'temperature.json')

IMG_SIZE   = (224, 224)
BATCH_SIZE = 32
SEED       = 42

from model_utils import build_model, load_weights_h5, build_logit_model
import json as _json
with open(NAMES_PATH) as _f:
    _class_names = _json.load(_f)
NUM_CLASSES = len(_class_names)

print("Building model and loading weights...")
model = build_model(num_classes=NUM_CLASSES, backbone_trainable=True)
load_weights_h5(model, MODEL_WEIGHTS)
logit_model = build_logit_model(model)
print("Logit model built.")

# ── Collect validation logits + labels ────────────────────────────────────────
def get_logits_labels(data_dir, val_split=0.2):
    ds = image_dataset_from_directory(
        data_dir, validation_split=val_split, subset='validation',
        seed=SEED, image_size=IMG_SIZE, batch_size=BATCH_SIZE, label_mode='int'
    )
    all_logits, all_labels = [], []
    for x, y in ds:
        x = eff_prep(x)
        logits = logit_model(x, training=False).numpy()
        all_logits.append(logits)
        all_labels.append(y.numpy())
    return np.concatenate(all_logits), np.concatenate(all_labels)

print("\nCollecting PlantVillage val logits...")
pv_logits, pv_labels = get_logits_labels(MAIZE_DIR)

print("Collecting field val logits...")
field_logits, field_labels = get_logits_labels(FIELD_DIR, val_split=0.25)

all_logits = np.concatenate([pv_logits, field_logits])
all_labels = np.concatenate([pv_labels, field_labels])
print(f"Total val samples: {len(all_labels)}")

# ── Calibrate temperature ──────────────────────────────────────────────────────
def nll(T):
    """Negative log-likelihood under temperature T."""
    scaled = all_logits / T
    # Stable softmax
    shifted = scaled - scaled.max(axis=1, keepdims=True)
    exp     = np.exp(shifted)
    probs   = exp / exp.sum(axis=1, keepdims=True)
    correct_probs = probs[np.arange(len(all_labels)), all_labels]
    return -np.mean(np.log(correct_probs + 1e-10))

result = minimize_scalar(nll, bounds=(0.1, 10.0), method='bounded')
T_opt  = float(result.x)

# Compare calibrated vs uncalibrated confidence
raw_probs  = np.exp(all_logits) / np.exp(all_logits).sum(axis=1, keepdims=True)
cal_logits = all_logits / T_opt
cal_probs  = np.exp(cal_logits - cal_logits.max(axis=1, keepdims=True))
cal_probs /= cal_probs.sum(axis=1, keepdims=True)

raw_conf = raw_probs.max(axis=1).mean() * 100
cal_conf = cal_probs.max(axis=1).mean() * 100
acc      = (all_logits.argmax(axis=1) == all_labels).mean() * 100

print(f"\nAccuracy:              {acc:.1f}%")
print(f"Avg confidence (raw):  {raw_conf:.1f}%  (gap: {raw_conf - acc:.1f}%)")
print(f"Optimal temperature T: {T_opt:.4f}")
print(f"Avg confidence (cal):  {cal_conf:.1f}%  (gap: {cal_conf - acc:.1f}%)")

with open(TEMP_PATH, 'w') as f:
    json.dump({'temperature': T_opt}, f)
print(f"\nTemperature saved: {TEMP_PATH}")
