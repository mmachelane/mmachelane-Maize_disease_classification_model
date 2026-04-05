"""
Maize Disease Classifier — Gradio Demo
Hugging Face Spaces deployment

Inference improvements:
  - Test-Time Augmentation (TTA): averages 8 augmented predictions
  - Temperature scaling: calibrated confidence from models/temperature.json
  - Uncertainty threshold: low-confidence predictions flagged
"""

import json
import numpy as np
import gradio as gr
import tensorflow as tf
from tensorflow.keras.applications.efficientnet import preprocess_input as eff_prep
from PIL import Image, ImageOps, ImageEnhance
import random

# ── Load model & class names ──────────────────────────────────────────────────
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model_utils import build_model, load_weights_h5, build_logit_model

with open('models/class_names.json') as f:
    class_names = json.load(f)

_WEIGHTS = 'models/best_model_joint_phaseB.weights.h5'
if not os.path.exists(_WEIGHTS):
    _WEIGHTS = 'models/best_model.keras.weights.h5'  # fallback to phase-1

model = build_model(num_classes=len(class_names), backbone_trainable=True)
load_weights_h5(model, _WEIGHTS)
print(f"Model loaded from {_WEIGHTS}")

# Temperature scaling — load if calibrated, else default T=1.0
try:
    with open('models/temperature.json') as f:
        TEMPERATURE = float(json.load(f)['temperature'])
    print(f"Temperature scaling loaded: T={TEMPERATURE:.4f}")
except FileNotFoundError:
    TEMPERATURE = 1.0
    print("No temperature file found — using T=1.0")

# Build logit model for temperature scaling
logit_model = build_logit_model(model)

LABELS = {
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot': 'Cercospora Leaf Spot / Gray Leaf Spot',
    'Corn_(maize)___Common_rust_':                        'Common Rust',
    'Corn_(maize)___healthy':                             'Healthy',
    'Corn_(maize)___Northern_Leaf_Blight':                'Northern Leaf Blight',
}

ADVICE = {
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot': (
        'Apply fungicides containing azoxystrobin or pyraclostrobin. '
        'Improve air circulation and avoid overhead irrigation.'
    ),
    'Corn_(maize)___Common_rust_': (
        'Apply fungicides early when pustules first appear. '
        'Resistant hybrids are the most effective long-term strategy.'
    ),
    'Corn_(maize)___healthy': (
        'No disease detected. Continue standard agronomic practices.'
    ),
    'Corn_(maize)___Northern_Leaf_Blight': (
        'Apply foliar fungicides at early whorl stage. '
        'Rotate crops and use resistant varieties in subsequent seasons.'
    ),
}

IMG_SIZE   = (224, 224)
TTA_RUNS   = 8
CONF_THRESHOLD = 65.0

def _tta_augment(img_pil):
    """Single random augmentation for TTA — PIL-based to avoid TF graph overhead."""
    # Random horizontal flip
    if random.random() > 0.5:
        img_pil = ImageOps.mirror(img_pil)
    # Random vertical flip
    if random.random() > 0.5:
        img_pil = ImageOps.flip(img_pil)
    # Random brightness ±20%
    factor = random.uniform(0.8, 1.2)
    img_pil = ImageEnhance.Brightness(img_pil).enhance(factor)
    # Random contrast ±20%
    factor = random.uniform(0.8, 1.2)
    img_pil = ImageEnhance.Contrast(img_pil).enhance(factor)
    # Random crop and resize (simulate zoom)
    w, h = img_pil.size
    margin = int(min(w, h) * 0.1)
    left   = random.randint(0, margin)
    top    = random.randint(0, margin)
    right  = w - random.randint(0, margin)
    bottom = h - random.randint(0, margin)
    img_pil = img_pil.crop((left, top, right, bottom)).resize(IMG_SIZE)
    return img_pil

def _apply_temperature(logits, T):
    """Scale logits by temperature then apply softmax."""
    scaled  = logits / T
    shifted = scaled - scaled.max()
    exp     = np.exp(shifted)
    return exp / exp.sum()

def classify(image):
    if image is None:
        return {}, "Upload a maize leaf image to get a prediction."

    img_pil = Image.fromarray(image).convert('RGB').resize(IMG_SIZE)

    # ── Test-Time Augmentation ────────────────────────────────────────────────
    all_logits = []

    # Original image
    arr = eff_prep(np.expand_dims(np.array(img_pil, dtype=np.float32), 0))
    all_logits.append(logit_model.predict(arr, verbose=0)[0])

    # TTA_RUNS - 1 augmented versions
    for _ in range(TTA_RUNS - 1):
        aug = _tta_augment(img_pil)
        arr = eff_prep(np.expand_dims(np.array(aug, dtype=np.float32), 0))
        all_logits.append(logit_model.predict(arr, verbose=0)[0])

    # Average logits then apply temperature-scaled softmax
    mean_logits = np.mean(all_logits, axis=0)
    preds       = _apply_temperature(mean_logits, TEMPERATURE)

    top_idx    = int(np.argmax(preds))
    top_class  = class_names[top_idx]
    confidence = float(preds[top_idx]) * 100

    confidences = {LABELS.get(c, c): float(p) for c, p in zip(class_names, preds)}

    if confidence < CONF_THRESHOLD:
        advice = (
            f"⚠️ **Low confidence ({confidence:.1f}%)** — the model is uncertain.\n\n"
            "Tip: retake the photo with the leaf filling the frame in good natural lighting, "
            "or consult an agronomist for a definitive diagnosis."
        )
    else:
        advice = (
            f"**{LABELS.get(top_class, top_class)}** ({confidence:.1f}% confidence)\n\n"
            f"{ADVICE.get(top_class, '')}"
        )

    return confidences, advice


# ── UI ────────────────────────────────────────────────────────────────────────
with gr.Blocks(title="Maize Disease Classifier") as demo:
    gr.Markdown(
        """
        # 🌽 Maize Disease Classifier
        Upload a photo of a maize leaf. The model will identify the disease (or confirm it's healthy)
        and suggest a treatment action.

        **Model:** EfficientNetB4 + TTA · **Classes:** 4 · **Built by:** Karabo Moswane
        """
    )

    with gr.Row():
        with gr.Column():
            image_input = gr.Image(label="Maize Leaf Photo", type="numpy")
            submit_btn  = gr.Button("Classify", variant="primary")
        with gr.Column():
            label_output  = gr.Label(label="Confidence Scores", num_top_classes=4)
            advice_output = gr.Markdown(label="Diagnosis & Advice")

    submit_btn.click(fn=classify, inputs=image_input, outputs=[label_output, advice_output])
    image_input.change(fn=classify, inputs=image_input, outputs=[label_output, advice_output])

    gr.Markdown(
        "---\n"
        "Model trained on [PlantVillage Dataset](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset) · "
        "[GitHub](https://github.com/mmachelane/Maize_disease_classification_model)"
    )

demo.launch(theme=gr.themes.Base())
