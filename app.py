"""
Maize Disease Classifier — Gradio Demo
Hugging Face Spaces deployment
"""

import json
import numpy as np
import gradio as gr
import tensorflow as tf
from PIL import Image

# ── Load model & class names ──────────────────────────────────────────────────
model      = tf.keras.models.load_model('models/best_model_joint.keras')
with open('models/class_names.json') as f:
    class_names = json.load(f)

# Clean display labels
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

IMG_SIZE = (224, 224)

def classify(image):
    if image is None:
        return {}, "Upload a maize leaf image to get a prediction."

    img = Image.fromarray(image).resize(IMG_SIZE)
    arr = np.array(img, dtype=np.float32)
    arr = np.expand_dims(arr, axis=0)  # (1, 224, 224, 3)

    preds       = model.predict(arr, verbose=0)[0]
    top_idx     = int(np.argmax(preds))
    top_class   = class_names[top_idx]
    confidence  = float(preds[top_idx]) * 100

    confidences = {LABELS.get(c, c): float(p) for c, p in zip(class_names, preds)}

    if confidence < 65:
        advice = (
            f"⚠️ **Low confidence ({confidence:.1f}%)** — the model is uncertain.\n\n"
            "This image may be outside the model's training distribution (field lighting, angle, or background). "
            "Consider retaking the photo with the leaf filling the frame in good lighting, or consult an agronomist."
        )
    else:
        advice = f"**{LABELS.get(top_class, top_class)}** ({confidence:.1f}% confidence)\n\n{ADVICE.get(top_class, '')}"

    return confidences, advice


# ── UI ────────────────────────────────────────────────────────────────────────
with gr.Blocks(title="Maize Disease Classifier") as demo:
    gr.Markdown(
        """
        # 🌽 Maize Disease Classifier
        Upload a photo of a maize leaf. The model will identify the disease (or confirm it's healthy)
        and suggest a treatment action.

        **Model:** ResNet50 fine-tuned on PlantVillage · **Classes:** 4 · **Built by:** Karabo Moswane
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
