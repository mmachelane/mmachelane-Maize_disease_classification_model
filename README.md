# 🌽 Maize Disease Classifier

**Real-World Maize Leaf Disease Detection — EfficientNetB4 + Joint Training + TTA + Temperature Scaling**

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.10-FF6F00?style=flat&logo=tensorflow&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=flat)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen?style=flat)
![PV Accuracy](https://img.shields.io/badge/PV_Val_Accuracy-95.1%25-blue?style=flat)
![Field Accuracy](https://img.shields.io/badge/Field_Val_Accuracy-70.0%25-orange?style=flat)
[![Demo](https://img.shields.io/badge/Live_Demo-HuggingFace-yellow?style=flat&logo=huggingface)](https://huggingface.co/spaces/KaraboMoswane/maize-disease-classifier)

---

## 🎯 Project Overview

Maize is Africa's most widely grown staple crop, yet smallholder farmers lose 20–40% of yields annually to preventable leaf diseases. Early, accurate diagnosis is the difference between a treated field and a failed harvest — but agronomists are scarce and lab testing is slow.

This project builds a **production-ready image classifier** that diagnoses maize leaf disease from a smartphone photo in under 3 seconds.

**What it does:**
- Classifies 4 conditions: Cercospora Leaf Spot, Common Rust, Northern Leaf Blight, Healthy
- Trained on both lab (PlantVillage) and real-world field images for domain robustness
- Provides calibrated confidence scores — low-confidence predictions trigger a retake prompt
- Delivers treatment advice alongside every diagnosis

***

## 📊 Key Results

| Metric | Value |
|---|---|
| PlantVillage validation accuracy | **95.1%** |
| Field image validation accuracy | **70.0%** |
| Combined calibrated accuracy | **88.4%** |
| Confidence gap (before calibration) | +3.2% overconfident |
| Confidence gap (after calibration, T=1.43) | **-0.4%** (near-perfect) |
| TTA runs per prediction | 8 |

***

## 🌍 African Agricultural Context

### The Disease Landscape

```
Maize leaf diseases (Sub-Saharan Africa)
│
├── Cercospora Leaf Spot / Gray Leaf Spot
│   └── Causes 30–60% yield loss in humid regions
│
├── Common Rust (Puccinia sorghi)
│   └── Spreads rapidly — early detection critical
│
└── Northern Leaf Blight (Turcicum)
    └── Most visually similar to healthy leaves — hardest class
```

### Why Field Images Matter

PlantVillage images are taken in controlled lab conditions — isolated leaves on white backgrounds. Real field photos have:
- Multiple overlapping leaves in frame
- Varying light, glare, and shadows
- Soil, background clutter, camera shake

Training only on PlantVillage gives 95%+ lab accuracy but poor field performance. This project bridges that gap with **joint domain training** using real field images and PlantDoc.

***

## 🏗️ Architecture

```
PlantVillage (3,852)  ──┐
Field Images (1,109)  ──┼──► Joint Training Pipeline ──► EfficientNetB4 Head
PlantDoc Corn (337)   ──┘         │
                                  ▼
                        Temperature Calibration (T=1.43)
                                  │
                                  ▼
                         Gradio App (HuggingFace)
                                  │
                    ┌─────────────┴─────────────┐
                    ▼                           ▼
              TTA (8 runs)            Confidence Threshold
              avg logits             < 65% → retake prompt
                    │                           │
                    └─────────────┬─────────────┘
                                  ▼
                        Diagnosis + Treatment Advice
```

***

## 📁 Project Structure

```
├── train.py                          # Stage 1: EfficientNetB4 base training
├── finetune_joint.py                 # Stage 2: Domain adaptation (PV + field + PlantDoc)
├── calibrate.py                      # Stage 3: Temperature scaling calibration
├── prepare_plantdoc.py               # Utility: Map PlantDoc classes → PV naming
├── model_utils.py                    # Shared build/load helpers (TF 2.10 compat)
├── app.py                            # Gradio demo: TTA + calibration + UI
├── requirements.txt
├── models/
│   ├── best_model_joint_phaseB.weights.h5   # Final weights (Git LFS)
│   ├── class_names.json
│   └── temperature.json                      # Calibrated T=1.43
└── data/                             # Gitignored — generated locally
    ├── maize_only/                   # PlantVillage maize subset
    ├── maize_in_field/               # Real-world field photos
    └── plantdoc_maize/               # PlantDoc corn (mapped classes)
```

***

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- NVIDIA GPU recommended (CPU works, slower)
- conda environment: `maize_tf210`

### Installation

```bash
git clone https://github.com/mmachelane/mmachelane-Maize_disease_classification_model.git
cd mmachelane-Maize_disease_classification_model
pip install -r requirements.txt
```

### Full Training Pipeline

```bash
# Stage 1: Base training on PlantVillage
python train.py

# Stage 2: Joint domain adaptation (PV + field + PlantDoc)
python finetune_joint.py

# Stage 3: Temperature scaling calibration
python calibrate.py
```

### Run the Demo Locally

```bash
python app.py
```

### Run on HuggingFace Spaces

[![Open in Spaces](https://huggingface.co/datasets/huggingface/badges/resolve/main/open-in-hf-spaces-sm.svg)](https://huggingface.co/spaces/KaraboMoswane/maize-disease-classifier)

***

## 📈 Training Pipeline

| Phase | Script | Description | LR | Epochs |
|---|---|---|---|---|
| 1A | `train.py` | Frozen EfficientNetB4, train head | 1e-3 | 50 (early stop) |
| 1B | `train.py` | Unfreeze top 30 layers, fine-tune | 1e-5 | 30 (early stop) |
| 2A | `finetune_joint.py` | Frozen backbone, fresh head on combined data | 1e-3 | 30 (early stop) |
| 2B | `finetune_joint.py` | Unfreeze top 15 layers on combined data | 1e-5 | 30 (early stop) |
| 3 | `calibrate.py` | NLL minimisation → optimal temperature T | — | — |

Auto-selects the better of Phase 2A/2B by combined val accuracy.

***

## 🔍 Key Capabilities

### EfficientNetB4 Backbone
Replaces ResNet50 for better domain generalisation. No preprocessing layer in the model — `preprocess_input` is applied in the data pipeline to avoid TF 2.10 EagerTensor serialisation bugs.

### Domain-Bridging Augmentation
```python
RandomFlip('horizontal_and_vertical')
RandomRotation(0.3)
RandomZoom(0.25)
RandomTranslation(0.1, 0.1)
RandomContrast(0.3)
RandomBrightness(0.3)
GaussianNoise(8.0)   # simulates sensor noise and real-world variation
```

### Test-Time Augmentation (TTA)
8 augmented versions of each input image are run through the model. Logits are averaged before temperature-scaled softmax — reduces prediction variance on ambiguous images.

### Temperature Scaling
Post-training calibration finds optimal T via NLL minimisation on the combined validation set. At T=1.43, the confidence gap drops from **+3.2%** to **-0.4%**.

***

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Backbone | EfficientNetB4 (ImageNet pretrained) |
| Framework | TensorFlow 2.10 / Keras |
| Training data | PlantVillage + field images + PlantDoc |
| Augmentation | Keras preprocessing layers |
| Calibration | SciPy `minimize_scalar` (bounded NLL) |
| Inference | TTA × 8 + temperature scaling |
| Demo | Gradio on HuggingFace Spaces |
| Model storage | Git LFS (73MB weights file) |
| Language | Python 3.10 |

***

## 🏆 What Sets This Apart

| Capability | Typical Approach | This Project |
|---|---|---|
| Training data | PlantVillage only | PV + field images + PlantDoc |
| Backbone | ResNet50 | EfficientNetB4 |
| Inference | Single forward pass | TTA (8 runs, averaged logits) |
| Confidence | Raw softmax | Temperature-calibrated (T=1.43) |
| Uncertain predictions | Shown anyway | Retake prompt below 65% |
| Domain gap | Ignored | Explicitly addressed via joint training |
| Save format | `.keras` / `.h5` | Weights-only (TF 2.10 compat) |

***

## ⚠️ Limitations

- **Healthy leaves in the field** remain the hardest class — close-up single-leaf photos in good natural lighting give best results
- **NLB vs Cercospora** — both produce elongated lesions; ambiguous cases are correctly flagged via the confidence threshold
- **No mobile export** — runs as full EfficientNetB4 (~2s per prediction with TTA on CPU)
- **Data imbalance** — only 513 Cercospora images vs 1,192 Common Rust in PlantVillage

***

*Built by [Karabo Moswane](https://github.com/mmachelane) · PlantVillage Dataset · PlantDoc Dataset*
