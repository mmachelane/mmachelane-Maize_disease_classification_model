# 🌽 Maize Disease Classification Model

### Detecting Maize Leaf Diseases with Deep Learning

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=flat&logo=tensorflow&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=flat)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen?style=flat)
![Accuracy](https://img.shields.io/badge/PV_Val_Accuracy-95.1%25-blue?style=flat)
![Field Accuracy](https://img.shields.io/badge/Field_Val_Accuracy-70.0%25-orange?style=flat)
[![HuggingFace](https://img.shields.io/badge/Demo-HuggingFace%20Spaces-yellow?style=flat&logo=huggingface)](https://huggingface.co/spaces/KaraboMoswane/maize-disease-classifier)

---

Maize is one of Africa's most important staple crops. This project builds a real-world image classifier that detects three common maize diseases — **Cercospora leaf spot**, **Common rust**, and **Northern leaf blight** — alongside healthy leaves, using transfer learning on EfficientNetB4 with joint training on both lab and field images.

## 🚀 Live Demo

Try it on [Hugging Face Spaces](https://huggingface.co/spaces/KaraboMoswane/maize-disease-classifier) — upload a maize leaf photo and get a diagnosis with treatment advice.

## 🔬 Architecture & Training Pipeline

### Model
- **Backbone:** EfficientNetB4 (ImageNet pretrained, `include_top=False`)
- **Head:** GlobalAveragePooling → BatchNorm → Dense(256, ReLU) → Dropout(0.4) → Dense(4, Softmax)

### Training — Two-stage pipeline

**Stage 1 — `train.py`** (PlantVillage base training)
1. Frozen backbone, train head only (LR=1e-3, up to 50 epochs, early stopping)
2. Unfreeze top 30 backbone layers, fine-tune (LR=1e-5, up to 30 epochs)

**Stage 2 — `finetune_joint.py`** (Domain adaptation)
- Phase A: Frozen backbone, fresh head, train on combined PV + field + PlantDoc data (LR=1e-3)
- Phase B: Unfreeze last 15 backbone layers, fine-tune on combined data (LR=1e-5)
- Auto-selects the better phase by combined val accuracy

**Stage 3 — `calibrate.py`** (Temperature scaling)
- Finds optimal temperature T via NLL minimisation on combined val set
- Calibrated T = 1.43 — reduces overconfidence gap from 3.2% to -0.4%

### Inference — `app.py`
- **Test-Time Augmentation (TTA):** 8 augmented runs averaged before softmax
- **Temperature scaling:** Calibrated confidence (T=1.43)
- **Uncertainty threshold:** Predictions below 65% confidence show a retake prompt

## 📊 Datasets

| Dataset | Images | Purpose |
|---|---|---|
| [PlantVillage](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset) | 3,852 | Base training (lab conditions) |
| Field images (`data/maize_in_field/`) | 1,109 | Real-world domain adaptation |
| [PlantDoc](https://github.com/pratikkayal/PlantDoc-Dataset) corn subset | 337 | Additional real-world coverage |

PlantVillage class distribution:

| Class | Images |
|---|---|
| Cercospora leaf spot (Gray leaf spot) | 513 |
| Common rust | 1,192 |
| Northern leaf blight | 985 |
| Healthy | 1,162 |

## 📈 Results

| Split | Accuracy |
|---|---|
| PlantVillage validation (80/20 split) | **95.1%** |
| Field validation (75/25 split) | **70.0%** |
| Combined val (PV + field, calibrated) | **88.4%** |

Temperature calibration reduced the confidence gap from **+3.2%** (overconfident) to **-0.4%** (near-perfect).

## ⚠️ Limitations

- **Healthy leaves in the field** remain the hardest class — PlantVillage healthy images have clean white backgrounds while field conditions introduce glare, overlapping leaves, and clutter. A close-up single-leaf photo in good lighting gives best results.
- **NLB vs Cercospora confusion** — both produce elongated lesions. The model is uncertain on ambiguous cases (correctly flagged via the confidence threshold).
- **No mobile-optimised export yet** — runs as a full EfficientNetB4 on CPU (~2s per prediction with TTA).

## 🗂️ Repository Structure

```
├── train.py                  # Stage 1: EfficientNetB4 base training on PlantVillage
├── finetune_joint.py         # Stage 2: Joint PV + field domain adaptation
├── calibrate.py              # Stage 3: Temperature scaling calibration
├── prepare_plantdoc.py       # Utility: map PlantDoc classes to PV naming
├── model_utils.py            # Shared: build/load helpers (TF 2.10 compat)
├── app.py                    # Gradio demo: TTA + temperature scaling + UI
├── requirements.txt
├── models/
│   ├── best_model_joint_phaseB.weights.h5   # Final joint-trained weights (LFS)
│   ├── class_names.json
│   └── temperature.json
└── data/                     # Gitignored — generated locally
    ├── maize_only/           # PlantVillage maize subset
    ├── maize_in_field/       # Real-world field photos
    └── plantdoc_maize/       # PlantDoc corn images (mapped classes)
```

> Large model files are tracked via Git LFS.
