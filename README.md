# Maize Disease Classification with ResNet50

Maize is one of Africa's most important staple crops. This project builds an image classifier that detects three common maize diseases — **Cercospora leaf spot**, **Common rust**, and **Northern leaf blight** — alongside healthy leaves, using transfer learning on a pre-trained ResNet50 backbone.

## Notebook

The main deliverable is a single Jupyter notebook that runs on both **Google Colab** and **Kaggle** without modification.

| Platform | Instructions |
|---|---|
| Google Colab | Add `KAGGLE_USERNAME` and `KAGGLE_KEY` as Colab Secrets, then run all cells |
| Kaggle | Add the [PlantVillage dataset](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset) via the "Add Data" sidebar, then run all cells |

## Dataset

[PlantVillage Dataset](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset) — 38 crop disease classes. We extract only the 4 maize/corn classes (3,852 images total).

| Class | Images |
|---|---|
| Cercospora leaf spot | 513 |
| Common rust | 1,192 |
| Northern leaf blight | 985 |
| Healthy | 1,162 |

## Approach

1. **Transfer learning** — frozen ResNet50 (ImageNet weights) + custom classification head
2. **Class weights** — balanced weighting to handle the Cercospora imbalance
3. **Data augmentation** — random flip, rotation, zoom, contrast
4. **Fine-tuning** — unfreeze top 30 ResNet50 layers at 10x lower learning rate

## Results

| Stage | Val Accuracy |
|---|---|
| Frozen backbone | ~76% |
| After fine-tuning | ~89% |

## Exports

The notebook saves three artefacts:
- `best_model.keras` — full Keras model
- `maize_disease_model.tflite` — optimised for mobile/edge deployment
- `class_names.json` — index-to-label mapping

## Repository Structure

```
├── notebook/
│   └── maize-disease-classification.ipynb
├── .gitignore
└── README.md
```

> Data, logs, and model files are gitignored and generated at runtime.
