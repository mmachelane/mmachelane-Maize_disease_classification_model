# Maize Disease Classification with ResNet50

Maize is one of Africa's most important staple crops. This project builds an image classifier that detects three common maize diseases — **Cercospora leaf spot**, **Common rust**, and **Northern leaf blight** — alongside healthy leaves, using transfer learning on a pre-trained ResNet50 backbone.

## Notebook

The main deliverable is a single Jupyter notebook that runs on both **Google Colab** and **Kaggle** without modification. All dependencies (TensorFlow, scikit-learn, matplotlib, etc.) are pre-installed in both environments — no `requirements.txt` needed.

| Platform | Instructions |
|---|---|
| Google Colab | Add `KAGGLE_USERNAME` and `KAGGLE_KEY` as Colab Secrets, then run all cells |
| Kaggle | Add the [PlantVillage dataset](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset) via the "Add Data" sidebar, then run all cells |

## Dataset

[PlantVillage Dataset](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset) — 38 crop disease classes. We extract only the 4 maize/corn classes (3,852 images total).

| Class | Images | Share |
|---|---|---|
| Cercospora leaf spot (Gray leaf spot) | 513 | 13.3% |
| Common rust | 1,192 | 30.9% |
| Northern leaf blight | 985 | 25.6% |
| Healthy | 1,162 | 30.2% |

The dataset is split 80/20 into 3,082 training and 770 validation images (97 and 25 batches at batch_size=32).

## Approach

1. **Transfer learning** — frozen ResNet50 (ImageNet weights) + custom classification head (GAP → Dense(256) → Dropout(0.5) → Softmax(4))
2. **Class weights** — balanced weighting to handle the Cercospora imbalance (weight: 1.80 vs ~0.85 for other classes)
3. **Data augmentation** — random flip, rotation, zoom, contrast
4. **Fine-tuning** — unfreeze top 30 of 175 ResNet50 layers at 10x lower learning rate (1e-4)

## Results

Training ran for 70 epochs total (50 initial + 20 fine-tuning) with early stopping.

| Metric | Value |
|---|---|
| Best validation accuracy | **90.13%** (epoch 62) |
| Final validation accuracy | 88.57% |
| Final training accuracy | 93.06% |
| Macro avg F1-score | 0.86 |

### Per-Class Performance

| Class | Precision | Recall | F1-Score |
|---|---|---|---|
| Cercospora leaf spot | 0.62 | 0.77 | 0.68 |
| Common rust | 0.99 | 0.99 | 0.99 |
| Northern leaf blight | 0.85 | 0.74 | 0.79 |
| Healthy | 0.97 | 0.97 | 0.97 |

Common rust and Healthy are near-perfect. Cercospora leaf spot is the weakest class due to limited training data (513 images, 13.3% of the dataset).

## Limitations

- **Lab-controlled images only:** PlantVillage images have uniform backgrounds and consistent lighting — the model has not been exposed to real-world field conditions
- **Limited geographic diversity:** The dataset does not capture regional variations in maize cultivars, soil types, or growing conditions
- **Class imbalance:** Cercospora leaf spot is under-represented, limiting the model's reliability for that disease despite class weighting

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
