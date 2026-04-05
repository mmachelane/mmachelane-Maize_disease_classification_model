"""
Training script — EfficientNetB4 backbone (replaces ResNet50).
Improvements:
  - EfficientNetB4 with built-in preprocessing (better domain generalization)
  - Stronger augmentation: GaussianNoise, RandomTranslation, perspective-sim
  - PlantDoc data merged into training set for real-world coverage
  - BatchNormalization in head for stability
  - Two-phase training: frozen base → fine-tune top 30 layers

Run with: python train.py
Saves: models/best_model.keras, models/class_names.json
"""

import os, json
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model, layers, Sequential
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.layers import (Input, GlobalAveragePooling2D, Dense,
                                     Dropout, BatchNormalization, Activation)
from tensorflow.keras.callbacks import (ModelCheckpoint, EarlyStopping,
                                        ReduceLROnPlateau, CSVLogger)
from tensorflow.keras.optimizers import Adam

REPO_DIR   = os.path.dirname(os.path.abspath(__file__))
MAIZE_DIR  = os.path.join(REPO_DIR, 'data', 'maize_only')
PLANTDOC_DIR = os.path.join(REPO_DIR, 'data', 'plantdoc_maize')  # merged PlantDoc corn images
OUTPUT_DIR = os.path.join(REPO_DIR, 'models')
MODEL_PATH = os.path.join(OUTPUT_DIR, 'best_model.keras')
NAMES_PATH = os.path.join(OUTPUT_DIR, 'class_names.json')
LOG_PATH   = os.path.join(OUTPUT_DIR, 'training_log.csv')
os.makedirs(OUTPUT_DIR, exist_ok=True)

IMG_SIZE   = (224, 224)   # EfficientNetB4 works at 224; upgrade to 380 if GPU available
BATCH_SIZE = 32
EPOCHS     = 50
FT_EPOCHS  = 30
LR         = 1e-3
FT_LR      = 1e-5         # lower than ResNet50 — EfficientNet is more sensitive
VAL_SPLIT  = 0.2
SEED       = 42

print(f"TF {tf.__version__} | GPU: {tf.config.list_physical_devices('GPU')}")

# ── Datasets ──────────────────────────────────────────────────────────────────
train_ds = image_dataset_from_directory(
    MAIZE_DIR, validation_split=VAL_SPLIT, subset='training',
    seed=SEED, image_size=IMG_SIZE, batch_size=BATCH_SIZE, label_mode='int'
)
val_ds = image_dataset_from_directory(
    MAIZE_DIR, validation_split=VAL_SPLIT, subset='validation',
    seed=SEED, image_size=IMG_SIZE, batch_size=BATCH_SIZE, label_mode='int'
)

class_names = train_ds.class_names
NUM_CLASSES = len(class_names)
print(f"Classes ({NUM_CLASSES}): {class_names}")

with open(NAMES_PATH, 'w') as f:
    json.dump(class_names, f, indent=2)

# ── Merge PlantDoc if available ───────────────────────────────────────────────
AUTOTUNE = tf.data.AUTOTUNE
if os.path.isdir(PLANTDOC_DIR):
    plantdoc_ds = image_dataset_from_directory(
        PLANTDOC_DIR, image_size=IMG_SIZE, batch_size=BATCH_SIZE, label_mode='int'
    )
    print(f"PlantDoc classes: {plantdoc_ds.class_names}")
    assert plantdoc_ds.class_names == class_names, "PlantDoc class order mismatch!"
    train_ds = tf.data.Dataset.sample_from_datasets(
        [train_ds, plantdoc_ds.repeat(3)], weights=[0.6, 0.4], seed=SEED
    )
    print("PlantDoc merged into training set.")

# ── Class weights ─────────────────────────────────────────────────────────────
labels_all = np.concatenate([y.numpy() for _, y in
                              image_dataset_from_directory(
                                  MAIZE_DIR, validation_split=VAL_SPLIT, subset='training',
                                  seed=SEED, image_size=IMG_SIZE, batch_size=BATCH_SIZE, label_mode='int'
                              )])
counts = np.bincount(labels_all)
total  = labels_all.size
class_weight_dict = {i: total / (NUM_CLASSES * c) for i, c in enumerate(counts)}
print(f"Class weights: {class_weight_dict}")

# ── Augmentation — domain-bridging ───────────────────────────────────────────
# GaussianNoise + RandomTranslation simulate real field conditions:
# camera shake, partial occlusion, varying distance.
data_augmentation = Sequential([
    layers.RandomFlip('horizontal_and_vertical'),
    layers.RandomRotation(0.3),
    layers.RandomZoom(0.25),
    layers.RandomTranslation(0.1, 0.1),
    layers.RandomContrast(0.3),
    layers.RandomBrightness(0.3),
    layers.GaussianNoise(8.0),   # stddev in [0,255] scale — simulates sensor noise
], name='augmentation')

# ── EfficientNet preprocessing in data pipeline ───────────────────────────────
# EfficientNetB4 expects ImageNet-normalized inputs. Applying in the data
# pipeline (not in the model) avoids the EagerTensor JSON serialization bug
# in TF 2.10 with Normalization layers.
from tensorflow.keras.applications.efficientnet import preprocess_input as eff_prep

def augment_and_prep(x, y):
    x = data_augmentation(x, training=True)
    x = eff_prep(x)   # [0,255] → ImageNet-normalised
    return x, y

def prep_only(x, y):
    return eff_prep(x), y

train_ds = train_ds.map(augment_and_prep, num_parallel_calls=AUTOTUNE)
train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds   = val_ds.map(prep_only, num_parallel_calls=AUTOTUNE)
val_ds   = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# ── Model — EfficientNetB4 ────────────────────────────────────────────────────
# No preprocessing layer in the model — handled in data pipeline above.
# Model inputs are already ImageNet-normalised (range roughly -2.1 to +2.6).
base_model = tf.keras.applications.EfficientNetB4(
    weights='imagenet', include_top=False,
    input_shape=(*IMG_SIZE, 3)
)
base_model.trainable = False

inputs  = Input(shape=(*IMG_SIZE, 3))
x       = base_model(inputs, training=False)
x       = GlobalAveragePooling2D()(x)
x       = BatchNormalization()(x)
x       = Dense(256, activation='relu')(x)
x       = Dropout(0.4)(x)
logits  = Dense(NUM_CLASSES)(x)
outputs = Activation('softmax')(logits)

model = Model(inputs, outputs, name='maize_efficientnetb4')
model.compile(
    optimizer=Adam(learning_rate=LR),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
model.summary()

# ── Phase 1: Train head ───────────────────────────────────────────────────────
print("\n── Phase 1: Training classification head ──")
callbacks = [
    ModelCheckpoint(MODEL_PATH + '.weights.h5', monitor='val_accuracy', save_best_only=True, mode='max', verbose=1, save_weights_only=True),
    EarlyStopping(monitor='val_loss', patience=12, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=6, min_lr=1e-7, verbose=1),
    CSVLogger(LOG_PATH)
]
model.fit(
    train_ds, validation_data=val_ds,
    epochs=EPOCHS, callbacks=callbacks,
    class_weight=class_weight_dict, verbose=1
)
from model_utils import load_weights_h5, save_savedmodel
load_weights_h5(model, MODEL_PATH + '.weights.h5')
SAVED_P1 = MODEL_PATH.replace('.keras', '_tf_p1')
save_savedmodel(model, SAVED_P1)
print(f"Phase 1 model saved: {SAVED_P1}")

# ── Phase 2: Fine-tune top layers ─────────────────────────────────────────────
# EfficientNetB4 is more sensitive than ResNet50 — use lower LR and
# unfreeze fewer layers to prevent catastrophic forgetting.
print("\n── Phase 2: Fine-tuning top EfficientNetB4 layers ──")
base_model.trainable = True
for layer in base_model.layers[:-30]:
    layer.trainable = False

trainable = sum(1 for l in base_model.layers if l.trainable)
print(f"Unfrozen layers: {trainable} / {len(base_model.layers)}")

model.compile(
    optimizer=Adam(learning_rate=FT_LR),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
ft_callbacks = [
    ModelCheckpoint(MODEL_PATH + '.weights.h5', monitor='val_accuracy', save_best_only=True, mode='max', verbose=1, save_weights_only=True),
    EarlyStopping(monitor='val_loss', patience=12, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=6, min_lr=1e-8, verbose=1),
    CSVLogger(LOG_PATH, append=True)
]
model.fit(
    train_ds, validation_data=val_ds,
    epochs=FT_EPOCHS, callbacks=ft_callbacks,
    class_weight=class_weight_dict, verbose=1
)

load_weights_h5(model, MODEL_PATH + '.weights.h5')
save_savedmodel(model, MODEL_PATH.replace('.keras', '_tf'))
print(f"\nModel saved: {MODEL_PATH.replace('.keras', '_tf')}")
print("Training complete.")
