"""
Joint fine-tuning: mix PlantVillage + field images so the head learns both distributions.
Loads best_model.keras, reinitialises the dense head, trains on combined data in two phases:
  Phase A — frozen ResNet50, fresh head trained on combined data  (LR=1e-3, 30 epochs)
  Phase B — unfreeze last ResNet50 conv block (~15 layers)        (LR=1e-5, 30 epochs)

Saves to models/best_model_joint.keras — never touches best_model.keras.
Logs: models/joint_log_a.csv (Phase A), models/joint_log_b.csv (Phase B)

Run with: conda run -n maize_tf210 python finetune_joint.py
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import layers, Sequential

REPO_DIR    = os.path.dirname(os.path.abspath(__file__))
PV_DIR      = os.path.join(REPO_DIR, 'data', 'maize_only')
FIELD_DIR   = os.path.join(REPO_DIR, 'data', 'maize_in_field')
MODEL_IN    = os.path.join(REPO_DIR, 'models', 'best_model.keras')
MODEL_OUT   = os.path.join(REPO_DIR, 'models', 'best_model_joint.keras')
LOG_PATH    = os.path.join(REPO_DIR, 'models', 'joint_log.csv')

IMG_SIZE    = (224, 224)
BATCH_SIZE  = 32
SEED        = 42
EPOCHS      = 30

print(f"TF {tf.__version__} | GPU: {tf.config.list_physical_devices('GPU')}")

# ── PlantVillage training split ───────────────────────────────────────────────
pv_train = image_dataset_from_directory(
    PV_DIR, validation_split=0.2, subset='training',
    seed=SEED, image_size=IMG_SIZE, batch_size=BATCH_SIZE, label_mode='int'
)
pv_val = image_dataset_from_directory(
    PV_DIR, validation_split=0.2, subset='validation',
    seed=SEED, image_size=IMG_SIZE, batch_size=BATCH_SIZE, label_mode='int'
)

# ── Field training split ──────────────────────────────────────────────────────
field_train = image_dataset_from_directory(
    FIELD_DIR, validation_split=0.25, subset='training',
    seed=SEED, image_size=IMG_SIZE, batch_size=BATCH_SIZE, label_mode='int'
)
field_val = image_dataset_from_directory(
    FIELD_DIR, validation_split=0.25, subset='validation',
    seed=SEED, image_size=IMG_SIZE, batch_size=BATCH_SIZE, label_mode='int'
)

print(f"\nPV classes:    {pv_train.class_names}")
print(f"Field classes: {field_train.class_names}")
assert pv_train.class_names == field_train.class_names, "Class order mismatch!"

# ── Augmentation ─────────────────────────────────────────────────────────────
augment = Sequential([
    layers.RandomFlip('horizontal_and_vertical'),
    layers.RandomRotation(0.15),
    layers.RandomZoom(0.12),
    layers.RandomContrast(0.15),
    layers.RandomBrightness(0.12),
], name='augmentation')

AUTOTUNE = tf.data.AUTOTUNE

def prep(ds):
    return ds.map(lambda x, y: (augment(x, training=True), y)).prefetch(AUTOTUNE)

# Oversample field images 3× so they're not drowned by PlantVillage
field_train_over = field_train.repeat(3)

# Combine: interleave PV and field batches
combined_train = tf.data.Dataset.sample_from_datasets(
    [pv_train, field_train_over], weights=[0.5, 0.5], seed=SEED
)
combined_train = combined_train.map(lambda x, y: (augment(x, training=True), y)).prefetch(AUTOTUNE)

# Validation: concatenate both — we want accuracy on both domains
combined_val = pv_val.concatenate(field_val).prefetch(AUTOTUNE)

# ── Load model ────────────────────────────────────────────────────────────────
print(f"\nLoading {MODEL_IN}")
model = tf.keras.models.load_model(MODEL_IN)

# Freeze entire ResNet50 — we only retrain the head
base_model = model.get_layer('resnet50')
base_model.trainable = False

# Reinitialise dense head so it learns both PV and field features from scratch
# (the PV-trained head weights are biased and cause the ~55% ceiling on field data)
for layer in model.layers:
    if hasattr(layer, 'kernel_initializer') and layer.name != 'resnet50':
        layer.kernel.assign(layer.kernel_initializer(layer.kernel.shape))
        if layer.bias is not None:
            layer.bias.assign(layer.bias_initializer(layer.bias.shape))

print(f"ResNet50 frozen. Head re-initialised. Trainable params: {sum(np.prod(v.shape) for v in model.trainable_variables):,}")

# ── Baseline on each domain ───────────────────────────────────────────────────
model.compile(optimizer=Adam(1e-3), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
_, acc_pv    = model.evaluate(pv_val,    verbose=0)
_, acc_field = model.evaluate(field_val, verbose=0)
print(f"\nBaseline — PlantVillage: {acc_pv*100:.1f}%  |  Field: {acc_field*100:.1f}%")

# ── Phase A: Head-only, LR=1e-3 ─────────────────────────────────────────────
print("\n══ Phase A: Joint head training (LR=1e-3, frozen ResNet50) ══")

LOG_A = os.path.join(os.path.dirname(LOG_PATH), 'joint_log_a.csv')
LOG_B = os.path.join(os.path.dirname(LOG_PATH), 'joint_log_b.csv')

callbacks_a = [
    ModelCheckpoint(MODEL_OUT, monitor='val_accuracy', save_best_only=True, mode='max', verbose=1),
    EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-7, verbose=1),
    CSVLogger(LOG_A),
]

model.fit(
    combined_train, validation_data=combined_val,
    epochs=EPOCHS, callbacks=callbacks_a, verbose=1
)

_, acc_pv_a    = model.evaluate(pv_val,    verbose=0)
_, acc_field_a = model.evaluate(field_val, verbose=0)
print(f"\nPhase A — PlantVillage: {acc_pv_a*100:.1f}%  |  Field: {acc_field_a*100:.1f}%")

# ── Phase B: Unfreeze last ResNet conv block, LR=1e-5 ────────────────────────
# Only fine-tune last ~15 ResNet50 layers — conservative to avoid forgetting
# on the tiny field dataset. LR is 100x lower than Phase A.
print("\n══ Phase B: Unfreeze last ResNet50 conv block (LR=1e-5) ══")
base_model.trainable = True
for layer in base_model.layers[:-15]:
    layer.trainable = False

trainable = sum(1 for l in base_model.layers if l.trainable)
print(f"Unfrozen ResNet50 layers: {trainable} / {len(base_model.layers)}")

model.compile(
    optimizer=Adam(learning_rate=1e-5),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

callbacks_b = [
    ModelCheckpoint(MODEL_OUT, monitor='val_accuracy', save_best_only=True, mode='max', verbose=1),
    EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-8, verbose=1),
    CSVLogger(LOG_B),
]

model.fit(
    combined_train, validation_data=combined_val,
    epochs=EPOCHS, callbacks=callbacks_b, verbose=1
)

_, acc_pv_final    = model.evaluate(pv_val,    verbose=0)
_, acc_field_final = model.evaluate(field_val, verbose=0)
print(f"\nFinal — PlantVillage: {acc_pv_final*100:.1f}%  |  Field: {acc_field_final*100:.1f}%")
print(f"Joint model saved to: {MODEL_OUT}")
