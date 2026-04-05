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
from tensorflow.keras.applications.efficientnet import preprocess_input as eff_prep
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import layers, Sequential

REPO_DIR    = os.path.dirname(os.path.abspath(__file__))
PV_DIR      = os.path.join(REPO_DIR, 'data', 'maize_only')
FIELD_DIR   = os.path.join(REPO_DIR, 'data', 'maize_in_field')
MODEL_IN_WEIGHTS = os.path.join(REPO_DIR, 'models', 'best_model.keras.weights.h5')
MODEL_OUT_A_W    = os.path.join(REPO_DIR, 'models', 'best_model_joint_phaseA.weights.h5')
MODEL_OUT_B_W    = os.path.join(REPO_DIR, 'models', 'best_model_joint_phaseB.weights.h5')
MODEL_OUT        = os.path.join(REPO_DIR, 'models', 'best_model_joint_tf')  # SavedModel dir
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

# ── Augmentation — domain-bridging ───────────────────────────────────────────
augment = Sequential([
    layers.RandomFlip('horizontal_and_vertical'),
    layers.RandomRotation(0.3),
    layers.RandomZoom(0.25),
    layers.RandomTranslation(0.1, 0.1),
    layers.RandomContrast(0.3),
    layers.RandomBrightness(0.3),
    layers.GaussianNoise(8.0),
], name='augmentation')

AUTOTUNE = tf.data.AUTOTUNE

# ── PlantDoc data (real-world images) ─────────────────────────────────────────
PLANTDOC_DIR = os.path.join(REPO_DIR, 'data', 'plantdoc_maize')
if os.path.isdir(PLANTDOC_DIR):
    plantdoc_ds = image_dataset_from_directory(
        PLANTDOC_DIR, image_size=IMG_SIZE, batch_size=BATCH_SIZE,
        label_mode='int', seed=SEED
    )
    print(f"PlantDoc classes: {plantdoc_ds.class_names}")
    assert plantdoc_ds.class_names == pv_train.class_names, "PlantDoc class order mismatch!"
    # Oversample PlantDoc 5× — it's real-world like field data
    field_train_over = tf.data.Dataset.sample_from_datasets(
        [field_train.repeat(3), plantdoc_ds.repeat(5)], weights=[0.5, 0.5], seed=SEED
    )
    print("PlantDoc merged with field data.")
else:
    field_train_over = field_train.repeat(3)

# Combine: interleave PV and real-world batches
combined_train = tf.data.Dataset.sample_from_datasets(
    [pv_train, field_train_over], weights=[0.5, 0.5], seed=SEED
)
def augment_and_prep(x, y):
    x = augment(x, training=True)
    x = eff_prep(x)
    return x, y

def prep_only(x, y):
    return eff_prep(x), y

combined_train = combined_train.map(augment_and_prep, num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)

# Validation: concatenate both — we want accuracy on both domains
combined_val = pv_val.concatenate(field_val).map(prep_only, num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)

# ── Build model + load weights ───────────────────────────────────────────────
from model_utils import build_model, load_weights_h5, save_savedmodel

print(f"\nBuilding EfficientNetB4 model...")
NUM_CLASSES = len(pv_train.class_names)
model = build_model(num_classes=NUM_CLASSES, backbone_trainable=True)

# Load trained head weights; backbone stays at ImageNet weights (was frozen in Phase 1)
print(f"Loading head weights from {MODEL_IN_WEIGHTS}")
load_weights_h5(model, MODEL_IN_WEIGHTS)

# Freeze backbone for Phase A
base_model = model.get_layer('efficientnetb4')
base_model.trainable = False

# Reinitialise dense head so it learns both PV and field features from scratch
for layer in model.layers:
    if hasattr(layer, 'kernel_initializer') and layer.name != 'efficientnetb4':
        layer.kernel.assign(layer.kernel_initializer(layer.kernel.shape))
        if layer.bias is not None:
            layer.bias.assign(layer.bias_initializer(layer.bias.shape))

print(f"Backbone frozen. Head re-initialised. Trainable params: {sum(np.prod(v.shape) for v in model.trainable_variables):,}")

# ── Baseline on each domain ───────────────────────────────────────────────────
model.compile(optimizer=Adam(1e-3), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
_, acc_pv    = model.evaluate(pv_val.map(prep_only).prefetch(AUTOTUNE),    verbose=0)
_, acc_field = model.evaluate(field_val.map(prep_only).prefetch(AUTOTUNE), verbose=0)
print(f"\nBaseline — PlantVillage: {acc_pv*100:.1f}%  |  Field: {acc_field*100:.1f}%")

# ── Phase A: Head-only, LR=1e-3 ─────────────────────────────────────────────
print("\n══ Phase A: Joint head training (LR=1e-3, frozen backbone) ══")

LOG_A = os.path.join(os.path.dirname(LOG_PATH), 'joint_log_a.csv')
LOG_B = os.path.join(os.path.dirname(LOG_PATH), 'joint_log_b.csv')

callbacks_a = [
    ModelCheckpoint(MODEL_OUT_A_W, monitor='val_accuracy', save_best_only=True, mode='max', verbose=1, save_weights_only=True),
    EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-7, verbose=1),
    CSVLogger(LOG_A),
]

model.fit(
    combined_train, validation_data=combined_val,
    epochs=EPOCHS, callbacks=callbacks_a, verbose=1
)

_, acc_pv_a    = model.evaluate(pv_val.map(prep_only).prefetch(AUTOTUNE),    verbose=0)
_, acc_field_a = model.evaluate(field_val.map(prep_only).prefetch(AUTOTUNE), verbose=0)
print(f"\nPhase A — PlantVillage: {acc_pv_a*100:.1f}%  |  Field: {acc_field_a*100:.1f}%")

# ── Phase B: Unfreeze last conv block, LR=1e-5 ───────────────────────────────
print("\n══ Phase B: Unfreeze last backbone conv block (LR=1e-5) ══")
base_model.trainable = True
for layer in base_model.layers[:-15]:
    layer.trainable = False
trainable = sum(1 for l in base_model.layers if l.trainable)
print(f"Unfrozen backbone layers: {trainable} / {len(base_model.layers)}")

model.compile(
    optimizer=Adam(learning_rate=1e-5),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

callbacks_b = [
    ModelCheckpoint(MODEL_OUT_B_W, monitor='val_accuracy', save_best_only=True, mode='max', verbose=1, save_weights_only=True),
    EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-8, verbose=1),
    CSVLogger(LOG_B),
]

model.fit(
    combined_train, validation_data=combined_val,
    epochs=EPOCHS, callbacks=callbacks_b, verbose=1
)

_, acc_pv_b    = model.evaluate(pv_val.map(prep_only).prefetch(AUTOTUNE),    verbose=0)
_, acc_field_b = model.evaluate(field_val.map(prep_only).prefetch(AUTOTUNE), verbose=0)
combined_b = acc_pv_b + acc_field_b

# ── Compare Phase A vs Phase B, save best as SavedModel ─────────────────────
# Rebuild Phase A model and evaluate
model_a = build_model(num_classes=NUM_CLASSES, backbone_trainable=True)
load_weights_h5(model_a, MODEL_OUT_A_W)
model_a.compile(loss='sparse_categorical_crossentropy', metrics=['accuracy'])
_, acc_pv_a2    = model_a.evaluate(pv_val.map(prep_only).prefetch(AUTOTUNE),    verbose=0)
_, acc_field_a2 = model_a.evaluate(field_val.map(prep_only).prefetch(AUTOTUNE), verbose=0)
combined_a = acc_pv_a2 + acc_field_a2

print(f"\nPhase A — PlantVillage: {acc_pv_a2*100:.1f}%  |  Field: {acc_field_a2*100:.1f}%  (combined score: {combined_a:.4f})")
print(f"Phase B — PlantVillage: {acc_pv_b*100:.1f}%  |  Field: {acc_field_b*100:.1f}%  (combined score: {combined_b:.4f})")

best_model = model_a if combined_a >= combined_b else model
phase_label = 'A' if combined_a >= combined_b else 'B'
save_savedmodel(best_model, MODEL_OUT)
print(f"\nPhase {phase_label} was better — saved as: {MODEL_OUT}")
print("Done.")
