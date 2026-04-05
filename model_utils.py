"""
Shared model utilities for Maize Disease Classifier.

Works around TF 2.10 limitations:
  1. EfficientNetB4 embeds a Normalization layer with EagerTensor params that
     cannot be JSON-serialised — so model.save() fails for all formats.
  2. model.load_weights() fails on depthwise-conv kernels due to a transpose
     axes bug in the H5 loader.

Solution:
  - Always save weights-only (.weights.h5) via ModelCheckpoint.
  - Reconstruct the architecture with build_model() and load via load_weights_h5().
  - Save final inference model with tf.saved_model.save() (protobuf, no JSON).
"""

import os
import json
import numpy as np
import h5py
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import (Input, GlobalAveragePooling2D,
                                     Dense, Dropout, BatchNormalization, Activation)


def build_model(num_classes: int = 4, img_size: tuple = (224, 224),
                backbone_trainable: bool = True) -> tf.keras.Model:
    """Build the EfficientNetB4-based classifier."""
    base = tf.keras.applications.EfficientNetB4(
        weights='imagenet', include_top=False,
        input_shape=(*img_size, 3)
    )
    base.trainable = backbone_trainable

    inputs  = Input(shape=(*img_size, 3))
    x       = base(inputs, training=False)
    x       = GlobalAveragePooling2D()(x)
    x       = BatchNormalization()(x)
    x       = Dense(256, activation='relu')(x)
    x       = Dropout(0.4)(x)
    # Separate Dense (no activation) + Activation so model.layers[-1].input = logits
    logits  = Dense(num_classes)(x)
    outputs = Activation('softmax')(logits)
    return Model(inputs, outputs, name='maize_efficientnetb4')


def _canonical(h5_path: str) -> str:
    """Strip 'efficientnetb4/' prefix and collapse duplicate layer-name segments.

    Examples:
      efficientnetb4/block1a_bn/beta:0          → block1a_bn/beta:0
      batch_normalization/batch_normalization/beta:0 → batch_normalization/beta:0
      dense/dense/kernel:0                       → dense/kernel:0
    """
    parts = h5_path.split('/')
    if parts[0] == 'efficientnetb4':
        parts = parts[1:]
    if len(parts) >= 2 and parts[0] == parts[1]:
        parts = [parts[0]] + parts[2:]
    return '/'.join(parts)


def load_weights_h5(model: tf.keras.Model, path: str) -> None:
    """Load weights from an H5 file, bypassing TF 2.10 transpose bugs.

    Matches weights by canonical name (strips layer prefixes and duplicate
    segments) so it works for both frozen-backbone and fine-tuned checkpoints.
    """
    # Flatten the whole H5 file into {canonical_path: ndarray}
    h5_map: dict[str, np.ndarray] = {}
    with h5py.File(path, 'r') as f:
        def _collect(grp, prefix=''):
            for key in grp.keys():
                full = f"{prefix}/{key}" if prefix else key
                item = grp[key]
                if isinstance(item, h5py.Dataset):
                    h5_map[_canonical(full)] = item[()]
                else:
                    _collect(item, full)
        _collect(f)

    loaded, skipped = 0, 0
    for weight in model.weights:
        # Model weight name examples: 'stem_conv/kernel:0', 'dense/kernel:0'
        name = weight.name  # already in canonical form
        if name in h5_map:
            weight.assign(h5_map[name])
            loaded += 1
        else:
            skipped += 1

    print(f"load_weights_h5: loaded={loaded}  skipped={skipped}  "
          f"(of {len(model.weights)} model weights, {len(h5_map)} in file)")


def build_logit_model(model: tf.keras.Model) -> tf.keras.Model:
    """Return a model whose output is logits (input to the final softmax)."""
    return tf.keras.Model(
        inputs=model.input,
        outputs=model.layers[-1].input,
        name='logit_model'
    )


def save_savedmodel(model: tf.keras.Model, path: str) -> None:
    """Save using tf.saved_model.save — avoids EagerTensor JSON bug."""
    tf.saved_model.save(model, path)
    size = sum(
        os.path.getsize(os.path.join(r, f))
        for r, _, files in os.walk(path)
        for f in files
    )
    print(f"SavedModel written: {path}  ({size / 1e6:.1f} MB)")
