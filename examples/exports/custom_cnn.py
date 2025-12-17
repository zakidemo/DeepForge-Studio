# DeepForge Studio - Exported Training Pipeline
# Generated: 2025-12-17T10:38:55.786Z

import os, random
import numpy as np

def set_seed(seed=42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except Exception:
        pass

set_seed(42)

import tensorflow as tf
from tensorflow.keras import layers, models

# Model architecture
model = models.Sequential([
    layers.Conv2D(32, (3, 3), strides=1, padding='valid', activation='relu', input_shape=(224, 224, 3)),
    layers.Conv2D(32, (3, 3), strides=1, padding='valid', activation='relu'),
    layers.MaxPooling2D((2, 2), strides=2, padding='valid'),
    layers.Conv2D(32, (3, 3), strides=1, padding='valid', activation='relu'),
    layers.Conv2D(32, (3, 3), strides=1, padding='valid', activation='relu'),
    layers.MaxPooling2D((2, 2), strides=2, padding='valid'),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(5, activation='softmax')
])

# Summary
model.summary()

# NOTE: Compilation and training steps are included in the exported training pipeline.


# ============================
# TRAINING PIPELINE (EDIT DATA PATH)
# ============================
import tensorflow as tf

# Expected folder structure:
# DATA_DIR/
#   class_a/
#   class_b/
#   ...
DATA_DIR = "path/to/your/image_dataset"
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 50

train_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.2,
    subset="training",
    seed=42,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="categorical",
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.2,
    subset="validation",
    seed=42,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="categorical",
)

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# Re-compile using UI-selected hyperparams (overrides template defaults safely)
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS)

model.save("deepforge_model.keras")
print("Saved model to deepforge_model.keras")
