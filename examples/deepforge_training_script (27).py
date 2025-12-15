# DeepForge Studio - Exported Training Pipeline
# Generated: 2025-12-15T21:31:43.533Z

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
from tensorflow.keras.applications import VGG16

from tensorflow.keras.applications.vgg16 import preprocess_input

# Load pretrained VGG16 model
base_model = VGG16(
    input_shape=(224, 224, 3),
    include_top=False,
    weights='imagenet'
)

# Create the complete model
inputs = tf.keras.Input(shape=(224, 224, 3))

# Data augmentation
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.2),
    layers.RandomZoom(0.2),
])

x = data_augmentation(inputs)
x = preprocess_input(x)
x = base_model(x, training=False)

# Default classification head
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.2)(x)
outputs = layers.Dense(10, activation='softmax')(x)

# Build the model
model = tf.keras.Model(inputs, outputs)

# Compile with different learning rates for fine-tuning
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Model summary
print(f"Total layers: {len(model.layers)}")
print(f"Trainable layers: {sum([layer.trainable for layer in model.layers])}")
model.summary()

# ============================
# TRAINING PIPELINE (EDIT DATA PATH)
# ============================
import tensorflow as tf

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
