# ============================================================
# COLAB SETUP — paste this as the FIRST cell, above the exported code
# ============================================================
# Downloads a public dataset (tf_flowers: 3,670 photos, 5 classes) in the
# class-folder layout the exported script expects. This is the "public
# dataset" the functional evaluation reports against.
#
# IMPORTANT: export from the tool with "Number of Classes" = 5,
# otherwise the classification head will not match the data.

import pathlib, tensorflow as tf

_archive = tf.keras.utils.get_file(
    "flower_photos.tgz",
    "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz",
    extract=True,
)
FLOWERS_DIR = str(next(pathlib.Path(_archive).parent.rglob("flower_photos")))
print("Dataset ready at:", FLOWERS_DIR)
print("Classes:", sorted(p.name for p in pathlib.Path(FLOWERS_DIR).iterdir() if p.is_dir()))

# ============================================================
# Then, in the exported script below, change ONE line:
#
#     DATA_DIR = "path/to/your/image_dataset"
# to:
#     DATA_DIR = FLOWERS_DIR
#
# and lower the epochs for a verification run:
#
#     EPOCHS = 2
#
# Change nothing else. The evaluation protocol reported in the paper is that
# the exported artefact runs with no edit other than the data path, so any
# further change you have to make is a finding worth recording.
# ============================================================
