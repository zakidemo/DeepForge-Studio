# DeepForge Studio - Exported Training Pipeline
# Generated: 2025-12-17T10:39:09.330Z

import os, random
import numpy as np

def set_seed(seed=42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

set_seed(42)


# ============================
# DATA LOADING (EDIT THIS)
# ============================
# Provide X (features) and y (labels) as numpy arrays.
# Example:
#   import pandas as pd
#   df = pd.read_csv("your_data.csv")
#   X = df.drop("label", axis=1).values
#   y = df["label"].values
#
# X = ...
# y = ...

# Guard: ensure X and y are defined before proceeding
try:
    X
    y
except NameError as e:
    raise NameError("Please define X and y before running. See the DATA LOADING section.") from e

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

# K-Nearest Neighbors (KNN)
knn = KNeighborsClassifier(
    n_neighbors=5,
    weights='uniform',
    algorithm='auto',
    metric='euclidean',
    n_jobs=-1
)

pipeline = Pipeline([("scaler", StandardScaler()), ("knn", knn)])

# Train
pipeline.fit(X_train, y_train)

# Predict
y_pred = pipeline.predict(X_test)

# Evaluate
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("Done.")
