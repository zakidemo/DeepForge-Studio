# DeepForge Studio - Exported Training Pipeline
# Generated: 2025-12-15T21:32:22.532Z


# ============================
# DATA LOADING (EDIT THIS)
# ============================
# Provide X (features) and y (labels) as numpy arrays.
# Example:
#   import pandas as pd
#   df = pd.read_csv("your_data.csv")
#   X = df.drop("label", axis=1).values
#   y = df["label"].values

# X = ...
# y = ...

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# K-Nearest Neighbors Classifier
model = KNeighborsClassifier(
    n_neighbors=5,
    weights='uniform',
    algorithm='auto',
    metric='euclidean',
    n_jobs=-1
)

# Data preprocessing
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the model
model.fit(X_train_scaled, y_train)

# Make predictions
y_pred = model.predict(X_test_scaled)

# Evaluate
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("Done.")
