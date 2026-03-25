from src.data.datagen import Data_object
from src.methods.drnet import DRNet
import torch
import numpy as np
import json

args = {
    "dataset": "data/loan_data",
    "confounding_bias": -1,
    "test_fraction": 0.2,
    "val_fraction": 0.1,
    "noise_std": 0.0,
    "gt": None,
    "seed": 42
}

# -------------------------
# Load & preprocess data
# -------------------------
data = Data_object(args)

# -------------------------
# Save feature schema
# -------------------------
with open("feature_schema.json", "w") as f:
    json.dump(data.feature_names, f, indent=2)

print("✅ Feature schema saved")

# -------------------------
# Class imbalance fix: compute pos_weight from training labels
# pos_weight = (# negative samples) / (# positive samples)
# This up-weights the minority class during MSE loss computation.
# -------------------------
y_train = data.dataset_train['y']
n_pos = np.sum(y_train == 1)
n_neg = np.sum(y_train == 0)
pos_weight = float(n_neg / n_pos) if n_pos > 0 else 1.0
print(f"✅ Class balance — Approved: {n_pos}, Rejected: {n_neg}, pos_weight: {pos_weight:.4f}")

# -------------------------
# Model config
# -------------------------
config = {
    "learningRate": 1e-3,
    "batchSize": 256,
    "numSteps": 5000,
    "numLayers": 3,
    "inputSize": data.x.shape[1],  # NUMBER OF FEATURES
    "hiddenSize": 64,
    "numHeads": 10,
    "dropoutRate": 0.1,            # overfitting fix
    "pos_weight": 1.0              # disabled: computed pos_weight over-inflated mid-range head outputs
}

model = DRNet(config)

# -------------------------
# Train
# -------------------------
model.trainModel(data.dataset_train, data.dataset_val)

# -------------------------
# Save model
# -------------------------
torch.save(model.state_dict(), "drnet.pth")
print("✅ Model saved as drnet.pth")
