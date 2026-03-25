import torch
import matplotlib.pyplot as plt

from src.data.datagen import Data_object
from src.methods.drnet import DRNet

# -----------------------
# Define args (same as training)
# -----------------------
args = {
    "dataset": "data/loan_data",
    "confounding_bias": -1,
    "test_fraction": 0.2,
    "val_fraction": 0.1,
    "noise_std": 0.0,
    "gt": None,
    "seed": 42
}

# -----------------------
# Load data
# -----------------------
data = Data_object(args)

# -----------------------
# Define config (inputSize MUST match data)
# -----------------------
config = {
    "learningRate": 1e-3,
    "batchSize": 256,
    "numSteps": 5000,          # not used here, but needed by DRNet
    "numLayers": 3,
    "inputSize": data.x.shape[1],
    "hiddenSize": 64,
    "numHeads": 10
}

# -----------------------
# Load trained model
# -----------------------
model = DRNet(config)
model.load_state_dict(torch.load("drnet.pth", map_location="cpu"))
model.eval()

# -----------------------
# Pick one test observation
# -----------------------
obs = data.dataset_test["x"][0]

# -----------------------
# Get dose–response curve
# -----------------------
d_values, y_values = model.getDR(obs)

# -----------------------
# Plot
# -----------------------
plt.figure(figsize=(8, 5))
plt.plot(d_values, y_values, marker="o")
plt.xlabel("Interest Rate (Dosage)")
plt.ylabel("Approval Probability")
plt.title("Dose–Response Curve (DRNet)")
plt.grid(True)
plt.show()
