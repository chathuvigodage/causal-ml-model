from fastapi import FastAPI
from pydantic import BaseModel
import torch
import numpy as np
import pandas as pd
import json
import logging
import time
import uuid

from src.methods.drnet import DRNet

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

# -----------------------
# Load feature schema
# -----------------------
with open("feature_schema.json") as f:
    FEATURE_ORDER = json.load(f)

# -----------------------
# Load normalization stats (saved during training by datagen.py)
# -----------------------
with open("norm_stats.json") as f:
    NORM_STATS = json.load(f)

# -----------------------
# FastAPI app
# -----------------------
app = FastAPI(title="DRNet Loan Approval API")

# -----------------------
# Input schema (STRICT)
# -----------------------
class LoanRequest(BaseModel):
    LoanAmount: float
    LoanDuration: float
    DebtToIncomeRatio: float
    CreditScore: float
    NumberOfOpenCreditLines: int
    AnnualIncome: float
    SavingsAccountBalance: float
    TotalLiabilities: float
    Age: int
    EducationLevel: str
    MaritalStatus: str
    EmploymentStatus: str
    PaymentHistory: float
    InterestRate: float

# -----------------------
# Load model
# -----------------------
model = None

@app.on_event("startup")
def load_model():
    global model
    logger.info("[startup] application starting...")

    config = {
        "learningRate": 1e-3,
        "batchSize": 256,
        "numSteps": 5000,
        "numLayers": 3,
        "inputSize": len(FEATURE_ORDER),
        "hiddenSize": 64,
        "numHeads": 10
    }

    logger.info("[startup] loading DRNet model...")
    model = DRNet(config)
    model.load_state_dict(torch.load("drnet.pth", map_location="cpu"))
    model.eval()

    logger.info("[startup] DRNet model loaded successfully")
    print("✅ DRNet loaded")

# -----------------------
# Preprocessing
# -----------------------
def preprocess(req: LoanRequest):
    raw = req.dict()
    d_raw = raw.pop("InterestRate")

    df = pd.DataFrame([raw])
    df = pd.get_dummies(df)

    # 🔒 FORCE same columns & order
    df = df.reindex(columns=FEATURE_ORDER, fill_value=0)

    x_arr = df.values.astype("float32")

    # --- Normalization fix: apply same min-max scaling used during training ---
    x_min = np.array(NORM_STATS["x_min"], dtype="float32")
    x_max = np.array(NORM_STATS["x_max"], dtype="float32")
    x_arr = (x_arr - x_min) / (x_max - x_min + 1e-8)

    d_min = NORM_STATS["d_min"]
    d_max = NORM_STATS["d_max"]
    d_norm = (d_raw - d_min) / (d_max - d_min + 1e-8)

    x = torch.tensor(x_arr, dtype=torch.float32)
    d = torch.tensor([d_norm], dtype=torch.float32)

    return x, d

# -----------------------
# Prediction endpoint
# -----------------------
@app.post("/predict")
def predict(req: LoanRequest):
    request_id = str(uuid.uuid4())[:8]
    logger.info(f"[predict][request_id={request_id}] request received on endpoint /predict")
    start_time = time.time()
    
    try:
        x, d = preprocess(req)
        
        logger.info(f"[predict][request_id={request_id}] prediction started")
        with torch.no_grad():
            prob = model.predictObservation(x, d)[0]
        
        end_time = time.time()
        duration = end_time - start_time
        logger.info(f"[predict][request_id={request_id}] prediction completed in {duration:.4f}s (start: {start_time:.2f}, end: {end_time:.2f})")
        
        return {
            "acceptance_probability": float(prob)
        }
    except Exception as e:
        logger.error(f"[predict][request_id={request_id}] prediction failed: {type(e).__name__} - {str(e)}", exc_info=True)
        raise
