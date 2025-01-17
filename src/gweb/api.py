import onnxruntime as ort
import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from torch_geometric.data import Data
import os
from http import HTTPStatus
# Initialize FastAPI app
app = FastAPI()

# Get the absolute path to the model file relative to the current file
current_dir = os.path.dirname(os.path.abspath(__file__))  # Path of the current file (api.py)
onnx_model_path = os.path.join(current_dir, "..", "..", "models", "model.onnx")
onnx_model_path = os.path.normpath(onnx_model_path)  # Normalize path for cross-platform compatibility

# Debugging: Print the model path to confirm
print(f"ONNX model path: {onnx_model_path}")

# Load the ONNX model
try:
    ort_session = ort.InferenceSession(onnx_model_path)
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")

class TransactionData(BaseModel):
    node_features: List[List[float]]  # List of node feature vectors
    edge_index: List[List[int]]  # List of edge indices (source, target)
    edge_attr: List[List[float]]  # List of edge attributes
    batch_size: int  # Number of nodes (batch size)
    transaction_id: str  # Transaction ID for reference
@app.get("/")
async def read_root():
    return {"message": "Welcome to the fraud detection API!"}

# Health check endpoint
@app.get("/healthcheck/")
async def healthcheck():
    response = {
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
    }
    return {"status": "Healthy", "message": "Model loaded successfully"}


# Prediction endpoint
@app.post("/predict/")
async def predict_money_laundering(data: TransactionData):
    try:
        # Convert the input into PyTorch Geometric Data object
        edge_index = torch.tensor(data.edge_index, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(data.edge_attr, dtype=torch.float32)
        x = torch.tensor(data.node_features, dtype=torch.float32)

        # Create the PyTorch Geometric Data object
        graph_data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

        # Ensure the input has the right shape and batch processing is applied
        # Prepare the data for the model (similar to the training process)
        input_data = graph_data.x.numpy()  # Convert to NumPy array for ONNX inference

        # Run inference using ONNX Runtime
        inputs = {ort_session.get_inputs()[0].name: input_data}
        outputs = ort_session.run(None, inputs)

        # Extract the prediction from the model output (e.g., if output > 0.5, it's suspicious)
        result = outputs[0]  # Assuming the first output is the prediction
        prediction = "Money laundering" if result[0] > 0.5 else "Not suspicious"
        confidence = result[0]

        # Return the result
        return {
            "transaction_id": data.transaction_id,
            "prediction": prediction,
            "confidence": confidence
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
