import onnxruntime as ort
import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from torch_geometric.data import Data
import os
from http import HTTPStatus
from pathlib import Path 
from model import GCN, load_checkpoint  # Replace 'gcn_model' with the correct file name

# Initialize FastAPI app
app = FastAPI()

# Get the absolute path to the model file relative to the current file
#current_dir = os.path.dirname(__file__)  # Get the current script's directory
#model_path = os.path.join(current_dir, '..', '..', '..', 'models', 'model_1_0.001_16.pth')
#current_dir = os.path.dirname(os.path.abspath(__file__))  # Get the absolute path of the current script
#model_path = os.path.join(current_dir, '..', '..', '..', 'models', 'model_1_0.001_16.pth')

#model_path = Path("G-WEB_Fraud_Detection/models/model_1_0_001_16.pth")

#C:\Users\jbirk\OneDrive\Skrivebord\MLops\G-WEB_Fraud_Detection\models\model_1_0_001_16.pth
#C:\\Users\\jbirk\\OneDrive\\Skrivebord\\MLops\\G-WEB_Fraud_Detection\\src\\models\\model_1_0_001_16.pth
#C:\\Users\\jbirk\\OneDrive\\Skrivebord\\MLops\\G-WEB_Fraud_Detection\\src\\src\\models\\model_1_0_001_16.pth'
model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..','..', 'models', 'model_2_0_001_16.pth'))

# Debugging: Print the model path to confirm
#print(f" model path: {model_path}")
#print(f"Current directory: {current_dir}")
print(f"Expected model path: {model_path}")


# Define the model architecture and input dimensions
def load_checkpoint(model_path, in_channels, hidden_channels, out_channels, dropout, heads):
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    print("Checkpoint keys:", checkpoint.keys())

    # Initialize the model with the parameters (like in_channels, hidden_channels, etc.)
    model = GCN(in_channels=in_channels, 
                hidden_channels=hidden_channels, 
                out_channels=out_channels, 
                heads=heads, 
                dropout=dropout)
    
    # Load the state_dict into the model
    model.load_state_dict(checkpoint)
    
    return model
# Load the model using the checkpoint


def load_model(model_path):
    if os.path.isfile(model_path):
        print(f"Loading model from: {model_path}")
        # Use the load_checkpoint function to load the model weights
        # Define the model architecture

        in_channels = 16  # Example: Number of features per node
        hidden_channels = 16
        out_channels = 1  # Binary classification output
        heads = 4 # Number of attention heads in GATConv layers
        dropout = 0.6  # Dropout rate
#    size mismatch for conv1.att_src: copying a param with shape torch.Size([1, 4, 16]) from checkpoint, the shape in current model is torch.Size([1, 16, 32]).
        # Load the model
        model_path = r'C:\Users\jbirk\OneDrive\Skrivebord\MLops\G-WEB_Fraud_Detection\models\model_1_0_001_16.pth'
        model = load_checkpoint(model_path, in_channels, hidden_channels, out_channels, dropout, heads)

        # Check the model architecture
        print(model)
        model = load_checkpoint(model_path)
        model.eval()  # Set to evaluation mode
        print("Model loaded successfully.")
        return model
    else:
        print(f"Model file not found: {model_path}")
        return None
# Load the model


load_model(model_path)
# Load the model (ONNX format)
#try:
#    ort_session = ort.InferenceSession(model_path)
#    print("Model loaded successfully!")
#except Exception as e:
#    print(f"Error loading model: {e}")

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
        
        
        
        # if .pth format
        
        model_input = graph_data.x.unsqueeze(0)  # Add batch dimension (assuming batch size = 1)

        # Run inference using the loaded PyTorch model
        with torch.no_grad():  # No need to track gradients for inference
            output = model(model_input)  # Assuming your model takes the graph input and outputs a prediction

        # Extract the prediction from the model output (e.g., if output > 0.5, it's suspicious)
        result = output.squeeze().item()  
        
        # end if .pth format
        
        # if ONNX format:

        # Ensure the input has the right shape and batch processing is applied
        # Prepare the data for the model (similar to the training process)
        #input_data = graph_data.x.numpy()  # Convert to NumPy array for ONNX inference

        # Run inference using ONNX Runtime
        #inputs = {ort_session.get_inputs()[0].name: input_data}
        #outputs = ort_session.run(None, inputs)
        
        

        # Extract the prediction from the model output (e.g., if output > 0.5, it's suspicious)
        #result = outputs[0]  # Assuming the first output is the prediction
        # end if ONNX format
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
