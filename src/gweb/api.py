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
from evaluate_API import test 
import streamlit as st
import requests
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import torch.optim.sgd
import torch_geometric.transforms as T
from torch_geometric.loader import NeighborLoader
import typer
from sklearn.metrics import f1_score
from model import GCN
from data import AMLtoGraph
import os
import time 
from evaluate_API import test
import seaborn as sns 
import logging
# Initialize FastAPI app
logging.basicConfig(level=logging.DEBUG)
app = FastAPI()

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
async def predict_money_laundering():
    try:
        # Call the test function to get accuracy and confusion matrix
        accuracy, cm = test()

        # Return accuracy and confusion matrix in the response
        return {"accuracy": accuracy, "confusion_matrix": cm.tolist()}  # cm is likely a numpy array, so convert it to list
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Prediction failed: {str(e)}")
    

# data-visualization endpoint
@app.post("/dataviz/")
async def dataviz():
    try:
        dataset = AMLtoGraph("/dtu/blackhole/0e/154958/data_small")  
        data = dataset[0]
        split = T.RandomNodeSplit(split="train_rest", num_val=0.2, num_test=0.1)
        data = split(data)
        test_loader = NeighborLoader(data,num_neighbors=[30] * 2,batch_size=600,input_nodes=data.test_mask)
        for batch in test_loader:
            break  # We only need the first batch for visualization
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"batch not found: {str(e)}")

    try: 
        # Extract the batch's node indices (n_id) and the corresponding node features (x) and labels (y)
        node_ids = batch.n_id.numpy()  # Node indices for the current batch
        node_features = data.x[node_ids]  # Features for the nodes in the batch
        node_labels = data.y[node_ids]  # Labels for the nodes in the batch

        # Find fraud nodes (assuming 1 is fraud)
        fraud_node_ids = node_ids[node_labels == 1]
        non_fraud_node_ids = node_ids[node_labels == 0]
        edge_index_batch = batch.edge_index .numpy()
    except Exception as e:
        raise HTTPException(status_code=503, detail=f" failed 2: {str(e)}")
    try:
        # the non fraud graph 
        non_fraud_edges = []
        for edge in zip(edge_index_batch[0], edge_index_batch[1]):
            if edge[0] in non_fraud_node_ids and edge[1] in non_fraud_node_ids:
                non_fraud_edges.append((int(edge[0]),int(edge[1])))


        
        # the fraud graph
        fraud_edges = []
        for edge in zip(edge_index_batch[0], edge_index_batch[1]):
            if edge[0] in fraud_node_ids and edge[1] in fraud_node_ids:
                fraud_edges.append((int(edge[0]),int(edge[1])))

        

        # the entire graph
        edges = []
        for edge in zip(edge_index_batch[0], edge_index_batch[1]):
            if (edge[0] in non_fraud_node_ids and edge[1] in non_fraud_node_ids) or (edge[0] in fraud_node_ids and edge[1] in fraud_node_ids) :
                edges.append((int(edge[0]),int(edge[1])))

        # Create a graph using NetworkX
        
    except Exception as e:
        raise HTTPException(status_code=504, detail=f" failed at last: {str(e)}")
    try:
        # Degrees
        
        #logging.debug({"edges": edges.tolist(), "fraud_edges": fraud_edges.tolist(), "non_fraud_edges": non_fraud_edges.tolist(), "degreesfraud": degreesfraud.tolist(), "degreesnonfraud": degreesnonfraud.tolist()})

        # Return accuracy and confusion matrix in the response
        #return {"degreesfraud": degreesfraud.tolist(), "degreesnonfraud": degreesnonfraud.tolist() }  # cm is likely a numpy array, so convert it}
        return {"edges": edges, "fraud_edges": fraud_edges, "non_fraud_edges": non_fraud_edges} # cm is likely a numpy array, so convert it to list
    except Exception as e:
        response = {"edges": edges, "fraud_edges": fraud_edges.tolist(), "non_fraud_edges": non_fraud_edges.tolist()}
        raise HTTPException(status_code=505, detail=f"Prediction failed: {str(e)} {response}")
    
