import torch
import torch.optim.sgd
import torch_geometric.transforms as T
from torch_geometric.loader import NeighborLoader
import typer
from sklearn.metrics import f1_score
from model import GCN
from data import AMLtoGraph
import wandb
import numpy as np
import os
dataset = AMLtoGraph("/dtu/blackhole/0e/154958/data_small")  # Adjust path

data = dataset[0]

print("data here___________")
# Assuming 'data' is a PyTorch Geometric Data object
print("First observation of x (node features):")
print(data.x[0])  # Prints the first node feature vector

print("\nFirst observation of edge_index (edge list):")
print(data.edge_index[:, 0])  # Prints the first edge (source and target node indices)

print("\nFirst observation of edge_attr (edge attributes):")
print(data.edge_attr[0])  # Prints the first edge's attribute values


print(data)