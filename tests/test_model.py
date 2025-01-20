import pytest
import torch
from src.gweb.model import GCN

def test_model():
    # Define the model architecture and input dimensions
    in_channels = 16  # Example input feature size per node
    hidden_channels = 32
    out_channels = 1  # Binary classification
    heads = 4
    dropout = 0.6

    # Instantiate the model
    model = GCN(in_channels, hidden_channels, out_channels, heads, dropout)

    # Create example input data
    num_nodes = 10  # Example number of nodes
    num_edges = 20  # Example number of edges

    x = torch.randn(num_nodes, in_channels)  # Node features
    edge_index = torch.randint(0, num_nodes, (2, num_edges))  # Edge index
    edge_attr = torch.randn(num_edges, 5)  # Edge attributes

    # Perform a forward pass
    y = model(x, edge_index, edge_attr)

    # Assert the output shape is correct
    assert y.shape == (num_nodes, out_channels), (
        f"Expected output shape ({num_nodes}, {out_channels}), got {y.shape}."
    )
