import pytest
import torch
from src.gweb.model import GCN

def test_model():
    # Define model architecture and input dimensions
    in_channels = 16 
    hidden_channels = 32
    out_channels = 1
    heads = 4
    dropout = 0.6

    # Instantiate the model
    model = GCN(in_channels, hidden_channels, out_channels, heads, dropout)

    # We use example data
    num_nodes = 10
    num_edges = 20

    x = torch.randn(num_nodes, in_channels)
    edge_index = torch.randint(0, num_nodes, (2, num_edges))
    edge_attr = torch.randn(num_edges, 5)

    # Do forward pass
    y = model(x, edge_index, edge_attr)

    # Check output shape
    assert y.shape == (num_nodes, out_channels), (
        f"Expected output shape ({num_nodes}, {out_channels}), got {y.shape}."
    )
