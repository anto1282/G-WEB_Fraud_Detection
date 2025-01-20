import pytest
import torch
from torch_geometric.loader import NeighborLoader
from src.gweb.data import AMLtoGraph
from torch_geometric.transforms import RandomNodeSplit
from tests import _PATH_DATA

# Loading dataset:
@pytest.fixture
def dataset():
    dataset = AMLtoGraph(_PATH_DATA)
    return dataset[0]  # Access the first data object

def test_data_loading(dataset):
    # Checking that data loads and is not empty
    assert dataset.x is not None, "Feature matrix (x) should not be None."
    assert dataset.edge_index is not None, "Edge index should not be None."
    assert dataset.y is not None, "Labels (y) should not be None."
    assert dataset.x.shape[0] > 0, "Dataset should contain nodes."
    assert dataset.edge_index.shape[1] > 0, "Dataset should contain edges."

def test_node_features_shape(dataset):
    # Verifying the feature matrix shape
    assert dataset.x.dim() == 2, "Feature matrix (x) should be 2-dimensional."
    assert dataset.x.size(1) > 0, "Feature matrix (x) should have at least one feature."

def test_edge_index_shape(dataset):
    # Verifying the edge index shape
    assert dataset.edge_index.dim() == 2, "Edge index should be 2-dimensional."
    assert dataset.edge_index.size(0) == 2, "Edge index should have exactly two rows."

def test_labels_shape(dataset):
    # Verifying the label vector shape
    assert dataset.y.dim() == 1, "Labels (y) should be 1-dimensional."
    assert dataset.y.size(0) == dataset.x.size(0), "Number of labels should match the number of nodes."

def test_random_node_split(dataset):
    # Checking if RandomNodeSplit works correctly
    split = RandomNodeSplit(split="train_rest", num_val=0.2, num_test=0.2)
    data = split(dataset)
    assert hasattr(data, "train_mask"), "Data should have a train_mask attribute."
    assert hasattr(data, "val_mask"), "Data should have a val_mask attribute."
    assert hasattr(data, "test_mask"), "Data should have a test_mask attribute."
    assert data.train_mask.sum() > 0, "Train mask should select some nodes."
    assert data.val_mask.sum() > 0, "Validation mask should select some nodes."
    assert data.test_mask.sum() > 0, "Test mask should select some nodes."

def test_edge_attributes(dataset):
    # Verifying edge attributes
    assert dataset.edge_attr is not None, "Edge attributes should not be None."
    assert dataset.edge_attr.dim() == 2, "Edge attributes should be 2-dimensional."
    assert dataset.edge_attr.size(0) == dataset.edge_index.size(1), "Number of edge attributes should match the number of edges."

def test_data_device_transfer(dataset):
    # Checking if data can be transferred to GPU (if available)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = dataset.to(device)
    assert dataset.x.is_cuda == torch.cuda.is_available(), "Feature matrix (x) should be on the correct device."
    assert dataset.edge_index.is_cuda == torch.cuda.is_available(), "Edge index should be on the correct device."
    if dataset.edge_attr is not None:
        assert dataset.edge_attr.is_cuda == torch.cuda.is_available(), "Edge attributes should be on the correct device."

def test_graph_connectivity(dataset):
    # Verifying graph connectivity
    num_nodes = dataset.x.size(0)
    edge_index = dataset.edge_index.cpu().numpy()
    connected_nodes = set(edge_index[0]).union(set(edge_index[1]))
    assert len(connected_nodes) == num_nodes, "All nodes should be part of the graph connectivity."
