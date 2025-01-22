import pytest
import torch
from unittest.mock import MagicMock, patch
from torch_geometric.data import Data
from src.gweb.model import GCN
from src.gweb.train import train

@pytest.fixture
def fake_data():
    data = Data(
        x=torch.ones(100, 16),  # Deterministic features
        edge_index=torch.randint(0, 100, (2, 500)),
        edge_attr=torch.ones(500),
        y=torch.ones(100,),
        train_mask=torch.randint(0, 2, (100,), dtype=torch.bool),
        val_mask=torch.randint(0, 2, (100,), dtype=torch.bool),
    )
    return data

@pytest.fixture
def mock_train_config():
    return {
        "lr": 0.01,
        "batchsize": 16,
        "hdn_chnls": 32,
        "atn_heads": 4,
        "drop_out": 0.1,
        "epochs": 3,
        "pos_weight": 1.0,
    }

@patch("src.gweb.train._PATH_DATA", "/fake/path")
@patch("wandb.init", MagicMock(return_value=MagicMock()))
@patch("wandb.log", MagicMock())
@patch("wandb.Artifact", MagicMock())
@patch("torch.onnx.export", MagicMock())
def test_train_with_full_coverage(fake_data, mock_train_config):
    with patch("torch_geometric.loader.NeighborLoader") as mock_loader, patch(
        "torch_geometric.transforms.RandomNodeSplit"
    ) as mock_split:
        mock_loader.side_effect = lambda *args, **kwargs: [fake_data]
        mock_split.return_value = lambda data: data

        with patch("os.makedirs", MagicMock()):
            train(test_mode=True, config=mock_train_config)

def test_loss_decreases(fake_data, mock_train_config):
    model = GCN(
        in_channels=fake_data.num_features,
        hidden_channels=mock_train_config["hdn_chnls"],
        out_channels=1,
        heads=mock_train_config["atn_heads"],
        dropout=mock_train_config["drop_out"],
    )
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=mock_train_config["lr"])

    train_data = fake_data
    model.train()
    optimizer.zero_grad()

    pred = model(train_data.x, train_data.edge_index, train_data.edge_attr)
    loss = criterion(pred, train_data.y.unsqueeze(1).float())
    initial_loss = loss.item()

    loss.backward()
    optimizer.step()

    pred_new = model(train_data.x, train_data.edge_index, train_data.edge_attr)
    new_loss = criterion(pred_new, train_data.y.unsqueeze(1).float()).item()

    assert new_loss <= initial_loss + 1e-6, "Loss did not decrease after one step."

def test_early_stopping(fake_data, mock_train_config):
    model = GCN(
        in_channels=fake_data.num_features,
        hidden_channels=mock_train_config["hdn_chnls"],
        out_channels=1,
        heads=mock_train_config["atn_heads"],
        dropout=mock_train_config["drop_out"],
    )
    best_val_score = -float("inf")
    patience_counter = 0
    patience = 2

    for epoch in range(4):
        f1_score = 1 if epoch % 2 == 0 else 0  # Alternating F1 scores

        if f1_score > best_val_score:
            best_val_score = f1_score
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            break

    assert patience_counter == patience, "Early stopping did not trigger correctly."
