import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, Linear


# GCN with attention mechanism
class GCN(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        heads: int,
        dropout: float,
    ) -> None:
        super().__init__()

        self.dropout = dropout
        self.conv1 = GATConv(
            in_channels, hidden_channels, heads, dropout=dropout
        )
        self.conv2 = GATConv(
            hidden_channels * heads,
            int(hidden_channels / 4),
            heads=1,
            concat=False,
            dropout=dropout,
        )
        self.lin = Linear(int(hidden_channels / 4), out_channels)
        self.sigmoid = nn.Sigmoid()

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.elu(self.conv1(x, edge_index, edge_attr))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.elu(self.conv2(x, edge_index, edge_attr))
        x = self.lin(x)
        x = self.sigmoid(x)

        return x


def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model = GCN(
        checkpoint["hidden_channels"],
        checkpoint["in_channels"],
        checkpoint["out_channels"],
        checkpoint["dropout"],
    )
    model.load_state_dict(checkpoint["state_dict"])

    return model


if __name__ == "__main__":
    # Example parameters for a simple graph
    in_channels = 16  # Number of features per node
    hidden_channels = 32
    out_channels = 1  # Binary classification or regression
    heads = 4  # Attention heads
    dropout = 0.6

    # Create an example graph with 5 nodes and 4 edges
    x = torch.randn(
        5, in_channels
    )  # Node features (5 nodes, 16 features each)
    edge_index = torch.tensor(
        [[0, 1, 2, 3], [1, 0, 3, 4]], dtype=torch.long
    )  # Edge list (4 edges)
    edge_attr = torch.randn(4, 5)  # Edge features (4 edges, 5 features each)

    # Instantiate the model
    model = GCN(in_channels, hidden_channels, out_channels, heads, dropout)
    print(f"Model architecture: {model}")
    print(
        f"Number of parameters: {sum(p.numel() for p in model.parameters())}"
    )

    # Perform a forward pass
    output = model(x, edge_index, edge_attr)
    print(f"Output shape: {output.shape}")
