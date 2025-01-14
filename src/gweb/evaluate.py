import torch
from torch_geometric.loader import NeighborLoader
from model import GCN
from data import AMLtoGraph
import torch_geometric.transforms as T
import typer

def test(model_path: str, batchsize: int = 256, hdn_chnls: int = 16, atn_heads: int = 4,drop_out: float = 0.6) -> None:
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

    # Load dataset
    dataset = AMLtoGraph('/dtu/blackhole/0e/154958/data') 
    data = dataset[0]

    # Model parameters
    hidden_channels = hdn_chnls
    heads = atn_heads
    dropout = drop_out


    model = GCN(in_channels=data.num_features, hidden_channels=hidden_channels, out_channels=1, heads=heads,dropout=dropout)
    model.load_state_dict(torch.load(model_path))
    model = model.to(device)
    model.eval()

    split = T.RandomNodeSplit(split='train_rest', num_val=0.2, num_test=0.2)
    data = split(data)

    test_loader = NeighborLoader(
        data,
        num_neighbors=[30] * 2,
        batch_size=batchsize,
        input_nodes=data.test_mask,
    )

    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for test_data in test_loader:
            test_data.to(device)
            pred = model(test_data.x, test_data.edge_index, test_data.edge_attr)
            ground_truth = test_data.y
            predictions = (pred > 0.5).float() 
            total_correct += (predictions == ground_truth.unsqueeze(1)).sum().item()
            total_samples += len(ground_truth)

    accuracy = total_correct / total_samples if total_samples > 0 else 0
    print(f"Test Accuracy: {accuracy:.4f}")
    
if __name__ == "__main__":
    typer.run(test)
    
    