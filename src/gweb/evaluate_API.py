import torch
from torch_geometric.loader import NeighborLoader
from model import GCN
from data import AMLtoGraph
import torch_geometric.transforms as T
import typer
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def test(
    model_path: str = "/zhome/45/0/155089/G-WEB_Fraud_Detection/models/model.pth",  # Assuming the model path is local
    batchsize: int = 256,
    hdn_chnls: int = 16,
    atn_heads: int = 4,
    drop_out: float = 0.6,
) -> None:
    torch.manual_seed(42)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    # Load dataset
    dataset = AMLtoGraph("/dtu/blackhole/0e/154958/data_small")
    data = dataset[0]

    # Model parameters
    hidden_channels = hdn_chnls
    heads = atn_heads
    dropout = drop_out

    model = GCN(
        in_channels=data.num_features,
        hidden_channels=hidden_channels,
        out_channels=1,
        heads=heads,
        dropout=dropout,
    )
    model = model.to(device)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    
    model.eval()

    split = T.RandomNodeSplit(split="train_rest", num_val=0.2, num_test=0.2)
    data = split(data)

    test_loader = NeighborLoader(
        data,
        num_neighbors=[30] * 2,
        batch_size=batchsize,
        input_nodes=data.test_mask,
    )

    total_correct = 0
    total_samples = 0
    all_preds = []
    all_labels = []
    i = 0 
    with torch.no_grad():
        for test_data in test_loader:
            i += 1 
            test_data.to(device)
            pred = model(
                test_data.x, test_data.edge_index, test_data.edge_attr
            )
            pred.to(device)
            ground_truth = test_data.y
            predictions = (pred > 0.5).float()
            all_preds.extend(predictions.flatten().cpu().numpy())
            all_labels.extend(ground_truth.flatten().cpu().numpy())

            total_correct += (
                (predictions == ground_truth.unsqueeze(1)).sum().item()
            )
            total_samples += len(ground_truth)
            

    all_preds = [int(i) for i in all_preds]
    all_labels = [int(i) for i in all_labels]

    accuracy = total_correct / total_samples if total_samples > 0 else 0
    print(f"Test Accuracy: {accuracy:.4f}")
    

    # Confusion Matrix and Plot
    cm = confusion_matrix(all_labels, all_preds)
    print("Confusion Matrix:")
    print(cm)

    class_names = ["Normal", "Fraud"]
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='g', cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.show()
    print("so many runs", i)

    # Optionally, save the confusion matrix plot
    plt.savefig("confusion_matrix.png")
    return accuracy, cm


