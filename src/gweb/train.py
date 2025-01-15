import torch
import torch_geometric.transforms as T
from torch_geometric.loader import NeighborLoader
import matplotlib.pyplot as plt
import torch
import typer
from model import GCN
from data import AMLtoGraph
import wandb
from sklearn.metrics import RocCurveDisplay, accuracy_score, f1_score, precision_score, recall_score




def train(epochs: int = 50, lr: float = 0.001,batchsize: int = 256, hdn_chnls: int = 16, atn_heads: int = 4, drop_out: float = 0.6) -> None:
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    dataset = AMLtoGraph('/dtu/blackhole/0e/154958/data') # write correct path. 
    data = dataset[0]
    
    # Hyperparameters
    epoch = epochs
    batch_size = batchsize
    learning_rate = lr
    hidden_channels = hdn_chnls
    heads = atn_heads
    dropout = drop_out

    run = wandb.init(
        project="G-WEB_Fraud_Detection",
        config={"lr": lr, "batch_size": batch_size, "epochs": epochs, "hidden_channels": hidden_channels, "heads": heads, "dropout": dropout},
        job_type="train",
        
    )
    
    model = GCN(in_channels=data.num_features, hidden_channels=hidden_channels, out_channels=1, heads=heads, dropout=dropout)
    model = model.to(device)
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    split = T.RandomNodeSplit(split='train_rest', num_val=0.2, num_test=0.2)
    data = split(data)

    train_loader = NeighborLoader(
        data,
        num_neighbors=[30] * 2,
        batch_size=batch_size,
        input_nodes=data.train_mask,
    )

    val_loader = NeighborLoader(
        data,
        num_neighbors=[30] * 2,
        batch_size=batch_size,
        input_nodes=data.val_mask,
    )
    
    statistics = {"train_loss": [], "val_accuracy": []}

    for i in range(epoch):
        total_loss = 0
        model.train()
        preds, targets = [], []
        for data in train_loader:
            optimizer.zero_grad()
            data.to(device)
            pred = model(data.x, data.edge_index, data.edge_attr)
            ground_truth = data.y
            loss = criterion(pred, ground_truth.unsqueeze(1))
            loss.backward()
            optimizer.step()
            total_loss += float(loss)
            
            preds.append(pred.detach().cpu())
            targets.append(ground_truth.detach().cpu())
        if i+1%4 == 0 or (i == epoch-1):
            print(f"Epoch: {i+1:03d}, Loss: {total_loss:.4f}")
            model.eval()
            acc = 0
            total = 0
            for test_data in val_loader:
                test_data.to(device)
                pred = model(test_data.x, test_data.edge_index, test_data.edge_attr)
                ground_truth = test_data.y
                correct = (pred == ground_truth.unsqueeze(1)).sum().item()
                total += len(ground_truth)
                acc += correct
            acc = acc/total
            print('accuracy:', acc)
            statistics["val_accuracy"].append
            wandb.log({"val_accuracy": acc})
        statistics["train_loss"].append(total_loss)
        wandb.log({"train_loss": total_loss})


    preds = torch.cat(preds, 0)
    targets = torch.cat(targets, 0)

    final_accuracy = accuracy_score(targets, preds.argmax(dim=1))
    final_precision = precision_score(targets, preds.argmax(dim=1), average="weighted")
    final_recall = recall_score(targets, preds.argmax(dim=1), average="weighted")
    final_f1 = f1_score(targets, preds.argmax(dim=1), average="weighted")
    
    torch.save(model.state_dict(), f"models/model_{epochs}_{lr}_{hdn_chnls}.pth")
    artifact = wandb.Artifact(
        name="G-web-fraud-detection-model",
        type="model",
        description="A model trained to classify transaction fraud",
        metadata={"accuracy": final_accuracy, "precision": final_precision, "recall": final_recall, "f1": final_f1},
    )
    
    print("Training complete")
    torch.save(model.state_dict(), "models/model.pth")
    artifact.add_file("models/model.pth")
    run.log_artifact(artifact)
        
    
    
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    axs[0].plot(statistics["train_loss"])
    axs[0].set_title("Train loss")
    axs[1].plot(statistics["val_accuracy"])
    axs[1].set_title("Train accuracy")
    fig.savefig("reports/figures/training_statistics.png")
    
    run.finish()

if __name__ == "__main__":
    typer.run(train)
    
    
    
    
    