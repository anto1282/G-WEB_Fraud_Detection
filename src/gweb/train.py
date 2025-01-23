import torch
import torch.optim.sgd
import torch_geometric.transforms as T
from torch_geometric.loader import NeighborLoader
import typer
from sklearn.metrics import f1_score

try:
    from .model import GCN  # Relative import
except ImportError:
    from model import GCN  # Absolute import
try:
    from .data import AMLtoGraph
except ImportError:
    from data import AMLtoGraph
import wandb
import numpy as np
import os
from gweb import _PATH_DATA


def train(test_mode=False, config=None) -> None:

    # Use smaller dataset for testing
    if test_mode:
        path = os.path.join(_PATH_DATA, "test")
    else:
        path = _PATH_DATA

    torch.manual_seed(42)
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    # Necessary to avoid error when running in a container
    try:
        dataset = AMLtoGraph(path)
    except FileNotFoundError:
        if test_mode:
            path = os.path.join("/app/data", "test")
        else:
            path = "/app/data"

        dataset = AMLtoGraph(path)

    data = dataset[0]

    # Extract hyperparameters from the WandB config
    run = wandb.init(config=config, entity="s203557-danmarks-tekniske-universitet-dtu")

    # Extract hyperparameters from the WandB config
    lr = wandb.config.lr
    batchsize = wandb.config.batchsize
    hdn_chnls = wandb.config.hdn_chnls
    atn_heads = wandb.config.atn_heads
    drop_out = wandb.config.drop_out
    epochs = wandb.config.epochs
    pos_weight = torch.tensor([wandb.config.pos_weight]).to(device)
    patience = 16


    print(
        f"Running with config: lr={lr}, batchsize={batchsize}, hdn_chnls={hdn_chnls}, atn_heads={atn_heads}, drop_out={drop_out}, epochs={epochs}, pos_weight={pos_weight}"
    )

    # Model, optimizer, and loss setup
    model = GCN(
        in_channels=data.num_features,
        hidden_channels=hdn_chnls,
        out_channels=1,
        heads=atn_heads,
        dropout=drop_out,
    ).to(device)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    # Data split and loaders
    split = T.RandomNodeSplit(split="train_rest", num_val=0.2, num_test=0.2)
    data = split(data)

    train_loader = NeighborLoader(
        data,
        num_neighbors=[30] * 2,
        batch_size=batchsize,
        input_nodes=data.train_mask,
    )

    val_loader = NeighborLoader(
        data,
        num_neighbors=[30] * 2,
        batch_size=batchsize,
        input_nodes=data.val_mask,
    )

    statistics = {"train_loss": [], "val_accuracy": [], "f1_score": []}

    best_val_score = float("-inf")
    patience_counter = 0  # Tracks epochs without improvement

    for i in range(epochs):
        total_loss = 0
        model.train()
        for data in train_loader:
            optimizer.zero_grad()
            data.to(device)
            pred = model(data.x, data.edge_index, data.edge_attr)
            loss = criterion(pred.to(device), data.y.unsqueeze(1).to(device))
            loss.backward()
            optimizer.step()
            total_loss += float(loss)

        model.eval()
        val_acc = 0
        total = 0
        all_preds = []
        all_labels = []
        if i % 2 == 0:

            with torch.no_grad():
                for val_data in val_loader:
                    val_data.to(device)
                    pred = model(
                        val_data.x, val_data.edge_index, val_data.edge_attr
                    )
                    pred_rounded = pred.round()

                    val_acc += (
                        (pred_rounded == val_data.y.to(device).unsqueeze(1))
                        .sum()
                        .item()
                    )
                    total += len(val_data.y)

                    # Collect predictions and labels for F1 score
                    all_preds.append(pred_rounded.cpu().numpy())
                    all_labels.append(val_data.y.cpu().numpy())

            # Calculate accuracy after processing all validation data
            val_acc /= total

            # Flatten the predictions and labels to compute F1 score
            all_preds = np.concatenate(all_preds).flatten()
            all_labels = np.concatenate(all_labels).flatten()

            f1 = f1_score(all_labels, all_preds, average="binary")

            statistics["val_accuracy"].append(val_acc)
            statistics["f1_score"].append(f1)

            wandb.log({"val_accuracy": val_acc, "f1_score": f1})

        ## Early stopping logic
        if f1 > best_val_score:
            best_val_score = f1
            patience_counter = 0
            print(f"Epoch {i}: New best model with F1 score: {f1:.4f}")
        else:
            patience_counter += 1

        if patience_counter >= patience:
            break

        statistics["train_loss"].append(total_loss)
        wandb.log({"train_loss": total_loss})

    data = next(iter(train_loader))
    dummy_input = (
        data.x.to(device),
        data.edge_index.to(device),
        data.edge_attr.to(device) if hasattr(data, "edge_attr") else None,
    )

    # Generate model name
    model_name = f"model_lr-{wandb.config.lr:.2e}_bs-{wandb.config.batchsize}_dropout-{wandb.config.drop_out:.2f}_epochs-{wandb.config.epochs}.onnx"
    model.eval()
    data = next(iter(train_loader))
    dummy_input = (
        data.x.to(device),
        data.edge_index.to(device),
        data.edge_attr.to(device),
    )

    # Define output path
    output_dir = "../../models"
    os.makedirs(output_dir, exist_ok=True)  # Ensure the directory exists
    output_path = os.path.join(output_dir, model_name)

    torch.onnx.export(
        model,
        args=dummy_input,  # Ensure arguments match the model's forward method
        f=output_path,  # Directly specify the file path
        opset_version=18,
        export_options=torch.onnx.ExportOptions(dynamic_shapes=True),
    )
    print(f"Model saved as {model_name}")

    artifact = wandb.Artifact(
        name="onnx_model",
        type="model",
        description="Exported ONNX model",
        metadata={
            "lr": wandb.config.lr,
            "batchsize": wandb.config.batchsize,
            "dropout": wandb.config.drop_out,
            "epochs": wandb.config.epochs,
        },
    )

    artifact.add_file(output_path)

    wandb.log_artifact(artifact)
    print(f"ONNX model {model_name} logged to W&B as an artifact")

    run.finish()

    # Define output path
    output_dir = "../../models"
    os.makedirs(output_dir, exist_ok=True)  # Ensure the directory exists
    output_path = os.path.join(output_dir, model_name)

    # Export to ONNX
    torch.onnx.export(
        model,
        args=dummy_input,  # Ensure arguments match the model's forward method
        f=output_path,  # Directly specify the file path
        opset_version=18,
        export_options=torch.onnx.ExportOptions(dynamic_shapes=True),
    )
    print(f"Model saved as {model_name}")



if __name__ == "__main__":
    print("Running training script")
    typer.run(train)
