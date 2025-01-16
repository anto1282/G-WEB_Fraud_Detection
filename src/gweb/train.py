import torch
import torch_geometric.transforms as T
from torch_geometric.loader import NeighborLoader
import typer
from model import GCN
from data import AMLtoGraph
import wandb


def weighted_bce_loss(pred, target, pos_weight):
    bce = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    return bce(pred, target)


def train(config=None) -> None:
    torch.manual_seed(42)
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    dataset = AMLtoGraph("/dtu/blackhole/0e/154958/data")  # Adjust path
    data = dataset[0]

    # Extract hyperparameters from the WandB config
    run = wandb.init(config=config, entity="s203557", mode="online")

    # Extract hyperparameters from the WandB config
    lr = wandb.config.lr
    batchsize = wandb.config.batchsize
    hdn_chnls = wandb.config.hdn_chnls
    atn_heads = wandb.config.atn_heads
    drop_out = wandb.config.drop_out
    epochs = wandb.config.epochs
    pos_weight = torch.tensor([wandb.config.pos_weight])

    print(
        f"Running with config: lr={lr}, batchsize={batchsize}, hdn_chnls={hdn_chnls}, atn_heads={atn_heads}, drop_out={drop_out}, epochs={epochs}"
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
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

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

    statistics = {"train_loss": [], "val_accuracy": []}

    for i in range(epochs):
        total_loss = 0
        model.train()
        for data in train_loader:
            optimizer.zero_grad()
            data.to(device)
            pred = model(data.x, data.edge_index, data.edge_attr)
            loss = criterion(pred, data.y)
            loss.backward()
            optimizer.step()
            total_loss += float(loss)

        model.eval()
        val_acc = 0
        total = 0
        with torch.no_grad():
            for val_data in val_loader:
                val_data.to(device)
                pred = model(
                    val_data.x, val_data.edge_index, val_data.edge_attr
                )
                val_acc += (
                    (pred.round() == val_data.y.unsqueeze(1)).sum().item()
                )
                total += len(val_data.y)

        val_acc /= total
        statistics["val_accuracy"].append(val_acc)
        statistics["train_loss"].append(total_loss)

        # Log metrics
        wandb.log({"train_loss": total_loss, "val_accuracy": val_acc})

    run.finish()


if __name__ == "__main__":
    # Just in case you need to debug or run standalone training
    print("Running training script as standalone.")
    typer.run(train)
