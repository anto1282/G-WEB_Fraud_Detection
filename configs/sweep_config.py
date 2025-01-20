import wandb

sweep_configuration = {
    "program": "train.py",
    "method": "bayes",
    "metric": {"goal": "maximize", "name": "f1_score"},
    "parameters": {
        "lr": {"min": 0.0001, "max": 0.001},
        "batchsize": {"values": [256, 512, 1024]},
        "hdn_chnls": {"values": [16, 32, 64]},
        "atn_heads": {"values": [4, 8, 16]},
        "drop_out": {"min": 0.4, "max": 0.8},
        "epochs": {"values": [50, 75, 100]},
        "pos_weight": {"values": [30, 50, 70, 90]},
    },
    "command": ["python", "train.py"],
}

if __name__ == "__main__":
    sweep_id = wandb.sweep(
        sweep=sweep_configuration,
        project="G-Web-Fraud-Detection",
        entity="s203557-danmarks-tekniske-universitet-dtu",
    )
