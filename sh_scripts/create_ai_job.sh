gcloud ai custom-jobs create \
    --region=europe-west1 \
    --display-name=test-run \
    --config=configs/gcp_gpu_config.yaml \
    # these are the arguments that are passed to the container, only needed if you want to change defaults
    # Commands should be compatible with wandb. Ie should be a wandb agent. Perhaps change train.py?
    --command 'python src/gweb/train.py' \
    --args '["--epochs", "10"]'