#!/bin/bash
gcloud ai custom-jobs create  --region=europe-west1  --display-name=test-run  --config=configs/vertex_ai_config.yaml 