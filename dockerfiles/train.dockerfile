# Base image
FROM python:3.11-slim AS base

RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt ./requirements.txt
COPY pyproject.toml ./pyproject.toml
COPY src/ ./src/
COPY data/test/ ./data/test/
COPY configs/default_experiment_config.yaml ./configs/default_experiment_config.yaml

RUN --mount=type=cache,target=/root/.cache/pip pip install torch torchvision torchaudio
RUN --mount=type=cache,target=/root/.cache/pip pip install torch-cluster torch-scatter torch-geometric torch-spline-conv torch-sparse
RUN --mount=type=cache,target=/root/.cache/pip pip install -r requirements.txt --no-cache-dir --verbose
RUN pip install . --no-deps --no-cache-dir --verbose

ENTRYPOINT ["python", "-u", "src/gweb/train.py", "--config", "configs/default_experiment_config.yaml", "--test-mode", "True"]
