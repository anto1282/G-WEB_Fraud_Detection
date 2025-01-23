# Change from latest to a specific version if your requirements.txt
FROM python:3.11-slim AS base

RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

COPY src src/
COPY requirements.txt requirements.txt
COPY dev_requirements.txt requirements_dev.txt
COPY README.md README.md
COPY pyproject.toml pyproject.toml
COPY data data/
COPY models/model.pth models/*.pth

RUN pip install torch torchvision torchaudio
RUN pip install torch-cluster torch-scatter torch-geometric torch-spline-conv torch-sparse

RUN pip install -r requirements.txt --no-cache-dir --verbose
RUN pip install -r requirements_dev.txt --no-cache-dir --verbose
RUN pip install . --no-deps --no-cache-dir --verbose

ENTRYPOINT ["uvicorn", "src/gweb/api:app", "--host", "0.0.0.0", "--port", "8000"]
