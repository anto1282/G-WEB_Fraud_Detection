# Use a Python slim image
FROM python:3.11-slim AS base

RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app


COPY requirements.txt requirements.txt
RUN pip install --upgrade pip && \
    pip install torch torchvision torchaudio && \
    pip install torch-cluster torch-scatter torch-geometric torch-spline-conv torch-sparse && \
    pip install -r requirements.txt --no-cache-dir && \
    pip install prometheus-client networkx streamlit seaborn && \
    pip install . --no-deps --no-cache-dir


# Copy source code and other necessary files
COPY src src/
COPY README.md README.md
COPY pyproject.toml pyproject.toml
COPY data data/
COPY models/model.pth models/

# Expose ports for Streamlit and Uvicorn
EXPOSE 8501 8000

WORKDIR /app/src/gweb

# Start both Streamlit and Uvicorn using a process manager
ENTRYPOINT ["sh", "-c", "uvicorn api:app --host 0.0.0.0 --port 8000 & streamlit run app_interface.py"]
