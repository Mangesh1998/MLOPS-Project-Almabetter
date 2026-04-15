FROM python:3.11-slim-buster

WORKDIR /app

# Ensure dependencies are isolated and installed without cache to reduce image size
COPY requirements.txt .
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir mlflow requests

# Pre-load standard SentenceTransformers dependencies so scaling instances don't hang downloading models!
# This builds it directly into an image layer.
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('flax-sentence-embeddings/all_datasets_v4_MiniLM-L6')"

# Push root project map into the isolated container instance
# (.dockerignore ensures .git and venv do not cause megabyte bloat)
COPY . .

# Expose Web Interface & API default ports
EXPOSE 8000 8501
