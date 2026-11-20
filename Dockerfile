FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install CPU-only PyTorch when available on this architecture to avoid multi-GB CUDA wheels.
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
 && (pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu || true) \
 && pip install --no-cache-dir -r requirements.txt

# Pre-cache embedding and cross-encoder weights so container runs work offline and start fast.
RUN python -c "from sentence_transformers import SentenceTransformer, CrossEncoder; \
    SentenceTransformer('all-MiniLM-L6-v2'); \
    CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')"

ENV HF_HUB_OFFLINE=1

COPY src/ ./src/
COPY data/ ./data/

ENTRYPOINT ["python", "-m", "src.rag_pipeline"]
