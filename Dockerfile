# Document and Identity Fraud Detection System - Docker image
FROM python:3.11-slim

# Tesseract for the MRZ / OCR checks
RUN apt-get update && apt-get install -y --no-install-recommends \
        tesseract-ocr libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# CPU-only PyTorch and torchvision first (keeps the image small), then the remaining dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

# Install the docfraud package and copy configs, SQL, scripts and tests
COPY pyproject.toml README.md Makefile run_all.sh ./
COPY src/ ./src/
RUN pip install --no-cache-dir --no-deps -e .
COPY configs/ ./configs/
COPY sql/ ./sql/
COPY scripts/ ./scripts/
COPY tests/ ./tests/

EXPOSE 8000 8501

# Serve the scoring API (data/, models/ and reports/ are mounted)
CMD ["uvicorn", "docfraud.api:app", "--host", "0.0.0.0", "--port", "8000"]
