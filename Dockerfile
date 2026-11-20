# Production Multi-Stage or Slim Python Container for Iris MLOps API
FROM python:3.11-slim

# Prevent Python from writing .pyc files and enable unbuffered logging
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency definition
COPY requirements.txt .

# Install Python packages
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application directories and files
COPY config/ config/
COPY data/ data/
COPY models/ models/
COPY reports/ reports/
COPY src/ src/
COPY params.yaml .
COPY dvc.yaml .

# Expose FastAPI listening port
EXPOSE 8000

# Health check to ensure container readiness
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Start Uvicorn ASGI server serving FastAPI application and dashboard
CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
