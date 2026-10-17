# Multi-stage production container build for InterviewIQ FastAPI Backend
FROM python:3.12-slim AS builder

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# Final runtime image
FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /install /usr/local

# Copy backend application code and question bank data
COPY backend/app /app/app
COPY backend/data /app/data

# Default environment configurations
ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PORT=8000 \
    USE_BEDROCK=true \
    BEDROCK_MODEL_ID=google.gemma-3-4b-it \
    AWS_REGION=ap-southeast-2 \
    USE_LOCAL_DECIDER=true \
    DECIDER_DEVICE=cpu \
    DECIDER_MODEL_ID=StrandsAgents/strands-decider-2B-hobson-v21 \
    DECIDER_CONFIDENCE_THRESHOLD=0.60

EXPOSE 8000

# Health check matching the FastAPI /api/health endpoint
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/api/health || exit 1

# Production server execution command
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
