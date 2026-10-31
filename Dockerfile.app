# ============================================================
# RetainIQ - Streamlit Dashboard Dockerfile
# Phase 8 Production UI Container
# ============================================================

FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MPLCONFIGDIR=/tmp

# Install system dependencies (libgomp1 for model runtime fallback, curl for healthchecks)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application assets, models, and evaluation reports
COPY app/ ./app/
COPY src/ ./src/
COPY models/ ./models/
COPY reports/ ./reports/
COPY config/ ./config/
COPY data/ ./data/
COPY preprocessing/ ./preprocessing/

RUN mkdir -p /tmp && chmod 777 /tmp

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

CMD ["streamlit", "run", "app/streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]

