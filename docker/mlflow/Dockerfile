# ══════════════════════════════════════════════════════════
# Dockerfile — MLflow Tracking Server
# ══════════════════════════════════════════════════════════

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN pip install --no-cache-dir \
    mlflow==2.15.1 \
    psycopg2-binary==2.9.9 \
    boto3==1.34.0

RUN groupadd --gid 1001 mlflow && \
    useradd --uid 1001 --gid mlflow --shell /bin/bash --create-home mlflow

RUN mkdir -p /mlflow/artifacts && \
    chown -R mlflow:mlflow /mlflow

USER mlflow
WORKDIR /mlflow

HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:5000/health')" || exit 1

EXPOSE 5000

CMD ["mlflow", "server", \
     "--host", "0.0.0.0", \
     "--port", "5000", \
     "--backend-store-uri", "postgresql://notam_user:notam_pass@postgres:5432/notam_db", \
     "--default-artifact-root", "/mlflow/artifacts", \
     "--serve-artifacts"]