# ══════════════════════════════════════════════════════════
# Dockerfile — NOTAM Classification API (FastAPI)
# Multi-stage build : builder + runtime léger
# ══════════════════════════════════════════════════════════

# ── Stage 1 : Builder ─────────────────────────────────────
FROM python:3.11-slim AS builder

# Variables d'environnement build
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_VERSION=0.4.18

# Installe uv
RUN pip install --no-cache-dir uv==${UV_VERSION}

WORKDIR /app

# Copie les fichiers de dépendances en premier (cache Docker)
COPY pyproject.toml uv.lock* ./

# Installe les dépendances dans un venv isolé
RUN uv venv /opt/venv && \
    . /opt/venv/bin/activate && \
    uv sync --frozen --no-dev

# ── Stage 2 : Runtime ─────────────────────────────────────
FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH" \
    PYTHONPATH="/app"

# Utilisateur non-root pour la sécurité
RUN groupadd --gid 1001 notam && \
    useradd --uid 1001 --gid notam --shell /bin/bash --create-home notam

# Copie le venv depuis le builder
COPY --from=builder /opt/venv /opt/venv

WORKDIR /app

# Copie le code source
COPY src/         ./src/
COPY config/      ./config/
COPY scripts/     ./scripts/

# Répertoires de données (montés en volume en prod)
RUN mkdir -p data/processed/models data/raw logs && \
    chown -R notam:notam /app

USER notam

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/v1/health')" || exit 1

EXPOSE 8000

CMD ["uvicorn", "src.api.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "2", \
     "--log-level", "info"]