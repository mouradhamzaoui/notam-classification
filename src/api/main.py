"""
main.py
Point d'entrée FastAPI — Application principale.
"""

import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse

from src.utils.config import Config
from src.utils.logger import get_logger
from src.api.routers import classify, health, monitoring

logger = get_logger(__name__)
cfg    = Config.get()


# ── Lifespan (startup / shutdown) ─────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Charge le modèle au démarrage, libère les ressources à l'arrêt."""
    logger.info("=" * 55)
    logger.info(f"  🚀 Starting {cfg.project.name} API v{cfg.project.version}")
    logger.info("=" * 55)

    # Pré-charge le modèle (évite la latence à la première requête)
    try:
        from src.api.dependencies import get_inference_pipeline, get_db
        get_inference_pipeline()
        get_db()
        logger.info("[Startup] ✅ Model and DB loaded")
    except Exception as e:
        logger.error(f"[Startup] ⚠️  Preload failed: {e}")

    yield

    logger.info("[Shutdown] API shutting down...")


# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title       = "NOTAM Classification API",
    description = """
## ✈️ Automatic NOTAM Classification System

API de classification automatique des **NOTAMs (Notice to Air Missions)**
vers les catégories opérationnelles ICAO.

### Catégories supportées
| Catégorie | Priorité | Description |
|---|---|---|
| `RUNWAY_CLOSURE` | CRITICAL | Fermeture de piste |
| `NAVIGATION_AID` | HIGH | Panne aide navigation |
| `AIRSPACE_RESTRICTION` | CRITICAL | Restriction espace aérien |
| `LIGHTING` | MEDIUM | Panne éclairage |
| `OBSTACLE` | HIGH | Nouvel obstacle |
| `AERODROME_PROCEDURE` | LOW | Procédure aérodrome |

### Endpoints principaux
- `POST /api/v1/classify` — Classifier un NOTAM
- `POST /api/v1/classify/batch` — Classifier par lot (max 100)
- `GET  /api/v1/health` — État de l'API
- `GET  /api/v1/monitoring/stats` — Statistiques des prédictions
    """,
    version     = cfg.project.version,
    lifespan    = lifespan,
    docs_url    = "/docs",
    redoc_url   = "/redoc",
    openapi_url = "/openapi.json",
)

# ── Middlewares ───────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["*"],   # En prod : restreindre aux domaines autorisés
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)
app.add_middleware(GZipMiddleware, minimum_size=1000)


# ── Middleware de logging des requêtes ────────────────────────────────────────
@app.middleware("http")
async def log_requests(request: Request, call_next):
    t0       = time.time()
    response = await call_next(request)
    duration = (time.time() - t0) * 1000
    logger.debug(
        f"[API] {request.method} {request.url.path} "
        f"→ {response.status_code} ({duration:.1f}ms)"
    )
    return response


# ── Global exception handler ─────────────────────────────────────────────────
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"[API] Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"error": "Internal server error", "detail": str(exc), "code": 500},
    )


# ── Routers ───────────────────────────────────────────────────────────────────
PREFIX = cfg.api.prefix or "/api/v1"
app.include_router(health.router,     prefix=PREFIX)
app.include_router(classify.router,   prefix=PREFIX)
app.include_router(monitoring.router, prefix=PREFIX)


# ── Root ──────────────────────────────────────────────────────────────────────
@app.get("/", include_in_schema=False)
async def root():
    return {
        "name":    cfg.project.name,
        "version": cfg.project.version,
        "docs":    "/docs",
        "health":  f"{PREFIX}/health",
    }