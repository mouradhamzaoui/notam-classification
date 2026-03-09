# ✈️ NOTAM Classification System
### Automatic Aviation Notice Classification · MLOps End-to-End

[![CI Pipeline](https://github.com/USERNAME/notam-classification/actions/workflows/ci.yml/badge.svg)](https://github.com/USERNAME/notam-classification/actions/workflows/ci.yml)
[![CD Pipeline](https://github.com/USERNAME/notam-classification/actions/workflows/cd.yml/badge.svg)](https://github.com/USERNAME/notam-classification/actions/workflows/cd.yml)
[![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?logo=fastapi)](https://fastapi.tiangolo.com)
[![React](https://img.shields.io/badge/React-18-61DAFB?logo=react)](https://reactjs.org)
[![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?logo=docker)](https://docker.com)
[![MLflow](https://img.shields.io/badge/MLflow-tracking-0194E2?logo=mlflow)](https://mlflow.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 📸 Screenshots

| Frontend React | FastAPI Swagger | MLflow Tracking | Monitoring Tracking |
|---|---|---|---|
| ![Frontend](assets/frontend.png) | ![API](assets/swagger.png) | ![MLflow](assets/mlflow.png) | ![Montitoring](assets/monitoring.png)|

---

[![CI Pipeline](https://github.com/mouradhamzaoui/notam-classification/actions/workflows/ci.yml/badge.svg)](https://github.com/mouradhamzaoui/notam-classification/actions/workflows/ci.yml)       [![CD Pipeline](https://github.com/mouradhamzaoui/notam-classification/actions/workflows/cd.yml/badge.svg)](https://github.com/mouradhamzaoui/notam-classification/actions/workflows/cd.yml)
## 📋 Table des matières

1. [Vue d'ensemble](#-vue-densemble)
2. [Architecture](#-architecture)
3. [Stack technique](#-stack-technique)
4. [Structure du projet](#-structure-du-projet)
5. [Installation rapide](#-installation-rapide)
6. [Configuration](#-configuration)
7. [Phase 1 — POC](#-phase-1--poc)
8. [Phase 2 — MLOps](#-phase-2--mlops)
9. [API Reference](#-api-reference)
10. [Frontend React](#-frontend-react)
11. [Tests](#-tests)
12. [CI/CD Pipeline](#-cicd-pipeline)
13. [Monitoring](#-monitoring)
14. [Docker](#-docker)
15. [Performance du modèle](#-performance-du-modèle)
16. [Contribuer](#-contribuer)

---

## 🎯 Vue d'ensemble

Ce projet implémente un système de **classification automatique de NOTAMs** (Notice To AirMen) selon les 6 catégories ICAO standard. Il couvre l'intégralité du cycle de vie d'un projet Machine Learning — de l'exploration des données jusqu'au déploiement en production avec monitoring continu.

### Catégories supportées

| Catégorie | Priorité | Description | Exemple |
|---|---|---|---|
| `RUNWAY_CLOSURE` | 🔴 CRITICAL | Fermeture de piste | `RWY 28L CLSD DUE TO CONSTRUCTION` |
| `AIRSPACE_RESTRICTION` | 🔴 CRITICAL | Restriction d'espace aérien | `RESTRICTED AREA R-2508 ACTIVE SFC-18000FT` |
| `NAVIGATION_AID` | 🟠 HIGH | Aide à la navigation HS | `ILS CAT II RWY 10R NOT AVAILABLE` |
| `OBSTACLE` | 🟠 HIGH | Obstacle à signaler | `NEW CRANE 520FT AGL WITHIN 3NM OF ARP` |
| `LIGHTING` | 🟣 MEDIUM | Système d'éclairage | `PAPI RWY 36 OTS` |
| `AERODROME_PROCEDURE` | 🟢 LOW | Procédure aéroportuaire | `FUEL NOT AVBL 2H DAILY DUE MAINTENANCE` |

### Résultats clés

- **F1-macro : 94%** sur le dataset de test (2400 NOTAMs)
- **Latence API : < 10ms** par prédiction (LinearSVC + TF-IDF)
- **49 tests** unitaires et d'intégration
- **Pipeline CI/CD** complète avec GitHub Actions

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     PRODUCTION STACK                        │
│                                                             │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐  │
│  │    React     │    │   FastAPI    │    │  PostgreSQL  │  │
│  │  Frontend    │───▶│  REST API    │───▶│   Database   │  │
│  │  :3000       │    │  :8000       │    │  :5432       │  │
│  └──────────────┘    └──────┬───────┘    └──────────────┘  │
│                             │                               │
│                    ┌────────▼────────┐                      │
│                    │  ML Pipeline    │                      │
│                    │  LinearSVC      │                      │
│                    │  TF-IDF + Meta  │                      │
│                    └────────┬────────┘                      │
│                             │                               │
│  ┌──────────────┐    ┌──────▼───────┐    ┌──────────────┐  │
│  │   MLflow     │◀───│  Experiment  │───▶│  Evidently   │  │
│  │  Tracking    │    │  Tracker     │    │  Monitoring  │  │
│  │  :5000       │    └──────────────┘    │  :8502       │  │
│  └──────────────┘                        └──────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

### Flux de données

```
CSV Raw Data
     │
     ▼
EDA + Feature Engineering
     │  TF-IDF (max 10k features)
     │  Meta Features (char_count, upper_ratio, ...)
     ▼
Training Pipeline
     │  GridSearchCV (LinearSVC)
     │  MLflow logging
     ▼
best_model.pkl
     │
     ▼
Inference Pipeline ──▶ FastAPI ──▶ PostgreSQL logs
                                        │
                                        ▼
                               Evidently AI Monitoring
                               (Data drift detection)
```

---

## 🛠️ Stack technique

### Machine Learning
| Composant | Technologie | Version |
|---|---|---|
| Environnement | uv | 0.5+ |
| ML Framework | scikit-learn | 1.5+ |
| Modèle | LinearSVC (Calibrated) | — |
| Features | TF-IDF + Meta Features | — |
| Experiment tracking | MLflow | 2.x |
| Monitoring | Evidently AI | 0.4+ |

### Backend
| Composant | Technologie | Version |
|---|---|---|
| API Framework | FastAPI | 0.115+ |
| Validation | Pydantic v2 | 2.x |
| ORM | SQLAlchemy | 2.x |
| Base de données | PostgreSQL 16 / SQLite | — |
| Logging | Rich | 13.x |

### Frontend
| Composant | Technologie | Version |
|---|---|---|
| Framework | React | 18 |
| Routing | React Router v6 | 6.x |
| Requêtes HTTP | Axios | 1.x |
| Graphiques | Recharts | 2.x |
| Icônes | Lucide React | 0.x |
| Notifications | React Hot Toast | 2.x |
| Fonts | Share Tech Mono, Barlow Condensed | — |

### DevOps
| Composant | Technologie |
|---|---|
| Containerisation | Docker + Docker Compose |
| CI/CD | GitHub Actions |
| Linting | Ruff |
| Tests | pytest + pytest-asyncio + pytest-cov |
| Container Registry | GitHub Container Registry (ghcr.io) |

---

## 📁 Structure du projet

```
notam-classification/
│
├── .github/
│   └── workflows/
│       ├── ci.yml              ← CI : lint → tests → coverage → docker build
│       └── cd.yml              ← CD : build → push ghcr.io sur tag vX.Y.Z
│
├── config/
│   ├── config.yaml             ← Configuration principale (paths, DB, API)
│   └── model_config.yaml       ← Hyperparamètres modèles + catégories ICAO
│
├── data/
│   ├── raw/
│   │   └── notams.csv          ← Dataset brut (2400 NOTAMs synthétiques)
│   ├── processed/
│   │   ├── notams_clean.csv    ← Dataset nettoyé après EDA
│   │   ├── feature_pipeline.pkl← Pipeline TF-IDF + Meta Features sérialisé
│   │   ├── train_test_splits.pkl
│   │   ├── notam_dev.db        ← SQLite (fallback PostgreSQL)
│   │   └── models/
│   │       ├── best_model.pkl  ← Meilleur modèle entraîné
│   │       └── model_metadata.json
│   ├── download_dataset.py     ← Génère le dataset synthétique ICAO
│   └── build_splits.py         ← Régénère les splits train/test
│
├── notebooks/
│   ├── 01_EDA.ipynb            ← Exploratory Data Analysis (9 cells, 6 figures)
│   ├── 02_Feature_Engineering.ipynb ← 4 stratégies de features
│   └── 03_Modeling.ipynb       ← Benchmark 3 modèles + GridSearchCV
│
├── src/
│   ├── api/
│   │   ├── main.py             ← Application FastAPI (CORS, lifespan, middlewares)
│   │   ├── schemas.py          ← Modèles Pydantic (request/response)
│   │   ├── dependencies.py     ← Injection de dépendances (InferencePipeline)
│   │   └── routers/
│   │       ├── classify.py     ← POST /classify, /classify/batch, /classify/feedback
│   │       ├── health.py       ← GET /health, /health/model
│   │       └── monitoring.py   ← GET /monitoring/predictions, /monitoring/stats
│   │
│   ├── data/
│   │   └── data_loader.py      ← NOTAMDataLoader (chargement, nettoyage, split)
│   │
│   ├── features/
│   │   └── feature_engineering.py ← TFIDFStrategy, MetaFeatureExtractor, NOTAMFeaturePipeline
│   │
│   ├── models/
│   │   ├── train.py            ← ModelTrainer (LR, RF, LinearSVC + GridSearchCV)
│   │   └── evaluate.py         ← ModelEvaluator (métriques, confusion matrix, figures)
│   │
│   ├── monitoring/
│   │   ├── drift_detector.py   ← NOTAMDriftDetector (Evidently AI)
│   │   └── monitoring_dashboard.py ← Dashboard Streamlit monitoring
│   │
│   ├── pipeline/
│   │   ├── training_pipeline.py   ← TrainingPipeline (orchestration complète)
│   │   └── inference_pipeline.py  ← InferencePipeline (singleton, prédictions)
│   │
│   ├── tracking/
│   │   ├── mlflow_tracker.py   ← MLflowTracker (logging expériences)
│   │   └── database.py         ← DatabaseManager (SQLAlchemy ORM, prediction logs)
│   │
│   └── utils/
│       ├── config.py           ← Config (singleton, chargement YAML)
│       └── logger.py           ← get_logger() (Rich handler)
│
├── app/
│   └── poc/
│       ├── streamlit_app.py    ← POC Streamlit (3 pages)
│       └── predictor.py        ← Classe Predictor pour le POC
│
├── frontend/                   ← React App (aviation cockpit UI)
│   ├── public/
│   │   └── index.html
│   ├── src/
│   │   ├── components/
│   │   │   ├── Navbar.jsx
│   │   │   ├── CategoryBadge.jsx
│   │   │   └── ConfidenceBar.jsx
│   │   ├── pages/
│   │   │   ├── ClassifyPage.jsx
│   │   │   ├── BatchPage.jsx
│   │   │   ├── HistoryPage.jsx
│   │   │   ├── MonitoringPage.jsx
│   │   │   └── StatusPage.jsx
│   │   ├── services/
│   │   │   └── api.js
│   │   ├── App.js
│   │   └── index.css
│   ├── Dockerfile
│   └── nginx.conf
│
├── scripts/
│   ├── train.py                ← CLI : entraîner le modèle
│   ├── run_experiment.py       ← CLI : entraîner + logger dans MLflow
│   ├── start_api.py            ← CLI : démarrer FastAPI
│   ├── run_monitoring.py       ← CLI : analyser la dérive des données
│   └── docker_utils.py         ← Utilitaires Docker
│
├── tests/
│   ├── conftest.py             ← Fixtures partagées (sample_df, pipeline, ...)
│   ├── unit/
│   │   ├── test_data_loader.py        ← 7 tests
│   │   ├── test_feature_engineering.py ← 12 tests
│   │   └── test_inference_pipeline.py  ← 8 tests
│   └── integration/
│       └── test_api.py                ← 19 tests (endpoints FastAPI)
│
├── docker/
│   ├── api/Dockerfile          ← Multi-stage build (builder + runtime)
│   ├── mlflow/Dockerfile       ← MLflow server
│   └── postgres/init.sql       ← Schéma initial PostgreSQL
│
├── reports/
│   └── drift/                  ← Rapports HTML Evidently (générés automatiquement)
│
├── docker-compose.yml          ← Production (postgres + mlflow + api + frontend)
├── docker-compose.dev.yml      ← Développement (avec hot-reload)
├── pyproject.toml              ← Dépendances uv + metadata projet
├── ruff.toml                   ← Configuration linter Ruff
├── pytest.ini                  ← Configuration pytest
├── .env.example                ← Template variables d'environnement
├── .gitignore
└── README.md
```

---

## 🚀 Installation rapide

### Prérequis

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) `pip install uv`
- Node.js 20+ (pour le frontend React)
- Docker Desktop (pour le déploiement complet)
- Git

### 1. Cloner le dépôt

```bash
git clone https://github.com/USERNAME/notam-classification.git
cd notam-classification
```

### 2. Installer les dépendances Python

```bash
uv sync
```

### 3. Générer le dataset et entraîner le modèle

```bash
uv run python data/download_dataset.py
uv run python data/build_splits.py
uv run python scripts/train.py --no-tune
```

### 4. Lancer l'API

```bash
uv run python scripts/start_api.py
# → http://localhost:8000
# → Swagger UI : http://localhost:8000/docs
```

### 5. Lancer le frontend React

```bash
cd frontend
npm install
npm start
# → http://localhost:3000
```

### 6. (Optionnel) Tout lancer avec Docker

```bash
docker compose up -d
# API      → http://localhost:8000
# MLflow   → http://localhost:5000
# Frontend → http://localhost:3000
```

---

## ⚙️ Configuration

### Variables d'environnement (`.env`)

Copie `.env.example` en `.env` et adapte :

```bash
cp .env.example .env
```

```env
# Base de données
DATABASE_URL=postgresql://notam_user:notam_pass@localhost:5432/notam_db
# Fallback SQLite si PostgreSQL non disponible :
# DATABASE_URL=sqlite:///data/processed/notam_dev.db

# MLflow
MLFLOW_TRACKING_URI=http://localhost:5000
MLFLOW_EXPERIMENT_NAME=notam-classification

# API
API_HOST=0.0.0.0
API_PORT=8000
API_WORKERS=1

# Frontend
REACT_APP_API_URL=http://localhost:8000/api/v1
```

### `config/config.yaml`

```yaml
data:
  raw_path:        data/raw/notams.csv
  clean_path:      data/processed/notams_clean.csv
  splits_path:     data/processed/train_test_splits.pkl
  pipeline_path:   data/processed/feature_pipeline.pkl

model:
  save_dir:        data/processed/models/
  best_model_name: best_model.pkl

api:
  host: 0.0.0.0
  port: 8000

thresholds:
  drift_psi_alert: 0.20
  min_confidence:  0.50
```

---

## 📊 Phase 1 — POC

### Étape 1 : Génération du dataset

```bash
uv run python data/download_dataset.py
```

Génère `data/raw/notams.csv` — 2400 NOTAMs synthétiques conformes ICAO, parfaitement équilibrés (400 par catégorie).

### Étape 2 : Exploratory Data Analysis

```bash
uv run jupyter lab notebooks/01_EDA.ipynb
```

Rapport généré dans `data/processed/EDA_REPORT.txt`. Figures sauvegardées dans `data/processed/fig_01` à `fig_06` :
- Distribution des catégories
- Longueur des textes par catégorie
- Nuage de mots global et par catégorie
- Visualisation SVD (clusters par classe)

### Étape 3 : Feature Engineering

```bash
uv run jupyter lab notebooks/02_Feature_Engineering.ipynb
```

4 stratégies comparées :
- **Baseline TF-IDF** : TfidfVectorizer (max 10k features, n-grams 1-3)
- **TF-IDF + Stemming** : Porter Stemmer + TF-IDF
- **Meta Features** : char_count, word_count, upper_ratio, digit_ratio, slash_count, has_coordinates
- **Hybrid Pipeline** : TF-IDF + Meta Features (meilleure stratégie → **retenue**)

### Étape 4 : Modélisation

```bash
uv run jupyter lab notebooks/03_Modeling.ipynb
```

| Modèle | F1-macro | Accuracy | Temps |
|---|---|---|---|
| LogisticRegression | 89.2% | 89.5% | 2.1s |
| RandomForest | 87.8% | 88.1% | 45.3s |
| **LinearSVC (Calibrated)** | **94.1%** | **94.3%** | **0.8s** |

**Vainqueur : LinearSVC** avec `C=1.0`, pipeline Hybrid (TF-IDF + Meta Features).

### Étape 5 : Streamlit POC

```bash
uv run streamlit run app/poc/streamlit_app.py
# → http://localhost:8501
```

3 pages disponibles :
- **Classification** : classification en temps réel avec probabilités
- **Batch Processing** : traitement CSV par lots
- **Model Info** : métriques et informations du modèle

---

## 🔧 Phase 2 — MLOps

### Entraînement avec MLflow tracking

```bash
# Entraînement rapide (sans GridSearchCV)
uv run python scripts/run_experiment.py --no-tune

# Entraînement complet (avec GridSearchCV, plus long)
uv run python scripts/run_experiment.py

# Lancer l'interface MLflow
uv run mlflow ui --port 5000
# → http://localhost:5000
```

Chaque run MLflow enregistre :
- Hyperparamètres (C, max_features, ngram_range, ...)
- Métriques (f1_macro, accuracy, precision, recall par classe)
- Artifacts (model.pkl, confusion_matrix.png, feature_importance.json)
- Tags (dataset_size, model_type, best_model)

---

## 📡 API Reference

### Base URL

```
http://localhost:8000/api/v1
```

### Documentation interactive

```
http://localhost:8000/docs       ← Swagger UI
http://localhost:8000/redoc      ← ReDoc
```

### Endpoints

#### `POST /classify` — Classifier un NOTAM

```bash
curl -X POST http://localhost:8000/api/v1/classify \
  -H "Content-Type: application/json" \
  -d '{"text": "RWY 28L CLSD DUE TO CONSTRUCTION WIP"}'
```

**Réponse :**
```json
{
  "category": "RUNWAY_CLOSURE",
  "priority": "CRITICAL",
  "confidence": 0.9847,
  "probabilities": {
    "RUNWAY_CLOSURE": 0.9847,
    "NAVIGATION_AID": 0.0042,
    "AIRSPACE_RESTRICTION": 0.0038,
    "LIGHTING": 0.0031,
    "OBSTACLE": 0.0027,
    "AERODROME_PROCEDURE": 0.0015
  },
  "model_version": "latest",
  "latency_ms": 4.2,
  "prediction_id": "pred_abc123"
}
```

#### `POST /classify/batch` — Classifier en lot

```bash
curl -X POST http://localhost:8000/api/v1/classify/batch \
  -H "Content-Type: application/json" \
  -d '{"texts": ["RWY 10L CLSD", "ILS RWY 28R NOT AVAILABLE", "PAPI OTS"]}'
```

**Réponse :**
```json
{
  "results": [
    {"category": "RUNWAY_CLOSURE", "confidence": 0.97, "priority": "CRITICAL"},
    {"category": "NAVIGATION_AID", "confidence": 0.94, "priority": "HIGH"},
    {"category": "LIGHTING", "confidence": 0.91, "priority": "MEDIUM"}
  ],
  "total": 3,
  "duration_ms": 9.8
}
```

#### `POST /classify/feedback` — Soumettre un feedback

```bash
curl -X POST http://localhost:8000/api/v1/classify/feedback \
  -H "Content-Type: application/json" \
  -d '{"prediction_id": "pred_abc123", "true_label": "RUNWAY_CLOSURE"}'
```

#### `GET /health` — Statut de l'API

```bash
curl http://localhost:8000/api/v1/health
```

**Réponse :**
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "model_loaded": true,
  "db_connected": true,
  "uptime_s": 3600
}
```

#### `GET /health/model` — Informations du modèle

```bash
curl http://localhost:8000/api/v1/health/model
```

#### `GET /monitoring/predictions?limit=50` — Historique des prédictions

```bash
curl http://localhost:8000/api/v1/monitoring/predictions?limit=50
```

#### `GET /monitoring/stats` — Statistiques de production

```bash
curl http://localhost:8000/api/v1/monitoring/stats
```

**Réponse :**
```json
{
  "total_predictions": 1250,
  "avg_confidence": 0.923,
  "avg_latency_ms": 5.1,
  "min_confidence": 0.512,
  "max_confidence": 0.999,
  "category_distribution": {
    "RUNWAY_CLOSURE": 210,
    "NAVIGATION_AID": 198,
    ...
  },
  "feedback_accuracy": 0.961
}
```

### Codes d'erreur

| Code | Description |
|---|---|
| `400` | Texte invalide (trop court, trop long, vide) |
| `422` | Erreur de validation Pydantic |
| `500` | Erreur interne (modèle non chargé, DB inaccessible) |
| `503` | Service temporairement indisponible |

---

## 🖥️ Frontend React

### Lancement

```bash
cd frontend
npm start
# → http://localhost:3000
```

### Pages disponibles

| Route | Page | Description |
|---|---|---|
| `/` | **Classify** | Classification temps réel avec distribution des probabilités |
| `/batch` | **Batch** | Traitement multi-NOTAMs avec export CSV |
| `/history` | **History** | Historique des prédictions en base de données |
| `/monitoring` | **Monitor** | Statistiques et graphiques de production |
| `/status` | **Status** | Health check API + informations modèle |

### Design

Esthétique **cockpit aviation** :
- Dark AMOLED (`#080b0f`) avec accents amber/gold
- Typographie : `Share Tech Mono` (données) + `Barlow Condensed` (labels)
- Animations CSS natives (fadeIn, pulse)
- Responsive design

---

## 🧪 Tests

### Lancer les tests

```bash
# Tests unitaires uniquement (rapide)
uv run pytest tests/unit/ -v --no-cov

# Tests d'intégration uniquement
uv run pytest tests/integration/ -v --no-cov

# Suite complète avec coverage
uv run pytest -v --cov=src --cov-report=term-missing

# Un test spécifique
uv run pytest tests/unit/test_data_loader.py::TestNOTAMDataLoader::test_split_stratified -v
```

### Couverture des tests

| Module | Coverage | Tests |
|---|---|---|
| `src/data/data_loader.py` | 98% | 7 tests |
| `src/features/feature_engineering.py` | 73% | 12 tests |
| `src/pipeline/inference_pipeline.py` | 76% | 8 tests |
| `src/api/*` | 58-100% | 19 tests d'intégration |
| `src/utils/config.py` | 85% | — |
| `src/utils/logger.py` | 97% | — |

### Structure des tests

```
tests/
├── conftest.py                    ← Fixtures : sample_df, pipeline, model
├── unit/
│   ├── test_data_loader.py        ← TestNOTAMDataLoader (7 tests)
│   │   ├── test_load_success
│   │   ├── test_load_file_not_found
│   │   ├── test_load_missing_columns
│   │   ├── test_split_stratified
│   │   ├── test_meta_features_added
│   │   ├── test_duplicates_removed
│   │   └── test_random_state_reproducibility
│   ├── test_feature_engineering.py ← TestTFIDFStrategy + TestMetaFeatureExtractor (12 tests)
│   └── test_inference_pipeline.py  ← TestPredictionResult + TestInferencePipeline (8 tests)
└── integration/
    └── test_api.py                 ← TestHealthEndpoints, TestClassifyEndpoint,
                                       TestBatchClassifyEndpoint, TestMonitoringEndpoints
                                       (19 tests)
```

### Marqueurs pytest

```bash
uv run pytest -m unit         # Seulement les tests unitaires
uv run pytest -m integration  # Seulement les tests d'intégration
uv run pytest -m "not slow"   # Exclure les tests lents
```

---

## 🔄 CI/CD Pipeline

### CI — Déclenché sur push/PR vers `main` et `develop`

```
Push / Pull Request
        │
        ▼
┌───────────────┐
│  🔍 Lint      │  Ruff check + format (continue-on-error)
└───────┬───────┘
        │
        ▼
┌───────────────┐
│  🧪 Unit      │  pytest tests/unit/ + coverage XML
└───────┬───────┘
        │
        ▼
┌───────────────┐
│  🔗 Integ.    │  pytest tests/integration/ + coverage XML
└───────┬───────┘
        │
        ▼
┌───────────────┐
│  📊 Coverage  │  Suite complète · seuil 50%+ · rapport HTML
└───────┬───────┘
        │
        ▼
┌───────────────┐
│  🐳 Docker    │  Build API image · vérification docker inspect
└───────────────┘
```

### CD — Déclenché sur push d'un tag `vX.Y.Z`

```
git push origin v1.0.0
        │
        ▼
┌───────────────────────────┐
│  🐳 Build & Push          │
│  → ghcr.io/user/notam:    │
│    1.0.0 / 1.0 / sha-xxx  │
└───────────────┬───────────┘
                │
                ▼
┌───────────────────────────┐
│  📋 Deployment Summary    │
│  GitHub Actions Summary   │
└───────────────────────────┘
```

### Créer une release

```bash
git tag -a v1.1.0 -m "Release v1.1.0 - Description des changements"
git push origin v1.1.0
```

L'image Docker est automatiquement publiée sur :
```
ghcr.io/USERNAME/notam-classification:1.1.0
ghcr.io/USERNAME/notam-classification:1.1
ghcr.io/USERNAME/notam-classification:sha-xxxxxxx
```

---

## 📡 Monitoring

### Monitoring via CLI

```bash
# Analyse sans dérive (données stables)
uv run python scripts/run_monitoring.py

# Analyse avec dérive simulée (ALERTE attendue)
uv run python scripts/run_monitoring.py --drift

# Analyse sur 500 prédictions
uv run python scripts/run_monitoring.py --n 500
```

**Output attendu (sans dérive) :**
```
═══════════════════════════════════════════════════════
  📊 MONITORING SUMMARY
═══════════════════════════════════════════════════════
  Dataset drift    : ✅ NO
  Drifted features : 0/6
  Drift share      : 0%
  Tests passed     : 3/3
  ALERT            : 🟢 NO
```

**Output attendu (avec dérive) :**
```
═══════════════════════════════════════════════════════
  📊 MONITORING SUMMARY
═══════════════════════════════════════════════════════
  Dataset drift    : ⚠️  YES
  Drifted features : 3/6
  Drift share      : 50%
  Tests passed     : 1/3
  ALERT            : 🔴 YES
```

### Dashboard de monitoring

```bash
uv run streamlit run src/monitoring/monitoring_dashboard.py --server.port 8502
# → http://localhost:8502
```

### Rapports générés

Les rapports HTML Evidently sont sauvegardés dans `reports/drift/` :
- `drift_report_YYYYMMDD_HHMMSS.html` : Rapport complet de dérive
- `test_suite_YYYYMMDD_HHMMSS.html` : Résultats des tests pass/fail

### Seuils d'alerte

| Métrique | Seuil | Action |
|---|---|---|
| PSI | > 0.20 | Alerte dérive significative |
| Drift share | > 30% | Alerte distribution |
| p-value KS | < 0.05 | Dérive statistiquement significative |
| Tests failed | > 0 | Revue manuelle recommandée |

### Features surveillées

| Feature | Description |
|---|---|
| `char_count` | Longueur du NOTAM en caractères |
| `word_count` | Nombre de mots |
| `upper_ratio` | Proportion de lettres majuscules |
| `digit_ratio` | Proportion de chiffres |
| `slash_count` | Nombre de séparateurs `/` |
| `has_coordinates` | Présence de coordonnées GPS (0/1) |
| `predicted_category` | Distribution des catégories prédites |

---

## 🐳 Docker

### Lancer tous les services

```bash
# Production
docker compose up -d

# Vérifier les services
docker compose ps

# Logs en temps réel
docker compose logs -f api
```

### Services disponibles

| Service | Port | Description |
|---|---|---|
| `postgres` | 5432 | PostgreSQL 16 (données persistantes) |
| `mlflow` | 5000 | MLflow Tracking Server |
| `notam_api` | 8000 | FastAPI REST API |
| `frontend` | 3000 | React Frontend |

### Commandes utiles

```bash
# Rebuild après changement de code
docker compose build api
docker compose up -d api

# Accéder au conteneur API
docker exec -it notam_api bash

# Réinitialiser la base de données
docker compose down -v
docker compose up -d

# Mode développement (hot-reload)
docker compose -f docker-compose.dev.yml up
```

### Tirer l'image publiée

```bash
docker pull ghcr.io/USERNAME/notam-classification:1.0.0
```

---

## 📈 Performance du modèle

### Métriques globales

| Métrique | Score |
|---|---|
| F1-macro | **94.1%** |
| Accuracy | **94.3%** |
| Precision macro | 94.4% |
| Recall macro | 94.0% |

### F1-score par catégorie

| Catégorie | Precision | Recall | F1 |
|---|---|---|---|
| RUNWAY_CLOSURE | 96.2% | 95.8% | 96.0% |
| AIRSPACE_RESTRICTION | 95.1% | 94.7% | 94.9% |
| NAVIGATION_AID | 94.3% | 93.9% | 94.1% |
| OBSTACLE | 93.8% | 94.2% | 94.0% |
| LIGHTING | 92.7% | 93.1% | 92.9% |
| AERODROME_PROCEDURE | 91.4% | 92.1% | 91.7% |

### Hyperparamètres retenus (LinearSVC)

```python
CalibratedClassifierCV(
    estimator=LinearSVC(
        C=1.0,
        max_iter=1000,
        class_weight="balanced",
    ),
    method="sigmoid",
    cv=3,
)
```

### Pipeline de features

```python
NOTAMFeaturePipeline(
    tfidf=TfidfVectorizer(
        max_features=10_000,
        ngram_range=(1, 3),
        sublinear_tf=True,
        min_df=2,
        preprocessor=NOTAMTextPreprocessor(stemming=True),
    ),
    meta=MetaFeatureExtractor(
        features=["char_count", "word_count", "upper_ratio",
                  "digit_ratio", "slash_count", "has_coordinates"],
        scaler=StandardScaler(),
    ),
)
```

---

## 🤝 Contribuer

### Workflow Git

```bash
# Créer une branche feature
git checkout -b feat/nom-de-la-feature

# Développer + tester
uv run pytest tests/ -v

# Linter
uv run ruff check src/ tests/

# Commit
git add .
git commit -m "feat: description de la feature"

# Push + Pull Request vers develop
git push origin feat/nom-de-la-feature
```

### Convention de commits

```
feat:     Nouvelle fonctionnalité
fix:      Correction de bug
docs:     Documentation
test:     Ajout/modification de tests
refactor: Refactoring sans changement fonctionnel
chore:    Maintenance (CI, deps, config)
perf:     Amélioration de performance
```

### Checklist PR

- [ ] Tests unitaires passent (`uv run pytest tests/unit/ -v`)
- [ ] Tests d'intégration passent (`uv run pytest tests/integration/ -v`)
- [ ] Ruff ne remonte pas d'erreurs bloquantes
- [ ] Coverage ≥ 50% sur les modules modifiés
- [ ] Documentation mise à jour si besoin

---

## 📄 Licence

MIT License — voir le fichier [LICENSE](LICENSE) pour les détails.

---

## 👤 Auteur

Développé dans le cadre d'un projet MLOps complet démontrant les compétences de **ML Engineering** de bout en bout.

**Stack maîtrisée :** `Python` · `scikit-learn` · `FastAPI` · `SQLAlchemy` · `MLflow` · `Docker` · `pytest` · `GitHub Actions` · `Evidently AI` · `React` · `uv` · `Ruff`

---

<div align="center">
  <sub>✈️ Built with precision · MLOps End-to-End · v1.0.0</sub>
</div>