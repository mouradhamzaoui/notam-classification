"""
train.py
Entraînement de 3 algorithmes de classification NOTAM.

Modèles candidats :
  1. Logistic Regression  — Baseline linéaire rapide
  2. Random Forest        — Ensemble non-linéaire, interprétable
  3. LinearSVC            — SVM linéaire, état de l'art sur texte sparse

Justifications mathématiques dans les docstrings de chaque classe.
"""

import time
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import (
    classification_report,
    f1_score,
    accuracy_score,
    confusion_matrix,
)

# ── Constantes ────────────────────────────────────────────────────────────────
MODELS_DIR = Path("data/processed/models")
MODELS_DIR.mkdir(parents=True, exist_ok=True)


# ── Dataclass résultat ────────────────────────────────────────────────────────
@dataclass
class TrainingResult:
    model_name:    str
    model:         Any
    train_time_s:  float
    cv_f1_mean:    float
    cv_f1_std:     float
    test_f1_macro: float
    test_accuracy: float
    report:        str
    confusion_mat: np.ndarray
    params:        dict = field(default_factory=dict)

    def summary(self) -> str:
        return (
            f"┌─ {self.model_name:30s} ──────────────────────────┐\n"
            f"│  CV  F1-macro  : {self.cv_f1_mean:.4f} ± {self.cv_f1_std:.4f}          │\n"
            f"│  Test F1-macro : {self.test_f1_macro:.4f}                          │\n"
            f"│  Test Accuracy : {self.test_accuracy:.4f}                          │\n"
            f"│  Train time    : {self.train_time_s:.2f}s                           │\n"
            f"└──────────────────────────────────────────────────┘"
        )


# ══════════════════════════════════════════════════════════════════════════════
# MODÈLE 1 — Logistic Regression
# ══════════════════════════════════════════════════════════════════════════════
class LogisticRegressionModel:
    """
    Régression Logistique Multinomiale (softmax).

    Formule (one-vs-rest) :
        P(y=k | x) = exp(wₖᵀx + bₖ) / Σⱼ exp(wⱼᵀx + bⱼ)

    Optimisation :
        Minimise la log-vraisemblance négative + régularisation L2 :
        L(w) = -Σᵢ log P(yᵢ|xᵢ) + (λ/2) ||w||²

    Pourquoi adapté aux NOTAMs ?
    - Linéaire dans l'espace TF-IDF haute dimension → très efficace.
    - Les probabilités calibrées permettent d'afficher le score de confiance
      dans l'interface (essentiel pour l'application aviation).
    - Solver 'saga' : optimal pour les matrices TF-IDF sparse et larges.
    - C (inverse de λ) : contrôle le compromis biais-variance.
    """

    NAME = "Logistic Regression"

    def __init__(self, C: float = 5.0, max_iter: int = 1000):
        self.model = LogisticRegression(
            C=C,
            solver="saga",
            max_iter=max_iter,
            n_jobs=-1,
            random_state=42,
            class_weight="balanced",
        )
        self.params = {"C": C, "max_iter": max_iter, "solver": "saga"}

    def fit(self, X, y):
        return self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)


# ══════════════════════════════════════════════════════════════════════════════
# MODÈLE 2 — Random Forest
# ══════════════════════════════════════════════════════════════════════════════
class RandomForestModel:
    """
    Forêt Aléatoire (Breiman, 2001).

    Principe :
        Entraîne B arbres de décision sur des sous-échantillons bootstrap.
        Prédiction finale = vote majoritaire :
            ŷ = argmax_k Σᵦ 𝟙[hᵦ(x) = k]

    Deux sources de randomisation :
        1. Bootstrap : chaque arbre voit ~63.2% des données (out-of-bag ≈ 36.8%)
        2. Feature subsampling : à chaque split, tire √p features au hasard
           → réduit la corrélation entre arbres (Théorème de la variance RF)

    Variance d'un RF :
        Var(ŷ) = ρ·σ² + (1-ρ)·σ²/B
        où ρ = corrélation entre arbres, σ² = variance d'un arbre seul.
        → Plus ρ est faible (via feature subsampling), plus la variance baisse.

    Pourquoi adapté aux NOTAMs ?
    - Capture les interactions non-linéaires entre features
      (ex: "has_coordinates=1 ET word_count>10" → OBSTACLE).
    - Feature importance interprétable → audit réglementaire possible.
    - Robuste aux features bruitées (digit_ratio sur textes variables).

    Limitation : lent à prédire sur de grosses matrices sparse TF-IDF.
    → n_estimators=200 est un bon compromis performance/vitesse.
    """

    NAME = "Random Forest"

    def __init__(self, n_estimators: int = 200, max_depth: int = None):
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features="sqrt",
            class_weight="balanced",
            n_jobs=-1,
            random_state=42,
            oob_score=True,
        )
        self.params = {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "max_features": "sqrt",
        }

    def fit(self, X, y):
        self.model.fit(X, y)
        print(f"   [RF] OOB Score : {self.model.oob_score_:.4f}")
        return self.model

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)


# ══════════════════════════════════════════════════════════════════════════════
# MODÈLE 3 — LinearSVC (état de l'art texte)
# ══════════════════════════════════════════════════════════════════════════════
class LinearSVCModel:
    """
    Support Vector Machine linéaire (Crammer & Singer multiclasse).

    Objectif d'optimisation :
        min_{w,b,ξ} (1/2)||w||² + C·Σᵢ ξᵢ
        s.t. wᵧᵢᵀxᵢ - wₖᵀxᵢ ≥ 1 - ξᵢ  ∀k ≠ yᵢ, ξᵢ ≥ 0

    La marge maximale garantit une généralisation optimale (théorie VC).
    Avec kernel linéaire + TF-IDF, LinearSVC est souvent l'état de l'art
    sur les tâches de classification de texte court (Joachims, 1998).

    Note : LinearSVC ne produit pas de probabilités nativement.
    → CalibratedClassifierCV avec Platt Scaling :
        P(y=k|x) = σ(Awₖᵀx + B)
        A, B estimés par régression logistique sur validation set.

    Pourquoi adapté aux NOTAMs ?
    - Optimal en haute dimension sparse (TF-IDF 5000 features).
    - Invariant à l'échelle → pas besoin de normaliser L2 en plus.
    - 10-50x plus rapide que SVM kernel RBF.
    - C=1.0 : marge standard. Tunable via GridSearch.
    """

    NAME = "LinearSVC (Calibrated)"

    def __init__(self, C: float = 1.0, max_iter: int = 2000):
        base_svc = LinearSVC(
            C=C,
            multi_class="crammer_singer",
            max_iter=max_iter,
            random_state=42,
            class_weight="balanced",
        )
        # Calibration pour obtenir des probabilités
        self.model = CalibratedClassifierCV(
            estimator=base_svc,
            cv=3,
            method="sigmoid",
        )
        self.params = {"C": C, "max_iter": max_iter, "calibration": "sigmoid"}

    def fit(self, X, y):
        return self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)


# ══════════════════════════════════════════════════════════════════════════════
# TRAINER — Orchestrateur d'entraînement
# ══════════════════════════════════════════════════════════════════════════════
class NOTAMTrainer:
    """
    Orchestre l'entraînement, la cross-validation et l'évaluation
    des 3 modèles candidats.
    """

    def __init__(self, cv_folds: int = 5):
        self.cv_folds = cv_folds
        self.cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        self.results: list[TrainingResult] = []

    def train_and_evaluate(
        self,
        model_wrapper,
        X_train, y_train,
        X_test,  y_test,
        classes: list,
    ) -> TrainingResult:

        name = model_wrapper.NAME
        print(f"\n{'='*55}")
        print(f"  🚀 Training : {name}")
        print(f"{'='*55}")

        # ── Cross-validation F1-macro ─────────────────────────────────────────
        print(f"  ⚙️  {self.cv_folds}-fold Stratified CV...")
        cv_scores = cross_val_score(
            model_wrapper.model, X_train, y_train,
            cv=self.cv, scoring="f1_macro", n_jobs=-1,
        )
        print(f"  CV F1-macro : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

        # ── Entraînement final ────────────────────────────────────────────────
        print(f"  📚 Final fit on full train set...")
        t0 = time.time()
        model_wrapper.fit(X_train, y_train)
        train_time = time.time() - t0
        print(f"  ✅ Done in {train_time:.2f}s")

        # ── Évaluation test ───────────────────────────────────────────────────
        y_pred = model_wrapper.predict(X_test)
        test_f1  = f1_score(y_test, y_pred, average="macro")
        test_acc = accuracy_score(y_test, y_pred)
        report   = classification_report(y_test, y_pred, target_names=classes)
        cm       = confusion_matrix(y_test, y_pred)

        result = TrainingResult(
            model_name    = name,
            model         = model_wrapper.model,
            train_time_s  = train_time,
            cv_f1_mean    = cv_scores.mean(),
            cv_f1_std     = cv_scores.std(),
            test_f1_macro = test_f1,
            test_accuracy = test_acc,
            report        = report,
            confusion_mat = cm,
            params        = model_wrapper.params,
        )

        print(result.summary())
        self.results.append(result)

        # Sauvegarde individuelle
        save_path = MODELS_DIR / f"{name.replace(' ', '_').replace('(','').replace(')','')}.pkl"
        joblib.dump(model_wrapper.model, save_path)
        print(f"  💾 Model saved → {save_path}")

        return result

    def get_best(self) -> TrainingResult:
        """Retourne le meilleur modèle selon le F1-macro test."""
        return max(self.results, key=lambda r: r.test_f1_macro)

    def comparison_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame([{
            "Model":        r.model_name,
            "CV F1-macro":  round(r.cv_f1_mean, 4),
            "CV Std":       round(r.cv_f1_std, 4),
            "Test F1-macro":round(r.test_f1_macro, 4),
            "Test Accuracy":round(r.test_accuracy, 4),
            "Train Time(s)":round(r.train_time_s, 2),
        } for r in self.results]).sort_values("Test F1-macro", ascending=False)