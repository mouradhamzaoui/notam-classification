"""
evaluate.py
Fonctions de visualisation et d'évaluation des modèles NOTAM.
"""

from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.model_selection import learning_curve

PALETTE = {
    "RUNWAY_CLOSURE": "#ef4444",
    "NAVIGATION_AID": "#3b82f6",
    "AIRSPACE_RESTRICTION": "#f59e0b",
    "LIGHTING": "#8b5cf6",
    "OBSTACLE": "#10b981",
    "AERODROME_PROCEDURE": "#ec4899",
}
MODEL_COLORS = ["#3b82f6", "#10b981", "#f59e0b"]


def plot_model_comparison(results: list, save_path: str = None):
    """Graphique comparatif des 3 modèles."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.patch.set_facecolor("#0d1117")

    names = [r.model_name.replace(" (Calibrated)", "") for r in results]
    cv_means = [r.cv_f1_mean for r in results]
    cv_stds = [r.cv_f1_std for r in results]
    test_f1s = [r.test_f1_macro for r in results]
    [r.test_accuracy for r in results]
    times = [r.train_time_s for r in results]

    def _bar_ax(ax, values, errs, title, ylabel, fmt=".4f"):
        ax.set_facecolor("#161b22")
        bars = ax.bar(
            range(len(names)),
            values,
            yerr=errs if errs else None,
            color=MODEL_COLORS[: len(names)],
            alpha=0.85,
            edgecolor="none",
            capsize=5,
            error_kw={"color": "white", "linewidth": 1.5},
        )
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, fontsize=9, color="#8b949e", rotation=10)
        ax.set_title(title, color="white", fontsize=11, pad=10)
        ax.set_ylabel(ylabel, color="#8b949e", fontsize=9)
        ax.spines[:].set_visible(False)
        ax.tick_params(colors="#555")
        best_idx = values.index(max(values))
        for i, (bar, val) in enumerate(zip(bars, values, strict=False)):
            color = "gold" if i == best_idx else "white"
            weight = "bold" if i == best_idx else "normal"
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + (max(values) * 0.01),
                f"{val:{fmt}}",
                ha="center",
                color=color,
                fontsize=10,
                fontweight=weight,
            )

    _bar_ax(axes[0], cv_means, cv_stds, "CV F1-macro (5-fold ± std)", "F1 Score")
    _bar_ax(axes[1], test_f1s, None, "Test F1-macro", "F1 Score")
    _bar_ax(axes[2], times, None, "Temps d'entraînement (s)", "Secondes", fmt=".2f")

    plt.suptitle("🏆 Comparaison des Modèles NOTAM — Baseline", color="white", fontsize=14, y=1.02)
    plt.tight_layout(pad=3)
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="#0d1117")
    plt.show()


def plot_confusion_matrices(results: list, classes: list, save_path: str = None):
    """Matrices de confusion normalisées pour les 3 modèles."""
    fig, axes = plt.subplots(1, len(results), figsize=(7 * len(results), 6))
    fig.patch.set_facecolor("#0d1117")

    for ax, result in zip(axes, results, strict=False):
        cm_norm = result.confusion_mat.astype(float)
        cm_norm = cm_norm / cm_norm.sum(axis=1, keepdims=True)

        short_classes = [c.replace("_", "\n") for c in classes]
        sns.heatmap(
            cm_norm,
            ax=ax,
            annot=True,
            fmt=".0%",
            cmap="Blues",
            linewidths=0.5,
            linecolor="#0d1117",
            xticklabels=short_classes,
            yticklabels=short_classes,
            annot_kws={"size": 8},
            cbar_kws={"shrink": 0.8},
        )
        ax.set_title(
            result.model_name.replace(" (Calibrated)", ""), color="white", fontsize=11, pad=10
        )
        ax.tick_params(colors="#8b949e", labelsize=7)
        ax.set_xlabel("Prédit", color="#8b949e")
        ax.set_ylabel("Réel", color="#8b949e")

    plt.suptitle("🔍 Matrices de Confusion Normalisées", color="white", fontsize=14, y=1.02)
    plt.tight_layout(pad=3)
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="#0d1117")
    plt.show()


def plot_per_class_f1(results: list, classes: list, save_path: str = None):
    """F1-score par classe pour chaque modèle."""
    from sklearn.metrics import f1_score as _f1

    fig, ax = plt.subplots(figsize=(14, 6))
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#161b22")

    x = np.arange(len(classes))
    width = 0.25

    for i, (result, color) in enumerate(zip(results, MODEL_COLORS, strict=False)):
        # Recalcul F1 par classe à partir de la confusion matrix
        cm = result.confusion_mat
        tp = np.diag(cm)
        fp = cm.sum(axis=0) - tp
        fn = cm.sum(axis=1) - tp
        f1_per_class = 2 * tp / (2 * tp + fp + fn + 1e-10)

        ax.bar(
            x + i * width,
            f1_per_class,
            width,
            label=result.model_name.replace(" (Calibrated)", ""),
            color=color,
            alpha=0.85,
            edgecolor="none",
        )

    ax.set_xticks(x + width)
    ax.set_xticklabels([c.replace("_", "\n") for c in classes], fontsize=9, color="#8b949e")
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("F1-Score", color="#8b949e")
    ax.set_title("📊 F1-Score par Classe et par Modèle", color="white", fontsize=13, pad=15)
    ax.legend(fontsize=9, framealpha=0.2, labelcolor="white", facecolor="#1c1c2c")
    ax.spines[:].set_visible(False)
    ax.tick_params(colors="#555")
    ax.axhline(0.85, color="gold", linestyle="--", linewidth=1, alpha=0.6, label="Cible ≥ 0.85")
    ax.grid(axis="y", alpha=0.08)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="#0d1117")
    plt.show()


def plot_learning_curve(model, X_train, y_train, model_name: str, save_path: str = None):
    """Courbe d'apprentissage pour diagnostiquer le biais/variance."""
    train_sizes, train_scores, val_scores = learning_curve(
        model,
        X_train,
        y_train,
        train_sizes=np.linspace(0.1, 1.0, 8),
        cv=5,
        scoring="f1_macro",
        n_jobs=-1,
        random_state=42,
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#161b22")

    train_mean = train_scores.mean(axis=1)
    train_std = train_scores.std(axis=1)
    val_mean = val_scores.mean(axis=1)
    val_std = val_scores.std(axis=1)

    ax.plot(
        train_sizes,
        train_mean,
        "o-",
        color="#3b82f6",
        linewidth=2.5,
        label="Train F1",
        markersize=7,
    )
    ax.fill_between(
        train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.15, color="#3b82f6"
    )

    ax.plot(
        train_sizes,
        val_mean,
        "s-",
        color="#10b981",
        linewidth=2.5,
        label="Validation F1",
        markersize=7,
    )
    ax.fill_between(
        train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.15, color="#10b981"
    )

    ax.axhline(0.85, color="gold", linestyle="--", linewidth=1.2, alpha=0.7, label="Cible ≥ 0.85")
    ax.set_xlabel("Taille du training set", color="#8b949e", fontsize=11)
    ax.set_ylabel("F1-macro Score", color="#8b949e", fontsize=11)
    ax.set_title(f"📈 Courbe d'apprentissage — {model_name}", color="white", fontsize=13, pad=15)
    ax.legend(fontsize=10, framealpha=0.2, labelcolor="white", facecolor="#1c1c2c")
    ax.spines[:].set_visible(False)
    ax.tick_params(colors="#555")
    ax.grid(alpha=0.08)
    ax.set_ylim(0.4, 1.05)

    gap = abs(train_mean[-1] - val_mean[-1])
    diagnosis = (
        "✅ Good fit"
        if gap < 0.05
        else ("⚠️  High variance" if train_mean[-1] > val_mean[-1] + 0.05 else "⚠️  High bias")
    )
    ax.text(
        0.02,
        0.05,
        f"Gap train/val = {gap:.3f} → {diagnosis}",
        transform=ax.transAxes,
        color="#8b949e",
        fontsize=9,
    )

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="#0d1117")
    plt.show()
