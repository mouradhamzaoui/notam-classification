"""
monitoring_dashboard.py
Dashboard Streamlit pour visualiser le monitoring en temps réel.
"""

from __future__ import annotations
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.monitoring.drift_detector import NOTAMDriftDetector
from src.utils.config import Config

# ── Config ────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NOTAM Monitor",
    page_icon="📡",
    layout="wide",
)

PALETTE = {
    "RUNWAY_CLOSURE":       "#ef4444",
    "NAVIGATION_AID":       "#3b82f6",
    "AIRSPACE_RESTRICTION": "#f59e0b",
    "LIGHTING":             "#8b5cf6",
    "OBSTACLE":             "#10b981",
    "AERODROME_PROCEDURE":  "#ec4899",
}

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  .main { background-color: #0d1117; }
  .metric-card {
    background: #161b22; border-radius: 12px;
    padding: 1.2rem; text-align: center;
    border: 1px solid #21262d;
  }
  .metric-value { font-size: 2rem; font-weight: 700; color: #e8f4fd; }
  .metric-label { font-size: 0.8rem; color: #8b949e; margin-top: 0.3rem; }
  .alert-box {
    background: #2d1515; border: 1px solid #ef4444;
    border-radius: 8px; padding: 1rem;
    color: #ef4444; font-weight: 600;
  }
  .ok-box {
    background: #0d1f13; border: 1px solid #10b981;
    border-radius: 8px; padding: 1rem;
    color: #10b981; font-weight: 600;
  }
</style>
""", unsafe_allow_html=True)


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("## 📡 NOTAM Classification — Monitoring Dashboard")
st.markdown("Surveillance de la dérive des données et du modèle en production.")
st.markdown("---")

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    n_prod    = st.slider("Données production simulées", 50, 500, 200, 50)
    with_drift= st.checkbox("Simuler une dérive artificielle", value=False)
    run_btn   = st.button("🔍 Lancer l'analyse", type="primary",
                          use_container_width=True)
    st.markdown("---")
    st.markdown("**Seuils d'alerte**")
    st.markdown("- PSI > 0.20 → ALERTE")
    st.markdown("- Drift share > 30% → ALERTE")
    st.markdown("- p-value < 0.05 → Dérive")


# ── Main ──────────────────────────────────────────────────────────────────────
if run_btn:
    with st.spinner("Analyse en cours..."):

        # ── Init detector ─────────────────────────────────────────────────────
        detector = NOTAMDriftDetector()
        try:
            detector.load_reference_from_csv("data/processed/notams_clean.csv")
        except Exception as e:
            st.error(f"❌ Impossible de charger les données de référence : {e}")
            st.stop()

        # ── Production data ───────────────────────────────────────────────────
        current_df = detector.generate_synthetic_production_data(
            n=n_prod, drift=with_drift
        )

        # ── Run drift analysis ────────────────────────────────────────────────
        try:
            metrics     = detector.run_data_drift_report(current_df, save=True)
            test_summary= detector.run_test_suite(current_df, save=True)
            alert       = detector.check_alert(metrics)
        except Exception as e:
            st.error(f"❌ Erreur lors de l'analyse : {e}")
            st.stop()

    # ── Alert Banner ──────────────────────────────────────────────────────────
    if alert or with_drift:
        st.markdown("""
        <div class="alert-box">
          ⚠️ ALERTE DÉRIVE DÉTECTÉE — La distribution des données de production
          s'écarte significativement du dataset d'entraînement.
          Envisagez un retraining du modèle.
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="ok-box">
          ✅ Distribution stable — Aucune dérive significative détectée.
          Le modèle est fiable sur les données actuelles.
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # ── Métriques clés ────────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    cols_data = [
        ("🔍 Dataset Drift",  "OUI ⚠️" if metrics["dataset_drift"] else "NON ✅",
         "#ef4444" if metrics["dataset_drift"] else "#10b981"),
        ("📊 Features driftées",
         f"{metrics['n_drifted']}/{metrics['n_features']}",
         "#f59e0b" if metrics['n_drifted'] > 0 else "#10b981"),
        ("📈 Drift Share",
         f"{metrics['drift_share']:.0%}",
         "#ef4444" if metrics['drift_share'] > 0.3 else "#10b981"),
        ("🧪 Tests passés",
         f"{test_summary['passed']}/{test_summary['total']}",
         "#ef4444" if test_summary['failed'] > 0 else "#10b981"),
    ]
    for col, (label, value, color) in zip([c1, c2, c3, c4], cols_data):
        col.markdown(f"""
        <div class="metric-card">
          <div class="metric-value" style="color:{color}">{value}</div>
          <div class="metric-label">{label}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")

    # ── Distribution comparaison ──────────────────────────────────────────────
    col_ref, col_cur = st.columns(2)

    with col_ref:
        st.markdown("#### 📚 Distribution de référence (Training)")
        ref_df = pd.read_csv("data/processed/notams_clean.csv")
        ref_counts = ref_df["category"].value_counts()
        fig_ref = go.Figure(go.Bar(
            x=[c.replace("_", " ") for c in ref_counts.index],
            y=ref_counts.values,
            marker_color=[PALETTE.get(c, "#555") for c in ref_counts.index],
            marker_line_width=0,
        ))
        fig_ref.update_layout(
            paper_bgcolor="#0d1117", plot_bgcolor="#161b22",
            height=300, margin=dict(l=0, r=0, t=10, b=60),
            xaxis=dict(color="#8b949e", tickangle=-20),
            yaxis=dict(color="#8b949e"),
            showlegend=False,
        )
        st.plotly_chart(fig_ref, use_container_width=True)

    with col_cur:
        st.markdown("#### 🏭 Distribution production (actuelle)")
        cur_counts = current_df["predicted_category"].value_counts()
        fig_cur = go.Figure(go.Bar(
            x=[c.replace("_", " ") for c in cur_counts.index],
            y=cur_counts.values,
            marker_color=[PALETTE.get(c, "#555") for c in cur_counts.index],
            marker_line_width=0,
        ))
        fig_cur.update_layout(
            paper_bgcolor="#0d1117", plot_bgcolor="#161b22",
            height=300, margin=dict(l=0, r=0, t=10, b=60),
            xaxis=dict(color="#8b949e", tickangle=-20),
            yaxis=dict(color="#8b949e"),
            showlegend=False,
        )
        st.plotly_chart(fig_cur, use_container_width=True)

    # ── Features numériques ───────────────────────────────────────────────────
    st.markdown("#### 📊 Distribution des features numériques")
    feat_cols = st.columns(3)
    features  = ["char_count", "word_count", "digit_ratio"]

    for i, feat in enumerate(features):
        with feat_cols[i]:
            ref_vals = ref_df[feat].dropna() if feat in ref_df.columns else pd.Series()
            cur_vals = current_df[feat].dropna()

            fig = go.Figure()
            if len(ref_vals) > 0:
                fig.add_trace(go.Histogram(
                    x=ref_vals, name="Référence",
                    opacity=0.6, marker_color="#3b82f6",
                    histnorm="probability",
                ))
            fig.add_trace(go.Histogram(
                x=cur_vals, name="Production",
                opacity=0.6, marker_color="#f59e0b",
                histnorm="probability",
            ))
            fig.update_layout(
                barmode="overlay",
                paper_bgcolor="#0d1117", plot_bgcolor="#161b22",
                height=220, margin=dict(l=0, r=0, t=25, b=30),
                title=dict(text=feat, font=dict(color="white", size=11)),
                xaxis=dict(color="#555"),
                yaxis=dict(color="#555"),
                legend=dict(font=dict(color="white", size=8),
                            bgcolor="#1c1c2c"),
                showlegend=True,
            )
            st.plotly_chart(fig, use_container_width=True)

    # ── Rapports HTML ─────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### 📄 Rapports générés")
    reports = sorted(Path("reports/drift").glob("*.html"), reverse=True)
    if reports:
        for r in reports[:5]:
            st.markdown(f"- 📄 `{r.name}`")
        st.info("Les rapports HTML Evidently sont dans `reports/drift/`")
    else:
        st.warning("Aucun rapport généré pour l'instant.")

else:
    st.markdown("""
    <div style="text-align:center;padding:4rem;color:#555">
      <div style="font-size:4rem">📡</div>
      <div style="font-size:1.1rem;margin-top:1rem">
        Configure les paramètres et clique sur <strong>Lancer l'analyse</strong>
      </div>
    </div>
    """, unsafe_allow_html=True)