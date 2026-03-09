"""
streamlit_app.py
Interface POC de démonstration pour la classification automatique de NOTAMs.
"""

import sys
import re
import time
from pathlib import Path

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import streamlit as st

# ── Path setup ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from app.poc.predictor import NOTAMPredictor, CATEGORY_META, PRIORITY_ORDER

# ══════════════════════════════════════════════════════════════════════════════
# CONFIG STREAMLIT
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="NOTAM Classifier",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS personnalisé ──────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Inter:wght@300;400;600;700&display=swap');

  html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

  .main { background-color: #0d1117; }

  .hero-banner {
    background: linear-gradient(135deg, #0d1b2e 0%, #1a1f35 50%, #0d1117 100%);
    border: 1px solid #1e3a5f;
    border-radius: 16px;
    padding: 2rem 2.5rem;
    margin-bottom: 1.5rem;
    position: relative;
    overflow: hidden;
  }
  .hero-banner::before {
    content: "✈";
    position: absolute;
    right: 2rem; top: 1rem;
    font-size: 6rem;
    opacity: 0.06;
  }
  .hero-title {
    font-size: 2.2rem; font-weight: 700;
    color: #e8f4fd; margin: 0; letter-spacing: -0.5px;
  }
  .hero-sub {
    color: #4a90d9; font-size: 0.95rem;
    margin-top: 0.3rem; font-weight: 300;
  }

  .result-card {
    background: #161b22;
    border-radius: 12px;
    padding: 1.5rem;
    border-left: 4px solid;
    margin-bottom: 1rem;
  }
  .result-title {
    font-size: 1.4rem; font-weight: 700;
    margin: 0 0 0.3rem 0;
  }
  .result-priority {
    display: inline-block;
    padding: 0.2rem 0.75rem;
    border-radius: 999px;
    font-size: 0.75rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    margin-bottom: 0.8rem;
  }
  .confidence-bar-bg {
    background: #21262d;
    border-radius: 999px;
    height: 8px;
    margin-top: 0.5rem;
  }
  .metric-pill {
    background: #21262d;
    border-radius: 8px;
    padding: 0.6rem 1rem;
    text-align: center;
  }
  .metric-value {
    font-size: 1.6rem; font-weight: 700; color: #e8f4fd;
  }
  .metric-label {
    font-size: 0.75rem; color: #8b949e; margin-top: 0.1rem;
  }
  .notam-text {
    font-family: 'JetBrains Mono', monospace;
    background: #0d1117;
    border: 1px solid #21262d;
    border-radius: 8px;
    padding: 1rem;
    font-size: 0.85rem;
    color: #c9d1d9;
    white-space: pre-wrap;
    word-break: break-word;
  }
  .action-box {
    background: #1c2128;
    border-radius: 8px;
    padding: 0.8rem 1rem;
    border-left: 3px solid #3b82f6;
    margin-top: 0.8rem;
    font-size: 0.88rem;
    color: #8b949e;
  }
  .batch-row-critical { background: #2d1515; border-radius: 6px; padding: 0.3rem 0.5rem; }
  .batch-row-high     { background: #1a1f0d; border-radius: 6px; padding: 0.3rem 0.5rem; }
  .stTextArea textarea {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.85rem !important;
    background: #0d1117 !important;
    color: #c9d1d9 !important;
  }
  div[data-testid="stSidebar"] { background: #0d1117; border-right: 1px solid #21262d; }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# INIT
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner=False)
def load_predictor():
    p = NOTAMPredictor()
    p.load()
    return p

# ── Exemples de NOTAMs ────────────────────────────────────────────────────────
EXAMPLES = {
    "🛬 Fermeture de piste":     "RWY 28L CLSD DUE TO CONSTRUCTION WIP",
    "📡 Panne ILS":              "ILS CAT II RWY 10R NOT AVAILABLE",
    "🚫 Zone restreinte":        "RESTRICTED AREA R-2508 ACTIVE SFC-18000FT MSL",
    "💡 PAPI hors service":      "PAPI RWY 36 OTS",
    "🏗️ Obstacle (grue)":       "NEW OBSTACLE CRANE 520FT AGL WITHIN 3.5NM OF LFPG ARP",
    "📋 Procédure aérodrome":    "FUEL NOT AVBL 2H DAILY DUE MAINTENANCE",
}

PRIORITY_COLORS = {
    "CRITICAL": "#ef4444",
    "HIGH":     "#f59e0b",
    "MEDIUM":   "#8b5cf6",
    "LOW":      "#10b981",
}

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("### ✈️ NOTAM Classifier")
    st.markdown("---")
    page = st.radio(
        "Navigation",
        ["🎯 Classification", "📦 Batch Processing", "📊 À propos du modèle"],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.markdown("**Stack technique**")
    st.markdown("""
    - 🧠 LinearSVC (Calibrated)
    - ⚙️ TF-IDF + Meta Features
    - 🏷️ 6 catégories ICAO
    - 📐 5,007 features
    """)
    st.markdown("---")
    st.markdown(
        "<div style='color:#555;font-size:0.75rem'>Phase 1 POC · notam-classification</div>",
        unsafe_allow_html=True,
    )

# ══════════════════════════════════════════════════════════════════════════════
# HERO BANNER
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="hero-banner">
  <div class="hero-title">Automatic NOTAM Classification</div>
  <div class="hero-sub">
    Système de classification automatique des avis aux navigants aériens · 
    Powered by LinearSVC + TF-IDF · ICAO Standard
  </div>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE : CLASSIFICATION
# ══════════════════════════════════════════════════════════════════════════════
if "🎯 Classification" in page:

    col_input, col_result = st.columns([1, 1], gap="large")

    with col_input:
        st.markdown("#### 📝 Saisie du NOTAM")

        # Exemples rapides
        st.markdown("**Exemples rapides :**")
        cols_ex = st.columns(3)
        example_text = ""
        for i, (label, text) in enumerate(EXAMPLES.items()):
            if cols_ex[i % 3].button(label, use_container_width=True, key=f"ex_{i}"):
                example_text = text

        notam_input = st.text_area(
            "Texte du NOTAM",
            value=example_text,
            height=140,
            placeholder="Ex: RWY 28L CLSD DUE TO CONSTRUCTION WIP",
            label_visibility="collapsed",
        )

        col_btn, col_clear = st.columns([3, 1])
        classify_btn = col_btn.button(
            "🔍 Classifier ce NOTAM",
            type="primary",
            use_container_width=True,
        )

        if notam_input.strip():
            st.markdown("**Texte brut (format ICAO) :**")
            st.markdown(
                f'<div class="notam-text">{notam_input.upper()}</div>',
                unsafe_allow_html=True,
            )

    with col_result:
        st.markdown("#### 🎯 Résultat de Classification")

        if classify_btn and notam_input.strip():
            with st.spinner("Analyse en cours..."):
                predictor = load_predictor()
                t0 = time.time()
                result = predictor.predict_one(notam_input)
                latency_ms = (time.time() - t0) * 1000

            cat    = result["category"]
            conf   = result["confidence"]
            meta   = result["meta"]
            probas = result["probabilities"]

            # ── Carte résultat ─────────────────────────────────────────────────
            priority_color = PRIORITY_COLORS.get(meta.get("priority", "LOW"), "#555")
            st.markdown(f"""
            <div class="result-card" style="border-color:{meta.get('color','#555')}">
              <div class="result-title" style="color:{meta.get('color','white')}">
                {meta.get('icon','✈️')}  {cat.replace('_', ' ').title()}
              </div>
              <span class="result-priority"
                style="background:{priority_color}22;color:{priority_color}">
                ⚡ {meta.get('priority','—')}
              </span>
              <p style="color:#8b949e;font-size:0.88rem;margin:0.3rem 0">
                {meta.get('description','')}
              </p>
              <div class="action-box">
                🔧 <strong>Action recommandée :</strong> {meta.get('action','')}
              </div>
            </div>
            """, unsafe_allow_html=True)

            # ── Métriques ──────────────────────────────────────────────────────
            m1, m2, m3 = st.columns(3)
            m1.markdown(f"""
            <div class="metric-pill">
              <div class="metric-value" style="color:{meta.get('color','white')}">
                {conf:.1%}
              </div>
              <div class="metric-label">Confiance</div>
            </div>""", unsafe_allow_html=True)
            m2.markdown(f"""
            <div class="metric-pill">
              <div class="metric-value">{latency_ms:.0f}ms</div>
              <div class="metric-label">Latence</div>
            </div>""", unsafe_allow_html=True)
            m3.markdown(f"""
            <div class="metric-pill">
              <div class="metric-value">{len(notam_input.split())}</div>
              <div class="metric-label">Mots</div>
            </div>""", unsafe_allow_html=True)

            # ── Graphique probabilités ─────────────────────────────────────────
            st.markdown("**Distribution des probabilités :**")
            sorted_probas = sorted(probas.items(), key=lambda x: x[1], reverse=True)
            cats_sorted   = [c.replace("_", " ") for c, _ in sorted_probas]
            vals_sorted   = [v for _, v in sorted_probas]
            colors_sorted = [
                CATEGORY_META.get(c, {}).get("color", "#555")
                for c, _ in sorted_probas
            ]

            fig = go.Figure(go.Bar(
                x=vals_sorted, y=cats_sorted,
                orientation="h",
                marker_color=colors_sorted,
                marker_line_width=0,
                text=[f"{v:.1%}" for v in vals_sorted],
                textposition="outside",
                textfont=dict(color="white", size=11),
            ))
            fig.update_layout(
                paper_bgcolor="#0d1117", plot_bgcolor="#161b22",
                height=260, margin=dict(l=0, r=60, t=10, b=10),
                xaxis=dict(showgrid=False, showticklabels=False,
                           zeroline=False, range=[0, 1.15]),
                yaxis=dict(color="#8b949e", tickfont=dict(size=10)),
                showlegend=False,
            )
            st.plotly_chart(fig, use_container_width=True)

        elif classify_btn:
            st.warning("⚠️ Saisis un texte NOTAM avant de classifier.")
        else:
            st.markdown("""
            <div style="text-align:center;padding:3rem 1rem;color:#555">
              <div style="font-size:3rem">✈️</div>
              <div style="margin-top:0.5rem;font-size:0.9rem">
                Saisis un NOTAM ou clique sur un exemple
              </div>
            </div>
            """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE : BATCH PROCESSING
# ══════════════════════════════════════════════════════════════════════════════
elif "📦 Batch" in page:
    st.markdown("#### 📦 Traitement par lot")
    st.markdown("Colle plusieurs NOTAMs (un par ligne) ou uploade un fichier CSV.")

    tab1, tab2 = st.tabs(["✍️ Saisie manuelle", "📁 Upload CSV"])

    with tab1:
        default_batch = "\n".join([
            "RWY 10L CLSD DUE TO MAINTENANCE",
            "ILS RWY 28R NOT AVAILABLE",
            "RESTRICTED AREA R-4009 ACTIVE SFC-10000FT",
            "PAPI RWY 18 OTS",
            "NEW CRANE 480FT AGL WITHIN 2NM OF EHAM ARP",
            "FUEL NOT AVBL 0600-1400 DAILY",
        ])
        batch_input = st.text_area(
            "NOTAMs (un par ligne)",
            value=default_batch,
            height=200,
        )
        if st.button("🔍 Classifier le lot", type="primary"):
            lines = [l.strip() for l in batch_input.strip().splitlines() if l.strip()]
            if lines:
                predictor = load_predictor()
                with st.spinner(f"Classification de {len(lines)} NOTAMs..."):
                    results = predictor.predict_batch(lines)

                # ── Tableau résultats ──────────────────────────────────────────
                rows = []
                for text, res in zip(lines, results):
                    meta = res["meta"]
                    rows.append({
                        "Priorité":   meta.get("priority", "—"),
                        "Catégorie":  f"{meta.get('icon','')} {res['category'].replace('_',' ').title()}",
                        "Confiance":  f"{res['confidence']:.1%}",
                        "NOTAM":      text[:70] + ("…" if len(text) > 70 else ""),
                    })

                df_results = pd.DataFrame(rows)
                priority_order = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}
                df_results["_sort"] = df_results["Priorité"].map(priority_order)
                df_results = df_results.sort_values("_sort").drop("_sort", axis=1)

                st.markdown(f"**{len(lines)} NOTAMs classifiés — triés par priorité :**")
                st.dataframe(
                    df_results,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "Confiance": st.column_config.ProgressColumn(
                            "Confiance", min_value=0, max_value=1, format="%.0%"
                        ),
                    },
                )

                # ── Résumé ─────────────────────────────────────────────────────
                st.markdown("**Résumé par catégorie :**")
                cats_count = pd.Series(
                    [r["category"] for r in results]
                ).value_counts()
                fig_pie = go.Figure(go.Pie(
                    labels=[c.replace("_"," ").title() for c in cats_count.index],
                    values=cats_count.values,
                    hole=0.55,
                    marker_colors=[
                        CATEGORY_META.get(c, {}).get("color", "#555")
                        for c in cats_count.index
                    ],
                    textfont=dict(color="white"),
                ))
                fig_pie.update_layout(
                    paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
                    height=300, showlegend=True,
                    legend=dict(font=dict(color="white"), bgcolor="#161b22"),
                    margin=dict(l=0, r=0, t=10, b=10),
                )
                st.plotly_chart(fig_pie, use_container_width=True)

    with tab2:
        uploaded = st.file_uploader("Fichier CSV (colonne 'body_text' requise)", type=["csv"])
        if uploaded:
            df_up = pd.read_csv(uploaded)
            if "body_text" not in df_up.columns:
                st.error("❌ Le CSV doit contenir une colonne 'body_text'.")
            else:
                predictor = load_predictor()
                with st.spinner(f"Classification de {len(df_up)} NOTAMs..."):
                    results = predictor.predict_batch(df_up["body_text"].tolist())
                df_up["predicted_category"] = [r["category"] for r in results]
                df_up["confidence"]          = [r["confidence"] for r in results]
                st.success(f"✅ {len(df_up)} NOTAMs classifiés !")
                st.dataframe(df_up, use_container_width=True, hide_index=True)
                csv_out = df_up.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "⬇️ Télécharger les résultats",
                    csv_out, "notams_classified.csv", "text/csv",
                )

# ══════════════════════════════════════════════════════════════════════════════
# PAGE : À PROPOS DU MODÈLE
# ══════════════════════════════════════════════════════════════════════════════
elif "📊" in page:
    st.markdown("#### 📊 Informations sur le modèle")

    c1, c2, c3, c4 = st.columns(4)
    for col, (val, label) in zip([c1,c2,c3,c4], [
        ("LinearSVC", "Algorithme"),
        ("5,007",     "Features"),
        ("6",         "Catégories"),
        ("~94%",      "F1-macro"),
    ]):
        col.markdown(f"""
        <div class="metric-pill" style="margin-bottom:1rem">
          <div class="metric-value">{val}</div>
          <div class="metric-label">{label}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("**Catégories gérées :**")
    for cat, meta in CATEGORY_META.items():
        p_color = PRIORITY_COLORS.get(meta["priority"], "#555")
        st.markdown(f"""
        <div class="result-card" style="border-color:{meta['color']};margin-bottom:0.5rem">
          <strong style="color:{meta['color']}">{meta['icon']} {cat.replace('_',' ').title()}</strong>
          <span class="result-priority"
            style="background:{p_color}22;color:{p_color};margin-left:0.5rem">
            {meta['priority']}
          </span>
          <p style="color:#8b949e;font-size:0.85rem;margin:0.3rem 0 0 0">
            {meta['description']}
          </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("**Pipeline de traitement :**")
    st.code("""
Input NOTAM Text
  └─► NOTAMTextPreprocessor  (lowercase · remove digits · Porter stemming)
        └─► TF-IDF Vectorizer  (5,000 features · n-gram (1,2) · sublinear_tf)
              └─►  ⊕  MetaFeatureExtractor  (7 features · StandardScaler)
                    └─► LinearSVC  (C=best · crammer_singer · calibrated)
                          └─► Category + Confidence Score
    """, language="text")