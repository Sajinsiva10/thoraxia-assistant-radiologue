"""THORAXIA Streamlit app — version dashboard pro pour la soutenance."""
from __future__ import annotations
import time
from pathlib import Path
import sys

import streamlit as st
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

from src.medgemma_inference import medgemma_predict
from src.guardrails import apply_safety_guardrails


# ============== CACHE MEDGEMMA
@st.cache_resource(show_spinner="⏳ Chargement initial de MedGemma 4B (1 fois au démarrage, ~1 min)...")
def _ensure_model_loaded():
    from src.medgemma_inference import _load_model
    _load_model()
    return True


# ============== CONFIG GLOBALE
st.set_page_config(
    page_title="THORAXIA — Assistant radiologue IA",
    page_icon="🩻",
    layout="wide",
    initial_sidebar_state="expanded",
)

_ensure_model_loaded()

# Charte THORAXIA
NAVY = "#1B2D5B"
ACCENT = "#C8553D"
ICE = "#EAF2F8"
LIGHT = "#F8FAFC"
GREEN = "#2E7D32"
RED = "#C62828"
GRAY_TEXT = "#475569"

# ============== CSS CUSTOM
st.markdown(f"""
<style>
.stApp {{
    background: linear-gradient(180deg, {LIGHT} 0%, #FFFFFF 30%);
}}

.hero-container {{
    background: linear-gradient(135deg, {NAVY} 0%, #2A3F7A 100%);
    border-radius: 16px;
    padding: 1.8rem 2rem;
    margin-bottom: 1.5rem;
    color: white;
    box-shadow: 0 8px 24px rgba(27,45,91,0.18);
}}
.hero-title {{ font-size: 2.6rem; font-weight: 800; margin: 0; letter-spacing: -0.5px; }}
.hero-subtitle {{ font-size: 1.1rem; opacity: 0.85; margin-top: 0.3rem; font-weight: 300; }}
.hero-tagline {{
    display: inline-block;
    background: rgba(200,85,61,0.25);
    border: 1px solid {ACCENT};
    padding: 0.3rem 0.9rem;
    border-radius: 999px;
    font-size: 0.85rem;
    margin-top: 0.8rem;
}}

.metric-card {{
    background: white; border-radius: 12px; padding: 1rem 1.2rem;
    border: 1px solid #E2E8F0; box-shadow: 0 2px 8px rgba(0,0,0,0.04); height: 100%;
}}
.metric-label {{ color: {GRAY_TEXT}; font-size: 0.8rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 0.3rem; }}
.metric-value {{ color: {NAVY}; font-size: 2rem; font-weight: 800; line-height: 1.1; margin: 0; }}
.metric-sub {{ color: {GREEN}; font-size: 0.85rem; font-weight: 600; margin-top: 0.2rem; }}

.warning-banner {{
    background: linear-gradient(90deg, #FFF4ED 0%, #FFE8DC 100%);
    border-left: 5px solid {ACCENT}; color: #7C2D12;
    padding: 1rem 1.3rem; border-radius: 10px; font-weight: 600;
    margin: 1rem 0; display: flex; align-items: center; gap: 0.6rem;
}}

.section-title {{
    color: {NAVY}; font-size: 1.25rem; font-weight: 700;
    margin: 0 0 1rem 0; display: flex; align-items: center; gap: 0.5rem;
}}

.result-badge {{
    border-radius: 14px; padding: 1.8rem; text-align: center; color: white;
    margin: 1rem 0; box-shadow: 0 6px 20px rgba(0,0,0,0.12);
}}
.result-class {{ font-size: 2.2rem; font-weight: 800; margin: 0; letter-spacing: -0.5px; }}
.result-conf {{ font-size: 1.1rem; opacity: 0.92; margin-top: 0.3rem; }}

/* ========== EXPANDERS — lisibilité forcée ========== */
[data-testid="stExpander"] {{
    border: 1px solid #E2E8F0 !important;
    border-radius: 10px !important;
    margin-bottom: 0.5rem !important;
    overflow: hidden !important;
    background: white !important;
}}
[data-testid="stExpander"] details summary {{
    background: {NAVY} !important;
    color: white !important;
    padding: 0.7rem 1rem !important;
    font-weight: 600 !important;
}}
[data-testid="stExpander"] details summary:hover {{
    background: #2A3F7A !important;
}}
[data-testid="stExpander"] details summary p,
[data-testid="stExpander"] details summary span,
[data-testid="stExpander"] details summary strong {{
    color: white !important;
    margin: 0 !important;
}}
[data-testid="stExpander"] details[open] > div {{
    background: white !important;
    padding: 1rem 1.2rem !important;
}}
[data-testid="stExpander"] details > div:not(summary) p,
[data-testid="stExpander"] details > div:not(summary) li,
[data-testid="stExpander"] details > div:not(summary) span,
[data-testid="stExpander"] details > div:not(summary) strong,
[data-testid="stExpander"] details > div:not(summary) {{
    color: #1F2937 !important;
}}

/* ========== JSON RENDER — lisibilité forcée ========== */
[data-testid="stJson"] {{
    background: #F8FAFC !important;
    border-radius: 8px !important;
    padding: 1rem !important;
    border: 1px solid #E2E8F0 !important;
}}
[data-testid="stJson"] * {{
    color: #1F2937 !important;
    background: transparent !important;
}}
[data-testid="stJson"] pre {{
    background: transparent !important;
    color: #1F2937 !important;
}}

/* ========== ÉTATS D'ANALYSE (bannière en cours/terminé) ========== */
.state-banner {{
    width: 100%; padding: 0.7rem 1rem; border-radius: 8px;
    text-align: center; font-weight: 700; font-size: 1rem;
    margin: 0.3rem 0 0.8rem 0; box-shadow: 0 2px 6px rgba(0,0,0,0.08);
}}
.analyzing-state {{
    background: linear-gradient(90deg, #F59E0B, #D97706);
    color: white;
    animation: pulse-bg 2s ease-in-out infinite;
}}
.done-state {{
    background: linear-gradient(90deg, {GREEN}, #43A047);
    color: white;
}}
.dots {{ display: inline-block; }}
.dots::after {{
    content: '';
    animation: dots-anim 1.4s infinite;
}}
@keyframes dots-anim {{
    0% {{ content: '.'; }}
    33% {{ content: '..'; }}
    66% {{ content: '...'; }}
    100% {{ content: '.'; }}
}}
@keyframes pulse-bg {{
    0%, 100% {{ box-shadow: 0 2px 6px rgba(0,0,0,0.08); }}
    50% {{ box-shadow: 0 4px 16px rgba(245,158,11,0.4); }}
}}

.footer {{
    text-align: center; color: {GRAY_TEXT}; font-size: 0.85rem;
    padding: 2rem 0 1rem 0; border-top: 1px solid #E2E8F0; margin-top: 2rem;
}}

#MainMenu, footer {{ visibility: hidden; }}
.stDeployButton {{ display: none; }}
</style>
""", unsafe_allow_html=True)

# ============== SIDEBAR
with st.sidebar:
    st.markdown(f"### 🩻 THORAXIA")
    st.markdown(f"**Assistant radiologue virtuel responsable**")
    st.divider()
    st.markdown("##### 📋 Pipeline d'inférence")
    st.markdown("""
1. Upload radio (PNG/JPG)
2. Préprocessing Pillow
3. **MedGemma 4B IT** (Google)
4. Garde-fous + seuil 0.60
5. Sortie JSON validée
""")
    st.divider()
    st.markdown("##### 👥 Équipe")
    st.markdown("""
- **Souhaib Sghaier** · Chef de projet
- **Kevin Sivaharan** · MOA
- **Sajin Sivasaranam** · MOE
- **Pratheep Parthepan** · Comm
- **Laurent Qiu** · Qualité
- **Yanick Shan** · Documentation
""")
    st.divider()
    st.caption("EFREI Paris · MasterCamp 2026")
    st.caption("Tuteur : Badr Tajini")
    st.caption("Big Data & IA · 2025-2026")

# ============== HERO
st.markdown(f"""
<div class="hero-container">
    <div class="hero-title">🩻 THORAXIA</div>
    <div class="hero-subtitle">Assistant radiologue virtuel responsable — Prototype pédagogique</div>
    <span class="hero-tagline">Classification de radiographies thoraciques · MedGemma 4B</span>
</div>
""", unsafe_allow_html=True)

# ============== MÉTRIQUES
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown(f"""<div class="metric-card">
    <div class="metric-label">Accuracy</div>
    <div class="metric-value">0.80</div>
    <div class="metric-sub">↑ Cible 0.70 atteinte</div>
</div>""", unsafe_allow_html=True)
with col2:
    st.markdown(f"""<div class="metric-card">
    <div class="metric-label">Macro-F1</div>
    <div class="metric-value">0.83</div>
    <div class="metric-sub">↑ Cible 0.68 atteinte</div>
</div>""", unsafe_allow_html=True)
with col3:
    st.markdown(f"""<div class="metric-card">
    <div class="metric-label">JSON valides</div>
    <div class="metric-value">100%</div>
    <div class="metric-sub">↑ Cible 95% atteinte</div>
</div>""", unsafe_allow_html=True)
with col4:
    st.markdown(f"""<div class="metric-card">
    <div class="metric-label">Hallucinations</div>
    <div class="metric-value">0</div>
    <div class="metric-sub">↑ Garde-fous actifs</div>
</div>""", unsafe_allow_html=True)
st.caption("📊 Mesuré sur 50 cas RSNA Pneumonia · Mode improved (prompt + seuil 0.60)")

# ============== WARNING
st.markdown("""
<div class="warning-banner">
    ⚠️ Prototype pédagogique. <strong>Non destiné au diagnostic médical.</strong> Validation par un professionnel qualifié requise.
</div>
""", unsafe_allow_html=True)

# ============== INIT SESSION STATE
for key, val in [
    ("selected_demo", None),
    ("uploaded_image", None),
    ("analyzing", False),
    ("last_pred", None),
    ("last_latency", None),
    ("last_error", None),
    ("last_source", None),
    ("active_demo_idx", None),
]:
    if key not in st.session_state:
        st.session_state[key] = val

# ============== GALERIE D'EXEMPLES
st.markdown(f"""<div class="section-title">🖼️ Exemples — Cliquez pour lancer l'analyse</div>""", unsafe_allow_html=True)

demo_dir = ROOT / "app" / "demo_images"
demo_files = sorted(demo_dir.glob("*.png")) if demo_dir.exists() else []

if demo_files:
    cols = st.columns(len(demo_files))
    for i, demo_file in enumerate(demo_files):
        with cols[i]:
            label = "normal" if "normal" in demo_file.stem else "suspected_opacity"
            st.image(str(demo_file), use_container_width=True)

            # États du bouton selon active_demo_idx + analyzing + last_pred
            is_active = (st.session_state.active_demo_idx == i)
            is_analyzing_this = (is_active and st.session_state.analyzing)
            has_result_this = (
                is_active
                and st.session_state.last_pred is not None
                and not st.session_state.analyzing
            )

            if is_analyzing_this:
                btn_label = "⏳ Analyse en cours..."
                btn_type = "secondary"
                btn_disabled = True
            elif has_result_this:
                btn_label = "✅ Analyse terminée"
                btn_type = "secondary"
                btn_disabled = True
            else:
                btn_label = f"🔬 Tester ({label})"
                btn_type = "primary" if is_active else "secondary"
                btn_disabled = False

            if st.button(btn_label, key=f"demo_{i}",
                         use_container_width=True,
                         type=btn_type,
                         disabled=btn_disabled):
                # Clic = sélection + lancement auto
                st.session_state.selected_demo = demo_file
                st.session_state.uploaded_image = None
                st.session_state.active_demo_idx = i
                st.session_state.analyzing = True
                st.session_state.last_pred = None
                st.session_state.last_error = None
                st.rerun()
else:
    st.info("Aucun exemple disponible. Lance `python scripts/prepare_demo_images.py` pour préparer la galerie.")

st.divider()

# ============== UPLOAD
st.markdown(f"""<div class="section-title">📤 Ou uploadez votre propre radiographie</div>""", unsafe_allow_html=True)

uploaded = st.file_uploader(
    "Formats acceptés : PNG, JPG, JPEG",
    type=["png", "jpg", "jpeg"],
    label_visibility="collapsed",
)
if uploaded is not None:
    if st.session_state.uploaded_image is None or uploaded.name != getattr(st.session_state.uploaded_image, "name", None):
        st.session_state.uploaded_image = uploaded
        st.session_state.selected_demo = None
        st.session_state.active_demo_idx = None
        st.session_state.last_pred = None
        st.session_state.last_error = None
        st.session_state.analyzing = False

# ============== TRAITEMENT
image_to_process = None
source_name = None
is_from_gallery = False

if st.session_state.uploaded_image is not None:
    temp_path = ROOT / "data" / "_temp_upload.png"
    temp_path.parent.mkdir(parents=True, exist_ok=True)
    with open(temp_path, "wb") as f:
        f.write(st.session_state.uploaded_image.getbuffer())
    image_to_process = temp_path
    source_name = st.session_state.uploaded_image.name
elif st.session_state.selected_demo is not None:
    image_to_process = st.session_state.selected_demo
    source_name = st.session_state.selected_demo.name
    is_from_gallery = True

# Auto-reset si source change
current_source = source_name if image_to_process else None
if st.session_state.last_source != current_source and not st.session_state.analyzing:
    if st.session_state.last_pred is not None and image_to_process is not None:
        # On garde le pred si on vient de finir l'inférence
        pass
    st.session_state.last_source = current_source

if image_to_process is not None:
    st.divider()
    col_img, col_res = st.columns([1, 1.3])

    analyzing = st.session_state.analyzing
    has_result = st.session_state.last_pred is not None

    with col_img:
        st.markdown(f"""<div class="section-title">🖼️ Image chargée</div>""", unsafe_allow_html=True)
        image = Image.open(image_to_process)
        st.image(image, caption=source_name, use_container_width=True)
        st.caption(f"📐 Dimensions : {image.size[0]} × {image.size[1]} px · Format : {image.format or 'PNG'}")

        st.markdown("")

        if analyzing:
            st.markdown(
                '<div class="state-banner analyzing-state">⏳ Analyse MedGemma en cours'
                '<span class="dots"></span></div>',
                unsafe_allow_html=True
            )

        elif has_result:
            st.markdown(
                '<div class="state-banner done-state">✅ Analyse terminée</div>',
                unsafe_allow_html=True
            )
            if st.button("🔄 Nouvelle analyse", use_container_width=True, key="btn_reset"):
                st.session_state.last_pred = None
                st.session_state.last_error = None
                st.session_state.analyzing = False
                st.session_state.active_demo_idx = None
                st.session_state.selected_demo = None
                st.session_state.uploaded_image = None
                st.rerun()

        elif not is_from_gallery:
            # IDLE et image vient d'un upload manuel → bouton classique
            if st.button("🚀 Lancer l'analyse MedGemma", type="primary",
                         use_container_width=True, key="btn_launch"):
                st.session_state.analyzing = True
                st.session_state.last_pred = None
                st.session_state.last_error = None
                st.rerun()

    with col_res:
        st.markdown(f"""<div class="section-title">🤖 Analyse MedGemma</div>""", unsafe_allow_html=True)

        if analyzing:
            with st.spinner("⏳ MedGemma examine la radiographie..."):
                t0 = time.perf_counter()
                try:
                    pred = apply_safety_guardrails(medgemma_predict(image_to_process, mode="improved"))
                    latency = round(time.perf_counter() - t0, 1)
                    st.session_state.last_pred = pred
                    st.session_state.last_latency = latency
                except Exception as e:
                    st.session_state.last_error = str(e)[:200]
            st.session_state.analyzing = False
            st.rerun()

        elif has_result:
            pred = st.session_state.last_pred
            latency = st.session_state.last_latency or 0
            pred_class = pred["predicted_class"]
            conf = pred.get("confidence", 0)
            color_map = {"normal": GREEN, "suspected_opacity": ACCENT, "uncertain": "#FFB74D"}
            color = color_map.get(pred_class, NAVY)
            icon = {"normal": "✅", "suspected_opacity": "⚠️", "uncertain": "❓"}.get(pred_class, "🔬")

            st.markdown(f"""
<div class="result-badge" style="background: linear-gradient(135deg, {color} 0%, {color}DD 100%);">
    <div class="result-class">{icon} {pred_class}</div>
    <div class="result-conf">Confiance : {conf:.2f} · Qualité image : {pred.get('image_quality','?')}</div>
    <div class="result-conf" style="font-size:0.85rem;margin-top:0.4rem;">⏱️ Latence : {latency} s</div>
</div>
""", unsafe_allow_html=True)

            with st.expander("👁️ **Observations visuelles** (visual_evidence)", expanded=True):
                evs = pred.get("visual_evidence", [])
                if evs:
                    for ev in evs:
                        st.markdown(f"- {ev}")
                else:
                    st.caption("Aucune observation retournée.")

            with st.expander("📝 **Justification du modèle**"):
                st.markdown(pred.get("justification", "—"))

            with st.expander("⚠️ **Limites identifiées**"):
                lims = pred.get("limitations", [])
                if lims:
                    for lim in lims:
                        st.markdown(f"- {lim}")
                else:
                    st.caption("Aucune limite explicite.")

            with st.expander("🔧 **Métadonnées techniques**"):
                st.markdown(f"- **Modèle** : `{pred.get('model_name','?')}`")
                st.markdown(f"- **Version prompt** : `{pred.get('prompt_version','?')}`")
                st.markdown(f"- **Latence mesurée** : {latency} s")
                st.markdown(f"- **Classe prédite** : `{pred_class}`")
                st.markdown(f"- **Confiance** : {conf:.4f}")

            with st.expander("📋 **Sortie JSON complète**"):
                st.json(pred)

        elif st.session_state.last_error:
            st.error(f"❌ Erreur : {st.session_state.last_error}")

        else:
            st.info("👆 Choisissez un exemple dans la galerie ou uploadez une radiographie.")

    if has_result:
        st.markdown("""
<div class="warning-banner">
    ⚠️ Ce résultat est expérimental. <strong>Il ne constitue PAS un diagnostic médical.</strong> Pour toute interprétation clinique, consultez un radiologue qualifié.
</div>
""", unsafe_allow_html=True)

# ============== FOOTER
st.markdown(f"""
<div class="footer">
    <strong>THORAXIA</strong> · EFREI MasterCamp 2026 · Filière Big Data & IA<br>
    Backbone : <code>google/medgemma-4b-it</code> · Tuteur : Badr Tajini<br>
    <em>Architecture en couches : Prompt amélioré → MedGemma → Garde-fous</em>
</div>
""", unsafe_allow_html=True)