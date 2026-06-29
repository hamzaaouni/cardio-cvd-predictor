# ============================================================
# STREAMLIT APP — CVD Prediction Dashboard
# PFE: Early Prediction of Chronic Diseases
# ============================================================
# Run with: streamlit run app.py
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import joblib
import os
import warnings

warnings.filterwarnings('ignore')

# ── PATHS (resolved from this file, so the app works regardless of the
#    directory it is launched from) ───────────────────────────
from pathlib import Path
import json
OUTPUTS = Path(__file__).resolve().parent / "outputs"
MODELS  = OUTPUTS / "models"
PLOTS   = OUTPUTS / "plots"

# ── PAGE CONFIGURATION ──────────────────────────────────────
st.set_page_config(
    page_title="CVD Risk Predictor",
    page_icon=":material/cardiology:",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CUSTOM CSS ──────────────────────────────────────────────
st.markdown("""
<style>
    /* Main background */
    .main { background-color: #F8FAFC; }

    /* Header card */
    .header-card {
        background: linear-gradient(135deg, #1a237e 0%, #283593 50%, #3949ab 100%);
        padding: 2rem 2.5rem;
        border-radius: 16px;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(26, 35, 126, 0.3);
    }
    .header-card h1 { color: white; font-size: 2rem; margin: 0; }
    .header-card p  { color: #C5CAE9; font-size: 1rem; margin: 0.5rem 0 0; }

    /* Metric cards */
    .metric-card {
        background: white;
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        box-shadow: 0 2px 12px rgba(0,0,0,0.06);
        border-left: 4px solid #3949ab;
        margin-bottom: 1rem;
    }
    .metric-card .label { color: #64748b; font-size: 0.8rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; }
    .metric-card .value { color: #1e293b; font-size: 1.6rem; font-weight: 700; margin-top: 0.2rem; }

    /* Risk badge */
    .risk-low    { background: #DCFCE7; color: #166534; padding: 0.4rem 1.2rem; border-radius: 999px; font-weight: 700; display: inline-block; }
    .risk-medium { background: #FEF9C3; color: #854D0E; padding: 0.4rem 1.2rem; border-radius: 999px; font-weight: 700; display: inline-block; }
    .risk-high   { background: #FEE2E2; color: #991B1B; padding: 0.4rem 1.2rem; border-radius: 999px; font-weight: 700; display: inline-block; }

    /* Section headings */
    .section-title {
        display: flex;
        align-items: center;
        gap: 0.55rem;
        font-size: 1.1rem;
        font-weight: 700;
        color: #1e293b;
        margin: 1.5rem 0 0.8rem;
        padding-bottom: 0.4rem;
        border-bottom: 2px solid #e2e8f0;
    }
    .section-title svg { color: #3949ab; flex: 0 0 auto; }
    /* Disclaimer */
    .disclaimer {
        background: #EFF6FF;
        border: 1px solid #BFDBFE;
        border-radius: 8px;
        padding: 0.8rem 1rem;
        color: #1e40af;
        font-size: 0.82rem;
        margin-top: 1.5rem;
    }
</style>
""", unsafe_allow_html=True)


# ── Icons: Lucide inline SVG (crisp, offline-safe, shadcn-style) ──
_LUCIDE = {
    "heart-pulse": '<path d="M19 14c1.49-1.46 3-3.21 3-5.5A5.5 5.5 0 0 0 16.5 3c-1.76 0-3 .5-4.5 2-1.5-1.5-2.74-2-4.5-2A5.5 5.5 0 0 0 2 8.5c0 2.3 1.5 4.05 3 5.5l7 7Z"/><path d="M3.22 12H9.5l.5-1 2 4.5 2-7 1.5 3.5h5.27"/>',
    "stethoscope": '<path d="M11 2v2"/><path d="M5 2v2"/><path d="M5 3H4a2 2 0 0 0-2 2v4a6 6 0 0 0 12 0V5a2 2 0 0 0-2-2h-1"/><path d="M8 15a6 6 0 0 0 12 0v-3"/><circle cx="20" cy="10" r="2"/>',
    "clipboard-list": '<rect width="8" height="4" x="8" y="2" rx="1" ry="1"/><path d="M16 4h2a2 2 0 0 1 2 2v14a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2V6a2 2 0 0 1 2-2h2"/><path d="M12 11h4"/><path d="M12 16h4"/><path d="M8 11h.01"/><path d="M8 16h.01"/>',
    "target": '<circle cx="12" cy="12" r="10"/><circle cx="12" cy="12" r="6"/><circle cx="12" cy="12" r="2"/>',
    "search": '<circle cx="11" cy="11" r="8"/><path d="m21 21-4.3-4.3"/>',
    "bar-chart": '<path d="M3 3v18h18"/><path d="M18 17V9"/><path d="M13 17V5"/><path d="M8 17v-3"/>',
    "line-chart": '<path d="M3 3v18h18"/><path d="m19 9-5 5-4-4-3 3"/>',
    "check-circle": '<path d="M21.801 10A10 10 0 1 1 17 3.335"/><path d="m9 11 3 3L22 4"/>',
    "alert-triangle": '<path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3Z"/><path d="M12 9v4"/><path d="M12 17h.01"/>',
    "octagon-alert": '<path d="M12 16h.01"/><path d="M12 8v4"/><path d="M15.312 2a2 2 0 0 1 1.414.586l4.688 4.688A2 2 0 0 1 22 8.688v6.624a2 2 0 0 1-.586 1.414l-4.688 4.688a2 2 0 0 1-1.414.586H8.688a2 2 0 0 1-1.414-.586l-4.688-4.688A2 2 0 0 1 2 15.312V8.688a2 2 0 0 1 .586-1.414l4.688-4.688A2 2 0 0 1 8.688 2z"/>',
}
def ic(name, size=18):
    return ('<svg xmlns="http://www.w3.org/2000/svg" width="%d" height="%d" viewBox="0 0 24 24" '
            'fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" '
            'stroke-linejoin="round" style="vertical-align:-0.18em">%s</svg>' % (size, size, _LUCIDE[name]))
def section(name, text):
    st.markdown('<p class="section-title">%s<span>%s</span></p>' % (ic(name, 20), text),
                unsafe_allow_html=True)


# ── LOAD MODEL & DATA ───────────────────────────────────────
@st.cache_resource
def load_model():
    model = joblib.load(MODELS / "best_model.pkl")
    with open(MODELS / "best_model_name.txt") as f:
        name = f.read().strip()
    return model, name

@st.cache_data
def load_feature_cols():
    with open(OUTPUTS / "data" / "feature_cols.json") as f:
        return json.load(f)

@st.cache_data
def load_results():
    df = pd.read_csv(OUTPUTS / "results" / "benchmark_results.csv")
    return df

@st.cache_resource
def load_shap_explainer(_model):
    # leading underscore: tells Streamlit not to hash the (unhashable) model.
    # shap is imported lazily here so the cold-start / wake does not pay the heavy
    # shap -> numba -> llvmlite import cost until an explanation is actually needed.
    import shap
    return shap.TreeExplainer(_model)

try:
    model, model_name = load_model()
    feature_cols      = load_feature_cols()
    results_df        = load_results()
    MODEL_LOADED = True
except Exception as e:
    MODEL_LOADED = False
    st.error(f"Could not load model. Please run step1→step4 first.\nError: {e}")


# ── HEADER ──────────────────────────────────────────────────
st.markdown(f"""
<div class="header-card">
    <h1>{ic('heart-pulse', 30)} Cardiovascular Disease Risk Predictor</h1>
    <p>PFE Project — Early Prediction of Chronic Diseases Using Big Data & Machine Learning</p>
</div>
""", unsafe_allow_html=True)


# ── Demo scenarios — shareable via ?scenario=low|moderate|high ──
_SCENARIOS = {
    "low":      dict(age=35, gender="Female", height=168, weight=60,  ap_hi=110, ap_lo=70,  cholesterol=1, gluc=1, smoke=False, alco=False, active=True),
    "moderate": dict(age=54, gender="Male",   height=174, weight=88,  ap_hi=138, ap_lo=88,  cholesterol=2, gluc=2, smoke=False, alco=False, active=True),
    "high":     dict(age=62, gender="Male",   height=170, weight=95,  ap_hi=165, ap_lo=100, cholesterol=3, gluc=3, smoke=True,  alco=True,  active=False),
}
_p = _SCENARIOS.get(st.query_params.get("scenario", "").lower(), {})


# ── SIDEBAR: PATIENT INPUT FORM ─────────────────────────────
with st.sidebar:
    section("stethoscope", "Patient Information")
    st.markdown("Enter the patient's medical data below:")

    st.markdown("### Demographics")
    age   = st.slider("Age (years)",   min_value=18, max_value=90, value=_p.get("age", 50), step=1)
    gender = st.radio("Gender", ["Female", "Male"], index=(1 if _p.get("gender") == "Male" else 0), horizontal=True)
    height = st.slider("Height (cm)",  min_value=140, max_value=220, value=_p.get("height", 170), step=1)
    weight = st.slider("Weight (kg)",  min_value=40,  max_value=180, value=_p.get("weight", 75),  step=1)

    st.markdown("### Blood Pressure")
    ap_hi = st.slider("Systolic BP (mmHg)",  min_value=80,  max_value=220, value=_p.get("ap_hi", 120), step=1)
    ap_lo = st.slider("Diastolic BP (mmHg)", min_value=40,  max_value=140, value=_p.get("ap_lo", 80),  step=1)

    st.markdown("### Lab Results")
    cholesterol = st.selectbox("Cholesterol Level",
                               options=[1, 2, 3],
                               index=[1, 2, 3].index(_p.get("cholesterol", 1)),
                               format_func=lambda x: {1: "Normal", 2: "Above Normal", 3: "Well Above Normal"}[x])
    gluc        = st.selectbox("Glucose Level",
                               options=[1, 2, 3],
                               index=[1, 2, 3].index(_p.get("gluc", 1)),
                               format_func=lambda x: {1: "Normal", 2: "Above Normal", 3: "Well Above Normal"}[x])

    st.markdown("### Lifestyle")
    smoke  = st.toggle("Smoker",            value=_p.get("smoke", False))
    alco   = st.toggle("Alcohol Drinker",   value=_p.get("alco", False))
    active = st.toggle("Physically Active", value=_p.get("active", True))

    predict_btn = st.button("Predict Risk", icon=":material/bolt:", type="primary", use_container_width=True)


# ── COMPUTE FEATURES ────────────────────────────────────────
def build_features(age, gender, height, weight, ap_hi, ap_lo,
                   cholesterol, gluc, smoke, alco, active):
    bmi            = round(weight / ((height / 100) ** 2), 2)
    pulse_pressure = ap_hi - ap_lo
    map_val        = round(ap_lo + (pulse_pressure / 3), 2)  # mean arterial pressure
    gender_val     = 2 if gender == "Male" else 1

    bmi_cat = 0
    if bmi < 18.5:   bmi_cat = 0
    elif bmi < 25:   bmi_cat = 1
    elif bmi < 30:   bmi_cat = 2
    else:            bmi_cat = 3

    bp_cat = 0
    if ap_hi < 120 and ap_lo < 80:         bp_cat = 0
    elif ap_hi < 130 and ap_lo < 80:       bp_cat = 1
    elif ap_hi < 140 or ap_lo < 90:        bp_cat = 2
    else:                                  bp_cat = 3

    return pd.DataFrame([{
        "age_years":      age,
        "gender":         gender_val,
        "height":         height,
        "weight":         weight,
        "ap_hi":          ap_hi,
        "ap_lo":          ap_lo,
        "cholesterol":    cholesterol,
        "gluc":           gluc,
        "smoke":          int(smoke),
        "alco":           int(alco),
        "active":         int(active),
        "bmi":            bmi,
        "pulse_pressure": pulse_pressure,
        "bmi_category":   bmi_cat,
        "bp_category":    bp_cat,
        "map":            map_val,
    }])


# ── MAIN CONTENT TABS ───────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    ":material/insights: Prediction",
    ":material/bar_chart: Model Benchmarking",
    ":material/show_chart: EDA Plots",
    ":material/info: About"
])


# ════════════════════════════════════════════════════════════
# TAB 1 — PREDICTION
# ════════════════════════════════════════════════════════════
with tab1:
    if not MODEL_LOADED:
        st.warning("Please run the training pipeline first (step1→step4).")
    elif predict_btn or True:   # show empty state on load

        features_df = build_features(
            age, gender, height, weight, ap_hi, ap_lo,
            cholesterol, gluc, smoke, alco, active
        )
        # enforce the exact column order the model was trained on
        features_df = features_df[feature_cols]

        bmi = features_df["bmi"].values[0]
        pulse_pressure = features_df["pulse_pressure"].values[0]

        # ── Computed stats ──────────────────────────────────
        section("clipboard-list", "Computed Patient Metrics")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(f'<div class="metric-card"><div class="label">BMI</div><div class="value">{bmi:.1f}</div></div>', unsafe_allow_html=True)
        with c2:
            st.markdown(f'<div class="metric-card"><div class="label">Pulse Pressure</div><div class="value">{pulse_pressure} mmHg</div></div>', unsafe_allow_html=True)
        with c3:
            bp_labels = {0: "Normal", 1: "Elevated", 2: "High Stage 1", 3: "High Stage 2"}
            bp_val    = features_df["bp_category"].values[0]
            st.markdown(f'<div class="metric-card"><div class="label">BP Category</div><div class="value">{bp_labels[bp_val]}</div></div>', unsafe_allow_html=True)
        with c4:
            bmi_labels = {0: "Underweight", 1: "Normal", 2: "Overweight", 3: "Obese"}
            bmi_val    = features_df["bmi_category"].values[0]
            st.markdown(f'<div class="metric-card"><div class="label">BMI Category</div><div class="value">{bmi_labels[bmi_val]}</div></div>', unsafe_allow_html=True)

        # ── Prediction ──────────────────────────────────────
        if predict_btn or _p:
            proba     = float(model.predict_proba(features_df)[0][1])
            risk_pct  = round(proba * 100, 1)

            if risk_pct < 30:
                risk_level = "LOW"
                risk_class = "risk-low"
                risk_emoji = ic("check-circle", 16)
                rec = "Continue healthy lifestyle. Maintain regular check-ups."
            elif risk_pct < 65:
                risk_level = "MODERATE"
                risk_class = "risk-medium"
                risk_emoji = ic("alert-triangle", 16)
                rec = "Monitor blood pressure and cholesterol. Consider lifestyle changes."
            else:
                risk_level = "HIGH"
                risk_class = "risk-high"
                risk_emoji = ic("octagon-alert", 16)
                rec = "Immediate medical consultation recommended. Review risk factors urgently."

            section("target", "Prediction Result")

            col_res, col_gauge = st.columns([1, 1])

            with col_res:
                st.markdown(f"""
                <div style="background:white; border-radius:16px; padding:2rem; box-shadow:0 4px 20px rgba(0,0,0,0.08); text-align:center;">
                    <div style="font-size:0.85rem; color:#64748b; text-transform:uppercase; font-weight:600; letter-spacing:0.05em;">CVD Risk Probability</div>
                    <div style="font-size:4rem; font-weight:900; color:{'#166534' if risk_level=='LOW' else '#854D0E' if risk_level=='MODERATE' else '#991B1B'}; margin:0.5rem 0;">{risk_pct:.1f}%</div>
                    <span class="{risk_class}">{risk_emoji} {risk_level} RISK</span>
                    <div style="margin-top:1.2rem; padding:0.8rem; background:#F8FAFC; border-radius:8px; font-size:0.9rem; color:#374151;">{rec}</div>
                </div>
                """, unsafe_allow_html=True)

            with col_gauge:
                # Simple horizontal progress bar visualization
                fig, ax = plt.subplots(figsize=(6, 3))
                ax.set_facecolor("#F8FAFC")
                fig.patch.set_facecolor("#F8FAFC")

                # Background bar
                ax.barh(0, 100, height=0.4, color="#E2E8F0", zorder=1)
                # Risk bar
                bar_color = "#4CAF50" if risk_pct < 30 else "#FF9800" if risk_pct < 65 else "#F44336"
                ax.barh(0, risk_pct, height=0.4, color=bar_color, zorder=2)

                ax.set_xlim(0, 100)
                ax.set_ylim(-0.5, 0.5)
                ax.set_xlabel("CVD Risk (%)", fontsize=11)
                ax.set_yticks([])
                ax.spines[["top", "right", "left"]].set_visible(False)
                ax.axvline(30, color="gray", linestyle="--", alpha=0.5, linewidth=0.8)
                ax.axvline(65, color="gray", linestyle="--", alpha=0.5, linewidth=0.8)
                ax.text(15, 0.35, "Low",      ha="center", fontsize=9, color="gray")
                ax.text(47, 0.35, "Moderate", ha="center", fontsize=9, color="gray")
                ax.text(82, 0.35, "High",     ha="center", fontsize=9, color="gray")
                ax.set_title(f"Risk Score: {risk_pct:.1f}%", fontweight="bold", fontsize=12)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()

            # ── SHAP local explanation ───────────────────────
            section("search", "Why This Prediction? (SHAP Explanation)")

            try:
                explainer = load_shap_explainer(model)   # built on first use, then cached
                shap_vals = explainer.shap_values(features_df)
                if isinstance(shap_vals, list):
                    sv = shap_vals[1][0]
                else:
                    sv = shap_vals[0]

                feature_labels = {
                    "age_years": "Age", "gender": "Gender", "height": "Height",
                    "weight": "Weight", "ap_hi": "Systolic BP", "ap_lo": "Diastolic BP",
                    "cholesterol": "Cholesterol", "gluc": "Glucose", "smoke": "Smoking",
                    "alco": "Alcohol", "active": "Physical Activity", "bmi": "BMI",
                    "pulse_pressure": "Pulse Pressure", "bmi_category": "BMI Category",
                    "bp_category": "BP Category", "map": "Mean Arterial Pressure"
                }

                feat_names = [feature_labels.get(c, c) for c in features_df.columns]
                sorted_idx = np.argsort(np.abs(sv))[::-1][:8]
                top_names  = [feat_names[i] for i in sorted_idx]
                top_shap   = [sv[i] for i in sorted_idx]
                bar_colors = ["#F44336" if v > 0 else "#4CAF50" for v in top_shap]

                fig, ax = plt.subplots(figsize=(9, 4))
                fig.patch.set_facecolor("white")
                ax.set_facecolor("white")
                bars = ax.barh(top_names[::-1], top_shap[::-1],
                               color=bar_colors[::-1], edgecolor="white", height=0.6)
                ax.axvline(0, color="#334155", linewidth=1)
                ax.set_xlabel("SHAP Value (impact on CVD risk prediction)", fontsize=10)
                ax.set_title("Top Contributing Factors to This Prediction", fontsize=11, fontweight="bold")
                ax.grid(axis="x", linestyle="--", alpha=0.4)
                ax.spines[["top", "right"]].set_visible(False)

                red_patch   = mpatches.Patch(color="#F44336", label="↑ Increases CVD risk")
                green_patch = mpatches.Patch(color="#4CAF50", label="↓ Decreases CVD risk")
                ax.legend(handles=[red_patch, green_patch], fontsize=9, loc="lower right")
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()

            except Exception as e:
                st.info(f"SHAP explanation not available: {e}")

            # Disclaimer
            st.markdown("""
            <div class="disclaimer">
                ⚕️ <strong>Medical Disclaimer:</strong> This tool is designed for research and educational purposes only.
                It is not a medical diagnostic tool and should not replace professional clinical judgment.
                Always consult a qualified healthcare professional for medical decisions.
            </div>
            """, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════
# TAB 2 — BENCHMARK RESULTS
# ════════════════════════════════════════════════════════════
with tab2:
    section("bar-chart", "Model Benchmarking Results")

    if MODEL_LOADED:
        st.markdown(f"**Best Model: {model_name}** 🏆")

        # Styled dataframe
        st.dataframe(
            results_df.style
                .highlight_max(subset=["Accuracy", "Recall", "F1", "ROC_AUC"],
                               color="#D1FAE5")
                .format({
                    "Accuracy":     "{:.2%}",
                    "Precision":    "{:.2%}",
                    "Recall":       "{:.2%}",
                    "Specificity":  "{:.2%}",
                    "F1":           "{:.2%}",
                    "ROC_AUC":      "{:.4f}",
                    "PR_AUC":       "{:.4f}",
                    "MCC":          "{:.4f}",
                    "Train_Time_s": "{:.2f}s",
                }),
            use_container_width=True, height=320
        )

        # Show pre-generated charts if available
        col1, col2 = st.columns(2)
        for path, col, caption in [
            (PLOTS / "benchmark_comparison.png", col1, "All Metrics Comparison"),
            (PLOTS / "roc_curves.png",           col2, "ROC Curves"),
            (PLOTS / "confusion_matrix_best.png", col1, f"Confusion Matrix — {model_name}"),
            (PLOTS / "cross_validation.png",      col2, "5-Fold Cross-Validation"),
        ]:
            if path.exists():
                col.image(str(path), caption=caption, use_container_width=True)


# ════════════════════════════════════════════════════════════
# TAB 3 — EDA PLOTS
# ════════════════════════════════════════════════════════════
with tab3:
    section("line-chart", "Exploratory Data Analysis")

    plot_files = {
        "Feature Distributions":          PLOTS / "feature_distributions.png",
        "Bivariate Analysis":              PLOTS / "bivariate_analysis.png",
        "Correlation Heatmap":             PLOTS / "correlation_heatmap.png",
        "Categorical Features vs Target":  PLOTS / "categorical_vs_target.png",
        "Age Distribution by Disease":     PLOTS / "age_distribution.png",
        "Blood Pressure Scatter":          PLOTS / "bp_scatter.png",
    }

    for caption, path in plot_files.items():
        if path.exists():
            st.image(str(path), caption=caption, use_container_width=True)
        else:
            st.info(f"Run step2_eda.py to generate: {caption}")

    section("search", "SHAP Global Explanations")
    shap_files = {
        "SHAP Feature Importance (Bar)":    PLOTS / "shap_importance_bar.png",
        "SHAP Summary (Beeswarm)":          PLOTS / "shap_summary.png",
        "SHAP Dependence — Top Feature":    PLOTS / "shap_dependence_top.png",
        "SHAP Local — High-Risk Patient":   PLOTS / "shap_local_explanation.png",
    }
    col1, col2 = st.columns(2)
    for i, (caption, path) in enumerate(shap_files.items()):
        col = col1 if i % 2 == 0 else col2
        if path.exists():
            col.image(str(path), caption=caption, use_container_width=True)
        else:
            col.info(f"Run step4_explainability.py to generate: {caption}")


# ════════════════════════════════════════════════════════════
# TAB 4 — ABOUT
# ════════════════════════════════════════════════════════════
with tab4:
    st.markdown("""
    ## :material/menu_book: About This Project

    **Title:** Prédiction précoce des maladies chroniques à partir de données massives de santé

    **Dataset:** Kaggle Cardiovascular Disease Dataset — 70,000 patient records

    ---

    ### :material/biotech: Models Evaluated
    | Model | Type |
    |-------|------|
    | Logistic Regression | Classical |
    | Decision Tree | Classical |
    | Random Forest | Ensemble |
    | SVM | Classical |
    | KNN | Classical |
    | AdaBoost | Boosting |
    | XGBoost | Boosting |
    | LightGBM | Boosting |

    ---

    ### :material/account_tree: Pipeline Overview
    1. **Preprocessing** — Outlier removal, missing values, scaling, SMOTE
    2. **EDA** — Distributions, correlations, feature vs target analysis
    3. **Training** — 8 models with cross-validation and hyperparameter tuning
    4. **Explainability** — SHAP global and local explanations
    5. **Deployment** — This Streamlit app

    ---

    ### :material/build: Technologies
    `Python` · `scikit-learn` · `XGBoost` · `LightGBM` · `SHAP` · `Streamlit` · `Pandas` · `Matplotlib` · `Seaborn`

    ---

    ### :material/warning: Disclaimer
    This application is developed for **academic research purposes only** as part of a PFE project.
    It is not intended for clinical use. Medical decisions must always be made by qualified healthcare professionals.
    """)
