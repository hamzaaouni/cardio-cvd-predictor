<div align="center">

# 🫀 Cardio CVD Risk Predictor

### Interactive clinical decision-support tool for cardiovascular-disease risk

*Enter a patient's routine clinical data and get a calibrated CVD-risk score with a per-patient **SHAP** explanation. Powered by an Optuna-tuned **XGBoost** model (ROC-AUC 0.802).*

[![Streamlit](https://img.shields.io/badge/Streamlit-app-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Model](https://img.shields.io/badge/XGBoost-ROC--AUC_0.802-success)](#-model)
[![License](https://img.shields.io/badge/License-Academic_(PFE_2026)-blue)](#-author)

**[🚀 Deploy on Streamlit Cloud](https://share.streamlit.io/deploy?repository=hamzaaouni/cardio-cvd-predictor&branch=main&mainModule=app.py)** · **[📊 Analytics Dashboard](https://cardio-decision-ml.vercel.app/)** · **[📄 Project & Thesis](https://github.com/hamzaaouni/pfe)**

</div>

---

This is the **standalone, deploy-ready** Streamlit predictor extracted from the [CardioDecision-ML](https://github.com/hamzaaouni/pfe) project. It is self-contained: the trained model and all assets live in `outputs/`, so it deploys on Streamlit Community Cloud with one click.

## ✨ Features

- **Real-time risk prediction** from 11 clinical inputs (auto-engineered into 16 features: BMI, pulse pressure, MAP, clinical categories).
- **Risk banding** — 🟢 LOW (< 30 %) · 🟡 MODERATE (30–65 %) · 🔴 HIGH (≥ 65 %).
- **Per-patient SHAP explanation** — which factors push the risk up or down.
- **Model benchmarking & EDA** tabs with the project's result figures.

## 🎬 Demo scenarios (shareable links)

Once deployed, append `?scenario=` to the URL to pre-load a patient and show the result:

| Link | Patient | Result |
| :--- | :--- | :---: |
| `…/?scenario=low` | healthy 35 y/o | 🟢 **7.1 %** |
| `…/?scenario=moderate` | 54 y/o, elevated BP | 🟡 **64.5 %** |
| `…/?scenario=high` | 62 y/o, multiple risk factors | 🔴 **81.0 %** |

## 🚀 Run locally

```bash
pip install -r requirements.txt
streamlit run app.py        # http://localhost:8501
```

## ☁️ Deploy (Streamlit Community Cloud)

1. Open **[share.streamlit.io](https://share.streamlit.io)** and sign in with GitHub.
2. **Create app** → Repository `hamzaaouni/cardio-cvd-predictor`, Branch `main`, **Main file `app.py`**.
3. **Deploy.** (One-click link: the *Deploy* badge above.)

No environment variables or secrets are required.

## 🧠 Model

`XGBoost_Tuned` (Optuna, 50 trials) — ROC-AUC **0.802**, accuracy 73.3 %, recall **90.8 %** at the recall-optimised operating threshold 0.27. Top predictors (SHAP): systolic BP · age · cholesterol · mean arterial pressure · BMI.

## 🗂️ Structure

```text
app.py                 # the Streamlit predictor
requirements.txt       # dependencies (Streamlit Cloud installs these)
outputs/
  models/              # best_model.pkl (XGBoost) + name
  data/                # feature_cols.json, optimal_threshold.txt
  results/             # benchmark_results.csv
  plots/               # benchmark & EDA & SHAP figures
```

## 🔗 Related

- **Analytics dashboard (Next.js):** [cardio-decision-ml.vercel.app](https://cardio-decision-ml.vercel.app/) · [repo](https://github.com/hamzaaouni/CardioDecision-ML)
- **Full ML pipeline, notebook & thesis:** [github.com/hamzaaouni/pfe](https://github.com/hamzaaouni/pfe)

## 👤 Author

**AOUNI Hamza** — Master 2, Big Data & Artificial Intelligence · Supervisor **Pr. Jilali ANTARI** · Multidisciplinary Faculty of Taroudant, Ibn Zohr University — **PFE 2026**

> ⚕️ For research and educational purposes only — not a substitute for professional clinical judgment.
