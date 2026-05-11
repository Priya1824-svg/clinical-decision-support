# clinical-decision-support
readme = """
# 🏥 AI-Powered Clinical Decision Support System

> Predicting Diabetes Risk with Explainable AI (XGBoost + SHAP)

![Python](https://img.shields.io/badge/Python-3.8+-blue)
![XGBoost](https://img.shields.io/badge/Model-XGBoost-green)
![Streamlit](https://img.shields.io/badge/UI-Streamlit-red)
![SHAP](https://img.shields.io/badge/Explainability-SHAP-orange)

---

## 📌 Problem Statement
Doctors often lack AI-assisted tools to quickly assess diabetes risk from
patient vitals. This project builds an explainable ML system that predicts
diabetes risk and tells the doctor **why** — not just what.

---

## 🎯 Key Features
- ✅ Predicts diabetes risk with ~80% accuracy
- ✅ ROC-AUC score of ~0.86
- ✅ SHAP-based explanation for every prediction
- ✅ Doctor-friendly Streamlit dashboard
- ✅ Works on real-world Pima Indians Diabetes Dataset

---

## 🧠 Tech Stack
| Layer | Tools |
|-------|-------|
| Language | Python 3.8+ |
| ML Model | XGBoost |
| Explainability | SHAP |
| UI | Streamlit |
| Data | Pima Indians Diabetes Dataset |
| Libraries | Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn |

---

## 📊 Model Performance
| Metric | Score |
|--------|-------|
| Accuracy | ~80% |
| ROC-AUC | ~0.86 |
| Precision | ~78% |
| Recall | ~72% |

---

## 🔬 How Explainability Works
This project uses **SHAP (SHapley Additive Explanations)** to explain
every prediction made by the model.

- 🔴 Red bars = features pushing toward HIGH RISK
- 🔵 Blue bars = features pushing toward LOW RISK
- Doctors can see exactly which vitals triggered the alert

---

## 🚀 How to Run Locally
```bash
