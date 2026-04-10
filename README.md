# PostuRisk

> **Classifying fallers vs non-fallers from posturography / accelerometer data
> using traditional ML with SHAP explainability.**

---

## 🎯 Project Goal

PostuRisk is a machine-learning pipeline that predicts **fall risk** in elderly
adults using 3D accelerometer signals recorded during daily living and
structured lab-walk assessments.  The project deliberately uses *traditional*
(non-deep-learning) classifiers — Random Forest, Gradient Boosting, SVM,
Logistic Regression — so that every prediction can be **explained** with
[SHAP](https://shap.readthedocs.io/) (SHapley Additive exPlanations).

### Why it matters

Falls are a leading cause of injury-related morbidity in older adults.
Identifying individuals at risk **before** a fall occurs allows for targeted
interventions (physiotherapy, environmental modifications, medication review).
Wearable accelerometers make continuous, objective monitoring feasible — but raw
sensor signals must be transformed into clinically interpretable features.

---

## 📊 Dataset

This project uses the **Long Term Movement Monitoring (LTMM) Database v1.0.0**
from [PhysioNet](https://physionet.org/content/ltmm/1.0.0/):

| Property | Value |
|---|---|
| Subjects | 71 community-living elders (age 65-87) |
| Sensors | 3D accelerometer on lower back |
| Recordings | 3-day free-living + 1-min lab walks |
| Labels | **Fallers** (≥ 2 falls/year) vs **Non-fallers** |
| Clinical scores | DGI, BBS, TUG, FSST, MMSE, ABC |
| Format | MIT/WFDB (`.dat` + `.hea`) + `.xlsx` metadata |
| License | Open Data Commons Attribution License v1.0 |

**Citation:**

> A. Weiss, M. Brozgol, M. Dorfman, T. Herman, S. Shema, N. Giladi,
> J.M. Hausdorff, *"Does the Evaluation of Gait Quality During Daily Life
> Provide Insight Into Fall Risk?"*, Neurorehabil Neural Repair, 2013.
> DOI: [10.1177/1545968313491004](https://doi.org/10.1177/1545968313491004)

---

## 🗂️ Project Structure

```
posturisk/
├── data/
│   ├── raw/            ← Downloaded PhysioNet files (git-ignored)
│   └── processed/      ← Engineered feature tables (git-ignored)
├── notebooks/          ← Exploratory analysis & visualization
├── src/
│   └── posturisk/
│       ├── __init__.py
│       └── fetch_data.py   ← Data download script
├── tests/
│   └── __init__.py
├── .gitignore
├── pyproject.toml
└── README.md
```

---

## 🚀 Quick Start

### 1. Install

```bash
# Clone the repository
git clone https://github.com/<your-org>/posturisk.git
cd posturisk

# Create a virtual environment & install in editable mode
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS / Linux

pip install -e ".[dev,notebooks]"
```

### 2. Download the data

```bash
# Download the full LTMM dataset (~20 GB)
posturisk-fetch

# Or download only the lightweight lab-walk subset + metadata (~50 MB)
posturisk-fetch --lab-walks-only
```

### 3. Run tests

```bash
pytest
```

---

## 🔬 Pipeline Overview

1. **Data acquisition** — `fetch_data.py` pulls the LTMM dataset from
   PhysioNet.
2. **Feature engineering** — Extract gait features (stride time, step
   regularity, RMS acceleration, frequency-domain metrics) from raw
   accelerometer windows.
3. **Modelling** — Train and compare Random Forest, Gradient Boosting, SVM,
   and Logistic Regression with nested cross-validation.
4. **Explainability** — Generate SHAP summary plots, dependence plots, and
   per-subject force plots to interpret model decisions.

---

## 📜 License

This project is licensed under the [MIT License](LICENSE).

The underlying LTMM dataset is distributed under the
[Open Data Commons Attribution License v1.0](https://opendatacommons.org/licenses/by/1-0/).
