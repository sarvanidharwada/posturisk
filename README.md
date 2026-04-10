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

The project is structured incrementally natively through modular `.py` scripts and Jupyter Notebooks natively aligned exactly to each stage:

1. **Stage 1: Data Acquisition (`fetch_data.py`)** 
   * Pulls the LTMM dataset from PhysioNet. Uses local Python execution parsing `RECORDS` to download `.dat`/`.hea` waveform components plus clinical meta-files.
2. **Stage 2 & 3: Preprocessing & Feature Engineering (`preprocess.py`, `features.py`)** 
   * A custom, dependency-free WFDB (`.hea` and `.dat`) parser extracts raw 16-bit physical signals.
   * Cleans clinical metadata (median imputations and missing logic drops). 
   * Engineers **129** distinct gait features (jerk profiles, spectral bandwidth distribution, and postural sway planar approximations).
   * Runs natively using `python -m posturisk.preprocess`.
3. **Stage 4: Modelling (Pending)** 
   * Train and compare Random Forest, Gradient Boosting, SVM, and Logistic Regression with nested cross-validation.
4. **Stage 5: Explainability (Pending)** 
   * Generate SHAP summary plots, dependence plots, and per-subject force plots to interpret model decisions.

### 📓 Notebooks
* `notebooks/01_eda.ipynb` — Explores the clinical demographics, visualizes the raw waveform signatures for Fallers vs. Non-Fallers.
* `notebooks/02_features.ipynb` — Interactive visual deep dives into advanced feature correlations and engineered differences.

---

## 📜 License

This project is licensed under the [MIT License](LICENSE).

The underlying LTMM dataset is distributed under the
[Open Data Commons Attribution License v1.0](https://opendatacommons.org/licenses/by/1-0/).
