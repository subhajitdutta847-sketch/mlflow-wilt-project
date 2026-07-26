# Wilt Classification — MLflow Full Lifecycle Project

A classic ML project demonstrating the full model lifecycle using **MLflow**: training, experiment tracking, model registration, serving, and automated retraining/promotion via GitHub Actions.

## Problem

Predict whether a tree is diseased ("wilt") or healthy, using 5 features derived from satellite (Quickbird) imagery:
- `GLCM_pan`, `Mean_Green`, `Mean_Red`, `Mean_NIR`, `SD_pan`

Binary classification, highly imbalanced dataset (74 diseased vs 4,265 healthy trees in training data).

**Dataset:** [UCI Wilt Dataset](https://archive.ics.uci.edu/dataset/285/wilt)

## Why Recall, Not Accuracy

Because diseased trees are rare (~1.5% of data), a model that always predicts "healthy" would score ~93% accuracy while catching zero diseased trees. We optimize for **recall on the diseased class** — missing a diseased tree is costlier than a false alarm.

## Project Structure

```
├── .github/workflows/
│   └── retrain.yml            # CI: auto train, register, retrain, promote
├── data/
│   ├── training.csv
│   └── testing.csv
├── train.py                   # Baseline: Logistic Regression
├── train_random_forest.py     # Random Forest
├── train_xgboost.py           # XGBoost
├── register_model.py          # Registers best model, sets "production" alias
├── retrain.py                 # Trains improved model, auto-promotes if better
├── mlruns/                    # MLflow tracking data (committed for CI portability)
└── requirements.txt
```

## Setup

```bash
python -m venv venv
source venv/Scripts/activate   # or venv/bin/activate on Mac/Linux
pip install mlflow scikit-learn pandas xgboost
export MLFLOW_ALLOW_FILE_STORE=true
```

## Step 1 — Model Comparison

Three algorithms were trained and tracked in MLflow:

| Model | Recall (diseased) | F1 (diseased) |
|---|---|---|
| **Logistic Regression** | **0.82** | 0.75 |
| Random Forest | 0.58 | 0.72 |
| XGBoost | 0.69 | 0.77 |

```bash
python train.py
python train_random_forest.py
python train_xgboost.py
mlflow ui   # view results at http://127.0.0.1:5000
```

**Chosen model: Logistic Regression** — highest recall on the diseased class, the priority metric for this problem.

## Step 2 — Register + Serve

```bash
python register_model.py
mlflow models serve -m "models:/wilt-model@production" -p 5001 --no-conda
```

Test inference:
```bash
curl -X POST http://127.0.0.1:5001/invocations \
  -H "Content-Type: application/json" \
  -d '{"dataframe_split": {"columns": ["GLCM_pan","Mean_Green","Mean_Red","Mean_NIR","SD_pan"], "data": [[120.36, 205.50, 119.39, 416.58, 20.68]]}}'
```

## Step 3 — Automated Improvement + Replacement

`retrain.py` trains an improved version (adds feature scaling via `StandardScaler`), compares its recall against the current production model, and **automatically promotes** it only if it performs better — no manual step.

| Model | Recall (diseased) |
|---|---|
| v1 — Logistic Regression | 0.8182 |
| v2 — Logistic Regression + Scaling | **0.8877** ✅ Promoted |

```bash
python retrain.py
```

## CI/CD — GitHub Actions

`.github/workflows/retrain.yml` runs the full pipeline (train → register → retrain → auto-promote → commit) automatically:
- On every push to `main`
- Weekly, every Sunday (scheduled)

No manual intervention required — this satisfies the "automatic replacement" requirement end-to-end.

## Tech Stack

MLflow, scikit-learn, XGBoost, pandas, GitHub Actions

## Author

Subhajit Dutta