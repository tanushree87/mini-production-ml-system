# Mini Production ML System — Credit Card Fraud Detection

**BITS Pilani WILP | MSc Data Science & Artificial Intelligence**  
**Course:** ML Model Engineering (MDSDF408) | **Assignment:** Graded Assignment (20 Marks)

**GitHub Repository:** https://github.com/tanushree87/mini-production-ml-system

> End-to-end production ML system for credit card fraud detection, covering temporal data splitting, model training and evaluation, FastAPI serving, input validation, latency profiling, and a proposed drift-monitoring blueprint.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Repository Structure](#repository-structure)
- [Quickstart](#quickstart)
- [Key Results](#key-results)
- [Design Decisions](#design-decisions)
- [API Reference](#api-reference)
- [Drift Monitoring Blueprint](#drift-monitoring-blueprint)

---

## Project Overview

This project builds a cohesive production-oriented ML system for credit card fraud detection. Rather than focusing only on model development, it covers the engineering lifecycle:

| Phase | What is built |
|---|---|
| 1 — Data & Splitting | Chronological train/validation/test split to avoid look-ahead leakage |
| 2 — Model Engineering | Dummy baseline and LightGBM model, imbalance handling, and validation-based threshold selection |
| 3 — Serving | FastAPI REST API with Pydantic input/output validation |
| 4 — Operations | Local latency profiling, automated schema tests, and a drift-monitoring blueprint |

**Dataset:** [Credit Card Fraud Detection (ULB, Kaggle)](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) — 284,807 transactions, with fraud representing approximately 0.17% of the data. Features `V1`–`V28` are PCA-transformed; `Time` and `Amount` are the original transaction fields.

---

## Repository Structure

```text
mini-production-ml-system/
├── README.md
├── requirements.txt
├── data/                        # Dataset goes here (not committed)
├── artifacts/                    # Generated model/evaluation outputs (not committed)
│   ├── scaler.joblib
│   ├── lgbm_model.joblib
│   ├── threshold.json
│   ├── pr_curve.png
│   ├── confusion_matrix.png
│   └── benchmark_results.json
├── src/
│   ├── __init__.py
│   ├── data_pipeline.py          # Temporal split and preprocessing
│   ├── train.py                 # Baseline, LightGBM, threshold tuning
│   └── evaluate.py               # Test evaluation and diagnostic plots
├── serving/
│   ├── __init__.py
│   ├── app.py                   # FastAPI service (/health and /predict)
│   ├── schemas.py               # Pydantic request/response validation
│   └── benchmark.py              # Sequential latency benchmark
└── tests/
    ├── __init__.py
    └── test_schemas.py           # Automated schema tests
```

---

## Quickstart

### 1. Clone the repository

```bash
git clone https://github.com/tanushree87/mini-production-ml-system.git
cd mini-production-ml-system
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Add the dataset

Download `creditcard.csv` from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) and place it in the `data/` directory:

```text
data/
└── creditcard.csv
```

### 4. Train the model

```bash
python src/train.py
```

This trains the baseline and LightGBM model, selects a classification threshold using the validation set, and saves the model artifacts to `artifacts/`.

### 5. Evaluate on the held-out test set

```bash
python src/evaluate.py
```

This prints the evaluation metrics and saves the precision-recall curve and confusion matrix to `artifacts/`.

### 6. Start the REST API

```bash
uvicorn serving.app:app --host 127.0.0.1 --port 8000
```

### 7. Health check

```bash
curl http://127.0.0.1:8000/health
```

Expected response (threshold shown at full precision):

```json
{
  "status": "ok",
  "model": "lgbm-v1",
  "threshold": 0.9999999837335847
}
```

### 8. Score a transaction

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "Time": 406, "Amount": 149.62,
    "V1": -1.36, "V2": -0.07, "V3": 2.54, "V4": 1.38,
    "V5": -0.34, "V6": 0.46, "V7": 0.24, "V8": 0.10,
    "V9": 0.36, "V10": 0.09, "V11": -0.55, "V12": -0.62,
    "V13": -0.99, "V14": -0.31, "V15": 1.47, "V16": -0.47,
    "V17": 0.21, "V18": 0.03, "V19": 0.40, "V20": 0.25,
    "V21": -0.02, "V22": 0.28, "V23": -0.11, "V24": 0.07,
    "V25": 0.13, "V26": -0.19, "V27": 0.13, "V28": -0.02
  }'
```

Example response:

```json
{
  "fraud_probability": 0.002388,
  "prediction": "LEGIT",
  "threshold_used": 0.9999999837335847,
  "model_version": "lgbm-v1"
}
```

### 9. Run latency benchmark

The API must be running in another terminal.

```bash
python serving/benchmark.py
```

### 10. Run automated tests

```bash
pytest tests/ -v
```

Latest verified result:

```text
12 passed, 2 warnings in 0.49s
```

The warnings are from dependency compatibility and a Pydantic protected-namespace warning; they did not cause test failures.

---

## Key Results

### Model Performance — Held-Out Test Set

| Metric | Dummy Baseline | LightGBM |
|---|---:|---:|
| PR-AUC | 0.0012 | **0.7049** |
| Fraud Precision | 0.00 | **0.6842** |
| Fraud Recall | 0.00 | **0.7500** |
| Fraud F1 | 0.00 | **0.7156** |
| Macro F1 | 0.4992 | **0.8576** |
| Accuracy | — | **0.9993** |
| PR-AUC Lift | — | **579.1×** |

*PR-AUC lift is calculated using the unrounded PR-AUC values; the table displays rounded metrics.*

### Data Partitions

| Partition | Rows | Fraud Transactions | Fraud Rate |
|---|---:|---:|---:|
| Train | 199,364 | 384 | 0.1926% |
| Validation | 42,721 | 56 | 0.1311% |
| Test | 42,722 | 52 | 0.1217% |

The data was split chronologically using `Time`, rather than randomly shuffled, to simulate evaluation on transactions occurring later in time.

### API Latency — Local Sequential Benchmark

The benchmark sent 500 sequential requests to the local API.

| Metric | Value |
|---|---:|
| Requests | 500 |
| Errors | 0 |
| Min latency | 9.639 ms |
| Mean latency | 17.710 ms |
| p50 latency | 16.320 ms |
| p95 latency | 25.219 ms |
| p99 latency | 40.993 ms |
| Max latency | 266.909 ms |
| Throughput | 56.5 requests/sec |

The observed p95 and p99 latencies were below 100 ms during this local sequential benchmark. The maximum latency was 266.909 ms. These measurements characterize the prototype under the tested conditions; they are not a concurrent load test or a production SLA guarantee.

### Automated Testing

The pytest suite covers valid payloads, required fields, missing fields, invalid data types, null values, boundary values, prediction labels, and default model-version handling.

| Test result | Value |
|---|---:|
| Total tests | 12 |
| Passed | 12 |
| Failed | 0 |
| Execution time | 0.49 seconds |

---

## Design Decisions

### Temporal split over random shuffle

Sorting transactions by `Time` before splitting preserves chronological order. A random shuffle could place later transactions into training and earlier transactions into evaluation, creating an unrealistic evaluation setup and potential look-ahead bias.

### PR-AUC over accuracy

Fraud represents approximately 0.17% of the dataset. A classifier predicting every transaction as legitimate can achieve very high accuracy while detecting no fraud. PR-AUC focuses on the precision-recall trade-off for the minority class and is more informative for this imbalanced problem.

### Class imbalance handling

The dataset exhibits severe class imbalance. Instead of using the full negative-to-positive class ratio, `scale_pos_weight` was set to the square root of the class ratio (approximately 22.76). This provides a moderated weighting strategy for the minority class during model training.

### F-beta (β = 2) threshold tuning

The classification threshold is selected on the validation set by maximising F2-score, which places greater emphasis on recall than precision. In fraud detection, missing a fraudulent transaction may carry a greater cost than investigating a false alert. The test set is held out for final evaluation and is not used for threshold selection.

### Scaler fitted on training partition only

The `StandardScaler` is fitted exclusively on the training partition and serialised. Validation, test, and inference data are transformed using the fitted scaler without refitting, preventing preprocessing leakage.

---

## API Reference

### `GET /health`

Returns service health and the active model/decision threshold.

Example:

```json
{
  "status": "ok",
  "model": "lgbm-v1",
  "threshold": 0.9999999837335847
}
```

### `POST /predict`

Scores a single transaction. All 30 numerical features are required.

**Request body:** `TransactionInput`

- `Time` — transaction time field
- `Amount` — transaction amount in USD
- `V1`–`V28` — PCA-transformed numerical features

**Response:** `PredictionOutput`

| Field | Type | Description |
|---|---|---|
| `fraud_probability` | float | Model-estimated fraud score/probability |
| `prediction` | string | `FRAUD` or `LEGIT` |
| `threshold_used` | float | Decision threshold applied |
| `model_version` | string | Model identifier |

**Error responses**

| HTTP status | Meaning |
|---|---|
| `422` | Missing field, invalid type, or null input rejected by validation |
| `503` | Model unavailable/not loaded |
| `500` | Internal scoring error |

---

## Drift Monitoring Blueprint

The following monitoring and retraining mechanisms are proposed as a production operations blueprint. They are not implemented as automated monitoring services in this prototype.

| Signal | Metric | Threshold | Proposed action |
|---|---|---|---|
| Feature drift | PSI per feature, weekly | PSI > 0.20 on 3 or more features | Trigger retraining review |
| Output drift | KS test on prediction scores | p < 0.05 | Manual review |
| Business metric | Fraud detection rate, weekly | Drop > 10% week-over-week | Alert and investigate |
| System health | p95 latency | > 100 ms | Investigate/scale |
| System health | Error rate (4xx/5xx) | > 1% | Alert and investigate |

**Retraining trigger:** Any drift signal exceeding its proposed threshold, or scheduled monthly retraining regardless of drift.

**Fallback:** If model scoring fails, the prototype returns HTTP 500. A production deployment could supplement this with a business-rule-based fallback mechanism, subject to risk and business approval. For example, transactions meeting selected high-risk rules could be routed for additional review while the model service recovers.

---

## Limitations

- The benchmark is a local, sequential, single-client test; concurrent load and production infrastructure were not evaluated.
- The dataset is anonymised and historical. Real-world fraud patterns, transaction systems, and feature availability may differ.
- Monitoring, alerting, and automated retraining are described as a blueprint and are not implemented.
- The selected decision threshold is validation-derived and should be reassessed against operational costs, fraud loss, alert volume, and calibration requirements before production deployment.

---

## Conclusion

This project demonstrates an end-to-end ML engineering workflow: leakage-conscious temporal data partitioning, preprocessing, baseline comparison, model training, validation-based threshold selection, held-out evaluation, model serialization, API inference, schema robustness tests, and latency profiling. It also documents a proposed operational monitoring and retraining strategy for a production deployment.
