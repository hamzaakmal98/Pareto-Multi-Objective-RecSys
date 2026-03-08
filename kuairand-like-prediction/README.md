# KuaiRand Multi-Objective Recommendation (Research)

This repository contains an end-to-end research implementation for multi-objective recommendation using KuaiRand-style interaction logs. The project investigates multi-task prediction models and several ranking strategies — including Pareto-frontier reranking — to produce principled trade-offs between engagement signals.

## Project Overview
- Objective: predict and rank items for users optimizing multiple engagement signals simultaneously (click, like, long view).
- Approach: train a multi-task model to predict three binary engagement targets, generate candidate scores, and compare ranking strategies that trade off objectives differently.

## Dataset
- Input: KuaiRand-style interaction logs (raw data in `real_data/` and project `data/` folders).
- Targets: `is_click`, `is_like`, `long_view` derived from interaction events.
- Artifacts: processed features, splits and metadata are stored in `data/processed/` and `artifacts/` (feature registry, scaler state, saved predictions).

## Problem Setup
- Predict per-item probabilities for three engagement signals for each user.
- Produce ranked lists for users using different ranking strategies and evaluate retrieval quality (NDCG@5, NDCG@10, NDCG@20) averaged across users.

## Architecture
- Multi-task neural network with a shared encoder and task-specific heads. Each head outputs a probability for its target and uses binary cross-entropy loss.
- Training features: per-task loss weighting, early stopping on validation, and checkpointing of best model. Test predictions are saved for downstream reranking experiments.

### Architecture Diagram

```mermaid
flowchart LR
  A[KuaiRand-Pure Dataset] --> B[Data Engineering]
  B --> B1[join interaction, user, and video features]
  B --> B2[remove leakage columns]
  B --> B3[create time and numeric features]
  B --> B4[scale and split data]
  B4 --> C[Multi-Task Prediction Model]
  C --> C1[click score]
  C --> C2[like score]
  C --> C3[long_view score]
  C --> D[Ranking Strategies]
  D --> D1[click-only]
  D --> D2[like-only]
  D --> D3[longview-only]
  D --> D4[weighted scalar]
  D --> D5[Pareto frontier]
  D --> D6[Pareto + weighted]
  D --> E[Evaluation]
  E --> E1[NDCG@5]
  E --> E2[NDCG@10]
  E --> E3[NDCG@20]
```

Diagram: an end-to-end pipeline showing preprocessing of the KuaiRand dataset, multi-task prediction producing three engagement scores, several reranking strategies, and NDCG-based evaluation.
## Ranking Strategies Evaluated
- click-only / like-only / longview-only: rank purely by a single-task score.
- weighted_scalar: scalarize multiple scores via a weighted sum, then sort.
- pareto_frontier: select non-dominated candidates on multi-objective scores to form a reranked list.
- pareto_weighted: order Pareto-frontier candidates by a secondary weighted score to control trade-offs.

## Evaluation Metrics
- NDCG@5, NDCG@10, NDCG@20 averaged across users — reported in `artifacts/tables/` and `reports/analysis/`.
- Per-task predictive metrics (ROC-AUC, PR-AUC) are reported for baseline models (logistic, LightGBM) and the multi-task model.

## Key Insights
- Weighted scalarization often yields the best single-number retrieval metric for a chosen objective, but requires choosing weights up front.
- Pareto reranking produces an interpretable set of non-dominated trade-off solutions and enables post-hoc selection without retraining.
- The `pareto_weighted` variant offers a practical compromise: near-scalarization performance with more transparent trade-offs.

## Repository Structure
- `data/` — raw and processed datasets and canonical train/val/test splits.
- `scripts/` — preprocessing, training, and reporting utilities (e.g., `prepare_kuairand_data.py`, `smoke_train_multitask.py`, `generate_final_tables.py`).
- `src/` — model, data loading, evaluation, and feature engineering code.
- `artifacts/` — generated tables, predictions, models and figures produced by experiments.
- `reports/analysis/` — final narrative, experiment summaries, and figure/table markdowns.

## Future Work
- Run systematic weight sweeps for scalarization and Pareto secondary-ordering; add statistical tests across users.
- Build a reproducible notebook that recreates key figures from saved predictions and tables.
- Explore personalized weightings (per-user or per-cohort) and evaluate in A/B tests or logged offline policy learning.

---
For details, see the analysis and generated artifacts in `reports/analysis/` and `artifacts/`.
Project Title
=============

KuaiRand-like Multi-Objective Recommendation — research prototype

Problem Overview
----------------
This repository is a compact research prototype for learning and evaluating multi-objective recommendation from KuaiRand-style interaction logs. The goal is to jointly consider multiple engagement signals so ranking and re-ranking strategies can balance trade-offs between objectives such as clicks, likes, and long views.

Why Multi-Objective Recommendation
---------------------------------
- Real-world recommender systems optimize multiple business and engagement objectives simultaneously.
- A single scalar objective (e.g., click-through-rate) can bias ranking toward short-term signals and may harm downstream long-view metrics. Multi-objective approaches let teams explore trade-offs and re-ranking strategies that respect multiple objectives.

Dataset
-------
- Source: KuaiRand-style interaction logs (raw datasets under `real_data/` and processing scripts in `scripts/`).
- Interaction logs: per-impression rows including timestamps, user id, video id, and pre-exposure metadata.
- User features: profile and historical summary features computed prior to each exposure (see `data/processed/meta.csv`).
- Video features: static metadata and aggregated statistics (duration, play-time statistics) computed without using post-exposure labels.

Final Objectives
----------------
The experiments in this repo use three targets (final objectives):
- `is_click` — whether the user clicked the candidate impression (used as a target in multi-task experiments).
- `is_like` — whether the user liked the item (sparse, difficult target).
- `long_view` — whether the impression resulted in a long watch.

Data Engineering Pipeline
-------------------------
- The pipeline reads raw KuaiRand-like CSVs and produces leakage-safe processed artifacts: `data/processed/X.csv`, `data/processed/y.csv`, `data/processed/meta.csv`, and split indices (`train_idx.csv`, `val_idx.csv`, `test_idx.csv`).
- Numeric features are z-score standardized using train-set statistics; the scaler is saved to `artifacts/feature_metadata/scaler.json`.
- Explicit leakage prevention: post-exposure signals (click indicators for candidate impression, watch-duration for the same impression, and target-derived proxies) are excluded from `X`.
- Dataset sizes used in reported runs: train = 4578, val = 1526, test = 1526.

Modeling Approach
-----------------
- Simplified multi-task model: a compact PyTorch model with a shared encoder and three prediction heads (one per target). Training uses per-task binary cross-entropy losses and early stopping.
- Shared encoder: a small feed-forward network that learns common representations across tasks.
- Three prediction heads: independent final layers for `is_click`, `is_like`, and `long_view`.

Ranking Strategies Evaluated
----------------------------
The repository evaluates multiple ranking and reranking strategies on the same candidate pool and aggregates NDCG across users.
- `click_only` — sort by predicted click score.
- `like_only` — sort by predicted like score.
- `longview_only` — sort by predicted long-view score.
- `weighted_scalar` — score = scalar combination of predicted objectives (used a trained scalar weighting in experiments).
- `pareto_frontier` — select Pareto-efficient candidates across objectives and flatten Pareto layers into a ranking.
- `pareto_weighted` — compute Pareto frontier, then order Pareto candidates by a secondary weighted score (the new pareto-weighted strategy implemented in this repo).

Evaluation
----------
Ranking performance is reported with NDCG at top-k positions aggregated across users. Reported metrics are taken from saved artifacts in `artifacts/tables/`.
- NDCG@5
- NDCG@10
- NDCG@20

Key Findings (canonical numbers from saved artifacts)
----------------------------------------------------
All numbers below are taken directly from saved artifacts and run reports in `artifacts/tables/` and `reports/analysis/`.

- Baseline highlights (`artifacts/tables/baseline_metrics.csv`):
  - LightGBM `is_like` PR-AUC = 0.30312505727387185 (best baseline for the sparse like target).

- Multi-task (weighted) training (`reports/analysis/weighted_training_run.md`):
  - Best epoch: 20 (train rows = 4578, val rows = 1526).
  - Test PR-AUCs (best checkpoint): `is_click` = 0.9963813120082162, `is_like` = 0.057604364398194635, `long_view` = 0.9862033610184631.

- Pareto / ranking evaluation (`artifacts/tables/pareto_ranking_results.csv` and `artifacts/tables/pareto_weighted_results.csv`):
  - `click_only` achieves `is_click` NDCG@5 = 0.9914812310000376 (mean across users).
  - `weighted_scalar` achieves `long_view` NDCG@20 = 0.6754049387402925.
  - `pareto_weighted` (Pareto front ordered by secondary weight) achieves `long_view` NDCG@20 = 0.6736256922589338 and `is_click` NDCG@5 = 0.9890577647563572.

Repo Structure (key paths)
-------------------------
- `data/` — raw and processed datasets; processed artifacts under `data/processed/`.
- `scripts/` — preprocessing and ad-hoc experiment scripts (e.g., `prepare_kuairand_data.py`, `smoke_train_multitask.py`).
- `src/` — core code: data loaders, feature builders, model definitions, training loops.
- `artifacts/` — generated metadata, saved models, prediction CSVs, and `artifacts/tables/` with canonical CSV summaries.
- `reports/analysis/` — human-readable run summaries and analysis files (including `weighted_training_run.md` and `pareto_ranking_results.md`).
- `docs/` — design and reproducibility notes (`DESIGN.md`, `DATA_ENGINEERING.md`, `EXPERIMENTS.md`).

How to Reproduce (quick)
------------------------
The `docs/EXPERIMENTS.md` file contains the short reproducibility commands. Minimal steps:

```powershell
# regenerate processed data
python scripts/prepare_kuairand_data.py --input <raw_csv> --out data/processed --targets is_click,is_like,long_view

# run baseline (example)
python src/train_baseline.py --model lgbm --config configs/default.yaml

# run the weighted multi-task trainer
python scripts/smoke_train_multitask.py --epochs 20 --batch_size 128 --lr 5e-4

# compute pareto reranking and evaluation
python scripts/pareto_rerank_and_evaluate.py
```

Limitations
-----------
- This is a research prototype, not a production MMoE system. The multi-task model is a simplified shared-encoder + heads implementation intended for exploration and curriculum experiments.
- `is_like` is a low-prevalence target; results should be interpreted with caution and larger datasets or specialized sampling/weighting may be required.
- The ranking experiments use offline NDCG aggregated across users and do not measure online impact or business KPIs.

Future Work
-----------
- Run a weight-sweep for scalarization and report statistical comparisons (paired tests) across users.
- Add a small notebook that reproduces Pareto ranking figures and per-user example cases for presentation slides.
- Experiment with richer MMoE-style gating and larger expert pools; evaluate transfer learning effects across tasks.

Where to find canonical results
--------------------------------
- `artifacts/tables/baseline_metrics.csv`
- `reports/analysis/weighted_training_run.md`
- `artifacts/tables/pareto_ranking_results.csv`
- `artifacts/tables/pareto_weighted_results.csv`
- `reports/analysis/pareto_ranking_results.md`

If you want me to produce a single-slide summary or a notebook that reproduces the Pareto figures, tell me which output format you prefer (PowerPoint, PDF, Jupyter notebook) and I will generate it from the saved artifacts.
