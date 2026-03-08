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