# Project Narrative

This document summarizes the end-to-end multi-objective recommendation project: goals, dataset processing, modeling and evaluation, and the main takeaways.

**Problem Statement**
- Goal: build and evaluate recommender approaches that jointly optimize multiple user engagement signals (click, like, long view) rather than a single objective. The project explores multi-task modeling plus post-hoc reranking using Pareto-frontier methods to produce principled trade-offs between objectives.

**Dataset & Processing**
- Source: KuaiRand-style interaction logs assembled under `data/` and `real_data/`.
- Targets: `is_click`, `is_like`, `long_view` derived from interaction columns.
- Leakage removal: removed proxy/leaky features (e.g., `creator_proxy`, `is_follow`, `is_profile_enter`) and any columns computed from future signals.
- Splits & artifacts: produced canonical train/validation/test splits and saved processed features and targets to `data/processed/` along with metadata files (feature registry, scaler state). Feature scaling and categorical encoding were applied consistently across splits.

**Modeling & Architecture**
- Multi-task neural model: a shared encoder feeding task-specific heads (one head per target). Each head outputs a probability; per-task binary cross-entropy losses are combined with configurable per-task weights.
- Training: early stopping on validation, checkpointing of best epoch, and saving test predictions for downstream reranking experiments.

**Baselines**
- Simple single-task baselines: models optimized for one objective (click-only, like-only, longview-only) used as reference rankings.
- Standard baselines: logistic regression and LightGBM trained per-task; reported ROC-AUC and PR-AUC in `artifacts/tables/baseline_metrics.csv`.
- Scalarization: weighted-sum of model scores (weighted_scalar) used to produce rank lists for a chosen weight vector.

**Pareto Reranking**
- Implemented Pareto frontier selection and a `pareto_weighted` variant that orders Pareto-frontier candidates by a secondary weighted score to control trade-offs.
- Evaluation: mean NDCG@k across users for multiple strategies (click_only, like_only, longview_only, weighted_scalar, pareto_frontier, pareto_weighted). Representative numbers (NDCG@10 for `long_view`):
  - `weighted_scalar`: 0.6753
  - `pareto_weighted`: 0.6723
  - `pareto_frontier`: 0.6689

**Results Summary & Main Takeaway**
- Weighted scalarization produced the best single-number NDCG for `long_view` in these experiments, but Pareto methods provide an interpretable set of non-dominated alternatives and allow post-hoc selection of trade-offs without retraining.
- Pareto reranking (especially `pareto_weighted`) achieves competitive performance while exposing options for stakeholders to prioritize different objectives dynamically.

**Recommendations & Next Steps**
- Run systematic weight sweeps for scalarization and Pareto secondary-ordering to locate Pareto-efficient weight regions and compute statistical comparisons across users.
- Produce a reproducible notebook that recreates key figures (Pareto front visualizations, NDCG comparisons) from saved predictions in `artifacts/predictions/` and tables in `artifacts/tables/`.
- Consider user- or cohort-level optimization (personalized weights) and log-based A/B evaluation for deployment.

This narrative is intentionally concise so it can be dropped into a final report section or used as a presentation script.
