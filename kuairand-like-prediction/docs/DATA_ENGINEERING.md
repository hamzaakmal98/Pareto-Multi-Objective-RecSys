Data engineering notes
======================

Scope
-----
This document records the preprocessing steps, leakage mitigations, and locations of processed datasets used by the experiments.

Preprocessing pipeline (high level)
----------------------------------
1. Identify the time column (used as the exposure timestamp) and normalize to a pandas datetime.
2. Derive light-weight temporal features: `hour_of_day`, `day_of_week`, and `time_norm`.
3. Convert duration-like fields to seconds and rename consistent columns (e.g., `duration_s`).
4. Drop identifiers and any post-exposure / target-derived columns. In particular the pipeline removes `is_follow` and `is_profile_enter` (leakage sources) and ensures `is_click` (when used as a target) is not present in `X`.
5. Z-score normalize numeric features using train-set mean/std; the scaler is saved to `artifacts/feature_metadata/scaler.json`.
6. Save final processed artifacts: `data/processed/X.csv`, `data/processed/y.csv`, `data/processed/meta.csv`, and split indices `train_idx.csv`, `val_idx.csv`, `test_idx.csv`.

Important saved metadata
-----------------------
- `artifacts/feature_metadata/scaler.json` — train-set mean/std for numeric features.
- `artifacts/feature_metadata/feature_registry.json` — canonical feature lists and types.
- `artifacts/feature_metadata/target_prevalence.json` — target positive counts and prevalences.

Dataset statistics (as used by experiments)
-----------------------------------------
- Train rows: 4578
- Validation rows: 1526
- Test rows: 1526

Target prevalence (from EDA)
- `is_like`: 239/3904 positive (6.1219%)
- `long_view`: 3621/3904 positive (92.7510%)
- `creator_proxy`: 205/3904 positive (5.2510%) — creator proxy was audited and removed from features; `is_click` is used as the third task in final runs.

Feature correlations (selected)
-------------------------------
Key numeric correlations found in `artifacts/tables/target_feature_correlations.csv`:
- `play_time_s` vs `long_view`: 0.522187067972118 (strong positive association with long view)
- `duration_s` vs `long_view`: -0.060930659514181605 (small negative association)

Data engineering notes
======================

Purpose
-------
This document describes the full data engineering pipeline used to produce the leakage-safe, multi-task dataset used by experiments in this repository. It documents raw inputs, join logic, feature engineering decisions, scaling, split construction, and the artifacts that the experiments consume.

1) Raw data files used
----------------------
- KuaiRand-style impression logs (CSV): per-impression rows containing `user_id`, `video_id`, timestamp, and candidate-level metadata. These raw CSVs live under `real_data/` (or are supplied via the `data` path in configs).
- Lightweight side tables (when available): pre-computed user summaries, video metadata, and historical aggregates. Where present these are joined by `user_id` and `video_id`.

2) Join logic
-------------
- The preprocessing script merges the primary impressions table with available side tables on `user_id` and `video_id` using a left join pattern so that each impression row contains the user/video features available before exposure.
- Temporal joins (historical aggregates): any historical summary features are computed using records that precede the exposure timestamp. This is implemented by filtering source events by timestamp < exposure_time before aggregation.

3) Feature engineering steps
---------------------------
- Time features: derive `hour_of_day`, `day_of_week`, `hourmin` (hour*100 + minute), and a normalized `time_norm` where appropriate.
- Duration conversions: convert millisecond duration fields to seconds and create `duration_s` and related play-time features.
- Aggregates: compute user- and video-level aggregates (e.g., past like-rate, average play_time) using strictly pre-exposure data.
- Categorical handling: keep low-cardinality categories as-is and export higher-cardinality ids only as hashed embeddings or dropped depending on the experiment settings.

4) Scaling and normalization
----------------------------
- Numeric features are standardized with train-set z-score normalization (zero mean, unit variance). The pipeline fits the scaler on the training split only and applies it to val/test.
- The fitted scaler (means and standard deviations) is written to `artifacts/feature_metadata/scaler.json` so runs are reproducible.

5) Time-based features created
-----------------------------
- `hour_of_day` — 0–23 value extracted from exposure timestamp.
- `day_of_week` — 0–6 value extracted from exposure timestamp.
- `hourmin` — compact hour/min representation useful for visualization and bucketing.
- `time_norm` — optional normalized time-of-day used in some experiments.

6) Columns removed for leakage prevention
-----------------------------------------
The preprocessing pipeline explicitly removes any columns that are either direct labels or post-exposure signals for the candidate impression. Notable removals include:
- Candidate-level post-exposure indicators: explicit click flags or watch-fraction values for the same impression.
- Features that directly encode or proxy targets derived from post-exposure events (for example, `is_follow` and `is_profile_enter` which were audited and excluded).
- Any column used to derive temporary `creator_proxy` signals that were found to leak label information — `creator_proxy` was removed and replaced by `is_click` as a target where appropriate.

7) Why leakage prevention mattered in this project
-------------------------------------------------
- Leakage (features that contain or are computed from future/post-exposure information) can produce deceptively strong offline metrics and cause deployed models to perform poorly in production.
- Because the project investigates multi-objective behavior and ranking, avoiding leakage is critical for honest comparisons between ranking strategies and to ensure Pareto analyses reflect real pre-exposure signals.

8) Final target definitions
---------------------------
- `is_click`: binary indicator whether the user clicked the candidate impression (used as one of the tasks in multi-task training).
- `is_like`: binary indicator whether the user liked the item — a sparse positive target (reported prevalence: 239 positives out of 3904 rows; see `reports/analysis/eda_summary.md`).
- `long_view`: binary indicator whether the impression led to a long watch.

9) Split construction
----------------------
- The pipeline creates train/validation/test splits and writes index files to `data/processed/train_idx.csv`, `data/processed/val_idx.csv`, and `data/processed/test_idx.csv`.
- The canonical run used in reports employed: train = 4578 rows, val = 1526 rows, test = 1526 rows. Splits are created to preserve temporal ordering where possible (validation and test sets typically cover later time windows) to mimic realistic evaluation.

10) Saved processed artifacts
----------------------------
The preprocessing pipeline writes the following artifacts (canonical paths used by experiments):
- `data/processed/X.csv` — feature matrix (leakage-filtered) used as model inputs.
- `data/processed/y.csv` — target columns (`is_click`, `is_like`, `long_view`).
- `data/processed/meta.csv` — metadata per impression (timestamps, `user_id`, `video_id`, and other fields used for grouping/evaluation).
- `data/processed/train_idx.csv`, `val_idx.csv`, `test_idx.csv` — split indices.
- `artifacts/feature_metadata/scaler.json` — saved z-score parameters.
- `artifacts/feature_metadata/feature_registry.json` — feature lists and types.
- `artifacts/feature_metadata/target_prevalence.json` — positive counts and prevalences used in reports.

Why users/interactions were filtered
----------------------------------
- EDA showed heavy right-skew in user activity (many one-off users and a small set of heavy users). Filtering or careful split construction reduces the risk that a handful of heavy users dominate aggregated metrics and helps create stable validation/test comparisons.
- Filtering also ensures that historical aggregates have sufficient pre-exposure data and reduces extreme cold-start cases during analysis.

Key assumptions
---------------
- Timestamps are correct and can be used to establish a strict pre-exposure cutoff for historical aggregations.
- Labels in the raw logs (`is_click`, `is_like`, `long_view`) are reliable and recorded at the time of or after each impression.
- Side tables (user/video metadata) reflect pre-exposure state or are timestamped appropriately when used for historical features.

Limitations of the preprocessing pipeline
-----------------------------------------
- Dataset size used in experiments is small (train ~4.5k rows); results may not generalize to larger-scale production logs without additional tuning.
- `is_like` is sparse (6.12% positive in the reported EDA), which makes stable learning and ranking for likes challenging without sampling or more data.
- Candidate generation / retrieval is out-of-scope; experiments assume a fixed candidate pool derived from the impressions log.
- The pipeline does not currently implement advanced de-duplication or temporal smoothing for user histories; these are possible future improvements.

Files to review
---------------
- Preprocessing script: `scripts/prepare_kuairand_data.py` (main entrypoint for dataset creation).
- Feature registry and scaler: `artifacts/feature_metadata/feature_registry.json`, `artifacts/feature_metadata/scaler.json`.
- Processed data folder: `data/processed/` (X, y, meta, and split indices).

If you want, I can produce a short notebook that shows the preprocessing checks (leakage tests, histogram of time features, and a small unit test that verifies `is_click` is not present in `X`).
