-# Data Engineering

This document describes the data engineering pipeline for the KuaiRand-Pure dataset used in the KuaiRand Pareto MMoE project. It is written to be rigorous and reproducible while remaining readable for presentations and reports.

Contents
- Overview of source tables
- Supervised interaction table
- Why random-exposure logs are preferred for evaluation
- Join strategy and entity keys
- Cleaning steps
- Missing-value handling
- Feature typing
- Leakage prevention policy
- Train / validation / test splitting
- Saved artifacts
- Why this pipeline supports multi-task recommendation

## 1) Overview of source tables

Typical source tables available (names vary by dataset ingest):

- interaction_log (impression-level rows): one row per candidate impression shown to a user; contains `impression_id`, `user_id`, `video_id`, `timestamp`, `exposure_flag`, and raw engagement outcomes.
- exposure_log / random_exposure: a randomized exposure subset where candidate exposures were randomized at serving time. Used to reduce exposure bias for evaluation.
- user_profile: static and slowly-changing user attributes (`user_id`, age_bin, country, signup_date, follow_count`, ...).
- video_meta: immutable video-level attributes (`video_id`, `creator_id`, `duration_secs`, `category`, `upload_ts`, ...).
- video_stats: historical or aggregated statistics for video performance (`video_id`, `avg_view_time`, `like_rate`, `impressions_last_7d`, ...).
- session_context / device_log: per-impression context such as device type, app version, placement, and local time.

## 2) Which table is used for supervised interaction modeling

- Primary supervised table: `interaction_log` (or the instrumented random-exposure subset when training a model intended to generalize under unbiased exposure). Each row corresponds to a candidate shown to a user at a specific time; target labels are recorded on this row (see Feature typing / Targets).

Notes:
- For model training we typically use `data/processed/train.parquet` derived from `interaction_log` after joins and feature engineering. For unbiased evaluation, prefer rows from `random_exposure` held-out splits (see section 3).

## 3) Why the random exposure log is preferred for evaluation

- Exposure bias: logs of production recommendations reflect the ranking and selection policy: items not shown generate no label and may bias offline evaluation.
- Random-exposure logs contain impressions assigned with some randomized policy (or A/B treatment) that breaks correlation between model ranking and exposure. This reduces selection bias and yields more reliable estimates of true user preferences.
- Using random-exposure rows in the test set produces more trustworthy NDCG estimates and clearer Pareto tradeoff analysis because the observed outcomes are less confounded by prior ranking policies.

## 4) Join strategy

Goal: produce a single, wide per-impression table that contains all features required by the model and downstream reranker while keeping provenance and reproducibility clear.

Join order (recommended):

1. Start with `interaction_log` (impression-level) as the left table — this preserves all impressions and their timestamps.
2. Left-join `session_context` / `device_log` on `impression_id` or (`user_id`, `timestamp`) as available.
3. Left-join `user_profile` on `user_id`.
4. Left-join `video_meta` on `video_id`.
5. Left-join `video_stats` on `video_id` (use timestamp-aware aggregation where possible, e.g., video stats computed up to impression time).

Rationale:
- Keeping `interaction_log` as the left table preserves impressions where downstream joins may be missing (we often want to keep impressions and impute missing features rather than drop rows).
- Joining in this order minimizes accidental leakage: video-level stats should be computed using only information available prior to the impression time (see Leakage prevention).

## 5) Entity keys used for joins

- impression-level primary key: `impression_id` (if available)
- user key: `user_id`
- item key: `video_id`
- creator key: `creator_id` (present in `video_meta` and optionally `video_stats`)
- session/time context key: `timestamp` (or `ts`) to support time-aware aggregations

When `impression_id` is missing, use composite keys (`user_id`, `video_id`, `timestamp`, `placement`) as a resilient alternative.

## 6) Cleaning steps

Apply these steps consistently and save intermediate artifacts for reproducibility:

1. Normalize column names to snake_case and canonical types.
2. Remove exact duplicate rows based on the impression primary key.
3. Filter out known bot or QA traffic using blacklists and heuristics (e.g., extremely high impression rates per user, impossible device ids).
4. Align timezones and convert timestamps to UTC; create derived time features (`hour_of_day`, `day_of_week`).
5. Truncate unrealistic numeric outliers (clip durations, rates) and optionally log-transform heavy-tailed features.
6. Deduplicate or collapse near-duplicate video_meta rows by taking the latest authoritative record.
7. Validate referential integrity: log and keep counts for unmatched `video_id` or `user_id` (do not silently drop rows without recording statistics).

Record all cleaning steps and their parameters in `data/interim/cleaning_log.yaml` for auditability.

## 7) Missing-value handling strategy

Follow a consistent, documented approach per feature type:

- For critical ID keys (`user_id`, `video_id`, `impression_id`, `timestamp`): drop rows or route to a quarantine file; log counts.
- Categorical features (e.g., `category`, `device_type`): replace missing with a sentinel string `"__MISSING__"` and record a missing indicator column (e.g., `category_missing` = 1/0).
- Numeric features (e.g., `duration_secs`, `avg_view_time`): impute with the median computed on the training set; add an `_is_missing` flag. For skewed metrics consider log1p before imputation.
- Aggregated statistics (e.g., `impressions_last_7d`): if historical windows are empty, default to 0 and add an `_is_missing` flag.
- Temporal/context fields: if missing, fall back to a conservative default (e.g., `hour_of_day` to -1) and add a missing flag.

Important: compute imputation statistics (median, category vocab, etc.) only on the training split and persist them (e.g., `artifacts/feature_metadata/encoders/`) so that validation/test transforms are identical.

## 8) Feature typing

Maintain a `feature_metadata.yaml` that records each feature's type, domain, and preprocessing steps. Example groupings below.

- ID columns (do not featurize as continuous):
	- `impression_id`, `user_id`, `video_id`, `creator_id`, `session_id`

- Categorical columns (one-hot, ordinal encoding, or embedding):
	- `category`, `device_type`, `country`, `placement`, `app_version`

- Numeric columns (continuous; scale and impute):
	- `duration_secs`, `avg_view_time`, `like_rate`, `impressions_last_7d`, `user_follow_count`

- Temporal / context columns:
	- `timestamp` (raw UTC ts), `hour_of_day`, `day_of_week`, `time_since_upload` (impression_ts - upload_ts), `session_length`

- Target columns (supervised outcomes; these are measured on the interaction row):
	- `like` (binary: user liked this impression)
	- `long_view` (binary or ordinal: viewed >= threshold or fraction of duration)
	- `creator_interest` (binary: user followed or interacted with creator)

Notes on encoding:
- Categorical cardinality should be recorded; for high-cardinality fields (e.g., `video_id`), use embeddings or frequency-bucketing with `__OTHER__` sentinel for rare values.
- For temporal features consider cyclical encodings for hour/day when used directly in models.

## 9) Leakage prevention policy

This project enforces strict rules to avoid information leakage between features and targets. The following interaction outcome columns are strictly banned as model inputs (they are only targets):

- `like` (current impression like flag)
- `long_view` (current impression long view indicator)
- `creator_interest` (current impression creator interaction)

Additional leakage rules:

1. Do not use any feature that materializes future behavior relative to the impression timestamp. All aggregated features (video_stats, user_stats) must be computed using data available strictly before the impression time.
2. When computing video or user aggregates (e.g., last-7-day impressions, like rates), use rolling-window computations with an exclusive end time (<= impression_ts - epsilon).
3. When training on the random-exposure subset for unbiased evaluation, avoid mixing policy logs that would reintroduce historical ranking effects as implicit features unless explicitly modeled and justified.
4. Keep feature generation code and the exact SQL/queries in `data/interim/` and record the run timestamp for reproducibility.

## 10) Train / validation / test splitting strategy

Recommended approach for robust multi-task evaluation:

- Primary split method: time-based split by impression timestamp. Use an earlier contiguous period for training, a subsequent contiguous period for validation, and the latest period for test. This preserves temporal causality and avoids peeking.
- For the final test set prefer rows from the `random_exposure` log to reduce exposure bias in evaluation metrics.
- When measuring generalization across users, consider a secondary user-holdout split (user-level split) or nested CV using GroupKFold by `user_id`.
- Keep stratification by rarity for rare targets (e.g., sample more positives into validation/test when computing diagnostics), but do not change the train distribution for training without documenting the sampling scheme.

Split artifact naming convention example:

- `data/processed/train_2025-01-01_2025-03-31.parquet`
- `data/processed/val_2025-04-01_2025-04-30.parquet`
- `data/processed/test_random_exposure_2025-05.parquet`

## 11) Saved artifacts

Persist the following artifacts with clear metadata (who ran it, timestamp, config id):

- Interim joined tables: `data/interim/joined_interaction_{config_id}.parquet` (contains raw joined columns plus derived time/context fields). Include a `provenance.json` alongside.
- Processed feature metadata: `artifacts/feature_metadata/feature_metadata.yaml` and encoder objects (e.g., `artifacts/feature_metadata/category_encoder.joblib`).
- Training-ready datasets: `data/processed/train.parquet`, `data/processed/val.parquet`, `data/processed/test.parquet` (each with a matching `*_manifest.json` that lists rows count, time range, and source hashes).
- Aggregation snapshots: `data/interim/video_stats_snapshot_{ts}.parquet` used to compute time-aware features.
- Leakage audit logs: `artifacts/leakage_checks/{run_id}.log` describing checks performed before training.

Store all artifacts with experiment identifiers and timestamps to ensure traceability and reproducibility.

## 12) Why this engineering pipeline supports a multi-task recommender

- Shared features: By producing a single joined table per impression, models can learn shared representations (e.g., via MMoE) while task-specific towers take advantage of specialized signals.
- Consistent preprocessing and persisted encoders ensure that the same features are available to all tasks and to reranking/evaluation stages.
- Time-aware aggregation and careful leakage prevention make objective comparisons fair: per-objective statistics are computed under the same temporal cutoffs, enabling valid Pareto frontier analysis.
- Saved artifacts separate raw data, interim joins, and model-ready splits so experiments (e.g., changing one target threshold or adding a new objective) can be repeated without re-running expensive joins.
- The pipeline explicitly stores per-objective evaluation artifacts (predictions, NDCG scores, and Pareto sets), which are necessary to plot and interpret tradeoffs between objectives.

---

