# EDA Summary

## Key figures
Figures saved to `artifacts/figures/eda/`

## Target Prevalence
- `is_like`: 239/3904 positive (6.1219% positive)
- `long_view`: 3621/3904 positive (92.7510% positive)
- `creator_proxy`: 205/3904 positive (5.2510% positive)

## User Activity
- `user_activity.png` shows distribution of impressions per user; heavy right-skew indicates many one-off users and a small set of heavy users.

## Item Popularity
- `item_popularity.png` shows per-item impression counts; long-tail popularity may require frequency-based handling in modeling.

## Temporal Distribution
- `temporal_distribution.png` shows interaction volume over time; useful to verify stationarity and choose split strategy.

## Missingness Summary
- `missingness_summary.png` highlights columns with missing data; consider imputation or feature removal for high-missing columns.

## Numeric Feature Distributions
- `numeric_histograms.png` and `correlation_heatmap.png` provide quick checks for scaling needs and multicollinearity.

## Target Interactions
- Not enough targets present to show pairwise interactions.

## Notes and next steps
- All detected targets are treated as labels and excluded from model inputs by the preprocessing pipeline.
- Extremely imbalanced targets (low positive rate) may require class weighting, sampling, or alternative metrics (NDCG/PR-AUC).