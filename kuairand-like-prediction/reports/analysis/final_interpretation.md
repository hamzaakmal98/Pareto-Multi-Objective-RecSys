Final reranking interpretation
=============================

What the final ranking table shows
---------------------------------
The NDCG@10 table summarizes how different reranking strategies perform on three business-relevant signals: click, like and long view. Each row is a single strategy (e.g., click-only, longview-only, weighted scalar, Pareto variants) and the numbers are mean NDCG@10 scores computed across users. Higher values mean the strategy tends to place relevant items for that objective higher in the ranked list. The table therefore makes the trade-offs explicit: strategies that prioritize clicks tend to have the highest click NDCG, while longview-focused strategies push up long-view NDCG at the expense of others.

Why Pareto is still useful
-------------------------
Even if a Pareto-based strategy does not produce the single highest mean NDCG for any one objective, it is valuable because it surfaces candidates that are not strictly worse across objectives. In practice that means the Pareto set gives a menu of balanced options rather than over-optimizing one metric. Ordering the Pareto set with a small secondary weight (the `pareto_weighted` variant) lets you nudge the list toward one objective while still keeping only non-dominated candidates — a pragmatic middle ground for multi-stakeholder decisions.

Assumptions
-----------
- The saved predictions and labels reflect the true offline log state and were generated from models and checkpoints recorded in `artifacts/`.
- User-grouping (for NDCG aggregation) is sensible and representative of the intended evaluation population.
- Preprocessing removed obvious leakage so features in `X.csv` are pre-exposure.

Limitations
-----------
- Offline NDCG does not measure downstream business impact — online A/B tests are required for deployment decisions.
- `is_like` is sparse in the dataset, so its NDCG estimates are noisy and small in absolute value.
- Candidate generation was not part of this study; results reflect reranking a fixed candidate pool (impressions log).
