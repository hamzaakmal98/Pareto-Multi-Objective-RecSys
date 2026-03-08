# Pareto Reranking Method

This document describes the Pareto reranking approach implemented in `src/rerank` and the runner `scripts/run_pareto_rerank.py`.

1. Candidate generation
- For each objective (e.g., `like_score`, `longview_score`, `creator_score`) we take the top-N items per user.
- The per-objective top-N lists are unioned into a candidate pool and deduplicated per `(user_id, video_id)`. For deduplication we keep the maximum observed score per objective.

2. Scalarized baseline
- Candidates can be ranked by a scalarized score: min-max normalize each objective score and compute a weighted sum using configurable weights. This provides a simple baseline for multi-objective ranking.

3. Pareto frontier extraction
- For each user, we treat each candidate as a point in objective space (e.g., [like_score, longview_score, creator_score]).
- Non-dominated sorting (Pareto frontier) returns items not dominated by any other candidate: an item A is dominated if there exists item B with scores >= A in all objectives and > in at least one.

4. Frontier ordering and tie-breaking
- The frontier is a set; to produce a deterministic ranked list we optionally order frontier items by a secondary score (e.g., `like_score`) or by a provided tie-breaker.

5. Outputs
- Candidate pool: `artifacts/tables/pareto_candidate_pool.csv`
- Scalarized candidates: `artifacts/tables/scalarized_candidates.csv`
- Pareto frontier: `artifacts/tables/pareto_frontier.csv`
- Reranked lists: `artifacts/tables/pareto_reranked.csv`
- Metrics: `reports/metrics/pareto_metrics.json` and `reports/analysis/pareto_analysis.md`

6. Evaluation
- Evaluate ranking quality with NDCG@K, Precision@K, Recall@K across users (averaged across users with at least one positive). The script evaluates single-objective ranking, scalarized ranking, and Pareto frontier outputs for comparison.

7. Notes and extensibility
- The implementation is intentionally modular: candidate generation, scalarization, Pareto extraction, and evaluation are separate functions and easy to extend.
- Future extensions: multi-frontier extraction (k-level Pareto layers), constrained Pareto selection (e.g., fairness or exposure constraints), and interactive visualization of frontier tradeoffs.
