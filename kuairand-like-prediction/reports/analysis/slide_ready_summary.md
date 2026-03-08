Slide-ready summary — KuaiRand-like multi-objective reranking
=============================================================

Project objective (2–3 lines)
- Build and evaluate a compact multi-task recommendation pipeline that predicts `is_click`, `is_like`, and `long_view` from KuaiRand-style impression logs.
- Compare single-objective and multi-objective reranking strategies (including Pareto-based methods) using NDCG@k.

Dataset summary (3–4 bullets)
- Processed impressions: train = 4578, val = 1526, test = 1526 (split indices in `data/processed/`).
- Targets: `is_click`, `is_like` (sparse; ≈239/3904 positives), `long_view` (high prevalence).
- Feature engineering: pre-exposure historical aggregates, temporal features, and train-set z-score scaling (scaler saved in `artifacts/feature_metadata/`).

Architecture summary (3–4 bullets)
- Simplified multi-task model: shared encoder + three heads (one per target) — fast to iterate and easy to debug.
- Baselines: per-target Logistic and LightGBM trained independently; LightGBM gives the strongest single-task baseline for `is_like`.
- Offline reranking performed on the impression candidate pool; candidate generation is out of scope.

Main baseline findings (3–4 bullets)
- LightGBM `is_like` PR-AUC = 0.303125 (best baseline for sparse likes).
- Logistic and LightGBM both report near-1.0 PR-AUC for `is_click` on this dataset (≈0.9997 and 0.99988 respectively).
- Baselines provide a robust reference for interpreting multi-task and reranking results.

Main reranking findings (3–4 bullets)
- `click_only` maximizes click NDCG@10: 0.99145 (mean across users).
- `longview_only` maximizes long_view NDCG@10: 0.67868.
- `like_only` slightly improves like NDCG@10 (0.03998) but substantially reduces long_view NDCG (≈0.5087).
- `pareto_weighted` and `weighted_scalar` provide balanced trade-offs; `pareto_weighted` yields click NDCG@10=0.99005 and long_view NDCG@10=0.67228.

Why Pareto matters (short)
- Pareto selection returns candidates that are not strictly worse across objectives; this preserves options with provable trade-offs.
- Ordering the Pareto set with a secondary weight (`pareto_weighted`) lets teams prioritize an objective while keeping multi-objective guarantees.

Limitations (short)
- Offline evaluation only — no online A/B validation.
- Small dataset (train ~4.6k) and sparse `is_like` signal (≈6.1% positive) limit generalization and stability.
- No candidate-generation stage; results evaluate reranking on the impression pool only.

Speaker notes (concise)
- We trained baselines (Logistic, LightGBM) and a compact multi-task model, then compared reranking strategies using NDCG@k aggregated by user. LightGBM gives a strong single-task baseline for likes; the multi-task model performs well on click and long_view but like remains challenging due to sparsity. Pareto-based reranking doesn't always yield the highest mean for a single metric, but it surfaces non-dominated candidates and supports fairer trade-offs when multiple stakeholders matter.

On-slide bullets (very short)
- Objective: evaluate reranking for click/like/long_view
- Dataset: train=4.6k / test=1.5k; `is_like` sparse
- Key result: click_only best for clicks (NDCG@10=0.99145); longview_only best for long views (0.67868)
- Pareto: balanced candidate set; `pareto_weighted` keeps strong click and good long_view
