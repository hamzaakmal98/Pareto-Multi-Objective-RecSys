# Final results — summary

This file consolidates the canonical metrics used in the project's analysis. All numbers are taken directly from saved artifacts in `artifacts/tables/` and `reports/analysis/`.

1) Baseline metrics (source: `artifacts/tables/baseline_metrics.csv` / `reports/analysis/baseline_model_comparison.md`)

| Model | Task | ROC-AUC | PR-AUC | Accuracy |
|---|---:|---:|---:|---:|
| Logistic | is_click | 0.9779282015678786 | 0.9996631834915211 | 0.9849279161205766 |
| Logistic | is_like | 0.5832300961282129 | 0.05124643288998091 | 0.971821756225426 |
| Logistic | long_view | 0.9760304226596362 | 0.9790126510177042 | 0.9285714285714286 |
| LightGBM | is_click | 0.9922618531053835 | 0.9998812351548039 | 0.9888597640891219 |
| LightGBM | is_like | 0.7659677899920023 | 0.30312505727387185 | 0.9711664482306684 |
| LightGBM | long_view | 0.9998930237694281 | 0.999886299821057 | 0.9960681520314548 |

2) Multi-task (weighted) training — best checkpoint (source: `reports/analysis/weighted_training_run.md`)

- Best epoch: 20
- Train rows: 4578, Val rows: 1526

Best validation metrics (from `weighted_training_run.md`):
- `is_click`: accuracy=0.9855832241153342, auc=0.8517254771415889, pr_auc=0.9966115390972482
- `is_like`: accuracy=0.9770642201834863, auc=0.6004215770815369, pr_auc=0.07294452423305936
- `long_view`: accuracy=0.9344692005242464, auc=0.9856363247745388, pr_auc=0.9857413955233557

Test metrics (best checkpoint):
- `is_click`: roc_auc=0.8215597789927391, pr_auc=0.9963813120082162
- `is_like`: roc_auc=0.5377220906710157, pr_auc=0.057604364398194635
- `long_view`: roc_auc=0.9860913646026006, pr_auc=0.9862033610184631

3) Pareto / ranking evaluation (source: `artifacts/tables/pareto_ranking_results.csv` and `artifacts/tables/pareto_weighted_results.csv`)

- `click_only` produced NDCG@5 for `is_click` = 0.9914812310000376 (mean across users).
- `weighted_scalar` achieved `long_view` NDCG@20 = 0.6754049387402925.
- `pareto_weighted` (secondary weighted ordering of Pareto front) yields `long_view` NDCG@20 = 0.6736256922589338 and `is_click` NDCG@5 = 0.9890577647563572.

Files used as canonical sources
--------------------------------
- `artifacts/tables/baseline_metrics.csv`
- `reports/analysis/baseline_model_comparison.md`
- `reports/analysis/weighted_training_run.md`
- `artifacts/tables/pareto_ranking_results.csv`
- `artifacts/tables/pareto_weighted_results.csv`

Notes and interpretation
------------------------
- The LightGBM baseline provides strong single-task baselines, notably for `is_like` (PR-AUC 0.3031).
- The multi-task neural run attains high PR-AUC for `is_click` and `long_view` in test predictions, while `is_like` remains challenging (low positive prevalence).
- Pareto-based reranking with a secondary weighted ordering (`pareto_weighted`) offers competitive long-view ranking performance while retaining reasonable click NDCG.

Next actions (recommended)
-------------------------
- Run additional scalar weight sweeps for the ranking stage and report statistical comparisons (paired tests) across users.
- Add a short notebook that reproduces the Pareto ranking figures for presentation slides.
---

## Final baseline comparison

Canonical per-task baseline metrics (source: `artifacts/tables/baseline_metrics.csv`)

| Model | Task | ROC-AUC | PR-AUC | Accuracy |
|---|---:|---:|---:|---:|
| Logistic | is_click | 0.9779282015678786 | 0.9996631834915211 | 0.9849279161205766 |
| Logistic | is_like | 0.5832300961282129 | 0.05124643288998091 | 0.971821756225426 |
| Logistic | long_view | 0.9760304226596362 | 0.9790126510177042 | 0.9285714285714286 |
| LightGBM | is_click | 0.9922618531053835 | 0.9998812351548039 | 0.9888597640891219 |
| LightGBM | is_like | 0.7659677899920023 | 0.30312505727387185 | 0.9711664482306684 |
| LightGBM | long_view | 0.9998930237694281 | 0.999886299821057 | 0.9960681520314548 |

**Key baseline takeaways**
- LightGBM is a strong single-task baseline, notably for `is_like` (PR-AUC = 0.3031).
- Logistic provides a simple, interpretable comparison point; both baselines achieve very high PR-AUC on `is_click` (≈0.9997).

## Final multi-task model summary

Source: `reports/analysis/weighted_training_run.md` (best checkpoint)

- Best epoch: 20
- Train rows: 4578; Val rows: 1526

Best validation metrics (selected):
- `is_click`: accuracy=0.9855832241153342, auc=0.8517254771415889, pr_auc=0.9966115390972482
- `is_like`: accuracy=0.9770642201834863, auc=0.6004215770815369, pr_auc=0.07294452423305936
- `long_view`: accuracy=0.9344692005242464, auc=0.9856363247745388, pr_auc=0.9857413955233557

Test metrics (best checkpoint):
- `is_click`: roc_auc=0.8215597789927391, pr_auc=0.9963813120082162
- `is_like`: roc_auc=0.5377220906710157, pr_auc=0.057604364398194635
- `long_view`: roc_auc=0.9860913646026006, pr_auc=0.9862033610184631

**Interpretation**: the multi-task model achieves very high PR-AUC on `is_click` and `long_view` in test predictions, while `is_like` remains difficult (low PR-AUC), consistent with its low prevalence.

## Pareto reranking comparison

Source: `artifacts/tables/pareto_ranking_results.csv` and `artifacts/tables/pareto_weighted_results.csv`.

Main NDCG@10 table (mean across users)

| Strategy | NDCG@10 (`is_click`) | NDCG@10 (`is_like`) | NDCG@10 (`long_view`) |
|---|---:|---:|---:|
| click_only | 0.9914496501106276 | 0.038442201535491424 | 0.6648164191847047 |
| like_only | 0.9873266550422473 | 0.03997893604465119 | 0.5087101538967954 |
| longview_only | 0.9909237944792726 | 0.03821201756271286 | 0.6786826239602339 |
| weighted_scalar | 0.9904696214595484 | 0.03719708377734195 | 0.6752954623752686 |
| pareto_frontier | 0.9898398884207323 | 0.037798344441099654 | 0.6688888264836388 |
| pareto_weighted | 0.9900522483620959 | 0.03750250342685043 | 0.6722804318801154 |

Short interpretation
- `click_only` maximizes click NDCG (≈0.99145) but yields moderate `long_view` NDCG (≈0.6648).
- `like_only` favors `is_like` NDCG (small absolute improvements) but substantially reduces `long_view` performance (≈0.509).
- `longview_only` maximizes `long_view` NDCG (≈0.6787) while keeping `is_click` near top baselines.
- `weighted_scalar` and `pareto_weighted` trade slightly in favor of `long_view` while retaining high `is_click` NDCG; `weighted_scalar` achieves the highest `long_view` NDCG@10 (≈0.6753).

Discussion of trade-offs
- `click_only`: best for click-centric objectives; small sacrifice in `long_view` compared to `longview_only`.
- `like_only`: improves `is_like` very slightly (absolute NDCG values are small due to sparsity) at the expense of `long_view`.
- `longview_only`: best for long engagement; preserves click performance close to `click_only`.
- `weighted_scalar`: when weights are chosen to emphasize `long_view`, this strategy achieves the best `long_view` NDCG@10 (0.6753) while keeping `is_click` high.
- `pareto_frontier`: surfaces non-dominated candidates; in our experiments it provides balanced performance but not the highest `long_view` or `click` NDCG.
- `pareto_weighted`: orders Pareto candidates by a secondary weighted score; it improves `long_view` NDCG over `pareto_frontier` while keeping `is_click` close to top.

Why Pareto helps
- Pareto selection filters out candidates that are strictly dominated across objectives, yielding a candidate set with provably non-worse trade-offs.
- Secondary ordering (e.g., `pareto_weighted`) lets practitioners impose preference within that non-dominated set, combining fairness across objectives with a tunable prioritization.

Limitations and caveats
- `is_like` is low-prevalence (reported positives ≈239/3904 in EDA), so NDCG and PR-AUC estimates for likes are noisy and small in absolute terms.
- Experiments are offline and limited in dataset size (train ≈4.6k rows); online A/B or larger-scale runs are needed to validate deployment impact.
- Pareto methods assume the candidate pool is representative; without candidate generation this evaluation reflects re-ranking quality on the impression pool only.

Artifacts referenced
- `artifacts/tables/baseline_metrics.csv`
- `reports/analysis/weighted_training_run.md`
- `artifacts/tables/pareto_ranking_results.csv`
- `artifacts/tables/pareto_weighted_results.csv`

If you want, I can produce a single-slide summary PNG that contains the NDCG@10 table and a short bullet summary suitable for presentation. Would you like that? 
