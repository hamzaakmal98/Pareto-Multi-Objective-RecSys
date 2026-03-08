# Final tables



## Baseline model comparison

Source: `artifacts/tables/baseline_metrics.csv`



| Model | Task | ROC-AUC | PR-AUC | Accuracy |
| --- | --- | --- | --- | --- |
| Logistic | is_click | 0.977928 | 0.999663 | 0.984928 |
| Logistic | is_like | 0.583230 | 0.051246 | 0.971822 |
| Logistic | long_view | 0.976030 | 0.979013 | 0.928571 |
| LightGBM | is_click | 0.992262 | 0.999881 | 0.988860 |
| LightGBM | is_like | 0.765968 | 0.303125 | 0.971166 |
| LightGBM | long_view | 0.999893 | 0.999886 | 0.996068 |



## Final reranking (NDCG@10)

Source: `artifacts/tables/pareto_ranking_results.csv`



| Method | Click | Like | LongView | Mean |
| --- | --- | --- | --- | --- |
| click_only | 0.991450 | 0.038442 | 0.664816 | 0.564903 |
| like_only | 0.987327 | 0.039979 | 0.508710 | 0.512005 |
| longview_only | 0.990924 | 0.038212 | 0.678683 | 0.569273 |
| weighted_scalar | 0.990470 | 0.037197 | 0.675295 | 0.567654 |
| pareto_frontier | 0.989840 | 0.037798 | 0.668889 | 0.565509 |
| pareto_weighted | 0.990052 | 0.037503 | 0.672280 | 0.566612 |



## Best strategy per objective

| Objective | BestMethod | NDCG@10 |
| --- | --- | --- |
| Click | click_only | 0.991450 |
| Like | like_only | 0.039979 |
| LongView | longview_only | 0.678683 |



**Notes**: Tables derived from saved artifacts; no models were re-run.