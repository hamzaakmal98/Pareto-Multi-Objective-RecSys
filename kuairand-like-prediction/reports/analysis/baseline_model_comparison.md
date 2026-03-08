# Baseline Model Comparison

| Model | Task | ROC-AUC | PR-AUC | Accuracy |
|---|---:|---:|---:|---:|
| Logistic | is_click | 0.9779 | 0.9997 | 0.9849 |
| Logistic | is_like | 0.5832 | 0.0512 | 0.9718 |
| Logistic | long_view | 0.9760 | 0.9790 | 0.9286 |
| LightGBM | is_click | 0.9923 | 0.9999 | 0.9889 |
| LightGBM | is_like | 0.7660 | 0.3031 | 0.9712 |
| LightGBM | long_view | 0.9999 | 0.9999 | 0.9961 |