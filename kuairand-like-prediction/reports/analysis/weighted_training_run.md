# Weighted Training Run

- best_epoch: 20
- train_rows: 4578
- val_rows: 1526

## Best validation metrics
- is_click: accuracy=0.9855832241153342, auc=0.8517254771415889, pr_auc=0.9966115390972482
- is_like: accuracy=0.9770642201834863, auc=0.6004215770815369, pr_auc=0.07294452423305936
- long_view: accuracy=0.9344692005242464, auc=0.9856363247745388, pr_auc=0.9857413955233557

## Test metrics (best checkpoint)
- is_click: roc_auc=0.8215597789927391, pr_auc=0.9963813120082162
- is_like: roc_auc=0.5377220906710157, pr_auc=0.057604364398194635
- long_view: roc_auc=0.9860913646026006, pr_auc=0.9862033610184631

## Notes
- Per-task loss weights used: {"is_click": 1.0, "is_like": 1.0, "long_view": 1.0}
- Early stopping patience: 3