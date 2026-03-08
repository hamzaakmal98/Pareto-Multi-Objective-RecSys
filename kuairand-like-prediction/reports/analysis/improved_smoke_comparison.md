# Improved Smoke Run Comparison

## Experimental setup
- Previous (baseline smoke): scaled features, lr=1e-3, batch_size=256, no positive weighting for `is_like`.
- Improved smoke: scaled features, lr=5e-4, batch_size=128, positive class weighting applied for `is_like`.

## Rows used
- train: 2342
- val: 781
- test: 781

## Train loss (per epoch)
- Previous: epoch1=0.633953, epoch2=0.556891, epoch3=0.480012
- Improved: epoch1=0.860533, epoch2=0.811884, epoch3=0.757360

## `is_like` (primary sparse target)
- Previous: AUC=0.53120, PR-AUC=0.06418
- Improved: AUC=0.75834, PR-AUC=0.31028
- Effect: substantial improvement in both AUC and PR-AUC after applying positive weighting and tuning LR/batch. This indicates the sparse `is_like` signal is better captured.

## `long_view`
- Previous: AUC=0.49285, PR-AUC=0.92824
- Improved: AUC=0.53346, PR-AUC=0.93426
- Effect: small improvement in AUC and PR-AUC remains high; signal already strong.

## `creator_proxy`
- Previous: AUC=0.95469, PR-AUC=0.92528
- Improved: AUC=0.87610, PR-AUC=0.46763
- Effect: AUC dropped, PR-AUC decreased — likely due to changed weighting and batch/LR; may need per-task loss weighting or task-specific tuning.

## Training stability
- Previous: training loss decreased steadily to 0.48 (stable).
- Improved: training loss decreased steadily from 0.86 → 0.76 (stable but higher absolute loss because of positive weighting and different hyperparams).

## Recommendation
- Ready for longer training: Yes, with caution. The pipeline is end-to-end functional; apply further hyperparameter tuning and task loss weighting before large runs.
- Baseline comparison: Yes — run baseline models (logistic / lightgbm) on the same processed dataset for quick benchmarking.
- Pareto reranking experiments: Proceed after a stable, higher-quality multi-task model is available (recommend after longer training and hyperparameter sweeps).

## Next suggested actions
- Run a 10-epoch training with per-task loss weights (increase weight for `is_like`), monitor validation PR-AUC.
- Add per-task learning rates or heads with separate optimizers if necessary.
- Produce baseline metrics from LightGBM / Logistic for robust comparison.

