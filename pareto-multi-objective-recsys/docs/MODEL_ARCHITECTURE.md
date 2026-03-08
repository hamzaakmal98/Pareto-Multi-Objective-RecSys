# Model Architecture — Custom MMoE-inspired Multi-Task Recommender

This document summarizes the custom multi-task architecture implemented in `src/models`.

Overview
- Inputs are grouped by semantic type: user categorical/id features, item categorical/id features, numeric features, and context/temporal features.
- The pipeline first encodes inputs into a shared latent representation via `SharedEncoder`.
- A custom expert system (in `CustomMMoE`) mixes expert outputs per task using task-specific gates.
- Task-specific towers produce final logits for `like`, `longview`, and (optionally) `creator` predictions.

Encoder
- Categorical and id-like columns are represented with learned embeddings. Each categorical column has its own embedding table; dimensions are configurable.
- Numeric features are normalized with `LayerNorm` and projected via a small feed-forward block.
- The encoder concatenates embedded categorical vectors and numeric projections, then projects the concat into a shared `projection_dim` latent vector. This shared vector is the input to the expert system.

Custom expert system
- `n_experts` small MLP experts process the shared representation in parallel.
- Each task (like, longview, creator) has a learned gate (linear layer) that outputs a softmax over experts; the gate weights produce a convex mixture of expert outputs per example.
- The resulting task-specific mixture is optionally combined with a task-specific residual projection of the shared representation; this enables task-adaptive routing beyond pure expert mixtures.
- Each task mixture is passed to a compact task tower (MLP) producing a scalar logit.

Design choices and rationale
- Sharing: the encoder learns a compact shared representation that captures common signal across tasks while expert mixtures allow specialization where needed.
- Task gates: soft mixtures avoid hard routing and remain differentiable while enabling task-specific focus on different experts.
- Residual routing: provides a lightweight task-adaptive path when expert outputs alone are insufficient.
- Configurability: number of experts, hidden sizes, and whether to enable residuals are configurable in `configs/model.yaml`.

Training and losses
- Each task uses `BCEWithLogitsLoss` (binary tasks). Loss contributions are weighted via configurable `loss_weights`.
- The training loop supports early stopping, checkpointing of the best model, and concise logging of epoch-level train/validation loss and per-task loss breakdown.

Artifacts saved by training
- Best model checkpoint (encoder, mmoe, heads state dicts)
- `training_history.json` with train/validation loss per epoch
- Config used for the run (saved alongside checkpoints)

Explainability
- The architecture is explainable: gates reveal which experts each task used on average; expert outputs and residuals can be inspected per-example.
- The separation between encoder, experts, gates, and towers makes the flow of information explicit for analysis and presentation.

Notes
- This model is intentionally richer than a single MLP but remains trainable on commodity GPUs for moderate-sized datasets.
- The design favors interpretability and modularity so that reranking, NDCG evaluation, and Pareto analysis can be performed using the same learned predictions.
