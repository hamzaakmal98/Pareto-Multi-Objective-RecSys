
Design notes — kuairand-like-prediction
=====================================

Overview
--------
This document captures the final design and reasoning for the repository: why the project uses a multi-objective setup, how the simplified multi-task architecture maps to a full MMoE design, the end-to-end flow, and why Pareto reranking and NDCG evaluation were chosen.

1) Problem statement
--------------------
Recommend candidate videos to users while balancing multiple engagement objectives (clicks, likes, long views). The core engineering challenge is to train models and evaluate reranking strategies that make explicit trade-offs between competing objectives rather than optimizing a single scalar surrogate.

2) Why multi-objective recommendation
-------------------------------------
- Multiple downstream signals capture different notions of success: clicks capture short-term attention, likes indicate user endorsement, and long views reflect sustained engagement.
- Optimizing for any single signal can harm others (e.g., click-bait that reduces long_view). A multi-objective setup enables systematic exploration of Pareto trade-offs and re-ranking strategies that consider multiple predicted objectives.

3) Why this architecture is a simplified alternative to full MMoE
----------------------------------------------------------------
- Full MMoE (Mixture-of-Experts with task-specific gates) is an expressive multi-task architecture. For a focused research prototype we implemented a lightweight, reproducible design:
	- Shared encoder: a compact feed-forward network that learns common representations across tasks.
	- Task heads: independent final layers for each objective (`is_click`, `is_like`, `long_view`).
- Rationale for simplification:
	- Faster iteration and smaller compute/engineering footprint for local experiments.
	- Easier to debug leakage and per-task behavior when heads are simple and isolated.
	- The simplified model captures the core multi-task idea (shared representation + task-specific outputs) while leaving MMoE gating/experimentation as a natural next step.

4) End-to-end system flow
-------------------------
The pipeline implemented in the repo follows these stages:

- Raw KuaiRand data
	- Per-impression CSV logs with timestamp, `user_id`, `video_id`, and other metadata.

- Preprocessing
	- `scripts/prepare_kuairand_data.py` normalizes timestamps, derives temporal features, removes leakage columns (post-exposure signals and target proxies), and splits the data into train/val/test indices.
	- Numeric columns are z-score standardized using train-set statistics; scaler saved to `artifacts/feature_metadata/scaler.json`.

- Feature engineering
	- Lightweight derived features (e.g., `hour_of_day`, `day_of_week`, `duration_s`, historical aggregates computed strictly from pre-exposure events).
	- All features are validated to ensure they are available before exposure (no candidate-level click/watch features in `X`).

- Multi-task prediction
	- A shared encoder produces a common embedding which is fed to three task-specific heads producing logits/scores for `is_click`, `is_like`, and `long_view`.
	- Training uses per-task BCEWithLogits losses and early stopping. Best checkpoint and predictions are saved in `artifacts/models/` and `artifacts/predictions/`.

- Ranking strategies
	- Candidate-level predicted scores are combined into final ranked lists per strategy (click-only, like-only, longview-only, weighted scalar).

- Pareto reranking
	- The Pareto frontier is computed across predicted objectives to find non-dominated candidates.
	- Two approaches used: flattening Pareto layers into a ranking, and a secondary weighted ordering of Pareto candidates (`pareto_weighted`).

- NDCG evaluation
	- Ranked lists are evaluated by NDCG@5/10/20 aggregated across users; canonical CSVs saved to `artifacts/tables/`.

5) Final task setup
-------------------
The project uses three final targets (labels):
- `is_click` — whether the user clicked on the impression (used as a task in multi-task training and in ranking evaluation).
- `is_like` — whether the user liked the item (sparse; difficult to predict).
- `long_view` — whether the impression resulted in a long watch.

6) Why candidate generation was omitted
-------------------------------------
- This repo focuses on modeling, re-ranking, and multi-objective evaluation for a pre-defined candidate pool (the impression set in the logs). Candidate generation (retrieval, nearest-neighbors, or large-scale candidate scoring) is outside scope because:
	- It introduces additional system complexity and infrastructure requirements (indexing, ANN stores) that distract from the core multi-objective evaluation.
	- The research questions here concern trade-offs across objectives for the same candidate set; those questions can be answered without a separate retrieval stage.

7) Why Pareto ranking is useful
------------------------------
- Pareto-efficient selection identifies candidates that are not strictly worse on all objectives — i.e., those offering non-dominated trade-offs.
- For multi-stakeholder or multi-objective products, Pareto-based reranking surfaces items that offer balanced trade-offs and avoids over-optimizing a single metric.
- Secondary weighted ordering of Pareto candidates (the `pareto_weighted` strategy) lets practitioners prioritize one objective among the Pareto set while preserving multi-objective filtering.

8) Why NDCG is used
--------------------
- NDCG (Normalized Discounted Cumulative Gain) measures quality of ranked lists while discounting lower positions and allowing graded relevance.
- For recommendations, top-ranked positions matter most; NDCG@k captures this and is robust to item popularity skew when aggregated by user.

Mermaid architecture diagram
---------------------------
Below is a Mermaid diagram that reflects the final architecture and artifact flow used in this repo.

```mermaid
flowchart TD
	subgraph RAW[Raw data]
		A[KuaiRand CSVs] --> B[Raw impressions]
	end
	B --> C[Preprocessing: scripts/prepare_kuairand_data.py]
	C --> D[data/processed/X.csv]
	C --> E[data/processed/y.csv]
	C --> F[data/processed/meta.csv]
	C --> G[artifacts/feature_metadata/scaler.json]

	D --> H[Shared encoder (PyTorch)]
	H --> I[Click head]
	H --> J[Like head]
	H --> K[Long_view head]

	I --> L[pred_score_is_click]
	J --> M[pred_score_is_like]
	K --> N[pred_score_long_view]

	L --> O[Ranking strategies]
	M --> O
	N --> O

	subgraph RANKS[Ranking strategies]
		O --> O1[click_only]
		O --> O2[like_only]
		O --> O3[longview_only]
		O --> O4[weighted_scalar]
		O --> O5[pareto_frontier]
		O --> O6[pareto_weighted]
	end

	O5 --> P[Pareto selection]
	P --> O6

	O1 --> Q[NDCG@k evaluation]
	O2 --> Q
	O3 --> Q
	O4 --> Q
	O5 --> Q
	O6 --> Q

	Q --> R[artifacts/tables/pareto_ranking_results.csv]
	Q --> S[reports/analysis/pareto_ranking_results.md]
```

Notes
-----
- The diagram intentionally excludes candidate-generation/retrieval and online serving subsystems; this repo is experimental and focuses on offline dataset creation, multi-task training, reranking strategies, and evaluation.

Reference artifacts
-------------------
- `artifacts/tables/baseline_metrics.csv`
- `reports/analysis/weighted_training_run.md`
- `artifacts/tables/pareto_ranking_results.csv`

If you want the diagram exported to a PNG/SVG for slides, I can generate it from the Mermaid source and add the image to `reports/figures/`.

