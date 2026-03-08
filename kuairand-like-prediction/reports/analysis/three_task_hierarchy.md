# Three-Task Engagement Hierarchy

Targets used for multi-task training (final):

- `is_click`: immediate engagement indicator (clicked/engaged). Treated strictly as a prediction target and excluded from model features to avoid leakage.
- `is_like`: positive long-term signal (explicit positive feedback).
- `long_view`: proxy for continued attention (user watched for long fraction of video).

Design rules and rationale:

- `is_click` is excluded from features and any post-exposure columns (e.g., `play_time_ms`, `is_comment`, `is_forward`, `is_hate`, `profile_stay_time`, `comment_stay_time`) to prevent leakage from downstream user actions.
- `is_like` and `long_view` are downstream engagement tasks that may be correlated with `is_click` but are modeled jointly to learn shared representations; this supports multi-objective ranking.
- The three-task ordering is: first predict immediate engagement (`is_click`), then downstream signals (`is_like`, `long_view`). When deploying, use strict feature separation to avoid using any variables that are recorded after exposure.

Recommendation:

- Use the 3-task setup for offline exploration and Pareto reranking experiments only after ensuring no post-exposure leakage remains in the feature set. If any leakage persists, fallback to the 2-task setup (`is_like` + `long_view`).
