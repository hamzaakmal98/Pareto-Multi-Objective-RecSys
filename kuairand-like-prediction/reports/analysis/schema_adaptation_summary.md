# Schema Adaptation Summary

**Assumptions removed:**
- Removed assumption that only `is_like` would be present; supporting multiple engagement signals.
- Removed creator-interest placeholder `creator_interest`; using explicit `is_follow` when present or synthesized as proxy.

**Detected targets:**
- `is_like` — prevalence: 239/43028 (0.0056)

**Feature groups available:**
- Numeric columns: 11
- Categorical columns: 0

**Implication for model design:**
- Keep multi-task architecture (shared encoder + per-task towers) to jointly predict the detected engagement targets.
- Ensure all detected target columns are excluded from model inputs (handled by preprocessing registry).
- If `is_follow` is missing, consider synthesizing a creator-interest proxy from follows or creator-level aggregates.