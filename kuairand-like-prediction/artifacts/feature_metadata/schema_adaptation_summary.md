# Schema Adaptation Summary

**Assumptions removed:**
- Removed assumption that only `is_like` would be present; supporting multiple engagement signals.
- Removed creator-interest placeholder `creator_interest`; using explicit `is_follow` when present or synthesized as proxy.

**Detected targets:**
- `is_like` — prevalence: 239/3904 (6.1219%)
- `long_view` — prevalence: 3621/3904 (92.7510%)
- `creator_proxy` — prevalence: 205/3904 (5.2510%)

**Feature groups available:**
- Numeric columns: 11
- Categorical columns: 0

**Implication for model design:**
- Keep multi-task architecture (shared encoder + per-task towers) to jointly predict the detected engagement targets.
- Ensure all detected target columns are excluded from model inputs (handled by preprocessing registry).
- `creator_proxy` is derived where possible and included as a target. See creator_proxy_note.md for the derivation rule.