# Creator Proxy Derivation

Rule used to derive `creator_proxy`:

- If both `is_follow` and `is_profile_enter` are present in the raw interaction row, set `creator_proxy = 1` when either is 1 (logical OR). Otherwise `creator_proxy = 0`.
- If only one of those columns exists, use the available column as the proxy (i.e., `creator_proxy = is_follow` or `creator_proxy = is_profile_enter`).
- If neither column is present, `creator_proxy` is set to 0 for all rows (documented fallback).

Rationale:
- Following or entering a creator's profile are defensible signals of creator interest that do not directly reflect immediate content consumption (less likely to be a post-exposure label).
- Using an OR preserves sensitivity to either explicit follow events or profile visits when both are available.

Notes:
- This proxy is intended for modeling creator-level interest as a secondary objective in multi-task setups. It should be excluded from input features via the preprocessing registry (it is saved only as a target label).
- If you prefer a stricter proxy, consider requiring both signals or combining with creator-level aggregates.
