# Creator Proxy Leakage Audit

Summary of automatic checks on processed dataset and registry.

## Key findings

- `is_follow` present in X: True
- `is_profile_enter` present in X: True
- `creator_proxy` present in X: False
- Targets accidentally present in X: []
- Post-exposure columns present in X: ['is_comment', 'is_forward', 'is_hate', 'play_time_ms', 'profile_stay_time', 'comment_stay_time']
- Duplicate rows (total): 0
- (user_id,video_id) pairs in multiple splits: 0
- Columns that exactly reveal `creator_proxy`: []

## Verdict

- **likely leakage found**

## Notes and recommendations

- The preprocessing was patched to drop `is_follow` and `is_profile_enter` from features; re-run `prepare_kuairand_data.py` to regenerate a cleaned processed dataset before further experiments.
- Consider removing `creator_proxy` from the training targets if it is trivially predictable or if it risks leaking from metadata.
- Recompute correlations and retrain baselines after regenerating processed data.

## Sample leaking (user_id,video_id) pairs across splits (up to 20)

- none found

## Where to find artifacts

- Correlations CSV: C:\Users\hamza\DLP\kuairand-like-prediction\artifacts\tables\creator_proxy_feature_correlations.csv
- Processed data: data/processed/X.csv and data/processed/y.csv
