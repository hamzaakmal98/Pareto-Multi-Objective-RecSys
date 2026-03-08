import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import sys
import json


def infer_time_col(df, hint=None):
    candidates = []
    if hint:
        candidates.append(hint)
    candidates += ['timestamp', 'ts', 'time', 'created_at', 'created', 'date', 'datetime']
    for c in candidates:
        if c in df.columns:
            return c
    return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input', type=str, required=False,
                   default='real_data/KuaiRand-1K/data/log_random_4_22_to_5_08_1k.csv')
    p.add_argument('--time-col', type=str, default=None)
    p.add_argument('--targets', type=str, default='is_click,is_like,long_view')
    p.add_argument('--out', type=str, default='data/processed')
    args = p.parse_args()

    inp = Path(args.input)
    if not inp.exists():
        print(f'Input file not found: {inp}', file=sys.stderr)
        sys.exit(2)

    df = pd.read_csv(inp)

    time_col = infer_time_col(df, args.time_col)
    if time_col is None:
        print('No time column detected. Please provide --time-col', file=sys.stderr)
        sys.exit(2)

    print(f'Using time column: {time_col}')
    df[time_col] = pd.to_datetime(df[time_col], errors='coerce')
    if df[time_col].isna().all():
        print('All parsed times are NaT. Check --time-col', file=sys.stderr)
        sys.exit(2)

    targets = [t.strip() for t in args.targets.split(',') if t.strip()]
    for t in targets:
        if t not in df.columns:
            df[t] = 0

    # Identify potential leakage columns (follow/profile) so they can be removed from features
    follow_col = 'is_follow' if 'is_follow' in df.columns else None
    profile_col = 'is_profile_enter' if 'is_profile_enter' in df.columns else None

    # keep only rows that have at least one positive in targets
    df_pos = df[df[targets].sum(axis=1) > 0].copy()
    if df_pos.empty:
        print('No rows with positive targets found after filtering. Exiting.', file=sys.stderr)
        sys.exit(2)

    # assign week buckets
    df_pos['week_start'] = df_pos[time_col].dt.to_period('W').apply(lambda p: p.start_time)
    weeks = sorted(df_pos['week_start'].dropna().unique())
    if len(weeks) >= 3:
        n = len(weeks)
        n_train = max(1, int(np.ceil(0.6 * n)))
        n_val = max(1, int(np.ceil(0.2 * n)))
        # remainder for test
        n_test = n - n_train - n_val
        if n_test < 1:
            n_test = 1
            if n_train + n_val + n_test > n:
                # adjust n_val
                n_val = max(1, n - n_train - n_test)

        train_weeks = weeks[:n_train]
        val_weeks = weeks[n_train:n_train + n_val]
        test_weeks = weeks[n_train + n_val: n_train + n_val + n_test]

        df_pos['split'] = 'test'
        df_pos.loc[df_pos['week_start'].isin(train_weeks), 'split'] = 'train'
        df_pos.loc[df_pos['week_start'].isin(val_weeks), 'split'] = 'val'
    else:
        # fallback: not enough distinct weeks — use chronological 60/20/20 split by timestamp
        df_pos = df_pos.sort_values(by=time_col).reset_index(drop=True)
        nrows = len(df_pos)
        cut1 = int(0.6 * nrows)
        cut2 = int(0.8 * nrows)
        df_pos['split'] = 'test'
        df_pos.loc[:cut1 - 1, 'split'] = 'train'
        df_pos.loc[cut1:cut2 - 1, 'split'] = 'val'

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    # build X (features) and y (targets)
    # drop obvious post-exposure / leakage columns from features
    post_exposure = ['is_click', 'is_comment', 'is_forward', 'is_hate', 'play_time_ms', 'profile_stay_time', 'comment_stay_time']

    # ensure df_pos index is contiguous so split masks align with saved indices
    df_pos = df_pos.reset_index(drop=True)

    # Create time-derived features if a parsed time column exists
    if time_col in df_pos.columns:
        try:
            ts = pd.to_datetime(df_pos[time_col], errors='coerce')
            if not ts.isna().all():
                df_pos['hour_of_day'] = ts.dt.hour
                df_pos['day_of_week'] = ts.dt.dayofweek
                # normalized time index in [0,1]
                tmin = ts.min()
                tmax = ts.max()
                denom = (tmax - tmin).total_seconds() if pd.notna(tmin) and pd.notna(tmax) and tmax > tmin else None
                if denom:
                    df_pos['time_norm'] = (ts - tmin).dt.total_seconds() / denom
                else:
                    df_pos['time_norm'] = 0.0
        except Exception:
            pass

    # convert millisecond durations to seconds if present
    if 'duration_ms' in df_pos.columns:
        df_pos['duration_s'] = pd.to_numeric(df_pos['duration_ms'], errors='coerce').fillna(0.0) / 1000.0
    if 'play_time_ms' in df_pos.columns:
        df_pos['play_time_s'] = pd.to_numeric(df_pos['play_time_ms'], errors='coerce').fillna(0.0) / 1000.0

    # determine columns to drop from features (targets, post-exposure, week/time raw)
    # Also drop any columns that were used to derive targets (e.g. is_follow, is_profile_enter)
    extra_leakage_cols = [c for c in [follow_col, profile_col] if c]
    drop_cols = list(dict.fromkeys(targets + post_exposure + extra_leakage_cols + ['week_start', time_col, 'time_ms', 'duration_ms', 'play_time_ms']))
    # build X and y
    X = df_pos.drop(columns=[c for c in drop_cols if c in df_pos.columns]).copy()
    y = df_pos[targets].copy()

    # Remove raw id columns from features (keep in meta)
    id_cols = [c for c in ['user_id', 'video_id'] if c in X.columns]
    for c in id_cols:
        X.drop(columns=[c], inplace=True, errors='ignore')

    # Standardize numeric features using statistics computed on the training split
    train_mask = df_pos['split'] == 'train'
    numeric_cols = X.select_dtypes(include=[float, int, 'number']).columns.tolist()
    # exclude any boolean-like targets accidentally present
    numeric_cols = [c for c in numeric_cols if c not in targets]
    scaler = {}
    if len(numeric_cols) > 0 and train_mask.any():
        means = X.loc[train_mask, numeric_cols].mean()
        stds = X.loc[train_mask, numeric_cols].std().replace(0, 1.0)
        X[numeric_cols] = (X[numeric_cols] - means) / stds
        scaler = {c: {'mean': float(means[c]), 'std': float(stds[c])} for c in numeric_cols}

    # Save scaler metadata
    feat_meta_dir = Path(outdir) / '../artifacts' / 'feature_metadata'
    feat_meta_dir.mkdir(parents=True, exist_ok=True)
    (feat_meta_dir / 'scaler.json').write_text(json.dumps(scaler, indent=2), encoding='utf-8')

    # Update and save feature registry (reflect cleaned feature groups)
    numeric_list = numeric_cols
    categorical_list = [c for c in X.columns if c not in numeric_list]
    fr = {
        'NUMERIC_COLUMNS': numeric_list,
        'CATEGORICAL_COLUMNS': categorical_list,
        'TARGET_COLUMNS': targets,
    }
    (feat_meta_dir / 'feature_registry.json').write_text(json.dumps(fr, indent=2), encoding='utf-8')

    # Also write feature metadata to repo-level artifacts directory for consistency
    repo_root = Path(outdir).resolve().parents[1]
    repo_feat_meta = repo_root / 'artifacts' / 'feature_metadata'
    repo_feat_meta.mkdir(parents=True, exist_ok=True)
    (repo_feat_meta / 'feature_registry.json').write_text(json.dumps(fr, indent=2), encoding='utf-8')

    # Compute and save target prevalence
    preval = {}
    for t in targets:
        try:
            vals = y[t].dropna().astype(float)
            preval[t] = {"n_positive": int(vals.sum()), "n_total": int(len(vals)), "positive_rate": float(vals.sum() / len(vals) if len(vals) else 0.0)}
        except Exception:
            preval[t] = {"n_positive": 0, "n_total": 0, "positive_rate": 0.0}
    (feat_meta_dir / 'target_prevalence.json').write_text(json.dumps(preval, indent=2), encoding='utf-8')
    (repo_feat_meta / 'target_prevalence.json').write_text(json.dumps(preval, indent=2), encoding='utf-8')

    # ensure index is unique and stable
    X.reset_index(drop=True, inplace=True)
    y.reset_index(drop=True, inplace=True)

    # save files
    X.to_csv(outdir / 'X.csv')
    y.to_csv(outdir / 'y.csv')

    # save split indices (positions)
    train_idx = (df_pos['split'] == 'train').to_numpy().nonzero()[0]
    val_idx = (df_pos['split'] == 'val').to_numpy().nonzero()[0]
    test_idx = (df_pos['split'] == 'test').to_numpy().nonzero()[0]
    np.savetxt(outdir / 'train_idx.csv', train_idx, fmt='%d', delimiter=',')
    np.savetxt(outdir / 'val_idx.csv', val_idx, fmt='%d', delimiter=',')
    np.savetxt(outdir / 'test_idx.csv', test_idx, fmt='%d', delimiter=',')

    # summary
    print('Split counts:')
    print(' train:', len(train_idx))
    print(' val:  ', len(val_idx))
    print(' test: ', len(test_idx))
    for t in targets:
        print(f"Target {t}: train_pos={int(y.loc[train_idx][t].sum())} val_pos={int(y.loc[val_idx][t].sum())} test_pos={int(y.loc[test_idx][t].sum())}")

    # Save processed X/y/meta
    X.to_csv(outdir / 'X.csv')
    y.to_csv(outdir / 'y.csv')

    # Save meta including id columns and timestamp
    meta = df_pos[[c for c in ['user_id', 'video_id', 'week_start', time_col] if c in df_pos.columns]].copy()
    meta.to_csv(outdir / 'meta.csv', index=False)

    # save split indices as CSV
    pd.Series(train_idx).to_csv(outdir / 'train_idx.csv', index=False, header=False)
    pd.Series(val_idx).to_csv(outdir / 'val_idx.csv', index=False, header=False)
    pd.Series(test_idx).to_csv(outdir / 'test_idx.csv', index=False, header=False)


if __name__ == '__main__':
    main()
