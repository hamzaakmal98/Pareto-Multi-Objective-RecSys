from pathlib import Path
from typing import List, Dict, Any
import json
import pandas as pd

from src.evaluation.metrics import classification_metrics
from src.evaluation.ranking import evaluate_ranking, scalarize_scores
from src.utils.io import ensure_dir, write_json, write_csv, write_text


def evaluate_predictions(predictions_path: Path, ks: List[int], scalar_weights: Dict[str, float] = None, user_col: str = 'user_id') -> Dict[str, Any]:
    df = pd.read_csv(predictions_path)

    # infer target and score column names
    target_candidates = ['is_like', 'long_view', 'creator_interest_proxy', 'creator_interest', 'creator_follow']
    score_candidates = ['like_score', 'longview_score', 'creator_score']

    targets = [c for c in target_candidates if c in df.columns]
    scores = { 'is_like': next((s for s in score_candidates if 'like' in s), 'like_score'),
               'long_view': next((s for s in score_candidates if 'long' in s), 'longview_score'),
               'creator_interest_proxy': next((s for s in score_candidates if 'creator' in s), 'creator_score') }

    # Classification metrics per task
    task_metrics = {}
    for t in ['is_like', 'long_view', 'creator_interest_proxy']:
        if t in df.columns and scores.get(t) in df.columns:
            cm = classification_metrics(df[t].fillna(0).values, df[scores[t]].fillna(0).values)
            task_metrics[t] = cm

    # Ranking metrics per task
    ranking_metrics = {}
    for t in ['is_like', 'long_view', 'creator_interest_proxy']:
        sc = scores.get(t)
        if t in df.columns and sc in df.columns:
            rm = evaluate_ranking(df, user_col, sc, t, ks)
            ranking_metrics[t] = rm

    # scalarized ranking
    scalar_metrics = None
    if scalar_weights:
        score_cols = [scores['is_like'], scores['long_view']]
        if 'creator_interest_proxy' in df.columns and scores.get('creator_interest_proxy') in df.columns:
            score_cols.append(scores['creator_interest_proxy'])
        weights = [scalar_weights.get(k, 1.0) for k in ['is_like', 'long_view', 'creator_interest_proxy'] if (k in df.columns and scores.get(k) in df.columns)]
        if weights and score_cols:
            df = scalarize_scores(df, score_cols, weights, out_col='scalar_score')
            scalar_metrics = evaluate_ranking(df, user_col, 'scalar_score', 'is_like', ks)

    # Save outputs
    repo_root = Path(__file__).resolve().parents[3]
    reports_metrics = ensure_dir(repo_root / 'reports' / 'metrics')
    artifacts_tables = ensure_dir(repo_root / 'artifacts' / 'tables')

    write_json(reports_metrics / 'task_metrics.json', task_metrics)
    write_json(reports_metrics / 'ranking_metrics.json', ranking_metrics)

    # produce metric summary CSV (flatten)
    rows = []
    for task, mets in task_metrics.items():
        row = {'task': task}
        row.update({f'class_{k}': v for k, v in mets.items()})
        rows.append(row)
    for task, mets in ranking_metrics.items():
        if not any(r['task'] == task for r in rows):
            rows.append({'task': task})
        for r in rows:
            if r['task'] == task:
                r.update({f'rank_{k}': v for k, v in mets.items()})

    write_csv(artifacts_tables / 'metric_summary.csv', [[r.get('task')] + [r.get(c) for c in sorted(r.keys()) if c!='task'] for r in rows], header=None)

    # markdown summary
    md_lines = ['# Evaluation Summary', '']
    md_lines.append('## Classification metrics (per task)')
    md_lines.append('')
    for t, m in task_metrics.items():
        md_lines.append(f'### {t}')
        md_lines.append(f'- ROC-AUC: {m.get("roc_auc"):.4f}')
        md_lines.append(f'- PR-AUC: {m.get("pr_auc"):.4f}')
        md_lines.append(f'- LogLoss: {m.get("log_loss"):.4f}')
        md_lines.append('')

    md_lines.append('## Ranking metrics (per task)')
    for t, m in ranking_metrics.items():
        md_lines.append(f'### {t}')
        for k, v in m.items():
            md_lines.append(f'- {k}: {v:.4f}')
        md_lines.append('')

    if scalar_metrics:
        md_lines.append('## Scalarized ranking metrics')
        for k, v in scalar_metrics.items():
            md_lines.append(f'- {k}: {v:.4f}')
        md_lines.append('')

    write_text(ensure_dir(repo_root / 'reports' / 'analysis') / 'evaluation_summary.md', '\n'.join(md_lines))

    return {'task_metrics': task_metrics, 'ranking_metrics': ranking_metrics, 'scalar_metrics': scalar_metrics}
