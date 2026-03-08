"""Generate a final experiment narrative by aggregating evaluation outputs.

Usage:
  python scripts/generate_final_analysis.py --out-root .

The script looks for evaluation outputs produced by prior steps and composes
`reports/analysis/final_experiment_narrative.md` with concise tables and discussion.
"""
from pathlib import Path
import json
import argparse
import pandas as pd

from src.utils.io import ensure_dir
from src.utils.runner import setup_run, handle_exceptions


def load_json(p: Path):
    if p.exists():
        return json.loads(p.read_text(encoding='utf-8'))
    return None


def load_csv(p: Path):
    if p.exists():
        return pd.read_csv(p)
    return None


def format_table_markdown(df: pd.DataFrame, caption: str = None) -> str:
    if df is None or df.empty:
        return "(no data available)\n"
    md = df.to_markdown(index=False)
    if caption:
        return f"{md}\n\n*{caption}*\n"
    return md + "\n"


def build_ndcg_table(baseline_csv: pd.DataFrame, ks=[5,10,20]):
    if baseline_csv is None:
        return None
    # baseline_csv expected columns: baseline, task, ndcg@5, ndcg@10, ndcg@20
    # Pivot to have baseline rows, task+metric columns
    records = []
    for _, r in baseline_csv.iterrows():
        baseline = r.get('baseline')
        task = r.get('task')
        row = {'baseline': baseline, 'task': task}
        for k in baseline_csv.columns:
            if isinstance(k, str) and k.startswith('ndcg@'):
                row[k] = r.get(k)
        records.append(row)
    if not records:
        return None
    df = pd.DataFrame(records)
    # create a readable table: baseline | task | ndcg@5 | ndcg@10 | ndcg@20
    cols = ['baseline', 'task'] + sorted([c for c in df.columns if c.startswith('ndcg@')])
    return df[cols]


def main(out_root: Path, logger=None):
    out_root = Path(out_root)
    reports_dir = ensure_dir(out_root / 'reports' / 'analysis')

    # load available artifacts
    baseline_csv = load_csv(out_root / 'artifacts' / 'tables' / 'baseline_comparison.csv')
    metric_summary = load_csv(out_root / 'artifacts' / 'tables' / 'metric_summary.csv')
    pareto_metrics = load_json(out_root / 'reports' / 'metrics' / 'pareto_metrics.json')
    evaluation_summary = load_json(out_root / 'reports' / 'metrics' / 'ranking_metrics.json')
    sweep = load_json(out_root / 'artifacts' / 'tables' / 'baseline_weight_sweep.json')

    ndcg_table = build_ndcg_table(baseline_csv)

    md_lines = []
    md_lines.append('# Final Experiment Narrative')
    md_lines.append('')
    md_lines.append('## Objective setup')
    md_lines.append('We study a multi-objective recommendation task with the following objectives:')
    md_lines.append('- `like`: whether the user liked the recommended item')
    md_lines.append('- `long_view`: whether the impression resulted in a long view')
    md_lines.append("- `creator_interest` or proxy: measurable creator-level engagement (follow/repeat interactions)")
    md_lines.append('')
    md_lines.append('## Evaluation methods')
    md_lines.append('We evaluate three families of ranking approaches:')
    md_lines.append('- Single-objective ranking (rank by one predicted score)')
    md_lines.append('- Weighted scalarization: min-max normalize per-objective scores and sum with weights')
    md_lines.append('- Pareto frontier reranking: extract non-dominated candidates and present frontier-ordered recommendations')
    md_lines.append('')
    md_lines.append('Evaluation measures: NDCG@K (primary), Precision@K, and Recall@K where applicable. Classification metrics (ROC-AUC, PR-AUC, log loss) are also reported per-task.')
    md_lines.append('')

    md_lines.append('## NDCG@K comparison (baselines)')
    md_lines.append('')
    if ndcg_table is not None:
        md_lines.append(format_table_markdown(ndcg_table, caption='NDCG@K per baseline and task'))
    else:
        md_lines.append('(No baseline comparison CSV found)')
    md_lines.append('')

    md_lines.append('## Summary of ranking metrics')
    if evaluation_summary:
        md_lines.append('Ranking metrics (per-task) extracted from evaluation outputs:')
        md_lines.append('')
        md_lines.append('```json')
        md_lines.append(json.dumps(evaluation_summary, indent=2))
        md_lines.append('```')
    else:
        md_lines.append('(No ranking metrics JSON found)')
    md_lines.append('')

    md_lines.append('## Pareto analysis summary')
    if pareto_metrics:
        md_lines.append('Pareto-based ranking metrics:')
        md_lines.append('')
        md_lines.append('```json')
        md_lines.append(json.dumps(pareto_metrics, indent=2))
        md_lines.append('```')
    else:
        md_lines.append('(No pareto metrics found)')
    md_lines.append('')

    md_lines.append('## Tradeoff interpretation')
    md_lines.append('We inspect how ranking for one objective affects others. In general:')
    md_lines.append('- Single-objective rankings maximize their target but can substantially reduce performance on other objectives.')
    md_lines.append('- Weighted scalarization provides a tunable tradeoff; performance smoothly varies with weights and allows selecting operating points.')
    md_lines.append('- Pareto frontier reranking exposes non-dominated candidate sets which allow the decision-maker to choose among alternatives without collapsing objectives into a single scalar.')
    md_lines.append('')

    md_lines.append('## Best scalar weights from sweep')
    if sweep:
        md_lines.append('A coarse weight sweep was performed; sample of results:')
        md_lines.append('```json')
        md_lines.append(json.dumps(sweep, indent=2))
        md_lines.append('```')
        md_lines.append('The best-performing normalized weights (by aggregate NDCG) are reported above. Use a finer grid for production tuning.')
    else:
        md_lines.append('(No weight sweep data found)')
    md_lines.append('')

    md_lines.append('## Scalarization vs Pareto: practical comparison')
    md_lines.append('- Scalarization simplifies deployment (single score) and is easy to tune, but requires picking weights and may hide tradeoffs.')
    md_lines.append('- Pareto methods preserve tradeoffs explicitly and present a set of non-dominated candidates; they are useful when multiple stakeholders value different objectives.')
    md_lines.append('')

    md_lines.append('## Assumptions, limitations, and biases')
    md_lines.append('- Evaluation relies on logged data; where possible we use random-exposure logs to reduce selection bias.')
    md_lines.append('- Creator interest is a proxy when explicit follow signals are not available; proxies introduce measurement noise.')
    md_lines.append('- Aggregated video/user statistics must be computed with temporal cutoffs to avoid leakage.')
    md_lines.append('- Results are dependent on candidate generation strategy (top-N per objective) and dataset splits.')
    md_lines.append('')

    md_lines.append('## Filtering inactive users')
    md_lines.append('If inactive users were filtered, document the threshold here. (No filtering applied by default in this pipeline.)')
    md_lines.append('')

    md_lines.append('## Conclusion')
    md_lines.append('This repository provides an end-to-end workflow to train multi-task recommenders, generate candidate pools, apply Pareto reranking, and evaluate tradeoffs across objectives. The artifacts produced (models, frontiers, metrics) support informed selection of operating points for deployment or further study.')

    out_path = reports_dir / 'final_experiment_narrative.md'
    out_path.write_text('\n'.join(md_lines), encoding='utf-8')
    if logger:
        logger.info('Wrote final experiment narrative: %s', out_path)
    else:
        print(f'Wrote final experiment narrative: {out_path}')


if __name__ == '__main__':
    cfg, out_root, logger = setup_run()
    decorator = handle_exceptions(logger, out_root)

    @decorator
    def _cli():
        parser = argparse.ArgumentParser()
        parser.add_argument('--out-root', default='.')
        args = parser.parse_args()
        main(args.out_root, logger=logger)

    _cli()
