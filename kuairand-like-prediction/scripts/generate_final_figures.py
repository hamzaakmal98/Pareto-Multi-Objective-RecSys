"""Generate presentation-quality figures from saved artifacts.

Saves PNGs to `artifacts/figures/final/` and writes an index markdown.
Only uses matplotlib and pandas; reads existing artifacts and predictions.
"""
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def pareto_front_mask(points):
    # points: (n, d) array, higher is better
    n = points.shape[0]
    is_dominated = np.zeros(n, dtype=bool)
    for i in range(n):
        if is_dominated[i]:
            continue
        better_or_equal = np.all(points >= points[i], axis=1)
        strictly_better = np.any(points > points[i], axis=1)
        dominates = better_or_equal & strictly_better
        if np.any(dominates):
            is_dominated[i] = True
    return ~is_dominated


def save_fig(fig, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(str(path), dpi=200)
    plt.close(fig)


def main():
    repo = Path(__file__).resolve().parents[1]
    figures_dir = repo / 'artifacts' / 'figures' / 'final'
    tables_dir = repo / 'artifacts' / 'tables'
    data_dir = repo / 'data' / 'processed'
    preds_dir = repo / 'artifacts' / 'predictions'
    reports_dir = repo / 'reports' / 'analysis'
    figures_dir.mkdir(parents=True, exist_ok=True)

    index_lines = ['# Final figures index', '']

    # 1) Target prevalence bar chart
    y_fp = data_dir / 'y.csv'
    if y_fp.exists():
        y = pd.read_csv(y_fp)
        preval = y.mean(axis=0).to_dict()
        fig, ax = plt.subplots(figsize=(6,3))
        keys = list(preval.keys())
        vals = [preval[k] for k in keys]
        ax.bar(keys, vals, color=['#4C72B0','#55A868','#C44E52'])
        ax.set_ylim(0, max(vals)*1.2 if len(vals)>0 else 1)
        ax.set_ylabel('Positive rate')
        ax.set_title('Target prevalence')
        fname = figures_dir / 'target_prevalence.png'
        save_fig(fig, fname)
        index_lines += [f'## Target prevalence', f'File: {fname.relative_to(repo)}', '', 'Shows per-target positive rate. Use on methodology slide.','']
    else:
        index_lines += ['- y.csv not found; skipping target prevalence', '']

    # 2) User activity distribution
    meta_fp = data_dir / 'meta.csv'
    if meta_fp.exists():
        meta = pd.read_csv(meta_fp)
        if 'user_id' in meta.columns:
            counts = meta['user_id'].value_counts()
            fig, ax = plt.subplots(figsize=(6,3))
            ax.hist(counts.values, bins=50, color='#4C72B0')
            ax.set_xscale('log')
            ax.set_xlabel('Impressions per user (log scale)')
            ax.set_ylabel('Number of users')
            ax.set_title('User activity distribution')
            fname = figures_dir / 'user_activity_distribution.png'
            save_fig(fig, fname)
            index_lines += [f'## User activity distribution', f'File: {fname.relative_to(repo)}', '', 'Shows impression counts per user (log scale). Useful for dataset characterization.','']
        else:
            index_lines += ['- meta.csv missing `user_id`; skipping user activity plot', '']
    else:
        index_lines += ['- meta.csv not found; skipping user activity plot', '']

    # 3) Item popularity distribution
    if meta_fp.exists():
        meta = pd.read_csv(meta_fp)
        if 'video_id' in meta.columns:
            vcounts = meta['video_id'].value_counts()
            fig, ax = plt.subplots(figsize=(6,3))
            ax.hist(vcounts.values, bins=50, color='#55A868')
            ax.set_xscale('log')
            ax.set_xlabel('Impressions per video (log scale)')
            ax.set_ylabel('Number of videos')
            ax.set_title('Item popularity distribution')
            fname = figures_dir / 'item_popularity_distribution.png'
            save_fig(fig, fname)
            index_lines += [f'## Item popularity distribution', f'File: {fname.relative_to(repo)}', '', 'Shows long-tail popularity of videos (log scale).','']
        else:
            index_lines += ['- meta.csv missing `video_id`; skipping item popularity plot', '']
    
    # 4) Baseline model comparison plot (PR-AUC by task)
    baseline_fp = tables_dir / 'baseline_metrics.csv'
    if baseline_fp.exists():
        dfb = pd.read_csv(baseline_fp)
        # pivot PR-AUC per model/task
        pr = dfb.pivot(index='Task', columns='Model', values='PR-AUC')
        fig, ax = plt.subplots(figsize=(7,3))
        width = 0.35
        x = np.arange(len(pr.index))
        models = list(pr.columns)
        n = len(models)
        for i, m in enumerate(models):
            ax.bar(x + (i - n/2)*width + width/2, pr[m].values, width=width, label=m)
        ax.set_xticks(x)
        ax.set_xticklabels(pr.index)
        ax.set_ylabel('PR-AUC')
        ax.set_title('Baseline PR-AUC by task')
        ax.legend(frameon=False)
        fname = figures_dir / 'baseline_pr_auc_by_task.png'
        save_fig(fig, fname)
        index_lines += [f'## Baseline model comparison (PR-AUC)', f'File: {fname.relative_to(repo)}', '', 'Bar chart of PR-AUC by model and task. Use in results comparison slides.','']
    else:
        index_lines += ['- baseline_metrics.csv not found; skipping baseline comparison plot', '']

    # 5) NDCG@10 comparison bar chart across ranking strategies
    rerank_fp = tables_dir / 'pareto_ranking_results.csv'
    if rerank_fp.exists():
        dfr = pd.read_csv(rerank_fp)
        d10 = dfr[dfr['k'] == 10]
        pivot = d10.pivot(index='strategy', columns='target', values='ndcg')
        # reorder strategies
        order = ['click_only','like_only','longview_only','weighted_scalar','pareto_frontier','pareto_weighted']
        pivot = pivot.reindex(order)
        fig, ax = plt.subplots(figsize=(8,3))
        width = 0.15
        x = np.arange(len(pivot.index))
        targets = ['is_click','is_like','long_view']
        colors = ['#4C72B0','#55A868','#C44E52']
        for i, t in enumerate(targets):
            vals = pivot[t].values
            ax.bar(x + (i-1)*width, vals, width=width, label=t, color=colors[i])
        ax.set_xticks(x)
        ax.set_xticklabels(pivot.index, rotation=25)
        ax.set_ylabel('NDCG@10')
        ax.set_title('NDCG@10 by ranking strategy and target')
        ax.legend(frameon=False)
        fname = figures_dir / 'ndcg10_ranking_comparison.png'
        save_fig(fig, fname)
        index_lines += [f'## NDCG@10 comparison', f'File: {fname.relative_to(repo)}', '', 'Compare NDCG@10 across strategies and targets; good for results slides.','']
    else:
        index_lines += ['- pareto_ranking_results.csv not found; skipping NDCG plot', '']

    # 6) Pareto trade-off plot (use predictions if available)
    preds_fp = preds_dir / 'best_weighted_test_predictions.csv'
    if preds_fp.exists():
        preds = pd.read_csv(preds_fp)
        # ensure required cols
        if {'pred_score_is_like','pred_score_long_view'}.issubset(preds.columns):
            x = preds['pred_score_is_like'].values
            y = preds['pred_score_long_view'].values
            points = np.vstack([x,y]).T
            mask = pareto_front_mask(points)
            fig, ax = plt.subplots(figsize=(6,4))
            ax.scatter(x, y, alpha=0.4, s=10, color='#7B68EE')
            ax.scatter(x[mask], y[mask], color='#FF7F0E', s=20, label='Pareto frontier')
            ax.set_xlabel('Predicted like score')
            ax.set_ylabel('Predicted long_view score')
            ax.set_title('Pareto trade-off: like vs long_view (predictions)')
            ax.legend(frameon=False)
            fname = figures_dir / 'pareto_like_longview.png'
            save_fig(fig, fname)
            index_lines += [f'## Pareto trade-off plot', f'File: {fname.relative_to(repo)}', '', 'Scatter of predicted scores; Pareto frontier highlighted. Use for methodology or results slides.','']
        else:
            index_lines += ['- prediction columns for pareto plot not found; skipping', '']
    else:
        index_lines += ['- predictions file not found; skipping pareto trade-off plot', '']

    # 7) Optional score distribution plots
    if preds_fp.exists():
        preds = pd.read_csv(preds_fp)
        score_cols = [c for c in preds.columns if c.startswith('pred_score_')]
        if score_cols:
            fig, axes = plt.subplots(1, len(score_cols), figsize=(4*len(score_cols),3))
            if len(score_cols) == 1:
                axes = [axes]
            for ax, col in zip(axes, score_cols):
                ax.hist(preds[col].dropna().values, bins=50, color='#4C72B0')
                ax.set_title(col.replace('pred_score_',''))
                ax.set_xlabel('Predicted score')
            fname = figures_dir / 'score_distributions.png'
            save_fig(fig, fname)
            index_lines += [f'## Score distributions', f'File: {fname.relative_to(repo)}', '', 'Distribution of model scores for each target; use for calibration/diagnostics.','']
        else:
            index_lines += ['- no pred_score_ columns found; skipping score distributions', '']
    
    # write index markdown
    md_path = reports_dir / 'final_figures_index.md'
    md_path.write_text('\n'.join(index_lines), encoding='utf-8')
    print('Wrote figures and index to', figures_dir, md_path)


if __name__ == '__main__':
    main()
