"""Generate final presentation tables from saved artifacts.

Reads existing baseline metrics and ranking NDCG outputs and writes
clean CSVs and a markdown report for slides.
"""
from pathlib import Path
import pandas as pd


def main():
    repo = Path(__file__).resolve().parents[1]
    tables_dir = repo / 'artifacts' / 'tables'
    reports_dir = repo / 'reports' / 'analysis'
    reports_dir.mkdir(parents=True, exist_ok=True)

    # Input files
    baseline_fp = tables_dir / 'baseline_metrics.csv'
    ranking_fp = tables_dir / 'pareto_ranking_results.csv'

    # Output files
    out_baseline = tables_dir / 'final_baseline_table.csv'
    out_rerank = tables_dir / 'final_reranking_table.csv'
    out_md = reports_dir / 'final_tables.md'

    def df_to_md(df, floatfmt='.6f'):
        if df is None or df.empty:
            return ''
        cols = list(df.columns)
        hdr = '| ' + ' | '.join(cols) + ' |'
        sep = '| ' + ' | '.join(['---'] * len(cols)) + ' |'
        lines = [hdr, sep]
        for _, r in df.iterrows():
            vals = []
            for c in cols:
                v = r[c]
                if pd.isna(v):
                    vals.append('')
                elif isinstance(v, float):
                    vals.append(f"{v:{floatfmt}}")
                else:
                    vals.append(str(v))
            lines.append('| ' + ' | '.join(vals) + ' |')
        return '\n'.join(lines)

    # Load baseline metrics and write cleaned CSV
    if baseline_fp.exists():
        df_baseline = pd.read_csv(baseline_fp)
        df_baseline.to_csv(out_baseline, index=False)
    else:
        df_baseline = pd.DataFrame()

    # Load ranking results and build NDCG@10 table
    df_rank = pd.read_csv(ranking_fp)
    df_k10 = df_rank[df_rank['k'] == 10].copy()
    # pivot to strategies x targets
    piv = df_k10.pivot(index='strategy', columns='target', values='ndcg')
    # ensure expected columns and order
    cols = ['is_click', 'is_like', 'long_view']
    for c in cols:
        if c not in piv.columns:
            piv[c] = float('nan')
    piv = piv[cols]
    # rename columns for presentation
    piv = piv.rename(columns={'is_click': 'Click', 'is_like': 'Like', 'long_view': 'LongView'})
    # desired strategy order
    order = ['click_only', 'like_only', 'longview_only', 'weighted_scalar', 'pareto_frontier', 'pareto_weighted']
    piv = piv.reindex(order)
    piv['Mean'] = piv.mean(axis=1)
    piv = piv.reset_index().rename(columns={'strategy': 'Method'})
    # round for presentation
    piv_rounded = piv.copy()
    for c in ['Click', 'Like', 'LongView', 'Mean']:
        piv_rounded[c] = piv_rounded[c].map(lambda v: f"{v:.6f}" if pd.notna(v) else '')
    piv.to_csv(out_rerank, index=False)

    # best-per-objective table
    bests = []
    for col in ['Click', 'Like', 'LongView']:
        best_row = piv[["Method", col]].sort_values(by=col, ascending=False).iloc[0]
        bests.append({'Objective': col, 'BestMethod': best_row['Method'], 'NDCG@10': float(best_row[col])})
    df_bests = pd.DataFrame(bests)

    # write markdown report
    md = []
    md.append('# Final tables')
    md.append('')
    md.append('## Baseline model comparison')
    md.append('Source: `artifacts/tables/baseline_metrics.csv`')
    md.append('')
    if not df_baseline.empty:
            md.append(df_to_md(df_baseline, floatfmt='.6f'))
    else:
        md.append('- baseline metrics not found')
    md.append('')
    md.append('## Final reranking (NDCG@10)')
    md.append('Source: `artifacts/tables/pareto_ranking_results.csv`')
    md.append('')
    md.append(df_to_md(piv_rounded.reset_index(drop=True), floatfmt='.6f'))
    md.append('')
    md.append('## Best strategy per objective')
    md.append(df_to_md(df_bests, floatfmt='.6f'))
    md.append('')
    md.append('**Notes**: Tables derived from saved artifacts; no models were re-run.')

    out_md.write_text('\n\n'.join(md), encoding='utf-8')
    print('Wrote:', out_baseline)
    print('Wrote:', out_rerank)
    print('Wrote:', out_md)


if __name__ == '__main__':
    main()
