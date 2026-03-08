"""Inspect KuaiRand-Pure CSV files and produce schema reports.

Usage (from repo root):
    python -m scripts.inspect_dataset
or
    python scripts/inspect_dataset.py --data-root <path_to_KuaiRand-Pure/data>

Outputs:
 - artifacts/tables/schema_summary.json
 - artifacts/tables/table_shapes.csv
 - reports/analysis/schema_report.md

This script locates the KuaiRand-Pure `data/` directory automatically when possible.
"""
from pathlib import Path
import argparse
import sys
import json
import pandas as pd

from src.data.schema import summarize_table
from src.utils.io import find_under_tree, write_json, write_csv, write_text, ensure_dir
from src.data.load_raw import find_kuairand_data_root
from src.utils.runner import setup_run, handle_exceptions


def locate_data_dir(provided: str = None) -> Path:
    if provided:
        p = Path(provided).expanduser().resolve()
        if p.exists() and p.is_dir():
            return p
        raise FileNotFoundError(f"Provided data root does not exist: {p}")

    # Delegate to load_raw helper which searches common neighboring repo layouts
    try:
        return find_kuairand_data_root(Path.cwd())
    except Exception:
        # fallback to original search
        cwd = Path.cwd()
        kr = find_under_tree(cwd, "KuaiRand-Pure")
        if kr:
            data_dir = kr / "data"
            if data_dir.exists():
                return data_dir
        # try parent
        parent = cwd.parent
        kr2 = find_under_tree(parent, "KuaiRand-Pure")
        if kr2:
            data_dir = kr2 / "data"
            if data_dir.exists():
                return data_dir

    raise FileNotFoundError("Could not locate KuaiRand-Pure/data directory. Pass --data-root explicitly.")


def human_md_for_table(summary: dict) -> str:
    lines = []
    lines.append(f"## {summary['table_name']}")
    lines.append("")
    lines.append(f"Path: {summary['path']}")
    lines.append(f"Rows: {summary['row_count']}")
    lines.append("")
    lines.append("| Column | Dtype | Null % | Distinct (sample/full) | Candidate target |")
    lines.append("|---:|:---:|:---:|:---:|:---:|")
    for c, meta in summary["columns_meta"].items():
        null_pct = f"{meta['null_pct']:.2f}" if meta['null_pct'] is not None else ""
        distinct = str(meta.get('distinct_count_sample_or_full', ""))
        lines.append(f"| {c} | {meta['dtype']} | {null_pct} | {distinct} | {meta['is_candidate_target']} |")
    lines.append("")
    return "\n".join(lines)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", help="Path to KuaiRand-Pure/data directory (optional)")
    p.add_argument("--out-root", help="Root for outputs (optional). Defaults to repository root containing this script.")
    args = p.parse_args(argv)

    try:
        data_dir = locate_data_dir(args.data_root)
    except Exception as e:
        print(str(e), file=sys.stderr)
        sys.exit(2)

    # Determine output root (repo root of pareto-multi-objective-recsys)
    repo_root = Path(__file__).resolve().parents[2]
    if args.out_root:
        repo_root = Path(args.out_root).expanduser().resolve()

    artifacts_tables = ensure_dir(repo_root / "artifacts" / "tables")
    reports_analysis = ensure_dir(repo_root / "reports" / "analysis")

    # discover csv files in data_dir
    csvs = sorted([p for p in data_dir.glob("*.csv") if p.is_file()])
    if not csvs:
        print(f"No CSV files found in {data_dir}", file=sys.stderr)
        sys.exit(1)

    summaries = {}
    shapes_rows = []

    for csv in csvs:
        logger.info('Inspecting: %s', csv.name)
        try:
            summary = summarize_table(csv)
        except Exception as e:
            logger.exception('Failed to read %s: %s', csv, e)
            continue
        summaries[csv.name] = summary
        shapes_rows.append([csv.name, summary['row_count'], len(summary['columns'])])

    # write machine-readable schema
    schema_json = artifacts_tables / "schema_summary.json"
    write_json(schema_json, summaries)

    # write shapes csv
    shapes_csv = artifacts_tables / "table_shapes.csv"
    write_csv(shapes_csv, shapes_rows, header=["table", "rows", "columns"])

    # write human readable markdown
    md = ["# KuaiRand-Pure Schema Report", ""]
    for name, s in summaries.items():
        md.append(human_md_for_table(s))

    md_path = reports_analysis / "schema_report.md"
    write_text(md_path, "\n".join(md))

    logger.info('Wrote: %s', schema_json)
    logger.info('Wrote: %s', shapes_csv)
    logger.info('Wrote: %s', md_path)


if __name__ == "__main__":
    cfg, repo_root, logger = setup_run()
    decorator = handle_exceptions(logger, repo_root)

    @decorator
    def _cli(argv=None):
        parser = argparse.ArgumentParser()
        parser.add_argument("--data-root", help="Path to KuaiRand-Pure/data directory (optional)")
        parser.add_argument("--out-root", help="Root for outputs (optional). Defaults to repository root containing this script.")
        args = parser.parse_args(argv)

        # Determine output root (repo root of pareto-multi-objective-recsys)
        repo_root = Path(__file__).resolve().parents[2]
        if args.out_root:
            repo_root = Path(args.out_root).expanduser().resolve()

        try:
            data_dir = locate_data_dir(args.data_root)
        except Exception as e:
            logger.error(str(e))
            raise

        # call main logic with logger
        # reuse the existing main body by writing a small wrapper
        # (we keep file-writing behavior same as before)
        artifacts_tables = ensure_dir(repo_root / "artifacts" / "tables")
        reports_analysis = ensure_dir(repo_root / "reports" / "analysis")

        # discover csv files in data_dir
        csvs = sorted([p for p in data_dir.glob("*.csv") if p.is_file()])
        if not csvs:
            logger.error('No CSV files found in %s', data_dir)
            raise FileNotFoundError(f'No CSV files found in {data_dir}')

        summaries = {}
        shapes_rows = []

        for csv in csvs:
            logger.info('Inspecting: %s', csv.name)
            try:
                summary = summarize_table(csv)
            except Exception as e:
                logger.exception('Failed to read %s: %s', csv, e)
                continue
            summaries[csv.name] = summary
            shapes_rows.append([csv.name, summary['row_count'], len(summary['columns'])])

        # write machine-readable schema
        schema_json = artifacts_tables / "schema_summary.json"
        write_json(schema_json, summaries)

        # write shapes csv
        shapes_csv = artifacts_tables / "table_shapes.csv"
        write_csv(shapes_csv, shapes_rows, header=["table", "rows", "columns"])

        # write human readable markdown
        md = ["# KuaiRand-Pure Schema Report", ""]
        for name, s in summaries.items():
            md.append(human_md_for_table(s))

        md_path = reports_analysis / "schema_report.md"
        write_text(md_path, "\n".join(md))

        logger.info('Wrote: %s', schema_json)
        logger.info('Wrote: %s', shapes_csv)
        logger.info('Wrote: %s', md_path)

    _cli()
