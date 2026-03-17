"""
Visualize VLMEvalKit benchmark results.

Scans a work_dir for evaluation result files, extracts accuracy/score
per model × benchmark, and saves comparison bar charts.

Usage:
    python scripts/visualize_results.py --work-dir /path/to/outputs --out-dir ./plots
    python scripts/visualize_results.py --work-dir /path/to/outputs  # saves to work_dir/plots
"""
import argparse
import json
import os
import os.path as osp
import re
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# --------------------------------------------------------------------------- #
# Result file parsing
# --------------------------------------------------------------------------- #

def _try_load(path: Path):
    """Return dict/list content of a json/csv file, or None on failure."""
    try:
        if path.suffix == '.json':
            with open(path) as f:
                return json.load(f)
        if path.suffix == '.csv':
            import pandas as pd
            return pd.read_csv(path)
    except Exception:
        return None


def _extract_score_from_json(data) -> float | None:
    """Pull the primary accuracy/score out of a result JSON."""
    if not isinstance(data, dict):
        return None
    # Common keys in VLMEvalKit result JSONs
    for key in ('accuracy', 'Accuracy', 'acc', 'score', 'Score', 'overall'):
        if key in data:
            val = data[key]
            if isinstance(val, (int, float)):
                return float(val)
    # Nested: {"overall": {"accuracy": ...}}
    for v in data.values():
        if isinstance(v, dict):
            result = _extract_score_from_json(v)
            if result is not None:
                return result
    return None


def _extract_score_from_csv(df) -> float | None:
    """Pull the primary accuracy out of a result CSV DataFrame."""
    import pandas as pd
    if df is None or df.empty:
        return None
    # Standard VLMEvalKit format: rows with 'category' column, one row is 'overall'
    if 'category' in df.columns and 'accuracy' in df.columns:
        overall = df[df['category'] == 'overall']
        if not overall.empty:
            try:
                return float(overall.iloc[0]['accuracy'])
            except (ValueError, TypeError):
                pass
    # Fallback: look for accuracy/score columns, take last row
    for col in df.columns:
        if col.lower() in ('accuracy', 'acc', 'score', 'overall', 'success'):
            vals = df[col].dropna()
            if len(vals):
                try:
                    return float(vals.iloc[-1])  # last row is often the total
                except (ValueError, TypeError):
                    pass
    return None


# --------------------------------------------------------------------------- #
# Directory scanning
# --------------------------------------------------------------------------- #

def scan_work_dir(work_dir: str) -> dict[str, dict[str, float]]:
    """
    Returns: {benchmark_name: {model_name: score}}

    Directory layout expected:
        work_dir/
          {model_name}/
            T{date}_G{hash}/           ← eval run dir
              {model_name}_{dataset}_acc.csv    ← or .json
              {model_name}_{dataset}_score.csv
              ...
    """
    work_dir = Path(work_dir)
    # bench -> model -> score
    results: dict[str, dict[str, float]] = defaultdict(dict)

    for model_dir in sorted(work_dir.iterdir()):
        if not model_dir.is_dir():
            continue
        model_name = model_dir.name

        # Walk all run sub-dirs (T{date}_G{hash}) and root
        candidate_dirs = [model_dir] + sorted(
            [d for d in model_dir.iterdir() if d.is_dir()],
            reverse=True  # newest first
        )

        seen_benchmarks: set[str] = set()

        for run_dir in candidate_dirs:
            for fpath in sorted(run_dir.iterdir()):
                if not fpath.is_file():
                    continue
                fname = fpath.name

                # Match: ModelName_DatasetName_acc.csv / _score.csv / _result.json
                # or:    ModelName_DatasetName.json  (some datasets dump bare json)
                m = re.match(
                    rf'^{re.escape(model_name)}_(.+?)(?:_acc|_score|_eval|_result)?\.(?:csv|json)$',
                    fname,
                )
                if not m:
                    continue
                bench = m.group(1)
                if bench in seen_benchmarks:
                    continue  # already got a score from a newer run

                score: float | None = None
                if fpath.suffix == '.json':
                    score = _extract_score_from_json(_try_load(fpath))
                elif fpath.suffix == '.csv':
                    score = _extract_score_from_csv(_try_load(fpath))

                if score is not None:
                    results[bench][model_name] = score
                    seen_benchmarks.add(bench)

    return results


# --------------------------------------------------------------------------- #
# Plotting
# --------------------------------------------------------------------------- #

COLORS = [
    '#4C72B0', '#DD8452', '#55A868', '#C44E52', '#8172B3',
    '#937860', '#DA8BC3', '#8C8C8C', '#CCB974', '#64B5CD',
]


def plot_benchmark(bench: str, model_scores: dict[str, float], out_path: str):
    """Save a grouped bar chart for one benchmark."""
    models = sorted(model_scores.keys())
    scores = [model_scores[m] for m in models]

    fig, ax = plt.subplots(figsize=(max(6, len(models) * 1.4), 5))
    x = np.arange(len(models))
    bars = ax.bar(x, scores, width=0.6,
                  color=[COLORS[i % len(COLORS)] for i in range(len(models))],
                  edgecolor='white', linewidth=0.8)

    # Value labels on top of bars
    for bar, val in zip(bars, scores):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.5,
                f'{val:.1f}', ha='center', va='bottom',
                fontsize=9, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=25, ha='right', fontsize=9)
    ax.set_ylabel('Score (%)', fontsize=11)
    ax.set_title(bench, fontsize=13, fontweight='bold', pad=12)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.0f'))
    ax.set_ylim(0, min(105, max(scores) * 1.18 + 2))
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    ax.spines[['top', 'right']].set_visible(False)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  saved → {out_path}')


def plot_overview(results: dict[str, dict[str, float]], out_path: str):
    """
    Heatmap / grouped bar: all benchmarks × all models.
    Only drawn when there are multiple benchmarks.
    """
    all_models = sorted({m for scores in results.values() for m in scores})
    benchmarks = sorted(results.keys())

    if len(all_models) == 0 or len(benchmarks) == 0:
        return

    data = np.full((len(benchmarks), len(all_models)), np.nan)
    for bi, bench in enumerate(benchmarks):
        for mi, model in enumerate(all_models):
            if model in results[bench]:
                data[bi, mi] = results[bench][model]

    fig, ax = plt.subplots(figsize=(max(8, len(all_models) * 1.6),
                                    max(5, len(benchmarks) * 0.6 + 2)))

    # grouped bars
    width = 0.8 / max(len(all_models), 1)
    x = np.arange(len(benchmarks))
    for mi, model in enumerate(all_models):
        offset = (mi - len(all_models) / 2 + 0.5) * width
        vals = data[:, mi]
        mask = ~np.isnan(vals)
        ax.bar(x[mask] + offset, vals[mask], width=width * 0.9,
               label=model,
               color=COLORS[mi % len(COLORS)],
               edgecolor='white', linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(benchmarks, rotation=30, ha='right', fontsize=9)
    ax.set_ylabel('Score (%)', fontsize=11)
    ax.set_title('Overview: All Models × All Benchmarks',
                 fontsize=13, fontweight='bold', pad=12)
    ax.legend(loc='upper right', fontsize=8, framealpha=0.7)
    ax.grid(axis='y', linestyle='--', alpha=0.35)
    ax.spines[['top', 'right']].set_visible(False)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  saved → {out_path}')


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser(description='Visualize VLMEvalKit results')
    parser.add_argument('--work-dir', required=True,
                        help='VLMEvalKit output directory (contains model subdirs)')
    parser.add_argument('--out-dir', default=None,
                        help='Directory to save plots (default: work_dir/plots)')
    parser.add_argument('--min-models', type=int, default=1,
                        help='Only plot benchmarks with at least this many models')
    args = parser.parse_args()

    out_dir = args.out_dir or osp.join(args.work_dir, 'plots')
    os.makedirs(out_dir, exist_ok=True)

    print(f'Scanning: {args.work_dir}')
    results = scan_work_dir(args.work_dir)

    if not results:
        print('No result files found. Make sure evaluation has completed and '
              'result CSV/JSON files exist in the work_dir subdirectories.')
        sys.exit(1)

    print(f'\nFound {len(results)} benchmark(s), '
          f'{len({m for s in results.values() for m in s})} model(s)\n')

    # Per-benchmark plots
    for bench, model_scores in sorted(results.items()):
        if len(model_scores) < args.min_models:
            continue
        safe_name = re.sub(r'[^\w\-]', '_', bench)
        out_path = osp.join(out_dir, f'{safe_name}.png')
        print(f'[{bench}]  {len(model_scores)} model(s): '
              + ', '.join(f'{m}={v:.1f}' for m, v in sorted(model_scores.items())))
        plot_benchmark(bench, model_scores, out_path)

    # Overview plot (all benchmarks × all models)
    if len(results) > 1:
        plot_overview(results, osp.join(out_dir, '_overview.png'))

    print(f'\nDone. Plots saved to: {out_dir}')


if __name__ == '__main__':
    main()
