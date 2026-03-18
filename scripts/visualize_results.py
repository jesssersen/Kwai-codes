"""
Visualize VLMEvalKit benchmark results.

Scans a work_dir for evaluation result files, extracts accuracy/score
per model × benchmark, and saves comparison bar charts.

Usage:
    python scripts/visualize_results.py --work-dir /m2v_intern/xuboshen/zgw/eval_8gpu --out-dir ./plots
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
    """Return dict/list content of a json/csv/xlsx file, or None on failure."""
    try:
        if path.suffix == '.json':
            with open(path) as f:
                return json.load(f)
        if path.suffix == '.csv':
            import pandas as pd
            return pd.read_csv(path)
        if path.suffix == '.xlsx':
            import pandas as pd
            return pd.read_excel(path)
    except Exception:
        return None


def _extract_score_from_json(data) -> float | None:
    """Pull the primary accuracy/score out of a result JSON."""
    if not isinstance(data, dict):
        return None
    # Common scalar keys
    for key in ('accuracy', 'Accuracy', 'acc', 'score', 'Score'):
        if key in data:
            val = data[key]
            if isinstance(val, (int, float)):
                return float(val)
    # Handle 'overall' key in various formats
    if 'overall' in data:
        val = data['overall']
        if isinstance(val, (int, float)):
            return float(val)
        # MVBench _rating.json: {"overall": [correct, total, "51.60%"]}
        if isinstance(val, list) and len(val) >= 3:
            try:
                return float(str(val[2]).rstrip('%'))
            except (ValueError, TypeError):
                pass
        # LongVideoBench _rating.json: {"overall": {"overall": "0.512", "question_category": {...}}}
        if isinstance(val, dict) and 'overall' in val:
            inner = val['overall']
            if isinstance(inner, str):
                try:
                    v = float(inner)
                    # If value <= 1.0, assume it's a fraction; convert to percentage
                    return v * 100 if v <= 1.0 else v
                except ValueError:
                    pass
            if isinstance(inner, (int, float)):
                v = float(inner)
                return v * 100 if v <= 1.0 else v
        # Plain string: "51.60%" or "0.512"
        if isinstance(val, str):
            clean = val.rstrip('%')
            try:
                v = float(clean)
                if val.endswith('%'):
                    return v
                return v * 100 if v <= 1.0 else v
            except ValueError:
                pass
    # Recursive search in nested dicts
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


def _normalize_per_axis(
    results: dict[str, dict[str, float]],
    benchmarks: list[str],
    models: list[str],
    padding_frac: float = 0.25,
    min_padding: float = 0.5,
) -> tuple[np.ndarray, dict[str, tuple[float, float]]]:
    """Per-axis normalization: map each benchmark's scores to [0, 100] independently.

    Returns (normalized, axis_ranges) where:
      - normalized: shape (num_models, num_benchmarks), values in [0, 100]
      - axis_ranges: {benchmark: (display_min, display_max)} in original score space
    """
    num_m, num_b = len(models), len(benchmarks)
    raw = np.full((num_m, num_b), np.nan)
    for bi, bench in enumerate(benchmarks):
        for mi, model in enumerate(models):
            if model in results[bench]:
                raw[mi, bi] = results[bench][model]

    normalized = np.full_like(raw, np.nan)
    axis_ranges: dict[str, tuple[float, float]] = {}

    for bi, bench in enumerate(benchmarks):
        col = raw[:, bi]
        valid = col[~np.isnan(col)]
        if len(valid) == 0:
            axis_ranges[bench] = (0.0, 100.0)
            continue
        lo, hi = float(valid.min()), float(valid.max())
        spread = hi - lo
        pad = max(min_padding, spread * padding_frac)
        axis_min = max(0.0, lo - pad)
        axis_max = min(100.0, hi + pad)
        rng = axis_max - axis_min
        if rng == 0:
            rng = 1.0
        axis_ranges[bench] = (axis_min, axis_max)
        for mi in range(num_m):
            if not np.isnan(raw[mi, bi]):
                normalized[mi, bi] = (raw[mi, bi] - axis_min) / rng * 100

    return normalized, axis_ranges


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


def plot_radar(
    results: dict[str, dict[str, float]],
    out_path: str,
    padding_frac: float = 0.25,
    min_padding: float = 0.5,
):
    """Radar (spider) chart with per-axis zoom to amplify small differences.

    Each benchmark axis is independently scaled so that even 0.5 pp gaps
    become clearly visible.  Actual score values are shown via overlay axes.
    """
    all_models = sorted({m for scores in results.values() for m in scores})
    benchmarks = sorted(results.keys())
    num_vars = len(benchmarks)

    if num_vars < 3 or len(all_models) == 0:
        return

    # Keep only models that have scores for ALL benchmarks
    complete_models = [m for m in all_models
                       if all(m in results[b] for b in benchmarks)]
    if len(complete_models) < 2:
        # Fallback: use all models even if incomplete
        complete_models = all_models
    if len(complete_models) == 0:
        return

    skipped = set(all_models) - set(complete_models)
    if skipped:
        print(f'  radar: skipping models with missing benchmarks: {skipped}')

    normalized, axis_ranges = _normalize_per_axis(
        results, benchmarks, complete_models, padding_frac, min_padding)

    # --- angles ---
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles_deg = np.linspace(0, 360, num_vars, endpoint=False).tolist()

    # --- main polar figure ---
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))

    use_fill = len(complete_models) <= 6

    for mi, model in enumerate(complete_models):
        vals = normalized[mi]
        if np.all(np.isnan(vals)):
            continue
        # Close the polygon
        loop_vals = np.append(vals, vals[0])
        loop_angles = angles + [angles[0]]
        color = COLORS[mi % len(COLORS)]
        ax.plot(loop_angles, loop_vals, color=color, linewidth=1.5,
                linestyle='solid', label=model)
        if use_fill:
            ax.fill(loop_angles, loop_vals, color=color, alpha=0.15)

    ax.set_ylim(0, 100)
    ax.set_yticks([20, 40, 60, 80, 100])
    ax.set_yticklabels([''] * 5)

    ax.tick_params(pad=20)
    ax.set_xticks(angles)
    ax.set_xticklabels(benchmarks, fontsize=10)

    # Legend
    leg = ax.legend(loc='center right', bbox_to_anchor=(1.45, 0.5),
                    fontsize=9, framealpha=0.8, ncol=1, labelspacing=1.0)
    for line in leg.get_lines():
        line.set_linewidth(2.5)

    # --- overlay axes for per-axis tick labels ---
    # Dynamically compute center & size from the main axes position
    bbox = ax.get_position()
    cx = bbox.x0 + bbox.width / 2
    cy = bbox.y0 + bbox.height / 2
    sz = bbox.width / 2

    for i in range(num_vars):
        oa = fig.add_axes(
            [cx - sz, cy - sz, sz * 2, sz * 2],
            projection='polar', label=f'overlay_{i}',
        )
        oa.patch.set_visible(False)
        oa.grid(False)
        oa.xaxis.set_visible(False)

        axis_min, axis_max = axis_ranges[benchmarks[i]]
        rng = axis_max - axis_min if axis_max != axis_min else 1.0
        # 4 evenly-spaced tick values in original score space (skip endpoints)
        tick_actual = [axis_min + rng / 5 * k for k in range(2, 6)]
        tick_norm = [(v - axis_min) / rng * 100 for v in tick_actual]
        tick_labels = [f'{v:.1f}' for v in tick_actual]

        oa.set_rgrids(tick_norm, angle=angles_deg[i],
                      labels=tick_labels, fontsize=8)
        oa.spines['polar'].set_visible(False)
        oa.set_ylim(0, 100)

    # Title & subtitle
    ax.set_title('Model Comparison (per-axis zoom)',
                 fontsize=14, fontweight='bold', pad=30, y=1.08)
    fig.text(0.5, 0.02,
             'Note: Each axis is independently scaled to amplify differences. '
             'Tick labels show actual scores.',
             ha='center', fontsize=8, fontstyle='italic', color='#666666')

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

    # Radar chart (needs >= 3 benchmarks for meaningful spider plot)
    if len(results) >= 3:
        plot_radar(results, osp.join(out_dir, '_radar.png'))

    # Overview plot (all benchmarks × all models)
    if len(results) > 1:
        plot_overview(results, osp.join(out_dir, '_overview.png'))

    print(f'\nDone. Plots saved to: {out_dir}')


if __name__ == '__main__':
    main()
