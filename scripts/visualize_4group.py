"""
4-Group Comparison Visualization: CoT × Temperature ablation.

Compares evaluation results across 4 configurations:
  1. No CoT + temperature=0   (greedy, direct answer)
  2. No CoT + temperature=0.7 (sampling, direct answer)
  3. CoT    + temperature=0   (greedy, chain-of-thought)
  4. CoT    + temperature=0.7 (sampling, chain-of-thought)

Generates:
  1. 4-way heatmap — groups × benchmarks score matrix
  2. 4-way grouped bars — per-benchmark comparison
  3. CoT effect bars — Δ(CoT − NoCoT) at each temperature
  4. Temperature effect bars — Δ(T=0.7 − T=0) for each CoT setting
  5. 2×2 interaction heatmap — CoT × Temperature per benchmark
  6. Radar chart — 4 groups on one radar

Usage:
    python scripts/visualize_4group.py \\
        --noCoT-temp0  /path/to/eval_noCoT_temp0 \\
        --noCoT-temp07 /path/to/eval_noCoT_temp0.7 \\
        --CoT-temp0    /path/to/eval_CoT_temp0 \\
        --CoT-temp07   /path/to/eval_CoT_temp0.7 \\
        --out-dir      ./plots_4group
"""
from __future__ import annotations

import argparse
import os
import os.path as osp
import re
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

# Reuse helpers from visualize_ablation
sys.path.insert(0, osp.dirname(osp.abspath(__file__)))
from visualize_ablation import (
    scan_work_dir,
    merge_aotbench,
    short_bench,
    BENCH_SHORT_NAMES,
    COLORS,
)

# =========================================================================== #
#  Group definitions
# =========================================================================== #

GROUP_LABELS = [
    'NoCoT T=0',
    'NoCoT T=0.7',
    'CoT T=0',
    'CoT T=0.7',
]

GROUP_COLORS = ['#4C72B0', '#55A868', '#DD8452', '#C44E52']


def _load_group(work_dir: str) -> dict[str, float]:
    """Load a single group's results, return {bench: score} for the first model found."""
    overall, _ = scan_work_dir(work_dir)
    merged = merge_aotbench(overall)
    flat: dict[str, float] = {}
    for bench, model_scores in merged.items():
        scores = list(model_scores.values())
        if scores:
            flat[bench] = scores[0]
    return flat


# =========================================================================== #
#  Plot 1: 4-way Heatmap
# =========================================================================== #

def plot_4way_heatmap(
    groups: list[dict[str, float]],
    labels: list[str],
    out_path: str,
):
    benchmarks = sorted({b for g in groups for b in g})
    if not benchmarks:
        return

    data = np.full((len(labels), len(benchmarks)), np.nan)
    for gi, g in enumerate(groups):
        for bi, bench in enumerate(benchmarks):
            if bench in g:
                data[gi, bi] = g[bench]

    fig, ax = plt.subplots(figsize=(max(8, len(benchmarks) * 1.6), max(3, len(labels) * 0.9 + 1.5)))

    valid = data[~np.isnan(data)]
    if len(valid) == 0:
        plt.close(fig)
        return
    vmin, vmax = np.nanmin(valid), np.nanmax(valid)
    pad = max(1, (vmax - vmin) * 0.1)
    norm = mcolors.Normalize(vmin=vmin - pad, vmax=vmax + pad)
    im = ax.imshow(data, cmap='RdYlGn', norm=norm, aspect='auto')

    for gi in range(len(labels)):
        for bi in range(len(benchmarks)):
            val = data[gi, bi]
            if np.isnan(val):
                ax.text(bi, gi, '—', ha='center', va='center', fontsize=10, color='#999')
            else:
                text_color = 'white' if (val - vmin) / max(vmax - vmin, 1) < 0.3 else 'black'
                ax.text(bi, gi, f'{val:.1f}', ha='center', va='center',
                        fontsize=11, fontweight='bold', color=text_color)

    ax.set_xticks(range(len(benchmarks)))
    ax.set_xticklabels([short_bench(b) for b in benchmarks], rotation=30, ha='right', fontsize=10)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=11)
    ax.set_title('4-Group Score Matrix (CoT × Temperature)', fontsize=14, fontweight='bold', pad=12)
    fig.colorbar(im, ax=ax, shrink=0.8, aspect=30, pad=0.02, label='Score')

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  saved -> {out_path}')


# =========================================================================== #
#  Plot 2: 4-way Grouped Bars
# =========================================================================== #

def plot_4way_grouped_bars(
    groups: list[dict[str, float]],
    labels: list[str],
    out_path: str,
):
    benchmarks = sorted({b for g in groups for b in g})
    if not benchmarks:
        return

    n_bench = len(benchmarks)
    n_groups = len(labels)
    width = 0.75 / n_groups
    x = np.arange(n_bench)

    fig, ax = plt.subplots(figsize=(max(10, n_bench * 2), 6))

    for gi, (g, label) in enumerate(zip(groups, labels)):
        offset = (gi - n_groups / 2 + 0.5) * width
        vals = [g.get(b, 0) for b in benchmarks]
        bars = ax.bar(x + offset, vals, width=width * 0.9,
                      label=label, color=GROUP_COLORS[gi],
                      edgecolor='white', linewidth=0.5)
        for bar, val in zip(bars, vals):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                        f'{val:.1f}', ha='center', va='bottom', fontsize=7, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels([short_bench(b) for b in benchmarks], rotation=20, ha='right', fontsize=10)
    ax.set_ylabel('Score', fontsize=11)
    ax.set_title('4-Group Comparison per Benchmark', fontsize=14, fontweight='bold', pad=10)
    ax.legend(fontsize=9, framealpha=0.8)
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    ax.spines[['top', 'right']].set_visible(False)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  saved -> {out_path}')


# =========================================================================== #
#  Plot 3: CoT Effect Bars — Δ(CoT − NoCoT) at each temperature
# =========================================================================== #

def plot_cot_effect(
    groups: list[dict[str, float]],
    out_path: str,
):
    benchmarks = sorted({b for g in groups for b in g})
    if not benchmarks:
        return

    # groups: [noCoT_t0, noCoT_t07, CoT_t0, CoT_t07]
    delta_t0 = {b: groups[2].get(b, 0) - groups[0].get(b, 0) for b in benchmarks
                if b in groups[0] and b in groups[2]}
    delta_t07 = {b: groups[3].get(b, 0) - groups[1].get(b, 0) for b in benchmarks
                 if b in groups[1] and b in groups[3]}
    benches = sorted(set(delta_t0) | set(delta_t07))
    if not benches:
        return

    x = np.arange(len(benches))
    width = 0.35

    fig, ax = plt.subplots(figsize=(max(10, len(benches) * 1.8), 5))
    vals_t0 = [delta_t0.get(b, 0) for b in benches]
    vals_t07 = [delta_t07.get(b, 0) for b in benches]

    bars1 = ax.bar(x - width / 2, vals_t0, width, label='CoT effect @ T=0',
                   color=GROUP_COLORS[0], edgecolor='white')
    bars2 = ax.bar(x + width / 2, vals_t07, width, label='CoT effect @ T=0.7',
                   color=GROUP_COLORS[3], edgecolor='white')

    for bars in [bars1, bars2]:
        for bar in bars:
            val = bar.get_height()
            sign = '+' if val > 0 else ''
            color = '#1a7a2e' if val > 0 else '#c0392b'
            ax.text(bar.get_x() + bar.get_width() / 2,
                    val + (0.3 if val >= 0 else -0.8),
                    f'{sign}{val:.1f}', ha='center', va='bottom' if val >= 0 else 'top',
                    fontsize=8, fontweight='bold', color=color)

    ax.axhline(0, color='black', linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([short_bench(b) for b in benches], rotation=20, ha='right', fontsize=10)
    ax.set_ylabel('Score Difference', fontsize=11)
    ax.set_title('CoT Effect: Δ(CoT − NoCoT)', fontsize=14, fontweight='bold', pad=10)
    ax.legend(fontsize=9, framealpha=0.8)
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    ax.spines[['top', 'right']].set_visible(False)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  saved -> {out_path}')


# =========================================================================== #
#  Plot 4: Temperature Effect Bars — Δ(T=0.7 − T=0) for each CoT setting
# =========================================================================== #

def plot_temp_effect(
    groups: list[dict[str, float]],
    out_path: str,
):
    benchmarks = sorted({b for g in groups for b in g})
    if not benchmarks:
        return

    # groups: [noCoT_t0, noCoT_t07, CoT_t0, CoT_t07]
    delta_nocot = {b: groups[1].get(b, 0) - groups[0].get(b, 0) for b in benchmarks
                   if b in groups[0] and b in groups[1]}
    delta_cot = {b: groups[3].get(b, 0) - groups[2].get(b, 0) for b in benchmarks
                 if b in groups[2] and b in groups[3]}
    benches = sorted(set(delta_nocot) | set(delta_cot))
    if not benches:
        return

    x = np.arange(len(benches))
    width = 0.35

    fig, ax = plt.subplots(figsize=(max(10, len(benches) * 1.8), 5))
    vals_nocot = [delta_nocot.get(b, 0) for b in benches]
    vals_cot = [delta_cot.get(b, 0) for b in benches]

    bars1 = ax.bar(x - width / 2, vals_nocot, width, label='Temp effect (NoCoT)',
                   color=GROUP_COLORS[0], edgecolor='white')
    bars2 = ax.bar(x + width / 2, vals_cot, width, label='Temp effect (CoT)',
                   color=GROUP_COLORS[2], edgecolor='white')

    for bars in [bars1, bars2]:
        for bar in bars:
            val = bar.get_height()
            sign = '+' if val > 0 else ''
            color = '#1a7a2e' if val > 0 else '#c0392b'
            ax.text(bar.get_x() + bar.get_width() / 2,
                    val + (0.3 if val >= 0 else -0.8),
                    f'{sign}{val:.1f}', ha='center', va='bottom' if val >= 0 else 'top',
                    fontsize=8, fontweight='bold', color=color)

    ax.axhline(0, color='black', linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([short_bench(b) for b in benches], rotation=20, ha='right', fontsize=10)
    ax.set_ylabel('Score Difference', fontsize=11)
    ax.set_title('Temperature Effect: Δ(T=0.7 − T=0)', fontsize=14, fontweight='bold', pad=10)
    ax.legend(fontsize=9, framealpha=0.8)
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    ax.spines[['top', 'right']].set_visible(False)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  saved -> {out_path}')


# =========================================================================== #
#  Plot 5: 2×2 Interaction Heatmap
# =========================================================================== #

def plot_interaction_heatmap(
    groups: list[dict[str, float]],
    out_path: str,
):
    benchmarks = sorted({b for g in groups for b in g})
    if not benchmarks:
        return

    n_bench = len(benchmarks)
    n_cols = min(4, n_bench)
    # +1 for the average panel
    total = n_bench + 1
    n_rows = (total + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3.5, n_rows * 3))
    if n_rows * n_cols == 1:
        axes = np.array([[axes]])
    axes = np.atleast_2d(axes)

    # Compute global color range
    all_vals = [v for g in groups for v in g.values()]
    if not all_vals:
        plt.close(fig)
        return
    vmin, vmax = min(all_vals), max(all_vals)
    pad = max(1, (vmax - vmin) * 0.1)
    norm = mcolors.Normalize(vmin=vmin - pad, vmax=vmax + pad)

    row_labels = ['NoCoT', 'CoT']
    col_labels = ['T=0', 'T=0.7']

    items = list(benchmarks) + ['Average']
    for idx, bench in enumerate(items):
        r, c = divmod(idx, n_cols)
        ax = axes[r, c]

        if bench == 'Average':
            grid = np.zeros((2, 2))
            counts = np.zeros((2, 2))
            for b in benchmarks:
                for gi, (ri, ci) in enumerate([(0, 0), (0, 1), (1, 0), (1, 1)]):
                    if b in groups[gi]:
                        grid[ri, ci] += groups[gi][b]
                        counts[ri, ci] += 1
            counts[counts == 0] = 1
            grid = grid / counts
        else:
            # groups order: [noCoT_t0, noCoT_t07, CoT_t0, CoT_t07]
            grid = np.array([
                [groups[0].get(bench, np.nan), groups[1].get(bench, np.nan)],
                [groups[2].get(bench, np.nan), groups[3].get(bench, np.nan)],
            ])

        im = ax.imshow(grid, cmap='RdYlGn', norm=norm, aspect='auto')
        for ri in range(2):
            for ci in range(2):
                val = grid[ri, ci]
                if np.isnan(val):
                    ax.text(ci, ri, '—', ha='center', va='center', fontsize=11, color='#999')
                else:
                    ax.text(ci, ri, f'{val:.1f}', ha='center', va='center',
                            fontsize=12, fontweight='bold')

        ax.set_xticks([0, 1])
        ax.set_xticklabels(col_labels, fontsize=9)
        ax.set_yticks([0, 1])
        ax.set_yticklabels(row_labels, fontsize=9)
        title = short_bench(bench) if bench != 'Average' else 'Average'
        ax.set_title(title, fontsize=10, fontweight='bold')

    # Hide empty subplots
    for idx in range(len(items), n_rows * n_cols):
        r, c = divmod(idx, n_cols)
        axes[r, c].set_visible(False)

    fig.suptitle('CoT × Temperature Interaction', fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  saved -> {out_path}')


# =========================================================================== #
#  Plot 6: Radar Chart
# =========================================================================== #

def plot_4way_radar(
    groups: list[dict[str, float]],
    labels: list[str],
    out_path: str,
):
    benchmarks = sorted({b for g in groups for b in g})
    num_vars = len(benchmarks)
    if num_vars < 3:
        return

    raw = np.full((len(labels), num_vars), np.nan)
    for gi, g in enumerate(groups):
        for bi, bench in enumerate(benchmarks):
            if bench in g:
                raw[gi, bi] = g[bench]

    # Per-axis normalization
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
        pad_val = max(0.5, spread * 0.25)
        axis_min = max(0.0, lo - pad_val)
        axis_max = min(100.0, hi + pad_val)
        rng = max(axis_max - axis_min, 1.0)
        axis_ranges[bench] = (axis_min, axis_max)
        for gi in range(len(labels)):
            if not np.isnan(raw[gi, bi]):
                normalized[gi, bi] = (raw[gi, bi] - axis_min) / rng * 100

    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles_deg = np.linspace(0, 360, num_vars, endpoint=False).tolist()
    theta_offset = np.pi / 4

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    ax.set_theta_offset(theta_offset)

    for gi, label in enumerate(labels):
        vals = normalized[gi]
        if np.all(np.isnan(vals)):
            continue
        loop_vals = np.append(vals, vals[0])
        loop_angles = angles + [angles[0]]
        color = GROUP_COLORS[gi]
        ax.plot(loop_angles, loop_vals, color=color, linewidth=2.2, label=label)
        ax.fill(loop_angles, loop_vals, color=color, alpha=0.08)

    ax.set_ylim(0, 100)
    ax.set_yticks([20, 40, 60, 80, 100])
    ax.set_yticklabels([''] * 5)
    ax.tick_params(pad=30)
    ax.set_xticks(angles)
    ax.set_xticklabels([short_bench(b) for b in benchmarks], fontsize=12, fontweight='bold')

    # Per-axis tick labels
    bbox = ax.get_position()
    cx, cy = bbox.x0 + bbox.width / 2, bbox.y0 + bbox.height / 2
    sz = bbox.width / 2
    for i in range(num_vars):
        oa = fig.add_axes([cx - sz, cy - sz, sz * 2, sz * 2],
                          projection='polar', label=f'ov_{i}')
        oa.patch.set_visible(False)
        oa.grid(False)
        oa.xaxis.set_visible(False)
        oa.set_theta_offset(theta_offset)
        axis_min, axis_max = axis_ranges[benchmarks[i]]
        rng = max(axis_max - axis_min, 1.0)
        tick_actual = [axis_min + rng / 5 * k for k in range(2, 6)]
        tick_norm = [(v - axis_min) / rng * 100 for v in tick_actual]
        tick_labels = [f'{v:.1f}' for v in tick_actual]
        oa.set_rgrids(tick_norm, angle=angles_deg[i], labels=tick_labels, fontsize=9)
        oa.spines['polar'].set_visible(False)
        oa.set_ylim(0, 100)

    leg = ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1),
                    fontsize=11, framealpha=0.9)
    for line in leg.get_lines():
        line.set_linewidth(2.5)
    ax.set_title('4-Group Radar (per-axis zoom)',
                 fontsize=14, fontweight='bold', pad=35, y=1.08)
    fig.text(0.5, 0.01, 'Each axis independently scaled. Tick labels = actual scores.',
             ha='center', fontsize=9, fontstyle='italic', color='#666')
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'  saved -> {out_path}')


# =========================================================================== #
#  Console summary
# =========================================================================== #

def print_summary(groups: list[dict[str, float]], labels: list[str]):
    benchmarks = sorted({b for g in groups for b in g})
    if not benchmarks:
        return

    print('\n' + '=' * 70)
    print('4-GROUP SCORE SUMMARY')
    print('=' * 70)
    header = f'{"Group":<20s}' + ''.join(f'{short_bench(b):>14s}' for b in benchmarks) + f'{"Avg":>10s}'
    print(header)
    print('-' * len(header))
    for gi, (g, label) in enumerate(zip(groups, labels)):
        row = f'{label:<20s}'
        vals = []
        for bench in benchmarks:
            val = g.get(bench)
            row += f'{val:>14.1f}' if val is not None else f'{"—":>14s}'
            if val is not None:
                vals.append(val)
        avg = sum(vals) / len(vals) if vals else 0
        row += f'{avg:>10.1f}'
        print(row)
    print()


# =========================================================================== #
#  Main
# =========================================================================== #

def main():
    parser = argparse.ArgumentParser(
        description='4-Group (CoT x Temperature) comparison visualization',
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument('--noCoT-temp0', required=True,
                        help='Work dir: No CoT + temperature=0')
    parser.add_argument('--noCoT-temp07', required=True,
                        help='Work dir: No CoT + temperature=0.7')
    parser.add_argument('--CoT-temp0', required=True,
                        help='Work dir: CoT + temperature=0')
    parser.add_argument('--CoT-temp07', required=True,
                        help='Work dir: CoT + temperature=0.7')
    parser.add_argument('--out-dir', default='./plots_4group',
                        help='Directory to save plots')
    args = parser.parse_args()

    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    dirs = [
        getattr(args, 'noCoT_temp0'),
        getattr(args, 'noCoT_temp07'),
        getattr(args, 'CoT_temp0'),
        getattr(args, 'CoT_temp07'),
    ]

    print('Loading 4 groups...')
    groups = []
    for label, d in zip(GROUP_LABELS, dirs):
        print(f'  {label}: {d}')
        g = _load_group(d)
        print(f'    -> {len(g)} benchmark(s)')
        groups.append(g)

    all_benches = {b for g in groups for b in g}
    if not all_benches:
        print('No result files found in any group.')
        sys.exit(1)

    print(f'\nTotal benchmarks: {len(all_benches)}')
    print(f'Output directory: {out_dir}\n')

    # 1. Heatmap
    print('[1/6] 4-way Heatmap')
    plot_4way_heatmap(groups, GROUP_LABELS, osp.join(out_dir, '01_4way_heatmap.png'))

    # 2. Grouped bars
    print('[2/6] 4-way Grouped Bars')
    plot_4way_grouped_bars(groups, GROUP_LABELS, osp.join(out_dir, '02_4way_grouped_bars.png'))

    # 3. CoT effect
    print('[3/6] CoT Effect')
    plot_cot_effect(groups, osp.join(out_dir, '03_cot_effect.png'))

    # 4. Temperature effect
    print('[4/6] Temperature Effect')
    plot_temp_effect(groups, osp.join(out_dir, '04_temp_effect.png'))

    # 5. 2x2 interaction
    print('[5/6] 2x2 Interaction Heatmap')
    plot_interaction_heatmap(groups, osp.join(out_dir, '05_interaction_2x2.png'))

    # 6. Radar
    print('[6/6] Radar Chart')
    plot_4way_radar(groups, GROUP_LABELS, osp.join(out_dir, '06_4way_radar.png'))

    # Console summary
    print_summary(groups, GROUP_LABELS)

    print(f'Done. All plots saved to: {out_dir}')


if __name__ == '__main__':
    main()
