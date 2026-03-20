"""
Plot spectral metrics from compute_metrics.py output.

Usage:
    python plot_metrics.py [--metrics_dir METRICS_DIR] [--output_dir OUTPUT_DIR]
"""

import re
import argparse
import math
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
from pathlib import Path

# ---------------------------------------------------------------------------
# LR scheduler (mirrors train.py warmup-stable-decay)
# ---------------------------------------------------------------------------
LEARNING_RATE = 6e-4
MIN_LR = 6e-5
WARMUP_ITERS = 2000
MAX_ITERS = 30000
LR_DECAY_ITERS = 2000        # decay starts at MAX_ITERS - LR_DECAY_ITERS = 28000
BETA1 = 0.9


def get_lr(it: int) -> float:
    if it < WARMUP_ITERS:
        return LEARNING_RATE * (it + 1) / (WARMUP_ITERS + 1)
    if it >= MAX_ITERS - LR_DECAY_ITERS:
        coeff = (MAX_ITERS - it - 1) / LR_DECAY_ITERS
        return MIN_LR + coeff * (LEARNING_RATE - MIN_LR)
    return LEARNING_RATE


def sharpness_bound(it: int) -> float:
    """Adam sharpness bound: 2(1+β1) / (η(1-β1))."""
    eta = get_lr(it)
    if eta <= 0:
        return math.inf
    return 2.0 * (1.0 + BETA1) / (eta * (1.0 - BETA1))


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def available_keys(all_data: dict) -> set:
    """Return the set of keys present in the first trial of the first checkpoint."""
    first_trials = next(iter(all_data.values()))
    return set(first_trials[0].keys())


def load_metrics(metrics_dir: Path) -> dict[int, list[dict]]:
    """Return {iteration: [trial_dict, ...]}, sorted by iteration."""
    pattern = re.compile(r'ckpt-(\d+)_trial-(\d+)\.pt')
    all_data: dict[int, list] = {}
    for fname in sorted(metrics_dir.glob('ckpt-*.pt')):
        m = pattern.match(fname.name)
        if not m:
            continue
        iteration = int(m.group(1))
        data = torch.load(fname, map_location='cpu', weights_only=False)
        all_data.setdefault(iteration, []).append(data)
    return dict(sorted(all_data.items()))


def aggregate(all_data: dict, key: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (iterations, mean, std) arrays for a scalar metric."""
    iters, means, stds = [], [], []
    for it, trials in all_data.items():
        vals = [d[key].item() for d in trials]
        iters.append(it)
        means.append(np.mean(vals))
        stds.append(np.std(vals))
    return np.array(iters), np.array(means), np.array(stds)


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------
sns.set_theme(style='whitegrid', font_scale=1.0)
COLORS = sns.color_palette('tab10')

BOUND_LABEL = r'Bound $\frac{2(1+\beta_1)}{\eta(1-\beta_1)}$'


def plot_line(ax, x, mean, std, label, color, lw=1.8):
    ax.plot(x, mean, label=label, color=color, lw=lw)
    ax.fill_between(x, mean - std, mean + std, alpha=0.18, color=color)


LOSS_DECEL_ITER = 3000

VLINES = [
    (WARMUP_ITERS,                   'End warmup',    'steelblue',  ':'),
    (MAX_ITERS - LR_DECAY_ITERS,     'Start decay',   'darkorange', ':'),
    (LOSS_DECEL_ITER,                 'Loss decel',    'forestgreen', ':'),
]


def add_vlines(ax):
    """Add vertical reference lines for training phase boundaries."""
    for x, label, color, ls in VLINES:
        ax.axvline(x, color=color, ls=ls, lw=1.4, label=label)


def add_bound(ax, iters, color='black', ls='--', lw=1.5):
    bound = np.array([sharpness_bound(it) for it in iters])
    finite = np.isfinite(bound)
    ax.plot(iters[finite], bound[finite], color=color, ls=ls, lw=lw, label=BOUND_LABEL)


def style_axes(ax, xscale, xlabel='Iteration', ylabel='Eigenvalue'):
    ax.set_xscale(xscale)
    ax.set_yscale('log')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if xscale == 'log':
        ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
        ax.xaxis.set_minor_formatter(ticker.NullFormatter())


def single_fig(height=4.2):
    fig, ax = plt.subplots(figsize=(7, height))
    return fig, ax


def save(fig, path: Path):
    fig.tight_layout()
    fig.savefig(path, bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f"  saved → {path}")


def make_plots(draw_fn, stem: str, output_dir: Path):
    """Call draw_fn(ax, xscale) for both x-scales and save two PNGs."""
    for xscale in ('linear', 'log'):
        fig, ax = single_fig()
        draw_fn(ax, xscale)
        save(fig, output_dir / f'{stem}_{xscale}.png')


# ---------------------------------------------------------------------------
# Individual plots
# ---------------------------------------------------------------------------

def plot1_precond_eigenvalues(all_data, iters, output_dir, keys=None):
    """Top precond-Hessian and (if available) top precond-GN eigenvalues."""
    if keys is None:
        keys = available_keys(all_data)
    _, ht_m, ht_s = aggregate(all_data, 'hessian_precond_top')
    has_gn = 'gn_precond_top' in keys
    if has_gn:
        _, gn_m, gn_s = aggregate(all_data, 'gn_precond_top')

    def draw(ax, xscale):
        plot_line(ax, iters, ht_m, ht_s, 'Hessian precond top', COLORS[0])
        if has_gn:
            plot_line(ax, iters, gn_m, gn_s, 'GN precond top', COLORS[1])
        add_bound(ax, iters)
        add_vlines(ax)
        style_axes(ax, xscale)
        ax.set_title('Precond eigenvalues')
        ax.legend(fontsize=9)

    make_plots(draw, 'plot1_precond_eigenvalues', output_dir)


def plot2_jtj(all_data, iters, output_dir):
    """J^T J top eigenvalue."""
    _, jm, js = aggregate(all_data, 'jtj_top')

    def draw(ax, xscale):
        plot_line(ax, iters, jm, js, r'$J^\top\!J$ top', COLORS[2])
        add_vlines(ax)
        style_axes(ax, xscale)
        ax.set_title(r'$J^\top\!J$ top eigenvalue')
        ax.legend(fontsize=9)

    make_plots(draw, 'plot2_jtj_top', output_dir)


def plot3_hessian_top_bottom(all_data, iters, output_dir):
    """Top and (negated) bottom precond-Hessian eigenvalues."""
    _, ht_m, ht_s = aggregate(all_data, 'hessian_precond_top')
    _, hb_m, hb_s = aggregate(all_data, 'hessian_precond_bottom')

    def draw(ax, xscale):
        plot_line(ax, iters, ht_m, ht_s, 'Hessian precond top', COLORS[0])
        plot_line(ax, iters, -hb_m, hb_s, r'$-$Hessian precond bottom', COLORS[3])
        add_bound(ax, iters)
        add_vlines(ax)
        style_axes(ax, xscale)
        ax.set_title('Hessian precond top & −bottom')
        ax.legend(fontsize=9)

    make_plots(draw, 'plot3_hessian_top_bottom', output_dir)


def plot4_max_res(all_data, iters, output_dir):
    """Max residuals for all eigenvalue estimates."""
    _, ht_m, ht_s = aggregate(all_data, 'hessian_precond_top_max_res')
    _, hb_m, hb_s = aggregate(all_data, 'hessian_precond_bottom_max_res')
    _, gn_m, gn_s = aggregate(all_data, 'gn_precond_top_max_res')
    _, jm,   js   = aggregate(all_data, 'jtj_top_max_res')

    def draw(ax, xscale):
        plot_line(ax, iters, ht_m, ht_s, 'Hessian precond top', COLORS[0])
        plot_line(ax, iters, hb_m, hb_s, 'Hessian precond bottom', COLORS[3])
        plot_line(ax, iters, gn_m, gn_s, 'GN precond top', COLORS[1])
        plot_line(ax, iters, jm, js, r'$J^\top\!J$ top', COLORS[2])
        add_vlines(ax)
        style_axes(ax, xscale, ylabel='Max residual')
        ax.set_title('Max residuals')
        ax.legend(fontsize=9)

    make_plots(draw, 'plot4_max_res', output_dir)


def plot6_solver_convergence(all_data, iters, output_dir, keys):
    """Plot solver convergence diagnostics for all solvers present in the metrics files.

    - ``*_cossim`` keys (power-iteration): final cosine similarity per run.
    - ``*_max_res`` keys (LOBPCG): max relative residual per run.

    Separate figures (linear + log x) are produced for each key type when present.
    """
    cossim_keys = sorted(k for k in keys if k.endswith('_cossim'))
    max_res_keys = sorted(k for k in keys if k.endswith('_max_res'))

    def _label(k):
        return k.replace('_cossim', '').replace('_max_res', '').replace('_', ' ')

    if cossim_keys:
        data = {k: aggregate(all_data, k) for k in cossim_keys}

        def draw_cossim(ax, xscale):
            for i, k in enumerate(cossim_keys):
                _, m, s = data[k]
                plot_line(ax, iters, m, s, _label(k), COLORS[i % len(COLORS)])
            add_vlines(ax)
            ax.set_xscale(xscale)
            if xscale == 'log':
                ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
                ax.xaxis.set_minor_formatter(ticker.NullFormatter())
            ax.set_xlabel('Iteration')
            ax.set_ylabel('Cosine similarity (final step)')
            ax.set_title('Solver convergence: cosine similarity')
            ax.legend(fontsize=9)

        make_plots(draw_cossim, 'plot6_cossim', output_dir)

    if max_res_keys:
        data = {k: aggregate(all_data, k) for k in max_res_keys}

        def draw_max_res(ax, xscale):
            for i, k in enumerate(max_res_keys):
                _, m, s = data[k]
                plot_line(ax, iters, m, s, _label(k), COLORS[i % len(COLORS)])
            add_vlines(ax)
            style_axes(ax, xscale, ylabel='Max relative residual')
            ax.set_title('Solver convergence: max residual (LOBPCG)')
            ax.legend(fontsize=9)

        make_plots(draw_max_res, 'plot6_max_res', output_dir)


def plot5_loss_hessian_ridge(all_data, output_dir):
    """Ridge histplot of the loss-Hessian spectrum; log-scale x-axis."""
    all_iters = sorted(all_data.keys())

    # Select a readable subset: skip very early iters, then sample up to ~18
    candidates = [it for it in all_iters if it >= 1]
    if len(candidates) > 18:
        # Log-space selection so early checkpoints are sampled more densely
        log_targets = np.logspace(np.log10(candidates[0]), np.log10(candidates[-1]), 18)
        idx = [np.argmin(np.abs(np.array(candidates) - t)) for t in log_targets]
        candidates = [candidates[i] for i in sorted(set(idx))]

    n = len(candidates)
    iter_order = candidates[::-1]   # index 0 = latest iteration (top row)

    # Shared log-spaced bins from all data
    all_evs = np.concatenate([
        np.concatenate([d['loss_hessian_spectrum'].numpy() for d in all_data[it]])
        for it in candidates
    ])
    pos_evs = all_evs[all_evs > 0]
    log_bins = np.logspace(np.log10(pos_evs.min()), np.log10(pos_evs.max()), 201)
    x_min, x_max = log_bins[0], log_bins[-1]

    # Per-iteration histograms
    hists = {}
    for it in iter_order:
        evs = np.concatenate([d['loss_hessian_spectrum'].numpy() for d in all_data[it]])
        evs = evs[evs > 0]
        counts, _ = np.histogram(evs, bins=log_bins)
        hists[it] = counts

    palette = sns.color_palette('viridis_r', n)

    # Ridge layout: overlapping rows, top row drawn last (highest z-order)
    row_h = 0.9           # inches per row (before overlap)
    overlap = 0.3         # fraction of row_h to overlap
    fig_h = max(4, n * row_h * (1 - overlap) + row_h * 2)
    fig = plt.figure(figsize=(9, fig_h))
    gs = fig.add_gridspec(n, 1, hspace=-(overlap), top=0.97)

    # Create axes bottom-first so the top row (index 0) is created last → highest z-order
    axes = [None] * n
    for i in range(n - 1, -1, -1):
        axes[i] = fig.add_subplot(gs[i])

    for i, it in enumerate(iter_order):
        ax = axes[i]
        color = palette[i]
        counts = hists[it]
        y_max = counts.max() if counts.max() > 0 else 1

        # White baseline fill to mask the row below (ridge effect)
        ax.fill_between([x_min, x_max], [0, 0], [-y_max * 0.15, -y_max * 0.15],
                        color='white', zorder=2)

        # Histogram bars
        ax.bar(log_bins[:-1], counts, width=np.diff(log_bins),
               color=color, align='edge', linewidth=0, zorder=3)

        ax.set_xscale('log')
        ax.set_xlim(1e-5, 1e-2)
        ax.set_ylim(-y_max * 0.15, y_max * 1.35)
        ax.set_yticks([])
        ax.patch.set_facecolor('white')

        ax.spines['left'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        if i < n - 1:
            ax.spines['bottom'].set_visible(False)
            ax.tick_params(bottom=False, labelbottom=False)
        else:
            ax.set_xlabel('Loss-Hessian eigenvalue', fontsize=12)

        ax.text(-0.01, 0.3, f'{it}',
                transform=ax.transAxes, ha='right', va='center', fontsize=11)

    fig.suptitle('Loss-Hessian spectrum evolution', fontsize=14, y=1.0)

    path = output_dir / 'plot5_loss_hessian_ridge.png'
    fig.savefig(path, bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f"  saved → {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--metrics_dir', default='metrics/metrics-2026.03.12',
                        help='Directory with ckpt-*_trial-*.pt files')
    parser.add_argument('--output_dir', default='plots',
                        help='Directory to save figures')
    args = parser.parse_args()

    metrics_dir = Path(args.metrics_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading metrics from {metrics_dir} ...")
    all_data = load_metrics(metrics_dir)
    iters = np.array(sorted(all_data.keys()))
    print(f"  found {len(iters)} checkpoints, "
          f"up to {max(d['trial'] for trials in all_data.values() for d in trials) + 1} trials")

    avail = available_keys(all_data)
    print(f"  available keys: {sorted(avail)}")

    print("Generating plots ...")
    plot1_precond_eigenvalues(all_data, iters, output_dir, avail)
    if 'jtj_top' in avail:
        plot2_jtj(all_data, iters, output_dir)
    else:
        print("  skipping plot2_jtj (key 'jtj_top' not found)")
    if 'hessian_precond_bottom' in avail:
        plot3_hessian_top_bottom(all_data, iters, output_dir)
    else:
        print("  skipping plot3_hessian_top_bottom (key 'hessian_precond_bottom' not found)")
    has_cossim = any(k.endswith('_cossim') for k in avail)
    has_max_res = any(k.endswith('_max_res') for k in avail)
    if has_cossim or has_max_res:
        plot6_solver_convergence(all_data, iters, output_dir, avail)
    else:
        print("  skipping plot6_solver_convergence (no _cossim or _max_res keys found)")
    if 'loss_hessian_spectrum' in avail:
        plot5_loss_hessian_ridge(all_data, output_dir)
    else:
        print("  skipping plot5_loss_hessian_ridge (key 'loss_hessian_spectrum' not found)")

    print("Done.")


if __name__ == '__main__':
    main()
