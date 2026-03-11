"""
compute_metrics.py

Compute spectral metrics over GPT checkpoints.

Each (checkpoint, trial) pair is saved as an individual file:
    <output_dir>/ckpt-{iteration:05d}_trial-{trial:02d}.pt

A config JSON is written once to:
    <output_dir>/config.json

Usage:
    # List available metrics and exit (no --metrics provided)
    python compute_metrics.py

    # Process all checkpoints with specific metrics
    python compute_metrics.py --metrics hessian gn_top loss_hessian

    # Use LOBPCG solver instead of power iteration
    python compute_metrics.py --metrics hessian gn_top --solver lobpcg

    # Process only specific iterations
    python compute_metrics.py --metrics hessian --iterations 1000 5000 10000

    # Skip already-computed (checkpoint, trial) pairs
    python compute_metrics.py --metrics hessian --skip_existing

    # Also log cosine similarity histories (power_iteration solver)
    python compute_metrics.py --metrics hessian gn_top --return_cossim

    # Also log max relative residuals (lobpcg solver)
    python compute_metrics.py --metrics hessian gn_top --solver lobpcg --return_max_res

Available metrics:
    hessian         -- top and bottom eigenvalues of the full Hessian
    hessian_precond -- top and bottom eigenvalues of the Adam-preconditioned Hessian
    gn_top          -- top eigenvalue of the Gauss-Newton matrix
    gn_precond_top  -- top eigenvalue of the Adam-preconditioned Gauss-Newton matrix
    jtj_top         -- top eigenvalue of J^T J
    loss_hessian    -- top-k eigenvalues of the cross-entropy Hessian w.r.t. logits
"""

import os
import re
import json
import argparse
import numpy as np
import torch

from model import GPTConfig, GPT, CausalSelfAttention
from metrics import (
    hessian_matvec, gn_matvec,
    top_eigenvalue, top_and_bottom_eigenvalue,
    top_eigenvalue_lobpcg,
    precond_vector, loss_hessian_top_eigenvalues,
)


AVAILABLE_METRICS = ['hessian', 'hessian_precond', 'gn_top', 'gn_precond_top', 'jtj_top', 'loss_hessian']


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description='Compute spectral metrics over GPT checkpoints')

    # Paths
    parser.add_argument('--ckpt_dir', type=str, default='ckpts')
    parser.add_argument('--data_dir', type=str, default=os.path.join('data', 'openwebtext'))
    parser.add_argument('--output_dir', type=str, default='metrics',
                        help='Directory for per-(checkpoint, trial) .pt files and config.json')
    parser.add_argument('--split', type=str, default='train', choices=['train', 'val'])

    # Checkpoint selection
    parser.add_argument('--iterations', type=int, nargs='*', default=None,
                        help='Iterations to process. If omitted, all discovered checkpoints are used.')
    parser.add_argument('--skip_existing', action='store_true', default=False,
                        help='Skip (checkpoint, trial) pairs whose output file already exists')

    # Metric selection
    parser.add_argument('--metrics', type=str, nargs='*', default=None,
                        choices=AVAILABLE_METRICS, metavar='METRIC',
                        help=f'Metrics to compute: {", ".join(AVAILABLE_METRICS)}. '
                             'If omitted, nothing is computed and the script exits.')

    # Batch config
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--block_size', type=int, default=1024)
    parser.add_argument('--num_batches', type=int, default=16,
                        help='Number of batches averaged per metric estimate')
    parser.add_argument('--num_trials', type=int, default=1)

    # Loss hessian spectrum
    parser.add_argument('--loss_hessian_k', type=int, default=1024)
    parser.add_argument('--loss_hessian_num_steps', type=int, default=2048)

    # Solver selection
    parser.add_argument('--solver', type=str, default='power_iteration',
                        choices=['power_iteration', 'lobpcg'],
                        help='Eigenvalue solver for spectral metrics (default: power_iteration)')

    # Power iteration config
    parser.add_argument('--max_iter', type=int, default=30)
    parser.add_argument('--rtol', type=float, default=1e-3)
    parser.add_argument('--cossim_thr', type=float, default=0.995)
    parser.add_argument('--return_cossim', action='store_true', default=False,
                        help='(power_iteration) Store per-run cosine similarity in output .pt file')

    # LOBPCG config
    parser.add_argument('--lobpcg_tol', type=float, default=1e-2,
                        help='(lobpcg) Relative residual tolerance for convergence')
    parser.add_argument('--return_max_res', action='store_true', default=False,
                        help='(lobpcg) Store per-run max relative residual in output .pt file')

    # Model config
    parser.add_argument('--n_layer', type=int, default=12)
    parser.add_argument('--n_head', type=int, default=12)
    parser.add_argument('--n_embd', type=int, default=768)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--bias', action='store_true', default=False)
    parser.add_argument('--vocab_size', type=int, default=50304)

    # Optimizer config (for preconditioner)
    parser.add_argument('--learning_rate', type=float, default=6e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-1)
    parser.add_argument('--beta1', type=float, default=0.9)
    parser.add_argument('--beta2', type=float, default=0.95)

    parser.add_argument('--verbose', action='store_true', default=False)

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Output file naming
# ---------------------------------------------------------------------------

def result_filename(output_dir, iteration, trial):
    return os.path.join(output_dir, f'ckpt-{iteration:05d}_trial-{trial:02d}.pt')


def config_filename(output_dir):
    return os.path.join(output_dir, 'config.json')


# ---------------------------------------------------------------------------
# Config persistence
# ---------------------------------------------------------------------------

def save_config(args):
    os.makedirs(args.output_dir, exist_ok=True)
    config = dict(
        metrics=args.metrics,
        batch_size=args.batch_size,
        block_size=args.block_size,
        num_batches=args.num_batches,
        num_trials=args.num_trials,
        loss_hessian_k=args.loss_hessian_k,
        loss_hessian_num_steps=args.loss_hessian_num_steps,
        solver=args.solver,
        max_iter=args.max_iter,
        rtol=args.rtol,
        cossim_thr=args.cossim_thr,
        return_cossim=args.return_cossim,
        lobpcg_tol=args.lobpcg_tol,
        return_max_res=args.return_max_res,
        split=args.split,
        ckpt_dir=args.ckpt_dir,
        data_dir=args.data_dir,
    )
    with open(config_filename(args.output_dir), 'w') as f:
        json.dump(config, f, indent=2)
    print(f'Config written to {config_filename(args.output_dir)}')


# ---------------------------------------------------------------------------
# Checkpoint discovery
# ---------------------------------------------------------------------------

def find_checkpoints(ckpt_dir):
    pattern = re.compile(r'^ckpt-(\d+)\.pt$')
    iterations = []
    for fname in os.listdir(ckpt_dir):
        m = pattern.match(fname)
        if m:
            iterations.append(int(m.group(1)))
    iterations.sort()
    return iterations


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def build_model(args, device):
    model_args = dict(n_layer=args.n_layer, n_head=args.n_head, n_embd=args.n_embd,
                      block_size=args.block_size, bias=args.bias,
                      vocab_size=args.vocab_size, dropout=args.dropout)
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    for module in model.modules():
        if isinstance(module, CausalSelfAttention):
            module.flash = False
            module.register_buffer(
                'bias',
                torch.tril(torch.ones(args.block_size, args.block_size))
                     .view(1, 1, args.block_size, args.block_size)
            )
    return model.to(device)


def load_checkpoint(model, ckpt_path, args, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    keys = list(ckpt['model'].keys())
    for key in keys:
        new_key = key.replace('_orig_mod.', '')
        if new_key != key:
            ckpt['model'][new_key] = ckpt['model'][key]
            del ckpt['model'][key]
    model.load_state_dict(ckpt['model'], strict=False)
    model.eval()
    optimizer = model.configure_optimizers(
        args.weight_decay, args.learning_rate, (args.beta1, args.beta2), 'cuda'
    )
    optimizer.load_state_dict(ckpt['optimizer'])
    return model, precond_vector(model, optimizer)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def make_get_batch(args, device):
    def get_batch():
        data = np.memmap(os.path.join(args.data_dir, f'{args.split}.bin'),
                         dtype=np.uint16, mode='r')
        ix = torch.randint(len(data) - args.block_size, (args.batch_size,))
        x = torch.stack([torch.from_numpy(data[i:i+args.block_size].astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy(data[i+1:i+1+args.block_size].astype(np.int64)) for i in ix])
        return (x.pin_memory().to(device, non_blocking=True),
                y.pin_memory().to(device, non_blocking=True))
    return get_batch


def sample_batches(get_batch, num_batches):
    return [get_batch() for _ in range(num_batches)]


# ---------------------------------------------------------------------------
# Metric computation for a single (checkpoint, trial)
# ---------------------------------------------------------------------------

def _jtj_matvec(model, inputs, targets, v):
    return gn_matvec(model, inputs, targets, v, loss_hessian=False)


def compute_metrics_single(model, precond, batches, args):
    requested = set(args.metrics)
    metrics = {}

    if args.solver == 'power_iteration':
        power_kwargs = dict(
            max_iter=args.max_iter,
            rtol=args.rtol,
            cossim_thr=args.cossim_thr,
            verbose=args.verbose,
            return_cossim=args.return_cossim,
        )

        def unpack(result):
            """Return (eigenvalue, cossim_or_None) regardless of return_cossim."""
            if args.return_cossim:
                return result[0], result[1]
            return result, None

        if 'hessian' in requested:
            result = top_and_bottom_eigenvalue(
                hessian_matvec, model, batches, precond=None, **power_kwargs
            )
            if args.return_cossim:
                top, bottom, top_cossim, bot_cossim = result
                metrics['hessian_top_cossim'] = torch.tensor(top_cossim)
                metrics['hessian_bottom_cossim'] = torch.tensor(bot_cossim)
            else:
                top, bottom = result
            metrics['hessian_top'] = torch.tensor(top)
            metrics['hessian_bottom'] = torch.tensor(bottom)

        if 'hessian_precond' in requested:
            result = top_and_bottom_eigenvalue(
                hessian_matvec, model, batches, precond=precond, **power_kwargs
            )
            if args.return_cossim:
                top, bottom, top_cossim, bot_cossim = result
                metrics['hessian_precond_top_cossim'] = torch.tensor(top_cossim)
                metrics['hessian_precond_bottom_cossim'] = torch.tensor(bot_cossim)
            else:
                top, bottom = result
            metrics['hessian_precond_top'] = torch.tensor(top)
            metrics['hessian_precond_bottom'] = torch.tensor(bottom)

        if 'gn_top' in requested:
            val, cossim = unpack(top_eigenvalue(gn_matvec, model, batches, precond=None, **power_kwargs))
            metrics['gn_top'] = torch.tensor(val)
            if args.return_cossim:
                metrics['gn_top_cossim'] = torch.tensor(cossim)

        if 'gn_precond_top' in requested:
            val, cossim = unpack(top_eigenvalue(gn_matvec, model, batches, precond=precond, **power_kwargs))
            metrics['gn_precond_top'] = torch.tensor(val)
            if args.return_cossim:
                metrics['gn_precond_top_cossim'] = torch.tensor(cossim)

        if 'jtj_top' in requested:
            val, cossim = unpack(top_eigenvalue(_jtj_matvec, model, batches, precond=None, **power_kwargs))
            metrics['jtj_top'] = torch.tensor(val)
            if args.return_cossim:
                metrics['jtj_top_cossim'] = torch.tensor(cossim)

    else:  # lobpcg
        lobpcg_kwargs = dict(
            k=1,
            max_iter=args.max_iter,
            tol=args.lobpcg_tol,
            verbose=args.verbose,
            return_max_res=True,
        )

        if 'hessian' in requested:
            top, top_res = top_eigenvalue_lobpcg(
                hessian_matvec, model, batches, precond=None, mode='top', **lobpcg_kwargs
            )
            bot, bot_res = top_eigenvalue_lobpcg(
                hessian_matvec, model, batches, precond=None, mode='bottom', **lobpcg_kwargs
            )
            metrics['hessian_top'] = torch.tensor(top)
            metrics['hessian_bottom'] = torch.tensor(bot)
            if args.return_max_res:
                metrics['hessian_top_max_res'] = torch.tensor(top_res)
                metrics['hessian_bottom_max_res'] = torch.tensor(bot_res)

        if 'hessian_precond' in requested:
            top, top_res = top_eigenvalue_lobpcg(
                hessian_matvec, model, batches, precond=precond, mode='top', **lobpcg_kwargs
            )
            bot, bot_res = top_eigenvalue_lobpcg(
                hessian_matvec, model, batches, precond=precond, mode='bottom', **lobpcg_kwargs
            )
            metrics['hessian_precond_top'] = torch.tensor(top)
            metrics['hessian_precond_bottom'] = torch.tensor(bot)
            if args.return_max_res:
                metrics['hessian_precond_top_max_res'] = torch.tensor(top_res)
                metrics['hessian_precond_bottom_max_res'] = torch.tensor(bot_res)

        if 'gn_top' in requested:
            val, res = top_eigenvalue_lobpcg(
                gn_matvec, model, batches, precond=None, mode='top', **lobpcg_kwargs
            )
            metrics['gn_top'] = torch.tensor(val)
            if args.return_max_res:
                metrics['gn_top_max_res'] = torch.tensor(res)

        if 'gn_precond_top' in requested:
            val, res = top_eigenvalue_lobpcg(
                gn_matvec, model, batches, precond=precond, mode='top', **lobpcg_kwargs
            )
            metrics['gn_precond_top'] = torch.tensor(val)
            if args.return_max_res:
                metrics['gn_precond_top_max_res'] = torch.tensor(res)

        if 'jtj_top' in requested:
            val, res = top_eigenvalue_lobpcg(
                _jtj_matvec, model, batches, precond=None, mode='top', **lobpcg_kwargs
            )
            metrics['jtj_top'] = torch.tensor(val)
            if args.return_max_res:
                metrics['jtj_top_max_res'] = torch.tensor(res)

    # Loss hessian is solver-independent (uses Lanczos)
    if 'loss_hessian' in requested:
        all_logits = []
        with torch.no_grad():
            for inputs, targets in batches[:1]:
                logits, _ = model(inputs, targets)
                all_logits.append(logits.reshape(-1, logits.shape[-1]))
        logits_flat = torch.cat(all_logits, dim=0)
        metrics['loss_hessian_spectrum'] = loss_hessian_top_eigenvalues(
            logits_flat, k=args.loss_hessian_k, num_steps=args.loss_hessian_num_steps,
        ).cpu()

    return metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    if not args.metrics:
        print('No metrics specified. Use --metrics to select what to compute.')
        print(f'Available metrics: {", ".join(AVAILABLE_METRICS)}')
        return

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    print(f'Solver: {args.solver}')
    print(f'Metrics: {args.metrics}')

    os.makedirs(args.output_dir, exist_ok=True)
    save_config(args)

    # Determine which iterations to process
    if args.iterations is not None:
        iterations = sorted(args.iterations)
        available = set(find_checkpoints(args.ckpt_dir))
        missing = [it for it in iterations if it not in available]
        if missing:
            raise RuntimeError(f'Requested iterations not found in {args.ckpt_dir}: {missing}')
    else:
        iterations = find_checkpoints(args.ckpt_dir)
        if not iterations:
            raise RuntimeError(f'No checkpoints found in {args.ckpt_dir}')

    print(f'Iterations to process: {iterations}')

    # Build todo list, optionally skipping existing files
    todo = []
    for iteration in iterations:
        for trial in range(args.num_trials):
            path = result_filename(args.output_dir, iteration, trial)
            if args.skip_existing and os.path.exists(path):
                print(f'  Skipping {os.path.basename(path)} (already exists)')
                continue
            todo.append((iteration, trial))

    if not todo:
        print('Nothing to do.')
        return

    print(f'\n{len(todo)} (checkpoint, trial) pairs to compute.')

    get_batch = make_get_batch(args, device)
    model = build_model(args, device)
    current_iteration = None

    for i, (iteration, trial) in enumerate(todo):
        out_path = result_filename(args.output_dir, iteration, trial)
        print(f'\n[{i + 1}/{len(todo)}] iteration={iteration}  trial={trial + 1}/{args.num_trials}')

        if iteration != current_iteration:
            ckpt_path = os.path.join(args.ckpt_dir, f'ckpt-{iteration:05d}.pt')
            print(f'  Loading checkpoint {ckpt_path}')
            model, precond = load_checkpoint(model, ckpt_path, args, device)
            current_iteration = iteration

        torch.manual_seed(trial)
        batches = sample_batches(get_batch, args.num_batches)
        metrics = compute_metrics_single(model, precond, batches, args)

        torch.save({'iteration': iteration, 'trial': trial, **metrics}, out_path)

        # Print summary of computed scalar metrics
        summary_keys = ['hessian_top', 'hessian_bottom', 'hessian_precond_top', 'hessian_precond_bottom',
                        'gn_top', 'gn_precond_top', 'jtj_top']
        parts = [f'{k}={metrics[k]:.4f}' for k in summary_keys if k in metrics]
        if parts:
            print(f'  {", ".join(parts)}')

        # Print convergence diagnostics
        if args.solver == 'power_iteration' and args.return_cossim:
            cossim_keys = [k for k in metrics if k.endswith('_cossim')]
            cossim_parts = [f'{k}={metrics[k]:.4f}' for k in cossim_keys]
            if cossim_parts:
                print(f'  cossim: {", ".join(cossim_parts)}')
        elif args.solver == 'lobpcg' and args.return_max_res:
            res_keys = [k for k in metrics if k.endswith('_max_res')]
            res_parts = [f'{k}={metrics[k]:.2e}' for k in res_keys]
            if res_parts:
                print(f'  max_res: {", ".join(res_parts)}')

        print(f'  Saved to {out_path}')

    print(f'\nAll done. Results in {args.output_dir}/')


if __name__ == '__main__':
    main()
