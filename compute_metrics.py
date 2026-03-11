"""
compute_metrics.py

Compute spectral metrics over GPT checkpoints.

Each (checkpoint, trial) pair is saved as an individual file:
    <output_dir>/ckpt-{iteration:05d}_trial-{trial:02d}.pt

A config JSON is written once to:
    <output_dir>/config.json

Usage:
    # Process all checkpoints
    python compute_metrics.py

    # Process only specific iterations
    python compute_metrics.py --iterations 1000 5000 10000

    # Skip already-computed (checkpoint, trial) pairs
    python compute_metrics.py --skip_existing

    # Also log cosine similarity histories for all power iteration runs
    python compute_metrics.py --return_cossim
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
    precond_vector, loss_hessian_top_eigenvalues,
)


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

    # Batch config
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--block_size', type=int, default=1024)
    parser.add_argument('--num_batches', type=int, default=16,
                        help='Number of batches averaged per metric estimate')
    parser.add_argument('--num_trials', type=int, default=1)

    # Loss hessian spectrum
    parser.add_argument('--loss_hessian_k', type=int, default=128)
    parser.add_argument('--loss_hessian_num_steps', type=int, default=512)

    # Power iteration config
    parser.add_argument('--max_iter', type=int, default=30)
    parser.add_argument('--rtol', type=float, default=1e-3)
    parser.add_argument('--cossim_thr', type=float, default=0.995)
    parser.add_argument('--return_cossim', action='store_true', default=False,
                        help='Log cosine similarity at the end of all power '
                             'iteration runs. Stored in the output .pt file under keys like '
                             'hessian_top_cossim, gn_top_cossim, etc.')

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
        batch_size=args.batch_size,
        block_size=args.block_size,
        num_batches=args.num_batches,
        num_trials=args.num_trials,
        loss_hessian_k=args.loss_hessian_k,
        loss_hessian_num_steps=args.loss_hessian_num_steps,
        max_iter=args.max_iter,
        rtol=args.rtol,
        cossim_thr=args.cossim_thr,
        return_cossim=args.return_cossim,
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

def compute_metrics_single(model, precond, batches, args):
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

    # 1. Hessian top and bottom
    result = top_and_bottom_eigenvalue(
        hessian_matvec, model, batches, precond=None, **power_kwargs
    )
    if args.return_cossim:
        hessian_top, hessian_bottom, hessian_top_cossim, hessian_bottom_cossim = result
    else:
        hessian_top, hessian_bottom = result
        hessian_top_cossim = hessian_bottom_cossim = None

    # 2. Preconditioned Hessian top and bottom
    result = top_and_bottom_eigenvalue(
        hessian_matvec, model, batches, precond=precond, **power_kwargs
    )
    if args.return_cossim:
        hessian_precond_top, hessian_precond_bottom, \
            hessian_precond_top_cossim, hessian_precond_bottom_cossim = result
    else:
        hessian_precond_top, hessian_precond_bottom = result
        hessian_precond_top_cossim = hessian_precond_bottom_cossim = None

    # 3. GN top
    gn_result = top_eigenvalue(gn_matvec, model, batches, precond=None, **power_kwargs)
    gn_top, gn_top_cossim = unpack(gn_result)

    # 4. Preconditioned GN top
    gn_precond_result = top_eigenvalue(gn_matvec, model, batches, precond=precond, **power_kwargs)
    gn_precond_top, gn_precond_top_cossim = unpack(gn_precond_result)

    # 5. J^T J top (GN with loss_hessian=False)
    def jtj_matvec(model, inputs, targets, v):
        return gn_matvec(model, inputs, targets, v, loss_hessian=False)

    jtj_result = top_eigenvalue(jtj_matvec, model, batches, precond=None, **power_kwargs)
    jtj_top, jtj_top_cossim = unpack(jtj_result)

    # 6. Loss hessian spectrum
    all_logits = []
    with torch.no_grad():
        for inputs, targets in batches[:1]:
            logits, _ = model(inputs, targets)
            all_logits.append(logits.reshape(-1, logits.shape[-1]))
    logits_flat = torch.cat(all_logits, dim=0)
    loss_hessian_spectrum = loss_hessian_top_eigenvalues(
        logits_flat, k=args.loss_hessian_k, num_steps=args.loss_hessian_num_steps,
    )

    metrics = dict(
        hessian_top=torch.tensor(hessian_top),
        hessian_bottom=torch.tensor(hessian_bottom),
        hessian_precond_top=torch.tensor(hessian_precond_top),
        hessian_precond_bottom=torch.tensor(hessian_precond_bottom),
        gn_top=torch.tensor(gn_top),
        gn_precond_top=torch.tensor(gn_precond_top),
        jtj_top=torch.tensor(jtj_top),
        loss_hessian_spectrum=loss_hessian_spectrum.cpu(),
    )

    # Attach cossim histories when requested — variable length tensors, one per run
    if args.return_cossim:
        metrics.update(
            hessian_top_cossim=torch.tensor(hessian_top_cossim),
            hessian_bottom_cossim=torch.tensor(hessian_bottom_cossim),
            hessian_precond_top_cossim=torch.tensor(hessian_precond_top_cossim),
            hessian_precond_bottom_cossim=torch.tensor(hessian_precond_bottom_cossim),
            gn_top_cossim=torch.tensor(gn_top_cossim),
            gn_precond_top_cossim=torch.tensor(gn_precond_top_cossim),
            jtj_top_cossim=torch.tensor(jtj_top_cossim),
        )

    return metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

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
    if args.return_cossim:
        print('Cosine similarity logging enabled.')

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

        batches = sample_batches(get_batch, args.num_batches)
        metrics = compute_metrics_single(model, precond, batches, args)

        torch.save({'iteration': iteration, 'trial': trial, **metrics}, out_path)

        print(f'  hessian_top={metrics["hessian_top"]:.2f}, '
              f'hessian_bottom={metrics["hessian_bottom"]:.2f}, '
              f'gn_top={metrics["gn_top"]:.2f}')
        if args.return_cossim:
            print(f'  hessian_top final cossim={metrics["hessian_top_cossim"]:.4f}, '
                  f'gn_top final cossim={metrics["gn_top_cossim"]:.4f}')
        print(f'  Saved to {out_path}')

    print(f'\nAll done. Results in {args.output_dir}/')


if __name__ == '__main__':
    main()
