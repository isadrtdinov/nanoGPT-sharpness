# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Simple PyTorch HVP for multiple vectors (no vmap complications)
"""

import torch
import torch.nn as nn
from torch.func import functional_call, grad, jvp
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from torch import Tensor
from typing import Tuple, Dict

import cupy as cp
from cupyx.scipy.sparse.linalg import LinearOperator, lobpcg

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def compute_hvp(model: nn.Module, batch: Tuple, vec: Tensor, P: Tensor = None):

    # extract the examples
    x, y = batch
    params_dict = dict(model.named_parameters())

    if P is not None:
        vec = vec / P.sqrt()

    vec_dict = {name:torch.empty_like(param) for name, param in params_dict.items()}
    vector_to_parameters(vec, vec_dict.values()) # convert the vector to parameters

    def compute_loss(params: Dict[str, Tensor]):
        """ Computes the loss for the given batch """
        with torch.amp.autocast(device_type = 'cuda', dtype = torch.bfloat16):
            _, loss = functional_call(model, params, (x, y))
            return loss

    # hvp computation
    with torch.amp.autocast(device_type = 'cuda', dtype = torch.bfloat16):
        grad_fn = grad(compute_loss)
        _, hvp = jvp(grad_fn, (params_dict,), (vec_dict,)) # jvp computes the Jacobian-vector product
        flat_hvp = parameters_to_vector(hvp.values()) # flatten the hvp
        if P is not None:
            flat_hvp = flat_hvp / P.sqrt()
        return flat_hvp.detach()

def create_hvp_operator(model: nn.Module, batch: Tuple, P = None):
    """
    Create HVP operator - both function and LinearOperator
    """
    num_params = len(parameters_to_vector(model.parameters()))

    def hessian_matvec(v_cupy):
        """ Matrix-vector product for LOBPCG """
        # Convert CuPy to PyTorch
        v_torch = torch.as_tensor(v_cupy, device = 'cuda', dtype = torch.float32).flatten()

        # Compute HVP using your existing function
        hvp_torch = compute_hvp(model, batch, v_torch, P)

        hvp_cupy = cp.asarray(hvp_torch.detach())

        # Convert back to CuPy
        return hvp_cupy.reshape(-1, 1)

    # Create LinearOperator
    linear_op = LinearOperator(shape = (num_params, num_params), matvec = hessian_matvec, dtype = cp.float32)

    return linear_op, num_params


def create_hvp_operator_avg(model: nn.Module, batches, P = None):
    """
    HVP LinearOperator that averages over a list of batches per matvec call.
    """
    num_params = len(parameters_to_vector(model.parameters()))

    def hessian_matvec(v_cupy):
        v_torch = torch.as_tensor(v_cupy, device='cuda', dtype=torch.float32).flatten()
        acc = None
        for batch in batches:
            hvp = compute_hvp(model, batch, v_torch, P)
            acc = hvp if acc is None else acc + hvp
        acc = acc / len(batches)
        return cp.asarray(acc.detach()).reshape(-1, 1)

    linear_op = LinearOperator(shape=(num_params, num_params), matvec=hessian_matvec, dtype=cp.float32)
    return linear_op, num_params


def lobpcg_solver(hvp_op: LinearOperator, num_params: int, eigvecs: cp.array, tol: float = 1e-9, max_iters: int = 1000):
    """ Solve the eigenvalue problem using LOBPCG"""

    # Run LOBPCG with LinearOperator
    mini_iters = 10
    num_iters = 0

    # Track best results (lowest residual)
    best_residual = cp.inf
    best_sharpness = 0.0

    # while residual > tol * num_params * sharpness:
    while best_residual > tol * num_params * best_sharpness and num_iters < max_iters:
        remaining_iters = min(mini_iters, max_iters - num_iters)

        # at initialization, eigvecs is None
        # sometimes lobpcg returns NaN, so we need to handle that
        if eigvecs is None or cp.any(cp.isnan(eigvecs)):
            eigvecs = cp.random.randn(num_params, 1).astype(cp.float32)
        else:
            eigvecs = cp.asarray(eigvecs, dtype = cp.float32)
    
        eigvecs = eigvecs / cp.linalg.norm(eigvecs)  # Normalize
        # Run LOBPCG with LinearOperator
        eigvals, eigvecs, eig_history, res_history = lobpcg(A = hvp_op, X = eigvecs, largest = True, tol = tol, maxiter = remaining_iters, retLambdaHistory = True, retResidualNormsHistory = True)
        # Check each mini iteration within this batch
        for sharpness, residual in zip(eig_history, res_history):
            # Update best if current has lower residual
            sharpness = sharpness.item()
            residual = residual.item()
            if residual < best_residual:
                best_residual = residual
                best_sharpness = sharpness
            
        num_iters += remaining_iters
    
    return best_sharpness, eigvecs, best_residual, num_iters


def get_sharpness_lobpcg(model: nn.Module, batch: Tuple, eigvecs: cp.array = None, tol: float = 1e-11, max_iters: int = 1000):
    """ Compute sharpness (top eigenvalue) using LOBPCG with LinearOperator """

    # Create the HVP LinearOperator
    hvp_op, num_params = create_hvp_operator(model, batch)

    # Run LOBPCG with LinearOperator
    sharpness, eigvecs, residual, num_iters = lobpcg_solver(hvp_op, num_params, eigvecs, tol, max_iters)

    return sharpness, eigvecs, num_iters

def get_adam_preconditioner(params: dict, loss: nn.Module, optim: torch.optim.Optimizer):
    "Assumes the existence of gradients at param.grad "
    P_dict = {}

    param_to_name = {param: name for name, param in params}
    # optim.param_groups is a list of groups. Each group element is a dict
    for group in optim.param_groups: 
        # group is a dict with keys: [params, lr, weight_decay, betas, eps ...]
        beta1, beta2 = group['betas']
        eps = group['eps']
        weight_decay = group['weight_decay']
        lr_scale = group.get('lr_scale', 1.0)

        for param in group['params']:
            # group[params] is a group of params
            if param.grad is None:
                continue

            # get name and gradients
            name = param_to_name[param]
            grad = param.grad

            # Get the optimizer state
            param_state = optim.state.get(param, {})
            step = param_state.get('step', 0) + 1

            nu = param_state.get('exp_avg_sq', torch.zeros_like(grad.data))

            # compute the bias corrected nu
            nu_next = beta2 * nu + (1-beta2) * grad**2
            nu_next = nu_next / (1-beta2**step) 

            P_dict[name] = (1 - beta1**step) * (nu_next.sqrt() + eps) / lr_scale
    P = parameters_to_vector(P_dict.values())     
    return P


def get_pre_sharpness_lobpcg(model: nn.Module, optim: torch.optim.Optimizer, batch: Tuple, eigvecs: cp.array = None, tol: float = 1e-9, max_iters: int = 1000):
    """ Compute sharpness (top eigenvalue) using LOBPCG with LinearOperator """

    x, y = batch

    def compute_loss():
        with torch.amp.autocast(device_type = 'cuda', dtype = torch.bfloat16):
            _, loss = model(x, y)
        return loss

    # Compute loss and gradients because Adam pre-conditioner requires gradients; Note grads can be utilized from critical LR computations
    # set gradients to None
    optim.zero_grad(set_to_none = True)

    loss = compute_loss()
    loss_params = loss.item()
    loss.backward()

    P = get_adam_preconditioner(model.named_parameters(), None, optim)

    optim.zero_grad(set_to_none = True) # free up the memory for sharpness computation
    # Create the HVP LinearOperator
    hvp_op, num_params = create_hvp_operator(model, batch, P)

    # Run LOBPCG with LinearOperator
    sharpness, eigvecs, residual, num_iters = lobpcg_solver(hvp_op, num_params, eigvecs, tol, max_iters)

    return sharpness, eigvecs, num_iters