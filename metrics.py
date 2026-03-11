import torch
import torch.nn.functional as F

# ------------------------------------------------------------------ #
# Helpers
# ------------------------------------------------------------------ #

def _dot(u, v):
    return sum((a * b).sum() for a, b in zip(u, v))

def _normalize(v):
    norm = _dot(v, v).sqrt()
    return [vi / norm for vi in v]

def _scale(v, scalar):
    return [vi * scalar for vi in v]

def _add(u, v, mult_u=1.0, mult_v=1.0):
    return [mult_u * a + mult_v * b for a, b in zip(u, v)]

def _zeros_like_params(model):
    return [torch.zeros_like(p) for p in model.parameters()]

def _randn_like_params(model):
    return [torch.randn_like(p) for p in model.parameters()]

def _gram(U, V):
    """Compute k×k Gram matrix M[i,j] = _dot(U[i], V[j]).
    U, V are lists of k list-of-tensor vectors."""
    k = len(U)
    device = U[0][0].device
    M = torch.zeros(k, k, device=device)
    for i in range(k):
        for j in range(k):
            M[i, j] = _dot(U[i], V[j])
    return M

def _block_lc(vectors, coeffs):
    """Linear combination sum_j coeffs[j] * vectors[j].
    vectors: list of k list-of-tensor vectors; coeffs: 1D tensor of length k."""
    result = [torch.zeros_like(vi) for vi in vectors[0]]
    for j, c in enumerate(coeffs.tolist()):
        result = _add(result, vectors[j], mult_v=c)
    return result

def _mult(u, v):
    """Element-wise product of two list-of-tensor vectors."""
    return [a * b for a, b in zip(u, v)]

def _block_orth(vectors):
    """Modified Gram-Schmidt orthonormalization.
    Returns a list of orthonormal list-of-tensor vectors (may be shorter if near-degenerate)."""
    result = []
    for v in vectors:
        w = [vi.clone() for vi in v]
        for u in result:
            proj = _dot(w, u).item()
            w = _add(w, u, mult_v=-proj)
        norm = _dot(w, w).sqrt().item()
        if norm > 1e-6:
            result.append(_scale(w, 1.0 / norm))
    return result

# ------------------------------------------------------------------ #
# HVP
# ------------------------------------------------------------------ #

def hessian_matvec(model, inputs, targets, v):
    """
    Full Hessian matvec product (1/n sum_{i=1}^n \nabla^2_{w} L(f_w(x_i), y_i)) v
    via double backward. Assumes model returns (logits, loss).

    Args:
        model:   nn.Module, returns (logits, loss)
        inputs:  input batch
        targets: label batch
        v:       list of tensors matching model.parameters()
    Returns:
        Hv:      list of tensors matching model.parameters()
    """
    model.zero_grad()
    params = list(model.parameters())

    _, loss = model(inputs, targets)

    grads = torch.autograd.grad(loss, params, create_graph=True)
    gv = _dot(grads, v)
    Hv = torch.autograd.grad(gv, params)

    return list(Hv)


# ------------------------------------------------------------------ #
# GN matvec
# ------------------------------------------------------------------ #

def gn_matvec(model, inputs, targets, v, loss_hessian=True):
    """
    Gauss-Newton matvec product (1/n sum_{i=1}^n J_i^T \nabla^2_{z_i} L(z_i, y_i) J_i) v
    via ghost variable trick. Assumes model returns (logits, loss).

    Args:
        model:        nn.Module, returns (logits, loss)
        inputs:       input batch
        targets:      label batch
        v:            list of tensors matching model.parameters()
        loss_hessian: whether to multiply by the loss hessian J^T \nabla^2 J
    Returns:
        Gv:            list of tensors matching model.parameters()
    """
    model.zero_grad()
    params = list(model.parameters())
    logits, _ = model(inputs, targets)

    # JVP via ghost variable
    u = torch.zeros_like(logits, requires_grad=True)
    JTu = torch.autograd.grad(logits, params, grad_outputs=u, create_graph=True)
    uTJv = _dot(JTu, v)
    Jv, = torch.autograd.grad(uTJv, u)

    # Apply H analytically
    if loss_hessian:
        probs = F.softmax(logits, dim=-1).detach()
        HJv = probs * Jv - probs * (probs * Jv).sum(dim=-1, keepdim=True)
    else:
        HJv = Jv

    # Normalize over all predictions: 1 for [], N for [N], N for [N, C], N*T for [N, T, C]
    if len(logits.shape) == 0:
        num_predictions = 1
    elif len(logits.shape) == 1:
        num_predictions = logits.numel()
    else:
        num_predictions = logits[..., 0].numel()
    HJv = HJv / num_predictions

    # VJP
    Gv = torch.autograd.grad(logits, params, grad_outputs=HJv.detach())
    return list(Gv)


# ------------------------------------------------------------------ #
# Top (in absolute value) eigenvalue via power iteration
# ------------------------------------------------------------------ #

def top_eigenvalue(matvec, model, batches, precond=None, shift=0.0, max_iter=50, batches_per_iter=None,
                   warmup_iters=0, rtol=1e-3, cossim_thr=0.995, verbose=False,
                   return_cossim=False):
    """
    Estimate the top eigenvalue (in absolute value) of a matrix defined by matvec
    using power iteration.

    Args:
        matvec:           callable(model, inputs, targets, v) -> Mv
                          where v and Mv are lists of tensors matching model.parameters()
        model:            nn.Module
        batches:          either an iterable of (inputs, targets) or a function to sample batches
        precond:          optional list of tensors representing P^{-1} diagonal; uses P^{-1/2}AP^{-1/2}
        max_iter:         max power iteration steps
        batches_per_iter: how many batches per iteration sample (if batches is a function)
        warmup_iters:     during warmup_iters stopping criterion is not checked (use when shift != 0)
        rtol:             relative tolerance on eigenvalue convergence
        cossim_thr:       cosine similarity threshold for convergence
        verbose:          print iteration info
        return_cossim:    if True, also return a 1D tensor of per-iteration cosine similarities
    Returns:
        eigenvalue (float), or (eigenvalue, cossim_history) if return_cossim=True
        cossim_history is a 1D tensor of length equal to the number of iterations run
    """
    if callable(batches):
        assert batches_per_iter is not None
    else:
        batches_per_iter = len(batches)

    if precond is not None:
        precond_sqrt = [pr.sqrt() for pr in precond]
    else:
        precond_sqrt = None

    v = _normalize(_randn_like_params(model))

    eigenvalue = None
    cossim = None

    for i in range(max_iter):
        # Average matvec over batches; input to A is P^{-1/2} v when preconditioning
        v_in = _mult(v, precond_sqrt) if precond_sqrt is not None else v
        Mv = _zeros_like_params(model)
        for j in range(batches_per_iter):
            if callable(batches):
                inputs, targets = batches()
            else:
                inputs, targets = batches[j]
            Mv_batch = matvec(model, inputs, targets, v_in)
            Mv = _add(Mv, Mv_batch)
        Mv = _scale(Mv, 1 / batches_per_iter)

        # Apply P^{-1/2} to get P^{-1/2} A P^{-1/2} v
        if precond_sqrt is not None:
            PMv = _mult(Mv, precond_sqrt)
        else:
            PMv = Mv

        # cossim w.r.t. (P^{-1/2} A P^{-1/2}) v: measures alignment with top eigenvector
        cossim = _dot(v, PMv) / _dot(PMv, PMv).sqrt()

        # Apply shift AFTER computing cossim
        if i >= warmup_iters and shift != 0.0:
            PMv = _add(PMv, v, mult_u=1.0, mult_v=shift)

        new_eigenvalue = _dot(v, PMv)

        if verbose:
            if i < warmup_iters:
                print(f'Iteration #{i}: eig={new_eigenvalue.item():.2f}, '
                      f'cossim={cossim.item():.5f}')
            else:
                if i == warmup_iters and shift != 0.0:
                    print(f'Warmup ended, applying shift')
                print(f'Iteration #{i}: eig={new_eigenvalue.item() - shift:.2f}, '
                      f'cossim={cossim.item():.5f}')

        v = _normalize(PMv)

        if i >= warmup_iters:
            if eigenvalue is not None and torch.abs(new_eigenvalue / eigenvalue - 1) < rtol:
                break
            if cossim.abs().item() > cossim_thr:
                break

        eigenvalue = new_eigenvalue

    result = new_eigenvalue.item() - shift
    if return_cossim:
        return result, cossim.item()
    return result


# ------------------------------------------------------------------ #
# Top and bottom eigenvalue via power iteration
# ------------------------------------------------------------------ #

def top_and_bottom_eigenvalue(matvec, model, batches, precond=None,
                              max_iter=50, rtol=1e-3, cossim_thr=0.995,
                              verbose=False, return_cossim=False):
    """
    Estimate both the top and bottom eigenvalues of a matrix defined by matvec
    using two runs of power iteration.

    Step 1: Run with shift=0 to get the dominant eigenvalue lambda_1
            (largest in absolute value, could be positive or negative).
    Step 2: Run with shift=-lambda_1 and warmup_iters=1 to find the eigenvalue
            from the opposite end of the spectrum.

    Args:
        matvec:        callable(model, inputs, targets, v) -> Mv
        model:         nn.Module
        batches:       iterable of (inputs, targets)
        precond:       optional list of tensors for diagonal preconditioning
        max_iter:      max power iteration steps per run
        rtol:          relative tolerance on eigenvalue convergence
        cossim_thr:    cosine similarity threshold for convergence
        verbose:       print iteration info
        return_cossim: if True, also return cossim histories for both runs
    Returns:
        (top, bottom), or (top, bottom, cossim_run1, cossim_run2) if return_cossim=True
        cossim_run1/2 are 1D tensors of per-iteration cosine similarities
    """
    shared_kwargs = dict(
        precond=precond,
        max_iter=max_iter,
        rtol=rtol,
        cossim_thr=cossim_thr,
        verbose=verbose,
        return_cossim=return_cossim,
    )

    # Step 1: dominant eigenvalue (largest in absolute value)
    result_1 = top_eigenvalue(
        matvec, model, batches,
        shift=0.0,
        warmup_iters=0,
        **shared_kwargs
    )

    # Step 2: shift by -lambda_1 to expose the opposite end of the spectrum
    lambda_1 = result_1[0] if return_cossim else result_1
    result_2 = top_eigenvalue(
        matvec, model, batches,
        shift=-lambda_1,
        warmup_iters=1,
        **shared_kwargs
    )

    lambda_2 = result_2[0] if return_cossim else result_2
    top = max(lambda_1, lambda_2)
    bottom = min(lambda_1, lambda_2)

    if return_cossim:
        # Return cossim histories in (top_run, bottom_run) order matching eigenvalue order
        cossim_1, cossim_2 = result_1[1], result_2[1]
        if lambda_1 >= lambda_2:
            return top, bottom, cossim_1, cossim_2
        else:
            return top, bottom, cossim_2, cossim_1

    return top, bottom


# ------------------------------------------------------------------ #
# Top-k eigenvalues via block subspace iteration (LOBPCG-style)
# ------------------------------------------------------------------ #

def top_eigenvalue_lobpcg(matvec, model, batches, precond=None,
                          k=1, max_iter=50, batches_per_iter=None,
                          tol=1e-3, verbose=False, mode='top'):
    """
    Estimate the top-k or bottom-k eigenvalues of a matrix defined by matvec
    using block subspace iteration (LOBPCG-style).

    When precond is provided, computes eigenvalues of P^{-1/2} A P^{-1/2}
    where P^{-1} = precond (diagonal).

    Args:
        matvec:           callable(model, inputs, targets, v) -> Mv
        model:            nn.Module
        batches:          iterable of (inputs, targets) or callable returning (inputs, targets)
        precond:          optional list of tensors representing P^{-1} diagonal
        k:                number of eigenvalues to compute
        max_iter:         maximum number of iterations
        batches_per_iter: required if batches is callable; number of batches to sample once
        tol:              relative residual tolerance for convergence
        verbose:          print iteration info
        mode:             'top' (default) for largest eigenvalues in descending order;
                          'bottom' for smallest eigenvalues in ascending order
                          (implemented by negating the matvec and negating results)
    Returns:
        float if k==1, else list of k floats
        (descending order for mode='top', ascending order for mode='bottom')
    """
    if callable(batches):
        assert batches_per_iter is not None, "batches_per_iter required when batches is callable"
        fixed_batches = [batches() for _ in range(batches_per_iter)]
    else:
        fixed_batches = list(batches)

    def apply_A(v):
        Mv = _zeros_like_params(model)
        for inputs, targets in fixed_batches:
            Mv_batch = matvec(model, inputs, targets, v)
            Mv = _add(Mv, Mv_batch)
        return _scale(Mv, 1.0 / len(fixed_batches))

    if precond is not None:
        precond_sqrt = [pr.sqrt() for pr in precond]
        def apply_A_tilde(v):
            # P^{-1/2} A P^{-1/2} v
            return _mult(apply_A(_mult(v, precond_sqrt)), precond_sqrt)
    else:
        apply_A_tilde = apply_A

    if mode == 'bottom':
        _apply_A_tilde_orig = apply_A_tilde
        def apply_A_tilde(v):
            return _scale(_apply_A_tilde_orig(v), -1.0)

    # Initialize subspace
    X = _block_orth([_randn_like_params(model) for _ in range(k)])
    AX = [apply_A_tilde(x) for x in X]

    E = None
    P, AP = None, None   # conjugate directions (None on first iteration)

    for i in range(max_iter):
        # Rayleigh-Ritz in X: rotate to eigenvector basis and get fresh eigenvalue estimates
        H = _gram(X, AX)
        E_raw, C_raw = torch.linalg.eigh(H)
        E_raw, C_raw = E_raw.flip(0), C_raw.flip(1)
        E = E_raw[:k]
        C = C_raw[:, :k]
        X  = [_block_lc(X,  C[:, j]) for j in range(k)]
        AX = [_block_lc(AX, C[:, j]) for j in range(k)]

        # Residuals R[j] = AX[j] - E[j] * X[j]
        R = [_add(AX[j], X[j], mult_v=-E[j].item()) for j in range(k)]

        # Convergence check
        rel_res = [(_dot(R[j], R[j]).sqrt() / max(E[j].abs().item(), 1e-30)).item()
                   for j in range(k)]
        max_res = max(rel_res)

        if verbose:
            eigs_str = ', '.join(f'{e:.4f}' for e in E.tolist())
            print(f'Iter {i}: eigenvalues=[{eigs_str}], max_rel_res={max_res:.2e}')

        if max_res < tol:
            break

        # Expand subspace: [X, R] on first iter, [X, R, P] thereafter (3-term recurrence)
        candidates = X + R + (P if P is not None else [])
        S = _block_orth(candidates)

        AS = [apply_A_tilde(s) for s in S]

        # Rayleigh-Ritz in S, keep top k
        H_S = _gram(S, AS)
        E_S, C_S = torch.linalg.eigh(H_S)
        E_S, C_S = E_S.flip(0), C_S.flip(1)
        X  = [_block_lc(S,  C_S[:, j]) for j in range(k)]
        AX = [_block_lc(AS, C_S[:, j]) for j in range(k)]
        E  = E_S[:k]

        # Extract conjugate direction: the non-X component of the new X vectors
        extra_S  = S[k:]
        if extra_S:
            P = [_block_lc(extra_S,  C_S[k:, j]) for j in range(k)]
        else:
            P = None

    if k == 1:
        result = E[0].item()
        return -result if mode == 'bottom' else result
    result = E[:k].tolist()
    return [-e for e in result] if mode == 'bottom' else result


# ------------------------------------------------------------------ #
# Pre-conditioning vector
# ------------------------------------------------------------------ #
def precond_vector(model, optimizer):
    """
    Compute diagonal preconditioner from Adam state,
    aligned to model.named_parameters() ordering.
    """
    def get_precond_factor(v, t, beta1, beta2, eps):
        return 1 / (torch.sqrt(v / (1 - beta2 ** t)) + eps) / (1 - beta1 ** t)

    if not optimizer.state:
        return None

    (beta1, beta2) = optimizer.param_groups[0]['betas']
    eps = optimizer.param_groups[0]['eps']

    # Build a name -> param mapping from the optimizer's param groups
    # optimizer.state_dict() stores state by index, but also stores
    # param_groups with named structure we can zip against model params
    named_params = dict(model.named_parameters())

    # Align optimizer state to model parameter names explicitly
    precond = []
    for name, p in model.named_parameters():
        state = optimizer.state.get(p)
        if state is None:
            # fallback: optimizer not yet stepped for this param
            precond.append(torch.ones_like(p))
        else:
            precond.append(get_precond_factor(
                state['exp_avg_sq'], state['step'], beta1, beta2, eps
            ))

    return precond


# ------------------------------------------------------------------ #
# Lanczos algorithm implementation
# ------------------------------------------------------------------ #
def lanczos_top_eigenvalues(matvec_fn, dim, k=1, num_steps=50, device='cpu'):
    """
    Lanczos algorithm for top-k eigenvalues of a symmetric operator.

    Args:
        matvec_fn: callable(v) -> Av, where A is symmetric, v is [dim]
        dim:       dimension of the operator
        k:         number of top eigenvalues
        num_steps: number of Lanczos steps (>= k, more = more accurate)
        device:    torch device
    Returns:
        eigenvalues: tensor of shape [k], top-k eigenvalues in descending order
    """
    num_steps = min(num_steps, dim)

    # Random unit starting vector
    v = torch.randn(dim, device=device)
    v = v / v.norm()

    # Storage for Lanczos vectors and tridiagonal entries
    V = torch.zeros(num_steps, dim, device=device)  # Lanczos basis [m, dim]
    alphas = torch.zeros(num_steps, device=device)  # diagonal
    betas = torch.zeros(num_steps - 1, device=device)  # off-diagonal

    V[0] = v
    w = matvec_fn(v)
    alphas[0] = v @ w
    w = w - alphas[0] * v

    for j in range(1, num_steps):
        beta = w.norm()
        if beta < 1e-10:
            num_steps = j
            break

        betas[j - 1] = beta
        v_new = w / beta

        # Full re-orthogonalization against all previous vectors
        v_new = v_new - V[:j].T @ (V[:j] @ v_new)
        v_new = v_new / v_new.norm()  # re-normalize after orthogonalization

        V[j] = v_new
        w = matvec_fn(v_new)
        alphas[j] = v_new @ w

        # Subtract projections onto previous two vectors
        w = w - alphas[j] * v_new - beta * V[j - 1]

    # Build tridiagonal matrix T
    T = torch.diag(alphas[:num_steps]) \
      + torch.diag(betas[:num_steps - 1], diagonal=1) \
      + torch.diag(betas[:num_steps - 1], diagonal=-1)

    eigenvalues = torch.linalg.eigvalsh(T)
    return eigenvalues.flip(0)[:k]


# ------------------------------------------------------------------ #
# Loss hessian top eigenvalues
# ------------------------------------------------------------------ #
def loss_hessian_top_eigenvalues(logits, k=1, num_steps=50):
    """
    Compute top-k eigenvalues of the mean cross-entropy Hessian w.r.t. logits:
        H = E_i[diag(p_i) - p_i p_i^T]
          = diag(E_i[p_i]) - E_i[p_i p_i^T]

    Args:
        logits:    tensor of shape [N, C]
        k:         number of top eigenvalues
        num_steps: number of Lanczos steps
    Returns:
        eigenvalues: tensor of shape [k], top-k eigenvalues in descending order
    """
    probs = torch.softmax(logits, dim=-1)       # [N, C]
    p_mean = probs.mean(dim=0)                  # [C]
    # E[p p^T] != p_mean p_mean^T in general
    # matvec for E[diag(p) - pp^T] v = E[p*v - p(p^T v)]
    #                                 = p_mean * v - E[p (p^T v)]
    #                                 = p_mean * v - (probs^T @ (probs @ v)) / N

    def matvec(v):
        # probs @ v: [N], per-sample dot products p_i^T v
        # probs.T @ (probs @ v): [C], sum_i p_i (p_i^T v) / N
        return p_mean * v - probs.T @ (probs @ v) / probs.shape[0]

    return lanczos_top_eigenvalues(
        matvec, dim=probs.shape[1], k=k,
        num_steps=num_steps, device=logits.device
    )
