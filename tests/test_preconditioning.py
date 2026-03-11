import unittest
import torch
from metrics import gn_matvec, top_eigenvalue, precond_vector
from tests.toy_models import ClassificationModel, TwoLayerClassificationModel, matvec_to_matrix


# ------------------------------------------------------------------ #
# Pre-conditioning tests — classification
# ------------------------------------------------------------------ #

class TestPreconditioning(unittest.TestCase):
    """
    For a diagonal preconditioner P, power iteration on PGv should return
    the top eigenvalue of PG. We verify this by comparing against
    torch.linalg.eigvalsh on the explicitly materialized PG matrix.
    """

    def setUp(self):
        torch.manual_seed(42)
        self.model = ClassificationModel(in_features=4, num_classes=3)
        self.inputs = torch.randn(8, 4)
        self.targets = torch.randint(0, 3, (8,))
        self.G = matvec_to_matrix(gn_matvec, self.model, self.inputs, self.targets)

    def _make_precond(self):
        """Random positive preconditioner with same structure as model parameters."""
        return [torch.rand_like(p) + 0.1 for p in self.model.parameters()]

    def test_preconditioned_top_eigenvalue(self):
        """
        Top eigenvalue of PG from power iteration should match
        top eigenvalue of explicitly materialized PG.
        """
        precond = self._make_precond()
        precond_flat = torch.cat([p.flatten() for p in precond])

        # Explicit PG matrix
        PG = precond_flat.unsqueeze(1) * self.G   # row-wise scaling = diag(P) @ G
        exact_top = torch.linalg.eig(PG).eigenvalues.real.max().item()

        estimated = top_eigenvalue(
            gn_matvec, self.model, [(self.inputs, self.targets)],
            precond=precond, max_iter=500, rtol=1e-5, cossim_thr=0.9999
        )

        self.assertAlmostEqual(
            estimated, exact_top, delta=abs(exact_top) * 1e-2,
            msg=f"Estimated: {estimated:.4f}, Exact: {exact_top:.4f}"
        )

    def test_identity_precond_unchanged(self):
        """
        A preconditioner of all ones should give the same result as no preconditioning.
        """
        precond_ones = [torch.ones_like(p) for p in self.model.parameters()]

        top_no_precond = top_eigenvalue(
            gn_matvec, self.model, [(self.inputs, self.targets)],
            max_iter=500, rtol=1e-5, cossim_thr=0.9999
        )
        top_with_ones = top_eigenvalue(
            gn_matvec, self.model, [(self.inputs, self.targets)],
            precond=precond_ones, max_iter=500, rtol=1e-5, cossim_thr=0.9999
        )

        self.assertAlmostEqual(
            top_no_precond, top_with_ones, delta=abs(top_no_precond) * 1e-2,
            msg=f"No precond: {top_no_precond:.4f}, Ones precond: {top_with_ones:.4f}"
        )

    def test_scalar_precond_scales_eigenvalue(self):
        """
        A uniform scalar preconditioner c should scale the top eigenvalue by c exactly.
        """
        c = 3.14
        precond_scalar = [torch.full_like(p, c) for p in self.model.parameters()]

        top_base = top_eigenvalue(
            gn_matvec, self.model, [(self.inputs, self.targets)],
            max_iter=500, rtol=1e-5, cossim_thr=0.9999
        )
        top_scaled = top_eigenvalue(
            gn_matvec, self.model, [(self.inputs, self.targets)],
            precond=precond_scalar, max_iter=500, rtol=1e-5, cossim_thr=0.9999
        )

        self.assertAlmostEqual(
            top_scaled, c * top_base, delta=abs(c * top_base) * 1e-2,
            msg=f"Expected: {c * top_base:.4f}, Got: {top_scaled:.4f}"
        )


# ------------------------------------------------------------------ #
# Generating pre-conditioning vector tests - classification
# ------------------------------------------------------------------ #

class TestPrecondVector(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(42)
        self.model = TwoLayerClassificationModel(in_features=4, hidden_size=8, num_classes=3)
        self.inputs = torch.randn(16, 4)
        self.targets = torch.randint(0, 3, (16,))
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)

    def _do_step(self):
        """Perform a single optimizer step to initialize the state."""
        self.optimizer.zero_grad()
        _, loss = self.model(self.inputs, self.targets)
        loss.backward()
        self.optimizer.step()

    def test_returns_none_before_step(self):
        """Preconditioner should be None before any optimizer step."""
        precond = precond_vector(self.model, self.optimizer)
        self.assertIsNone(precond)

    def test_structure_matches_params(self):
        """Preconditioner should have same structure as model parameters."""
        self._do_step()
        precond = precond_vector(self.model, self.optimizer)

        self.assertEqual(len(precond), len(list(self.model.parameters())))
        for pr, p in zip(precond, self.model.parameters()):
            self.assertEqual(pr.shape, p.shape,
                             f"Shape mismatch: {pr.shape} vs {p.shape}")

    def test_values_match_analytical_formula(self):
        """
        Preconditioner values should match (1 / (sqrt(v / (1 - beta2^t)) + eps)) / (1 - beta1^t).
        """
        self._do_step()
        precond = precond_vector(self.model, self.optimizer)

        beta1, beta2 = self.optimizer.param_groups[0]['betas']
        eps = self.optimizer.param_groups[0]['eps']

        for pr, p in zip(precond, self.model.parameters()):
            state = self.optimizer.state[p]
            t = state['step']
            v = state['exp_avg_sq']
            expected = 1 / (torch.sqrt(v / (1 - beta2 ** t)) + eps) / (1 - beta1 ** t)
            self.assertTrue(
                torch.allclose(pr, expected, atol=1e-6),
                f"Max diff: {(pr - expected).abs().max():.2e}"
            )

    def test_precond_is_positive(self):
        """Preconditioner values should be strictly positive."""
        self._do_step()
        precond = precond_vector(self.model, self.optimizer)
        for pr in precond:
            self.assertTrue(
                (pr > 0).all(),
                f"Non-positive preconditioner values found: min={pr.min():.2e}"
            )

    def test_larger_gradient_gives_smaller_precond(self):
        """
        A parameter with larger gradients should accumulate larger second moments
        and therefore receive a smaller preconditioner value.
        """
        self._do_step()

        # Find which parameter has the largest vs smallest second moment
        states = [self.optimizer.state[p] for p in self.model.parameters()]
        mean_sq_moments = [s['exp_avg_sq'].mean().item() for s in states]

        precond = precond_vector(self.model, self.optimizer)
        mean_precond = [pr.mean().item() for pr in precond]

        # Parameter with larger second moment should have smaller preconditioner
        max_moment_idx = max(range(len(mean_sq_moments)), key=lambda i: mean_sq_moments[i])
        min_moment_idx = min(range(len(mean_sq_moments)), key=lambda i: mean_sq_moments[i])

        self.assertLess(
            mean_precond[max_moment_idx], mean_precond[min_moment_idx],
            f"Larger moment ({mean_sq_moments[max_moment_idx]:.4f}) should give "
            f"smaller precond ({mean_precond[max_moment_idx]:.4f}) than "
            f"smaller moment ({mean_sq_moments[min_moment_idx]:.4f}) -> "
            f"precond ({mean_precond[min_moment_idx]:.4f})"
        )

    def test_uniform_moments_give_uniform_precond(self):
        """
        If all second moments are equal, preconditioner should be
        the same scalar value for all parameters.
        """
        self._do_step()

        # Manually set all second moments to the same value
        uniform_v = 0.01
        for p in self.model.parameters():
            self.optimizer.state[p]['exp_avg_sq'].fill_(uniform_v)

        precond = precond_vector(self.model, self.optimizer)

        beta1, beta2 = self.optimizer.param_groups[0]['betas']
        eps = self.optimizer.param_groups[0]['eps']
        t = self.optimizer.state[next(self.model.parameters())]['step']
        expected_scalar = 1 / (torch.sqrt(torch.tensor(uniform_v / (1 - beta2 ** t))) + eps) / (1 - beta1 ** t)

        for pr in precond:
            self.assertTrue(
                torch.allclose(pr, expected_scalar.expand_as(pr), atol=1e-6),
                f"Max diff: {(pr - expected_scalar).abs().max():.2e}"
            )


if __name__ == '__main__':
    unittest.main()
