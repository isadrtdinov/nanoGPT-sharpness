import unittest
import torch
from metrics import lanczos_top_eigenvalues, loss_hessian_top_eigenvalues


class TestCrossEntropyHessian(unittest.TestCase):
    """
    Tests for the cross-entropy Hessian w.r.t. logits:
        H = E_i[diag(p_i) - p_i p_i^T]
    """

    def setUp(self):
        torch.manual_seed(42)
        self.N = 32
        self.C = 100
        self.k = 10
        self.steps = 500
        self.logits = torch.randn(self.N, self.C)
        self.probs = torch.softmax(self.logits, dim=-1)  # [N, C]

        # Materialize the exact mean Hessian for reference
        self.H_exact = self._exact_hessian()

    def _exact_hessian(self):
        """Materialize E[diag(p_i) - p_i p_i^T] explicitly."""
        H = torch.zeros(self.C, self.C)
        for p in self.probs:
            H += torch.diag(p) - torch.outer(p, p)
        return H / self.N

    def _matvec(self, v):
        """Reference matvec implementation."""
        p_mean = self.probs.mean(dim=0)
        return p_mean * v - self.probs.T @ (self.probs @ v) / self.N

    # ------------------------------------------------------------------ #
    # Structural tests
    # ------------------------------------------------------------------ #

    def test_hessian_is_symmetric(self):
        """Mean cross-entropy Hessian should be symmetric."""
        self.assertTrue(
            torch.allclose(self.H_exact, self.H_exact.T, atol=1e-6),
            f"Max asymmetry: {(self.H_exact - self.H_exact.T).abs().max():.2e}"
        )

    def test_hessian_is_psd(self):
        """Mean cross-entropy Hessian should be positive semidefinite."""
        eigenvalues = torch.linalg.eigvalsh(self.H_exact)
        self.assertGreaterEqual(
            eigenvalues.min().item(), -1e-6,
            f"Most negative eigenvalue: {eigenvalues.min().item():.2e}"
        )

    def test_hessian_has_one_zero_eigenvalue(self):
        """
        diag(p) - pp^T always has 1^T as a null vector since
        H * 1 = p - p * (p^T 1) = p - p = 0.
        So H always has at least one zero eigenvalue.
        """
        ones = torch.ones(self.C)
        Hv = self._matvec(ones)
        self.assertTrue(
            torch.allclose(Hv, torch.zeros(self.C), atol=1e-6),
            f"H * 1 should be zero, got norm: {Hv.norm():.2e}"
        )

    def test_eigenvalues_bounded_by_probs(self):
        """
        Eigenvalues of diag(p) - pp^T are bounded above by max(p),
        since H <= diag(p) in the PSD sense.
        """
        exact_top = torch.linalg.eigvalsh(self.H_exact).max().item()
        p_mean = self.probs.mean(dim=0)
        self.assertLessEqual(
            exact_top, p_mean.max().item() + 1e-6,
            f"Top eigenvalue {exact_top:.4f} exceeds max(p_mean) {p_mean.max().item():.4f}"
        )

    # ------------------------------------------------------------------ #
    # Matvec correctness
    # ------------------------------------------------------------------ #

    def test_matvec_matches_explicit(self):
        """Matvec should match explicitly materialized H @ v."""
        v = torch.randn(self.C)
        Hv = self._matvec(v)
        Hv_exact = self.H_exact @ v
        self.assertTrue(
            torch.allclose(Hv, Hv_exact, atol=1e-6),
            f"Max diff: {(Hv - Hv_exact).abs().max():.2e}"
        )

    def test_mean_hessian_differs_from_mean_prob_hessian(self):
        """
        E[diag(p_i) - p_i p_i^T] should differ from diag(p_mean) - p_mean p_mean^T
        unless all p_i are identical.
        """
        p_mean = self.probs.mean(dim=0)
        H_mean_prob = torch.diag(p_mean) - torch.outer(p_mean, p_mean)

        self.assertFalse(
            torch.allclose(self.H_exact, H_mean_prob, atol=1e-6),
            "E[diag(p_i) - p_i p_i^T] should differ from diag(p_mean) - p_mean p_mean^T"
        )

    def test_mean_hessian_equals_mean_prob_hessian_when_identical(self):
        """
        When all p_i are identical, E[diag(p_i) - p_i p_i^T] = diag(p_mean) - p_mean p_mean^T.
        """
        # All examples get the same logits -> identical probabilities
        uniform_logits = torch.zeros(self.N, self.C)
        probs = torch.softmax(uniform_logits, dim=-1)  # all rows identical

        H = torch.zeros(self.C, self.C)
        for p in probs:
            H += torch.diag(p) - torch.outer(p, p)
        H /= self.N

        p_mean = probs.mean(dim=0)
        H_mean_prob = torch.diag(p_mean) - torch.outer(p_mean, p_mean)

        self.assertTrue(
            torch.allclose(H, H_mean_prob, atol=1e-6),
            f"Max diff: {(H - H_mean_prob).abs().max():.2e}"
        )

    # ------------------------------------------------------------------ #
    # Lanczos correctness
    # ------------------------------------------------------------------ #

    def test_lanczos_on_known_matrix(self):
        """
        Test Lanczos on a matrix with known eigenvalues,
        to isolate whether the bug is in Lanczos or the matvec.
        """
        # Diagonal matrix with known eigenvalues
        eigenvalues_true = torch.tensor([5.0, 3.0, 1.0, 0.5, 0.1])
        A = torch.diag(eigenvalues_true)

        def matvec(v):
            return A @ v

        estimated = lanczos_top_eigenvalues(matvec, dim=5, k=3, num_steps=100)
        exact = eigenvalues_true[:3]

        self.assertTrue(
            torch.allclose(estimated, exact, atol=1e-3),
            f"Estimated: {estimated}, Exact: {exact}"
        )

    def test_top_eigenvalue_matches_exact(self):
        exact_top = torch.linalg.eigvalsh(self.H_exact).max().item()

        estimated = lanczos_top_eigenvalues(
            self._matvec, dim=self.C, k=1, num_steps=self.steps
        )[0].item()

        self.assertAlmostEqual(
            estimated, exact_top, delta=abs(exact_top) * 1e-3,
            msg=f"Estimated: {estimated:.4f}, Exact: {exact_top:.4f}"
        )

    def test_top_k_eigenvalues_match_exact(self):
        exact_top_k = torch.linalg.eigvalsh(self.H_exact).flip(0)[:self.k]

        estimated = lanczos_top_eigenvalues(
            self._matvec, dim=self.C, k=self.k, num_steps=self.steps
        )

        self.assertTrue(
            torch.allclose(estimated, exact_top_k, atol=1e-3),
            f"Estimated: {estimated}, Exact: {exact_top_k}"
        )

    def test_loss_hessian_top_eigenvalues_fn(self):
        exact_top_k = torch.linalg.eigvalsh(self.H_exact).flip(0)[:self.k]

        estimated = loss_hessian_top_eigenvalues(
            self.logits, k=self.k, num_steps=self.steps
        )

        self.assertTrue(
            torch.allclose(estimated, exact_top_k, atol=1e-3),
            f"Estimated: {estimated}, Exact: {exact_top_k}"
        )


if __name__ == '__main__':
    unittest.main()
