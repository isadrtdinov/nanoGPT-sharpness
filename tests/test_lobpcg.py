import unittest
import torch
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from metrics import top_eigenvalue_lobpcg, hessian_matvec, gn_matvec
from tests.toy_models import LinearRegressionModel, TwoLayerClassificationModel, matvec_to_matrix


def _make_lr_data(N=20, d=4, seed=0):
    torch.manual_seed(seed)
    inputs  = torch.randn(N, d)
    targets = torch.randn(N)
    return inputs, targets


def _analytical_hessian_eigenvalues(inputs):
    """Hessian of (1/N)||Xw-y||^2 / 2 is (1/N) X^T X."""
    N = inputs.shape[0]
    H = inputs.T @ inputs / N
    return torch.linalg.eigvalsh(H).flip(0)  # descending


class TestLobpcgEigenvalue(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.N, self.d = 30, 5
        self.inputs, self.targets = _make_lr_data(self.N, self.d, seed=7)
        self.model = LinearRegressionModel(self.d)
        self.batches = [(self.inputs, self.targets)]
        self.analytical = _analytical_hessian_eigenvalues(self.inputs)

    def test_top_eigenvalue_matches_analytical(self):
        eig = top_eigenvalue_lobpcg(
            hessian_matvec, self.model, self.batches,
            k=1, max_iter=100, tol=1e-5,
        )
        self.assertIsInstance(eig, float)
        self.assertAlmostEqual(eig, self.analytical[0].item(), delta=1e-3)

    def test_top_k_eigenvalues(self):
        k = 3
        eigs = top_eigenvalue_lobpcg(
            hessian_matvec, self.model, self.batches,
            k=k, max_iter=200, tol=1e-5,
        )
        self.assertIsInstance(eigs, list)
        self.assertEqual(len(eigs), k)
        for j in range(k):
            self.assertAlmostEqual(eigs[j], self.analytical[j].item(), delta=1e-2)

    def test_with_all_ones_precond(self):
        """precond = ones ⟹ same eigenvalue as without precond."""
        ones_precond = [torch.ones_like(p) for p in self.model.parameters()]
        eig_no_precond = top_eigenvalue_lobpcg(
            hessian_matvec, self.model, self.batches,
            k=1, max_iter=100, tol=1e-5,
        )
        eig_ones_precond = top_eigenvalue_lobpcg(
            hessian_matvec, self.model, self.batches,
            precond=ones_precond, k=1, max_iter=100, tol=1e-5,
        )
        self.assertAlmostEqual(eig_no_precond, eig_ones_precond, delta=1e-3)

    def test_callable_batches(self):
        """callable batches + batches_per_iter."""
        def batch_fn():
            return self.inputs, self.targets

        eig = top_eigenvalue_lobpcg(
            hessian_matvec, self.model, batch_fn,
            k=1, max_iter=100, batches_per_iter=2, tol=1e-5,
        )
        self.assertAlmostEqual(eig, self.analytical[0].item(), delta=1e-3)

    def test_returns_descending_order(self):
        k = 3
        eigs = top_eigenvalue_lobpcg(
            hessian_matvec, self.model, self.batches,
            k=k, max_iter=200, tol=1e-5,
        )
        for j in range(k - 1):
            self.assertGreaterEqual(eigs[j], eigs[j + 1])


class TestLobpcgConjugateDirection(unittest.TestCase):
    def test_conjugate_direction_convergence(self):
        """Conjugate direction (3-term recurrence) converges on an ill-conditioned problem."""
        torch.manual_seed(0)
        N, d = 40, 5
        # Scale columns so condition number is large (~16^2 = 256)
        inputs = torch.randn(N, d) * torch.tensor([1.0, 2.0, 4.0, 8.0, 16.0])
        targets = torch.randn(N)
        model = LinearRegressionModel(d)
        batches = [(inputs, targets)]
        analytical = _analytical_hessian_eigenvalues(inputs)

        # With conjugate directions, should converge within a modest iteration budget
        eig = top_eigenvalue_lobpcg(
            hessian_matvec, model, batches,
            k=1, max_iter=30, tol=1e-4,
        )
        self.assertAlmostEqual(eig, analytical[0].item(), delta=1e-2)


class TestLobpcgMultiParam(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        N, d, h, C = 40, 6, 8, 3
        self.inputs  = torch.randn(N, d)
        self.targets = torch.randint(0, C, (N,))
        self.model   = TwoLayerClassificationModel(d, h, C)
        self.batches = [(self.inputs, self.targets)]

    def _dense_top_eigenvalues(self, k):
        M = matvec_to_matrix(hessian_matvec, self.model, self.inputs, self.targets)
        eigs = torch.linalg.eigvalsh(M).flip(0)
        return eigs[:k].tolist()

    def test_multi_param_model(self):
        k = 10
        eigs_lobpcg = top_eigenvalue_lobpcg(
            hessian_matvec, self.model, self.batches,
            k=k, max_iter=200, tol=1e-5,
        )
        eigs_exact = self._dense_top_eigenvalues(k)
        self.assertEqual(len(eigs_lobpcg), k)
        for j in range(k):
            self.assertAlmostEqual(eigs_lobpcg[j], eigs_exact[j], delta=1e-2)


class TestLobpcgGaussNewton(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        N, d, h, C = 40, 6, 8, 3
        self.inputs  = torch.randn(N, d)
        self.targets = torch.randint(0, C, (N,))
        self.model   = TwoLayerClassificationModel(d, h, C)
        self.batches = [(self.inputs, self.targets)]

    def _dense_top_eigenvalues(self, k):
        M = matvec_to_matrix(gn_matvec, self.model, self.inputs, self.targets)
        eigs = torch.linalg.eigvalsh(M).flip(0)
        return eigs[:k].tolist()

    def test_top_k_eigenvalues(self):
        k = 10
        eigs_lobpcg = top_eigenvalue_lobpcg(
            gn_matvec, self.model, self.batches,
            k=k, max_iter=200, tol=1e-5,
        )
        eigs_exact = self._dense_top_eigenvalues(k)
        self.assertIsInstance(eigs_lobpcg, list)
        self.assertEqual(len(eigs_lobpcg), k)
        for j in range(k):
            self.assertAlmostEqual(eigs_lobpcg[j], eigs_exact[j], delta=1e-2)

    def test_eigenvalues_are_nonnegative(self):
        """GN matrix is PSD; all top eigenvalues must be >= 0."""
        k = 10
        eigs = top_eigenvalue_lobpcg(
            gn_matvec, self.model, self.batches,
            k=k, max_iter=200, tol=1e-5,
        )
        for j, e in enumerate(eigs):
            self.assertGreaterEqual(e, -1e-4, msg=f"eigenvalue[{j}]={e:.4f} is negative")

    def test_returns_descending_order(self):
        k = 3
        eigs = top_eigenvalue_lobpcg(
            gn_matvec, self.model, self.batches,
            k=k, max_iter=100, tol=1e-5,
        )
        for j in range(k - 1):
            self.assertGreaterEqual(eigs[j], eigs[j + 1])


class TestLobpcgBottomEigenvalue(unittest.TestCase):
    """Bottom eigenvalue mode on LinearRegressionModel (all eigenvalues non-negative)."""

    def setUp(self):
        torch.manual_seed(42)
        self.N, self.d = 30, 5
        self.inputs, self.targets = _make_lr_data(self.N, self.d, seed=7)
        self.model = LinearRegressionModel(self.d)
        self.batches = [(self.inputs, self.targets)]
        self.analytical = _analytical_hessian_eigenvalues(self.inputs)  # descending

    def test_bottom_eigenvalue_matches_analytical(self):
        eig = top_eigenvalue_lobpcg(
            hessian_matvec, self.model, self.batches,
            k=1, max_iter=100, tol=1e-5, mode='bottom',
        )
        self.assertIsInstance(eig, float)
        self.assertAlmostEqual(eig, self.analytical[-1].item(), delta=1e-3)

    def test_bottom_k_eigenvalues_match_analytical(self):
        k = 3
        eigs = top_eigenvalue_lobpcg(
            hessian_matvec, self.model, self.batches,
            k=k, max_iter=200, tol=1e-5, mode='bottom',
        )
        self.assertIsInstance(eigs, list)
        self.assertEqual(len(eigs), k)
        exact = self.analytical.flip(0)[:k].tolist()  # ascending (smallest first)
        for j in range(k):
            self.assertAlmostEqual(eigs[j], exact[j], delta=1e-2)

    def test_bottom_eigenvalues_ascending_order(self):
        k = 3
        eigs = top_eigenvalue_lobpcg(
            hessian_matvec, self.model, self.batches,
            k=k, max_iter=200, tol=1e-5, mode='bottom',
        )
        for j in range(k - 1):
            self.assertLessEqual(eigs[j], eigs[j + 1])


class TestLobpcgBottomEigenvalueWithNegatives(unittest.TestCase):
    """Bottom eigenvalue mode on TwoLayerClassificationModel (Hessian can be indefinite)."""

    def setUp(self):
        torch.manual_seed(0)
        N, d, h, C = 40, 6, 8, 3
        self.inputs = torch.randn(N, d)
        self.targets = torch.randint(0, C, (N,))
        self.model = TwoLayerClassificationModel(d, h, C)
        self.batches = [(self.inputs, self.targets)]
        H = matvec_to_matrix(hessian_matvec, self.model, self.inputs, self.targets)
        # eigvalsh returns eigenvalues in ascending order
        self.exact_all = torch.linalg.eigvalsh(H)

    def test_bottom_k_eigenvalues_match_exact(self):
        k = 5
        eigs = top_eigenvalue_lobpcg(
            hessian_matvec, self.model, self.batches,
            k=k, max_iter=500, tol=1e-5, mode='bottom',
        )
        self.assertEqual(len(eigs), k)
        exact = self.exact_all[:k].tolist()  # ascending (most negative first)
        for j in range(k):
            self.assertAlmostEqual(eigs[j], exact[j], delta=1e-2)

    def test_bottom_eigenvalues_include_negative(self):
        k = 5
        eigs = top_eigenvalue_lobpcg(
            hessian_matvec, self.model, self.batches,
            k=k, max_iter=500, tol=1e-5, mode='bottom',
        )
        self.assertTrue(
            any(e < 0 for e in eigs),
            f"Expected at least one negative eigenvalue, got {eigs}"
        )

    def test_bottom_eigenvalues_ascending_order(self):
        k = 5
        eigs = top_eigenvalue_lobpcg(
            hessian_matvec, self.model, self.batches,
            k=k, max_iter=500, tol=1e-5, mode='bottom',
        )
        for j in range(k - 1):
            self.assertLessEqual(eigs[j], eigs[j + 1])


if __name__ == '__main__':
    unittest.main()
