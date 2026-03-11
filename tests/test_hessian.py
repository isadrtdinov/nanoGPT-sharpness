import unittest
import torch
from metrics import hessian_matvec, top_eigenvalue
from tests.toy_models import ClassificationModel, LinearRegressionModel, matvec_to_matrix


# ------------------------------------------------------------------ #
# Hessian tests — classification
# ------------------------------------------------------------------ #

class TestHessianMatvec(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(42)
        self.model = ClassificationModel(in_features=4, num_classes=3)
        self.inputs = torch.randn(8, 4)
        self.targets = torch.randint(0, 3, (8,))
        self.H = matvec_to_matrix(hessian_matvec, self.model, self.inputs, self.targets)

    def test_matvec_matches_exact(self):
        """HVP output should match materialized Hessian times v."""
        params = list(self.model.parameters())
        v = [torch.randn_like(p) for p in params]
        v_flat = torch.cat([vi.flatten() for vi in v])

        Hv = hessian_matvec(self.model, self.inputs, self.targets, v)
        Hv_flat = torch.cat([hvi.flatten() for hvi in Hv])
        Hv_exact = self.H @ v_flat

        self.assertTrue(
            torch.allclose(Hv_flat, Hv_exact, atol=1e-5),
            f"Max diff: {(Hv_flat - Hv_exact).abs().max():.2e}"
        )

    def test_hessian_is_symmetric(self):
        """Hessian matrix should be symmetric."""
        self.assertTrue(
            torch.allclose(self.H, self.H.T, atol=1e-5),
            f"Max asymmetry: {(self.H - self.H.T).abs().max():.2e}"
        )

    def test_top_eigenvalue(self):
        """Top eigenvalue from power iteration should match torch.linalg.eigvalsh."""
        exact_top = torch.linalg.eigvalsh(self.H).max().item()

        estimated = top_eigenvalue(
            hessian_matvec, self.model, [(self.inputs, self.targets)],
            max_iter=500, rtol=1e-5, cossim_thr=0.9999
        )

        self.assertAlmostEqual(
            estimated, exact_top, delta=abs(exact_top) * 1e-2,
            msg=f"Estimated: {estimated:.4f}, Exact: {exact_top:.4f}"
        )


# ------------------------------------------------------------------ #
# Hessian tests — linear regression (analytical ground truth)
# ------------------------------------------------------------------ #

class TestHessianLinearRegression(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(42)
        self.N = 20
        self.D = 5
        self.inputs = torch.randn(self.N, self.D)
        self.targets = torch.randn(self.N)
        self.model = LinearRegressionModel(self.D)
        # Analytical Hessian: (1/N) X^T X
        self.H_analytical = (1 / self.N) * self.inputs.T @ self.inputs  # [D, D]

    def test_matvec_matches_analytical(self):
        """HVP should match (1/N) X^T X v."""
        params = list(self.model.parameters())
        v = [torch.randn_like(p) for p in params]
        v_flat = torch.cat([vi.flatten() for vi in v])

        Hv = hessian_matvec(self.model, self.inputs, self.targets, v)
        Hv_flat = torch.cat([hvi.flatten() for hvi in Hv])

        Hv_analytical = self.H_analytical @ v_flat

        self.assertTrue(
            torch.allclose(Hv_flat, Hv_analytical, atol=1e-5),
            f"Max diff: {(Hv_flat - Hv_analytical).abs().max():.2e}"
        )

    def test_hessian_matrix_matches_analytical(self):
        """Materialized Hessian should match (1/N) X^T X."""
        H = matvec_to_matrix(hessian_matvec, self.model, self.inputs, self.targets)

        self.assertTrue(
            torch.allclose(H, self.H_analytical, atol=1e-5),
            f"Max diff: {(H - self.H_analytical).abs().max():.2e}"
        )

    def test_top_eigenvalue_matches_analytical(self):
        """Top eigenvalue should match largest eigenvalue of (1/N) X^T X."""
        exact_top = torch.linalg.eigvalsh(self.H_analytical).max().item()

        estimated = top_eigenvalue(
            hessian_matvec, self.model, [(self.inputs, self.targets)],
            max_iter=500, rtol=1e-5, cossim_thr=0.9999
        )

        self.assertAlmostEqual(
            estimated, exact_top, delta=abs(exact_top) * 1e-2,
            msg=f"Estimated: {estimated:.4f}, Exact: {exact_top:.4f}"
        )

    def test_hessian_independent_of_params(self):
        """
        Hessian of MSE is parameter-independent —
        it should be the same before and after a random parameter perturbation.
        """
        H1 = matvec_to_matrix(hessian_matvec, self.model, self.inputs, self.targets)

        with torch.no_grad():
            for p in self.model.parameters():
                p.add_(torch.randn_like(p))

        H2 = matvec_to_matrix(hessian_matvec, self.model, self.inputs, self.targets)

        self.assertTrue(
            torch.allclose(H1, H2, atol=1e-5),
            f"Max diff: {(H1 - H2).abs().max():.2e}"
        )


if __name__ == '__main__':
    unittest.main()
