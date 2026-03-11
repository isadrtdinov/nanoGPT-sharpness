import unittest
import torch
from metrics import hessian_matvec, gn_matvec, top_eigenvalue
from tests.toy_models import ClassificationModel, LinearRegressionModel, matvec_to_matrix


# ------------------------------------------------------------------ #
# GN tests — classification
# ------------------------------------------------------------------ #

class TestGNMatvec(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(42)
        self.model = ClassificationModel(in_features=4, num_classes=3)
        self.inputs = torch.randn(8, 4)
        self.targets = torch.randint(0, 3, (8,))
        self.G = matvec_to_matrix(gn_matvec, self.model, self.inputs, self.targets)
        self.H = matvec_to_matrix(hessian_matvec, self.model, self.inputs, self.targets)

    def test_matvec_matches_explicit(self):
        """GN matvec should match explicitly materialized G."""
        params = list(self.model.parameters())
        v = [torch.randn_like(p) for p in params]
        v_flat = torch.cat([vi.flatten() for vi in v])

        Gv = gn_matvec(self.model, self.inputs, self.targets, v)
        Gv_flat = torch.cat([gvi.flatten() for gvi in Gv])
        Gv_exact = self.G @ v_flat

        self.assertTrue(
            torch.allclose(Gv_flat, Gv_exact, atol=1e-5),
            f"Max diff: {(Gv_flat - Gv_exact).abs().max():.2e}"
        )

    def test_gn_is_symmetric(self):
        """GN matrix should be symmetric."""
        self.assertTrue(
            torch.allclose(self.G, self.G.T, atol=1e-5),
            f"Max asymmetry: {(self.G - self.G.T).abs().max():.2e}"
        )

    def test_gn_is_psd(self):
        """GN matrix should be positive semidefinite."""
        min_eigenvalue = torch.linalg.eigvalsh(self.G).min().item()
        self.assertGreaterEqual(
            min_eigenvalue, -1e-5,
            f"Most negative eigenvalue: {min_eigenvalue:.2e}"
        )

    def test_top_eigenvalue(self):
        """Top GN eigenvalue from power iteration should match torch.linalg.eigvalsh."""
        exact_top = torch.linalg.eigvalsh(self.G).max().item()

        estimated = top_eigenvalue(
            gn_matvec, self.model, [(self.inputs, self.targets)],
            max_iter=500, rtol=1e-5, cossim_thr=0.9999
        )

        self.assertAlmostEqual(
            estimated, exact_top, delta=abs(exact_top) * 1e-2,
            msg=f"Estimated: {estimated:.4f}, Exact: {exact_top:.4f}"
        )


# ------------------------------------------------------------------ #
# GN tests — linear regression
# For MSE loss, GN = Hessian = (1/N) X^T X exactly
# ------------------------------------------------------------------ #

class TestGNLinearRegression(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(42)
        self.N = 20
        self.D = 5
        self.inputs = torch.randn(self.N, self.D)
        self.targets = torch.randn(self.N)
        self.model = LinearRegressionModel(self.D)
        self.H_analytical = (1 / self.N) * self.inputs.T @ self.inputs

    def test_gn_equals_hessian_for_mse(self):
        """
        For MSE with a linear model, GN = Hessian = (1/N) X^T X exactly.
        We use loss_hessian=False since the loss Hessian w.r.t. predictions is I.
        """
        G = matvec_to_matrix(
            lambda model, inputs, targets, v: gn_matvec(
                model, inputs, targets, v, loss_hessian=False),
            self.model, self.inputs, self.targets
        )

        H = matvec_to_matrix(hessian_matvec, self.model, self.inputs, self.targets)

        self.assertTrue(
            torch.allclose(G, H, atol=1e-5),
            f"Max diff between GN and Hessian: {(G - H).abs().max():.2e}"
        )
        self.assertTrue(
            torch.allclose(G, self.H_analytical, atol=1e-5),
            f"Max diff between GN and analytical: {(G - self.H_analytical).abs().max():.2e}"
        )


if __name__ == '__main__':
    unittest.main()
