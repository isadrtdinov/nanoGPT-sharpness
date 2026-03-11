import unittest
import torch
from metrics import hessian_matvec, gn_matvec, top_eigenvalue, top_and_bottom_eigenvalue
from tests.toy_models import TwoLayerClassificationModel, matvec_to_matrix


# ------------------------------------------------------------------ #
# Multi-parameter Hessian tests
# ------------------------------------------------------------------ #

class TestMultiParamHessian(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(42)
        self.model = TwoLayerClassificationModel(in_features=4, hidden_size=8, num_classes=3)
        self.inputs = torch.randn(16, 4)
        self.targets = torch.randint(0, 3, (16,))
        self.H = matvec_to_matrix(hessian_matvec, self.model, self.inputs, self.targets)

    def test_matvec_matches_exact(self):
        """HVP should match materialized Hessian for multi-param model."""
        params = list(self.model.parameters())
        v = [torch.randn_like(p) for p in params]
        v_flat = torch.cat([vi.flatten() for vi in v])

        Hv = hessian_matvec(self.model, self.inputs, self.targets, v)
        Hv_flat = torch.cat([hvi.flatten() for hvi in Hv])

        self.assertTrue(
            torch.allclose(Hv_flat, self.H @ v_flat, atol=1e-5),
            f"Max diff: {(Hv_flat - self.H @ v_flat).abs().max():.2e}"
        )

    def test_hessian_is_symmetric(self):
        self.assertTrue(
            torch.allclose(self.H, self.H.T, atol=1e-5),
            f"Max asymmetry: {(self.H - self.H.T).abs().max():.2e}"
        )

    def test_top_eigenvalue(self):
        exact_top = torch.linalg.eigvalsh(self.H).max().item()
        estimated = top_eigenvalue(
            hessian_matvec, self.model, [(self.inputs, self.targets)],
            max_iter=500, rtol=1e-5, cossim_thr=0.9999
        )
        self.assertAlmostEqual(
            estimated, exact_top, delta=abs(exact_top) * 1e-2,
            msg=f"Estimated: {estimated:.4f}, Exact: {exact_top:.4f}"
        )

    def test_preconditioned_top_eigenvalue(self):
        """Top eigenvalue of P^{-1}H should match torch.linalg.eig."""
        precond = [torch.rand_like(p) + 0.1 for p in self.model.parameters()]
        precond_flat = torch.cat([p.flatten() for p in precond])

        PinvH = precond_flat.unsqueeze(1) * self.H
        exact_top = torch.linalg.eig(PinvH).eigenvalues.real.max().item()

        estimated = top_eigenvalue(
            hessian_matvec, self.model, [(self.inputs, self.targets)],
            precond=precond, max_iter=500, rtol=1e-5, cossim_thr=0.9999
        )
        self.assertAlmostEqual(
            estimated, exact_top, delta=abs(exact_top) * 1e-2,
            msg=f"Estimated: {estimated:.4f}, Exact: {exact_top:.4f}"
        )

    def test_top_and_bottom_eigenvalue(self):
        """
        top_and_bottom_eigenvalue should recover both the largest and smallest
        eigenvalues of the Hessian, including negative ones.
        """
        exact_eigenvalues = torch.linalg.eigvalsh(self.H)
        exact_top = exact_eigenvalues.max().item()
        exact_bottom = exact_eigenvalues.min().item()

        estimated_top, estimated_bottom = top_and_bottom_eigenvalue(
            hessian_matvec, self.model, [(self.inputs, self.targets)],
            max_iter=500, rtol=1e-5, cossim_thr=0.9999
        )

        self.assertAlmostEqual(
            estimated_top, exact_top, delta=abs(exact_top) * 1e-2,
            msg=f"Top — Estimated: {estimated_top:.4f}, Exact: {exact_top:.4f}"
        )
        self.assertAlmostEqual(
            estimated_bottom, exact_bottom, delta=abs(exact_bottom) * 1e-2,
            msg=f"Bottom — Estimated: {estimated_bottom:.4f}, Exact: {exact_bottom:.4f}"
        )

    def test_top_and_bottom_eigenvalue_ordering(self):
        """Top eigenvalue should always be >= bottom eigenvalue."""
        top, bottom = top_and_bottom_eigenvalue(
            hessian_matvec, self.model, [(self.inputs, self.targets)],
            max_iter=500, rtol=1e-5, cossim_thr=0.9999
        )
        self.assertGreaterEqual(top, bottom,
                                f"Top ({top:.4f}) should be >= bottom ({bottom:.4f})")

    def test_top_and_bottom_eigenvalue_signs(self):
        """
        For cross-entropy Hessian, the top eigenvalue should be positive.
        The bottom eigenvalue may be negative (Hessian is not guaranteed PSD
        away from a minimum).
        """
        top, bottom = top_and_bottom_eigenvalue(
            hessian_matvec, self.model, [(self.inputs, self.targets)],
            max_iter=500, rtol=1e-5, cossim_thr=0.9999
        )
        self.assertGreater(top, 0.0,
                           f"Top eigenvalue should be positive, got {top:.4f}")


# ------------------------------------------------------------------ #
# Multi-parameter GN tests
# ------------------------------------------------------------------ #

class TestMultiParamGN(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(42)
        self.model = TwoLayerClassificationModel(in_features=4, hidden_size=8, num_classes=3)
        self.inputs = torch.randn(16, 4)
        self.targets = torch.randint(0, 3, (16,))
        self.G = matvec_to_matrix(gn_matvec, self.model, self.inputs, self.targets)
        self.H = matvec_to_matrix(hessian_matvec, self.model, self.inputs, self.targets)

    def test_matvec_matches_explicit(self):
        params = list(self.model.parameters())
        v = [torch.randn_like(p) for p in params]
        v_flat = torch.cat([vi.flatten() for vi in v])

        Gv = gn_matvec(self.model, self.inputs, self.targets, v)
        Gv_flat = torch.cat([gvi.flatten() for gvi in Gv])

        self.assertTrue(
            torch.allclose(Gv_flat, self.G @ v_flat, atol=1e-5),
            f"Max diff: {(Gv_flat - self.G @ v_flat).abs().max():.2e}"
        )

    def test_gn_is_symmetric(self):
        self.assertTrue(
            torch.allclose(self.G, self.G.T, atol=1e-5),
            f"Max asymmetry: {(self.G - self.G.T).abs().max():.2e}"
        )

    def test_gn_is_psd(self):
        min_eigenvalue = torch.linalg.eigvalsh(self.G).min().item()
        self.assertGreaterEqual(min_eigenvalue, -1e-5,
                                f"Most negative eigenvalue: {min_eigenvalue:.2e}")

    def test_top_eigenvalue(self):
        exact_top = torch.linalg.eigvalsh(self.G).max().item()
        estimated = top_eigenvalue(
            gn_matvec, self.model, [(self.inputs, self.targets)],
            max_iter=500, rtol=1e-5, cossim_thr=0.9999
        )
        self.assertAlmostEqual(
            estimated, exact_top, delta=abs(exact_top) * 1e-2,
            msg=f"Estimated: {estimated:.4f}, Exact: {exact_top:.4f}"
        )

    def test_preconditioned_top_eigenvalue(self):
        """Top eigenvalue of P^{-1}G should match torch.linalg.eig."""
        precond = [torch.rand_like(p) + 0.1 for p in self.model.parameters()]
        precond_flat = torch.cat([p.flatten() for p in precond])

        PinvG = precond_flat.unsqueeze(1) * self.G
        exact_top = torch.linalg.eig(PinvG).eigenvalues.real.max().item()

        estimated = top_eigenvalue(
            gn_matvec, self.model, [(self.inputs, self.targets)],
            precond=precond, max_iter=500, rtol=1e-5, cossim_thr=0.9999
        )
        self.assertAlmostEqual(
            estimated, exact_top, delta=abs(exact_top) * 1e-2,
            msg=f"Estimated: {estimated:.4f}, Exact: {exact_top:.4f}"
        )

    def test_scalar_precond_scales_eigenvalue(self):
        """Uniform scalar preconditioner c should scale the top eigenvalue by c."""
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
# Multi-batch averaging tests
# ------------------------------------------------------------------ #

class TestMultiParamMultipleBatches(unittest.TestCase):
    """
    Test that averaging over multiple batches works correctly
    for both hessian and GN matvec.
    """

    def setUp(self):
        torch.manual_seed(42)
        self.model = TwoLayerClassificationModel(in_features=4, hidden_size=8, num_classes=3)
        # Two batches
        self.batches = [
            (torch.randn(8, 4), torch.randint(0, 3, (8,))),
            (torch.randn(8, 4), torch.randint(0, 3, (8,))),
        ]
        # Single combined batch for reference
        self.combined_inputs = torch.cat([b[0] for b in self.batches])
        self.combined_targets = torch.cat([b[1] for b in self.batches])

    def test_hessian_multibatch_matches_combined(self):
        """
        Averaging hessian matvec over two batches should match
        a single matvec on the combined batch.
        """
        params = list(self.model.parameters())
        v = [torch.randn_like(p) for p in params]

        Hv_multibatch = [torch.zeros_like(p) for p in params]
        for inputs, targets in self.batches:
            Hv_batch = hessian_matvec(self.model, inputs, targets, v)
            Hv_multibatch = [hv_multibatch + hv_batch for hv_multibatch, hv_batch in zip(Hv_multibatch, Hv_batch)]
        Hv_multibatch = [hv / len(self.batches) for hv in Hv_multibatch]

        Hv_combined = hessian_matvec(self.model, self.combined_inputs, self.combined_targets, v)

        for hv_mb, hv_c in zip(Hv_multibatch, Hv_combined):
            self.assertTrue(
                torch.allclose(hv_mb, hv_c, atol=1e-5),
                f"Max diff: {(hv_mb - hv_c).abs().max():.2e}"
            )

    def test_top_eigenvalue_multibatch(self):
        """
        Top eigenvalue estimated over multiple batches should be close
        to that estimated on the combined batch.
        """
        top_multibatch = top_eigenvalue(
            hessian_matvec, self.model, self.batches,
            max_iter=500, rtol=1e-5, cossim_thr=0.9999
        )
        top_combined = top_eigenvalue(
            hessian_matvec, self.model, [(self.combined_inputs, self.combined_targets)],
            max_iter=500, rtol=1e-5, cossim_thr=0.9999
        )
        self.assertAlmostEqual(
            top_multibatch, top_combined, delta=abs(top_combined) * 1e-2,
            msg=f"Multibatch: {top_multibatch:.4f}, Combined: {top_combined:.4f}"
        )


if __name__ == '__main__':
    unittest.main()
