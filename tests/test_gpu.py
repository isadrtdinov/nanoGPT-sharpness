import unittest
import torch
from functools import partial
from metrics import (
    hessian_matvec, gn_matvec, top_eigenvalue,
    top_and_bottom_eigenvalue, loss_hessian_top_eigenvalues,
    top_eigenvalue_lobpcg,
)
from tests.toy_models import TwoLayerClassificationModel, SmallGPT, matvec_to_matrix


# ------------------------------------------------------------------ #
# GPU tests — simple models
# ------------------------------------------------------------------ #

@unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
class TestGPU(unittest.TestCase):
    """
    Test that all three eigenvalue computation methods work correctly on GPU.
    Results should match CPU computation up to floating point tolerance.
    """

    def setUp(self):
        torch.manual_seed(42)
        self.device = torch.device('cuda')

        # Classification model on GPU
        self.model_gpu = TwoLayerClassificationModel(in_features=64, hidden_size=128, num_classes=128)
        self.model_gpu = self.model_gpu.to(self.device)
        self.inputs_gpu = torch.randn(32, 64, device=self.device)
        self.targets_gpu = torch.randint(0, 128, (32,), device=self.device)

        # Same model/data on CPU for reference
        torch.manual_seed(42)
        self.model_cpu = TwoLayerClassificationModel(in_features=64, hidden_size=128, num_classes=128)
        self.inputs_cpu = self.inputs_gpu.cpu()
        self.targets_cpu = self.targets_gpu.cpu()
        # Copy GPU model weights to CPU model
        self.model_cpu.load_state_dict(self.model_gpu.state_dict())

        # Logits for loss hessian tests
        self.logits_gpu = torch.randn(16, 5, device=self.device)
        self.logits_cpu = self.logits_gpu.cpu()

        self.k = 32
        self.steps = 1000

    # ------------------------------------------------------------------ #
    # Hessian
    # ------------------------------------------------------------------ #

    def test_hessian_matvec_on_gpu(self):
        """hessian_matvec should run on GPU and match CPU result."""
        params_cpu = list(self.model_cpu.parameters())
        params_gpu = list(self.model_gpu.parameters())

        v_cpu = [torch.randn_like(p) for p in params_cpu]
        v_gpu = [v.to(self.device) for v in v_cpu]

        Hv_cpu = hessian_matvec(self.model_cpu, self.inputs_cpu, self.targets_cpu, v_cpu)
        Hv_gpu = hessian_matvec(self.model_gpu, self.inputs_gpu, self.targets_gpu, v_gpu)

        for hv_cpu, hv_gpu in zip(Hv_cpu, Hv_gpu):
            self.assertTrue(
                torch.allclose(hv_cpu, hv_gpu.cpu(), atol=1e-5),
                f"Max diff: {(hv_cpu - hv_gpu.cpu()).abs().max():.2e}"
            )

    def test_hessian_top_eigenvalue_on_gpu(self):
        """top_eigenvalue with hessian_matvec should run on GPU and match CPU."""
        top_cpu = top_eigenvalue(
            hessian_matvec, self.model_cpu, [(self.inputs_cpu, self.targets_cpu)],
            max_iter=500, rtol=1e-5, cossim_thr=0.9999
        )
        top_gpu = top_eigenvalue(
            hessian_matvec, self.model_gpu, [(self.inputs_gpu, self.targets_gpu)],
            max_iter=500, rtol=1e-5, cossim_thr=0.9999
        )
        self.assertAlmostEqual(
            top_cpu, top_gpu, delta=abs(top_cpu) * 1e-2,
            msg=f"CPU: {top_cpu:.4f}, GPU: {top_gpu:.4f}"
        )

    # ------------------------------------------------------------------ #
    # Gauss-Newton
    # ------------------------------------------------------------------ #

    def test_gn_matvec_on_gpu(self):
        """gn_matvec should run on GPU and match CPU result."""
        params_cpu = list(self.model_cpu.parameters())
        params_gpu = list(self.model_gpu.parameters())

        v_cpu = [torch.randn_like(p) for p in params_cpu]
        v_gpu = [v.to(self.device) for v in v_cpu]

        Gv_cpu = gn_matvec(self.model_cpu, self.inputs_cpu, self.targets_cpu, v_cpu)
        Gv_gpu = gn_matvec(self.model_gpu, self.inputs_gpu, self.targets_gpu, v_gpu)

        for gv_cpu, gv_gpu in zip(Gv_cpu, Gv_gpu):
            self.assertTrue(
                torch.allclose(gv_cpu, gv_gpu.cpu(), atol=1e-5),
                f"Max diff: {(gv_cpu - gv_gpu.cpu()).abs().max():.2e}"
            )

    def test_gn_top_eigenvalue_on_gpu(self):
        """top_eigenvalue with gn_matvec should run on GPU and match CPU."""
        top_cpu = top_eigenvalue(
            gn_matvec, self.model_cpu, [(self.inputs_cpu, self.targets_cpu)],
            max_iter=500, rtol=1e-5, cossim_thr=0.9999
        )
        top_gpu = top_eigenvalue(
            gn_matvec, self.model_gpu, [(self.inputs_gpu, self.targets_gpu)],
            max_iter=500, rtol=1e-5, cossim_thr=0.9999
        )
        self.assertAlmostEqual(
            top_cpu, top_gpu, delta=abs(top_cpu) * 1e-2,
            msg=f"CPU: {top_cpu:.4f}, GPU: {top_gpu:.4f}"
        )

    # ------------------------------------------------------------------ #
    # Loss Hessian (cross-entropy w.r.t. logits)
    # ------------------------------------------------------------------ #

    def test_loss_hessian_matvec_on_gpu(self):
        """Loss hessian matvec should run on GPU and match CPU."""
        probs_cpu = torch.softmax(self.logits_cpu, dim=-1)
        probs_gpu = torch.softmax(self.logits_gpu, dim=-1)
        p_mean_cpu = probs_cpu.mean(dim=0)
        p_mean_gpu = probs_gpu.mean(dim=0)

        v_cpu = torch.randn(self.logits_cpu.shape[1])
        v_gpu = v_cpu.to(self.device)

        def matvec_cpu(v):
            return p_mean_cpu * v - probs_cpu.T @ (probs_cpu @ v) / probs_cpu.shape[0]

        def matvec_gpu(v):
            return p_mean_gpu * v - probs_gpu.T @ (probs_gpu @ v) / probs_gpu.shape[0]

        Hv_cpu = matvec_cpu(v_cpu)
        Hv_gpu = matvec_gpu(v_gpu)

        self.assertTrue(
            torch.allclose(Hv_cpu, Hv_gpu.cpu(), atol=1e-5),
            f"Max diff: {(Hv_cpu - Hv_gpu.cpu()).abs().max():.2e}"
        )

    def test_loss_hessian_top_eigenvalues_on_gpu(self):
        """loss_hessian_top_eigenvalues should run on GPU and match CPU."""
        top_cpu = loss_hessian_top_eigenvalues(self.logits_cpu, k=self.k, num_steps=self.steps)
        top_gpu = loss_hessian_top_eigenvalues(self.logits_gpu, k=self.k, num_steps=self.steps)

        self.assertTrue(
            torch.allclose(top_cpu, top_gpu.cpu(), atol=1e-3),
            f"CPU: {top_cpu}, GPU: {top_gpu.cpu()}"
        )

    def test_hessian_lobpcg_on_gpu(self):
        """top_eigenvalue_lobpcg with hessian_matvec should run on GPU and match CPU."""
        top_cpu = top_eigenvalue_lobpcg(
            hessian_matvec, self.model_cpu, [(self.inputs_cpu, self.targets_cpu)],
            k=1, max_iter=200, tol=1e-4,
        )
        top_gpu = top_eigenvalue_lobpcg(
            hessian_matvec, self.model_gpu, [(self.inputs_gpu, self.targets_gpu)],
            k=1, max_iter=200, tol=1e-4,
        )
        self.assertAlmostEqual(
            top_cpu, top_gpu, delta=abs(top_cpu) * 1e-2,
            msg=f"CPU: {top_cpu:.4f}, GPU: {top_gpu:.4f}"
        )

    def test_results_on_correct_device(self):
        """Outputs should reside on the same device as the inputs."""
        # Hessian matvec
        params_gpu = list(self.model_gpu.parameters())
        v_gpu = [torch.randn_like(p) for p in params_gpu]
        Hv_gpu = hessian_matvec(self.model_gpu, self.inputs_gpu, self.targets_gpu, v_gpu)
        for hv in Hv_gpu:
            self.assertEqual(hv.device.type, 'cuda',
                             f"Expected cuda, got {hv.device}")

        # GN matvec
        Gv_gpu = gn_matvec(self.model_gpu, self.inputs_gpu, self.targets_gpu, v_gpu)
        for gv in Gv_gpu:
            self.assertEqual(gv.device.type, 'cuda',
                             f"Expected cuda, got {gv.device}")

        # Loss hessian eigenvalues
        top_gpu = loss_hessian_top_eigenvalues(self.logits_gpu, k=3, num_steps=100)
        self.assertEqual(top_gpu.device.type, 'cuda',
                         f"Expected cuda, got {top_gpu.device}")


# ------------------------------------------------------------------ #
# GPU tests — SmallGPT integration
# ------------------------------------------------------------------ #

@unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
class TestGPTSequentialGPU(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(42)
        self.device = torch.device('cuda')

        self.N = 4
        self.T = 8
        self.C = 16
        self.embed_dim = 8
        self.num_heads = 2
        self.num_layers = 2
        self.ffn_dim = 8

        self.model = SmallGPT(
            vocab_size=self.C,
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            seq_len=self.T,
            num_layers=self.num_layers,
            ffn_dim=self.ffn_dim
        ).to(self.device)

        self.inputs = torch.randint(0, self.C, (self.N, self.T), device=self.device)
        self.targets = torch.randint(0, self.C, (self.N, self.T), device=self.device)

        # Materialize matrices on GPU once for all tests
        self.G = matvec_to_matrix(gn_matvec, self.model, self.inputs, self.targets)
        self.H = matvec_to_matrix(hessian_matvec, self.model, self.inputs, self.targets)

    # ------------------------------------------------------------------ #
    # Sanity checks
    # ------------------------------------------------------------------ #

    def test_logits_shape(self):
        logits, _ = self.model(self.inputs, self.targets)
        self.assertEqual(logits.shape, (self.N, self.T, self.C))

    def test_num_predictions_is_n_times_t(self):
        logits, _ = self.model(self.inputs, self.targets)
        self.assertEqual(logits[..., 0].numel(), self.N * self.T)

    def test_gn_matvec_output_shape(self):
        params = list(self.model.parameters())
        v = [torch.randn_like(p) for p in params]
        Gv = gn_matvec(self.model, self.inputs, self.targets, v)
        for gv, p in zip(Gv, params):
            self.assertEqual(gv.shape, p.shape)

    def test_hessian_matvec_output_shape(self):
        params = list(self.model.parameters())
        v = [torch.randn_like(p) for p in params]
        Hv = hessian_matvec(self.model, self.inputs, self.targets, v)
        for hv, p in zip(Hv, params):
            self.assertEqual(hv.shape, p.shape)

    def test_output_device_is_gpu(self):
        params = list(self.model.parameters())
        v = [torch.randn_like(p) for p in params]

        for gv in gn_matvec(self.model, self.inputs, self.targets, v):
            self.assertEqual(gv.device.type, 'cuda')

        for hv in hessian_matvec(self.model, self.inputs, self.targets, v):
            self.assertEqual(hv.device.type, 'cuda')

        logits, _ = self.model(self.inputs, self.targets)
        top_eigs = loss_hessian_top_eigenvalues(
            logits.reshape(-1, self.C), k=3, num_steps=50
        )
        self.assertEqual(top_eigs.device.type, 'cuda')

    # ------------------------------------------------------------------ #
    # Matrix structure
    # ------------------------------------------------------------------ #

    def test_gn_is_symmetric(self):
        self.assertTrue(
            torch.allclose(self.G, self.G.T, atol=1e-4),
            f"Max asymmetry: {(self.G - self.G.T).abs().max():.2e}"
        )

    def test_gn_is_psd(self):
        min_eigenvalue = torch.linalg.eigvalsh(self.G).min().item()
        self.assertGreaterEqual(min_eigenvalue, -1e-4,
                                f"Most negative eigenvalue: {min_eigenvalue:.2e}")

    def test_hessian_is_symmetric(self):
        self.assertTrue(
            torch.allclose(self.H, self.H.T, atol=1e-4),
            f"Max asymmetry: {(self.H - self.H.T).abs().max():.2e}"
        )

    # ------------------------------------------------------------------ #
    # Matvec correctness
    # ------------------------------------------------------------------ #

    def test_gn_matvec_matches_materialized(self):
        params = list(self.model.parameters())
        v = [torch.randn_like(p) for p in params]
        v_flat = torch.cat([vi.flatten() for vi in v])

        Gv = gn_matvec(self.model, self.inputs, self.targets, v)
        Gv_flat = torch.cat([gv.flatten() for gv in Gv])

        self.assertTrue(
            torch.allclose(Gv_flat, self.G @ v_flat, atol=1e-4),
            f"Max diff: {(Gv_flat - self.G @ v_flat).abs().max():.2e}"
        )

    def test_hessian_matvec_matches_materialized(self):
        params = list(self.model.parameters())
        v = [torch.randn_like(p) for p in params]
        v_flat = torch.cat([vi.flatten() for vi in v])

        Hv = hessian_matvec(self.model, self.inputs, self.targets, v)
        Hv_flat = torch.cat([hv.flatten() for hv in Hv])

        self.assertTrue(
            torch.allclose(Hv_flat, self.H @ v_flat, atol=1e-4),
            f"Max diff: {(Hv_flat - self.H @ v_flat).abs().max():.2e}"
        )

    # ------------------------------------------------------------------ #
    # Eigenvalue correctness
    # ------------------------------------------------------------------ #

    def test_gn_top_eigenvalue_matches_materialized(self):
        exact_top = torch.linalg.eigvalsh(self.G).max().item()

        estimated = top_eigenvalue(
            gn_matvec, self.model, [(self.inputs, self.targets)],
            max_iter=500, rtol=1e-5, cossim_thr=0.9999
        )
        self.assertAlmostEqual(
            estimated, exact_top, delta=abs(exact_top) * 1e-2,
            msg=f"Estimated: {estimated:.4f}, Exact: {exact_top:.4f}"
        )

    def test_top_and_bottom_eigenvalue_matches_materialized(self):
        """
        top_and_bottom_eigenvalue should recover both the largest and smallest
        eigenvalues of the Hessian, matching eigvalsh on the materialized matrix.
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

    # ------------------------------------------------------------------ #
    # Normalization
    # ------------------------------------------------------------------ #

    def test_gn_matvec_normalization(self):
        """Doubling batch size should give the same GN matvec."""
        params = list(self.model.parameters())
        v = [torch.randn_like(p) for p in params]

        Gv1 = gn_matvec(self.model, self.inputs, self.targets, v)

        inputs_double = self.inputs.repeat(2, 1)
        targets_double = self.targets.repeat(2, 1)
        Gv2 = gn_matvec(self.model, inputs_double, targets_double, v)

        for gv1, gv2 in zip(Gv1, Gv2):
            self.assertTrue(
                torch.allclose(gv1, gv2, atol=1e-4),
                f"Max diff: {(gv1 - gv2).abs().max():.2e}"
            )

    # ------------------------------------------------------------------ #
    # Loss hessian
    # ------------------------------------------------------------------ #

    def test_loss_hessian_top_eigenvalues(self):
        logits, _ = self.model(self.inputs, self.targets)
        logits_flat = logits.reshape(-1, self.C).detach()

        k = 3
        top_eigs = loss_hessian_top_eigenvalues(logits_flat, k=k, num_steps=50)

        self.assertEqual(top_eigs.shape, (k,))
        self.assertTrue(
            (top_eigs >= -1e-6).all(),
            f"Loss hessian eigenvalues should be non-negative, got {top_eigs}"
        )
        self.assertTrue(
            (top_eigs[:-1] >= top_eigs[1:] - 1e-6).all(),
            f"Eigenvalues should be in descending order, got {top_eigs}"
        )

    # ------------------------------------------------------------------ #
    # LOBPCG eigenvalues
    # ------------------------------------------------------------------ #

    def test_lobpcg_hessian_top_eigenvalues(self):
        """top_eigenvalue_lobpcg with hessian_matvec matches materialized Hessian."""
        k = 5
        exact_top_k = torch.linalg.eigvalsh(self.H).flip(0)[:k].tolist()

        eigs = top_eigenvalue_lobpcg(
            hessian_matvec, self.model, [(self.inputs, self.targets)],
            k=k, max_iter=200, tol=1e-4,
        )
        self.assertEqual(len(eigs), k)
        for j in range(k):
            self.assertAlmostEqual(
                eigs[j], exact_top_k[j], delta=abs(exact_top_k[j]) * 1e-2,
                msg=f"j={j}: LOBPCG={eigs[j]:.4f}, exact={exact_top_k[j]:.4f}"
            )

    def test_lobpcg_gn_top_eigenvalues(self):
        """top_eigenvalue_lobpcg with gn_matvec matches materialized GN matrix."""
        k = 5
        exact_top_k = torch.linalg.eigvalsh(self.G).flip(0)[:k].tolist()

        eigs = top_eigenvalue_lobpcg(
            gn_matvec, self.model, [(self.inputs, self.targets)],
            k=k, max_iter=200, tol=1e-4,
        )
        self.assertEqual(len(eigs), k)
        for j in range(k):
            self.assertAlmostEqual(
                eigs[j], exact_top_k[j], delta=abs(exact_top_k[j]) * 1e-2,
                msg=f"j={j}: LOBPCG={eigs[j]:.4f}, exact={exact_top_k[j]:.4f}"
            )

    def test_lobpcg_hessian_bottom_eigenvalues(self):
        """top_eigenvalue_lobpcg with mode='bottom' matches smallest Hessian eigenvalues."""
        k = 5
        # eigvalsh returns ascending order; take the k smallest
        exact_bottom_k = torch.linalg.eigvalsh(self.H)[:k].tolist()

        eigs = top_eigenvalue_lobpcg(
            hessian_matvec, self.model, [(self.inputs, self.targets)],
            k=k, max_iter=200, tol=1e-4, mode='bottom',
        )
        self.assertEqual(len(eigs), k)
        for j in range(k):
            self.assertAlmostEqual(
                eigs[j], exact_bottom_k[j], delta=max(abs(exact_bottom_k[j]) * 1e-2, 1e-4),
                msg=f"j={j}: LOBPCG={eigs[j]:.4f}, exact={exact_bottom_k[j]:.4f}"
            )

    def test_lobpcg_bottom_eigenvalues_ascending_order(self):
        """mode='bottom' returns eigenvalues in ascending order."""
        k = 5
        eigs = top_eigenvalue_lobpcg(
            hessian_matvec, self.model, [(self.inputs, self.targets)],
            k=k, max_iter=200, tol=1e-4, mode='bottom',
        )
        for j in range(k - 1):
            self.assertLessEqual(
                eigs[j], eigs[j + 1],
                msg=f"eigs[{j}]={eigs[j]:.4f} > eigs[{j+1}]={eigs[j+1]:.4f}"
            )

    def test_loss_hessian_top_eigenvalues_match_exact(self):
        """Loss hessian top eigenvalues should match explicitly materialized H_loss."""
        logits, _ = self.model(self.inputs, self.targets)
        logits_flat = logits.reshape(-1, self.C).detach()
        probs = torch.softmax(logits_flat, dim=-1)      # [N*T, C]

        # Materialize E[diag(p) - pp^T] on GPU
        p_mean = probs.mean(dim=0)                      # [C]
        H_loss = torch.diag(p_mean) - probs.T @ probs / probs.shape[0]  # [C, C]
        exact_top_k = torch.linalg.eigvalsh(H_loss).flip(0)[:3]

        estimated = loss_hessian_top_eigenvalues(logits_flat, k=3, num_steps=100)

        self.assertTrue(
            torch.allclose(estimated, exact_top_k, atol=1e-3),
            f"Estimated: {estimated}, Exact: {exact_top_k}"
        )


if __name__ == '__main__':
    unittest.main()
