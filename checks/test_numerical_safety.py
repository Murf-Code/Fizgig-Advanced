"""CPU regression tests for the shared trainer's numerical guards."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import torch
from fizgig.training.numerical_safety import NumericalError, checked_clip_grad_norm, require_finite, require_finite_tensors, require_finite_score


class SafetyTests(unittest.TestCase):
    def parameter(self, values):
        p = torch.nn.Parameter(torch.zeros(len(values)))
        p.grad = torch.tensor(values, dtype=p.dtype)
        return p

    def test_normal_clip_identical(self):
        p, q = self.parameter([3., 4.]), self.parameter([3., 4.])
        a = checked_clip_grad_norm([p], 1., 'step')
        b = torch.nn.utils.clip_grad_norm_([q], 1.)
        torch.testing.assert_close(a, b)
        torch.testing.assert_close(p.grad, q.grad)

    def test_finite_overflow(self):
        p = self.parameter([1e20, -1e20])
        n = checked_clip_grad_norm([p], 1., 'step')
        self.assertTrue(torch.isfinite(n))
        torch.testing.assert_close(p.grad.norm(), torch.tensor(1.))

    def test_nonfinite_gradient(self):
        for invalid in (float('nan'), float('inf')):
            for clip in (0., 1.):
                p = self.parameter([1., invalid])
                with self.assertRaises(NumericalError):
                    checked_clip_grad_norm([p], clip, 'step')
                self.assertEqual(float(p.grad[0]), 1.)

    def test_no_clip(self):
        p = self.parameter([3., 4.])
        self.assertIsNone(checked_clip_grad_norm([p], 0., 'step'))
        torch.testing.assert_close(p.grad, torch.tensor([3., 4.]))

    def test_loss_and_score(self):
        require_finite(torch.tensor(1.), 'loss')
        require_finite_score(1., 'validation')
        with self.assertRaises(NumericalError):
            require_finite(torch.tensor(float('nan')), 'loss')
        with self.assertRaises(NumericalError):
            require_finite_score(float('inf'), 'validation')

    def test_named_tensor(self):
        require_finite_tensors([('mask', torch.ones(2, dtype=torch.bool)), ('weight', torch.ones(2))], 'cache')
        with self.assertRaisesRegex(NumericalError, 'embedding'):
            require_finite_tensors([('embedding', torch.tensor([float('nan')]))], 'cache')

    def test_checkpoint_preserved(self):
        import tempfile
        from pathlib import Path
        from safetensors.torch import load_file
        from fizgig.training.numerical_safety import save_finite_safetensors
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "model.safetensors"
            save_finite_safetensors({"w": torch.ones(2)}, p)
            before = p.read_bytes()
            with self.assertRaises(NumericalError):
                save_finite_safetensors({"w": torch.tensor([float("nan")])}, p)
            self.assertEqual(p.read_bytes(), before)
            torch.testing.assert_close(load_file(p)["w"], torch.ones(2))
            self.assertEqual(list(Path(d).iterdir()), [p])

    def test_nested_cache(self):
        from fizgig.training.numerical_safety import tensor_leaves
        with self.assertRaisesRegex(NumericalError, "caption/0"):
            require_finite_tensors(tensor_leaves({"caption": [torch.tensor(float("inf"))]}), "cache")

    def test_fp8_cache(self):
        if not hasattr(torch, "float8_e4m3fn"):
            self.skipTest("float8 unavailable")
        require_finite_tensors([("fp8", torch.ones(2).to(torch.float8_e4m3fn))], "cache")
        with self.assertRaises(NumericalError):
            require_finite_tensors([("fp8", torch.tensor([float("nan")]).to(torch.float8_e4m3fn))], "cache")

    def test_no_grad(self):
        self.assertEqual(float(checked_clip_grad_norm([torch.nn.Parameter(torch.zeros(1))], 1., 'step')), 0.)


if __name__ == '__main__':
    unittest.main()
