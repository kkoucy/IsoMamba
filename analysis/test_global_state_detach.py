"""Check the global-context gradient boundary without a selective-scan kernel."""

import unittest

import torch

from basicsr.models.archs.clusterM_arch import SS2D_dh0


class GlobalStateDetachTests(unittest.TestCase):
    def test_context_is_detached_but_film_and_features_train(self):
        torch.manual_seed(19)
        expert = SS2D_dh0(d_model=4, d_state=3, expand=2)
        # Exercise non-identity modulation, not just the zero-initialized FiLM.
        torch.nn.init.normal_(expert.FiLM[-1].weight, std=0.2)
        torch.nn.init.normal_(expert.FiLM[-1].bias, std=0.2)
        features = torch.randn(1, 3, 2, 8, requires_grad=True)
        context = torch.randn(8, 3, requires_grad=True)
        reference = expert.FiLM(context)
        expected = features * (1 + reference[:, 0].view(1, 1, 1, 8))
        expected = expected + reference[:, 1].view(1, 1, 1, 8)
        actual = expert.fusion_lastState(features, context)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        actual.square().mean().backward()
        self.assertIsNone(context.grad)
        self.assertGreater(features.grad.abs().sum().item(), 0)
        for parameter in expert.FiLM.parameters():
            self.assertIsNotNone(parameter.grad)
            self.assertGreater(parameter.grad.abs().sum().item(), 0)


if __name__ == '__main__':
    unittest.main()
