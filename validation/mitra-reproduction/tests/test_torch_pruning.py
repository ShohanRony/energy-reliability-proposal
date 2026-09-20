"""These checks are explicitly skipped if PyTorch is absent."""
import importlib.util
import unittest


@unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch unavailable; pruning backend UNTESTED")
class PruningTest(unittest.TestCase):
    def test_mask_survives_optimizer_updates(self):
        import torch
        from torch import nn
        from torch.nn.utils import prune
        layer = nn.Conv2d(1, 1, 2, bias=False)
        with torch.no_grad(): layer.weight.copy_(torch.tensor([[[[1., 2.], [3., 4.]]]]))
        prune.global_unstructured([(layer, "weight")], pruning_method=prune.L1Unstructured, amount=0.5)
        self.assertEqual(layer.weight_mask.flatten().tolist(), [0., 0., 1., 1.])
        opt = torch.optim.SGD(layer.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
        for _ in range(3):
            opt.zero_grad(); layer(torch.ones(1, 1, 3, 3)).sum().backward(); opt.step()
        layer(torch.ones(1, 1, 3, 3))
        self.assertEqual(layer.weight.flatten()[:2].tolist(), [0., 0.])


if __name__ == "__main__": unittest.main()
