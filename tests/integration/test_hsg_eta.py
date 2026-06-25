import numpy as np
import torch

from reference_eta.models.hsg_eta import HSGETA, HSGConfig


def test_hsg_eta_output_contract() -> None:
    model = HSGETA(HSGConfig(hidden_dim=16, max_tasks=6))
    context = torch.randn(4, 19)
    tasks = torch.randn(4, 6, 8)
    mask = torch.ones(4, 6, dtype=torch.bool)
    output = model(context, tasks, mask)
    quantiles = output["quantiles"].detach().numpy()
    assert quantiles.shape == (4, 3)
    assert np.all(quantiles[:, 0] <= quantiles[:, 1])
    assert np.all(quantiles[:, 1] <= quantiles[:, 2])
    assert output["route_logits"].shape == (4, 6)
