import torch
import torch.nn as nn

from .metrics import align_output


def pgd(model, normalize, x, y, eps=4.0 / 255.0, step=1.0 / 255.0, steps=50):
    """Untargeted L_inf PGD with random start; same logic as evaluate_equi_defense.generate_pgd_perturbation."""
    criterion = nn.NLLLoss(ignore_index=255)
    x = x.detach()
    delta = torch.empty_like(x).uniform_(-eps, eps)
    delta = torch.clamp(delta, -x, 1.0 - x)
    for _ in range(steps):
        delta = delta.detach().requires_grad_(True)
        out = align_output(model(normalize(x + delta))[0], y.shape[-2:])
        loss = criterion(out, y)
        model.zero_grad()
        loss.backward()
        delta = delta.detach() + step * delta.grad.detach().sign()
        delta = torch.clamp(delta, -eps, eps)
        delta = torch.clamp(delta, -x, 1.0 - x)
    return delta.detach()
