import torch
import torch.nn as nn

from .metrics import align_output

# Standard PGD-50 setting (REPRODUCTION_REFERENCE.md section 1). The cache validates against these.
PGD_PARAMS = {"eps": 4.0 / 255.0, "step": 1.0 / 255.0, "steps": 50}


def pgd(model, normalize, x, y, eps=PGD_PARAMS["eps"], step=PGD_PARAMS["step"], steps=PGD_PARAMS["steps"]):
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


def segpgd(model, normalize, x, y, eps=PGD_PARAMS["eps"], step=PGD_PARAMS["step"], steps=PGD_PARAMS["steps"]):
    """SegPGD (Gu et al., ECCV 2022). Same L_inf update as pgd(), but the loss splits valid pixels
    into currently correct (P_T) and wrong (P_F) ones at every iteration t = 1..T:
        L = (1 - lam_t) / N * sum_{P_T} CE + lam_t / N * sum_{P_F} CE,   lam_t = (t - 1) / (2T),
    with N the number of valid (non-ignore) pixels. Early steps push correct pixels; later steps
    keep wrong ones wrong."""
    x = x.detach()
    valid = y != 255
    y_safe = torch.where(valid, y, torch.zeros_like(y))
    n_valid = valid.sum().clamp_min(1)
    delta = torch.empty_like(x).uniform_(-eps, eps)
    delta = torch.clamp(delta, -x, 1.0 - x)
    for t in range(1, steps + 1):
        lam = (t - 1) / (2 * steps)
        delta = delta.detach().requires_grad_(True)
        logp = align_output(model(normalize(x + delta))[0], y.shape[-2:])   # log-softmax output
        ce = -logp.gather(1, y_safe.unsqueeze(1)).squeeze(1)                  # per-pixel CE
        correct = (logp.argmax(1) == y) & valid
        wrong = (~correct) & valid
        loss = ((1 - lam) * (ce * correct).sum() + lam * (ce * wrong).sum()) / n_valid
        model.zero_grad()
        loss.backward()
        delta = delta.detach() + step * delta.grad.detach().sign()
        delta = torch.clamp(delta, -eps, eps)
        delta = torch.clamp(delta, -x, 1.0 - x)
    return delta.detach()
