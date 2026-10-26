import torch
import torch.nn as nn
from torchvision.transforms import transforms

from learning.attack_new_single import Equi_Set_Defense_good_inloop


class OriginalDefense:
    """Upstream equivariance defense, called exactly as in evaluate_equi_defense.py (step 1.0, 20 steps)."""

    def __init__(self, model, normalize, eps_v, num_views, steps=20, device="cuda"):
        self.model, self.normalize = model, normalize
        self.eps_v, self.num_views, self.steps = eps_v, num_views, steps
        jitter = nn.Sequential(transforms.ColorJitter(0.8, 0.8, 0.8, 0.2), transforms.RandomGrayscale(p=0.2))
        self.scripted = torch.jit.script(jitter).to(device)
        self.criterion = nn.NLLLoss(ignore_index=255)

    def __call__(self, x, y):
        delta = Equi_Set_Defense_good_inloop(
            x=x, y=y, net_list=[self.model] * self.num_views, Loss=self.criterion,
            epsilon=self.eps_v, steps=self.steps, dataset="cityscape", step_size=1.0, info=None,
            using_noise=True, SGD_wN=True, attack_type="equi", innormalize=self.normalize,
            norm="l_inf", scripted_transforms=self.scripted, transform_delta=True)
        return torch.clamp(x + delta, 0.0, 1.0)


DEFENSES = {"original": OriginalDefense}
