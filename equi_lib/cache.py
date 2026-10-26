"""Caches standard PGD deltas per (model, seed, image index), so every defense variant
is evaluated on the same attacked images without regenerating them.
Never cache adaptive attacks: those depend on the defense."""
import os

import torch

from . import RESULTS_DIR

CACHE_DIR = os.path.join(RESULTS_DIR, "cache")


class PGDCache:
    def __init__(self, model_tag: str, seed: int):
        os.makedirs(CACHE_DIR, exist_ok=True)
        self.path = os.path.join(CACHE_DIR, f"pgd50_{model_tag}_s{seed}.pt")
        self.store = torch.load(self.path) if os.path.exists(self.path) else {}
        self.dirty = False

    def has(self, indices) -> bool:
        return all(i in self.store for i in indices)

    def get(self, indices, device):
        return torch.stack([self.store[i] for i in indices]).to(device).float()

    def put(self, indices, deltas):
        for i, d in zip(indices, deltas):
            self.store[i] = d.detach().to("cpu", torch.float16)
        self.dirty = True

    def save(self):
        if self.dirty:
            torch.save(self.store, self.path)
            self.dirty = False
