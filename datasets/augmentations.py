import random
import torch


def augment(x: torch.Tensor) -> torch.Tensor:
    """Tensor-only CIFAR augmentation, avoiding torchvision."""
    if random.random() < 0.5:
        x = torch.flip(x, [-1])
    if random.random() < 0.8:
        scale = 0.8 + 0.4 * random.random()
        x = torch.clamp(x * scale, 0, 1)
    if random.random() < 0.2:
        x = x + torch.randn_like(x) * 0.03
    return x.clamp(0, 1)
