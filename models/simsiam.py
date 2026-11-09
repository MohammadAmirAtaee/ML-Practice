import torch
from torch import nn
from .encoder import ConvEncoder
from .projector import MLP
from .predictor import Predictor


def negative_cosine(p: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    z = z.detach()
    return -(nn.functional.normalize(p, dim=1) * nn.functional.normalize(z, dim=1)).sum(dim=1).mean()


class SimSiam(nn.Module):
    def __init__(self, feature_dim=512, projection_dim=256, predictor_dim=128):
        super().__init__()
        self.encoder = ConvEncoder(feature_dim)
        self.projector = MLP(feature_dim, projection_dim, projection_dim)
        self.predictor = Predictor(projection_dim, predictor_dim)

    def encode(self, x):
        return self.encoder(x)

    def forward(self, x1, x2):
        z1, z2 = self.projector(self.encoder(x1)), self.projector(self.encoder(x2))
        return negative_cosine(self.predictor(z1), z2) / 2 + negative_cosine(self.predictor(z2), z1) / 2
