from torch import nn


class Predictor(nn.Module):
    def __init__(self, dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, hidden_dim, bias=False), nn.BatchNorm1d(hidden_dim), nn.ReLU(True),
                                 nn.Linear(hidden_dim, dim))

    def forward(self, x):
        return self.net(x)
