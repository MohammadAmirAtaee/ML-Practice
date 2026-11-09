from torch import nn


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hidden_dim, bias=False), nn.BatchNorm1d(hidden_dim), nn.ReLU(True),
                                 nn.Linear(hidden_dim, out_dim, bias=False), nn.BatchNorm1d(out_dim, affine=False))

    def forward(self, x):
        return self.net(x)
