import torch
import torch.nn.functional as F
from torch import nn, einsum

class GVP(nn.Module):
    def __init__(
        self,
        *,
        dim_v,
        dim_n,
        dim_m,
        dim_u,
        σ = nn.Sigmoid(),
        σ_plus = nn.Sigmoid()
    ):
        super().__init__()
        self.dim_v = dim_v
        self.dim_n = dim_n

        self.dim_u = dim_u
        dim_h = max(dim_v, dim_u)

        self.Wh = nn.Parameter(torch.randn(dim_v, dim_h))
        self.Wu = nn.Parameter(torch.randn(dim_h, dim_u))

        self.Wm = nn.Parameter(torch.randn(dim_h + dim_n, dim_m))
        self.Bm = nn.Parameter(torch.randn(1, dim_m))

        self.σ = σ
        self.σ_plus = σ_plus

    def forward(self, feats, coors):
        b, n, _, v, c = *feats.shape, *coors.shape

        assert c == 3 and v == self.dim_v, 'coordinates have wrong dimensions'
        assert n == self.dim_n, 'scalar features have wrong dimensions'

        Vh = einsum('b v c, v h -> b h c', coors, self.Wh)
        Vu = einsum('b h c, h u -> b u c', Vh, self.Wu)

        sh = torch.norm(Vh, p = 2, dim = -1)
        vu = torch.norm(Vu, p = 2, dim = -1, keepdim = True)

        s = torch.cat((feats, sh), dim = 1)
        sm = einsum('b h, h m -> b m', s, self.Wm) + self.Bm

        feats_out = self.σ(sm)
        coors_out = self.σ_plus(vu) * Vu

        return feats_out, coors_out
