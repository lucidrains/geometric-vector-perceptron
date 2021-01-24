import torch
from torch import nn, einsum

class GVP(nn.Module):
    def __init__(
        self,
        *,
        dim_vectors_in,
        dim_vectors_out,
        dim_feats_in,
        dim_feats_out,
        feats_activation = nn.Sigmoid(),
        vectors_activation = nn.Sigmoid()
    ):
        super().__init__()
        self.dim_vectors_in = dim_vectors_in
        self.dim_feats_in = dim_feats_in

        self.dim_vectors_out = dim_vectors_out
        dim_h = max(dim_vectors_in, dim_vectors_out)

        self.Wh = nn.Parameter(torch.randn(dim_vectors_in, dim_h))
        self.Wu = nn.Parameter(torch.randn(dim_h, dim_vectors_out))

        self.vectors_activation = vectors_activation

        self.to_feats_out = nn.Sequential(
            nn.Linear(dim_h + dim_feats_in, dim_feats_out),
            feats_activation
        )

    def forward(self, feats, vectors):
        b, n, _, v, c = *feats.shape, *vectors.shape

        assert c == 3 and v == self.dim_vectors_in, 'vectors have wrong dimensions'
        assert n == self.dim_feats_in, 'scalar features have wrong dimensions'

        Vh = einsum('b v c, v h -> b h c', vectors, self.Wh)
        Vu = einsum('b h c, h u -> b u c', Vh, self.Wu)

        sh = torch.norm(Vh, p = 2, dim = -1)
        vu = torch.norm(Vu, p = 2, dim = -1, keepdim = True)

        s = torch.cat((feats, sh), dim = 1)

        feats_out = self.to_feats_out(s)
        vectors_out = self.vectors_activation(vu) * Vu

        return feats_out, vectors_out


class GVPDropout(nn.Module):
    """ Separate dropout for scalars and vectors. """
    def __init__(self, rate):
        super(GVPDropout, self).__init__()
        self.vdropout = nn.Dropout2d(p = rate)
        self.fdropout = nn.Dropout(p = rate)

    def forward(self, feats, vectors, training=None):
        if not training: 
            return x
        return self.fdropout(feats), self.vdropout(vectors)


class GVPLayerNorm(nn.Module):
    """ Normal layer norm for scalars, nontrainable norm for vectors. """
    def __init__(self, feats_h_size):
        super(GVPLayerNorm, self).__init__()
        self.fnorm = nn.LayerNorm(feats_h_size)

    def forward(self, feats, vectors):
        vnorm = torch.linalg.norm(vectors, dim=(-1,-2), keepdim=True)
        return self.fnorm(feats), vectors/vnorm






