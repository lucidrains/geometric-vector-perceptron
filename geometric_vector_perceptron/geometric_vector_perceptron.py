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
        super().__init__()
        self.vector_dropout = nn.Dropout2d(rate)
        self.feat_dropout = nn.Dropout(rate)

    def forward(self, feats, vectors):
        return self.feat_dropout(feats), self.vector_dropout(vectors)

class GVPLayerNorm(nn.Module):
    """ Normal layer norm for scalars, nontrainable norm for vectors. """
    def __init__(self, feats_h_size, eps = 1e-8):
        super().__init__()
        self.eps = eps
        self.feat_norm = nn.LayerNorm(feats_h_size)

    def forward(self, feats, vectors):
        vector_norm = vectors.norm(dim=(-1,-2), keepdim=True)
        normed_feats = self.feat_norm(feats)
        normed_vectors = vectors / (vector_norm + self.eps)
        return normed_feats, normed_vectors
