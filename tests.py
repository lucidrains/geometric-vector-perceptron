import torch
from geometric_vector_perceptron import GVP

TOL = 1e-2

def random_rotation():
    q, r = torch.qr(torch.randn(3, 3))
    return q

def test_equivariance():
    R = random_rotation()

    model = GVP(
        dim_coors_in = 1024,
        dim_feats_in = 512,
        dim_coors_out = 256,
        dim_feats_out = 512
    )

    feats = torch.randn(1, 512)
    coors = torch.randn(1, 1024, 3)

    feats_out, coors_out = model(feats, coors)
    feats_out_r, coors_out_r = model(feats, coors @ R)

    err = ((coors_out @ R) - coors_out_r).max()
    assert err < TOL, 'equivariance must be respected'
