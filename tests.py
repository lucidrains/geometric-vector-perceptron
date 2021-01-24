import torch
from geometric_vector_perceptron import GVP, GVPDropout, GVPLayerNorm

TOL = 1e-2

def random_rotation():
    q, r = torch.qr(torch.randn(3, 3))
    return q

def diff_matrix(vectors):
    b, _, d = vectors.shape
    diff = vectors[..., None, :] - vectors[:, None, ...]
    return diff.reshape(b, -1, d)

def test_equivariance():
    R = random_rotation()

    model = GVP(
        dim_vectors_in = 1024,
        dim_feats_in = 512,
        dim_vectors_out = 256,
        dim_feats_out = 512
    )

    feats = torch.randn(1, 512)
    vectors = torch.randn(1, 32, 3)

    feats_out, vectors_out = model(feats, diff_matrix(vectors))
    feats_out_r, vectors_out_r = model(feats, diff_matrix(vectors @ R))

    err = ((vectors_out @ R) - vectors_out_r).max()
    assert err < TOL, 'equivariance must be respected'

def test_all_layer_types():
    R = random_rotation()

    model = GVP(
        dim_vectors_in = 1024,
        dim_feats_in = 512,
        dim_vectors_out = 256,
        dim_feats_out = 512
    )
    dropout = GVPDropout(0.2)
    layer_norm = GVPLayerNorm(512)

    feats = torch.randn(1, 512)
    message = torch.randn(1, 512)
    vectors = torch.randn(1, 32, 3)


    # GVP layer
    feats_out, vectors_out = model( (feats, diff_matrix(vectors)) )
    assert list(feats_out.shape) == [1, 512] and list(vectors_out.shape) == [1, 256, 3]

    # GVP Dropout
    feats_out, vectors_out = dropout(feats_out, vectors_out)
    assert list(feats_out.shape) == [1, 512] and list(vectors_out.shape) == [1, 256, 3]

    # GVP Layer Norm
    feats_out, vectors_out = layer_norm(feats_out, vectors_out)
    assert list(feats_out.shape) == [1, 512] and list(vectors_out.shape) == [1, 256, 3]


if __name__ == "__main__":
    test_equivariance()
    test_all_layer_types()

