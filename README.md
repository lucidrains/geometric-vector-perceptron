## Geometric Vector Perceptron (wip)

Implementation of <a href="https://openreview.net/forum?id=1YLJDvSx6J4">Geometric Vector Perceptron</a>, a simple circuit for 3d rotation equivariance for learning over large biomolecules, in Pytorch. The repository may also contain experimentation to see if this could be easily extended to self-attention.

## Install

```bash
$ pip install geometric-vector-perceptron
```

## Usage

```python
import torch
from geometric_vector_perceptron import GVP
from pytorch3d.transforms import random_rotation

model = GVP(
    dim_v = 1024,
    dim_n = 512,
    dim_m = 256,
    dim_u = 512
)

feats = torch.randn(1, 512)
coors = torch.randn(1, 1024, 3)

feats_out, coors_out = model(feats, coors)
# (1, 256), (1, 512, 3)
```

## Citations

```bibtex
@inproceedings{
    anonymous2021learning,
    title={Learning from Protein Structure with Geometric Vector Perceptrons},
    author={Anonymous},
    booktitle={Submitted to International Conference on Learning Representations},
    year={2021},
    url={https://openreview.net/forum?id=1YLJDvSx6J4},
    note={under review}
}
```
