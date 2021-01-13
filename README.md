## Geometric Vector Perceptron (wip)

Implementation of <a href="https://openreview.net/forum?id=1YLJDvSx6J4">Geometric Vector Perceptron</a>, a simple circuit with 3d rotation equivariance for learning over large biomolecules, in Pytorch. The repository may also contain experimentation to see if this could be easily extended to self-attention.

## Install

```bash
$ pip install geometric-vector-perceptron
```

## Usage

```python
import torch
from geometric_vector_perceptron import GVP

model = GVP(
    dim_coors_in = 1024,
    dim_feats_in = 512,
    dim_coors_out = 256,
    dim_feats_out = 512
)

feats, coors = (torch.randn(1, 512), torch.randn(1, 1024, 3))

feats_out, coors_out = model(feats, coors) # (1, 256), (1, 512, 3)
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
