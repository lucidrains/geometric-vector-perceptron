import torch
from torch import nn, einsum
from torch_geometric import MessagePassing
# types
from typing import Optional, List, Union
from torch_geometric.typing import OptPairTensor, Adj, Size, OptTensor

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

    def forward(self, data):
        feats, vectors = data
        b, n, _, v, c  = *feats.shape, *vectors.shape

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


class GVP_MPNN(MessagePassing):
    r"""The Geometric Vector Perceptron message passing layer
        introduced in https://openreview.net/forum?id=1YLJDvSx6J4.
        
        Uses a Geometric Vector Perceptron instead of the normal 
        MLP in aggregation phase.

        Args:
        * dim_vectors_in: int. number of dimensions in the vector inputs.
        * dim_vectors_in: int. number of dimensions in the vector hidden state.
        * dim_feats_in: int. number of dimensions in the feature inputs.
        * dim_feats_in: int. number of dimensions in the feature hidden state.
        * dropout: float. dropout rate.
        * verbose: bool. verbosity level.
    """
    def __init__(self, dim_vectors_in, dim_vectors_h,
                       dim_feats_out, dim_feats_h,
                       dropout, verbose=False, **kwargs):
        super(MessagePassing, self).__init__(aggr="mean",**kwargs)
        self.verbose = verbose
        # layer params
        self.dim_vectors_h = dim_vectors_h
        self.dim_vectors_in = dim_vectors_in
        self.dim_feats_h = dim_feats_h
        self.dim_feats_in = dim_feats_in
        self.vo, self.fo = vo, fo = dim_vectors_h, dim_feats_h
        self.norm = [GVPLayerNorm(vo) for _ in range(2)]
        self.dropout = GVPDropout(dropout)
        # this receives the vec_in message AND the receiver node
        self.W_EV = Sequential([GVP(dim_vectors_in=dim_vectors_in, dim_vectors_out=vo, dim_feats_in=dim_feats_in, dim_feats_out=fo), 
                                GVP(dim_vectors_in=vo, dim_vectors_out=vo, dim_feats_in=fo, dim_feats_out=fo),
                                GVP(dim_vectors_in=vo, dim_vectors_out=vo, dim_feats_in=fo, dim_feats_out=fo)])
        
        self.W_dh = Sequential([GVP(dim_vectors_in=vo, dim_vectors_out=2*vo, dim_feats_in=fo, dim_feats_out=4*fo),
                                GVP(dim_vectors_in=2*vo, dim_vectors_out=vo, dim_feats_in=4*fo, dim_feats_out=fo)])


    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj,
                edge_attr: OptTensor = None, size: Size = None) -> Tensor:
        """"""
        if isinstance(edge_index, SparseTensor):
            edge_attr = edge_index.storage.value()
            if edge_attr is not None:
                assert x[0].size(-1) == edge_attr.size(-1)

        # TODO: customize aggr so that sums are performed both in nodes and edges
        # return x, modified x, edge_attrs
        out_nodes, out_edges = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        # aggregate
        aggr_nodes, aggr_edges = self.dropout(out_nodes, out_edges)
        out_nodes, out_edges = self.norm[0]( x + aggr_nodes, edge_attr + aggr_edges )
        # update position-wise feedforward
        pw_nodes, pw_edges = self.dropout( self.W_dh(out) )
        out_nodes, out_edges = self.norm[1]( x + pw_nodes, edge_attr + pw_edges )

        return (out_nodes, out_edges)


    def message(self, x_j, edge_attr) -> Tensor:
        msg, edges = self.W_EV( (x_j, edge_attr) )
        return msg, edges


    def propagate(self, edge_index: Adj, size: Size = None, **kwargs):
        r"""The initial call to start propagating messages.
        Args:
            adj (Tensor or SparseTensor): `edge_index` holds the indices of a general (sparse)
                assignment matrix of shape :obj:`[N, M]`.
            size (tuple, optional): If set to :obj:`None`, the size will be automatically inferred
                and assumed to be quadratic.
                This argument is ignored in case :obj:`edge_index` is a
                :obj:`torch_sparse.SparseTensor`. (default: :obj:`None`)
            **kwargs: Any additional data which is needed to construct and
                aggregate messages, and to update node embeddings.
        """
        msg, edges = self.message(**msg_kwargs)
        # aggregate them
        aggr_kwargs = self.inspector.distribute('aggregate', coll_dict)
        out_msg     = self.aggregate(msg, **aggr_kwargs)
        out_edges   = self.aggregate(edges, **aggr_kwargs)
        
        return out_msg, out_edges

        
    def __repr__(self):
        return 'GVP_MPNN Layer with the following attributes: ' + \
                'dim_vectors_in={0}, dim_vectors_h={1}, dropout={2}'.format(self.dim_vectors_in,
                                                                            self.dim_vectors_h,
                                                                            self.dropout)

