import torch
from torch import nn, einsum
from torch_geometric.nn import MessagePassing
# types
from typing import Optional, List, Union
from torch_geometric.typing import OptPairTensor, Adj, Size, OptTensor, Tensor

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

        return (feats_out, vectors_out)

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


class GVP_MPNN(MessagePassing):
    r"""The Geometric Vector Perceptron message passing layer
        introduced in https://openreview.net/forum?id=1YLJDvSx6J4.
        
        Uses a Geometric Vector Perceptron instead of the normal 
        MLP in aggregation phase.

        Inputs will be a concatenation of (vectors, features)

        Args:
        * feats_x_in: int. number of scalar dimensions in the x inputs.
        * vectors_x_in: int. number of vector dimensions in the x inputs.
        * feats_x_out: int. number of scalar dimensions in the x outputs.
        * vectors_x_out: int. number of vector dimensions in the x outputs.
        * feats_edge_in: int. number of scalar dimensions in the edge_attr inputs.
        * vectors_edge_in: int. number of vector dimensions in the edge_attr inputs.
        * feats_edge_out: int. number of scalar dimensions in the edge_attr outputs.
        * vectors_edge_out: int. number of vector dimensions in the edge_attr outputs.
        * dropout: float. dropout rate.
        * vector_dim: int. dimensions of the space containing the vectors.
        * verbose: bool. verbosity level.
    """
    def __init__(self, feats_x_in, vectors_x_in,
                       feats_x_out, vectors_x_out,
                       feats_edge_in, vectors_edge_in,
                       feats_edge_out, vectors_edge_out,
                       dropout, vector_dim=3, verbose=False, **kwargs):
        super(GVP_MPNN, self).__init__(aggr="mean",**kwargs)
        self.verbose = verbose
        # record x dimensions ( vector + scalars )
        self.feats_x_in    = feats_x_in 
        self.vectors_x_in  = vectors_x_in # N vectors features in input
        self.feats_x_out   = feats_x_out 
        self.vectors_x_out = vectors_x_out # N vectors features in output
        # record edge_attr dimensions ( vector + scalars )
        self.feats_edge_in    = feats_edge_in 
        self.vectors_edge_in  = vectors_edge_in # N vectors features in input
        self.feats_edge_out   = feats_edge_out 
        self.vectors_edge_out = vectors_edge_out # N vectors features in output
        # aux layers
        self.vector_dim = vector_dim
        self.norm = nn.ModuleList([GVPLayerNorm(self.feats_x_out), # + self.feats_edge_out
                                   GVPLayerNorm(self.feats_x_out)])
        self.dropout = GVPDropout(dropout)
        # this receives the vec_in message AND the receiver node
        self.W_EV = nn.Sequential(GVP(
                                      dim_vectors_in = self.vectors_x_in + self.vectors_edge_in, 
                                      dim_vectors_out = self.vectors_x_out + self.feats_edge_out,
                                      dim_feats_in = self.feats_x_in + self.feats_edge_in, 
                                      dim_feats_out = self.feats_x_out + self.feats_edge_out
                                  ), 
                                  GVP(
                                      dim_vectors_in = self.vectors_x_out + self.feats_edge_out, 
                                      dim_vectors_out = self.vectors_x_out + self.feats_edge_out,
                                      dim_feats_in = self.feats_x_out + self.feats_edge_out,
                                      dim_feats_out = self.feats_x_out + self.feats_edge_out
                                  ),
                                  GVP(
                                      dim_vectors_in = self.vectors_x_out + self.feats_edge_out, 
                                      dim_vectors_out = self.vectors_x_out + self.feats_edge_out,
                                      dim_feats_in = self.feats_x_out + self.feats_edge_out,
                                      dim_feats_out = self.feats_x_out + self.feats_edge_out
                                  ))
        
        self.W_dh = nn.Sequential(GVP(
                                      dim_vectors_in = self.vectors_x_out,
                                      dim_vectors_out = 2*self.vectors_x_out,
                                      dim_feats_in = self.feats_x_out,
                                      dim_feats_out = 4*self.feats_x_out
                                  ),
                                  GVP(
                                      dim_vectors_in = 2*self.vectors_x_out,
                                      dim_vectors_out = self.vectors_x_out,
                                      dim_feats_in = 4*self.feats_x_out,
                                      dim_feats_out = self.feats_x_out
                                  ))


    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj,
                edge_attr: OptTensor = None, size: Size = None) -> Tensor:
        """"""
        x_size = list(x.shape)[-1]
        # aggregate feats and vectors separately
        feats, vectors = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        # aggregate
        feats, vectors = self.dropout(feats, vectors.reshape(vectors.shape[0], -1, self.vector_dim))
        # get the information relative to the nodes - edges not returned
        feats_nodes  = feats[:, :self.feats_x_in]
        vector_nodes = vectors[:, :self.vectors_x_in]
        # reshapes the vector part to last 3d
        x_vectors    = x[:, :self.vectors_x_in * self.vector_dim].reshape(x.shape[0], -1, self.vector_dim)
        feats, vectors = self.norm[0]( x[:, self.vectors_x_in * self.vector_dim:]+feats_nodes, x_vectors+vector_nodes )
        # update position-wise feedforward
        feats_, vectors_ = self.dropout( *self.W_dh( (feats, vectors) ) )
        feats, vectors   = self.norm[1]( feats+feats_, vectors+vectors_ )
        # replace in the original
        x = torch.cat( [feats, vectors.flatten(start_dim=-2)], dim=-1 )
        return x


    def message(self, x_j, edge_attr) -> Tensor:
        feats   = torch.cat([ x_j[:, self.vectors_x_in * self.vector_dim:],
                              edge_attr[:, self.vectors_edge_in * self.vector_dim:] ], dim=-1)
        vectors = torch.cat([ x_j[:, :self.vectors_x_in * self.vector_dim], 
                              edge_attr[:, :self.vectors_edge_in * self.vector_dim] ], dim=-1).reshape(x_j.shape[0],-1,self.vector_dim)
        feats, vectors = self.W_EV( (feats, vectors) )
        return feats, vectors.flatten(start_dim=-2)


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
        size = self.__check_input__(edge_index, size)
        coll_dict = self.__collect__(self.__user_args__,
                                     edge_index, size, kwargs)
        msg_kwargs = self.inspector.distribute('message', coll_dict)
        out = self.message(**msg_kwargs)
        feats, vectors = self.message(**msg_kwargs)
        # aggregate them
        aggr_kwargs = self.inspector.distribute('aggregate', coll_dict)
        out_feats   = self.aggregate(feats, **aggr_kwargs)
        out_vectors = self.aggregate(vectors, **aggr_kwargs)
        # return tuple
        update_kwargs = self.inspector.distribute('update', coll_dict)
        return self.update((out_feats, out_vectors), **update_kwargs)

        
    def __repr__(self):
        dict_print = { "feats_x_in": feats_x_in, 
                       "vectors_x_in": vectors_x_in,
                       "feats_x_out": feats_x_out,
                       "vectors_x_out": vectors_x_out, 
                       "feats_edge_in": feats_edge_in,
                       "vectors_edge_in": vectors_edge_in, 
                       "feats_edge_out": feats_edge_out,
                       "vectors_edge_out": vectors_edge_out,
                       "vector_dim": vector_dim }
        return  'GVP_MPNN Layer with the following attributes: ' + str(dict_print)


class GVP_Network():
    def __init__(self, n_layers, 
                       feats_x_in, vectors_x_in,
                       feats_x_out, vectors_x_out,
                       feats_edge_in, vectors_edge_in,
                       feats_edge_out, vectors_edge_out,
                       dropout, vector_dim=3, verbose=False):
        super().__init__()
        self.n_layers         = n_layers  
        self.fc_layers        = torch.nn.ModuleList()
        self.gcnn_layers      = torch.nn.ModuleList()
        self.feats_x_in       = feats_x_in
        self.vectors_x_in     = vectors_x_in
        self.feats_x_out      = feats_x_out
        self.vectors_x_out    = vectors_x_out
        self.feats_edge_in    = feats_edge_in
        self.vectors_edge_in  = vectors_edge_in
        self.feats_edge_out   = feats_edge_out
        self.vectors_edge_out = vectors_edge_out
        self.dropout          = dropout
        self.vector_dim       = vector_dim
        self.verbose          = verbose

        # instantiate layers
        for i in range(n_layers):
            layer = GVP_MPNN(feats_x_in, vectors_x_in,
                             feats_x_out, vectors_x_out,
                             feats_edge_in, vectors_edge_in,
                             feats_edge_out, vectors_edge_out,
                             dropout, vector_dim=vector_dim, verbose=verbose)
            self.gcnn_layers.append(layer)

    def forward(self, x, edge_index, batch, edge_attr, bsize=None):
        # pass layers
        for i,layer in enumerate(self.gcnn_layers):
          x = layer(x, edge_index, edge_attr, size=bsize)
        return x

    def __repr__(self):
        return 'GVP_Network of: {0} layers'.format(len(self.gcnn_layers))
