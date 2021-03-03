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
                       dropout, residual=False, vector_dim=3, 
                       verbose=False, **kwargs):
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
        self.residual = residual
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
        # make it residual
        new_x = torch.cat( [feats, vectors.flatten(start_dim=-2)], dim=-1 )
        if self.residual:
          return new_x + x
        return new_x


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
        feats, vectors = self.message(**msg_kwargs)
        # aggregate them
        aggr_kwargs = self.inspector.distribute('aggregate', coll_dict)
        out_feats   = self.aggregate(feats, **aggr_kwargs)
        out_vectors = self.aggregate(vectors, **aggr_kwargs)
        # return tuple
        update_kwargs = self.inspector.distribute('update', coll_dict)
        return self.update((out_feats, out_vectors), **update_kwargs)

        
    def __repr__(self):
        dict_print = { "feats_x_in": self.feats_x_in,
                       "vectors_x_in": self.vectors_x_in,
                       "feats_x_out": self.feats_x_out,
                       "vectors_x_out": self.vectors_x_out,
                       "feats_edge_in": self.feats_edge_in,
                       "vectors_edge_in": self.vectors_edge_in,
                       "feats_edge_out": self.feats_edge_out,
                       "vectors_edge_out": self.vectors_edge_out,
                       "vector_dim": self.vector_dim }
        return  'GVP_MPNN Layer with the following attributes: ' + str(dict_print)


class GVP_Network(nn.Module):
    r"""Sample GNN model architecture that uses the Geometric Vector Perceptron
        message passing layer to learn over point clouds. 
        Main MPNN layer introduced in https://openreview.net/forum?id=1YLJDvSx6J4.

        Inputs will be standard GNN: x, edge_index, edge_attr, batch, ...

        Args:
        * n_layers: int. number of MPNN layers
        * feats_x_in: int. number of scalar dimensions in the x inputs.
        * vectors_x_in: int. number of vector dimensions in the x inputs.
        * feats_x_out: int. number of scalar dimensions in the x outputs.
        * vectors_x_out: int. number of vector dimensions in the x outputs.
        * feats_edge_in: int. number of scalar dimensions in the edge_attr inputs.
        * vectors_edge_in: int. number of vector dimensions in the edge_attr inputs.
        * feats_edge_out: int. number of scalar dimensions in the edge_attr outputs.
        * embedding_nums: list. number of unique keys to embedd. for points
                          1 entry per embedding needed. 
        * embedding_dims: list. point - number of dimensions of
                          the resulting embedding. 1 entry per embedding needed. 
        * edge_embedding_nums: list. number of unique keys to embedd. for edges.
                               1 entry per embedding needed. 
        * edge_embedding_dims: list. point - number of dimensions of
                               the resulting embedding. 1 entry per embedding needed. 
        * vectors_edge_out: int. number of vector dimensions in the edge_attr outputs.
        * dropout: float. dropout rate.
        * vector_dim: int. dimensions of the space containing the vectors.
        * recalc: bool. Whether to recalculate edge features between MPNN layers.
        * verbose: bool. verbosity level.
    """
    def __init__(self, n_layers, 
                       feats_x_in, vectors_x_in,
                       feats_x_out, vectors_x_out,
                       feats_edge_in, vectors_edge_in,
                       feats_edge_out, vectors_edge_out,
                       embedding_nums=[], embedding_dims=[],
                       edge_embedding_nums=[], edge_embedding_dims=[],
                       dropout=0.0, residual=False, vector_dim=3,
                       recalc=True, verbose=False):
        super().__init__()

        self.n_layers         = n_layers 
        # Embeddings? solve here
        self.embedding_nums   = embedding_nums
        self.embedding_dims   = embedding_dims
        self.emb_layers       = torch.nn.ModuleList()
        self.edge_embedding_nums = edge_embedding_nums
        self.edge_embedding_dims = edge_embedding_dims
        self.edge_emb_layers     = torch.nn.ModuleList()
        # instantiate point and edge embedding layers
        for i in range( len(self.embedding_dims) ):
            self.emb_layers.append(nn.Embedding(num_embeddings = embedding_nums[i],
                                                embedding_dim  = embedding_dims[i]))
            feats_x_in += embedding_dims[i] - 1
            feats_x_out += embedding_dims[i] - 1
        for i in range( len(self.edge_embedding_dims) ):
            self.edge_emb_layers.append(nn.Embedding(num_embeddings = edge_embedding_nums[i],
                                                     embedding_dim  = edge_embedding_dims[i]))
            feats_edge_in += edge_embedding_dims[i] - 1
            feats_edge_out += edge_embedding_dims[i] - 1
        # rest
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
        self.residual         = residual
        self.vector_dim       = vector_dim
        self.recalc           = recalc
        self.verbose          = verbose
        
        # instantiate layers
        for i in range(n_layers):
            layer = GVP_MPNN(feats_x_in, vectors_x_in,
                             feats_x_out, vectors_x_out,
                             feats_edge_in, vectors_edge_in,
                             feats_edge_out, vectors_edge_out,
                             dropout, residual=residual,
                             vector_dim=vector_dim, verbose=verbose)
            self.gcnn_layers.append(layer)

    def forward(self, x, edge_index, batch, edge_attr,
                bsize=None, recalc_edge=None, verbose=0):
        """ Embedding of inputs when necessary, then pass layers.
            Recalculate edge features every time with the
            `recalc_edge` function.
        """
        # do embeddings when needed
        # pick to embedd. embedd sequentially and add to input
        
        # points:
        to_embedd = x[:, -len(self.embedding_dims):].long()
        for i,emb_layer in enumerate(self.emb_layers):
            # the portion corresponding to `to_embedd` part gets dropped
            # at first iter
            stop_concat = -len(self.embedding_dims) if i == 0 else x.shape[-1]
            x = torch.cat([ x[:, :stop_concat], 
                            emb_layer( to_embedd[:, i] ) 
                          ], dim=-1)
        # pass layers
        for i,layer in enumerate(self.gcnn_layers):
            # embedd edge items (needed everytime since edge_attr and idxs
            # are recalculated every pass)
            to_embedd = edge_attr[:, -len(self.edge_embedding_dims):].long()
            for i,edge_emb_layer in enumerate(self.edge_emb_layers):
                # the portion corresponding to `to_embedd` part gets dropped
                # at first iter
                stop_concat = -len(self.edge_embedding_dims) if i == 0 else x.shape[-1]
                edge_attr = torch.cat([ edge_attr[:, :-len(self.edge_embedding_dims) + i], 
                                        edge_emb_layer( to_embedd[:, i] ) 
                              ], dim=-1)
            # pass layers
            x = layer(x, edge_index, edge_attr, size=bsize)

            # recalculate edge info - not needed if last layer
            if i < len(self.gcnn_layers)-1 and self.recalc:
                edge_attr, edge_index, _ = recalc_edge(x.detach()) # returns attr, idx, embedd_info
            
            if verbose:
                print("========")
                print(i, "layer, nlinks:", edge_attr.shape)
            
        return x

    def __repr__(self):
        return 'GVP_Network of: {0} layers'.format(len(self.gcnn_layers))
