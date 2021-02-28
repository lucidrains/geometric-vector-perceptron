# Author: Eric Alcaide

import os 
import sys
# science
import torch
import numpy as np 
from einops import repeat, rearrange
# custom utils - from https://github.com/EleutherAI/mp_nerf
from data_handler import *

# new data builders
def get_atom_ids_dict():
    """ Get's a dict mapping each atom to a token. """
    ids = set(["N", "CA", "C", "O"])

    for k,v in SC_BUILD_INFO.items():
        for name in v["atom-names"]:
            ids.add(name)
            
    return {k: i for i,k in enumerate(sorted(ids))}


#################################
##### ORIGINAL PROJECT DATA #####
#################################

AAS = "ARNDCQEGHILKMFPSTWYV_"
ATOM_IDS = get_atom_ids_dict()
# numbers follow the same order as sidechainnet atoms
GVP_DATA = { 
    'A': {
        'bonds': [[0,1], [1,2], [2,3], [1,4]] 
         },
    'R': {
        'bonds': [[0,1], [1,2], [2,3], [2,4], [4,5], [5,6],
                  [6,7], [7,8], [8,9], [8,10]] 
         },
    'N': {
        'bonds': [[0,1], [1,2], [2,3], [1,4], [4,5], [5,6], [5,7]] 
         },
    'D': {
        'bonds': [[0,1], [1,2], [2,3], [1,4], [4,5], [4,6], [4,6]] 
         },
    'C': {
        'bonds': [[0,1], [1,2], [2,3], [1,4], [4,5]] 
        },
    'Q': {
        'bonds': [[0,1], [1,2], [2,3], [1,4], [4,5], [5,6],
                  [6,7], [6,8]] 
        },
    'E': {
        'bonds': [[0,1], [1,2], [2,3], [1,4], [4,5], [5,6],
                  [6,7], [6,8]] 
        },
    'G': {
        'bonds': [[0,1], [1,2], [2,3]] 
        },
    'H': {
        'bonds': [[0,1], [1,2], [2,3], [1,4], [4,5], [5,6],
                  [6,7], [7,8], [8,9], [5,9]] 
        },
    'I': {
        'bonds': [[0,1], [1,2], [2,3], [1,4], [4,5], [5,6],
                  [4,7]] 
         },
    'L': {
        'bonds': [[0,1], [1,2], [2,3], [1,4], [4,5], [5,6],
                  [5,7]] 
         },
    'K': {
        'bonds': [[0,1], [1,2], [2,3], [1,4], [4,5], [5,6],
                  [6,7], [7,8]] 
         },
    'M': {
        'bonds': [[0,1], [1,2], [2,3], [1,4], [4,5], [5,6],
                  [6,7]] 
         },
    'F': {
        'bonds': [[0,1], [1,2], [2,3], [1,4], [4,5], [5,6],
                  [6,7], [7,8], [8,9], [9,10], [5,10]] 
         },
    'P': {
        'bonds': [[0,1], [1,2], [2,3], [1,4], [4,5], [5,6],
                  [0,6]] 
         },
    'S': {
        'bonds': [[0,1], [1,2], [2,3], [1,4], [4,5]] 
         },
    'T': {
        'bonds': [[0,1], [1,2], [2,3], [1,4], [4,5], [4,6]] 
         },
    'W': {
        'bonds': [[0,1], [1,2], [2,3], [1,4], [4,5], [5,6],
                  [6,7], [7,8], [8,9], [9,10], [10,11], [11,12],
                  [12, 13], [5,13], [8,13]] 
         },
    'Y': {
        'bonds': [[0,1], [1,2], [2,3], [1,4], [4,5], [5,6],
                  [6,7], [7,8], [8,9], [8,10], [10,11], [5,11]] 
         },
    'V': {
        'bonds': [[0,1], [1,2], [2,3], [1,4], [4,5], [4,6]] 
         },
    '_': {
        'bonds': []
        }
    }


#################################
##### ORIGINAL PROJECT DATA #####
#################################

def graph_laplacian_embedds(edges, eigen_k, center_idx=1, norm=False):
    """ Returns the embeddings of points in the K 
        first eigenvectors of the graph Laplacian.
        Inputs:
        * edges: (2, N). long tensor or list. undirected edges are enough.
        * eigen_k: int. N of first eigenvectors to return embeddings for.
        * center_idx: int. index to take as center for the embeddings
        * norm: bool. Whether to use the normalized Laplacian. Not recommended.
        Output: (n_points, eigen_k)
    """
    if isinstance(edges, list):
        edges = torch.tensor(edges).long()
        # correct dims
        if edges.shape[0] != 2:
            edges = edges.t()
        # early stopping if empty entry
        if edges.shape[0] == 0:
            return torch.zeros(1, eigen_k)
    # get params
    size = torch.max(edges)+1
    device = edges.device
    # crate laplacian
    adj_mat = torch.eye(size, device=device) 
    for i,j in edges.t():
        adj_mat[i,j] = adj_mat[j,i] = 1.
        
    deg_mat = torch.eye(size) * adj_mat.sum(dim=-1, keepdim=True)
    laplace = deg_mat - adj_mat
    # use norm-laplace if arg passed
    if norm:
        for i,j in edges.t():
            laplace[i,j] = laplace[j,i] = -1 / (deg_mat[i,i] * deg_mat[j,j])**0.5
    # get laplacian basis - eigendecomposition - order importance by eigenvalue
    e, v = torch.symeig(laplace, eigenvectors=True)
    idxs = torch.sort( e.abs(), descending=True)[1]
    # take embedds and center
    embedds = v[:, idxs[:eigen_k]]
    embedds = embedds - embedds[center_idx].unsqueeze(-2)
    return embedds


def make_atom_id_embedds(k):
    """ Return the tokens for each atom in the aa. """
    mask = torch.zeros(14).long()
    atom_list = ["N", "CA", "C", "O"] + SC_BUILD_INFO[k]["atom-names"]
    for i,atom in enumerate(atom_list):
        mask[i] = ATOM_IDS[atom]
    return mask


#################################
########## SAVE INFO ############
#################################

SUPREME_INFO = {k: {"cloud_mask": make_cloud_mask(k),
                    "bond_mask": make_bond_mask(k),
                    "theta_mask": make_theta_mask(k),
                    "torsion_mask": make_torsion_mask(k),
                    "idx_mask": make_idx_mask(k),
                    #
                    "eigen_embedd": graph_laplacian_embedds(GVP_DATA[k]["bonds"], eigen_k = 3),
                    "atom_id_embedd": make_atom_id_embedds(k)
                    } 
                for k in "ARNDCQEGHILKMFPSTWYV_"}

#################################
######### RANDOM UTILS ##########
#################################


def encode_dist(x, scales=[1,2,4,8]):
    """ Encodes a distance with sines and cosines. 
        Inputs:
        * x: (batch, N) or (N,). data to encode.
              Infer devic and type (f16, f32, f64) from here.
        * scales: (s,) or list. lower or higher depending on distances.
        Output: (..., number_of_scales*2)
    """
    # infer device
    device, precise = x.device, x.type()
    # convert to tensor
    if isinstance(scales, list):
        scales = torch.tensor([scales], device=device).type(precise)
    # get pos encodings
    sines   = torch.sin(x.unsqueeze(-1) / scales)
    cosines = torch.cos(x.unsqueeze(-1) / scales)
    # concat and return
    return torch.cat([sines, cosines], dim=-1)


def decode_dist(x, scales=[1,2,4,8]):
    """ Decodes a distance with from sines and cosines. 
        Inputs:
        * x: (..., n_scales * 2) data to decode.
        * scales: (s,) or list. scales used to encode the data.
        Outputs: (..., 1). same shape as the encoded distance
    """
    # infer device
    device, precise = x.device, x.type()
    # convert to tensor
    if isinstance(scales, list):
        scales = torch.tensor([scales], device=device).type(precise)
    # decode with atan
    half = x.shape[-1]//2
    x_new = (torch.atan2(x[..., :half], x[..., half:]) * scales) % (2*np.pi)
    return x_new


def prot_covalent_bond(seq, cloud_mask=None):
    """ Returns the idxs of covalent bonds for a protein.
        Inputs 
        * seq: str. Protein sequence in 1-letter AA code.
        * cloud_mask: mask selecting the present atoms.
        Outputs: edge_idxs
    """
    # create or infer
    if cloud_mask is None: 
        cloud_mask = scn_cloud_mask(seq).bool()
    device, precise = cloud_mask.device, cloud_mask.type()
    # get starting poses for every aa
    scaff = torch.zeros_like(cloud_mask)
    scaff[:, 0] = 1
    idxs = scaff[cloud_mask].nonzero().view(-1)
    # get poses + idxs from the dict with GVP_DATA - return all edges
    return torch.cat( [ idx + torch.tensor( GVP_DATA[seq[i]]['bonds'] ).long().t() \
                        for i,idx in enumerate(idxs) ] , dim=-1).to(device)


def dist2ca(x, mask=None, eps=1e-7):
    """ Calculates distance from each point to C-alfa.
        Inputs:
        * x: (L, 14, D)
        * mask: boolean mask of (L, 14)
        Returns unit vectors and norm. 
    """
    x = x - x[:, 1].unsqueeze(1)
    norm = torch.norm(x, dim=-1, keepdim=True)
    x_norm = x / (norm+eps)
    if mask:
        return x_norm[mask], norm[mask]
    return x_norm, norm


def orient_aa(x, mask=None, eps=1e-7):
    """ Calculates unit vectors and norms of features for backbone.
        Inputs:
        * x: (L, 14, D). Cordinates in Sidechainnet format.
        Returns unit vectors (5) and norms (3). 
    """
    # get tensor info
    device, precise = x.device, x.type()

    vec_wrap  = torch.zeros(5, x.shape[0], 3, device=device) # (feats, L, dims+1)
    norm_wrap = torch.zeros(3, x.shape[0], device=device)
    # first feat is CB-CA
    vec_wrap[0]  = x[:, 4] - x[:, 1]
    norm_wrap[0] = torch.norm(vec_wrap[0], dim=-1)
    vec_wrap[0] /= norm_wrap[0].unsqueeze(dim=-1) + eps
    # second is CA+ - CA :
    vec_wrap[1, :-1]  = x[:-1, 1] - x[1:, 1]
    norm_wrap[1, :-1] = torch.norm(vec_wrap[1, :-1], dim=-1)
    vec_wrap[1, :-1] /= norm_wrap[1, :-1].unsqueeze(dim=-1) + eps
    # same but reverse vectors
    vec_wrap[2] = (-1)*vec_wrap[1]
    # third is CA - CA-
    vec_wrap[3, 1:]  = x[:-1, 1] - x[1:, 1]
    norm_wrap[2, 1:] = torch.norm(vec_wrap[3, 1:], dim=-1)
    vec_wrap[3, 1:] /= norm_wrap[2, 1:].unsqueeze(dim=-1) + eps
    # now vectors in reverse order
    vec_wrap[4] = (-1)*vec_wrap[3]

    return vec_wrap, norm_wrap


def chain2atoms(x, mask=None):
    """ Expand from (L, other) to (L, C, other). """
    device, precise = x.device, x.type()
    # get mask
    wrap = torch.ones(x.shape[0], 14, *x.shape[1:]).type(precise).to(device)
    # assign
    wrap = wrap * x.unsqueeze(1)
    if mask is not None:
        return wrap[mask]
    return wrap


def from_encode_to_pred(whole_point_enc, embedd_info, needed_info, vec_dim=3):
    """ Turns the encoding from the above func into a label / prediction format.
        Containing only the essential for position recovery (radial unit vec + norm)
        Inputs: input_tuple containing:
        * whole_point_enc: (atoms, vector_dims+scalar_dims)
                           Same shape from the function above. 
                           Radial unit vector must be be the first vector dims
        * embedd_info: dict. contains the number of scalar and vector feats.
    """
    vec_dims = vec_dim * embedd_info["point_n_vectors"]
    return torch.cat([# unit radial vector
                      whole_point_enc[:, :3], 
                      # encoding of vector norm
                      whole_point_enc[:, vec_dims : 2*len(needed_info["atom_pos_scales"])+vec_dims ] 
                     ], dim=-1)


def encode_whole_bonds(x, x_format="coords", embedd_info={},
                       needed_info = {"cutoffs": [2,5,10],
                                      "bond_scales": [.5, 1, 2]}, free_mem=False):
    """ Given some coordinates, and the needed info,
        encodes the bonds from point information.
        * x: (N, 3) or prediction format
        * x_format: one of ["coords" or "prediction"]
        * embedd_info: dict. contains the needed embedding info
        * needed_info: dict. contains additional needed info
            * cutoffs: list. cutoff distances for bonds
    """ 
    device, precise = x.device, x.type()
    # convert to 3d coords if passed as preds
    if x_format == "prediction":
        pred_x = from_encode_to_pred(x, embedd_info, needed_info)
        dist_x = decode_dist(pred_x[:, 3:], scales=needed_info["atom_pos_scales"]).mean(dim=-1)
        x = pred_x[:, :3] * dist_x.unsqueeze(-1)

    # encode bonds

    # 1. BONDS: find the covalent bond_indices
    native_bond_idxs = prot_covalent_bond(needed_info["seq"])

    # points under cutoff = d(i - j) < X 
    cutoffs = torch.tensor(needed_info["cutoffs"], device=device).type(precise)
    dist_mat = torch.cdist(x, x, p=2)
    bond_buckets = torch.bucketize(dist_mat, cutoffs) 
    # assign native bonds the extra token - don't repeat them
    bond_buckets[native_bond_idxs[0], native_bond_idxs[1]] = cutoffs.shape[0]
    bond_buckets[native_bond_idxs[1], native_bond_idxs[0]] = cutoffs.shape[0]
    # find the indexes - symmetric and we dont want the diag
    bond_buckets   += len(cutoffs) * torch.eye(bond_buckets.shape[0]).long()
    close_bond_idxs = ( bond_buckets < len(cutoffs) ).nonzero().t()
    # merge all bonds
    if close_bond_idxs.shape[0] > 0:
        whole_bond_idxs = torch.cat([native_bond_idxs, close_bond_idxs], dim=-1)
    else:
        whole_bond_idxs = native_bond_idxs

    # 2. ATTRS: encode bond -> attrs
    bond_vecs  = x[ whole_bond_idxs[0] ] - x[ whole_bond_idxs[1] ]
    bond_norms = dist_mat[ whole_bond_idxs[0] , whole_bond_idxs[1] ]
    bond_norms_enc = encode_dist(bond_norms, scales=needed_info["bond_scales"]).squeeze()
    # bond unit vector
    bond_vecs /= bond_norms.unsqueeze(-1)

    # pack scalars and vectors
    bond_n_vectors = 1
    bond_n_scalars = 2 * len(needed_info["bond_scales"]) + 1 # last one is an embedding of size 1+len(cutoffs)
    whole_bond_enc = torch.cat([bond_vecs, # 1 vector - no need of reverse - we do 2x bonds (symmetry)
                                # scalars
                                bond_norms_enc, # 2 * len(scales)
                                bond_buckets[ whole_bond_idxs[0], whole_bond_idxs[1] ].unsqueeze(-1) # 1
                               ], dim=-1) 
    # free gpu mem
    if free_mem:
        del bond_buckets, bond_norms_enc, bond_vecs, dist_mat

    embedd_info = {"bond_n_vectors": bond_n_vectors, 
                   "bond_n_scalars": bond_n_scalars, 
                   "bond_embedding_nums": [ len(cutoffs) + 1 ]} # extra one for covalent (default)

    return whole_bond_enc, whole_bond_idxs, embedd_info


def encode_whole_protein(seq, true_coords, angles, padding_seq,
                         needed_info = { "cutoffs": [2, 5, 10],
                                          "bond_scales": [0.5, 1, 2]}, free_mem=False):
    """ Encodes a whole protein. In points + vectors. """
    device, precise = true_coords.device, true_coords.type()
    #################
    # encode points #
    #################
    scaffolds = build_scaffolds_from_scn_angles(seq[:-padding_seq], angles[:-padding_seq])
    flat_mask = rearrange(scaffolds["cloud_mask"], 'l c -> (l c)')
    # embedd everything

    # general position embedding
    center_coords = true_coords - true_coords.mean(dim=0)
    pos_unit_norms = torch.norm(center_coords, dim=-1, keepdim=True)
    pos_unit_vecs  = center_coords / pos_unit_norms
    pos_unit_norms_enc = encode_dist(pos_unit_norms, scales=needed_info["atom_pos_scales"]).squeeze()
    # reformat coordinates to scn (L, 14, 3) - TODO: solve if padding=0
    coords_wrap = rearrange(center_coords, '(l c) d -> l c d', c=14)[:-padding_seq] 

    # position in backbone embedding
    aa_pos = encode_dist( torch.arange(len(seq[:-padding_seq]), device=device), scales=needed_info["aa_pos_scales"])
    atom_pos = chain2atoms(aa_pos)[scaffolds["cloud_mask"]]

    # atom identity embedding
    atom_id_embedds = torch.stack([SUPREME_INFO[k]["atom_id_embedd"] for k in seq[:-padding_seq]], 
                                  dim=0)[scaffolds["cloud_mask"]].to(device)
    # aa embedding
    seq_int = torch.tensor([AAS.index(aa) for aa in seq[:-padding_seq]], device=device).long()
    aa_id_embedds   = chain2atoms(seq_int, mask=scaffolds["cloud_mask"])

    # CA - SCN distance
    dist2ca_vec, dist2ca_norm = dist2ca(coords_wrap) 
    dist2ca_norm_enc = encode_dist(dist2ca_norm, scales=needed_info["dist2ca_norm_scales"]).squeeze()

    # BACKBONE feats
    vecs, norms    = orient_aa(coords_wrap)
    bb_vecs_atoms  = chain2atoms(torch.transpose(vecs, 0, 1), mask=scaffolds["cloud_mask"])
    bb_norms_atoms = chain2atoms(torch.transpose(norms, 0, 1), mask=scaffolds["cloud_mask"])
    bb_norms_atoms_enc = encode_dist(bb_norms_atoms, scales=[0.5])

    ################
    # encode bonds #
    ################
    bond_info = encode_whole_bonds(x = coords_wrap[scaffolds["cloud_mask"]],
                                   x_format = "coords",
                                   embedd_info = {},
                                   needed_info = needed_info )
    whole_bond_enc, whole_bond_idxs, bond_embedd_info = bond_info
    #########
    # merge #
    #########

    # concat so that final is [vector_dims, scalar_dims]
    point_n_vectors = 1 + 1 + 5
    point_n_scalars = 12 + 16 + 6 + 6 + 2 # the last 2 are to be embedded yet
    whole_point_enc = torch.cat([ pos_unit_vecs[ :-padding_seq*14 ][ flat_mask ], # 1
                                  dist2ca_vec[scaffolds["cloud_mask"]], # 1
                                  rearrange(bb_vecs_atoms, 'atoms n d -> atoms (n d)'), # 5
                                  # scalars
                                  pos_unit_norms_enc[ :-padding_seq*14 ][ flat_mask ], # 14
                                  atom_pos, # 16
                                  dist2ca_norm_enc[scaffolds["cloud_mask"]], # 6
                                  rearrange(bb_norms_atoms_enc, 'atoms feats encs -> atoms (feats encs)'), # 6
                                  atom_id_embedds.unsqueeze(-1),
                                  aa_id_embedds.unsqueeze(-1) ], dim=-1) # the last 2 are yet to be embedded
    if free_mem:
        del pos_unit_vecs, dist2ca_vec, bb_vecs_atoms, pos_unit_norms_enc, \
            atom_pos, dist2ca_norm_enc, bb_norms_atoms_enc, atom_id_embedds, aa_id_embedds


    # record embedding dimensions
    point_embedd_info = {"point_n_vectors": point_n_vectors,
                         "point_n_scalars": point_n_scalars,}

    embedd_info = {**point_embedd_info, **bond_embedd_info}

    return whole_point_enc, whole_bond_idxs, whole_bond_enc, embedd_info


def get_prot(dataloader_=None, vocab_=None, min_len=80, max_len=150, verbose=True):
    """ Gets a protein from sidechainnet and returns
        the right attrs for training. 
        Inputs: 
        * dataloader_: sidechainnet iterator over dataset
        * vocab_: sidechainnet VOCAB class
        * min_len: int. minimum sequence length
        * max_len: int. maximum sequence length
        * verbose: bool. verbosity level
    """
    for batch in dataloader_['train']:
        real_seqs = [''.join([vocab_.int2char(aa) for aa in seq]) \
                     for seq in batch.int_seqs.numpy()]
        # try for breaking from 2 loops at once
        try:
            for i in range(len(batch.int_seqs.numpy())):
                # get variables
                seq     = real_seqs[i]
                int_seq = batch.int_seqs[i]
                angles  = batch.angs[i]
                # get padding
                padding_angles = (torch.abs(angles).sum(dim=-1) == 0).long().sum()
                padding_seq    = (np.array([x for x in seq]) == "_").sum()
                # only accept sequences with right dimensions and no missing coords
                # # bigger than 0 to avoid errors  with negative indexes later
                if list(batch.crds[i].shape)[0]//14 == len(int_seq):
                    if ( max_len > len(seq) and len(seq) > min_len ) and \
                       ( padding_seq == padding_angles and padding_seq > 0): 
                        if verbose:
                            print("stopping at sequence of length", len(seq))
                            # print(len(seq), angles.shape, "paddings: ", padding_seq, padding_angles)
                        raise StopIteration
                    else:
                        # print("found a seq of length:", len(seq),
                        #        "but oustide the threshold:", min_len, max_len)
                        pass
        except StopIteration:
            break
            
    return seq, batch.crds[i], angles, padding_seq


