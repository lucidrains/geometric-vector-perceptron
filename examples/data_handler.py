# Author: Eric Alcaide


# A substantial part has been borrowed from
# https://github.com/jonathanking/sidechainnet
#
# Here's the License for it:
#
# Copyright 2020 Jonathan King
# Redistribution and use in source and binary forms, with or without modification, are permitted provided that the
# following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following
# disclaimer in the documentation and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote
# products derived from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
# INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF
# THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import warnings
warnings.filterwarnings("ignore")

import torch
import numpy as np 
from einops import repeat, rearrange

######################
## structural utils ##
######################

def get_dihedral(c1, c2, c3, c4):
    """ Returns the dihedral angle in radians.
        Will use atan2 formula from: 
        https://en.wikipedia.org/wiki/Dihedral_angle#In_polymer_physics
        Inputs: 
        * c1: (batch, 3) or (3,)
        * c2: (batch, 3) or (3,)
        * c3: (batch, 3) or (3,)
        * c4: (batch, 3) or (3,)
    """
    u1 = c2 - c1
    u2 = c3 - c2
    u3 = c4 - c3

    return torch.atan2( ( (torch.norm(u2, dim=-1, keepdim=True) * u1) * torch.cross(u2,u3, dim=-1) ).sum(dim=-1) ,  
                        (  torch.cross(u1,u2, dim=-1) * torch.cross(u2, u3, dim=-1) ).sum(dim=-1) )


def get_angle(c1, c2, c3):
    """ Returns the angle in radians.
        Inputs: 
        * c1: (batch, 3) or (3,)
        * c2: (batch, 3) or (3,)
        * c3: (batch, 3) or (3,)
    """
    u1 = c2 - c1
    u2 = c3 - c2

    # don't use trad arccos since it gets the "smallest angle", 
    # not necessarily the one we want
    # return torch.acos( (u1*u2).sum(dim=-1) / (torch.norm(u1, dim=-1)*torch.norm(u2, dim=-1)) )+

    # better use atan2 formula: atan2(cross, dot) from here: 
    # https://johnblackburne.blogspot.com/2012/05/angle-between-two-3d-vectors.html

    # add a minus since we want the angle in reversed order - sidechainnet issues
    return torch.atan2( torch.norm(torch.cross(u1,u2, dim=-1), dim=-1), 
                        -(u1*u2).sum(dim=-1) ) 


def kabsch_torch(X, Y):
    """ Kabsch alignment of X into Y. 
        Assumes X,Y are both (D, N) - usually (3, N)
    """
    #  center X and Y to the origin
    X_ = X - X.mean(dim=-1, keepdim=True)
    Y_ = Y - Y.mean(dim=-1, keepdim=True)
    # calculate convariance matrix (for each prot in the batch)
    C = torch.matmul(X_, Y_.t())
    # Optimal rotation matrix via SVD - warning! W must be transposed
    V, S, W = torch.svd(C.detach())
    # determinant sign for direction correction
    d = (torch.det(V) * torch.det(W)) < 0.0
    if d:
        S[-1]    = S[-1] * (-1)
        V[:, -1] = V[:, -1] * (-1)
    # Create Rotation matrix U
    U = torch.matmul(V, W.t())
    # calculate rotations
    X_ = torch.matmul(X_.t(), U).t()
    # return centered and aligned
    return X_, Y_


def rmsd_torch(X, Y):
    """ Assumes x,y are both (batch, d, n) - usually (batch, 3, N). """
    return torch.sqrt( torch.mean((X - Y)**2, axis=(-1, -2)) )


############
### INFO ###
############

SC_BUILD_INFO = {
    'A': {
        'angles-names': ['N-CA-CB'],
        'angles-types': ['N -CX-CT'],
        'angles-vals': [1.9146261894377796],
        'atom-names': ['CB'],
        'bonds-names': ['CA-CB'],
        'bonds-types': ['CX-CT'],
        'bonds-vals': [1.526],
        'torsion-names': ['C-N-CA-CB'],
        'torsion-types': ['C -N -CX-CT'],
        'torsion-vals': ['p']
    },
    'R': {
        'angles-names': [
            'N-CA-CB', 'CA-CB-CG', 'CB-CG-CD', 'CG-CD-NE', 'CD-NE-CZ', 'NE-CZ-NH1',
            'NE-CZ-NH2'
        ],
        'angles-types': [
            'N -CX-C8', 'CX-C8-C8', 'C8-C8-C8', 'C8-C8-N2', 'C8-N2-CA', 'N2-CA-N2',
            'N2-CA-N2'
        ],
        'angles-vals': [
            1.9146261894377796, 1.911135530933791, 1.911135530933791, 1.9408061282176945,
            2.150245638457014, 2.0943951023931953, 2.0943951023931953
        ],
        'atom-names': ['CB', 'CG', 'CD', 'NE', 'CZ', 'NH1', 'NH2'],
        'bonds-names': ['CA-CB', 'CB-CG', 'CG-CD', 'CD-NE', 'NE-CZ', 'CZ-NH1', 'CZ-NH2'],
        'bonds-types': ['CX-C8', 'C8-C8', 'C8-C8', 'C8-N2', 'N2-CA', 'CA-N2', 'CA-N2'],
        'bonds-vals': [1.526, 1.526, 1.526, 1.463, 1.34, 1.34, 1.34],
        'torsion-names': [
            'C-N-CA-CB', 'N-CA-CB-CG', 'CA-CB-CG-CD', 'CB-CG-CD-NE', 'CG-CD-NE-CZ',
            'CD-NE-CZ-NH1', 'CD-NE-CZ-NH2'
        ],
        'torsion-types': [
            'C -N -CX-C8', 'N -CX-C8-C8', 'CX-C8-C8-C8', 'C8-C8-C8-N2', 'C8-C8-N2-CA',
            'C8-N2-CA-N2', 'C8-N2-CA-N2'
        ],
        'torsion-vals': ['p', 'p', 'p', 'p', 'p', 'p', 'i']
    },
    'N': {
        'angles-names': ['N-CA-CB', 'CA-CB-CG', 'CB-CG-OD1', 'CB-CG-ND2'],
        'angles-types': ['N -CX-2C', 'CX-2C-C ', '2C-C -O ', '2C-C -N '],
        'angles-vals': [
            1.9146261894377796, 1.9390607989657, 2.101376419401173, 2.035053907825388
        ],
        'atom-names': ['CB', 'CG', 'OD1', 'ND2'],
        'bonds-names': ['CA-CB', 'CB-CG', 'CG-OD1', 'CG-ND2'],
        'bonds-types': ['CX-2C', '2C-C ', 'C -O ', 'C -N '],
        'bonds-vals': [1.526, 1.522, 1.229, 1.335],
        'torsion-names': ['C-N-CA-CB', 'N-CA-CB-CG', 'CA-CB-CG-OD1', 'CA-CB-CG-ND2'],
        'torsion-types': ['C -N -CX-2C', 'N -CX-2C-C ', 'CX-2C-C -O ', 'CX-2C-C -N '],
        'torsion-vals': ['p', 'p', 'p', 'i']
    },
    'D': {
        'angles-names': ['N-CA-CB', 'CA-CB-CG', 'CB-CG-OD1', 'CB-CG-OD2'],
        'angles-types': ['N -CX-2C', 'CX-2C-CO', '2C-CO-O2', '2C-CO-O2'],
        'angles-vals': [
            1.9146261894377796, 1.9390607989657, 2.0420352248333655, 2.0420352248333655
        ],
        'atom-names': ['CB', 'CG', 'OD1', 'OD2'],
        'bonds-names': ['CA-CB', 'CB-CG', 'CG-OD1', 'CG-OD2'],
        'bonds-types': ['CX-2C', '2C-CO', 'CO-O2', 'CO-O2'],
        'bonds-vals': [1.526, 1.522, 1.25, 1.25],
        'torsion-names': ['C-N-CA-CB', 'N-CA-CB-CG', 'CA-CB-CG-OD1', 'CA-CB-CG-OD2'],
        'torsion-types': ['C -N -CX-2C', 'N -CX-2C-CO', 'CX-2C-CO-O2', 'CX-2C-CO-O2'],
        'torsion-vals': ['p', 'p', 'p', 'i']
    },
    'C': {
        'angles-names': ['N-CA-CB', 'CA-CB-SG'],
        'angles-types': ['N -CX-2C', 'CX-2C-SH'],
        'angles-vals': [1.9146261894377796, 1.8954275676658419],
        'atom-names': ['CB', 'SG'],
        'bonds-names': ['CA-CB', 'CB-SG'],
        'bonds-types': ['CX-2C', '2C-SH'],
        'bonds-vals': [1.526, 1.81],
        'torsion-names': ['C-N-CA-CB', 'N-CA-CB-SG'],
        'torsion-types': ['C -N -CX-2C', 'N -CX-2C-SH'],
        'torsion-vals': ['p', 'p']
    },
    'Q': {
        'angles-names': ['N-CA-CB', 'CA-CB-CG', 'CB-CG-CD', 'CG-CD-OE1', 'CG-CD-NE2'],
        'angles-types': ['N -CX-2C', 'CX-2C-2C', '2C-2C-C ', '2C-C -O ', '2C-C -N '],
        'angles-vals': [
            1.9146261894377796, 1.911135530933791, 1.9390607989657, 2.101376419401173,
            2.035053907825388
        ],
        'atom-names': ['CB', 'CG', 'CD', 'OE1', 'NE2'],
        'bonds-names': ['CA-CB', 'CB-CG', 'CG-CD', 'CD-OE1', 'CD-NE2'],
        'bonds-types': ['CX-2C', '2C-2C', '2C-C ', 'C -O ', 'C -N '],
        'bonds-vals': [1.526, 1.526, 1.522, 1.229, 1.335],
        'torsion-names': [
            'C-N-CA-CB', 'N-CA-CB-CG', 'CA-CB-CG-CD', 'CB-CG-CD-OE1', 'CB-CG-CD-NE2'
        ],
        'torsion-types': [
            'C -N -CX-2C', 'N -CX-2C-2C', 'CX-2C-2C-C ', '2C-2C-C -O ', '2C-2C-C -N '
        ],
        'torsion-vals': ['p', 'p', 'p', 'p', 'i']
    },
    'E': {
        'angles-names': ['N-CA-CB', 'CA-CB-CG', 'CB-CG-CD', 'CG-CD-OE1', 'CG-CD-OE2'],
        'angles-types': ['N -CX-2C', 'CX-2C-2C', '2C-2C-CO', '2C-CO-O2', '2C-CO-O2'],
        'angles-vals': [
            1.9146261894377796, 1.911135530933791, 1.9390607989657, 2.0420352248333655,
            2.0420352248333655
        ],
        'atom-names': ['CB', 'CG', 'CD', 'OE1', 'OE2'],
        'bonds-names': ['CA-CB', 'CB-CG', 'CG-CD', 'CD-OE1', 'CD-OE2'],
        'bonds-types': ['CX-2C', '2C-2C', '2C-CO', 'CO-O2', 'CO-O2'],
        'bonds-vals': [1.526, 1.526, 1.522, 1.25, 1.25],
        'torsion-names': [
            'C-N-CA-CB', 'N-CA-CB-CG', 'CA-CB-CG-CD', 'CB-CG-CD-OE1', 'CB-CG-CD-OE2'
        ],
        'torsion-types': [
            'C -N -CX-2C', 'N -CX-2C-2C', 'CX-2C-2C-CO', '2C-2C-CO-O2', '2C-2C-CO-O2'
        ],
        'torsion-vals': ['p', 'p', 'p', 'p', 'i']
    },
    'G': {
        'angles-names': [],
        'angles-types': [],
        'angles-vals': [],
        'atom-names': [],
        'bonds-names': [],
        'bonds-types': [],
        'bonds-vals': [],
        'torsion-names': [],
        'torsion-types': [],
        'torsion-vals': []
    },
    'H': {
        'angles-names': [
            'N-CA-CB', 'CA-CB-CG', 'CB-CG-ND1', 'CG-ND1-CE1', 'ND1-CE1-NE2', 'CE1-NE2-CD2'
        ],
        'angles-types': [
            'N -CX-CT', 'CX-CT-CC', 'CT-CC-NA', 'CC-NA-CR', 'NA-CR-NB', 'CR-NB-CV'
        ],
        'angles-vals': [
            1.9146261894377796, 1.9739673840055867, 2.0943951023931953,
            1.8849555921538759, 1.8849555921538759, 1.8849555921538759
        ],
        'atom-names': ['CB', 'CG', 'ND1', 'CE1', 'NE2', 'CD2'],
        'bonds-names': ['CA-CB', 'CB-CG', 'CG-ND1', 'ND1-CE1', 'CE1-NE2', 'NE2-CD2'],
        'bonds-types': ['CX-CT', 'CT-CC', 'CC-NA', 'NA-CR', 'CR-NB', 'NB-CV'],
        'bonds-vals': [1.526, 1.504, 1.385, 1.343, 1.335, 1.394],
        'torsion-names': [
            'C-N-CA-CB', 'N-CA-CB-CG', 'CA-CB-CG-ND1', 'CB-CG-ND1-CE1', 'CG-ND1-CE1-NE2',
            'ND1-CE1-NE2-CD2'
        ],
        'torsion-types': [
            'C -N -CX-CT', 'N -CX-CT-CC', 'CX-CT-CC-NA', 'CT-CC-NA-CR', 'CC-NA-CR-NB',
            'NA-CR-NB-CV'
        ],
        'torsion-vals': ['p', 'p', 'p', 3.141592653589793, 0.0, 0.0]
    },
    'I': {
        'angles-names': ['N-CA-CB', 'CA-CB-CG1', 'CB-CG1-CD1', 'CA-CB-CG2'],
        'angles-types': ['N -CX-3C', 'CX-3C-2C', '3C-2C-CT', 'CX-3C-CT'],
        'angles-vals': [
            1.9146261894377796, 1.911135530933791, 1.911135530933791, 1.911135530933791
        ],
        'atom-names': ['CB', 'CG1', 'CD1', 'CG2'],
        'bonds-names': ['CA-CB', 'CB-CG1', 'CG1-CD1', 'CB-CG2'],
        'bonds-types': ['CX-3C', '3C-2C', '2C-CT', '3C-CT'],
        'bonds-vals': [1.526, 1.526, 1.526, 1.526],
        'torsion-names': ['C-N-CA-CB', 'N-CA-CB-CG1', 'CA-CB-CG1-CD1', 'N-CA-CB-CG2'],
        'torsion-types': ['C -N -CX-3C', 'N -CX-3C-2C', 'CX-3C-2C-CT', 'N -CX-3C-CT'],
        'torsion-vals': ['p', 'p', 'p', 'p']
    },
    'L': {
        'angles-names': ['N-CA-CB', 'CA-CB-CG', 'CB-CG-CD1', 'CB-CG-CD2'],
        'angles-types': ['N -CX-2C', 'CX-2C-3C', '2C-3C-CT', '2C-3C-CT'],
        'angles-vals': [
            1.9146261894377796, 1.911135530933791, 1.911135530933791, 1.911135530933791
        ],
        'atom-names': ['CB', 'CG', 'CD1', 'CD2'],
        'bonds-names': ['CA-CB', 'CB-CG', 'CG-CD1', 'CG-CD2'],
        'bonds-types': ['CX-2C', '2C-3C', '3C-CT', '3C-CT'],
        'bonds-vals': [1.526, 1.526, 1.526, 1.526],
        'torsion-names': ['C-N-CA-CB', 'N-CA-CB-CG', 'CA-CB-CG-CD1', 'CA-CB-CG-CD2'],
        'torsion-types': ['C -N -CX-2C', 'N -CX-2C-3C', 'CX-2C-3C-CT', 'CX-2C-3C-CT'],
        'torsion-vals': ['p', 'p', 'p', 'p']
    },
    'K': {
        'angles-names': ['N-CA-CB', 'CA-CB-CG', 'CB-CG-CD', 'CG-CD-CE', 'CD-CE-NZ'],
        'angles-types': ['N -CX-C8', 'CX-C8-C8', 'C8-C8-C8', 'C8-C8-C8', 'C8-C8-N3'],
        'angles-vals': [
            1.9146261894377796, 1.911135530933791, 1.911135530933791, 1.911135530933791,
            1.9408061282176945
        ],
        'atom-names': ['CB', 'CG', 'CD', 'CE', 'NZ'],
        'bonds-names': ['CA-CB', 'CB-CG', 'CG-CD', 'CD-CE', 'CE-NZ'],
        'bonds-types': ['CX-C8', 'C8-C8', 'C8-C8', 'C8-C8', 'C8-N3'],
        'bonds-vals': [1.526, 1.526, 1.526, 1.526, 1.471],
        'torsion-names': [
            'C-N-CA-CB', 'N-CA-CB-CG', 'CA-CB-CG-CD', 'CB-CG-CD-CE', 'CG-CD-CE-NZ'
        ],
        'torsion-types': [
            'C -N -CX-C8', 'N -CX-C8-C8', 'CX-C8-C8-C8', 'C8-C8-C8-C8', 'C8-C8-C8-N3'
        ],
        'torsion-vals': ['p', 'p', 'p', 'p', 'p']
    },
    'M': {
        'angles-names': ['N-CA-CB', 'CA-CB-CG', 'CB-CG-SD', 'CG-SD-CE'],
        'angles-types': ['N -CX-2C', 'CX-2C-2C', '2C-2C-S ', '2C-S -CT'],
        'angles-vals': [
            1.9146261894377796, 1.911135530933791, 2.0018926520374962, 1.726130630222392
        ],
        'atom-names': ['CB', 'CG', 'SD', 'CE'],
        'bonds-names': ['CA-CB', 'CB-CG', 'CG-SD', 'SD-CE'],
        'bonds-types': ['CX-2C', '2C-2C', '2C-S ', 'S -CT'],
        'bonds-vals': [1.526, 1.526, 1.81, 1.81],
        'torsion-names': ['C-N-CA-CB', 'N-CA-CB-CG', 'CA-CB-CG-SD', 'CB-CG-SD-CE'],
        'torsion-types': ['C -N -CX-2C', 'N -CX-2C-2C', 'CX-2C-2C-S ', '2C-2C-S -CT'],
        'torsion-vals': ['p', 'p', 'p', 'p']
    },
    'F': {
        'angles-names': [
            'N-CA-CB', 'CA-CB-CG', 'CB-CG-CD1', 'CG-CD1-CE1', 'CD1-CE1-CZ', 'CE1-CZ-CE2',
            'CZ-CE2-CD2'
        ],
        'angles-types': [
            'N -CX-CT', 'CX-CT-CA', 'CT-CA-CA', 'CA-CA-CA', 'CA-CA-CA', 'CA-CA-CA',
            'CA-CA-CA'
        ],
        'angles-vals': [
            1.9146261894377796, 1.9896753472735358, 2.0943951023931953,
            2.0943951023931953, 2.0943951023931953, 2.0943951023931953, 2.0943951023931953
        ],
        'atom-names': ['CB', 'CG', 'CD1', 'CE1', 'CZ', 'CE2', 'CD2'],
        'bonds-names': [
            'CA-CB', 'CB-CG', 'CG-CD1', 'CD1-CE1', 'CE1-CZ', 'CZ-CE2', 'CE2-CD2'
        ],
        'bonds-types': ['CX-CT', 'CT-CA', 'CA-CA', 'CA-CA', 'CA-CA', 'CA-CA', 'CA-CA'],
        'bonds-vals': [1.526, 1.51, 1.4, 1.4, 1.4, 1.4, 1.4],
        'torsion-names': [
            'C-N-CA-CB', 'N-CA-CB-CG', 'CA-CB-CG-CD1', 'CB-CG-CD1-CE1', 'CG-CD1-CE1-CZ',
            'CD1-CE1-CZ-CE2', 'CE1-CZ-CE2-CD2'
        ],
        'torsion-types': [
            'C -N -CX-CT', 'N -CX-CT-CA', 'CX-CT-CA-CA', 'CT-CA-CA-CA', 'CA-CA-CA-CA',
            'CA-CA-CA-CA', 'CA-CA-CA-CA'
        ],
        'torsion-vals': ['p', 'p', 'p', 3.141592653589793, 0.0, 0.0, 0.0]
    },
    'P': {
        'angles-names': ['N-CA-CB', 'CA-CB-CG', 'CB-CG-CD'],
        'angles-types': ['N -CX-CT', 'CX-CT-CT', 'CT-CT-CT'],
        'angles-vals': [1.9146261894377796, 1.911135530933791, 1.911135530933791],
        'atom-names': ['CB', 'CG', 'CD'],
        'bonds-names': ['CA-CB', 'CB-CG', 'CG-CD'],
        'bonds-types': ['CX-CT', 'CT-CT', 'CT-CT'],
        'bonds-vals': [1.526, 1.526, 1.526],
        'torsion-names': ['C-N-CA-CB', 'N-CA-CB-CG', 'CA-CB-CG-CD'],
        'torsion-types': ['C -N -CX-CT', 'N -CX-CT-CT', 'CX-CT-CT-CT'],
        'torsion-vals': ['p', 'p', 'p']
    },
    'S': {
        'angles-names': ['N-CA-CB', 'CA-CB-OG'],
        'angles-types': ['N -CX-2C', 'CX-2C-OH'],
        'angles-vals': [1.9146261894377796, 1.911135530933791],
        'atom-names': ['CB', 'OG'],
        'bonds-names': ['CA-CB', 'CB-OG'],
        'bonds-types': ['CX-2C', '2C-OH'],
        'bonds-vals': [1.526, 1.41],
        'torsion-names': ['C-N-CA-CB', 'N-CA-CB-OG'],
        'torsion-types': ['C -N -CX-2C', 'N -CX-2C-OH'],
        'torsion-vals': ['p', 'p']
    },
    'T': {
        'angles-names': ['N-CA-CB', 'CA-CB-OG1', 'CA-CB-CG2'],
        'angles-types': ['N -CX-3C', 'CX-3C-OH', 'CX-3C-CT'],
        'angles-vals': [1.9146261894377796, 1.911135530933791, 1.911135530933791],
        'atom-names': ['CB', 'OG1', 'CG2'],
        'bonds-names': ['CA-CB', 'CB-OG1', 'CB-CG2'],
        'bonds-types': ['CX-3C', '3C-OH', '3C-CT'],
        'bonds-vals': [1.526, 1.41, 1.526],
        'torsion-names': ['C-N-CA-CB', 'N-CA-CB-OG1', 'N-CA-CB-CG2'],
        'torsion-types': ['C -N -CX-3C', 'N -CX-3C-OH', 'N -CX-3C-CT'],
        'torsion-vals': ['p', 'p', 'p']
    },
    'W': {
        'angles-names': [
            'N-CA-CB', 'CA-CB-CG', 'CB-CG-CD1', 'CG-CD1-NE1', 'CD1-NE1-CE2',
            'NE1-CE2-CZ2', 'CE2-CZ2-CH2', 'CZ2-CH2-CZ3', 'CH2-CZ3-CE3', 'CZ3-CE3-CD2'
        ],
        'angles-types': [
            'N -CX-CT', 'CX-CT-C*', 'CT-C*-CW', 'C*-CW-NA', 'CW-NA-CN', 'NA-CN-CA',
            'CN-CA-CA', 'CA-CA-CA', 'CA-CA-CA', 'CA-CA-CB'
        ],
        'angles-vals': [
            1.9146261894377796, 2.0176006153054447, 2.181661564992912, 1.8971728969178363,
            1.9477874452256716, 2.3177972466484698, 2.0943951023931953,
            2.0943951023931953, 2.0943951023931953, 2.0943951023931953
        ],
        'atom-names': [
            'CB', 'CG', 'CD1', 'NE1', 'CE2', 'CZ2', 'CH2', 'CZ3', 'CE3', 'CD2'
        ],
        'bonds-names': [
            'CA-CB', 'CB-CG', 'CG-CD1', 'CD1-NE1', 'NE1-CE2', 'CE2-CZ2', 'CZ2-CH2',
            'CH2-CZ3', 'CZ3-CE3', 'CE3-CD2'
        ],
        'bonds-types': [
            'CX-CT', 'CT-C*', 'C*-CW', 'CW-NA', 'NA-CN', 'CN-CA', 'CA-CA', 'CA-CA',
            'CA-CA', 'CA-CB'
        ],
        'bonds-vals': [1.526, 1.495, 1.352, 1.381, 1.38, 1.4, 1.4, 1.4, 1.4, 1.404],
        'torsion-names': [
            'C-N-CA-CB', 'N-CA-CB-CG', 'CA-CB-CG-CD1', 'CB-CG-CD1-NE1', 'CG-CD1-NE1-CE2',
            'CD1-NE1-CE2-CZ2', 'NE1-CE2-CZ2-CH2', 'CE2-CZ2-CH2-CZ3', 'CZ2-CH2-CZ3-CE3',
            'CH2-CZ3-CE3-CD2'
        ],
        'torsion-types': [
            'C -N -CX-CT', 'N -CX-CT-C*', 'CX-CT-C*-CW', 'CT-C*-CW-NA', 'C*-CW-NA-CN',
            'CW-NA-CN-CA', 'NA-CN-CA-CA', 'CN-CA-CA-CA', 'CA-CA-CA-CA', 'CA-CA-CA-CB'
        ],
        'torsion-vals': [
            'p', 'p', 'p', 3.141592653589793, 0.0, 3.141592653589793, 3.141592653589793,
            0.0, 0.0, 0.0
        ]
    },
    'Y': {
        'angles-names': [
            'N-CA-CB', 'CA-CB-CG', 'CB-CG-CD1', 'CG-CD1-CE1', 'CD1-CE1-CZ', 'CE1-CZ-OH',
            'CE1-CZ-CE2', 'CZ-CE2-CD2'
        ],
        'angles-types': [
            'N -CX-CT', 'CX-CT-CA', 'CT-CA-CA', 'CA-CA-CA', 'CA-CA-C ', 'CA-C -OH',
            'CA-C -CA', 'C -CA-CA'
        ],
        'angles-vals': [
            1.9146261894377796, 1.9896753472735358, 2.0943951023931953,
            2.0943951023931953, 2.0943951023931953, 2.0943951023931953,
            2.0943951023931953, 2.0943951023931953
        ],
        'atom-names': ['CB', 'CG', 'CD1', 'CE1', 'CZ', 'OH', 'CE2', 'CD2'],
        'bonds-names': [
            'CA-CB', 'CB-CG', 'CG-CD1', 'CD1-CE1', 'CE1-CZ', 'CZ-OH', 'CZ-CE2', 'CE2-CD2'
        ],
        'bonds-types': [
            'CX-CT', 'CT-CA', 'CA-CA', 'CA-CA', 'CA-C ', 'C -OH', 'C -CA', 'CA-CA'
        ],
        'bonds-vals': [1.526, 1.51, 1.4, 1.4, 1.409, 1.364, 1.409, 1.4],
        'torsion-names': [
            'C-N-CA-CB', 'N-CA-CB-CG', 'CA-CB-CG-CD1', 'CB-CG-CD1-CE1', 'CG-CD1-CE1-CZ',
            'CD1-CE1-CZ-OH', 'CD1-CE1-CZ-CE2', 'CE1-CZ-CE2-CD2'
        ],
        'torsion-types': [
            'C -N -CX-CT', 'N -CX-CT-CA', 'CX-CT-CA-CA', 'CT-CA-CA-CA', 'CA-CA-CA-C ',
            'CA-CA-C -OH', 'CA-CA-C -CA', 'CA-C -CA-CA'
        ],
        'torsion-vals': [
            'p', 'p', 'p', 3.141592653589793, 0.0, 3.141592653589793, 0.0, 0.0
        ]
    },
    'V': {
        'angles-names': ['N-CA-CB', 'CA-CB-CG1', 'CA-CB-CG2'],
        'angles-types': ['N -CX-3C', 'CX-3C-CT', 'CX-3C-CT'],
        'angles-vals': [1.9146261894377796, 1.911135530933791, 1.911135530933791],
        'atom-names': ['CB', 'CG1', 'CG2'],
        'bonds-names': ['CA-CB', 'CB-CG1', 'CB-CG2'],
        'bonds-types': ['CX-3C', '3C-CT', '3C-CT'],
        'bonds-vals': [1.526, 1.526, 1.526],
        'torsion-names': ['C-N-CA-CB', 'N-CA-CB-CG1', 'N-CA-CB-CG2'],
        'torsion-types': ['C -N -CX-3C', 'N -CX-3C-CT', 'N -CX-3C-CT'],
        'torsion-vals': ['p', 'p', 'p']
    },
    '_': {
        'angles-names': [],
        'angles-types': [],
        'angles-vals': [],
        'atom-names': [],
        'bonds-names': [],
        'bonds-types': [],
        'bonds-vals': [],
        'torsion-names': [],
        'torsion-types': [],
        'torsion-vals': []
    }
}

BB_BUILD_INFO = {
    "BONDLENS": {
        # the updated is according to crystal data from 1DPE_1_A and validated with other structures
        # the commented is the sidechainnet one
        'n-ca': 1.4664931, # 1.442, 
        'ca-c': 1.524119,  # 1.498,
        'c-n': 1.3289373,  # 1.379,
        'c-o': 1.229,  # From parm10.dat || huge variability according to structures
        # we get 1.3389416 from 1DPE_1_A but also 1.2289 from 2F2H_d2f2hf1
        'c-oh': 1.364
    },  # From parm10.dat, for OXT
    # For placing oxygens
    "BONDANGS": {
        'ca-c-o': 2.0944,  # Approximated to be 2pi / 3; parm10.dat says 2.0350539
        'ca-c-oh': 2.0944
    },  # Equal to 'ca-c-o', for OXT
    "BONDTORSIONS": {
        'n-ca-c-n': -0.785398163
    }  # A simple approximation, not meant to be exact.
}

#################
##### DOERS #####
#################

def make_cloud_mask(aa):
    """ relevent points will be 1. paddings will be 0. """
    mask = np.zeros(14)
    if aa != "_":
        n_atoms = 4+len( SC_BUILD_INFO[aa]["atom-names"] )
        mask[:n_atoms] = 1
    return mask

def make_bond_mask(aa):
    """ Gives the length of the bond originating each atom. """
    mask = np.zeros(14)
    # backbone
    mask[0] = BB_BUILD_INFO["BONDLENS"]['c-n']
    mask[1] = BB_BUILD_INFO["BONDLENS"]['n-ca']
    mask[2] = BB_BUILD_INFO["BONDLENS"]['ca-c']
    mask[3] = BB_BUILD_INFO["BONDLENS"]['c-o']
    # sidechain - except padding token 
    if aa in SC_BUILD_INFO.keys():
        for i,bond in enumerate(SC_BUILD_INFO[aa]['bonds-vals']):
            mask[4+i] = bond
    return mask

def make_theta_mask(aa):
    """ Gives the theta of the bond originating each atom. """
    mask = np.zeros(14)
    # backbone
    #
    # sidechain
    for i,theta in enumerate(SC_BUILD_INFO[aa]['angles-vals']):
        mask[4+i] = theta
    return mask

def make_torsion_mask(aa):
    """ Gives the dihedral of the bond originating each atom. """
    mask = np.zeros(14)
    # backbone
    #
    # sidechain
    for i, torsion in enumerate(SC_BUILD_INFO[aa]['torsion-vals']):
        if torsion == 'p':
            mask[4+i] = np.nan 
        elif torsion == "i":
            # https://github.com/jonathanking/sidechainnet/blob/master/sidechainnet/structure/StructureBuilder.py#L372
            mask[4+i] =  999  # anotate to change later # mask[4+i-1] - np.pi
        else:
            mask[4+i] = torsion
    return mask

def make_idx_mask(aa):
    """ Gives the idxs of the 3 previous points. """
    mask = np.zeros((11, 3))
    # backbone
    mask[0, :] = np.arange(3) 
    # sidechain
    mapper = {"N": 0, "CA": 1, "C":2,  "CB": 4}
    for i, torsion in enumerate(SC_BUILD_INFO[aa]['torsion-names']):
        # get all the atoms forming the dihedral
        torsions = [x.rstrip(" ") for x in torsion.split("-")]
        # for each atom
        for n, torsion in enumerate(torsions[:-1]):
            # get the index of the atom in the coords array
            loc = mapper[torsion] if torsion in mapper.keys() else 4 + SC_BUILD_INFO[aa]['atom-names'].index(torsion)
            # set position to index
            mask[i+1][n] = loc
    return mask


###################
##### GETTERS #####
###################
SUPREME_INFO = {k: {"cloud_mask": make_cloud_mask(k),
                    "bond_mask": make_bond_mask(k),
                    "theta_mask": make_theta_mask(k),
                    "torsion_mask": make_torsion_mask(k),
                    "idx_mask": make_idx_mask(k),
                    } 
                for k in "ARNDCQEGHILKMFPSTWYV_"}

# @jit()
def scn_cloud_mask(seq, coords=None):
    """ Gets the boolean mask atom positions (not all aas have same atoms). 
        Inputs: 
        * seqs: (length) iterable of 1-letter aa codes of a protein
        * coords: optional .(batch, lc, 3). sidechainnet coords.
                  returns the true mask (solves potential atoms that might not be provided)
        Outputs: (length, 14) boolean mask 
    """ 
    if coords is not None:
        return  (( rearrange(coords, '... (l c) d -> ... l c d', c=14) == 0 ).sum(dim=-1)  < coords.shape[-1]).float().cpu()
    return torch.tensor([SUPREME_INFO[aa]['cloud_mask'] for aa in seq])


def scn_bond_mask(seq):
    """ Inputs: 
        * seqs: (length). iterable of 1-letter aa codes of a protein
        Outputs: (L, 14) maps point to bond length
    """ 
    return torch.tensor([SUPREME_INFO[aa]['bond_mask'] for aa in seq])


def scn_angle_mask(seq, angles):
    """ Inputs: 
        * seq: (length). iterable of 1-letter aa codes of a protein
        * angles: (length, 12). [phi, psi, omega, b_angle(n_ca_c), b_angle(ca_c_n), b_angle(c_n_ca), 6_scn_torsions]
        Outputs: (L, 14) maps point to theta and dihedral.
                 first angle is theta, second is dihedral
    """ 
    device, precise = angles.device, angles.type()
    angles = angles
    # get masks
    theta_mask   = torch.tensor([SUPREME_INFO[aa]['theta_mask'] for aa in seq]).type(precise)
    torsion_mask = torch.tensor([SUPREME_INFO[aa]['torsion_mask'] for aa in seq]).type(precise)
    # fill masks with angle values
    theta_mask[:, 0] = angles[:, 4] # ca_c_n
    theta_mask[1:, 1] = angles[:-1, 5] # c_n_ca
    theta_mask[:, 2] = angles[:, 3] # n_ca_c
    theta_mask[:, 3] = BB_BUILD_INFO["BONDANGS"]["ca-c-o"]
    # backbone_torsions
    torsion_mask[:, 0] = angles[:, 1] # n determined by psi of previous
    torsion_mask[1:, 1] = angles[:-1, 2] # ca determined by omega of previous
    torsion_mask[:, 2] = angles[:, 0] # c determined by phi
    # O placement - same as in sidechainnet
    # https://github.com/jonathanking/sidechainnet/blob/master/sidechainnet/structure/StructureBuilder.py#L313der.py#L313
    torsion_mask[:, 3] = angles[:, 1] - np.pi 
    torsion_mask[-1, 3] += np.pi              
    # add torsions to sidechains
    to_fill = torsion_mask != torsion_mask # "p" fill with passed values
    to_pick = torsion_mask == 999          # "i" infer from previous one
    for i in range(len(seq)):
        # check if any is nan -> fill the holes
        number = to_fill[i].long().sum()
        torsion_mask[i, to_fill[i]] = angles[i, 6:6+number]

        # pick previous value for inferred torsions
        for j, val in enumerate(to_pick[i]):
            if val:
                torsion_mask[i, j] = torsion_mask[i, j-1] - np.pi # pick values from last one.

    return torch.stack([theta_mask, torsion_mask], dim=0).to(device)


def scn_index_mask(seq):
    """ Inputs: 
        * seq: (length). iterable of 1-letter aa codes of a protein
        Outputs: (L, 11, 3) maps point to theta and dihedral.
                 first angle is theta, second is dihedral
    """ 
    idxs = torch.tensor([SUPREME_INFO[aa]['idx_mask'] for aa in seq])
    return rearrange(idxs, 'l s d -> d l s')


def build_scaffolds_from_scn_angles(seq, angles, coords=None, device="auto"):
    """ Builds scaffolds for fast access to data
        Inputs: 
        * seq: string of aas (1 letter code)
        * angles: (L, 12) tensor containing the internal angles.
                  Distributed as follows (following sidechainnet convention):
                  * (L, 3) for torsion angles
                  * (L, 3) bond angles
                  * (L, 6) sidechain angles
        * coords: (L, 3) sidechainnet coords. builds the mask with those instead
                  (better accuracy if modified residues present).
        Outputs:
        * cloud_mask: (L, 14 ) mask of points that should be converted to coords 
        * point_ref_mask: (3, L, 11) maps point (except n-ca-c) to idxs of
                                     previous 3 points in the coords array
        * angles_mask: (2, L, 14) maps point to theta and dihedral
        * bond_mask: (L, 14) gives the length of the bond originating that atom
    """
    # auto infer device and precision
    precise = angles.type()
    if device == "auto":
        device = angles.device

    if coords is not None: 
        cloud_mask = scn_cloud_mask(seq, coords=coords)
    else: 
        cloud_mask = scn_cloud_mask(seq)

    cloud_mask = torch.tensor(cloud_mask).bool().to(device)
    
    point_ref_mask = torch.tensor(scn_index_mask(seq)).long().to(device)
     
    angles_mask = torch.tensor(scn_angle_mask(seq, angles)).type(precise).to(device)
     
    bond_mask = torch.tensor(scn_bond_mask(seq)).type(precise).to(device)
    # return all in a dict
    return {"cloud_mask":     cloud_mask, 
            "point_ref_mask": point_ref_mask,
            "angles_mask":    angles_mask,
            "bond_mask":      bond_mask }


#############################
####### ENCODERS ############
#############################


def modify_scaffolds_with_coords(scaffolds, coords):
    """ Gets scaffolds and fills in the right data.
        Inputs: 
        * scaffolds: dict. as returned by `build_scaffolds_from_scn_angles`
        * coords: (L, 14, 3). sidechainnet tensor. same device as scaffolds
        Outputs: corrected scaffolds
    """


    # calculate distances and update: 
    # N, CA, C
    scaffolds["bond_mask"][1:, 0] = torch.norm(coords[1:, 0] - coords[:-1, 2], dim=-1) # N
    scaffolds["bond_mask"][ :, 1] = torch.norm(coords[ :, 1] - coords[:  , 0], dim=-1) # CA
    scaffolds["bond_mask"][ :, 2] = torch.norm(coords[ :, 2] - coords[:  , 1], dim=-1) # C
    # O, CB, side chain
    selector = np.arange(len(coords))
    for i in range(3, 14):
        # get indexes
        idx_a, idx_b, idx_c = scaffolds["point_ref_mask"][:, :, i-3] # (3, L, 11) -> 3 * (L, 11)
        # correct distances
        scaffolds["bond_mask"][:, i] = torch.norm(coords[:, i] - coords[selector, idx_c], dim=-1)
        # get angles
        scaffolds["angles_mask"][0, :, i] = get_angle(coords[selector, idx_b], 
                                                      coords[selector, idx_c], 
                                                      coords[:, i])
        # handle C-beta, where the C requested is from the previous aa
        if i == 4:
            # for 1st residue, use position of the second residue's N
            first_next_n     = coords[1, :1] # 1, 3
            # the c requested is from the previous residue
            main_c_prev_idxs = coords[selector[:-1], idx_a[1:]]# (L-1), 3
            # concat
            coords_a = torch.cat([first_next_n, main_c_prev_idxs])
        else:
            coords_a = coords[selector, idx_a]
        # get dihedrals
        scaffolds["angles_mask"][1, :, i] = get_dihedral(coords_a,
                                                         coords[selector, idx_b], 
                                                         coords[selector, idx_c], 
                                                         coords[:, i])
    # correct angles and dihedrals for backbone 
    scaffolds["angles_mask"][0, :-1, 0] = get_angle(coords[:-1, 1], coords[:-1, 2], coords[1: , 0]) # ca_c_n
    scaffolds["angles_mask"][0, 1:,  1] = get_angle(coords[:-1, 2], coords[1:,  0], coords[1: , 1]) # c_n_ca
    scaffolds["angles_mask"][0,  :,  2] = get_angle(coords[:,   0], coords[ :,  1], coords[ : , 2]) # n_ca_c
    
    # N determined by previous psi = f(n, ca, c, n+1)
    scaffolds["angles_mask"][1, :-1, 0] = get_dihedral(coords[:-1, 0], coords[:-1, 1], coords[:-1, 2], coords[1:, 0])
    # CA determined by omega = f(ca, c, n+1, ca+1)
    scaffolds["angles_mask"][1,  1:, 1] = get_dihedral(coords[:-1, 1], coords[:-1, 2], coords[1:, 0], coords[1:, 1])
    # C determined by phi = f(c-1, n, ca, c)
    scaffolds["angles_mask"][1,  1:, 2] = get_dihedral(coords[:-1, 2], coords[1:, 0], coords[1:, 1], coords[1:, 2])

    return scaffolds



if __name__ == "__main__":
    print(scn_cloud_mask("AAAA"))
