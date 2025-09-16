"""
    Codebase adapted from adapted from https://github.com/mahmoodlab/MMP/blob/main/src/mil_models/PANTHER/layers.py
"""

import torch
import torch.nn as nn
import torch.nn.init
from .network import DirNIWNet


class PANTHERBase(nn.Module):
    """
    Args:
    - p (int): Number of prototypes
    - d (int): Feature dimension
    - L (int): Number of EM iterations
    - out (str): Ways to merge features
    - ot_eps (float): eps
    """
    def __init__(self, d, prototypes, p=5, L=3, tau=10.0, ot_eps=0.1, fix_proto=True):
        super(PANTHERBase, self).__init__()

        self.L = L
        self.tau = tau

        self.priors = DirNIWNet(p, d, prototypes, ot_eps, fix_proto)
        self.outdim = p + 2*p*d

    def forward(self, S, mask=None):
        """
        Args
        - S: data
        """
        B, N_max, d = S.shape
        
        if mask is None:
            mask = torch.ones(B, N_max).to(S)
        
        pis, mus, Sigmas, qqs = [], [], [], []
        pi, mu, Sigma, qq = self.priors.map_em(S, 
                                                    mask=mask, 
                                                    num_iters=self.L, 
                                                    tau=self.tau, 
                                                    prior=self.priors())

        pis.append(pi)
        mus.append(mu)
        Sigmas.append(Sigma)
        qqs.append(qq)

        pis = torch.stack(pis, dim=2) # pis: (n_batch x n_proto x n_head)
        mus = torch.stack(mus, dim=3) # mus: (n_batch x n_proto x embed_dim x n_head)
        Sigmas = torch.stack(Sigmas, dim=3) # Sigmas: (n_batch x n_proto x embed_dim x n_head)
        qqs = torch.stack(qqs, dim=3)
            
        out = torch.cat([pis.reshape(B,-1), mus.reshape(B,-1), Sigmas.reshape(B,-1)], dim=1)
        return out, qqs