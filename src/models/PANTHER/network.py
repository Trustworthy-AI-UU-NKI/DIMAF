"""
    Codebase adapted from https://github.com/mahmoodlab/MMP/blob/main/src/mil_models/PANTHER/networks.py
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def mog_eval(mog, data):
    """
    This evaluates the log-likelihood of mixture of Gaussians
    """    
    B, N, d = data.shape    
    pi, mu, Sigma = mog    
    if len(pi.shape)==1:
        pi = pi.unsqueeze(0).repeat(B,1)
        mu = mu.unsqueeze(0).repeat(B,1,1)
        Sigma = Sigma.unsqueeze(0).repeat(B,1,1)
    
    # compute the log(prior * N(data; mean, cov))
    jll = -0.5 * ( d * np.log(2*np.pi) + 
        Sigma.log().sum(-1).unsqueeze(1) +
        torch.bmm(data**2, 1./Sigma.permute(0,2,1)) + 
        ((mu**2) / Sigma).sum(-1).unsqueeze(1) + 
        -2. * torch.bmm(data, (mu/Sigma).permute(0,2,1))
    ) + pi.log().unsqueeze(1) 
    
    # compute the log(sum(prior * N(data; mean, cov)))
    mll = jll.logsumexp(-1) 
    # compute the log posterior prob
    cll = jll - mll.unsqueeze(-1)
    
    return jll, cll, mll

class DirNIWNet(nn.Module):
    """
    Conjugate prior for the Gaussian mixture model

    Args:
    - p (int): Number of prototypes
    - d (int): Embedding dimension
    - eps (float): initial covariance (similar function to sinkorn entropic regularizer)
    """
    
    def __init__(self, p, d, prototypes, eps=0.1, fix_proto=True):
        """
        self.m: prior mean (p x d)
        self.V_: prior covariance (diagonal) (p x d)
        """
        super(DirNIWNet, self).__init__()

        self.eps = eps
        self.m = nn.Parameter(torch.from_numpy(prototypes), requires_grad=not fix_proto)

        self.V_ = nn.Parameter(np.log(np.exp(1) - 1) * torch.ones((p, d)), requires_grad=not fix_proto)
        # All values are 0.5413

        self.p, self.d = p, d
    
    def forward(self):
        """
        Return prior mean and covariance
        """
        V = self.eps * F.softplus(self.V_)
        # V == filled with 0.1
        return self.m, V
    
    def mode(self, prior=None):
        if prior is None:
            m, V = self.forward()
        else:
            m, V = prior
        pi = torch.ones(self.p).to(m) / self.p
        mu = m
        Sigma = V
        return pi.float(), mu.float(), Sigma.float()

        
    def map_m_step(self, data, weight, tau=1.0, prior=None):
        # Update rules are obtained from Kim, M. "Differentiable Expectation-Maximization for Set Representation Learning ", ICLR, 2022
        B, N, d = data.shape
        
        if prior is None:
            m, V = self.forward()
        else:
            m, V = prior

        wsum = weight.sum(1)
        wsum_reg = wsum + tau 
        wxsum = torch.bmm(weight.permute(0,2,1), data) 
        wxxsum = torch.bmm(weight.permute(0,2,1), data**2) 

        pi = wsum_reg / wsum_reg.sum(1, keepdim=True) 
        mu = (wxsum + m.unsqueeze(0)*tau) / wsum_reg.unsqueeze(-1)
        Sigma = (wxxsum + (V+m**2).unsqueeze(0)*tau) / wsum_reg.unsqueeze(-1) - mu**2

        return pi.float(), mu.float(), Sigma.float()
    
    def map_em(self, data, mask=None, num_iters=3, tau=1.0, prior=None):
        # EM algorithm
        B, N, d = data.shape
        
        if mask is None:
            mask = torch.ones(B, N).to(data)

        # Need to set to the mode for initial starting point
        pi, mu, Sigma = self.mode(prior)
        pi = pi.unsqueeze(0).repeat(B,1)
        mu = mu.unsqueeze(0).repeat(B,1,1)
        Sigma = Sigma.unsqueeze(0).repeat(B,1,1)
        
        for emiter in range(num_iters):
            # E-step: Evaluate the log likelihood of the model given the data. 
            _, qq, _ = mog_eval((pi, mu, Sigma), data)
            # qq = posterior probability
            qq = qq.exp() * mask.unsqueeze(-1)

            # M-step: Update prior prob, mean and covariance
            pi, mu, Sigma = self.map_m_step(data, weight=qq, tau=tau, prior=prior)
            
        return pi, mu, Sigma, qq
    