import torch
import torch.nn as nn
from einops import rearrange


class MultiSNN(nn.Module):
    """ Block with multiple SNNs (Ensemble). """
    def __init__(self, in_dims, out_dim):
        super().__init__()
        multi_snn_network = []
        for input_dim in in_dims:
            snn_network_pathway = [SNN_Block(input_dim, out_dim), SNN_Block(out_dim, out_dim)]
            multi_snn_network.append(nn.Sequential(*snn_network_pathway))
        
        self.ensemble_snn = nn.ModuleList(multi_snn_network)
        
    def forward(self, x):
        outputs = []
        for i, net in enumerate(self.ensemble_snn):
            outputs.append(net(x[i]).float())  
        return torch.stack(outputs, dim=1)


class SNN_Block(nn.Module):
    """ Multilayer Reception Block with Self-Normalization (Self Normalizing Network). """
    def __init__(self, in_dim, out_dim, dropout=0.25):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim), 
            nn.ELU(), 
            nn.AlphaDropout(p=dropout))

    def forward(self, x):
        return self.net(x)
    

class CrossAttentionLayer(nn.Module):
    """ Single attention layer in the attention module. """

    def __init__(
            self,
            dim=512,
            dim_head=64,
            heads=1
    ):
        super().__init__()
        self.norm_x = nn.LayerNorm(dim)
        self.norm_y = nn.LayerNorm(dim)
        self.inner_dim = heads * dim_head
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.to_q = nn.Linear(dim, self.inner_dim, bias=False)
        self.to_k = nn.Linear(dim, self.inner_dim, bias=False)
        self.to_v = nn.Linear(dim, self.inner_dim, bias=False)

    
    def forward(self, x, y, return_attention=False):
        x_norm = self.norm_x(x)
        y_norm = self.norm_y(y)

        # derive query, keys, values 
        q = self.to_q(x_norm)
        k = self.to_k(y_norm)
        v = self.to_v(y_norm)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), (q, k, v))
        # regular transformer scaling
        q = q * self.scale

        einops_eq = '... i d, ... j d -> ... i j'
        pre_soft_attn_matrix = torch.einsum(einops_eq, q, k)

        attn_matrix = pre_soft_attn_matrix.softmax(dim=-1)

        out  = attn_matrix @ v

        # merge and combine heads
        out = rearrange(out, 'b h n d -> b n (h d)', h=self.heads)

        if return_attention:
            # Also return the attention weights
            return out, attn_matrix.squeeze().detach().cpu()
    
        return out
    

    




                








