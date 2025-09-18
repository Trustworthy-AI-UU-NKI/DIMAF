import torch
import torch.nn as nn

from .layers import MultiSNN, CrossAttentionLayer
from survival.losses import NLLSurvLoss, CoxLoss,  DisentangledSurvLoss

class DIMAF(nn.Module):
    """ Main model of Disentangled and Interpretable Multimodal Attention Fusion. """
    def __init__(
            self,
            rna_dims,
            histo_dim,
            device,
            num_classes=1,
            single_out_dim=256,
            loss_fn='cox',
            num_proto_wsi=16):
        """
        Args:
            rna_dims: number of genes in each pathway (list of ints)
            histo_dim: dimension of histology feature vector (int)
            device: torch device (torch.device)
            num_classes: number of output classes (int)
            single_out_dim: output dimension of single modality embedding (int)
            loss_fn: loss function to use (str)
            num_proto_wsi: number of histology prototypes (int)
        """

        super(DIMAF, self).__init__()

        self.device = device

        # Input args
        self.rna_dims = rna_dims
        self.histo_dim = histo_dim

        # Output args
        self.num_classes = num_classes

        # Architecture args
        self.single_out_dim = single_out_dim
        self.nr_wsi_prototypes = num_proto_wsi
        self.nr_rna_prototypes = len(rna_dims)   

        # Loss function     
        self.loss_fn = loss_fn

        # Create the architecture
        self.create_mm_architecture()
    

    def create_mm_architecture(self):
        """ Create the DIMAF architecture. """
        # Histology embeddings
        self.f_h = nn.Sequential(nn.Linear(self.histo_dim, self.single_out_dim))

        # Rna embeddings
        self.f_g = MultiSNN(self.rna_dims, self.single_out_dim)

        # Add type embeddings to each prototype
        self.single_out_dim, self.wsi_pt_embedding, self.rna_pt_embedding = self.get_pt_embeddings()

        multi_out_dim = self.single_out_dim // 2

        # 4 seperate attention blocks
        self.rna_attention = CrossAttentionLayer(
                dim=self.single_out_dim,
                dim_head=multi_out_dim,
                heads=1)
        
        self.wsi_attention = CrossAttentionLayer(
                dim=self.single_out_dim,
                dim_head=multi_out_dim,
                heads=1)

        self.cross_attention_rna_wsi = CrossAttentionLayer(
                dim=self.single_out_dim,
                dim_head=multi_out_dim,
                heads=1)

        self.cross_attention_wsi_rna = CrossAttentionLayer(
                dim=self.single_out_dim,
                dim_head=multi_out_dim,
                heads=1)
    
        # Layer normalization
        self.layer_norm = nn.LayerNorm(multi_out_dim)

        # Cox PH risk predictor
        out_classifier_dim = 4 * multi_out_dim
        self.f_surv = nn.Linear(out_classifier_dim, self.num_classes, bias=False)
        

    def get_pt_embeddings(self):
        """ Create learnable prototype embeddings to append to each prototype. """
  
        # Learnable encoding per prototype
        append_dim = 32
        path_proj_dim_new = self.single_out_dim + append_dim

        histo_embedding = torch.nn.Parameter(torch.randn(1, self.nr_wsi_prototypes, append_dim), requires_grad=True)
        gene_embedding = torch.nn.Parameter(torch.randn(1, self.nr_rna_prototypes, append_dim), requires_grad=True)

        return path_proj_dim_new, histo_embedding, gene_embedding


    def append_pt_embeddings(self, wsi_embed, rna_embed):
        """ Append the learnable prototype embeddings to each feature. """
        bs = wsi_embed.size(0)
        # RNA
        rna_pt_embedding_exp = self.rna_pt_embedding.expand(bs, -1, -1)
        z_g = torch.cat([rna_embed, rna_pt_embedding_exp], dim=-1)

        # WSI
        wsi_pt_embedding_exp = self.wsi_pt_embedding.expand(bs, -1, -1)
        z_h = torch.cat([wsi_embed, wsi_pt_embedding_exp], dim=-1)
    
        return z_g, z_h

    def disentangled_attention_fusion(self, z_h, z_g):
        """ Pass through disentangled fusion.  """
        # B, 50, multi_out_dim
        zp_gg = self.rna_attention(z_g, z_g)

        # B, 50, multi_out_dim
        zp_hg = self.cross_attention_wsi_rna(z_g, z_h)

        # B, 16, multi_out_dim
        zp_gh = self.cross_attention_rna_wsi(z_h, z_g)

        # B, 16, multi_out_dim
        zp_hh = self.wsi_attention(z_h, z_h)

        # Aggregate sinlge and multimodal --> concat
        post_attn_tokens = torch.cat([zp_gg, zp_hg, zp_gh, zp_hh], dim=1)
        return post_attn_tokens


    def compute_post_attn_tokens(self, wsi, rna):
        """ Compute multimodal features (Zp_) for SHAP calculation. """
        # wsi embeddings
        wsi_emb = self.f_h(wsi)

        # Rna embeddings
        rna_emb = self.f_g(rna)

        # Append prototype embeddings
        z_g, z_h = self.append_pt_embeddings(wsi_emb, rna_emb)

        # Pass through disentangled fusion 
        post_attn_tokens = self.disentangled_attention_fusion(z_h, z_g)
        return post_attn_tokens

    def forward_shap_post_attn(self, post_attn_tokens):
        """ Forward function for SHAP calculation after fusion. """
        mm_norm_tokens = self.layer_norm(post_attn_tokens)

        # Aggregate the protypes per 'modality'
        rna_mm_tokens = mm_norm_tokens[:, :self.nr_rna_prototypes, :]
        z_gg = torch.mean(rna_mm_tokens, dim=1) # B, dim

        count = self.nr_rna_prototypes
        wsi_rna_mm_tokens = mm_norm_tokens[:, count:count + self.nr_rna_prototypes, :]
        z_hg = torch.mean(wsi_rna_mm_tokens, dim=1) # B, dim

        count = self.nr_rna_prototypes + self.nr_rna_prototypes
        rna_wsi_mm_tokens = mm_norm_tokens[:, count:count+self.nr_wsi_prototypes, :]
        z_gh = torch.mean(rna_wsi_mm_tokens, dim=1) # B, dim

        count = self.nr_rna_prototypes + self.nr_rna_prototypes + self.nr_wsi_prototypes
        wsi_mm_tokens = mm_norm_tokens[:, count:, :]
        z_hh = torch.mean(wsi_mm_tokens, dim=1) # B, dim
        
        # Concatenate disentangled embeddings 
        disentangled_embedding = torch.concat([z_hg, z_gh, z_gg, z_hh], dim=1)

        # Risk prediction
        logits = self.f_surv(disentangled_embedding)

        return logits
    
    def compute_post_attn_tokens_av(self, wsi, rna):
        """ Compute aggregated multimodal features (Z_) for SHAP calculation. """
        post_attn_tokens = self.compute_post_attn_tokens(wsi, rna)

        mm_norm_tokens = self.layer_norm(post_attn_tokens)

        # Aggregate the protypes per 'modality'
        rna_mm_tokens = mm_norm_tokens[:, :self.nr_rna_prototypes, :]
        z_gg = torch.mean(rna_mm_tokens, dim=1) # B, dim

        count = self.nr_rna_prototypes
        wsi_rna_mm_tokens = mm_norm_tokens[:, count:count + self.nr_rna_prototypes, :]
        z_hg = torch.mean(wsi_rna_mm_tokens, dim=1) # B, dim

        count = self.nr_rna_prototypes + self.nr_rna_prototypes
        rna_wsi_mm_tokens = mm_norm_tokens[:, count:count+self.nr_wsi_prototypes, :]
        z_gh = torch.mean(rna_wsi_mm_tokens, dim=1) # B, dim

        count = self.nr_rna_prototypes + self.nr_rna_prototypes + self.nr_wsi_prototypes
        wsi_mm_tokens = mm_norm_tokens[:, count:, :]
        z_hh = torch.mean(wsi_mm_tokens, dim=1) # B, dim
        
        # Concatenate disentangled embeddings 
        disentangled_embedding = torch.concat([z_hg, z_gh, z_gg, z_hh], dim=1)
        return disentangled_embedding


    def forward_mm_no_loss(self, wsi, rna, return_attn):
        # wsi embeddings
        wsi_emb = self.f_h(wsi)

        # Rna embeddings
        rna_emb = self.f_g(rna)

        # Append prototype embedding to the features
        z_g, z_h = self.append_pt_embeddings(wsi_emb, rna_emb)


        # Required for visualization
        if return_attn:
            with torch.no_grad():
                # B, 50, 50
                _, attention_matrix_self_rna = self.rna_attention(z_g, z_g, return_attention=True)
                # B, 16, dim: Pathway --> Histology attention
                _, cross_attn_wsi_rna = self.cross_attention_wsi_rna(z_g, z_h, return_attention=True)
                # B, 16, dim: Histology --> Pathway attention
                _, cross_attn_rna_wsi = self.cross_attention_rna_wsi(z_h, z_g, return_attention=True)
                # B, 16, dim
                _, attention_matrix_self_wsi = self.wsi_attention(z_h, z_h, return_attention=True)
      
        # Pass through disentangled fusion 
        post_attn_tokens = self.disentangled_attention_fusion(z_h, z_g)
        mm_norm_tokens = self.layer_norm(post_attn_tokens)

        # Aggregate the protypes per 'modality'
        rna_mm_tokens = mm_norm_tokens[:, :self.nr_rna_prototypes, :]
        z_gg = torch.mean(rna_mm_tokens, dim=1) # B, dim

        count = self.nr_rna_prototypes
        wsi_rna_mm_tokens = mm_norm_tokens[:, count:count + self.nr_rna_prototypes, :]
        z_hg = torch.mean(wsi_rna_mm_tokens, dim=1) # B, dim

        count = self.nr_rna_prototypes + self.nr_rna_prototypes
        rna_wsi_mm_tokens = mm_norm_tokens[:, count:count+self.nr_wsi_prototypes, :]
        z_gh = torch.mean(rna_wsi_mm_tokens, dim=1) # B, dim

        count = self.nr_rna_prototypes + self.nr_rna_prototypes + self.nr_wsi_prototypes
        wsi_mm_tokens = mm_norm_tokens[:, count:, :]
        z_hh = torch.mean(wsi_mm_tokens, dim=1) # B, dim
        
        # Concatenate disentangled embeddings for prediction
        disentangled_embedding = torch.concat([z_hg, z_gh, z_gg, z_hh], dim=1)

        # Risk prediction
        logits = self.f_surv(disentangled_embedding)
        
        results = {"wsi_rna_repr": z_hg,
                            "rna_wsi_repr": z_gh,
                            "logits": logits,
                            "wsi_repr": z_hh,
                            "rna_repr": z_gg
                            }
        
        if return_attn:
            # Attention matrices
            results['self_attn_rna'] = attention_matrix_self_rna
            results['self_attn_wsi'] = attention_matrix_self_wsi
            results['cross_attn_rna_wsi'] = cross_attn_rna_wsi
            results['cross_attn_wsi_rna'] = cross_attn_wsi_rna

        return results


    def forward(self, wsi, rna, label, censorship, return_attn=False):
        """ Main forward function"""
        # Forward pass
        output = self.forward_mm_no_loss(wsi, rna, return_attn)
        
        # Compute the total loss and update results dict
        output_results, output_log = self.compute_loss(output, label, censorship)
        output.update(output_results)

        return output, output_log
    
    def compute_loss(self, output, label, censorship):
        """Compute the loss given the output of the model."""
        logits = output['logits']
        results_dict = {'logits': logits}

        if isinstance(self.loss_fn, NLLSurvLoss):
            total_loss, log_dict = self.loss_fn(logits=logits, times=label, censorships=censorship)
            hazards = torch.sigmoid(logits)
            survival = torch.cumprod(1 - hazards, dim=1)
            risk = -torch.sum(survival, dim=1).unsqueeze(dim=1)
            results_dict.update({'hazards': hazards,
                                    'survival': survival,
                                    'risk': risk})

        elif isinstance(self.loss_fn, CoxLoss):
            total_loss, log_dict = self.loss_fn(logits=logits, times=label, censorships=censorship)
            risk = torch.exp(logits)
            results_dict['risk'] = risk
        
        elif isinstance(self.loss_fn, DisentangledSurvLoss):
            total_loss, log_dict = self.loss_fn(output=output, times=label, censorships=censorship)
            risk = torch.exp(logits)
            results_dict['risk'] = risk

        results_dict['loss'] = total_loss

        return results_dict, log_dict

    def from_pretrained(self, cp_path):
        # Load weights fro pretrained model
        state_dict = torch.load(cp_path, map_location=self.device)
    
        # Load the weights into the model
        self.load_state_dict(state_dict)