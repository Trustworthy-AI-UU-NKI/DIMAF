import torch
import torch.nn as nn
import sys
from torch.utils.data import DataLoader

from .main_model import DIMAF
from utils.train_utils import list_to_device

class SHAP_DIMAF(nn.Module):
    """SHAP wrapper for DIMAF. """
    def __init__(
            self,
            rna_dims,
            histo_dim,
            device, 
            bs,
            shap_type,
            n_label_bins=4,
            single_out_dim=256,
            loss_fn='cox',
            num_proto_wsi=16,
            num_workers=2):

        super(SHAP_DIMAF, self).__init__()
        
        # Defines for which features we want to compute SHAP values
        self.shap_type = shap_type

        # Define number of classes based on the loss function
        if loss_fn.split("_")[0] == 'cox':
            num_classes = 1
        else:
            num_classes = n_label_bins

        # Data params
        self.batch_size = bs
        self.pathway_sizes = rna_dims

        # Create the survival model
        self.model = DIMAF(rna_dims=rna_dims,
                        histo_dim=histo_dim,
                        device=device,
                        loss_fn=None,
                        single_out_dim=single_out_dim,
                        num_classes=num_classes,
                        num_proto_wsi=num_proto_wsi)
        self.device = device
        self.num_workers = num_workers

        # Put trained model to eval
        self.model.eval()

    def forward_post_attn(self, data):
        """ Forward pass shap after attention fusion. """
        all_logits = []

        # Get the total number of samples
        num_samples = data.size(0)
        bs = self.batch_size

        # Loop through the dataset in batches
        for i in range(0, num_samples, bs):
            # if count > total number of samples, reduce the batch size
            if bs + i > num_samples:
                bs = num_samples-i
            
            # Get the batch
            post_attn_batch = data[i:i+bs] # [B, 132, Dz]

            # Forward pass
            out = self.model.forward_shap_post_attn(post_attn_batch)
            all_logits.append(out)
        
        logits = torch.cat(all_logits, dim=0)
        return logits
    
    def forward_post_attn_av(self, data):
        """ Forward pass shap after prototype aggregation. """
        all_logits = []

        # Get the total number of samples
        num_samples = data.size(0)
        bs = self.batch_size

        # Loop through the dataset in batches
        for i in range(0, num_samples, bs):
            # if count > total number of samples, reduce the batch size
            if bs + i > num_samples:
                bs = num_samples-i

            # Get the batch
            post_attn_av_batch = data[i:i+bs] # [B, 4, Dz]
            reshaped_tensor = post_attn_av_batch.reshape(bs, 4*post_attn_av_batch.shape[-1])

            # Forward pass
            out = self.model.f_surv(reshaped_tensor)
            all_logits.append(out)
        
        logits = torch.cat(all_logits, dim=0)
        return logits

        
    def prep_data_post_attn_av(self, dataset):
        """ Prepares the data for post attention prototype aggregation SHAP. """
        proc_dataset = []
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
        # For each batch in the dataset
        for batch in dataloader:
            batch_wsi = batch['img'].to(self.device)
            batch_rna = list_to_device(batch['rna'], self.device)

            # Compute post attention tokens with prototype aggregation
            post_attn_tokens = self.model.compute_post_attn_tokens_av(batch_wsi, batch_rna)
            proc_dataset.append(post_attn_tokens)
        
        mm_data_tensor = torch.cat(proc_dataset, dim=0)
        return mm_data_tensor
    

    def prep_data_post_attn(self, dataset):
        """ Prepares the data for post attention SHAP. """
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
        proc_dataset = []
        # For each batch in the dataset
        for batch in dataloader:
            batch_wsi = batch['img'].to(self.device)
            batch_rna = list_to_device(batch['rna'], self.device)

            # Compute post attention tokens
            post_attn_tokens = self.model.compute_post_attn_tokens(batch_wsi, batch_rna)
            proc_dataset.append(post_attn_tokens)
        
        mm_data_tensor = torch.cat(proc_dataset, dim=0)
        return mm_data_tensor


    def from_pretrained(self, pretrained_model_path):
        """ Load the pretrained DIMAF model. """
        self.model.from_pretrained(pretrained_model_path)

    def __call__(self, data):
        """ Forward pass depending on the SHAP type. """
        if self.shap_type == 'post_attn':
            risk_score = self.forward_post_attn(data)
        elif self.shap_type == 'post_attn_av':
            risk_score = self.forward_post_attn_av(data)
        else:
            sys.exit("SHAP mode is not implemented, abborting....")

        return risk_score

