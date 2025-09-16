"""
    Codebase adapted from https://github.com/mahmoodlab/MMP/blob/main/src/mil_models/model_PANTHER.py
"""

import torch
from torch import nn

from .layers import PANTHERBase
from tqdm import tqdm
from sksurv.util import Surv


class PANTHER(nn.Module):
    """
    Wrapper for PANTHER model
    """
    def __init__(self, args, prototypes, device):
        super(PANTHER, self).__init__()
        self.emb_dim = args.in_dim
        self.outsize = args.n_proto
        self.prototypes = prototypes
        self.device = device

        # This module contains the EM step
        self.panther = PANTHERBase(self.emb_dim, prototypes, p=self.outsize, L=args.em_iter,
                         tau=args.tau, ot_eps=args.ot_eps, fix_proto=args.fix_proto)

    def representation(self, x):
        """ Construct unsupervised slide representation. """
        out, qqs = self.panther(x)
        # out = slide embeddings (GMM parameters), qqs = posterior probabilities
        return {'repr': out, 'qq': qqs}

    def forward(self, x):
        out = self.representation(x)
        return out['repr']
    
    def create_emb_surv(self, data_loader):
        """ Create slide embeddings to use in the survival prediction model. """
        X = []
        label_output = []
        censor_output = []
        time_output = []
        # We need a BatchSize of 1 for creating the slide summaries!
        dataset = data_loader.dataset
        for i in tqdm(range(len(dataset))):
            batch = dataset.__getitem__(i)
            data, label, censorship, time = batch['img'].unsqueeze(dim=0), batch['label'].unsqueeze(dim=0), batch['censorship'].unsqueeze(dim=0), batch['survival_time'].unsqueeze(dim=0)
            data = data.to(self.device)
        
            with torch.no_grad():
                # Obtain slide embeddings (GMM parameters)
                out = self.representation(data)
                out = out['repr'].data.detach().cpu()

            X.append(out)
            label_output.append(label)
            censor_output.append(censorship)
            time_output.append(time)

        X = torch.cat(X)
        label_output = torch.cat(label_output)
        censor_output = torch.cat(censor_output)
        time_output = torch.cat(time_output)

        # Desired format
        y = Surv.from_arrays(~censor_output.numpy().astype('bool').squeeze(),
                            time_output.numpy().squeeze())


        return X, y

    def predict(self, data_loader):
        output = self.create_emb_surv(data_loader)
        
        return output
    
    