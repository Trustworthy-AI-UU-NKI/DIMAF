import torch
import sys

from sksurv.util import Surv
from sksurv.metrics import concordance_index_censored, concordance_index_ipcw


def compute_survival_metrics(all_censorships, all_event_times, all_risk_scores, survival_info_train):
    """ Compute the survival metrics: C-index and C-index IPCW. """
    # C-index
    c_index = concordance_index_censored((1 - all_censorships).astype(bool), all_event_times, all_risk_scores, tied_tol=1e-08)[0]
    # C-index IPCW
    c_index_ipcw = 0.
    if survival_info_train:
        structured_survival_data_test = Surv.from_arrays(event=(1-all_censorships).astype(bool), time=all_event_times)
        structured_survival_data_train = Surv.from_arrays(event=(1-survival_info_train['censorship']).astype(bool), time=survival_info_train['time'])

        c_index_ipcw = concordance_index_ipcw(structured_survival_data_train, structured_survival_data_test, estimate=all_risk_scores)[0]

    return c_index, c_index_ipcw


def compute_orth(x, y):
    """Compute the orthogonality score (linear disentanglement). """
    # Normalize to unit vectors
    x_norm = torch.nn.functional.normalize(x, dim=1)
    y_norm = torch.nn.functional.normalize(y, dim=1)

    # Compute cosine similarity between corresponding samples
    sim = (x_norm * y_norm).sum(dim=1)  # (batch_size,)

    # Mean absolute similarity
    orth_score = sim.abs().mean()

    return orth_score

def normalize_data(X):
    """ Normalize the data to have zero mean and unit variance. """
    return (X - X.mean(dim=0)) / (X.std(dim=0) + 1e-8)

def compute_dist_corr(x, y, epsilon=1e-8):
        """ Adapted from https://github.com/vios-s/CSDisentanglement_Metrics_Library/blob/master/metrics/distance_correlation.py """
        
        # Compute pairwise distance matrices
        a = torch.cdist(x, x)
        b = torch.cdist(y, y)

        # Double-centering
        A = a - a.mean(dim=0, keepdim=True) - a.mean(dim=1, keepdim=True) + a.mean()
        B = b - b.mean(dim=0, keepdim=True) - b.mean(dim=1, keepdim=True) + b.mean()

        # Compute distance covariance
        dcov = (A * B).mean().sqrt()

        # Compute distance variances
        dvar_x = (A * A).mean().sqrt()
        dvar_y = (B * B).mean().sqrt()

        # Compute distance correlation
        dcor = dcov / torch.sqrt(dvar_x * dvar_y + epsilon)
        return dcor

def compute_disentanglement(rna_specific, wsi_specific, wsi_rna_mm, rna_wsi_mm, type='dcor'):     
    shared_repr = torch.cat([wsi_rna_mm, rna_wsi_mm], dim=-1)
    single_repr = torch.cat([rna_specific, wsi_specific], dim=-1)

    if type == 'dcor':
        # Disentanglement between the modality-specific and modality-shared representations (D2 disentanglement)
        D2 = compute_dist_corr(shared_repr, single_repr)

        # Disentanglement between the two modality-specific representations (D1 disentanglement)
        D1 = compute_dist_corr(rna_specific, wsi_specific)
    
    elif type == 'orth':
        # Disentanglement between the modality-specific and modality-shared representations (D2 disentanglement)
        D2 = compute_orth(shared_repr, single_repr)

        # Disentanglement between the two modality-specific representations (D1 disentanglement)
        D1 = compute_orth(rna_specific, wsi_specific)
    else:
        sys.exit("Disentanglement metric is not implemented ....")
    

    total = (D2 + D1) /2


    return {f'Total Disentanglement {type}': total.item(), f'D1 Disentanglement {type}': D1.item(), f'D2 Disentanglement {type}': D2.item()}
