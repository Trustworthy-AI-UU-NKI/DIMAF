import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

# NLL Survival loss
class NLLSurvLoss(nn.Module):
    """
    The negative log-likelihood loss function for the discrete time to event model.
    Zadeh, S.G. and Schmid, M., 2020. Bias in cross-entropy-based training of deep survival networks. IEEE transactions on pattern analysis and machine intelligence.
    Code obtained from https://github.com/mahmoodlab/MMP/blob/main/src/utils/losses.py
    """
    def __init__(self, alpha=0.0, eps=1e-7):
        super().__init__()
        self.alpha = alpha
        self.eps = eps

    def __call__(self, logits, times, censorships):
        """
        Parameters
        ----------
        logits: (n_batches, n_classes)
            The neural network output discrete survival predictions such that hazards = sigmoid(logits).
        y: (n_batches, )
            The true time bin index label.
        c: (n_batches, )
            The censoring status indicator.
        alpha: float
            TODO: document
        eps: float
            Numerical constant; lower bound to avoid taking logs of tiny numbers.
        """
        times = times.long()
        censorships = censorships.long()

        hazards = torch.sigmoid(logits)

        S = torch.cumprod(1 - hazards, dim=1)

        S_padded = torch.cat([torch.ones_like(censorships), S], 1)

        s_prev = torch.gather(S_padded, dim=1, index=times).clamp(min=self.eps)
        h_this = torch.gather(hazards, dim=1, index=times).clamp(min=self.eps)
        s_this = torch.gather(S_padded, dim=1, index=times+1).clamp(min=self.eps)

        uncensored_loss = -(1 - censorships) * (torch.log(s_prev) + torch.log(h_this))
        censored_loss = - censorships * torch.log(s_this)

        neg_l = censored_loss + uncensored_loss
        
        if self.alpha is not None:
            loss = (1 - self.alpha) * neg_l + self.alpha * uncensored_loss

        
        loss = loss.sum()
        censored_loss = censored_loss.sum()
        uncensored_loss = uncensored_loss.sum()

        return loss,  {'loss': loss.item(), 'uncensored_loss': uncensored_loss.item(), 'censored_loss': censored_loss.item()}


class CoxLoss(nn.Module):
    """
    The Cox proportional hazards loss.
    Code obtained from https://github.com/mahmoodlab/MMP/blob/main/src/utils/losses.py
    """
    def __init__(self):
        super().__init__()

    def __call__(self, logits, times, censorships):
        """
        logits: log risks, B x 1
        times: survival time, B x 1
        censorships: censorship indicator, B x 1
        """    
        event_indicators=(1-censorships).float()
        num_uncensored = torch.sum(event_indicators, 0)

        if num_uncensored.item() == 0:
            loss = torch.sum(logits) * 0
            return loss, {'loss': loss.item()}
        
        times = times.squeeze(1)
        event_indicators = event_indicators.squeeze(1)
        logits = logits.squeeze(1)

        sindex = torch.argsort(-times)
        times = times[sindex]
        event_indicators = event_indicators[sindex]
        logits = logits[sindex]

        log_risk_stable = torch.logcumsumexp(logits, 0)

        likelihood = logits - log_risk_stable
        uncensored_likelihood = likelihood * event_indicators
        logL = -torch.sum(uncensored_likelihood)
        # negative average log-likelihood
        loss = logL / num_uncensored
        return loss, {'loss': loss.item()}


class DisentangledSurvLoss(nn.Module):
    """ Wrapper total loss function for different combinations of losses. """
    def __init__(self, survival_type,  disentanglement_type, weight_surv=0.5, weight_disentanglement=0.5, n_label_bins=4, alpha=0.5):
        super().__init__()
        if disentanglement_type == 'orthogonal':
            self.disentanglement_loss = OrthogonalLoss()
        elif disentanglement_type == 'distcor':
            self.disentanglement_loss = DistanceCorrelationLoss()   
        elif disentanglement_type == 'hsic':
            self.disentanglement_loss = HSICLoss()
        else:
            sys.exit("Disentanglement loss is not implemented, aborting... ")
        
        if survival_type == 'nll':
            self.surv_loss = NLLSurvLoss(alpha=alpha)
            self.num_classes = n_label_bins
        elif survival_type == 'cox':
            self.surv_loss = CoxLoss()
            self.num_classes = 1
        else:
            sys.exit("Survival loss is not implemented, aborting... ")
        
        self.beta1 = weight_disentanglement
        self.beta2 = weight_surv
    
    def get_num_classes(self):
        return self.num_classes

    def __call__(self, output, times, censorships):
        # Compute the survival loss
        loss_surv, log_dict = self.surv_loss(logits=output['logits'], times=times, censorships=censorships)
        log_dict['survival_loss'] = log_dict.pop('loss')

        # Compute the disentanglement loss
        loss_dis, log_dict_dis = self.disentanglement_loss(uni_repr_rna=output['rna_repr'], uni_repr_wsi=output['wsi_repr'], uni_repr_rna_wsi=output['rna_wsi_repr'], uni_repr_wsi_rna=output['wsi_rna_repr'])

        # Total loss
        loss = self.beta2*loss_surv + self.beta1*loss_dis

        # Logging
        for item, value in log_dict_dis.items():
            log_dict[item] = value
        
        log_dict['loss'] = loss.item()

        return loss, log_dict


class OrthogonalLoss(nn.Module):
    """ Orthogonal loss to enforce linear disentanglement. """
    def __init__(self, weight_D1=0.5, weight_D2=0.5):
        super().__init__()
        self.weight_D1 = weight_D1
        self.weight_D2 = weight_D2

    def __call__(self, uni_repr_rna, uni_repr_wsi, uni_repr_rna_wsi, uni_repr_wsi_rna):
 
        shared_repr = torch.concat([uni_repr_wsi_rna, uni_repr_rna_wsi], dim=1)
        single_repr = torch.concat([uni_repr_rna, uni_repr_wsi], dim=1)

        # Normalize the representations
        shared_repr_norm = F.normalize(shared_repr, dim=1)
        single_repr_norm = F.normalize(single_repr, dim=1)

        # Disentanglement between the modality-specific and modality-shared representations (D2 disentanglement)
        dot_product_D2 = torch.bmm(shared_repr_norm.unsqueeze(2), single_repr_norm.unsqueeze(1))
        loss_D2 = torch.norm(dot_product_D2, dim=(1, 2), p='fro') ** 2

        # Normalize the modality-specific representations
        uni_repr_rna_norm = F.normalize(uni_repr_rna, dim=1)
        uni_repr_wsi_norm = F.normalize(uni_repr_wsi, dim=1)

        # Disentanglement between the two modality-specific representations (D1 disentanglement)
        dot_product_D1 = torch.bmm(uni_repr_rna_norm.unsqueeze(2), uni_repr_wsi_norm.unsqueeze(1))
        loss_D1 = torch.norm(dot_product_D1, dim=(1, 2), p='fro') ** 2

        # Average loss across the batch
        loss_D2 = loss_D2.mean()
        loss_D1 = loss_D1.mean()

        loss = self.weight_D2*loss_D2 + self.weight_D1*loss_D1

        return loss, {'disentanglement_loss': loss.item(), 'disentanglement_D1_loss': loss_D1.item(), 'disentanglement_D2_loss': loss_D2.item()}

class DistanceCorrelationLoss(nn.Module):
    """ Distance correlation loss to enforce linear and nonlinear disentanglement. Implemented from Liu, Xiao, et al. "Measuring the biases and effectiveness of content-style disentanglement." arXiv preprint arXiv:2008.12378 (2020)."""
    def __init__(self, weight_D1=0.5, weight_D2=0.5, epsilon=1e-8):
        super().__init__()
        self.weight_D1 = weight_D1
        self.weight_D2 = weight_D2
        self.epsilon = epsilon
    
    def compute_dist_corr(self, x, y):
        # Compute Euclidean pairwise distance matrices
        a = torch.cdist(x, x, p=2)
        b = torch.cdist(y, y, p=2)

        # Double-centering
        A = a - a.mean(dim=0, keepdim=True) - a.mean(dim=1, keepdim=True) + a.mean()
        B = b - b.mean(dim=0, keepdim=True) - b.mean(dim=1, keepdim=True) + b.mean()

        # Compute distance covariance
        dcov = (A * B).mean().sqrt()

        # Compute distance variances
        dvar_x = (A * A).mean().sqrt()
        dvar_y = (B * B).mean().sqrt()

        # Compute distance correlation
        dcor = dcov / torch.sqrt(dvar_x * dvar_y + self.epsilon) 
        return dcor

    def __call__(self, uni_repr_rna, uni_repr_wsi, uni_repr_rna_wsi, uni_repr_wsi_rna):
        """ Compute the distance correlation loss. """ 
        shared_repr = torch.concat([uni_repr_wsi_rna, uni_repr_rna_wsi], dim=1)
        single_repr = torch.concat([uni_repr_rna, uni_repr_wsi], dim=1)

        # Disentanglement between the modality-specific and modality-shared representations (D2 disentanglement)
        dcor_D2 = self.compute_dist_corr(shared_repr, single_repr)

        # Disentanglement between the two modality-specific representations (D1 disentanglement)
        dcor_D1 = self.compute_dist_corr(uni_repr_rna, uni_repr_wsi)

        dcor_total = self.weight_D2*dcor_D2 + self.weight_D1*dcor_D1

        return dcor_total, {'disentanglement_loss': dcor_total.item(), 'disentanglement_D1_loss': dcor_D1.item(), 'disentanglement_D2_loss': dcor_D2.item()}

class HSICLoss(nn.Module):
    """
    HSIC-based loss. Implemented from Gretton, Arthur, et al. "A kernel statistical test of independence." Advances in neural information processing systems 20 (2007).
    """
    def __init__(self, sigma=None, weight_D1=0.5, weight_D2=0.5, unbiased=True):
        """ Initialize the HSIC loss. """
        super(HSICLoss, self).__init__()
        self.sigma = sigma
        self.weight_D1 = weight_D1
        self.weight_D2 = weight_D2
        self.unbiased = unbiased

    def rbf_kernel(self, Z):
        """ Compute the RBF kernel matrix for a given tensor Z. """
        pairwise_dists = torch.cdist(Z, Z, p=2) ** 2
        if self.sigma is None:
            sigma = torch.sqrt(torch.median(pairwise_dists[pairwise_dists > 0]))
        else:
            sigma = self.sigma

        return torch.exp(-pairwise_dists / (2 * sigma ** 2 + 1e-8))
    
    def normalize_data(self, X):
        """ Normalize the data to have zero mean and unit variance. """
        return (X - X.mean(dim=0)) / (X.std(dim=0) + 1e-8)
    
    def compute_hsic(self, X, Y):
        """ Compute the HSIC loss between X and Y. """

        B = X.size(0)  # Batch size

        # Normalize
        X_norm = self.normalize_data(X)
        Y_norm = self.normalize_data(Y)

        # Apply kernels
        K_X = self.rbf_kernel(X_norm)
        K_Y = self.rbf_kernel(Y_norm)

        # Center the kernel matrices
        H = torch.eye(B, device=X.device) - (1.0 / B) * torch.ones(B, B, device=X.device)
        K_X_centered = H @ K_X @ H
        K_Y_centered = H @ K_Y @ H

        # Compute HSIC
        if self.unbiased:
            # Heuristic version of unbiased HSIC of Song et al (2012)
            hsic_value = torch.trace(K_X_centered @ K_Y_centered) / (B - 1) ** 2
        else:
            hsic_value = torch.trace(K_X_centered @ K_Y_centered) / (B) ** 2
        return hsic_value
    
    def __call__(self, uni_repr_rna, uni_repr_wsi, uni_repr_rna_wsi, uni_repr_wsi_rna):
        """ Compute the HSIC loss. """
        shared_repr = torch.concat([uni_repr_wsi_rna, uni_repr_rna_wsi], dim=1)
        single_repr = torch.concat([uni_repr_rna, uni_repr_wsi], dim=1)

        # Disentanglement between the modality-specific and modality-shared representations (D2 disentanglement)
        hsic_D2 = self.compute_hsic(shared_repr, single_repr)

        # Disentanglement between the two modality-specific representations (D1 disentanglement)
        hsic_D1 = self.compute_hsic(uni_repr_rna, uni_repr_wsi)

        hsic_total = self.weight_D2*hsic_D2 + self.weight_D1*hsic_D1

        return hsic_total, {'disentanglement_loss': hsic_total.item(), 'disentanglement_D1_loss': hsic_D1.item(), 'disentanglement_D2_loss': hsic_D2.item()}