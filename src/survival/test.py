import os
import torch
import numpy as np
from collections import defaultdict

from .losses import NLLSurvLoss, CoxLoss, DisentangledSurvLoss
from .metrics import compute_disentanglement, compute_survival_metrics
from utils.train_utils import list_to_device, LoggingMeter
from utils.general_utils import save_json, save_pkl
from embeddings.embeddings import prepare_embeddings
from models.DIMAF import DIMAF

def test_survival_model(model, test_dl, device, result_dir=None, survival_info_train=None, return_attn=False):
    """ Test a survival prediction model for a single fold. """
    model.eval()

    outputs = defaultdict(list)  # Stores attentions, scores, labels, etc.
    test_log = {}

    # Loop over data
    with torch.no_grad():
        for idx, batch in enumerate(test_dl):
            # Get the data and labels
            wsi = batch['img'].to(device)
            rna = list_to_device(batch['rna'], device)
            label = batch['label'].to(device)
            event_time = batch['survival_time'].to(device)
            censorship = batch['censorship'].to(device)

            # forward pass
            out, log_dict = model(wsi=wsi, rna=rna, label=label, censorship=censorship, return_attn=return_attn)

            # IDs
            outputs['case_ids'].append(np.array(batch['case_id']))
            outputs['slide_ids'].append(np.array(batch['slide_id']))

            # Risk scores and labels
            outputs['risk'].append(out['risk'].detach().cpu().numpy())
            outputs['censorship'].append(censorship.cpu().numpy())
            outputs['event_time'].append(event_time.cpu().numpy())
            
            # Attention matrices
            if return_attn:
                for key in ['self_attn_rna', 'cross_attn_rna_wsi', 'cross_attn_wsi_rna', 'self_attn_wsi']:
                    val = out[key]
                    if len(val.shape) == 2:
                        val = val.unsqueeze(0)
                    outputs[key].append(val.detach().cpu().numpy())

            # Embeddings
            for key in ['rna_repr', 'rna_wsi_repr', 'wsi_rna_repr', 'wsi_repr']:
                outputs[key].append(out[key].detach())
                
            # Logging
            for key, val in log_dict.items():
                if key not in test_log:
                    test_log[key] = LoggingMeter(key)
                test_log[key].update(val, n=len(wsi))
            
    
        # Concatenate everything
        all_results = {}
        for key, vals in outputs.items():
            if 'repr' in key:  # embeddings
                all_results[key] = torch.cat(vals, dim=0)
            else:  # numpy arrays
                all_results[key] = np.concatenate(vals, axis=0).squeeze(1)

        # Compute disentanglement metrics
        dcor_dict = compute_disentanglement(all_results['rna_repr'], all_results['wsi_repr'],
                                            all_results['wsi_rna_repr'], all_results['rna_wsi_repr'])
        orth_dict = compute_disentanglement(all_results['rna_repr'], all_results['wsi_repr'],
                                            all_results['wsi_rna_repr'], all_results['rna_wsi_repr'], type='orth')

        # Compute survival metrics
        c_index, c_index_ipcw = compute_survival_metrics(all_results['censorship'], all_results['event_time'],
                                                        all_results['risk'], survival_info_train)

        # Combine all results
        results = {item: meter.avg for item, meter in test_log.items()}
        results.update({'c_index': c_index, 'c_index_ipcw': c_index_ipcw})
        results.update(dcor_dict)
        results.update(orth_dict)

        # Save predicted risk scores
        risk_scores_dict = {k: all_results[k] for k in ['case_ids', 'slide_ids', 'risk', 'censorship', 'event_time']}
        save_pkl(result_dir, "predicted_risk_scores_test.pkl", risk_scores_dict)

        # Save attention data
        if return_attn:
            assert result_dir is not None, "Result dir is not specified."
            attention_data = {k: np.concatenate(outputs[k], axis=0) for k in ['self_attn_rna', 'cross_attn_rna_wsi',
                                                                            'cross_attn_wsi_rna', 'self_attn_wsi']}
            attention_data['case_ids'] = all_results['case_ids']
            attention_data['slide_ids'] = all_results['slide_ids']
            save_pkl(result_dir, "attention_matrices_test.pkl", attention_data)

    return results

def survival_test(args, test_dl, fold, survival_info_train):
    """ Load and test survival prediction model. """
 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Set up results and log dir.
    results_dir_fold =  os.path.join(args.result_dir, f"Fold_{fold}/")
    pretrained_model_path = os.path.join(results_dir_fold, "model_checkpoint.pth")

    # Initialize loss function
    if args.loss_fn == 'nll':
        loss_fn = NLLSurvLoss(alpha=args.nll_alpha)
        num_classes = args.n_label_bins
    elif args.loss_fn == 'cox':
        loss_fn = CoxLoss()
        num_classes = 1
    else:
        loss_fn_split = args.loss_fn.split("_")
        loss_fn = DisentangledSurvLoss(loss_fn_split[0], loss_fn_split[1], weight_surv=args.w_surv, weight_disentanglement=args.w_dis, n_label_bins=args.n_label_bins, alpha=args.nll_alpha)
        num_classes = loss_fn.get_num_classes()

    # Create unimodal representations
    test_dl, data_info  = prepare_embeddings(args, 'test', test_dl)

    # Load model
    model = DIMAF(rna_dims=data_info['Pathway sizes'],
                       histo_dim=data_info['Dim wsi'],
                       device=device,
                       single_out_dim=256,
                       num_classes=num_classes,
                       loss_fn=loss_fn
                       )
    model.to(device)
    model.from_pretrained(pretrained_model_path)

    # Test the model
    results = test_survival_model(model, test_dl, device, result_dir=results_dir_fold, survival_info_train=survival_info_train, return_attn=args.return_attn)
    save_json(results_dir_fold, f"test_summary.json", results)
    return results


