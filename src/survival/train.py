import os
import torch
import numpy as np

from torch.utils.tensorboard import SummaryWriter
from sksurv.metrics import concordance_index_censored
from collections import defaultdict

from .losses import NLLSurvLoss, CoxLoss, DisentangledSurvLoss
from .test import test_survival_model
from embeddings.embeddings import prepare_embeddings
from models.DIMAF import DIMAF
from utils.general_utils import save_json
from utils.train_utils import get_optim, get_lr_scheduler, list_to_device, LoggingMeter, log_results


def survival_train(args, fold, train_dl, test_dl=None):
    """ Train a survival prediction model for a single fold. """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Set up results and log dir.
    result_dir_fold = os.path.join(args.result_dir, f"Fold_{fold}")
    log_dir_fold = os.path.join(args.log_dir, f"Fold_{fold}")

    os.makedirs(result_dir_fold, exist_ok=True)
    os.makedirs(log_dir_fold, exist_ok=True)

    # Tensorboard writer
    writer = SummaryWriter(log_dir=log_dir_fold)

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

    # Create unimodal unsupervised representations
    print('\nCreate unimodal representations...', end=' ')
    train_dl, data_info  = prepare_embeddings(args, 'train', train_dl)
    
    if not test_dl == None:
        test_dl, _ = prepare_embeddings(args, 'test', test_dl)
    
    # Initialize model
    print('\nInit Model...', end=' ')
    model = DIMAF(rna_dims=data_info['Pathway sizes'],
                       histo_dim=data_info['Dim wsi'],
                       device=device,
                       single_out_dim=256,
                       num_classes=num_classes,
                       loss_fn=loss_fn)
    model.to(device)
    
    # Initialize optimizer and lr scheduler
    print('\nInit optimizer ...')
    optimizer = get_optim(model=model, args=args)
    lr_scheduler = get_lr_scheduler(args, optimizer, len(train_dl))

    #####################
    # The training loop #
    #####################
    print('\nStart training ...')
    for epoch in range(args.max_epochs):
        # Train
        print('#' * 10, f'TRAIN Epoch: {epoch}', '#' * 10)
        train_results, train_data_info = train_loop(model, train_dl, optimizer, lr_scheduler, device)
        log_results(writer, train_results, epoch, mode='train')
        
        # Save last model
        torch.save(model.state_dict(), os.path.join(result_dir_fold, "model_checkpoint.pth"))

    # Test the model
    print(f'End of training')
    if not test_dl == None:
        print(f'Testing on Split {fold}...:')
        results = test_survival_model(model, test_dl, device, result_dir=result_dir_fold, survival_info_train=train_data_info, return_attn=True)
        save_json(result_dir_fold, f"train_test_summary.json", results)

    writer.close()
    

def train_loop(model, dataloader, optimizer, lr_scheduler, device):
    """ Train loop for survival prediction. """
    model.train()
    train_log = {}
    train_outputs = defaultdict(list)  # Stores risks, censorships, event_times for all samples

    # Loop over all data samples
    for idx, batch in enumerate(dataloader):
        # Get the data
        wsi = batch['img'].to(device)
        rna = list_to_device(batch['rna'], device)

        # Get the labels
        label = batch['label'].to(device)
        event_time = batch['survival_time'].to(device)
        censorship = batch['censorship'].to(device)

        # Forward pass
        output_results, log_dict = model(wsi=wsi, rna=rna, label=label, censorship=censorship)

        # Backward pass
        loss = output_results['loss']
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

        # Logging
        for key, val in log_dict.items():
            if key not in train_log:
                train_log[key] = LoggingMeter(key)
            train_log[key].update(val, n=len(wsi))

        train_outputs['risk'].append(output_results['risk'].detach().cpu().numpy())
        train_outputs['censorship'].append(censorship.cpu().numpy())
        train_outputs['event_time'].append(event_time.cpu().numpy())

    # Concatenate everything
    all_train_results = {}
    for key, vals in train_outputs.items():
        all_train_results[key] = np.concatenate(vals, axis=0).squeeze(1)
    
    # Compute c-index
    c_index = concordance_index_censored(
        (1 - all_train_results['censorship']).astype(bool), all_train_results['event_time'], all_train_results['risk'], tied_tol=1e-08)[0]

    results = {item: meter.avg for item, meter in train_log.items()}
    results.update({'c_index': c_index})
    results['lr'] = optimizer.param_groups[0]['lr']
    train_data_info = {'censorship': all_train_results['censorship'], 'time': all_train_results['event_time']}

    return results, train_data_info
    