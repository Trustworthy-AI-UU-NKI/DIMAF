"""
    This is the main file to run the survival prediction.
"""

import os
import sys 
import argparse
from torch.utils.data import DataLoader

from utils.general_utils import set_seed, save_json, save_exp_settings
from survival.train import survival_train
from survival.test import survival_test
from interpretability.compute_shap import survival_shap
from data.mm_survival_dataset import MMSurvivalDataset


def create_dataloader(args, fold, mode="train", type='dl'):
    """ Obtain the dataset and dataloader. """
    dataset = MMSurvivalDataset(args, mode, fold)

    print(f"Dataset for fold {fold} is constructed and checked!")
    print(f'Split: {fold}, n: {len(dataset)}')

    # return dataloader
    if type == 'dl':
        shuffle_mode = mode == "train"
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=shuffle_mode, num_workers=args.num_workers)
        return dataloader
    # return data information for metrics computation
    elif type == 'data_info':
        all_censorships, all_event_times = dataset.get_all_labels()
        return {'censorship': all_censorships, 'time':all_event_times}
    # return dataset
    else: 
        return dataset


def k_fold_shap(args):
    """ Obtain shap values for k-folds. """
    for i in range(args.folds):
        print(f"Fold {i}")
        # SHAP values of Z^p_
        survival_shap(args, i, shap_type='post_attn')
        # SHAP values of Z_
        survival_shap(args, i, shap_type='post_attn_av')


def k_fold_test(args):
    """ K-fold cross-validation - Test only. """
    final_res = {}
    for i in range(args.folds):
        print("Testing fold ", i)
        # Get test dataloader & train data info
        train_data_info = create_dataloader(args, fold=i, mode="train", type='data_info')
        test_dl = create_dataloader(args, fold=i, mode='test')
        # Test for this fold
        results = survival_test(args, test_dl, survival_info_train=train_data_info, fold=i)
        final_res[f'Fold{i}'] = results

    # Save all results
    save_json(args.result_dir, 'Final_results.json', final_res)


def k_fold_train(args):
    """ K-fold cross-validation - Train only. """
    for i in range(args.folds):
        # Get train dataloader
        train_dl = create_dataloader(args, fold=i, mode="train")
        # Train for this fold
        survival_train(args, i, train_dl)


def k_fold_train_test(args):
    """ K-fold cross-validation - Train and Test. """
    save_exp_settings(args)
    for i in range(args.folds):
        # Get train and test dataloader
        train_dl = create_dataloader(args, fold=i, mode="train")
        test_dl = create_dataloader(args, fold=i, mode="test")
        # Train & test for this fold
        survival_train(args, i, train_dl, test_dl)

 
def main(args):
    """ K-fold cross-validation for Survival Prediction. """
    set_seed(args.seed)

    # Result dirs
    args.result_dir = os.path.join(args.result_dir, args.task, args.exp_code)
    args.log_dir = os.path.join(args.log_dir, args.task, args.exp_code)
    os.makedirs(args.result_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    # If not using NLL loss, set n_label_bins to 0
    if args.loss_fn != 'nll':
        args.n_label_bins = 0

    # Run the specified mode
    if args.mode == "train_test":
        k_fold_train_test(args)
    elif args.mode == "train":
        k_fold_train(args)
    elif args.mode == "test":
        k_fold_test(args)
    elif args.mode == "shap":
        k_fold_shap(args)
    else:
        sys.exit("Unspecified mode! Abborting..")
    
    print("FINISHED!\n\n\n")
    
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Survival Prediction with DIMAF')

    # General args 
    parser.add_argument('--seed', type=int, default=1, help='random seed for reproducible experiment (default: 1)')
    parser.add_argument('--num_workers', type=int, default=2)


    # Learning args
    parser.add_argument('--max_epochs', type=int, default=30, help='maximum number of epochs to train (default: 20)')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--wd', type=float, default=1e-5, help='weight decay')
    parser.add_argument('--lr_scheduler', type=str, choices=['cosine', 'linear', 'constant'], default='cosine')
    parser.add_argument('--warmup_steps', type=int, default=-1, help='warmup iterations')


    # PANTHER args
    parser.add_argument('--ot_eps', default=0.1, type=float, help='Strength for entropic constraint regularization for OT')
    parser.add_argument('--em_iter', type=int, default=1)
    parser.add_argument('--tau', type=float, default=0.001)
    parser.add_argument('--proto_file', type=str, help="Path to prototypes")
    parser.add_argument('--fix_proto', type=bool, default=True)
    parser.add_argument('--n_proto', type=int, default=16)


    # Loss args
    parser.add_argument('--w_dis', type=float, default=7)
    parser.add_argument('--w_surv', type=float, default=1)
    parser.add_argument('--n_label_bins', type=int, default=4, help='number of bins for event time discretization')
    parser.add_argument('--loss_fn', type=str, default='cox_distcor', choices=['nll', 'cox', 'cox_orthogonal', 'cox_distcor', 'cox_hsic', 'nll_orthogonal', 'nll_distcor'],help='which loss function to use')
    parser.add_argument('--nll_alpha', type=float, default=0.5, help='Balance between censored / uncensored loss (in NLL)')
    

    # Experiment args / label args
    parser.add_argument('--task', type=str, default='dss_survival_brca')
    parser.add_argument('--target_col', type=str, default='dss_survival_days')
    parser.add_argument('--folds', type=int, default=5)
    parser.add_argument('--mode', type=str, default='train_test', choices=['test', 'train_test', 'train', 'shap']) 


    # dataset args
    parser.add_argument('--omics_type', type=str, default='rna_data')
    parser.add_argument('--data_source', type=str, default='data/data_files/tcga_brca/', help='manually specify the data source')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--in_dim', default=1024, type=int, help='dim of input features (output of WSI patch foundation model)')


    # logging args
    parser.add_argument('--result_dir', default='results',help='results directory')
    parser.add_argument('--log_dir', default='logs',help='results directory')
    parser.add_argument('--return_attn', action='store_true', default=False)
    parser.add_argument('--exp_code', type=str, default='test', help='experiment code for saving results')


    # SHAP args
    parser.add_argument('--shap_refdist_n', type=int, default=512) # BRCA: 512, BLCA: 256, LUAD: 320, KIRC: 192
    parser.add_argument('--shap_bs', type=int, default=64)
    parser.add_argument('--explainer', type=str, default='shap', choices=['eg', 'shap'])


    args = parser.parse_args()
    main(args)