
import torch
import os
import shap
import sys
import numpy as np


from embeddings.embeddings import get_mixture_params
from models.DIMAF import SHAP_DIMAF
from data.mm_survival_dataset import MMSurvivalDataset
from utils.general_utils import load_pkl, save_pkl

def get_multimodal_dataset(args, mode, fold):
    # Obtain multimodal dataset
    dataset = MMSurvivalDataset(args, mode, fold)

    # Load WSI embeddings from PANTHER
    embeddings_name = f"{mode}_uni_embeddings_wsi_proto_{args.n_proto}_em_{args.em_iter}_tau_{args.tau}.pkl"
    embedding_dir = os.path.join(dataset.split_dir, 'embeddings_DIMAF')

    try:
        embeddings = load_pkl(embedding_dir, embeddings_name)
    except:
        print("Something is wrong; there are no embeddings for the defined WSI's. Please check if the embeddings exist.")

    dataset.X, dataset.Y = embeddings['X'], embeddings['y']
    new_in_dim = dataset.X.shape[-1]
    # Use only the proportion and mean of each mixture as WSI representation
    prob, mean = get_mixture_params(dataset.X, args.n_proto)
    dataset.X = torch.cat([torch.Tensor(prob).unsqueeze(dim=-1), torch.Tensor(mean)], dim=-1)
    in_dim = (new_in_dim // args.n_proto) - mean.shape[-1]

    return dataset, in_dim

def get_sample_ids(data):
    """ Get the sample IDs (case_id, slide_id). """
    samples = []
    for i in range(data.__len__()):
        item = data.get_sample_id(i)
        samples.append(item)
    return samples

def prepare_data_shap_post_attn(data, model):
    """ Function that prepares the data to obtain shap values of embeddings after fusion. """
    preproc_dataset = model.prep_data_post_attn(data)
    # preproc_dataset: [n_samples, [zp_gg, zp_hg, zp_gh, zp_hh]]
    # rna specific == zp_gg
    # wsi_rna == zp_hg
    # rna_wsi == zp_gh
    # wsi_specific == zp_hh
    feature_names = [f"rna_specific_{i}" for i in range(50)] + [f"wsi_rna_{i}" for i in range(50)] + [f"rna_wsi_{i}" for i in range(16)] + [f"wsi_specific_{i}" for i in range(16)]
    # Get sample ids
    samples = get_sample_ids(data)
    return preproc_dataset, feature_names, samples

def prepare_data_shap_post_attn_av(data, model):
    """ Function that prepares the data to obtain shap values of embeddings after fusion. """
    preproc_dataset = model.prep_data_post_attn_av(data)
    # preproc_dataset: [n_samples, [z_hg, z_gh, z_gg, z_hh]]
    feature_names = ["wsi_rna_zhg", "rna_wsi_zgh", "rna_specific_zgg", "wsi_specific_zhh"]
    # Get sample ids
    samples = get_sample_ids(data)

    return preproc_dataset, feature_names, samples

def survival_shap(args, fold, shap_type='start'):
    """ Compute shap values for a trained survival prediction model. """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define paths
    results_dir_fold =  os.path.join(args.result_dir, f"Fold_{fold}/")
    pretrained_model_path = os.path.join(results_dir_fold, "model_checkpoint.pth")
    results_dir_shap = os.path.join(results_dir_fold, f'shap')
    os.makedirs(results_dir_shap, exist_ok=True)

    # Create multimodal dataset 
    train_data, wsi_dim  = get_multimodal_dataset(args, 'train', fold)
    test_data, wsi_dim  = get_multimodal_dataset(args, 'test', fold)

    # Obtain wrapper for SHAP 
    model = SHAP_DIMAF(rna_dims=train_data.pathway_sizes,
                       histo_dim=wsi_dim,
                       bs=args.shap_bs,
                       device=device,
                       shap_type=shap_type,
                       single_out_dim=256,
                       n_label_bins=args.n_label_bins,
                       loss_fn=args.loss_fn,
                       num_proto_wsi=args.n_proto,
                       num_workers=args.num_workers)
    
    model.to(device)
    model.from_pretrained(pretrained_model_path)

    # Obtain train and test input data for the shap_module
    if shap_type == 'post_attn':  
        train_data, feature_names, _ = prepare_data_shap_post_attn(train_data, model)
        test_data, feature_names_test, samples_test = prepare_data_shap_post_attn(test_data, model)
        assert feature_names_test == feature_names
    elif shap_type == 'post_attn_av':  
        train_data, feature_names, _ = prepare_data_shap_post_attn_av(train_data, model)
        test_data, feature_names_test, samples_test = prepare_data_shap_post_attn_av(test_data, model)
        assert feature_names_test == feature_names
    else:
        sys.exit("SHAP mode is not implemented, abborting....")

    # Background samples for SHAP
    mask = shap.sample(train_data.to(device), args.shap_refdist_n)
    test_data = test_data.to(device)
    
    # Compute SHAP values with the given explainer
    if args.explainer == 'shap':
        print("Computing SHAP values....")
        explainer = shap.DeepExplainer(model, mask)
        shap_values = explainer.shap_values(test_data, check_additivity=False)
    elif args.explainer == 'eg':
        print("Computing Expected Gradients values....")
        explainer = shap.GradientExplainer(model, mask)
        shap_values = explainer.shap_values(test_data)
    else:
        sys.exit("Unspecified explainer! Abborting..")

    # Squeeze the SHAP values
    shap_values_sq = np.squeeze(shap_values) # [B, n_feats, Dz]

    # Save shap values
    print("Saving shap values..")
    name = f'{shap_type}_{args.explainer}'
    save_pkl(results_dir_shap, f"{name}.pkl", {'shap values': shap_values_sq, 'Feature names': feature_names, "Samples": samples_test})
  



