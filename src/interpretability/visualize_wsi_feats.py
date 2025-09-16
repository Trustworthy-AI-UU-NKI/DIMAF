import os
import torch
import argparse
import numpy as np
import h5py
import openslide
from tqdm import tqdm
import matplotlib.pyplot as plt
import sys

sys.path.append('../')
from embeddings.embeddings import get_mixture_params
import math
from utils.visualization_utils import get_panther_encoder, get_dataset, get_mixture_plot_figure


def find_patch_size(next_coords, prev_coords, coords_patch):
    """Find the patch size by comparing the coords with the next patch"""
    y_ps = 0
    for y_cord, x_cord in next_coords:
        if y_ps > 0:
            break

        if y_cord > coords_patch[0]:
            y_ps = y_cord-coords_patch[0]

    # If the patch is at the last edge
    if y_ps == 0:
        for y_cord, x_cord in prev_coords[::-1]:
            if y_ps > 0:
                break

            if y_cord < coords_patch[0]:
                y_ps = coords_patch[0] - y_cord
    
    assert y_ps > 0, "patch size is 0. Somthing is going wrong!"

    # a patch is always square
    return y_ps
    
def find_patch(all_ids, all_patch_lens, idx, h5_feats_fpath):
    """Find the coordinates and patch size of the patchyou want to visualize."""
    counter = 0
    for i, patch_len in enumerate(all_patch_lens):
        # Find the slide the specific patch belongs to
        if idx < (counter + patch_len):
            slide_id = all_ids[i]
            patch_i = idx - counter
            break
        else:
            counter += patch_len

    # Open the slide
    h5_feats_fpath_slide = os.path.join(h5_feats_fpath, f'{slide_id}.h5')
    h5_file = h5py.File(h5_feats_fpath_slide, 'r')

    # Get coords of the specific patch
    coords = h5_file['coords']
    coords_patch = coords[patch_i]

    # Get the patchsize of this patch
    patch_size = find_patch_size(coords[patch_i+1:], coords[:patch_i], coords_patch)
    return slide_id, coords_patch, patch_size

def get_grid(n_figures):
    """Return (rows, cols) for a grid layout. """
    cols = math.ceil(math.sqrt(n_figures))
    rows = math.ceil(n_figures / cols)
    return rows, cols

def plot_pt(result_file, h5_feats_fpath, slides_fpath, all_ids, all_patch_lens, indices):
    """Visualize the given patches to a specified prototype."""
    grid_shape = get_grid(len(indices))
    fig, axes = plt.subplots(*grid_shape, figsize=(3.5*grid_shape[1], 3.8*grid_shape[0]))  # Adjust size as needed
    axes = axes.flatten() 

    # We visualize x patches per prototype
    for i, (idx, ax) in enumerate(zip(indices, axes)):
        slide = None
        try:
            # Find slide and patch information
            slide_id, h5_coord, patch_size = find_patch(all_ids, all_patch_lens, idx, h5_feats_fpath)
            slide_path = os.path.join(slides_fpath, f'{slide_id}.svs')

            # Open slide and extract patch
            slide = openslide.OpenSlide(slide_path)
            patch = slide.read_region(
                (h5_coord[0], h5_coord[1]), 
                level=0, 
                size=(patch_size, patch_size)
            ).convert("RGB")
            
            
            # Display patch in subplot
            ax.imshow(patch)
            ax.axis("off")
        except Exception as e:
            print(f"Error processing patch {i}: {e}")
        finally:
            if slide is not None:
                slide.close()  # Explicitly close the OpenSlide object
    
    # Save and close the plot
    plt.tight_layout()
    plt.savefig(result_file, format='pdf', dpi=300, bbox_inches='tight')
    plt.close()


def get_info_panther_embeddings(split_folder, data_type, fold):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Get PANTHER model and the wsi's to obtain the embeddings
    panther_encoder = get_panther_encoder(split_folder=split_folder)
    dataset = get_dataset(data_type, fold, type)

    # Loop over the dataset
    with torch.inference_mode():
        for idx in tqdm(range(len(dataset))):
            batch = dataset.__getitem__(idx)
            data, slide_id = batch['img'].unsqueeze(dim=0), batch['slide_id']
            data = data.to(device)

            with torch.no_grad():
                # Obtain slide embeddings (GMM parameters)
                out, qqs = panther_encoder.representation(data).values()
                # Obtain the posterior probabilities of each patch given each prototype
                qq = qqs[0,:,:,0].cpu().numpy()
            
            all_qq.append(qq)
            all_ids.append(slide_id)
            all_patch_lens.append(qq.shape[0])
    
    all_qq = np.vstack(all_qq)
    all_ids = np.array(all_ids)
    all_patch_lens = np.array(all_patch_lens)

    return all_qq, all_patch_lens, all_ids


def visualize_pt_assignment_general(type, fold, data_type):
    """ Visualize the mixture proportion distribution of all data in train or test splits. """
    split_folder = f"../data/data_files/tcga_{type}/splits/{fold}"
    result_dir_all = os.path.join(split_folder, f'prototypes_DIMAF/visualization_{data_type}')
    os.makedirs(result_dir_all, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    all_props = []

    # Get PANTHER model and the wsi's to obtain the embeddings
    panther_encoder = get_panther_encoder(split_folder=split_folder)
    dataset = get_dataset(mode=data_type, fold=fold, type=type)

    # Loop over dataset
    with torch.inference_mode():
        for idx in tqdm(range(len(dataset))):
            batch = dataset.__getitem__(idx)
            data = batch['img'].unsqueeze(dim=0)
            data = data.to(device)

            with torch.no_grad():
                # Obtain slide embeddings (GMM parameters)
                out, qqs = panther_encoder.representation(data).values()
                pis, mus = get_mixture_params(out.detach().cpu(), p=16)
                # We obtain the mixture probabilities
                pis = pis[0].detach().cpu().numpy()

            # Save distribution importances of each slide
            all_props.append(pis)

    # take mean over all samples
    all_props = torch.Tensor(np.array(all_props))
    all_props_mean = torch.mean(all_props, dim=0)

    assert all_props_mean.shape == torch.Size([16])

    # Visualize general prototype assignment
    get_mixture_plot_figure(all_props_mean, plot_path=os.path.join(result_dir_all, 'mixture_proportion_distribution.pdf'))


def visualize_pt(type, fold, data_type, nr_patches=25):
    """Visualize the closest patches to each prototype for a specific fold."""
    # input paths
    slides_fpath = f'../data/data_files/tcga_{type}/wsi/images'
    h5_feats_fpath = f'../data/data_files/tcga_{type}/wsi/extracted_res0_5_patch256_uni/feats_h5/'

    # Output paths
    split_folder = f"../data/data_files/tcga_{type}/splits/{fold}"
    result_dir = os.path.join(split_folder, f'prototypes_DIMAF/visualization_{data_type}')
    os.makedirs(result_dir, exist_ok=True)

    # get all probabilities (qq), nr of patches per wsi and wsi id (slide id)
    all_qq, all_patch_lens, all_ids = get_info_panther_embeddings(split_folder, data_type, fold)

    # For all prototypes
    for i in range(16):
        # Obtain the patches with the highest posterior probability given this prototype
        rev_top_indices = np.argsort(all_qq[:,i])[-nr_patches:]
        top_indices = np.flip(rev_top_indices)

        # Visualize those patches
        result_img_pt_name = os.path.join(result_dir, f'pt_{i}.pdf')
        plot_pt(result_img_pt_name, h5_feats_fpath, slides_fpath, all_ids, all_patch_lens, top_indices)

        
def visualize_pt_per_sample(type, fold, data_type, nr_patches=25):
    """Visualize the closest patches to each prototype for a specific fold."""
    # input paths
    slides_fpath = f'../data/data_files/tcga_{type}/wsi/images'
    h5_feats_fpath = f'../data/data_files/tcga_{type}/wsi/extracted_res0_5_patch256_uni/feats_h5/'

    # Output paths
    split_folder = f"../data/data_files/tcga_{type}/splits/{fold}"
    result_dir = os.path.join(split_folder, f'prototypes_DIMAF/visualization_{data_type}')
    os.makedirs(result_dir, exist_ok=True)

    # get all probabilities (qq), nr of patches per wsi and wsi id (slide id)
    all_qq, all_patch_lens, all_ids = get_info_panther_embeddings(split_folder, data_type, fold)

    for i in range(16):
        print(f"Prototype {i}")
        slide_best_patches = []  # Track best patch per slide for this prototype

        # Group patches by slide and find the best patch for each slide
        start_idx = 0
        all_persons = []
        for slide_idx, num_patches in enumerate(all_patch_lens):
            person_id = "-".join(all_ids[slide_idx].split('-')[:3]) 
            if person_id in all_persons:
                end_idx = start_idx + num_patches
                start_idx = end_idx
                continue

            all_persons.append(person_id)
            end_idx = start_idx + num_patches
            slide_qq = all_qq[start_idx:end_idx, i]
            best_patch_idx = np.argmax(slide_qq)
            best_patch_likelihood = slide_qq[best_patch_idx]
            slide_best_patches.append((all_ids[slide_idx], start_idx + best_patch_idx, best_patch_likelihood))
            start_idx = end_idx

        # Sort all slides by their best patch likelihood for this prototype
        slide_best_patches.sort(key=lambda x: x[2], reverse=True)

        # Find the top patches across different slides
        top_patches = slide_best_patches[:nr_patches]
        top_indices = [entry[1] for entry in top_patches]

        # Visualize those patches
        result_img_pt_name = os.path.join(result_dir, f'spec_pt_{i}.pdf')
        plot_pt(result_img_pt_name, h5_feats_fpath, slides_fpath, all_ids, all_patch_lens, top_indices)

        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--type', type=str, default='brca')
    parser.add_argument('--exp_name', type=str, default='DIMAF')
    parser.add_argument('--fold', type=int, default=2)
    parser.add_argument('--task', type=str, default='pt_assignment_test', choices=['pt_assignment_test', 'pt_assignment_train', 'pt_vis_test', 'pt_vis_train', 'pt_vis_train_spec', 'pt_vis_test_spec'])
    parser.add_argument('--nr_patches', type=int, default=25)
    args = parser.parse_args()

    # Visualize mixture proportion distribution for test data
    if args.task == 'pt_assignment_test':
        visualize_pt_assignment_general(args.type, args.fold, 'test')

    # Visualize mixture proportion distribution for train data
    elif args.task == 'pt_assignment_train':
        visualize_pt_assignment_general(args.type, args.fold, 'train')

    # Visualize prototypes using closest test patches
    elif args.task == 'pt_vis_test':
        visualize_pt(args.type, args.fold, 'test', nr_patches=args.nr_patches)

    # Visualize prototypes using closest train patches
    elif args.task == 'pt_vis_train':
        visualize_pt(args.type, args.fold, 'train', nr_patches=args.nr_patches)

    # Visualize prototypes using closest train patches with max one patch per sample
    elif args.task == 'pt_vis_train_spec':
        visualize_pt_per_sample(args.type,  args.fold, 'train', nr_patches=args.nr_patches)

    # Visualize prototypes using closest test patches with max one patch per sample
    elif args.task == 'pt_vis_test_spec':
        visualize_pt_per_sample(args.type, args.fold, 'test', nr_patches=args.nr_patches)
