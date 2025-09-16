"""
    This is the main file to create the WSI prototypes.
"""

import os
import argparse
from torch.utils.data import DataLoader

from data.prototype_dataset import WSIDataset
from utils.general_utils import set_seed, save_pkl
from embeddings.prototype import cluster


def create_dataloader(args, split_dir):
    """ Create the dataloader for the clustering. """
    split_file = os.path.join(split_dir, "train.csv")
    wsi_path = os.path.join(args.data_source, args.wsi_dir)

    # Get the dataset and dataloader
    dataset = WSIDataset(wsi_path, split_file)
    dataloader = DataLoader(dataset, batch_size=1, num_workers=args.num_workers)
    return dataloader

def save_prototypes(args, pt_weights, split_dir):
    """ Save the generated prototypes in a file """
    name = f"prototypes_{args.n_proto}_type_{args.mode}_init_{args.n_init}_nr_{args.n_proto_patches}.pkl"
    output_dir = os.path.join(split_dir, "prototypes_DIMAF")
    os.makedirs(output_dir, exist_ok=True)

    # Save the prototypes
    save_pkl(output_dir, name, {'prototypes': pt_weights})
   


def main(args):
    """ Generate prototypes for the different folds """
    set_seed(args.seed)
    # For each fold, create the prototypes
    for i in range(args.folds):
        print("Generating prototypes for fold ", i)
        split_dir = os.path.join(args.data_source, f'splits/{i}')
        dataloader = create_dataloader(args, split_dir)
        # Cluster the patches to obtain prototypes
        pt_weights = cluster(dataloader, 
                            n_proto=args.n_proto,
                            n_iter=args.n_iter,
                            n_init=args.n_init,
                            feature_dim=args.in_dim,
                            mode=args.mode,
                            n_proto_patches=args.n_proto_patches)
        # Save the cluster centers as prototypes
        save_prototypes(args, pt_weights, split_dir)
        print(f"Saved the prototypes for fold {i}!")

    print("DONE")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create WSI prototypes')

    # General args
    parser.add_argument('--seed', type=int, default=1, help='Random seed for reproducible experiment (default: 1)')
    parser.add_argument('--folds', type=int, default=5)
    parser.add_argument('--num_workers', type=int, default=2)

    # Clustering args
    parser.add_argument('--n_proto', type=int, default=16, help='Number of prototypes')
    parser.add_argument('--n_proto_patches', type=int, default=100000,help='Number of patches per prototype to use. Total patches = n_proto * n_proto_patches')
    parser.add_argument('--n_init', type=int, default=3, help='Number of different KMeans initialization (for FAISS)')
    parser.add_argument('--n_iter', type=int, default=50, help='Number of iterations for Kmeans clustering')
    parser.add_argument('--mode', type=str, choices=['kmeans', 'faiss'], default='faiss', help='Clustering mode')

    # Data args
    parser.add_argument('--data_source', type=str, default='data/data_files/tcga_brca/', help='The data source')
    parser.add_argument('--wsi_dir', type=str, default='wsi/extracted_res0_5_patch256_uni/feats_h5/', help='The WSI directory')
    parser.add_argument('--in_dim', type=int, default=1024)

    args = parser.parse_args()

    main(args)