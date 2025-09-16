import cv2
import torch
import numpy as np
from tqdm import tqdm
from types import SimpleNamespace
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from models.PANTHER import PANTHER
from embeddings.embeddings import get_prototypes
from data.single_survival_dataset import WSISurvivalDataset



def get_dataset(mode, fold, type):
    """ Obtain dataset for visualization. """
    # Define args
    args = SimpleNamespace(
        data_source=f'../../../data/tcga_{type}/',
        target_col='dss_survival_days',
        n_label_bins=4
    )

    dataset = WSISurvivalDataset(args, mode, fold)

    return dataset

def get_panther_encoder(split_folder):
    """ Load the PANTHER model for obtaining WSI embeddings. Needed for visualization of WSI prototypes. """
    # Define args
    args = SimpleNamespace(
        in_dim=1024,
        n_proto=16,
        em_iter=1,
        tau=0.001,
        ot_eps=0.1,
        fix_proto=True
    )

    # Load the prototypes
    prototypes = get_prototypes(args.n_proto, args.in_dim, split_folder, f'prototypes_DIMAF/prototypes_{args.n_proto}_type_faiss_init_3_nr_100000.pkl')

    # Create the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PANTHER(args, prototypes, device).to(device)
    model.eval()
    return model

def get_mixture_plot_figure(mixtures, plot_path=None):
    """Create a barplot for the mixture coefficients Pi_c."""
    labels = [f'W{i}' for i in range(len(mixtures))]

    # Create dataframe for plotting
    mixtures_df = pd.DataFrame(mixtures, index=labels).T

    # Plot
    fig, ax = plt.subplots()
    sns.barplot(mixtures_df, color='navy', ax=ax)

    # Spine visibility
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    # Labels and ticks
    ax.set_ylabel(r'$\pi$', fontsize=15)
    max_val = np.max(mixtures)  
    ylim = np.ceil(max_val * 10) / 10.0  
    ax.set_ylim([0, ylim + 0.02])
    yticks = np.arange(0, ylim + 0.02, 0.1)
    ax.set_yticks(yticks)
    plt.xticks(rotation=45)

    # Save plot if path is provided
    if plot_path:
        fig.savefig(plot_path, format='pdf', dpi=300)

    plt.close(fig)
    return fig