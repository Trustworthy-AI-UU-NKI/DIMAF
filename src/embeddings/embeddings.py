import os
import torch

from utils.general_utils import load_pkl, save_pkl
from models.PANTHER import PANTHER

def get_prototypes(n_proto, embed_dim, split_folder, proto_file):
    """ Load, check and return the specified prototypes. """
    assert os.path.exists(os.path.join(split_folder, proto_file)), "{} does not exist!".format(proto_file)

    if proto_file.endswith('pkl'):
        prototypes = load_pkl(split_folder, proto_file)['prototypes'].squeeze()
    else:
        print("File format for given prototypes is not supported. Abborting...")

    assert (n_proto == prototypes.shape[0]) and (embed_dim == prototypes.shape[1]), "Prototype dimensions do not match! Aborting..."

    return prototypes


def create_slide_embeddings(args, prototypes, dataloader):
    """ Generate the slide embeddings in an unsupervised way using PANTHER (https://github.com/mahmoodlab/PANTHER). """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PANTHER(args, prototypes, device).to(device)

    # Generate the slide embeddings
    X, y = model.predict(dataloader)
    dataloader.dataset.X, dataloader.dataset.y = X, y
    embeddings = {'X': X, 'y': y}
    return embeddings

def get_mixture_params(X, p):
    """ Get the mixture probabilities and mean. """
    d = (X.shape[1] - p) // (2 * p)
    prob = X[:, : p]
    mean = X[:, p: p * (1 + d)].reshape(-1, p, d)
    return prob, mean


def reshape_embeddings(dataloader, n_proto):
    """ Reshape the embeddings to be concatenated features of the prob and mean per prototype. """
    prob, mean = get_mixture_params(dataloader.dataset.X, n_proto)

    # Embeddings with importance and mean per mixture 
    dataloader.dataset.X = torch.cat([torch.Tensor(prob).unsqueeze(dim=-1), torch.Tensor(mean)], dim=-1)

    # New input dimension after reshaping
    in_dim = prob.shape[-1] + mean.shape[-1]

    return dataloader, in_dim


def get_slide_embeddings(args, mode, dataloader):
    """ Obtain slide embeddings. """

    # Preparing file path for loading or saving embeddings
    print('\nConstructing unsupervised slide embedding...', end=' ')
    embeddings_name = f"{mode}_uni_embeddings_wsi_proto_{args.n_proto}_em_{args.em_iter}_tau_{args.tau}.pkl"
    embedding_dir = os.path.join(dataloader.dataset.split_dir, 'embeddings_DIMAF')

    
    if os.path.isfile(os.path.join(embedding_dir, embeddings_name)):
        # Load existing embeddings if already created
        embeddings = load_pkl(embedding_dir, embeddings_name)
        print(f'\n\tEmbedding already exists! Loading', end=' ')
    else:
        # Else, create them using an adaptation of PANTHER
        split_folder = dataloader.dataset.get_split_folder()
        prototypes = get_prototypes(args.n_proto, args.in_dim, split_folder, args.proto_file)
        embeddings = create_slide_embeddings(args, prototypes, dataloader)

        # Save the embeddings for efficiency
        os.makedirs(embedding_dir, exist_ok=True)
        save_pkl(embedding_dir, embeddings_name, embeddings)
    
    # Save the embeddings to the dataloader
    dataloader.dataset.X, dataloader.dataset.Y = embeddings['X'], embeddings['y']
    assert dataloader.dataset.X is not None
    
    # Reshape the embeddings to be concatenated features of the prob and mean per prototype
    dataloader, in_dim = reshape_embeddings(dataloader, args.n_proto, args.wsi_embed_no_var)

    return dataloader, in_dim

def prepare_embeddings(args, mode, dataloader):
    """ Prepare unimodal embeddings for both the wsi and rna-seq data. """
    data_info = {}

    # Obtain the wsi embeddings
    dataloader, in_dim_wsi = get_slide_embeddings(args, mode, dataloader)
    data_info['Dim wsi'] = in_dim_wsi
    
    # The pathway embeddings are made on the fly
    pathway_sizes = dataloader.dataset.pathway_sizes
    data_info['Pathway sizes'] = pathway_sizes
    
    return dataloader, data_info
