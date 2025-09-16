import os
import torch
import pandas as pd
import h5py
import numpy as np

from torch.utils.data import Dataset
from utils.data_utils import compute_discretization, pd_diff, overlap_col_df, series_intersection
from utils.general_utils import save_pkl, load_pkl
from sklearn.preprocessing import StandardScaler

import warnings
warnings.filterwarnings('ignore')

class MMSurvivalDataset(Dataset):
    """
        Multi Modal Dataset using RNA-seq data and WSI 
        Used for multi modal survival prediction
    """
    def __init__(self,
                 args, 
                 mode, 
                 fold,
                 slide_col='slide_id'):
        """
        Args:
            - args          : All the arguments given by the user (Obj)
            - mode          : Specifies if we are in 'train' or 'test' mode (str)
            - fold          : Specifies the fold number (int)
            - slide_col     : Name of the column storing the slide ids (str)
        """

        self.mode = mode
        self.fold = fold

        # Data sources
        self.data_source = args.data_source
        self.split_dir = os.path.join(self.data_source, f'splits/{self.fold}/')
        self.data_file = f'{self.mode}.csv'

        # WSI args
        self.slide_col = slide_col
        self.X = None
        self.y = None

        # RNA args
        self.omics_type = args.omics_type
        self.scaler = None

        # Label args
        self.survival_time_col = args.target_col
        self.target_col = args.target_col
        self.censorship_col = args.target_col.split('_')[0] + '_censorship'
        self.n_label_bins = args.n_label_bins
        self.label_bins = None
        
        # Setup and check GT discrete labels in NLL loss
        if self.n_label_bins > 0:
            self.init_disc_labels()

        # Setup and check clinical data
        self.init_df()
        # Setup and check WSI data
        self.init_df_wsi()
        # Setup and check RNA data
        self.init_df_rna()


    def check_data_file(self):
        """Check the clinical survival dataset. """

        # Filter out NAN labels
        is_nan_censorship = self.data_df[self.censorship_col].isna()
        if sum(is_nan_censorship) > 0:
            print('# of NaNs in Censorship col, dropping:', sum(is_nan_censorship))
            self.data_df = self.data_df[~is_nan_censorship]

        is_nan_survival = self.data_df[self.survival_time_col].isna()
        if sum(is_nan_survival) > 0:
            print('# of NaNs in Survival time col, dropping:', sum(is_nan_survival))
            self.data_df = self.data_df[~is_nan_survival]

        # Check that each case_id has only one survival value
        num_unique_surv_times = self.data_df.groupby(self.slide_col)[self.survival_time_col].unique().apply(len)
        assert (num_unique_surv_times == 1).all(), 'Each case_id must have only one unique survival value.'

        # check that all survival values are numeric
        assert not pd.to_numeric(self.data_df[self.survival_time_col], errors='coerce').isna().any(), 'Survival values must be numeric.'

        # check that all survival values are positive
        assert (self.data_df[self.survival_time_col] >= 0).all(), 'Survival values must be positive.'

        # check that all censorship values are binary integers
        assert self.data_df[self.censorship_col].isin([0, 1]).all(), 'Censorship values must be binary integers.'

        # Should be no duplicates in splits file
        assert len(list(self.data_df[self.slide_col].astype(str).unique())) == len(list(self.data_df[self.slide_col].astype(str))), 'There are duplicates in the given splits file...'

    def check_wsi_files(self, feats_wsi_df):
        """ Check that the wsi files are complete and that there are no duplicates. """
        # Should be no missing files
        missing_feats_in_split = pd_diff(self.data_df[self.slide_col], feats_wsi_df[self.slide_col])
        assert len(missing_feats_in_split) == 0, f'Missing Features in Split:\n{missing_feats_in_split}'

        # Should be no duplicates
        duplicates = feats_wsi_df[self.slide_col].duplicated()
        assert duplicates.sum() == 0, f'Features duplicated in data source(s):{feats_wsi_df[duplicates].to_string()}'

    def check_rna_files(self):
        """ Check that the rna files have no duplicates. """
        assert not self.df_rna['case_id'].duplicated().any(), "There are duplicates in the rna data!"

    def init_df(self):
        """ Set up clinical data of this split. """
        split_file = os.path.join(self.split_dir, self.data_file)
        self.data_df = pd.read_csv(split_file)
        self.check_data_file()

    def init_df_wsi(self):
        """ Set up WSI data of this split. """
        # Obtain directory containing the patch features
        self.feat_dir_wsi = os.path.join(self.data_source, "wsi/extracted_res0_5_patch256_uni/feats_h5")
        self.data_df[self.slide_col] = self.data_df[self.slide_col].astype(str)

        # Store feature paths 
        feats_wsi_df = pd.DataFrame([(e.path, os.path.splitext(e.name)[0]) for e in os.scandir(self.feat_dir_wsi)], columns=['fpath', self.slide_col]).reset_index(drop=True)
        self.check_wsi_files(feats_wsi_df)

        # All slide ids to feature paths should have a one-to-one mapping. Raises ValueError if not.
        # Add feature paths to data frame
        self.data_df = self.data_df.merge(feats_wsi_df, how='inner', on=self.slide_col, validate='1:1')
        self.data_df = self.data_df[list(self.data_df.columns[-1:]) + list(self.data_df.columns[:-1])]
    
    def init_df_rna(self):
        """ Set up RNA data of this split. """
        # Read RNA data
        self.feat_dir_rna = os.path.join(self.data_source, f"rna/{self.omics_type}.csv")
        if os.path.isfile(self.feat_dir_rna):
            self.df_rna = pd.read_csv(self.feat_dir_rna, engine='python', index_col=0)
            self.df_rna = self.df_rna.rename(columns={'Unnamed: 0': 'case_id'})
        else:
            raise FileNotFoundError(f"{self.feat_dir_rna} not found!")
        
        # Check for duplicates
        self.check_rna_files()

        # Keep only the patients that are in this split and have WSI data
        case_ids_overlap = overlap_col_df(self.df_rna, self.data_df, 'case_id')
        sample_list = sorted(case_ids_overlap)

        self.data_df = self.data_df[self.data_df['case_id'].isin(sample_list)].reset_index(drop=True)
        self.df_rna = self.df_rna[self.df_rna['case_id'].isin(sample_list)].reset_index(drop=True)
        self.df_rna = self.df_rna.set_index('case_id')

        # Set up hallmark pathways
        self.setup_rna_pathways()

        # Initialize and apply scaler for RNA data
        self.setup_scaler()
        self.apply_scaler()

    def init_disc_labels(self):
        """ Compute discrete time labels from continuous survival times. """
        disc_labels, label_bins = compute_discretization(df=self.data_df,
                                                            survival_time_col=self.survival_time_col,
                                                            censorship_col=self.censorship_col,
                                                            n_label_bins=self.n_label_bins,
                                                            label_bins=self.label_bins)
        self.data_df = self.data_df.join(disc_labels)
        self.label_bins = label_bins
        self.target_col = disc_labels.name

    def get_labels(self, idx):
        """ Get the survival time (days), censorship and target label (either continuous or discrete time labels). """
        labels = self.data_df.loc[idx][[self.survival_time_col, self.censorship_col, self.target_col]]
        return list(labels)
    
    def get_sample_id(self, idx):
        """ Get the sample ID (case_id, slide_id). """
        sample = self.data_df.loc[idx][['case_id', 'slide_id']]
        return list(sample)
    
    def get_all_labels(self):
        """ Get all survival times and censorships for this dataset. """
        cs_list = list(self.data_df.loc[:][self.censorship_col])
        st_list = list(self.data_df.loc[:][self.survival_time_col])
        return np.array(cs_list), np.array(st_list)

    def setup_scaler(self):
        """ Fit or load scaler for RNA data. """
        if self.mode == 'train':
            # Fit the scaler on the training data
            self.scaler = StandardScaler().fit(self.df_rna)
            save_pkl(self.split_dir, f'{self.omics_type}_scaler_fold_{self.fold}.pkl', self.scaler)
        else:
            try:
                # Read the scaler from the pickle file
                print("Load scaler")
                self.scaler = load_pkl(self.split_dir, f'{self.omics_type}_scaler_fold_{self.fold}.pkl')
            except FileNotFoundError:
                print(f"Cannot access the scaler from training. Make sure '{self.split_dir}scaler_fold_{self.fold}.pkl' exists.")
                self.scaler = None  
    
    def apply_scaler(self):
        """ Apply fitted scaler to RNA data. """
        assert not self.scaler == None, "Cannot scale the data, scaler is not defined!"

        cols = self.df_rna.columns
        case_list = self.df_rna.index.values
        self.df_rna = pd.DataFrame(self.scaler.transform(self.df_rna), columns=cols)
        self.df_rna.insert(0, 'case_id', case_list)
        self.df_rna = self.df_rna.set_index('case_id')

    def setup_rna_pathways(self):
        """ Load Hallmarks biological pathways, which serve as the prototypes. """
        signatures = pd.read_csv(os.path.join(os.path.dirname(self.data_source), f"hallmarks_signatures.csv"))
        # Stores which genes are in which pathway (list of genes per pathway)
        self.genes_in_pathways = []
        # Stores the names of the pathways
        self.pathway_names = []
        # Stores the sizes of the pathways
        self.pathway_sizes = []

        # For each pathway 
        for pathway_name in signatures.columns:
            genes = signatures[pathway_name].dropna().unique()
            genes = sorted(series_intersection(genes, self.df_rna.columns))

            # Store the genes involved in this pathway
            self.genes_in_pathways.append(genes)

            # Store the name of the pathway
            self.pathway_names.append(pathway_name)

            # Store the number of genes in the pathway
            self.pathway_sizes.append(len(genes))

    def get_label_bins(self):
        """ Get the time bins for the discrete time labels."""
        return self.label_bins
    
    def get_split_folder(self):
        """ Get the dir of this specific fold. """
        return self.split_dir

    def __len__(self):
        """ Get the number of samples. """
        return len(self.data_df)
    
    def get_whole_rna_dataset(self):
        """ Get the whole rna dataset for all samples - Used for computing SHAP. """
        pathway_summs_all = []

        # For each pathway
        for i in range(len(self.pathway_names)):
            pathway_summary = []
            
            # Obtain the expression values for all genes in this pathway for all samples 
            for j in range(self.__len__()):
                case_id = self.data_df.loc[j]['case_id']
                pathway_summary.append(torch.Tensor(self.df_rna.loc[case_id, self.genes_in_pathways[i]]))
            
            # Expression values for all samples in this pathway
            pathway_summary = torch.stack(pathway_summary)
            pathway_summs_all.append(pathway_summary)

        return pathway_summs_all

    def __getitem__(self, idx):
        # Obtain labels
        survival_time, censorship, label = self.get_labels(idx)
        out = {'survival_time': torch.Tensor([survival_time]),
            'censorship': torch.Tensor([censorship]),
            'label': torch.Tensor([label])}
        
        # Obtain case and slide ids
        case_id = self.data_df.loc[idx]['case_id']
        slide_id = self.data_df.loc[idx][self.slide_col]
        out['case_id'] = case_id
        out['slide_id'] = slide_id


        # Obtain RNA data
        pathway_summary = []
        # For each pathway, obtain the expression values for all genes in this pathway
        for i in range(len(self.pathway_names)):
            pathway_summary.append(torch.Tensor(self.df_rna.loc[case_id, self.genes_in_pathways[i]]))
        
        out['rna'] = pathway_summary
    
        # Obtain WSI data
        if self.X is not None:
            # We already created the unsupervised slide embedding
            out['img'] = self.X[idx]
        else:
            # Else obtain the image features, used to create the unsupervised slide embedding
            feat_path = self.data_df.loc[idx]['fpath']
            with h5py.File(feat_path, 'r') as f:
                features = f['features'][:]

            if len(features.shape) > 2:
                assert features.shape[0] == 1, f'{features.shape} is not compatible! It has to be (1, numOffeats, feat_dim) or (numOffeats, feat_dim)'
                features = np.squeeze(features, axis=0)
            
            features = torch.from_numpy(features)
            out['img'] = features

        return out
    







    



        



