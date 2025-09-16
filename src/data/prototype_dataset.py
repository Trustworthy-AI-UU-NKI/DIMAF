import os
import torch
import pandas as pd
import h5py
import numpy as np

from torch.utils.data import Dataset
from utils.data_utils import pd_diff

class WSIDataset(Dataset):
    """
        WSI Cluster Dataset
        Used for creating the prototypes
        Adapted from https://github.com/mahmoodlab/MMP/blob/main/src/wsi_datasets/wsi_prototype.py
        """
    def __init__(self,
                 wsi_file_path,
                 split_file,
                 slide_col='slide_id'):
        """
        Args:
            - wsi_file_path : Dir of the WSI features (str)
            - split_file    : Dir of the split file (str)
            - slide_col     : Name of the column storing the slide ids (str)
        """

        # Dataframe with the splits and all the clinical data
        self.data_df = pd.read_csv(split_file)
        self.slide_col = slide_col
        self.feat_dir = wsi_file_path

        # Make sure sample col is a string
        self.data_df[slide_col] = self.data_df[slide_col].astype(str)

        # Check the dataframe and features directory
        self.check_df_file()
        # Obtain the paths of the features
        self.obtain_feat_paths()
        
    def obtain_feat_paths(self):
        """ Obtain paths of features used for creating the prototypes. """
        self.feats_df = pd.DataFrame([(e.path, os.path.splitext(e.name)[0]) for e in os.scandir(self.feat_dir)], columns=['fpath', self.slide_col]).reset_index(drop=True)

        self.check_wsi_files()
        self.data_df = self.data_df[list(self.data_df.columns[-1:]) + list(self.data_df.columns[:-1])]
    
    def check_df_file(self):
        """ Check that the dataframe and features directory are correct. """
        assert 'Unnamed: 0' not in self.data_df.columns
        assert 'feats_h5' in self.feat_dir
        assert len(list(self.data_df[self.slide_col].astype(str).unique())) == len(list(self.data_df[self.slide_col].astype(str))), 'There are duplicates in the given splits file...'
    
    def check_wsi_files(self):
        """ Check that the wsi files are complete and that there are no duplicates """
        # Should be no missing files
        missing_feats_in_split = pd_diff(self.data_df[self.slide_col], self.feats_df[self.slide_col])
        assert len(missing_feats_in_split) == 0, f'Missing Features in Split:\n{missing_feats_in_split}'

        # All slide ids to feature paths should have a one-to-one mapping. Raises ValueError if not.
        # Add feature paths to data frame
        self.data_df = self.data_df.merge(self.feats_df, how='left', on=self.slide_col, validate='1:1')
    
        duplicates = self.feats_df[self.slide_col].duplicated()
        assert duplicates.sum() == 0, f'Features duplicated in data source(s):{self.feats_df[duplicates].to_string()}'

        print("Dataset check is complete!")

    def __len__(self):
        """ Get the total number of samples """
        return len(self.data_df)
    
    def __getitem__(self, idx):
        # Get the features for a given index
        feat_path = self.data_df.loc[idx]['fpath']
        with h5py.File(feat_path, 'r') as f:
            features = f['features'][:]

        # Check the shape of the features
        if len(features.shape) > 2:
            assert features.shape[0] == 1, f'{features.shape} is not compatible! It has to be (1, numOffeats, feat_dim) or (numOffeats, feat_dim)'
            features = np.squeeze(features, axis=0)
        
        features = torch.from_numpy(features)
        return features