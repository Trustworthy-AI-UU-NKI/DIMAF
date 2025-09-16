import pandas as pd
import numpy as np

def compute_discretization(df, survival_time_col='dss_survival_days', censorship_col='dss_censorship', n_label_bins=4, label_bins=None):
    """
    Compute discrete survival prediction time labels.
    Obtained from https://github.com/mahmoodlab/MMP/blob/main/src/wsi_datasets/wsi_survival.py
    """
    if label_bins is not None:
        assert len(label_bins) == n_label_bins + 1
        q_bins = label_bins
    else:
        uncensored_df = df[df[censorship_col] == 0]
        disc_labels, q_bins = pd.qcut(uncensored_df[survival_time_col], q=n_label_bins, retbins=True, labels=False)
        q_bins[-1] = 1e6  # set rightmost edge to be infinite
        q_bins[0] = -1e-6  # set leftmost edge to be 0

    disc_labels, q_bins = pd.cut(df[survival_time_col], bins=q_bins,
                                retbins=True, labels=False,
                                include_lowest=True)
    
    assert isinstance(disc_labels, pd.Series) and (disc_labels.index.name == df.index.name)
    disc_labels.name = 'disc_label'
    return disc_labels, q_bins

def pd_diff(df_1, df_2):
    """ Returns set difference of two pd.Series. """
    return pd.Series(list(set(df_1).difference(set(df_2))), dtype='O')

def overlap_col_df(df_1, df_2, col):
    """ Returns overlap of rows in a specific column. """
    return np.intersect1d(np.unique(df_1[col].values), np.unique(df_2[col].values))

def series_intersection(s1, s2):
    """ Return intersection of two sets. """
    return pd.Series(list(set(s1) & set(s2)))
