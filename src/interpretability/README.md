# Interpretability 

This folder contains all files related to the interpretability of DIMAF, including:
- Computation and interpretation of SHAP values
- Visualization of WSI prototypes
- Visualization of RNA prototypes

We are currently working on visualizations for further interpretability of DIMAF to provide a deeper understanding of multimodal interactions, both intra- and inter-modal. This upcoming feature will offer more robust insights into how different data modalities contribute to survival prediction. **Coming soon – stay tuned!**

![Overview of DIMAF](../../docs/dimaf.png)

## SHAP
- `compute_shap.py` – Script that computes SHAP values for the multimodal disentangled representations. SHAP values can be calculated per feature (Z^p_, see figure above) or after feature aggregation in the representations (Z_, see figure above). Used by `main_survival.py`; see the README in `src` for more information.
- `visualize_shap.ipynb` – Notebook to visualize the normalized mean absolute SHAP values of the disentangled representations. Use this notebook after computing the SHAP values.

## Visualize WSI prototypes
- `visualize_wsi_feats.ipynb` – Notebook for visualizing the features of one WSI: # TODO
    - Visualizing the mixture proportion distribution
    - Visualizing the prototypes by using the closest patches # TODO
- `visualize_wsi_feats.py` – Code for visualizing the features of a group (train/test samples) of WSIs:
    - Visualizing the mixture proportion distribution (args.task == pt_assignment_{train OR test})
    - Visualizing the prototypes by using the closest patches (args.task == pt_vis_{train OR test})
    - Visualizing the prototypes by using the closest patches, with max one patch per prototype per WSI (args.task == pt_assignment_{train OR test}_spec)


## Visualize RNA prototypes
`plot_pathways.ipynb` – With this notebook, you can visualize the pathway features by
**(1)**  Plotting the mean pathway expression of all samples per pathway together with the predicted risk score (group level).
**(2)**  Plotting the gene expression distribution per pathway of one sample (sample level).