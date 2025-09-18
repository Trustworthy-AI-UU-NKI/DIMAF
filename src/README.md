# Instructions on running DIMAF
## 1. Data preprocessing
First , prepare the data such that DIMAF can use it. Please see the [README](data/README.md) in the `data` folder for detailed instructions on how to download, preprocesss and structure the data. 
Currently, it supports the 4 [TCGA](https://portal.gdc.cancer.gov) data cohorts used in the paper, i.e., BRCA, BLCA, LUAD and KIRC. However, it can easily be adapted to other cohorts also. After this step, cd back to the current directory (`src`).


## 2. Constructing initial histology prototypes
DIMAF obtains initial prototypes as means for the mixture distributions by clustering the train data and taking the cluster centres as initial means. To construct these initial mixture distribution means (prototypes), run 
```
python main_prototype.py --data_source data/data_files/tcga_brca \
                         --wsi_dir wsi/extracted_res0_5_patch256_uni/feats_h5/ \
                         --mode faiss \
                         --n_proto 16 
```
**List of arguments:**
- `data_source`: Path to the dataset.
- `wsi_dir`: Subpath to extracted WSI features.
- `mode`: Clustering method (faiss for GPU, kmeans for CPU, default = faiss).
- `n_proto`: Number of prototypes (default = 16).

For all possible arguments, see `main_prototype.py`. The default values reproduce the DIMAF settings used in our paper.


## 3. Training DIMAF for survival prediction
To train DIMAF on the TCGA-BRCA dataset, run
```
python main_survival.py --data_source data/data_files/tcga_brca/ \
                        --max_epochs 30 \
                        --proto_file prototypes_DIMAF/prototypes_16_type_faiss_init_3_nr_100000.pkl \
                        --task dss_survival_brca \
                        --exp_code DIMAF \
                        --loss_fn cox_distcor \
                        --omics_type rna_data \
                        --w_dis 7 \
                        --w_surv 1 \
                        --mode train
```
**List of arguments:**
- `data_source`: Path of the data, default is _data/data_files/tcga_brca/_. Adjust to use other cohorts by replacing `brca` with `blca`, `kirc` or `luad`.
- `max_epochs`: Number of epochs the model will train for, default is 30.
- `proto_file`: Path to the file which stores the obtained prototypes of step 2.
- `task`: Defines the task of the model. Options are [`dss_survival_brca`, `dss_survival_blca`, `dss_survival_luad`, `dss_survival_kirc`].
- `exp_code`: Name of the experiment.
- `loss_fn`: Loss function used, default is cox_distcor (Cox + DC loss).
- `omics_type`: Name of the rna data file.
- `w_dis`: Weight of the disentanglement loss in the total loss, default is 7.
- `w_surv`: Weight of the survival loss in the total loss, default is 1.
- `mode`: Determines the mode of the experiment. Options are train, test, train_test and shap.

For all possible arguments, see `main_survival.py`. The default values reproduce the DIMAF settings used in our paper.


## 4. Testing DIMAF for survival prediction
To test a trained model, run the same command as in training but set `--mode test` and add `--return_attn` to save attention matrices for interpretability:
```
python main_survival.py --data_source data/data_files/tcga_brca/ \
                        --proto_file prototypes_DIMAF/prototypes_16_type_faiss_init_3_nr_100000.pkl \
                        --task dss_survival_brca \
                        --exp_code DIMAF \
                        --loss_fn cox_distcor \
                        --omics_type rna_data \
                        --w_dis 7 \
                        --w_surv 1 \
                        --mode test \
                        --return_attn
```

To evaluate survival and disentanglement performance across folds (mean ± std), use `get_results.ipynb`.

## 5. Running SHAP for feature importance of the disentangled representations
To compute SHAP values, run the same command as training but set `--mode shap` and include `--shap_refdist_n` and `--explainer`:
- `shap_refdist_n`: Size of the background samples (training set). Recommended:
    - BRCA: 512
    - BLCA: 256
    - LUAD: 320
    - KIRC: 192
- `explainer`: Explanation technique (_shap_ by default, also supports _eg_ for Expected Gradients).

Example for BRCA:

```
python main_survival.py --data_source data/data_files/tcga_brca/ \
                        --proto_file prototypes_DIMAF/prototypes_16_type_faiss_init_3_nr_100000.pkl \
                        --task dss_survival_brca \
                        --exp_code DIMAF \
                        --loss_fn cox_distcor \
                        --omics_type rna_data \
                        --w_dis 7 \
                        --w_surv 1 \
                        --mode shap \
                        --explainer shap \
                        --shap_refdist_n 512
```

### 6. Visualization and further interpretability of DIMAF
For unimodal feature visualization and interpretation of the obtained SHAP values, see the [README](interpretability/README.md) in  the `interpretability` folder.