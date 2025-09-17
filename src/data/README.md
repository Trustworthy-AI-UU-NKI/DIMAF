# Data 

This folder contains all data files, from preprocessing to dataset classes.  

- `data_files/` – Contains all data files:  
  - `hallmarks_signatures.csv`: gene sets for all pathways.  
  - Cohort-specific folders (`tcga_blca`, `tcga_brca`, `tcga_luad`, `tcga_kirc`) containing:  
    - Train/test splits for 5-fold cross-validation (stratified by site).  
    - Clinical data of all samples (`clinical_data_all_filtered.csv`).  
- `preprocess_TCGA_rna.py` – Script with all steps for preprocessing the RNA data.  
- `preprocess_TCGA_rna.ipynb` – Notebook version with additional visualization options.  
- `mm_survival_dataset.py`– Dataset class for the multimodal data used during training and testing.  

## Prepare the data

The data should be structured the following.
```
tcga_cohort/ 
    ├─ splits/
        ├─ 0/
        ├─ 1/
        ├─ 2/
        ├─ 3/
        └─ 4/
            ├─ train.csv
            └─ test.csv
    ├─ rna/
        ├─ HiSeqV2_PANCAN_BRCA
        └─ rna_data.csv
    ├─ wsi/
        ├─ images/
        └─ extracted_res0_5_patch256_uni/
            └─ feats_h5/
    └─ clinical_data_all.csv
```

- The `splits` folder stores sample, slide, and clinical endpoints for 5-fold cross-validation.  
- `clinical_data_all.csv` contains all samples with additional clinical information (tumor type, etc.).  

We now need to obtain **RNA data** and **WSI data**.  
After activating the conda environment, navigate to this folder (`src/data`).

### RNA
Download and preprocess the RNA-seq data for each cohort: 

#### TCGA-BRCA
```
curl -o data_files/tcga_brca/rna/HiSeqV2_PANCAN_BRCA.gz https://tcga-xena-hub.s3.us-east-1.amazonaws.com/download/TCGA.BRCA.sampleMap%2FHiSeqV2_PANCAN.gz
gunzip data_files/tcga_brca/rna/HiSeqV2_PANCAN_BRCA.gz
python preprocess_TCGA_rna.py --data brca --name rna_data
```

#### TCGA-BLCA
```
curl -o data_files/tcga_blca/rna/HiSeqV2_PANCAN_BLCA.gz https://tcga-xena-hub.s3.us-east-1.amazonaws.com/download/TCGA.BLCA.sampleMap%2FHiSeqV2_PANCAN.gz
gunzip data_files/tcga_blca/rna/HiSeqV2_PANCAN_BLCA.gz
python preprocess_TCGA_rna.py --data blca --name rna_data
```

#### TCGA-LUAD
```
curl -o data_files/tcga_luad/rna/HiSeqV2_PANCAN_LUAD.gz https://tcga-xena-hub.s3.us-east-1.amazonaws.com/download/TCGA.LUAD.sampleMap%2FHiSeqV2_PANCAN.gz
gunzip data_files/tcga_luad/rna/HiSeqV2_PANCAN_LUAD.gz
python preprocess_TCGA_rna.py --data luad --name rna_data
```

#### TCGA-KIRC
```
curl -o data_files/tcga_kirc/rna/HiSeqV2_PANCAN_KIRC.gz https://tcga-xena-hub.s3.us-east-1.amazonaws.com/download/TCGA.KIRC.sampleMap%2FHiSeqV2_PANCAN.gz
gunzip data_files/tcga_kirc/rna/HiSeqV2_PANCAN_KIRC.gz
python preprocess_TCGA_rna.py --data kirc --name rna_data
```

### WSI

For the whole slide images, follow these steps:
- **Download the WSIs**: from [The GDC data portal](https://portal.gdc.cancer.gov)
- **Ensure unified resolution:** images/patches must have a resolution of 0.5 micrometer per pixel.
- **Process & store the WSIs:** Segment & patch the images, and extract features:
    - We used the [CLAM](https://github.com/mahmoodlab/CLAM) framework and the [UNI](https://github.com/mahmoodlab/UNI) WSI foundation model to extract patch features.
    - Each WSI should be represented as a set of .h5 patch feature files stored in `feats_h5/`.



## Tips & Notes
- Make sure all folder names match exactly for scripts to work.
- We used [UNI](https://github.com/mahmoodlab/UNI) to extract features from 256x256 patches at 0.5 μm resolution, stored in `extracted_res0_5_patch256_uni/feats_h5/`.
- As a substitute of CLAM, [TRIDENT](https://github.com/mahmoodlab/TRIDENT) can be used with a variety of patch encoders. However, the code has not been tested for this yet.
- For the visualizaiton of the WSIs, store the WSI files (`.svs`) in the `images` subfolder