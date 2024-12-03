# Prepping data for STG-NF

We provide code to prepare the 3 biological datasets used in the paper into the format expected by STG-NF. Train/test splits used for FishLarvae1 and Meerkat are provided in `./FishLarvae1/metadata_csvs/` and `./Meerkat/metadata_csvs/`. For PoseR we used the partitions specified in their data.

You'll need to download the data yourself from their corresponding repos and adhere to any licensing restrictions specified by the dataset creators.

[FishLarvae1 data](https://data.mendeley.com/datasets/8sb4ywbx7f/1),     [Paper by Johnson et al. 2020](https://www.cell.com/current-biology/fulltext/S0960-9822(19)31465-4)

[PoseR data](https://doi.org/10.5281/zenodo.7807968),        [Pre-print by Mullen et al. 2023](https://doi.org/10.1101/2023.04.07.535991)

[Meerkat data](https://doi.org/10.5061/dryad.7q294p8),       [Paper by Chakaravaty et al. 2019](https://besjournals.onlinelibrary.wiley.com/doi/full/10.1111/2041-210X.13172)

## Synthetic dataset 
Our synthetic dataset is a nice little sandbox where you can test how your method reacts to different levels of behavior similarity and behavior rarity.
You can get it at: [10.5281/zenodo.14266407](https://doi.org/10.5281/zenodo.14266407)
## Example of dataset directory structure
Our code creates for a given dataset the following data structure:

```
.
├── [FishLarvae1 OR PoseR OR Meerkat]
│   ├── metadata_csvs
│   └── training
│       ├── test
│       └── train
|           |──normal
|           └──abnormal
```
Our code separates the abnormal/normal classes in the train because this was necessary in early iterations of our code for the supervised variant of the STG-NF.
Importantly, the unsupervised version is trained on all contents of the `train` folder, not just the `normal` category, as you would expect from a properly unsupervised setting.
The metadata_csvs folder contains csvs with the data splits and file metadata.

## Data prep for FishLarvae1
Replace "/path/to/save/output/" with the path to where you want to save the dataset, and replace the "/path/to/raw/files/" with where you downloaded the raw files to.
Note that this will create the dataset with our estimate of the eye coordinates, which is what we used in our paper. If you'd like to opt-out of this use the `--no_eyes` flag.

```
python ./data_prep/FishLarvae1/fishlarvae1_data_prep.py /path/to/save/output/FishLarvae1 /path/to/raw/files
 
```
## Data prep for PoseR
Replace "/path/to/save/output/" with the path to where you want to save the dataset, and replace the "/path/to/raw/files/" with where you downloaded the raw files to. Note that this automatically removes outlier samples that were overly noisy due to the pose detector (and preventing us from benchmarking our pipeline for rare behaviors properly). If you'd like to keep these outliers use the flag `--with_outliers`.

```
python ./data_prep_testing/PoseR/poser_data_prep.py /path/to/save/output/PoseR /path/to/raw/files
```

## Data prep for Meerkat
Replace "/path/to/save/output/" with the path to where you want to save the dataset, and replace the "/path/to/raw/files/" with where you downloaded the raw files to.

```
python ./data_prep_testing/Meerkat/meerkat_data_prep.py /path/to/save/output/Meerkat /path/to/raw/files
```