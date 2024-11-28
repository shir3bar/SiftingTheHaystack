# 


The official PyTorch implementation of the paper [**"Sifting through the haystack - efficiently finding rare animal behaviors in large-scale dataset"**]().





## Getting Started

This code was tested on `Ubuntu 20.04.4 LTS` and requires:
* Python 3.8
* conda3 or miniconda3
* CUDA capable GPU (one is enough)

### Setup Conda Environment:
```
git clone https://github.com/@@
cd @@@

# Conda environment setup
conda env create -f environment.yml
conda activate STG-NF
```

### Dataset Directory Structure
```
.
├── PoseR
│   ├── metadata_csvs
│   └── training
│       ├── test
│       └── train
|           |──normal
|           └──abnormal
├── models
│   └── STG_NF
|   |   ├── graph.py
└── utils

```

### Data Prep
We provide codes to save data into the format expected by STG-NF and divide it into the partitions used in our paper. 
You'll need to download the data seperately from the corresponding paper's repositories.
[FishLarvae1](), [PoseR](), [Meerkat]()
Full details in the data_prep folder.

### Custom Dataset
To get started with your own data, you'll need to customize a data prep script to process your input data into the required data formats `.npy` and `.json`.
You'll need to add a skeleton layout to the `./models/STG-NF/graph.py`
And, you'll probably want to edit the `dataset.py` file to add some metadata regarding your dataset. 

```
python train_eval.py --dataset Custom --model_layout your_layout --data_dir /path/to/your/data
```


## Training/Testing
Training and Evaluating is run using:
```
python train_eval.py --dataset [Larvae2019\PoseR\Meerkat2019\SimulatedWave]
```
Note that the dataset called FishLarvae1 in the paper is referred to as Larvae2019 in the code.

We provide a bash script with all the experiments we ran for the paper, including the arguments used: `./paper_experiments/unsupervised_experiments.sh`

Evaluation of our pretrained model can be done using:

FishLarvae1:
```
python train_eval.py --dataset Larvae2019 --model_layout larvaeposeeyes --seg_len 8 --checkpoint @@@checkpoints/ShanghaiTech_85_9.tar
```
PoseR:
```
python train_eval.py --dataset PoseR --model_layout poser --seg_len 8 --checkpoint @@checkpoints/UBnormal_unsupervised_71_8.tar 
```
Meerkat:
```
python train_eval.py --dataset Meerkat2019  --model_layout meerkat_connected --seg_len 8 --checkpoint @@checkpoints/UBnormal_supervised_79_2.tar
```

## Rarity experiments
To recreate all of the results of the rarity experiments conducted in the paper, use the bash script: `./paper_experiments/rarity_exps.sh`. Note it takes a very long time to run.
Example call for a single experiment:

```
python rarity_rarefaction_final_exp.py --dataset Larvae2019 --data_dir /path/to/data --model_layout larvaeposeeyes --epochs 6 --classifier 
```

## Acknowledgments
Our code is an adaptation of the original STG-NF repository:
- [Normalizing Flows for Human Pose Anomaly Detection](https://github.com/orhir/STG-NF)
Additionally, their code is based on code from:
- [Graph Embedded Pose Clustering for Anomaly Detection](https://github.com/amirmk89/gepc)
- [Glow](https://github.com/y0ast/Glow-PyTorch)


## Citation
If you find this useful, please cite this work as follows:
```
@article{bar2025sifting,
  title = {Sifting through the haystack - efficiently finding rare animal behaviors in large-scale datasets},
  author = {Bar, Shir, Hirschorn, Or, Holzman, Roi and Avidan, Shai},
  journal={Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision},
  year = {2025},
}
```
## License
This code is distributed under a [Creative Commons LICENSE](LICENSE).

Note that our code depends on other libraries and uses datasets that each have their own respective licenses that must also be followed.