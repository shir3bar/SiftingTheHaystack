python rarity_exp.py --dataset PoseR --data_dir /path/to/PoseR --seed 42 --model_layout poser --adj_strategy spatial --seg_stride 4 --K 4 --model_lr 0.01 --model_lr_decay 0.5 --epochs 2 --model_hidden_dim 16 --classifier --normal_ratio 1

python rarity_exp.py --dataset PoseR --data_dir /path/to/PoseR --seed 16 --model_layout poser --adj_strategy spatial --seg_stride 4 --K 4 --model_lr 0.01 --model_lr_decay 0.5 --epochs 2 --model_hidden_dim 16 --classifier --normal_ratio 1

python rarity_exp.py --dataset PoseR --data_dir /path/to/PoseR --seed 82000 --model_layout poser --adj_strategy spatial --seg_stride 4 --K 4 --model_lr 0.01 --model_lr_decay 0.5 --epochs 2 --model_hidden_dim 16 --classifier --normal_ratio 1

python rarity_exp.py --dataset Larvae2019 --data_dir /path/to/FishLarvae1 --seed 42 --model_layout larvaeposeeyes --adj_strategy spatial --seg_stride 1 --K 4 --model_hidden_dim 16 --max_hops 8 --model_lr 0.01 --model_lr_decay 0.5 --epochs 2 --classifier --normal_ratio 1

python rarity_exp.py --dataset Larvae2019 --data_dir /path/to/FishLarvae1 --seed 16 --model_layout larvaeposeeyes --adj_strategy spatial --seg_stride 1 --K 4 --model_hidden_dim 16 --max_hops 8 --model_lr 0.01 --model_lr_decay 0.5 --epochs 2 --classifier --normal_ratio 1

python rarity_exp.py --dataset Larvae2019 --data_dir /path/to/FishLarvae1 --seed 82000 --model_layout larvaeposeeyes --adj_strategy spatial --seg_stride 1 --K 4 --model_hidden_dim 16 --max_hops 8 --model_lr 0.01 --model_lr_decay 0.5 --epochs 2 --classifier --normal_ratio 1

python rarity_exp.py --dataset Meerkat2019 --data_dir /path/to/Meerkat --seed 42 --model_layout meerkat_connected --seg_stride 2 --K 4  --epochs 2 --classifier --model_lr 0.01 --model_lr_decay 0.5 --normal_ratio 1

python rarity_exp.py --dataset Meerkat2019 --data_dir /path/to/Meerkat --seed 16 --model_layout meerkat_connected --seg_stride 2 --K 4  --epochs 2 --classifier --model_lr 0.01 --model_lr_decay 0.5 --normal_ratio 1

python rarity_exp.py --dataset Meerkat2019 --data_dir /path/to/Meerkat --seed 82000 --model_layout meerkat_connected --seg_stride 2 --K 4  --epochs 2 --classifier --model_lr 0.01 --model_lr_decay 0.5 --normal_ratio 1

