import random
import math
import numpy as np
import os
import torch
from torch.utils.tensorboard import SummaryWriter
from models.STG_NF.model_pose import STG_NF
from models.training import Trainer
from utils.data_utils import trans_list
from utils.optim_init import init_optimizer, init_scheduler
from args import create_exp_dirs
from args import init_parser, init_sub_args
from dataset import  get_dataset_and_loader
from utils.train_utils import dump_args, init_model_params
from utils.scoring_utils import score_dataset, score_dataset_classifier
from utils.train_utils import calc_num_of_params
from models.STG_NF.stgcn import ST_GCN_18
import pandas as pd
from datetime import datetime
import copy

SAMPLE_SIZES = {'Larvae2019':[30, 200, 1000,# 3000, 6000, #10000, 30000, 100000,
                              127379],#changed to match abnormal+normal in train
                        'PoseR':[30, 200, 1000,
                                 # 3000,# 10000,
                                    26854],#28912],Poser with outliers has 28912 samples, Poser without outliers has 21312 samples
                        'Meerkat2019':[30, 200, 1000,# 3000, 6000, 
                                       61912],
                        'SimulatedWave':[30, 200, 1000,
                                         31500]
                        }

def train_supervised(args, trans_list=trans_list):
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    torch.manual_seed(args.seed)
        #torch.set_num_threads(1)
    np.random.seed(args.seed)
    train2_dataset, train2_loader = get_dataset_and_loader(args, trans_list=trans_list, only_test=False)
    model_args_sup = init_model_params(args, train2_dataset)
    if args.classifier:
        model2 = ST_GCN_18(in_channels=2, num_class=args.num_classes, layout=args.model_layout, 
                          max_hop=args.max_hops, strategy=args.adj_strategy)
        args.ckpt_dir = os.path.join(args.ckpt_dir,'classifier')
    else:
        model2 = STG_NF(**model_args_sup)
        args.ckpt_dir = os.path.join(args.ckpt_dir,'supervised_NF')
    os.makedirs(args.ckpt_dir,exist_ok=True)
    num_of_params = calc_num_of_params(model2)
    print(f'Model has {num_of_params} parameters')
    trainer = Trainer(args, model2, train2_loader['train'], train2_loader['test'],
                      optimizer_f=init_optimizer(args.model_optimizer, lr=args.model_lr),
                      scheduler_f=init_scheduler(args.model_sched, lr=args.model_lr, epochs=args.epochs))
    writer = SummaryWriter()
    trainer.train(log_writer=writer, save_every=args.save_every)
    dump_args(args, args.ckpt_dir)
    normality_scores = trainer.test()
    current_time = datetime.now().strftime("%d.%m %H:%M:%S")
    if args.classifier:
        auroc, auprc, scores = score_dataset_classifier(normality_scores, list(train2_dataset["test"].metadata), args=args)

        # Logging and recording results
        print(f"\n----------------------{current_time}-------------------------")
        print("\033[92m Done with {}% AuROC and {}% AuPRC for {} samples\033[0m".format(auroc * 100, auprc * 100, scores.shape[0]))
        print("-------------------------------------------------------\n\n")        
    else:
        auroc,auprc, scores = score_dataset(normality_scores, list(train2_dataset["test"].metadata), args=args)
        # Logging and recording results
        print(f"\n----------------------{current_time}-------------------------")
        print("\033[92m Done with {}% AuROC and {}% AuPRC for {} samples\033[0m".format(auroc * 100, auprc * 100, scores.shape[0]))
        print("-------------------------------------------------------\n\n")

def main():
    parser = init_parser()
    args = parser.parse_args()
    if args.seed == 999:  # Record and init seed
        args.seed = torch.initial_seed()
        np.random.seed(0)
    else:
        random.seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True
        torch.manual_seed(args.seed)
        #torch.set_num_threads(1)
        np.random.seed(args.seed)

    args, model_args = init_sub_args(args)
    assert os.path.exists(args.exp_dir), f'Experiment directory {args.exp_dir} does not exist'
    args.ckpt_dir = args.exp_dir
    #args.ckpt_dir = create_exp_dirs(args.exp_dir, dirmap=os.path.join(args.dataset))
    normal_dir = os.path.join(args.data_dir,'training','train','normal')
    abnormal_dir = os.path.join(args.data_dir,'training','train','abnormal')
    sample_sizes = SAMPLE_SIZES[args.dataset]
    # last sample size is the total number of samples, which is divided by two cause each samples has a data and metadata file in the folder
    # we're overriding because some datasets were cleaned and the numbers above are not accurate
    sample_sizes[-1] = int(len(os.listdir(normal_dir))/2 + len(os.listdir(abnormal_dir))/2) 
    args.subsample_size = sample_sizes[-1]
    rarities = sorted([float(f.strip('rarity_')) for f in os.listdir(args.exp_dir) if f.startswith('rarity')])
    for rarity_percent in rarities:
        print('Working on rarity:',rarity_percent)
        unsupervised_dir = os.path.join(args.ckpt_dir,f'rarity_{rarity_percent}',f'unsupervised_subsample_{args.subsample_size}')
        # Load unsupervised results
        grouped_score_df = pd.read_csv(os.path.join(unsupervised_dir,'grouped_res.csv'))
        grouped_score_df =  grouped_score_df.sort_values(by='mean_scores', ascending=True).reset_index(drop=True)
        #grouped_score_df = grouped_score_df.loc[args.discard_abnormal:] # discard the top args.discard_abnormal samples to get rid of outlies
        supervised_args = copy.deepcopy(args)
        supervised_args.rarity_percent = rarity_percent
        supervised_args.classifier = True
        prev_size = 0
        file_dict = {'normal':[],'abnormal':[]}
                        #,'ours_v2':{'normal':[],'abnormal':[]}}
        ablat_df = grouped_score_df.copy()
        normal_save = pd.DataFrame()
        abnormal_save = pd.DataFrame()
        for subsample_size in sample_sizes[:-1]:
            supervised_args, _ = init_sub_args(supervised_args)
            supervised_args.pose_path_train_abnormal = None
            # select files for traditional dataset:
            # We sample the amount of file we need to add on top of the previous subsample size
            # select files for Ours V1 dataset:
            selected_abnormal = ablat_df.head(int(subsample_size-prev_size))
            ablat_df = ablat_df.drop(selected_abnormal.index)
            
            review_filelist = selected_abnormal.apply(lambda row: os.path.join(args.data_dir,'training','train',row.label,row.filename+'.npy'),axis=1).tolist()
            abnormal_filelist = [n for n in review_filelist if 'abnormal' in n] # review abnormal and remove normal samples
            file_dict['abnormal'] = file_dict['abnormal']+abnormal_filelist

            extra_normal = [n for n in review_filelist if 'abnormal' not in n] # review abnormal and remove normal samples
            normal_sample_size = int((subsample_size-prev_size)*args.normal_ratio)
            selected_normal = ablat_df.tail(normal_sample_size)
            ablat_df = ablat_df.drop(selected_normal.index)
            abnormal_save = pd.concat([abnormal_save,selected_abnormal[selected_abnormal.label=='abnormal']])
            normal_save = pd.concat([normal_save,selected_normal,
                                     selected_abnormal[selected_abnormal.label=='normal']])
            prev_size = subsample_size
            normal_filelist = selected_normal.apply(lambda row: os.path.join(args.data_dir,'training','train',row.label,row.filename+'.npy'),axis=1).tolist() 
            normal_filelist = normal_filelist + extra_normal
            file_dict['normal'] = file_dict['normal']+normal_filelist
            print(f'@@@@@@@@@@@@@@ Starting second training phase - subsample size {subsample_size} @@@@@@@@@@@@@@@@@@')
            supervised_args.file_list['train'] = file_dict['normal']
            supervised_args.file_list['train_abnormal'] = file_dict['abnormal']
            supervised_args.ckpt_dir = os.path.join(args.ckpt_dir,f'rarity_{rarity_percent}','ablation',f'subsample_{subsample_size}')
            os.makedirs(supervised_args.ckpt_dir,exist_ok=True)
            train_supervised(supervised_args)
            abnormal_save.to_csv(os.path.join(supervised_args.ckpt_dir,f'top_{len(file_dict["abnormal"])}_abnormal.csv'),index=False)
            normal_save.to_csv(os.path.join(supervised_args.ckpt_dir,f'top_{len(file_dict["normal"])}_normal.csv'),index=False)


   
if __name__ == '__main__':
    main()
