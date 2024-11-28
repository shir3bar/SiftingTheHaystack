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

SAMPLE_SIZES = {'Larvae2019':[30, 60, 100, 300, 600, 1000,# 3000, 6000, #10000, 30000, 100000,
                              127379],#c
                        'PoseR':[30, 60, 100, 300, 600, 1000,
                                 # 3000,# 10000,
                                    26854],#28912],Poser with outliers has 28912 samples, Poser without outliers has 21312 samples
                        'Meerkat2019':[30, 60, 100, 300, 600, 1000,# 3000, 6000, 
                                       61912],
                        'SimulatedWave':[30,60,100,300,600,1000,
                                         31500]
                        }
########### Rarity experiments ##############
# In these experiments we set out to test the effectiveness of using STG-NF for finding abnormal samples, and training downstream classifiers using them.
# We repeat the tests using different underlying frequencies of the behavior of interest in the data (i.e., different 'rarities') and with different labeling budgets.
# For each rarity level, we compare our approach of selecting data using the STG-NF likelihood scores to the traditional approach of randomly/naively selecting data for labeling.
# We compare the methods using different labeling budgets specified above in "SAMPLE_SIZES".
# We sequentially increase the size of the dataset, adding additional samples to the existing ones as the budget grows, 
# but training a new classifier for each dataset size (thus we're not dealing with an active learning scenario).
# We start at (roughly) the baseline rarity of the dataset (i.e., the observed frequency of the "abnormal" or rare behaviors) and reduce it by half till
# the number of samples in the entire dataset is VERY small. We keep the test set stable at the baseline rarity of the dataset, 
# to allow easy comparison between different rarities (AuPRC, our chosen metric, is sensitive to data imbalance).
# Note that this experiment takes a long time to run on a single standard GPU.
# Experiment is saved in the following folder structure:
#├── exp_root
#|   ├── rarity_$baseline_rarity_percent$
#|   │   ├── subsample_30
#|   |   ...
#|   │   ├── subsample_1000
#|   │   └── unsupervised_$ds_size$
#|   |...
#|   ├── rarity_$lowest_rarity_percent$
#|   |  ...
#############################################

def train_supervised(args, trans_list=trans_list):
    '''
    Train a supervised model (either classifier or supervised STG-NF) on a selected dataset. 
    All parameters necessary for training, including the files for the training dataset, are specified in the args variable.
    The dataset file are set in the file_list dictionary in the args object (see below in main).
    The function has no output arguments but saves model checkpoints and evaluation results in args.exp_dir.
    '''
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
    writer = SummaryWriter(log_dir=args.ckpt_dir)
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
    '''
    Run the main loop of the rarity experiment.
    See details regarding the experiment and the results' folder structure above.
    '''
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
    if args.dataset== "SimulatedWave":
        # separate results according to overlap between behaviors (data stds), for the simulated dataset
        std_rare = '_'.join(args.data_dir.split('_')[-2:])
        args.ckpt_dir = create_exp_dirs(args.exp_dir, dirmap=os.path.join(args.dataset,'rarity_exps',std_rare))
    else:
        args.ckpt_dir = create_exp_dirs(args.exp_dir, dirmap=os.path.join(args.dataset,'rarity_exps'))
    
    normal_dir = os.path.join(args.data_dir,'training','train','normal')
    abnormal_dir = os.path.join(args.data_dir,'training','train','abnormal')
    sample_sizes = SAMPLE_SIZES[args.dataset]
    # last sample size is the total number of samples, which is divided by two cause each samples has a data and metadata file in the folder
    # we're overriding because some datasets were cleaned and the numbers above are not accurate
    sample_sizes[-1] = int(len(os.listdir(normal_dir))/2 + len(os.listdir(abnormal_dir))/2) 
    if args.subsample_size==0:
        args.subsample_size = sample_sizes[-1]
    else:
        sample_sizes = sample_sizes[:sample_sizes.index(args.subsample_size)+1]+[sample_sizes[-1]]
        print(sample_sizes)


    normal_files = [os.path.abspath(os.path.join(normal_dir, p)) for p in os.listdir(normal_dir) if p.endswith('.npy')]
    abnormal_files = [os.path.abspath(os.path.join(abnormal_dir, p)) for p in os.listdir(abnormal_dir) if p.endswith('.npy')]
    num_to_sample = lambda rarity,norm_size: rarity*norm_size/(1-rarity) # given a desired rarity and the number of normal samples, return the number of abnormal samples to sample
    rarities = [math.floor(100*len(abnormal_files)/(len(normal_files)+len(abnormal_files)))/100] #start with observsed rarity
    while num_to_sample(rarities[-1]/2,len(normal_files))>=5:
        rarities.append(rarities[-1]/2)
    print(f'going to sample {rarities} rarities')
    for rarity_percent in rarities:
        # start by creating an args object for the unsupervised model:
        num_abnormal_to_sample = round(num_to_sample(rarity_percent,len(normal_files)))
        unsupervised_args=copy.deepcopy(args) 
        unsupervised_args.epochs = 4 # we trained our STG-NF for 4 epochs
        unsupervised_args.pose_path_train_abnormal = None# used when we train a supervised STG-NF, not the case here
        unsupervised_args.classifier = False
        unsupervised_args, model_args = init_sub_args(unsupervised_args)
        unsupervised_args.rarity_percent = rarity_percent
        # Subsample the abnormal class to manipulate rarity level:
        unsupervised_train_files = normal_files + np.random.choice(abnormal_files,num_abnormal_to_sample,replace=False).tolist()
        # test and train for the first training round are the same because we want to then select samples from the train set according to abnormality scores:
        unsupervised_args.file_list['train'] = unsupervised_train_files#.tolist()
        unsupervised_args.file_list['test'] = unsupervised_train_files#.tolist()
        contamination = np.vectorize(lambda x: 'abnormal' in x)(unsupervised_train_files).sum()/len(unsupervised_train_files)
        print(f'Sample contamination: {contamination}')
        train1_dataset, train1_loader = get_dataset_and_loader(unsupervised_args, trans_list=trans_list, only_test=False)
        print(f'@@@@@@@@@@@@@@@@@ Test set size: {len(train1_dataset["test"].file_list)}@@@@@@@@@@')
        # Train STG-NF:
        model_args = init_model_params(unsupervised_args, train1_dataset)
        model = STG_NF(**model_args)
        unsupervised_args.ckpt_dir = os.path.join(args.ckpt_dir,f'rarity_{rarity_percent}',f'unsupervised_subsample_{args.subsample_size}')
        
        os.makedirs(unsupervised_args.ckpt_dir)
        #num_of_params = calc_num_of_params(model)
        #print(f'Model has {num_of_params} parameters')
        trainer = Trainer(unsupervised_args, model, train1_loader['train'], train1_loader['test'],
                        optimizer_f=init_optimizer(unsupervised_args.model_optimizer, lr=unsupervised_args.model_lr),
                        scheduler_f=init_scheduler(unsupervised_args.model_sched, lr=unsupervised_args.model_lr, epochs=unsupervised_args.epochs))
        writer = SummaryWriter(log_dir=unsupervised_args.ckpt_dir) #save loss logs in the unsupervised folder
        trainer.train(log_writer=writer, save_every=unsupervised_args.save_every)
        dump_args(unsupervised_args, unsupervised_args.ckpt_dir, clear_file_list=True)
        normality_scores = trainer.test()
        train1_auroc,train1_auprc, train1_scores = score_dataset(normality_scores, list(train1_dataset["test"].metadata), args=unsupervised_args)
        # Logging and recording results
        current_time = datetime.now().strftime("%d.%m %H:%M:%S")
        print(f"\n----------------------{current_time}-------------------------")
        print("\033[92m Done with {}% AuROC and {}% AuPRC for {} samples\033[0m".format(train1_auroc * 100, train1_auprc * 100, train1_scores.shape[0]))
        print("-------------------------------------------------------\n\n")
        # Load unsupervised results
        grouped_score_df = pd.read_csv(os.path.join(unsupervised_args.ckpt_dir,'grouped_res.csv'))
        grouped_score_df =  grouped_score_df.sort_values(by='mean_scores', ascending=True).reset_index(drop=True)
        grouped_score_df = grouped_score_df.loc[args.discard_abnormal:] # discard the top args.discard_abnormal samples to get rid of outliers, defaults to 0, aka no discarding

        # Now that we have the abnormality scores, we can start playing around with the rarefaction and the classifier training
        # For each subsample size we will create three datasets to train on:
        # 1. Traditional - sample [subsample_size] samples randomly from all possible train files (normal and abnormal)
        # 2. Ours - sample the top [subsample_size*(1-normal_ratio)]] abnormal samples, review them (remove any normal samples) 
        #               and take the top [subsample_size-abnormal_sample_size] normal samples **without** reviewing them
        supervised_args = copy.deepcopy(args)
        supervised_args.rarity_percent = rarity_percent
        supervised_args.classifier = True
        abnormal_coeff = 2 
        prev_size = 0
        file_dict = {'traditional':[]
                        ,'ours_v1':{'normal':[],'abnormal':[]}}
        trad_df = grouped_score_df.copy()
        ours_df = grouped_score_df.copy()
        trad_save = pd.DataFrame()
        normal_save = pd.DataFrame()
        abnormal_save = pd.DataFrame()
        score_mean = grouped_score_df.mean_scores.mean()
        score_std = grouped_score_df.mean_scores.std()
        for subsample_size in sample_sizes[:-1]:
            supervised_args, _ = init_sub_args(supervised_args)
            supervised_args.pose_path_train_abnormal = None
            # select files for traditional dataset:
            # We sample the amount of file we need to add on top of the previous subsample size
            traditional_files = trad_df.sample(subsample_size-prev_size,replace=False)#np.random.choice(files, subsample_size, replace=False).tolist()
            trad_df = trad_df.drop(traditional_files.index)
            trad_save = pd.concat([trad_save,traditional_files])
            #print(len(trad_df), len(grouped_score_df))
            traditional_filelist = traditional_files.apply(lambda row: os.path.join(args.data_dir,'training','train',row.label,row.filename+'.npy'),axis=1).tolist()
            file_dict['traditional'] = file_dict['traditional'] + traditional_filelist
            # select files for Ours V1 dataset:
            most_abnormal = ours_df[ours_df.mean_scores<(score_mean-abnormal_coeff*score_std)]
            while len(most_abnormal) < int(subsample_size-prev_size) and abnormal_coeff>=0.5:
                abnormal_coeff-=0.1
                most_abnormal = ours_df[ours_df.mean_scores<(score_mean-abnormal_coeff*score_std)]
                # if there aren't enough samples in the interval, take the tail (it will include the interval samples too)
            if len(most_abnormal) < int(subsample_size-prev_size):
                print('taking abnormal tailllllllllllll')
                selected_abnormal = ours_df.head(int(subsample_size-prev_size))
            else:
                selected_abnormal = most_abnormal.sample(int(subsample_size-prev_size),replace=False)#*(1-args.normal_ratio)))
            ours_df = ours_df.drop(selected_abnormal.index)
            
            abnormal_filelist_v1 = selected_abnormal.apply(lambda row: os.path.join(args.data_dir,'training','train',row.label,row.filename+'.npy'),axis=1).tolist()
            abnormal_filelist_v11 = [n for n in abnormal_filelist_v1 if 'abnormal' in n] # review abnormal and remove normal samples
            file_dict['ours_v1']['abnormal'] = file_dict['ours_v1']['abnormal']+abnormal_filelist_v11

            extra_normal = [n for n in abnormal_filelist_v1 if 'abnormal' not in n] # review abnormal and remove normal samples
            normal_sample_size = int((subsample_size-prev_size)*args.normal_ratio)
            most_normal = ours_df[ours_df.mean_scores.between(score_mean-0.05*score_std,
                                                        score_mean+0.25*score_std)]
            if len(most_normal)+len(extra_normal) < normal_sample_size:
                # if there aren't enough samples in the interval, take the tail (it will include the interval samples too)
                print('taking normal tailllllllllllll')
                selected_normal = ours_df.tail(normal_sample_size)
            else:
                selected_normal= most_normal.sample(normal_sample_size,replace=False)#-len(abnormal_filelist_v1))
            ours_df = ours_df.drop(selected_normal.index)
            abnormal_save = pd.concat([abnormal_save,selected_abnormal[selected_abnormal.label=='abnormal']])
            normal_save = pd.concat([normal_save,selected_normal,
                                     selected_abnormal[selected_abnormal.label=='normal']])
            prev_size = subsample_size
            normal_filelist_v1 = selected_normal.apply(lambda row: os.path.join(args.data_dir,'training','train',row.label,row.filename+'.npy'),axis=1).tolist() 
            normal_filelist_v1 = normal_filelist_v1 + extra_normal
            file_dict['ours_v1']['normal'] = file_dict['ours_v1']['normal']+normal_filelist_v1
            # Now we have the filelists for the two datasets, we can start training the classifiers
            # First we will train the traditional classifier
            print(f'@@@@@@@@@@@@@@ Starting second training phase - subsample size {subsample_size} @@@@@@@@@@@@@@@@@@')
            abnormal_traditional_filelist = [n for n in file_dict['traditional'] if 'abnormal' in n]
            normal_traditional_filelist = [n for n in file_dict['traditional'] if 'abnormal' not in n]
            supervised_args.file_list['train'] = normal_traditional_filelist
            supervised_args.file_list['train_abnormal'] = abnormal_traditional_filelist
            supervised_args.ckpt_dir = os.path.join(args.ckpt_dir,f'rarity_{rarity_percent}',f'subsample_{subsample_size}','traditional_train2')
            print(f'Traditional sampled {(traditional_files.label=="abnormal").sum()} abnormal samples')
            train_supervised(supervised_args)
            # Now we will train the Ours classifier
            print(f'@@@@ Ours V1 {len(normal_filelist_v1)} - {len(abnormal_filelist_v1)}')
            supervised_args.file_list['train'] = file_dict['ours_v1']['normal']
            supervised_args.file_list['train_abnormal'] = file_dict['ours_v1']['abnormal']
            supervised_args.ckpt_dir = os.path.join(args.ckpt_dir,f'rarity_{rarity_percent}',f'subsample_{subsample_size}','ours_v1_train2')
            train_supervised(supervised_args)
            # save metadata on the files used for classifier training:
            trad_save.to_csv(os.path.join(args.ckpt_dir,f'rarity_{rarity_percent}',f'subsample_{subsample_size}',f'traditional_{len(traditional_files)}_files.csv'),index=False)
            abnormal_save.to_csv(os.path.join(args.ckpt_dir,f'rarity_{rarity_percent}',f'subsample_{subsample_size}',f'top_{len(selected_abnormal)}_abnormal.csv'),index=False)
            normal_save.to_csv(os.path.join(args.ckpt_dir,f'rarity_{rarity_percent}',f'subsample_{subsample_size}',f'top_{len(selected_normal)}_normal.csv'),index=False)

   
if __name__ == '__main__':
    main()
