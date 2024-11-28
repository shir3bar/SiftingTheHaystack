import os
import pandas as pd
import json
from utils.scoring_utils import save_results_by_bout
from argparse import Namespace
from utils.train_utils import dump_args

def aggregate_exp_results(exp_dir,exp_name):
    # collect results from seperate folders to a single data frame with specified hyperparameter arguments
    keys = ['K','L','seed','seg_len','seg_stride','model_layout',
            'dataset','adj_strategy','model_hidden_dim','max_hops',
            'model_lr','classifier','pose_path_train_abnormal']
    dirs = os.listdir(exp_dir)
    all_df_list = []
    for folder in dirs:
        folder_path = os.path.join(exp_dir,folder)
        df = pd.read_csv(os.path.join(folder_path,'grouped_res.csv'))
        df['folder'] = folder
        with open(os.path.join(folder_path,'args.json'),'r') as f:
            args = json.load(f)
        for key in keys:
            df[key] = args[key]
        df['exp_dir'] = exp_dir
        all_df_list.append(df)
    all_df = pd.concat(all_df_list)
    filepath = os.path.join(exp_dir,f'{exp_name}_results_agg.csv')
    all_df.to_csv(filepath, index=False)
    print(f'Saved aggregated results to {filepath}')

def aggregate_rarefaction_results(exp_dir,exp_name):
    # collect results from seperate folders to a single data frame with specified hyperparameter arguments
    keys = ['K','L','seed','seg_len','seg_stride','model_layout','epochs',
            'dataset','adj_strategy','model_hidden_dim','max_hops',
            'model_lr', 'subsample_size','classifier']
    subsamples = os.listdir(exp_dir)
    all_df_list = []
    for folder in subsamples:
        subsample_path = os.path.join(exp_dir,folder)
        models = os.listdir(subsample_path)
        for model_category in models:
            folder_path = os.path.join(subsample_path,model_category)
            df = pd.read_csv(os.path.join(folder_path,'grouped_res.csv'))
            df['folder'] = folder
            with open(os.path.join(folder_path,'args.json'),'r') as f:
                args = json.load(f)
            for key in keys:
                df[key] = args[key]
            df['exp_dir'] = exp_dir
            df['model_type'] = model_category
            all_df_list.append(df)
    all_df = pd.concat(all_df_list)
    filepath = os.path.join(exp_dir,f'{exp_name}_results_agg.csv')
    all_df.to_csv(filepath, index=False)
    print(f'Saved aggregated results to {filepath}')

def aggregate_final_rarefaction_results(exp_dir,exp_name, rarity=False):
    # collect results from seperate folders to a single data frame with specified hyperparameter arguments
    keys = ['K','L','seed','seg_len','seg_stride','model_layout','epochs',
            'dataset','adj_strategy','model_hidden_dim','max_hops',
            'model_lr']#, 'subsample_size']
    if rarity:
        keys.append('rarity_percent')
    subsamples =  [f for f in os.listdir(exp_dir) if os.path.isdir(os.path.join(exp_dir,f))]#os.listdir(exp_dir)
    all_df_list = []
    for folder in subsamples:
        subsample_path = os.path.join(exp_dir,folder)
        strategies = [f for f in os.listdir(subsample_path) if 'train2' in f]
        for strategy in strategies:
            folder_path = os.path.join(subsample_path,strategy)
            model_category = os.listdir(folder_path)[0]
            df = pd.read_csv(os.path.join(folder_path,model_category,'grouped_res.csv'))
            df['folder'] = folder
            with open(os.path.join(folder_path,model_category,'args.json'),'r') as f:
                args = json.load(f)
            for key in keys:
                df[key] = args[key]
            df['exp_dir'] = exp_dir
            df['model_type'] = model_category
            df['sampling_strategy'] = strategy.replace('_train2',"")
            df['subsample_size'] = folder.replace('subsample_','')
            all_df_list.append(df)
    all_df = pd.concat(all_df_list)
    filepath = os.path.join(exp_dir,f'{exp_name}_results_agg.csv')
    all_df.to_csv(filepath, index=False)
    print(f'Saved aggregated results to {filepath}')
    if rarity:
        return all_df

def aggregate_rarity_train_data(exp_dir,exp_name):
    rarities = [f for f in os.listdir(exp_dir) if os.path.isdir(os.path.join(exp_dir,f))]
    all_df_list = []
    for folder in rarities:
        subsamples = [f for f in os.listdir(os.path.join(exp_dir,folder)) if os.path.isdir(os.path.join(exp_dir,folder,f))]
        print(subsamples)
        for subsample in subsamples:

            folder_path = os.path.join(exp_dir,folder,subsample)
            csvs = [f for f in os.listdir(folder_path) if f.endswith('.csv') and 'res' not in f]
            print(csvs)
            pass
            for csv in csvs:
                df = pd.read_csv(os.path.join(folder_path,csv))
                df['rarity_percent'] = folder
                df['rep_folder'] = exp_dir
                df['subsample'] = subsample
                if 'abnormal' in csv:
                    category = 'ours_abnormal'
                elif 'normal' in csv:
                    category = 'ours_normal'
                else:
                    category = 'traditional'
                df['category'] = category
                    
                all_df_list.append(df)
    all_df = pd.concat(all_df_list)
    print(len(all_df))
    filepath = os.path.join(exp_dir,f'{exp_name}_train_files_agg.csv')
    all_df.to_csv(filepath, index=False)
    print(f'Saved aggregated results to {filepath}')


def aggregate_rarity_results(exp_dir,exp_name,remove=False):
    rarities = [f for f in os.listdir(exp_dir) if os.path.isdir(os.path.join(exp_dir,f))]
    filepath = os.path.join(exp_dir,f'{exp_name}_results_agg.csv')
    if os.path.exists(filepath) and remove:
        os.remove(filepath)
    elif os.path.exists(filepath):
        print(f'File already exists at {filepath} and remove is set to False, returning...')
        return
    all_df_list = []
    for folder in rarities:
        folder_df = aggregate_final_rarefaction_results(os.path.join(exp_dir,folder),exp_name+f'_{folder}', rarity=True)
        all_df_list.append(folder_df)
    all_df = pd.concat(all_df_list)
    
    all_df.to_csv(filepath, index=False)
    print(f'Saved aggregated results to {filepath}')

def aggregate_single_simulation_results(exp_dir,exp_name):
    # collect results from seperate folders to a single data frame with specified hyperparameter arguments
    keys = ['K','L','seed','seg_len','seg_stride','model_layout','epochs',
            'dataset','adj_strategy','model_hidden_dim','max_hops',
            'model_lr']#, 'subsample_size']
    dirs = [f for f in os.listdir(exp_dir) if os.path.isdir(os.path.join(exp_dir,f))]
    all_df_list = []
    for folder in dirs:
        folder_path = os.path.join(exp_dir,folder)
        df = pd.read_csv(os.path.join(folder_path,'grouped_res.csv'))
        df['folder'] = folder
        with open(os.path.join(folder_path,'args.json'),'r') as f:
            args = json.load(f)
            print(args.keys())
        for key in keys:
            df[key] = args[key]
        df['exp_dir'] = exp_dir
        all_df_list.append(df)
    all_df = pd.concat(all_df_list)
    filepath = os.path.join(exp_dir,f'{exp_name}_results_agg.csv')
    all_df.to_csv(filepath, index=False)
    print(f'Saved aggregated results to {filepath}')
    return all_df  

def aggregate_all_simulation_results(exp_dir,exp_name):
    sim_categories = [f for f in os.listdir(exp_dir) if os.path.isdir(os.path.join(exp_dir,f))]
    stds_key = {'5std':5,
                '25std':2.5,
                '15std':1.5,
                '05std':0.5}
    rare_key = {'24rare':0.24,
                '12rare':0.12,
                'simulatedDS':0.05,# I f*ed up the DS folder names so the 0.05 rarity is this thing
                '015rare':0.015}
    all_df_list = []
    for folder in sim_categories:
        folder_df = aggregate_single_simulation_results(os.path.join(exp_dir,folder),exp_name+f'_{folder}')
        if 'rare' not in folder:
            std_str = folder.split('_')[-1]
            rare_str = folder.split('_')[0]
        else:
            std_str = folder.split('_')[0]
            rare_str = folder.split('_')[1]
        folder_df['data_std'] = stds_key[std_str]
        folder_df['data_rarity'] = rare_key[rare_str]
        all_df_list.append(folder_df)
    all_df = pd.concat(all_df_list)
    filepath = os.path.join(exp_dir,f'{exp_name}_results_agg.csv')
    all_df.to_csv(filepath, index=False)
    print(f'Saved aggregated results to {filepath}')


def add_std_calc_rarefaction(exp_dir):
    subsamples = [d for d in os.listdir(exp_dir) if '.' not in d]
    for folder in subsamples:
        subsample_path = os.path.join(exp_dir,folder)
        models = os.listdir(subsample_path)
        for model_category in models:
            folder_path = os.path.join(subsample_path,model_category)
            df = pd.read_csv(os.path.join(folder_path,'res_test.csv'))
            grouped_df = pd.read_csv(os.path.join(folder_path,'grouped_res.csv'))
            grouped_df.to_csv(os.path.join(folder_path,'grouped_res_backup.csv'))
            with open(os.path.join(folder_path,'args.json'),'r') as f:
                args_dict = json.load(f)
            args = Namespace(**args_dict)
            args.ckpt_dir = folder_path
            save_results_by_bout(df, args)
    print(f'finished {exp_dir}')


def add_subsample_size_to_args(exp_dir):
    # collect results from seperate folders to a single data frame with specified hyperparameter arguments

    subsamples = os.listdir(exp_dir)
    sizes = [int(s.split('_')[-1]) for s in subsamples] #all folders have the naming convention: 'abnormal_subsample_{size}'
    for i,folder in enumerate(subsamples):
        subsample_path = os.path.join(exp_dir,folder)
        models = os.listdir(subsample_path)
        for model_category in models:
            folder_path = os.path.join(subsample_path,model_category)
            with open(os.path.join(folder_path,'args.json'),'r') as f:
                args_dict = json.load(f)
            args_dict['subsample_size'] = sizes[i]
            args = Namespace(**args_dict)
            dump_args(args, folder_path)

def aggregate_single_ablation_results(exp_dir,exp_name, rarity=True):
    # collect results from seperate folders to a single data frame with specified hyperparameter arguments
    keys = ['K','L','seed','seg_len','seg_stride','model_layout','epochs',
            'dataset','adj_strategy','model_hidden_dim','max_hops',
            'model_lr','rarity_percent']
    subsamples =  [f for f in os.listdir(exp_dir) if os.path.isdir(os.path.join(exp_dir,f))]#os.listdir(exp_dir)
    all_df_list = []
    for folder in subsamples:
        subsample_path = os.path.join(exp_dir,folder)
        model_category = os.listdir(subsample_path)[0]
        df = pd.read_csv(os.path.join(subsample_path,model_category,'grouped_res.csv'))
        df['folder'] = folder
        with open(os.path.join(subsample_path,model_category,'args.json'),'r') as f:
            args = json.load(f)
        for key in keys:
            df[key] = args[key]
        df['exp_dir'] = exp_dir
        df['model_type'] = model_category
        df['sampling_strategy'] = 'ablation'
        df['subsample_size'] = folder.replace('subsample_','')
        all_df_list.append(df)
    all_df = pd.concat(all_df_list)
    filepath = os.path.join(exp_dir,f'{exp_name}_results_agg.csv')
    all_df.to_csv(filepath, index=False)
    print(f'Saved aggregated results to {filepath}')
    if rarity:
        return all_df
    
def aggregate_all_ablation_results(exp_dir,exp_name):
    rarities = [f for f in os.listdir(exp_dir) if os.path.isdir(os.path.join(exp_dir,f))]
    all_df_list = []
    exp_name= exp_name+'_ablation'
    for folder in rarities:
        folder_df = aggregate_single_ablation_results(os.path.join(exp_dir,folder,'ablation'),exp_name+f'_{folder}', rarity=True)
        all_df_list.append(folder_df)
    all_df = pd.concat(all_df_list)
    filepath = os.path.join(exp_dir,f'{exp_name}_results_agg.csv')
    all_df.to_csv(filepath, index=False)
    print(f'Saved aggregated results to {filepath}')