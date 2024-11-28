import os
import json
import numpy as np
import pandas as pd
import argparse
import h5py
from tqdm import tqdm

BEHAVIOR_KEY = {0: 'vigilance',
               1: 'resting',
               2:'foraging',
               3: 'running'}

NORMAL_BEHAVIORS = ['vigilance','foraging']

def create_dataset_folder(save_dir, subfolders):
    for sub in subfolders.keys():
        if not 'test' in sub:
            os.makedirs(os.path.join(save_dir,sub,'normal'),exist_ok=True)
            os.makedirs(os.path.join(save_dir,sub,'abnormal'),exist_ok=True)
        else:
            os.makedirs(os.path.join(save_dir,sub),exist_ok=True)
        os.makedirs(os.path.join(save_dir,'metadata_csvs'),exist_ok=True)


def save_data(save_dir, subfolders, dataset_ref,session_ref):
    for partition, bout_df in subfolders.items():
        print(f'Now working on {partition} on {len(bout_df)} bouts')
        for i,entry in tqdm(bout_df.iterrows(), total=bout_df.shape[0]):
            data_entry = dict(entry)
            data_entry['num_keypoints'] = 3 # Our keypoints are the planar accelarations along XY, YZ, XZ plains (because STG-NF expects 2D keypoints as input)
            if entry.normal:
                category = 'normal'
            else:
                category = 'abnormal'
            if ('/train/' in partition):
                path = os.path.join(save_dir,partition,category)
            else:
                path = os.path.join(save_dir,partition)
            meta_path = os.path.join(path,entry.unique_file_id+'.json')
            bout_path = os.path.join(path,entry.unique_file_id+'.npy')
            if not os.path.exists(meta_path):
                with open(meta_path, 'w') as outfile:
                    json.dump(data_entry, outfile)
            if not os.path.exists(bout_path):
                session_ds = dataset_ref[session_ref[entry.session_id-1]][0] #remove 1 to match file numbering, the indexing in the DataFrame match the paper's numbers
                behavior_ref = dataset_ref[session_ds[entry.behavior_id]][0]
                bout = np.array(dataset_ref[behavior_ref[entry.session_bout_id]])
                bout = np.expand_dims(bout, 1) 
                xy = np.concatenate([bout[0,:],bout[1,:],],axis=0)
                yz = np.concatenate([bout[1,:],bout[2,:]],axis=0)
                xz = np.concatenate([bout[0,:],bout[2,:],],axis=0)
                dst_arr = np.concatenate([xy[None,...],yz[None,...],xz[None,...]])# the matrix is 3x200, fpr STG-NF to work we need to add a channel dimension: 3x1x200
                with open(bout_path, 'wb') as outfile2:
                    np.save(outfile2, dst_arr)

def init_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('save_dir', help='directory to store save processed dataset')
    parser.add_argument('data_dir', help='directory where raw data files are stored')
    return parser

def main():
    parser = init_parser()
    args = parser.parse_args()
    dataset_ref = h5py.File(os.path.join(args.data_dir, 'labelledTriaxialAccData.mat'))
    session_ref= dataset_ref['sessionWiseAccData_fourBehaviours'][0] 
    script_path = os.path.dirname(os.path.realpath(__file__))
    train_set = pd.read_csv(os.path.join(script_path,'metadata_csvs','train_meta.csv'))
    test_set = pd.read_csv(os.path.join(script_path,'metadata_csvs','test_meta.csv'))
    subfolders = {'training/train/':train_set,
              'training/test':test_set}
    create_dataset_folder(args.save_dir,subfolders)
    save_data(args.save_dir, subfolders,dataset_ref,session_ref)
    train_set.to_csv(os.path.join(args.save_dir,'metadata_csvs','train_meta.csv'))
    test_set.to_csv(os.path.join(args.save_dir,'metadata_csvs','test_meta.csv'))


if __name__ == '__main__':
    main()