### Prep data from Johnson et al. 2020, see their original code for explanations on the data files and visualizations

import os
import json
import pandas as pd
import numpy as np
import pickle
import argparse
from tqdm import tqdm
from fish_pose_utils import get_skeleton_with_eyes

CLUSTER_KEY={0:'e0',1:'e0',
               2:'e1',3:'e1',
               4:'e2',5:'e2',
               6:'e3',7:'e3',
               8:'e4',9:'e4',
               10:'e5',11:'e5',
               12:'e6',13:'e6',
               14:'e7',15:'e7',
               16:'e8',17:'e8',
               18:'e9',19:'e9',
              20:'j1',21:'j1',
               22:'j2',23:'j2',
               24:'p1',25:'p1',
               26:'p2',27:'p2',
               28:'p3',29:'p3',
               30:'p4',31:'p4',
               32:'a',33:'a',
               34:'s',35:'s'
               }
NORMAL_CLUSTERS = list(range(20))+list(range(24,32)) #Explore and Pursuit will be normal; J-turns, aborts and strikes will be abnormal

# these are keys which we renamed from the original dataset:
POSE_KEYS = {#'bSkeleton':'keypoints',
             'bPosition':'head_position',
             'bEyes':'eye_vergence',
             'bHeading':'heading'} #heading and position are in the coordinate system of the arena, keypoints is normalized
META_KEYS = {'fishNum':'fish_id',
             'trialNum':'trial_id',
             'uniqueSeqNum':'track_id',
             #'boutInd':'bout_id',
             'boutLabel':'cluster_id',
             'boutDuration':'bout_duration',
             'fed':'fed'
             }
# This script creates a dataset in the format expected by STG-NF, from the raw data published by Johnson et al. (2020).
# For each sample we save two files:  
# 1. An .npy array containing a temporal sequence of 2D keypoints (keypoints, channels, frames):
#    i.e., each row is a different landmark, the columns are X and Y and the last dimension is time
# 2. Metadata is saved as a .json file with the same name containing all swim bout metadata:
#    including cluster label, behavior label, and whether the behavior is considered normal/abnormal in our setting.

def get_behavior_label(cluster_id):
  assert cluster_id>=0 and cluster_id<36, print('Cluster id not in known range')
  if cluster_id<20:
    label = 'explore'
  elif cluster_id<24:
    label = 'j_turn'
  elif cluster_id<32:
    label = 'pursuit'
  elif cluster_id<34:
    label = 'abort'
  else:
    label = 'strike'
  return label

def create_dataset_folder(save_dir, subfolders):
    for sub in subfolders.keys():
        if not 'test' in sub:
            os.makedirs(os.path.join(save_dir,sub,'normal'),exist_ok=True)
            os.makedirs(os.path.join(save_dir,sub,'abnormal'),exist_ok=True)
        else:
            os.makedirs(os.path.join(save_dir,sub),exist_ok=True)
    os.makedirs(os.path.join(save_dir,'metadata_csvs'),exist_ok=True)

def save_data(poseData, boutData, save_dir, subfolders, add_eyes=True):
    for partition, bout_df in subfolders.items():
    #if 'test' in partition or 'val' in partition:
        print(f'Now working on {partition} on {len(bout_df)} bouts')
        for i,entry in tqdm(bout_df.iterrows(), total=bout_df.shape[0]):
            data_entry = {'bout_id':entry.bout_id, 'num_keypoints':31}
            if entry.normal:
                category = 'normal'
            else:
                category = 'abnormal'
            if ('/train/' in partition ):
                path = os.path.join(save_dir,partition,category)
            else:
                path = os.path.join(save_dir,partition)
            for pose_key,data_key in POSE_KEYS.items():
                if type(poseData[entry.bout_id][pose_key])==np.ndarray:
                    value = poseData[entry.bout_id][pose_key].tolist()
                else:
                    value = poseData[entry.bout_id][pose_key]
                data_entry[data_key] = value
            for meta_key,data_key in META_KEYS.items():
                try:
                    data_entry[data_key] = entry[meta_key]
                except:
                    data_entry[data_key] = entry[data_key]
                assert data_entry[data_key] == boutData[meta_key][entry.bout_id]
            data_entry['cluster_label'] = CLUSTER_KEY[data_entry['cluster_id']]
            data_entry['behavior_label'] = get_behavior_label(data_entry['cluster_id'])
            
            if add_eyes:
                # This is what we did for the paper, we calculated the 4 eye points byusing the eye angles measured by the compilers of the dataset.
                # Using this angle we estimate the eye as a tilted ellipse with a set diameter. 
                # The 4 points are located on the major and minor axes on the perimeter of the ellipse.
                skeleton = get_skeleton_with_eyes(poseData[entry.bout_id])
            else:
                skeleton = poseData[entry.bout_id]['bSkeleton']
            file_name = f'bout_{data_entry["bout_id"]}_track_{data_entry["track_id"]}_trial_{data_entry["trial_id"]}_fish_{data_entry["fish_id"]}_cluster_{data_entry["cluster_id"]}'
            meta_path = os.path.join(path,file_name+'.json')
            skeleton_path = os.path.join(path,file_name+'.npy')
            if not os.path.exists(meta_path):
                with open(meta_path, 'w') as outfile:
                    json.dump(data_entry, outfile)
            if not os.path.exists(skeleton_path):
                with open(skeleton_path, 'wb') as outfile2:
                    np.save(outfile2, skeleton)
      
def init_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('save_dir', help='directory to store save processed dataset')
    parser.add_argument('data_dir', help='directory where raw data files are stored')
    parser.add_argument('--no_eyes', action='store_true')
    return parser

def main():
    parser = init_parser()
    args = parser.parse_args()
    print(args)
    boutDataFileName = os.path.join(args.data_dir,"boutData.pkl")
    with open(boutDataFileName, "rb") as input_file:
        boutData = pickle.load(input_file)
    poseDataFileName = os.path.join(args.data_dir,"poseData.pkl")
    with open(poseDataFileName, "rb") as input_file:
        poseData = pickle.load(input_file)
    #del boutData['tSNE']
    #boutsMeta = pd.DataFrame.from_dict(boutData)
    #boutsMeta['normal'] = boutsMeta.boutLabel.map(lambda x: x in NORMAL_CLUSTERS)
    #boutsMeta.index.name = 'boutInd'
    #boutsMeta.to_csv(os.path.join(args.data_dir,'boutsMeta.csv'))
    #holdout_set.to_csv(os.path.join(data_src_folder,'new_dataset','holdout_meta.csv'))

    #read the metadata files which include train and test partitions:
    script_path = os.path.dirname(os.path.realpath(__file__))
    train_set = pd.read_csv(os.path.join(script_path,'metadata_csvs','train_meta.csv'))
    test_set = pd.read_csv(os.path.join(script_path,'metadata_csvs','test_meta.csv'))
    subfolders = {'training/train/':train_set,
              'training/test':test_set}
    create_dataset_folder(args.save_dir,subfolders)
    save_data(poseData, boutData, args.save_dir, subfolders, add_eyes=~args.no_eyes)
    train_set.to_csv(os.path.join(args.save_dir,'metadata_csvs','train_meta.csv'))
    test_set.to_csv(os.path.join(args.save_dir,'metadata_csvs','test_meta.csv'))

if __name__ == '__main__':
    main()