import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import imageio
import json
import argparse

BEHAVIOR_CLUSTERS = { 2: "scoot",14: "scoot",
                     9: "routine-turn",11:  "routine-turn",15:  "routine-turn",18:  "routine-turn",22:  "routine-turn",27:  "routine-turn",
                     1: "burst",4: "burst",6: "burst",7: "burst", 12: "burst", 13: "burst",23: "burst",28: "burst",
                     20: "o-bend", 30: "o-bend",
                     3: "j-turn",5: "j-turn", 29: "j-turn",25: "j-turn",
                     16: "slc",24: "slc",
                     8: "llc",10: "llc",17: "llc",19: "llc",
                     21: "noise",26: "noise"}
ABNORMAL_CLUSTERS = [20, 30, #o-bend
                     #3, 5, 29, 25, #j-turn
                     16, 24, #slc
                     8, 10, 17, 19, #llc
                    21, 26]#noise

def create_dataset_folder(data_dir,subfolders):
    for sub in subfolders.keys():
        if not 'test' in sub:
            os.makedirs(os.path.join(data_dir,sub,'normal'),exist_ok=True)
            os.makedirs(os.path.join(data_dir,sub,'abnormal'),exist_ok=True)
        else:
            os.makedirs(os.path.join(data_dir,sub),exist_ok=True)
    os.makedirs(os.path.join(data_dir,'metadata_csvs'),exist_ok=True)


def save_data(save_dir,subfolders,remove_outliers):
    for partition, boutslist in subfolders.items():
        if remove_outliers:
            boutst, labels, outlier_frames, indices_to_remove = boutslist
        else:
            boutst, labels = boutslist
            indices_to_remove = []
        path = os.path.join(save_dir,partition)
        print(f'Now working on {partition} on {boutst.shape[0]-len(indices_to_remove)} bouts')
        for i,entry in enumerate(boutst):
            if i not in indices_to_remove:
                entry = entry.squeeze().transpose(2,0,1)
                entry = entry[:,:2,...] #remove conf for now
                label = labels[i]+1 # according to the dataset docs we need to add 1 to map the clusters to the labels
                file_name = f'bout_{i}_cluster{label}'
                if label in ABNORMAL_CLUSTERS:
                    category = 'abnormal'
                else:
                    category = 'normal'
                if partition.endswith('train/'):
                    filepath = os.path.join(path,category)
                else:
                    filepath = path
                data_entry = {'bout_id':int(i),
                            'num_keypoints':int(entry.shape[0]),
                            'cluster_id': int(label),
                            'behavior_label':BEHAVIOR_CLUSTERS[label],
                            'bout_duration':int(entry.shape[-1]),
                            'partition':partition}
                if remove_outliers:
                    entry = entry[...,~outlier_frames[i]] # remove outlier frames
                    data_entry['bout_duration'] = int(entry.shape[-1])
                    data_entry['num_outlier_frames'] = int(outlier_frames[i].sum()), 
                    data_entry['outlier_frame_indices'] = [int(ind) for ind in np.where(outlier_frames[i])[0]]

                meta_path = os.path.join(filepath,file_name+'.json')
                skeleton_path = os.path.join(filepath,file_name+'.npy')
                if not os.path.exists(meta_path):
                    with open(meta_path, 'w') as outfile:
                        json.dump(data_entry, outfile)
                if not os.path.exists(skeleton_path):
                    with open(skeleton_path, 'wb') as outfile2:
                        np.save(outfile2, entry)


#calculate distances between adjacent points in each frame
def get_outliers1(clip_array):
    all_points = np.squeeze(clip_array[:,0:2,...]).T
    all_diffs = np.diff(all_points,axis=0)
    all_dists = np.sqrt(np.sum(all_diffs**2,axis=2))
    mean_dist = np.mean(all_dists[9:,...].flatten()) #without eye points, they are more close to each other
    delta_dist = all_dists[9:,...] - mean_dist
    #print(delta_dist.min(), delta_dist.max())
    #take only distances on the left side of the distribution, i.e., smaller than the mean
    # this prevents undue influence of very large distances (i.e., outliers) on the std
    left_dists = delta_dist[delta_dist<=0]
    #print(f'left dists stats min {left_dists.min()}, max {left_dists.max()}, length {left_dists.shape}')
    left_std = np.sqrt((left_dists**2).mean())
    #print(f'mean dist: {mean_dist}, std dist: {left_std}')
    outlier_points = all_dists>(mean_dist+6*left_std) #it's actually outlier distances between adjacent points, but ok
    outlier_frames = np.any(outlier_points,axis=0).T#.reshape(-1,300)
    outlier_frames_per_video = np.sum(outlier_frames,axis=1)
    num_adjacent_outlier_frames = np.logical_and(outlier_frames[:,:-1], outlier_frames[:,1:]).sum(axis=1) #how many two frame sequences are outliers
    indices_to_remove = np.where((num_adjacent_outlier_frames>50)| (outlier_frames_per_video>50))[0]
    #print('bouts to remove',len(indices_to_remove))
    return outlier_frames, indices_to_remove


# now do the same but for the same point between frames
def get_outliers2(clip_array):
    all_points = np.squeeze(clip_array[:,0:2,...]).T
    all_diffs = np.diff(all_points,axis=1)
    all_dists = np.sqrt(np.sum(all_diffs**2,axis=2))
    mean_dist = np.mean(all_dists.flatten())
    delta_dist = all_dists-mean_dist
    print(delta_dist.min(), delta_dist.max())
    #take only distances on the left side of the distribution, i.e., smaller than the mean
    # this prevents undue influence of very large distances (i.e., outliers) on the std
    left_dists = delta_dist[delta_dist<=0]
    #print(f'left dists stats min {left_dists.min()}, max {left_dists.max()}, length {left_dists.shape}')
    left_std = np.sqrt((left_dists**2).mean())
    #print(f'mean dist: {mean_dist}, std dist: {left_std}')
    outlier_points = all_dists>(mean_dist+12*left_std) #3 sds because movement is smaller between frames, leading to smaller mean
    # assume first frame is never an outlier (for convenience so we can merge with other outlier array)
    outlier_frames = np.concatenate([np.array([False]*outlier_points.shape[2])[None,...],np.any(outlier_points,axis=0)]).T
    outlier_frames_per_video = np.sum(outlier_frames,axis=1)
    num_adjacent_outlier_frames = np.logical_and(outlier_frames[:,:-1], outlier_frames[:,1:]).sum(axis=1) #how many two frame sequences are outliers
    indices_to_remove = np.where((num_adjacent_outlier_frames>50)| (outlier_frames_per_video>=50))[0]
    #print('bouts to remove',len(indices_to_remove))
    return outlier_frames, indices_to_remove


# all together now
def get_outliers(clip_array):
    outlier_frames1, indices_to_remove1 = get_outliers1(clip_array) #outliers in distances between adjacent points in the same frame
    outlier_frames2, indices_to_remove2 = get_outliers2(clip_array) #outliers in distances between the same point in adjacent frames
    outlier_frames = np.logical_or(outlier_frames1,outlier_frames2)
    indices_to_remove = np.unique(np.concatenate([indices_to_remove1,indices_to_remove2])) 
    indices_to_remove.sort()
    outlier_frames_per_video = np.sum(outlier_frames,axis=1)
    num_adjacent_outlier_frames = np.logical_and(outlier_frames[:,:-1], outlier_frames[:,1:]).sum(axis=1) #how many two frame sequences are outliers
    indices_to_remove_tot = np.where((num_adjacent_outlier_frames>20)| (outlier_frames_per_video>=50))[0]
    print('bouts to remove',len(indices_to_remove_tot))
    perc_outliers = outlier_frames.sum()/(outlier_frames.shape[0]*outlier_frames.shape[1])
    print(f'{perc_outliers*100:.2f}% of frames are outliers, {len(indices_to_remove_tot)} bouts to remove')
    return outlier_frames, indices_to_remove, indices_to_remove_tot


def init_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('save_dir', help='directory to store save processed dataset')
    parser.add_argument('data_dir', help='directory where raw data files are stored')
    parser.add_argument('--remove_outliers',  action='store_true')
    parser.set_defaults(with_outliers=True)
    parser.add_argument('--with_outliers',dest='remove_outliers', action='store_false')
    return parser


def main():
    parser = init_parser()
    args = parser.parse_args()
    print(args)
    train = np.load(os.path.join(args.data_dir,'Zebtrain.npy'))
    train_labels = np.load(os.path.join(args.data_dir,'Zebtrain_labels.npy'))
    test = np.load(os.path.join(args.data_dir,'Zebtest.npy'))
    test_labels = np.load(os.path.join(args.data_dir,'Zebtest_labels.npy'))
    train_df = pd.DataFrame(train_labels,columns=['cluster_id'])
    # according to the dataset docs we need to add 1 to map the clusters to the labels:
    train_df['behavior_label'] = train_df.cluster_id.map(lambda x: BEHAVIOR_CLUSTERS[x+1])
    train_df['bout_id'] = range(len(train_df))
    train_df['file_key'] = train_df.apply(lambda x: f'bout_{x.bout_id}_cluster{x.cluster_id+1}',axis=1)
    test_df = pd.DataFrame(test_labels,columns=['cluster_id'])
    test_df['behavior_label'] = test_df.cluster_id.map(lambda x: BEHAVIOR_CLUSTERS[x+1])
    test_df['bout_id'] = range(len(test_df))
    test_df['file_key'] = test_df.apply(lambda x: f'bout_{x.bout_id}_cluster{x.cluster_id+1}',axis=1)
    # calculate outlier frames (where pose estimator was noisy) and flag them for removal from the data:
    if args.remove_outliers:
        train_outlier_frames, _, train_tot = get_outliers(train)
        test_outlier_frames, _, test_tot = get_outliers(test)
        # save data structure for dataset construction:
        subfolders = {
              'training/train/':[train,train_labels, train_outlier_frames, train_tot],
              'training/test':[test,test_labels, test_outlier_frames, test_tot]}
        # remove outliers from metadata
        train_df = train_df.drop(train_tot)
        test_df = test_df.drop(test_tot)
    else:
        # save data structure for dataset construction:
        subfolders = {'training/train/':[train,train_labels],
                    'training/test':[test,test_labels]}
    # create new dataset folders:
    create_dataset_folder(args.save_dir,subfolders)
    # save data in required format i.e., .npy for the keypoints and .json for the metadata
    save_data(args.save_dir,subfolders,args.remove_outliers)
    # save metadata
    train_df['cluster_id'] += 1
    test_df['cluster_id'] += 1
    train_df.to_csv(os.path.join(args.save_dir,'metadata_csvs','train.csv'),index=False)
    test_df.to_csv(os.path.join(args.save_dir,'metadata_csvs','test.csv'),index=False)


if __name__ == '__main__':
    main()