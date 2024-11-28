import os
import re
import numpy as np
from scipy.ndimage import gaussian_filter1d
from sklearn.metrics import (roc_auc_score, precision_recall_curve, top_k_accuracy_score,
                              average_precision_score, confusion_matrix, ConfusionMatrixDisplay,
                              balanced_accuracy_score)
import sklearn.metrics as metrics
from sklearn import preprocessing
from tqdm import tqdm
import pandas as pd
from dataset import CLASSIFIER_CLASS2IDX, CLASSIFIER_IDX2CLASS
import matplotlib.pyplot as plt

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def score_dataset(score, metadata, args=None):
    gt, scores = get_dataset_scoresL(score, metadata, args=args)
    #print(gt, scores)
    #scores_arr = smooth_scores(scores_arr)
    #gt_np = np.concatenate(gt_arr)
    #scores_np = np.concatenate(scores_arr)
    auroc = roc_auc_score(gt, -scores)#score_auc(scores, gt)
    auprc = prc_auc_score(gt, -scores)
    return auroc, auprc, scores

def score_dataset_classifier(score, metadata, args=None):
    num_classes=args.num_classes
    gt, scores = get_dataset_scores_classifier(score, metadata, args)
    if args.num_classes>2:
        encoded_gt = preprocessing.label_binarize(gt, classes=range(args.num_classes))
        top1_acc = top_k_accuracy_score(gt,scores,k=1,labels=range(num_classes))
        top2_acc = top_k_accuracy_score(gt,scores,k=2,labels=range(num_classes))
        ap = average_precision_score(encoded_gt,scores)
        balanced_acc = balanced_accuracy_score(gt,scores.argmax(axis=1))
        return top1_acc,top2_acc, ap, balanced_acc, scores
    else:
        auroc = roc_auc_score(gt, scores[:,1])
        auprc = prc_auc_score(gt, scores[:,1])
        return auroc, auprc, scores

def get_dataset_scoresL(score, metadata, args=None):
    # labels are bout/clip level so we need to aggregate scores from all segments by bouts and score each bout
    # each metadata has a bout_id field and a segment field with which we can determine the clip it came from
    # and the location within the clip
    dataset_gt_arr = []
    dataset_scores_arr = []

    #metadata_np = np.array(metadata)
    res_df = pd.DataFrame(metadata)
    if args.dataset == 'Larvae2019':
        res_df = res_df.drop(['heading','eye_vergence','head_position'],axis=1)
        res_df['filename']=res_df.apply(lambda row: f'bout_{row.bout_id}_track_{row.track_id}_trial_{row.trial_id}_fish_{row.fish_id}_cluster_{row.cluster_id}',axis=1)
    elif args.dataset == 'PoseR':
        res_df['filename'] = res_df.apply(lambda row: f'bout_{row.bout_id}_cluster{row.cluster_id}',axis=1)
    elif args.dataset == 'Meerkat2019' or args.dataset.startswith('Simulated'):
        res_df['filename'] = res_df.unique_file_id
    res_df['scores'] = score
    gt = (np.concatenate(res_df.groupby('bout_id').label.aggregate('unique').values)=='abnormal').astype(int)
    scores = res_df.groupby('bout_id').scores.aggregate('mean').values
    res_df.to_csv(os.path.join(args.ckpt_dir,'res_test.csv'),index=False)
    #plot_confusion_mat(gt,scores,args)
    save_results_by_bout(res_df, args)
    return gt, scores

def get_dataset_scores_classifier(score, metadata,args=None):
    res_df = pd.DataFrame(metadata)
    if args.dataset == 'Larvae2019':
        res_df = res_df.drop(['heading','eye_vergence','head_position'],axis=1)
        res_df['filename']=res_df.apply(lambda row: f'bout_{row.bout_id}_track_{row.track_id}_trial_{row.trial_id}_fish_{row.fish_id}_cluster_{row.cluster_id}',axis=1)
    elif args.dataset == 'PoseR':
        res_df['filename'] = res_df.apply(lambda row: f'bout_{row.bout_id}_cluster{row.cluster_id}',axis=1)
    elif args.dataset == 'Meerkat2019' or args.dataset.startswith('Simulated'):
        res_df['filename'] = res_df.unique_file_id
    res_df[[f'score{i}' for i in range(args.num_classes)]] = score  #creates a column for each class score
    if args.num_classes>2:
        gt = res_df.groupby('bout_id').label.aggregate('unique').map(lambda x: CLASSIFIER_CLASS2IDX[args.dataset][x[0]]).values
    else:
        gt = (np.concatenate(res_df.groupby('bout_id').label.aggregate('unique').values)=='abnormal').astype(int)
    scores = res_df.groupby('bout_id').aggregate({f'score{i}':'mean' for i in range(args.num_classes)}).values
    plot_confusion_mat(gt,scores,args)
    res_df.to_csv(os.path.join(args.ckpt_dir,'res_test.csv'),index=False)
    save_results_by_bout(res_df, args)
    return gt, scores

def plot_confusion_mat(gt, scores, args):
    hardscores = scores.argmax(axis=1)
    cm = confusion_matrix(gt, hardscores)
    if args.num_classes>2:
        disp_labels = [CLASSIFIER_IDX2CLASS[args.dataset][i] for i in range(args.num_classes)]
    else:
        disp_labels = ['normal','abnormal']
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=disp_labels)
    disp.plot()
    #plt.show()
    plt.savefig(os.path.join(args.ckpt_dir, 'conf_mat.png'))
    #plt.close()


def save_results_by_bout(res_df, args):
    #grouped_df = res_df.groupby('bout_id').bout_id.aggregate('unique')#.reset_index()
    grouped_df = pd.DataFrame({'bout_id' : np.concatenate(res_df.groupby('bout_id').bout_id.aggregate('unique').values)})
    if args.classifier:
        grouped_df[[f'max_score{i}' for i in range(args.num_classes)]] = res_df.groupby('bout_id').aggregate({f'score{i}':'max' for i in range(args.num_classes)}).values
        grouped_df[[f'mean_score{i}' for i in range(args.num_classes)]] = res_df.groupby('bout_id').aggregate({f'score{i}':'mean' for i in range(args.num_classes)}).values
        grouped_df[[f'std_score{i}' for i in range(args.num_classes)]] = res_df.groupby('bout_id').aggregate({f'score{i}':'std' for i in range(args.num_classes)}).values

        #grouped_df['label'] = res_df.groupby('bout_id').behavior_label.aggregate('unique').map(lambda x: CLASSIFIER_CLASS2IDX[x[0]]).values
    else:
        #grouped_df = res_df.groupby('bout_id').scores.aggregate('mean').reset_index()
        grouped_df['min_scores'] = res_df.groupby('bout_id').scores.aggregate('min').values
        grouped_df['mean_scores'] = res_df.groupby('bout_id').scores.aggregate('mean').values
        grouped_df['std_scores'] = res_df.groupby('bout_id').scores.aggregate('std').values
    grouped_df['label'] = np.concatenate(res_df.groupby('bout_id').label.aggregate('unique').values)
    grouped_df['behavior_label'] = np.concatenate(res_df.groupby('bout_id').behavior_label.aggregate('unique').values)
    if args.dataset == 'Larvae2019':
        grouped_df['cluster_label'] = np.concatenate(res_df.groupby('bout_id').cluster_label.aggregate('unique').values)
        grouped_df['bout_duration'] = np.concatenate(res_df.groupby('bout_id').bout_duration.aggregate('unique').values)
        #grouped_df['filename'] = np.concatenate(res_df.groupby('bout_id').filename.aggregate('unique').values)
    elif 'meerkat' in args.model_layout:
        grouped_df['unique_file_id'] = np.concatenate(res_df.groupby('bout_id').unique_file_id.aggregate('unique').values)
        grouped_df['session_id'] = np.concatenate(res_df.groupby('bout_id').session_id.aggregate('unique').values)
        grouped_df['individual_id'] = np.concatenate(res_df.groupby('bout_id').individual_id.aggregate('unique').values)
        grouped_df['session_bout_id'] = np.concatenate(res_df.groupby('bout_id').session_bout_id.aggregate('unique').values)
        #grouped_df['filename'] = np.concatenate(res_df.groupby('bout_id').filename.aggregate('unique').values)
    elif args.dataset == 'SimulatedWave':
        res_df['A_x'] = res_df.A_x.apply(str)
        res_df['A_y'] = res_df.A_y.apply(str)
        res_df['F'] = res_df.F.apply(str)
        grouped_df['unique_file_id'] = np.concatenate(res_df.groupby('bout_id').unique_file_id.aggregate('unique').values)
        grouped_df['A_x'] = np.concatenate(res_df.groupby('bout_id').A_x.aggregate('unique').values) 
        grouped_df['A_y'] = np.concatenate(res_df.groupby('bout_id').A_y.aggregate('unique').values) 
        grouped_df['F'] = np.concatenate(res_df.groupby('bout_id').F.aggregate('unique').values) 
    else:
        grouped_df['cluster_id'] =  np.concatenate(res_df.groupby('bout_id').cluster_id.aggregate('unique').values)
    grouped_df['filename'] = np.concatenate(res_df.groupby('bout_id').filename.aggregate('unique').values)
    grouped_df.to_csv(os.path.join(args.ckpt_dir, 'grouped_res.csv'),index=False)

def get_dataset_scores(scores, metadata, args=None):
    dataset_gt_arr = []
    dataset_scores_arr = []
    metadata_np = np.array(metadata)

    if args.dataset == 'UBnormal':
        pose_segs_root = 'data/UBnormal/pose/test'
        clip_list = os.listdir(pose_segs_root)
        clip_list = sorted(
            fn.replace("alphapose_tracked_person.json", "tracks.txt") for fn in clip_list if fn.endswith('.json'))
        per_frame_scores_root = 'data/UBnormal/gt/'
    elif args.dataset == 'Larvae2019':
        pass
    else:
        per_frame_scores_root = 'data/ShanghaiTech/gt/test_frame_mask/'
        clip_list = os.listdir(per_frame_scores_root)
        clip_list = sorted(fn for fn in clip_list if fn.endswith('.npy'))

    print("Scoring {} clips".format(len(clip_list)))
    for clip in tqdm(clip_list):
        clip_gt, clip_score = get_clip_score(scores, clip, metadata_np, metadata, per_frame_scores_root, args)
        if clip_score is not None:
            dataset_gt_arr.append(clip_gt)
            dataset_scores_arr.append(clip_score)

    scores_np = np.concatenate(dataset_scores_arr, axis=0)
    scores_np[scores_np == np.inf] = scores_np[scores_np != np.inf].max()
    scores_np[scores_np == -1 * np.inf] = scores_np[scores_np != -1 * np.inf].min()
    index = 0
    for score in range(len(dataset_scores_arr)):
        for t in range(dataset_scores_arr[score].shape[0]):
            dataset_scores_arr[score][t] = scores_np[index]
            index += 1

    return dataset_gt_arr, dataset_scores_arr


def score_auc(scores_np, gt):
    scores_np[scores_np == np.inf] = scores_np[scores_np != np.inf].max()
    scores_np[scores_np == -1 * np.inf] = scores_np[scores_np != -1 * np.inf].min()
    auc = roc_auc_score(gt, scores_np)
    return auc

def prc_auc_score(gt,scores_np):
    #precision, recall, _ = precision_recall_curve(gt, scores_np)
    # auc = metrics.auc(recall, precision) this is the wrong way to calculate it, using average_precision_score instead
    auc = average_precision_score(gt, scores_np)
    return auc

def smooth_scores(scores_arr, sigma=7):
    for s in range(len(scores_arr)):
        for sig in range(1, sigma):
            scores_arr[s] = gaussian_filter1d(scores_arr[s], sigma=sig)
    return scores_arr


def get_clip_score(scores, clip, metadata_np, metadata, per_frame_scores_root, args):
    if args.dataset == 'UBnormal':
        type, scene_id, clip_id = re.findall('(abnormal|normal)_scene_(\d+)_scenario(.*)_tracks.*', clip)[0]
        clip_id = type + "_" + clip_id
    else:
        scene_id, clip_id = [int(i) for i in clip.replace("label", "001").split('.')[0].split('_')]
        if shanghaitech_hr_skip((args.dataset == 'ShanghaiTech-HR'), scene_id, clip_id):
            return None, None
    clip_metadata_inds = np.where((metadata_np[:, 1] == clip_id) &
                                  (metadata_np[:, 0] == scene_id))[0]
    clip_metadata = metadata[clip_metadata_inds]
    clip_fig_idxs = set([arr[2] for arr in clip_metadata])
    clip_res_fn = os.path.join(per_frame_scores_root, clip)
    clip_gt = np.load(clip_res_fn)
    if args.dataset != "UBnormal":
        clip_gt = np.ones(clip_gt.shape) - clip_gt  # 1 is normal, 0 is abnormal
    scores_zeros = np.ones(clip_gt.shape[0]) * np.inf
    if len(clip_fig_idxs) == 0:
        clip_person_scores_dict = {0: np.copy(scores_zeros)}
    else:
        clip_person_scores_dict = {i: np.copy(scores_zeros) for i in clip_fig_idxs}

    for person_id in clip_fig_idxs:
        person_metadata_inds = \
            np.where(
                (metadata_np[:, 1] == clip_id) & (metadata_np[:, 0] == scene_id) & (metadata_np[:, 2] == person_id))[0]
        pid_scores = scores[person_metadata_inds]

        pid_frame_inds = np.array([metadata[i][3] for i in person_metadata_inds]).astype(int)
        clip_person_scores_dict[person_id][pid_frame_inds + int(args.seg_len / 2)] = pid_scores

    clip_ppl_score_arr = np.stack(list(clip_person_scores_dict.values()))
    clip_score = np.amin(clip_ppl_score_arr, axis=0)

    return clip_gt, clip_score
