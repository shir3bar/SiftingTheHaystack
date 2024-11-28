import json
import math
import os
import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm
from utils.pose_utils import gen_clip_seg_data_np
from torch.utils.data import DataLoader
import glob

ABNORMAL_CLUSTERS = {'Larvae2019':[#6,7,12,13,18,19,30,31, #e3,e6,e9,p4
                                    20,21,22,23,# J-turn,
                                   32,33, # Abort,
                                   34,35],# Strike
                    'PoseR':[20, 30, #o-bend
                     #3, 5, 29, 25, #j-turn
                     16, 24, #slc
                     8, 10, 17, 19, #llc
                    21, 26],#noise
                    'Meerkat2019':[1, #resting
                                   3], #running
                    'Simulated':[1], #low AR
                    'SimulatedWave':[1] #a>>f
                    } 
CLASSIFIER_CLASS2IDX = {'Larvae2019':{'explore':0,'pursuit':1, 'j_turn':2,'abort':3,'strike':4 },
                        'PoseR':{'burst':0, 'routine-turn':1, 'j-turn':2, 'scoot':3, 'llc':4, 'slc':5, 'o-bend':6, 'noise':7},
                        'Meerkat2019':{'vigilance':0,'resting':1,'foraging':2,'running':3},
                        'Simulated':{'high_AR':0,'low_AR':1},
                        'SimulatedWave':{'f_gt_A':0,'A_gt_f':1}}
CLASSIFIER_IDX2CLASS = {dataset :{v:k for k,v in CLASSIFIER_CLASS2IDX[dataset].items()} for dataset in CLASSIFIER_CLASS2IDX.keys()}

class PoseSegDataset(Dataset):
    """
    Generates a dataset with two objects, a np array holding sliced pose sequences
    and an object array holding file name, person index and start time for each sliced sequence.
    """

    def __init__(self, path_to_json_dir, return_indices=False,
                 debug=False, return_global=True, evaluate=False, abnormal_train_path=None,
                 subsample_abnormal=[],subsample_normal=[],file_list=[],abnormal_file_list=[],
                 **dataset_args):
        super().__init__()
        self.args = dataset_args
        self.path_to_json = path_to_json_dir
        self.eval = evaluate
        self.debug = debug
        num_clips = dataset_args.get('specific_clip', None)
        self.return_indices = return_indices
        self.transform_list = dataset_args.get('trans_list', None)
        if self.transform_list is None or len(self.transform_list)==0:
            self.apply_transforms = False
            self.num_transform = 1
        else:
            self.apply_transforms = True
            self.num_transform = len(self.transform_list)
        self.train_seg_conf_th = dataset_args.get('train_seg_conf_th', 0.0)
        self.seg_len = dataset_args.get('seg_len', 12)
        self.seg_stride = dataset_args.get('seg_stride', 1)
        self.abnormal_file_list = abnormal_file_list 
        self.segs_data_np, self.segs_meta_lst, self.file_list = gen_dataset(path_to_json_dir, num_clips=num_clips, 
                                                            subsample_abnormal=subsample_abnormal,#for classifier
                                                            subsample_normal=subsample_normal,
                                                            file_list=file_list,
                                                            **dataset_args)
    
        if abnormal_train_path is not None or len(abnormal_file_list)>0:
            print('loading abnormal data')
            # supervised anomaly detection, labels are 1 for normal -1 for abnormal
            self.segs_data_np_ab, self.segs_meta_lst_ab, self.file_list_ab  = gen_dataset(abnormal_train_path, num_clips=num_clips, 
                                                                       subsample_abnormal=subsample_abnormal,#ret_keys=True,
                                                                       file_list=abnormal_file_list,
                                               #ret_global_data=return_global, 
                                               **dataset_args)
            num_normal_samp = self.segs_data_np.shape[0]
            num_abnormal_samp = self.segs_data_np_ab.shape[0]
            total_num_samp = num_normal_samp + num_abnormal_samp
            print("Num of abnormal samples: {}  | Num of normal samples: {}  |  Precent abnormal: {}".format(
                num_abnormal_samp, num_normal_samp, num_abnormal_samp / total_num_samp))
            if dataset_args.get('classifier') and dataset_args.get('num_classes')==2:
                self.labels = np.concatenate((np.zeros(num_normal_samp), np.ones(num_abnormal_samp)),
                                         axis=0).astype(int)
            else:
                self.labels = np.concatenate((np.ones(num_normal_samp), -np.ones(num_abnormal_samp)),
                                         axis=0).astype(int)
            self.segs_data_np = np.concatenate((self.segs_data_np, self.segs_data_np_ab), axis=0)
            self.segs_meta_lst = np.concatenate((self.segs_meta_lst, self.segs_meta_lst_ab), axis=0)
            self.file_list = np.concatenate((self.file_list, self.file_list_ab), axis=0)
        else:
            if dataset_args.get('classifier'):
                #supervised classifier, labels are either behaviors or normal/abnormal with the abnormal=1
                if dataset_args.get('num_classes')>2:
                    self.labels = [CLASSIFIER_CLASS2IDX[dataset_args.get('dataset')][m['label']] for m in self.segs_meta_lst] 
                else:
                    self.labels = [m['label']=='abnormal' for m in self.segs_meta_lst]
                self.labels = np.array(self.labels).astype(int)
            else:
                #unsupervised - all labels are 1
                if not evaluate:
                    self.labels = np.ones(self.segs_data_np.shape[0]).astype(int)
                else:
                    #test set:
                    self.labels = [-1 if m['label']=='abnormal' else 1 for m in self.segs_meta_lst]
                    self.labels = np.array(self.labels).astype(int)

        self.metadata = self.segs_meta_lst
        self.num_samples, self.C, self.T, self.V = self.segs_data_np.shape

    def __getitem__(self, index):
        # Select sample and augmentation. I.e. given 5 samples and 2 transformations,
        # sample 7 is data sample 7%5=2 and transform is 7//5=1
        if self.apply_transforms:
            sample_index = index % self.num_samples
            trans_index = math.floor(index / self.num_samples)
            data_numpy = np.array(self.segs_data_np[sample_index])
            data_transformed = self.transform_list[trans_index](data_numpy)
        else:
            sample_index = index
            data_transformed = np.array(self.segs_data_np[index])

        ret_arr = [data_transformed, self.labels[sample_index]]
        return ret_arr

    def __len__(self):
        return self.num_transform * self.num_samples


def get_dataset_and_loader(args, trans_list, only_test=False):
    loader_args = {'batch_size': args.batch_size, 'num_workers': args.num_workers, 'pin_memory': False}
    dataset_args = {'headless': args.headless, 'scale': args.norm_scale, 'scale_proportional': args.prop_norm_scale,
                    'seg_len': args.seg_len, 'return_indices': True, 'return_metadata': True, "dataset": args.dataset,
                    'train_seg_conf_th': args.train_seg_conf_th, 'specific_clip': args.specific_clip,
                    'arena_coords': args.arena_coords, 'model_layout': args.model_layout, 
                    'classifier': args.classifier, 'num_classes': args.num_classes, 'individual_ids': args.individual_ids,
                    }
    print(dataset_args)
    dataset, loader = dict(), dict()
    splits = ['train', 'test'] if not only_test else ['test']
    if len(args.file_list['train'])==0 and not args.classifier and args.pose_path_train_abnormal==None:
        # vanilla unsupervised. we'll override the current mechanisms to give the alg both abnormal and normal data and make sure they get the label normal
        # we'll do this by giving a file list 
        lambda_file = lambda folder: sorted(glob.glob(os.path.join(folder,'*.npy')))
        args.file_list['train'] = lambda_file(os.path.join(args.pose_path['train'], 'normal'))+lambda_file(os.path.join(args.pose_path['train'], 'abnormal'))
    for split in splits:
        evaluate = split == 'test'
        if evaluate or (args.dataset != 'Larvae2019'):
            dataset_args['individual_ids'] = [] #make sure we use all individuals in the test set, also individual analysis only supported for Larvae2019?
        if split == 'train' or 'test_abnormal' in args.pose_path.keys():
            abnormal_train_path = args.pose_path_train_abnormal  #IMPORTANT:*This isto help with the weird test set*
        else:
            abnormal_train_path = None
        abnormal_file_list = args.file_list['train_abnormal'] if split == 'train' else []
        dataset_args['trans_list'] = trans_list[:args.num_transform] if (split == 'train' and args.num_transform!=0) else None
        dataset_args['seg_stride'] = args.seg_stride if split == 'train' else 1  # No strides for test set
        dataset[split] = PoseSegDataset(args.pose_path[split], evaluate=evaluate,
                                        abnormal_train_path=abnormal_train_path,
                                        subsample_abnormal=args.subsample_abnormal[split],
                                        subsample_normal=args.subsample_normal[split],
                                        file_list=args.file_list[split],
                                        abnormal_file_list = abnormal_file_list,
                                        **dataset_args)
        print(f'Loaded {split} dataset with {len(dataset[split])} samples')
        loader[split] = DataLoader(dataset[split], **loader_args, shuffle=(split == 'train'))
    if only_test:
        loader['train'] = None
    return dataset, loader

def get_file_list(person_json_root,subsample_abnormal=[],subsample_normal=[],**dataset_args):
    #subsample abnormal is a list of indices to use as a subsample out of the whole abnormal class
    # this is use for a rarefaction study comparing the performance of the classifier and supervised anomaly detection
    # in low data regimes
    lambda_file = lambda folder: sorted(glob.glob(os.path.join(folder,'*.npy')))
    if person_json_root.endswith('test') or ('abnormal' in person_json_root):
        search_folder = person_json_root
        file_list = lambda_file(search_folder)
        if len(subsample_abnormal)>0 and (not person_json_root.endswith('test')):
            file_list = np.array(file_list)
            file_list = list(file_list[subsample_abnormal])
    elif dataset_args.get('classifier'):
        search_folder = os.path.join(person_json_root,'normal')#search normal first subfolders
        file_list = lambda_file(search_folder)
        if len(subsample_normal)>0:
            file_list = np.array(file_list)
            file_list = list(file_list[subsample_normal])
        search_folder = os.path.join(person_json_root,'abnormal')#then abnormal
        print(search_folder)
        tmp = lambda_file(search_folder)
        if len(subsample_abnormal)>0:
            tmp = np.array(tmp)
            tmp = list(tmp[subsample_abnormal])
        file_list = file_list + tmp
    else:
        search_folder = os.path.join(person_json_root,'normal')
        file_list = lambda_file(search_folder)
        if len(subsample_normal)>0:
            file_list = np.array(file_list)
            file_list = list(file_list[subsample_normal])
       # dir_list = os.listdir(person_json_root)
    #print(search_folder)
    #file_list = #sorted([fn for fn in dir_list if fn.endswith('.npy')])
    print(len(file_list))
    return file_list


def gen_dataset(person_json_root, num_clips=None, subsample_abnormal=[], subsample_normal=[], file_list=[],**dataset_args):
    segs_data_lst = []
    segs_meta_lst = []
    individual_ids = dataset_args.get('individual_ids', [])
    keep_all_individuals = len(individual_ids)==0 
    start_ofst = dataset_args.get('start_ofst', 0)
    seg_stride = dataset_args.get('seg_stride', 1)
    seg_len = dataset_args.get('seg_len', 12)
    seg_conf_th = dataset_args.get('train_seg_conf_th', 0.0)
    if len(file_list)==0:
        file_list = get_file_list(person_json_root,subsample_abnormal=subsample_abnormal,subsample_normal=subsample_normal, **dataset_args)
    if num_clips is not None:
        file_list = [file_list[num_clips]]  # For debugging purposes
    for pose_clip_path in tqdm(file_list):
        clip_data = np.load(pose_clip_path)#os.path.join(person_json_root, pose_clip_path))
        with open(pose_clip_path.replace('.npy', '.json'), 'r') as f:#os.path.join(person_json_root, pose_clip_path.replace('.npy', '.json')), 'r') as f:
            meta_data = json.load(f)
            try:
                ind_to_keep = meta_data['fish_id'] in individual_ids # won't work for meerkat and PoseR because they don't have fish_id
            except:
                ind_to_keep = True
        if keep_all_individuals or ind_to_keep:#works only for Johnson dataset
            if dataset_args.get('classifier') and dataset_args.get('num_classes')>2:
                label = meta_data['behavior_label']
            else:
                key = 'cluster_id' if 'cluster_id' in meta_data.keys() else 'behavior_id' #fish datasets use cluster_id, meerkat uses behavior_id
                label = 'abnormal' if meta_data[key] in ABNORMAL_CLUSTERS[dataset_args.get('dataset')] else 'normal'
            clip_segs_data_np, clip_segs_meta = gen_clip_seg_data_np(
                clip_data,
                meta_data,
                start_ofst,
                seg_stride,
                seg_len,
                label,
                arena_coords=dataset_args.get('arena_coords'),
                model_layout=dataset_args.get('model_layout')
            )

            segs_data_lst.append(clip_segs_data_np)
            segs_meta_lst += clip_segs_meta

    # Global data
    segs_data_np = np.concatenate(segs_data_lst, axis=0)
    # Format is B, C, T, V
    segs_data_np = np.transpose(segs_data_np, (0, 2, 3, 1)).astype(np.float32)

    # if seg_conf_th > 0.0:
    #     segs_data_np, segs_meta, segs_score_np = \
    #         seg_conf_th_filter(segs_data_np, segs_meta_lst, seg_conf_th)
    return segs_data_np, segs_meta_lst, file_list


def keypoints17_to_coco18(kps):
    """
    Convert a 17 keypoints coco format skeleton to an 18 keypoint one.
    New keypoint (neck) is the average of the shoulders, and points
    are also reordered.
    """
    kp_np = np.array(kps)
    neck_kp_vec = 0.5 * (kp_np[..., 5, :] + kp_np[..., 6, :])
    kp_np = np.concatenate([kp_np, neck_kp_vec[..., None, :]], axis=-2)
    opp_order = [0, 17, 6, 8, 10, 5, 7, 9, 12, 14, 16, 11, 13, 15, 2, 1, 4, 3]
    opp_order = np.array(opp_order, dtype=np.int)
    kp_coco18 = kp_np[..., opp_order, :]
    return kp_coco18


def seg_conf_th_filter(segs_data_np, segs_meta, segs_score_np, seg_conf_th=2.0):
    # seg_len = segs_data_np.shape[2]
    # conf_vals = segs_data_np[:, 2]
    # sum_confs = conf_vals.sum(axis=(1, 2)) / seg_len
    sum_confs = segs_score_np.mean(axis=1)
    seg_data_filt = segs_data_np[sum_confs > seg_conf_th]
    seg_meta_filt = list(np.array(segs_meta)[sum_confs > seg_conf_th])
    segs_score_np = segs_score_np[sum_confs > seg_conf_th]

    return seg_data_filt, seg_meta_filt, segs_score_np
