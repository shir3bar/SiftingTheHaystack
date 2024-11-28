import random
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from models.STG_NF.model_pose import STG_NF
from models.training import Trainer
from utils.data_utils import trans_list
from utils.optim_init import init_optimizer, init_scheduler
from args import create_exp_dirs
from args import init_parser, init_sub_args
from dataset import get_dataset_and_loader
from utils.train_utils import dump_args, init_model_params
from utils.scoring_utils import score_dataset, score_dataset_classifier
from utils.train_utils import calc_num_of_params
from models.STG_NF.stgcn import ST_GCN_18
import os
ind_ids = [  1,   2,   3,   4,   6,   7,   8,   9,  10,  11,  12,  13,  14,
        15,  17,  19,  21,  22,  24,  25,  26,  27,  28,  29,  30,  31,
         32,  33,  34,  36,  37,  38,  39,  40,  41,  43,  46,  47,  50,
         51,  52,  53,  55,  56,  58,  59,  60,  62,  63,  64,  65,  66,
         67,  68,  69,  70,  71,  73,  75,  78,  79,  80,  81,  82,  83,
         84,  86,  88,  89,  90,  91,  92,  93,  95,  97,  98,  99, 103,
        104, 105, 108, 109, 110, 111, 112, 114, 115, 116, 118, 119, 120,
      122, 123, 124, 125, 126, 127, 128, 129, 113]
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
        np.random.seed(args.seed)

    args, model_args = init_sub_args(args)
    # added for simulation data organization:
    #std_rare = '_'.join(args.data_dir.split('_')[-2:])
    #print(f'vamossss {std_rare}')
    args.ckpt_dir = create_exp_dirs(args.exp_dir, dirmap=os.path.join(args.dataset,'new_unsupervised_exps'))#'simulation_exps_classifier',std_rare))# remove when done
    if args.individuals_to_sample>0 and len(args.individual_ids)==0:
        args.individual_ids = np.random.choice(ind_ids,args.individuals_to_sample,replace=False)
        print(f'Using individuals {args.individual_ids}')

    pretrained = vars(args).get('checkpoint', None)
    dataset, loader = get_dataset_and_loader(args, trans_list=trans_list, only_test=(pretrained is not None))
    model_args = init_model_params(args, dataset)
    print(model_args)
    if args.classifier:
        model = ST_GCN_18(in_channels=2, num_class=args.num_classes, layout=args.model_layout, 
                          max_hop=args.max_hops, strategy=args.adj_strategy
                          )
    else:
        model = STG_NF(**model_args)
    num_of_params = calc_num_of_params(model)
    args.num_of_params = num_of_params
    trainer = Trainer(args, model, loader['train'], loader['test'],
                      optimizer_f=init_optimizer(args.model_optimizer, lr=args.model_lr),
                      scheduler_f=init_scheduler(args.model_sched, lr=args.model_lr, epochs=args.epochs))
    if pretrained:
        trainer.load_checkpoint(pretrained)
    else:
        writer = SummaryWriter(log_dir=args.ckpt_dir)
        trainer.train(log_writer=writer,save_every=args.save_every)
        if type(args.individual_ids)==np.ndarray:
            args.individual_ids = args.individual_ids.astype(np.uint8).tolist()
        dump_args(args, args.ckpt_dir)

    normality_scores = trainer.test()
    if args.classifier:
        if args.num_classes>2:
            top1_acc,top2_acc, ap, balanced_acc, scores = score_dataset_classifier(normality_scores, dataset["test"].metadata, args=args)

            # Logging and recording results
            print("\n-------------------------------------------------------")
            print("\033[92m Done with {} top1_acc, {} top2_acc and {} AP for {} samples\033[0m".format(top1_acc, top2_acc, ap, scores.shape[0]))
            print("-------------------------------------------------------\n\n")
        else:
            auroc, auprc, scores = score_dataset_classifier(normality_scores, dataset["test"].metadata, args=args)

            # Logging and recording results
            print("\n-------------------------------------------------------")
            print("\033[92m Done with {}% AuROC and {}% AuPRC for {} samples\033[0m".format(auroc * 100, auprc * 100, scores.shape[0]))
            print("-------------------------------------------------------\n\n")

    else:
        auroc,auprc, scores = score_dataset(normality_scores, dataset["test"].metadata, args=args)

        # Logging and recording results
        print("\n-------------------------------------------------------")
        print("\033[92m Done with {}% AuROC and {}% AuPRC for {} samples\033[0m".format(auroc * 100, auprc * 100, scores.shape[0]))
        print("-------------------------------------------------------\n\n")


if __name__ == '__main__':
    main()
