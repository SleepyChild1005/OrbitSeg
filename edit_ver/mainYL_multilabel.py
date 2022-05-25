import os
import random
import argparse
import time
import datetime
import dateutil.tz
import torch
import pandas as pd
import numpy as np

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu_id", type=str, default='0', help='GPU number')
    parser.add_argument("--num_workers", type=int, default=4, help='worker number')
    parser.add_argument("--epochs", type=int, default=500, help="epochs")
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    
    parser.add_argument('--learning_rate', type=float, default=5e-5, help='generator learning rate')     
    parser.add_argument('--total_fold_num', type=int, default=4, help='num of total fold')
    parser.add_argument('--fold_num', type=int, nargs='+', required=True, help='num of target fold')
    
    parser.add_argument("--num_sequence", type=int, default=3, help='number of sequence')
    parser.add_argument("--image_size", type=int, default=64, help='image size')

    parser.add_argument("--bAttention", type=int, default=0, help='attention model or not')
    parser.add_argument("--bTransfer_learning", type=int, default=0, help='transfer learning or not')

    parser.add_argument("--exp_name", type=str, default='test', help='experiment name')
    parser.add_argument("--pretrained_model_path", type=str, default='pretrained_model/nodule_sensor3d_attention_final.pth', help='pretrained_model_path')
    parser.add_argument("--dataset_path", type=str,
                        default='/home/ubuntu/Workspace/Wonseo/Orbit_Seg_bySJ/Orbit_Dataset_multilabel/Sequential_Slices_Clipping_5/',
                        help='tg_dataset_path')


    opt = parser.parse_args()
    print(opt)

    from tg_dataset_multilabel import TargetDataset
    from trainer_multilabel import sequentialSegTrainer as trainer
    from torch.utils.tensorboard import SummaryWriter

    now = datetime.datetime.now(dateutil.tz.tzlocal())
    timestamp = now.strftime('%Y_%m%d__%H_%M___')
    exp_folder_name = str(timestamp) + opt.exp_name

    train_dataset = []
    test_dataset = []
    print('check opt.total_fold_num = ',opt.total_fold_num)
    for fold in range(opt.total_fold_num):
        train_dataset.append(TargetDataset(opt.dataset_path, 'train', fold, opt.total_fold_num))
        test_dataset.append(TargetDataset(opt.dataset_path,'test', fold, opt.total_fold_num))
        print(train_dataset[fold].get_num_patient(), test_dataset[fold].get_num_patient())

    for fold in opt.fold_num:
        print(fold, '-th fold ::: training start')

        print(fold, 'th fold train/test seq:: ', len(train_dataset[fold]), len(test_dataset[fold]))
        print(fold, 'th fold train/test patient:: ', train_dataset[fold].get_num_patient(), test_dataset[fold].get_num_patient())

        train_dataloader = torch.utils.data.DataLoader(train_dataset[fold], batch_size=opt.batch_size, drop_last=False, shuffle=True, num_workers=opt.num_workers)
        test_dataloader = torch.utils.data.DataLoader(test_dataset[fold], batch_size=opt.batch_size, drop_last=False, shuffle=True, num_workers=opt.num_workers)

        fold_timestamp = datetime.datetime.now(dateutil.tz.tzlocal()).strftime('%m_%d_%H_%M__')
        fold_exp_name = str(fold) + 'th_fold_'

        output_dir = './experiments_sensor3d_multilabel_final/%s/%s' % (exp_folder_name, fold_exp_name)
        writer_path = './log/%s/%s' % (exp_folder_name, fold_exp_name)
        os.makedirs(writer_path)
        writer = SummaryWriter(writer_path)



        algo = trainer(epochs= opt.epochs,
                        gpu= opt.gpu_id,
                        batch_size= opt.batch_size,
                        image_size= opt.image_size,
                        learning_rate= opt.learning_rate,
                        output_dir= output_dir,
                        bAttention = opt.bAttention,
                        bTransfer_learning = opt.bTransfer_learning,
                        pretrained_model_dir=  opt.pretrained_model_path,
                        train_dataloader= train_dataloader,
                        test_dataloader= test_dataloader,
                        writer= writer)

        start_t = time.time()
        algo.train()
        end_t = time.time()

        print(fold, '-th fold ::: total time for training: ', end_t - start_t)