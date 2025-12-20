# -- coding: utf-8 --**
# train the Run EV-Gait-3DGraph model

import numpy as np
import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.data import DataLoader
import argparse
import pdb
from tqdm import tqdm
import os
import torchvision
import torch.distributed as dist

import logging
import sys
sys.path.append("..")
from config import Config
from model.dual_model import Net as dual_model
from datasets.img_voxel_dataset import EV_Gait_3DGraph_Dataset as dual_dataset
from datasets.spatial_transforms import *
from datasets.temporal_transforms import *


if __name__ == '__main__':
    # pdb.set_trace()
    if not os.path.exists(Config.log_dir):
        os.makedirs(Config.log_dir)

    if not os.path.exists(Config.model_dir):
        os.makedirs(Config.model_dir)

    parser = argparse.ArgumentParser()
    # parser.add_argument("--cuda", default="0", help="The GPU ID")
    parser.add_argument("--epoch", default=150, type=int, help="The GPU ID")
    parser.add_argument("--batch_size", default=8, type=int, help="batch size")
    parser.add_argument("--num_works", default=16, type=int)
    parser.add_argument("--clip_len", default=8, type=int)
    parser.add_argument('--base_model', default='resnet50', type=str)
    parser.add_argument('--local_rank', default=-1, type=int,
                    help='node rank for distributed training')

    args = parser.parse_args()
    torch.multiprocessing.set_sharing_strategy('file_system')
    torch.backends.cudnn.benchmark = False
    # logging.basicConfig(filename=Config.graph_train_log_path, level=logging.DEBUG)
    dist.init_process_group(backend='nccl')
    args.nprocs = torch.cuda.device_count()
    # model = model_dataset[args.type]["model"]()
    model = dual_model(

    dim = 768, 
    num_heads = 4,
    mlp_ratio=4., 
    qkv_bias=False, 
    clip_len = args.clip_len,
    drop=0., 
    attn_drop=0., 
    init_values=1e-5,
    drop_path=0.
)

    torch.cuda.set_device(args.local_rank)
    model.cuda(args.local_rank)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],find_unused_parameters=True)
    # ,find_unused_parameters=True

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    input_mean=[.485, .456, .406]
    input_std=[.229, .224, .225]
    normalize = GroupNormalize(input_mean, input_std)
        
    scales = [1, .875, .75, .66]
    trans_train  = torchvision.transforms.Compose([
                            GroupScale(256),
                            GroupMultiScaleCrop(224, scales),
                            Stack(roll=(args.base_model in ['BNInception', 'InceptionV3'])),
                            ToTorchFormatTensor(div=(args.base_model not in ['BNInception', 'InceptionV3'])),
                            normalize
                            ])
    temporal_transform_train = torchvision.transforms.Compose([
                                        TemporalUniformCrop_train(args.clip_len)
                                        ])    
    trans_test  = torchvision.transforms.Compose([
                            GroupScale(256),
                            GroupCenterCrop(224),
                            Stack(roll=(args.base_model in ['BNInception', 'InceptionV3'])),
                            ToTorchFormatTensor(div=(args.base_model not in ['BNInception', 'InceptionV3'])),
                            normalize
                            ])
    temporal_transform_test = torchvision.transforms.Compose([
                                        TemporalUniformCrop_val(args.clip_len)
                                    ])
    train_data_aug = T.Compose([T.Cartesian(cat=False), T.RandomScale([0.96, 1]), T.RandomTranslate(0.001)])

    dataset_factory = dual_dataset
    train_dataset = dataset_factory(
        # need to modify if switching datasets
        Config.Nmnist_root_dir, mode='train',split='dir',transform=train_data_aug,spatial_transform=trans_train,
        # Config.ASL_root_dir, mode='train',split='txt',transform=train_data_aug,spatial_transform=trans_train,
         temporal_transform = temporal_transform_train
    )
    num_samples = len(train_dataset)

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler, num_workers=args.num_works, pin_memory=True, drop_last=True)
    
    test_data_aug = T.Compose([T.Cartesian(cat=False), T.RandomScale([0.999, 1])])
    test_dataset = dataset_factory(
        # need to modify if switching datasets
        Config.Nmnist_root_dir, mode='test',split='dir',transform=test_data_aug,
        # Config.ASL_root_dir, mode='test',split='txt',transform=test_data_aug,
        spatial_transform=trans_test, temporal_transform = temporal_transform_test
    )
    
    test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, sampler=test_sampler,num_workers=args.num_works, pin_memory=True)
  
    # train
    
    for epoch in range(1, args.epoch+1):
        model.train()

        if epoch == 60:
            for param_group in optimizer.param_groups:
                param_group["lr"] = 0.0001
        if epoch == 110:
            for param_group in optimizer.param_groups:
                param_group["lr"] = 0.00001

        correct = 0
        total = 0
        total_loss = 0
        train_sampler.set_epoch(epoch)
        # train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler, num_workers=args.num_works, pin_memory=True, drop_last=True)
        with tqdm(total=num_samples, desc=f'Epoch {epoch}/{args.epoch}', unit='sample') as pbar:
            for batch in train_loader:

                optimizer.zero_grad()
 

                label = batch[0].cuda(args.local_rank,non_blocking=True)
                data_voxel = batch[1].cuda(args.local_rank,non_blocking=True)
                data_frame = batch[2].cuda(args.local_rank,non_blocking=True)
                # end_point = model(data_voxel)
                end_point = model(data_voxel, data_frame)
                # pdb.set_trace()
                loss = F.nll_loss(end_point, label)
                pred = end_point.max(1)[1]
                total += len(label)
                correct += pred.eq(label).sum()
                total_loss += float(loss)
                loss.backward()
                optimizer.step()
                pbar.set_postfix(**{'loss (batch)': loss.item()})
                pbar.update(len(label))
        if dist.get_rank() == 0:
            # logging.info("epoch: {}, train acc is {}, loss is {}".format(epoch, float(correct) / total,total_loss/len(train_loader)))
            print("epoch: {}, train acc is {},loss is {}".format(epoch, float(correct) / total,total_loss/len(train_loader)))
        if epoch % 10 ==0:
            model.eval()
            correct_t = 0
            total_t = 0
            with torch.no_grad():
                for index,batch in enumerate(tqdm(test_loader)):
                    label = batch[0].cuda(args.local_rank,non_blocking=True)
                    data_voxel = batch[1].cuda(args.local_rank,non_blocking=True)
                    data_frame = batch[2].cuda(args.local_rank,non_blocking=True)
                    end_point = model(data_voxel, data_frame)
                    pred_t = end_point.max(1)[1]
                    total_t += len(label)
                    correct_t += pred_t.eq(label).sum().item()

                if dist.get_rank() == 0:
                    # logging.info("test acc is {}".format(float(correct_t) / total_t))
                    print("test acc is {}".format(float(correct_t) / total_t))

        # accuracy of each epoch
        if dist.get_rank() == 0 and  epoch %10==0:    
            torch.save(model.module.state_dict(), os.path.join(Config.model_dir, 'NMnist_{}_{}_add_neck.pkl'.format(args.type,epoch)))
    # test
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for index,batch in enumerate(tqdm(test_loader)):
            label = batch[0].cuda(args.local_rank,non_blocking=True)
            data_voxel = batch[1].cuda(args.local_rank,non_blocking=True)
            data_frame = batch[2].cuda(args.local_rank,non_blocking=True)
            end_point = model(data_voxel, data_frame)
            pred = end_point.max(1)[1]
            total += len(label)
            correct += pred.eq(label).sum().item()

        # logging.info("test acc is {}".format(float(correct) / total))
        if dist.get_rank() == 0:
            print("test acc is {}".format(float(correct) / total))
            torch.save(model.module.state_dict(), os.path.join(Config.model_dir, 'NMnist_{}_{}_add_neck.pkl'.format(args.type,epoch)))
        # torch.save(model.state_dict(), Config.gcn_model_name.format(epoch))
