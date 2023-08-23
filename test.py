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
from models.dual_model import Net as dual_model
from datasets.img_voxel_dataset import EV_Gait_3DGraph_Dataset as dual_dataset
from datasets.spatial_transforms import *
from datasets.temporal_transforms import *
from ptflops import get_model_complexity_info
from thop import profile


if __name__ == '__main__':
    if not os.path.exists(Config.log_dir):
        os.makedirs(Config.log_dir)

    if not os.path.exists(Config.model_dir):
        os.makedirs(Config.model_dir)

    parser = argparse.ArgumentParser()
    # parser.add_argument("--cuda", default="0", help="The GPU ID")
    parser.add_argument("--model_name", default="Test_EV_Gait_3DGraph.pkl")
    parser.add_argument("--batch_size", default=8, type=int, help="batch size")
    parser.add_argument("--type", default="dual", type=str, help="train type",required=True)
    parser.add_argument('--base_model', default='resnet50', type=str)
    parser.add_argument("--clip_len", default=8, type=int)
    parser.add_argument("--num_works", default=8, type=int)
    parser.add_argument('--local_rank', default=-1, type=int,
                    help='node rank for distributed training')
    args = parser.parse_args()

    # seed_everything(1234)
    torch.multiprocessing.set_sharing_strategy('file_system')
    torch.backends.cudnn.benchmark = False
    logging.basicConfig(filename=Config.graph_train_log_path, level=logging.DEBUG)

    # device = torch.device("cuda:" + args.cuda if torch.cuda.is_available() else "cpu")

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
    model.load_state_dict(torch.load(os.path.join(Config.model_dir, args.model_name),map_location='cpu'),True)
    torch.cuda.set_device(args.local_rank)
    
    model.cuda(args.local_rank)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank])

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    input_mean=[.485, .456, .406]
    input_std=[.229, .224, .225]
    normalize = GroupNormalize(input_mean, input_std)
        
    scales = [1, .875, .75, .66]   
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
    dataset_factory = dual_dataset
    test_data_aug = T.Compose([T.Cartesian(cat=False), T.RandomScale([0.999, 1])])
    test_dataset = dataset_factory(
        Config.ASL_root_dir, mode='test',split='txt',transform=test_data_aug,
        spatial_transform=trans_test, temporal_transform = temporal_transform_test
    )
    
    test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, sampler=test_sampler,num_workers=args.num_works, pin_memory=True)

    # test
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for index,batch in enumerate(tqdm(test_loader)):
            label = batch[0].cuda(args.local_rank,non_blocking=True)
            data_voxel = batch[1].cuda(args.local_rank,non_blocking=True)
            data_frame = batch[2].cuda(args.local_rank,non_blocking=True)
            end_point = model(data_voxel,data_frame)
            pred = end_point.max(1)[1]
            total += len(label)
            correct += pred.eq(label).sum().item()
            # pdb.set_trace()
        logging.info("test acc is {}".format(float(correct) / total))
        print("test acc is {}".format(float(correct) / total))
