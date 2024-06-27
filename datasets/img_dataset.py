# -- coding: utf-8 --**
# the dataset class for EV-Gait-3DGraph model


import os
import numpy as np
import glob
import pdb
import scipy.io as sio
import torch
import torch.utils.data
from torch_geometric.data import Data, Dataset
import os.path as osp
from PIL import Image
import random
# import voxel
# import geome
# import tri


def files_exist(files):
    return all([osp.exists(f) for f in files])
def load_video(root,mode):
    root_path = os.path.join(root,'rawframes')
    labels = []
    rgb_samples = []
    anno_file = os.path.join(root,'ASL_{}.txt'.format(mode))
    #anno_file = os.path.join(root,'Ncal_{}.txt'.format(mode))
    with open(anno_file, 'r') as fin:
        for line in fin:
           
            line_split = line.strip().split()
            idx = 0
            frame_dir = line_split[idx]
            img_list = os.listdir(os.path.join(root_path,frame_dir))
            img_path = []
            for img in img_list:
                
                img_path.append(os.path.join(root_path,frame_dir,img))
                img_path.sort()
         
            rgb_samples.append(img_path) 
            label = line_split[idx+2]
            labels.append(label)
    return rgb_samples, labels

class EV_Gait_3DGraph_Dataset(Dataset):
    def __init__(self, root_path, mode, split=None, transform=None,spatial_transform=None, temporal_transform=None):
        self.root_path = root_path
        self.rgb_samples, self.labels = load_video(root_path, mode)
        self.sample_num = len(self.rgb_samples)
        self.spatial_transform = spatial_transform
        self.temporal_transform = temporal_transform

    def __getitem__(self, idx):
        rgb_name = self.rgb_samples[idx]
        label = self.labels[idx]
        indices = [i for i in range(len(rgb_name))]
        selected_indice = self.temporal_transform(indices)
        clip_frames = []
        for i, frame_name_i in enumerate(selected_indice):
            rgb_cache = Image.open(rgb_name[frame_name_i]).convert("RGB")
            clip_frames.append(rgb_cache)
        clip_frames = self.spatial_transform(clip_frames)
        # pdb.set_trace()
        n, h, w = clip_frames.size()
        return int(label),clip_frames.view(-1, 3, h, w)
    def __len__(self):
        return int(self.sample_num)
        

    @property
    def raw_file_names(self):
        pass

    # get all file names in  self.processed_dir
    @property
    def processed_file_names(self):
        pass
    def _process(self):
        pass
    def _download(self):
        pass

    def download(self):
        pass
    def process(self):
        pass
    def get():
        pass