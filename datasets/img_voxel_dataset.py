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

class EV_Gait_3DGraph_Dataset(Dataset):

    def __init__(self, root, mode, split, transform=None, spatial_transform=None, temporal_transform=None,pre_transform=None):
        self.root = root

        self.spatial_transform = spatial_transform
        self.temporal_transform = temporal_transform
        self.img_root = root
        self.train_test = mode
        self.split = split
        
        self.branch_v = 'v2g_8_r3_base'
        self.labels = []
        self.G_path_list=[]
        if self.split == 'txt':
            txt_file = os.path.join(self.root,'ASL_cls5_{}.txt'.format(self.train_test))

            with open(txt_file,'r') as anno_file:
                while(1):
                    anno = anno_file.readline()
                    if not anno:
                        break
                    repath = anno.split(' ')[0]
                    cls_name = repath.split(os.sep)[0]
                    file_name =  repath.split(os.sep)[1]+'.pt'
                    label = anno.split(' ')[2]
                    p_file_path = os.path.join(self.root,self.branch_v,cls_name,file_name)
                    self.labels.append(label)
                    self.G_path_list.append(p_file_path)
        else:
            path = os.path.join(self.root,'{}'.format(self.train_test),self.branch_v)
            cls_list = os.listdir(path)
            for cls_id in range(len(cls_list)):
                cls = cls_list[cls_id]
                file_list = os.listdir(os.path.join(path,cls))
                for file_id in range(len(file_list)):
                    file_name = file_list[file_id]
                    # if cls.find('cars')!=-1:
                    #     label=1
                    # else:
                    #     label=0
                    label=int(cls)
                    self.G_path_list.append(os.path.join(path,cls,file_name))
                    self.labels.append(label)

        super(EV_Gait_3DGraph_Dataset, self).__init__(root, transform, pre_transform)
        
    def __len__(self):
        return len(self.G_path_list)

    def __getitem__(self, idx):
        v_file_path = self.G_path_list[idx]
        data_v = torch.load(v_file_path)

        if self.transform is not None:
            data_v = self.transform(data_v)
        
        cls_name = v_file_path.split(os.sep)[-2]
        file_name = v_file_path.split(os.sep)[-1]
        #/DATA/gblav4/yuanchengguo/Gait_Identification-main/data/celex/celex_dual_graph_cls150
        # /voxel2geometric/action_001_pull_up_weeds/action_001_20220221_110904108_EI_70M.pt
        video = file_name.split('.')[0]
        label = int(self.labels[idx])
        img_dir = os.listdir(os.path.join(self.img_root,'rawframes',cls_name,video))
        img_dir.sort()
        rgb_name = []
        for img in img_dir:
            rgb_name.append(os.path.join(self.img_root,'rawframes',cls_name,video,img))
  
        indices = [i for i in range(len(rgb_name))]
        selected_indice = self.temporal_transform(indices)
        clip_frames = []
        for i, frame_name_i in enumerate(selected_indice):
            rgb_cache = Image.open(rgb_name[frame_name_i]).convert("RGB")
            clip_frames.append(rgb_cache)
  
        clip_frames = self.spatial_transform(clip_frames)
       
        n, h, w = clip_frames.size()
        return int(label),data_v,clip_frames.view(-1, 3, h, w)

        # return file list of self.raw_dir
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