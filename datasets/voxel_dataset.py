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
import random
# import voxel
# import geome
# import tri


def files_exist(files):
    return all([osp.exists(f) for f in files])

class EV_Gait_3DGraph_Dataset(Dataset):

    def __init__(self, root, mode,split, transform=None, spatial_transform=None, temporal_transform=None,pre_transform=None):
        self.root = root
        self.train_test = mode
        self.split = split
        
        self.branch_v = 'v2g_151515_2048_max3w'

        self.G_path_list=[]
        self.labels=[]
        if self.split == 'txt':
            txt_file = os.path.join(self.root,'Ncal_{}.txt'.format(self.train_test))

            with open(txt_file,'r') as anno_file:
                while(1):
                    anno = anno_file.readline()
                    if not anno:
                        break
                    repath = anno.split(' ')[0]
                    cls_name = repath.split(os.sep)[0]
                    label = anno.split(' ')[2]
                    file_name =  repath.split(os.sep)[1]+'.pt'
                    g_file_path = os.path.join(self.root,self.branch_v,cls_name,file_name)

                    self.G_path_list.append(g_file_path)
                    self.labels.append(label)
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
        label = int(self.labels[idx])
        data_v = torch.load(v_file_path)
        if self.transform is not None:
            data_v = self.transform(data_v)
  
        return label,data_v
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