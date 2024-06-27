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
from abc import abstractmethod


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
        
        self.branch_v = 'v2g_151515_2048_max3w'#v2g_8_r3_base,v2g_2022_512_8_r2,v2g_151515_2048_max3w,poker_graph/v2g_8_r3_base,HARDVS_graph/v2g_8_r3_base
        self.labels = []
        self.G_path_list=[]
        self.path_names = []
        if self.split == 'txt':
            #txt_file = os.path.join(self.root,'ASL_{}.txt'.format(self.train_test))
            #txt_file = os.path.join(self.root,'Ncal_{}.txt'.format(self.train_test))
            #txt_file = os.path.join('/media/amax/c08a625b-023d-436f-b33e-9652dc1bc7c0/DATA/dataset/Poker_Event/Poker_list/{}_label.txt'.format(self.train_test))
            #txt_file = os.path.join('/media/amax/c08a625b-023d-436f-b33e-9652dc1bc7c0/DATA/dataset/HARDVS/HarDvs300_list/{}_label.txt'.format(self.train_test))
            txt_file = os.path.join('/media/amax/c08a625b-023d-436f-b33e-9652dc1bc7c0/DATA/dataset/bully10k/{}_label.txt'.format(self.train_test))
            #breakpoint()
            with open(txt_file,'r') as anno_file:
                while(1):
                    anno = anno_file.readline()
                    if not anno:
                        break
                    #repath = anno.split(' ')[0]
                    repath = anno.split(' ')[0].split('\t')[0] #punching/punching_bf_hx_L_dark/dvSave-2022_09_07_20_43_28
                    cls_name = repath.split(os.sep)[0]          #punching
                    file_name =  repath.split(os.sep)[2]+'.pt' #dvSave-2022_09_07_20_43_28.pt
                    path_name = repath.split(os.sep)[1]  # 从repath中获取path_name
                    self.path_names.append(path_name)  # 将path_name添加到列表中
                    #breakpoint()
                    #label = anno.split(' ')[2]
                    label = anno.split('\t')[2].strip()  #0
                    #breakpoint()
                    p_file_path = os.path.join(self.root,self.branch_v,cls_name,file_name)
                    self.labels.append(label)
                    self.G_path_list.append(p_file_path)
        else:
            #breakpoint()
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
        #breakpoint()
        super(EV_Gait_3DGraph_Dataset, self).__init__(root, transform, pre_transform)
        
    def __len__(self):
        return len(self.G_path_list)

    def __getitem__(self, idx):
        v_file_path = self.G_path_list[idx]
        data_v = torch.load(v_file_path)
        path_name = self.path_names[idx]
        #print(idx)
        if self.transform is not None:
            data_v = self.transform(data_v)
        #print (v_file_path)  #/media/amax/c08a625b-023d-436f-b33e-9652dc1bc7c0/DATA/dataset/bully10k/bully10k_graph/v2g_8_r3_base/slapping/dvSave-2022_02_28_14_52_54.pt
        cls_name = v_file_path.split(os.sep)[-2]
        file_name = v_file_path.split(os.sep)[-1]
           
        video = file_name.split('.')[0]
        label = int(self.labels[idx]) 
         
        #img_dir = os.listdir(os.path.join(self.img_root,'rawframes',cls_name,video))#,self.train_test
        #img_dir = os.listdir(os.path.join('/media/amax/c08a625b-023d-436f-b33e-9652dc1bc7c0/DATA/dataset/Poker_Event/Poker_EventImg',cls_name,video))#,self.train_test
        #img_dir = os.listdir(os.path.join('/media/amax/c08a625b-023d-436f-b33e-9652dc1bc7c0/DATA/dataset/HARDVS/rawframes',cls_name,video,video + '_dvs'))
        img_dir = os.listdir(os.path.join('/media/amax/c08a625b-023d-436f-b33e-9652dc1bc7c0/DATA/dataset/bully10k/bully10kimg',cls_name,path_name,video))
        # print(cls_name)#action_014
        #print(video)#dvSave-2021_08_31_17_10_46
        # print(video + '_dvs/')
        # breakpoint()
        img_dir.sort()
        rgb_name = []
        for img in img_dir:
            #rgb_name.append(os.path.join(self.img_root,'rawframes',cls_name,video,img))
            #rgb_name.append(os.path.join('/media/amax/c08a625b-023d-436f-b33e-9652dc1bc7c0/DATA/dataset/Poker_Event/Poker_EventImg',cls_name,video,img))
            #rgb_name.append(os.path.join('/media/amax/c08a625b-023d-436f-b33e-9652dc1bc7c0/DATA/dataset/HARDVS/rawframes',cls_name,video, video + '_dvs/',img))
            rgb_name.append(os.path.join('/media/amax/c08a625b-023d-436f-b33e-9652dc1bc7c0/DATA/dataset/bully10k/bully10kimg',cls_name,path_name,video,img))
            #print (rgb_name)
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

    