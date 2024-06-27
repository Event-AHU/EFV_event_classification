import os
import pdb
import csv
import numpy as np
import random
import cv2
import torch
import pandas as pd
from tqdm import tqdm
import scipy.io as sio
from torch_geometric.data import Data
from spconv.pytorch.utils import PointToVoxel
import math
def calculate_edges(data, r=5):
    # threshold of radius
    d = 32
    # scaling factor to tune the difference between temporal and spatial resolution
    alpha = 1
    beta = 1
    data_size = data.shape[0]
    # max number of edges is 1000000,
    edges = np.zeros([100000, 2])
    # get t, x,y
    points = data[:, 0:3]
    row_num = 0
    edge_sum=0
    for i in range(data_size - 1):
        count = 0
        distance_matrix = points[i + 1 : data_size + 1, 0:3]
        distance_matrix[:, 1:3] = distance_matrix[:, 1:3] - points[i, 1:3]
        distance_matrix[:, 0] = distance_matrix[:, 0] - points[i, 0]
        distance_matrix = np.square(distance_matrix)
        distance_matrix[:, 0] *= alpha
        distance_matrix[:, 1:3] *= beta
        # calculate the distance of each pair of events
        distance = np.sqrt(np.sum(distance_matrix, axis=1))
        index = np.where(distance <= r)
        # save the edges
        if index:
            index = index[0].tolist()
            for id in index:
                edges[row_num, 0] = i
                edges[row_num + 1, 1] = i
                edges[row_num, 1] = int(id) + i + 1
                edges[row_num + 1, 0] = int(id) + i + 1
                row_num = row_num + 2
                count = count + 1
                edge_sum+=2
                if count > d:
                    break
        if edge_sum>40000:
            break
    edges = edges[~np.all(edges == 0, axis=1)]
    edges = np.transpose(edges)

    return edges

def generate_graph(data):
    feature = data['features']

    position = data['coor']

    edges = calculate_edges(data['coor'], 3)
    return feature,position,edges


if __name__ == '__main__':
    #data_path = '/media/amax/c08a625b-023d-436f-b33e-9652dc1bc7c0/DATA/dataset/HARDVS/HARDVS_voxel/HARDVS_voxel'
    data_path = '/media/amax/c08a625b-023d-436f-b33e-9652dc1bc7c0/DATA/dataset/bully10k/bully10k_voxel'
    save_path = '/media/amax/c08a625b-023d-436f-b33e-9652dc1bc7c0/DATA/dataset/bully10k/bully10k_graph'
    device = torch.device("cuda:6")

    class_dir = os.listdir(data_path)
    class_dir.sort()
    # label_dict = {lbl: i for i, lbl in enumerate(class_dir)}
    for classID in range(len(class_dir)): #['greeting', 'handshake', 'pushing']
        cls = class_dir[classID]
        label = classID
        cls_name = cls
        videoLIST = os.listdir(os.path.join(data_path, cls))
        if not os.path.exists(os.path.join(save_path,'v2g_8_r3_base', cls_name)):
            os.makedirs(os.path.join(save_path,'v2g_8_r3_base',  cls_name))
        for videoID in range(len(videoLIST)):
            video_Name = videoLIST[videoID]
            fileLIST = os.listdir(os.path.join(data_path, cls, video_Name))
            for FileID in tqdm(range(len(fileLIST))):#['dvSave-2021_11_23_16_26_15.npy_voxel'
                file_Name = fileLIST[FileID]
                save_name = file_Name.split('.')[0]
                if not os.path.exists(os.path.join(save_path,'v2g_8_r3_base', cls_name, video_Name)):
                    os.makedirs(os.path.join(save_path,'v2g_8_r3_base', cls_name, video_Name))
                if os.path.exists(os.path.join(save_path,'v2g_8_r3_base', cls_name, video_Name, '{}.pt'.format(save_name))):
                    continue
                #breakpoint() 
                file_subfolders = os.listdir(os.path.join(data_path, cls, video_Name ,file_Name))  # 使用file_Name作为子文件夹的名字
                                
                # ...
                for subfolder in file_subfolders:
                    mat_file_name = '{}'.format(str(subfolder).zfill(8))  # 根据你提供的文件路径,mat文件名应该是8位数字
                    G_save_path = os.path.join(save_path, 'v2g_8_r3_base', cls_name, '{}.pt'.format(save_name)) # move this line up

                    # Check if the file has been processed already
                    if os.path.exists(G_save_path):
                        print(f"{G_save_path} already exists. Skipping this file.")
                        continue

                    if os.path.exists(os.path.join(data_path, cls, video_Name ,file_Name, mat_file_name)):
                        read_path = os.path.join(data_path, cls, video_Name ,file_Name, mat_file_name)
                        try:
                            data = sio.loadmat(read_path)
                            feature, position, edges = generate_graph(data)

                            # Process and save the data only if it was successfully loaded and processed
                            feature = torch.tensor(feature)[:, :].float()
                            edge_index = torch.tensor(np.array(edges).astype(np.int32), dtype=torch.long)
                            pos = torch.tensor(np.array(position), dtype=torch.float32)
                            label_idx = torch.tensor(int(label), dtype=torch.long)
                            data = Data(x=feature, edge_index=edge_index, pos=pos, y=label_idx.unsqueeze(0))
                            torch.save(data, G_save_path)
                        except OSError:
                            print(f"Exception occurred when processing {read_path}. Skipping this file.")
