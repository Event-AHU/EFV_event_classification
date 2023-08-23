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
    data_path = 'xxx'
    save_path = 'xxx'
    device = torch.device("cuda:0")

    class_dir = os.listdir(data_path)
    class_dir.sort()
    # label_dict = {lbl: i for i, lbl in enumerate(class_dir)}
    for classID in range(len(class_dir)):
        cls = class_dir[classID]
        label = classID
        cls_name = cls
        fileLIST = os.listdir(os.path.join(data_path, cls))
        if not os.path.exists(os.path.join(save_path,'v2g_8_r3_base', cls_name)):
            os.makedirs(os.path.join(save_path,'v2g_8_r3_base',  cls_name))
        for FileID in tqdm(range(len(fileLIST))):
            file_Name = fileLIST[FileID]
            save_name = file_Name.split('.')[0]
            if not os.path.exists(os.path.join(save_path,'v2g_8_r3_base', cls_name)):
                os.makedirs(os.path.join(save_path,'v2g_8_r3_base', cls_name))
            if os.path.exists(os.path.join(save_path,'v2g_8_r3_base', cls_name, '{}.pt'.format(save_name))):
                continue
            read_path = os.path.join(data_path, cls,'{}.mat'.format(save_name))
            data = sio.loadmat(read_path)
            feature,position,edges = generate_graph(data)

            feature = torch.tensor(feature)[:, :].float()
            edge_index = torch.tensor(
                np.array(edges).astype(np.int32), dtype=torch.long
            )
            pos = torch.tensor(np.array(position), dtype=torch.float32)
            label_idx = torch.tensor(int(label), dtype=torch.long)
            data = Data(
                x=feature, edge_index=edge_index, pos=pos, y=label_idx.unsqueeze(0)
            )

            G_save_path =os.path.join(os.path.join(save_path,'v2g_8_r3_base', cls_name, '{}.pt'.format(save_name)))

            torch.save(data, G_save_path)