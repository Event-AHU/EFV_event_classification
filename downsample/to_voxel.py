import os
import pdb
import csv
import numpy as np
import cv2
import torch
import pandas as pd
from tqdm import tqdm
import scipy.io as sio
from spconv.pytorch.utils import PointToVoxel
data_path = "xxx"
save_path = "xxx"
video_files = os.listdir(data_path)


def transform_points_to_voxels(data_dict = {}, device = torch.device("cuda:0")):
    points = data_dict['points']
    # 将points打乱
    shuffle_idx = np.random.permutation(points.shape[0])
    points = points[shuffle_idx]
    data_dict['points'] = points

    voxel_generator = PointToVoxel(
        # 给定每个voxel的长宽高  
        vsize_xyz=[10, 10, 10], 
        # 给定的范围 
        coors_range_xyz=[0, 0, 0, 200, 239, 179],
        # 给定每个event的特征维度 t,x,y,p
        num_point_features=4,
        # 最多选取多少个voxel，
        max_num_voxels=10000,  
        # 给定每个voxel中有采样多少个点，不够则补0
        max_num_points_per_voxel=8,  # 32
        device=device
    )
    # 使用spconv生成voxel输出
    points = torch.tensor(data_dict['points'])
    points = points.to(device)
    voxel_output = voxel_generator(points)

    voxels, coordinates, num_points = voxel_output
    # pdb.set_trace()
    voxels = voxels.to(device)
    coordinates = coordinates.to(device)
    num_points = num_points.to(device)
    # 选event数量在前5000的voxel
    if num_points.shape[0] < 512:
        features = voxels[:,:,3]
        coor = coordinates[:,:]
    else:
        _, voxels_idx = torch.topk(num_points, 512)
    # 将每个voxel的1024个p拼接作为voxel初始特征
        features = voxels[voxels_idx][:, :, 3]
    # 前5000个voxel的三维坐标
        coor = coordinates[voxels_idx]
    # 将y.x.t改为t,x,y
    coor[:, [0, 1, 2]] = coor[:, [2, 1, 0]]

    return coor, features

if __name__ == '__main__':
    device = torch.device("cuda:0")
    video_files = os.listdir(data_path)

    label_dict = {lbl: i for i, lbl in enumerate(video_files)}
    for videoID in range(len(video_files)):
        videoName = video_files[videoID]
        fileLIST = os.listdir(os.path.join(data_path, videoName))
        if not os.path.exists(os.path.join(save_path,  videoName)):
            os.makedirs(os.path.join(save_path,  videoName))
        mat_save = os.path.join(save_path,  videoName)
        for FileID in tqdm(range(len(fileLIST))):
            file = fileLIST[FileID]
            save_name = file.split('.')[0]
            if os.path.exists(os.path.join(mat_save,'{}.mat'.format(save_name))):
                continue
            label = label_dict.get(videoName, None)
            read_path = os.path.join(data_path, videoName, file)
            mat_file = sio.loadmat(read_path)
            t = mat_file["ts"]
            time_length = t[-1]-t[0]
            t = ((t-t[0]) / time_length) * 200.0
            x = mat_file["x"].astype(np.float64)
            y = mat_file["y"].astype(np.float64)
            p = mat_file["pol"].astype(np.float64)
            p[np.where(p==0)]=-1
            all_events = np.hstack((t,x,y,p))
            all_events = torch.from_numpy(all_events).float()
            all_events = all_events.to(device)
            data_dict = {'points': all_events}
            coor, features = transform_points_to_voxels(data_dict=data_dict, device=device)
            coor = coor.cpu()
            features = features.cpu()
            coor = coor.numpy()
            features = features.numpy()
            sio.savemat(os.path.join(mat_save , '{}.mat'.format(save_name)), mdict={'coor': coor, 'features': features})