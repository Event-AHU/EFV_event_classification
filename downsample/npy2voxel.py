import os
import pdb
# import csv
import numpy as np
import cv2
import torch
import pandas as pd
from tqdm import tqdm
import scipy.io
from spconv.pytorch.utils import PointToVoxel
from dv import AedatFile
import numpy as np


def transform_points_to_voxels(data_dict={}, voxel_generator=None, device=torch.device("cuda:1")):
    """
    将点云转换为voxel,调用spconv的VoxelGeneratorV2
    """
    
    points = data_dict['points']
    # 将points打乱
    shuffle_idx = np.random.permutation(points.shape[0])
    points = points[shuffle_idx]
    data_dict['points'] = points

    # 使用spconv生成voxel输出
    points = torch.as_tensor(data_dict['points']).to(device)
    #breakpoint()
    voxel_output = voxel_generator(points)

    # 假设一份点云数据是N*4，那么经过pillar生成后会得到三份数据
    # voxels代表了每个生成的voxel数据，维度是[M, 5, 4]
    # coordinates代表了每个生成的voxel所在的zyx轴坐标，维度是[M,3]
    # num_points代表了每个生成的voxel中有多少个有效的点维度是[m,]，因为不满5会被0填充
    voxels, coordinates, num_points = voxel_output
    voxels = voxels.to(device)
    coordinates = coordinates.to(device)
    num_points = num_points.to(device)
    # 选event数量在前5000的voxel  8000 from(4k+,6k+)
    # print(torch.where(num_points>=16)[0].shape)
    if num_points.shape[0] < save_voxel:
        features = voxels[:, :, 3]
        coor = coordinates[:, :]
    else:
        _, voxels_idx = torch.topk(num_points, save_voxel)
        # 将每个voxel的1024个p拼接作为voxel初始特征   16
        features = voxels[voxels_idx][:, :, 3]
        # 前5000个voxel的三维坐标
        coor = coordinates[voxels_idx]
    # 将y.x.t改为t,x,y
    coor[:, [0, 1, 2]] = coor[:, [2, 1, 0]]

    return coor, features


if __name__ == '__main__':
    save_voxel = 1000 # 切到50 
    #save_voxel = 5000
    use_mode = 'frame_exposure_time'
    device = torch.device("cuda:1")
    data_path = r"/media/amax/c08a625b-023d-436f-b33e-9652dc1bc7c0/DATA/dataset/bully10k/bully10k_event"
    #save_path = r"/media/amax/c08a625b-023d-436f-b33e-9652dc1bc7c0/DATA/dataset/Poker_Event/poker_voxel/"
    save_path = r"/media/amax/c08a625b-023d-436f-b33e-9652dc1bc7c0/DATA/dataset/bully10k/bully10k_voxel"
    action_dirs = os.listdir(data_path)
    #action_dirs = os.listdir(data_path)[:2]  # 抽取前两个类
    dvs_img_interval = 1
    voxel_generator = PointToVoxel(
        vsize_xyz=[50, 10, 10],
        coors_range_xyz=[0, 0, 0, 1000, 345, 259],
        num_point_features=4,
        max_num_voxels=16000,
        max_num_points_per_voxel=9, #16
        device=device
    )

    save_interval = 5

    for action_dir in action_dirs:
        action_path = os.path.join(data_path, action_dir)
        video_dirs = os.listdir(action_path)
        #breakpoint()
        for video_dir in video_dirs:
            video_path = os.path.join(action_path, video_dir)
            video_files = os.listdir(video_path)
            
            for videoID in range(len(video_files)):
                foldName = video_files[videoID]
                print("==>> foldName: ", foldName)

                if not os.path.exists(os.path.join(save_path, action_dir, video_dir, foldName+'_voxel')):
                    os.makedirs(os.path.join(save_path, action_dir, video_dir, foldName+'_voxel'))
                else:
                    continue
                mat_save = os.path.join(save_path, action_dir, video_dir, foldName+'_voxel/')

                read_path = os.path.join(action_path, video_dir, foldName)
                # Assume your data is in the form of a Numpy array
                data = np.load(read_path,allow_pickle=True)

                # Convert the events data to PyTorch tensors
                
                for i in range(len(data)):
                    event = data[i]
                    t_all = torch.tensor(event['timestamp']).unsqueeze(0).to(device) 
                    x_all = torch.tensor(event['x']).unsqueeze(0).to(device)
                    y_all = torch.tensor(event['y']).unsqueeze(0).to(device)
                    p_all = torch.tensor(event['polarity']).float().unsqueeze(0).to(device)

                    # Normalize timestamps to be between 0 and 1000
                    t_min = torch.min(t_all).item()
                    t_max = torch.max(t_all).item()
                    t_all = ((t_all - t_min) / (t_max - t_min)) * 1000


                    current_events = torch.cat((t_all, x_all, y_all, p_all), dim=0)
                    current_events = current_events.transpose(0, 1)

                    
                    data_dict = {'points': current_events}
                    #breakpoint()
                    coor, features = transform_points_to_voxels(data_dict=data_dict, voxel_generator=voxel_generator,device=device)

                    coor = coor.cpu().numpy()
                    features = features.cpu().numpy()

                    scipy.io.savemat(mat_save + '{:0>8d}.mat'.format(i), mdict={'coor': coor, 'features': features})

                    
                
