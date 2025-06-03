import os 
import pdb 
import csv 
import numpy as np 
import cv2
import torch
import pandas as pd
from tqdm import tqdm
from PIL import Image
from numpy.lib import recfunctions

data_path = r'/media/amax/c08a625b-023d-436f-b33e-9652dc1bc7c0/DATA/dataset/bully10k/bully10k_event/fingerguess/fingerguess_bf_hzq_L_light'
save_path = r"/media/amax/c08a625b-023d-436f-b33e-9652dc1bc7c0/DATA/dataset/bully10k/bully10k_imgblue"
#action_004_20220219_101957781_EI_70M`
if __name__=='__main__':
	cls_dirs = os.listdir(data_path)
	for cls_ID in range(len(cls_dirs)):
		cls = cls_dirs[cls_ID]
		fileLIST = os.listdir(os.path.join(data_path))
		save_cls_path = os.path.join(save_path)
		if not os.path.exists(save_cls_path):
			os.makedirs(save_cls_path)
		for FileID in tqdm(range(len(fileLIST))):
			csv_Name = fileLIST[FileID]
			video_save_path = os.path.join(save_cls_path,csv_Name.split('.')[0])
			if os.path.exists(video_save_path):
				continue
			if not os.path.exists(video_save_path):
				os.makedirs(video_save_path)
			read_path = os.path.join(data_path,csv_Name)
			recordMODE = "EI"
			dt = np.load(read_path, allow_pickle=True)
			#breakpoint()
			dt = np.concatenate(dt)
			breakpoint()
			dt_modified = np.asarray([tuple(map(int, x)) for x in dt], dtype=np.int32)
			dt_modified = dt_modified[:, :4]
			dt = torch.tensor(dt_modified, dtype=torch.int)
			t, x, y, p = torch.chunk(dt, 4, dim=1)
			all_events = torch.cat((x, y, p, t), dim=1)
			all_events = all_events.numpy()
			frameRATE = 100
			height,width = 260,346
			finalTIME_stamp = int(all_events[all_events.shape[0]-1][3])
			time_length = all_events[-1,3] - all_events[0,3]
			eventMODnum = 50000
			frameNUM = int(5000000//33333)
			start_idx = []
			deltaT = int(time_length/10)
			i = 1
			for j in range(len(all_events)):
				if all_events[j][-1]-all_events[0][-1] > deltaT * i:
					start_idx.append(j)
					i += 1
			start_time_stamp = 0
			saved_event_timeStamp = []
			count_csvsample = 0
			count_IMG = 0
			assert len(start_idx)!=0,'{} get 0 img!'.format(csv_Name)
			for imgID in range(len(start_idx)-1):
				event_frame = 255 * np.ones((height, width, 3), dtype=np.uint8)
				start_time_stamp = start_idx[imgID]
				end_time_stamp = start_idx[imgID+1]
				saved_event_timeStamp.append([start_time_stamp, end_time_stamp])
				if recordMODE == "EI":
					event = all_events[start_time_stamp:end_time_stamp]
					on_idx = np.where(event[:, 2] == 1)
					off_idx = np.where(event[:, 2] == 0)
					event_frame[height - 1 - event[:, 1][on_idx],  event[:, 0][on_idx], :] = [30, 30, 220]
					event_frame[height - 1 - event[:, 1][off_idx], event[:, 0][off_idx], :] = [200, 30, 30]
					event_frame=cv2.flip(event_frame,0)  ##垂直翻转
					cv2.imwrite(os.path.join(video_save_path, '{:04d}'.format(count_IMG)+'.png'), event_frame)
				
				count_IMG += 1
