import os

class Config():
    curPath = os.path.abspath(__file__)
    rootPath = os.path.split(curPath)[0]

    ASL_root_dir = os.path.join(rootPath, 'data/ASL_Img_Voxel')
    #Nmnist_root_dir = os.path.join(rootPath, 'data/Nmnist_img_voxel')
    Nmnist_root_dir = os.path.join(rootPath, 'data/Nmnist_img_voxel')
    Ncal_root_dir = os.path.join(rootPath, 'data/Ncaltech')
    poker_root_dir = os.path.join('/media/amax/c08a625b-023d-436f-b33e-9652dc1bc7c0/DATA/dataset/Poker_Event')
    hardvs_root_dir = os.path.join('/media/amax/c08a625b-023d-436f-b33e-9652dc1bc7c0/DATA/dataset/HARDVS')
    billy_root_dir = os.path.join('/media/amax/c08a625b-023d-436f-b33e-9652dc1bc7c0/DATA/dataset/bully10k')
    log_dir = os.path.join(rootPath, 'log_dir')
    graph_train_log_path = os.path.join(log_dir, 'train_hardvs++1.log')
    model_dir = os.path.join(rootPath, 'trained_model')
