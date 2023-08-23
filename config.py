import os

class Config():
    curPath = os.path.abspath(__file__)
    rootPath = os.path.split(curPath)[0]

    ASL_root_dir = os.path.join(rootPath, 'data/ASL_Img_Voxel')
    Nmnist_root_dir = os.path.join(rootPath, 'data/Nmnist_img_voxel')
    Ncal_root_dir = os.path.join(rootPath, 'data/Ncaltech')
    model_dir = os.path.join(rootPath, 'trained_model')
