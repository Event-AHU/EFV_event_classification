# Official PyTorch Implementation of EFV++

**Retain, Blend, and Exchange: A Quality-aware Spatial-Stereo Fusion Approach for Event Stream Recognition**, 
Lan Chen, Dong Li, Xiao Wang*, Pengpeng Shao, Wei Zhang, Yaowei Wang, Yonghong Tian, Jin Tang, 
**IEEE Transactions on Multimedia (TMM) 2025**, arXiv:2406.18845
[[arXiv](https://arxiv.org/abs/2406.18845)]


## :dart: Abstract 
Existing event-based pattern recognition models usually represent the event stream as the point cloud, voxel, image, etc, and design various deep neural networks to learn their features. Although considerable results can be achieved in simple cases, however, the model performance may be limited by monotonous modal expressions, sub-optimal fusion, and readout mechanisms. 

In this paper, we propose a novel dual-stream framework for event stream-based pattern recognition via differentiated fusion, termed EFV++. It models two common event representations simultaneously, i.e., event images and event voxels. The spatial and three-dimensional stereo information can be learned separately by utilizing Transformer and Structured Graph Neural Network (GNN). We believe the features of each representation still contain both efficient and redundant features and the sub-optimal solution may be obtained if we directly fuse them without differentiation. Thus, we divide each feature into three levels and retain high-quality features, blend medium-quality features, and exchange low-quality features. The enhanced dual features will be fed into the fusion Transformer together with bottleneck features. In addition, we also introduce a novel GRU-based readout layer to enhance the diversity of features as final representations. Extensive experiments demonstrate that our proposed framework achieves state-of-the-art performance on multiple widely used or newly proposed event-based classification datasets. Specifically, we achieve new state-of-the-art performance on the Bullying10k dataset, i.e., 90.51%, which exceeds the second place by +2.21%. 


<p align="center">
  <img width="90%" src="https://github.com/Event-AHU/EFV_event_classification/blob/EFVpp/figure/firstIMG.jpg" alt="feature_vis"/>
</p> 


## :construction_worker: Environment Setting 
```   
Python 3.8
Pytorch 
numpy
scipy
Pytorch Geometric
torch-cluster 1.5.9
torch-geometric 1.7.0
torch-scatter 2.0.6
torch-sparse 0.6.9
torch-spline-conv 1.2.1
spconv
pandas
pillow
Matlab
```

## :triangular_flag_on_post: Framework 
<p align="center">
  <img width="90%" src="https://github.com/Event-AHU/EFV_event_classification/blob/EFVpp/figure/RBE_framework.png" alt="feature_vis"/>
</p> 

## :floppy_disk: Dataset Download and Pre-processing 

### Bullying10k

```
 Bullying10k:https:https://www.brain-cog.network/dataset/Bullying10k/
```

### HARDVS
```
Event Images：https://pan.baidu.com/s/1OhlhOBHY91W2SwE6oWjDwA?pwd=1234 提取码：1234

Compact Event file：https://pan.baidu.com/s/1iw214Aj5ugN-arhuxjmfOw?pwd=1234 提取码：1234
```

### POKER
```
BaiduYun (178GB): https://pan.baidu.com/s/1vQnHZUqQ1o58SajvtE-uHw?pwd=AHUE 提取码：AHUE 

DropBox (178GB): https://www.dropbox.com/scl/fo/w658kwhfi3qa8naul3eeb/h?rlkey=zjn4b69wa1e3mhid8p6hh8v75&dl=0
```

### ASL-DVS

```
 ASL-DVS:https: //www.dropbox.com/sh/ibq0jsicatn7l6r/AACNrNELV56rs1YInMWUs9CAa?dl=0
```

### N-MNIST

```
 N-MNIST: https://www.garrickorchard.com/datasets/n-mnist
```

## :hourglass: Training and Testing 

```
    train
    CUDA_VISIBLE_DEVICES=3 python -m torch.distributed.launch --nproc_per_node=1 --master_port=30000 train.py poker

    test
    CUDA_VISIBLE_DEVICES=3 python -m torch.distributed.launch --nproc_per_node=1 --master_port=30000 eval.py poker
```

## :chart_with_upwards_trend: Experimental Results 
<p align="center">
  <img width="90%" src="https://github.com/Event-AHU/EFV_event_classification/blob/EFVpp/figure/top5.png" alt="feature_vis"/>
</p> 

## :newspaper:License

Code is released for non-commercial and research purposes **only**. For commercial purposes, please contact the authors.

## :icecream:Acknowledgement

Our code is implemented based on 
<a href="https://github.com/Event-AHU/EFV_event_classification/tree/main">EFV_event_classification</a>, 

## :pencil2: Citation 

If you think this work contributes to your research, please give us a star :star2: and cite this paper via: 
```
@misc{chen2024retainblendexchangequalityaware,
      title={Retain, Blend, and Exchange: A Quality-aware Spatial-Stereo Fusion Approach for Event Stream Recognition}, 
      author={Lan Chen and Dong Li and Xiao Wang and Pengpeng Shao and Wei Zhang and Yaowei Wang and Yonghong Tian and Jin Tang},
      year={2024},
      eprint={2406.18845},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2406.18845}, 
}
```

If you have any questions about this work, please leave an issue. 

