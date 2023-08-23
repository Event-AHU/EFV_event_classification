# EFV_event_classification
[**PRCV-2023**] **Learning Bottleneck Transformer for Event Image-Voxel Feature Fusion based Classification**, Chengguo Yuan, Yu Jin, Zongzhen Wu, Fanting Wei, Yangzirui Wang, Lan Chen, and Xiao Wang. [[Paper]()] 


## Abstract 
```
Recognizing target objects using an event-based camera draws more and more attention in recent years. Existing works usually represent the event streams into point-cloud, voxel, image, etc, and learn the feature representations using various deep neural networks. Their final results may be limited by the following factors: monotonous modal expressions and the design of the network structure. To address the aforementioned challenges, this paper proposes a novel dual-stream framework for event representation, extraction, and fusion. This framework simultaneously models two common representations: event images and event voxels. By utilizing Transformer and Structured Graph Neural Network (GNN) architectures, spatial information and three-dimensional stereo information can be learned separately. Additionally, a bottleneck Transformer is introduced to facilitate the fusion of the dual-stream information. Extensive experiments demonstrate that our proposed framework achieves state-of-the-art performance on two widely used event-based classification datasets.
```

<p align="center">
  <img width="90%" src="https://github.com/Event-AHU/EFV_event_classification/blob/main/figure/firstIMG.jpg" alt="feature_vis"/>
</p> 


## Environment Setting 
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

## Framework 
<p align="center">
  <img width="90%" src="https://github.com/Event-AHU/EFV_event_classification/blob/main/figure/frameworkV4.jpg" alt="feature_vis"/>
</p> 

## Dataset Download and Pre-processing 
```
    ASL-DVS:https: //www.dropbox.com/sh/ibq0jsicatn7l6r/AACNrNELV56rs1YInMWUs9CAa?dl=0
    N-MNIST: https://www.garrickorchard.com/datasets/n-mnist
```
    
### event to voxel
```
    python downsample/to_voxel.py
```
    
### voxel to graph
```
    python generate_graph/voxel2graph.py
```
    
## Training and Testing 
    * **train**
    ```
    CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --master_port 1568 --nproc_per_node=1 train.py --epoch 150 --batch_size 8
    ```
    * **test**
    ```
    CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 test.py --model_name xxx.pkl --batch_size 8
    ```
    
## Experimental Results 
<p align="center">
  <img width="90%" src="https://github.com/Event-AHU/EFV_event_classification/blob/main/figure/ASLtop5TSNE.jpg" alt="feature_vis"/>
</p> 

## Citation 
If you think this work contributes to your research, please cite this paper via: 
```

```










