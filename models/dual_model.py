# -- coding: utf-8 --**
# GCN model

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.transforms as T
from itertools import permutations
from torch_geometric.nn import GMMConv,global_mean_pool
import os
import sys
import pdb
from models import torchvision_resnet
from models import crossreplacetransformer
from models.crossreplacetransformer import CrossReplaceTransformerLayer,CrossReplaceTransformer
from timm.models.layers import PatchEmbed, Mlp, DropPath, trunc_normal_, lecun_normal_


import cv2
import numpy as np
import torch
from tqdm import tqdm
from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, EigenGradCAM, LayerCAM, FullGrad
from pytorch_grad_cam import GuidedBackpropReLUModel
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image


curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

class CustomGRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout_rate):
        super(CustomGRU, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)  
        out, _ = self.gru(x, h0)
        out = self.dropout(out)
        return out

class Attention(nn.Module):
    # fused_attn: Final[bool]

    def __init__(
            self,
            dim,
            num_heads=6,
            qkv_bias=False,
            qk_norm=False,
            attn_drop=0.,
            proj_drop=0.,
            norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.fused_attn = False

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        if self.fused_attn:
            x = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.attn_drop.p,
            )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class ResidualBlock(torch.nn.Module):
    def __init__(self, in_channel, out_channel):
        super(ResidualBlock, self).__init__()
        self.left_conv1 = GMMConv(in_channel, out_channel, dim=3, kernel_size=5)
        self.left_bn1 = torch.nn.SyncBatchNorm(out_channel)
        self.left_conv2 = GMMConv(out_channel, out_channel, dim=3, kernel_size=5)
        self.left_bn2 = torch.nn.SyncBatchNorm(out_channel)

        self.shortcut_conv = GMMConv(in_channel, out_channel, dim=3, kernel_size=1)
        self.shortcut_bn = torch.nn.SyncBatchNorm(out_channel)

    def forward(self, data):
        data.x = F.elu(self.left_bn2(
            self.left_conv2(F.elu(self.left_bn1(self.left_conv1(data.x, data.edge_index, data.edge_attr))),
                            data.edge_index, data.edge_attr)) + self.shortcut_bn(
            self.shortcut_conv(data.x, data.edge_index, data.edge_attr)))

        return data

class PatchEmbed(nn.Module):
    """ 2D Image to Patch Embedding
    """
    def __init__(self, img_size=8, patch_size=4, in_chans=512, embed_dim=768, norm_layer=None, flatten=True):
        super().__init__()
        img_size = (img_size,img_size)
        patch_size = (patch_size,patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0], f"Input image height ({H}) doesn't match model ({self.img_size[0]})."
        assert W == self.img_size[1], f"Input image width ({W}) doesn't match model ({self.img_size[1]})."
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        return x
class LayerScale(nn.Module):
    def __init__(self, dim, init_values=1e-5, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        return x.mul_(self.gamma) if self.inplace else x * self.gamma

class ST_Transformer_block(torch.nn.Module):
    def __init__(self,dim=768,num_heads=4,mlp_ratio=4,attn_drop=0.,drop=0.,drop_path=0.,
                 qkv_bias=False,init_values=None):
        super(ST_Transformer_block, self).__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.mlp1 = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=nn.GELU, drop=drop)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()

    def forward(self,input):
        temporal_form = input + self.drop_path1(self.ls1(self.attn(self.norm1(input))))
        temporal_form = self.norm2(temporal_form)
        temporal_form_out = temporal_form + self.drop_path2(self.ls2(self.mlp1(temporal_form)))#torch.Size([4, 9, 512])

        return temporal_form_out
    
class Net(torch.nn.Module):
    def __init__(self,pretrained=None, batchNorm=True, output_layers=None, init_std=0.05,
                dim=24, clip_len=8, num_heads=6, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0., init_values=None, drop_path=0.,embed_dim=768):
        super(Net, self).__init__()
        self.patch_emd = PatchEmbed(img_size=8, patch_size=4, in_chans=512, embed_dim=embed_dim, norm_layer=None, flatten=True)
        self.drop = drop
        drop_rate = 0.
        self.ST_Transformer_block_1 = ST_Transformer_block(
            dim=dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            attn_drop=attn_drop,
            drop=drop,
            qkv_bias=qkv_bias,
            init_values=init_values,
            drop_path=drop_path)
        self.ST_Transformer_block_2 = ST_Transformer_block(
            dim=dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            attn_drop=attn_drop,
            drop=drop,
            qkv_bias=qkv_bias,
            init_values=init_values,
            drop_path=drop_path)
        self.ST_Transformer_block_3 = ST_Transformer_block(
            dim=dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            attn_drop=attn_drop,
            drop=drop,
            qkv_bias=qkv_bias,
            init_values=init_values,
            drop_path=drop_path)
        self.ST_Transformer_block_4 = ST_Transformer_block(
            dim=dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            attn_drop=attn_drop,
            drop=drop,
            qkv_bias=qkv_bias,
            init_values=init_values,
            drop_path=drop_path)
        self.ST_Transformer_block_5 = ST_Transformer_block(
            dim=dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            attn_drop=attn_drop,
            drop=drop,
            qkv_bias=qkv_bias,
            init_values=init_values,
            drop_path=drop_path)
        
        self.clip_len = clip_len
        self.batchNorm = batchNorm

        self.resnet18_feature_extractor = torchvision_resnet.resnet18(pretrained = True)

        self.base_layer = CrossReplaceTransformerLayer(d_model=768, nhead=8, theta=(0.025,0.031))

        # 创建 CrossReplaceTransformer
        num_layers = 2  
        replace_start = 1  # 从第2层开始进行替换操作
        replace_end = 2 # 在第3层结束替换操作
        self.cross_replace_transformer = CrossReplaceTransformer(self.base_layer, num_layers, replace_start, replace_end)


        self.pos_drop = nn.Dropout(p=drop_rate)

        self.pos_embed = nn.Parameter(torch.zeros([1, self.clip_len*4, embed_dim],requires_grad=True))
        self.cls_token = nn.Parameter(torch.randn([1, 1, embed_dim], requires_grad=True))
        #voxel#
        self.conv_0 = GMMConv(16, 64, dim=3, kernel_size=5)
        self.bn_0 = torch.nn.SyncBatchNorm(64)


        self.block1 = ResidualBlock(64,128)
        self.block2 = ResidualBlock(128,256)

        self.block3 = ResidualBlock(256,512)
        #self.block4 = ResidualBlock(400,512)
        self.bottle_neck = nn.Parameter(torch.randn([1, 32, 768], requires_grad=True))
        self.fc0 = torch.nn.Linear(768,768)
        self.bn0 = torch.nn.SyncBatchNorm(768)
        self.fc1 = torch.nn.Linear(768,512)
        self.bn = torch.nn.SyncBatchNorm(512)
        self.drop_out = torch.nn.Dropout()
        self.fc2 = torch.nn.Linear(512, 114)
        self.embed_adjust = nn.Linear(512, 768)
        self.gru = CustomGRU(input_size=768, hidden_size=768, num_layers=1, dropout_rate=0.2)
    def forward(self,data_voxel,data_frame):
                
        ##################################################################################
        data_frame = data_frame.permute(0,2,1,3,4)
        B,C,N,H,W=data_frame.shape # B 3 8 240 240)
        res_img_F = F.interpolate(data_frame, size = [self.clip_len, 240, 240], mode='trilinear') #([b, 3,8, 240, 240])

        res_img = res_img_F.permute(0,2,1,3,4).reshape(self.clip_len*B,C,240,240)   #([b*t, 3, h, w]) 5 D -> 4 D
        img_with_feature = self.resnet18_feature_extractor(res_img) #[b*n,512,8,8]
        ###############get feature map by resnet#########################################
        patched_data = self.patch_emd(img_with_feature)   # [b*n N C][64,4,768]
        ###############get pathed token by path embeding##################################
        input_former_data_ori = patched_data.reshape(B,-1,patched_data.shape[2]) #[b,N*n,C][8,32,768]
 
        ############### Former Block#####################################################
        input_former_data = self.pos_drop(input_former_data_ori + self.pos_embed) #

        former_b3_out = self.ST_Transformer_block_1(input_former_data)
               
        ################# GCN Block ######################################################
        #breakpoint()
        data_voxel.x = F.elu(self.bn_0(self.conv_0(data_voxel.x, data_voxel.edge_index, data_voxel.edge_attr)))
        #breakpoint()
        data_voxel = self.block1(data_voxel)
        data_voxel = self.block2(data_voxel) 
        data_voxel = self.block3(data_voxel)
        # data_voxel = self.block4(data_voxel)

        #################################交换模块##################################################
        # 对于 data_frame
        src1 = former_b3_out  
        src1 = src1.transpose(0, 1)  
        # 对于 data_voxel
        graph_sizes = data_voxel.ptr[1:] - data_voxel.ptr[:-1]  
        graphs = torch.split(data_voxel.x, graph_sizes.tolist())  
        # 对每个图的节点特征进行处理
        src2_list = []
        for graph in graphs:
            graph_mean = graph.mean(dim=0)  
            src2_list.append(graph_mean)
                
        src2 = torch.stack(src2_list, dim=0)
        src2 = self.embed_adjust(src2)
        src2 = src2.unsqueeze(1) 
        src2 = src2.expand(-1, 32, -1)  
        src2 = src2.transpose(0, 1)
        
        x1, x2 = self.cross_replace_transformer(src1, src2) 
        
        # 将 former_b3_out 换回为 x1
        x1 = x1.transpose(0, 1)#torch.Size([2, 32, 768])
        #######################bottle_nc###########################################
        
        bottle_nc = self.bottle_neck.expand_as(x1)#torch.Size([2, 32, 768])
        fusion_input = torch.cat((x1, bottle_nc), 1)
        bn_fusion_out = self.ST_Transformer_block_4(fusion_input) #torch.Size([2, 64, 768])
        # 将 data_voxel.x 换回为 x2
        x2 = x2.transpose(0, 1)
        x2 = torch.mean(x2, dim=1) #[2,768] 
        x_input = self.bn0(x2)  #[2,768]
        
        ################ get mean vector of GCN Block's output##############################
        x_input = torch.unsqueeze(x_input,1)
        bn_fused = bn_fusion_out[:,32:,:] #torch.Size([2, 32, 768])
        bn_input_gcn = torch.cat((bn_fused,x_input),1) #torch.Size([2, 33, 768])
        
        bn_fusion_out_2 = self.ST_Transformer_block_5(bn_input_gcn) #torch.Size([2, 33, 768])
        former_fused = bn_fusion_out[:,:32,:] #torch.Size([2, 32, 768])
        gcn_fused = bn_fusion_out_2[:,:32,:] #torch.Size([2, 32, 768])
        x_fused = torch.cat((gcn_fused,former_fused),1) #torch.Size([2, 64, 768])

        ###############################   GRU   ############################################# 
       
        x_fused_split = [bn_fusion_out[:,:32,:], bn_fusion_out[:,32:,:], gcn_fused] 
        
        perms = list(permutations(x_fused_split, 3))[:5]
        gru_inputs = torch.stack([torch.cat(perm, dim=1) for perm in perms])
       
        gru_inputs = gru_inputs.view(-1, 96, 768) #torch.Size([5, 8, 96, 768])
        gru_output = self.gru(gru_inputs)
        gru_output = gru_output.view(5, -1, 96, 768)
        gru_output_mean = gru_output.mean(dim=0)
        x_fused = gru_output_mean

        x = torch.mean(x_fused,1) # [b,768]
        ################cat vector of dual branch and classification ##################################################
        x = self.fc1(x)
        x = F.elu(x)
        x = self.bn(x)
        x = self.drop_out(x)
        x = self.fc2(x)
        
        return F.log_softmax(x, dim=1)
 
