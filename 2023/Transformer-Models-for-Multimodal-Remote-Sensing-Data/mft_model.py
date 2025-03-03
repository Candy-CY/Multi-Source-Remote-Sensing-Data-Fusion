# Multimodal Fusion Transformer (MFT) PyTorch Implementation (modified, original code is from https://github.com/AnkurDeria/MFT)
# This is an updated version of MFT PyTorch code for both serial and distributed training
# Link for the original paper is: https://arxiv.org/abs/2203.16952

# All the changes are commented and are mainly to facilitate hypaer parameters to be passed through the main function
# Added both 'Channel' and 'Pixel' tokenization for the other multimodal data (like the LiDAR data stream) in the same code so that we can call it using the parameter 'LiDAR_token_type' from the main function

# Import all the desired packages

from torch.nn import LayerNorm, Linear, Dropout, Softmax
from einops import rearrange, repeat
import copy
from torchsummary import summary
import math
import time
import torchvision.transforms.functional as TF
from torch.nn.parameter import Parameter
import torch.utils.data as dataf
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch import einsum
import random
import numpy as np
import os
import torch.backends.cudnn as cudnn
cudnn.deterministic = True
cudnn.benchmark = False
cudnn.enabled = False

#random_seed = 42
#random.seed(random_seed)
#torch.manual_seed(random_seed)
#torch.cuda.manual_seed_all(random_seed)

# HetConv layer for the HSI data processing     
class HetConv(nn.Module):
    def __init__(self, in_channels, out_channels, p = 64, g = 64):
        super().__init__()
        # Groupwise Convolution
        self.gwc = nn.Conv2d(in_channels, out_channels, kernel_size=3,groups=g,padding = 1)
        # Pointwise Convolution
        self.pwc = nn.Conv2d(in_channels, out_channels, kernel_size=1,groups=p)
    def forward(self, x):
        return self.gwc(x) + self.pwc(x)   

# Attention Module in the Tramsformer Encoder    
class MCrossAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.1, proj_drop=0.1):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.wq = nn.Linear(head_dim, dim , bias=qkv_bias)
        self.wk = nn.Linear(head_dim, dim , bias=qkv_bias)
        self.wv = nn.Linear(head_dim, dim , bias=qkv_bias)
#         self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim * num_heads, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):

        B, N, C = x.shape
        q = self.wq(x[:, 0:1, ...].reshape(B, 1, self.num_heads, C // self.num_heads)).permute(0, 2, 1, 3)  # B1C -> B1H(C/H) -> BH1(C/H)
        k = self.wk(x.reshape(B, N, self.num_heads, C // self.num_heads)).permute(0, 2, 1, 3)  # BNC -> BNH(C/H) -> BHN(C/H)
        v = self.wv(x.reshape(B, N, self.num_heads, C // self.num_heads)).permute(0, 2, 1, 3)  # BNC -> BNH(C/H) -> BHN(C/H)
        attn = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
#         attn = (q @ k.transpose(-2, -1)) * self.scale  # BH1(C/H) @ BH(C/H)N -> BH1N
        attn = attn.softmax(dim=-1)
#         attn = self.attn_drop(attn)
        x = torch.einsum('bhij,bhjd->bhid', attn, v).transpose(1, 2)
#         x = (attn @ v).transpose(1, 2)
        x = x.reshape(B, 1, C * self.num_heads)   # (BH1N @ BHN(C/H)) -> BH1(C/H) -> B1H(C/H) -> B1C
        x = self.proj(x)
        x = self.proj_drop(x)        
        return x

# MLP Module in the Tramsformer Encoder 
class Mlp(nn.Module):
    def __init__(self, dim, mlp_dim):
        super().__init__()
        self.fc1 = Linear(dim, mlp_dim)
        self.fc2 = Linear(mlp_dim, dim)
        self.act_fn = nn.GELU()
        self.dropout = Dropout(0.1)

        self._init_weights()

    def _init_weights(self):

        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)

        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x

# Single Tramsformer Encoder Block that combines the Attention and Mlp layers     
class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_dim):
        super().__init__()
        self.hidden_size = dim
        self.hidden_dim_size = mlp_dim
        self.attention_norm = LayerNorm(dim, eps=1e-6)
        self.ffn_norm = LayerNorm(dim, eps=1e-6)
        self.ffn = Mlp(dim, mlp_dim)
        self.attn = MCrossAttention(dim, num_heads)
    def forward(self, x):
        h = x
        x = self.attention_norm(x)
        x= self.attn(x)
        x = x + h

        h = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = x + h
        
        return x

# Transformer Encoder Block with repetition    
class TransformerEncoder(nn.Module):

    def __init__(self, dim, num_heads=8, mlp_dim=512, depth=2):
        super().__init__()
        self.layer = nn.ModuleList()
        self.encoder_norm = LayerNorm(dim, eps=1e-6)
        for _ in range(depth):
            layer = Block(dim, num_heads, mlp_dim)
            self.layer.append(copy.deepcopy(layer))       

    def forward(self, x):
        for layer_block in self.layer:
            x= layer_block(x)
            
        encoded = self.encoder_norm(x)

        return encoded[:,0]

# The Final MFT Implementation with cls from other modalities
class MFT(nn.Module):
    def __init__(self, FM, NC, NCLidar, Classes, ntokens, token_type, num_heads, mlp_dim, depth):
        super().__init__()
        #self.HSIOnly = HSIOnly
        self.ntokens = ntokens
        self.FM = FM
        
        self.conv5 = nn.Sequential(
            nn.Conv3d(1, 8, (9, 3, 3), padding=(0,1,1), stride = 1),
            nn.BatchNorm3d(8),
            #nn.GroupNorm(4,8),
            nn.ReLU()
        )
        
        self.conv6 = nn.Sequential(
            HetConv(8 * (NC - 8), FM*4,
                p = 1,
                g = (FM*4)//4 if (8 * (NC - 8))%FM == 0 else (FM*4)//8,
                   ),
            nn.BatchNorm2d(FM*4),
            #nn.GroupNorm(4,FM*4),
            nn.ReLU()
        )
        
        self.lidarConv = nn.Sequential(
                        nn.Conv2d(NCLidar,FM*4,3,1,1),
                        nn.BatchNorm2d(FM*4),
                        nn.GELU()
                        )
        self.ca = TransformerEncoder(FM*4, num_heads, mlp_dim, depth)
        self.out3 = nn.Linear(FM*4 , Classes)
        #self.cls_token = nn.Parameter(torch.zeros(1, 1, FM*4))

        self.position_embeddings = nn.Parameter(torch.randn(1, ntokens + 1, FM*4))
        self.dropout = nn.Dropout(0.1)

        torch.nn.init.xavier_uniform_(self.out3.weight)

        torch.nn.init.normal_(self.out3.bias, std=1e-6)
        self.token_wA = nn.Parameter(torch.empty(1, ntokens, FM*4),
                                     requires_grad=True)  # Tokenization parameters
                             
        torch.nn.init.xavier_normal_(self.token_wA)
        self.token_wV = nn.Parameter(torch.empty(1, FM*4, FM*4),
                                     requires_grad=True)  # Tokenization parameters
        torch.nn.init.xavier_normal_(self.token_wV)
        
        if token_type == "pixel":
            
            self.token_wA_L = nn.Parameter(torch.empty(1, 1, 1),
                                     requires_grad=True)  # Tokenization parameters
                                 
            torch.nn.init.xavier_normal_(self.token_wA_L)
            self.token_wV_L = nn.Parameter(torch.empty(1, 1, FM*4),
                                     requires_grad=True)  # Tokenization parameters
                                 
            torch.nn.init.xavier_normal_(self.token_wV_L)
        
        elif token_type == "channel":
            
            self.token_wA_L = nn.Parameter(torch.empty(1, 1, FM*4),
                                     requires_grad=True)  # Tokenization parameters

            torch.nn.init.xavier_normal_(self.token_wA_L)
            self.token_wV_L = nn.Parameter(torch.empty(1, FM*4, FM*4),
                                     requires_grad=True)  # Tokenization parameters

            torch.nn.init.xavier_normal_(self.token_wV_L)
        
        else:
            raise ValueError(
                print("unknown Lidar_token_type {token_type}, acceptable pixel, channel")
            )                                
        
    def forward(self, x1, x2):
        x1 = x1.reshape(x1.shape[0],-1,11,11)
        x1 = x1.unsqueeze(1)
        x1 = self.conv5(x1)
        x1 = x1.reshape(x1.shape[0],-1,11,11)        
        x1 = self.conv6(x1)
        
        x1 = x1.flatten(2)        
        x1 = x1.transpose(-1, -2)
        wa = self.token_wA.expand(x1.shape[0],-1,-1)
        wa = rearrange(wa, 'b h w -> b w h')  # Transpose
        A = torch.einsum('bij,bjk->bik', x1, wa)
        A = rearrange(A, 'b h w -> b w h')  # Transpose
        A = A.softmax(dim=-1)
        wv = self.token_wV.expand(x1.shape[0],-1,-1)
        VV = torch.einsum('bij,bjk->bik', x1, wv)
        T = torch.einsum('bij,bjk->bik', A, VV)
        
        x2 = x2.reshape(x2.shape[0],-1,11,11)        
        x2 = self.lidarConv(x2)
        x2 = x2.reshape(x2.shape[0],-1,11**2)
        x2 = x2.transpose(-1, -2)   
        
        wa_L = self.token_wA_L.expand(x2.shape[0],-1,-1)
        wa_L = rearrange(wa_L, 'b h w -> b w h')  # Transpose
        A_L = torch.einsum('bij,bjk->bik', x2, wa_L)
        A_L = rearrange(A_L, 'b h w -> b w h')  # Transpose
        A_L = A_L.softmax(dim=-1)
        wv_L = self.token_wV_L.expand(x2.shape[0],-1,-1)
        VV_L = torch.einsum('bij,bjk->bik', x2, wv_L)
        L = torch.einsum('bij,bjk->bik', A_L, VV_L)
        
        x = torch.cat((L, T), dim = 1) #[b,n+1,dim]
        x = x + self.position_embeddings
        x = self.dropout(x)
        x = self.ca(x)
        x = x.reshape(x.shape[0],-1)
        out3 = self.out3(x)
        return out3
  
# The Final MFT Implementation with cls from random
class Transformer(nn.Module):
    def __init__(self, FM, NC, Classes, ntokens, num_heads, mlp_dim, depth):
        super().__init__()
        #self.HSIOnly = HSIOnly
        self.ntokens = ntokens
        self.FM = FM
        
        self.conv5 = nn.Sequential(
            nn.Conv3d(1, 8, (9, 3, 3), padding=(0,1,1), stride = 1),
            #nn.BatchNorm3d(8),
            nn.GroupNorm(4,8),
            nn.ReLU()
        )
        
        self.conv6 = nn.Sequential(
            HetConv(8 * (NC - 8), FM*4,
                p = 1,
                g = (FM*4)//4 if (8 * (NC - 8))%FM == 0 else (FM*4)//8,
                   ),
            #nn.BatchNorm2d(FM*4),
            nn.GroupNorm(4,FM*4),
            nn.ReLU()
        )
        
        self.last_BandSize = NC//2//2//2
        
        self.ca = TransformerEncoder(FM*4, num_heads, mlp_dim, depth)
        self.out3 = nn.Linear(FM*4 , Classes)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, FM*4))


        self.position_embeddings = nn.Parameter(torch.randn(1, ntokens + 1, FM*4))
        self.dropout = nn.Dropout(0.1)

        torch.nn.init.xavier_uniform_(self.out3.weight)

        torch.nn.init.normal_(self.out3.bias, std=1e-6)
        self.token_wA = nn.Parameter(torch.empty(1, ntokens, FM*4),
                                     requires_grad=True)  # Tokenization parameters
                                  
        torch.nn.init.xavier_normal_(self.token_wA)
        self.token_wV = nn.Parameter(torch.empty(1, FM*4, FM*4),
                                     requires_grad=True)  # Tokenization parameters

        torch.nn.init.xavier_normal_(self.token_wV)
        
        
    def forward(self, x1):
        x1 = x1.reshape(x1.shape[0],-1,11,11)
        x1 = x1.unsqueeze(1)
        x1 = self.conv5(x1)
        x1 = x1.reshape(x1.shape[0],-1,11,11)        
        x1 = self.conv6(x1)
        
        x1 = x1.flatten(2)        
        x1 = x1.transpose(-1, -2)
        wa = self.token_wA.expand(x1.shape[0],-1,-1)
        wa = rearrange(wa, 'b h w -> b w h')  # Transpose
        A = torch.einsum('bij,bjk->bik', x1, wa)
        A = rearrange(A, 'b h w -> b w h')  # Transpose
        A = A.softmax(dim=-1)
        wv = self.token_wV.expand(x1.shape[0],-1,-1)
        VV = torch.einsum('bij,bjk->bik', x1, wv)
        T = torch.einsum('bij,bjk->bik', A, VV)
        
        L = self.cls_token.repeat(x1.shape[0], 1, 1)
        
        x = torch.cat((L, T), dim = 1) #[b,n+1,dim]
        embeddings = x + self.position_embeddings
        embeddings = self.dropout(embeddings)
        x = self.ca(embeddings)
        x = x.reshape(x.shape[0],-1)
        out3 = self.out3(x)
        return out3
  
