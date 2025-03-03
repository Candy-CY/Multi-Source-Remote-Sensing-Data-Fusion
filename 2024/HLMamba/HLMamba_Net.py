import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm import Mamba
from torch.nn import Linear

def img2seq(x):
    [b, c, h, w] = x.shape
    x = x.reshape((b, c, h*w))
    return x

def seq2img(x):
    [b, c, d] = x.shape
    p = int(d ** .5)
    x = x.reshape((b, c, p, p))
    return x

class CNN_Encoder(nn.Module):
    def __init__(self, l1, l2):
        super(CNN_Encoder, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(l1, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),  # No effect on order
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(l2, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),  # No effect on order
        )
        self.conv1_1 = nn.Sequential(
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),  # No effect on order
        )
        self.conv2_1 = nn.Sequential(
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),  # No effect on order
        )
        self.conv1_2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),  # No effect on order
        )
        self.conv2_2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),  # No effect on order
        )
        self.conv1_3 = nn.Sequential(
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),  # No effect on order
        )
        self.conv2_3 = nn.Sequential(
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),  # No effect on order
        )
        self.xishu1 = torch.nn.Parameter(torch.Tensor([0.5]))  # lamda
        self.xishu2 = torch.nn.Parameter(torch.Tensor([0.5]))  # 1 - lamda
        self.xishu3 = torch.nn.Parameter(torch.Tensor([0.5]))  # 1 - lamda

        self.convg1 = nn.Sequential(
            nn.Conv2d(l2, 32, 1, 1, 0),
            nn.ReLU(),  # No effect on order
        )
        self.convg2 = nn.Sequential(
            nn.Conv2d(32, 64, 1, 1, 0),
            nn.ReLU(),  # No effect on order
        )

    def forward(self, x11, x21,x31, x12, x22,x32, x13, x23,x33):
        
        x11 = self.conv1(x11)
        x21 = self.conv2(x21) 
        x31 = self.convg1(x31)

        x12 = self.conv1(x12)
        x22 = self.conv2(x22)
        x32 = self.convg1(x32)

        x13 = self.conv1(x13)
        x23 = self.conv2(x23)
        x33 = self.convg1(x33)

        x1_1 = self.conv1_1(x11)
        x2_1 = self.conv2_1(x21)
        x3_1 = self.convg2(x31)

        x2_1 = x3_1 * self.xishu1 + x2_1 * self.xishu2
        x_add1 = x1_1 * self.xishu1 + x2_1 * self.xishu2

        x1_2 = self.conv1_2(x12)
        x2_2 = self.conv2_2(x22)
        x3_2 = self.convg2(x32)
        x2_2 = x3_2 * self.xishu1 + x2_2 * self.xishu2
        x_add2 = x1_2 * self.xishu1 + x2_2 * self.xishu2

        x1_3 = self.conv1_3(x13)
        x2_3 = self.conv2_3(x23)
        x3_3 = self.convg2(x33)
        x2_3 = x2_3 * self.xishu1 + x3_3 * self.xishu2
        x_add3 = x1_3 * self.xishu1 + x2_3 * self.xishu2
        return x_add1, x_add2, x_add3

class CNN_Classifier(nn.Module):
    def __init__(self, Classes):
        super(CNN_Classifier, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(64, 32, 1),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, Classes, 1),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.FC = nn.Sequential(
            Linear(32, Classes),
        )

    def forward(self, x):   # [64, 66, 8, 8]
        x = self.conv1(x)
        x = self.pool(x).view(x.size(0), -1)
        x = self.FC(x)
        x_out = F.softmax(x, dim=1)

        return x_out

class FuseMamba(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_head, dropout, num_channel):
        super().__init__()

        self.xishu1 = torch.nn.Parameter(torch.Tensor([0.5]))  # lamda
        self.xishu2 = torch.nn.Parameter(torch.Tensor([0.5]))  # lamda
        self.xishu3 = torch.nn.Parameter(torch.Tensor([0.5]))  # lamda

        self.mamba = Mamba(dim)
        
        self.conv1 = nn.Conv2d(64,64,1)
        self.bn = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(64)
        self.conv2 = nn.Conv2d(64,64,1)
        self.sigmoid = nn.Sigmoid()


    def forward(self, x1,x2,x3, mask=None):


        b,c,h = x3.shape

        x3 = self.conv1(x3.reshape(b,c,8,8)).reshape(b,c,h)
        x2 = self.conv1(x2.reshape(b,c,8,8)).reshape(b,c,h)
        x1 = self.conv1(x1.reshape(b,c,8,8)).reshape(b,c,h)

        x2 = x1 * x2
        x3 = x1 * x3

        # # x3 = self.Transformer1(x3)
        # # x3 = self.Transformer2(x3)
 
        x3 = self.mamba(x3) + x3
        x3 = self.relu(self.bn(self.conv1(x3.reshape(b,c,8,8)))).reshape(b,c,h)

        # # x2 = self.Transformer1(x2)
        # # x2 = self.Transformer2(x2)        
        x2 = self.mamba(x2) + x2
        x2 = self.relu(self.bn(self.conv1(x2.reshape(b,c,8,8)))).reshape(b,c,h)

        # # x1 = self.Transformer1(x1)
        # # x1 = self.Transformer2(x1)
        x1 = self.mamba(x1) + x1
        x1 = self.relu(self.bn(self.conv1(x1.reshape(b,c,8,8)))).reshape(b,c,h)
        x = x1 + self.xishu1 * x2 + self.xishu2 * x3

        return x

class HLMamba(nn.Module):
    def __init__(self, l1, l2, patch_size, num_patches, num_classes, encoder_embed_dim, decoder_embed_dim, en_depth, en_heads,
                 de_depth, de_heads, mlp_dim, dim_head=16, dropout=0., emb_dropout=0.):
        super().__init__()
        self.cnn_encoder = CNN_Encoder(l1, l2)
        self.cnn_classifier = CNN_Classifier(num_classes)
        self.coefficient1 = torch.nn.Parameter(torch.Tensor([0.5]))
        self.coefficient2 = torch.nn.Parameter(torch.Tensor([0.5]))
        self.loss_fun2 = nn.MSELoss()
        self.num_patches = num_patches
        self.patch_size = patch_size
        self.encoder_embed_dim = encoder_embed_dim
        self.encoder_pos_embed = nn.Parameter(torch.randn(1, self.patch_size ** 2 + 1 + 2, encoder_embed_dim))
        self.decoder_pos_embed = nn.Parameter(torch.randn(1, self.patch_size ** 2 + 1 + 2, decoder_embed_dim))
        self.encoder_embedding1 = nn.Linear(((patch_size) * 1) ** 2, self.patch_size ** 2)
        self.encoder_embedding2 = nn.Linear(((patch_size) * 2) ** 2, self.patch_size ** 2)
        self.encoder_embedding3 = nn.Linear(((patch_size) * 3) ** 2, self.patch_size ** 2)
        self.cls_token = nn.Parameter(torch.randn(1, 1, encoder_embed_dim))
        self.dropout = nn.Dropout(emb_dropout)
        self.FuseMamba = FuseMamba(encoder_embed_dim, en_depth, en_heads, dim_head, mlp_dim, dropout,
                                          num_patches)
        self.to_latent = nn.Identity()
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(encoder_embed_dim),
            nn.Linear(encoder_embed_dim, num_classes)
        )

    def encoder(self, x11, x21,x31, x12, x22,x32, x13, x23,x33):   #

        x_fuse1, x_fuse2, x_fuse3 = self.cnn_encoder(x11, x21,x31, x12, x22,x32, x13, x23,x33)  # x_fuse1:64*64*8*8, x_fuse2:64*64*4*4, x_fuse2:64*64*2*2
        x_flat1 = x_fuse1.flatten(2)
        x_flat2 = x_fuse2.flatten(2)
        x_flat3 = x_fuse3.flatten(2)

        x_1  = self.encoder_embedding1(x_flat1)  # [64, 64, 64]
        x_2 = self.encoder_embedding2(x_flat2)   # [64, 64, 64]
        x_3 = self.encoder_embedding3(x_flat3)   # [64, 64, 64]

        x = torch.einsum('nld->ndl', x_1)  # [64, 66, 64]
        x2 = torch.einsum('nld->ndl', x_2)  # [64, 66, 64]
        x3 = torch.einsum('nld->ndl', x_3)  # [64, 66, 64]

        b, n, _ = x.shape
        
        # add position embedding
        x += self.encoder_pos_embed[:, :1]
        x = self.dropout(x)    # [64, 66, 64]

        x2 += self.encoder_pos_embed[:, :1]
        x2 = self.dropout(x2)    # [64, 66, 64]

        x3 += self.encoder_pos_embed[:, :1]
        x3 = self.dropout(x3)    # [64, 66, 64]

        x = self.FuseMamba(x,x2,x3,mask=None)

        return x+x2+x3, x_1

    def classifier(self, x, x_cnn):

        x = self.to_latent(x [:, 0])
        x_cls1 = self.mlp_head(x)

        x_cnn = torch.einsum('ndl->nld', x_cnn)
        x_cls2 = self.cnn_classifier(seq2img(x_cnn))

        x_cls = x_cls1 * self.coefficient1 + x_cls2 * self.coefficient2

        return x_cls

    def forward(self, img11, img21, img31, img12, img22, img32, img13, img23, img33):

        x_vit, x_cnn = self.encoder(img11, img21, img31, img12, img22, img32, img13, img23, img33)
        x_cls = self.classifier(x_vit, x_cnn)
        return x_cls

