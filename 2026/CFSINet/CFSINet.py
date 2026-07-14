import torch
import torch.nn as nn
from torch.nn import functional as F
from timm.layers import DropPath
from timm.layers.helpers import to_2tuple
from einops import rearrange

# 特征处理与特征提取
def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)
class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None, fist_dilation=1, multi_grid=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=dilation*multi_grid, dilation=dilation*multi_grid, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=False)
        self.relu_inplace = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.dilation = dilation
        self.stride = stride
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out = out + residual
        out = self.relu_inplace(out)
        return out

# 细粒度区域聚焦模块
class PredictorConv(nn.Module):  
    def __init__(self, embed_dim=256, num_modals=4):  # 将 embed_dim 改为 256，匹配输入
        super().__init__()
        self.num_modals = num_modals
        self.score_nets = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, 3, 1, 1, groups=embed_dim),  # 深度卷积，保持通道数 256 不变
            nn.Conv2d(embed_dim, embed_dim, 1),  # 保持通道数不变为 256
            nn.Sigmoid()  # 使用 Sigmoid 限制输出范围为 [0, 1]
        )
    def forward(self, x):
        x_ = self.score_nets(x)  # 输入形状 (B, 256, 57, 57), 输出形状仍为 (B, 256, 57, 57)
        return x_
    
# 跨分支特征融合阶段
class CovBlock(nn.Module):
    def __init__(self, feature_dimension, features_num, hidden_dim, dropout=0.05):
        super().__init__()
        self.cov_mlp = nn.Sequential(
            nn.Linear(feature_dimension, feature_dimension),
            nn.Dropout(dropout, inplace=True),
            nn.LeakyReLU(inplace=True),
            nn.Linear(feature_dimension, hidden_dim),
            nn.LeakyReLU(inplace=True),
            nn.Linear(hidden_dim, features_num)
        )

    def forward(self, x):
        x = x - x.mean(dim=-1, keepdim=True)
        cov = x.transpose(-2, -1) @ x
        cov_norm = torch.norm(x, p=2, dim=-2, keepdim=True)
        cov_norm = cov_norm.transpose(-2, -1) @ cov_norm
        cov /= cov_norm
        weight = self.cov_mlp(cov)
        return weight

class BandSelectBlock(nn.Module):
    def __init__(self, feature_dimension, features_num):
        super().__init__()
        self.CovBlockList = nn.ModuleList([])
        for _ in range(features_num):
            self.CovBlockList.append(
                CovBlock(feature_dimension, 1, round(feature_dimension * 0.6), 0)
            )
        self.global_covblock = CovBlock(features_num, 1, features_num, 0)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
    def forward(self, feature_maps):
        H = feature_maps[0].shape[2]
        W = feature_maps[0].shape[3]
        C_weights = []
        for feature_map, block in zip(feature_maps, self.CovBlockList):
            input = rearrange(feature_map, 'B C H W -> B (H W) C', H=H) / (H * W - 1)
            C_weights.append(block(input).squeeze_(-1))
        weight_matrix = torch.stack(C_weights, dim=1)
        feature_maps = torch.stack(feature_maps, dim=1)
        output = weight_matrix.unsqueeze_(-1).unsqueeze_(-1) * feature_maps
        global_weight = self.global_pool(feature_maps).squeeze_(-1).squeeze_(-1)
        global_weight = F.softmax(
            self.global_covblock(global_weight.transpose_(-1, -2)),
            dim=-2
        )
        output = torch.sum(output * global_weight.unsqueeze(-1).unsqueeze(-1), dim=1)

        return output

class FeatureExtract(nn.Module):
    def __init__(self, block = Bottleneck, layers = [3, 4, 6, 3], LS_c = 14, ST_c=20, num_classes=14):
        super(FeatureExtract, self).__init__()
        # LS_branch
        self.LS_branch = nn.Sequential(
            conv3x3(LS_c,32),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=False),
            conv3x3(32, 64),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)
        )
        # ST_branch
        self.ST_branch = nn.Sequential(
            conv3x3(ST_c, 32), # 下采样
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=False),
            conv3x3(32, 64),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)
        )
        # 定义细粒度区域聚焦模块，逐个写出每个 embed_dim 对应的 PredictorConv
        score_predictor_0 = PredictorConv(embed_dim=128, num_modals=ST_c)
        score_predictor_1 = PredictorConv(embed_dim=256, num_modals=ST_c)
        score_predictor_2 = PredictorConv(embed_dim=512, num_modals=ST_c)
        score_predictor_3 = PredictorConv(embed_dim=1024, num_modals=ST_c)
        # 将每个细粒度区域聚焦模块加入 ModuleList中
        self.extra_score_predictor = nn.ModuleList([
            score_predictor_0,
            score_predictor_1,
            score_predictor_2,
            score_predictor_3
        ])
        # 定义跨模态特征交互融合FeatureFusionModule
        num_heads = [1, 2, 4, 8]
        self.FFMs = nn.ModuleList([
            BandSelectBlock(feature_dimension=128, features_num=2),
            BandSelectBlock(feature_dimension=256, features_num=2),
            BandSelectBlock(feature_dimension=512, features_num=2),
            BandSelectBlock(feature_dimension=1024, features_num=2)])

        self.layer1_LS = self._make_layer_LS(block, 64, 32, layers[0], stride=2)
        self.layer1_ST = self._make_layer_ST(block, 64, 32, layers[0], stride=2)
        self.attention1_LS = self.attention(128)
        self.attention1_ST = self.attention(128)

        self.layer2_LS = self._make_layer_LS(block, 128, 64, layers[1], stride=2)
        self.layer2_ST = self._make_layer_ST(block, 128, 64, layers[1], stride=2)
        self.attention2_LS = self.attention(256)
        self.attention2_ST = self.attention(256)

        self.layer3_LS = self._make_layer_LS(block, 256, 128, layers[2], stride=2, dilation=2)
        self.layer3_ST = self._make_layer_ST(block, 256, 128, layers[2], stride=2, dilation=2)
        self.attention3_LS = self.attention(512)
        self.attention3_ST = self.attention(512)

        self.layer4_LS = self._make_layer_LS(block, 512, 256, layers[3], stride=1, dilation=3, multi_grid=(1, 1, 1))
        self.layer4_ST = self._make_layer_ST(block, 512, 256, layers[3], stride=1, dilation=3, multi_grid=(1, 1, 1))
        self.attention4_LS = self.attention(1024)
        self.attention4_ST = self.attention(1024)

    
    def _make_layer_LS(self, block, inplanes, planes, blocks, stride=1, dilation=1, multi_grid=1):
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, affine=True))
        layers = []
        generate_multi_grid = lambda index, grids: grids[index%len(grids)] if isinstance(grids, tuple) else 1
        layers.append(block(inplanes, planes, stride,dilation=dilation, downsample=downsample, multi_grid=generate_multi_grid(0, multi_grid)))
        inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(inplanes, planes, dilation=dilation, multi_grid=generate_multi_grid(i, multi_grid)))
        return nn.Sequential(*layers)

    def _make_layer_ST(self, block, inplanes, planes, blocks, stride=1, dilation=1, multi_grid=1):
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, affine=True))
        layers = []
        generate_multi_grid = lambda index, grids: grids[index % len(grids)] if isinstance(grids, tuple) else 1
        layers.append(block(inplanes, planes, stride, dilation=dilation, downsample=downsample,
                            multi_grid=generate_multi_grid(0, multi_grid)))
        inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(inplanes, planes, dilation=dilation, multi_grid=generate_multi_grid(i, multi_grid)))
        return nn.Sequential(*layers)
    
    def attention(self, num_channels):
        pool_attention = nn.AdaptiveAvgPool2d(1)
        conv_attention = nn.Conv2d(num_channels, num_channels, kernel_size=1)
        activate = nn.Sigmoid()
        return nn.Sequential(pool_attention, conv_attention, activate)

    def tokenselect(self, x_ext, module):
        x_scores = module(x_ext)
        # 将小于阈值值替换为 0，建议选择比较小的值，较大的值可能导致有用特征被丢弃
        x_scores = torch.where(x_scores < 1e-5, torch.zeros_like(x_scores), x_scores)
        x_ext = x_scores * x_ext + x_ext
        return x_ext
    
    def forward(self, LS, ST):
        # stage1   输入数据形状 torch.Size([18, 3, 224, 224])
        x_LS1 = self.LS_branch(LS)  # x_LS1.shape: torch.Size([18, 64, 112, 112])    3 ----> 32 ----->64
        y_ST1 = self.ST_branch(ST)

        x_LS2 = self.layer1_LS(x_LS1)  # torch.Size([18, 128, 56, 56])  32 -----> 128  expansion=4
        y_ST2 = self.layer1_ST(y_ST1)
        x_LS_atten1 = self.attention1_LS(x_LS2)  # torch.Size([18, 128, 1, 1])  
        y_ST_atten1 = self.attention1_ST(y_ST2)
        # 将小于阈值值替换为 0，建议选择比较小的值，较大的值可能导致有用特征被丢弃
        LS_atten1 = torch.where(x_LS_atten1 < 1e-5, torch.zeros_like(x_LS_atten1), x_LS_atten1)
        ST_atten1 = torch.where(y_ST_atten1 < 1e-5, torch.zeros_like(y_ST_atten1), y_ST_atten1)
        x_LS3 = torch.mul(x_LS2, LS_atten1) # torch.Size([18, 128, 56, 56])
        y_ST3 = torch.mul(y_ST2, ST_atten1) # 注意力增强特征
        y_sta1 = self.tokenselect(y_ST3, self.extra_score_predictor[0]) # torch.Size([18, 128, 56, 56])
        # 跨分支融合1
        feature_list1 = [x_LS3, y_sta1]
        x_LS_4 = self.FFMs[0](feature_list1)  # torch.Size([18, 128, 56, 56])
        x_LS_s1 = x_LS_4 + y_sta1 # torch.Size([18, 128, 56, 56])
        x_low_feature = x_LS_s1

        # stage2  输入数据形状 torch.Size([18, 128, 56, 56])
        s2_LS1 = self.layer2_LS(x_LS_s1) # torch.Size([18, 256, 28, 28])
        s2_ST1 = self.layer2_ST(y_sta1)

        s2_LS_atten1 = self.attention2_LS(s2_LS1) # torch.Size([18, 256, 1, 1])
        s2_ST_atten1 = self.attention2_ST(s2_ST1)
        # 将小于阈值值替换为 0，建议选择比较小的值，较大的值可能导致有用特征被丢弃
        LS_atten2 = torch.where(s2_LS_atten1 < 1e-5, torch.zeros_like(s2_LS_atten1), s2_LS_atten1)
        ST_atten2 = torch.where(s2_ST_atten1 < 1e-5, torch.zeros_like(s2_ST_atten1), s2_ST_atten1)
        s2_LS2 = torch.mul(s2_LS1,LS_atten2) # torch.Size([18, 256, 28, 28])
        s2_ST2 = torch.mul(s2_ST1,ST_atten2)
        s2_ST_sta = self.tokenselect(s2_ST2,self.extra_score_predictor[1]) # torch.Size([18, 256, 28, 28])
        # 跨分支融合2
        feature_list2 = [s2_LS2, s2_ST_sta]
        s2_LS3 = self.FFMs[1](feature_list2)
        s2_LS = s2_LS3 + s2_ST_sta # torch.Size([18, 256, 28, 28])
        x_midF_dsn = s2_LS

        # stage3  输入数据形状 torch.Size([18, 256, 28, 28])
        s3_LS1 = self.layer3_LS(s2_LS) # torch.Size([18, 512, 28, 28])
        s3_ST1 = self.layer3_ST(s2_ST_sta)
        s3_LS_atten1 = self.attention3_LS(s3_LS1) # torch.Size([18, 512, 1, 1])
        s3_ST_atten1 = self.attention3_ST(s3_ST1)
        # 将小于阈值值替换为 0，建议选择比较小的值，较大的值可能导致有用特征被丢弃
        LS_atten3 = torch.where(s3_LS_atten1 < 1e-5, torch.zeros_like(s3_LS_atten1), s3_LS_atten1)
        ST_atten3 = torch.where(s3_ST_atten1 < 1e-5, torch.zeros_like(s3_ST_atten1), s3_ST_atten1)
        s3_LS2 = torch.mul(s3_LS1,LS_atten3) # torch.Size([18, 512, 28, 28])
        s3_ST2 = torch.mul(s3_ST1,ST_atten3)
        s3_ST_sta = self.tokenselect(s3_ST2,self.extra_score_predictor[2]) # torch.Size([18, 512, 28, 28])
        # 跨分支融合3
        feature_list3 = [s3_LS2, s3_ST_sta]
        s3_LS3 = self.FFMs[2](feature_list3)
        s3_LS = s3_LS3 + s3_ST_sta # s3_LS.shape: torch.Size([18, 512, 14, 14])

        # stage4  输入数据形状 torch.Size([18, 512, 14, 14])
        s4_LS1 = self.layer4_LS(s3_LS)  # torch.Size([18, 1024, 28, 28])
        s4_ST1 = self.layer4_ST(s3_ST_sta)
        s4_LS_atten1 = self.attention4_LS(s4_LS1) # torch.Size([18, 1024, 1, 1])
        s4_ST_atten1 = self.attention4_ST(s4_ST1)
        # 将小于阈值值替换为 0，建议选择比较小的值，较大的值可能导致有用特征被丢弃
        LS_atten4 = torch.where(s4_LS_atten1 < 1e-5, torch.zeros_like(s4_LS_atten1), s4_LS_atten1)
        ST_atten4 = torch.where(s4_ST_atten1 < 1e-5, torch.zeros_like(s4_ST_atten1), s4_ST_atten1)
        s4_LS2 = torch.mul(s4_LS1,LS_atten4) # torch.Size([18, 1024, 28, 28])
        s4_ST2 = torch.mul(s4_ST1,ST_atten4)
        s4_ST_sta = self.tokenselect(s4_ST2,self.extra_score_predictor[3]) # torch.Size([18, 1024, 28, 28])
        # 跨分支融合4
        feature_list4 = [s4_LS2, s4_ST_sta]
        s4_LS3 = self.FFMs[3](feature_list4)
        s4_LS = s4_LS3 + s4_ST_sta # torch.Size([18, 1024, 28, 28])
        # 输出特征大小：torch.Size([18, 128, 56, 56]) torch.Size([18, 256, 28, 28]) torch.Size([18, 1024, 14, 14])
        return x_low_feature, x_midF_dsn, s4_LS

class PyramidPoolAgg(nn.Module):
    def __init__(self, stride=2):
        super().__init__()
        self.stride = stride
    def forward(self, inputs):
        B, C, H, W = inputs[-1].shape
        H = (H - 1) // self.stride + 1
        W = (W - 1) // self.stride + 1
        return torch.cat([nn.functional.adaptive_avg_pool2d(inp, (H, W)) for inp in inputs], dim=1)

class ConvMlp(nn.Module):
    """ MLP using 1x1 convs that keeps spatial dims"""
    def __init__(
            self, in_features, hidden_features=None, out_features=None, act_layer=nn.ReLU,
            norm_layer=None, bias=True, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = to_2tuple(bias)
        self.fc1 = nn.Conv2d(in_features, hidden_features, kernel_size=1, bias=bias[0])
        self.norm = norm_layer(hidden_features) if norm_layer else nn.Identity()
        self.act = act_layer()
        self.drop = nn.Dropout(drop)
        self.fc2 = nn.Conv2d(hidden_features, out_features, kernel_size=1, bias=bias[1])
    def forward(self, x):
        x = self.fc1(x)
        x = self.norm(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        return x

class RCA(nn.Module):
    def __init__(self, inp,  kernel_size=1, ratio=1, band_kernel_size=11,dw_size=(1,1), padding=(0,0), stride=1, square_kernel_size=2, relu=True):
        super(RCA, self).__init__()
        self.dwconv_hw = nn.Conv2d(inp, inp, square_kernel_size, padding=square_kernel_size//2, groups=inp)
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        gc=inp//ratio
        self.excite = nn.Sequential(
                nn.Conv2d(inp, gc, kernel_size=(1, band_kernel_size), padding=(0, band_kernel_size//2), groups=gc),
                nn.BatchNorm2d(gc),
                nn.ReLU(inplace=True),
                nn.Conv2d(gc, inp, kernel_size=(band_kernel_size, 1), padding=(band_kernel_size//2, 0), groups=gc),
                nn.Sigmoid()
            )
    def sge(self, x):
        x_h = self.pool_h(x) #[N, D, C, 1]
        x_w = self.pool_w(x)
        x_gather = x_h + x_w #.repeat(1,1,1,x_w.shape[-1])
        ge = self.excite(x_gather) # [N, 1, C, 1]
        return ge
    def forward(self, x):
        loc=self.dwconv_hw(x)
        att=self.sge(x)
        out = att*loc
        
        return out

class RCM(nn.Module):
    """ MetaNeXtBlock Block
    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        ls_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """
    def __init__(self,dim,token_mixer=RCA,norm_layer=nn.BatchNorm2d,mlp_layer=ConvMlp,
                 mlp_ratio=2,act_layer=nn.GELU,ls_init_value=1e-6,drop_path=0.,dw_size=11,
                 square_kernel_size=3,ratio=1,):
        super().__init__()
        self.token_mixer = token_mixer(dim, band_kernel_size=dw_size, square_kernel_size=square_kernel_size, ratio=ratio)
        self.norm = norm_layer(dim)
        self.mlp = mlp_layer(dim, int(mlp_ratio * dim), act_layer=act_layer)
        self.gamma = nn.Parameter(ls_init_value * torch.ones(dim)) if ls_init_value else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
    def forward(self, x):
        shortcut = x
        x = self.token_mixer(x)
        x = self.norm(x)
        x = self.mlp(x)
        if self.gamma is not None:
            x = x.mul(self.gamma.reshape(1, -1, 1, 1))
        x = self.drop_path(x) + shortcut
        return x

class SplitAndUpsample(nn.Module):
    def __init__(self, in_channels, target_sizes):
        super().__init__()
        self.in_channels = in_channels
        self.target_sizes = target_sizes  # 原始输入的空间尺寸：[(56,56), (28,28), (14,14)]
        # 上采样层（双线性插值）
        self.upsamplers = nn.ModuleList([
            nn.Upsample(size=size, mode='bilinear', align_corners=False)
            for size in target_sizes
        ])
    def forward(self, x):
        # 1. 分割通道
        splits = x.split(self.in_channels, dim=1)
        # 2. 分别上采样到原始空间尺寸
        outputs = [self.upsamplers[i](split) for i, split in enumerate(splits)]
        return outputs

# 定义EMA模块: Multi-Scale Attention (EMA) Module
class EMA(nn.Module):
    def __init__(self, channels, factor=8):
        super(EMA, self).__init__()
        # 设置分组数量，用于特征分组
        self.groups = factor
        # 确保分组后的通道数大于0
        assert channels // self.groups > 0
        # softmax激活函数，用于归一化
        self.softmax = nn.Softmax(-1)
        # 全局平均池化，生成通道描述符
        self.agp = nn.AdaptiveAvgPool2d((1, 1))
        # 水平方向的平均池化，用于编码水平方向的全局信息
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        # 垂直方向的平均池化，用于编码垂直方向的全局信息
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        # GroupNorm归一化，减少内部协变量偏移
        self.gn = nn.GroupNorm(channels // self.groups, channels // self.groups)
        # 1x1卷积，用于学习跨通道的特征
        self.conv1x1 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=1, stride=1, padding=0)
        # 3x3卷积，用于捕捉更丰富的空间信息
        self.conv3x3 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        b, c, h, w = x.size()
        # 对输入特征图进行分组处理
        group_x = x.reshape(b * self.groups, -1, h, w)  # b*g,c//g,h,w
        # 应用水平和垂直方向的全局平均池化
        x_h = self.pool_h(group_x)
        x_w = self.pool_w(group_x).permute(0, 1, 3, 2)
        # 通过1x1卷积和sigmoid激活函数，获得注意力权重
        hw = self.conv1x1(torch.cat([x_h, x_w], dim=2))
        x_h, x_w = torch.split(hw, [h, w], dim=2)
        # 应用GroupNorm和注意力权重调整特征图
        x1 = self.gn(group_x * x_h.sigmoid() * x_w.permute(0, 1, 3, 2).sigmoid())
        x2 = self.conv3x3(group_x)
        # 将特征图通过全局平均池化和softmax进行处理，得到权重
        x11 = self.softmax(self.agp(x1).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x12 = x2.reshape(b * self.groups, c // self.groups, -1)  # b*g, c//g, hw
        x21 = self.softmax(self.agp(x2).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x22 = x1.reshape(b * self.groups, c // self.groups, -1)  # b*g, c//g, hw
        # 通过矩阵乘法和sigmoid激活获得最终的注意力权重，调整特征图
        weights = (torch.matmul(x11, x12) + torch.matmul(x21, x22)).reshape(b * self.groups, 1, h, w)
        # 将调整后的特征图重塑回原始尺寸
        return (group_x * weights.sigmoid()).reshape(b, c, h, w)

# High-Frequency Prompt Encoder
class LearnableHighPassFilter(nn.Module):
    def __init__(self, H, W, channels=1, per_channel=False, theta_init=0.5, eps=1e-6):
        super().__init__()
        self.H = H
        self.W = W
        self.eps = eps
        self.per_channel = per_channel
        # For rfft2 output, width dimension length is W_r = W//2 + 1
        self.W_r = W // 2 + 1

        if per_channel:
            # store log(theta) per channel for positivity
            self.theta_log = nn.Parameter(torch.log(torch.ones(channels) * theta_init + eps))
        else:
            self.theta_log = nn.Parameter(torch.log(torch.tensor(theta_init + eps)))
        u = torch.fft.fftfreq(self.H)  # cycles per sample, length H
        v = torch.fft.rfftfreq(self.W)  # length W_r
        # create meshgrid (u^2 + v^2)
        uu = u.unsqueeze(1)  # (H,1)
        vv = v.unsqueeze(0)  # (1,W_r)
        R2 = (uu ** 2) + (vv ** 2)  # (H, W_r)
        # store as buffer (float32)
        self.register_buffer("R2", R2.float())

    def forward(self, device=None):
        theta = torch.exp(self.theta_log) + self.eps  # positive
        R2 = self.R2.to(theta.device)
        # Compute Hmap = 1 - exp( - R2 / (2 theta^2) )
        if self.per_channel:
            # theta shape (C,) -> reshape to (C,1,1) then expand
            C = theta.shape[0]
            theta_sq = (theta ** 2).view(C, 1, 1)
            Hmap_c = 1.0 - torch.exp(-R2.unsqueeze(0) / (2.0 * theta_sq))  # (C, H, W_r)
            Hmap = Hmap_c.unsqueeze(0)  # (1, C, H, W_r)
        else:
            theta_sq = (theta ** 2)
            Hmap = 1.0 - torch.exp(-R2 / (2.0 * theta_sq))  # (H, W_r)
            Hmap = Hmap.unsqueeze(0).unsqueeze(0)  # (1,1,H,W_r)

        return Hmap.to(device if device is not None else next(self.parameters()).device)


class HFPE(nn.Module):
    def __init__(self, in_channels: int, H: int, W: int,
                 hidden_channels: int = 64, out_channels: int = 64,
                 conv_kernel: int = 3, per_channel_filter: bool = False):
        super().__init__()
        self.in_channels = in_channels
        self.H = H
        self.W = W
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        # High-pass filter generator (builds H(u,v;theta) for rfft2 grid)
        self.hpf = LearnableHighPassFilter(H=H, W=W, channels=in_channels,
                                           per_channel=per_channel_filter, theta_init=0.5)
        # Conv + LayerNorm block (ConvLN)
        pad = (conv_kernel - 1) // 2
        self.conv_ln = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=conv_kernel, padding=pad, bias=False),
            # Use Channel-wise LayerNorm: GroupNorm(1) acts as LayerNorm over channels for each pixel
            nn.GroupNorm(num_groups=1, num_channels=hidden_channels)
        )

        # Further refinement block: LeakyReLU + Conv (two convs advisable)
        self.refine = nn.Sequential(
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )

        # Final 1x1 conv to produce high-frequency prompt Ph
        self.ph_conv = nn.Conv2d(hidden_channels, out_channels, kernel_size=1, bias=True)

        # Initialization
        self._init_weights()
    def _init_weights(self):
        # initialize convs with kaiming
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=0.2, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            if isinstance(m, nn.GroupNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        # 1) FFT -> Xf: complex tensor shape (B, C, H, W_r)
        Xf = torch.fft.rfft2(x, dim=(-2, -1))  # complex tensor
        # 2) Build Hmap and multiply in freq domain
        Hmap = self.hpf(device=x.device)  # (1,1 or C, H, W_r)
        # Broadcast to (B, C, H, W_r)
        if Hmap.shape[1] == 1 and C > 1:
            Hmap = Hmap.expand(1, C, -1, -1)
        Hmap = Hmap.expand(B, -1, -1, -1)  # (B, C, H, W_r)
        # Multiply complex spectrum by real-valued Hmap (broadcasting works; real scales complex)
        Xf_filtered = Xf * Hmap  # complex
        # 3) inverse FFT -> G(x,y) real tensor (B, C, H, W)
        G = torch.fft.irfft2(Xf_filtered, s=(H, W), dim=(-2, -1))  # real
        # 4) Conv + LayerNorm on spatial domain: ConvLN(G)
        X_high = self.conv_ln(G)  # (B, hidden, H, W)
        # 5) LeakyReLU + conv refine
        X_ref = self.refine(X_high)
        # 6) final 1x1 conv -> Ph
        Ph = self.ph_conv(X_ref)  # (B, out_channels, H, W)
        return Ph

class Net(nn.Module):
    def __init__(self, Fblock=Bottleneck, LS_channels=9, ST_channels=4, n_classes=11):
        super(Net, self).__init__()
        # this module extract remote sensing image features
        self.feaMoudle = FeatureExtract(Fblock,layers=[3,4,6,3],LS_c=LS_channels,ST_c=ST_channels,num_classes=n_classes)
        # this module enhance remote sensing image features
        # 金字塔上下文提取
        self.PPA = PyramidPoolAgg(stride=1)
        self.SPCU = SplitAndUpsample(in_channels=[128, 256, 1024],target_sizes=[(56,56), (28,28), (14,14)])
        # 空间特征增强与重建
        self.rcms = nn.ModuleList([
            RCM(dim=1408),
            RCM(dim=1024),
            RCM(dim=256),
            RCM(dim=128)])
        self.fema = EMA(128)
        # 频域信息提取
        self.hfpes = nn.ModuleList([
            HFPE(in_channels=1024, H=14, W=14, hidden_channels=1408, out_channels=1024, per_channel_filter=True),
            HFPE(in_channels=256, H=28, W=28, hidden_channels=512, out_channels=256, per_channel_filter=True),
            HFPE(in_channels=128, H=56, W=56, hidden_channels=256, out_channels=128, per_channel_filter=True)])
        # torch.Size([18, 1024, 14, 14])    torch.Size([18, 256, 28, 28])  torch.Size([18, 128, 56, 56])
        self.PKFMs = nn.ModuleList([
            BandSelectBlock(feature_dimension=128, features_num=2),
            BandSelectBlock(feature_dimension=256, features_num=2),
            BandSelectBlock(feature_dimension=512, features_num=2),
            BandSelectBlock(feature_dimension=1024, features_num=2)])
        # 通道处理函数
        self.pconv1 = nn.Sequential(
            nn.Conv2d(1024, 1024, 3, 1, 1, groups=1024),
            nn.Conv2d(1024, 512, 3, 1, 1),
            nn.BatchNorm2d(512, momentum=0.1),
            nn.Conv2d(512, 256, 3, 1, 1),
            nn.BatchNorm2d(256, momentum=0.1),
            nn.Conv2d(256, 256, 1),
            nn.BatchNorm2d(256),
            nn.Dropout2d(p=0.1)
        )
        self.pconv2 = nn.Sequential(
            nn.Conv2d(256, 256, 3, 1, 1, groups=256),
            nn.Conv2d(256, 128, 3, 1, 1),
            nn.BatchNorm2d(128, momentum=0.1),
            nn.Conv2d(128, 128, 1),
            nn.BatchNorm2d(128),
            nn.Dropout2d(p=0.1)
        )
        self.seghead = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1,bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, 3, padding=1,bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, n_classes, 1,bias=False),
            nn.BatchNorm2d(n_classes, momentum=0.1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.1)
            )

        # init weights
        self._init_weight()
    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, LS, ST):
        # 输入特征大小： LS and ST: torch.randn(18, 3, 224, 224)
        # 经历特征提取模块后输出的特征大小     x_low_feature：torch.Size([18, 128, 56, 56]) 
        # x_midF_dsn：torch.Size([18, 256, 28, 28])  x_fin：torch.Size([18, 1024, 14, 14])
        x_low_feature, x_midF_dsn, x_fin = self.feaMoudle(LS, ST)
        x_cat = [x_low_feature, x_midF_dsn, x_fin]
        # 金字塔上下文提取
        x_PPA = self.PPA(x_cat) # torch.Size([18, 1408, 14, 14])
        F_RCM1 = self.rcms[0](x_PPA)
        out1, out2, out3 = self.SPCU(F_RCM1)
        # torch.Size([18, 128, 56, 56]) torch.Size([18, 256, 28, 28]) torch.Size([18, 1024, 14, 14])
        # 先频域，Cat上采样特征，再FRM
        x_finP1 = self.hfpes[0](x_fin)
        x_finK1 = self.rcms[1](x_fin) # # torch.Size([18, 1024, 14, 14])
        x_fin1 = self.PKFMs[3]([x_finP1, x_finK1])
        fus1 = torch.mul(x_fin1,out3) # torch.Size([18, 1024, 14, 14])
        up1 = F.interpolate(fus1, size=x_midF_dsn.size()[2:], mode='bilinear', align_corners=True)
        fus1_up = self.pconv1(up1) # torch.Size([18, 256, 28, 28])

        x_midF_dsnP1 = self.hfpes[1](x_midF_dsn)
        x_midF_dsnK1 = self.rcms[2](x_midF_dsn)
        x_midF_dsn1 = self.PKFMs[1]([x_midF_dsnP1, x_midF_dsnK1])
        x_fin2 = self.rcms[2](x_midF_dsn1+fus1_up)
        fus2 = torch.mul(x_fin2, out2) # torch.Size([18, 256, 28, 28])
        up2 = F.interpolate(fus2, size=x_low_feature.size()[2:], mode='bilinear', align_corners=True)
        fus2_up = self.pconv2(up2) # torch.Size([18, 128, 56, 56])

        x_low_featureP1 = self.hfpes[2](x_low_feature)
        x_low_featureK1 = self.rcms[3](x_low_feature)
        x_low_feature1 = self.PKFMs[0]([x_low_featureP1, x_low_featureK1])
        x_fin3 = self.rcms[3](x_low_feature1+fus2_up)
        fus3 = torch.mul(x_fin3, out1) # torch.Size([18, 128, 56, 56])

        fus3_up = F.interpolate(fus3, size=[112,112], mode='bilinear', align_corners=True)
        x4 = self.fema(fus3_up) # torch.Size([18, 128, 112, 112])
        x4_up = F.interpolate(x4, size=[224,224], mode='bilinear', align_corners=True)
        fin_out = self.seghead(x4_up)
        loss_f = [fin_out, fus2]

        return loss_f, fin_out


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image1 = torch.randn(18, 9, 224, 224).to(device)
    image2 = torch.randn(18, 4, 224, 224).to(device)
    model = Net().to(device)
    out1,out2 = model(image1,image2)
    print(len(out1),out2.shape)
