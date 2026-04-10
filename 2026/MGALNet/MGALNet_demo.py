import torch
import torch.nn as nn
from torch.nn import functional as F
from timm.models.layers import trunc_normal_
import math
from torch.nn import Softmax
from timm.layers import DropPath
from timm.layers.helpers import to_2tuple

class MLPBlock(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, input_dim)
        )
    def forward(self, x):
        return self.mlp(x)    

class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None):
        super(CrossAttention, self).__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.kv1 = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.kv2 = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.mlp = MLPBlock(input_dim=dim // num_heads, hidden_dim=dim // 2)
    def forward(self, x1, x2):
        B, N, C = x1.shape
        q1 = x1.reshape(B, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3).contiguous()
        q2 = x2.reshape(B, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3).contiguous()
        k1, v1 = self.kv1(x1).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4).contiguous()
        k2, v2 = self.kv2(x2).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4).contiguous()
        ctx1 = (k1.transpose(-2, -1) @ v1) * self.scale
        ctx1 = ctx1.softmax(dim=-2)
        ctx2 = (k2.transpose(-2, -1) @ v2) * self.scale
        ctx2 = ctx2.softmax(dim=-2)
        ctx1 = self.mlp(ctx1)               # (B,1,256,256)-(B,1,256,256)
        ctx2 = self.mlp(ctx2)
        x1 = (q1 @ ctx2).permute(0, 2, 1, 3).reshape(B, N, C).contiguous() 
        x2 = (q2 @ ctx1).permute(0, 2, 1, 3).reshape(B, N, C).contiguous() 
        return x1, x2
class CrossPath(nn.Module):
    def __init__(self, dim, reduction=1, num_heads=None, norm_layer=nn.LayerNorm):
        super().__init__()
        self.channel_proj1 = nn.Linear(dim, dim // reduction * 2)
        self.channel_proj2 = nn.Linear(dim, dim // reduction * 2)
        self.act1 = nn.ReLU(inplace=True)
        self.act2 = nn.ReLU(inplace=True)
        self.cross_attn = CrossAttention(dim // reduction, num_heads=num_heads)
        self.end_proj1 = nn.Linear(dim // reduction * 2, dim)
        self.end_proj2 = nn.Linear(dim // reduction * 2, dim)
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)
    def forward(self, x1, x2):
        y1, u1 = self.act1(self.channel_proj1(x1)).chunk(2, dim=-1)
        y2, u2 = self.act2(self.channel_proj2(x2)).chunk(2, dim=-1)
        v1, v2 = self.cross_attn(u1, u2)
        y1 = torch.cat((y1, v1), dim=-1)
        y2 = torch.cat((y2, v2), dim=-1)
        out_x1 = self.norm1(x1 + self.end_proj1(y1))
        out_x2 = self.norm2(x2 + self.end_proj2(y2))
        return out_x1, out_x2
# Stage 2
class ChannelEmbed(nn.Module):
    def __init__(self, in_channels, out_channels, reduction=1, norm_layer=nn.BatchNorm2d):
        super(ChannelEmbed, self).__init__()
        self.out_channels = out_channels
        self.residual = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.channel_embed = nn.Sequential(
                        nn.Conv2d(in_channels, out_channels//reduction, kernel_size=1, bias=True),
                        nn.Conv2d(out_channels//reduction, out_channels//reduction, kernel_size=3, stride=1, padding=1, bias=True, groups=out_channels//reduction),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(out_channels//reduction, out_channels, kernel_size=1, bias=True),
                        norm_layer(out_channels)
                        )
        self.norm = norm_layer(out_channels)
    def forward(self, x, H, W):
        B, N, _C = x.shape
        x = x.permute(0, 2, 1).reshape(B, _C, H, W).contiguous()
        residual = self.residual(x)
        x = self.channel_embed(x)
        out = self.norm(residual + x)
        return out
    
class FeatureFusionModule(nn.Module):
    def __init__(self, dim, reduction=1, num_heads=None, norm_layer=nn.BatchNorm2d):
        super().__init__()
        self.cross = CrossPath(dim=dim, reduction=reduction, num_heads=num_heads)
        self.channel_emb = ChannelEmbed(in_channels=dim*2, out_channels=dim, reduction=reduction, norm_layer=norm_layer)
        self.apply(self._init_weights)
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()
    def forward(self, x1, x2):
        B, C, H, W = x1.shape
        x1 = x1.flatten(2).transpose(1, 2)
        x2 = x2.flatten(2).transpose(1, 2)
        x1, x2 = self.cross(x1, x2) 
        merge = torch.cat((x1, x2), dim=-1)
        merge = self.channel_emb(merge, H, W)
        return merge

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

class PredictorConv(nn.Module):  
    def __init__(self, embed_dim=256, num_modals=4):
        super().__init__()
        self.num_modals = num_modals
        self.score_nets = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, 3, 1, 1, groups=embed_dim),
            nn.Conv2d(embed_dim, embed_dim, 1), 
            nn.Sigmoid()
        )
    def forward(self, x):
        x_ = self.score_nets(x) 
        return x_

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

        score_predictor_0 = PredictorConv(embed_dim=128, num_modals=ST_c)
        score_predictor_1 = PredictorConv(embed_dim=256, num_modals=ST_c)
        score_predictor_2 = PredictorConv(embed_dim=512, num_modals=ST_c)
        score_predictor_3 = PredictorConv(embed_dim=1024, num_modals=ST_c)
        self.extra_score_predictor = nn.ModuleList([
            score_predictor_0,
            score_predictor_1,
            score_predictor_2,
            score_predictor_3
        ])
        num_heads = [1, 2, 4, 8]
        self.FFMs = nn.ModuleList([
            FeatureFusionModule(dim=128, reduction=1, num_heads=num_heads[0], norm_layer=nn.BatchNorm2d),
            FeatureFusionModule(dim=256, reduction=1, num_heads=num_heads[1], norm_layer=nn.BatchNorm2d),
            FeatureFusionModule(dim=512, reduction=1, num_heads=num_heads[2], norm_layer=nn.BatchNorm2d),
            FeatureFusionModule(dim=1024, reduction=1, num_heads=num_heads[3], norm_layer=nn.BatchNorm2d)])

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
        x_scores = torch.where(x_scores < 1e-5, torch.zeros_like(x_scores), x_scores)
        x_ext = x_scores * x_ext + x_ext
        return x_ext
    
    def forward(self, LS, ST):
        x_LS1 = self.LS_branch(LS) 
        y_ST1 = self.ST_branch(ST)

        x_LS2 = self.layer1_LS(x_LS1)
        y_ST2 = self.layer1_ST(y_ST1)
        x_LS_atten1 = self.attention1_LS(x_LS2)  # torch.Size([18, 128, 1, 1])  
        y_ST_atten1 = self.attention1_ST(y_ST2)
        LS_atten1 = torch.where(x_LS_atten1 < 1e-5, torch.zeros_like(x_LS_atten1), x_LS_atten1)
        ST_atten1 = torch.where(y_ST_atten1 < 1e-5, torch.zeros_like(y_ST_atten1), y_ST_atten1)
        x_LS3 = torch.mul(x_LS2, LS_atten1) 
        y_ST3 = torch.mul(y_ST2, ST_atten1)
        y_sta1 = self.tokenselect(y_ST3, self.extra_score_predictor[0])
        x_LS_4 = self.FFMs[0](x_LS3, y_sta1)
        x_LS_s1 = x_LS_4 + y_sta1 
        x_low_feature = x_LS_s1

        s2_LS1 = self.layer2_LS(x_LS_s1)
        s2_ST1 = self.layer2_ST(y_sta1)

        s2_LS_atten1 = self.attention2_LS(s2_LS1) # torch.Size([18, 256, 1, 1])
        s2_ST_atten1 = self.attention2_ST(s2_ST1)
        LS_atten2 = torch.where(s2_LS_atten1 < 1e-5, torch.zeros_like(s2_LS_atten1), s2_LS_atten1)
        ST_atten2 = torch.where(s2_ST_atten1 < 1e-5, torch.zeros_like(s2_ST_atten1), s2_ST_atten1)
        s2_LS2 = torch.mul(s2_LS1,LS_atten2)
        s2_ST2 = torch.mul(s2_ST1,ST_atten2)
        s2_ST_sta = self.tokenselect(s2_ST2,self.extra_score_predictor[1])
        s2_LS3 = self.FFMs[1](s2_LS2, s2_ST_sta)
        s2_LS = s2_LS3 + s2_ST_sta 
        x_midF_dsn = s2_LS

        s3_LS1 = self.layer3_LS(s2_LS) # torch.Size([18, 512, 28, 28])
        s3_ST1 = self.layer3_ST(s2_ST_sta)
        s3_LS_atten1 = self.attention3_LS(s3_LS1) # torch.Size([18, 512, 1, 1])
        s3_ST_atten1 = self.attention3_ST(s3_ST1)
        LS_atten3 = torch.where(s3_LS_atten1 < 1e-5, torch.zeros_like(s3_LS_atten1), s3_LS_atten1)
        ST_atten3 = torch.where(s3_ST_atten1 < 1e-5, torch.zeros_like(s3_ST_atten1), s3_ST_atten1)
        s3_LS2 = torch.mul(s3_LS1,LS_atten3) # torch.Size([18, 512, 28, 28])
        s3_ST2 = torch.mul(s3_ST1,ST_atten3)
        s3_ST_sta = self.tokenselect(s3_ST2,self.extra_score_predictor[2]) # torch.Size([18, 512, 28, 28])
        s3_LS3 = self.FFMs[2](s3_LS2, s3_ST_sta)
        s3_LS = s3_LS3 + s3_ST_sta # s3_LS.shape: torch.Size([18, 512, 14, 14])

        s4_LS1 = self.layer4_LS(s3_LS)  # torch.Size([18, 1024, 28, 28])
        s4_ST1 = self.layer4_ST(s3_ST_sta)
        s4_LS_atten1 = self.attention4_LS(s4_LS1) # torch.Size([18, 1024, 1, 1])
        s4_ST_atten1 = self.attention4_ST(s4_ST1)
        LS_atten4 = torch.where(s4_LS_atten1 < 1e-5, torch.zeros_like(s4_LS_atten1), s4_LS_atten1)
        ST_atten4 = torch.where(s4_ST_atten1 < 1e-5, torch.zeros_like(s4_ST_atten1), s4_ST_atten1)
        s4_LS2 = torch.mul(s4_LS1,LS_atten4) # torch.Size([18, 1024, 28, 28])
        s4_ST2 = torch.mul(s4_ST1,ST_atten4)
        s4_ST_sta = self.tokenselect(s4_ST2,self.extra_score_predictor[3]) # torch.Size([18, 1024, 28, 28])
        s4_LS3 = self.FFMs[3](s4_LS2, s4_ST_sta)
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

class EMA(nn.Module):
    def __init__(self, channels, factor=8):
        super(EMA, self).__init__()
        self.groups = factor
        assert channels // self.groups > 0
        self.softmax = nn.Softmax(-1)
        self.agp = nn.AdaptiveAvgPool2d((1, 1))
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.gn = nn.GroupNorm(channels // self.groups, channels // self.groups)
        self.conv1x1 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=1, stride=1, padding=0)
        self.conv3x3 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        b, c, h, w = x.size()
        group_x = x.reshape(b * self.groups, -1, h, w)  # b*g,c//g,h,w
        x_h = self.pool_h(group_x)
        x_w = self.pool_w(group_x).permute(0, 1, 3, 2)
        hw = self.conv1x1(torch.cat([x_h, x_w], dim=2))
        x_h, x_w = torch.split(hw, [h, w], dim=2)
        x1 = self.gn(group_x * x_h.sigmoid() * x_w.permute(0, 1, 3, 2).sigmoid())
        x2 = self.conv3x3(group_x)
        x11 = self.softmax(self.agp(x1).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x12 = x2.reshape(b * self.groups, c // self.groups, -1)  # b*g, c//g, hw
        x21 = self.softmax(self.agp(x2).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x22 = x1.reshape(b * self.groups, c // self.groups, -1)  # b*g, c//g, hw
        weights = (torch.matmul(x11, x12) + torch.matmul(x21, x22)).reshape(b * self.groups, 1, h, w)
        return (group_x * weights.sigmoid()).reshape(b, c, h, w)

class Net(nn.Module):
    def __init__(self, Fblock=Bottleneck, size=[224,224], LS_channels=10, ST_channels=20,
                 n_classes=14, recurrence=1):
        super(Net, self).__init__()
        self.feaMoudle = FeatureExtract(Fblock,layers=[3,4,6,3],LS_c=LS_channels,ST_c=ST_channels,num_classes=n_classes)
        self.PPA = PyramidPoolAgg(stride=1)
        self.rcm1 = RCM(dim=1408)
        self.SPCU = SplitAndUpsample(in_channels=[128, 256, 1024],target_sizes=[(56,56), (28,28), (14,14)])
        self.rcm2 = RCM(dim=1024)
        self.rcm3 = RCM(dim=256)
        self.rcm4 = RCM(dim=128)
        self.fema = EMA(128)
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
            nn.BatchNorm2d(14, momentum=0.1),
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
        x_PPA = self.PPA(x_cat) # torch.Size([18, 1408, 14, 14])
        F_RCM1 = self.rcm1(x_PPA)
        out1, out2, out3 = self.SPCU(F_RCM1)
        # torch.Size([18, 128, 56, 56]) torch.Size([18, 256, 28, 28]) torch.Size([18, 1024, 14, 14])
        x_fin1 = self.rcm2(x_fin) # torch.Size([18, 1024, 14, 14])
        fus1 = torch.mul(x_fin1,out3) # torch.Size([18, 1024, 14, 14])
        up1 = F.interpolate(fus1, size=x_midF_dsn.size()[2:], mode='bilinear', align_corners=True)
        fus1_up = self.pconv1(up1) # torch.Size([18, 256, 28, 28])
        x2 = x_midF_dsn + fus1_up
        x_fin2 = self.rcm3(x2)
        fus2 = torch.mul(x_fin2, out2) # torch.Size([18, 256, 28, 28])
        up2 = F.interpolate(fus2, size=x_low_feature.size()[2:], mode='bilinear', align_corners=True)
        fus2_up = self.pconv2(up2) # torch.Size([18, 128, 56, 56])
        x3 = x_low_feature + fus2_up
        x_fin3 = self.rcm4(x3) # torch.Size([18, 128, 56, 56])
        fus3 = torch.mul(x_fin3, out1)
        fus3_up = F.interpolate(fus3, size=[112,112], mode='bilinear', align_corners=True)
        x4 = self.fema(fus3_up) # torch.Size([18, 128, 112, 112])
        x4_up = F.interpolate(x4, size=[224,224], mode='bilinear', align_corners=True)
        fin_out = self.seghead(x4_up)
        loss_f = [fin_out, fus2]

        return loss_f, fin_out
    
if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image1 = torch.randn(18, 10, 224, 224).to(device)
    image2 = torch.randn(18, 20, 224, 224).to(device)
    model = Net().to(device)
    out1,out2 = model(image1,image2)
    print(len(out1),out2.shape)
