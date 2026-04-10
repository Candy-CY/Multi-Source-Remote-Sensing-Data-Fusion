import torch
from torch import nn
from mmengine.model import constant_init, kaiming_init,trunc_normal_init
import torch.nn.functional as F
from timm.models.layers import DropPath


def autopad(k, p=None, d=1):  # kernel, padding, dilation
    # Pad to 'same' shape outputs
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p

class AdConv(nn.Module):
    # Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)
    default_act = nn.SiLU()  # default activation
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))
    def forward_fuse(self, x):
        return self.act(self.conv(x))

class ADown(nn.Module):
    def __init__(self, c1, c2):  # ch_in, ch_out, shortcut, kernels, groups, expand
        super().__init__()
        self.c = c2 // 2
        self.cv1 = AdConv(c1 // 2, self.c, 3, 2, 1)
        self.cv2 = AdConv(c1 // 2, self.c, 1, 1, 0)
    def forward(self, x):
        x = torch.nn.functional.avg_pool2d(x, 2, 1, 0, False, True)
        x1,x2 = x.chunk(2, 1)
        x1 = self.cv1(x1)
        x2 = torch.nn.functional.max_pool2d(x2, 3, 2, 1)
        x2 = self.cv2(x2)
        return torch.cat((x1, x2), 1)

class MSAG(nn.Module):
    def __init__(self, channel):
        super(MSAG, self).__init__()
        self.channel = channel
        self.pointwiseConv = nn.Sequential(
            nn.Conv2d(self.channel, self.channel, kernel_size=1, padding=0, bias=True),
            nn.BatchNorm2d(self.channel),
        )
        self.ordinaryConv = nn.Sequential(
            nn.Conv2d(self.channel, self.channel, kernel_size=3, padding=1, stride=1, bias=True),
            nn.BatchNorm2d(self.channel),
        )
        self.dilationConv = nn.Sequential(
            nn.Conv2d(self.channel, self.channel, kernel_size=3, padding=2, stride=1, dilation=2, bias=True),
            nn.BatchNorm2d(self.channel),
        )
        self.voteConv = nn.Sequential(
            nn.Conv2d(self.channel * 3, self.channel, kernel_size=(1, 1)),
            nn.BatchNorm2d(self.channel),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        x1 = self.pointwiseConv(x)
        x2 = self.ordinaryConv(x)
        x3 = self.dilationConv(x)
        _x = self.relu(torch.cat((x1, x2, x3), dim=1))
        _x = self.voteConv(_x)
        x = x + x * _x
        return x

class ConvX(nn.Module):
    def __init__(self, in_planes, out_planes, kernel=3, stride=1, dilation=1):
        super(ConvX, self).__init__()
        if dilation==1:
            self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel, stride=stride, bias=False)
        else:
            self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel, stride=stride, dilation=dilation, padding=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes, momentum=0.1)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        out = self.relu(self.bn(self.conv(x)))
        return out

class MFACB(nn.Module):
    def __init__(self, in_planes, inter_planes, out_planes, block_num=3, stride=1, dilation=[2,2,2]):
        super(MFACB, self).__init__()
        assert block_num > 1, print("block number should be larger than 1.")
        self.conv_list = nn.ModuleList()
        self.stride = stride
        self.conv_list.append(ConvX(in_planes, inter_planes, stride=stride, dilation=dilation[0]))
        self.conv_list.append(ConvX(inter_planes, inter_planes, stride=stride, dilation=dilation[1]))
        self.conv_list.append(ConvX(inter_planes, inter_planes, stride=stride, dilation=dilation[2]))
        self.process1 = nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=1, padding=0, bias=False ),
            nn.BatchNorm2d(out_planes, momentum=0.1),
            nn.ReLU(inplace=True)
        )
        self.process2 = nn.Sequential(
            nn.Conv2d(inter_planes *3, out_planes, kernel_size=1, padding=0, bias=False ),
            nn.BatchNorm2d(out_planes, momentum=0.1),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        out_list = []
        out = x
        out1 = self.process1(x)
        for idx in range(3):
            out = self.conv_list[idx](out)
            out_list.append(out)
        out = torch.cat(out_list, dim=1)
        return self.process2(out) + out1

# 编码器卷积块
class Aconv_block(nn.Module):
    def __init__(self,in_ch,out_ch):
        super(Aconv_block,self).__init__()
        self.conv_two = nn.Sequential(
            # 第一次卷积
            nn.Conv2d(in_ch, out_ch, kernel_size=1,bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            # 第二次卷积
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
        self.adown = ADown(out_ch * 2,out_ch)
        self.msag = MSAG(out_ch)
        self.mfacb = MFACB(out_ch, out_ch*2, out_ch, dilation=[3,3,3])
        self.maff = Muti_AFF(out_ch)
    def forward(self,x):
        x1 = self.conv_two(x)
        x2 = self.msag(x1)
        x3 = self.mfacb(x2)
        x4 = self.maff(x3,x1)
        xout1 = torch.cat([x1,x4],dim=1)
        xout = self.adown(xout1)
        return xout

class ConvolutionalAttention(nn.Module):
    def __init__(self,in_channels,out_channels,inter_channels,num_heads=2):
        super(ConvolutionalAttention,self).__init__()
        assert out_channels % num_heads == 0, \
            "out_channels ({}) should be be a multiple of num_heads ({})".format(out_channels, num_heads)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.inter_channels = inter_channels
        self.num_heads = num_heads
        self.norm = nn.BatchNorm2d(in_channels, eps=1e-06)
        self.kv =nn.Parameter(torch.zeros(inter_channels, in_channels, 7, 1))
        self.kv3 =nn.Parameter(torch.zeros(inter_channels, in_channels, 1, 7))
        trunc_normal_init(self.kv, std=0.001)
        trunc_normal_init(self.kv3, std=0.001)
        self.apply(self._init_weights)
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_init(m.weight, std=.001)
            if m.bias is not None:
                constant_init(m.bias, val=0.)
        elif isinstance(m, (nn.SyncBatchNorm, nn.BatchNorm2d)):
            constant_init(m.weight, val=1.)
            constant_init(m.bias, val=.0)
        elif isinstance(m, nn.Conv2d):
            trunc_normal_init(m.weight, std=.001)
            if m.bias is not None:
                constant_init(m.bias, val=0.)
    def _act_dn(self, x):
        x_shape = x.shape  # n,c_inter,h,w
        h, w = x_shape[2], x_shape[3]
        x = x.reshape(
            [x_shape[0], self.num_heads, self.inter_channels // self.num_heads, -1])   #n,c_inter,h,w -> n,heads,c_inner//heads,hw
        x = F.softmax(x, dim=3)   
        x = x / (torch.sum(x, dim =2, keepdim=True) + 1e-06)  
        x = x.reshape([x_shape[0], self.inter_channels, h, w]) 
        return x
    def forward(self, x):
        x = self.norm(x)
        x1 = F.conv2d(x,self.kv,bias=None,stride=1,padding=(3,0))  
        x1 = self._act_dn(x1)  
        x1 = F.conv2d(x1, self.kv.transpose(1, 0), bias=None, stride=1,padding=(3,0))  
        x3 = F.conv2d(x,self.kv3,bias=None,stride=1,padding=(0,3)) 
        x3 = self._act_dn(x3)
        x3 = F.conv2d(x3, self.kv3.transpose(1, 0), bias=None, stride=1,padding=(0,3)) 
        x=x1+x3
        return x

class MLP(nn.Module):
    def __init__(self,in_channels,hidden_channels=None,out_channels=None,drop_rate=0.):
        super(MLP,self).__init__()
        hidden_channels = hidden_channels or in_channels
        out_channels = out_channels or in_channels
        self.norm = nn.BatchNorm2d(in_channels, eps=1e-06)
        self.conv1 = nn.Conv2d(in_channels, hidden_channels, 3, 1, 1)
        self.act = nn.GELU()
        self.conv2 = nn.Conv2d(hidden_channels, out_channels, 3, 1, 1)
        self.drop = nn.Dropout(drop_rate)
        self.apply(self._init_weights)
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_init(m.weight, std=.02)
            if m.bias is not None:
                constant_init(m.bias, val=0)
        elif isinstance(m, (nn.SyncBatchNorm, nn.BatchNorm2d)):
            constant_init(m.weight, val=1.0)
            constant_init(m.bias, val=0)
        elif isinstance(m, nn.Conv2d):
            kaiming_init(m.weight)
            if m.bias is not None:
                constant_init(m.bias, val=0)
    def forward(self, x):
        x = self.norm(x)
        x = self.conv1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.conv2(x)
        x = self.drop(x)
        return x

class CFBlock(nn.Module):
    def __init__(self,in_channels,out_channels,num_heads=2,drop_rate=0,drop_path_rate=0):
        super(CFBlock,self).__init__()
        in_channels_l = in_channels
        out_channels_l = out_channels
        self.attn_l = ConvolutionalAttention(in_channels_l,out_channels_l,inter_channels=64,num_heads=num_heads)
        self.mlp_l = MLP(out_channels_l, drop_rate=drop_rate)
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()
    def _init_weights_kaiming(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_init(m.weight, std=.02)
            if m.bias is not None:
                constant_init(m.bias, val=0)
        elif isinstance(m, (nn.SyncBatchNorm, nn.BatchNorm2d)):
            constant_init(m.weight, val=1.0)
            constant_init(m.bias, val=0)
        elif isinstance(m, nn.Conv2d):
            kaiming_init(m.weight)
            if m.bias is not None:
                constant_init(m.bias, val=0)
    def forward(self, x):
        x_res = x
        x = x_res + self.drop_path(self.attn_l(x))
        x = x + self.drop_path(self.mlp_l(x)) 
        return x

def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.data.size()
    channels_per_group = num_channels // groups
    # reshape
    x = x.view(batchsize, groups, channels_per_group, height, width)
    x = torch.transpose(x, 1, 2).contiguous()
    # flatten
    x = x.view(batchsize, -1, height, width)
    return x
def act_layer(act, inplace=False, neg_slope=0.2, n_prelu=1):
    # activation layer
    act = act.lower()
    if act == 'relu':
        layer = nn.ReLU(inplace)
    elif act == 'relu6':
        layer = nn.ReLU6(inplace)
    elif act == 'leakyrelu':
        layer = nn.LeakyReLU(neg_slope, inplace)
    elif act == 'prelu':
        layer = nn.PReLU(num_parameters=n_prelu, init=neg_slope)
    elif act == 'gelu':
        layer = nn.GELU()
    elif act == 'hswish':
        layer = nn.Hardswish(inplace)
    else:
        raise NotImplementedError('activation layer [%s] is not found' % act)
    return layer


class EUCB(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, activation='relu'):
        super(EUCB, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.up_dwc = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(self.in_channels, self.in_channels, kernel_size=kernel_size, stride=stride,
                      padding=kernel_size // 2, groups=self.in_channels, bias=False),
            nn.BatchNorm2d(self.in_channels),
            act_layer(activation, inplace=True)
        )
        self.pwc = nn.Sequential(
            nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1, stride=1, padding=0, bias=True)
        )
    def forward(self, x):
        x = self.up_dwc(x)
        x = channel_shuffle(x, self.in_channels)
        x = self.pwc(x)
        return x

class Dec_block(nn.Module):
    def __init__(self,in_ch,out_ch):
        super(Dec_block,self).__init__()
        self.conv = nn.Sequential(
            # 第一次卷积
            nn.Conv2d(out_ch, out_ch, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            # 第二次卷积
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
        self.ecub = EUCB(in_ch,out_ch)
        self.msag = MSAG(out_ch)
    def forward(self,x):
        xup = self.ecub(x)
        x1 = self.conv(xup)
        x2 = self.msag(x1)
        xout = xup + x2
        return xout

# SegHead
class SeparableConvBNReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1,
                 norm_layer=nn.BatchNorm2d):
        super(SeparableConvBNReLU, self).__init__(
            nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride, dilation=dilation,
                      padding=((stride - 1) + dilation * (kernel_size - 1)) // 2,
                      groups=in_channels, bias=False),
            norm_layer(in_channels),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            norm_layer(out_channels),
            nn.ReLU6()
        )

class Convhead(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, bias=False):
        super(Convhead, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride, padding=((stride - 1) + dilation * (kernel_size - 1)) // 2)
        )

class MAF(nn.Module):
    def __init__(self, dim, fc_ratio, dilation=[1, 3, 5], dropout=0.1, num_classes=14):
        super(MAF, self).__init__()
        self.conv0 = nn.Conv2d(dim, dim//fc_ratio, 1)
        self.bn0 = nn.BatchNorm2d(dim//fc_ratio)
        self.conv1_1 = nn.Conv2d(dim//fc_ratio, dim//fc_ratio, 3, padding=dilation[-3], dilation=dilation[-3], groups=dim//fc_ratio)
        self.bn1_1 = nn.BatchNorm2d(dim//fc_ratio)
        self.conv1_2 = nn.Conv2d(dim//fc_ratio, dim, 1)
        self.bn1_2 = nn.BatchNorm2d(dim)
        self.conv2_1 = nn.Conv2d(dim//fc_ratio, dim//fc_ratio, 3, padding=dilation[-2], dilation=dilation[-2], groups=dim//fc_ratio)
        self.bn2_1 = nn.BatchNorm2d(dim//fc_ratio)
        self.conv2_2 = nn.Conv2d(dim//fc_ratio, dim, 1)
        self.bn2_2 = nn.BatchNorm2d(dim)
        self.conv3_1 = nn.Conv2d(dim//fc_ratio, dim//fc_ratio, 3, padding=dilation[-1], dilation=dilation[-1], groups=dim//fc_ratio)
        self.bn3_1 = nn.BatchNorm2d(dim//fc_ratio)
        self.conv3_2 = nn.Conv2d(dim//fc_ratio, dim, 1)
        self.bn3_2 = nn.BatchNorm2d(dim)
        self.relu = nn.ReLU6()
        self.conv4 = nn.Conv2d(dim, dim, 1)
        self.bn4 = nn.BatchNorm2d(dim)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(dim, dim//fc_ratio, 1, 1),
            nn.ReLU6(),
            nn.Conv2d(dim//fc_ratio, dim, 1, 1),
            nn.Sigmoid()
        )
        self.s_conv = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=5, padding=2)
        self.sigmoid = nn.Sigmoid()
        self.head = nn.Sequential(SeparableConvBNReLU(dim, dim, kernel_size=3),
                                  nn.Dropout2d(p=dropout, inplace=True),
                                  Convhead(dim, num_classes, kernel_size=1))
    def forward(self, x):
        u = x.clone()
        attn1_0 = self.relu(self.bn0(self.conv0(x)))
        attn1_1 = self.relu(self.bn1_1(self.conv1_1(attn1_0)))
        attn1_1 = self.relu(self.bn1_2(self.conv1_2(attn1_1)))
        attn1_2 = self.relu(self.bn2_1(self.conv2_1(attn1_0)))
        attn1_2 = self.relu(self.bn2_2(self.conv2_2(attn1_2)))
        attn1_3 = self.relu(self.bn3_1(self.conv3_1(attn1_0)))
        attn1_3 = self.relu(self.bn3_2(self.conv3_2(attn1_3)))
        c_attn = self.avg_pool(x)
        c_attn = self.fc(c_attn)
        c_attn = u * c_attn
        s_max_out, _ = torch.max(x, dim=1, keepdim=True)
        s_avg_out = torch.mean(x, dim=1, keepdim=True)
        s_attn = torch.cat((s_avg_out, s_max_out), dim=1)
        s_attn = self.s_conv(s_attn)
        s_attn = self.sigmoid(s_attn)
        s_attn = u * s_attn
        attn = attn1_1 + attn1_2 +  attn1_3
        attn = self.relu(self.bn4(self.conv4(attn)))
        attn = u * attn
        out = self.head(attn + c_attn + s_attn)
        return out

# 多特征融合 AFF, 一个像素级尺度，多个语义级尺度
class Muti_AFF(nn.Module):
    def __init__(self,channels, r=4):
        super(Muti_AFF, self).__init__()
        inter_channels = int(channels // r)
        self.local_att = nn.Sequential(
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )
        self.context1 = nn.Sequential(
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels)
        )
        self.context2 = nn.Sequential(
            nn.AdaptiveAvgPool2d((8, 8)),
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels)
        )
        self.context3 = nn.Sequential(
            nn.AdaptiveAvgPool2d((16, 16)),
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels)
        )
        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )
        self.sigmoid = nn.Sigmoid()
    def forward(self, x, residual):
        h, w = x.shape[2], x.shape[3]  # 获取输入 x 的高度和宽度
        xa = x + residual
        xl = self.local_att(xa)
        c1 = self.context1(xa)
        c2 = self.context2(xa)
        c3 = self.context3(xa)
        xg = self.global_att(xa)
        # 将 c1, c2, c3 还原到原本的大小，按均匀分布
        c1 = F.interpolate(c1, size=[h, w], mode='nearest')
        c2 = F.interpolate(c2, size=[h, w], mode='nearest')
        c3 = F.interpolate(c3, size=[h, w], mode='nearest')
        xlg = xl + xg + c1 + c2 + c3
        wei = self.sigmoid(xlg)
        xo = 2 * x * wei + 2 * residual * (1 - wei)
        return xo

class Net(nn.Module):
    def __init__(self, Cin_N, num_class=12): # 3,14
        super(Net, self).__init__()
        # 定义编码器部分
        self.convb1 = Aconv_block(Cin_N,32)
        self.convb2 = Aconv_block(32, 64)
        self.convb3 = Aconv_block(64, 128)
        self.convb4 = Aconv_block(128,256)
        # 定义解码器部分
        self.dconv1 = Dec_block(256,128)
        self.dconv2 = Dec_block(128,64)
        self.dconv3 = Dec_block(64,32)
        self.dconv4 = Dec_block(32,32)
        # 定义Conv-Former
        self.cfb = CFBlock(256,256)
        self.patch2 = nn.Sequential(
            nn.Conv2d(Cin_N, 32, kernel_size=1, bias=True),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            CFBlock(32,32)
        )
        # 定义分割头
        self.seg_head = MAF(dim =42, fc_ratio=4, dilation=[1, 3, 5], dropout=0.1, num_classes=num_class)
        # 定义数据融合模块
        self.Fea_F1 = Muti_AFF(256)
        self.Fea_F2 = Muti_AFF(128)
        self.Fea_F3 = Muti_AFF(64)
        self.Fea_F4 = Muti_AFF(32)

    def forward(self,x):
        # 编码路径 x:(18, 10, 224, 224)
        conv1 = self.convb1(x) # # (10,32) torch.Size([18, 32, 112, 112])
        conv2 = self.convb2(conv1) # (35, 64) torch.Size([18, 64, 56, 56])
        conv3 = self.convb3(conv2) # (99, 128) torch.Size([18, 128, 28, 28])
        conv4 = self.convb4(conv3) # (227,256) torch.Size([18, 256, 14, 14])
        # 解码路径
        xcfb = self.cfb(conv4) # torch.Size([18, 256, 14, 14])
        fup = self.Fea_F1(xcfb,conv4) # torch.Size([18, 256, 14, 14])
        d1upf = self.dconv1(fup) # torch.Size([18, 128, 28, 28])
        fup1 = self.Fea_F2(d1upf,conv3) # torch.Size([18, 128, 28, 28])
        d2upf = self.dconv2(fup1) # torch.Size([18, 64, 56, 56])
        fup2 = self.Fea_F3(d2upf,conv2) # torch.Size([18, 64, 56, 56])
        d3upf = self.dconv3(fup2) # torch.Size([18, 32, 112, 112])
        fup3 = self.Fea_F4(d3upf,conv1) # torch.Size([18, 32, 224, 224])
        d4upf = self.dconv4(fup3) # torch.Size([18, 32, 224, 224])
        xcfb1 = self.patch2(x) # torch.Size([18, 32, 224, 224])
        fin = self.Fea_F4(d4upf, xcfb1) # torch.Size([18, 32, 224, 224])
        fse = torch.cat([fin,x],dim=1) # torch.Size([18, 42, 224, 224])
        out = self.seg_head(fse)
        return out

if __name__ == '__main__':
    x = torch.randn(18, 10, 224, 224)
    model = Net(10,12)
    output = model(x)
    print(output.shape)
