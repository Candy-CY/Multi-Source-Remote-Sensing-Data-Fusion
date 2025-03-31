from torch.nn import functional as F
import torch
affine_par = True
import functools
from semseg.models.modules.ffm import FeatureFusionModule as FFM
from torch import nn, Tensor
from cc_attention import CrissCrossAttention
from inplace_abn import InPlaceABN, InPlaceABNSync
BatchNorm2d = functools.partial(InPlaceABNSync, activation='identity')


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

def predict_whole(outs, tile_size):
    interp = nn.Upsample(size=tile_size, mode='bilinear', align_corners=True)
    if isinstance(outs, list):
        outs = outs[0]
    prediction = interp(outs)
    return prediction

class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None, fist_dilation=1, multi_grid=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=dilation*multi_grid, dilation=dilation*multi_grid, bias=False)
        self.bn2 = BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = BatchNorm2d(planes * 4)
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

class ConvLayerNorm(nn.Module):
    """Channel first layer norm
    """
    def __init__(self, normalized_shape, eps=1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x

class ASPP_module(nn.Module):
    def __init__(self, inplanes, planes):
        super(ASPP_module, self).__init__()

        self.aspp0 = nn.Sequential(nn.Conv2d(inplanes, planes, kernel_size=1, stride=1, padding=0, dilation=1, bias=False), InPlaceABNSync(planes))
        self.aspp1 = nn.Sequential(nn.Conv2d(inplanes, planes, kernel_size=3, stride=1, padding=6, dilation=6, bias=False), InPlaceABNSync(planes))
        self.aspp2 = nn.Sequential(nn.Conv2d(inplanes, planes, kernel_size=3, stride=1, padding=12, dilation=12, bias=False), InPlaceABNSync(planes))
        self.aspp3 = nn.Sequential(nn.Conv2d(inplanes, planes, kernel_size=3, stride=1, padding=18, dilation=18, bias=False), InPlaceABNSync(planes))

    def forward(self, x):
        x0 = self.aspp0(x)
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)

        return torch.cat((x0, x1, x2, x3), dim=1)


class RCCAModule(nn.Module):
    def __init__(self, in_channels, out_channels, num_classes):
        super(RCCAModule, self).__init__()
        inter_channels = in_channels // 4
        self.conva = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False), InPlaceABNSync(inter_channels))
        self.cca = CrissCrossAttention(inter_channels)
        self.convb = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False), InPlaceABNSync(inter_channels))

        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels + inter_channels, out_channels, kernel_size=3, padding=1, dilation=1, bias=False),
            InPlaceABNSync(out_channels),
            nn.Dropout2d(0.1),
            nn.Conv2d(512, num_classes, kernel_size=1, stride=1, padding=0, bias=True)
        )

    def forward(self, x, recurrence=1):
        output = self.conva(x)
        for i in range(recurrence):
            output = self.cca(output)
        output = self.convb(output)

        output = self.bottleneck(torch.cat([x, output], 1))
        return output

class PredictorConv(nn.Module):  # 生成不同模态数据的评分图（score maps）
    def __init__(self, embed_dim=256, num_modals=4):  # 将 embed_dim 改为 256，匹配输入
        super().__init__()
        self.num_modals = num_modals
        self.score_nets = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, 3, 1, 1, groups=embed_dim),  # 深度卷积，保持通道数 256 不变
            nn.Conv2d(embed_dim, embed_dim, 1),  # 保持通道数不变为 256
            nn.Sigmoid()  # 使用 Sigmoid 限制输出范围为 [0, 1]
        )
    def forward(self, x):
        B, C, H, W = x.shape  # 输入形状 (B, 256, 57, 57)
        x_ = self.score_nets(x)  # 输出形状仍为 (B, 256, 57, 57)
        return x_

class ResNet(nn.Module):
    def __init__(self, block, layers, img_channels, ex_channels, num_classes=11):
        super(ResNet, self).__init__()
        # self.CMNeXt = CMNeXt(model_name='B2', modals=['img', 'basic', 'dem', 'landcover', 'NDVI', 'slope', 'cluster', 'texture'])
        # img branch
        self.img_branch = nn.Sequential(
            conv3x3(img_channels, 64, stride=2),
            BatchNorm2d(64),
            nn.ReLU(inplace=False),
            conv3x3(64, 64),
            BatchNorm2d(64),
            nn.ReLU(inplace=False),
            conv3x3(64, 128),
            BatchNorm2d(128),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=True)
        )

        #Dist&Dem branch
        self.ex_branch = nn.Sequential(
            conv3x3(ex_channels, 64, stride=2),
            BatchNorm2d(64),
            nn.ReLU(inplace=False),
            conv3x3(64, 64),
            BatchNorm2d(64),
            nn.ReLU(inplace=False),
            conv3x3(64, 128),
            BatchNorm2d(128),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=True)
        )

        self.layer1 = self._make_layer_img(block, 128, 64, layers[0])
        self.layer1_d = self._make_layer_d(block, 128, 64, layers[0])
        self.attention_1 = self.attention(256)
        self.attention_1_d = self.attention(256)
        self.attention_ex = self.attention(7)

        self.layer2 = self._make_layer_img(block, 256, 128, layers[1], stride=2)
        self.layer2_d = self._make_layer_d(block, 256, 128, layers[1], stride=2)
        self.attention_2 = self.attention(512)
        self.attention_2_d = self.attention(512)

        self.layer3 = self._make_layer_img(block, 512, 256, layers[2], stride=1, dilation=2)
        self.layer3_d = self._make_layer_d(block, 512, 256, layers[2], stride=1, dilation=2)
        self.attention_3 = self.attention(1024)
        self.attention_3_d = self.attention(1024)

        self.layer4 = self._make_layer_img(block, 1024, 512, layers[3], stride=1, dilation=4, multi_grid=(1, 1, 1))
        self.layer4_d = self._make_layer_d(block, 1024, 512, layers[3], stride=1, dilation=4, multi_grid=(1, 1, 1))
        self.attention_4 = self.attention(2048)
        self.attention_4_d = self.attention(2048)

        self.dsn = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=3, stride=1, padding=1),
            InPlaceABNSync(512),
            nn.Dropout2d(0.1),
            nn.Conv2d(512, num_classes, kernel_size=1, stride=1, padding=0, bias=True)
            )

        num_heads = [1, 2, 4, 8]
        self.FFMs = nn.ModuleList([
            FFM(dim=256, reduction=1, num_heads=num_heads[0], norm_layer=nn.BatchNorm2d),
            FFM(dim=512, reduction=1, num_heads=num_heads[1], norm_layer=nn.BatchNorm2d),
            FFM(dim=1024, reduction=1, num_heads=num_heads[2], norm_layer=nn.BatchNorm2d),
            FFM(dim=2048, reduction=1, num_heads=num_heads[3], norm_layer=nn.BatchNorm2d)])

        # num_heads = [1, 2, 4, 8]
        self.conv1 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=1)
        # self.conv2 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=1)
        # self.conv3 = nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=1)

        # 定义 PredictorConv 模块，逐个写出每个 embed_dim 对应的 PredictorConv
        score_predictor_0 = PredictorConv(embed_dim=256, num_modals=ex_channels)
        score_predictor_1 = PredictorConv(embed_dim=512, num_modals=ex_channels)
        score_predictor_2 = PredictorConv(embed_dim=1024, num_modals=ex_channels)
        score_predictor_3 = PredictorConv(embed_dim=2048, num_modals=ex_channels)
        # 将每个 PredictorConv 模块加入 ModuleList
        self.extra_score_predictor = nn.ModuleList([
            score_predictor_0,
            score_predictor_1,
            score_predictor_2,
            score_predictor_3
        ])

    def attention(self, num_channels):
        pool_attention = nn.AdaptiveAvgPool2d(1)
        conv_attention = nn.Conv2d(num_channels, num_channels, kernel_size=1)
        activate = nn.Sigmoid()
        return nn.Sequential(pool_attention, conv_attention, activate)

    def tokenselect(self, x_ext, module, stage, labels, attention):       #从M个模态中选择最有效的块;list7(22,128,56,56)
        x_scores = module(x_ext)                #list7(B,128,57,57)-list7(B,1,57,57)
        # 将小于阈值值替换为 0，建议选择比较小的值，较大的值可能导致有用特征被丢弃
        x_scores = torch.where(x_scores < 1e-5, torch.zeros_like(x_scores), x_scores)
        x_ext = x_scores * x_ext + x_ext    #输出list7(22,128,56,56)
        return x_ext

    def _make_layer_img(self, block, inplanes, planes, blocks, stride=1, dilation=1, multi_grid=1):
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                BatchNorm2d(planes * block.expansion, affine=affine_par))

        layers = []
        generate_multi_grid = lambda index, grids: grids[index%len(grids)] if isinstance(grids, tuple) else 1
        layers.append(block(inplanes, planes, stride,dilation=dilation, downsample=downsample, multi_grid=generate_multi_grid(0, multi_grid)))
        inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(inplanes, planes, dilation=dilation, multi_grid=generate_multi_grid(i, multi_grid)))

        return nn.Sequential(*layers)

    def _make_layer_d(self, block, inplanes, planes, blocks, stride=1, dilation=1, multi_grid=1):
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                BatchNorm2d(planes * block.expansion, affine=affine_par))

        layers = []
        generate_multi_grid = lambda index, grids: grids[index % len(grids)] if isinstance(grids, tuple) else 1
        layers.append(block(inplanes, planes, stride, dilation=dilation, downsample=downsample,
                            multi_grid=generate_multi_grid(0, multi_grid)))
        inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(inplanes, planes, dilation=dilation, multi_grid=generate_multi_grid(i, multi_grid)))

        return nn.Sequential(*layers)

    def forward(self, x, x_ex, labels):
        # stage 1
        x = self.img_branch(x)                          #(B,3,224,224)-(B,128,57,57)
        y = self.ex_branch(x_ex)                        #(B,7,224,224)-(B,128,57,57)
        x = self.layer1(x)                              #(B,128,57,57)-(B,256,57,57)
        y = self.layer1_d(y)                            #(B,128,57,57)-(B,256,57,57)
        x_attention = self.attention_1(x)               #(B,256,57,57)-(B,256,1,1)
        y_attention = self.attention_1_d(y)
        # 将小于阈值值替换为 0，建议选择比较小的值，较大的值可能导致有用特征被丢弃
        x_attention = torch.where(x_attention < 1e-5, torch.zeros_like(x_attention), x_attention)
        y_attention = torch.where(y_attention < 1e-5, torch.zeros_like(y_attention), y_attention)
        x = torch.mul(x, x_attention)                   #(B,256,57,57)
        y = torch.mul(y, y_attention)
        y = self.tokenselect(y, self.extra_score_predictor[0], 1, labels, y_attention)
        x = self.FFMs[0](x, y)                          #(B,256,57,57)
        x = x + y
        x_low = x

        # stage 2
        x = self.layer2(x)                          #(B,256,57,57)—#(B,512,29,29)
        y = self.layer2_d(y)                     #(B,256,57,57)—#(B,512,29,29)
        x_attention = self.attention_2(x)
        y_attention = self.attention_2_d(y)
        # 将小于阈值值替换为 0，建议选择比较小的值，较大的值可能导致有用特征被丢弃
        x_attention = torch.where(x_attention < 1e-5, torch.zeros_like(x_attention), x_attention)
        y_attention = torch.where(y_attention < 1e-5, torch.zeros_like(y_attention), y_attention)
        x = torch.mul(x, x_attention)
        y = torch.mul(y, y_attention)
        y = self.tokenselect(y, self.extra_score_predictor[1], 2, labels, y_attention)
        x = self.FFMs[1](x, y)                      #(B,512,29,29)
        x = x + y

        # stage 3
        x = self.layer3(x)                                  #(B,512,29,29)—#(B,1024,29,29)
        y = self.layer3_d(y)
        x_attention = self.attention_3(x)
        y_attention = self.attention_3_d(y)
        # 将小于阈值值替换为 0，建议选择比较小的值，较大的值可能导致有用特征被丢弃
        x_attention = torch.where(x_attention < 1e-5, torch.zeros_like(x_attention), x_attention)
        y_attention = torch.where(y_attention < 1e-5, torch.zeros_like(y_attention), y_attention)
        x = torch.mul(x, x_attention)
        y = torch.mul(y, y_attention)
        y = self.tokenselect(y, self.extra_score_predictor[2], 2, labels, y_attention)
        x = self.FFMs[2](x, y)                          #(B,1024,29,29)
        x = x + y
        x_dsn = self.dsn(x)

        # stage 4
        x = self.layer4(x)                                  #(B,1024,29,29)—(B,2048,29,29)
        y = self.layer4_d(y)
        x_attention = self.attention_4(x)
        y_attention = self.attention_4_d(y)
        # 将小于阈值值替换为 0，建议选择比较小的值，较大的值可能导致有用特征被丢弃
        x_attention = torch.where(x_attention < 1e-5, torch.zeros_like(x_attention), x_attention)
        y_attention = torch.where(y_attention < 1e-5, torch.zeros_like(y_attention), y_attention)
        x = torch.mul(x, x_attention)
        y = torch.mul(y, y_attention)
        y = self.tokenselect(y, self.extra_score_predictor[3], 3, labels, y_attention)
        x = self.FFMs[3](x, y)  # [8,64,256,256]
        x = x + y

        return x, x_dsn, x_low

class SFA_DFNet(nn.Module):
    def __init__(self, block, size, in_channels, ex_channels,
                 n_classes, criterion=None, recurrence=2, use_rcca=True):
        super(SFA_DFNet, self).__init__()

        # Main stream and branch(branch extract dist features)
        self.backbone = ResNet(block, [3, 4, 6, 3], img_channels=in_channels, ex_channels=ex_channels, num_classes=n_classes)
        # ASPP
        self.aspp = ASPP_module(2048, 256)
        # RCCA
        self.rcca = RCCAModule(2048, 512, num_classes=n_classes)

        # global pooling
        self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)), nn.Conv2d(2048, 256, 1, stride=1, bias=False))

        self.conv = nn.Conv2d(1280, 256, 1, bias=False)
        self.bn = BatchNorm2d(256)

        self.conv2 = nn.Conv2d(256, 48, 1, bias=False)
        self.bn2 = BatchNorm2d(48)

        self.conv3 = nn.Sequential(nn.Conv2d(304, 256, kernel_size=3, stride=1, padding=1, bias=False), BatchNorm2d(256), nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False), BatchNorm2d(256), nn.Conv2d(256, n_classes, kernel_size=1, stride=1))

        self.last_conv = nn.Conv2d(n_classes*2, n_classes, kernel_size=1, bias=False)

        self.size = size
        self.criterion = criterion
        self.recurrence = recurrence
        self.use_rcca = use_rcca                #是否用CCA：False，use_rcca

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

    # def forward(self, x, x_ex, labels=None):
    def forward(self, x, x_ex, labels):
        # x                  8x2048x29x29
        # low_level_features 8x256x57x57
        # x_dsn              8x5x29x29
        x, x_dsn, low_level_features = self.backbone(x, x_ex, labels)
        x_r = x

        # x                  8x2048x57x57
        x = F.interpolate(x, size=low_level_features.size()[2:], mode='bilinear', align_corners=True)

        ################################## MSFE########################
        # x_aspp             8x1024x57x57
        x_aspp = self.aspp(x)
        aspp_params = sum(p.numel() for p in self.aspp.parameters())
        # print(f"aspp模块的参数量: {aspp_params}")
        # x_                 8x256x57x57
        x_ = self.global_avg_pool(x)
        x_ = F.interpolate(x_, size=x.size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat((x_aspp, x_), dim=1)

        # x                  8x256x57x57
        x = self.conv(x)
        x = self.bn(x)

        ################################## low_level_features########################
        # 8x48x57x57
        low_level_features = self.conv2(low_level_features)
        low_level_features = self.bn2(low_level_features)

        ##################################MSFE与low_level_features融合#################
        x = torch.cat((x, low_level_features), dim=1)

        # x                  8x5x57x57
        x = self.conv3(x)

        ##################################GFDM########################
        # x_rcca             8x5x29x29
        if self.use_rcca:
            x_rcca = self.rcca(x_r)
            rcca_params = sum(p.numel() for p in self.rcca.parameters())
            # print(f"rcca模块的参数量: {rcca_params}")

            x_rcca = F.interpolate(x_rcca, size=x.size()[2:], mode='bilinear', align_corners=True)
            x = torch.cat((x, x_rcca), dim=1)
            x = self.last_conv(x)

        x_dsn = F.interpolate(x_dsn, size=x.size()[2:], mode='bilinear', align_corners=True)
        outs = [x, x_dsn]

        if self.criterion is not None and labels is not None:
            return self.criterion(outs, labels), predict_whole(outs, self.size)
        else:
            return predict_whole(outs, self.size)


def Seg_Model(in_channel, ex_channels, num_classes, size, criterion=None, recurrence=0, use_rcca=True, **kwargs):
    model = SFA_DFNet(Bottleneck, size=size, in_channels=in_channel, ex_channels=ex_channels, n_classes=num_classes, criterion=criterion, recurrence=recurrence, use_rcca=use_rcca)
    return model