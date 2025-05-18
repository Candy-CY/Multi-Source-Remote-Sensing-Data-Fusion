import torch
from torch import nn
from einops import rearrange


class LayerNorm(nn.Module):
    def __init__(self, dim):
        super(LayerNorm, self).__init__()
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        x = rearrange(x, 'b c h w -> b (h w) c')
        x = self.norm(x)
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
        return x


class SCFA(nn.Module):
    def __init__(self, dim, num_heads, bias=True, dropout=0.1):
        super(SCFA, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.hsi_qv = nn.Conv2d(dim, dim * 2, kernel_size=1, bias=bias)
        self.lidar_kv = nn.Conv2d(dim, dim * 2, kernel_size=1, bias=bias)
        self.hsi_dw_conv = nn.Conv2d(
            dim * 2, dim * 2, kernel_size=3, stride=1, padding=1, groups=dim * 2, bias=bias)
        self.lidar_dw_conv = nn.Conv2d(
            dim * 2, dim * 2, kernel_size=3, stride=1, padding=1, groups=dim * 2, bias=bias)

        self.dropout = nn.Dropout(dropout)
        # self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, hsi, lidar):
        assert hsi.shape[-2:] == lidar.shape[-2:], "H,W of hsi and lidar must be the same."

        b, c, h, w = hsi.shape

        hsi_qv = self.hsi_qv(hsi)
        hsi_qv = self.hsi_dw_conv(hsi_qv)
        hsi_q, hsi_v = hsi_qv.chunk(2, dim=1)

        lidar_kv = self.lidar_kv(lidar)
        lidar_kv = self.lidar_dw_conv(lidar_kv)
        lidar_k, lidar_v = lidar_kv.chunk(2, dim=1)

        hsi_q = rearrange(hsi_q, 'b (head c) h w -> b head c (h w)',
                          head=self.num_heads)
        lidar_k = rearrange(lidar_k, 'b (head c) h w -> b head c (h w)',
                            head=self.num_heads)
        hsi_v = rearrange(hsi_v, 'b (head c) h w -> b head c (h w)',
                          head=self.num_heads)
        lidar_v = rearrange(lidar_v, 'b (head c) h w -> b head c (h w)',
                            head=self.num_heads)

        q = torch.nn.functional.normalize(hsi_q, dim=-1)
        k = torch.nn.functional.normalize(lidar_k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)

        hsi_out = (attn @ hsi_v ) + hsi_v
        lidar_out = (attn @ lidar_v) + lidar_v

        hsi_out = rearrange(hsi_out, 'b head c (h w) -> b (head c) h w',
                            head=self.num_heads, h=h, w=w)+ hsi
        lidar_out = rearrange(lidar_out, 'b head c (h w) -> b (head c) h w',
                              head=self.num_heads, h=h, w=w) + lidar
        return hsi_out, lidar_out

# class FeedForward(nn.Module):
#     def __init__(self, dim, ffn_expansion_factor, dropout=0.):
#         super().__init__()
#         hidden_dim = int(dim * ffn_expansion_factor)
#         self.net = nn.Sequential(
#             nn.Conv2d(dim, hidden_dim, kernel_size=1),
#             nn.GELU(),
#             nn.Dropout(dropout),
#             nn.Conv2d(hidden_dim, dim, kernel_size=1),
#             nn.Dropout(dropout)
#         )
#
#     def forward(self, x):
#         return self.net(x)


class SCFTransBlock(nn.Module):
    def __init__(self, dim=64, num_heads=8, ffn_expansion_factor=2):
        super(SCFTransBlock, self).__init__()

        self.norm1_1 = LayerNorm(dim)
        self.norm1_2 = LayerNorm(dim)
        self.attn = SCFA(dim, num_heads)
        self.project_out = nn.Conv2d(dim * 2, dim, kernel_size=1)

        # self.norm2 = LayerNorm(dim)
        # self.ffn = FeedForward(dim, ffn_expansion_factor)
    def forward(self, x, y):
        x, y = self.attn(self.norm1_1(x), self.norm1_2(y))
        attn = torch.cat((x, y), dim=1)
        attn = self.project_out(attn)
        # out = attn + self.ffn(self.norm2(attn))
        return attn


class LGM(nn.Module):
    def __init__(self, in_chan=64, out_chan=64, num_heads=8, dropout=0.1):
        super(LGM, self).__init__()

        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(self.num_heads, 1, 1))

        self.layer_norm = LayerNorm(in_chan)

        self.conv1 = nn.Conv2d(in_chan, out_chan, kernel_size=1)
        self.conv2 = nn.Conv2d(in_chan, out_chan, kernel_size=1)
        self.conv3 = nn.Conv2d(in_chan, out_chan, kernel_size=1)

        self.conv4 = nn.Sequential(nn.Conv2d(in_chan, out_chan, kernel_size=1),
                                   nn.BatchNorm2d(out_chan),
                                   nn.GELU(), )
        self.dw_conv = nn.Sequential(nn.Conv2d(out_chan, out_chan, kernel_size=3, padding=1, groups=out_chan),
                                     nn.BatchNorm2d(out_chan),
                                     nn.GELU(),)

        self.dw_conv_local = nn.Sequential(nn.Conv2d(out_chan, out_chan, kernel_size=1, groups=out_chan),
                                     nn.BatchNorm2d(out_chan),
                                     nn.GELU(), )
        self.dw_conv_global = nn.Sequential(nn.Conv2d(out_chan, out_chan, kernel_size=1, groups=out_chan),
                                     nn.BatchNorm2d(out_chan),
                                     nn.GELU(), )
        self.conv5 = nn.Sequential(nn.Conv2d(out_chan, out_chan, kernel_size=1),
                                   nn.BatchNorm2d(out_chan),
                                   nn.GELU(), )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        b, c, h, w = x.shape
        x_res = x
        # x = self.layer_norm(x)
        q, k, v = self.conv1(x), self.conv2(x), self.conv3(x)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q.transpose(-2, -1) @ k) * self.temperature
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)

        attn_out = (v @ attn) + v
        attn_out = rearrange(attn_out, 'b head c (h w) -> b (head c) h w',
                            head=self.num_heads, h=h, w=w)
        attn_out = attn_out + x_res

        conv_out1 =self.conv4(x)
        conv_out = self.dw_conv(conv_out1) + conv_out1

        conv_out = self.dw_conv_local(conv_out)
        attn_out = self.dw_conv_global(attn_out)
        out = self.conv5(conv_out+attn_out)
        return out


class Proposed(nn.Module):
    def __init__(self, dataset_name):
        super(Proposed, self).__init__()

        if dataset_name == 'Houston2013':
            hsi_dim = 144
            lidar_dim = 1
            num_classes = 15
        elif dataset_name == 'MUUFL':
            hsi_dim = 64
            lidar_dim = 2
            num_classes = 11
        elif dataset_name == 'Trento':
            hsi_dim = 63
            lidar_dim = 1
            num_classes = 6
        elif dataset_name == 'Berlin':
            hsi_dim = 244
            lidar_dim = 4
            num_classes = 8
        elif dataset_name == 'Augsburg':
            hsi_dim = 180
            lidar_dim = 4
            num_classes = 7
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}. Dataset does not exist.")

        self.conv3d_hsi = nn.Sequential(
            nn.Conv3d(in_channels=1, out_channels=8, kernel_size=(5, 3, 3), padding=(2, 1, 1)),
            nn.BatchNorm3d(8),
            nn.GELU(),
        )

        self.conv2d_hsi = nn.Sequential(
            nn.Conv2d(in_channels=8*hsi_dim, out_channels=64, kernel_size=(3, 3)),
            nn.BatchNorm2d(64),
            nn.GELU(),
        )

        self.conv2d_lidar = nn.Sequential(
            nn.Conv2d(lidar_dim, out_channels=64, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(64),
            nn.GELU(),
        )
        self.conv2d_lidar2 = nn.Sequential(
            nn.Conv2d(64, out_channels=64, kernel_size=(3, 3),),
            nn.BatchNorm2d(64),
            nn.GELU(),
        )

        self.lgt_hsi1 = LGM()
        self.lgt_hsi2 = LGM()
        self.lgt_lidar1 = LGM()
        self.lgt_lidar2 = LGM()

        self.scfa1 = SCFTransBlock()
        self.scfa2 = SCFTransBlock()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x1, x2):

        x1 = x1.unsqueeze(1)
        x1 = self.conv3d_hsi(x1)
        x1 = rearrange(x1, 'b c h w y ->b (c h) w y')
        x1 = self.conv2d_hsi(x1)

        x2 = self.conv2d_lidar(x2)
        x2 = self.conv2d_lidar2(x2)

        x1 = self.lgt_hsi1(x1)
        x2 = self.lgt_lidar1(x2)
        out1 = self.scfa1(x1, x2)
        x1 = x1+out1
        x2 = x2+out1

        x1 = self.lgt_hsi2(x1)
        x2 = self.lgt_lidar2(x2)
        out2 = self.scfa2(x1, x2)
        x1 = x1+out2
        x2 = x2+out2

        x1, x2 = self.avg_pool(x1), self.avg_pool(x2)
        out = torch.cat((x1, x2), dim=1)
        out = self.fc(out.squeeze())
        return out

if __name__ == '__main__':

    model = Proposed('Houston2013')
    model.eval()
    x1, x2 = torch.rand((1, 144, 11, 11)), torch.rand((1, 1, 11, 11))
    y = model(x1, x2)
    print(y.shape)

# class CrossAttention(nn.Module):
#     def __init__(self, dim=64, num_heads=8):
#         super(CrossAttention, self).__init__()
#
#         self.linear_q = nn.Conv2d(dim, dim, 1)
#         self.linear_k = nn.Conv2d(dim, dim, 1)
#         self.linear_v = nn.Conv2d(dim, dim, 1)
#
#         self.scale = np.power(dim, 0.5)
#
#     def forward(self, hsi, lidar):
#         assert hsi.shape[-2:] == lidar.shape[-2:], "H,W of hsi and lidar must be the same."
#         b, c, h, w = hsi.shape
#
#         q = self.linear_q(hsi)
#         k = self.linear_k(hsi)
#         v = self.linear_v(lidar)
#
#         q = rearrange(q, 'b c h w -> b c (h w)')
#         k = rearrange(k, 'b c h w -> b (h w) c')
#         v = rearrange(v, 'b c h w -> b c (h w)')
#
#         attn = (q @ k) / self.scale
#         attn = attn.softmax(-1)
#
#         out = attn @ v
#         out = rearrange(out, 'b c (h w) -> b c h w',h=h,w=w)
#         return out

# class MultipleCrossAttention(nn.Module):
#     def __init__(self, dim=64):
#         super(MultipleCrossAttention, self).__init__()
#
#         self.cross_1 = CrossAttention()
#         self.cross_2 = CrossAttention()
#         self.cross_3 = CrossAttention()
#
#     def forward(self, x, y):
#         out1 = self.cross_1(x, y)
#         out2 = self.cross_2(y, x)
#         out = self.cross_3(out1, out2)
#         return out