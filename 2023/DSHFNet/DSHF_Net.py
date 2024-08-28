# -*- coding:utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F



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
            nn.LeakyReLU(),  # No effect on order
            # nn.MaxPool2d(2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(l2, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),  # No effect on order
            # nn.MaxPool2d(2),
        )
        self.conv1_1 = nn.Sequential(
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),  # No effect on order
            nn.MaxPool2d(2),
            nn.Dropout(0.5),
        )
        self.conv2_1 = nn.Sequential(
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),  # No effect on order
            nn.MaxPool2d(2),
            nn.Dropout(0.5),
        )
        self.conv1_2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),  # No effect on order
            nn.MaxPool2d(2),
            nn.Dropout(0.5),
        )
        self.conv2_2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),  # No effect on order
            nn.MaxPool2d(2),
            nn.Dropout(0.5),
        )
        self.conv1_3 = nn.Sequential(
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),  # No effect on order
            nn.MaxPool2d(2),
            nn.Dropout(0.5),
        )
        self.conv2_3 = nn.Sequential(
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),  # No effect on order
            nn.MaxPool2d(2),
            nn.Dropout(0.5),
        )


    def forward(self, x11, x21, x12, x22, x13, x23):

        x11 = self.conv1(x11)
        x21 = self.conv2(x21)
        x12 = self.conv1(x12)
        x22 = self.conv2(x22)
        x13 = self.conv1(x13)
        x23 = self.conv2(x23)

        x1_1 = self.conv1_1(x11)
        x2_1 = self.conv1_1(x21)

        x1_2 = self.conv1_2(x12)
        x2_2 = self.conv2_2(x22)


        x1_3 = self.conv1_3(x13)
        x2_3 = self.conv2_3(x23)


        return x1_1, x2_1, x1_2, x2_2, x1_3, x2_3





class CNN_Classifier(nn.Module):
    def __init__(self, Classes):
        super(CNN_Classifier, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(64, 32, 1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, Classes, 1),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)

        x_out = F.softmax(x, dim=1)

        return x_out




class DSFE(nn.Module):
    def __init__(self, kernel_size=7):
        super(DSFE, self).__init__()
        assert kernel_size in (3, 7), "kernel size must be 3 or 7"
        padding = 3 if kernel_size == 7 else 1
        self.conv2d = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3)
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.leakyReLU = nn.LeakyReLU()
        self.dconv3 = nn.Sequential(
            nn.Upsample(scale_factor=2),  # add Upsample
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
# No effect on order

        )
        self.dconv4 = nn.Sequential(
            nn.Upsample(scale_factor=2),  # add Upsample
            nn.Conv2d(64, 64, 6, 1, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),  # No effect on order

        )



        self.dconv5 = nn.Sequential(
            nn.Upsample(scale_factor=2),  # add Upsample
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),  # No effect on order

        )
        self.dconv6 = nn.Sequential(
            nn.Upsample(scale_factor=2),  # add Upsample
            nn.Conv2d(64, 64, 6, 1, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),  # No effect on order

        )
        #
        self.xishu1 = torch.nn.Parameter(torch.Tensor([0.5]))  # lamda
        self.xishu2 = torch.nn.Parameter(torch.Tensor([0.5]))  # 1 - lamda
        self.xishu3 = torch.nn.Parameter(torch.Tensor([0.5]))  # lamda
        self.xishu4 = torch.nn.Parameter(torch.Tensor([0.5]))  # 1 - lamda
        self.xishu5 = torch.nn.Parameter(torch.Tensor([0.5]))  # lamda
        self.xishu6 = torch.nn.Parameter(torch.Tensor([0.5]))  # 1 - lamda
        self.encoder_embedding1 = nn.Linear(((6 // 2) * 1) ** 2, 6 ** 2)
        self.encoder_embedding2 = nn.Linear(((6 // 2) * 2) ** 2, 6 ** 2)
        self.encoder_embedding3 = nn.Linear(((6 // 2) * 3) ** 2, 6 ** 2)
    def forward(self, x1_1, x2_1, x1_2, x2_2, x1_3, x2_3, dim):
        b3, c3, h13, w13 = x1_1.size()
        b2, c2, h12, w12 = x1_2.size()
        b, c, h1, w1 = x1_3.size()
        avgout = torch.mean(x1_1, dim=1, keepdim=True)
        maxout, _ = torch.max(x1_1, dim=1, keepdim=True)
        out1 = torch.cat([avgout, maxout], dim=1)
        out1 = self.sigmoid(self.conv2d(out1))
        z = torch.zeros(size=(b3, c3, h13, w13), dtype=torch.float32).cuda()
        beta = 0.3
        out11 = torch.where(out1.data >= beta, out1, z)
        y1_1 = torch.nonzero(out11==0)
        n1_1 = y1_1.shape[0]
        y1_1a = torch.nonzero(out11 != 100)
        n1_1a = y1_1a.shape[0]
        if n1_1/n1_1a <=0.7:
            x1 = torch.zeros(size=(b, c, h1, w1), dtype=torch.float32).cuda()
            x2 = torch.zeros(size=(b2, c2, h12, w12), dtype=torch.float32).cuda()
            x3 = x1_1
        else:
            x1_11 = self.dconv3(x1_1 * out1 * self.xishu3)
            x1_2 = x1_11 *(1-(n1_1/n1_1a))  + x1_2
            avgout = torch.mean(x1_2, dim=1, keepdim=True)
            maxout, _ = torch.max(x1_2, dim=1, keepdim=True)
            out2 = torch.cat([avgout, maxout], dim=1)
            out2 = self.sigmoid(self.conv2d(out2))
            z = torch.zeros(size=(b2, c2, h12, w12), dtype=torch.float32).cuda()
            beta = 0.3
            out22 = torch.where(out2.data >= beta, out2, z)
            y1_2 = torch.nonzero(out22 == 0)
            n1_2 = y1_2.shape[0]
            y1_2a = torch.nonzero(out22 != 100)
            n1_2a = y1_2a.shape[0]
            if n1_2 / n1_2a <= 0.7:
                x1 = torch.zeros(size=(b, c, h1, w1), dtype=torch.float32).cuda()
                x2 = x1_2
                x3 = x1_1 * out1 * self.xishu3
            else:
                x1_21 = self.dconv4(x1_2 * out2 * self.xishu4)
                x1_3 = x1_21 *(1-(n1_2 / n1_2a)) + x1_3
                avgout = torch.mean(x1_3, dim=1, keepdim=True)
                maxout, _ = torch.max(x1_3, dim=1, keepdim=True)
                out3 = torch.cat([avgout, maxout], dim=1)
                out3 = self.sigmoid(self.conv2d(out3))
                x1 = x1_3 * out3
                x2 = x1_2 * out2 * self.xishu4
                x3 = x1_1 * out1 * self.xishu3

        b3ll, c3ll, h13ll, w13ll = x2_1.size()
        b2ll, c2ll, h12ll, w12ll = x2_2.size()
        bll, cll, h1ll, w1ll = x2_3.size()
        avgout = torch.mean(x2_1, dim=1, keepdim=True)
        maxout, _ = torch.max(x2_1, dim=1, keepdim=True)
        out1l = torch.cat([avgout, maxout], dim=1)
        out1l = self.sigmoid(self.conv2d(out1l))
        z = torch.zeros(size=(b3ll, c3ll, h13ll, w13ll), dtype=torch.float32).cuda()
        beta = 0.7

        out11l = torch.where(out1l.data >= beta, out1, z)
        y1_1 = torch.nonzero(out11l==0)
        n1_1 = y1_1.shape[0]
        y1_1a = torch.nonzero(out11l != 100)
        n1_1a = y1_1a.shape[0]
        if n1_1/n1_1a <=0.3:
            x1l = torch.zeros(size=(bll, cll, h1ll, w1ll), dtype=torch.float32).cuda()
            x2l = torch.zeros(size=(b2ll, c2ll, h12ll, w12ll), dtype=torch.float32).cuda()
            x3l = x2_1
        else:
            x2_11 = self.dconv5(x2_1 * out1l * self.xishu5)
            x2_2 = x2_11 *(1-(n1_1/n1_1a))+ x2_2
            avgout = torch.mean(x2_2, dim=1, keepdim=True)
            maxout, _ = torch.max(x2_2, dim=1, keepdim=True)
            out2l = torch.cat([avgout, maxout], dim=1)
            out2l = self.sigmoid(self.conv2d(out2l))
            z = torch.zeros(size=(b2ll, c2ll, h12ll, w12ll), dtype=torch.float32).cuda()
            beta = 0.7
            out22l = torch.where(out2l.data >= beta, out2l, z)
            y1_2 = torch.nonzero(out22l == 0)
            n1_2 = y1_2.shape[0]
            y1_2a = torch.nonzero(out22l != 100)
            n1_2a = y1_2a.shape[0]
            if n1_2 / n1_2a <= 0.3:
                x1l = torch.zeros(size=(b, c, h1, w1), dtype=torch.float32).cuda()
                x2l = x2_2
                x3l = x2_1 * out1 * self.xishu5
            else:
                x2_21 = self.dconv6(x2_2 * out2l * self.xishu6)
                x2_3 = x2_21 *(1-(n1_2 / n1_2a))+ x2_3
                avgout = torch.mean(x2_3, dim=1, keepdim=True)
                maxout, _ = torch.max(x2_3, dim=1, keepdim=True)
                out3l = torch.cat([avgout, maxout], dim=1)
                out3l = self.leakyReLU(self.conv2d(out3l))
                x1l = x2_3 * out3l
                x2l = x2_2 * out2l * self.xishu6
                x3l = x2_1 * out1l * self.xishu5
        x_add1 = x1 * self.xishu1 + x1l * self.xishu2
        x_add2 = x2 * self.xishu1 + x2l * self.xishu2
        x_add3 = x3 * self.xishu1 + x3l * self.xishu2

        x_add1 = x_add1.flatten(2)
        x_add2 = x_add2.flatten(2)
        x_add3 = x_add3.flatten(2)

        x_add1 = self.encoder_embedding3(x_add1)
        x_add2 = self.encoder_embedding2(x_add2)
        x_add3 = self.encoder_embedding1(x_add3)

        p = int(x_add1.shape[2] ** .5)
        x1_add = x_add1.reshape((x_add1.shape[0], x_add1.shape[1], p, p))
        x2_add = x_add2.reshape((x_add2.shape[0], x_add2.shape[1], p, p))
        x3_add = x_add3.reshape((x_add3.shape[0], x_add3.shape[1], p, p))
        num1 = x1_add.shape[1] // dim
        num2 = x2_add.shape[1] // dim
        num3 = x3_add.shape[1] // dim
        x_out = torch.empty(x1_add.shape[0], dim, p, p).cuda()
        for i in range(dim):
            x1_tmp = x1_add[:, i * num1:(i + 1) * num1, :, :]
            x2_tmp = x2_add[:, i * num2:(i + 1) * num2, :, :]
            x3_tmp = x3_add[:, i * num3:(i + 1) * num3, :, :]
            x_tmp1 = torch.cat((x1_tmp, x2_tmp, x3_tmp), dim=1)
            avgout = torch.mean(x_tmp1, dim=1, keepdim=True)
            maxout, _ = torch.max(x_tmp1, dim=1, keepdim=True)
            x_tmp2 = torch.cat([avgout, maxout], dim=1)
            x_tmp3 = self.conv(x_tmp2)
            x_tmp4 = self.sigmoid(x_tmp3)
            x_out[:, i:i+1, :, :] = x_tmp4
        x_out = x_out.reshape(x_out.shape[0], dim, p*p)

        x_h1 = x1.flatten(2)
        x_h2 = x2.flatten(2)
        x_h3 = x3.flatten(2)

        x_h1 = self.encoder_embedding3(x_h1)
        x_h2 = self.encoder_embedding2(x_h2)
        x_h3 = self.encoder_embedding1(x_h3)

        ph = int(x_h1.shape[2] ** .5)
        x1_h = x_h1.reshape((x_h1.shape[0], x_h1.shape[1], ph, ph))
        x2_h = x_h2.reshape((x_h2.shape[0], x_h2.shape[1], ph, ph))
        x3_h = x_h3.reshape((x_h3.shape[0], x_h3.shape[1], ph, ph))
        num1h = x1_h.shape[1] // dim
        num2h = x2_h.shape[1] // dim
        num3h = x3_h.shape[1] // dim
        x_outh = torch.empty(x1_h.shape[0], dim, ph, ph).cuda()
        for i in range(dim):
            x1_tmph = x1_h[:, i * num1h:(i + 1) * num1h, :, :]
            x2_tmph = x2_h[:, i * num2h:(i + 1) * num2h, :, :]
            x3_tmph = x3_h[:, i * num3h:(i + 1) * num3h, :, :]
            x_tmp1h = torch.cat((x1_tmph, x2_tmph, x3_tmph), dim=1)
            avgouth = torch.mean(x_tmp1h, dim=1, keepdim=True)
            maxouth, _ = torch.max(x_tmp1h, dim=1, keepdim=True)
            x_tmp2h = torch.cat([avgouth, maxouth], dim=1)
            x_tmp3h = self.conv(x_tmp2h)
            x_tmp4h = self.sigmoid(x_tmp3h)
            x_outh[:, i:i+1, :, :] = x_tmp4h

        x_outh = x_outh.reshape(x_outh.shape[0], dim, ph*ph)

        x_l1 = x1l.flatten(2)
        x_l2 = x2l.flatten(2)
        x_l3 = x3l.flatten(2)

        x_l1 = self.encoder_embedding3(x_l1)
        x_l2 = self.encoder_embedding2(x_l2)
        x_l3 = self.encoder_embedding1(x_l3)

        pl = int(x_l1.shape[2] ** .5)
        x1_l = x_l1.reshape((x_l1.shape[0], x_l1.shape[1], pl, pl))
        x2_l = x_l2.reshape((x_l2.shape[0], x_l2.shape[1], pl, pl))
        x3_l = x_l3.reshape((x_l3.shape[0], x_l3.shape[1], pl, pl))
        num1l = x1_l.shape[1] // dim
        num2l = x2_l.shape[1] // dim
        num3l = x3_l.shape[1] // dim
        x_outl = torch.empty(x1_l.shape[0], dim, pl, pl).cuda()
        for i in range(dim):
            x1_tmpl = x1_l[:, i * num1l:(i + 1) * num1l, :, :]
            x2_tmpl = x2_l[:, i * num2l:(i + 1) * num2l, :, :]
            x3_tmpl = x3_l[:, i * num3l:(i + 1) * num3l, :, :]
            x_tmp1l = torch.cat((x1_tmpl, x2_tmpl, x3_tmpl), dim=1)
            avgoutl = torch.mean(x_tmp1l, dim=1, keepdim=True)
            maxoutl, _ = torch.max(x_tmp1l, dim=1, keepdim=True)
            x_tmp2l = torch.cat([avgoutl, maxoutl], dim=1)
            x_tmp3l = self.conv(x_tmp2l)
            x_tmp4l = self.sigmoid(x_tmp3l)
            x_outl[:, i:i + 1, :, :] = x_tmp4l
        x_outl = x_outl.reshape(x_outl.shape[0], dim, pl * pl)


        return x_out,x_outh,x_outl






class PAM_Module(nn.Module):
    """ Position attention module"""
    #Ref from SAGAN
    def __init__(self, in_dim):
        super(PAM_Module, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma1 = torch.nn.Parameter(torch.Tensor([0.5]))
        self.gamma2= torch.nn.Parameter(torch.Tensor([0.5]))
    def forward(self, x,xh1,xl1):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X (HxW) X (HxW)
        """
        batchsize, channle, PF = xh1.size()

        x = x.reshape(x.shape[0], self.chanel_in, int(PF** .5),int(PF** .5))
        xh = xh1.reshape(x.shape[0], self.chanel_in, int(PF** .5), int(PF** .5))
        xl = xl1.reshape(x.shape[0], self.chanel_in, int(PF** .5), int(PF** .5))
        x = x.squeeze(-1)
        xh = xh.squeeze(-1)
        xl = xl.squeeze(-1)

        m_batchsize, C, height, width = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)
        energy = torch.bmm(proj_query, proj_key)
        attention = energy.softmax(dim=-1)
        proj_valueh = self.value_conv(xh).view(m_batchsize, -1, width * height)
        proj_valuel = self.value_conv(xl).view(m_batchsize, -1, width * height)
        outh = torch.bmm(proj_valueh, attention.permute(0, 2, 1))
        outh = outh.view(m_batchsize, C, height*width)
        outl = torch.bmm(proj_valuel, attention.permute(0, 2, 1))
        outl = outl.view(m_batchsize, C, height*width)
        outh = torch.mul(xh1, self.gamma1*outh)
        outl = torch.mul(xl1, self.gamma2*outl)

        return outh, outl








class DSHF(nn.Module):
    def __init__(self, l1, l2, patch_size, num_patches, num_classes, encoder_embed_dim, decoder_embed_dim, en_depth, en_heads,
                 de_depth, de_heads, mlp_dim, dim_head=16, dropout=0., emb_dropout=0.):
        super().__init__()
        self.cnn_encoder = CNN_Encoder(l1, l2)
        self.cnn_classifier = CNN_Classifier(num_classes)
        self.dsfe = DSFE()
        self.encoder_embed_dim = encoder_embed_dim
        self.PAM = PAM_Module(64)
    def encoder(self, x11, x21, x12, x22, x13, x23):


        x1_1, x2_1, x1_2, x2_2, x1_3, x2_3,= self.cnn_encoder(x11, x21, x12, x22, x13, x23)

        x_cnn,x_h,x_l  = self.dsfe(x1_1, x2_1, x1_2, x2_2, x1_3, x2_3, self.encoder_embed_dim)
        return  x_cnn,x_h,x_l

    def classifier(self, x_cnn):

        x_cls1 = self.cnn_classifier(seq2img(x_cnn))
        return x_cls1


    def forward(self, img11, img21, img12, img22, img13, img23):

        x_cnn,x_h,x_l = self.encoder(img11, img21, img12, img22, img13, img23)
        x_attentionh, x_attentionl = self.PAM(x_cnn,x_h, x_l)
        x_cnn = x_cnn + x_attentionh + x_attentionl
        self.features = x_cnn
        x_cls = self.classifier(x_cnn)
        return x_cls



