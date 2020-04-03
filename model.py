import math
import random
import torch

import numpy as np
import torch.nn as nn
from torch.nn.utils import spectral_norm

import block as B
from pytorch_radon import Radon, IRadon
from mypackage.model.unet_standard import UNet

class Inpaint_Net(nn.Module):
    def __init__(self, in_nc=2, out_nc=1, nf=16, nb=8, gc=16, norm_type=None, act_type='leakyrelu', mode='CNA'):
        super(Inpaint_Net, self).__init__()
        fea_conv = B.conv_block(in_nc, nf, kernel_size=3, norm_type=None, act_type=None)
        rb_blocks = [B.RRDB(nf, kernel_size=3, gc=gc, stride=1, bias=True, pad_type='zero',
                            norm_type=norm_type, act_type=act_type, mode='CNA', dilation=(1, 2, 4), groups=1) for _ in
                     range(nb)]
        LR_conv = B.conv_block(nf, nf, kernel_size=3, norm_type=norm_type, act_type=None, mode=mode)
        HR_conv0 = B.conv_block(nf, nf, kernel_size=3, norm_type=None, act_type=act_type)
        HR_conv1 = B.conv_block(nf, out_nc, kernel_size=3, norm_type=None, act_type=None)

        self.model = B.sequential(fea_conv, B.ShortcutBlock(B.sequential(*rb_blocks, LR_conv))
                                  , HR_conv0, HR_conv1)
        # self.model = B.sequential(fea_conv, HR_conv0, HR_conv1)

    def forward(self, x1, x2):
        # x [0, 1]
        x = torch.cat([x1, x2], dim=1)
        x = self.model(x)
        return x


class NLD_LG_inpaint(nn.Module):
    def __init__(self, depth=64, input_nc=3, output_nc=1):
        super(NLD_LG_inpaint, self).__init__()
        self.norm = nn.GroupNorm
        group = 4
        # 256 x 128 => 128 x 128
        self.layer0 = nn.Sequential(
            spectral_norm(nn.Conv2d(input_nc, depth, kernel_size=3, stride=1, padding=1),
                          n_power_iterations=3),
            nn.MaxPool2d(2),
            self.norm(group, depth),
            nn.LeakyReLU())
        # 128 => 128
        self.layer1 = nn.Sequential(
            spectral_norm(nn.Conv2d(depth, depth * 2, kernel_size=3, stride=1, padding=2, dilation=2),
                          n_power_iterations=3),
            self.norm(group, depth * 2),
            nn.LeakyReLU())
        # 128 => 64
        self.layer2 = nn.Sequential(
            spectral_norm(nn.Conv2d(depth * 2, depth * 2, kernel_size=3, stride=1, padding=2, dilation=2),
                          n_power_iterations=3),
            nn.MaxPool2d(2),
            self.norm(group, depth * 2),
            nn.LeakyReLU())
        # 64 => 64
        self.layer3 = nn.Sequential(
            spectral_norm(nn.Conv2d(depth * 2, depth * 4, kernel_size=3, stride=1, padding=2, dilation=2),
                          n_power_iterations=3),
            # nn.MaxPool2d(2),
            self.norm(group, depth * 4),
            nn.LeakyReLU())
        # # 64 => 32
        # self.layer4 = nn.Sequential(
        #     spectral_norm(nn.Conv2d(depth * 4, depth * 8, kernel_size=4, stride=2, padding=1, dilation=1),
        #                   n_power_iterations=3),
        #     self.norm(group, depth * 8),
        #     nn.LeakyReLU())
        # ===============================================
        # 128 => 64
        # self.fast_layer1 = nn.Sequential(
        #     spectral_norm(nn.Conv2d(depth * 1, depth * 2, kernel_size=4, stride=2, padding=1, dilation=1),
        #                   n_power_iterations=3),
        #     nn.MaxPool2d(2),
        #     self.norm(group, depth * 2),
        #     nn.LeakyReLU())
        # # 64 = > 32
        # self.fast_layer2 = nn.Sequential(
        #     spectral_norm(nn.Conv2d(depth * 2, depth * 4, kernel_size=4, stride=2, padding=1, dilation=1),
        #                   n_power_iterations=3),
        #     nn.MaxPool2d(2),
        #     self.norm(group, depth * 4),
        #     nn.LeakyReLU())
        # ===============================================
        # 32 => 32
        self.final_layer = nn.Sequential(
            spectral_norm(nn.Conv2d(depth * 4, output_nc, kernel_size=3, stride=1, padding=1, dilation=1),
                          n_power_iterations=3))

    def forward(self, input, x, g):
        out = torch.cat((input, x, g), 1)
        # print(x.size())  # torch.Size([3, 1, 256, 180])
        out = self.layer0(out)  # 256 x 128 => 128 x 128
        # print(out.size())  # torch.Size([3, 32, 128, 180])
        # fast_out = self.fast_layer1(out)  # 128 => 64
        # print("fast_out_layer1", fast_out.size())
        # fast_out = self.fast_layer2(fast_out)  # 64 => 32
        # print("fast_out_layer2", fast_out.size())
        out = self.layer1(out)  # 128 => 128
        # print(out.size())  # torch.Size([3, 64, 62, 88])
        out = self.layer2(out)  # 128 => 64
        # print(out.size())  # torch.Size([3, 64, 26, 39])
        out = self.layer3(out)  # 64 => 64
        # print(out.size())  # torch.Size([3, 128, 8, 11])
        # out = self.layer4(out)  # 64 => 32
        # print(out.size())
        # out = self.final_layer(torch.cat((out, fast_out), 1))  # 32 => 32
        out = self.final_layer(out)  # 32 => 32
        # print(out.size())
        return out


class Due_Generator(nn.Module):
    def __init__(self, angles, img_size=128, in_nc=2, out_nc=1, nf=16, nb=16, gc=16, norm_type=None,
                 act_type='leakyrelu',
                 mode='CNA'):
        super(Due_Generator, self).__init__()
        # self.de_recon = Inpaint_Net(in_nc, out_nc, nf, nb, gc, norm_type, act_type, mode)
        self.de_recon = UNet(2)
        self.de_sino = Inpaint_Net(in_nc, out_nc, nf, nb, gc, norm_type, act_type, mode)
        self.angles = angles
        self.iradon_trans = IRadon(img_size, self.angles)
        self.radon_trans = Radon(img_size, self.angles)

    def forward(self, recon, sinogram, iter=3):
        dn_recon = self.de_recon(recon, recon)
        dn_sinogram = self.de_sino(sinogram, sinogram)
        for _ in range(iter - 1):
            dn_recon2 = self.iradon_trans(dn_sinogram * 255)
            dn_sinogram2 = self.radon_trans(dn_recon) / 255
            dn_recon = self.de_recon(dn_recon, dn_recon2)
            dn_sinogram = self.de_sino(dn_sinogram, dn_sinogram2)

        return dn_recon, dn_sinogram


class Due_Discriminator(nn.Module):
    def __init__(self, angles, img_size=128, depth=1):
        super(Due_Discriminator, self).__init__()
        self.d_recon = NLD_LG_inpaint(depth)
        self.d_sino = NLD_LG_inpaint(depth)
        # self.final = nn.Sequential(nn.Linear(128, 1), nn.Sigmoid())
        self.angles = angles
        self.iradon_trans = IRadon(img_size, self.angles)
        self.radon_trans = Radon(img_size, self.angles)

    def forward(self, in_recon, in_sinogram, recon, sinogram):
        recon2 = self.iradon_trans(sinogram * 255)
        sinogram2 = self.radon_trans(recon) / 255

        dn_recon = self.d_recon(in_recon, recon, recon2)
        dn_sinogram = self.d_sino(in_sinogram, sinogram, sinogram2)
        return dn_recon, dn_sinogram


def get_batch(delta, gap_degree, input_size=(1, 1, 128, 128), img_size=128, mask_deg=50):
    dense_angles = np.arange(0., 180., delta)
    total_len = len(dense_angles)
    mask_start = math.floor((90 - gap_degree / 2.) / 180 * total_len)
    mask_end = math.ceil((90 + gap_degree / 2.) / 180 * total_len)
    dense_angles[mask_start:mask_end] = 0
    real_recon = torch.zeros(*input_size)
    a = random.randint(20, 50)
    b = random.randint(50, 90)
    c = random.randint(20, 50)
    d = random.randint(50, 90)
    real_recon[:, :, a:b, c:d] = 1

    real_sino = Radon(img_size, dense_angles)(real_recon) / 255  # 96
    in_sino = Radon(img_size, dense_angles)(real_recon) / 255  # 96
    in_sino[:, :, :, 32:96] = 0  # 128 - 64  # 48,80
    in_reon = IRadon(img_size, dense_angles)(in_sino * 255)
    in_sino = in_sino.cuda(1)
    in_reon = in_reon.cuda(1)
    real_sino = real_sino.cuda(1)
    real_recon = real_recon.cuda(1)
    return in_sino, in_reon, real_sino, real_recon


if __name__ == '__main__':
    import torch

    delta = 3
    dense_angles = np.arange(0., 180., delta)

    g = Due_Generator(dense_angles, img_size=256, nf=4, nb=2, gc=4)
    g = g.cuda(1)

    d = Due_Discriminator(dense_angles, depth=4, img_size=256)
    d = d.cuda(1)

    in_sino, in_reon, real_sino, real_recon = get_batch(delta, 45, input_size=(1, 1, 256, 256), img_size=256)
    fake_recon, fake_sino = g(in_reon, in_sino)
    print(fake_recon.size())
    print(fake_sino.size())
    dfake_recon, dfake_sino = d(in_reon, in_sino, fake_recon, fake_sino)

    print(dfake_recon.size())
    print(dfake_sino.size())
