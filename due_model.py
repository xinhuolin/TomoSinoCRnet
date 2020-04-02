import random

import numpy as np
import torch.nn as nn
from torch.nn.functional import mse_loss
from torch.nn.init import kaiming_uniform_, constant_
from torch.nn.utils import spectral_norm
from torch.optim import lr_scheduler
from torch.utils.tensorboard import SummaryWriter

import block as B
from loss_func import get_adversarial_losses_fn
from pytorch_radon import Radon, IRadon


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
                                  , HR_conv0, HR_conv1, nn.Sigmoid())
        # self.model = B.sequential(fea_conv, HR_conv0, HR_conv1)

    def forward(self, x1, x2):
        # x [0, 1]
        x = torch.cat([x1, x2], dim=1)
        x = self.model(x)
        return x


class Due_Net(nn.Module):
    def __init__(self, angles, img_size=128, in_nc=1, out_nc=1, nf=16, nb=16, gc=16, norm_type=None,
                 act_type='leakyrelu',
                 mode='CNA'):
        super(Due_Net, self).__init__()
        self.de_recon = Inpaint_Net()
        self.de_sino = Inpaint_Net()
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


# Discriminator
class NLD(nn.Module):
    def __init__(self, depth=16, input_nc=3):
        super(NLD, self).__init__()
        self.norm = nn.GroupNorm
        group = 4
        # 128
        self.layer1 = nn.Sequential(
            spectral_norm(nn.Conv2d(input_nc, depth, kernel_size=5, stride=1, padding=2, dilation=1),
                          n_power_iterations=3),
            self.norm(group, depth),
            nn.MaxPool2d(2),
            nn.LeakyReLU())
        # 64
        self.layer2 = nn.Sequential(
            spectral_norm(nn.Conv2d(depth, depth * 2, kernel_size=3, stride=1, padding=2, dilation=2),
                          n_power_iterations=3),
            self.norm(group, depth * 2),
            nn.MaxPool2d(2),
            nn.LeakyReLU())
        # 32
        self.layer3 = nn.Sequential(
            spectral_norm(nn.Conv2d(depth * 2, depth * 4, kernel_size=3, stride=1, padding=2, dilation=2),
                          n_power_iterations=3),
            self.norm(group, depth * 4),
            nn.MaxPool2d(2),
            nn.LeakyReLU())
        # 16
        self.layer4 = nn.Sequential(
            spectral_norm(nn.Conv2d(depth * 4, 1, kernel_size=5, stride=1, padding=2, dilation=1),
                          n_power_iterations=3),
            # nn.Sigmoid()
        )
        # # 16 x 8
        # self.layer5 = nn.Sequential(
        #     nn.Conv2d(depth * 4, depth * 8, kernel_size=4, stride=2, padding=1, dilation=1),
        #     self.norm(group, depth * 8),
        #     nn.LeakyReLU())
        # # 8 x 4
        # self.layer6 = nn.Sequential(
        #     nn.Conv2d(depth * 8, output_nc, kernel_size=(8, 4), stride=1, padding=0, dilation=1))

    def forward(self, in_x, x, x2):
        out = torch.cat([in_x, x, x2], 1)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        # out = self.layer5(out)
        # out = self.layer6(out)
        return out


class Due_Discriminator(nn.Module):
    def __init__(self, angles, img_size=128, depth=1):
        super(Due_Discriminator, self).__init__()
        self.d_recon = NLD(depth)
        self.d_sino = NLD(depth)
        self.final = nn.Sequential(nn.Linear(128, 1), nn.Sigmoid())
        self.angles = angles
        self.iradon_trans = IRadon(img_size, self.angles)
        self.radon_trans = Radon(img_size, self.angles)

    def forward(self, in_recon, in_sinogram, recon, sinogram):
        recon2 = self.iradon_trans(sinogram * 255)
        sinogram2 = self.radon_trans(recon) / 255

        dn_recon = self.d_recon(in_recon, recon, recon2)
        dn_sinogram = self.d_sino(in_sinogram, sinogram, sinogram2)

        # dn_recon = torch.flatten(dn_recon, start_dim=1)
        # dn_sinogram = torch.flatten(dn_sinogram, start_dim=1)

        # final = self.final(torch.cat([dn_sinogram, dn_recon], 1))
        return dn_recon, dn_sinogram


def normlize(img_np_256):
    return (img_np_256 - img_np_256.min()) / (img_np_256.max() - img_np_256.min() + 1e-8)


def lsgan_loss(d_real, d_fake, t):
    if t == "d":
        loss = 0.5 * mse_loss(d_real - d_fake.mean(), torch.ones_like(d_real)) + mse_loss(d_fake - d_real.mean(),
                                                                                          -torch.ones_like(d_real))
    else:
        loss = 0.5 * mse_loss(d_real - d_fake.mean(), -torch.ones_like(d_real)) + mse_loss(d_fake - d_real.mean(),
                                                                                           torch.ones_like(d_real))
    return loss


def get_batch(input_size =(1, 1, 128, 128), img_size = 128):
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


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        kaiming_uniform_(m.weight)
        constant_(m.bias, 0)


if __name__ == '__main__':
    import torch
    from tqdm import tqdm

    # m = RRDB_Net(1, 1, 8, 8, 16, 2, upsample_mode="horpixelshuffle")
    # x = torch.ones(1, 1, 4, 4)
    # y = m(x)
    # dense_angles = np.arange(0., 180., 1.40625)  # 128
    dense_angles = np.arange(0., 180., 3)
    # angles = np.concatenate((np.arange(0, 67.5, 1.40625), np.arange(112.5, 180, 1.40625)))  # 96

    g = Due_Net(dense_angles, img_size=256, nf=8, nb=2, gc=8)
    g = g.cuda(1)
    g.apply(weights_init)

    d = Due_Discriminator(dense_angles, depth=8)
    d = d.cuda(1)
    d.apply(weights_init)

    opt_g = torch.optim.Adam(g.parameters(), 1e-4, weight_decay=3e-4)
    lrs_g = lr_scheduler.MultiStepLR(opt_g, [2000], 0.1)
    opt_d = torch.optim.RMSprop(d.parameters(), 1e-4, weight_decay=3e-4)
    lrs_d = lr_scheduler.MultiStepLR(opt_d, [2000], 0.1)

    d_loss_fn, g_loss_fn = get_adversarial_losses_fn("lsgan")
    # real_recon = torch.zeros(2, 1, 128, 128)
    # real_recon[:, :, 50:80, 50:80] = 1
    # real_sino = Radon(128, dense_angles)(real_recon) / 255  # 96
    # # plt.imshow(real_recon.detach()[0][0], cmap="gray")
    # # plt.show()
    # # plt.imshow(real_sino.detach()[0][0], cmap="gray")
    # # plt.show()
    #
    # in_sino = Radon(128, dense_angles)(real_recon)  # 96
    # in_sino[:, :, :, 48:80] = 0
    # in_reon = IRadon(128, dense_angles)(in_sino)
    # in_sino = in_sino.cuda(1)
    # in_reon = in_reon.cuda(1)
    # real_sino = real_sino.cuda(1)
    # real_recon = real_recon.cuda(1)

    # plt.imshow(in_reon.detach()[0][0] , cmap="gray")
    # plt.show()
    # plt.imshow(in_sino.detach()[0][0] , cmap="gray")
    # plt.show()
    # in_reon.requires_grad=True
    # in_sino.requires_grad=True
    # print(gradcheck(m, (in_reon, in_sino), raise_exception=False, eps=1e-6))
    # plt.imshow(real_recon.detach().cpu()[0][0], cmap="gray")
    # plt.show()
    # plt.imshow(real_sino.detach().cpu()[0][0], cmap="gray")
    # plt.show()
    # writer = SummaryWriter("log/iter")
    in_sino, in_reon, real_sino, real_recon = get_batch(input_size=(1,1,256,256), img_size=256)
    # writer.add_graph(g, [in_reon, in_sino])
    fake_recon, fake_sino = g(in_reon, in_sino)
    # dfake_recon, dfake_sino = d(in_reon, in_sino, fake_recon, fake_sino)
    print(fake_recon.size())
    print(fake_sino.size())
    # print(dfake_recon.size())
    # print(dfake_sino.size())
    exit(1)
    for i in tqdm(range(20000)):
        in_sino, in_reon, real_sino, real_recon = get_batch()

        opt_g.zero_grad()
        fake_recon, fake_sino = g(in_reon, in_sino)
        dfake_recon, dfake_sino = d(in_reon, in_sino, fake_recon, fake_sino)
        # dreal_recon, dreal_sino = d(real_recon, real_sino)
        mse = mse_loss(fake_recon, real_recon)+mse_loss(fake_sino, real_sino)
        loss_g = (g_loss_fn(dfake_recon) + g_loss_fn(dfake_sino))*0.08 + mse
        # loss_g = mse_loss(fake_recon, real_recon) + mse_loss(fake_sino, real_sino)
        loss_g.backward()
        opt_g.step()

        opt_d.zero_grad()
        fake_recon, fake_sino = g(in_reon, in_sino)
        dfake_recon, dfake_sino = d(in_reon, in_sino, fake_recon.detach(), fake_sino.detach())
        dreal_recon, dreal_sino = d(in_reon, in_sino, real_recon, real_sino)
        loss_d = d_loss_fn(dreal_recon, dfake_recon) + d_loss_fn(dreal_sino, dfake_sino)
        loss_d.backward()
        opt_d.step()

        lrs_g.step(i)
        lrs_d.step(i)

        writer.add_scalar("loss_d/loss_d", loss_d, i)
        writer.add_scalar("loss_g/loss_g", loss_g, i)
        writer.add_scalar("loss_g/mse", mse, i)

        writer.add_scalar("lr", opt_d.param_groups[0]['lr'], i)

        if i % 10 == 0:
            writer.add_image("train/in_reon", normlize(in_reon.detach().cpu()[0]), i)
            writer.add_image("train/in_sino", normlize(in_sino.detach().cpu()[0]), i)

            writer.add_image("train/out_recon", normlize(fake_recon.detach().cpu()[0]), i)
            writer.add_image("train/out_sinogram", normlize(fake_sino.detach().cpu()[0]), i)

            writer.add_image("gt/real_recon", normlize(real_recon.detach().cpu()[0]), i)
            writer.add_image("gt/real_sino", normlize(real_sino.detach().cpu()[0]), i)
