import os
import shutil

import numpy as np
import torch
from jdit import Optimizer, Model
from jdit.trainer import Pix2pixGanTrainer
from skimage.transform import iradon
from skimage.restoration import denoise_tv_bregman
from torch.nn.functional import mse_loss

from dataset import RadonInnerInpaintDatasets
from model import Inpaint_Net, NLD_LG_inpaint
from tools.ssim import ms_ssim, get_SNR
from torch.nn.utils import spectral_norm

# pytorch.set_default_tensor_type('torch.DoubleTensor')
class RadonInpaintPix2pixGanTrainer(Pix2pixGanTrainer):
    def __init__(self, logdir, nepochs, gpu_ids_abs, netG, netD, optG, optD, datasets):
        super(RadonInpaintPix2pixGanTrainer, self).__init__(logdir, nepochs, gpu_ids_abs, netG, netD, optG, optD,
                                                            datasets)

        self.mask = torch.ones((batch_size, 1, 256, 128), requires_grad=False) * 0.01
        self.mask[:, :, :, 48:80] = 1
        self.mask = self.mask.to(self.device)

    def compute_d_loss(self):
        d_fake = self.netD(self.fake.detach(), self.input)
        d_real = self.netD(self.ground_truth, self.input)
        var_dic = {}
        var_dic["LOSS_D/RaLS_GAN"] = loss_d = (torch.mean((d_real - d_fake - 1) ** 2) +
                                               torch.mean((d_fake - d_real + 1) ** 2)) / 2
        return loss_d, var_dic

    def compute_g_loss(self):
        var_dic = {}
        d_fake = self.netD(self.fake, self.input)
        d_real = self.netD(self.ground_truth, self.input)
        var_dic["LOSS_G/MSE"] = mse_loss(self.fake * self.mask, self.ground_truth * self.mask)
        var_dic["LOSS_G/GAN"] = (torch.mean((d_real - d_fake + 1) ** 2) + torch.mean((d_fake - d_real - 1) ** 2))
        var_dic["LOSS_G/RALS_GAN"] = loss_g = var_dic["LOSS_G/MSE"] + var_dic["LOSS_G/GAN"] * 0.004
        return loss_g, var_dic

    def compute_valid(self):
        var_dic = {}
        d_fake = self.netD(self.fake, self.input)
        d_real = self.netD(self.ground_truth, self.input)
        var_dic["LOSS_G/MSE"] = mse_loss(self.fake * self.mask, self.ground_truth * self.mask)
        var_dic["LOSS_G/GAN"] = (torch.mean((d_real - d_fake + 1) ** 2) + torch.mean((d_fake - d_real - 1) ** 2))
        var_dic["LOSS_G/RALS_GAN"] = loss_g = var_dic["LOSS_G/MSE"] + var_dic["LOSS_G/GAN"] * 0.004
        var_dic["LOSS_D/RaLS_GAN"] = loss_d = (torch.mean((d_real - d_fake - 1) ** 2) +
                                               torch.mean((d_fake - d_real + 1) ** 2)) / 2
        var_dic["Eval/SNR"] = get_SNR(self.fake.detach(), self.ground_truth.detach())
        var_dic["Eval/MS-SSIM"] = ms_ssim(self.fake.detach(), self.ground_truth.detach(), 3)
        return var_dic

    def valid_epoch(self):
        super(Pix2pixGanTrainer, self).valid_epoch()
        self.netG.eval()
        self.netD.eval()
        if self.fixed_input is None:
            for batch in self.datasets.loader_test:
                if isinstance(batch, (list, tuple)):
                    self.fixed_input, self.fixed_ground_truth = self.get_data_from_batch(batch, self.device)
                    self.watcher.image(self.fixed_ground_truth, self.current_epoch, tag="Fixed/groundtruth",
                                       grid_size=(6, 6),
                                       shuffle=False)
                    dense_angles = np.arange(0., 180., 1.40625)  # 128 ， 128 +16 +16
                    angles = np.concatenate((np.arange(0, 67.5, 1.40625), np.arange(112.5, 180, 1.40625)))
                    input_iradon = self.iradon_trans(
                        torch.cat((self.fixed_input[:, :, :, :(self.fixed_input.size()[3] // 2 - hor_pad)],
                                   self.fixed_input[:, :, :, -(self.fixed_input.size()[3] // 2 - hor_pad):]), 3),
                        angles)
                    real_iradon = self.iradon_trans(self.fixed_ground_truth, dense_angles)
                    self.watcher.image(input_iradon, self.current_epoch, tag="Fixed/input_recon",
                                       grid_size=(6, 6),
                                       shuffle=False)
                    self.watcher.image(real_iradon, self.current_epoch, tag="Fixed/groundtruth_recon",
                                       grid_size=(6, 6),
                                       shuffle=False)

                else:
                    self.fixed_input = batch.to(self.device)
                self.watcher.image(self.fixed_input, self.current_epoch, tag="Fixed/input",
                                   grid_size=(6, 6),
                                   shuffle=False)
                break

        # watching the variation during training by a fixed input
        with torch.no_grad():
            fake = self.netG(self.fixed_input).detach()
        dense_angles = np.arange(0., 180., 1.40625)  # 128 ， 128 +16 +16
        angles = np.concatenate((np.arange(0, 67.5, 1.40625), np.arange(112.5, 180, 1.40625)))
        fake_iradon = self.iradon_trans(fake, dense_angles)
        self.watcher.image(fake, self.current_epoch, tag="Fixed/fake", grid_size=(6, 6), shuffle=False)
        self.watcher.image(fake_iradon, self.current_epoch, tag="Fixed/fake_recon", grid_size=(6, 6),
                           shuffle=False,
                           save_file=False)
        # saving training processes to build a .gif.
        self.watcher.set_training_progress_images(fake, grid_size=(6, 6))
        fake[:, :, :, :48] = self.fixed_ground_truth[:, :, :, :48]
        fake[:, :, :, 80:] = self.fixed_ground_truth[:, :, :, 80:]
        self.watcher.image(fake, self.current_epoch, tag="Fixed/fake_replace", grid_size=(6, 6), shuffle=False)
        fake_iradon = self.iradon_trans(fake, dense_angles)
        self.watcher.image(fake_iradon, self.current_epoch, tag="Fixed/fake_replace_recon", grid_size=(6, 6),
                           shuffle=False)
        self.netG.train()
        self.netD.train()

    def test(self):
        pass

    def _watch_images(self, tag, grid_size=(3, 3), shuffle=False, save_file=True):
        super(RadonInpaintPix2pixGanTrainer, self)._watch_images(tag, grid_size=(3, 3), shuffle=False, save_file=True)
        # angles = np.arange(-68., 67., 1.40625)  # 96 ， 96 +16 +16
        # angles = np.arange(67.5, 112.5, 1.40625)  # 32 ， 16 +16, angles_128[48:80]
        dense_angles = np.arange(0., 180., 1.40625)  # 128 ， 128 +16 +16
        angles = np.concatenate((np.arange(0, 67.5, 1.40625), np.arange(112.5, 180, 1.40625)))
        # torch.cat((self.input[:, :, :, :hor_pad], self.input[:, :, :, -hor_pad:]), 3)
        # input_iradon = self.iradon_trans(self.input[:, :, :, hor_pad:-hor_pad], angles)
        # print(len(angles))
        # print(self.input.size())
        # print(torch.cat((self.input[:, :, :, :(self.input.size()[3]//2 - hor_pad)], self.input[:, :, :, -(self.input.size()[3]//2 - hor_pad):]), 3).size())
        input_iradon = self.iradon_trans(torch.cat((self.input[:, :, :, :(self.input.size()[3] // 2 - hor_pad)],
                                                    self.input[:, :, :, -(self.input.size()[3] // 2 - hor_pad):]), 3), angles)
        real_iradon = self.iradon_trans(self.ground_truth, dense_angles)


        fake_iradon = self.iradon_trans(self.fake, dense_angles)
        # fakedenoise_iradon = denoise_tv_bregman(fake_iradon[0][0], 0.15)
        # import  matplotlib.pyplot as plt
        # plt.imshow(fakedenoise_iradon, cmap="gray")
        # plt.show()

        input_iradon_fft = self.fft(input_iradon)
        real_iradon_fft = self.fft(real_iradon)
        fake_iradon_fft = self.fft(fake_iradon)
        #
        self.watcher.image(input_iradon,
                           self.current_epoch,
                           tag="%s/input_iradon" % tag,
                           grid_size=grid_size,
                           shuffle=shuffle,
                           save_file=save_file)
        self.watcher.image(real_iradon,
                           self.current_epoch,
                           tag="%s/real_iradon" % tag,
                           grid_size=grid_size,
                           shuffle=shuffle,
                           save_file=save_file)
        self.watcher.image(fake_iradon,
                           self.current_epoch,
                           tag="%s/fake_iradon" % tag,
                           grid_size=grid_size,
                           shuffle=shuffle,
                           save_file=save_file)

        self.watcher.image(input_iradon_fft,
                           self.current_epoch,
                           tag="%s/input_iradon_fft" % tag,
                           grid_size=grid_size,
                           shuffle=shuffle,
                           save_file=save_file)
        self.watcher.image(real_iradon_fft,
                           self.current_epoch,
                           tag="%s/real_iradon_fft" % tag,
                           grid_size=grid_size,
                           shuffle=shuffle,
                           save_file=save_file)
        self.watcher.image(fake_iradon_fft,
                           self.current_epoch,
                           tag="%s/fake_iradon_fft" % tag,
                           grid_size=grid_size,
                           shuffle=shuffle,
                           save_file=save_file)

    @staticmethod
    def iradon_trans(tensor, angles):
        titlers = tensor.cpu().detach().numpy()
        result = np.ones((titlers.shape[0], 1, 256, 256))
        for index, tt in enumerate(titlers):  # [bs,1,256,256], tt=>[1,256,256]
            result[index][0] = iradon(tt[0] * 255, theta=angles, circle=True)
        result = torch.from_numpy(result).float() / 255.
        return result

    @staticmethod
    def fft(tensor):
        titlers = tensor.cpu().detach().numpy()
        result = np.ones((titlers.shape[0], 1, 256, 256))
        for index, tt in enumerate(titlers):  # [bs,1,256,256], tt=>[1,256,256]

            original_img = normlize(tt[0])
            dfgram = np.fft.fftshift(np.fft.fft2(original_img))
            disdfgram = abs(dfgram)
            disdfgram_np = np.log(disdfgram / disdfgram.max() + 1e-5)

            result[index][0] = disdfgram_np
        result = torch.from_numpy(result).float() / 255.
        return result


def normlize(img_np_256):
    return (img_np_256 - img_np_256.min()) / (img_np_256.max() - img_np_256.min() + 1e-5)


def clear_all_images(paths):
    paths = [paths] if isinstance(paths, str) else paths
    for path in paths:
        if os.path.exists(path):
            shutil.rmtree(path)
        os.makedirs(path)


if __name__ == '__main__':
    gpus = [0, 1]
    logdir = "log/iter_denoise_1"  # -90., 90., 11.25)16 =>(-90., 90., 1.40625)128
    clear_all_images(logdir)
    batch_size = 4  # 4
    # 900/4 = 225      225 * 200epochs = 45000 steps, 120, 180
    # 37000/8 = 4625, 4625 * 9.7epochs = 45000 steps, 6, 9,   3s/step, 3 * 4625 * 10 /3600 = 38.5h
    # 1e-4, 70K
    valid_size = 5000
    nepochs = 25
    hor_pad = 16
    G_hprams = {"optimizer": "Adam",
                "lr_decay": 0.1,
                "decay_position": [15, 22],
                "position_type": "epoch",
                "lr": 1e-4,
                "lr_reset": {1: 1e-4, 2: 1.5e-4, 3: 2e-4},
                "weight_decay": 1e-4,
                "betas": (0.9, 0.999),
                "amsgrad": False
                }
    D_hprams = {"optimizer": "RMSprop",
                "lr_decay": 0.1,
                "decay_position": [15, 22],
                "position_type": "epoch",
                "lr": 1e-4,
                "lr_reset": {1: 1e-4, 2: 1.5e-4, 3: 2e-4},
                "weight_decay": 1e-4,
                "alpha": 0.99,
                "momentum": 0
                }

    torch.backends.cudnn.benchmark = True
    print('===> Build dataset')
    radon_inpaint_datasets = RadonInnerInpaintDatasets(r"/data/dataset/data_inpaint", batch_size=batch_size,
                                                       valid_size=valid_size,
                                                       hor_pad=hor_pad, shuffle=True)
    print('===> Building model')
    # G_net = Inpaint_Net(1, 1, 16, 16, 16)
    G_net = Inpaint_Net(1, 1, 16, 16, 16)
    D_net = NLD_LG_inpaint(64)
    net_G = Model(G_net, gpus, verbose=True, check_point_pos=5)
    net_D = Model(D_net, gpus, verbose=True, check_point_pos=5)
    # shutil.copy("block.py", logdir)
    # shutil.copy("model.py", logdir)
    print('===> Building optimizer')
    optG = Optimizer(net_G.parameters(), **G_hprams)
    optD = Optimizer(net_D.parameters(), **D_hprams)
    print('===> Training')
    Trainer = RadonInpaintPix2pixGanTrainer(logdir, nepochs, gpus, net_G, net_D, optG, optD, radon_inpaint_datasets)

    import sys

    _DEBUG_ = len(sys.argv) > 1 and sys.argv[1].strip().lower() == "-d"
    if _DEBUG_:
        Trainer.debug()
    else:
        import warnings

        warnings.filterwarnings('ignore')
        Trainer.train(subbar_disable=False)
