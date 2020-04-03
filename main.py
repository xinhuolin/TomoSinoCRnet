import os
import shutil

import matplotlib.pyplot as plt
import torch
from jdit import Optimizer, Model
from jdit.trainer import Pix2pixGanTrainer
from torch.nn.functional import mse_loss
from tqdm import tqdm

from dataset import RadonInnerInpaintDatasets
from model import Due_Discriminator, Due_Generator
from pytorch_radon import Radon, IRadon
from tools.evaluate import *


def lsgan_d(d_real, d_fake):
    return 0.5 * (
            mse_loss(d_real - d_fake.mean(), torch.ones_like(d_real)) +
            mse_loss(d_fake - d_real.mean(), torch.zeros_like(d_real)))


def lsgan_g(d_real, d_fake):
    return 0.5 * (
            mse_loss(d_real - d_fake.mean(), torch.zeros_like(d_real)) +
            mse_loss(d_fake - d_real.mean(), torch.ones_like(d_real)))


class RadonIterDenoiseGanTrainer(Pix2pixGanTrainer):
    def __init__(self, logdir, nepochs, gpu_ids_abs, netG, netD, optG, optD, datasets):
        super(RadonIterDenoiseGanTrainer, self).__init__(logdir, nepochs, gpu_ids_abs, netG, netD, optG, optD,
                                                         datasets)
        self.w_gan = 0.004
        self.mask = torch.ones((batch_size, 1, 256, 128), requires_grad=False) * 0.01
        self.mask[:, :, :, 48:80] = 1
        self.mask = self.mask.to(self.device)
        self.fixed_in_sino = None

    def train_epoch(self, subbar_disable=False):
        for iteration, batch in tqdm(enumerate(self.datasets.loader_train, 1), unit="step", disable=subbar_disable):
            self.step += 1
            self.in_sino, self.real_sino = self.get_data_from_batch(batch, self.device)
            # plt.subplot(1, 2, 1)
            # plt.imshow(self.in_sino.detach().cpu()[0][0], cmap="gray")
            # plt.subplot(1, 2, 2)
            # plt.imshow(self.real_sino.detach().cpu()[0][0], cmap="gray")
            # plt.show()
            self.in_recon, self.real_recon = iradon_trans(self.in_sino * 255).detach(), iradon_trans(self.real_sino* 255).detach()
            # plt.subplot(2, 2, 3)
            # plt.imshow(self.in_recon.detach().cpu()[0][0], cmap="gray")
            # plt.subplot(2, 2, 4)
            # plt.imshow(self.real_recon.detach().cpu()[0][0], cmap="gray")

            # plt.subplot(2, 2, 3)
            # plt.imshow(self.fake_recon.detach().cpu()[0][0], cmap="gray")
            # plt.subplot(2, 2, 4)
            # plt.imshow(self.fake_sino.detach().cpu()[0][0], cmap="gray")
            # plt.show()

            self.fake_recon, self.fake_sino = self.netG(self.in_recon, self.in_sino)

            self._train_iteration(self.optD, self.compute_d_loss, csv_filename="Train_D")
            if (self.step % self.d_turn) == 0:
                self._train_iteration(self.optG, self.compute_g_loss, csv_filename="Train_G")
            if iteration == 1000:
                self._watch_images("Train")

    def compute_d_loss(self):
        d_fake_recon, d_fake_sino = self.netD(self.in_recon, self.in_sino, self.fake_recon.detach(),
                                              self.fake_sino.detach())
        d_real_recon, d_real_sino = self.netD(self.in_recon, self.in_sino, self.real_recon.detach(),
                                              self.real_sino.detach())
        var_dic = {}
        var_dic["LOSS_D/RaLS_GAN"] = loss_d = lsgan_d(d_real_recon, d_fake_recon) + lsgan_d(d_real_sino, d_fake_sino)
        return loss_d, var_dic

    def compute_g_loss(self):
        var_dic = {}
        d_fake_recon, d_fake_sino = self.netD(self.in_recon, self.in_sino, self.fake_recon, self.fake_sino)
        d_real_recon, d_real_sino = self.netD(self.in_recon, self.in_sino, self.real_recon, self.real_sino)
        # var_dic["LOSS_G/MSE"] = (mse_loss(self.fake_sino * self.mask, self.real_sino * self.mask) +
        #                          mse_loss(self.fake_recon, self.real_recon))
        # var_dic["LOSS_G/GAN"] = lsgan_g(d_real_recon, d_fake_recon) + lsgan_g(d_real_sino, d_fake_sino)
        var_dic["LOSS_G/MSE"] = mse_loss(self.fake_recon, self.real_recon)
        var_dic["LOSS_G/GAN"] = lsgan_g(d_real_recon, d_fake_recon)
        var_dic["LOSS_G/RALS_GAN"] = loss_g = var_dic["LOSS_G/MSE"] + var_dic["LOSS_G/GAN"] * self.w_gan
        return loss_g, var_dic

    def compute_valid(self):
        var_dic = {}
        d_fake_recon, d_fake_sino = self.netD(self.in_recon, self.in_sino, self.fake_recon, self.fake_sino)
        d_real_recon, d_real_sino = self.netD(self.in_recon, self.in_sino, self.real_recon, self.real_sino)
        var_dic["LOSS_G/MSE"] = (mse_loss(self.fake_sino * self.mask, self.real_sino * self.mask) +
                                 mse_loss(self.fake_recon, self.real_recon))
        var_dic["LOSS_G/GAN"] = lsgan_g(d_real_recon, d_fake_recon) + lsgan_g(d_real_sino, d_fake_sino)
        var_dic["LOSS_G/RALS_GAN"] = loss_g = var_dic["LOSS_G/MSE"] + var_dic["LOSS_G/GAN"] * self.w_gan
        var_dic["LOSS_D/RaLS_GAN"] = loss_d = lsgan_d(d_real_recon, d_fake_recon) + lsgan_d(d_real_sino, d_fake_sino)

        var_dic["Eval/PSNR"] = torch.from_numpy(get_psnr(self.real_recon.detach(), self.fake_recon.detach()))
        var_dic["Eval/SSIM"] = torch.from_numpy(get_ssim(self.real_recon.detach(), self.fake_recon.detach()))
        var_dic["Eval/NRMSE"] = torch.from_numpy(get_nrmse(self.real_recon.detach(), self.fake_recon.detach()))
        return var_dic

    def valid_epoch(self):
        self.netG.eval()
        self.netD.eval()
        avg_dic = {}
        for iteration, batch in enumerate(self.datasets.loader_valid, 1):
            self.input, self.ground_truth = self.get_data_from_batch(batch, self.device)
            with torch.no_grad():
                self.in_sino, self.real_sino = self.get_data_from_batch(batch, self.device)
                self.in_recon = iradon_trans(self.in_sino).detach()
                self.real_recon = iradon_trans(self.real_sino).detach()
                self.fake_recon, self.fake_sino = self.netG(self.in_recon, self.in_sino)
                dic = self.compute_valid()
            if avg_dic == {}:
                avg_dic = dic
            else:
                # 求和
                for key in dic.keys():
                    avg_dic[key] += dic[key]

        for key in avg_dic.keys():
            avg_dic[key] = avg_dic[key] / self.datasets.nsteps_valid

        self.watcher.scalars(avg_dic, self.step, tag="Valid")
        self.loger.write(self.step, self.current_epoch, avg_dic, "Valid", header=self.current_epoch <= 1)
        self._watch_images(tag="Valid")

        if self.fixed_in_sino is None:
            for batch in self.datasets.loader_test:
                self.fixed_in_sino, self.fixed_real_sino = self.get_data_from_batch(batch, self.device)

                self.fixed_real_recon = iradon_trans(self.fixed_real_sino).detach()
                self.fixed_in_recon = iradon_trans(self.fixed_in_sino).detach()

                # fixed_in_recon = iradon_trans(self.fixed_in_sino).detach()
                self.watcher.image(self.fixed_real_recon, self.current_epoch, tag="Fixed/real_recon",
                                   grid_size=(6, 6),
                                   shuffle=False)
                self.watcher.image(self.fixed_real_sino, self.current_epoch, tag="Fixed/real_sino",
                                   grid_size=(6, 6),
                                   shuffle=False)
                self.watcher.image(self.fixed_in_sino, self.current_epoch, tag="Fixed/in_sino",
                                   grid_size=(6, 6),
                                   shuffle=False)
                self.watcher.image(self.fixed_in_recon, self.current_epoch, tag="Fixed/in_recon",
                                   grid_size=(6, 6),
                                   shuffle=False)

        # watching the variation during training by a fixed input
        with torch.no_grad():
            fixed_fake_recon, fixed_fake_sino = self.netG(self.fixed_in_recon, self.fixed_in_sino, iter=3)

        self.watcher.image(fixed_fake_sino.detach(), self.current_epoch, tag="Fixed/fake_sino3", grid_size=(6, 6),
                           shuffle=False)
        self.watcher.image(fixed_fake_recon.detach(), self.current_epoch, tag="Fixed/fake_recon3", grid_size=(6, 6),
                           shuffle=False)
        # saving training processes to build a .gif.
        self.watcher.set_training_progress_images(fixed_fake_recon, grid_size=(6, 6))
        fixed_fake_sino[:, :, :, :48] = self.fixed_real_sino[:, :, :, :48]
        fixed_fake_sino[:, :, :, 80:] = self.fixed_real_sino[:, :, :, 80:]
        self.watcher.image(fixed_fake_sino, self.current_epoch, tag="Fixed/fake_sino_replace3", grid_size=(6, 6),
                           shuffle=False)
        self.watcher.image(iradon_trans(fixed_fake_sino).detach(), self.current_epoch, tag="Fixed/fake_recon_replace3",
                           grid_size=(6, 6),
                           shuffle=False)

        with torch.no_grad():
            fixed_fake_recon, fixed_fake_sino = self.netG(self.fixed_in_recon, self.fixed_in_sino, iter=9)

        self.watcher.image(fixed_fake_sino.detach(), self.current_epoch, tag="Fixed/fake_sino9", grid_size=(6, 6),
                           shuffle=False)
        self.watcher.image(fixed_fake_recon.detach(), self.current_epoch, tag="Fixed/fake_recon9", grid_size=(6, 6),
                           shuffle=False)
        # saving training processes to build a .gif.
        # self.watcher.set_training_progress_images(fixed_fake_recon, grid_size=(6, 6))
        fixed_fake_sino[:, :, :, :48] = self.fixed_real_sino[:, :, :, :48]
        fixed_fake_sino[:, :, :, 80:] = self.fixed_real_sino[:, :, :, 80:]
        self.watcher.image(fixed_fake_sino, self.current_epoch, tag="Fixed/fake_sino_replace9", grid_size=(6, 6),
                           shuffle=False)
        self.watcher.image(iradon_trans(fixed_fake_sino).detach(), self.current_epoch, tag="Fixed/fake_recon_replace9",
                           grid_size=(6, 6),
                           shuffle=False)
        self.netG.train()
        self.netD.train()

    def test(self):
        pass

    def _watch_images(self, tag, grid_size=(3, 3), shuffle=False, save_file=True):
        self.watcher.image(self.in_recon,
                           self.current_epoch,
                           tag="%s_input/in_recon" % tag,
                           grid_size=grid_size,
                           shuffle=shuffle,
                           save_file=save_file)
        self.watcher.image(self.in_sino,
                           self.current_epoch,
                           tag="%s_input/in_sino" % tag,
                           grid_size=grid_size,
                           shuffle=shuffle,
                           save_file=save_file)
        self.watcher.image(self.fake_recon,
                           self.current_epoch,
                           tag="%s_fake/fake_recon" % tag,
                           grid_size=grid_size,
                           shuffle=shuffle,
                           save_file=save_file)
        self.watcher.image(self.fake_sino,
                           self.current_epoch,
                           tag="%s_fake/fake_sino" % tag,
                           grid_size=grid_size,
                           shuffle=shuffle,
                           save_file=save_file)
        self.watcher.image(self.real_recon,
                           self.current_epoch,
                           tag="%s_real/real_recon" % tag,
                           grid_size=grid_size,
                           shuffle=shuffle,
                           save_file=save_file)
        self.watcher.image(self.real_sino,
                           self.current_epoch,
                           tag="%s_real/real_sino" % tag,
                           grid_size=grid_size,
                           shuffle=shuffle,
                           save_file=save_file)

        input_iradon_fft = self.fft(self.in_recon)
        real_iradon_fft = self.fft(self.real_recon)
        fake_iradon_fft = self.fft(self.fake_recon)
        self.watcher.image(input_iradon_fft,
                           self.current_epoch,
                           tag="%s_input/in_recon_fft" % tag,
                           grid_size=grid_size,
                           shuffle=shuffle,
                           save_file=save_file)
        self.watcher.image(real_iradon_fft,
                           self.current_epoch,
                           tag="%s_real/real_iradon_fft" % tag,
                           grid_size=grid_size,
                           shuffle=shuffle,
                           save_file=save_file)
        self.watcher.image(fake_iradon_fft,
                           self.current_epoch,
                           tag="%s_fake/fake_iradon_fft" % tag,
                           grid_size=grid_size,
                           shuffle=shuffle,
                           save_file=save_file)

    # @staticmethod
    # def iradon_trans(tensor, angles):
    #     titlers = tensor.cpu().detach().numpy()
    #     result = np.ones((titlers.shape[0], 1, 256, 256))
    #     for index, tt in enumerate(titlers):  # [bs,1,256,256], tt=>[1,256,256]
    #         result[index][0] = iradon(tt[0] * 255, theta=angles, circle=True)
    #     result = torch.from_numpy(result).float() / 255.
    #     return result

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


radon_trans = Radon(256, np.arange(0., 180., 1.40625))
iradon_trans = IRadon(256, np.arange(0., 180., 1.40625))

if __name__ == '__main__':
    gpus = [0, 1]
    logdir = "log/iter_D1"  # -90., 90., 11.25)16 =>(-90., 90., 1.40625)128
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
    G_net = Due_Generator(np.arange(0., 180., 1.40625), 256, nf=16, nb=2, gc=16)  # Due_Discriminator,Due_Generator
    D_net = Due_Discriminator(np.arange(0., 180., 1.40625), 256, 32)
    net_G = Model(G_net, gpus, verbose=True, check_point_pos=5)
    net_D = Model(D_net, gpus, verbose=True, check_point_pos=5)

    print('===> Building optimizer')
    optG = Optimizer(net_G.parameters(), **G_hprams)
    optD = Optimizer(net_D.parameters(), **D_hprams)
    print('===> Training')
    Trainer = RadonIterDenoiseGanTrainer(logdir, nepochs, gpus, net_G, net_D, optG, optD, radon_inpaint_datasets)

    import sys

    _DEBUG_ = len(sys.argv) > 1 and sys.argv[1].strip().lower() == "-d"
    if _DEBUG_:
        Trainer.debug()
    else:
        import warnings

        warnings.filterwarnings('ignore')
        Trainer.train(subbar_disable=False)
