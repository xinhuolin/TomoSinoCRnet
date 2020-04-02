from jdit import Optimizer, Model
from jdit.trainer import Pix2pixGanTrainer
import torch
from torch.nn import L1Loss, MSELoss
from dataset import RadonInpaintDatasets
from model import Inpaint_Net, NLD_inpaint, NLD_LG_inpaint
from tools.ssim import ms_ssim, get_SNR
import numpy as np
from skimage.transform import iradon
from PIL import Image, ImageOps
import torchvision
from torch.nn.functional import mse_loss


# pytorch.set_default_tensor_type('torch.DoubleTensor')
class RadonInpaintPix2pixGanTrainer(Pix2pixGanTrainer):
    def __init__(self, logdir, nepochs, gpu_ids_abs, netG, netD, optG, optD, datasets):
        super(RadonInpaintPix2pixGanTrainer, self).__init__(logdir, nepochs, gpu_ids_abs, netG, netD, optG, optD,
                                                            datasets)

    def compute_d_loss(self):
        d_fake = self.netD(self.fake.detach(), self.input)
        d_real = self.netD(self.ground_truth, self.input)
        var_dic = {}
        var_dic["LOSS_D/RaLS_GAN"] = loss_d = (torch.mean((d_real - d_fake - 1) ** 2) +
                                               torch.mean((d_fake - d_real + 1) ** 2)) / 2
        # (torch.mean((d_real - torch.mean(d_fake) - 1) ** 2) +
        #  torch.mean((d_fake - torch.mean(d_real) + 1) ** 2)) / 2

        return loss_d, var_dic

    def compute_g_loss(self):
        var_dic = {}
        d_fake = self.netD(self.fake, self.input)
        d_real = self.netD(self.ground_truth, self.input)
        mask = torch.ones_like(self.fake, requires_grad=False)
        mask[:, :, :, hor_pad:-hor_pad] = 0.1
        # if self.current_epoch <= 2:
        #     var_dic["LOSS_G/MSE"] = mse_loss(self.fake * mask, self.ground_truth * mask)
        #     var_dic["LOSS_G/RALS_GAN"] = loss_g = var_dic["LOSS_G/MSE"]
        # else:
        #     var_dic["LOSS_G/MSE"] = mse_loss(self.fake * mask, self.ground_truth * mask)
        #     var_dic["LOSS_G/GAN"] = (torch.mean((d_real - d_fake + 1) ** 2) +
        #                              torch.mean((d_fake - d_real - 1) ** 2))
        #     # (torch.mean((d_real - torch.mean(d_fake) + 1) ** 2) +
        #     #                      torch.mean((d_fake - torch.mean(d_real) - 1) ** 2)) / 2
        #
        #     var_dic["LOSS_G/RALS_GAN"] = loss_g = 0.004 * var_dic["LOSS_G/GAN"] + var_dic["LOSS_G/MSE"]
        #     # 0.0004 * var_dic["LOSS_G/GAN"] + var_dic["LOSS_G/MSE"]
        var_dic["LOSS_G/MSE"] = mse_loss(self.fake * mask, self.ground_truth * mask)
        var_dic["LOSS_G/GAN"] = (torch.mean((d_real - d_fake + 1) ** 2) + torch.mean((d_fake - d_real - 1) ** 2))
        var_dic["LOSS_G/RALS_GAN"] = loss_g = var_dic["LOSS_G/MSE"] + var_dic["LOSS_G/GAN"] * 0.04
        return loss_g, var_dic

    def compute_valid(self):
        var_dic = {}
        var_dic["Eval/SNR"] = get_SNR(self.fake.detach(), self.ground_truth.detach())
        var_dic["Eval/MS-SSIM"] = ms_ssim(self.fake.detach(), self.ground_truth.detach(), 3)
        return var_dic

    def test(self):
        pass

    def _watch_images(self, tag, grid_size=(3, 3), shuffle=False, save_file=True):
        super(RadonInpaintPix2pixGanTrainer, self)._watch_images(tag, grid_size=(3, 3), shuffle=False, save_file=True)
        angles = np.arange(-68., 67., 1.40625)  # 96 ， 96 +16 +16
        dense_angles = np.arange(-90., 90., 1.40625)  # 128 ， 128 +16 +16
        input_iradon = self.iradon_trans(self.input[:, :, :, hor_pad:-hor_pad], angles)
        real_iradon = self.iradon_trans(self.ground_truth, dense_angles)
        fake_iradon = self.iradon_trans(self.fake, dense_angles)

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

    @property
    def configure(self):
        config_dic = super(RadonInpaintPix2pixGanTrainer, self).configure
        config_dic["info"] = """
        Inpainting task:
        angles = np.arange(-68., 67., 1.40625)  # 96 ， 96 +16 +16
        dense_angles = np.arange(-90., 90., 1.40625)  # 128 ， 128 +16 +16
        pad:32,(0,255)
        """
        return config_dic


#
# def fft(original_img):
#     original_img = normlize(original_img)
#     dfgram = np.fft.fftshift(np.fft.fft2(original_img))
#     disdfgram = abs(dfgram)
#     disdfgram_np = np.log(disdfgram / disdfgram.max() + 1e-5)
#     return disdfgram_np

def normlize(img_np_256):
    return (img_np_256 - img_np_256.min()) / (img_np_256.max() - img_np_256.min() + 1e-5)


if __name__ == '__main__':
    gpus = [0, 1]
    logdir = "log_inpaint/Radon_mse_2"  # -90., 90., 11.25)16 =>(-90., 90., 1.40625)128

    batch_size = 4
    valid_size = 100
    hor_pad = 16
    # nepochs = 200
    # G_hprams = {"optimizer": "Adam",
    #             "lr_decay": 0.1,
    #             "decay_position": [120, 180],
    #             "position_type": "epoch",
    #             "lr": 2e-5,
    #             "lr_reset": {1: 2e-5, 2: 4e-5, 3: 6e-5, 4: 8e-5, 5: 1e-4},
    #             "weight_decay": 1e-4,
    #             "betas": (0.9, 0.999),
    #             "amsgrad": False
    #             }
    # D_hprams = {"optimizer": "Adam",
    #             "lr_decay": 0.1,
    #             "decay_position": [120, 180],
    #             "position_type": "epoch",
    #             "lr": 2e-5,
    #             "lr_reset": {1: 2e-5, 2: 4e-5, 3: 6e-5, 4: 8e-5, 5: 1e-4},
    #             "weight_decay": 1e-4,
    #             "betas": (0, 0.999),
    #             "amsgrad": False
    #             }

    nepochs = 100
    G_hprams = {"optimizer": "Adam",
                "lr_decay": 0.1,
                "decay_position": [20, 80],
                "position_type": "epoch",
                "lr": 1e-4,
                # "lr_reset": {1: 2e-5, 2: 4e-5, 3: 6e-5, 4: 8e-5, 5: 1e-5},
                "weight_decay": 1e-4,
                "betas": (0.9, 0.999),
                "amsgrad": False
                }
    D_hprams = {"optimizer": "Adam",
                "lr_decay": 0.1,
                "decay_position": [20, 80],
                "position_type": "epoch",
                "lr": 1e-4,
                # "lr_reset": {1: 2e-5, 2: 4e-5, 3: 6e-5, 4: 8e-5, 5: 1e-4},
                "weight_decay": 1e-4,
                "betas": (0, 0.999),
                "amsgrad": False
                }

    torch.backends.cudnn.benchmark = True
    print('===> Build dataset')
    radon_inpaint_datasets = RadonInpaintDatasets("data_inpaint", batch_size=batch_size, valid_size=valid_size,
                                                  hor_pad=hor_pad)
    print('===> Building model')
    G_net = Inpaint_Net(1, 1, 16, 16, 16)
    D_net = NLD_LG_inpaint(32)
    net_G = Model(G_net, gpus, verbose=True, check_point_pos=50)
    net_D = Model(D_net, gpus, verbose=True, check_point_pos=50)
    net_D.load_point("netD", 100, "log_inpaint/Radon_mse_2")
    net_G.load_point("netG", 100, "log_inpaint/Radon_mse_2")
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
        Trainer.train(subbar_disable=True)
