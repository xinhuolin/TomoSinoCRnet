import os
import shutil
from math import log10, exp
import numpy as np
from tqdm import tqdm
import torchvision.transforms as ts
from PIL import Image
from jdit import Model
from skimage.transform import radon, iradon, iradon_sart
from skimage.restoration import denoise_tv_bregman
from model import Inpaint_Net
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import sys

def clear_all_images():
    paths = ['img_show/narrow/sinogram',
             'img_show/narrow/recon_wbp',
             'img_show/narrow/recon_wbp_fft',
             'img_show/narrow/recon_sart',
             'img_show/narrow/recon_sart_fft',
             'img_show/narrow/recon_tvm',
             'img_show/narrow/recon_tvm_fft',

             'img_show/wide/sinogram',
             'img_show/wide/recon_wbp',
             'img_show/wide/recon_wbp_fft',
             'img_show/wide/recon_sart',
             'img_show/wide/recon_sart_fft',
             'img_show/wide/recon_tvm',
             'img_show/wide/recon_tvm_fft',

             'img_show/inpaint/input',
             'img_show/inpaint/sinogram',
             'img_show/inpaint/recon_wbp',
             'img_show/inpaint/recon_wbp_fft',
             'img_show/inpaint/recon_sart',
             'img_show/inpaint/recon_sart_fft',
             'img_show/inpaint/recon_tvm',
             'img_show/inpaint/recon_tvm_fft',

             # 'img_show/original',
             'img_show/original_fft',

             ]
    for path in paths:
        if os.path.exists(path):
            shutil.rmtree(path)
        os.makedirs(path)
    print("clear all images!")


# def save_real_images(original_img_path, angles, result_path=None):
#     original_paths_check = ["", "/tiltser", "/recon_fft", "/recon_bp"]
#     for p in original_paths_check:
#         if result_path is None:
#             result_path = "real_%s_%s" % (len(angles), str(angles))
#         if not os.path.exists(result_path + p):
#             os.mkdir(result_path + p)
#     with Image.open(original_img_path) as original:
#         original = original.convert("F")
#     (original.convert("L")).save("%s/original/%s" % (root, filename))

def save_real_images(original_img_path, root="img_show"):
    filename = path.split("/")[-1]
    with Image.open(original_img_path) as original:
        original = original.convert("F")

    (original.convert("L")).save("%s/original/%s" % (root, filename))
    print("save to %s/original/: %s" % (root, filename))
    # ----------------------------------------
    original_np = np.asarray(original) / 255  # 0,1
    original_fft = fft(original_np)
    save_np2img("%s/original_fft/%s" % (root, filename), original_fft, norm=True)
    angles_16 = np.arange(-90., 90., 11.25)  # 16
    angles_32 = np.arange(-90., 90., 5.625)  # 32
    angles_128 = np.arange(-90., 90., 1.40625)  # 128

    real_16_tiltser = radon(original_np, theta=angles_16, circle=True)
    real_32_tiltser = radon(original_np, theta=angles_32, circle=True)
    real_128_tiltser = radon(original_np, theta=angles_128, circle=True)
    save_np2img("%s/real_16/tiltser/%s" % (root, filename), real_16_tiltser, norm=False)
    save_np2img("%s/real_32/tiltser/%s" % (root, filename), real_32_tiltser, norm=False)
    save_np2img("%s/real_128/tiltser/%s" % (root, filename), real_128_tiltser, norm=False)

    real_16_recon_bp = iradon(real_16_tiltser, theta=angles_16, circle=True)
    real_32_recon_bp = iradon(real_32_tiltser, theta=angles_32, circle=True)
    real_128_recon_bp = iradon(real_128_tiltser, theta=angles_128, circle=True)
    save_np2img("%s/real_16/recon_bp/%s" % (root, filename), real_16_recon_bp, norm=True)
    save_np2img("%s/real_32/recon_bp/%s" % (root, filename), real_32_recon_bp, norm=True)
    save_np2img("%s/real_128/recon_bp/%s" % (root, filename), real_128_recon_bp, norm=True)

    real_16_fft = fft(real_16_recon_bp)
    real_32_fft = fft(real_32_recon_bp)
    real_128_fft = fft(real_128_recon_bp)
    save_np2img("%s/real_16/fft/%s" % (root, filename), real_16_fft, norm=True)
    save_np2img("%s/real_32/fft/%s" % (root, filename), real_32_fft, norm=True)
    save_np2img("%s/real_128/fft/%s" % (root, filename), real_128_fft, norm=True)


def save_gan_images_(original_img_path, inter_model, in_angles, out_angles, result_path="img_show/gan_inter",
                     use_gpu=True):
    result_paths_check = ["", "/recon_bp", "/recon_fft", "/tiltser"]
    for p in result_paths_check:
        if not os.path.exists(result_path + p):
            os.mkdir(result_path + p)

    filename = path.split("/")[-1]
    with Image.open(original_img_path) as original:
        original = original.convert("F")
    # ----------------------------------------
    original_np = np.asarray(original) / 255
    # in_angles = np.arange(-90., 90., 5.625)  # 32
    # out_angles = np.arange(-90., 90., 1.40625)  # 128
    input_tiltser = radon(original_np, theta=in_angles, circle=True)
    with torch.no_grad():
        input_tiltser = input_tiltser[np.newaxis, np.newaxis, :]
        input_tiltser = torch.Tensor(input_tiltser / 255)
        if use_gpu:
            input_tiltser = input_tiltser.cuda()
        output_tiltser = inter_model(input_tiltser).cpu().detach().numpy()[0][0]

    save_np2img("%s/tiltser/%s" % (result_path, filename), output_tiltser)

    gan_recon_bp = iradon(output_tiltser, theta=out_angles, circle=True)
    save_np2img("%s/recon_bp/%s" % (result_path, filename), gan_recon_bp)

    gan_fft = fft(gan_recon_bp)
    save_np2img("%s/recon_fft/%s" % (result_path, filename), gan_fft, norm=True)


def save_gan_images(model_inter_4, model_inter_8, original_img_path, root="img_show", use_gpu=True):
    filename = path.split("/")[-1]
    with Image.open(original_img_path) as original:
        original = original.convert("F")
    # ----------------------------------------
    original_np = np.asarray(original) / 255
    angles_16 = np.arange(-90., 90., 11.25)  # 16
    angles_32 = np.arange(-90., 90., 5.625)  # 32
    angles_128 = np.arange(-90., 90., 1.40625)  # 128
    input_16 = radon(original_np, theta=angles_16, circle=True)
    input_32 = radon(original_np, theta=angles_32, circle=True)
    with torch.no_grad():
        input_16 = input_16[np.newaxis, np.newaxis, :]
        input_32 = input_32[np.newaxis, np.newaxis, :]
        input_16 = torch.Tensor(input_16 / 255)
        input_32 = torch.Tensor(input_32 / 255)
        if use_gpu:
            input_16 = input_16.cuda()
            input_32 = input_32.cuda()
        output_inter_4 = model_inter_4(input_32).cpu().detach().numpy()[0][0]
        output_inter_8 = model_inter_8(input_16).cpu().detach().numpy()[0][0]

    save_np2img("%s/gan_inter_4_32to128/tiltser/%s" % (root, filename), output_inter_4)
    save_np2img("%s/gan_inter_8_16to128/tiltser/%s" % (root, filename), output_inter_8)

    gan_inter_4_recon_bp = iradon(output_inter_4, theta=angles_128, circle=True)
    gan_inter_8_recon_bp = iradon(output_inter_8, theta=angles_128, circle=True)
    save_np2img("%s/gan_inter_4_32to128/recon_bp/%s" % (root, filename), gan_inter_4_recon_bp)
    save_np2img("%s/gan_inter_8_16to128/recon_bp/%s" % (root, filename), gan_inter_8_recon_bp)

    gan_inter_4_fft = fft(gan_inter_4_recon_bp)
    gan_inter_8_fft = fft(gan_inter_8_recon_bp)
    save_np2img("%s/gan_inter_4_32to128/fft/%s" % (root, filename), gan_inter_4_fft, norm=True)
    save_np2img("%s/gan_inter_8_16to128/fft/%s" % (root, filename), gan_inter_8_fft, norm=True)


def save_np2img(path, img_np_256, norm=True, show=False):
    if norm:
        img_np_256 = 255 * normlize(img_np_256)
    image = Image.fromarray(img_np_256).convert("L")
    if show:
        image.show()
    image.save(path)


def normlize(img_np_256):
    return (img_np_256 - img_np_256.min()) / (img_np_256.max() - img_np_256.min() + 1e-5)


def fft(original_img):
    original_img = normlize(original_img)
    dfgram = np.fft.fftshift(np.fft.fft2(original_img))
    disdfgram = abs(dfgram)
    disdfgram_np = np.log(disdfgram / disdfgram.max() + 1e-5)
    # disdfgram_np = normlize(disdfgram_np)
    # fft_img = Image.fromarray(disdfgram_np * 255)
    # fft_img.show()
    return disdfgram_np


def prepare_model(weights_path, gpu_ids_abs):
    net = Inpaint_Net(1, 1, 16, 16, 16)
    model = Model(net, gpu_ids_abs=gpu_ids_abs)
    model.load_weights(weights_path, strict=True)
    print("load model successfully!")
    model.eval()
    return model


class NPInnerPad():
    def __init__(self, inner_width=32, pad_value=None):
        self.inner_width = inner_width
        self.pad_value = pad_value

    def __call__(self, img_np):
        hight, width = img_np.shape
        pad = np.mean(img_np) if self.pad_value is None else self.pad_value
        # print(img_np.max(), img_np.min(), np.mean(img_np), img_np.shape)
        np_pad = np.ones((self.inner_width, hight)) * pad
        # 96//2 =  48     angles_128[48:80]
        result_np = np.insert(img_np, width // 2, values=np_pad, axis=1)
        return result_np


def porcess_input(img, name):
    img_np = np.asarray(img)
    img_np = NPInnerPad()(img_np)
    img = Image.fromarray(img_np)
    # img.convert("L").save(os.path.join('img_show/inpaint/input', name))
    trans = ts.Compose([ts.ToTensor(), ts.Normalize([0.], [255.])])
    img_tensor = trans(img)
    return img_tensor.unsqueeze(0)


def porcess_output(img_np, name, show=False):
    img = Image.fromarray(img_np * 255)
    if show:
        img.show()
    img.convert("L").save(os.path.join('img_show/inpaint/sinogram', name))


class Recon(object):
    @staticmethod
    def wbp(sinogram_np, angles):
        # assert sinogram_np.max() <= 1, str(img_np.max())
        recon_np = iradon(sinogram_np, theta=angles, circle=True)  # iradon 0. 255.
        return recon_np

    @staticmethod
    def sart(sinogram_np, angles, relaxation=0.25, step=10):
        # assert sinogram_np.max() <= 1, str(img_np.max())
        recon_np = None
        for _ in range(step):
            recon_np = iradon_sart(sinogram_np.astype(np.float), angles, image=recon_np, relaxation=relaxation)
        return recon_np

    #
    # @staticmethod
    # def tvm(recon_np, weight, max_iter, eps=1e-3, isotropic=True):
    #     recon_TVM = denoise_tv_bregman(recon_np, weight=weight, max_iter=max_iter, eps=eps, isotropic=isotropic)
    #     return recon_TVM

    @classmethod
    def sart_tvm_bayes(cls, sinogram_np, recon_reference_np, angles, eval_type="psnr"):
        sinogram_np = normlize(sinogram_np)
        recon_reference_np = normlize(recon_reference_np)
        from ax.service.managed_loop import optimize
        def eval(params):
            iter = params["iter"]
            relaxation = params["relaxation"]
            tv_weight = params["tv_weight"]
            tv_max_iter = params["tv_max_iter"]
            # print("sinogram_np:", sinogram_np.max(), sinogram_np.min())
            # print("recon_reference_np", recon_reference_np.max(), recon_reference_np.min())
            recon_sartvm = cls.sart_tvm(sinogram_np, angles,
                                        iter=iter, relaxation=relaxation, tv_weight=tv_weight, tv_max_iter=tv_max_iter)
            recon_sartvm = normlize(recon_sartvm)
            # print("recon_sartvm", recon_sartvm.max(), recon_sartvm.min())
            if eval_type == "psnr":
                evaluate = Eval.psnr(recon_sartvm, recon_reference_np)
            elif eval_type == "ssim":
                evaluate = Eval.ms_ssim(recon_sartvm, recon_reference_np)
            print(evaluate, iter, relaxation, tv_weight, tv_max_iter)
            return evaluate

        best_parameters, values, experiment, model = optimize(
            parameters=[
                # 注意！！！一定不要搜索初始的随机数种子！！
                # name 参数名 ， type， 搜索类型（range范围搜索）， bounnds（上下界，写出小数点，即按照浮点数搜索）
                # 参数名在eval函数中做key值， 搜索类型还有其他的，参照官网，
                {"name": "iter", "type": "range", "bounds": [0, 30]},
                {"name": "tv_max_iter", "type": "range", "bounds": [0, 10]},
                {"name": "relaxation", "type": "range", "bounds": [0.1, 1.0]},
                {"name": "tv_weight", "type": "range", "bounds": [0.1, 1.0]},
            ],
            #
            # parameter_constraints=["x1 + x2 <= 10.0"],
            # 对于输出结果的约束。例如：优化后，末尾5%用户RSRP不低于XXX，可以用在这。
            outcome_constraints=[],
            evaluation_function=eval,
            objective_name='psnr',
            total_trials=30
        )
        print(best_parameters, values)

    @staticmethod
    def sart_tvm(sinogram_np, angles, iter=8, relaxation=0.5598, tv_weight=0.667, tv_max_iter=3):
        sinogram_np = sinogram_np.astype(np.float)
        recon_SARTVM = None
        for i in range(iter):
            recon_SARTVM = iradon_sart(sinogram_np, theta=angles, image=recon_SARTVM, relaxation=relaxation)
            recon_SARTVM = denoise_tv_bregman(recon_SARTVM, weight=tv_weight, max_iter=tv_max_iter)
            # frames.append(recon_SART)
            # now_error = np.mean((recon_SART - P) ** 2)
            # if abs(last_error - now_error) > epsilon:
            # print("ERROR:%04f" % now_error)
            # last_error = now_error
            # else:
            #     print("Finish!")
            #     break
        return recon_SARTVM


def analysis_sinorgram(root, name, sinogram, angles, show=True):
    if sinogram.max() <= 1:
        sinogram = sinogram * 255.0
    recon_wbp = Recon.wbp(sinogram, angles)
    save_np2img(os.path.join(root, "recon_wbp", name), recon_wbp, norm=True, show=show)
    save_np2img(os.path.join(root, "recon_wbp_fft", name), fft(recon_wbp), norm=True, show=show)

    recon_sart = Recon.sart(sinogram, angles)
    save_np2img(os.path.join(root, "recon_sart", name), recon_sart, norm=True, show=show)
    save_np2img(os.path.join(root, "recon_sart_fft", name), fft(recon_sart), norm=True, show=show)

    recon_sart_tvm = Recon.sart_tvm(sinogram, angles)
    save_np2img(os.path.join(root, "recon_tvm", name), recon_sart_tvm, norm=True, show=show)
    save_np2img(os.path.join(root, "recon_tvm_fft", name), fft(recon_sart_tvm), norm=True, show=show)
    return recon_wbp, recon_sart, recon_sart_tvm


def save_sinogram(sinogram, root, filename, show=False):
    if sinogram.max() <= 1:
        sinogram = sinogram * 255.0
    img = Image.fromarray(sinogram)
    if show:
        img.show()
    img.convert("L").save(os.path.join(root, filename))


class Eval():

    @staticmethod
    def snr(fake, real):
        fake_sq = (fake ** 2).sum()
        mse = ((fake - real) ** 2).sum()
        if mse < 1e-8:
            mse = 1e-8
        if fake_sq < 1e-8:
            fake_sq = 1e-8
        snr = 10 * log10(fake_sq / mse + 1e-8)
        return snr

    @staticmethod
    def psnr(fake, real):
        # print(fake.max(),real.max())
        fake_sq = (fake ** 2).mean()
        snr = Eval.snr(fake, real)
        psnr = snr + 10 * log10(255 ** 2 / fake_sq + 1e-8)
        return psnr

    @classmethod
    def ms_ssim(cls, img_1, img_2, levels=3):
        weight = torch.Tensor([0.0448, 0.2856, 0.3001, 0.2363, 0.1333])
        msssim = torch.Tensor(levels, )
        mcs = torch.Tensor(levels, )
        if not isinstance(img_1, torch.Tensor):
            img1 = torch.Tensor(img_1)
            img2 = torch.Tensor(img_2)
            img1 = img1.unsqueeze(0)
            img1 = img1.unsqueeze(0)
            img2 = img2.unsqueeze(0)
            img2 = img2.unsqueeze(0)

        for i in range(levels):
            ssim_map, mcs_map = cls._ssim(img1, img2, 11, True)
            msssim[i] = ssim_map
            mcs[i] = mcs_map
            filtered_im1 = F.avg_pool2d(img1, kernel_size=2, stride=2)
            filtered_im2 = F.avg_pool2d(img2, kernel_size=2, stride=2)
            img1 = filtered_im1
            img2 = filtered_im2

        value = (
                torch.prod(abs(mcs[0:levels - 1]) ** weight[0:levels - 1]) *
                (msssim[levels - 1] ** weight[levels - 1])
        )
        return value.cpu().numpy()

    @classmethod
    def _ssim(cls, img1, img2, window_size, size_average=True):
        if len(img1.size()) == 4:
            (_, channel, _, _) = img1.size()
        else:
            raise ValueError("rank of `img1` is %d, it should be 4" % len(img1.size()))
        window = cls._create_window(window_size, channel)

        if img1.is_cuda:
            window = window.cuda(img1.get_device())
        window = window.type_as(img1)

        mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
        mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        V1 = 2.0 * sigma12 + C2
        V2 = sigma1_sq + sigma2_sq + C2
        ssim_map = ((2 * mu1_mu2 + C1) * V1) / ((mu1_sq + mu2_sq + C1) * V2)
        mcs_map = V1 / V2
        if size_average:
            return ssim_map.mean(), mcs_map.mean()
        else:
            return ssim_map.mean(1).mean(1).mean(1), mcs_map.mean()

    @staticmethod
    def _gaussian(window_size, sigma):
        gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
        return gauss / gauss.sum()

    @classmethod
    def _create_window(cls, window_size, channel):
        _1D_window = cls._gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
        return window

    @staticmethod
    def evaluate(targets_dic, filername, eval_funcs, reference):
        with open("img_show/eval_results.csv", mode="a") as result:
            result.write(filername)
            for name, target in targets_dic.items():
                for fuc in eval_funcs:
                    value = fuc(target, reference)
                    result.write("," + str(value))
            result.write("\n")


if __name__ == '__main__':
    gpus = []
    if len(sys.argv) > 1:
        nums = int(sys.argv[1])
        gpus = list(range(nums))
    else:
        gpus = []


    weights_path = "log_Dataset/Radon_inpaint_4/checkpoint/Weights_netG_25.pth"
    dense_angles = np.arange(0., 180., 1.40625)  # 128 ， 128 +16 +16
    angles = np.concatenate((np.arange(0, 67.5, 1.40625), np.arange(112.5, 180, 1.40625)))

    model = prepare_model(weights_path, gpus)
    clear_all_images()
    # original_imgs = os.listdir("data/original_img")[-100:]
    original_imgs = os.listdir("img_show/original")

    with open("img_show/eval_results.csv", mode="w") as result:
        result.write(",".join(["filename",
                               "inpaint_wbp_psnr", "inpaint_wbp_snr", "inpaint_wbp_msssim",
                               "inpaint_sart_psnr", "inpaint_sart_snr", "inpaint_wbp_msssim",
                               "inpaint_tvm_psnr", "inpaint_tvm_snr", "inpaint_wbp_msssim",
                               "narrow_wbp_psnr", "narrow_wbp_snr", "narrow_wbp_msssim",
                               "narrow_sart_psnr", "narrow_sart_snr", "narrow_wbp_msssim",
                               "narrow_tvm_psnr", "narrow_tvm_snr", "narrow_wbp_msssim", "\n"]))

    for name in tqdm(original_imgs):
        path = os.path.join("img_show/original", name)
        with Image.open(path) as img:
            img_np = np.asarray(img.convert("L"))
            disdfgram_np = fft(img_np)
            save_np2img(os.path.join("img_show/original_fft", name), disdfgram_np, norm=True, show=False)
        narrow_sinogram = radon(img_np / img_np.max(), theta=angles, circle=True)  # radon 0.255.
        wide_sinogram = radon(img_np / img_np.max(), theta=dense_angles, circle=True)  # radon 0.255.
        input_sinogram = porcess_input(narrow_sinogram, name)
        save_sinogram(narrow_sinogram, "img_show/narrow/sinogram", name, show=False)
        save_sinogram(wide_sinogram, "img_show/wide/sinogram", name, show=False)
        save_sinogram(input_sinogram.cpu().detach().numpy()[0][0], "img_show/inpaint/input", name, show=False)
        with torch.no_grad():
            output_sinogram = model(input_sinogram).cpu().detach().numpy()[0][0]
            output_sinogram[:, :48] = input_sinogram[:, :, :, :48]
            output_sinogram[:, 80:] = input_sinogram[:, :, :, 80:]
        img = Image.fromarray(output_sinogram * 255)
        img.convert("L").save(os.path.join('img_show/inpaint/sinogram', name))

        reference, _, _ = analysis_sinorgram("img_show/wide", name, wide_sinogram, dense_angles, False)
        inpaint_wbp, inpaint_sart, inpaint_tvm = analysis_sinorgram("img_show/inpaint", name, output_sinogram,
                                                                    dense_angles,
                                                                    False)
        narrow_wbp, narrow_sart, narrow_tvm = analysis_sinorgram("img_show/narrow", name, narrow_sinogram, angles,
                                                                 False)
        # Recon.sart_tvm_bayes(narrow_sinogram, reference, angles)
        # exit(1)
        Eval.evaluate({"inpaint_wbp": inpaint_wbp,
                       "inpaint_sart": inpaint_sart,
                       "inpaint_tvm": inpaint_tvm,
                       "narrow_wbp": narrow_wbp,
                       "narrow_sart": narrow_sart,
                       "narrow_tvm": narrow_tvm,
                       },
                      name,
                      [Eval.psnr, Eval.snr, Eval.ms_ssim],
                      reference)
