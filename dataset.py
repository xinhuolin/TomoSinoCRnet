from torch.utils.data import Dataset, random_split
from torchvision import transforms
from PIL import Image
import os
import numpy as np
from jdit.dataset import DataLoadersFactory
import skimage


class _RadonDataset(Dataset):
    def __init__(self, transform, root="data"):
        self.transform = transforms.Compose(transform)
        names = os.listdir(r"%s/tiltser" % root)
        self.paths_input = [os.path.join(r"%s/tiltser" % root, name) for name in names]
        self.paths_groundtruth = [os.path.join(r"%s/dense_tiltser" % root, name) for name in names]
        self.total = len(self.paths_input)

    def __len__(self):
        return len(self.paths_input)

    def __getitem__(self, i):
        path_input = self.paths_input[i]
        path_output = self.paths_groundtruth[i]

        img_input = Image.fromarray(np.load(path_input))
        img_input = self.transform(img_input)

        img_groundtruth = Image.fromarray(np.load(path_output))
        img_groundtruth = self.transform(img_groundtruth)
        return img_input, img_groundtruth


class RadonDatasets(DataLoadersFactory):
    def __init__(self, root, batch_size=32, num_workers=-1, valid_size=100, shuffle=True, ):
        self.valid_size = valid_size
        super(RadonDatasets, self).__init__(root, batch_size, num_workers=num_workers, shuffle=shuffle,
                                            subdata_size=valid_size)

    def build_datasets(self):
        randon_dataset = _RadonDataset(self.test_transform_list)
        self.dataset_train, self.dataset_valid = random_split(randon_dataset,
                                                              [randon_dataset.total - self.valid_size, self.valid_size])
        print("train size:%d    valid size:%d" % (randon_dataset.total - self.valid_size, self.valid_size))
        self.dataset_test = self.dataset_valid

    def build_transforms(self, resize=256):
        self.train_transform_list = self.vaild_transform_list = self.test_transform_list = [
            transforms.ToTensor(),  # (H x W x C)=> (C x H x W)
            transforms.Normalize([0.], [255.])
        ]


# ============================================
class _RadonInpaintDataset(Dataset):
    def __init__(self, input_transform, output_transform, root=r"/home/dgl/dataset"):
        self.input_transform = transforms.Compose(input_transform)
        self.output_transform = transforms.Compose(output_transform)
        # names = os.listdir(r"%s/tiltser" % root)
        # self.paths_input = [os.path.join(r"%s/tiltser" % root, name) for name in names]
        # self.paths_groundtruth = [os.path.join(r"%s/dense_tiltser" % root, name) for name in names]

        ## narrow_sinogram_pad_96
        print(r"%s/narrow_sinogram_pad_96" % root)
        names = os.listdir(r"%s/narrow_sinogram_pad_96" % root)
        self.paths_input = [os.path.join(r"%s/narrow_sinogram_pad_96" % root, name) for name in names]
        self.paths_groundtruth = [os.path.join(r"%s/wide_sinogram_128" % root, name) for name in names]
        self.total = len(self.paths_input)

    def __len__(self):
        return len(self.paths_input)

    def __getitem__(self, i):
        path_input = self.paths_input[i]
        path_output = self.paths_groundtruth[i]

        img_input = Image.fromarray(np.load(path_input))
        img_input = self.input_transform(img_input)
        img_groundtruth = Image.fromarray(np.load(path_output))
        img_groundtruth = self.output_transform(img_groundtruth)
        return img_input, img_groundtruth


class RadonInnerInpaintDatasets(DataLoadersFactory):
    def __init__(self, root, batch_size=32, hor_pad=16, num_workers=-1, valid_size=100, shuffle=False):
        self.valid_size = valid_size
        self.hor_pad = hor_pad
        self.root = root
        super(RadonInnerInpaintDatasets, self).__init__(root, batch_size, num_workers=num_workers, shuffle=shuffle,
                                                        subdata_size=valid_size)

    def build_datasets(self):
        randon_dataset = _RadonInpaintDataset(self.train_transform_list_input, self.train_transform_list,
                                              root=self.root+"/train")
        self.dataset_train, self.dataset_valid = random_split(randon_dataset,
                                                              [randon_dataset.total - self.valid_size, self.valid_size])
        print("train size:%d    valid size:%d" % (randon_dataset.total - self.valid_size, self.valid_size))
        self.dataset_test = _RadonInpaintDataset(self.train_transform_list_input, self.train_transform_list,
                                              root=self.root+"/test")

    def build_transforms(self, resize=256):
        self.train_transform_list = self.vaild_transform_list = self.test_transform_list = [
            transforms.ToTensor(),  # (H x W x C)=> (C x H x W)
            transforms.Normalize([0.], [255.])
        ]
        self.train_transform_list_input = self.vaild_transform_list_input = self.test_transform_list_input = [
            # InnerPad(32),
            transforms.ToTensor(),  # (H x W x C)=> (C x H x W)
            transforms.Normalize([0.], [255.])
        ]

#
# class AddNoise():
#     def __init__(self, mode, mean=0, var=0.002):
#         self.mode = mode
#         self.mean = mean
#         self.var = var
#
#     def __call__(self, tensor):
#         # tensor (0, 1)
#         if self.mode == "gaussian":
#             tensor = skimage.util.random_noise(tensor, mode=self.mode, mean=self.mean, var=self.var)
#         else:
#             tensor = skimage.util.random_noise(tensor, mode=self.mode)
#
#         return Image.fromarray(tensor)


# class InnerPad():
#     def __init__(self, inner_width=32):
#         self.inner_width = inner_width
#
#     def __call__(self, img):
#         width, hight = img.size
#         np_img = np.asarray(img)
#         np_zeropad = np.zeros((self.inner_width, hight))
#         # 96//2 =  48     angles_128[48:80]
#         result_np = np.insert(np_img, width // 2, values=np_zeropad, axis=1)
#         return Image.fromarray(result_np)

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