from PIL import Image
import sys
import numpy as np

path_t = r"data\tiltser\0000.npy"
path_r = r"data\recon_bp\0000.npy"
def show_img(path):
    (Image.fromarray(np.load(path))).show()
show_img(path_t)
show_img(path_r)
