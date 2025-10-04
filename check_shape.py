import numpy as np

def load_data(dir):
    ft = np.load(dir)
    print(ft.shape)

dir = "/mnt/nvme-fast/datasets/imagenet_siglip/00000/img-feature-00000236.npy"

load_data(dir)