import numpy as np


dir = "/mnt/nvme-fast/datasets/imagenet_siglip/00000/img-feature-00000499.npy"

def load_siglip_feature(dir):
    with open(dir, "rb") as f:
        return np.load(f)

def main():
    feature = load_siglip_feature(dir)
    print(feature.shape)
    print(feature.dtype)
    print(feature.min())
    print(feature.max())
    
if __name__ == "__main__":
    main()






