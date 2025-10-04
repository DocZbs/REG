




#256
python preprocessing/dataset_tools.py convert \
    --source=/mnt/nvme-fast/datasets/imagenet-2010\
    --dest=/mnt/nvme-fast/datasets/imagenet_256_vae \
    --resolution=256x256 \
    --transform=center-crop-dhariwal