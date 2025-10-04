python preprocessing/dataset_image_to_siglip.py \
    --source=/mnt/nvme-fast/datasets/imagenet-2010 \
    --dest=/mnt/nvme-fast/datasets/imagenet_siglip \
    --model-name=google/siglip-so400m-patch14-384 \
    --batch-size=256