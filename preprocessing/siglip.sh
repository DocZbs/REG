#!/bin/bash

# Multi-GPU SigLIP feature extraction with accelerate
# Usage: bash siglip.sh

NUM_GPUS=4  # Adjust based on your available GPUs

accelerate launch \
    --multi_gpu \
    --num_processes $NUM_GPUS \
    preprocessing/dataset_image_to_siglip.py \
    --source=/mnt/nvme-fast/datasets/imagenet-2010 \
    --dest=/mnt/nvme-fast/datasets/imagenet_siglip \
    --model-name=google/siglip-so400m-patch14-384 \
    --batch-size=64 \
    --save-patch-tokens  # Add this flag to save patch tokens (vision_outputs[0])