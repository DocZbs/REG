#!/bin/bash

# Accelerated SigLIP feature extraction.
# Override env vars or edit paths before launch.

NUM_GPUS=${NUM_GPUS:-2}
SOURCE_DIR=${SOURCE_DIR:-/mnt/nvme-fast/datasets/imagenet_256}
DEST_DIR=${DEST_DIR:-/mnt/nvme-fast/datasets/imagenet_256_siglip}
ENC_TYPE=${ENC_TYPE:-siglip-so400m-patch14-384}
RESOLUTION=${RESOLUTION:-256}
BATCH_SIZE=${BATCH_SIZE:-32}
MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE:-8}

accelerate launch \
    --multi_gpu \
    --num_processes "${NUM_GPUS}" \
    preprocessing/dataset_image_encoder_accel.py \
    --source "${SOURCE_DIR}" \
    --dest "${DEST_DIR}" \
    --enc-type "${ENC_TYPE}" \
    --resolution ${RESOLUTION} \
    --batch-size ${BATCH_SIZE} \
    --micro-batch-size ${MICRO_BATCH_SIZE}
