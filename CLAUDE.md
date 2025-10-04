# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

REG (Representation Entanglement for Generation) is a research implementation for training Diffusion Transformers on ImageNet. The method entangles low-level image latents with high-level class tokens from pretrained foundation models during denoising, achieving significant training acceleration (63× faster than baseline SiT-XL/2).

**Key Paper**: "Representation Entanglement for Generation: Training Diffusion Transformers Is Much Easier Than You Think" (NeurIPS 2025 Oral)

## Architecture Overview

### Core Pipeline
1. **VAE Encoding**: Images are encoded to latent space using Stable Diffusion VAE (4-channel latents at 32×32 for 256×256 images)
2. **Foundation Model Features**: Extract dense features + class tokens from pretrained models (DINOv2, CLIP)
3. **Entangled Denoising**: SiT model denoises both image latents AND class tokens jointly
4. **Loss Components**:
   - Denoising loss for image latents (v-prediction)
   - Denoising loss for class tokens (controlled by `--cls` coefficient, default 0.03)
   - Projection loss for feature alignment (controlled by `--proj-coeff`, default 0.5)

### Key Model Components

**SiT Architecture** (`models/sit.py`):
- Standard transformer blocks with adaptive layer norm (adaLN-Zero)
- Class token is prepended to patch tokens and flows through all layers
- Projectors extract features at encoder depth (depth 8 for L/XL, depth 4 for B)
- Final layer outputs both denoised latents AND denoised class token

**Loss Function** (`loss.py:SILoss`):
- Supports linear/cosine interpolation paths
- V-prediction formulation: `v = d_alpha_t * x + d_sigma_t * noise`
- Projection loss uses normalized cosine similarity between predicted and target features

**Sampling** (`samplers.py:euler_maruyama_sampler`):
- SDE-based Euler-Maruyama sampler
- Dual classifier-free guidance: separate scales for image (`--cfg-scale`) and class token (`--cls-cfg-scale`)
- Guidance interval controlled by `--guidance-low` and `--guidance-high`

## Commands

### Environment Setup
```bash
conda create -n reg python=3.10.16 -y
conda activate reg
pip install torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1
pip install -r requirements.txt
```

### Data Preprocessing

**Convert ImageNet to 256×256 ZIP archive**:
```bash
cd preprocessing
bash dataset_prepare_convert.sh
```
Edit the script to set:
- `--source`: Path to raw ImageNet dataset
- `--dest`: Output path for processed images
- `--resolution=256x256` (or `512x512`)

**Encode to VAE latents**:
```bash
bash dataset_prepare_encode.sh
```
Edit to set source (processed images) and dest (VAE latents) paths.

**Expected directory structure** after preprocessing:
```
data_path/imagenet_vae/
├── imagenet_256_vae/     # RGB images (256×256)
└── vae-sd/               # VAE latent vectors (32×32×4)
    └── dataset.json      # Contains label mappings
```

### Training

**Basic training command**:
```bash
bash train.sh
```

**Key training arguments**:
- `--model`: `SiT-B/2`, `SiT-L/2`, `SiT-XL/2` (depth/hidden_size: B=12/768, L=24/1024, XL=28/1152)
- `--enc-type`: Foundation model type (`dinov2-vit-b`, `clip-vit-L`, etc.)
- `--encoder-depth`: Layer to extract projections (4 for SiT-B, 8 for SiT-L/XL)
- `--proj-coeff`: Weight for projection loss (default 0.5)
- `--cls`: Weight for class token denoising loss (default 0.03)
- `--batch-size`: Global batch size across all GPUs (e.g., 256)
- `--data-dir`: Path to preprocessed data directory
- `--path-type`: Interpolation schedule (`linear` or `cosine`)
- `--prediction`: Currently only supports `v` (v-prediction)

**Multi-GPU training**: Uses `accelerate` with automatic DDP setup
```bash
NUM_GPUS=8
accelerate launch --multi_gpu --num_processes $NUM_GPUS train.py [args...]
```

### Evaluation & Sampling

**Generate samples and compute FID**:
```bash
bash eval.sh
```

**Key evaluation parameters**:
- `--ckpt`: Path to checkpoint (e.g., `path/to/checkpoints/4000000.pt`)
- `--num-fid-samples`: Number of samples to generate (default 50000)
- `--mode`: Sampling mode (`sde` for Euler-Maruyama, `ode` not fully supported)
- `--num-steps`: Number of sampling steps (default 250)
- `--cfg-scale`: Classifier-free guidance scale for images (e.g., 2.3)
- `--cls-cfg-scale`: Classifier-free guidance scale for class tokens (e.g., 2.3)
- `--guidance-high`/`--guidance-low`: Guidance application interval (default 0.85/0.0)
- `--cls`: Dimension of class token (should match encoder, e.g., 768 for DINOv2-base)

**Evaluation pipeline**:
1. `generate.py` creates samples and saves as `.npz`
2. `evaluations/evaluator.py` computes FID against reference statistics

## Important Implementation Details

### Dataset (`dataset.py:CustomDataset`)
- Expects paired RGB images and VAE latents with matching filenames
- Labels loaded from `dataset.json` in the features directory
- Returns: `(raw_image, vae_latent, label)`

### Feature Extraction (`train.py:preprocess_raw_image`)
- Different normalization per encoder type:
  - CLIP: Resize to 224, normalize with CLIP stats
  - DINOv2/MAE/MoCoV3: ImageNet normalization, resize to 224 for DINOv2
  - Handles both 256×256 and 512×512 resolutions

### Class Token Handling
- DINOv2: `cls_token` extracted separately, then concatenated with patch tokens
- Class token goes through linear projection + LayerNorm before prepending to patches
- During inference, random class tokens are sampled: `torch.randn(n, cls_dim)`

### Checkpointing
- Saves every `--checkpointing-steps` (default 10000)
- Checkpoint contains: `model`, `ema`, `opt`, `args`, `steps`
- EMA model used for inference/evaluation
- Resume with `--resume-step`

### Configuration Compatibility
- `--encoder-depth` must match model depth constraints:
  - SiT-B: typically 4
  - SiT-L/XL: typically 8
- `--projector-embed-dims` in eval must match training encoder embedding dim
- VAE scale/bias hardcoded to `[0.18215] * 4` and `[0] * 4`

## Pretrained Models

- Download from [Baidu link](https://pan.baidu.com/s/1QX2p3ybh1KfNU7wsp5McWw?pwd=khpp)
- Also available on Hugging Face: `Martinser/REG`
- Official checkpoint: SiT-XL/2 + REG trained for 4M iterations (800 epochs)

## Common Gotchas

1. **Dataset paths**: `--data-dir` should point to parent containing both `imagenet_256_vae/` and `vae-sd/` subdirectories
2. **Encoder depth mismatch**: If projector parameters don't load, check `--encoder-depth` matches training config
3. **Class token dimension**: `--cls` in eval.sh must equal encoder's embedding dim (768 for base models, 1024 for large)
4. **Legacy flag**: Use `--legacy` if loading old checkpoints with different label dropout behavior
5. **Resolution**: Only 256×256 and 512×512 supported; 512×512 currently only works with DINOv2
6. **Accelerate config**: Project uses mixed precision fp16 by default; requires `--allow-tf32` for speed on Ampere GPUs

## Logging & Monitoring

- Weights & Biases integration via `--report-to="wandb"`
- Project name: "REG" (hardcoded in train.py:253)
- Logged metrics: `loss_final`, `loss_mean`, `proj_loss`, `loss_mean_cls`, `grad_norm`
- Checkpoints and logs saved to `{output_dir}/{exp_name}/`
- "你必须时刻检查你实现的功能的简洁可用性，以及你定变量名命名的合理性，用简洁且符合项目要求的风格编写代码"