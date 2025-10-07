# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

"""Multi-GPU image encoding utility with Accelerate support."""

from __future__ import annotations

import io
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, List, Optional, Sequence, Tuple

import click
import numpy as np
import PIL.Image
import torch
from accelerate import Accelerator
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# Ensure repository root is importable so we can reuse helper utilities.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in os.sys.path:
    os.sys.path.insert(0, str(REPO_ROOT))

from preprocessing.dataset_image_encoder import preprocess_raw_image  # reuse normalisation
from utils import load_encoders


@dataclass
class ImageRecord:
    path: str
    label: Optional[int]
    index: int


def _is_image(fname: str) -> bool:
    try:
        ext = Path(fname).suffix.lower()
        return ext in PIL.Image.EXTENSION
    except Exception:
        return False


def collect_image_records(source: str, *, max_images: Optional[int]) -> List[ImageRecord]:
    """Walk ``source`` (folder or zip) and collect image paths with labels."""

    def _scan_folder(folder: str) -> Sequence[str]:
        files: list[str] = []
        for root, _dirs, filenames in os.walk(folder):
            for name in filenames:
                full = os.path.join(root, name)
                if _is_image(full):
                    files.append(full)
        files.sort()
        return files

    if os.path.isdir(source):
        image_files = _scan_folder(source)
    else:
        raise click.ClickException('Only directory inputs are supported for Accelerate encoding.')

    if max_images is not None:
        image_files = image_files[:max_images]

    # Derive labels from optional dataset.json for compatibility with existing tooling.
    labels_map: dict[str, int] = {}
    meta_path = os.path.join(source, 'dataset.json')
    if os.path.isfile(meta_path):
        with open(meta_path, 'r') as file:
            data = json.load(file).get('labels')
            if data is not None:
                labels_map = {entry[0]: entry[1] for entry in data}

    records: list[ImageRecord] = []
    for idx, path in enumerate(image_files):
        rel = os.path.relpath(path, source).replace('\\', '/')
        label = labels_map.get(rel)
        records.append(ImageRecord(path=path, label=label, index=idx))
    return records


class ImageTensorDataset(Dataset):
    """Dataset that loads raw RGB images and keeps track of global index."""

    def __init__(self, samples: Sequence[ImageRecord]):
        self.samples = list(samples)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, int]:
        record = self.samples[idx]
        image = np.array(PIL.Image.open(record.path).convert('RGB'))
        tensor = torch.from_numpy(image).permute(2, 0, 1).float()  # [C, H, W], 0-255
        label = record.label if record.label is not None else -1
        return tensor, label, record.index


def _ensure_dir(path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)


def _empty_cache() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _split_batches(tensor: torch.Tensor, chunk: Optional[int]) -> Iterable[torch.Tensor]:
    if chunk is None or tensor.size(0) <= chunk:
        yield tensor
        return
    for start in range(0, tensor.size(0), chunk):
        yield tensor[start:start + chunk]


@click.command()
@click.option('--source', type=str, required=True, help='Directory with raw images.')
@click.option('--dest', type=str, required=True, help='Output directory for encoded features.')
@click.option('--enc-type', type=str, required=True, help='Encoder spec, e.g. siglip-so400m-patch14-384.')
@click.option('--resolution', type=int, default=256, help='Input resolution used for the encoder pipeline.')
@click.option('--batch-size', type=int, default=32, help='Batch size per process for loading images.')
@click.option('--micro-batch-size', type=int, default=None,
              help='Optional micro batch size for forward passes to limit VRAM usage.')
@click.option('--max-images', type=int, default=None, help='Optionally cap number of images processed.')
@click.option('--save-cls', is_flag=True, default=True, help='Save pooled/CLS outputs when available.')
def main(
    source: str,
    dest: str,
    enc_type: str,
    resolution: int,
    batch_size: int,
    micro_batch_size: Optional[int],
    max_images: Optional[int],
    save_cls: bool,
) -> None:
    accelerator = Accelerator()
    device = accelerator.device

    if accelerator.is_main_process:
        if dest == '':
            raise click.ClickException('--dest must not be empty')
        os.makedirs(dest, exist_ok=True)

    accelerator.wait_for_everyone()

    encoders, encoder_types, architectures = load_encoders(enc_type, device, resolution)
    encoder = encoders[0]
    encoder_type = encoder_types[0]
    encoder.eval()

    # Only load image list once and broadcast.
    if accelerator.is_main_process:
        samples = collect_image_records(source, max_images=max_images)
    else:
        samples = None

    obj_list = [samples]
    accelerator.broadcast_object_list(obj_list)
    samples = obj_list[0]
    assert samples is not None, 'Failed to broadcast samples list.'

    dataset = ImageTensorDataset(samples)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        drop_last=False,
    )
    dataloader = accelerator.prepare(dataloader)

    local_labels: list[Optional[Tuple[str, Optional[int]]]] = []

    for batch in tqdm(dataloader, disable=not accelerator.is_main_process, desc='Encoding'):
        images, label_batch, idx_batch = batch
        images = images.to(device, non_blocking=True)
        labels_np = label_batch.cpu().numpy()
        indices_np = idx_batch.cpu().numpy()

        cls_outputs: list[np.ndarray] = []
        patch_outputs: list[np.ndarray] = []

        with torch.no_grad():
            for micro in _split_batches(images, micro_batch_size):
                processed = preprocess_raw_image(micro, encoder_type).to(device)
                outputs = encoder.forward_features(processed)

                cls_tensor = None
                features = outputs
                if isinstance(outputs, dict):
                    cls_tensor = outputs.get('pooled_output')
                    if 'patch_tokens' in outputs:
                        features = outputs['patch_tokens']
                    elif 'x_norm_patchtokens' in outputs:
                        features = outputs['x_norm_patchtokens']
                    elif 'last_hidden_state' in outputs:
                        features = outputs['last_hidden_state']
                elif hasattr(outputs, 'last_hidden_state'):
                    features = outputs.last_hidden_state

                if encoder_type == 'siglip':
                    # SigLIP returns CLS in the pooled output; keep full token grid.
                    pass

                features_cpu = features.detach().to('cpu')
                patch_outputs.extend(torch.unbind(features_cpu, dim=0))

                if save_cls and cls_tensor is not None:
                    cls_cpu = cls_tensor.detach().to('cpu')
                    cls_outputs.extend(torch.unbind(cls_cpu, dim=0))
                else:
                    cls_outputs.extend([None] * features_cpu.size(0))

                del processed, outputs, features_cpu
                _empty_cache()

        for patch_tensor, cls_tensor, label, global_idx in zip(
            patch_outputs, cls_outputs, labels_np, indices_np
        ):
            idx_str = f'{int(global_idx):08d}'
            patch_fname = f'{idx_str[:5]}/img-patches-{idx_str}.npy'
            patch_path = os.path.join(dest, patch_fname)
            _ensure_dir(patch_path)

            patch_np = patch_tensor.numpy()
            np.save(patch_path, patch_np)

            if save_cls and cls_tensor is not None:
                cls_fname = f'{idx_str[:5]}/img-cls-{idx_str}.npy'
                cls_path = os.path.join(dest, cls_fname)
                _ensure_dir(cls_path)
                np.save(cls_path, cls_tensor.numpy())

            label_value = int(label) if label >= 0 else None
            local_labels.append([patch_fname, label_value] if label_value is not None else None)

        del images
        _empty_cache()

    accelerator.wait_for_everyone()

    gathered = accelerator.gather_object(local_labels)

    if accelerator.is_main_process:
        merged: list[Optional[Tuple[str, Optional[int]]]] = []
        for chunk in gathered:
            if chunk:
                merged.extend(chunk)

        merged.sort(key=lambda x: x[0] if x is not None else '')

        metadata = {'labels': merged if all(x is not None for x in merged) else None}
        meta_path = os.path.join(dest, 'dataset.json')
        with open(meta_path, 'w') as f:
            json.dump(metadata, f)

    accelerator.wait_for_everyone()


if __name__ == '__main__':
    main()
