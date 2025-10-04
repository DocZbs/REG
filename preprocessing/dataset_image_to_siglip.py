# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

"""Tool for encoding images using SigLIP."""

from collections.abc import Iterator
from dataclasses import dataclass
import functools
import io
import json
import os
import re
import zipfile
from pathlib import Path
from typing import Callable, Optional, Tuple, Union
import click
import numpy as np
import PIL.Image
import torch
from tqdm import tqdm
from torchvision.transforms import Normalize, Compose, Resize, CenterCrop, ToTensor

SIGLIP_DEFAULT_MEAN = (0.48145466, 0.4578275, 0.40821073)
SIGLIP_DEFAULT_STD = (0.26862954, 0.26130258, 0.27577711)

@dataclass
class ImageEntry:
    img: np.ndarray
    label: Optional[int]

def parse_tuple(s: str) -> Tuple[int, int]:
    m = re.match(r'^(\d+)[x,](\d+)$', s)
    if m:
        return int(m.group(1)), int(m.group(2))
    raise click.ClickException(f'cannot parse tuple {s}')

def maybe_min(a: int, b: Optional[int]) -> int:
    if b is not None:
        return min(a, b)
    return a

def file_ext(name: Union[str, Path]) -> str:
    return str(name).split('.')[-1]

def is_image_ext(fname: Union[str, Path]) -> bool:
    ext = file_ext(fname).lower()
    return f'.{ext}' in PIL.Image.EXTENSION

def open_image_folder(source_dir, *, max_images: Optional[int]) -> tuple[int, Iterator[ImageEntry]]:
    input_images = []
    def _recurse_dirs(root: str):
        with os.scandir(root) as it:
            for e in it:
                if e.is_file():
                    input_images.append(os.path.join(root, e.name))
                elif e.is_dir():
                    _recurse_dirs(os.path.join(root, e.name))
    _recurse_dirs(source_dir)
    input_images = sorted([f for f in input_images if is_image_ext(f)])

    arch_fnames = {fname: os.path.relpath(fname, source_dir).replace('\\', '/') for fname in input_images}
    max_idx = maybe_min(len(input_images), max_images)

    labels = dict()
    meta_fname = os.path.join(source_dir, 'dataset.json')
    if os.path.isfile(meta_fname):
        with open(meta_fname, 'r') as file:
            data = json.load(file)['labels']
            if data is not None:
                labels = {x[0]: x[1] for x in data}

    if len(labels) == 0:
        toplevel_names = {arch_fname: arch_fname.split('/')[0] if '/' in arch_fname else '' for arch_fname in arch_fnames.values()}
        toplevel_indices = {toplevel_name: idx for idx, toplevel_name in enumerate(sorted(set(toplevel_names.values())))}
        if len(toplevel_indices) > 1:
            labels = {arch_fname: toplevel_indices[toplevel_name] for arch_fname, toplevel_name in toplevel_names.items()}

    def iterate_images():
        for idx, fname in enumerate(input_images):
            img = np.array(PIL.Image.open(fname).convert('RGB'))
            yield ImageEntry(img=img, label=labels.get(arch_fnames[fname]))
            if idx >= max_idx - 1:
                break
    return max_idx, iterate_images()

def open_image_zip(source, *, max_images: Optional[int]) -> tuple[int, Iterator[ImageEntry]]:
    with zipfile.ZipFile(source, mode='r') as z:
        input_images = [str(f) for f in sorted(z.namelist()) if is_image_ext(f)]
        max_idx = maybe_min(len(input_images), max_images)

        labels = dict()
        if 'dataset.json' in z.namelist():
            with z.open('dataset.json', 'r') as file:
                data = json.load(file)['labels']
                if data is not None:
                    labels = {x[0]: x[1] for x in data}

    def iterate_images():
        with zipfile.ZipFile(source, mode='r') as z:
            for idx, fname in enumerate(input_images):
                with z.open(fname, 'r') as file:
                    img = np.array(PIL.Image.open(file).convert('RGB'))
                yield ImageEntry(img=img, label=labels.get(fname))
                if idx >= max_idx - 1:
                    break
    return max_idx, iterate_images()

def open_dataset(source, *, max_images: Optional[int]):
    if os.path.isdir(source):
        return open_image_folder(source, max_images=max_images)
    elif os.path.isfile(source):
        if file_ext(source) == 'zip':
            return open_image_zip(source, max_images=max_images)
        else:
            raise click.ClickException(f'Only zip archives are supported: {source}')
    else:
        raise click.ClickException(f'Missing input file or directory: {source}')

def open_dest(dest: str) -> Tuple[str, Callable[[str, Union[bytes, str]], None], Callable[[], None]]:
    dest_ext = file_ext(dest)

    if dest_ext == 'zip':
        if os.path.dirname(dest) != '':
            os.makedirs(os.path.dirname(dest), exist_ok=True)
        zf = zipfile.ZipFile(file=dest, mode='w', compression=zipfile.ZIP_STORED)
        def zip_write_bytes(fname: str, data: Union[bytes, str]):
            zf.writestr(fname, data)
        return '', zip_write_bytes, zf.close
    else:
        if os.path.isdir(dest) and len(os.listdir(dest)) != 0:
            raise click.ClickException('--dest folder must be empty')
        os.makedirs(dest, exist_ok=True)

        def folder_write_bytes(fname: str, data: Union[bytes, str]):
            os.makedirs(os.path.dirname(fname), exist_ok=True)
            with open(fname, 'wb') as fout:
                if isinstance(data, str):
                    data = data.encode('utf8')
                fout.write(data)
        return dest, folder_write_bytes, lambda: None

def load_siglip_model(model_name: str, device: torch.device):
    from transformers import AutoProcessor, AutoModelForZeroShotImageClassification
    
    processor = AutoProcessor.from_pretrained(model_name)
    model = AutoModelForZeroShotImageClassification.from_pretrained(model_name).to(device)
    
    model.eval()
    return model, processor

def preprocess_image_for_siglip(img: np.ndarray, processor) -> torch.Tensor:
    pil_img = PIL.Image.fromarray(img)
    inputs = processor(images=pil_img, return_tensors="pt")
    return inputs['pixel_values']

@click.command()
@click.option('--source', help='Input directory or archive name', metavar='PATH', type=str, required=True)
@click.option('--dest', help='Output directory or archive name', metavar='PATH', type=str, required=True)
@click.option('--max-images', help='Maximum number of images to output', metavar='INT', type=int)
@click.option('--model-name', help='SigLIP model name from HuggingFace', metavar='STR', type=str, default='google/siglip-so400m-patch14-384')
@click.option('--batch-size', help='Batch size for encoding', metavar='INT', type=int, default=32)
def encode(
    source: str,
    dest: str,
    max_images: Optional[int],
    model_name: str,
    batch_size: int
):
    """Encode images using SigLIP model."""
    
    PIL.Image.init()
    if dest == '':
        raise click.ClickException('--dest output filename or directory must not be an empty string')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Loading SigLIP model: {model_name}")
    model, processor = load_siglip_model(model_name, device)
    print("Model loaded successfully!")

    num_files, input_iter = open_dataset(source, max_images=max_images)
    archive_root_dir, save_bytes, close_dest = open_dest(dest)
    print(f"Processing {num_files} images...")

    labels = []
    image_batch = []
    image_entries = []
    idx_batch = []

    def process_batch(batch_images, batch_entries, batch_indices):
        with torch.no_grad():
            pixel_values = torch.cat(batch_images, dim=0).to(device)
            vision_outputs = model.vision_model(pixel_values=pixel_values)
            image_embeds = vision_outputs[1]
            image_embeds = model.vision_model.post_layernorm(image_embeds)
            
            for i, (feature, entry, idx) in enumerate(zip(image_embeds, batch_entries, batch_indices)):
                feature_np = feature.cpu().numpy()
                idx_str = f'{idx:08d}'
                archive_fname = f'{idx_str[:5]}/img-feature-{idx_str}.npy'

                f = io.BytesIO()
                np.save(f, feature_np)
                save_bytes(os.path.join(archive_root_dir, archive_fname), f.getvalue())
                labels.append([archive_fname, entry.label] if entry.label is not None else None)

    for idx, image_entry in tqdm(enumerate(input_iter), total=num_files):
        pixel_values = preprocess_image_for_siglip(image_entry.img, processor)
        image_batch.append(pixel_values)
        image_entries.append(image_entry)
        idx_batch.append(idx)

        if len(image_batch) >= batch_size:
            process_batch(image_batch, image_entries, idx_batch)
            image_batch = []
            image_entries = []
            idx_batch = []

    if len(image_batch) > 0:
        process_batch(image_batch, image_entries, idx_batch)

    metadata = {'labels': labels if all(x is not None for x in labels) else None}
    save_bytes(os.path.join(archive_root_dir, 'dataset.json'), json.dumps(metadata))
    close_dest()
    
    print(f"Encoding completed! Output saved to {dest}")

if __name__ == "__main__":
    encode()
