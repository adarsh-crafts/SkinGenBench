# GAN Models

> Please refer to the paper to understand the abbreviations such as BSGT, ADGT, BSGN, and ADGN.

This is a guide on how to train the diffusion models used in this research with the `StyleGAn2-ADA_training.ipynb` notebook.

## Setup
### Scripts
The training notebook was modified from [dvschultz](https://github.com/dvschultz)'s repository: https://github.com/dvschultz/stylegan2-ada-pytorch

## Training setup

### Training Parameters

- `training_dataset_path` is the path to the real images after applying either of the two preprocessing techniques. `(ADGT or BSGT)`
- resume_from is the path to the pretrained model. Here, a StyleGAN2-ADA pretrained on FFHQ dataset was selected.

```python
# Set variables
dataset_path='./datasets/256x256_mel_images-basic.zip'
resume_from='./pretrained/ffhq-res256-mirror-paper256-noaug.pkl'
```

```python
# Execute training
!python train.py \
    --gpus=1 \
    --outdir=./results \
    --data=$dataset_path \
    --resume=$resume_from \
    --kimg=240 \
    --batch=32 \
    --lrate=0.0025 \
    --mirror=True \
    --target=0.7
```

### Resume Training from Checkpoint

```python
# Set variables
dataset_path='./datasets/256x256_mel_images-basic.zip'
resume_from='./results/00003-256x256_mel_images-basic-mirror-auto1-kimg240-batch16-target0.7-resumecustom/network-snapshot-000200.pkl'
last_augment_strength = 0.080 # from latest checkpoint

# Execute training
!python train.py \
    --gpus=1 \
    --outdir=./results \
    --data=$dataset_path \
    --resume=$resume_from \
    --kimg=40 \
    --batch=16 \
    --lrate=0.0025 \
    --mirror=True \
    --target=0.7 \
    --initstrength=$last_augment_strength
```

## Hardware

- **GPU**: NVIDIA L4 (22GB VRAM)

## Generate Synthetic Images
```python
!pip install opensimplex
```

- `network_path` is the path to the latest checkpoint.

```python
import os
import subprocess
from tqdm import tqdm

network_path = "./results/00000-256x256_mel_images-basic-mirror-auto1-kimg240-batch32-target0.7-resumecustom/network-snapshot-000240.pkl"
outdir = "./generated_mel_images/resolution_256x256"
truncation = 0.8

os.makedirs(outdir, exist_ok=True)

# Generate upto 5000 images starting from 0
start_seed = 0
end_seed = 5000
batch_size = 300

for batch_start in tqdm(range(start_seed, end_seed, batch_size), desc="Generating batches"):
    batch_end = min(batch_start + batch_size - 1, end_seed - 1)
    subprocess.run(
        f'python generate.py --outdir="{outdir}" --trunc={truncation} --seeds={batch_start}-{batch_end} --network="{network_path}"',
        shell=True
    )
```