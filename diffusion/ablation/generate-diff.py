import argparse
from datetime import datetime
import random
import torch

import os
from PIL import Image
from torchvision import utils
import numpy as np

from simple_diffusion.scheduler import DDIMScheduler
from simple_diffusion.model import UNet

n_timesteps = 1000
n_inference_timesteps = 250

def generate_random_alphanumeric_string(length=8):
    characters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789'
    return ''.join(random.choice(characters) for _ in range(length))

def save_images_from_tensor(images_tensor, out_dir):
    """
    images_tensor: torch.Tensor or numpy array in range [0,1], shape (N, C, H, W) or (N, H, W, C)
    Saves individual images to out_dir with unique random names.
    """
    # ensure numpy array in HWC uint8
    if isinstance(images_tensor, torch.Tensor):
        imgs = images_tensor.detach().cpu()
        # if shape is (N,C,H,W) convert to (N,H,W,C)
        if imgs.ndim == 4 and imgs.shape[1] in (1, 3):
            imgs = imgs.permute(0, 2, 3, 1).numpy()
        else:
            imgs = imgs.numpy()
    else:
        imgs = np.array(images_tensor)

    # assumed in [0,1] float -> convert to uint8
    imgs = (imgs * 255.0).round().astype("uint8")
    for img in imgs:
        # handle grayscale single channel -> PIL handles both 2D and 3D arrays
        pil_im = Image.fromarray(img)
        rnd = generate_random_alphanumeric_string()
        path = os.path.join(out_dir, f"{rnd}.jpeg")
        pil_im.save(path)

def generate_for_seed(noise_scheduler, model, device, out_dir, seed, args):
    print("Generating samples for seed:", seed)

    # Create a CPU generator because the scheduler generates CPU noise internally
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)

    with torch.no_grad():
        generated = noise_scheduler.generate(
            model,
            num_inference_steps=n_inference_timesteps,
            generator=gen,
            eta=1.0,
            batch_size=args.eval_batch_size
        )

        images = generated.get("sample")
        sample_pt = generated.get("sample_pt", None)

        # fallback if only one present
        if images is None and sample_pt is not None:
            images = sample_pt

        if images is None:
            raise RuntimeError("Scheduler.generate returned no image tensors (keys 'sample'/'sample_pt' missing).")

        # save images into the shared ep469 folder
        save_images_from_tensor(images, out_dir)

        # save a grid per seed to avoid overwriting
        if sample_pt is not None:
            grid_path = os.path.join(out_dir, f"grid_seed{seed}.jpeg")
            utils.save_image(sample_pt, grid_path, nrow=max(1, args.eval_batch_size // 4))

    print("Saved samples (seed {}) to {}".format(seed, out_dir))

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # instantiate model once
    model = UNet(3, image_size=(args.resolution, args.resolution), hidden_dims=[64, 128, 256, 512],
                 use_flash_attn=args.use_flash_attn)

    noise_scheduler = DDIMScheduler(num_train_timesteps=n_timesteps,
                                   beta_schedule="cosine")

    # --- Load checkpoint safely if possible ---
    try:
        ckpt = torch.load(args.pretrained_model_path, map_location="cpu", weights_only=True)
    except TypeError:
        # older PyTorch: no weights_only kwarg
        ckpt = torch.load(args.pretrained_model_path, map_location="cpu")

    # support multiple ckpt layouts
    if isinstance(ckpt, dict) and "ema_model_state" in ckpt:
        pretrained = ckpt["ema_model_state"]
    elif isinstance(ckpt, dict) and "model_state" in ckpt:
        pretrained = ckpt["model_state"]
    else:
        pretrained = ckpt

    model.load_state_dict(pretrained, strict=False)
    model = model.to(device)
    model.eval()

    # single shared ep469 folder (no datetime)
    out_dir = os.path.join(".", args.samples_dir, "ep469")
    os.makedirs(out_dir, exist_ok=True)

    # loop seeds
    for seed in range(args.seedvalue):
        generate_for_seed(noise_scheduler, model, device, out_dir, seed, args)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simple script for image generation (single ep469 folder).")
    parser.add_argument("--samples_dir", type=str, default="generated_samples-adv")
    parser.add_argument("--resolution", type=int, default=224)
    parser.add_argument("--pretrained_model_path",
                        type=str,
                        default="trained_models-adv/ddpm-model-ep469.pth",
                        help="Path to pretrained model")
    parser.add_argument("--eval_batch_size", type=int, default=1)
    parser.add_argument('--use_flash_attn', action='store_true')
    parser.add_argument('--seedvalue', type=int, default=1564,
                        help="Number of different seeds to generate (will loop seeds 0..seedvalue-1)")
    args = parser.parse_args()

    main(args)
