import os
from datetime import datetime

from PIL import Image
import torch
import numpy as np
from torchvision import utils
import matplotlib.pyplot as plt


def save_images(generated_images, epoch, args, contexts=None):
    """
    Save individual images from generated_images['sample'] and a grid from ['sample_pt'].
    - Safe for torch.Tensor or numpy arrays.
    - Handles (C,H,W) or (H,W,C) and channel-first (N,C,H,W).
    - Normalizes from [-1,1] -> [0,1] if needed.
    - Ensures nrow >= 1 to avoid ZeroDivisionError for small eval_batch_size.
    """
    import os
    from datetime import datetime
    from PIL import Image
    import numpy as np
    import torch
    from torchvision import utils

    # Prepare output directory
    current_date = datetime.today().strftime('%Y%m%d_%H%M%S')
    out_dir = f"./{args.samples_dir}/{current_date}_{args.dataset_name}_{epoch}/"
    os.makedirs(out_dir, exist_ok=True)

    # ---- Helper to convert batch to uint8 HWC numpy ----
    def batch_to_uint8_np(x):
        # torch -> numpy
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
        x = np.asarray(x)
        if x.ndim == 3:           # single image, (C,H,W) or (H,W,C)
            x = x[None, ...]
        if x.ndim != 4:
            raise ValueError(f"Expected 3 or 4 dims for images, got {x.shape}")
        # channel-first N,C,H,W -> N,H,W,C
        if x.shape[1] in (1, 3):
            x = np.transpose(x, (0, 2, 3, 1))
        # if values look like [-1,1] convert to [0,1]
        if x.min() < -0.5 and x.max() <= 1.5:
            x = (x + 1.0) * 0.5
        x = np.clip(x, 0.0, 1.0)
        x = (x * 255.0).round().astype("uint8")
        return x

    # ---- Save individual images ----
    imgs = generated_images.get("sample")
    if imgs is None:
        print("Warning: 'sample' missing; skipping individual image saves.")
    else:
        try:
            imgs_np = batch_to_uint8_np(imgs)
            for idx, im in enumerate(imgs_np):
                pil = Image.fromarray(im)
                if contexts:
                    pil.save(f"{out_dir}/{epoch}_{contexts[idx]}_{idx}.jpeg")
                else:
                    pil.save(f"{out_dir}/{epoch}_{idx}.jpeg")
        except Exception as e:
            print(f"Warning: failed saving individual images: {e}")

    # ---- Save grid image safely ----
    grid_tensor = generated_images.get("sample_pt")
    if grid_tensor is None:
        # nothing to do
        return

    # If it's a torch tensor, ensure there is at least one image
    try:
        is_torch = isinstance(grid_tensor, torch.Tensor)
    except Exception:
        is_torch = False

    valid_grid = True
    try:
        if is_torch:
            if grid_tensor.dim() == 0 or grid_tensor.size(0) == 0:
                valid_grid = False
        else:
            # numpy-like
            if getattr(grid_tensor, "shape", None) is None or len(grid_tensor) == 0:
                valid_grid = False
    except Exception:
        valid_grid = False

    if not valid_grid:
        print("Skipping grid: 'sample_pt' empty or invalid.")
        return

    # compute nrow defensively (must be >=1)
    try:
        candidate = int(args.eval_batch_size) // 4
    except Exception:
        candidate = 1
    nrow = max(1, candidate)

    # Cap nrow to number of images if possible
    try:
        num_images = grid_tensor.size(0) if is_torch else len(grid_tensor)
        nrow = min(nrow, max(1, int(num_images)))
    except Exception:
        pass

    try:
        utils.save_image(grid_tensor, f"{out_dir}/{epoch}_grid.jpeg", nrow=nrow)
    except Exception as e:
        print(f"Warning: failed saving grid image: {e}")



def unnormalize_to_zero_to_one(t):
    return (t + 1) * 0.5


def numpy_to_pil(images):
    if images.ndim == 3:
        images = images[None, ...]
    images = (images * 255).round().astype("uint8")
    pil_images = [Image.fromarray(image) for image in images]

    return pil_images


def match_shape(values, broadcast_array, tensor_format="pt"):
    values = values.flatten()

    while len(values.shape) < len(broadcast_array.shape):
        values = values[..., None]
    if tensor_format == "pt":
        values = values.to(broadcast_array.device)

    return values


def clip(tensor, min_value=None, max_value=None):
    if isinstance(tensor, np.ndarray):
        return np.clip(tensor, min_value, max_value)
    elif isinstance(tensor, torch.Tensor):
        return torch.clamp(tensor, min_value, max_value)

    raise ValueError("Tensor format is not valid is not valid - " \
        f"should be numpy array or torch tensor. Got {type(tensor)}.")
