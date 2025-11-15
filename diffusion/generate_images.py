import torch
from diffusers import DDPMPipeline, DDPMScheduler, UNet2DModel
import os
from tqdm import tqdm
import math
from IPython.display import clear_output
import time
import gc

# --- Disable internal diffusers progress bars ---
from diffusers.utils import logging
logging.disable_progress_bar()

# --- 1. Configuration ---
LOCAL_MODEL_PATH = "/content/drive/MyDrive/colab-huggingface-diffusers/diffusers/examples/unconditional_image_generation/ddpm-finetuned-melanoma-advanced"  # UPDATE THIS PATH
CHECKPOINT = "checkpoint-11500"  # Which checkpoint to use
USE_EMA = True  # Use EMA weights (usually better quality)
OUTPUT_DIR = "/content/drive/MyDrive/colab-huggingface-diffusers/diffusers/examples/unconditional_image_generation/advanced_mel_generated_images"
NUM_IMAGES = 5000
BATCH_SIZE = 128
CHECKPOINT_INTERVAL = 128

# --- 2. Setup ---
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Check for existing images to resume
existing_images = sorted([f for f in os.listdir(OUTPUT_DIR) if f.endswith('.png')])
start_index = len(existing_images)

if start_index > 0:
    print(f"Found {start_index} existing images. Resuming generation...")
    images_saved = start_index
else:
    images_saved = 0

# Determine which model to use
unet_subfolder = f"{CHECKPOINT}/unet_ema" if USE_EMA else f"{CHECKPOINT}/unet"
unet_path = os.path.join(LOCAL_MODEL_PATH, unet_subfolder)

print(f"Loading model from local path: {LOCAL_MODEL_PATH}")
print(f"Using checkpoint: {CHECKPOINT}")
print(f"Using {'EMA' if USE_EMA else 'standard'} weights from: {unet_subfolder}\n")

# Verify paths exist
if not os.path.exists(LOCAL_MODEL_PATH):
    print(f"❌ Error: Model path does not exist: {LOCAL_MODEL_PATH}")
    print("Please update LOCAL_MODEL_PATH to point to your model folder")
    exit()

if not os.path.exists(os.path.join(LOCAL_MODEL_PATH, "model_index.json")):
    print(f"❌ Error: model_index.json not found in {LOCAL_MODEL_PATH}")
    print(f"Contents of {LOCAL_MODEL_PATH}:")
    for item in os.listdir(LOCAL_MODEL_PATH):
        print(f"  {item}")
    exit()

if not os.path.exists(unet_path):
    print(f"❌ Error: UNet path does not exist: {unet_path}")
    print(f"Available checkpoints in {LOCAL_MODEL_PATH}:")
    for item in os.listdir(LOCAL_MODEL_PATH):
        if item.startswith("checkpoint-"):
            print(f"  {item}")
    exit()

try:
    # Load the base pipeline from local files
    print("Loading base pipeline from local files...")
    pipeline = DDPMPipeline.from_pretrained(
        LOCAL_MODEL_PATH,
        torch_dtype=torch.float16,
        use_safetensors=True,
        local_files_only=True  # Only use local files
    )
    print("✓ Base pipeline loaded")
    
    # Replace with checkpoint UNet
    print(f"Loading UNet from {unet_path}...")
    unet = UNet2DModel.from_pretrained(
        unet_path,
        torch_dtype=torch.float16,
        use_safetensors=True,
        local_files_only=True
    )
    pipeline.unet = unet
    print("✓ UNet weights loaded from checkpoint")
    
    # Move to GPU
    pipeline.to("cuda")
    print("✓ Model moved to CUDA\n")
    
    # Print pipeline info
    print(f"Pipeline type: {type(pipeline).__name__}")
    print(f"Image size: {pipeline.unet.config.sample_size}")
    print()
    
except Exception as e:
    print(f"❌ Error loading model: {e}")
    import traceback
    traceback.print_exc()
    exit()

# --- 3. Generation Loop ---
print(f"Generating {NUM_IMAGES} images in batches of {BATCH_SIZE}...\n")

num_batches = math.ceil((NUM_IMAGES - images_saved) / BATCH_SIZE)
start_time = time.time()

for i in range(num_batches):
    images_to_generate = min(BATCH_SIZE, NUM_IMAGES - images_saved)
    
    if images_to_generate <= 0:
        break

    try:
        # Generate images
        with torch.no_grad():
            images = pipeline(
                batch_size=images_to_generate,
                num_inference_steps=1000  # DDPM typically uses 1000 steps
            ).images
        
        # Save images
        for img in images:
            filename = f"image_{images_saved:05d}.png"
            save_path = os.path.join(OUTPUT_DIR, filename)
            img.save(save_path)
            images_saved += 1
        
        # Clear GPU cache periodically
        if i % 10 == 0:
            torch.cuda.empty_cache()
            gc.collect()
        
        # Force flush to Google Drive
        os.sync()
        time.sleep(0.5)
        
        # Verify images were saved
        actual_files = len([f for f in os.listdir(OUTPUT_DIR) if f.endswith('.png')])
        
        # Calculate ETA
        elapsed_time = time.time() - start_time
        images_per_second = images_saved / elapsed_time if elapsed_time > 0 else 0
        remaining_images = NUM_IMAGES - images_saved
        eta_seconds = remaining_images / images_per_second if images_per_second > 0 else 0
        eta_hours = eta_seconds / 3600
        
        # Clear output and show progress
        clear_output(wait=True)
        print(f"Batch {i+1}/{num_batches} complete")
        print(f"Images generated: {images_saved}/{NUM_IMAGES}")
        print(f"Images verified on disk: {actual_files}")
        print(f"Progress: {(images_saved/NUM_IMAGES)*100:.1f}%")
        print(f"Speed: {images_per_second:.2f} images/sec")
        print(f"Estimated time remaining: {eta_hours:.2f} hours")
        
    except Exception as e:
        print(f"\n⚠️ Error in batch {i+1}: {e}")
        print(f"Progress saved up to image {images_saved}")
        print("You can re-run the script to resume from this point.")
        import traceback
        traceback.print_exc()
        break

# Final verification
final_count = len([f for f in os.listdir(OUTPUT_DIR) if f.endswith('.png')])
total_time = time.time() - start_time
print(f"\n✓ Done! {final_count} images saved in '{OUTPUT_DIR}'")
print(f"Total time: {total_time/3600:.2f} hours")

if final_count != NUM_IMAGES:
    print(f"⚠️ Warning: Expected {NUM_IMAGES} images but found {final_count} on disk")