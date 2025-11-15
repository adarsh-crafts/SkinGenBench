# Diffusion Models
> Please refer to the paper to understand the abbreviations such as BSGT, ADGT, BSDF, and ADDF.

This is a guide on how to train the diffusion models used in this research with the `DDPM_training.ipynb` notebook.

## Setup

### HuggingFace (optional)
Log into huggingface hub and generate a token with `write` permissions to upload model weights to huggingface.

### Scripts
We use the official training scripts from https://github.com/huggingface/diffusers

## Training setup

### Accelerate Configuration
After running `!accelerate config`. Choose the accelerate configuration as follows

```bash
--------------------------------------------------------------------------------In which compute environment are you running?
Please input a choice index (starting from 0), and press enter
 ➔  This machine
    AWS (Amazon SageMaker)
0
This machine
--------------------------------------------------------------------------------Which type of machine are you using?
Please input a choice index (starting from 0), and press enter
 ➔  No distributed training
    multi-CPU
    multi-XPU
    multi-HPU
    multi-GPU
    multi-NPU
    multi-MLU
    multi-SDAA
    multi-MUSA
    TPU
0
No distributed training
Do you want to run your training on CPU only (even if a GPU / Apple Silicon / Ascend NPU device is available)? [yes/NO]:NO
Do you wish to optimize your script with torch dynamo?[yes/NO]:yes
--------------------------------------------------------------------------------Which dynamo backend would you like to use?
Please input a choice index (starting from 0), and press enter
    eager
    aot_eager
 ➔  inductor
    aot_ts_nvfuser
    nvprims_nvfuser
    cudagraphs
    ofi
    fx2trt
    onnxrt
    tensorrt
    aot_torchxla_trace_once
    torhchxla_trace_once
    ipex
    tvm
2
inductor
Do you want to customize the defaults sent to torch.compile? [yes/NO]: NO
Do you want to use DeepSpeed? [yes/NO]: NO
What GPU(s) (by id) should be used for training on this machine as a comma-separated list? [all]:0
Would you like to enable numa efficiency? (Currently only supported on NVIDIA hardware). [yes/NO]: NO
--------------------------------------------------------------------------------Do you wish to use mixed precision?
Please input a choice index (starting from 0), and press enter
 ➔  no
    fp16
    bf16
    fp8
1
fp16
accelerate configuration saved at /root/.cache/huggingface/accelerate/default_config.yaml
```

### Training Parameters
`training_dataset_path` is the path to the real images after applying either of the two preprocessing techniques. `(ADGT or BSGT)`

```python
!accelerate launch train_unconditional.py \
  --dataset_name= "training_dataset_path" \
  --model_config_name_or_path="google/ddpm-celebahq-256" \
  --resolution=256 --center_crop --random_flip \
  --output_dir="output_folder" \
  --train_batch_size=16 \
  --num_epochs=120 \
  --gradient_accumulation_steps=1 \
  --use_ema \
  --learning_rate=1e-5 \
  --lr_warmup_steps=500 \
  --mixed_precision="fp16" \
  --push_to_hub
  ```

### Resume Training from Checkpoint

Example: To resume from checkpoint-9500:
```
  --resume_from_checkpoint="checkpoint-9500"
```

## Hardware

- **GPU**: NVIDIA L4 (22GB VRAM)

## Generate Synthetic Images
1. Install libraries
```python
!pip install datasets tqdm diffusers torch accelerate
```
2. Generate images
```python
!python generate_images.py
```