# Classifier Training

This directory contains the trained classification models and configuration files used in our research. We trained five state-of-the-art deep learning models for image classification to ensure robust and comprehensive evaluation.

## Models

We trained the following architectures:

- **ResNet18** - Lightweight residual network
- **ResNet50** - Deeper residual network with better capacity
- **ViT-B/16** - Vision Transformer with 16×16 patch size
- **EfficientNet-B0** - Efficient convolutional network
- **VGG16** - Classic deep convolutional network

## Training Setup

All models were trained using the [pytorch-classification-extended](https://github.com/adarsh-crafts/pytorch-classification-extended) framework, which provides a unified interface for training multiple architectures.

### Prerequisites

```bash
# Clone the training framework
git clone https://github.com/adarsh-crafts/pytorch-classification-extended.git
cd pytorch-classification-extended

# Install dependencies
pip install -r requirements.txt
```

### Dataset Preparation

Organize your dataset in the following structure:

```
dataset/
├── train/
│   ├── class1/
│   ├── class2/
│   └── ...
└── val/
    ├── class1/
    ├── class2/
    └── ...
```

## Training Commands

### Example: ResNet18
```python
python customdata.py `
    -a resnet18 `
    -d "path_to_dataset_folder" `
    --pretrained `
    --epochs 30 `
    --schedule 15 25 `
    --gamma 0.1 `
    --lr 0.001 `
    -c "checkpoints/my_model/resnet18"
```
Example: EfficientNet-B0
```python
python customdata.py `
    -a efficientnet_b0 `
    -d "path_to_dataset_folder" `
    --pretrained `
    --epochs 30 `
    --schedule 15 25 `
    --gamma 0.1 `
    --lr 0.001 `
    --train-batch 64 `
    --test-batch 64 `
    -c "checkpoints/my_model/efficientnet_b0"
```

## Hyperparameters

| Parameter | ResNet18 | ResNet50 | ViT-B/16 | EfficientNet-B0 | VGG16 |
|-----------|----------|----------|----------|-----------------|-------|
| Epochs | `30` | `30` | `30` | `30` | `30` |
| schedule | `15 25` | `15 25` | `15 25` | `15 25` | `15 25` |
| gamma | `0.1` | `0.1` | `0.1` | `0.1` | `0.1` |
| lr | `0.001` | `0.001` | `0.001` | `0.001` | `0.001` |
| train-batch | [default] | `64` | `64` | `64` | `64` |
| test-batch | [default] | `64` | `64` | `64` | `64` |

## Hardware

- **GPU**: NVIDIA RTX 4060



## Inference

To use the trained models for inference:

```python
import torch
from torchvision import transforms
from PIL import Image

# Load model
model = torch.load('checkpoints/resnet18/best_model.pth')
model.eval()

# Prepare image
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                       std=[0.229, 0.224, 0.225])
])

image = Image.open('path/to/image.jpg')
input_tensor = transform(image).unsqueeze(0)

# Inference
with torch.no_grad():
    output = model(input_tensor)
    prediction = output.argmax(dim=1)
```