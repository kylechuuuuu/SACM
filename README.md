# SACM

SACM is a PyTorch-based image segmentation project that integrates the Segment Anything Model (SAM) for both training and inference. The project provides custom training and testing scripts, and wraps SAM for easy extension and application.

## Features

- ViT-based image segmentation model (SAM)
- Custom dataset training and evaluation
- Automatic mask generation and prediction
- Detailed logging and evaluation metrics (F1, Precision, Recall)

## Directory Structure

```
segment_anything/         # SAM model and components
    __init__.py
    build_sam.py          # Build different SAM model variants
    predictor.py          # Inference and mask prediction
    automatic_mask_generator.py # Automatic mask generation
    modeling/             # Model architecture
    utils/                # Utility functions
train_sam.py              # Training script
test_sam.py               # Testing script
README.md
```

## Quick Start

1. Install dependencies:
   ```bash
   pip install torch torchvision scikit-learn pillow tqdm
   ```

2. Train the model:
   ```bash
   python train_sam.py --your_args
   ```

3. Test the model:
   ```bash
   python test_sam.py --your_args
   ```

## Reference

- [Meta Segment Anything](https://github.com/facebookresearch/segment-anything)