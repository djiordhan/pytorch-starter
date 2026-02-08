# Example Project: Semantic Segmentation

This example provides an end-to-end, lightweight PyTorch semantic segmentation workflow using `torchvision`'s FCN-ResNet50.
It includes a **toy dataset generator** so you can run the entire pipeline immediately.

## Project Structure

- `create_toy_data.py`: Generates synthetic images and pixel-wise masks.
- `dataset.py`: Dataset and data loading helpers for semantic segmentation.
- `model.py`: Model builder for an FCN-ResNet50 segmentation model.
- `train.py`: Training loop scaffold (saves model weights).
- `predict.py`: Inference script to visualize predicted masks on one image.
- `requirements.txt`: Dependencies for this example.

## Setup

```bash
pip install -r requirements.txt
```

## Quick Start (Runnable Example)

### 1) Create a Toy Dataset

```bash
python create_toy_data.py --output-dir toy_data --num-images 40
```

This creates:

- `toy_data/images/*.png`
- `toy_data/masks/*.png`

### 2) Train a Segmenter

```bash
python train.py \
  --data-root toy_data \
  --num-classes 2 \
  --epochs 1 \
  --batch-size 2 \
  --save-path segmenter.pth
```

### 3) Run Inference on One Image

```bash
python predict.py \
  --weights segmenter.pth \
  --image toy_data/images/img_0000.png \
  --output prediction.png
```

The script saves `prediction.png` with a red overlay showing the predicted mask.

## Using Your Dataset

Update `dataset.py` to load your images and segmentation masks (e.g., COCO-style masks, VOC, or a custom format).
The dataset should return:

- `image`: a tensor of shape `[3, H, W]`.
- `mask`: a tensor of shape `[H, W]` with integer class IDs.

## Next Steps

- Add a validation split and IoU evaluation.
- Experiment with DeepLabV3 or U-Net architectures.
- Add stronger augmentations and longer training.
