# Example Project: Object Detection

This example shows an end-to-end, lightweight PyTorch object detection workflow using `torchvision`'s Faster R-CNN.
It includes a **toy dataset generator** so you can run the entire pipeline immediately.

## Project Structure

- `create_toy_data.py`: Generates synthetic images with simple rectangle objects and `.pt` annotations.
- `dataset.py`: Dataset and data loading helpers for object detection.
- `model.py`: Model builder for a `torchvision` Faster R-CNN model.
- `train.py`: Training loop scaffold (saves model weights).
- `predict.py`: Inference script to visualize predicted boxes on one image.
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

- `toy_data/images/*.jpg`
- `toy_data/annotations/*.pt`

### 2) Train a Detector

```bash
python train.py \
  --data-root toy_data \
  --num-classes 2 \
  --epochs 1 \
  --batch-size 2 \
  --save-path detector.pth
```

### 3) Run Inference on One Image

```bash
python predict.py \
  --weights detector.pth \
  --image toy_data/images/img_0000.jpg \
  --output prediction.jpg
```

The script saves `prediction.jpg` with predicted boxes drawn in red.

## Using Your Dataset

Update `dataset.py` to load your images and annotations (e.g., COCO-style JSON, VOC XML, or a custom format).
The dataset should return:

- `image`: a tensor of shape `[3, H, W]`.
- `target`: a dict containing at least `boxes` (`FloatTensor[N, 4]`) and `labels` (`Int64Tensor[N]`).

## Next Steps

- Add a validation split and mAP evaluation.
- Experiment with different backbones (e.g., `fasterrcnn_mobilenet_v3_large_fpn`).
- Add stronger augmentations and longer training.
