# Example Project: Diffusion Model (MNIST)

This example provides a minimal diffusion training loop that learns to predict noise on MNIST digits.
It is intentionally lightweight so you can adapt the schedule, model size, or sampling routine.

## Project Structure

- `dataset.py`: MNIST data loader normalized to [-1, 1].
- `model.py`: Small convolutional noise predictor with timestep embeddings.
- `train.py`: Training loop for a DDPM-style objective.

## Setup

```bash
pip install -r requirements.txt
```

## Training

```bash
python train.py --epochs 5 --batch-size 128
```

By default, the model trains for 5 epochs and saves `diffusion_mnist.pth` in the example directory.

## Next Steps

- Add a sampling script to generate digits from pure noise.
- Increase the number of diffusion steps for higher-quality samples.
- Swap MNIST for Fashion-MNIST or your own grayscale dataset.
