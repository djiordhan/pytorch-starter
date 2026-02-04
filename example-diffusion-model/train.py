"""Train a tiny diffusion model on MNIST."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn, optim
from tqdm import tqdm

from dataset import get_mnist_dataloader
from model import NoisePredictor


def linear_beta_schedule(timesteps: int, start: float = 1e-4, end: float = 0.02) -> torch.Tensor:
    return torch.linspace(start, end, timesteps)


def get_diffusion_constants(timesteps: int, device: torch.device) -> dict[str, torch.Tensor]:
    betas = linear_beta_schedule(timesteps).to(device)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    return {
        "betas": betas,
        "alphas_cumprod": alphas_cumprod,
        "sqrt_alphas_cumprod": torch.sqrt(alphas_cumprod),
        "sqrt_one_minus_alphas_cumprod": torch.sqrt(1.0 - alphas_cumprod),
    }


def q_sample(x_start: torch.Tensor, t: torch.Tensor, constants: dict[str, torch.Tensor], noise: torch.Tensor) -> torch.Tensor:
    sqrt_alpha = constants["sqrt_alphas_cumprod"][t].view(-1, 1, 1, 1)
    sqrt_one_minus = constants["sqrt_one_minus_alphas_cumprod"][t].view(-1, 1, 1, 1)
    return sqrt_alpha * x_start + sqrt_one_minus * noise


def train(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataloader = get_mnist_dataloader(args.batch_size, args.data_dir)

    model = NoisePredictor().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = nn.MSELoss()

    constants = get_diffusion_constants(args.timesteps, device)
    model.train()

    for epoch in range(1, args.epochs + 1):
        epoch_loss = 0.0
        progress = tqdm(dataloader, desc=f"Epoch {epoch}/{args.epochs}")
        for images, _ in progress:
            images = images.to(device)
            t = torch.randint(0, args.timesteps, (images.size(0),), device=device)
            noise = torch.randn_like(images)
            noisy_images = q_sample(images, t, constants, noise)

            optimizer.zero_grad()
            noise_pred = model(noisy_images, t)
            loss = criterion(noise_pred, noise)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            progress.set_postfix(loss=loss.item())

        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch}: average loss {avg_loss:.4f}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_path)
    print(f"Saved model to {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a minimal diffusion model on MNIST")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--timesteps", type=int, default=200)
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--output", type=str, default="diffusion_mnist.pth")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
