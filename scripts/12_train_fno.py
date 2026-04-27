"""Train Fourier Neural Operator on full-field battery data."""

import argparse
import logging
import json
import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

from src.operator.fno import FNO2d
from src.operator.dataset import FullFieldDataset
from src.operator.trainer import FNOTrainer


def main():
    parser = argparse.ArgumentParser(description="Train FNO")
    parser.add_argument("--data", type=str, default="data/fullfield/fullfield_lmb.h5")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--mid-channels", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--checkpoint-dir", type=str, default="outputs/checkpoints")
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(args.gpu)}")

    dataset = FullFieldDataset(
        args.data,
        fields=["c_e", "phi_e"],
        normalize=True,
    )
    print(f"Dataset: {len(dataset)} sims, {dataset.n_time} time pts, {dataset.nx_full} spatial pts")

    model = FNO2d(
        num_params=7,
        in_channels=2,
        out_channels=2,
        mid_channels=args.mid_channels,
        num_layers=args.num_layers,
        modes1=min(16, dataset.nx_full),
        modes2=min(32, dataset.n_time),
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(f"FNO: {args.num_layers} layers, {args.mid_channels} channels, {n_params:,} params")

    trainer = FNOTrainer(
        model=model,
        dataset=dataset,
        device=device,
        lr=args.lr,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        checkpoint_dir=args.checkpoint_dir,
    )

    history = trainer.train()

    with open("outputs/fno_history.json", "w") as f:
        json.dump(history, f, indent=2)
    print("Training complete. History saved to outputs/fno_history.json")


if __name__ == "__main__":
    main()
