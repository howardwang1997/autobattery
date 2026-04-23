"""Train forward PINN (fast P2D solver)."""

import argparse
import logging
import torch
import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

from src.simulation.parameters import load_config
from src.data.dataset import SyntheticDataset, CollocationDataset
from src.pinn.network import MultiDomainPINN
from src.pinn.pdes import MetalBatteryPDE
from src.pinn.losses import PINNLoss
from src.pinn.forward import ForwardTrainer


def main():
    parser = argparse.ArgumentParser(description="Train forward PINN")
    parser.add_argument("--config", type=str, default="configs/lmb.yaml")
    parser.add_argument("--data", type=str, default="data/synthetic/synthetic_lmb.npz")
    parser.add_argument("--gpus", type=str, default="0")
    args = parser.parse_args()

    config = load_config(args.config)

    device = torch.device(f"cuda:{args.gpus}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    dataset = SyntheticDataset(args.data)
    print(f"Dataset: {len(dataset)} simulations, {dataset.num_params} parameters")

    loader = torch.utils.data.DataLoader(
        dataset, batch_size=8, shuffle=True, num_workers=2, pin_memory=True
    )

    model_cfg = config["model"]
    model = MultiDomainPINN(
        num_params=dataset.num_params,
        hidden_dim=model_cfg.get("hidden_dim", 128),
        num_layers=model_cfg.get("num_layers", 6),
        activation=model_cfg.get("activation", "silu"),
    )
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    pde = MetalBatteryPDE()

    train_cfg = config["training"]["forward"]
    loss_fn = PINNLoss(
        lambda_data=train_cfg.get("lambda_data", 10.0),
        lambda_pde=train_cfg.get("lambda_pde", 1.0),
        lambda_bc=train_cfg.get("lambda_bc", 5.0),
        lambda_ic=train_cfg.get("lambda_ic", 5.0),
    )

    colloc_dataset = CollocationDataset(n_points=10000, seed=42)

    trainer = ForwardTrainer(
        model=model,
        pde=pde,
        loss_fn=loss_fn,
        device=device,
        lr=train_cfg.get("learning_rate", 1e-3),
        scheduler=train_cfg.get("scheduler", "cosine"),
        num_epochs=train_cfg.get("num_epochs", 5000),
        log_every=config.get("logging", {}).get("log_every", 50),
        save_every=config.get("logging", {}).get("save_every", 500),
        checkpoint_dir=config.get("logging", {}).get("checkpoint_dir", "outputs/checkpoints"),
    )

    history = trainer.train(
        train_loader=loader,
        collocation_fn=colloc_dataset.sample_batch,
    )

    import json
    with open("outputs/forward_history.json", "w") as f:
        json.dump(history, f, indent=2)
    print("Training complete. History saved to outputs/forward_history.json")


if __name__ == "__main__":
    main()
