"""Train forward PINN (fast P2D solver)."""

import argparse
import logging
import json
import torch
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

from src.simulation.parameters import load_config
from src.pinn.network import MultiDomainPINN, VoltageMLP, VoltagePredictor
from src.pinn.pdes import MetalBatteryPDE
from src.pinn.losses import PINNLoss
from src.pinn.forward import ForwardTrainer


def main():
    parser = argparse.ArgumentParser(description="Train forward PINN")
    parser.add_argument("--config", type=str, default="configs/lmb.yaml")
    parser.add_argument("--data", type=str, default="data/synthetic/synthetic_lmb.npz")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--use-pde", action="store_true",
                        help="Add PDE residual loss term (only meaningful with --model pinn)")
    parser.add_argument("--pde-warmup-epochs", type=int, default=0,
                        help="Hold the PDE term out of the loss for this many epochs")
    parser.add_argument("--model", type=str, default="predictor",
                        choices=["mlp", "pinn", "predictor"])
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--norm-mode", type=str, default="global",
                        choices=["global", "per_sim"],
                        help="Voltage normalisation: 'global' (default, leak-free) or "
                             "'per_sim' (legacy, leaks per-curve mean/std)")
    parser.add_argument("--adaptive-weighting", type=str, default="none",
                        choices=["none", "softadapt"],
                        help="Adaptive loss-term weighting strategy")
    args = parser.parse_args()

    config = load_config(args.config)

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(args.gpu)}")

    model_cfg = config["model"]
    train_cfg = config["training"]["forward"]

    num_params = len(config["simulation"]["parameter_sweep"])
    hidden_dim = model_cfg.get("hidden_dim", 128)
    num_layers = model_cfg.get("num_layers", 6)

    if args.model == "mlp":
        model = VoltageMLP(
            num_params=num_params,
            hidden_dim=256,
            num_layers=4,
            activation="silu",
        )
    elif args.model == "predictor":
        model = VoltagePredictor(
            num_params=num_params,
            hidden_dim=256,
            num_layers=4,
            activation="silu",
        )
    else:
        model = MultiDomainPINN(
            num_params=num_params,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            activation=model_cfg.get("activation", "silu"),
        )
    print(f"Model: {args.model} | {sum(p.numel() for p in model.parameters()):,} params")

    pde = MetalBatteryPDE()

    loss_fn = PINNLoss(
        lambda_data=train_cfg.get("lambda_data", 10.0),
        lambda_pde=train_cfg.get("lambda_pde", 1.0),
        lambda_bc=train_cfg.get("lambda_bc", 5.0),
        lambda_ic=train_cfg.get("lambda_ic", 5.0),
    )

    batch_size = args.batch_size or train_cfg.get("batch_size_data", 256)
    num_epochs = args.epochs or train_cfg.get("num_epochs", 500)
    lr = args.lr or train_cfg.get("learning_rate", 1e-3)

    trainer = ForwardTrainer(
        model=model,
        pde=pde,
        loss_fn=loss_fn,
        device=device,
        lr=lr,
        scheduler=train_cfg.get("scheduler", "cosine"),
        num_epochs=num_epochs,
        log_every=config.get("logging", {}).get("log_every", 10),
        save_every=config.get("logging", {}).get("save_every", 100),
        checkpoint_dir=config.get("logging", {}).get("checkpoint_dir", "outputs/checkpoints"),
        use_pde=args.use_pde,
        pde_collocation_points=train_cfg.get("batch_size_collocation", 256),
        pde_warmup_epochs=args.pde_warmup_epochs,
        norm_mode=args.norm_mode,
        adaptive_weighting=args.adaptive_weighting,
    )

    history = trainer.train(
        data_path=args.data,
        batch_size=batch_size,
    )

    with open("outputs/forward_history.json", "w") as f:
        json.dump(history, f, indent=2)
    print(f"Training complete. History saved to outputs/forward_history.json")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    if history["train_loss"]:
        axes[0].plot(history["train_loss"], label="Train")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Total Loss")
        axes[0].set_yscale("log")
        axes[0].legend()
    if history["data_loss"]:
        axes[1].plot(history["data_loss"], label="Data Loss")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("MSE")
        axes[1].set_yscale("log")
        axes[1].legend()
    val_epochs = list(range(
        config.get("logging", {}).get("log_every", 10),
        len(history["train_loss"]) + 1,
        config.get("logging", {}).get("log_every", 10),
    ))
    if history["val_loss"]:
        axes[2].plot(val_epochs[:len(history["val_loss"])], history["val_loss"], label="Val Loss")
        axes[2].set_xlabel("Epoch")
        axes[2].set_ylabel("Val MSE")
        axes[2].set_yscale("log")
        axes[2].legend()
    plt.tight_layout()
    plt.savefig("outputs/forward_training_curves.png", dpi=150)
    print("Training curves saved to outputs/forward_training_curves.png")

    _plot_predictions(trainer, args.data, config)

    plt.close("all")


def _plot_predictions(trainer, data_path, config):
    """Plot sample predictions vs ground truth."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from src.data.loader import SyntheticDataLoader

    loader = SyntheticDataLoader(data_path)
    device = trainer.device

    v_mean = trainer.normalizer.v_mean_global
    v_std = trainer.normalizer.v_std_global
    param_mean = trainer._param_mean
    param_std = trainer._param_std

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    indices = np.linspace(0, loader.num_simulations - 1, 6, dtype=int)

    for ax, idx in zip(axes.flat, indices):
        sim = loader.get_simulation(idx)
        t = sim["time"]
        v_true = sim["voltage"]

        t_end = t[-1] if t[-1] > 0 else 1.0
        t_norm = t / t_end
        t_uniform = np.linspace(0, 1, 200)
        v_interp = np.interp(t_uniform, t_norm, v_true)

        params = np.array([sim["params"][n] for n in loader.param_names])
        log_params = np.log10(np.clip(params, 1e-30, None))
        p_norm = ((log_params - param_mean) / param_std).astype(np.float32)

        trainer.model.eval()
        with torch.no_grad():
            from src.pinn.network import MultiDomainPINN, VoltageMLP, VoltagePredictor
            t_t = torch.from_numpy(t_uniform.astype(np.float32)).reshape(-1, 1).to(device)
            p_flat = torch.from_numpy(np.tile(p_norm, (200, 1))).to(device)
            if isinstance(trainer.model, (VoltageMLP, VoltagePredictor)):
                v_pred_raw = trainer.model(t_t, p_flat).cpu().numpy().flatten()
            else:
                N = 200
                x_pos = torch.full((N, 1), 0.75, device=device)
                r_mid = torch.full((N, 1), 0.5, device=device)
                domain = torch.full((N,), MultiDomainPINN.DOMAIN_POS, dtype=torch.long, device=device)
                v_pred_raw = trainer.model(t_t, x_pos, r_mid, p_flat, domain)["V"].cpu().numpy().flatten()
            v_pred = v_pred_raw * v_std + v_mean

        rmse = np.sqrt(np.mean((v_pred - v_interp) ** 2)) * 1000
        ax.plot(t_uniform, v_interp, "b-", label="True", linewidth=1.5)
        ax.plot(t_uniform, v_pred, "r--", label="PINN", linewidth=1.5)
        c_rate = sim["c_rate"]
        ax.set_title(f"Sim {idx} (C={c_rate:.1f}, RMSE={rmse:.0f}mV)")
        ax.set_xlabel("t/t_end")
        ax.set_ylabel("Voltage (V)")
        ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig("outputs/forward_predictions.png", dpi=150)
    print("Predictions saved to outputs/forward_predictions.png")


if __name__ == "__main__":
    main()
