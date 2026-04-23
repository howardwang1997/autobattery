"""Train inverse PINN (parameter identification from experimental data)."""

import argparse
import logging
import torch
import json

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

from src.simulation.parameters import load_config, parse_learnable_params
from src.data.loader import ExperimentalDataLoader
from src.data.preprocessor import Preprocessor
from src.data.dataset import CollocationDataset
from src.pinn.network import MultiDomainPINN, InversePINN
from src.pinn.pdes import MetalBatteryPDE
from src.pinn.losses import PINNLoss
from src.pinn.inverse import InverseTrainer


def main():
    parser = argparse.ArgumentParser(description="Train inverse PINN")
    parser.add_argument("--config", type=str, default="configs/lmb.yaml")
    parser.add_argument("--data-dir", type=str, default="data/raw")
    parser.add_argument("--forward-ckpt", type=str, default=None,
                        help="Pretrained forward PINN checkpoint (optional)")
    parser.add_argument("--gpus", type=str, default="0")
    args = parser.parse_args()

    config = load_config(args.config)

    device = torch.device(f"cuda:{args.gpus}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    learnable_params = parse_learnable_params(config)
    print(f"Learnable parameters: {list(learnable_params.keys())}")

    loader = ExperimentalDataLoader()
    import pathlib
    data_files = sorted(pathlib.Path(args.data_dir).glob("*.csv"))
    if not data_files:
        print(f"No CSV files found in {args.data_dir}")
        return

    print(f"Found {len(data_files)} data files")

    first_file = data_files[0]
    data = loader.load_csv(first_file)
    print(f"Loaded {len(data['time'])} data points from {first_file.name}")

    preprocessor = Preprocessor()
    prepped = preprocessor.prepare_inverse_training(
        data["time"], data["voltage"], data["current"], n_points=200
    )

    t_colloc = torch.from_numpy(prepped["t_colloc"]).unsqueeze(-1)
    v_obs = torch.from_numpy(prepped["v_obs"]).unsqueeze(-1)
    i_input = torch.from_numpy(prepped["i_input"]).unsqueeze(-1)

    base_model = MultiDomainPINN(
        num_params=len(learnable_params),
        hidden_dim=config["model"].get("hidden_dim", 128),
        num_layers=config["model"].get("num_layers", 6),
        activation=config["model"].get("activation", "silu"),
    )

    if args.forward_ckpt:
        ckpt = torch.load(args.forward_ckpt, map_location=device)
        base_model.load_state_dict(ckpt["model_state_dict"])
        print(f"Loaded forward PINN checkpoint from {args.forward_ckpt}")

    init_values = {name: p.initial_value for name, p in learnable_params.items()}
    bounds = {name: p.bounds for name, p in learnable_params.items() if p.bounds}

    inverse_model = InversePINN(
        base_model=base_model,
        learnable_param_names=list(learnable_params.keys()),
        init_values=init_values,
        bounds=bounds,
    )

    pde = MetalBatteryPDE()

    train_cfg = config["training"]["inverse"]
    loss_fn = PINNLoss(
        lambda_data=train_cfg.get("lambda_data", 50.0),
        lambda_pde=train_cfg.get("lambda_pde", 1.0),
    )

    trainer = InverseTrainer(
        model=inverse_model,
        pde=pde,
        loss_fn=loss_fn,
        device=device,
        lr=train_cfg.get("learning_rate", 1e-3),
        param_lr_scale=train_cfg.get("param_lr_scale", 0.1),
        phase1_epochs=train_cfg.get("phase1_epochs", 5000),
        phase2_epochs=train_cfg.get("phase2_epochs", 5000),
        log_every=config.get("logging", {}).get("log_every", 50),
    )

    result = trainer.train(
        t_colloc=t_colloc,
        v_obs=v_obs,
        i_input=i_input,
    )

    with open("outputs/inverse_results.json", "w") as f:
        json.dump(result["params"], f, indent=2)
    print("Training complete. Results saved to outputs/inverse_results.json")
    print("Identified parameters:")
    for k, v in result["params"].items():
        print(f"  {k} = {v:.6e}")


if __name__ == "__main__":
    main()
