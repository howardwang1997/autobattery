"""Validate PINN predictions and generate comparison plots."""

import argparse
import logging
import torch
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

from src.data.loader import ExperimentalDataLoader, SyntheticDataLoader
from src.data.preprocessor import Preprocessor
from src.pinn.network import MultiDomainPINN
from src.utils.visualization import plot_voltage_comparison, plot_multi_crate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data-dir", type=str, default="data/raw")
    parser.add_argument("--output-dir", type=str, default="outputs/figures")
    parser.add_argument("--num-params", type=int, default=7)
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    import pathlib
    pathlib.Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    model = MultiDomainPINN(num_params=args.num_params)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    loader = ExperimentalDataLoader()
    preprocessor = Preprocessor()
    data_files = sorted(pathlib.Path(args.data_dir).glob("*.csv"))

    for f in data_files:
        data = loader.load_csv(f)
        prepped = preprocessor.prepare_inverse_training(
            data["time"], data["voltage"], data["current"], n_points=200
        )

        t = torch.from_numpy(prepped["t_colloc"]).unsqueeze(-1).to(device)
        params = torch.zeros(1, args.num_params, device=device)

        with torch.no_grad():
            v_pred = model.forward_voltage_only(t, params.expand(t.shape[0], -1))

        v_pred_np = v_pred.cpu().numpy().flatten()

        fig = plot_voltage_comparison(
            t_exp=prepped["t_physical"],
            v_exp=prepped["v_obs"],
            t_pred=prepped["t_physical"],
            v_pred=v_pred_np,
            title=f"Validation: {f.name}",
            save_path=f"{args.output_dir}/validation_{f.stem}.png",
        )
        print(f"Saved plot for {f.name}")

    print(f"All validation plots saved to {args.output_dir}")


if __name__ == "__main__":
    main()
