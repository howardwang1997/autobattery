"""Inverse parameter identification using trained forward surrogate."""

import argparse
import logging
import json
import time
import torch
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

from src.pinn.network import VoltageMLP


def main():
    parser = argparse.ArgumentParser(description="Inverse parameter identification")
    parser.add_argument("--checkpoint", type=str, default="outputs/checkpoints/forward_pinn_final.pt")
    parser.add_argument("--data", type=str, default="/root/data/raw/exapmles/NEWAREA_1205XXL01006.xlsx")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--num-epochs", type=int, default=5000)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--cycle", type=int, default=1, help="Which cycle to fit")
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)

    num_params = len(ckpt["param_names"])
    model = VoltageMLP(num_params=num_params, hidden_dim=256, num_layers=4)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    param_mean = np.array(ckpt["param_mean"]).astype(np.float64).flatten()
    param_std = np.array(ckpt["param_std"]).astype(np.float64).flatten() + 1e-12
    param_names = [str(p) for p in ckpt["param_names"]]

    v_obs, t_norm = load_single_cycle(args.data, args.cycle)
    print(f"Cycle {args.cycle}: {len(v_obs)} points, V=[{v_obs.min():.3f}, {v_obs.max():.3f}]V")

    v_mean_exp = float(v_obs.mean())
    v_std_exp = float(v_obs.std()) + 1e-3
    v_obs_norm = ((v_obs - v_mean_exp) / v_std_exp).astype(np.float32)

    t_tensor = torch.from_numpy(t_norm.reshape(-1, 1).astype(np.float32)).to(device)
    v_tensor = torch.from_numpy(v_obs_norm.reshape(-1, 1)).to(device)

    raw_params = torch.nn.Parameter(torch.zeros(1, num_params, dtype=torch.float32, device=device))

    optimizer = torch.optim.Adam([raw_params], lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.num_epochs, eta_min=args.lr * 0.01
    )

    print(f"\nStarting inverse optimization: {args.num_epochs} epochs, lr={args.lr}")
    t_start = time.time()

    best_loss = float("inf")
    best_params_norm = None
    history = {"loss": [], "params": {n: [] for n in param_names}}

    for epoch in range(args.num_epochs):
        params_norm = raw_params
        params_tiled = params_norm.expand(t_tensor.shape[0], -1)

        v_pred_norm = model(t_tensor, params_tiled)
        loss = torch.nn.functional.mse_loss(v_pred_norm, v_tensor)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_([raw_params], max_norm=1.0)
        optimizer.step()
        scheduler.step()

        loss_val = loss.item()
        if loss_val < best_loss:
            best_loss = loss_val
            best_params_norm = raw_params.detach().clone()

        history["loss"].append(loss_val)

        if (epoch + 1) % 500 == 0:
            rmse_mV = np.sqrt(loss_val) * v_std_exp * 1000
            lr = optimizer.param_groups[0]["lr"]
            elapsed = time.time() - t_start
            print(
                f"Epoch {epoch+1}/{args.num_epochs} | "
                f"Loss: {loss_val:.6f} | RMSE: {rmse_mV:.1f}mV | "
                f"LR: {lr:.2e} | {elapsed:.0f}s"
            )

    params_norm_np = best_params_norm.cpu().numpy().flatten()
    params_physical = params_norm_np * param_std + param_mean
    results = {name: float(params_physical[i]) for i, name in enumerate(param_names)}

    print(f"\nIdentified parameters ({time.time()-t_start:.0f}s):")
    for name, val in results.items():
        print(f"  {name} = {val:.6e}")

    rmse_final = np.sqrt(best_loss) * v_std_exp * 1000
    print(f"Final RMSE: {rmse_final:.1f}mV")

    with open("outputs/inverse_results.json", "w") as f:
        json.dump({"params": results, "rmse_mV": rmse_final, "cycle": args.cycle}, f, indent=2)

    plot_results(model, t_tensor, v_obs, v_mean_exp, v_std_exp,
                 best_params_norm, device)


def load_single_cycle(data_path, cycle_num=1):
    """Load a single discharge cycle from experimental data."""
    from src.data.loader import ExperimentalDataLoader

    loader = ExperimentalDataLoader()
    data = loader.load_neware_xlsx(data_path, sheet_name="record")

    cycles = data["cycle"]
    currents = data["current"]
    voltages = data["voltage"]
    times = data["time"]

    cycle_mask = cycles == cycle_num
    if not cycle_mask.any():
        available = np.unique(cycles)
        print(f"Cycle {cycle_num} not found. Available: {available[:10]}...")
        cycle_mask = cycles == available[1]

    c_cycle = currents[cycle_mask]
    v_cycle = voltages[cycle_mask]
    t_cycle = times[cycle_mask]

    discharge_mask = c_cycle < -1e-6
    if not discharge_mask.any():
        discharge_mask = np.abs(c_cycle) > 1e-6

    v_discharge = v_cycle[discharge_mask]
    t_discharge = t_cycle[discharge_mask]

    if len(v_discharge) < 10:
        t_discharge = t_cycle
        v_discharge = v_cycle

    t_start = t_discharge[0]
    t_end = t_discharge[-1]
    t_duration = t_end - t_start if t_end > t_start else 1.0

    n_points = 200
    t_uniform = np.linspace(0, 1, n_points)
    t_physical_norm = (t_discharge - t_start) / t_duration
    v_interp = np.interp(t_uniform, t_physical_norm, v_discharge)

    return v_interp.astype(np.float32), t_uniform.astype(np.float32)


def plot_results(model, t_tensor, v_obs, v_mean, v_std, best_params_norm, device):
    """Plot inverse fitting results."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with torch.no_grad():
        params_tiled = best_params_norm.expand(t_tensor.shape[0], -1)
        v_pred_norm = model(t_tensor, params_tiled).cpu().numpy().flatten()
    v_pred = v_pred_norm * v_std + v_mean

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(t_tensor.cpu().numpy().flatten(), v_obs, "b-", label="Experimental", linewidth=1.5)
    axes[0].plot(t_tensor.cpu().numpy().flatten(), v_pred, "r--", label="PINN Prediction", linewidth=1.5)
    axes[0].set_xlabel("t/t_end")
    axes[0].set_ylabel("Voltage (V)")
    rmse = np.sqrt(np.mean((v_pred - v_obs) ** 2)) * 1000
    axes[0].set_title(f"Parameter Identification (RMSE={rmse:.1f}mV)")
    axes[0].legend()

    axes[1].plot(v_obs, v_pred, ".", alpha=0.3, markersize=2)
    v_min = min(v_obs.min(), v_pred.min())
    v_max = max(v_obs.max(), v_pred.max())
    axes[1].plot([v_min, v_max], [v_min, v_max], "k--", linewidth=0.5)
    axes[1].set_xlabel("Experimental V (V)")
    axes[1].set_ylabel("Predicted V (V)")
    axes[1].set_title("Parity Plot")

    plt.tight_layout()
    plt.savefig("outputs/inverse_results.png", dpi=150)
    print("Results saved to outputs/inverse_results.png")


if __name__ == "__main__":
    main()
