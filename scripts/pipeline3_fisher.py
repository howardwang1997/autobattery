"""Pipeline 3: Fisher Information Analysis + Sensitivity-Weighted Identification.

Computes the Fisher information matrix for each parameter from each observable,
providing a rigorous theoretical framework for parameter identifiability analysis.

Runs on GPU 2.
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

import torch
import numpy as np
import time
import logging
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [P3] %(message)s")
logger = logging.getLogger(__name__)


def compute_fisher_matrix(model, dataset, device, n_samples=200):
    """
    Compute Fisher Information Matrix using finite differences.

    FIM[i,j] = E[ (∂log p(y|θ)/∂θ_i) (∂log p(y|θ)/∂θ_j) ]
             ≈ (1/N) Σ_n (∂f/∂θ_i)^T Σ^{-1} (∂f/∂θ_j)

    For MSE loss: FIM ∝ J^T J where J is the Jacobian.
    """
    import h5py

    stats = dataset._stats
    param_mean = torch.tensor(stats["params"]["mean"], device=device, dtype=torch.float32)
    param_std = torch.tensor(stats["params"]["std"], device=device, dtype=torch.float32)
    n_params = len(dataset.param_names)

    eps = 0.01  # perturbation in normalized space

    fisher_V = torch.zeros(n_params, n_params, device=device)
    fisher_ce = torch.zeros(n_params, n_params, device=device)
    fisher_phie = torch.zeros(n_params, n_params, device=device)
    fisher_combined = torch.zeros(n_params, n_params, device=device)

    indices = np.random.choice(len(dataset), min(n_samples, len(dataset)), replace=False)

    for sample_i, idx in enumerate(indices):
        batch = dataset[idx]
        coord = batch["coord"].unsqueeze(0).to(device)
        params_base = batch["params"].unsqueeze(0).to(device)
        c_rate = batch["c_rate"].unsqueeze(0).to(device)

        # Compute Jacobian via finite differences
        with torch.no_grad():
            fields_base, v_base = model(coord, params_base, c_rate)

            J_V = []    # (n_params, n_time)
            J_ce = []   # (n_params, nx*nt)
            J_phie = [] # (n_params, nx*nt)

            for j in range(n_params):
                p_plus = params_base.clone()
                p_plus[0, j] += eps
                p_minus = params_base.clone()
                p_minus[0, j] -= eps

                f_plus, v_plus = model(coord, p_plus, c_rate)
                f_minus, v_minus = model(coord, p_minus, c_rate)

                dv = ((v_plus - v_minus) / (2 * eps)).flatten()
                dce = ((f_plus[0, 0] - f_minus[0, 0]) / (2 * eps)).flatten()
                dphie = ((f_plus[0, 1] - f_minus[0, 1]) / (2 * eps)).flatten()

                J_V.append(dv)
                J_ce.append(dce)
                J_phie.append(dphie)

            J_V = torch.stack(J_V)      # (n_params, n_time)
            J_ce = torch.stack(J_ce)    # (n_params, nx*nt)
            J_phie = torch.stack(J_phie)

            fisher_V += J_V @ J_V.T
            fisher_ce += J_ce @ J_ce.T
            fisher_phie += J_phie @ J_phie.T
            fisher_combined += J_V @ J_V.T + 0.1 * (J_ce @ J_ce.T + J_phie @ J_phie.T)

        if (sample_i + 1) % 50 == 0:
            logger.info(f"  Fisher: {sample_i+1}/{len(indices)}")

    fisher_V /= len(indices)
    fisher_ce /= len(indices)
    fisher_phie /= len(indices)
    fisher_combined /= len(indices)

    return {
        "V": fisher_V.cpu().numpy(),
        "c_e": fisher_ce.cpu().numpy(),
        "phi_e": fisher_phie.cpu().numpy(),
        "combined": fisher_combined.cpu().numpy(),
    }


def analyze_fisher(fisher_dict, param_names, output_dir):
    """Analyze and visualize Fisher information."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    results = {}
    for obs_name, fim in fisher_dict.items():
        diag = np.diag(fim)
        diag_normalized = diag / (diag.max() + 1e-20)

        # Condition number
        eigvals = np.linalg.eigvalsh(fim)
        eigvals = eigvals[eigvals > 0]
        cond = eigvals.max() / (eigvals.min() + 1e-20) if len(eigvals) > 0 else np.inf

        # Identifiability (inverse of Cramer-Rao bound)
        try:
            cov = np.linalg.inv(fim + 1e-8 * np.eye(len(param_names)))
            cr_bound = np.sqrt(np.diag(cov))
        except:
            cr_bound = np.full(len(param_names), np.inf)

        results[obs_name] = {
            "diag": diag,
            "diag_normalized": diag_normalized,
            "condition_number": cond,
            "cr_bound": cr_bound,
            "eigenvalues": eigvals,
        }

        logger.info(f"\n=== Fisher Information ({obs_name}) ===")
        logger.info(f"Condition number: {cond:.2e}")
        logger.info(f"Diagonal (normalized):")
        for j, pn in enumerate(param_names):
            logger.info(f"  {pn[:35]:<36}: {diag_normalized[j]:.6f}  (CR bound: {cr_bound[j]:.4f})")

    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    obs_names = list(fisher_dict.keys())

    # Diagonal comparison
    ax = axes[0, 0]
    x = np.arange(len(param_names))
    width = 0.2
    for i, on in enumerate(obs_names):
        vals = results[on]["diag_normalized"]
        ax.bar(x + i * width, vals, width, label=on, alpha=0.8)
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels([n[:15] for n in param_names], rotation=45, ha='right', fontsize=8)
    ax.set_ylabel("Fisher Info (normalized)")
    ax.set_title("Parameter Identifiability per Observable")
    ax.legend()
    ax.set_yscale("log")

    # Eigenvalue spectrum
    ax = axes[0, 1]
    for on in obs_names:
        eigs = results[on]["eigenvalues"]
        ax.semilogy(range(len(eigs)), sorted(eigs, reverse=True), 'o-', label=on)
    ax.set_xlabel("Eigenvalue index")
    ax.set_ylabel("Eigenvalue")
    ax.set_title("FIM Eigenvalue Spectrum")
    ax.legend()

    # Fisher matrix heatmap (combined)
    ax = axes[1, 0]
    fim = fisher_dict["combined"]
    fim_norm = fim / (np.sqrt(np.diag(fim)[:, None] * np.diag(fim)[None, :]) + 1e-20)
    im = ax.imshow(fim_norm, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(param_names)))
    ax.set_yticks(range(len(param_names)))
    ax.set_xticklabels([n[:15] for n in param_names], rotation=45, ha='right', fontsize=8)
    ax.set_yticklabels([n[:15] for n in param_names], fontsize=8)
    ax.set_title("FIM Correlation (Combined)")
    plt.colorbar(im, ax=ax)

    # Information per observable
    ax = axes[1, 1]
    info_matrix = np.zeros((len(obs_names), len(param_names)))
    for i, on in enumerate(obs_names):
        info_matrix[i] = results[on]["diag_normalized"]
    im = ax.imshow(info_matrix, cmap="YlOrRd", aspect="auto")
    ax.set_yticks(range(len(obs_names)))
    ax.set_yticklabels(obs_names)
    ax.set_xticks(range(len(param_names)))
    ax.set_xticklabels([n[:15] for n in param_names], rotation=45, ha='right', fontsize=8)
    ax.set_title("Information Content per Observable per Parameter")
    plt.colorbar(im, ax=ax)

    fig.suptitle("Fisher Information Analysis", fontsize=14)
    fig.tight_layout()
    fig.savefig(f"{output_dir}/fisher_analysis.png", dpi=150)
    plt.close(fig)
    logger.info(f"Saved Fisher analysis to {output_dir}/fisher_analysis.png")

    return results


def sensitivity_weighted_identification(dataset, device, output_dir):
    """Use sensitivity-weighted loss for improved parameter identification."""
    from src.operator.fno import FNO2d
    import h5py

    stats = dataset._stats
    param_mean = stats["params"]["mean"]
    param_std = stats["params"]["std"]
    param_names = dataset.param_names

    model = FNO2d(
        num_params=len(param_names), in_channels=2, out_channels=2,
        mid_channels=64, num_layers=4,
        modes1=min(16, dataset.nx_full), modes2=min(32, dataset.n_time),
    ).to(device)
    ckpt = torch.load("outputs/checkpoints/fno_final.pt", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    with h5py.File("data/fullfield/fullfield_lmb_v2.h5", "r") as f:
        ps_ids = f["param_set_ids"][:]
        c_rates_all = f["c_rates"][:]

    # Compute per-region sensitivity weights for c_e
    # Neg electrode (indices 0-19) has highest sensitivity
    nx = dataset.nx_full
    spatial_weight = torch.ones(1, 1, nx, 1, device=device)
    spatial_weight[0, 0, :20, :] = 5.0   # neg electrode: 5× weight
    spatial_weight[0, 0, 20:40, :] = 2.0  # separator: 2× weight
    spatial_weight[0, 0, 40:, :] = 1.0    # pos electrode: 1× weight

    val_indices = dataset.val_idx.numpy() if isinstance(dataset.val_idx, torch.Tensor) else dataset.val_idx
    test_ps = np.unique(ps_ids[val_indices])[:15]

    # Test different weighting schemes
    schemes = {
        "baseline_w01": {"field_weight": 0.1, "spatial": False},
        "equal_w1": {"field_weight": 1.0, "spatial": False},
        "spatial_w5": {"field_weight": 5.0, "spatial": True},
    }

    scheme_results = {}

    for scheme_name, config in schemes.items():
        errs_list = []

        for ps_id in test_ps:
            mask = ps_ids == ps_id
            sim_indices = np.where(mask)[0]
            if len(sim_indices) < 2:
                continue

            gt_params = dataset.params[sim_indices[0]] * param_std + param_mean

            target_data = []
            for si in sim_indices:
                batch = dataset[si]
                target_data.append({
                    "coord": batch["coord"].unsqueeze(0).to(device),
                    "fields": batch["fields"].unsqueeze(0).to(device),
                    "voltage": batch["voltage"].unsqueeze(0).to(device),
                    "c_rate": batch["c_rate"].unsqueeze(0).to(device),
                })

            pred_p = torch.zeros(1, len(param_names), device=device, requires_grad=True)
            opt = torch.optim.Adam([pred_p], lr=0.01)
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=2000)
            best_loss, best_p = float("inf"), pred_p.detach().clone()

            for step in range(2000):
                total_loss = torch.tensor(0.0, device=device)
                for td in target_data:
                    fp, vp = model(td["coord"], pred_p, td["c_rate"])
                    total_loss = total_loss + torch.nn.functional.mse_loss(vp, td["voltage"])

                    if config["spatial"]:
                        weighted_diff = (fp - td["fields"]) * spatial_weight
                        total_loss = total_loss + config["field_weight"] * (weighted_diff ** 2).mean()
                    else:
                        total_loss = total_loss + config["field_weight"] * torch.nn.functional.mse_loss(fp, td["fields"])

                opt.zero_grad(); total_loss.backward(); opt.step(); sched.step()
                if total_loss.item() < best_loss:
                    best_loss = total_loss.item()
                    best_p = pred_p.detach().clone()

            rec = best_p.cpu().numpy()[0] * param_std + param_mean
            err = np.abs(rec - gt_params) / (np.abs(gt_params) + 1e-10) * 100
            errs_list.append(err)

        errs = np.array(errs_list)
        scheme_results[scheme_name] = errs

        logger.info(f"\n=== {scheme_name} ===")
        for j, pn in enumerate(param_names):
            logger.info(f"  {pn[:35]:<36}: {errs[:, j].mean():>8.1f}%")
        logger.info(f"  Overall: {errs.mean():>8.1f}%")

    # Save results
    np.savez(f"{output_dir}/sensitivity_weighted_results.npz",
             scheme_results=scheme_results, param_names=param_names, allow_pickle=True)

    return scheme_results


def main():
    import sys
    data_path = sys.argv[1] if len(sys.argv) > 1 else "data/fullfield/fullfield_lmb_v2.h5"
    ckpt_path = sys.argv[2] if len(sys.argv) > 2 else "outputs/checkpoints/fno_final.pt"
    gpu_id = sys.argv[3] if len(sys.argv) > 3 else "0"

    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id

    logger.info("=" * 60)
    logger.info("Pipeline 3: Fisher Information + Sensitivity Analysis")
    logger.info("=" * 60)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    from src.operator.fno import FNO2d
    from src.operator.dataset import FullFieldDataset

    dataset = FullFieldDataset(data_path, fields=["c_e", "phi_e"], normalize=True)
    param_names = dataset.param_names
    stats = dataset._stats

    model = FNO2d(
        num_params=len(param_names), in_channels=2, out_channels=2,
        mid_channels=64, num_layers=4,
        modes1=min(16, dataset.nx_full), modes2=min(32, dataset.n_time),
    ).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # Step 1: Fisher information
    logger.info("Step 1: Computing Fisher Information Matrix...")
    t0 = time.time()
    fisher_dict = compute_fisher_matrix(model, dataset, device, n_samples=300)
    logger.info(f"Fisher computation took {time.time()-t0:.0f}s")

    # Step 2: Analyze
    logger.info("Step 2: Analyzing Fisher information...")
    fisher_results = analyze_fisher(fisher_dict, param_names, "outputs/fisher")

    # Step 3: Sensitivity-weighted identification
    logger.info("Step 3: Sensitivity-weighted identification...")
    scheme_results = sensitivity_weighted_identification(dataset, device, "outputs/fisher")

    logger.info("Pipeline 3 COMPLETE!")


if __name__ == "__main__":
    main()
