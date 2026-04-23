import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional, Union


def plot_voltage_comparison(
    t_exp: np.ndarray,
    v_exp: np.ndarray,
    t_pred: np.ndarray,
    v_pred: np.ndarray,
    title: str = "Voltage Comparison",
    save_path: Optional[Union[str, Path]] = None,
):
    """Plot experimental vs predicted voltage curves."""
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    axes[0].plot(t_exp, v_exp, "k-", label="Experimental", linewidth=1.5)
    axes[0].plot(t_pred, v_pred, "r--", label="PINN Prediction", linewidth=1.5)
    axes[0].set_ylabel("Voltage [V]")
    axes[0].legend()
    axes[0].set_title(title)
    axes[0].grid(True, alpha=0.3)

    t_interp = np.interp(t_pred, t_exp, v_exp)
    error = v_pred - t_interp
    axes[1].plot(t_pred, error * 1000, "b-", linewidth=1.0)
    axes[1].axhline(y=0, color="k", linestyle="--", linewidth=0.5)
    axes[1].set_xlabel("Time [s]")
    axes[1].set_ylabel("Error [mV]")
    axes[1].set_title(f"Prediction Error (MAE={np.abs(error).mean()*1000:.2f} mV)")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig


def plot_training_history(
    history: dict,
    save_path: Optional[Union[str, Path]] = None,
):
    """Plot training loss curves."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    if "train_loss" in history:
        axes[0].semilogy(history["train_loss"], label="Total Loss")
    if "data_loss" in history:
        axes[0].semilogy(history["data_loss"], label="Data Loss")
    if "pde_loss" in history:
        axes[0].semilogy(history["pde_loss"], label="PDE Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training Losses")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    if "params" in history:
        for name, values in history["params"].items():
            axes[1].plot(values, label=name)
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Parameter Value")
        axes[1].set_title("Learnable Parameters")
        axes[1].legend()
        axes[1].set_yscale("log")
        axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig


def plot_multi_crate(
    results: dict[float, tuple[np.ndarray, np.ndarray]],
    save_path: Optional[Union[str, Path]] = None,
):
    """Plot voltage curves at multiple C-rates."""
    fig, ax = plt.subplots(figsize=(10, 6))

    for c_rate, (t, v) in sorted(results.items()):
        t_hours = t / 3600.0
        ax.plot(t_hours, v, label=f"{c_rate}C")

    ax.set_xlabel("Time [hours]")
    ax.set_ylabel("Voltage [V]")
    ax.set_title("Voltage Curves at Multiple C-rates")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig


def plot_parameter_heatmap(
    param_matrix: np.ndarray,
    param_names: list[str],
    metric: np.ndarray,
    metric_name: str = "MAE [mV]",
    save_path: Optional[Union[str, Path]] = None,
):
    """Plot heatmap of prediction error vs parameter combinations."""
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(metric.reshape(-1, 1) if metric.ndim == 1 else metric, aspect="auto")
    ax.set_ylabel("Parameter Combination")
    ax.set_title(f"{metric_name} across Parameter Space")
    plt.colorbar(im, ax=ax, label=metric_name)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig
