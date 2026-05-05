"""Rank robustness study for the Fisher rank-3 identifiability claim.

This script directly answers the four reviewer attacks on the rank=3
result documented in the publication roadmap:

  1. **η-threshold ambiguity** — "rank=3" hides the choice of cutoff.
     We sweep η ∈ {1e-2, 1e-3, 1e-6, 1e-9, 1e-12} and report η-rank at
     every level. Sloppy-model spectra decay roughly geometrically
     (Transtrum & Sethna 2014, *PRE*); reporting the full spectrum
     plus discrete rank counts makes the claim falsifiable.

  2. **Parameterisation dependence** — Fisher rank is *not* invariant
     under non-linear reparameterisation. We compute rank under four
     parameterisations: raw linear, log10, standardised log, and PCA-
     whitened. If rank stays ≈3 across all four, the claim is robust;
     if not, we report it as parameterisation-dependent.

  3. **Operating-condition dependence** — single-condition Fisher
     under-counts the rank a multi-condition protocol can achieve.
     Sulzer et al. 2021 already showed this for DFN. We compute Fisher
     per C-rate, then a joint multi-C-rate Fisher, and report the
     rank-gain. This is *also* the physical justification for adding
     GITT/HPPC observables.

  4. **Bootstrap stability** — eigenvalues are noisy point estimates.
     We bootstrap-resample the synthetic dataset 200× and report
     median + 95% CI on each rank.

Inputs
------
``--data-h5``  HDF5 file with at least ``V`` (n_sim, n_time),
                ``params`` (n_sim, n_params), and ``c_rates`` (n_sim,).
                Default: ``data/fullfield/fullfield_lfp_degradation.h5``.
``--param-names``  comma list, default
                   ``"SEI,LAM_neg,LAM_pos,D_n,D_p,t+,R_mult"``.
``--output``   output dir, default ``outputs/rank_robustness``.

Outputs
-------
``rank_table.json``     full rank dict, indexed by
                        (parameterisation, condition, eta).
``rank_table.md``       paper-ready markdown table.
``spectrum.png``        eigenvalue spectra (one curve per
                        parameterisation, log-y).
``rank_vs_eta.png``     rank-as-a-function-of-η for each
                        parameterisation.
``rank_gain_multirate.png``   single-rate vs multi-rate rank lift.

Run
---
    conda activate autobattery
    python scripts/99_rank_robustness.py \
        --data-h5 data/fullfield/fullfield_lfp_degradation.h5 \
        --output outputs/rank_robustness/lfp

Pure-numpy + h5py + matplotlib; CPU-only; ~2-5 minutes for 1200 sims.
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Sequence

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger("rank_robustness")

ETA_LEVELS = [1e-2, 1e-3, 1e-6, 1e-9, 1e-12]
PARAMETERISATIONS = ("raw", "log", "log_standardised", "pca_whitened")
DEFAULT_RIDGE_ALPHA = 1e-3                # tiny ridge to keep solve well-posed
DEFAULT_BOOTSTRAP = 200


# ---------------------------------------------------------------------------
# Core math
# ---------------------------------------------------------------------------


def transform_params(P: np.ndarray, mode: str) -> np.ndarray:
    """Map raw params to a working representation."""
    if mode == "raw":
        return P.astype(np.float64).copy()
    if mode == "log":
        return np.log10(np.clip(P, 1e-30, None)).astype(np.float64)
    if mode == "log_standardised":
        L = np.log10(np.clip(P, 1e-30, None))
        L_centered = L - L.mean(axis=0, keepdims=True)
        std = L_centered.std(axis=0, keepdims=True) + 1e-12
        return L_centered / std
    if mode == "pca_whitened":
        L = np.log10(np.clip(P, 1e-30, None))
        L_centered = L - L.mean(axis=0, keepdims=True)
        cov = (L_centered.T @ L_centered) / max(L.shape[0] - 1, 1)
        eigvals, eigvecs = np.linalg.eigh(cov)
        eigvals = np.clip(eigvals, 1e-12, None)
        whitener = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
        return L_centered @ whitener
    raise ValueError(f"unknown parameterisation: {mode}")


def signature_matrix(
    V: np.ndarray, P: np.ndarray, alpha: float = DEFAULT_RIDGE_ALPHA,
) -> np.ndarray:
    """Empirical Jacobian J = ∂V/∂θ via ridge regression.

    Both V and P are demeaned column-wise; the regression has an
    implicit intercept. Returns shape (n_params, n_time).
    """
    P_centered = P - P.mean(axis=0, keepdims=True)
    V_centered = V - V.mean(axis=0, keepdims=True)
    n = P.shape[1]
    PtP = P_centered.T @ P_centered
    return np.linalg.solve(PtP + alpha * np.eye(n), P_centered.T @ V_centered)


def fisher_eigenvalues(J: np.ndarray, sigma_v: float = 1.0) -> np.ndarray:
    """λ_i of F = J Jᵀ / σ², sorted descending."""
    F = (J @ J.T) / (sigma_v ** 2)
    eigs = np.linalg.eigvalsh(F)
    return np.sort(np.abs(eigs))[::-1]


def eta_rank(eigenvalues: np.ndarray, eta: float) -> int:
    if len(eigenvalues) == 0 or eigenvalues[0] == 0:
        return 0
    return int((eigenvalues / eigenvalues[0] > eta).sum())


# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------


def bootstrap_eigenvalues(
    V: np.ndarray, P: np.ndarray, n_boot: int, alpha: float, seed: int = 0,
) -> np.ndarray:
    """Return shape (n_boot, n_params) of sorted-descending eigenvalues."""
    rng = np.random.default_rng(seed)
    n_sim, n_params = P.shape
    out = np.zeros((n_boot, n_params))
    for b in range(n_boot):
        idx = rng.integers(0, n_sim, size=n_sim)
        J = signature_matrix(V[idx], P[idx], alpha=alpha)
        out[b] = fisher_eigenvalues(J)
    return out


# ---------------------------------------------------------------------------
# Multi-rate joint Fisher
# ---------------------------------------------------------------------------


def joint_multi_rate_fisher(
    V: np.ndarray,
    P_transformed: np.ndarray,
    c_rates: np.ndarray,
    available_rates: Sequence[float],
    alpha: float,
    min_per_rate: int = 50,
) -> tuple[np.ndarray, list[float]]:
    """Sum per-rate Fisher matrices to get the joint protocol rank."""
    n_params = P_transformed.shape[1]
    F_joint = np.zeros((n_params, n_params))
    used = []
    for cr in available_rates:
        mask = np.abs(c_rates - cr) < 0.01
        if mask.sum() < min_per_rate:
            continue
        J_cr = signature_matrix(V[mask], P_transformed[mask], alpha=alpha)
        F_joint += (J_cr @ J_cr.T)
        used.append(float(cr))
    return F_joint, used


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_spectrum(
    spectra: dict[str, np.ndarray], out_path: Path, param_names: list[str],
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 5))
    indices = np.arange(1, len(param_names) + 1)
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    for k, (mode, eigs) in enumerate(spectra.items()):
        normed = eigs / eigs[0]
        ax.semilogy(indices, normed, "o-", label=mode, color=colors[k % len(colors)])
    for eta in ETA_LEVELS:
        ax.axhline(eta, color="grey", lw=0.4, ls=":")
        ax.text(indices[-1], eta, f"η={eta:.0e}",
                fontsize=7, color="grey", va="bottom", ha="right")
    ax.set_xlabel("eigenvalue index (sorted descending)")
    ax.set_ylabel("λ_i / λ_1")
    ax.set_title("Fisher information spectrum across parameterisations")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_rank_vs_eta(
    rank_dict: dict, out_path: Path,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 5))
    for mode, by_eta in rank_dict.items():
        etas = sorted(by_eta.keys(), reverse=True)
        ranks = [by_eta[e]["rank_median"] for e in etas]
        lo = [by_eta[e]["rank_q05"] for e in etas]
        hi = [by_eta[e]["rank_q95"] for e in etas]
        ax.errorbar(
            etas, ranks,
            yerr=[np.array(ranks) - np.array(lo), np.array(hi) - np.array(ranks)],
            marker="o", capsize=3, label=mode,
        )
    ax.set_xscale("log")
    ax.set_xlabel("η (relative eigenvalue threshold)")
    ax.set_ylabel("η-rank")
    ax.set_title("Rank vs threshold (with bootstrap 95% CI)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_multi_rate_gain(
    single_ranks: dict[float, int], joint_rank: int,
    out_path: Path,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rates = sorted(single_ranks.keys())
    bars_x = list(range(len(rates) + 1))
    bars_y = [single_ranks[r] for r in rates] + [joint_rank]
    labels = [f"single C/{1/r:.0f}" if r < 1 else f"single {r:.1f}C"
              for r in rates] + ["joint multi-rate"]
    colors = ["tab:blue"] * len(rates) + ["tab:red"]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(bars_x, bars_y, color=colors)
    ax.set_xticks(bars_x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("η-rank @ η=1e-6")
    ax.set_title("Rank gain from joint multi-rate Fisher")
    for x, y in zip(bars_x, bars_y):
        ax.text(x, y + 0.05, str(y), ha="center", va="bottom", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Markdown table
# ---------------------------------------------------------------------------


@dataclass
class RankCell:
    rank_median: int
    rank_q05: int
    rank_q95: int


def _render_md_table(table: dict, etas: list[float]) -> str:
    rows = ["| parameterisation | condition | " + " | ".join(f"η={e:.0e}" for e in etas) + " |",
            "|---" * (2 + len(etas)) + "|"]
    for mode, conds in table.items():
        for cond, by_eta in conds.items():
            cells = [
                f"**{by_eta[e]['rank_median']}** [{by_eta[e]['rank_q05']}–{by_eta[e]['rank_q95']}]"
                for e in etas
            ]
            rows.append(f"| {mode} | {cond} | " + " | ".join(cells) + " |")
    return "\n".join(rows)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-h5",
                        default="data/fullfield/fullfield_lfp_degradation.h5")
    parser.add_argument("--param-names",
                        default="SEI,LAM_neg,LAM_pos,D_n,D_p,t+,R_mult")
    parser.add_argument("--output", default="outputs/rank_robustness")
    parser.add_argument("--bootstrap", type=int, default=DEFAULT_BOOTSTRAP)
    parser.add_argument("--ridge-alpha", type=float, default=DEFAULT_RIDGE_ALPHA)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--c-rate-key", default="c_rates",
                        help="HDF5 dataset name for per-sim C-rate")
    parser.add_argument("--no-multi-rate", action="store_true",
                        help="Skip multi-rate joint Fisher (set if data is single-rate)")
    args = parser.parse_args()

    import h5py
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    param_names = args.param_names.split(",")
    n_params = len(param_names)

    # Load -----------------------------------------------------------
    logger.info("loading %s", args.data_h5)
    with h5py.File(args.data_h5, "r") as f:
        V = f["V"][:].astype(np.float64)
        P = f["params"][:].astype(np.float64)
        try:
            c_rates = f[args.c_rate_key][:].astype(np.float64)
        except KeyError:
            logger.warning("no %s in h5 — multi-rate analysis disabled",
                           args.c_rate_key)
            c_rates = None

    if P.shape[1] != n_params:
        raise SystemExit(f"param count mismatch: data has {P.shape[1]}, "
                         f"--param-names lists {n_params}")
    logger.info("loaded V%s, P%s", V.shape, P.shape)

    # Spectrum + per-η rank table ------------------------------------
    spectra: dict[str, np.ndarray] = {}
    full_table: dict[str, dict[str, dict[float, dict]]] = {}

    for mode in PARAMETERISATIONS:
        logger.info("parameterisation: %s", mode)
        P_t = transform_params(P, mode)

        # Single condition = all data, full rank Fisher.
        J_full = signature_matrix(V, P_t, alpha=args.ridge_alpha)
        eigs_full = fisher_eigenvalues(J_full)
        spectra[mode] = eigs_full

        # Bootstrap distribution.
        eigs_boot = bootstrap_eigenvalues(
            V, P_t, args.bootstrap, args.ridge_alpha, seed=args.seed,
        )

        cond_table: dict[str, dict[float, dict]] = {}
        cond_table["all_data_singleC_pooled"] = {}
        for eta in ETA_LEVELS:
            ranks = np.array([eta_rank(eigs_boot[b], eta)
                              for b in range(args.bootstrap)])
            cond_table["all_data_singleC_pooled"][eta] = {
                "rank_median": int(np.median(ranks)),
                "rank_q05": int(np.quantile(ranks, 0.05)),
                "rank_q95": int(np.quantile(ranks, 0.95)),
                "rank_mean": float(ranks.mean()),
                "rank_std": float(ranks.std()),
            }

        # Multi-rate joint Fisher (only meaningful for 'log_standardised'
        # since rank is invariant under affine transforms but the joint
        # operation needs a consistent scale).
        if c_rates is not None and not args.no_multi_rate:
            unique_rates = sorted(set(np.round(c_rates, 2).tolist()))
            # Per-rate single-condition rank (no bootstrap, just point).
            for cr in unique_rates:
                mask = np.abs(c_rates - cr) < 0.01
                if mask.sum() < 30:
                    continue
                J_cr = signature_matrix(V[mask], P_t[mask], alpha=args.ridge_alpha)
                eigs_cr = fisher_eigenvalues(J_cr)
                key = f"single_C={cr:g}"
                cond_table.setdefault(key, {})
                for eta in ETA_LEVELS:
                    cond_table[key][eta] = {
                        "rank_median": eta_rank(eigs_cr, eta),
                        "rank_q05": eta_rank(eigs_cr, eta),
                        "rank_q95": eta_rank(eigs_cr, eta),
                    }

            # Joint multi-rate.
            F_joint, used = joint_multi_rate_fisher(
                V, P_t, c_rates, unique_rates, alpha=args.ridge_alpha,
            )
            eigs_joint = np.sort(np.abs(np.linalg.eigvalsh(F_joint)))[::-1]
            cond_table[f"joint_multi_rate({len(used)}_rates)"] = {
                eta: {
                    "rank_median": eta_rank(eigs_joint, eta),
                    "rank_q05": eta_rank(eigs_joint, eta),
                    "rank_q95": eta_rank(eigs_joint, eta),
                } for eta in ETA_LEVELS
            }

        full_table[mode] = cond_table

    # Save -----------------------------------------------------------
    out_json = {
        "param_names": param_names,
        "n_simulations": int(V.shape[0]),
        "n_time": int(V.shape[1]),
        "eta_levels": ETA_LEVELS,
        "bootstrap": args.bootstrap,
        "spectra": {m: e.tolist() for m, e in spectra.items()},
        "rank_table": {
            m: {c: {f"{e:g}": v for e, v in by_eta.items()}
                for c, by_eta in conds.items()}
            for m, conds in full_table.items()
        },
    }
    with (out / "rank_table.json").open("w") as f:
        json.dump(out_json, f, indent=2, default=str)

    # Markdown table.
    md_lines = [
        "# Fisher rank robustness study",
        "",
        f"Dataset: `{args.data_h5}` ({V.shape[0]} sims, {V.shape[1]} time pts)",
        f"Bootstrap: {args.bootstrap} resamples, ridge α={args.ridge_alpha:.0e}",
        "",
        "## η-rank table (median [5th–95th] from bootstrap)",
        "",
    ]
    md_lines.append(_render_md_table(full_table, ETA_LEVELS))
    md_lines += [
        "",
        "## Per-parameterisation eigenvalue spectra",
        "",
        "See `spectrum.png`.",
        "",
        "## Reading the table for the paper",
        "",
        "- If rank stays at 3 across all four parameterisations and across "
        "all η levels above 1e-6, the universal-rank-3 claim is robust.",
        "- If rank rises in `joint_multi_rate(...)` rows above the per-rate "
        "rank, you have a quantitative argument for adding new operating "
        "conditions (multi-C-rate, multi-T) — this is the physical "
        "justification for GITT/HPPC retrofits.",
        "- η=1e-12 should generally show full rank (= n_params); if it "
        "doesn't, your dataset has degenerate parameter directions.",
    ]
    with (out / "rank_table.md").open("w") as f:
        f.write("\n".join(md_lines))

    # Plots ----------------------------------------------------------
    plot_spectrum(spectra, out / "spectrum.png", param_names)

    rank_for_plot = {m: full_table[m]["all_data_singleC_pooled"]
                     for m in PARAMETERISATIONS}
    plot_rank_vs_eta(rank_for_plot, out / "rank_vs_eta.png")

    if c_rates is not None and not args.no_multi_rate:
        # Pick η=1e-6 and the "log_standardised" parameterisation for the bar plot.
        single_ranks: dict[float, int] = {}
        joint_rank = -1
        for cond, by_eta in full_table["log_standardised"].items():
            if cond.startswith("single_C="):
                cr = float(cond.split("=")[1])
                single_ranks[cr] = by_eta[1e-6]["rank_median"]
            elif cond.startswith("joint_multi_rate"):
                joint_rank = by_eta[1e-6]["rank_median"]
        if single_ranks and joint_rank >= 0:
            plot_multi_rate_gain(single_ranks, joint_rank,
                                 out / "rank_gain_multirate.png")

    logger.info("done. summary at %s/rank_table.md", out)


if __name__ == "__main__":
    main()
