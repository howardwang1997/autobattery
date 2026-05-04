"""Build a differential-voltage signature library for LMB.

Reads a synthetic LMB dataset (produced by ``01_generate_synthetic.py``
under the new plating-aware ``configs/lmb.yaml``), filters to the
signature C-rate, normalises parameters per the config, and writes a
:class:`src.diagnosis.SignatureLibrary` to disk.

Usage::

    python scripts/32_signature_library_lmb.py \
        --config configs/lmb.yaml \
        --data data/synthetic/synthetic_lmb.npz \
        --output outputs/diagnosis/signature_library_lmb.npz \
        --bootstrap 200
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("signature_library_lmb")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/lmb.yaml")
    parser.add_argument("--data", default="data/synthetic/synthetic_lmb.npz")
    parser.add_argument("--output", default="outputs/diagnosis/signature_library_lmb.npz")
    parser.add_argument("--bootstrap", type=int, default=0,
                        help="Bootstrap resamples for signature CIs (0 to skip)")
    parser.add_argument("--c-rate", type=float, default=None,
                        help="Override the signature C-rate. Defaults to config value")
    parser.add_argument("--ridge-alpha", type=float, default=None)
    parser.add_argument("--n-time", type=int, default=None)
    args = parser.parse_args()

    from src.simulation.parameters import load_config
    from src.diagnosis import build_signature_library

    cfg = load_config(args.config)
    sig_cfg = cfg.get("signature_library", {})
    c_rate = args.c_rate if args.c_rate is not None else sig_cfg.get("c_rate", 1.0)
    n_time = args.n_time if args.n_time is not None else sig_cfg.get("n_time", 200)
    ridge_alpha = args.ridge_alpha if args.ridge_alpha is not None \
        else sig_cfg.get("ridge_alpha", 0.1)
    log_scale_params = tuple(sig_cfg.get("log_scale_params", ()))

    logger.info("Loading synthetic data from %s", args.data)
    npz = np.load(args.data, allow_pickle=False)
    times = npz["times"]
    voltages = npz["voltages"]
    masks = npz["masks"]
    c_rates = npz["c_rates"]
    param_values = npz["param_values"]
    param_names = tuple(str(s) for s in npz["param_names"])

    # Select rows at the signature C-rate, resample to a common time grid.
    sel = np.where(np.abs(c_rates - c_rate) < 0.01)[0]
    if len(sel) < 50:
        raise SystemExit(
            f"Only {len(sel)} curves at C={c_rate}; need >=50 for stable signatures."
        )
    logger.info("Using %d curves at C=%.2f", len(sel), c_rate)

    target_t = np.linspace(0.0, 1.0, n_time)
    V = np.zeros((len(sel), n_time), dtype=np.float64)
    for i, idx in enumerate(sel):
        m = masks[idx]
        t = times[idx][m]
        v = voltages[idx][m]
        if len(t) < 5:
            continue
        t_norm = (t - t[0]) / max(t[-1] - t[0], 1e-9)
        V[i] = np.interp(target_t, t_norm, v)

    P = param_values[sel]

    library = build_signature_library(
        V_sim=V,
        P_sim=P,
        param_names=param_names,
        log_scale_params=log_scale_params,
        time_grid=target_t,
        ridge_alpha=ridge_alpha,
        bootstrap_samples=args.bootstrap,
        rng=np.random.default_rng(0),
        meta={
            "c_rate": float(c_rate),
            "n_curves": int(len(sel)),
            "config": str(args.config),
            "data": str(args.data),
        },
    )

    out_path = Path(args.output)
    library.save(out_path)
    logger.info("Saved signature library to %s", out_path)

    # Log a quick sanity summary: signature L2 norm per parameter.
    for name, sig in zip(library.param_names, library.signatures):
        logger.info("  %-55s ||sig||=%.4f", name, float(np.linalg.norm(sig)))


if __name__ == "__main__":
    main()
