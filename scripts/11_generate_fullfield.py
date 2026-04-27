"""Generate full-field P2D simulation data for Neural Operator training."""

import argparse
import logging
import time
import numpy as np
import h5py
from pathlib import Path
from multiprocessing import Pool
from functools import partial

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

from src.simulation.models import MetalBatteryDFN
from src.simulation.solver import PybammSolver
from src.simulation.parameters import load_config, parse_sweep_params


def _run_single(args):
    """Run a single full-field simulation. Accepts tuple for multiprocessing."""
    idx, param_vector, param_names, c_rate, temperature, t_end, n_time = args

    try:
        model = MetalBatteryDFN(chemistry="lmb")
        params = model.get_default_parameters()
        for name, value in zip(param_names, param_vector):
            params[name] = value
        params["Nominal cell capacity [A.h]"] = 3.0

        solver = PybammSolver(model)
        result = solver.solve_full_field(
            params, c_rate=c_rate, temperature=temperature,
            t_end=t_end, n_time=n_time,
        )
        if result is None:
            return None

        result["param_vector"] = param_vector
        result["sim_idx"] = idx
        return result

    except Exception as e:
        logger.warning(f"Sim {idx} failed: {e}")
        return None


def _resample_1d(arr, n_out):
    """Resample 1D array to n_out points."""
    n_in = arr.shape[-1]
    if n_in == n_out:
        return arr.astype(np.float32)
    x_in = np.linspace(0, 1, n_in)
    x_out = np.linspace(0, 1, n_out)
    return np.interp(x_out, x_in, arr).astype(np.float32)


def _resample_2d(arr, n_out):
    """Resample 2D array (Nx, Nt) → (Nx, n_out) along time axis."""
    n_in = arr.shape[-1]
    if n_in == n_out:
        return arr.astype(np.float32)
    x_in = np.linspace(0, 1, n_in)
    x_out = np.linspace(0, 1, n_out)
    result = np.zeros((arr.shape[0], n_out), dtype=np.float32)
    for i in range(arr.shape[0]):
        result[i] = np.interp(x_out, x_in, arr[i])
    return result


def _resample_3d(arr, n_out):
    """Resample 3D array (Nr, Nx, Nt) → (Nr, Nx, n_out) along time axis."""
    n_in = arr.shape[-1]
    if n_in == n_out:
        return arr.astype(np.float32)
    x_in = np.linspace(0, 1, n_in)
    x_out = np.linspace(0, 1, n_out)
    result = np.zeros((arr.shape[0], arr.shape[1], n_out), dtype=np.float32)
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            result[i, j] = np.interp(x_out, x_in, arr[i, j])
    return result


def main():
    parser = argparse.ArgumentParser(description="Generate full-field data")
    parser.add_argument("--config", type=str, default="configs/lmb.yaml")
    parser.add_argument("--output", type=str, default="data/fullfield/fullfield_lmb.h5")
    parser.add_argument("--num-samples", type=int, default=5000)
    parser.add_argument("--n-time", type=int, default=100)
    parser.add_argument("--num-workers", type=int, default=32)
    args = parser.parse_args()

    config = load_config(args.config)
    sim_cfg = config["simulation"]

    sweep_params = parse_sweep_params(config)
    param_names = [p.name for p in sweep_params]
    param_ranges = sweep_params
    c_rates = sim_cfg.get("c_rates", [0.1, 0.2, 0.5, 1.0])
    t_end = sim_cfg.get("t_end", 3600)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(42)
    n_per_crate = args.num_samples // len(c_rates)
    total = n_per_crate * len(c_rates)

    logger.info(
        f"Generating {total} full-field simulations "
        f"({n_per_crate} per C-rate × {len(c_rates)} C-rates), "
        f"{args.n_time} time points, {args.num_workers} workers"
    )

    param_vectors = []
    for pr in param_ranges:
        if pr.distribution == "log_uniform":
            values = 10 ** rng.uniform(np.log10(pr.min), np.log10(pr.max), total)
        else:
            values = rng.uniform(pr.min, pr.max, total)
        param_vectors.append(values)
    param_vectors = np.stack(param_vectors, axis=-1)

    tasks = []
    idx = 0
    for c_rate in c_rates:
        for i in range(n_per_crate):
            tasks.append((
                idx, param_vectors[idx], param_names,
                c_rate, 25.0, t_end, args.n_time,
            ))
            idx += 1

    t_start = time.time()
    results = []
    failed = 0

    if args.num_workers > 1:
        with Pool(processes=args.num_workers) as pool:
            for i, result in enumerate(pool.imap_unordered(_run_single, tasks, chunksize=8)):
                if result is not None:
                    results.append(result)
                else:
                    failed += 1
                if (i + 1) % 200 == 0:
                    elapsed = time.time() - t_start
                    rate = (i + 1) / elapsed
                    logger.info(
                        f"Progress: {i+1}/{len(tasks)} "
                        f"({len(results)} ok, {failed} failed) "
                        f"[{rate:.1f} sim/s, {elapsed:.0f}s]"
                    )
    else:
        for i, task in enumerate(tasks):
            result = _run_single(task)
            if result is not None:
                results.append(result)
            else:
                failed += 1
            if (i + 1) % 100 == 0:
                elapsed = time.time() - t_start
                logger.info(f"Progress: {i+1}/{len(tasks)} ({len(results)} ok)")

    elapsed = time.time() - t_start
    logger.info(f"Generated {len(results)} simulations ({failed} failed) in {elapsed:.0f}s")

    ref = results[0]
    nt = args.n_time

    nx_full = ref["c_e"].shape[0]
    nx_neg = ref["phi_s_neg"].shape[0]
    nx_pos = ref["phi_s_pos"].shape[0]
    nr_pos = ref["c_s_pos"].shape[0]

    n = len(results)
    c_e_all = np.zeros((n, nx_full, nt), dtype=np.float32)
    phi_e_all = np.zeros((n, nx_full, nt), dtype=np.float32)
    phi_s_neg_all = np.zeros((n, nx_neg, nt), dtype=np.float32)
    phi_s_pos_all = np.zeros((n, nx_pos, nt), dtype=np.float32)
    c_s_pos_all = np.zeros((n, nr_pos, nx_pos, nt), dtype=np.float32)
    j_pos_all = np.zeros((n, nx_pos, nt), dtype=np.float32)
    j_neg_all = np.zeros((n, nx_neg, nt), dtype=np.float32)
    L_sei_all = np.zeros((n, nx_neg, nt), dtype=np.float32)
    V_all = np.zeros((n, nt), dtype=np.float32)
    params_all = np.zeros((n, len(param_names)), dtype=np.float64)
    crates_all = np.zeros(n, dtype=np.float32)

    for i, r in enumerate(results):
        c_e_all[i] = _resample_2d(r["c_e"], nt)
        phi_e_all[i] = _resample_2d(r["phi_e"], nt)
        phi_s_neg_all[i] = _resample_2d(r["phi_s_neg"], nt)
        phi_s_pos_all[i] = _resample_2d(r["phi_s_pos"], nt)
        c_s_pos_all[i] = _resample_3d(r["c_s_pos"], nt)
        j_pos_all[i] = _resample_2d(r["j_pos"], nt)
        j_neg_all[i] = _resample_2d(r["j_neg"], nt)
        L_sei_all[i] = _resample_2d(r["L_sei"], nt)
        V_all[i] = _resample_1d(r["V"], nt)
        params_all[i] = r["param_vector"]
        crates_all[i] = r["c_rate"]

    logger.info("Compressing and saving to HDF5...")
    with h5py.File(args.output, "w") as f:
        f.attrs["n_sims"] = n
        f.attrs["n_time"] = nt
        f.attrs["nx_full"] = nx_full
        f.attrs["nx_neg"] = nx_neg
        f.attrs["nx_pos"] = nx_pos
        f.attrs["nr_pos"] = nr_pos
        f.attrs["param_names"] = np.array(param_names, dtype="S")
        f.attrs["c_rates_used"] = np.unique(crates_all)

        kw = dict(compression="gzip", compression_opts=4)
        f.create_dataset("c_e", data=c_e_all, **kw)
        f.create_dataset("phi_e", data=phi_e_all, **kw)
        f.create_dataset("phi_s_neg", data=phi_s_neg_all, **kw)
        f.create_dataset("phi_s_pos", data=phi_s_pos_all, **kw)
        f.create_dataset("c_s_pos", data=c_s_pos_all, **kw)
        f.create_dataset("j_pos", data=j_pos_all, **kw)
        f.create_dataset("j_neg", data=j_neg_all, **kw)
        f.create_dataset("L_sei", data=L_sei_all, **kw)
        f.create_dataset("V", data=V_all, **kw)
        f.create_dataset("params", data=params_all, **kw)
        f.create_dataset("c_rates", data=crates_all, **kw)

    fsize = Path(args.output).stat().st_size / 1e9
    logger.info(f"Saved {n} simulations to {args.output} ({fsize:.2f} GB)")
    logger.info(
        f"Shapes: c_e={c_e_all.shape}, phi_e={phi_e_all.shape}, "
        f"c_s_pos={c_s_pos_all.shape}, V={V_all.shape}"
    )


if __name__ == "__main__":
    main()
