"""Regenerate full-field data with shared parameter sets across C-rates.

Key change: N_param_sets unique parameters, each simulated at ALL C-rates.
This enables multi-C-rate joint parameter identification.
"""

import logging
import time
import numpy as np
import h5py
from pathlib import Path
from multiprocessing import Pool

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

from src.simulation.models import MetalBatteryDFN
from src.simulation.solver import PybammSolver
from src.simulation.parameters import load_config, parse_sweep_params


def _run_single(args):
    idx, param_vector, param_names, c_rate, t_end, n_time = args
    try:
        model = MetalBatteryDFN(chemistry="lmb")
        params = model.get_default_parameters()
        for name, value in zip(param_names, param_vector):
            params[name] = value
        params["Nominal cell capacity [A.h]"] = 3.0

        solver = PybammSolver(model)
        result = solver.solve_full_field(
            params, c_rate=c_rate, temperature=25.0,
            t_end=t_end, n_time=n_time,
        )
        if result is None:
            return None
        result["param_vector"] = param_vector
        result["sim_idx"] = idx
        result["c_rate"] = c_rate
        return result
    except Exception as e:
        return None


def _resample_2d(arr, n_out):
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
    config = load_config("configs/lmb.yaml")
    sim_cfg = config["simulation"]
    sweep_params = parse_sweep_params(config)
    param_names = [p.name for p in sweep_params]
    c_rates = [0.1, 0.2, 0.5, 1.0]
    t_end = sim_cfg.get("t_end", 3600)
    n_time = 100
    n_param_sets = 750

    output_path = "data/fullfield/fullfield_lmb_v2.h5"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(42)

    param_vectors = []
    for pr in sweep_params:
        if pr.distribution == "log_uniform":
            values = 10 ** rng.uniform(np.log10(pr.min), np.log10(pr.max), n_param_sets)
        else:
            values = rng.uniform(pr.min, pr.max, n_param_sets)
        param_vectors.append(values)
    param_vectors = np.stack(param_vectors, axis=-1).astype(np.float64)

    total = n_param_sets * len(c_rates)
    logger.info(f"Generating {total} sims: {n_param_sets} params × {len(c_rates)} C-rates")

    tasks = []
    idx = 0
    for pi in range(n_param_sets):
        for c_rate in c_rates:
            tasks.append((idx, param_vectors[pi], param_names, c_rate, t_end, n_time))
            idx += 1

    t_start = time.time()
    results = []
    failed = 0

    with Pool(processes=24) as pool:
        for i, result in enumerate(pool.imap_unordered(_run_single, tasks, chunksize=8)):
            if result is None:
                failed += 1
            else:
                results.append(result)
            if (i + 1) % 200 == 0:
                elapsed = time.time() - t_start
                rate = (i + 1) / elapsed
                logger.info(f"Progress: {i+1}/{total} ({failed} failed), {rate:.1f} sims/s")

    elapsed = time.time() - t_start
    logger.info(f"Completed: {len(results)} success, {failed} failed, {elapsed:.0f}s")

    results.sort(key=lambda r: r["sim_idx"])

    n_sims = len(results)
    nx_full = results[0]["spatial_info"]["c_e_shape"][0]
    phi_s_neg_shape = results[0]["spatial_info"]["phi_s_neg_shape"]
    phi_s_pos_shape = results[0]["spatial_info"]["phi_s_pos_shape"]
    c_s_pos_shape = results[0]["spatial_info"]["c_s_pos_shape"]
    nx_neg = phi_s_neg_shape[0]
    nx_pos = phi_s_pos_shape[0]
    nr_pos = c_s_pos_shape[0]

    actual_nt = []
    for r in results:
        actual_nt.append(r["V"].shape[0])
    n_time_actual = min(actual_nt)
    logger.info(f"Resampling to n_time={n_time_actual}")

    params_arr = np.zeros((n_sims, len(param_names)), dtype=np.float32)
    c_rates_arr = np.zeros(n_sims, dtype=np.float32)
    V_arr = np.zeros((n_sims, n_time_actual), dtype=np.float32)
    c_e_arr = np.zeros((n_sims, nx_full, n_time_actual), dtype=np.float32)
    phi_e_arr = np.zeros((n_sims, nx_full, n_time_actual), dtype=np.float32)
    j_pos_arr = np.zeros((n_sims, nx_pos, n_time_actual), dtype=np.float32)

    for i, r in enumerate(results):
        params_arr[i] = r["param_vector"][:len(param_names)]
        c_rates_arr[i] = r["c_rate"]
        V_arr[i] = r["V"][:n_time_actual] if r["V"].shape[0] >= n_time_actual else np.interp(
            np.linspace(0, 1, n_time_actual), np.linspace(0, 1, r["V"].shape[0]), r["V"])
        c_e_arr[i] = _resample_2d(r["c_e"][:, :n_time_actual] if r["c_e"].shape[1] >= n_time_actual else r["c_e"], n_time_actual)
        phi_e_arr[i] = _resample_2d(r["phi_e"][:, :n_time_actual] if r["phi_e"].shape[1] >= n_time_actual else r["phi_e"], n_time_actual)
        j_pos_arr[i] = _resample_2d(r["j_pos"][:, :n_time_actual] if r["j_pos"].shape[1] >= n_time_actual else r["j_pos"], n_time_actual)

    # Build param_set_id: which param set each sim belongs to
    param_set_ids = np.zeros(n_sims, dtype=np.int32)
    for i, r in enumerate(results):
        param_set_ids[i] = r["sim_idx"] // len(c_rates)

    logger.info(f"Saving to {output_path}...")
    with h5py.File(output_path, "w") as f:
        f.create_dataset("params", data=params_arr, compression="gzip")
        f.create_dataset("c_rates", data=c_rates_arr, compression="gzip")
        f.create_dataset("V", data=V_arr, compression="gzip")
        f.create_dataset("c_e", data=c_e_arr, compression="gzip")
        f.create_dataset("phi_e", data=phi_e_arr, compression="gzip")
        f.create_dataset("j_pos", data=j_pos_arr, compression="gzip")
        f.create_dataset("param_set_ids", data=param_set_ids, compression="gzip")

        f.attrs["n_sims"] = n_sims
        f.attrs["n_time"] = n_time_actual
        f.attrs["nx_full"] = nx_full
        f.attrs["nx_neg"] = nx_neg
        f.attrs["nx_pos"] = nx_pos
        f.attrs["nr_pos"] = nr_pos
        f.attrs["n_param_sets"] = n_param_sets
        f.attrs["c_rates"] = c_rates
        f.attrs["param_names"] = [p.encode() for p in param_names]

    logger.info(f"Done! {output_path}")
    logger.info(f"  {n_sims} sims, {n_param_sets} param sets × {len(c_rates)} C-rates")


if __name__ == "__main__":
    main()
