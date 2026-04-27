"""Pipeline 1: LIB full pipeline - simplified wrapper.

Chains data gen → train → eval → inverse → fisher.
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["MKL_THREADING_LAYER"] = "GNU"

import sys
import time
import logging
import subprocess

logging.basicConfig(level=logging.INFO, format="%(asctime)s [P1] %(message)s")
logger = logging.getLogger(__name__)

PYTHON = sys.executable
BASE = "/root/autobattery"
LIB_CKPT = "outputs/checkpoints_lib/fno_final.pt"


def run(cmd, desc, timeout=3600):
    logger.info(f"  Starting: {desc}")
    t0 = time.time()
    result = subprocess.run(
        f"cd {BASE} && MKL_THREADING_LAYER=GNU PYTHONPATH={BASE} {cmd}",
        shell=True, capture_output=True, text=True, timeout=timeout,
    )
    elapsed = time.time() - t0
    logger.info(f"  Done in {elapsed:.0f}s (rc={result.returncode})")
    if result.returncode != 0:
        logger.error(f"  STDERR: {result.stderr[-2000:]}")
        return False
    logger.info(f"  OUTPUT (last 500): {result.stdout[-500:]}")
    return True


def main():
    logger.info("=" * 60)
    logger.info("Pipeline 1: LIB (Graphite Anode) Full Pipeline")
    logger.info("=" * 60)

    import h5py
    from pathlib import Path
    lib_data = Path(f"{BASE}/data/fullfield/fullfield_lib.h5")

    # Step 1: Generate LIB data if not already present
    if not lib_data.exists():
        logger.info("Step 1/5: Generating LIB data...")
        run(f"{PYTHON} -u scripts/19_gen_lib_data.py", "LIB data generation", timeout=1800)
    else:
        logger.info("Step 1/5: LIB data already exists, skipping generation")

    # Step 2: Train FNO
    logger.info("Step 2/5: Training LIB FNO (500 epochs)...")
    run(
        f"{PYTHON} -u scripts/12_train_fno.py "
        f"--data data/fullfield/fullfield_lib.h5 --epochs 500 --gpu 0 "
        f"--checkpoint-dir outputs/checkpoints_lib",
        "LIB FNO training",
        timeout=3600,
    )

    # Step 3: Evaluate
    logger.info("Step 3/5: Evaluating LIB FNO...")
    run(
        f"{PYTHON} -u scripts/13_evaluate_fno.py "
        f"--data data/fullfield/fullfield_lib.h5 --checkpoint {LIB_CKPT} --gpu 0",
        "LIB FNO evaluation",
        timeout=300,
    )

    # Step 4: Inverse
    logger.info("Step 4/5: Parameter identification...")
    run(
        f"{PYTHON} -u scripts/14_fno_inverse.py "
        f"--data data/fullfield/fullfield_lib.h5 --checkpoint {LIB_CKPT} --gpu 0 --n-tests 10 --steps 1500",
        "LIB inverse",
        timeout=1200,
    )

    # Step 5: Fisher
    logger.info("Step 5/5: Fisher analysis...")
    run(
        f"{PYTHON} -u scripts/18_fisher_analysis.py "
        f"--data data/fullfield/fullfield_lib.h5 --checkpoint {LIB_CKPT} --gpu 0",
        "Fisher analysis",
        timeout=600,
    )

    logger.info("Pipeline 1 COMPLETE!")


if __name__ == "__main__":
    main()
