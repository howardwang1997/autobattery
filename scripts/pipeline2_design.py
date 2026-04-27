"""Pipeline 2: Battery Design Optimization using existing LMB FNO."""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["MKL_THREADING_LAYER"] = "GNU"

import sys
import time
import logging
import subprocess
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [P2] %(message)s")
logger = logging.getLogger(__name__)

PYTHON = sys.executable
BASE = "/root/autobattery"


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
    logger.info(f"  OUTPUT: {result.stdout[-1000:]}")
    return result.returncode == 0


def main():
    logger.info("=" * 60)
    logger.info("Pipeline 2: Battery Design Optimization")
    logger.info("=" * 60)

    logger.info("Running design optimization with existing LMB FNO...")
    run(f"{PYTHON} -u scripts/21_design_optimize.py", "Design optimization", timeout=1800)

    logger.info("Pipeline 2 COMPLETE!")


if __name__ == "__main__":
    main()
