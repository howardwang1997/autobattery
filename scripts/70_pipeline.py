#!/usr/bin/env python3
"""
Compute pipeline runner — 3 days of continuous computation.
Orchestrates all tasks across 2×H20 GPUs.

Tasks:
  Phase 1 (0-8h):   Data preparation + Bayesian model training
  Phase 2 (8-24h):  Synthetic data generation + Adaptive experiment design
  Phase 3 (24-48h): Experimental data analysis + Few-shot transfer
  Phase 4 (48-72h): Final evaluation + Paper figures
"""

import os
import sys
import time
import json
import subprocess
import logging
from pathlib import Path
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("/root/autobattery/logs/pipeline.log"),
    ],
)
logger = logging.getLogger(__name__)

PROJECT = "/root/autobattery"
CONDA = "source /root/miniconda3/bin/activate autobattery && "
LOG_DIR = f"{PROJECT}/logs"


def run(cmd, desc, timeout=None, gpu=None):
    env = os.environ.copy()
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    logger.info(f"START: {desc}")
    t0 = time.time()
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            executable="/bin/bash",
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
            cwd=PROJECT,
        )
        elapsed = time.time() - t0
        if result.returncode == 0:
            logger.info(f"DONE: {desc} ({elapsed:.0f}s)")
            return True
        else:
            logger.error(f"FAIL: {desc} ({elapsed:.0f}s)")
            logger.error(f"  stderr: {result.stderr[:500]}")
            return False
    except subprocess.TimeoutExpired:
        logger.error(f"TIMEOUT: {desc}")
        return False


def phase1():
    logger.info("=" * 60)
    logger.info("PHASE 1: Data prep + Bayesian model + Dependencies")
    logger.info("=" * 60)

    run(
        f"{CONDA} pip install zuko -q",
        "Install zuko (normalizing flows)",
        timeout=120,
    )

    run(
        f"{CONDA} cd {PROJECT} && python3 scripts/60_parse_experimental.py",
        "Parse experimental data (already done)",
        timeout=300,
    )

    run(
        f"{CONDA} cd {PROJECT} && CUDA_VISIBLE_DEVICES=0 python3 scripts/61_train_bayesian.py "
        "--epochs 500 --hidden 256 --n-flows 8 --data data/fullfield/fullfield_lfp_degradation.h5 "
        "--output outputs/bayesian/cnf_degradation.pt",
        "Train Conditional Normalizing Flow on degradation data",
        timeout=14400,
        gpu=0,
    )

    run(
        f"{CONDA} cd {PROJECT} && CUDA_VISIBLE_DEVICES=0 python3 scripts/62_train_ensemble.py "
        "--n-members 50 --epochs 300 --data data/fullfield/fullfield_lfp_degradation.h5 "
        "--output outputs/bayesian/ensemble/",
        "Train 50-member ensemble baseline",
        timeout=14400,
        gpu=0,
    )


def phase2():
    logger.info("=" * 60)
    logger.info("PHASE 2: Synthetic data + Adaptive design")
    logger.info("=" * 60)

    run(
        f"{CONDA} cd {PROJECT} && python3 scripts/50_gen_synthetic_cycling.py "
        "--n-cells 2000 --n-workers 16 --seed 123 "
        "--output data/synthetic_cycling/lfp_2000.h5",
        "Generate 2000 LFP cycling cells",
        timeout=7200,
    )

    run(
        f"{CONDA} cd {PROJECT} && CUDA_VISIBLE_DEVICES=0 python3 scripts/63_adaptive_design.py "
        "--n-eval 1000 --output outputs/adaptive_design/",
        "Adaptive experiment design simulation",
        timeout=14400,
        gpu=0,
    )

    run(
        f"{CONDA} cd {PROJECT} && CUDA_VISIBLE_DEVICES=0 python3 scripts/64_noisy_bayesian.py "
        "--noise-levels 1,5,10,50 --output outputs/bayesian/noisy/",
        "Train Bayesian models on noisy data",
        timeout=14400,
        gpu=0,
    )


def phase3():
    logger.info("=" * 60)
    logger.info("PHASE 3: Experimental validation + Few-shot")
    logger.info("=" * 60)

    run(
        f"{CONDA} cd {PROJECT} && CUDA_VISIBLE_DEVICES=0 python3 scripts/65_experimental_eval.py "
        "--synth-data data/synthetic_cycling/lfp_2000.h5 "
        "--exp-data data/experimental/experimental_cycling.h5 "
        "--output outputs/experimental_eval/",
        "Evaluate early-cycle prediction on experimental data",
        timeout=7200,
        gpu=0,
    )

    run(
        f"{CONDA} cd {PROJECT} && CUDA_VISIBLE_DEVICES=0 python3 scripts/66_bayesian_experimental.py "
        "--exp-data data/experimental/experimental_cycling.h5 "
        "--ckpt outputs/bayesian/cnf_degradation.pt "
        "--output outputs/bayesian/experimental/",
        "Bayesian diagnosis on experimental data",
        timeout=7200,
        gpu=0,
    )


def phase4():
    logger.info("=" * 60)
    logger.info("PHASE 4: Final evaluation + Paper figures")
    logger.info("=" * 60)

    run(
        f"{CONDA} cd {PROJECT} && python3 scripts/67_final_figures.py "
        "--output outputs/paper_figures/",
        "Generate publication-quality figures",
        timeout=3600,
    )

    run(
        f"{CONDA} cd {PROJECT} && python3 scripts/68_results_table.py "
        "--output outputs/final_results.json",
        "Generate final results comparison table",
        timeout=1800,
    )


if __name__ == "__main__":
    logger.info("Pipeline started")
    phase1()
    phase2()
    phase3()
    phase4()
    logger.info("Pipeline completed!")
