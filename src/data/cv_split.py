"""Cross-validation splits with no leakage.

Phase A4 of the publication roadmap. The original training pipeline used
a uniform 80/20 split *after* per-simulation normalisation, which both
(a) leaked target statistics into training and (b) couldn't answer
"does the model generalise to a new cell?". This module provides three
strict splits used by the H20 training scripts:

* :func:`leave_one_cell_out` — primary protocol for the paper, drops one
  cell at a time from training and evaluates on it.
* :func:`cycle_block_split` — chronological split per cell, used to test
  early-cycle prediction (train on cycles 1..k, predict 80%-of-life).
* :func:`stratified_chemistry_split` — for the eventual multi-chemistry
  generalisation experiment.

All splitters return arrays of indices into the supplied container; the
caller is responsible for slicing the underlying arrays. Splits are
deterministic given a seed and never share cells across folds.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CVFold:
    name: str
    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray

    def __post_init__(self) -> None:
        # Defensive: ensure no overlap.
        for a, b in (
            (self.train_idx, self.val_idx),
            (self.train_idx, self.test_idx),
            (self.val_idx, self.test_idx),
        ):
            if len(np.intersect1d(a, b)) > 0:
                raise ValueError(f"Overlap detected in fold '{self.name}'")


def _to_array(x: Iterable, dtype=None) -> np.ndarray:
    arr = np.asarray(list(x))
    return arr.astype(dtype) if dtype is not None else arr


def leave_one_cell_out(
    cell_ids: Sequence,
    val_fraction: float = 0.0,
    seed: int = 0,
) -> list[CVFold]:
    """Generate one fold per unique cell.

    Args:
        cell_ids: per-sample cell identifier (any hashable). Length N.
        val_fraction: fraction of training cells (NOT samples) to hold out
            as validation in each fold.
        seed: rng seed for the validation split.

    Returns:
        list of :class:`CVFold`. Length = number of unique cells.
    """
    cell_arr = _to_array(cell_ids)
    unique = np.array(sorted(set(cell_arr.tolist()), key=str))
    rng = np.random.default_rng(seed)
    folds: list[CVFold] = []

    for held in unique:
        test_idx = np.where(cell_arr == held)[0]
        train_pool = np.array([c for c in unique if c != held])

        if val_fraction > 0 and len(train_pool) > 1:
            n_val = max(1, int(round(val_fraction * len(train_pool))))
            perm = rng.permutation(train_pool)
            val_cells = perm[:n_val]
            train_cells = perm[n_val:]
        else:
            val_cells = np.array([], dtype=train_pool.dtype)
            train_cells = train_pool

        train_idx = np.where(np.isin(cell_arr, train_cells))[0]
        val_idx = np.where(np.isin(cell_arr, val_cells))[0]

        folds.append(
            CVFold(
                name=f"loo_{held}",
                train_idx=train_idx,
                val_idx=val_idx,
                test_idx=test_idx,
            )
        )
    return folds


def cycle_block_split(
    cell_ids: Sequence,
    cycles: Sequence[int],
    train_cycles: int,
    val_cycles: int = 0,
    min_test_cycles: int = 1,
) -> list[CVFold]:
    """Per-cell chronological split.

    For every cell, the first ``train_cycles`` cycles go to train, the
    next ``val_cycles`` go to validation, and the rest become test. Used
    for the early-cycle-life prediction benchmark in Phase C.
    """
    cell_arr = _to_array(cell_ids)
    cyc_arr = _to_array(cycles, dtype=np.int64)
    if len(cell_arr) != len(cyc_arr):
        raise ValueError("cell_ids and cycles must have the same length")
    unique = sorted(set(cell_arr.tolist()), key=str)
    folds: list[CVFold] = []

    for cell in unique:
        cell_mask = cell_arr == cell
        cell_idx = np.where(cell_mask)[0]
        # Sort by cycle number to enforce chronology.
        order = np.argsort(cyc_arr[cell_idx], kind="stable")
        ordered = cell_idx[order]

        if len(ordered) < train_cycles + val_cycles + min_test_cycles:
            logger.warning(
                "Cell %s only has %d cycles, skipping (need %d)",
                cell, len(ordered), train_cycles + val_cycles + min_test_cycles,
            )
            continue

        train_idx = ordered[:train_cycles]
        val_idx = ordered[train_cycles:train_cycles + val_cycles]
        test_idx = ordered[train_cycles + val_cycles:]

        folds.append(
            CVFold(
                name=f"chron_{cell}_{train_cycles}",
                train_idx=train_idx,
                val_idx=val_idx,
                test_idx=test_idx,
            )
        )
    return folds


def stratified_chemistry_split(
    cell_ids: Sequence,
    chemistries: Sequence[str],
    test_chemistry: str,
    val_fraction: float = 0.1,
    seed: int = 0,
) -> CVFold:
    """Hold one chemistry out entirely for the cross-chemistry experiment."""
    cell_arr = _to_array(cell_ids)
    chem_arr = _to_array(chemistries)
    if len(cell_arr) != len(chem_arr):
        raise ValueError("cell_ids and chemistries must have the same length")

    test_idx = np.where(chem_arr == test_chemistry)[0]
    train_pool_idx = np.where(chem_arr != test_chemistry)[0]
    train_cells = np.unique(cell_arr[train_pool_idx])

    rng = np.random.default_rng(seed)
    n_val = max(1, int(round(val_fraction * len(train_cells))))
    perm = rng.permutation(train_cells)
    val_cells = perm[:n_val]
    train_cells_keep = perm[n_val:]

    train_idx = np.where(np.isin(cell_arr, train_cells_keep))[0]
    val_idx = np.where(np.isin(cell_arr, val_cells))[0]

    return CVFold(
        name=f"chem_{test_chemistry}",
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
    )
