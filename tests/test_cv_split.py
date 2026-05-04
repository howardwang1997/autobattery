"""Tests for src.data.cv_split — cross-validation splits with no leakage."""

from __future__ import annotations

import numpy as np
import pytest

from src.data.cv_split import (
    leave_one_cell_out,
    cycle_block_split,
    stratified_chemistry_split,
)


def test_loo_produces_one_fold_per_cell():
    cell_ids = ["A", "A", "B", "B", "C", "C"]
    folds = leave_one_cell_out(cell_ids)
    assert len(folds) == 3
    for fold in folds:
        # Every fold's test set is exactly one cell.
        held = [cell_ids[i] for i in fold.test_idx]
        assert len(set(held)) == 1
        # Train indices never include the held cell.
        train_cells = {cell_ids[i] for i in fold.train_idx}
        assert held[0] not in train_cells


def test_loo_no_overlap_between_train_val_test():
    cell_ids = list("ABCDE") * 3
    folds = leave_one_cell_out(cell_ids, val_fraction=0.5, seed=7)
    for fold in folds:
        all_idx = np.concatenate([fold.train_idx, fold.val_idx, fold.test_idx])
        assert len(all_idx) == len(set(all_idx.tolist()))


def test_cycle_block_split_chronology():
    cell_ids = ["A"] * 10 + ["B"] * 10
    cycles = list(range(10)) + list(range(10))
    folds = cycle_block_split(cell_ids, cycles, train_cycles=5, val_cycles=2)
    assert len(folds) == 2
    for fold in folds:
        train_max = max(cycles[i] for i in fold.train_idx)
        val_min = min(cycles[i] for i in fold.val_idx) if len(fold.val_idx) else 99
        test_min = min(cycles[i] for i in fold.test_idx)
        assert train_max < val_min
        assert val_min <= test_min


def test_cycle_block_split_skips_short_cells():
    cell_ids = ["A"] * 3 + ["B"] * 50
    cycles = list(range(3)) + list(range(50))
    folds = cycle_block_split(cell_ids, cycles, train_cycles=10, val_cycles=5)
    assert len(folds) == 1
    assert all(cell_ids[i] == "B" for i in folds[0].train_idx)


def test_stratified_chemistry_split_holds_chemistry_out():
    cell_ids = list("ABCDE") + list("FGHIJ")
    chems = ["lmb"] * 5 + ["lib"] * 5
    fold = stratified_chemistry_split(cell_ids, chems, test_chemistry="lib")
    train_chems = {chems[i] for i in fold.train_idx}
    test_chems = {chems[i] for i in fold.test_idx}
    assert train_chems == {"lmb"}
    assert test_chems == {"lib"}


def test_overlap_validation_raises():
    from src.data.cv_split import CVFold
    with pytest.raises(ValueError):
        CVFold(
            name="bad",
            train_idx=np.array([1, 2, 3]),
            val_idx=np.array([3, 4]),  # overlaps with train
            test_idx=np.array([5, 6]),
        )
