"""Curve I/O, alignment, and normalisation for battery Q(N) trajectories.

All downstream modules expect curves as :class:`AlignedCurves` — a dict-like
container that stores:
    - cell_ids:  (n_cells,) str
    - cycles:    (n_cells, n_cycles) int   — aligned cycle grid
    - capacities:(n_cells, n_cycles) float — Q(N), NaN-padded where cells ended early
    - Q0:        (n_cells,) float          — initial capacity per cell
    - meta:      dict[str, np.ndarray]     — formulation features (optional)
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class AlignedCurves:
    cell_ids: np.ndarray          # (n,) str
    cycles: np.ndarray            # (n, T) int
    capacities: np.ndarray        # (n, T) float
    Q0: np.ndarray                # (n,) float
    meta: dict[str, np.ndarray] = field(default_factory=dict)  # feature_name -> (n,)

    @property
    def n_cells(self) -> int:
        return len(self.cell_ids)

    @property
    def n_cycles(self) -> int:
        return self.cycles.shape[1]

    def capacity_retention(self) -> np.ndarray:
        """Q(N) / Q0, with NaN where capacity is NaN."""
        return self.capacities / self.Q0[:, None]

    def fade_pct(self) -> np.ndarray:
        """(Q0 - Q_final) / Q0 * 100, using last non-NaN cycle per cell."""
        Q_final = np.full(self.n_cells, np.nan)
        for i in range(self.n_cells):
            valid = ~np.isnan(self.capacities[i])
            if valid.any():
                Q_final[i] = self.capacities[i, valid][-1]
        return (self.Q0 - Q_final) / self.Q0 * 100.0

    def cycles_to_first_below(self, threshold: float = 0.8) -> np.ndarray:
        """For each cell: first cycle where Q(N)/Q0 < threshold.  NaN if never reached."""
        ret = self.capacity_retention()
        out = np.full(self.n_cells, np.nan)
        for i in range(self.n_cells):
            below = np.where(ret[i] < threshold)[0]
            if len(below):
                out[i] = self.cycles[i, below[0]]
        return out

    def feature_matrix(self, names: Optional[list[str]] = None) -> np.ndarray:
        """Stack named formulation features into (n, d) matrix."""
        if names is None:
            names = sorted(self.meta.keys())
        return np.column_stack([self.meta[k] for k in names]) if names else np.zeros((self.n_cells, 0))


# ---------------------------------------------------------------------------
# Alignment
# ---------------------------------------------------------------------------


def align_curves(
    cell_ids: list[str],
    cycle_lists: list[np.ndarray],
    capacity_lists: list[np.ndarray],
    max_cycles: int = 1000,
    grid: Optional[np.ndarray] = None,
) -> AlignedCurves:
    """Align per-cell Q(N) trajectories to a common integer cycle grid.

    Parameters
    ----------
    cell_ids       : per-cell identifiers
    cycle_lists    : per-cell cycle numbers (int or float, will be cast to int)
    capacity_lists : per-cell discharge capacities (Ah)
    max_cycles     : upper bound on the grid if *grid* is None
    grid           : explicit cycle grid to interpolate onto (int array).
                     Default: [1, 2, …, max_cycles].
    """
    if grid is None:
        all_max = max(int(c[-1]) for c in cycle_lists if len(c))
        upper = min(all_max, max_cycles)
        grid = np.arange(1, upper + 1, dtype=int)

    n = len(cell_ids)
    T = len(grid)
    caps = np.full((n, T), np.nan)
    Q0 = np.zeros(n)

    for i, (cyc, cap) in enumerate(zip(cycle_lists, capacity_lists)):
        cyc_i = np.asarray(cyc, dtype=int)
        cap_f = np.asarray(cap, dtype=float)
        valid = ~np.isnan(cap_f) & (cyc_i >= 1)
        if valid.sum() < 2:
            continue
        caps[i] = np.interp(grid, cyc_i[valid], cap_f[valid], left=np.nan, right=np.nan)
        Q0[i] = cap_f[valid][0]

    return AlignedCurves(
        cell_ids=np.asarray(cell_ids, dtype=str),
        cycles=np.broadcast_to(grid[None, :], (n, T)).copy(),
        capacities=caps,
        Q0=Q0,
    )


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------


def load_csvs(cycling_csv: Path, meta_csv: Path, max_cycles: int = 1000) -> AlignedCurves:
    """Load aligned curves from two CSV files.

    *cycling_csv*:   columns = cell_id, cycle_number, discharge_capacity
    *meta_csv*:      columns = cell_id, feature_1, feature_2, …
    """
    import csv

    def _read(path):
        with open(path) as f:
            return list(csv.DictReader(f))

    cyc_rows = _read(cycling_csv)
    met_rows = _read(meta_csv)

    meta_cells = {r["cell_id"]: r for r in met_rows}

    # group cycling rows by cell_id
    from collections import defaultdict
    by_cell: dict[str, tuple[list, list]] = defaultdict(lambda: ([], []))
    for r in cyc_rows:
        cid = r["cell_id"]
        by_cell[cid][0].append(int(float(r["cycle_number"])))
        by_cell[cid][1].append(float(r["discharge_capacity"]))

    cell_ids = []
    cycle_lists = []
    cap_lists = []
    for cid in sorted(by_cell.keys()):
        cycles, caps = by_cell[cid]
        order = np.argsort(cycles)
        cell_ids.append(cid)
        cycle_lists.append(np.array(cycles)[order])
        cap_lists.append(np.array(caps)[order])

    curves = align_curves(cell_ids, cycle_lists, cap_lists, max_cycles=max_cycles)

    # attach formulation features
    feature_cols = [c for c in met_rows[0].keys() if c != "cell_id"] if met_rows else []
    for col in feature_cols:
        vals = []
        for cid in curves.cell_ids:
            r = meta_cells.get(cid, {})
            try:
                vals.append(float(r.get(col, np.nan)))
            except (TypeError, ValueError):
                vals.append(np.nan)
        curves.meta[col] = np.array(vals)

    return curves


def load_severson_h5(path: Path) -> AlignedCurves:
    """Load Severson (2019) parsed HDF5 as AlignedCurves.

    Schema expected (per ``scripts/94_parse_severson.py``):
        /<cell_id>/capacity  (T,) float
        /<cell_id>.attrs:    cycle_life, n_cycles, cap_initial_Ah, fade_pct, batch
    """
    import h5py

    cell_ids = []
    cycle_lists = []
    cap_lists = []
    batches = []
    cycle_lives = []

    with h5py.File(path, "r") as f:
        for cid in sorted(f.keys()):
            g = f[cid]
            cap = np.array(g["capacity"])
            n = len(cap)
            cell_ids.append(cid)
            cycle_lists.append(np.arange(1, n + 1, dtype=int))
            cap_lists.append(cap)
            batches.append(int(g.attrs.get("batch", 0)))
            cycle_lives.append(int(g.attrs.get("cycle_life", n)))

    curves = align_curves(cell_ids, cycle_lists, cap_lists)
    curves.meta["batch"] = np.array(batches)
    curves.meta["cycle_life"] = np.array(cycle_lives, dtype=float)
    return curves
