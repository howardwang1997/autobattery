"""Landmark-cell metadata schema for cryo-EM / XPS / SEM tie-in.

Phase B B1/B2 of the publication roadmap. The morphology-anchoring story
needs a clean record of *which* cells were dismantled at *which* cycle
and what was measured. We keep this as a JSON-backed dataclass instead
of a database so the file lives next to the cycling data and is easy to
diff / review.

Expected on-disk layout::

    data/landmark/
      ├── manifest.json          # list of LandmarkMeasurement
      ├── cell_A_cycle_100/
      │     ├── cryoem.npz       # dead-Li volume fraction, SEI thickness
      │     ├── xps.csv          # SEI species depth profile
      │     └── sem.npz          # crack fraction
      └── ...

The model's per-cycle outputs (dead-Li, SEI thickness, LAM_pos) are
compared against these landmark measurements; see
``src/diagnosis/morphology_constraint.py`` (added in Phase B).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Iterable, Optional


@dataclass
class LandmarkMeasurement:
    """One post-mortem measurement on a specific cell at a specific cycle."""

    cell_id: str
    cycle: int
    technique: str            # "cryoem" | "xps" | "sem" | "icp" | "tof_sims"
    measurement_path: str     # file path under data/landmark/
    operator: str = ""
    date: str = ""            # ISO8601
    notes: str = ""
    quantities: dict[str, float] = field(default_factory=dict)
    uncertainty: dict[str, float] = field(default_factory=dict)

    SUPPORTED_TECHNIQUES = ("cryoem", "xps", "sem", "icp", "tof_sims")
    EXPECTED_QUANTITIES = {
        "cryoem": ("dead_li_volume_fraction", "sei_thickness_nm", "porosity"),
        "xps":    ("sei_lif_atomic_pct", "sei_li2co3_atomic_pct", "sei_thickness_nm"),
        "sem":    ("crack_area_fraction", "particle_radius_um", "porosity"),
        "icp":    ("li_lost_mol_per_m2",),
        "tof_sims": ("sei_thickness_nm",),
    }

    def __post_init__(self) -> None:
        if self.technique not in self.SUPPORTED_TECHNIQUES:
            raise ValueError(
                f"Unsupported technique '{self.technique}'. "
                f"Choose from {self.SUPPORTED_TECHNIQUES}"
            )
        for q in self.quantities:
            if q not in self.EXPECTED_QUANTITIES.get(self.technique, ()):
                # Not fatal — we allow extra quantities, just warn.
                pass

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_manifest(path: str | Path) -> list[LandmarkMeasurement]:
    path = Path(path)
    if not path.exists():
        return []
    with path.open() as f:
        data = json.load(f)
    return [LandmarkMeasurement(**row) for row in data]


def save_manifest(
    measurements: Iterable[LandmarkMeasurement],
    path: str | Path,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [m.to_dict() for m in measurements]
    with path.open("w") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)
    return path


def measurements_for_cell(
    measurements: Iterable[LandmarkMeasurement],
    cell_id: str,
) -> list[LandmarkMeasurement]:
    return sorted(
        (m for m in measurements if m.cell_id == cell_id),
        key=lambda m: (m.cycle, m.technique),
    )


def quantity_at_cycle(
    measurements: Iterable[LandmarkMeasurement],
    cell_id: str,
    quantity: str,
    cycle: int,
    tolerance: int = 5,
) -> Optional[tuple[float, float]]:
    """Return (value, sigma) for a quantity at the requested cycle, if any
    measurement falls within ``tolerance`` cycles. Returns None otherwise."""
    best = None
    best_dist = tolerance + 1
    for m in measurements:
        if m.cell_id != cell_id or quantity not in m.quantities:
            continue
        d = abs(m.cycle - cycle)
        if d <= tolerance and d < best_dist:
            best = (m.quantities[quantity], m.uncertainty.get(quantity, 0.0))
            best_dist = d
    return best
