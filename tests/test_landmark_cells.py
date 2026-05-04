"""Tests for the landmark-cell metadata schema."""

from __future__ import annotations

import json

import pytest

from src.data.landmark_cells import (
    LandmarkMeasurement,
    load_manifest,
    save_manifest,
    measurements_for_cell,
    quantity_at_cycle,
)


def _example_measurements():
    return [
        LandmarkMeasurement(
            cell_id="cell_A", cycle=100, technique="cryoem",
            measurement_path="data/landmark/cell_A_cycle_100/cryoem.npz",
            quantities={"dead_li_volume_fraction": 0.05, "sei_thickness_nm": 12.0},
            uncertainty={"dead_li_volume_fraction": 0.01, "sei_thickness_nm": 2.0},
        ),
        LandmarkMeasurement(
            cell_id="cell_A", cycle=500, technique="cryoem",
            measurement_path="data/landmark/cell_A_cycle_500/cryoem.npz",
            quantities={"dead_li_volume_fraction": 0.18, "sei_thickness_nm": 25.0},
        ),
        LandmarkMeasurement(
            cell_id="cell_A", cycle=500, technique="xps",
            measurement_path="data/landmark/cell_A_cycle_500/xps.csv",
            quantities={"sei_lif_atomic_pct": 12.5, "sei_li2co3_atomic_pct": 6.0},
        ),
        LandmarkMeasurement(
            cell_id="cell_B", cycle=1000, technique="sem",
            measurement_path="data/landmark/cell_B_cycle_1000/sem.npz",
            quantities={"crack_area_fraction": 0.07},
        ),
    ]


def test_save_and_load_manifest_roundtrip(tmp_path):
    measurements = _example_measurements()
    path = save_manifest(measurements, tmp_path / "manifest.json")
    loaded = load_manifest(path)
    assert len(loaded) == len(measurements)
    assert {m.cell_id for m in loaded} == {"cell_A", "cell_B"}


def test_measurements_for_cell_sorted_by_cycle_then_technique():
    measurements = _example_measurements()
    a = measurements_for_cell(measurements, "cell_A")
    assert [m.cycle for m in a] == [100, 500, 500]
    assert a[1].technique == "cryoem"
    assert a[2].technique == "xps"


def test_quantity_at_cycle_within_tolerance():
    measurements = _example_measurements()
    val = quantity_at_cycle(measurements, "cell_A", "dead_li_volume_fraction", cycle=502)
    assert val is not None
    value, sigma = val
    assert value == pytest.approx(0.18)


def test_quantity_at_cycle_out_of_tolerance_returns_none():
    measurements = _example_measurements()
    assert quantity_at_cycle(
        measurements, "cell_A", "dead_li_volume_fraction", cycle=900,
    ) is None


def test_unsupported_technique_raises():
    with pytest.raises(ValueError):
        LandmarkMeasurement(
            cell_id="x", cycle=1, technique="ramen",
            measurement_path="x",
        )


def test_load_manifest_missing_file_returns_empty(tmp_path):
    assert load_manifest(tmp_path / "nope.json") == []
