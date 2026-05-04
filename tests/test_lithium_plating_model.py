"""Smoke + behaviour tests for the new LMB plating model wiring.

These tests are PyBaMM-heavy and slow (~10-30s each); they run on the
H20 box as part of the setup pipeline. CI workers without PyBaMM will
be skipped automatically.
"""

from __future__ import annotations

import numpy as np
import pytest

pybamm = pytest.importorskip("pybamm")


def test_metal_battery_dfn_summary_reports_plating_keys():
    from src.simulation.models import MetalBatteryDFN

    battery = MetalBatteryDFN(chemistry="lmb", mode="plating_dominant")
    summary = battery.summary()

    assert summary["chemistry"] == "lmb"
    assert summary["mode"] == "plating_dominant"
    assert summary["parameter_set"] in ("OKane2022", "Chen2020")
    # OKane2022 should have plating + dead-Li parameters; Chen2020 fallback
    # only has SEI. Allow either as long as something plating-related shows
    # up when the upstream set has it.
    if summary["parameter_set"] == "OKane2022":
        assert summary["n_plating_keys"] >= 5


def test_intercalation_mode_omits_plating_options():
    from src.simulation.models import MetalBatteryDFN

    battery = MetalBatteryDFN(chemistry="lmb", mode="intercalation")
    assert "lithium plating" not in battery._options


def test_lmb_solve_smoke():
    from src.simulation.models import quick_lmb_smoke_test

    res = quick_lmb_smoke_test(c_rate=0.5, t_end=1200, n_points=51)
    assert res is not None
    assert res["voltage"].shape == (51,)
    # Plating-dominant LMB on OKane2022 discharges in the ~3.0–4.2 V window.
    assert res["voltage"].min() > 1.5
    assert res["voltage"].max() < 5.0


def test_unknown_chemistry_raises():
    from src.simulation.models import MetalBatteryDFN
    with pytest.raises(ValueError):
        MetalBatteryDFN(chemistry="zinc")


def test_lithium_plating_keys_filter():
    from src.simulation.models import MetalBatteryDFN
    battery = MetalBatteryDFN(chemistry="lmb")
    keys = battery.lithium_plating_keys()
    for k in keys:
        kl = k.lower()
        assert any(t in kl for t in ("plating", "sei", "dead lithium", "lithium metal"))
