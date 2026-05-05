"""Tests for scripts/35_detect_gitt_hppc.py.

We can't test the script's main() end-to-end without real xlsx fixtures,
but we can hammer the segmentation + detection rules using synthetic
in-memory traces. The synthetic protocols below mimic the three
patterns the script is supposed to recognise.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Load the script as a module (it lives outside src/, so importable=False).
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def detect():
    import sys
    repo_root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "detect_gitt_hppc",
        repo_root / "scripts" / "35_detect_gitt_hppc.py",
    )
    mod = importlib.util.module_from_spec(spec)
    # Must be in sys.modules before exec_module so dataclasses can
    # resolve forward references (Python 3.13 quirk).
    sys.modules["detect_gitt_hppc"] = mod
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# Helpers: build synthetic V/I traces with known protocol structures.
# ---------------------------------------------------------------------------


def _append_segment(t, v, i, cyc, step, *, dur_s, dt_s, current_A, kind, cycle):
    """Append a constant-current segment to growing arrays."""
    n = max(2, int(round(dur_s / dt_s)))
    t_seg = (t[-1] if t else 0.0) + np.linspace(dt_s, dur_s, n)
    t.extend(t_seg.tolist())
    # Voltage doesn't matter for segmentation — fill with a slow drift.
    v_start = v[-1] if v else 3.6
    v.extend(np.linspace(v_start, v_start + 0.001 * current_A, n).tolist())
    i.extend([float(current_A)] * n)
    cyc.extend([cycle] * n)
    step.extend([kind] * n)


def _make_raw():
    return {"t": [], "v": [], "i": [], "cyc": [], "step": []}


def _finalize(buf):
    return {
        "time": np.array(buf["t"], dtype=np.float64),
        "voltage": np.array(buf["v"], dtype=np.float64),
        "current": np.array(buf["i"], dtype=np.float64),
        "cycle": np.array(buf["cyc"], dtype=np.int64),
        "step_type": np.array(buf["step"]),
    }


# ---------------------------------------------------------------------------
# Tests for the segmentation primitive.
# ---------------------------------------------------------------------------


def test_segment_by_step_uses_step_type_when_available(detect):
    buf = _make_raw()
    _append_segment(**buf, dur_s=600, dt_s=10, current_A=+1.0, kind="charge", cycle=1)
    _append_segment(**buf, dur_s=300, dt_s=10, current_A=0.0, kind="rest", cycle=1)
    _append_segment(**buf, dur_s=900, dt_s=10, current_A=-1.0, kind="discharge", cycle=1)
    raw = _finalize(buf)

    segs = detect._segment_by_step(raw)
    kinds = [s.kind for s in segs]
    assert kinds == ["charge", "rest", "discharge"]
    assert segs[0].duration_s == pytest.approx(600, abs=15)
    assert segs[1].duration_s == pytest.approx(300, abs=15)


def test_segment_falls_back_to_current_sign(detect):
    # No step_type column.
    n = 60
    t = np.arange(n) * 10.0
    v = np.full(n, 3.6)
    i = np.concatenate([np.full(20, 1.0), np.zeros(20), np.full(20, -1.0)])
    cyc = np.zeros(n, dtype=np.int64)
    raw = {"time": t, "voltage": v, "current": i, "cycle": cyc}

    segs = detect._segment_by_step(raw)
    kinds = [s.kind for s in segs]
    assert kinds == ["charge", "rest", "discharge"]


# ---------------------------------------------------------------------------
# Tests for retrospective GITT scoring.
# ---------------------------------------------------------------------------


def test_retrospective_gitt_score_high_for_long_cycle_end_rests(detect):
    buf = _make_raw()
    for c in range(1, 11):
        _append_segment(**buf, dur_s=600,  dt_s=10, current_A=+1.0, kind="charge", cycle=c)
        _append_segment(**buf, dur_s=600,  dt_s=10, current_A=-1.0, kind="discharge", cycle=c)
        _append_segment(**buf, dur_s=3600, dt_s=10, current_A=0.0,  kind="rest", cycle=c)
    raw = _finalize(buf)

    segs = detect._segment_by_step(raw)
    score, med, frac = detect._retrospective_gitt_quality(segs)

    assert score > 0.5
    assert med == pytest.approx(3600, abs=60)
    assert frac == pytest.approx(1.0, abs=0.01)


def test_retrospective_gitt_score_low_for_short_cycle_end_rests(detect):
    buf = _make_raw()
    for c in range(1, 11):
        _append_segment(**buf, dur_s=600, dt_s=10, current_A=+1.0, kind="charge", cycle=c)
        _append_segment(**buf, dur_s=600, dt_s=10, current_A=-1.0, kind="discharge", cycle=c)
        _append_segment(**buf, dur_s=120, dt_s=10, current_A=0.0,  kind="rest", cycle=c)
    raw = _finalize(buf)

    segs = detect._segment_by_step(raw)
    score, _, _ = detect._retrospective_gitt_quality(segs)
    assert score < 0.4


# ---------------------------------------------------------------------------
# Explicit GITT pulse detection.
# ---------------------------------------------------------------------------


def test_detects_textbook_gitt_protocol(detect):
    """Big charge → 10× (small pulse + 2h rest) → discharge.

    Mimics a C/3 charge to ~80% SOC followed by C/20 GITT titration with
    long rests at intermediate SOCs.
    """
    buf = _make_raw()
    # Bring cell up to ~0% then pulse-rest-pulse... up.
    _append_segment(**buf, dur_s=300,  dt_s=10, current_A=-1.0, kind="discharge", cycle=1)
    for k in range(10):
        _append_segment(**buf, dur_s=600,  dt_s=10, current_A=+0.05, kind="charge", cycle=1)
        _append_segment(**buf, dur_s=2*3600, dt_s=20, current_A=0.0, kind="rest", cycle=1)
    raw = _finalize(buf)

    segs = detect._segment_by_step(raw)
    cands = detect._detect_gitt_pulses(segs)
    assert len(cands) >= 5
    # All candidates should be the small C/20 pulses (the C/3 prep
    # discharge that brought the cell up should not appear because it
    # ends at boundary SOC, not intermediate).
    assert all(abs(c["pulse_current_A"]) < 0.2 for c in cands)
    # Rests are long.
    assert all(c["rest_duration_s"] >= 30 * 60 for c in cands)


def test_does_not_flag_normal_cycling_as_gitt(detect):
    buf = _make_raw()
    for c in range(1, 6):
        _append_segment(**buf, dur_s=3600, dt_s=10, current_A=+1.0, kind="charge", cycle=c)
        _append_segment(**buf, dur_s=300,  dt_s=10, current_A=0.0,  kind="rest", cycle=c)
        _append_segment(**buf, dur_s=3600, dt_s=10, current_A=-1.0, kind="discharge", cycle=c)
        _append_segment(**buf, dur_s=300,  dt_s=10, current_A=0.0,  kind="rest", cycle=c)
    raw = _finalize(buf)

    segs = detect._segment_by_step(raw)
    cands = detect._detect_gitt_pulses(segs)
    assert len(cands) == 0


# ---------------------------------------------------------------------------
# HPPC detection.
# ---------------------------------------------------------------------------


def test_detects_hppc_paired_pulses(detect):
    """USABC-style HPPC: rest → 10s discharge → 40s rest → 10s charge → rest."""
    buf = _make_raw()
    # Bring SOC up.
    _append_segment(**buf, dur_s=600, dt_s=5, current_A=+1.0, kind="charge", cycle=1)

    for soc_step in range(8):
        _append_segment(**buf, dur_s=120, dt_s=5, current_A=0.0,  kind="rest", cycle=1)
        _append_segment(**buf, dur_s=10,  dt_s=1, current_A=-2.0, kind="discharge", cycle=1)
        _append_segment(**buf, dur_s=40,  dt_s=2, current_A=0.0,  kind="rest", cycle=1)
        _append_segment(**buf, dur_s=10,  dt_s=1, current_A=+2.0, kind="charge", cycle=1)
        _append_segment(**buf, dur_s=120, dt_s=5, current_A=0.0,  kind="rest", cycle=1)
        # SOC drift down to next step.
        _append_segment(**buf, dur_s=600, dt_s=5, current_A=-1.0, kind="discharge", cycle=1)
    raw = _finalize(buf)

    segs = detect._segment_by_step(raw)
    cands = detect._detect_hppc_pulses(segs)
    assert len(cands) >= 4
    for c in cands:
        assert c["pulse_a_kind"] != c["pulse_b_kind"]
        assert 5 <= c["pulse_a_duration_s"] <= 60
        assert 5 <= c["pulse_b_duration_s"] <= 60


def test_does_not_flag_normal_cycling_as_hppc(detect):
    buf = _make_raw()
    for c in range(1, 6):
        _append_segment(**buf, dur_s=3600, dt_s=10, current_A=+1.0, kind="charge", cycle=c)
        _append_segment(**buf, dur_s=300,  dt_s=10, current_A=0.0,  kind="rest", cycle=c)
        _append_segment(**buf, dur_s=3600, dt_s=10, current_A=-1.0, kind="discharge", cycle=c)
        _append_segment(**buf, dur_s=300,  dt_s=10, current_A=0.0,  kind="rest", cycle=c)
    raw = _finalize(buf)

    segs = detect._segment_by_step(raw)
    cands = detect._detect_hppc_pulses(segs)
    assert len(cands) == 0


# ---------------------------------------------------------------------------
# Verdict thresholds.
# ---------------------------------------------------------------------------


def test_verdict_explicit_gitt(detect):
    audit = detect.CellAudit(
        cell_id="x", file_path="x", n_points=0, n_cycles=0, duration_h=0.0,
        has_step_type_column=True,
        explicit_gitt_score=0.6, retrospective_gitt_score=0.0,
    )
    assert "explicit GITT" in detect._summarise_verdict(audit)


def test_verdict_retrospective_only(detect):
    audit = detect.CellAudit(
        cell_id="x", file_path="x", n_points=0, n_cycles=0, duration_h=0.0,
        has_step_type_column=True,
        explicit_gitt_score=0.1, explicit_hppc_score=0.1,
        retrospective_gitt_score=0.7,
    )
    assert "single-SOC GITT" in detect._summarise_verdict(audit)


def test_verdict_no_observable(detect):
    audit = detect.CellAudit(
        cell_id="x", file_path="x", n_points=0, n_cycles=0, duration_h=0.0,
        has_step_type_column=True,
        explicit_gitt_score=0.0, explicit_hppc_score=0.0,
        retrospective_gitt_score=0.05,
    )
    assert "no usable" in detect._summarise_verdict(audit)
