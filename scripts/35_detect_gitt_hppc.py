"""Detect GITT / HPPC patterns in existing cycling data.

Scans every Neware xlsx (or CSV) under --data-dir and reports whether the
recorded protocol contains:

  * **explicit GITT**            small pulse (~C/20) + long rest (>30 min)
                                 at multiple intermediate SOC points
  * **explicit HPPC**            short pulses (10-30 s) at multiple SOC,
                                 paired discharge + charge with short rest
  * **retrospective GITT**       cycle-end rests long enough to fit a
                                 single Weppner-Huggins relaxation per
                                 cycle (the freebie hidden in many
                                 standard protocols)

This is a read-only diagnostic. It does NOT extract D_s — once you know
which cells have usable data, the extraction script (TODO: 36_extract_*)
runs separately. Output goes to ``outputs/observable_audit/``:

  summary.json            per-cell scores + global rollup
  summary.md              human-readable table for the runbook
  per_cell/<cell_id>/
      structure.png       I, V, segment-type vs time (overview)
      rest_distribution.png  histogram of rest durations
      gitt_candidates.json   list of segments that look like GITT pulses
      hppc_candidates.json   list of segments that look like HPPC pulses

Usage on H20::

    conda activate autobattery
    python scripts/35_detect_gitt_hppc.py \
        --data-dir data/raw/lmb_long_cycle \
        --output outputs/observable_audit \
        --max-cells 50
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Optional

import numpy as np

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
)
logger = logging.getLogger("detect_gitt_hppc")


# Tunable thresholds ---------------------------------------------------------

CURRENT_REST_THRESHOLD_A = 5e-3        # |I| below this counts as rest
GITT_REST_MIN_S = 30 * 60              # 30 min — minimum to call a rest "GITT-like"
GITT_REST_IDEAL_S = 2 * 3600           # ≥ 2 h is the textbook value
GITT_PULSE_MAX_S = 30 * 60             # GITT pulses are usually < 30 min
HPPC_PULSE_MIN_S = 5
HPPC_PULSE_MAX_S = 60
HPPC_REST_BETWEEN_PULSES_MAX_S = 5 * 60   # short rest between paired pulses
SOC_INTERMEDIATE_BAND = (0.10, 0.90)   # rests outside this band are "boundary" rests
RELAX_MIN_POINTS = 30                  # need ≥ 30 V samples in a rest to fit relaxation


# Data structures ------------------------------------------------------------


@dataclass
class Segment:
    cycle: int
    kind: str                 # "charge" | "discharge" | "rest" | "unknown"
    t_start: float            # seconds, absolute
    t_end: float
    duration_s: float
    n_points: int
    soc_start: float          # 0..1, estimated by Coulomb counting
    soc_end: float
    i_mean_A: float
    v_start: float
    v_end: float


@dataclass
class CellAudit:
    cell_id: str
    file_path: str
    n_points: int
    n_cycles: int
    duration_h: float
    has_step_type_column: bool

    # Counts
    n_rest_segments: int = 0
    n_charge_segments: int = 0
    n_discharge_segments: int = 0

    # Quality scores (0..1)
    explicit_gitt_score: float = 0.0
    explicit_hppc_score: float = 0.0
    retrospective_gitt_score: float = 0.0

    # Diagnostic raw counts
    n_gitt_candidate_segments: int = 0
    n_hppc_candidate_pulses: int = 0
    n_long_rests_at_intermediate_soc: int = 0
    n_long_rests_at_boundary: int = 0
    median_cycle_end_rest_s: float = 0.0
    fraction_cycles_with_usable_relaxation: float = 0.0

    # Verdict
    verdict: str = ""
    notes: list[str] = field(default_factory=list)


# Loading & segmentation -----------------------------------------------------


def _load(loader, path: Path) -> dict[str, np.ndarray]:
    if path.suffix.lower() == ".xlsx":
        return loader.load_neware_xlsx(path)
    if path.suffix.lower() == ".csv":
        return loader.load_csv(path)
    raise ValueError(f"unsupported extension: {path.suffix}")


def _segment_by_step(raw: dict[str, np.ndarray]) -> list[Segment]:
    """Split the V/I trace into homogeneous segments.

    Prefers the Neware ``step_type`` column when present (the 工步类型
    field is the canonical source of truth — Neware decides charge /
    discharge / rest at protocol-design time). Falls back to
    sign-of-current heuristics otherwise.
    """
    t = raw["time"].astype(np.float64)
    v = raw["voltage"].astype(np.float64)
    i = raw["current"].astype(np.float64)
    cyc = raw.get("cycle", np.zeros_like(t)).astype(np.int64)
    step_type = raw.get("step_type", None)

    # Coulomb counting → SOC estimate. Use sign-corrected current,
    # normalise by the maximum cumulative charge (≈ nominal capacity)
    # to get a 0..1 scale even when the spec capacity is unknown.
    dt = np.diff(t, prepend=t[0])
    q_cum = np.cumsum(i * dt) / 3600.0
    q_min = q_cum.min()
    q_max = q_cum.max()
    capacity_proxy = max(q_max - q_min, 1e-6)
    soc = (q_cum - q_min) / capacity_proxy
    soc = np.clip(soc, 0.0, 1.0)

    # Build segment label per sample.
    if step_type is not None and len(step_type) == len(t):
        labels = np.array([_normalise_step_label(s) for s in step_type])
    else:
        labels = np.where(
            np.abs(i) < CURRENT_REST_THRESHOLD_A, "rest",
            np.where(i > 0, "charge", "discharge"),
        )

    # Find boundaries where (label, cycle) changes.
    boundaries = [0]
    for k in range(1, len(t)):
        if labels[k] != labels[k - 1] or cyc[k] != cyc[k - 1]:
            boundaries.append(k)
    boundaries.append(len(t))

    segments: list[Segment] = []
    for k in range(len(boundaries) - 1):
        a, b = boundaries[k], boundaries[k + 1]
        if b - a < 2:
            continue
        segments.append(Segment(
            cycle=int(cyc[a]),
            kind=labels[a],
            t_start=float(t[a]),
            t_end=float(t[b - 1]),
            duration_s=float(t[b - 1] - t[a]),
            n_points=int(b - a),
            soc_start=float(soc[a]),
            soc_end=float(soc[b - 1]),
            i_mean_A=float(np.mean(i[a:b])),
            v_start=float(v[a]),
            v_end=float(v[b - 1]),
        ))
    return segments


def _normalise_step_label(raw: Any) -> str:
    s = str(raw or "").strip().lower()
    # Neware Chinese
    if any(t in s for t in ("搁置", "rest", "static", "open")):
        return "rest"
    if any(t in s for t in ("放电", "discharge", "dchg", "cc-dchg", "ccdchg")):
        return "discharge"
    if any(t in s for t in ("充电", "charge", "chg", "cc-chg", "ccchg", "cv")):
        return "charge"
    return "unknown"


# Detection rules ------------------------------------------------------------


def _classify_rest_segments(
    segs: list[Segment], capacity_proxy_A: float,
) -> tuple[list[Segment], list[Segment]]:
    """Return (intermediate_long_rests, cycle_end_rests).

    "Long" = ≥ GITT_REST_MIN_S. "Intermediate" SOC means the rest occurs
    away from the top/bottom of the SOC window — these are GITT-like
    even if the protocol wasn't named so.
    """
    intermediate = []
    boundary = []
    for s in segs:
        if s.kind != "rest" or s.duration_s < GITT_REST_MIN_S:
            continue
        # Use the SOC at the start of the rest as the location label.
        if SOC_INTERMEDIATE_BAND[0] <= s.soc_start <= SOC_INTERMEDIATE_BAND[1]:
            intermediate.append(s)
        else:
            boundary.append(s)
    return intermediate, boundary


def _detect_hppc_pulses(segs: list[Segment]) -> list[dict[str, Any]]:
    """Identify HPPC-style short pulse pairs.

    Heuristic: a charge OR discharge pulse of duration HPPC_PULSE_MIN_S
    .. HPPC_PULSE_MAX_S, sandwiched between rests, with the next pulse
    of the *opposite* polarity within HPPC_REST_BETWEEN_PULSES_MAX_S.
    """
    candidates = []
    for k, seg in enumerate(segs):
        if seg.kind not in ("charge", "discharge"):
            continue
        if not (HPPC_PULSE_MIN_S <= seg.duration_s <= HPPC_PULSE_MAX_S):
            continue
        # Must be flanked by rests (or be the first / last meaningful segment).
        prev = segs[k - 1] if k > 0 else None
        nxt = segs[k + 1] if k + 1 < len(segs) else None
        if prev is None or nxt is None:
            continue
        if prev.kind != "rest" or nxt.kind != "rest":
            continue
        # Look for a paired pulse of the opposite polarity within the
        # short rest budget.
        m = k + 2
        if m < len(segs):
            mate = segs[m]
            if (
                mate.kind in ("charge", "discharge")
                and mate.kind != seg.kind
                and HPPC_PULSE_MIN_S <= mate.duration_s <= HPPC_PULSE_MAX_S
                and nxt.duration_s <= HPPC_REST_BETWEEN_PULSES_MAX_S
            ):
                candidates.append({
                    "cycle": seg.cycle,
                    "soc_start": seg.soc_start,
                    "pulse_a_kind": seg.kind,
                    "pulse_a_duration_s": seg.duration_s,
                    "pulse_a_current_A": seg.i_mean_A,
                    "pulse_b_kind": mate.kind,
                    "pulse_b_duration_s": mate.duration_s,
                    "pulse_b_current_A": mate.i_mean_A,
                    "rest_between_s": nxt.duration_s,
                })
    return candidates


def _detect_gitt_pulses(segs: list[Segment]) -> list[dict[str, Any]]:
    """Identify GITT-style {pulse, long-rest} pairs.

    The defining structural signature of a GITT step is:

      (charge|discharge) of duration ≤ GITT_PULSE_MAX_S, ending at an
      intermediate SOC, immediately followed by a rest of ≥
      GITT_REST_MIN_S.

    GITT is typically run at C/20 while normal cycling is C/3..1C, so we
    *also* report whether the pulse magnitude is small relative to the
    cell's max operating current — this is a quality flag, not a filter,
    because synthetic / test traces may consist purely of GITT pulses
    with no normal cycling for comparison.
    """
    if not segs:
        return []

    op_currents = np.array([
        abs(s.i_mean_A) for s in segs if s.kind in ("charge", "discharge")
    ])
    if len(op_currents) == 0:
        return []
    op_max = float(op_currents.max())

    candidates = []
    for k in range(len(segs) - 1):
        pulse = segs[k]
        rest = segs[k + 1]
        if pulse.kind not in ("charge", "discharge"):
            continue
        if rest.kind != "rest":
            continue
        if not (5 <= pulse.duration_s <= GITT_PULSE_MAX_S):
            continue
        if rest.duration_s < GITT_REST_MIN_S:
            continue
        if not (SOC_INTERMEDIATE_BAND[0] <= pulse.soc_end <= SOC_INTERMEDIATE_BAND[1]):
            continue
        is_small_pulse = abs(pulse.i_mean_A) <= 0.6 * op_max
        candidates.append({
            "cycle": pulse.cycle,
            "soc_start": pulse.soc_start,
            "soc_end": pulse.soc_end,
            "pulse_kind": pulse.kind,
            "pulse_duration_s": pulse.duration_s,
            "pulse_current_A": pulse.i_mean_A,
            "is_small_pulse_relative_to_max": is_small_pulse,
            "rest_duration_s": rest.duration_s,
            "rest_n_points": rest.n_points,
        })
    return candidates


def _retrospective_gitt_quality(segs: list[Segment]) -> tuple[float, float, float]:
    """Cycle-end rests as 'free' single-SOC GITT.

    Returns (score, median_rest_s, frac_with_usable_relaxation).
    """
    rest_durs = []
    usable_count = 0
    n_cycles = max((s.cycle for s in segs), default=0)
    if n_cycles <= 0:
        return 0.0, 0.0, 0.0

    by_cycle: dict[int, list[Segment]] = {}
    for s in segs:
        by_cycle.setdefault(s.cycle, []).append(s)

    for cyc, segs_c in by_cycle.items():
        # Take the longest rest in this cycle.
        rests = [s for s in segs_c if s.kind == "rest"]
        if not rests:
            continue
        longest = max(rests, key=lambda s: s.duration_s)
        rest_durs.append(longest.duration_s)
        if (
            longest.duration_s >= 10 * 60   # at least 10 min
            and longest.n_points >= RELAX_MIN_POINTS
        ):
            usable_count += 1

    if not rest_durs:
        return 0.0, 0.0, 0.0
    med = float(np.median(rest_durs))
    frac_usable = usable_count / max(n_cycles, 1)
    score = float(min(1.0, (med / GITT_REST_IDEAL_S) * 0.5 + frac_usable * 0.5))
    return score, med, frac_usable


# Per-cell pipeline ----------------------------------------------------------


def audit_cell(path: Path, loader) -> tuple[CellAudit, list[Segment], list[dict], list[dict]]:
    raw = _load(loader, path)
    segments = _segment_by_step(raw)

    has_step_type = "step_type" in raw and len(raw["step_type"]) == len(raw["time"])

    # Counts
    n_rest = sum(1 for s in segments if s.kind == "rest")
    n_chg = sum(1 for s in segments if s.kind == "charge")
    n_dchg = sum(1 for s in segments if s.kind == "discharge")

    # Long rests at intermediate vs boundary SOC.
    inter, boundary = _classify_rest_segments(segments, capacity_proxy_A=1.0)

    # Explicit GITT and HPPC.
    gitt_cands = _detect_gitt_pulses(segments)
    hppc_cands = _detect_hppc_pulses(segments)

    # Retrospective GITT (cycle-end rests).
    retro_score, med_end_rest, frac_usable = _retrospective_gitt_quality(segments)

    # Heuristic scores --------------------------------------------------
    # GITT explicit: at least 5 candidates spanning ≥ 3 SOC bins.
    if len(gitt_cands) >= 5:
        soc_bins = {round(c["soc_start"] * 10) for c in gitt_cands}
        explicit_gitt = min(1.0, len(gitt_cands) / 30.0) * (1.0 if len(soc_bins) >= 3 else 0.5)
    else:
        explicit_gitt = 0.0

    # HPPC explicit: at least 3 paired pulses, ideally ≥ 5 SOC bins.
    if len(hppc_cands) >= 3:
        soc_bins = {round(c["soc_start"] * 10) for c in hppc_cands}
        explicit_hppc = min(1.0, len(hppc_cands) / 20.0) * (1.0 if len(soc_bins) >= 5 else 0.5)
    else:
        explicit_hppc = 0.0

    n_cycles = int(max((s.cycle for s in segments), default=0))

    audit = CellAudit(
        cell_id=path.stem,
        file_path=str(path),
        n_points=int(len(raw["time"])),
        n_cycles=n_cycles,
        duration_h=float((raw["time"][-1] - raw["time"][0]) / 3600.0),
        has_step_type_column=has_step_type,
        n_rest_segments=n_rest,
        n_charge_segments=n_chg,
        n_discharge_segments=n_dchg,
        explicit_gitt_score=explicit_gitt,
        explicit_hppc_score=explicit_hppc,
        retrospective_gitt_score=retro_score,
        n_gitt_candidate_segments=len(gitt_cands),
        n_hppc_candidate_pulses=len(hppc_cands),
        n_long_rests_at_intermediate_soc=len(inter),
        n_long_rests_at_boundary=len(boundary),
        median_cycle_end_rest_s=med_end_rest,
        fraction_cycles_with_usable_relaxation=frac_usable,
    )
    audit.verdict = _summarise_verdict(audit)
    return audit, segments, gitt_cands, hppc_cands


def _summarise_verdict(a: CellAudit) -> str:
    if a.explicit_gitt_score > 0.4:
        return "explicit GITT present — extract directly"
    if a.explicit_hppc_score > 0.4:
        return "explicit HPPC present — extract DCIR(SOC, cycle) directly"
    if a.retrospective_gitt_score > 0.5:
        return "no formal GITT, but cycle-end rests are usable as single-SOC GITT"
    if a.retrospective_gitt_score > 0.2:
        return "marginal — short rests; relaxation fits will be noisy"
    return "no usable additional observable hidden in this cell"


# Plotting -------------------------------------------------------------------


def plot_cell(
    audit: CellAudit, raw: dict[str, np.ndarray], segments: list[Segment],
    out_dir: Path,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)

    t = raw["time"]
    v = raw["voltage"]
    i = raw["current"]
    t_h = (t - t[0]) / 3600.0

    fig, axes = plt.subplots(3, 1, figsize=(13, 8), sharex=True)
    axes[0].plot(t_h, v, lw=0.6, color="black")
    axes[0].set_ylabel("V (V)")
    axes[0].set_title(f"{audit.cell_id} — n_cycles={audit.n_cycles}, dur={audit.duration_h:.0f}h")

    axes[1].plot(t_h, i, lw=0.6, color="tab:blue")
    axes[1].axhline(0, color="grey", lw=0.5)
    axes[1].set_ylabel("I (A)")

    color_map = {"charge": "tab:green", "discharge": "tab:red",
                 "rest": "tab:orange", "unknown": "lightgrey"}
    for s in segments:
        axes[2].axvspan(
            (s.t_start - t[0]) / 3600.0,
            (s.t_end - t[0]) / 3600.0,
            color=color_map.get(s.kind, "lightgrey"),
            alpha=0.5 if s.kind != "rest" else 0.8,
            lw=0,
        )
    axes[2].set_yticks([])
    axes[2].set_xlabel("time (h)")
    axes[2].set_ylabel("step type")
    fig.tight_layout()
    fig.savefig(out_dir / "structure.png", dpi=120)
    plt.close(fig)

    # Rest-duration histogram.
    rests = [s.duration_s / 60.0 for s in segments if s.kind == "rest"]
    if rests:
        fig2, ax = plt.subplots(figsize=(7, 4))
        ax.hist(rests, bins=40, color="tab:orange", edgecolor="black")
        ax.axvline(GITT_REST_MIN_S / 60, color="red", lw=1, ls="--",
                   label=f"GITT threshold ({GITT_REST_MIN_S//60} min)")
        ax.set_xlabel("rest duration (min)")
        ax.set_ylabel("count")
        ax.set_title(f"{audit.cell_id} — rest segment durations")
        ax.legend()
        fig2.tight_layout()
        fig2.savefig(out_dir / "rest_distribution.png", dpi=120)
        plt.close(fig2)


# Top-level driver -----------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True, help="dir with *.xlsx or *.csv")
    parser.add_argument("--output", default="outputs/observable_audit",
                        help="output directory")
    parser.add_argument("--max-cells", type=int, default=999,
                        help="stop after N cells (debug)")
    parser.add_argument("--no-plot", action="store_true", help="skip plotting")
    args = parser.parse_args()

    from src.data.loader import ExperimentalDataLoader
    loader = ExperimentalDataLoader()

    out = Path(args.output)
    (out / "per_cell").mkdir(parents=True, exist_ok=True)

    files = sorted(p for p in Path(args.data_dir).iterdir()
                   if p.suffix.lower() in (".xlsx", ".csv"))
    if args.max_cells:
        files = files[: args.max_cells]
    if not files:
        raise SystemExit(f"no xlsx/csv files under {args.data_dir}")

    audits: list[CellAudit] = []
    for path in files:
        cell_dir = out / "per_cell" / path.stem
        try:
            logger.info("auditing %s", path.name)
            audit, segments, gitt_cands, hppc_cands = audit_cell(path, loader)
        except Exception as exc:                # noqa: BLE001
            logger.warning("FAILED %s: %s", path.name, exc)
            continue
        audits.append(audit)

        cell_dir.mkdir(parents=True, exist_ok=True)
        with (cell_dir / "gitt_candidates.json").open("w") as f:
            json.dump(gitt_cands, f, indent=2)
        with (cell_dir / "hppc_candidates.json").open("w") as f:
            json.dump(hppc_cands, f, indent=2)
        with (cell_dir / "audit.json").open("w") as f:
            json.dump(asdict(audit), f, indent=2, default=str)
        if not args.no_plot:
            try:
                raw = _load(loader, path)
                plot_cell(audit, raw, segments, cell_dir)
            except Exception as exc:           # noqa: BLE001
                logger.warning("plot failed for %s: %s", path.name, exc)

    _write_summary(audits, out)
    logger.info("done. %d cells audited, summary at %s/summary.md",
                len(audits), out)


def _write_summary(audits: list[CellAudit], out: Path) -> None:
    rollup = {
        "n_cells": len(audits),
        "n_with_explicit_gitt": sum(1 for a in audits if a.explicit_gitt_score > 0.4),
        "n_with_explicit_hppc": sum(1 for a in audits if a.explicit_hppc_score > 0.4),
        "n_with_usable_retrospective_gitt": sum(
            1 for a in audits if a.retrospective_gitt_score > 0.5
        ),
        "median_cycles_per_cell": float(np.median([a.n_cycles for a in audits]) if audits else 0),
    }

    with (out / "summary.json").open("w") as f:
        json.dump({
            "rollup": rollup,
            "cells": [asdict(a) for a in audits],
        }, f, indent=2, default=str)

    lines = [
        "# Observable audit — GITT / HPPC / retrospective relaxation",
        "",
        f"- Cells audited: **{rollup['n_cells']}**",
        f"- Explicit GITT detected in: **{rollup['n_with_explicit_gitt']}** cells",
        f"- Explicit HPPC detected in: **{rollup['n_with_explicit_hppc']}** cells",
        f"- Usable retrospective GITT (cycle-end rests): "
        f"**{rollup['n_with_usable_retrospective_gitt']}** cells",
        f"- Median cycles / cell: {rollup['median_cycles_per_cell']:.0f}",
        "",
        "## Per-cell verdicts",
        "",
        "| cell | cycles | GITT | HPPC | retro | median end-rest (min) | verdict |",
        "|------|--------|------|------|-------|------------------------|---------|",
    ]
    for a in sorted(audits, key=lambda x: -max(
        x.explicit_gitt_score, x.explicit_hppc_score, x.retrospective_gitt_score,
    )):
        lines.append(
            f"| {a.cell_id} | {a.n_cycles} | "
            f"{a.explicit_gitt_score:.2f} ({a.n_gitt_candidate_segments}) | "
            f"{a.explicit_hppc_score:.2f} ({a.n_hppc_candidate_pulses}) | "
            f"{a.retrospective_gitt_score:.2f} | "
            f"{a.median_cycle_end_rest_s/60:.0f} | "
            f"{a.verdict} |"
        )

    lines += [
        "",
        "## How to read",
        "",
        "- **GITT** column: score (0..1) followed by the count of small-pulse "
        "+ long-rest segments at intermediate SOC. Score > 0.4 → explicit GITT.",
        "- **HPPC** column: score (0..1) and the count of paired-pulse short-rest "
        "patterns. Score > 0.4 → explicit HPPC.",
        "- **retro** column: 0..1 score for using cycle-end rests as a "
        "single-SOC GITT. Score > 0.5 → free observable, run extraction.",
        "- **median end-rest**: typical rest length between full charge and "
        "full discharge. < 10 min → too short for relaxation fits.",
        "",
        "Per-cell artefacts under `per_cell/<cell_id>/` (structure plot, "
        "rest histogram, JSON candidate lists).",
    ]

    with (out / "summary.md").open("w") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    main()
