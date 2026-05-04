"""Differential-voltage degradation signature library.

Phase A3 of the publication roadmap. The original prototype in
``scripts/28_degradation_modes.py`` showed that fitting per-cycle
ΔV(t) = V_cycle(t) − V_ref(t) as a linear combination of simulator-derived
``∂V/∂param`` signatures cuts the sim-to-real gap from ~178 mV (direct
voltage matching) to ~12.7 mV. This module promotes that prototype to
something publishable:

* deterministic fit on a fixed signature library, plus three regressors
  (NNLS, ridge, elastic net) for ablation;
* bootstrap confidence intervals over both signatures and per-cycle
  coefficients;
* leave-one-signature-out ablation API (used to back the claim in the
  paper that every mode is necessary);
* clean separation between *building* a library from simulation data and
  *applying* it to an experimental cell, so the same library can be
  reused across many cells without recomputation.

The expected workflow from the H20 runbook is::

    library = build_signature_library(V_sim, P_sim, ...)
    library.save("outputs/signature_library_lmb.npz")

    diag = DegradationDiagnosis(library)
    results = diag.diagnose(experimental_voltage_curves)
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

import numpy as np

logger = logging.getLogger(__name__)


DEFAULT_SIGNATURE_NAMES_LMB: tuple[str, ...] = (
    "D_e",                # electrolyte transport
    "D_s_pos",            # cathode solid diffusion
    "t_plus",             # transference number
    "k_plating",          # plating kinetics
    "k_dead_li",          # dead-Li decay
    "k_sei",              # SEI growth rate
    "L_sei_0",            # initial SEI thickness
    "j0_pos",             # cathode exchange current
)


# ---------------------------------------------------------------------------
# Signature library
# ---------------------------------------------------------------------------


@dataclass
class SignatureLibrary:
    """A library of ``∂V/∂param`` signatures + the normalisation context
    they were computed under.

    Attributes:
        signatures: shape ``(n_params, n_time)``. Row j is ``∂V/∂p_j``.
        param_names: names matching the rows of ``signatures``.
        log_scale_mask: boolean array marking parameters that were
            log-transformed before regression.
        param_ref: shape ``(n_params,)``. Reference parameter values
            (median of the simulation sweep, in the same scale as the
            sweep — i.e. log10 if log_scale, linear otherwise).
        param_scale: shape ``(n_params,)``. Scale used to normalise.
        time_grid: shape ``(n_time,)``. Normalised discharge time
            (0..1).
        ridge_alpha: regularisation strength used to build the library.
        bootstrap_signatures: optional shape
            ``(n_bootstrap, n_params, n_time)`` for UQ.
        meta: free-form metadata (cycle, c_rate, simulator settings).
    """

    signatures: np.ndarray
    param_names: tuple[str, ...]
    log_scale_mask: np.ndarray
    param_ref: np.ndarray
    param_scale: np.ndarray
    time_grid: np.ndarray
    ridge_alpha: float = 0.1
    bootstrap_signatures: Optional[np.ndarray] = None
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def n_params(self) -> int:
        return len(self.param_names)

    @property
    def n_time(self) -> int:
        return self.signatures.shape[1]

    def signature_uncertainty(self) -> Optional[np.ndarray]:
        if self.bootstrap_signatures is None:
            return None
        return self.bootstrap_signatures.std(axis=0)

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            signatures=self.signatures,
            param_names=np.array(self.param_names),
            log_scale_mask=self.log_scale_mask,
            param_ref=self.param_ref,
            param_scale=self.param_scale,
            time_grid=self.time_grid,
            ridge_alpha=np.float64(self.ridge_alpha),
            bootstrap_signatures=(
                self.bootstrap_signatures
                if self.bootstrap_signatures is not None
                else np.empty(0)
            ),
            meta=np.array(json.dumps(self.meta)),
        )
        return path

    @classmethod
    def load(cls, path: str | Path) -> "SignatureLibrary":
        path = Path(path)
        data = np.load(path, allow_pickle=False)
        boot = data["bootstrap_signatures"]
        return cls(
            signatures=data["signatures"],
            param_names=tuple(str(s) for s in data["param_names"]),
            log_scale_mask=data["log_scale_mask"].astype(bool),
            param_ref=data["param_ref"],
            param_scale=data["param_scale"],
            time_grid=data["time_grid"],
            ridge_alpha=float(data["ridge_alpha"]),
            bootstrap_signatures=boot if boot.size > 0 else None,
            meta=json.loads(str(data["meta"])),
        )


# ---------------------------------------------------------------------------
# Library construction
# ---------------------------------------------------------------------------


def _normalise_params(
    params: np.ndarray,
    log_scale_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply log10 to selected columns and standardise to zero mean / unit
    std. Returns (normalised, ref, scale)."""

    p = params.copy().astype(np.float64)
    if log_scale_mask.any():
        p[:, log_scale_mask] = np.log10(np.clip(p[:, log_scale_mask], 1e-30, None))
    ref = np.median(p, axis=0)
    scale = p.std(axis=0)
    scale[scale < 1e-12] = 1.0
    return (p - ref) / scale, ref, scale


def _ridge_fit(P_norm: np.ndarray, V: np.ndarray, alpha: float) -> np.ndarray:
    """Solve the per-time-point ridge regression in closed form.

    Equivalent to ``sklearn.linear_model.Ridge(alpha=alpha,
    fit_intercept=True)`` fit independently at every time index,
    vectorised. Returns shape ``(n_params, n_time)``. Both V and P_norm
    are demeaned across simulations before solving so that the constant
    voltage offset (e.g. mean discharge plateau) does not leak into the
    signatures.
    """
    P_centered = P_norm - P_norm.mean(axis=0, keepdims=True)
    V_centered = V - V.mean(axis=0, keepdims=True)
    PtP = P_centered.T @ P_centered
    n = PtP.shape[0]
    A = PtP + alpha * np.eye(n)
    return np.linalg.solve(A, P_centered.T @ V_centered)


def build_signature_library(
    V_sim: np.ndarray,
    P_sim: np.ndarray,
    param_names: Sequence[str],
    log_scale_params: Iterable[str] = (),
    time_grid: Optional[np.ndarray] = None,
    ridge_alpha: float = 0.1,
    bootstrap_samples: int = 0,
    rng: Optional[np.random.Generator] = None,
    meta: Optional[dict[str, Any]] = None,
) -> SignatureLibrary:
    """Compute degradation signatures from simulation data.

    Args:
        V_sim: shape ``(n_sim, n_time)``. Voltage curves on a common time
            grid. Caller is responsible for selecting curves at a single
            C-rate (otherwise the regression conflates protocol with
            parameter sensitivity).
        P_sim: shape ``(n_sim, n_params)``. Parameter values used to
            generate ``V_sim``.
        param_names: parameter names for each column of ``P_sim``.
        log_scale_params: subset of ``param_names`` to log-transform
            before regression. Required for parameters that span many
            orders of magnitude (D_e, k_sei, ...).
        ridge_alpha: ridge regularisation strength. Larger => smoother
            but less responsive signatures.
        bootstrap_samples: if > 0, repeat the regression on bootstrap
            re-samples of the simulations to provide signature CIs.
    """
    if V_sim.ndim != 2 or P_sim.ndim != 2:
        raise ValueError("V_sim and P_sim must both be 2D arrays")
    if V_sim.shape[0] != P_sim.shape[0]:
        raise ValueError("V_sim and P_sim must share their first dimension")
    if len(param_names) != P_sim.shape[1]:
        raise ValueError("param_names length must match P_sim columns")

    n_sim, n_time = V_sim.shape
    log_set = set(log_scale_params)
    log_mask = np.array([n in log_set for n in param_names], dtype=bool)

    if time_grid is None:
        time_grid = np.linspace(0.0, 1.0, n_time)

    P_norm, p_ref, p_scale = _normalise_params(P_sim, log_mask)
    signatures = _ridge_fit(P_norm, V_sim, ridge_alpha)

    boot = None
    if bootstrap_samples > 0:
        rng = rng or np.random.default_rng(0)
        boot = np.zeros((bootstrap_samples, len(param_names), n_time), dtype=np.float64)
        for b in range(bootstrap_samples):
            idx = rng.integers(0, n_sim, size=n_sim)
            P_b = P_norm[idx]
            V_b = V_sim[idx]
            boot[b] = _ridge_fit(P_b, V_b, ridge_alpha)

    return SignatureLibrary(
        signatures=signatures,
        param_names=tuple(param_names),
        log_scale_mask=log_mask,
        param_ref=p_ref,
        param_scale=p_scale,
        time_grid=time_grid,
        ridge_alpha=float(ridge_alpha),
        bootstrap_signatures=boot,
        meta=dict(meta or {}),
    )


# ---------------------------------------------------------------------------
# Per-cell diagnosis
# ---------------------------------------------------------------------------


@dataclass
class CycleDiagnosis:
    cycle: int
    coeffs: np.ndarray                 # shape (n_params,)
    coeffs_ci_low: Optional[np.ndarray] = None
    coeffs_ci_high: Optional[np.ndarray] = None
    dV_rmse_mV: float = 0.0
    capacity_Ah: Optional[float] = None
    n_points: int = 0

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        for k in ("coeffs", "coeffs_ci_low", "coeffs_ci_high"):
            if d[k] is not None:
                d[k] = np.asarray(d[k]).tolist()
        return d


def _solve_with_regressor(
    A: np.ndarray,
    b: np.ndarray,
    regressor: str,
    alpha: float,
    nonneg_mask: Optional[np.ndarray],
) -> np.ndarray:
    """Solve A @ x ≈ b under one of the supported regressors."""
    if regressor == "ridge":
        AtA = A.T @ A + alpha * np.eye(A.shape[1])
        return np.linalg.solve(AtA, A.T @ b)
    if regressor == "nnls":
        from scipy.optimize import nnls
        if nonneg_mask is None or nonneg_mask.all():
            x, _ = nnls(A, b)
            return x
        # Mixed-sign: split each column into +/- copies, NNLS, recombine.
        cols = []
        signs = []
        for j in range(A.shape[1]):
            if nonneg_mask[j]:
                cols.append(A[:, j])
                signs.append((j, +1))
            else:
                cols.append(A[:, j]); signs.append((j, +1))
                cols.append(-A[:, j]); signs.append((j, -1))
        A_aug = np.stack(cols, axis=1)
        x_aug, _ = nnls(A_aug, b)
        x = np.zeros(A.shape[1])
        for (j, s), val in zip(signs, x_aug):
            x[j] += s * val
        return x
    if regressor == "elastic_net":
        from sklearn.linear_model import ElasticNet
        model = ElasticNet(alpha=alpha, l1_ratio=0.5, fit_intercept=False, max_iter=10000)
        model.fit(A, b)
        return model.coef_
    raise ValueError(f"Unknown regressor '{regressor}'")


class DegradationDiagnosis:
    """Apply a signature library to experimental discharge curves.

    Parameters:
        library: precomputed :class:`SignatureLibrary`.
        regressor: ``"ridge"`` (default), ``"nnls"`` or ``"elastic_net"``.
            Use NNLS when you have strong prior that all modes monotonically
            grow (e.g. SEI, dead Li).
        alpha: regularisation parameter for ridge / elastic net.
        nonneg_signatures: subset of param names whose coefficients are
            constrained to be non-negative (only meaningful when
            ``regressor != "ridge"``).
    """

    def __init__(
        self,
        library: SignatureLibrary,
        regressor: str = "ridge",
        alpha: float = 0.1,
        nonneg_signatures: Optional[Sequence[str]] = None,
    ):
        self.library = library
        self.regressor = regressor
        self.alpha = alpha
        if nonneg_signatures is None:
            self._nonneg_mask = None
        else:
            nonneg = set(nonneg_signatures)
            self._nonneg_mask = np.array(
                [n in nonneg for n in library.param_names], dtype=bool
            )

    # ---- Reference (early-cycle) curve ----------------------------------

    @staticmethod
    def reference_curve(
        curves: Sequence[dict],
        n_time: int,
        n_ref_cycles: int = 5,
    ) -> np.ndarray:
        """Average the first ``n_ref_cycles`` curves to a common grid."""
        if len(curves) == 0:
            raise ValueError("curves is empty")
        n_ref = min(n_ref_cycles, len(curves))
        target_t = np.linspace(0.0, 1.0, n_time)
        v_refs = np.zeros((n_ref, n_time))
        for i, cc in enumerate(curves[:n_ref]):
            v = np.asarray(cc["voltage"], dtype=np.float64)
            t_norm = np.linspace(0.0, 1.0, len(v))
            v_refs[i] = np.interp(target_t, t_norm, v)
        return v_refs.mean(axis=0)

    # ---- Per-cycle fit --------------------------------------------------

    def diagnose_cycle(
        self,
        voltage: np.ndarray,
        v_ref: np.ndarray,
        cycle: int = 0,
        capacity_Ah: Optional[float] = None,
        smooth_sigma: float = 3.0,
        bootstrap_samples: int = 0,
        rng: Optional[np.random.Generator] = None,
    ) -> CycleDiagnosis:
        v_arr = np.asarray(voltage, dtype=np.float64)
        n_time = len(v_ref)
        target_t = np.linspace(0.0, 1.0, n_time)
        v_resampled = np.interp(target_t, np.linspace(0.0, 1.0, len(v_arr)), v_arr)
        dV = v_resampled - v_ref

        if smooth_sigma > 0:
            from scipy.ndimage import gaussian_filter1d
            dV = gaussian_filter1d(dV, sigma=smooth_sigma)

        A = self.library.signatures.T            # (n_time, n_params)
        coeffs = _solve_with_regressor(A, dV, self.regressor, self.alpha, self._nonneg_mask)

        recon = A @ coeffs
        rmse = float(np.sqrt(np.mean((recon - dV) ** 2)) * 1000.0)

        ci_lo = ci_hi = None
        if bootstrap_samples > 0:
            rng = rng or np.random.default_rng(0)
            boots = np.zeros((bootstrap_samples, len(coeffs)))
            for b in range(bootstrap_samples):
                idx = rng.integers(0, n_time, size=n_time)
                Ab = A[idx]
                bb = dV[idx]
                boots[b] = _solve_with_regressor(
                    Ab, bb, self.regressor, self.alpha, self._nonneg_mask,
                )
            ci_lo = np.quantile(boots, 0.05, axis=0)
            ci_hi = np.quantile(boots, 0.95, axis=0)

        return CycleDiagnosis(
            cycle=int(cycle),
            coeffs=coeffs,
            coeffs_ci_low=ci_lo,
            coeffs_ci_high=ci_hi,
            dV_rmse_mV=rmse,
            capacity_Ah=capacity_Ah,
            n_points=len(v_arr),
        )

    def diagnose(
        self,
        curves: Sequence[dict],
        n_ref_cycles: int = 5,
        smooth_sigma: float = 3.0,
        bootstrap_samples: int = 0,
        rng: Optional[np.random.Generator] = None,
    ) -> list[CycleDiagnosis]:
        n_time = self.library.n_time
        v_ref = self.reference_curve(curves, n_time=n_time, n_ref_cycles=n_ref_cycles)
        results = []
        for cc in curves:
            results.append(
                self.diagnose_cycle(
                    voltage=cc["voltage"],
                    v_ref=v_ref,
                    cycle=int(cc.get("cycle", 0)),
                    capacity_Ah=cc.get("capacity"),
                    smooth_sigma=smooth_sigma,
                    bootstrap_samples=bootstrap_samples,
                    rng=rng,
                )
            )
        return results

    # ---- Ablation -------------------------------------------------------

    def leave_one_out_ablation(
        self,
        curves: Sequence[dict],
        n_ref_cycles: int = 5,
        smooth_sigma: float = 3.0,
    ) -> dict[str, dict[str, float]]:
        """Drop each signature in turn and report mean RMSE increase.

        Returns a dict ``{param_name: {"rmse_full": x, "rmse_dropped": y,
        "delta_mV": y - x}}``. Used to back the paper claim that every
        mode is necessary.
        """
        n_time = self.library.n_time
        v_ref = self.reference_curve(curves, n_time=n_time, n_ref_cycles=n_ref_cycles)

        full = self.diagnose(curves, n_ref_cycles=n_ref_cycles, smooth_sigma=smooth_sigma)
        rmse_full = float(np.mean([d.dV_rmse_mV for d in full]))

        out: dict[str, dict[str, float]] = {}
        for j, name in enumerate(self.library.param_names):
            sigs = np.delete(self.library.signatures, j, axis=0)
            mask = (
                np.delete(self._nonneg_mask, j) if self._nonneg_mask is not None else None
            )
            tmp_lib = SignatureLibrary(
                signatures=sigs,
                param_names=tuple(n for k, n in enumerate(self.library.param_names) if k != j),
                log_scale_mask=np.delete(self.library.log_scale_mask, j),
                param_ref=np.delete(self.library.param_ref, j),
                param_scale=np.delete(self.library.param_scale, j),
                time_grid=self.library.time_grid,
                ridge_alpha=self.library.ridge_alpha,
            )
            tmp_diag = DegradationDiagnosis(
                tmp_lib,
                regressor=self.regressor,
                alpha=self.alpha,
                nonneg_signatures=tuple(
                    n for k, n in enumerate(self.library.param_names)
                    if mask is not None and mask[k - (1 if k > j else 0)]
                ) if mask is not None else None,
            )
            dropped = tmp_diag.diagnose(curves, n_ref_cycles=n_ref_cycles, smooth_sigma=smooth_sigma)
            rmse_drop = float(np.mean([d.dV_rmse_mV for d in dropped]))
            out[name] = {
                "rmse_full_mV": rmse_full,
                "rmse_dropped_mV": rmse_drop,
                "delta_mV": rmse_drop - rmse_full,
            }
        return out


# ---------------------------------------------------------------------------
# Convenience
# ---------------------------------------------------------------------------


def fit_cycle_modes(
    library: SignatureLibrary,
    curves: Sequence[dict],
    regressor: str = "ridge",
    alpha: float = 0.1,
    n_ref_cycles: int = 5,
    smooth_sigma: float = 3.0,
    bootstrap_samples: int = 0,
    nonneg_signatures: Optional[Sequence[str]] = None,
    rng: Optional[np.random.Generator] = None,
) -> list[CycleDiagnosis]:
    """One-call helper used by H20 launcher scripts."""
    diag = DegradationDiagnosis(
        library, regressor=regressor, alpha=alpha, nonneg_signatures=nonneg_signatures,
    )
    return diag.diagnose(
        curves,
        n_ref_cycles=n_ref_cycles,
        smooth_sigma=smooth_sigma,
        bootstrap_samples=bootstrap_samples,
        rng=rng,
    )
