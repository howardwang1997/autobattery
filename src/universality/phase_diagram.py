"""Phase diagram + N★ regression in formulation space.

Given each cell's archetype label and characteristic cycle N★, this module
fits two complementary models:

1. **Multinomial logit on archetype** — gives per-formulation probability
   of each death mode; the decision boundaries are the *phase boundaries*.
2. **N★ = f(formulation)** — both linear (interpretable) and symbolic
   (PySR if available, with a polynomial fallback). The closed-form law
   is the headline result for the paper.

Both models report performance on a held-out fold to make the
extrapolation claim quantitative.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class PhaseDiagramResult:
    feature_names: list[str]
    archetype_classifier: object       # fitted sklearn estimator
    archetype_accuracy: float
    archetype_auc_macro: float
    nstar_model: object                # fitted regression on log10(N★)
    nstar_r2: float
    nstar_mae: float
    coefficients: dict                 # per-feature coefficients (linear models)
    symbolic_law: Optional[dict] = None  # {'expr': str, 'r2': float}


def _safe_log10(x: np.ndarray) -> np.ndarray:
    return np.log10(np.clip(x, 1e-9, None))


def fit_phase_diagram(
    formulation: np.ndarray,        # (n, d)
    archetype_labels: np.ndarray,   # (n,) int
    N_star: np.ndarray,             # (n,) float (NaN allowed)
    feature_names: list[str],
    test_frac: float = 0.2,
    seed: int = 0,
    try_symbolic: bool = True,
) -> PhaseDiagramResult:
    """Fit the two formulation-space models on cells with valid labels."""
    from sklearn.linear_model import LogisticRegression, RidgeCV
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import train_test_split

    rng = np.random.default_rng(seed)

    # archetype classifier ---------------------------------------------
    valid_arch = archetype_labels >= 0
    Xa = formulation[valid_arch]
    ya = archetype_labels[valid_arch]
    Xa_tr, Xa_te, ya_tr, ya_te = train_test_split(
        Xa, ya, test_size=test_frac, random_state=seed, stratify=ya if len(set(ya)) > 1 else None,
    )
    clf = LogisticRegression(max_iter=2000, C=1.0)
    clf.fit(Xa_tr, ya_tr)
    acc = float(clf.score(Xa_te, ya_te))

    auc = np.nan
    if len(set(ya_te)) > 1:
        try:
            proba = clf.predict_proba(Xa_te)
            auc = float(roc_auc_score(
                ya_te, proba, multi_class="ovr", average="macro",
            ))
        except Exception:
            auc = np.nan

    # N★ regression ----------------------------------------------------
    valid_n = ~np.isnan(N_star) & (N_star > 0)
    Xn = formulation[valid_n]
    yn = _safe_log10(N_star[valid_n])
    if len(yn) > 5:
        Xn_tr, Xn_te, yn_tr, yn_te = train_test_split(
            Xn, yn, test_size=test_frac, random_state=seed,
        )
        reg = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0])
        reg.fit(Xn_tr, yn_tr)
        yn_pred = reg.predict(Xn_te)
        ss_res = float(np.sum((yn_te - yn_pred) ** 2))
        ss_tot = float(np.sum((yn_te - yn_te.mean()) ** 2))
        r2 = 1.0 - ss_res / max(ss_tot, 1e-15)
        mae = float(np.mean(np.abs(yn_te - yn_pred)))
    else:
        reg, r2, mae = None, np.nan, np.nan

    coeffs = {}
    if reg is not None and hasattr(reg, "coef_"):
        coeffs = {name: float(c) for name, c in zip(feature_names, reg.coef_)}
        coeffs["__intercept__"] = float(reg.intercept_)

    # symbolic law (best effort) ---------------------------------------
    sym = None
    if try_symbolic and len(yn) >= 50:
        sym = _try_symbolic_regression(Xn, yn, feature_names)

    return PhaseDiagramResult(
        feature_names=feature_names,
        archetype_classifier=clf,
        archetype_accuracy=acc,
        archetype_auc_macro=float(auc) if not np.isnan(auc) else np.nan,
        nstar_model=reg,
        nstar_r2=float(r2) if not np.isnan(r2) else np.nan,
        nstar_mae=float(mae) if not np.isnan(mae) else np.nan,
        coefficients=coeffs,
        symbolic_law=sym,
    )


def _try_symbolic_regression(
    X: np.ndarray, y: np.ndarray, feature_names: list[str],
) -> Optional[dict]:
    """Best-effort symbolic regression. Tries PySR; falls back to polynomial."""
    try:
        from pysr import PySRRegressor

        model = PySRRegressor(
            niterations=40,
            binary_operators=["+", "*", "-", "/"],
            unary_operators=["log", "exp", "sqrt"],
            verbosity=0,
            progress=False,
            populations=8,
        )
        model.fit(X, y, variable_names=feature_names)
        eq = model.get_best()
        return {"engine": "pysr", "expr": str(eq["equation"]), "loss": float(eq["loss"])}
    except Exception:
        pass

    # polynomial fallback (degree 2 with interactions)
    try:
        from sklearn.preprocessing import PolynomialFeatures
        from sklearn.linear_model import LassoCV

        poly = PolynomialFeatures(degree=2, interaction_only=False, include_bias=False)
        Xp = poly.fit_transform(X)
        names = poly.get_feature_names_out(feature_names)
        las = LassoCV(cv=5, max_iter=5000, n_alphas=30)
        las.fit(Xp, y)
        terms = []
        for n, c in zip(names, las.coef_):
            if abs(c) > 1e-4:
                terms.append(f"{c:+.3g}·{n}")
        if abs(las.intercept_) > 1e-9:
            terms.append(f"{las.intercept_:+.3g}")
        expr = " ".join(terms) if terms else f"{las.intercept_:.3g}"
        r2 = float(las.score(Xp, y))
        return {"engine": "lasso_poly2", "expr": f"log10(N*) = {expr}", "r2": r2}
    except Exception as exc:
        return {"engine": "none", "error": str(exc)}


# ---------------------------------------------------------------------------
# Decision-boundary grid for plotting (2-D phase diagram)
# ---------------------------------------------------------------------------


def boundary_grid(
    classifier,
    feature_ranges: dict[str, tuple[float, float]],
    feature_names: list[str],
    grid_n: int = 80,
    fixed: Optional[dict[str, float]] = None,
) -> dict:
    """Sweep two features on a grid (others held at *fixed* values) and return
    archetype probabilities for plotting.

    Use the *first two* names in feature_ranges as the swept axes.
    """
    if len(feature_ranges) < 2:
        raise ValueError("need ≥ 2 features in feature_ranges")
    fixed = fixed or {}

    swept = list(feature_ranges.keys())[:2]
    f1, f2 = swept
    g1 = np.linspace(*feature_ranges[f1], grid_n)
    g2 = np.linspace(*feature_ranges[f2], grid_n)
    G1, G2 = np.meshgrid(g1, g2)
    n_pts = grid_n * grid_n

    X = np.zeros((n_pts, len(feature_names)))
    for j, name in enumerate(feature_names):
        if name == f1:
            X[:, j] = G1.ravel()
        elif name == f2:
            X[:, j] = G2.ravel()
        else:
            X[:, j] = fixed.get(name, 0.0)

    probs = classifier.predict_proba(X)
    labels = classifier.predict(X).reshape(grid_n, grid_n)
    return {
        "axis_names": [f1, f2],
        "axis_values": [g1, g2],
        "labels": labels,
        "probs": probs.reshape(grid_n, grid_n, -1),
    }
