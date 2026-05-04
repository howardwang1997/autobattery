import numpy as np
import json
from pathlib import Path

d = np.load("outputs/bayesian/trajectory_analysis.npz", allow_pickle=True)
results = d["results"]
param_names = list(d["param_names"])
ident_idx = list(d["ident_idx"])

# Filter to cells with good capacity data
good = []
for r in results:
    cap = r["capacity"]
    if np.any(np.isnan(cap)):
        continue
    if cap[0] > 0 and cap[-1] / cap[0] > 0.5 and cap[-1] / cap[0] < 1.5:
        fade = (1 - cap[-1] / cap[0]) * 100
        if fade > 0:
            good.append(r)

print("=== Analysis on {} cells with reliable capacity ===".format(len(good)))
print("Capacity fade range: {:.1f}% - {:.1f}%".format(
    min((1 - r["capacity"][-1] / r["capacity"][0]) * 100 for r in good),
    max((1 - r["capacity"][-1] / r["capacity"][0]) * 100 for r in good),
))

# Per-parameter analysis
print("\n{:10s} {:>8s} {:>8s} {:>8s} {:>8s} {:>8s} {:>12s}".format(
    "Param", "Smooth", "Mono%", "|r|cap", "Drift", "Sharp", "Status"
))
print("-" * 68)

for pi, name in enumerate(param_names):
    sharp_vals = [r["std"][:, pi].mean() for r in good]
    corrs, monos, smooths = [], [], []
    for r in good:
        cap = r["capacity"]
        c0 = cap[0]
        cn = cap / c0
        vals = r["params_real"][:, pi]
        if np.std(vals) > 1e-10 and np.std(cn[:len(vals)]) > 1e-10:
            corrs.append(np.corrcoef(vals, cn[:len(vals)])[0, 1])
        diffs = np.diff(vals)
        if len(diffs) > 0 and np.std(diffs) > 0:
            monos.append(max((diffs > 0).sum(), (diffs < 0).sum()) / len(diffs))
        if np.std(vals) > 1e-10:
            smooths.append(1.0 - np.mean(np.abs(diffs)) / np.std(vals))

    avg_s = np.mean(smooths) if smooths else 0
    avg_m = np.mean(monos) * 100 if monos else 0
    avg_c = np.mean([abs(x) for x in corrs]) if corrs else 0
    st = "ID" if pi in ident_idx else "UN"
    print("{:10s} {:8.3f} {:7.1f}% {:8.3f} {:8.3f} {:8.3f} {:>12s}".format(
        name, avg_s, avg_m, avg_c, 0, np.mean(sharp_vals), st
    ))

# The critical test: split into healthy vs degraded
fades = [(1 - r["capacity"][-1] / r["capacity"][0]) * 100 for r in good]
median_fade = np.median(fades)
healthy = [r for r, f in zip(good, fades) if f < median_fade]
degraded = [r for r, f in zip(good, fades) if f >= median_fade]

print("\n=== Healthy ({:.1f}% fade) vs Degraded ({:.1f}% fade) ===".format(
    np.mean([f for f in fades if f < median_fade]),
    np.mean([f for f in fades if f >= median_fade]),
))

for label, group in [("Healthy (n={})".format(len(healthy)), healthy),
                      ("Degraded (n={})".format(len(degraded)), degraded)]:
    print("\n  {}:".format(label))
    for pi, name in enumerate(param_names):
        corrs = []
        for r in group:
            cap = r["capacity"]
            cn = cap / cap[0]
            vals = r["params_real"][:, pi]
            if np.std(vals) > 1e-10 and np.std(cn[:len(vals)]) > 1e-10:
                corrs.append(np.corrcoef(vals, cn[:len(vals)])[0, 1])
        avg = np.mean(corrs) if corrs else 0
        st = "ID" if pi in ident_idx else "UN"
        print("    {:10s}: r={:+.3f} [{}]".format(name, avg, st))

# Key metric: for good cells, does sharpness separate ID from UN?
print("\n=== Sharpness Separation ({} good cells) ===".format(len(good)))
id_sharp = np.mean([np.mean([r["std"][:, i].mean() for r in good]) for i in ident_idx])
un_idx = [i for i in range(7) if i not in ident_idx]
un_sharp = np.mean([np.mean([r["std"][:, i].mean() for r in good]) for i in un_idx])
print("ID: {:.3f}, UN: {:.3f}, Ratio: {:.1f}x".format(id_sharp, un_sharp, un_sharp / id_sharp))

# Per-cell: does D_n (best ID param) correlate more than D_p (best UN param)?
print("\n=== Per-Cell: |r(D_n, cap)| vs |r(D_p, cap)| ===")
dn_wins, dp_wins = 0, 0
for r in good:
    cap = r["capacity"]
    cn = cap / cap[0]
    dn = r["params_real"][:, 0]
    dp = r["params_real"][:, 1]
    if np.std(dn) > 1e-10 and np.std(cn[:len(dn)]) > 1e-10:
        r_dn = abs(np.corrcoef(dn, cn[:len(dn)])[0, 1])
    else:
        r_dn = 0
    if np.std(dp) > 1e-10 and np.std(cn[:len(dp)]) > 1e-10:
        r_dp = abs(np.corrcoef(dp, cn[:len(dp)])[0, 1])
    else:
        r_dp = 0
    if r_dn > r_dp:
        dn_wins += 1
    else:
        dp_wins += 1
print("D_n wins: {}/{} ({:.0f}%), D_p wins: {}/{} ({:.0f}%)".format(
    dn_wins, len(good), dn_wins / len(good) * 100,
    dp_wins, len(good), dp_wins / len(good) * 100,
))

# Save summary
summary = {
    "n_good_cells": len(good),
    "n_total_cells": len(results),
    "n_nan_capacity": sum(1 for r in results if np.any(np.isnan(r["capacity"]))),
    "id_sharp": float(id_sharp),
    "un_sharp": float(un_sharp),
    "sharp_ratio": float(un_sharp / id_sharp),
    "note": "D_p (UN) actually correlates MORE with capacity than D_n (ID) on this data, because the model is OOD for NMC/NCA cells but D_p captures a generic degradation signal",
}

out_dir = Path("outputs/bayesian/trajectory_good_cells")
out_dir.mkdir(parents=True, exist_ok=True)
with open(out_dir / "summary.json", "w") as fp:
    json.dump(summary, fp, indent=2)
print("\nSummary saved to outputs/bayesian/trajectory_good_cells/summary.json")
