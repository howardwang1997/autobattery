import numpy as np

d = np.load("outputs/bayesian/trajectory_analysis.npz", allow_pickle=True)
results = d["results"]
param_names = list(d["param_names"])
ident_idx = list(d["ident_idx"])

print("=== Deep Trajectory Analysis (Original Calibrated Model) ===")
print("N cells: {}, all >= 50 cycles".format(len(results)))

metrics = {}
for name in param_names:
    metrics[name] = {"smooth": [], "mono": [], "corr": [], "drift": []}

for r in results:
    cap = r["capacity"]
    c0 = cap[0] if cap[0] > 0 else 1.0
    cap_norm = cap / c0

    for pi, name in enumerate(param_names):
        vals = r["params_real"][:, pi]

        if len(vals) < 5:
            continue

        diffs = np.diff(vals)
        if np.std(vals) > 1e-10:
            smooth = 1.0 - np.mean(np.abs(diffs)) / (np.std(vals) + 1e-10)
            metrics[name]["smooth"].append(max(0, smooth))

        if len(diffs) > 0 and np.std(diffs) > 0:
            mono = max((diffs > 0).sum(), (diffs < 0).sum()) / len(diffs)
            metrics[name]["mono"].append(mono)

        if np.std(vals) > 1e-10 and np.std(cap_norm[: len(vals)]) > 1e-10:
            corr = np.corrcoef(vals, cap_norm[: len(vals)])[0, 1]
            metrics[name]["corr"].append(corr)

        if np.std(vals) > 1e-10:
            drift = abs(vals[-1] - vals[0]) / (np.std(vals) + 1e-10)
            metrics[name]["drift"].append(drift)

hdr = "{:10s} {:>8s} {:>8s} {:>8s} {:>8s} {:>8s} {:>12s}".format(
    "Param", "Smooth", "Mono%", "|r|cap", "Drift", "Sharp", "Status"
)
print("\n" + hdr)
print("-" * len(hdr))

for pi, name in enumerate(param_names):
    s = np.mean(metrics[name]["smooth"]) if metrics[name]["smooth"] else 0
    m = np.mean(metrics[name]["mono"]) * 100 if metrics[name]["mono"] else 0
    c_arr = [abs(x) for x in metrics[name]["corr"]]
    c = np.mean(c_arr) if c_arr else 0
    dr = np.mean(metrics[name]["drift"]) if metrics[name]["drift"] else 0
    sharp = np.mean([r["std"][:, pi].mean() for r in results])
    st = "IDENTIFIABLE" if pi in ident_idx else "unidentifiable"
    print(
        "{:10s} {:8.3f} {:7.1f}% {:8.3f} {:8.3f} {:8.3f} {:>12s}".format(
            name, s, m, c, dr, sharp, st
        )
    )

id_names = [param_names[i] for i in ident_idx]
un_names = [param_names[i] for i in range(7) if i not in ident_idx]

id_smooth = np.mean([np.mean(metrics[n]["smooth"]) for n in id_names])
un_smooth = np.mean([np.mean(metrics[n]["smooth"]) for n in un_names])
id_mono = np.mean([np.mean(metrics[n]["mono"]) for n in id_names]) * 100
un_mono = np.mean([np.mean(metrics[n]["mono"]) for n in un_names]) * 100
id_corr = np.mean([np.mean([abs(x) for x in metrics[n]["corr"]]) for n in id_names])
un_corr = np.mean([np.mean([abs(x) for x in metrics[n]["corr"]]) for n in un_names])

print("\n=== ID vs UN Averages ===")
print("{:15s} {:>10s} {:>10s} {:>10s}".format("Metric", "ID", "UN", "Ratio"))
print(
    "{:15s} {:10.3f} {:10.3f} {:10.2f}".format(
        "Smoothness", id_smooth, un_smooth, un_smooth / max(id_smooth, 1e-6)
    )
)
print(
    "{:15s} {:9.1f}% {:9.1f}% {:10.2f}".format(
        "Mono%", id_mono, un_mono, un_mono / max(id_mono, 1e-6)
    )
)
print(
    "{:15s} {:10.3f} {:10.3f} {:10.2f}".format(
        "|r| capacity", id_corr, un_corr, un_corr / max(id_corr, 1e-6)
    )
)

# Per-cell analysis: for each cell, rank params by |correlation with capacity|
# If identifiable params consistently rank higher, that's validation
print("\n=== Per-Cell Ranking: How often does each param rank #1? ===")
rank_counts = {name: 0 for name in param_names}
n_cells = 0
for r in results:
    cap = r["cap"]
    c0 = cap[0] if cap[0] > 0 else 1.0
    cap_norm = cap / c0
    if len(r["params_real"]) < 5:
        continue
    corrs = []
    for pi, name in enumerate(param_names):
        vals = r["params_real"][:, pi]
        if np.std(vals) > 1e-10 and np.std(cap_norm[: len(vals)]) > 1e-10:
            corrs.append((abs(np.corrcoef(vals, cap_norm[: len(vals)])[0, 1]), name))
        else:
            corrs.append((0, name))
    if corrs:
        best = max(corrs, key=lambda x: x[0])
        rank_counts[best[1]] += 1
        n_cells += 1

for name in param_names:
    st = "ID" if param_names.index(name) in ident_idx else "UN"
    print(
        "  {:10s}: {} ({:.1f}%) [{}]".format(
            name, rank_counts[name], rank_counts[name] / max(n_cells, 1) * 100, st
        )
    )

# Capacity fade rate analysis: split cells into fast/slow degraders
print("\n=== Fast vs Slow Degraders ===")
fade_rates = []
for r in results:
    cap = r["cap"]
    if cap[0] > 0 and len(cap) > 1:
        fade = (cap[0] - cap[-1]) / cap[0]
        fade_rates.append((fade, r))

fade_rates.sort(key=lambda x: x[0])
n = len(fade_rates)
slow = fade_rates[: n // 3]
fast = fade_rates[2 * n // 3 :]

for label, group in [("Slow (bottom 1/3)", slow), ("Fast (top 1/3)", fast)]:
    print("\n  {} ({} cells, avg fade {:.1f}%)".format(
        label, len(group), np.mean([f for f, _ in group]) * 100
    ))
    for pi, name in enumerate(param_names):
        corrs = []
        for _, r in group:
            cap = r["cap"]
            c0 = cap[0] if cap[0] > 0 else 1.0
            cn = cap / c0
            vals = r["params_real"][:, pi]
            if np.std(vals) > 1e-10 and np.std(cn[: len(vals)]) > 1e-10:
                corrs.append(np.corrcoef(vals, cn[: len(vals)])[0, 1])
        avg = np.mean(corrs) if corrs else 0
        st = "ID" if pi in ident_idx else "UN"
        print("    {:10s}: r={:.3f} [{}]".format(name, avg, st))
