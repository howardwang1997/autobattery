import numpy as np
import h5py
import torch
import torch.nn as nn
import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

PARAM_NAMES = ["D_n", "D_p", "t+", "SEI", "LAM_neg", "LAM_pos", "R_mult"]
IDENT_IDX = [0, 3, 4]


class CalibratedModel(nn.Module):
    def __init__(self, n_time=100, hidden=256):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(n_time + 1, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
        )
        self.mu_head = nn.Linear(hidden, 7)
        self.logvar_head = nn.Linear(hidden, 7)

    def forward(self, V, cr):
        h = self.encoder(torch.cat([V, cr], dim=-1))
        return self.mu_head(h), self.logvar_head(h)


def main():
    device = torch.device("cuda")
    ckpt = torch.load(
        "outputs/bayesian/calibrated/best.pt", map_location=device, weights_only=False
    )
    model = CalibratedModel().to(device)
    model.load_state_dict(ckpt["model"])
    p_mean = ckpt["p_mean"]
    p_std = ckpt["p_std"]
    model.eval()
    logger.info("Loaded calibrated model from epoch %d", ckpt.get("epoch", -1))

    all_res = []
    with h5py.File("data/experimental/experimental_cycling.h5", "r") as f:
        for key in sorted(f["cells"].keys()):
            grp = f["cells"][key]
            V = grp["V"][:]
            cap = grp["capacity"][:]
            ncyc = int(grp.attrs["n_cycles"])
            bc = grp.attrs["barcode"]
            if isinstance(bc, bytes):
                bc = bc.decode()
            cap_init = float(grp.attrs["cap_initial"])

            sample_idx = list(range(0, ncyc, max(1, ncyc // 20)))
            cr_t = torch.ones(1, 1, device=device)
            mu_l, std_l = [], []
            with torch.no_grad():
                for si in sample_idx:
                    vt = torch.tensor(V[si : si + 1], dtype=torch.float32).to(device)
                    mu, lv = model(vt, cr_t)
                    mu_l.append(mu.cpu().numpy()[0])
                    std_l.append(torch.exp(0.5 * lv).cpu().numpy()[0])

            all_res.append(
                {
                    "barcode": bc,
                    "ncyc": ncyc,
                    "cap": cap[sample_idx[: len(mu_l)]],
                    "cap_init": cap_init,
                    "mu": np.array(mu_l),
                    "std": np.array(std_l),
                }
            )

    logger.info("Processed %d experimental cells", len(all_res))

    # Sharpness analysis
    print("\n" + "=" * 70)
    print("PARAMETER IDENTIFIABILITY ON EXPERIMENTAL DATA (n={})".format(len(all_res)))
    print("=" * 70)

    hdr = "{:10s} {:>10s} {:>10s} {:>8s} {:>12s}".format(
        "Param", "Sharpness", "|r| cap", "Mono%", "Status"
    )
    print(hdr)
    print("-" * len(hdr))

    summary = {}
    for pi, name in enumerate(PARAM_NAMES):
        sharp = np.mean([r["std"][:, pi].mean() for r in all_res])
        corrs, monos = [], []
        for r in all_res:
            if len(r["mu"]) < 5:
                continue
            cn = r["cap"] / r["cap_init"] if r["cap_init"] > 0 else r["cap"]
            p = r["mu"][:, pi]
            if np.std(p) > 1e-10 and np.std(cn[: len(p)]) > 1e-10:
                corrs.append(abs(np.corrcoef(p, cn[: len(p)])[0, 1]))
            diffs = np.diff(p)
            if len(diffs) > 0 and np.std(diffs) > 0:
                mono = max((diffs > 0).sum(), (diffs < 0).sum()) / len(diffs)
                monos.append(mono)

        avg_r = np.mean(corrs) if corrs else 0
        avg_m = np.mean(monos) * 100 if monos else 0
        st = "ID" if pi in IDENT_IDX else "UN"
        print(
            "{:10s} {:10.3f} {:10.3f} {:7.1f}% {:>12s}".format(
                name, sharp, avg_r, avg_m, st
            )
        )
        summary[name] = {
            "sharpness": float(sharp),
            "corr_cap": float(avg_r),
            "mono_pct": float(avg_m),
            "status": st,
        }

    id_s = np.mean(
        [np.mean([r["std"][:, i].mean() for r in all_res]) for i in IDENT_IDX]
    )
    un_idx = [i for i in range(7) if i not in IDENT_IDX]
    un_s = np.mean(
        [np.mean([r["std"][:, i].mean() for r in all_res]) for i in un_idx]
    )
    print(
        "\nID sharpness: {:.3f}  UN: {:.3f}  Ratio: {:.1f}x".format(
            id_s, un_s, un_s / id_s
        )
    )
    summary["id_sharp"] = float(id_s)
    summary["un_sharp"] = float(un_s)
    summary["ratio"] = float(un_s / id_s)
    summary["n_cells"] = len(all_res)

    # Capacity fade analysis
    fades = []
    for r in all_res:
        if r["cap_init"] > 0 and len(r["cap"]) > 0:
            fade = (1 - r["cap"][-1] / r["cap_init"]) * 100
            fades.append(fade)
    fades = np.array(fades)
    print(
        "\nCapacity fade: [{:.1f}%, {:.1f}%], mean={:.1f}%".format(
            fades.min(), fades.max(), fades.mean()
        )
    )

    # Fast/slow degraders
    if len(fades) > 10:
        median_fade = np.median(fades)
        slow = [
            r
            for r, f in zip(all_res, fades)
            if f < median_fade and len(r["mu"]) > 5
        ]
        fast = [
            r
            for r, f in zip(all_res, fades)
            if f >= median_fade and len(r["mu"]) > 5
        ]
        print(
            "\nSlow ({:.1f}% avg fade, n={}) vs Fast ({:.1f}% avg fade, n={})".format(
                np.mean([f for f in fades if f < median_fade]),
                len(slow),
                np.mean([f for f in fades if f >= median_fade]),
                len(fast),
            )
        )

        for label, group in [("Slow", slow), ("Fast", fast)]:
            print("  {}:".format(label))
            for pi, name in enumerate(PARAM_NAMES):
                corrs = []
                for r in group:
                    cn = r["cap"] / r["cap_init"] if r["cap_init"] > 0 else r["cap"]
                    p = r["mu"][:, pi]
                    if np.std(p) > 1e-10 and np.std(cn[: len(p)]) > 1e-10:
                        corrs.append(np.corrcoef(p, cn[: len(p)])[0, 1])
                avg = np.mean(corrs) if corrs else 0
                st = "ID" if pi in IDENT_IDX else "UN"
                print("    {:10s}: r={:+.3f} [{}]".format(name, avg, st))

    # Per-cell ranking
    print("\nPer-Cell #1 Ranking:")
    rank_counts = {n: 0 for n in PARAM_NAMES}
    for r in all_res:
        if len(r["mu"]) < 5:
            continue
        cn = r["cap"] / r["cap_init"] if r["cap_init"] > 0 else r["cap"]
        best_name, best_corr = "", 0
        for pi, name in enumerate(PARAM_NAMES):
            p = r["mu"][:, pi]
            if np.std(p) > 1e-10 and np.std(cn[: len(p)]) > 1e-10:
                c = abs(np.corrcoef(p, cn[: len(p)])[0, 1])
                if c > best_corr:
                    best_corr = c
                    best_name = name
        if best_name:
            rank_counts[best_name] += 1

    for name in PARAM_NAMES:
        st = "ID" if PARAM_NAMES.index(name) in IDENT_IDX else "UN"
        print(
            "  {:10s}: {} ({:.1f}%) [{}]".format(
                name,
                rank_counts[name],
                rank_counts[name] / max(len(all_res), 1) * 100,
                st,
            )
        )

    out_dir = Path("outputs/bayesian/calibrated_exp_v2")
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_dir / "trajectory_analysis.npz",
        results=all_res,
        param_names=PARAM_NAMES,
        ident_idx=IDENT_IDX,
    )
    with open(out_dir / "summary.json", "w") as fp:
        json.dump(summary, fp, indent=2)
    print("\nSaved to {}".format(out_dir))


if __name__ == "__main__":
    main()
