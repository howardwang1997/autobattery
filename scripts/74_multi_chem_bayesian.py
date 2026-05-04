import numpy as np
import h5py
import torch
import torch.nn as nn
import logging
import json
from pathlib import Path
from torch.utils.data import Dataset, DataLoader

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

PARAM_NAMES = ["D_n", "D_p", "t+", "SEI", "LAM_neg", "LAM_pos", "R_mult"]
LOG_PARAMS = [0, 1, 3]
IDENT_IDX = [0, 3, 4]
FISHER_IDENT = torch.tensor([1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0])
FISHER_RECON = torch.tensor([1.0, 0.014, 0.02, 1.0, 0.999, 0.022, 0.016])


class MultiChemCycleDataset(Dataset):
    def __init__(self, h5_path, split="train", frac=0.8, seed=42):
        all_V, all_params, all_cr = [], [], []
        with h5py.File(h5_path, "r") as f:
            for key in sorted(f["cells"].keys()):
                grp = f["cells"][key]
                V = grp["V"][:].astype(np.float32)
                gt = grp["gt_params"][:].astype(np.float32)
                n = V.shape[0]
                cr_val = float(grp.attrs.get("c_rate", 1.0))
                for i in range(n):
                    all_V.append(V[i])
                    all_params.append(gt[i])
                    all_cr.append(cr_val)

        all_V = np.array(all_V)
        all_params = np.array(all_params)
        all_cr = np.array(all_cr)

        params_log = all_params.copy()
        for i in LOG_PARAMS:
            params_log[:, i] = np.log10(np.maximum(all_params[:, i], 1e-30))

        self.p_mean = params_log.mean(0)
        self.p_std = params_log.std(0) + 1e-8
        pn = (params_log - self.p_mean) / self.p_std
        self.params_norm = torch.tensor(pn, dtype=torch.float32)
        self.V = torch.tensor(all_V, dtype=torch.float32)
        self.cr = torch.tensor(all_cr.reshape(-1, 1), dtype=torch.float32)

        rng = np.random.RandomState(seed)
        idx = rng.permutation(len(all_V))
        nt = int(len(all_V) * frac)
        self.idx = sorted(idx[:nt]) if split == "train" else sorted(idx[nt:])

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        j = self.idx[i]
        return self.V[j], self.cr[j], self.params_norm[j]


class BayesianModel(nn.Module):
    def __init__(self, n_time=100):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(n_time + 1, 256),
            nn.SiLU(),
            nn.Linear(256, 256),
            nn.SiLU(),
            nn.Linear(256, 256),
        )
        self.mu_head = nn.Linear(256, 7)
        self.logvar_head = nn.Linear(256, 7)

    def forward(self, V, cr):
        h = self.encoder(torch.cat([V, cr], dim=-1))
        return self.mu_head(h), self.logvar_head(h)

    def loss_fn(self, V, cr, params, fw=2.0):
        mu, lv = self.forward(V, cr)
        prec = torch.exp(-lv)
        nll = 0.5 * (prec * (params - mu) ** 2 + lv)
        fi = FISHER_IDENT.to(V.device)
        ident_loss = (nll * (1.0 + fw * fi)).sum(-1).mean()
        ps = torch.exp(0.5 * lv)
        fr = FISHER_RECON.to(V.device)
        cal_loss = ((ps - fr.detach()) ** 2).mean()
        return ident_loss + 0.1 * cal_loss


def train():
    device = torch.device("cuda")
    h5 = "data/synthetic_cycling/multi_chem_cycling.h5"
    train_ds = MultiChemCycleDataset(h5, split="train")
    val_ds = MultiChemCycleDataset(h5, split="val")
    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=128, shuffle=False, num_workers=0)
    logger.info(
        "Train: %d, Val: %d samples across 5 chemistries",
        len(train_ds),
        len(val_ds),
    )

    model = BayesianModel().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, 500)

    best_val = float("inf")
    for ep in range(1, 501):
        model.train()
        for V, cr, params in train_loader:
            V, cr, params = V.to(device), cr.to(device), params.to(device)
            loss = model.loss_fn(V, cr, params)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()

        if ep % 50 == 0:
            model.eval()
            si, su, vloss, nb = [], [], 0.0, 0
            with torch.no_grad():
                for V, cr, params in val_loader:
                    V, cr, params = V.to(device), cr.to(device), params.to(device)
                    vloss += model.loss_fn(V, cr, params).item()
                    nb += 1
                    _, lv = model(V, cr)
                    s = torch.exp(0.5 * lv)
                    si.append(s[:, IDENT_IDX].mean().item())
                    un_idx = [i for i in range(7) if i not in IDENT_IDX]
                    su.append(s[:, un_idx].mean().item())
            vloss /= nb
            logger.info(
                "Epoch %d: vloss=%.4f id=%.3f un=%.3f ratio=%.1fx",
                ep,
                vloss,
                np.mean(si),
                np.mean(su),
                np.mean(su) / max(np.mean(si), 1e-6),
            )
            if vloss < best_val:
                best_val = vloss
                out_dir = Path("outputs/bayesian/multi_chem")
                out_dir.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {
                        "model": model.state_dict(),
                        "p_mean": train_ds.p_mean,
                        "p_std": train_ds.p_std,
                        "epoch": ep,
                        "val_loss": vloss,
                    },
                    out_dir / "best.pt",
                )

    logger.info("Best val_loss=%.4f", best_val)
    return model, train_ds


def evaluate_synthetic(model, p_mean, p_std):
    device = next(model.parameters()).device
    model.eval()

    with h5py.File("data/synthetic_cycling/multi_chem_cycling.h5", "r") as f:
        chem_results = {}
        for key in sorted(f["cells"].keys()):
            grp = f["cells"][key]
            V = grp["V"][:].astype(np.float32)
            gt = grp["gt_params"][:].astype(np.float32)
            cap = grp["capacity"][:].astype(np.float32)
            chem_name = grp.attrs["chem_name"]
            if isinstance(chem_name, bytes):
                chem_name = chem_name.decode()

            cr_t = torch.ones(1, 1, device=device)
            mu_l, std_l = [], []
            with torch.no_grad():
                for i in range(V.shape[0]):
                    vt = torch.tensor(V[i : i + 1], device=device)
                    mu, lv = model(vt, cr_t)
                    mu_l.append(mu.cpu().numpy()[0])
                    std_l.append(torch.exp(0.5 * lv).cpu().numpy()[0])

            if chem_name not in chem_results:
                chem_results[chem_name] = {"mu": [], "std": [], "gt": [], "cap": []}
            chem_results[chem_name]["mu"].append(np.array(mu_l))
            chem_results[chem_name]["std"].append(np.array(std_l))
            chem_results[chem_name]["gt"].append(gt)
            chem_results[chem_name]["cap"].append(cap)

    print("\n=== Per-Chemistry Validation (Ground Truth Available) ===")
    print(
        "{:10s} {:>10s} {:>10s} {:>10s} {:>10s}".format(
            "Chem", "n_cells", "MAE(ID)", "MAE(UN)", "ID/UN Sharp"
        )
    )
    print("-" * 55)

    for chem, data in sorted(chem_results.items()):
        n_cells = len(data["mu"])
        gt_all = np.concatenate(data["gt"])
        mu_all = np.concatenate(data["mu"])
        std_all = np.concatenate(data["std"])

        gt_log = gt_all.copy()
        for i in LOG_PARAMS:
            gt_log[:, i] = np.log10(np.maximum(gt_all[:, i], 1e-30))
        gt_norm = (gt_log - p_mean) / p_std

        mae_id = np.mean(
            [np.abs(gt_norm[:, i] - mu_all[:, i]).mean() for i in IDENT_IDX]
        )
        un_idx = [i for i in range(7) if i not in IDENT_IDX]
        mae_un = np.mean(
            [np.abs(gt_norm[:, i] - mu_all[:, i]).mean() for i in un_idx]
        )
        sharp_id = np.mean([std_all[:, i].mean() for i in IDENT_IDX])
        sharp_un = np.mean([std_all[:, i].mean() for i in un_idx])

        print(
            "{:10s} {:10d} {:10.3f} {:10.3f} {:10.1f}x".format(
                chem,
                n_cells,
                mae_id,
                mae_un,
                sharp_un / max(sharp_id, 1e-6),
            )
        )

    return chem_results


def evaluate_experimental(model, p_mean, p_std):
    device = next(model.parameters()).device
    model.eval()
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
            if ncyc < 20:
                continue

            sample_idx = list(range(0, ncyc, max(1, ncyc // 15)))
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
                    "mu": np.array(mu_l),
                    "std": np.array(std_l),
                    "cap": cap[: len(sample_idx)],
                }
            )

    logger.info("Experimental: %d cells", len(all_res))

    header = "{:10s} {:>10s} {:>10s} {:>8s} {:>16s}".format(
        "Param", "Sharpness", "|r| cap", "Mono%", "Status"
    )
    print("\n=== Experimental Trajectory Analysis ===")
    print(header)
    print("-" * len(header))

    summary = {}
    for pi, name in enumerate(PARAM_NAMES):
        sharp = np.mean([r["std"][:, pi].mean() for r in all_res])
        corrs, monos = [], []
        for r in all_res:
            if len(r["mu"]) > 5:
                c0 = r["cap"][0]
                cn = r["cap"] / c0 if c0 > 0 else r["cap"]
                p = r["mu"][:, pi]
                if np.std(p) > 1e-10 and np.std(cn[: len(p)]) > 1e-10:
                    corrs.append(abs(np.corrcoef(p, cn[: len(p)])[0, 1]))
                diffs = np.diff(p)
                if len(diffs) > 0 and np.std(diffs) > 0:
                    mono_pct = max(
                        (diffs > 0).sum() / len(diffs),
                        (diffs < 0).sum() / len(diffs),
                    )
                    monos.append(mono_pct)

        avg_r = np.mean(corrs) if corrs else 0
        avg_m = np.mean(monos) * 100 if monos else 0
        st = "IDENTIFIABLE" if pi in IDENT_IDX else "unidentifiable"
        print(
            "{:10s} {:10.3f} {:10.3f} {:7.1f}% {:>16s}".format(
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
            id_s, un_s, un_s / max(id_s, 1e-6)
        )
    )

    summary["id_sharp"] = float(id_s)
    summary["un_sharp"] = float(un_s)
    summary["ratio"] = float(un_s / max(id_s, 1e-6))
    summary["n_cells"] = len(all_res)

    out_dir = Path("outputs/bayesian/multi_chem_exp")
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_dir / "trajectory_analysis.npz",
        results=all_res,
        param_names=PARAM_NAMES,
        ident_idx=IDENT_IDX,
    )
    with open(out_dir / "summary.json", "w") as fp:
        json.dump(summary, fp, indent=2)

    return all_res


if __name__ == "__main__":
    model, train_ds = train()
    evaluate_synthetic(model, train_ds.p_mean, train_ds.p_std)
    evaluate_experimental(model, train_ds.p_mean, train_ds.p_std)
