#!/usr/bin/env python3
"""
Train calibrated Bayesian model on NMC fullfield data and evaluate on experimental cells.
"""

import os
os.environ["MKL_THREADING_LAYER"] = "GNU"

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


class FullfieldDataset(Dataset):
    def __init__(self, h5_path, split="train", frac=0.8, seed=42):
        with h5py.File(h5_path, "r") as f:
            params = f["params"][:].astype(np.float32)
            V = f["V"][:].astype(np.float32)
            cr = f["c_rates"][:].astype(np.float32)

        params_log = params.copy()
        for i in LOG_PARAMS:
            params_log[:, i] = np.log10(np.maximum(params[:, i], 1e-30))

        self.p_mean = params_log.mean(0)
        self.p_std = params_log.std(0) + 1e-8
        pn = (params_log - self.p_mean) / self.p_std
        self.params_norm = torch.tensor(pn, dtype=torch.float32)
        self.V = torch.tensor(V, dtype=torch.float32)
        self.cr = torch.tensor(cr.reshape(-1, 1), dtype=torch.float32)

        rng = np.random.RandomState(seed)
        idx = rng.permutation(len(V))
        nt = int(len(V) * frac)
        self.idx = sorted(idx[:nt]) if split == "train" else sorted(idx[nt:])

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        j = self.idx[i]
        return self.V[j], self.cr[j], self.params_norm[j]


class BayesianModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(101, 256), nn.SiLU(),
            nn.Linear(256, 256), nn.SiLU(),
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
        return ident_loss + 0.1 * ((ps - fr.detach()) ** 2).mean()


def train():
    device = torch.device("cuda")
    train_ds = FullfieldDataset("data/fullfield/fullfield_nmc_degradation.h5", split="train")
    val_ds = FullfieldDataset("data/fullfield/fullfield_nmc_degradation.h5", split="val")
    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=128, shuffle=False, num_workers=0)
    logger.info("Train: %d, Val: %d NMC samples", len(train_ds), len(val_ds))

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

        if ep % 100 == 0:
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
                ep, vloss, np.mean(si), np.mean(su),
                np.mean(su) / max(np.mean(si), 1e-6),
            )
            if vloss < best_val:
                best_val = vloss
                out_dir = Path("outputs/bayesian/nmc_calibrated")
                out_dir.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {
                        "model": model.state_dict(),
                        "p_mean": train_ds.p_mean,
                        "p_std": train_ds.p_std,
                        "epoch": ep,
                        "val_loss": vloss,
                        "chemistry": "NMC811",
                    },
                    out_dir / "best.pt",
                )

    logger.info("Best val_loss=%.4f", best_val)
    return model, train_ds


def validate_synthetic(model, p_mean, p_std):
    device = next(model.parameters()).device
    model.eval()

    with h5py.File("data/fullfield/fullfield_nmc_degradation.h5", "r") as f:
        V = f["V"][:].astype(np.float32)
        params = f["params"][:].astype(np.float32)

    params_log = params.copy()
    for i in LOG_PARAMS:
        params_log[:, i] = np.log10(np.maximum(params[:, i], 1e-30))
    pn = (params_log - p_mean) / p_std

    cr_t = torch.ones(1, 1, device=device)
    mu_all, std_all = [], []
    with torch.no_grad():
        for i in range(0, len(V), 64):
            vt = torch.tensor(V[i:i+64], dtype=torch.float32).to(device)
            crt = torch.ones(len(vt), 1, device=device)
            mu, lv = model(vt, crt)
            mu_all.append(mu.cpu().numpy())
            std_all.append(torch.exp(0.5 * lv).cpu().numpy())

    mu_all = np.concatenate(mu_all)
    std_all = np.concatenate(std_all)

    print("\n=== Synthetic NMC Validation ===")
    hdr = "{:10s} {:>10s} {:>10s} {:>10s} {:>10s}".format(
        "Param", "MAE", "Sharp", "Cov1sig", "Status"
    )
    print(hdr)
    print("-" * len(hdr))

    for pi, name in enumerate(PARAM_NAMES):
        mae = np.abs(mu_all[:, pi] - pn[:, pi]).mean()
        sharp = std_all[:, pi].mean()
        in_1sig = (np.abs(mu_all[:, pi] - pn[:, pi]) < std_all[:, pi]).mean()
        st = "ID" if pi in IDENT_IDX else "UN"
        print("{:10s} {:10.3f} {:10.3f} {:10.3f} {:>10s}".format(
            name, mae, sharp, in_1sig, st
        ))


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
            cap_init = float(grp.attrs.get("cap_initial", grp.attrs.get("cap_init", cap[0])))

            sample_idx = list(range(0, ncyc, max(1, ncyc // 20)))
            cr_t = torch.ones(1, 1, device=device)
            mu_l, std_l = [], []
            with torch.no_grad():
                for si in sample_idx:
                    vt = torch.tensor(V[si:si+1], dtype=torch.float32).to(device)
                    mu, lv = model(vt, cr_t)
                    mu_l.append(mu.cpu().numpy()[0])
                    std_l.append(torch.exp(0.5 * lv).cpu().numpy()[0])

            all_res.append({
                "barcode": bc, "ncyc": ncyc,
                "cap": cap[sample_idx[:len(mu_l)]],
                "cap_init": cap_init,
                "mu": np.array(mu_l), "std": np.array(std_l),
            })

    logger.info("Experimental: %d cells", len(all_res))

    print("\n" + "=" * 70)
    print("NMC-BAYESIAN ON EXPERIMENTAL DATA (n={})".format(len(all_res)))
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
            if np.std(p) > 1e-10 and np.std(cn[:len(p)]) > 1e-10:
                corrs.append(abs(np.corrcoef(p, cn[:len(p)])[0, 1]))
            diffs = np.diff(p)
            if len(diffs) > 0 and np.std(diffs) > 0:
                mono = max((diffs > 0).sum(), (diffs < 0).sum()) / len(diffs)
                monos.append(mono)

        avg_r = np.mean(corrs) if corrs else 0
        avg_m = np.mean(monos) * 100 if monos else 0
        st = "ID" if pi in IDENT_IDX else "UN"
        print("{:10s} {:10.3f} {:10.3f} {:7.1f}% {:>12s}".format(
            name, sharp, avg_r, avg_m, st
        ))
        summary[name] = {
            "sharpness": float(sharp),
            "corr_cap": float(avg_r),
            "mono_pct": float(avg_m),
            "status": st,
        }

    id_s = np.mean([np.mean([r["std"][:, i].mean() for r in all_res]) for i in IDENT_IDX])
    un_idx = [i for i in range(7) if i not in IDENT_IDX]
    un_s = np.mean([np.mean([r["std"][:, i].mean() for r in all_res]) for i in un_idx])
    print("\nID sharpness: {:.3f}  UN: {:.3f}  Ratio: {:.1f}x".format(id_s, un_s, un_s / id_s))

    summary["id_sharp"] = float(id_s)
    summary["un_sharp"] = float(un_s)
    summary["ratio"] = float(un_s / id_s)
    summary["n_cells"] = len(all_res)

    # Compare with LFP-trained model
    print("\n=== Comparison: NMC-trained vs LFP-trained ===")
    print("  NMC model: ID={:.3f} UN={:.3f} Ratio={:.1f}x".format(id_s, un_s, un_s/id_s))
    print("  LFP model: ID=0.241 UN=0.907 Ratio=3.8x")

    out_dir = Path("outputs/bayesian/nmc_exp")
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
    validate_synthetic(model, train_ds.p_mean, train_ds.p_std)
    evaluate_experimental(model, train_ds.p_mean, train_ds.p_std)
