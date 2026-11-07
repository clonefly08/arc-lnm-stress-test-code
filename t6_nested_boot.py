#!/usr/bin/env python3
"""T5-9 (R4 fallback): nested bootstrap of the leakage-free LOOCV, B = 200.
Each bootstrap resamples 226 patients with replacement and reruns the full
fold-internal LOOCV (selection |t| > 3.34, weights, model fit) for the
connectivity-weighted, lesion-overlap and covariates-only models.
Reports percentile CIs and Monte-Carlo SE for R and delta-R.
Run in WSL: python3 /mnt/n/ARC/analysis/t6_nested_boot.py"""
import os, json, time
os.environ.setdefault("OMP_NUM_THREADS", "8")
import numpy as np, pandas as pd
from scipy import stats

BASE = "/mnt/n/ARC"; LNM = f"{BASE}/lnm"; OUT = f"{BASE}/analysis/out"
B = 200
T_THRESH = stats.t.isf(0.0005, 200)
rng = np.random.default_rng(20261107)

summ = pd.read_csv(f"{LNM}/connmaps/gsp_sbc_summary.csv")
ph0 = pd.read_csv(f"{LNM}/stats/phenotype.csv")
names = [t for t in summ["lesion"] if t in set(ph0["tag"])]
ph = ph0.set_index("tag").loc[names].reset_index()
n = len(ph)
Y = ph["aq"].to_numpy(float)
Xc = np.column_stack([np.ones(n), ph["age"], ph["sex"],
                      ph["log_days"], ph["vol_mm3"] / 1e5])

ck = np.load(f"{LNM}/connmaps/.sbc_checkpoint.npz")
keep = [t in set(ph0["tag"]) for t in summ["lesion"]]
idx = [i for i, k in enumerate(keep) if k]
F = (ck["sum"][idx] / ck["n"]).astype(np.float64)

gm = nib = None
import nibabel as nib
gmb = np.asanyarray(nib.load(f"{LNM}/template/gm_mask_2mm.nii.gz").dataobj) > 0
gmflat = np.where(gmb.ravel())[0]
seed_idx = []
for t in names:
    m = np.asanyarray(nib.load(f"{LNM}/masks_mni/{t}/mask_mni.nii.gz").dataobj) > 0
    seed_idx.append(np.where((m & gmb).ravel()[gmflat])[0])

def residue(M, X):
    beta, *_ = np.linalg.lstsq(X, M, rcond=None)
    return M - X @ beta

def loocv(pidx):
    """Full leakage-free LOOCV on the given (resampled) patient indices."""
    m = len(pidx)
    pred_d = np.zeros(m); pred_o = np.zeros(m); pred_c = np.zeros(m)
    Fsub, Ysub, Xsub = F[pidx], Y[pidx], Xc[pidx]
    seeds = [seed_idx[i] for i in pidx]
    for j in range(m):
        tr = np.delete(np.arange(m), j)
        Xr = Xc[pidx[tr]]
        Yr = residue(Y[pidx[tr], None], Xr).ravel()
        Fr = residue(Fsub[tr], Xr)
        dof = len(tr) - Xr.shape[1] - 1
        sd = Fr.std(0); sd[sd == 0] = 1.0
        fz = (Fr - Fr.mean(0)) / sd
        yz = (Yr - Yr.mean()) / Yr.std()
        r = np.clip(yz @ fz / (len(tr) - 1), -1 + 1e-9, 1 - 1e-9)
        tk = np.nan_to_num(r * np.sqrt(dof / (1 - r ** 2)))
        w = tk[tk > 0]
        vox = np.where(tk > T_THRESH)[0]
        if len(vox) == 0:
            pred_d[j] = pred_o[j] = pred_c[j] = np.nan
            continue
        fz_sel = fz[:, vox]
        fz_sel = (fz_sel - fz_sel.mean(0)) / np.maximum(fz_sel.std(0), 1e-12)
        yzk = (Yr - Yr.mean()) / Yr.std()
        rr = np.clip(yzk @ fz_sel / (len(tr) - 1), -1 + 1e-9, 1 - 1e-9)
        tv = np.nan_to_num(rr * np.sqrt(dof / (1 - rr ** 2)))
        # connectivity-weighted score for held-out patient j on selected voxels
        c_j = Fsub[j, vox]
        wsel = tv
        pred_d[j] = float((wsel * c_j).sum() / max(np.abs(wsel).sum(), 1e-12))
        # lesion-overlap score: mean T inside held-out patient's own lesion voxels
        sj = seeds[j]
        pos_in_vox = np.searchsorted(vox, sj)
        pos_in_vox = pos_in_vox[(pos_in_vox < len(vox)) & (vox[np.minimum(pos_in_vox, len(vox)-1)] == sj)]
        pred_o[j] = float(tv[pos_in_vox].mean()) if len(pos_in_vox) else 0.0
        # covariates-only: fit on train, predict held-out
        beta_c, *_ = np.linalg.lstsq(Xc[pidx[tr]], Y[pidx[tr]], rcond=None)
        pred_c[j] = float(Xc[pidx[j]] @ beta_c)
    def rr(a, b):
        ok = np.isfinite(a) & np.isfinite(b)
        return float(np.corrcoef(a[ok], b[ok])[0, 1])
    y_true = Y[pidx]
    return rr(pred_d, y_true), rr(pred_o, y_true), rr(pred_c, y_true)

# sanity check: plain LOOCV on the full cohort must reproduce production values
t0 = time.time()
rd0, ro0, rc0 = loocv(np.arange(n))
print(f'sanity LOOCV (no resample): R_d={rd0:.3f} (expect 0.654) R_o={ro0:.3f} (0.640) R_c={rc0:.3f} (0.636)', flush=True)
assert abs(rd0 - 0.654) < 0.01, 'sanity check failed vs production values'

Rs_d, Rs_o, Rs_c, dRs = [], [], [], []
for b in range(B):
    pidx = np.sort(rng.choice(n, size=n, replace=True))
    try:
        rd, ro, rc = loocv(pidx)
        Rs_d.append(rd); Rs_o.append(ro); Rs_c.append(rc); dRs.append(rd - ro)
    except Exception as e:
        print(f"  boot {b} failed: {e}", flush=True)
    if (b + 1) % 20 == 0:
        print(f"[{b+1}/{B}] elapsed={time.time()-t0:.0f}s "
              f"R_d so far {np.mean(Rs_d):.3f}", flush=True)

out = dict(B=len(Rs_d),
           R_conn_mean=round(float(np.mean(Rs_d)), 4),
           R_conn_ci=[round(float(np.percentile(Rs_d, 2.5)), 4),
                      round(float(np.percentile(Rs_d, 97.5)), 4)],
           R_conn_mcse=round(float(np.std(Rs_d) / np.sqrt(len(Rs_d))), 4),
           R_overlap_mean=round(float(np.mean(Rs_o)), 4),
           R_cov_mean=round(float(np.mean(Rs_c)), 4),
           deltaR_mean=round(float(np.mean(dRs)), 4),
           deltaR_ci=[round(float(np.percentile(dRs, 2.5)), 4),
                      round(float(np.percentile(dRs, 97.5)), 4)],
           deltaR_mcse=round(float(np.std(dRs) / np.sqrt(len(dRs))), 4),
           elapsed_s=round(time.time() - t0, 1))
print(json.dumps(out, indent=1))
json.dump(out, open(f"{OUT}/t6_nested_boot.json", "w"), indent=1)
np.savez(f"{OUT}/t6_nested_boot.npz", R_d=Rs_d, R_o=Rs_o, R_c=Rs_c, dR=dRs)
print("T6_DONE")
