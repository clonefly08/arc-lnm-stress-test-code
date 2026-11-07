#!/usr/bin/env python3
"""C2 + C3 using Yeo-7 composition entropy.

C2: c_i(v) vs H_i correlation at 10 reference voxels — does the connectivity
    regressor track composition entropy?
C3: low-entropy tertile (Yeo-7) rerun of the main regression — does the map
    survive in the cleanest subset?
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "4")
import json
import numpy as np
import pandas as pd
import nibabel as nib
from scipy import stats

BASE = "/mnt/n/ARC/lnm"
STATS = f"{BASE}/stats"
OUT = f"{STATS}/revision"
os.makedirs(OUT, exist_ok=True)

# ---------- load data ----------
summ = pd.read_csv(f"{BASE}/connmaps/gsp_sbc_summary.csv")
ph0 = pd.read_csv(f"{STATS}/phenotype.csv")
names = [t for t in summ["lesion"] if t in set(ph0["tag"])]
ph = ph0.set_index("tag").loc[names].reset_index()
n = len(ph)
Y = ph["aq"].to_numpy(float)
Xc = np.column_stack([np.ones(n), ph["age"], ph["sex"],
                      ph["log_days"], ph["vol_mm3"] / 1e5])

ck = np.load(f"{BASE}/connmaps/.sbc_checkpoint.npz")
keep = [t in set(ph0["tag"]) for t in summ["lesion"]]
idx = [i for i, k in enumerate(keep) if k]
F = (ck["sum"][idx] / ck["n"]).astype(np.float64)     # n × V

gm = nib.load(f"{BASE}/template/gm_mask_2mm.nii.gz")
gmb = np.asanyarray(gm.dataobj) > 0
gm_flat = np.where(gmb.ravel())[0]
V = len(gm_flat)
inv = np.linalg.inv(gm.affine)

H7 = np.load(f"{OUT}/H_yeo.npy")     # Yeo-7 composition entropy (226,)
print(f"n={n}  V={V}  H mean={H7.mean():.3f} SD={H7.std():.3f}", flush=True)

# ---------- C2: c_i(v) vs H_i at reference voxels ----------
ref_mni = [(-34, -4, -8), (12, -8, -46), (-45, 25, 8), (45, 25, 8),
           (-50, -25, 5), (50, -25, 5), (-5, 50, 25), (5, 50, 25),
           (-20, -70, 40), (20, -70, 40)]
C2 = []
for xyz in ref_mni:
    vx = np.round((inv @ np.array([*xyz, 1.0]))[:3]).astype(int)
    vidx = np.ravel_multi_index(tuple(vx), gmb.shape)
    pos = np.searchsorted(gm_flat, vidx)
    if pos >= V or gm_flat[pos] != vidx:
        C2.append(dict(mni=list(xyz), status="not_in_GM"))
        continue
    r, p = stats.pearsonr(F[:, pos], H7)
    C2.append(dict(mni=list(xyz), r_c_H=round(float(r), 3), p=float(f"{p:.2e}")))
mean_abs_r = float(np.mean([abs(x["r_c_H"]) for x in C2 if "r_c_H" in x]))
print(f"C2: mean |r(c_i, H_i)| at 10 refs = {mean_abs_r:.3f}", flush=True)
for x in C2:
    print(f"  {x}", flush=True)

# ---------- C3: low-entropy tertile rerun ----------
thr_lo = np.quantile(H7, 1 / 3)
sel = np.where(H7 <= thr_lo)[0]
ns = len(sel)
print(f"\nC3: low-entropy tertile n={ns} (H <= {thr_lo:.3f})", flush=True)

def residue(M, X):
    beta, *_ = np.linalg.lstsq(X, M, rcond=None)
    return M - X @ beta

Ys = Y[sel]
Xs = Xc[sel]
Yr = residue(Ys[:, None], Xs).ravel()
Fr = residue(F[sel], Xs)
dof = ns - Xs.shape[1] - 1
yz = (Yr - Yr.mean()) / Yr.std()
sd = Fr.std(0); sd[sd == 0] = 1.0
fz = (Fr - Fr.mean(0)) / sd
fz[~np.isfinite(fz)] = 0.0
r_all = np.clip(yz @ fz / (ns - 1), -1 + 1e-9, 1 - 1e-9)
tobs = np.nan_to_num(r_all * np.sqrt(dof / (1 - r_all ** 2)))
print(f"uncorrected: max|t| = {np.abs(tobs).max():.2f}", flush=True)

# max-T permutation on this subset
K = 5000
rng = np.random.default_rng(42)
maxpos = np.empty(K); maxneg = np.empty(K)
for k in range(K):
    perm = rng.permutation(ns)
    yzp = (Yr[perm] - Yr[perm].mean()) / Yr[perm].std()
    rr = np.clip(yzp @ fz / (ns - 1), -1 + 1e-9, 1 - 1e-9)
    tk = np.nan_to_num(rr * np.sqrt(dof / (1 - rr ** 2)))
    maxpos[k] = tk.max(); maxneg[k] = tk.min()
    if (k + 1) % 1000 == 0:
        print(f"  perm {k+1}/{K}", flush=True)

tp = np.sort(maxpos)[::-1][int(np.ceil(0.025 * K)) - 1]
tn = np.sort(maxneg)[int(np.ceil(0.025 * K)) - 1]
pos = tobs > tp; neg = tobs < tn
n_sig = int(pos.sum() + neg.sum())
print(f"C3 low-entropy tertile: n_sig = {n_sig} (pos {int(pos.sum())} + neg {int(neg.sum())})", flush=True)

# peaks
if n_sig > 0:
    for sgn, nm in ((1, "pos"), (-1, "neg")):
        tt = np.where((pos if sgn == 1 else neg) if False else
                       ((tobs > tp) & (tobs > 0)) if sgn == 1 else
                       ((tobs < tn) & (tobs < 0)), tobs * sgn, -np.inf)
        pk = int(np.argmax(tt))
        vx = np.unravel_index(pk, (91, 109, 91))
        xyz = np.round((inv @ np.array([*vx, 1.0]))[:3]).astype(int)
        print(f"  {nm} peak: t={tobs[pk]:.2f} at MNI {list(xyz)}", flush=True)

json.dump(dict(n=ns, n_sig=n_sig, n_pos=int(pos.sum()), n_neg=int(neg.sum()),
               mean_abs_r=mean_abs_r, threshold_t_pos=round(float(tp), 3),
               threshold_t_neg=round(float(tn), 3)),
          open(f"{OUT}/C2_C3_yeo.json", "w"), indent=1)
print("C2_C3_DONE")
