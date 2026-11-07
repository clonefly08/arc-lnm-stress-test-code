#!/usr/bin/env python3
"""T5 tail items: (7) exclude 15 flagged registrations (n=211); (8) AQ ceiling
(drop 32 AQ>=93.8 + Tobit at 12 voxels); (6) C2 full table + variance
heterogeneity + leave-one-out influence on the two peaks.
Run in WSL: python3 /mnt/n/ARC/analysis/t5_tail.py"""
import os, json
os.environ.setdefault("OMP_NUM_THREADS", "8")
import numpy as np, pandas as pd, nibabel as nib
from scipy import stats, optimize

BASE = "/mnt/n/ARC"; LNM = f"{BASE}/lnm"; OUT = f"{BASE}/analysis/out"
K = 5000
rng = np.random.default_rng(20261107)

summ = pd.read_csv(f"{LNM}/connmaps/gsp_sbc_summary.csv")
ph0 = pd.read_csv(f"{LNM}/stats/phenotype.csv")
names = [t for t in summ["lesion"] if t in set(ph0["tag"])]
L = len(names)
tpos = {t: i for i, t in enumerate(names)}
ph = ph0.set_index("tag").loc[names].reset_index()
Y = ph["aq"].to_numpy(float)
Xc = np.column_stack([np.ones(L), ph["age"], ph["sex"], ph["log_days"], ph["vol_mm3"] / 1e5])

ck = np.load(f"{LNM}/connmaps/.sbc_checkpoint.npz")
idx = [i for i, t in enumerate(summ["lesion"]) if t in set(ph0["tag"])]
F = (ck["sum"][idx] / ck["n"]).astype(np.float64)
PERMS = {L: [rng.permutation(L) for _ in range(K)]}

def residue(M, X):
    beta, *_ = np.linalg.lstsq(X, M, rcond=None)
    return M - X @ beta

def run_variant(Fm, pidx, label):
    Xs = Xc[pidx]
    Yr = residue(Y[pidx, None], Xs).ravel()
    Fr = residue(Fm, Xs)
    dof = len(pidx) - Xs.shape[1] - 1
    yz = (Yr - Yr.mean()) / Yr.std()
    sd = Fr.std(0); sd[sd == 0] = 1.0
    fz = (Fr - Fr.mean(0)) / sd
    fz[~np.isfinite(fz)] = 0.0
    r = np.clip(yz @ fz / (len(pidx) - 1), -1 + 1e-9, 1 - 1e-9)
    tobs = np.nan_to_num(r * np.sqrt(dof / (1 - r ** 2)))
    perms = PERMS.get(len(pidx)) or [rng.permutation(len(pidx)) for _ in range(K)]
    maxpos = np.empty(K); maxneg = np.empty(K)
    for k in range(K):
        perm = perms[k]
        yzp = (Yr[perm] - Yr[perm].mean()) / Yr[perm].std()
        rr = np.clip(yzp @ fz / (len(pidx) - 1), -1 + 1e-9, 1 - 1e-9)
        tk = np.nan_to_num(rr * np.sqrt(dof / (1 - rr ** 2)))
        maxpos[k] = tk.max(); maxneg[k] = tk.min()
    tp5 = float(np.sort(maxpos)[::-1][int(np.ceil(0.05 * K)) - 1])
    tn5 = float(-np.sort(maxneg)[int(np.ceil(0.05 * K)) - 1])
    tp25 = float(np.sort(maxpos)[::-1][int(np.ceil(0.025 * K)) - 1])
    tn25 = float(-np.sort(maxneg)[int(np.ceil(0.025 * K)) - 1])
    n5 = int(((tobs > tp5) | (tobs < -tn5)).sum())
    n25 = int(((tobs > tp25) | (tobs < -tn25)).sum())
    print(f"{label:30s} n={len(pidx):3d}  0.05/tail: {n5:6,}  0.025/tail: {n25:6,}", flush=True)
    return dict(n=len(pidx), sig_005=n5, sig_0025=n25), tobs

# ---- (7) exclude 15 flagged registrations ----
audit = pd.read_csv(f"{LNM}/qc/phase2_full_report.csv")
flagged = set(audit.loc[audit["status"] != "PASS", "tag"])
drop = [tpos[t] for t in names if t in flagged]
print(f"flagged tags within cohort: {len(drop)}")
pidx = np.setdiff1d(np.arange(L), np.array(drop, dtype=int))
res_drop, _ = run_variant(F[pidx], pidx, "exclude flagged registrations")

# ---- (8a) AQ ceiling: drop 32 patients with AQ >= 93.8 ----
ceil = np.where(Y >= 93.8)[0]
print(f"AQ >= 93.8: {len(ceil)} patients")
pidx = np.setdiff1d(np.arange(L), ceil)
res_ceil, _ = run_variant(F[pidx], pidx, "drop AQ-ceiling patients")

# ---- (8b) Tobit (right-censored at 100) at 12 voxels, full cohort ----
gm_img = nib.load(f"{LNM}/template/gm_mask_2mm.nii.gz")
gmb = np.asanyarray(gm_img.dataobj) > 0
gmflat = np.where(gmb.ravel())[0]
t3d = np.asanyarray(nib.load(f"{LNM}/stats/obs_tmap.nii.gz").dataobj).astype(np.float64)
aff = gm_img.affine; inv = np.linalg.inv(aff)
refs = [(-34, -4, -8), (12, -8, -46), (-45, 25, 8), (45, 25, 8), (-50, -25, 5), (50, -25, 5),
        (-5, 50, 25), (5, 50, 25), (-20, -70, 40), (20, -70, 40)]
ref_pos = []
for xyz in refs:
    v = np.round((inv @ np.array([*xyz, 1.0]))[:3]).astype(int)
    vidx = np.ravel_multi_index(v, gmb.shape)
    pos = int(np.searchsorted(gmflat, vidx))
    if gmflat[pos] == vidx:
        ref_pos.append(pos)
peaks = [int(np.argmax(np.abs(t3d[gmb]) * (t3d[gmb] < 0))), int(np.argmax(t3d[gmb]))]
# use the actual primary peak voxels: strongest negative and strongest positive t
tg = t3d[gmb]
peaks = [int(np.argmin(tg)), int(np.argmax(tg))]
test_pos = sorted(set(ref_pos + peaks))

def tobit_loglik(par, y, X):
    beta, sigma = par[:-1], max(par[-1], 1e-6)
    mu = X @ beta
    unc = y < 100
    ll = 0.0
    if unc.any():
        ll += np.sum(stats.norm.logpdf(y[unc], mu[unc], sigma))
    if (~unc).any():  # censored at 100
        z = (100 - mu[~unc]) / sigma
        ll += np.sum(np.log(stats.norm.sf(z) + 1e-300))
    return -ll

tob = {}
for pos in test_pos:
    yv = np.minimum(Y, 100.0)  # observed; censoring handled in likelihood
    censored = (Y >= 99.999)
    if censored.sum() == 0:
        tob[pos] = "no censored obs"; continue
    Xd = np.column_stack([Xc, F[:, pos]])
    x0 = np.r_[np.linalg.lstsq(Xd, yv, rcond=None)[0], 20.0]
    try:
        res = optimize.minimize(tobit_loglik, x0, args=(yv, Xd), method="Nelder-Mead",
                                options=dict(maxiter=4000, xatol=1e-6, fatol=1e-6))
        b = res.x[:-1]
        tob[pos] = dict(beta_conn=round(float(b[-1]), 3),
                        n_censored=int(censored.sum()))
    except Exception as e:
        tob[pos] = f"fail {e}"
print(f"Tobit (right-censored at 100): {len(censored)} censored observations; "
      f"beta_conn at peaks/refs: {json.dumps({str(k): v for k, v in tob.items()})[:400]}")

# ---- (6) C2 full table + variance heterogeneity + LOO influence ----
H7 = np.load(f"{LNM}/stats/revision/H_yeo.npy")
inv_aff = inv
refs_mni = [(-34, -4, -8), (12, -8, -46), (-45, 25, 8), (45, 25, 8), (-50, -25, 5), (50, -25, 5),
            (-5, 50, 25), (5, 50, 25), (-20, -70, 40), (20, -70, 40)]
c2 = []
for xyz in refs_mni:
    v = np.round((inv_aff @ np.array([*xyz, 1.0]))[:3]).astype(int)
    vidx = np.ravel_multi_index(v, gmb.shape)
    pos = int(np.searchsorted(gmflat, vidx))
    if gmflat[pos] != vidx:
        c2.append(dict(mni=list(xyz), status="not_in_GM")); continue
    r, pv = stats.pearsonr(F[:, pos], H7)
    lo, hi = H7 <= np.quantile(H7, 1/3), H7 >= np.quantile(H7, 2/3)
    var_lo, var_hi = float(np.var(F[lo, pos])), float(np.var(F[hi, pos]))
    lev = float(stats.levene(F[lo, pos], F[hi, pos]).pvalue)
    c2.append(dict(mni=list(xyz), r_c_H=round(float(r), 3), p=float(f"{pv:.2e}"),
                   var_ratio_low_high=round(var_lo / max(var_hi, 1e-12), 2), levene_p=round(lev, 4)))
print("C2 full (signed):")
for cc in c2:
    print("  ", cc)

# LOO influence on the two peaks
pks = {"neg_peak": int(np.argmin(tg)), "pos_peak": int(np.argmax(tg))}
infl = {k: np.zeros(L) for k in pks}
Yr = residue(Y[:, None], Xc).ravel()
Fr = residue(F, Xc)
dof = L - Xc.shape[1] - 1
sd = Fr.std(0); sd[sd == 0] = 1.0
fz = (Fr - Fr.mean(0)) / sd; fz[~np.isfinite(fz)] = 0.0
yz = (Yr - Yr.mean()) / Yr.std()
for i in range(L):
    keep = np.delete(np.arange(L), i)
    yzk = (Yr[keep] - Yr[keep].mean()) / Yr[keep].std()
    for kname, pk in pks.items():
        fzk = fz[keep, pk] / max(sd[pk], 1e-12)
        rr = np.clip(yzk @ fzk / (len(keep) - 1), -1 + 1e-9, 1 - 1e-9)
        infl[kname][i] = rr * np.sqrt((len(keep) - 1) / (1 - rr ** 2))
base = {k: float(tg[pk]) for k, pk in pks.items()}
summary = {}
for k in pks:
    d = infl[k] - base[k]
    summary[k] = dict(base_t=round(base[k], 2),
                      max_abs_shift=round(float(np.abs(d).max()), 2),
                      worst_patient=str(names[int(np.argmax(np.abs(d)))]))
print("LOO influence on peaks:", summary)

json.dump(dict(exclude_flagged=res_drop, drop_ceiling=res_ceil,
               tobit={str(k): v for k, v in tob.items()},
               c2_full=c2, loo_influence=summary),
          open(f"{OUT}/t5_tail.json", "w"), indent=1)
print("T5_TAIL_DONE")
