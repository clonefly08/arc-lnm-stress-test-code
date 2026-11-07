#!/usr/bin/env python3
"""T2 (D-b power calibration): is the low-entropy tertile's null result a power artefact?
- 5 stratified calibration subsets (n=76): 1,000 permutations each -> per-tail 95th-pct
  thresholds; require range < 0.2, then t_crit = mean of the two tail thresholds
  (extend to 20 subsets if spread >= 0.2).
- 200 random n=76 subsets: t-maps only, count |t| > t_crit.
- low-entropy (Yeo-7 H <= 1.36, n=76) counted with the SAME t_crit.
- percentile of the low-entropy count within the 200-draw null.
Run in WSL: python3 /mnt/n/ARC/analysis/t2_power_calib.py"""
import os, json
os.environ.setdefault("OMP_NUM_THREADS", "4")
import numpy as np, pandas as pd

BASE = "/mnt/n/ARC"; LNM = f"{BASE}/lnm"; OUT = f"{BASE}/analysis/out"
SEED = 42
N_PERM_CAL = 1000
N_DRAWS = 200
K = 76
rng = np.random.default_rng(SEED)

summ = pd.read_csv(f"{LNM}/connmaps/gsp_sbc_summary.csv")
ph0 = pd.read_csv(f"{LNM}/stats/phenotype.csv")
names = [t for t in summ["lesion"] if t in set(ph0["tag"])]
ph = ph0.set_index("tag").loc[names].reset_index()
n = len(ph)
Y = ph["aq"].to_numpy(float)
Xc = np.column_stack([np.ones(n), ph["age"], ph["sex"], ph["log_days"], ph["vol_mm3"] / 1e5])
sex = ph["sex"].to_numpy()
vol = ph["vol_mm3"].to_numpy(float)

ck = np.load(f"{LNM}/connmaps/.sbc_checkpoint.npz")
keep = [t in set(ph0["tag"]) for t in summ["lesion"]]
idx = [i for i, k in enumerate(keep) if k]
F = (ck["sum"][idx] / ck["n"]).astype(np.float64)
print(f"n={n}  F={F.shape}")

H7 = np.load(f"{LNM}/stats/revision/H_yeo.npy")
thr_lo = float(np.quantile(H7, 1 / 3))
sel_low = np.where(H7 <= thr_lo)[0]
assert len(sel_low) == K, f"low-entropy tertile size {len(sel_low)} != {K}"
print(f"low-entropy tertile: H <= {thr_lo:.3f}, n={K}")

def residue(M, X):
    beta, *_ = np.linalg.lstsq(X, M, rcond=None)
    return M - X @ beta

def tmap_and_null(sel):
    Yr = residue(Y[sel, None], Xc[sel]).ravel()
    Fr = residue(F[sel], Xc[sel])
    dof = len(sel) - Xc.shape[1] - 1
    yz = (Yr - Yr.mean()) / Yr.std()
    sd = Fr.std(0); sd[sd == 0] = 1.0
    fz = (Fr - Fr.mean(0)) / sd
    fz[~np.isfinite(fz)] = 0.0
    r = np.clip(yz @ fz / (len(sel) - 1), -1 + 1e-9, 1 - 1e-9)
    tobs = np.nan_to_num(r * np.sqrt(dof / (1 - r ** 2)))
    return tobs, yz, fz, dof

def perm_thresholds(yz, fz, dof, nperm, rrng):
    maxpos = np.empty(nperm); maxneg = np.empty(nperm)
    for k in range(nperm):
        perm = rrng.permutation(K)
        yzp = (yz[perm] - yz[perm].mean()) / yz[perm].std()
        rr = np.clip(yzp @ fz / (K - 1), -1 + 1e-9, 1 - 1e-9)
        tk = np.nan_to_num(rr * np.sqrt(dof / (1 - rr ** 2)))
        maxpos[k] = tk.max(); maxneg[k] = tk.min()
    tp = float(np.sort(maxpos)[::-1][int(np.ceil(0.05 * nperm)) - 1])
    tn = float(-np.sort(maxneg)[int(np.ceil(0.05 * nperm)) - 1])
    return tp, tn

# ---- calibration: 5 stratified subsets (volume tertile x sex) ----
vter = pd.qcut(vol, 3, labels=False)
strata = [np.where((vter == v) & (sex == s))[0] for v in (0, 1, 2) for s in np.unique(sex)]
def stratified_draw():
    sel = []
    for st in strata:
        k = round(K * len(st) / n)
        sel.extend(rng.choice(st, size=k, replace=False))
    sel = np.array(sorted(sel))
    while len(sel) != K:
        if len(sel) > K:
            sel = np.delete(sel, rng.integers(len(sel)))
        else:
            cand = np.setdiff1d(np.arange(n), sel)
            sel = np.append(sel, rng.choice(cand))
    return np.sort(sel)

crits = []
for c in range(5):
    sel = stratified_draw()
    _, yz, fz, dof = tmap_and_null(sel)
    tp, tn = perm_thresholds(yz, fz, dof, N_PERM_CAL, rng)
    crits.append([tp, tn])
    print(f"calib {c+1}: t_pos {tp:.3f}  t_neg {tn:.3f}", flush=True)

crits = np.array(crits)
spread = max(crits[:, 0].max() - crits[:, 0].min(), crits[:, 1].max() - crits[:, 1].min())
print(f"threshold spread: {spread:.3f}")
if spread >= 0.2:
    print("spread >= 0.2 -> extend calibration to 20 subsets")
    for c in range(5, 20):
        sel = stratified_draw()
        _, yz, fz, dof = tmap_and_null(sel)
        tp, tn = perm_thresholds(yz, fz, dof, N_PERM_CAL, rng)
        crits = np.vstack([crits, [[tp, tn]]])
        print(f"calib {c+1}: t_pos {tp:.3f}  t_neg {tn:.3f}", flush=True)
t_crit = float(np.mean(np.abs(crits)))
print(f"t_crit (unified |t|) = {t_crit:.3f}")

# ---- 200 random draws ----
counts = np.empty(N_DRAWS, dtype=int); maxabs = np.empty(N_DRAWS)
for d in range(N_DRAWS):
    sel = np.sort(rng.choice(n, size=K, replace=False))
    tk, _, _, _ = tmap_and_null(sel)
    counts[d] = int((np.abs(tk) > t_crit).sum())
    maxabs[d] = float(np.abs(tk).max())
print(f"random subsets: count median {np.median(counts):.0f} "
      f"(IQR {np.percentile(counts,25):.0f}-{np.percentile(counts,75):.0f}), max {counts.max()}")

# ---- low-entropy tertile, same threshold ----
tk_low, _, _, _ = tmap_and_null(sel_low)
count_low = int((np.abs(tk_low) > t_crit).sum())
pct = float((counts < count_low).mean() * 100)
print(f"low-entropy tertile: count = {count_low}  (|t| max {np.abs(tk_low).max():.2f})")
print(f"percentile within 200 random draws = {pct:.1f}%")

json.dump(dict(t_crit=round(t_crit, 3), threshold_spread=round(float(spread), 3),
               n_crit_subsets=int(len(crits)),
               random_count_median=float(np.median(counts)),
               random_count_iqr=[float(np.percentile(counts, 25)), float(np.percentile(counts, 75))],
               random_count_max=int(counts.max()),
               low_entropy_count=count_low,
               low_entropy_abs_t_max=round(float(np.abs(tk_low).max()), 3),
               low_entropy_percentile=round(pct, 1)),
          open(f"{OUT}/t2_summary.json", "w"), indent=1)
pd.DataFrame({"draw": np.arange(N_DRAWS), "count_suprathreshold": counts,
              "max_abs_t": maxabs}).to_csv(f"{OUT}/t2_null_dist.csv", index=False)
print("T2_DONE")
