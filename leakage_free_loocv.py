#!/usr/bin/env python3
"""Leakage-free LOOCV: within EVERY training fold, reselect significant voxels,
recompute T weights, and refit the prediction model.

Fold-level significance uses an uncorrected two-sided |t| threshold corresponding to
p < 0.001 (stated explicitly in the manuscript), because per-fold permutation FWE
across 226 folds is not feasible.

Scores (all recomputed per fold):
  d  connectivity-weighted damage  = sum_{v in M+} T(v) c_i(v) / sum T(v)      (T>0)
  o  lesion-overlap               = mean_{v in S_i ∩ M+} T(v)
  n  negative-lobe damage         = sum_{v in M-} |T(v)| c_i(v) / sum |T(v)|    (T<0)
where M+/M- are fold-specific selected voxel sets.

Split-half uses the same per-half selection.
Bootstrap CIs for R, MAE and delta-R.
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
OUT = f"{BASE}/stats/leakage_free"
os.makedirs(OUT, exist_ok=True)

T_THRESH = stats.t.isf(0.0005, 200)   # two-sided p=0.001, df~220 -> |t| ~ 3.33
print("fold-level |t| threshold:", round(T_THRESH, 3))

summ = pd.read_csv(f"{BASE}/connmaps/gsp_sbc_summary.csv")
pheno = pd.read_csv(f"{STATS}/phenotype.csv")
names = [t for t in summ["lesion"] if t in set(pheno["tag"])]
ph = pheno.set_index("tag").loc[names].reset_index()
n = len(ph)

ck = np.load(f"{BASE}/connmaps/.sbc_checkpoint.npz")
keep = [t in set(pheno["tag"]) for t in summ["lesion"]]
idx = [i for i, k in enumerate(keep) if k]
F = (ck["sum"][idx] / ck["n"]).astype(np.float64)      # n x V

gm = nib.load(f"{BASE}/template/gm_mask_2mm.nii.gz")
gmb = np.asanyarray(gm.dataobj) > 0
V = int(gmb.sum())
assert F.shape == (n, V)

Y = ph["aq"].to_numpy(float)
Xc = np.column_stack([np.ones(n), ph["age"], ph["sex"],
                      ph["log_days"], ph["vol_mm3"] / 1e5])

# per-patient lesion voxel indices (into GM flat space)
seed = []
for t in names:
    m = np.asanyarray(nib.load(f"{BASE}/masks_mni/{t}/mask_mni.nii.gz").dataobj) > 0
    seed.append(np.where((m & gmb).ravel()[np.where(gmb.ravel())[0]])[0])

def residue(M, X):
    beta, *_ = np.linalg.lstsq(X, M, rcond=None)
    return M - X @ beta

def tmap(Fsub, Ysub, Xsub):
    Yr = residue(Ysub[:, None], Xsub).ravel()
    Fr = residue(Fsub, Xsub)
    yz = (Yr - Yr.mean()) / Yr.std()
    sd = Fr.std(0); sd[sd == 0] = 1.0
    fz = (Fr - Fr.mean(0)) / sd
    fz[~np.isfinite(fz)] = 0.0
    nn = len(Ysub)
    r = np.clip(yz @ fz / (nn - 1), -1 + 1e-9, 1 - 1e-9)
    dof = nn - Xsub.shape[1] - 1
    return np.nan_to_num(r * np.sqrt(dof / (1 - r ** 2)))

def scores_for(train_idx, test_i, t):
    """Compute the three scores for patient test_i using fold-specific maps."""
    pos = t > T_THRESH
    neg = t < -T_THRESH
    out = {}
    if pos.any():
        w = t[pos] / t[pos].sum()
        out["d"] = float(w @ F[test_i][pos])
        out["o"] = float(t[pos][np.isin(np.where(pos)[0], seed[test_i])].mean()) if np.isin(np.where(pos)[0], seed[test_i]).any() else 0.0
    else:
        out["d"] = out["o"] = 0.0
    if neg.any():
        w = (-t[neg]) / (-t[neg]).sum()
        out["n"] = float(w @ F[test_i][neg])
    else:
        out["n"] = 0.0
    return out, int(pos.sum()), int(neg.sum())

preds = {k: np.full(n, np.nan) for k in ("d", "o", "n")}
nsel = []
for i in range(n):
    tr = np.delete(np.arange(n), i)
    t = tmap(F[tr], Y[tr], Xc[tr])
    sc, npos, nneg = scores_for(tr, i, t)
    nsel.append((npos, nneg))
    # training scores (same fold map)
    tr_scores = {k: np.empty(len(tr)) for k in ("d", "o", "n")}
    for j, jj in enumerate(tr):
        s2, _, _ = scores_for(tr, jj, t)
        for k in tr_scores:
            tr_scores[k][j] = s2[k]
    for k in ("d", "o", "n"):
        D = np.column_stack([Xc[tr], tr_scores[k]])
        beta, *_ = np.linalg.lstsq(D, Y[tr], rcond=None)
        preds[k][i] = np.array([*Xc[i], sc[k]]) @ beta
    if (i + 1) % 25 == 0:
        print(f"  fold {i+1}/{n}", flush=True)

def stats_of(y, p):
    m = np.isfinite(p)
    r = float(np.corrcoef(y[m], p[m])[0, 1])
    mae = float(np.abs(y[m] - p[m]).mean())
    r2p = float(1 - ((y[m] - p[m]) ** 2).sum() / ((y[m] - y[m].mean()) ** 2).sum())
    return round(r, 3), round(r2p, 3), round(mae, 2)

res = {}
for k, lab in (("d", "conn_weighted"), ("o", "lesion_overlap"), ("n", "neg_lobe")):
    r, r2, mae = stats_of(Y, preds[k])
    res[lab] = dict(R=r, R2_press=r2, MAE=mae)
    print(f"{lab}: R={r} R2press={r2} MAE={mae}")

# covariates-only baseline
predc = np.full(n, np.nan)
for i in range(n):
    tr = np.delete(np.arange(n), i)
    beta, *_ = np.linalg.lstsq(Xc[tr], Y[tr], rcond=None)
    predc[i] = Xc[i] @ beta
res["covariates"] = dict(zip(("R", "R2_press", "MAE"), stats_of(Y, predc)))

# bootstrap CI for R and delta R
rng = np.random.default_rng(42)
B = 5000
rs = {k: np.empty(B) for k in ("d", "o")}
deltas = np.empty(B)
for b in range(B):
    ii = rng.integers(0, n, n)
    rs["d"][b] = np.corrcoef(Y[ii], preds["d"][ii])[0, 1]
    rs["o"][b] = np.corrcoef(Y[ii], preds["o"][ii])[0, 1]
    deltas[b] = rs["d"][b] - rs["o"][b]
res["R_conn_CI"] = [round(float(np.percentile(rs["d"], 2.5)), 3), round(float(np.percentile(rs["d"], 97.5)), 3)]
res["deltaR"] = round(res["conn_weighted"]["R"] - res["lesion_overlap"]["R"], 3)
res["deltaR_CI"] = [round(float(np.percentile(deltas, 2.5)), 3), round(float(np.percentile(deltas, 97.5)), 3)]
res["deltaR_p"] = round(float(2 * min((deltas <= 0).mean(), (deltas >= 0).mean())), 4)
res["n_selected_mean"] = [int(np.mean([x[0] for x in nsel])), int(np.mean([x[1] for x in nsel]))]
res["threshold_t"] = round(float(T_THRESH), 3)

# split-half with per-half selection
half_rs = []
for s in range(20):
    r2_ = np.random.default_rng(1000 + s)
    perm = r2_.permutation(n)
    A, Bx = perm[:n // 2], perm[n // 2:]
    for src, dst in ((A, Bx), (Bx, A)):
        t = tmap(F[src], Y[src], Xc[src])
        pos = t > T_THRESH
        if not pos.any():
            continue
        scA = F[src][:, pos] @ (t[pos] / t[pos].sum())
        scB = F[dst][:, pos] @ (t[pos] / t[pos].sum())
        D = np.column_stack([Xc[src], scA])
        beta, *_ = np.linalg.lstsq(D, Y[src], rcond=None)
        pb = np.column_stack([Xc[dst], scB]) @ beta
        half_rs.append(np.corrcoef(Y[dst], pb)[0, 1])
res["splithalf_R_mean"] = round(float(np.mean(half_rs)), 3)
res["splithalf_R_sd"] = round(float(np.std(half_rs)), 3)
res["splithalf_R_min"] = round(float(np.min(half_rs)), 3)
print("splithalf:", res["splithalf_R_mean"], "+-", res["splithalf_R_sd"])

np.savez(f"{OUT}/predictions.npz", **{k: v for k, v in preds.items()}, cov=predc)
json.dump(res, open(f"{OUT}/leakage_free_results.json", "w"), indent=1)
print(json.dumps(res, indent=1))
print("LEAKAGE_FREE_DONE")
