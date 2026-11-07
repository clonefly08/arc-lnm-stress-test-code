#!/usr/bin/env python3
"""P0-2: classic VLSM head-to-head (same covariates, same permutation framework).

Per tested voxel v (lesion prevalence >=5%):
    AQ_i = b0 + b1*lesion_i(v) + b2*vol + b3*age + b4*sex + b5*log_days + e
Freedman-Lane 5000 (max-T) + TFCE 1000, identical to the LNM analysis.
Head-to-head: LOOCV lesion-overlap score R for VLSM vs LNM (0.668).

Outputs: stats/vlsm_tmap.nii.gz, vlsm_fwe_p_maxT.npy, vlsm_fwe_p_tfce.nii.gz,
         vlsm_summary.csv
"""
import os, sys, time
os.environ.setdefault("OMP_NUM_THREADS", "4")
for v in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(v, "4")
import numpy as np
import pandas as pd
import nibabel as nib
from scipy import stats, ndimage

sys.path.insert(0, "/mnt/n/ARC/lnm/scripts")
from phase4_regression import residualize, tfce_score

BASE = "/mnt/n/ARC/lnm"
STATS = f"{BASE}/stats"
N_MAXT, N_TFCE = 5000, 1000
E, H = 0.5, 2.0
MIN_PREV = 0.05
SEED = 42


def tmap_lr(Lr, Yr, n, dof):
    sd = Lr.std(0); sd[sd == 0] = 1.0
    lz = (Lr - Lr.mean(0)) / sd
    lz[~np.isfinite(lz)] = 0.0
    yz = (Yr - Yr.mean()) / Yr.std()
    r = (yz @ lz) / (n - 1)
    r = np.clip(r, -1 + 1e-9, 1 - 1e-9)
    return np.nan_to_num(r * np.sqrt(dof / (1 - r ** 2)), nan=0.0, posinf=0.0, neginf=0.0)


def main():
    t00 = time.time()
    summ = pd.read_csv(f"{BASE}/connmaps/gsp_sbc_summary.csv")
    pheno = pd.read_csv(f"{STATS}/phenotype.csv")
    names = [t for t in summ["lesion"] if t in set(pheno["tag"])]
    ph = pheno.set_index("tag").loc[names].reset_index()
    n = len(ph)

    gm_img = nib.load(f"{BASE}/template/gm_mask_2mm.nii.gz")
    gm_bool = np.asanyarray(gm_img.dataobj) > 0
    gm_flat = np.where(gm_bool.ravel())[0]
    V = len(gm_flat)

    L = np.zeros((n, V), dtype=np.float32)
    for i, t in enumerate(names):
        m = np.asanyarray(nib.load(f"{BASE}/masks_mni/{t}/mask_mni.nii.gz").dataobj) > 0
        L[i] = (m & gm_bool).ravel()[gm_flat]
    prev = L.mean(0)
    tested = prev >= MIN_PREV
    print(f"tested voxels (prevalence>={MIN_PREV:.0%}): {int(tested.sum())}", flush=True)

    Y = ph["aq"].to_numpy(float)
    Xc = np.column_stack([np.ones(n), ph["age"], ph["sex"],
                          ph["log_days"], ph["vol_mm3"] / 1e5])
    P = np.eye(n) - Xc @ np.linalg.pinv(Xc)
    Yr = P @ Y
    Lt = L[:, tested].astype(np.float64)
    Lr = P @ Lt
    dof = n - Xc.shape[1] - 1

    tobs = tmap_lr(Lr, Yr, n, dof)
    rng = np.random.default_rng(SEED)
    maxpos = np.empty(N_MAXT); maxneg = np.empty(N_MAXT)
    for p in range(N_MAXT):
        t = tmap_lr(Lr[rng.permutation(n)], Yr, n, dof)
        maxpos[p] = t.max(); maxneg[p] = t.min()
        if (p + 1) % 1000 == 0:
            print(f"maxT {p+1}/{N_MAXT} elapsed={time.time()-t00:.0f}s", flush=True)
    fwe = np.ones(int(tested.sum()), dtype=np.float32)
    pos = tobs > 0; neg = tobs < 0
    fwe[pos] = (1 + (maxpos[None, :] >= tobs[pos][:, None]).sum(1)) / (N_MAXT + 1)
    fwe[neg] = (1 + (maxneg[None, :] <= tobs[neg][:, None]).sum(1)) / (N_MAXT + 1)
    np.save(f"{STATS}/vlsm_fwe_p_maxT.npy", fwe)
    print(f"VLSM maxT FWE p<0.05 voxels: {int((fwe < 0.05).sum())} "
          f"(of {int(tested.sum())} tested)", flush=True)

    # TFCE on the full-FOV map (untested voxels = 0)
    t3 = np.zeros(gm_bool.shape, dtype=np.float64)
    t3.ravel()[gm_flat[tested]] = tobs
    s_obs = tfce_score(t3, pos=True) + tfce_score(t3, pos=False)
    s_gm = s_obs.ravel()[gm_flat[tested]]
    tfce_max = np.empty(N_TFCE)
    for p in range(N_TFCE):
        t = tmap_lr(Lr[rng.permutation(n)], Yr, n, dof)
        t3 = np.zeros(gm_bool.shape, dtype=np.float64)
        t3.ravel()[gm_flat[tested]] = t
        s = tfce_score(t3, pos=True) + tfce_score(t3, pos=False)
        tfce_max[p] = s.max()
        if (p + 1) % 100 == 0:
            print(f"TFCE {p+1}/{N_TFCE} elapsed={time.time()-t00:.0f}s", flush=True)
    np.save(f"{STATS}/vlsm_tfce_max_perms.npy", tfce_max)
    fwe_t = np.empty(int(tested.sum()), dtype=np.float32)
    CH = 20000
    for s0 in range(0, len(s_gm), CH):
        e0 = min(s0 + CH, len(s_gm))
        fwe_t[s0:e0] = (1 + (s_gm[s0:e0, None] >= tfce_max[None, :]).sum(1)) / (N_TFCE + 1)
    np.save(f"{STATS}/vlsm_fwe_p_tfce.npy", fwe_t)
    print(f"VLSM TFCE FWE p<0.05 voxels: {int((fwe_t < 0.05).sum())}", flush=True)

    tmap_full = np.zeros(gm_bool.shape, dtype=np.float32)
    tmap_full.ravel()[gm_flat[tested]] = tobs.astype(np.float32)
    nib.save(nib.Nifti1Image(tmap_full, gm_img.affine), f"{STATS}/vlsm_tmap.nii.gz")

    # head-to-head LOOCV (VLSM 版 lesion-overlap score) 移至 phase5b 单独跑；
    # 此处只落盘图级比较
    pd.DataFrame(dict(
        metric=["sig_voxels_maxT", "sig_voxels_tfce", "tested_voxels"],
        value=[int((fwe < 0.05).sum()), int((fwe_t < 0.05).sum()), int(tested.sum())],
    )).to_csv(f"{STATS}/vlsm_summary.csv", index=False)
    print(f"VLSM_DONE elapsed={time.time()-t00:.0f}s", flush=True)


if __name__ == "__main__":
    main()
