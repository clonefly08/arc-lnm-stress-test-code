#!/usr/bin/env python3
"""P0-3a (revised): location-vs-connectivity dissociation test.

Per voxel:  AQ_i = b0 + b1*c_i(v) + b2*lesion_i(v) + covariates
Both regressors vary across patients. If b1 (connectivity) survives with b2
(voxel is inside the patient's own lesion) in the model, the association reflects
network connectivity beyond mere lesion location — a direct answer to the
"degree fingerprint" critique (F_mean itself cannot enter an across-patient
model: it is constant across patients at each voxel).

maxT 5000, Freedman-Lane. Output: stats/loc_conn_test_tmap.nii.gz, sig count.
"""
import os, time
os.environ.setdefault("OMP_NUM_THREADS", "4")
for v in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(v, "4")
import numpy as np
import pandas as pd
import nibabel as nib

BASE = "/mnt/n/ARC/lnm"
STATS = f"{BASE}/stats"
N_MAXT = 5000
SEED = 42


def main():
    t0 = time.time()
    summ = pd.read_csv(f"{BASE}/connmaps/gsp_sbc_summary.csv")
    pheno = pd.read_csv(f"{STATS}/phenotype.csv")
    names = [t for t in summ["lesion"] if t in set(pheno["tag"])]
    ph = pheno.set_index("tag").loc[names].reset_index()
    n = len(ph)

    ck = np.load(f"{BASE}/connmaps/.sbc_checkpoint.npz")
    keep = [t in set(pheno["tag"]) for t in summ["lesion"]]
    idx = [i for i, k in enumerate(keep) if k]
    F = (ck["sum"][idx] / ck["n"]).astype(np.float64)

    gm_img = nib.load(f"{BASE}/template/gm_mask_2mm.nii.gz")
    gm_bool = np.asanyarray(gm_img.dataobj) > 0
    gm_flat = np.where(gm_bool.ravel())[0]
    V = int(gm_bool.sum())
    assert F.shape == (n, V)

    L = np.zeros((n, V), dtype=np.float64)
    for i, t in enumerate(names):
        m = np.asanyarray(nib.load(f"{BASE}/masks_mni/{t}/mask_mni.nii.gz").dataobj) > 0
        L[i] = (m & gm_bool).ravel()[gm_flat]

    Y = ph["aq"].to_numpy(float)
    Xc = np.column_stack([np.ones(n), ph["age"], ph["sex"],
                          ph["log_days"], ph["vol_mm3"] / 1e5])

    def res(M):
        beta, *_ = np.linalg.lstsq(Xc, M, rcond=None)
        return M - Xc @ beta

    Yr = res(Y)
    Fr = res(F)
    Lr = res(L)
    dof = n - Xc.shape[1] - 2

    def tmap_c(y):
        """t of the connectivity regressor in the two-predictor per-voxel model."""
        yz = (y - y.mean()) / y.std()
        # partial out Lr from Fr, per voxel
        bL = (Fr * Lr).sum(0) / (Lr ** 2).sum(0)
        c_part = Fr - Lr * bL[None, :]
        sd = c_part.std(0); sd[sd == 0] = 1.0
        cz = (c_part - c_part.mean(0)) / sd
        cz[~np.isfinite(cz)] = 0.0
        r = (yz @ cz) / (n - 1)
        r = np.clip(r, -1 + 1e-9, 1 - 1e-9)
        return np.nan_to_num(r * np.sqrt(dof / (1 - r ** 2)),
                             nan=0.0, posinf=0.0, neginf=0.0)

    tobs = tmap_c(Yr)
    rng = np.random.default_rng(SEED)
    maxpos = np.empty(N_MAXT); maxneg = np.empty(N_MAXT)
    for p in range(N_MAXT):
        t = tmap_c(Yr[rng.permutation(n)])
        maxpos[p] = t.max(); maxneg[p] = t.min()
        if (p + 1) % 1000 == 0:
            print(f"perm {p+1}/{N_MAXT} elapsed={time.time()-t0:.0f}s", flush=True)
    fwe = np.ones(V, dtype=np.float32)
    pos = tobs > 0; neg = tobs < 0
    fwe[pos] = (1 + (maxpos[None, :] >= tobs[pos][:, None]).sum(1)) / (N_MAXT + 1)
    fwe[neg] = (1 + (maxneg[None, :] <= tobs[neg][:, None]).sum(1)) / (N_MAXT + 1)
    np.save(f"{STATS}/loc_conn_fwe_p.npy", fwe)
    out = np.zeros(gm_bool.shape, dtype=np.float32)
    out.ravel()[np.where(gm_bool.ravel())[0]] = tobs.astype(np.float32)
    nib.save(nib.Nifti1Image(out, gm_img.affine), f"{STATS}/loc_conn_test_tmap.nii.gz")
    n_sig = int((fwe < 0.05).sum())
    # overlap with the main LNM network
    fwe_main = np.load(f"{STATS}/fwe_p_maxT.npy")
    both = int(((fwe < 0.05) & (fwe_main < 0.05)).sum())
    print(f"location-vs-connectivity: FWE p<0.05 voxels = {n_sig}; "
          f"overlapping main LNM network = {both} (baseline 2469) "
          f"elapsed={time.time()-t0:.0f}s", flush=True)
    with open(f"{STATS}/loc_conn_sigcount.txt", "w") as f:
        f.write(f"{n_sig}\t{both}")
    print("LOC_CONN_DONE", flush=True)


if __name__ == "__main__":
    main()
