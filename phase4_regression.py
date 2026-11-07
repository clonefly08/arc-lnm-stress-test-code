#!/usr/bin/env python3
"""Phase 4: voxelwise behavioural regression AQ ~ c_i(v) + covariates (LNM primary analysis).

Model (per voxel v, across patients i=1..n):
    AQ_i = b0 + b1*c_i(v) + b2*lesion_vol_i + b3*age_i + b4*sex_i + b5*log10(wab_days_i) + e_i
c_i(v) = mean Fisher-z connectivity between lesion i and voxel v across 500 normative
subjects (from the SBC checkpoint acc_sum/acc_n).

Inference: Freedman-Lane permutation (residuals of the covariate-only reduced model),
- max-T FWE over 5000 permutations (voxelwise),
- TFCE (E=0.5, H=2) FWE over 1000 permutations.
FSL randomise cannot take voxelwise-varying regressors, hence this implementation.

Degeneracy check (van den Heuvel 2026 critique): distribution of corr(c_i, mean-map)
and across-patient variance — if c_i collapses to the connectome degree fingerprint
the regression is uninformative by construction.

Outputs under /mnt/n/ARC/lnm/stats/:
  phenotype.csv, obs_tmap.nii.gz, obs_zmap_fwe_maxT.nii.gz, fwe_p_tfce.nii.gz,
  degeneracy.csv, permutations.log
"""
import os, time
os.environ.setdefault("OMP_NUM_THREADS", "4")
for v in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(v, "4")
import numpy as np
import pandas as pd
import nibabel as nib
from scipy import ndimage, stats

BASE = "/mnt/n/ARC/lnm"
DS = "/mnt/e/download/ds004884-download"
STATS = f"{BASE}/stats"
os.makedirs(STATS, exist_ok=True)

N_MAXT = 5000
N_TFCE = 1000
E, H = 0.5, 2.0
SEED = 42


def load_phenotype():
    summ = pd.read_csv(f"{BASE}/connmaps/gsp_sbc_summary.csv")
    pheno = pd.read_csv(f"{DS}/participants.tsv", sep="\t")
    audit = pd.read_csv(f"{BASE}/qc/phase0_mask_audit.csv")
    vol = {r["sub"]: r["vol_mm3"] for _, r in audit.iterrows()}
    rows = []
    for tag in summ["lesion"]:
        sub = tag.split("_ses-")[0]
        p = pheno[pheno["participant_id"] == sub]
        if p.empty:
            print(f"  no WAB for {sub}")
            continue
        p = p.iloc[0]
        rows.append(dict(tag=tag, sub=sub, aq=float(p["wab_aq"]),
                         age=float(p["age_at_stroke"]),
                         sex=1.0 if str(p["sex"]).strip() == "M" else 0.0,
                         log_days=np.log10(max(float(p["wab_days"]), 1.0)),
                         vol_mm3=float(vol.get(sub, np.nan))))
    df = pd.DataFrame(rows)
    df.to_csv(f"{STATS}/phenotype.csv", index=False)
    print(f"phenotype: {len(df)} patients (AQ complete for all)", flush=True)
    return df


def residualize(M, Xc):
    """M: (n x k) -> residuals of each column on Xc."""
    beta, *_ = np.linalg.lstsq(Xc, M, rcond=None)
    return M - Xc @ beta


def tfce_score(x3d, pos=True):
    """TFCE score map (E=0.5, H=2), 100 thresholds, scipy label per threshold."""
    X = x3d if pos else -x3d
    X = np.where(np.isfinite(X), X, 0.0)
    m = X[X > 0]
    if m.size == 0:
        return np.zeros_like(X)
    hmax = m.max()
    thresholds = np.linspace(hmax * 0.001, hmax, 100)
    score = np.zeros_like(X, dtype=np.float32)
    dh = thresholds[1] - thresholds[0] if len(thresholds) > 1 else hmax
    struct = np.ones((3, 3, 3), dtype=bool)
    for h in thresholds:
        mask = X > h
        if not mask.any():
            continue
        lab, n = ndimage.label(mask, structure=struct)
        sizes = np.bincount(lab.ravel())
        s = sizes[lab]
        score += (dh * (h ** E) * (s.astype(np.float32) ** H))
    return score


def main():
    t00 = time.time()
    pheno = load_phenotype()
    TAG = os.environ.get("P4_TAG", "")
    STAG = os.environ.get("P4_SUM_TAG", TAG)
    summ = pd.read_csv(f"{BASE}/connmaps/gsp_sbc_summary{STAG}.csv")
    names = summ["lesion"].tolist()
    drop = {s.strip() for s in os.environ.get("P4_DROP", "").split(",") if s.strip()}
    # keep/idx MUST align with the full summary order (= checkpoint rows); drop AFTER
    keep = [t in set(pheno["tag"]) for t in names]
    names_k = [t for t, k in zip(names, keep) if k]
    idx = [i for i, k in enumerate(keep) if k]
    if drop:
        sel = [j for j, t in enumerate(names_k) if t not in drop]
        names_k = [names_k[j] for j in sel]
        idx = [idx[j] for j in sel]
    pheno = pheno.set_index("tag").loc[names_k].reset_index()
    n = len(pheno)
    print(f"analysis n={n}", flush=True)

    ck = np.load(f"{BASE}/connmaps/.sbc_checkpoint{STAG}.npz")
    F = (ck["sum"][idx] / ck["n"]).astype(np.float64)
    if os.environ.get("P4_DEMEAN") == "1":  # partial-degeneracy: patient-specific deviation only
        F = F - F.mean(axis=0, keepdims=True)     # n x V mean Fisher-z
    print(f"F matrix: {F.shape}", flush=True)

    Y = pheno["aq"].to_numpy(dtype=np.float64)
    vol_term = (np.log10(pheno["vol_mm3"]) if os.environ.get("P4_VOLLOG") == "1"
                else pheno["vol_mm3"] / 1e5)
    Xc = np.column_stack([np.ones(n), pheno["age"], pheno["sex"],
                          pheno["log_days"], vol_term])
    Xc_names = ["intercept", "age", "sex", "log_days", "vol"]

    # degeneracy diagnostics (2026 critique)
    F_mean = F.mean(axis=0)
    deg = pd.DataFrame({
        "tag": names_k,
        "corr_with_groupmean": [np.corrcoef(F[i], F_mean)[0, 1] for i in range(n)],
        "c_variance_across_patients": F.std(axis=0).mean(),
    })
    deg.to_csv(f"{STATS}/degeneracy.csv", index=False)
    cc = np.array([np.corrcoef(F[i], F_mean)[0, 1] for i in range(n)])
    print(f"degeneracy: corr(c_i, group mean) median={np.median(cc):.3f} "
          f"IQR=({np.percentile(cc,25):.3f},{np.percentile(cc,75):.3f})", flush=True)
    print(f"across-patient SD of c_i(v): mean={F.std(axis=0).mean():.4f}", flush=True)

    # residualize on covariates
    Yr = residualize(Y[:, None], Xc).ravel()
    Fr = residualize(F, Xc)
    dof = n - Xc.shape[1] - 1

    def tmap(y):
        yz = (y - y.mean()) / y.std()
        sd = Fr.std(0)
        sd[sd == 0] = 1.0                    # finalmask 外体素：500 人全为 0 -> t=0 中性
        fz = (Fr - Fr.mean(0)) / sd
        fz[~np.isfinite(fz)] = 0.0
        r = (yz @ fz) / (n - 1)
        r = np.clip(r, -1 + 1e-9, 1 - 1e-9)
        return np.nan_to_num(r * np.sqrt(dof / (1 - r ** 2)),
                             nan=0.0, posinf=0.0, neginf=0.0)

    gm_img = nib.load(os.environ.get("P4_GM") or f"{BASE}/template/gm_mask_2mm.nii.gz")
    gm_bool = np.asanyarray(gm_img.dataobj) > 0
    shape3 = gm_bool.shape
    flat_idx = np.where(gm_bool.ravel())[0]
    V = int(gm_bool.sum())
    assert Fr.shape[1] == V

    tobs = tmap(Yr)
    out = np.zeros(shape3, dtype=np.float32)
    out.ravel()[flat_idx] = tobs.astype(np.float32)
    nib.save(nib.Nifti1Image(out, gm_img.affine), f"{STATS}/obs_tmap{TAG}.nii.gz")
    print(f"observed t map saved; range=({tobs.min():.2f},{tobs.max():.2f}) "
          f"elapsed={time.time()-t00:.0f}s", flush=True)

    rng = np.random.default_rng(SEED)

    # ---- max-T FWE (5000) ---- (skipped if already computed by an earlier run)
    if os.path.exists(f"{STATS}/fwe_p_maxT{TAG}.npy"):
        print("maxT FWE already computed; loading", flush=True)
    else:
        maxpos = np.empty(N_MAXT); maxneg = np.empty(N_MAXT)
        for p in range(N_MAXT):
            perm = rng.permutation(n)
            yp = Yr[perm]
            t = tmap(yp)
            maxpos[p] = t.max(); maxneg[p] = t.min()
            if (p + 1) % 500 == 0:
                print(f"maxT perm {p+1}/{N_MAXT} elapsed={time.time()-t00:.0f}s", flush=True)
        fwe_p = np.ones(V, dtype=np.float32)
        fwe_p[tobs > 0] = (1 + np.sum(maxpos[None, :] >= tobs[tobs > 0][:, None], axis=1)) / (N_MAXT + 1)
        fwe_p[tobs < 0] = (1 + np.sum(maxneg[None, :] <= tobs[tobs < 0][:, None], axis=1)) / (N_MAXT + 1)
        np.save(f"{STATS}/fwe_p_maxT{TAG}.npy", fwe_p)
        z_fwe = stats.norm.isf(np.clip(fwe_p, 1e-12, 1)).astype(np.float32)
        zfwe_map = np.zeros(shape3, dtype=np.float32)
        zfwe_map.ravel()[flat_idx] = np.where(tobs > 0, z_fwe, -z_fwe)
        nib.save(nib.Nifti1Image(zfwe_map, gm_img.affine), f"{STATS}/obs_zmap_fwe_maxT{TAG}.nii.gz")
        print(f"maxT FWE p<0.05 voxels: {int((fwe_p < 0.05).sum())}", flush=True)
    fwe_p = np.load(f"{STATS}/fwe_p_maxT{TAG}.npy")
    nSig = int((fwe_p < 0.05).sum())
    print(f"maxT FWE p<0.05 voxels: {nSig}", flush=True)

    # ---- TFCE FWE (1000) ---- [deprecated for this data; set P4_TFCE=1 to run]
    if os.environ.get("P4_TFCE", "0") != "1":
        print("TFCE skipped (deprecated)", flush=True)
        print(f"PHASE4_DONE total elapsed={time.time()-t00:.0f}s", flush=True)
        return
    tobs3 = np.zeros(shape3, dtype=np.float64)
    tobs3.ravel()[flat_idx] = tobs
    s_obs = tfce_score(tobs3, pos=True) + tfce_score(tobs3, pos=False)
    s_obs_gm = s_obs.ravel()[flat_idx]                      # V_gm
    tfce_max = np.empty(N_TFCE)
    for p in range(N_TFCE):
        perm = rng.permutation(n)
        t = tmap(Yr[perm])
        t3 = np.zeros(shape3, dtype=np.float64)
        t3.ravel()[flat_idx] = t
        s = tfce_score(t3, pos=True) + tfce_score(t3, pos=False)
        tfce_max[p] = s.max()
        if (p + 1) % 100 == 0:
            print(f"TFCE perm {p+1}/{N_TFCE} elapsed={time.time()-t00:.0f}s", flush=True)
    np.save(f"{STATS}/tfce_max_perms.npy", tfce_max)
    # 比较限 GM 体素；分块防内存峰值
    fwe_tfce_gm = np.empty(V, dtype=np.float32)
    CH = 20000
    for s0 in range(0, V, CH):
        e0 = min(s0 + CH, V)
        fwe_tfce_gm[s0:e0] = (1 + (tfce_max[None, :] >= s_obs_gm[s0:e0, None]).sum(axis=1)) / (N_TFCE + 1)
    fwe_tfce = np.ones(shape3, dtype=np.float32)
    fwe_tfce.ravel()[flat_idx] = fwe_tfce_gm
    nib.save(nib.Nifti1Image(fwe_tfce, gm_img.affine), f"{STATS}/fwe_p_tfce.nii.gz")
    print(f"TFCE FWE p<0.05 voxels: {int((fwe_tfce_gm < 0.05).sum())}", flush=True)

    # covariate stability (observed)
    beta_cov, *_ = np.linalg.lstsq(np.column_stack([Xc[:, 1:], Fr]), Y, rcond=None)
    pd.DataFrame({"term": Xc_names[1:] + ["c_v"], "beta_full_model": beta_cov}).to_csv(
        f"{STATS}/covariate_betas.csv", index=False)

    print(f"PHASE4_DONE total elapsed={time.time()-t00:.0f}s", flush=True)


if __name__ == "__main__":
    main()
