#!/usr/bin/env python3
"""T3 (A3): remove the direct self-correlation term (target voxel inside its own seed).

For patient i, subject j, run r: seed s = mean of z-scored lesion rows (N voxels),
sigma_s = sample std of s (ddof=1). For v in S_i, with r = Pearson(s, z_v):
    R_v   = N * sigma_s * r - 1                      (sum of gamma_uv over u != v)
    var'  = (N^2 sigma_s^2 - 2 R_v - 1) / (N-1)^2    (sample var of s' = mean over S_i \\ {v})
    r'    = R_v / ((N-1) * sqrt(var'))               (Pearson(s', z_v))
(gamma = sample covariance; z-rows have unit sample variance so gamma_vv = 1 exactly.)

Pipeline: correct per run -> Fisher-z -> average runs per subject -> accumulate over
500 subjects -> splice corrected values into lesion voxels of the checkpoint mean map
-> covariates + Freedman-Lane 5,000 permutations -> n_sig at 0.05/tail and 0.025/tail,
peaks, Dice vs the original map. Resumable checkpoint every 20 subjects.
Run in WSL: python3 /mnt/n/ARC/analysis/t3_selfcorr.py
"""
import os, glob, json, time, traceback
os.environ.setdefault("MPLBACKEND", "Agg")
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(v, "4")
import numpy as np, pandas as pd, nibabel as nib

BASE = "/mnt/n/ARC"; LNM = f"{BASE}/lnm"; OUT = f"{BASE}/analysis/out"
GSP = f"{LNM}/template/gsp1000"
CKPT = f"{BASE}/analysis/out/.t3_checkpoint.npz"
CLIP = 1.0 - 1e-6
CKPT_EVERY = 20
K_PERM = 5000
rng = np.random.default_rng(42)

# ---------------- analytic correction ----------------
def correct_r_vector(r_vec, sigma_s, N):
    """Analytic leave-one-voxel-out correction for all lesion voxels of one run."""
    R = N * sigma_s * r_vec - 1.0
    varp = (N * N * sigma_s ** 2 - 2.0 * R - 1.0) / ((N - 1) ** 2)
    bad = ~(np.isfinite(varp) & (varp > 1e-12)) | ~np.isfinite(R)
    out = np.full_like(r_vec, np.nan, dtype=np.float64)
    ok = ~bad
    out[ok] = R[ok] / ((N - 1) * np.sqrt(varp[ok]))
    return np.clip(out, -CLIP, CLIP), bad

def selftest(ntest=2000, T=120, trials=6):
    """Brute-force vs analytic on synthetic data."""
    worst = 0.0
    for _ in range(trials):
        # random correlation structure among N unit-variance series
        N = int(rng.integers(4, 400))
        A = rng.normal(size=(T, N))
        A = (A - A.mean(0)) / A.std(0, ddof=1, keepdims=True)
        C = A.T @ A / (T - 1)                       # sample covariance, diag = 1
        # random target v
        v = int(rng.integers(N))
        s = A.mean(1)
        sd_s = s.std(ddof=1)
        r = float((A[:, v] - A[:, v].mean()).std(ddof=1) and
                  np.corrcoef(s, A[:, v])[0, 1])
        r_an, bad = correct_r_vector(np.array([r]), sd_s, N)
        keep = np.arange(N) != v
        sp = A[:, keep].mean(1)
        spz = (sp - sp.mean()) / sp.std(ddof=1)
        zv = (A[:, v] - A[:, v].mean()) / A[:, v].std(ddof=1)
        r_br = float(np.corrcoef(spz, zv)[0, 1])
        worst = max(worst, abs(r_an[0] - r_br))
    print(f"selftest worst |analytic - brute| = {worst:.3e}")
    assert worst < 1e-8, "analytic correction failed self-test"
    return worst

worst = selftest()
json.dump({"selftest_worst_absdiff": worst}, open(f"{OUT}/t3_selftest.json", "w"))

# ---------------- data structures ----------------
gm_img = nib.load(f"{LNM}/template/gm_mask_2mm.nii.gz")
gmb = np.asanyarray(gm_img.dataobj) > 0
gm_idx_bool = np.zeros(gmb.shape, dtype=bool); gm_idx_bool[gmb] = True

summ = pd.read_csv(f"{LNM}/connmaps/gsp_sbc_summary.csv")
ph0 = pd.read_csv(f"{LNM}/stats/phenotype.csv")
names = [t for t in summ["lesion"] if t in set(ph0["tag"])]
L = len(names)
name_pos = {t: i for i, t in enumerate(names)}

ck = np.load(f"{LNM}/connmaps/.sbc_checkpoint.npz")
F_mean_all = ck["sum"] / ck["n"]                       # (all lesions) x V
sel_rows = [i for i, t in enumerate(summ["lesion"]) if t in set(ph0["tag"])]
F_mean = F_mean_all[sel_rows]                          # L x V original (correctness baseline)

seeds = np.zeros((int(gmb.sum()), L), dtype=bool)
Ns = np.zeros(L, dtype=int)
for i, t in enumerate(names):
    m = np.asanyarray(nib.load(f"{LNM}/masks_mni/{t}/mask_mni.nii.gz").dataobj) > 0
    mg = m[gmb]
    seeds[:, i] = mg
    Ns[i] = int(mg.sum())
print(f"patients {L}; N min/median/max = {Ns.min()}/{np.median(Ns):.0f}/{Ns.max()}")

def find_subjects():
    runs = {}
    for p in sorted(glob.glob(f"{GSP}/sub-*/func/*_finalmask.nii.gz")):
        sid = p.split("/")[-3]
        runs.setdefault(sid, []).append(p)
    return {k: v for k, v in sorted(runs.items()) if v}

def process_run(path):
    img = nib.load(path)
    data = np.asanyarray(img.dataobj, dtype=np.float32)
    ts = data[gm_idx_bool]
    del data
    ts = ts[:, np.any(np.isfinite(ts), axis=0)]
    ts = np.nan_to_num(ts, nan=0.0, posinf=0.0, neginf=0.0)
    mu = ts.mean(axis=1, keepdims=True)
    sd = ts.std(axis=1, ddof=1, keepdims=True); sd[sd == 0] = 1.0
    return (ts - mu) / sd

# ---------------- full data pass ----------------
acc_sum = np.zeros((L, int(gmb.sum())), dtype=np.float64)
done = set()
if os.path.exists(CKPT):
    z = np.load(CKPT, allow_pickle=True)
    acc_sum = z["sum"]; done = set(list(z["done"]))
    print(f"resume: {len(done)} subjects done")
VALIDATE_SID = None  # validate on the first processed subject

subjects = find_subjects()
print(f"normative subjects: {len(subjects)}")
t0 = time.time(); n_proc = 0
for i, (sid, files) in enumerate(subjects.items(), 1):
    if sid in done:
        continue
    try:
        F_subj = []
        for rp in files:
            Z = process_run(rp)                            # V x T
            do_validate = (VALIDATE_SID is None)
            if do_validate:
                VALIDATE_BRUTE = {}
            # raw seed mean series + std per patient (L x T, L)
            seed_ts = seeds.astype(np.float32).T @ Z / Ns[:, None].astype(np.float32)
            mu_s = seed_ts.mean(axis=1, keepdims=True)
            sd_s = seed_ts.std(axis=1, ddof=1, keepdims=True)
            sd_s[sd_s == 0] = 1.0
            seed_z = (seed_ts - mu_s) / sd_s
            corr = (seed_z @ Z.T) / Z.shape[1]             # L x V Pearson
            # corrected Fisher-z maps (only lesion voxels differ)
            Fc = np.empty((L, corr.shape[1]), dtype=np.float64)
            n_bad = 0
            for li in range(L):
                N = Ns[li]
                idx_v = np.where(seeds[:, li])[0]
                r_les = corr[li, idx_v].astype(np.float64)
                if N >= 3:
                    rc, bad = correct_r_vector(r_les, float(sd_s[li, 0]), N)
                    n_bad += int(bad.sum())
                else:
                    rc = np.full(r_les.shape, np.nan); n_bad += r_les.size
                vals = np.arctanh(np.clip(rc, -CLIP, CLIP))
                vals[~np.isfinite(vals)] = 0.0
                Fc[li, idx_v] = vals
                nonles = ~seeds[:, li]
                Fc[li, nonles] = np.arctanh(np.clip(corr[li, nonles], -CLIP, CLIP))
            if do_validate:
                # brute-force check: one patient with decent N, 100 lesion voxels
                li = int(np.argmax(Ns))
                N = Ns[li]
                idx_v = np.where(seeds[:, li])[0]
                r_les = corr[li, idx_v].astype(np.float64)
                rc_a, _ = correct_r_vector(r_les, float(sd_s[li, 0]), N)
                pick = rng.choice(N, size=min(100, N), replace=False)
                Zles = Z[idx_v, :]
                worst_d = 0.0
                for v in pick:
                    keepm = np.ones(N, dtype=bool); keepm[v] = False
                    sp = Zles[keepm].mean(0)
                    spz = (sp - sp.mean()) / sp.std(ddof=1)
                    zv = Zles[v]
                    zv = (zv - zv.mean()) / zv.std(ddof=1)
                    rb = float((spz * zv).mean())
                    worst_d = max(worst_d, abs(rb - rc_a[v]))
                VALIDATE_BRUTE = {"sid": sid, "patient": names[li], "N": int(N),
                                  "n_voxels_checked": int(len(pick)),
                                  "worst_absdiff": float(worst_d)}
                print(f"  validate {sid}: worst |analytic-brute| = {worst_d:.2e}", flush=True)
                json.dump(VALIDATE_BRUTE, open(f"{OUT}/t3_validate.json", "w"))
                VALIDATE_SID = sid
            F_subj.append(Fc.astype(np.float32))
            del Z, corr, seed_z
        F = np.mean(F_subj, axis=0).astype(np.float64)
        F = np.nan_to_num(F, nan=0.0, posinf=0.0, neginf=0.0)
        acc_sum += F
        done.add(sid); n_proc += 1
        if n_proc % 10 == 0:
            print(f"[{i}/{len(subjects)}] {sid} elapsed={time.time()-t0:.0f}s", flush=True)
        if n_proc % CKPT_EVERY == 0:
            np.savez(CKPT, sum=acc_sum, done=np.array(sorted(done)))
            print("  checkpoint saved", flush=True)
    except Exception as e:
        print(f"  FAIL {sid}: {type(e).__name__}: {e}", flush=True)
        traceback.print_exc()

np.savez(CKPT, sum=acc_sum, done=np.array(sorted(done)))
print(f"accumulated over {len(done)} subjects", flush=True)

# ---------------- post pipeline ----------------
ph = ph0.set_index("tag").loc[names].reset_index()
Y = ph["aq"].to_numpy(float)
Xc = np.column_stack([np.ones(L), ph["age"], ph["sex"], ph["log_days"], ph["vol_mm3"] / 1e5])

def residue(M, X):
    beta, *_ = np.linalg.lstsq(X, M, rcond=None)
    return M - X @ beta

def run_pipeline(Fm, label):
    Yr = residue(Y[:, None], Xc).ravel()
    Fr = residue(Fm, Xc)
    dof = L - Xc.shape[1] - 1
    yz = (Yr - Yr.mean()) / Yr.std()
    sd = Fr.std(0); sd[sd == 0] = 1.0
    fz = (Fr - Fr.mean(0)) / sd
    fz[~np.isfinite(fz)] = 0.0
    r = np.clip(yz @ fz / (L - 1), -1 + 1e-9, 1 - 1e-9)
    tobs = np.nan_to_num(r * np.sqrt(dof / (1 - r ** 2)))
    maxpos = np.empty(K_PERM); maxneg = np.empty(K_PERM)
    for k in range(K_PERM):
        perm = rng.permutation(L)
        yzp = (Yr[perm] - Yr[perm].mean()) / Yr[perm].std()
        rr = np.clip(yzp @ fz / (L - 1), -1 + 1e-9, 1 - 1e-9)
        tk = np.nan_to_num(rr * np.sqrt(dof / (1 - rr ** 2)))
        maxpos[k] = tk.max(); maxneg[k] = tk.min()
    tp05 = float(np.sort(maxpos)[::-1][int(np.ceil(0.05 * K_PERM)) - 1])
    tn05 = float(-np.sort(maxneg)[int(np.ceil(0.05 * K_PERM)) - 1])
    tp025 = float(np.sort(maxpos)[::-1][int(np.ceil(0.025 * K_PERM)) - 1])
    tn025 = float(-np.sort(maxneg)[int(np.ceil(0.025 * K_PERM)) - 1])
    n05 = int(((tobs > tp05) | (tobs < -tn05)).sum())
    n025 = int(((tobs > tp025) | (tobs < -tn025)).sum())
    print(f"{label}: n_sig 0.05/tail = {n05}, 0.025/tail = {n025}", flush=True)
    return dict(n05=n05, n025=n025, t_obs=tobs, th=(tp05, tn05, tp025, tn025))

res_corr = run_pipeline(acc_sum / len(done), "corrected")
res_orig = run_pipeline(F_mean, "original")

def dice(a, b):
    ia = int((a & b).sum())
    return 2 * ia / max(int(a.sum()) + int(b.sum()), 1)

for al, key in (("0.05", "n05"), ("0.025", "n025")):
    tp, tn, tp2, tn2 = res_corr["th"]
    tp_o, tn_o, tp2_o, tn2_o = res_orig["th"]
    if al == "0.05":
        mc = (res_corr["t_obs"] > tp) | (res_corr["t_obs"] < -tn)
        mo = (res_orig["t_obs"] > tp_o) | (res_orig["t_obs"] < -tn_o)
    else:
        mc = (res_corr["t_obs"] > tp2) | (res_corr["t_obs"] < -tn2)
        mo = (res_orig["t_obs"] > tp2_o) | (res_orig["t_obs"] < -tn2_o)
    print(f"Dice corrected vs original @ {al}/tail: {dice(mc, mo):.3f} "
          f"({int(mc.sum())} vs {int(mo.sum())})")

# per-patient correction magnitude vs N
delta = np.abs(acc_sum / len(done) - F_mean).mean(axis=1)
dd = pd.DataFrame({"tag": names, "N": Ns, "mean_abs_delta_fisherz": delta})
dd.to_csv(f"{OUT}/t3_delta_per_patient.csv", index=False)
rr = float(np.corrcoef(delta, Ns)[0, 1])
print(f"corr(mean|delta|, N) = {rr:.3f}")

json.dump(dict(n_subjects=len(done), selftest_worst=worst,
               corrected=res_corr["n05"], corrected_025=res_corr["n025"],
               original=res_orig["n05"], original_025=res_orig["n025"],
               corr_delta_N=rr,
               n_patients_N_lt_3=int((Ns < 3).sum())),
          open(f"{OUT}/t3_before_after.json", "w"), indent=1)
print("T3_DONE")
