#!/usr/bin/env python3
"""Canonical rerun: ALL fast analysis variants under ONE permutation stream
(seed 20261107, K=5000 Freedman-Lane, per-tail thresholds at 0.05 and 0.025),
so every number in the revised manuscript shares the same null.

Variants: primary, A3-corrected, left-only, gm04 (own checkpoint), log-volume,
drop-cerebellar, drop-unrepairable, demean, C3 low-entropy tertile, ventricle-
exclusion count (primary, post-hoc intersection), occupancy-retention from the
production loc_conn array (provisional; joint-model rerun queued separately).
Run in WSL: python3 /mnt/n/ARC/analysis/canonical_rerun.py"""
import os, json, time
os.environ.setdefault("OMP_NUM_THREADS", "8")
import numpy as np, pandas as pd, nibabel as nib

BASE = "/mnt/n/ARC"; LNM = f"{BASE}/lnm"; OUT = f"{BASE}/analysis/out"
K = 5000
SEED = 20261107
rng = np.random.default_rng(SEED)

gm_img = nib.load(f"{LNM}/template/gm_mask_2mm.nii.gz")
gmb = np.asanyarray(gm_img.dataobj) > 0
gmflat = np.where(gmb.ravel())[0]; V = len(gmflat)

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
F_main = (ck["sum"][idx] / ck["n"]).astype(np.float64)

ckc = np.load(f"{OUT}/.t3_checkpoint.npz", allow_pickle=True)
F_corr = ckc["sum"].astype(np.float64) / 500.0

ckg = np.load(f"{LNM}/connmaps/.sbc_checkpoint_gm04.npz")
summ_g = pd.read_csv(f"{LNM}/connmaps/gsp_sbc_summary_gm04.csv")
row_g = {t: i for i, t in enumerate(summ_g["lesion"])}
gm04_names = [t for t in names if t in row_g]          # stricter mask dropped ~1 patient
gm04_pidx = np.array([tpos[t] for t in gm04_names])
F_gm04 = (ckg["sum"][[row_g[t] for t in gm04_names]] / ckg["n"]).astype(np.float64)
print(f"gm04 rows: {F_gm04.shape[0]} (missing: {sorted(set(names) - set(row_g))})")
print(f"F matrices: main {F_main.shape}, corr {F_corr.shape}, gm04 {F_gm04.shape}")

# tag sets
cere_tags = open(f"{LNM}/tmp/cere_tags.txt").read().strip().split(",")
cere_tags = [t for t in cere_tags if t in tpos]
print(f"cerebellar-overlap tags: {len(cere_tags)}")

audit = pd.read_csv(f"{LNM}/qc/phase0_mask_audit.csv").set_index("sub")
nonleft_tags = []
for t in names:
    sub = t.split("_ses-")[0]
    if str(audit.loc[sub, "side_x0"]) != "L":
        nonleft_tags.append(t)
print(f"non-left-hemisphere tags: {len(nonleft_tags)} (expect 7): {nonleft_tags}")

H7 = np.load(f"{LNM}/stats/revision/H_yeo.npy")
sel_low = np.where(H7 <= float(np.quantile(H7, 1 / 3)))[0]
noext_tag = "sub-M2046_ses-2499"

def residue(M, X):
    beta, *_ = np.linalg.lstsq(X, M, rcond=None)
    return M - X @ beta

PERM_CACHE = {}
def get_perms(n):
    """One shared permutation set per sample size: every variant with the same n
    sees the identical null draw, so matched comparisons (e.g. demeaning) are exact."""
    if n not in PERM_CACHE:
        PERM_CACHE[n] = [rng.permutation(n) for _ in range(K)]
    return PERM_CACHE[n]

def perm_counts(Fm, pidx=None, Xdesign=None, label=""):
    """Fm rows must be aligned to pidx (indices into the patient tables)."""
    t0 = time.time()
    if pidx is None:
        pidx = np.arange(L)
    X = Xdesign if Xdesign is not None else Xc
    Xs = X[pidx]
    Yr = residue(Y[pidx, None], Xs).ravel()
    Fr = residue(Fm, Xs)
    dof = len(pidx) - Xs.shape[1] - 1
    yz = (Yr - Yr.mean()) / Yr.std()
    sd = Fr.std(0); sd[sd == 0] = 1.0
    fz = (Fr - Fr.mean(0)) / sd
    fz[~np.isfinite(fz)] = 0.0
    r = np.clip(yz @ fz / (len(pidx) - 1), -1 + 1e-9, 1 - 1e-9)
    tobs = np.nan_to_num(r * np.sqrt(dof / (1 - r ** 2)))
    maxpos = np.empty(K); maxneg = np.empty(K)
    perms = get_perms(len(pidx))
    for k in range(K):
        perm = perms[k]
        yzp = (Yr[perm] - Yr[perm].mean()) / Yr[perm].std()
        rr = np.clip(yzp @ fz / (len(pidx) - 1), -1 + 1e-9, 1 - 1e-9)
        tk = np.nan_to_num(rr * np.sqrt(dof / (1 - rr ** 2)))
        maxpos[k] = tk.max(); maxneg[k] = tk.min()
    out = {"label": label, "n": int(len(pidx)), "max_t": round(float(np.abs(tobs).max()), 3)}
    for al, key in ((0.05, "005"), (0.025, "0025")):
        tp = float(np.sort(maxpos)[::-1][int(np.ceil(al * K)) - 1])
        tn = float(-np.sort(maxneg)[int(np.ceil(al * K)) - 1])
        m = (tobs > tp) | (tobs < -tn)
        out[f"sig_{key}"] = int(m.sum())
        out[f"mask_{key}"] = m
    print(f"{label:28s} n={len(pidx):3d}  0.05/tail: {out['sig_005']:6,}  "
          f"0.025/tail: {out['sig_0025']:6,}  ({time.time()-t0:.0f}s)", flush=True)
    return out

variants = {}
variants["primary"] = perm_counts(F_main, label="primary (linear volume)")
drop = np.array([tpos[t] for t in cere_tags if t in tpos], dtype=int)
pidx = np.setdiff1d(np.arange(L), drop)
variants["drop_cerebellar"] = perm_counts(F_main[pidx], pidx=pidx, label="drop cerebellar-overlap")
drop = np.array([tpos[t] for t in nonleft_tags], dtype=int)
pidx = np.setdiff1d(np.arange(L), drop)
variants["left_only"] = perm_counts(F_main[pidx], pidx=pidx, label="left-hemisphere only")
drop = np.array([tpos[noext_tag]], dtype=int) if noext_tag in tpos else np.array([], dtype=int)
pidx = np.setdiff1d(np.arange(L), drop)
variants["drop_unrepairable"] = perm_counts(F_main[pidx], pidx=pidx, label="drop unrepairable reg.")
Xl = Xc.copy(); Xl[:, 4] = np.log10(np.maximum(ph["vol_mm3"].to_numpy(float), 1.0))
variants["log_volume"] = perm_counts(F_main, Xdesign=Xl, label="log-volume term")
variants["demean"] = perm_counts(F_main - F_main.mean(0, keepdims=True), label="demeaning check")
variants["corrected_A3"] = perm_counts(F_corr, label="A3 selfcorr-corrected")
variants["gm04"] = perm_counts(F_gm04, pidx=gm04_pidx, label="GM prior > 0.4")
pidx = np.sort(sel_low)
variants["C3_low_entropy"] = perm_counts(F_main[pidx], pidx=pidx, label="low-entropy tertile")

# Dice between primary and key variants at 0.025
def dice(a, b):
    return 2 * int((a & b).sum()) / max(int(a.sum()) + int(b.sum()), 1)
for v in ("left_only", "log_volume", "corrected_A3"):
    d = dice(variants["primary"]["mask_0025"], variants[v]["mask_0025"])
    variants[v]["dice_vs_primary_0025"] = round(d, 3)
    print(f"Dice(primary, {v}) @0.025/tail = {d:.3f}")

# ventricle exclusion on the canonical primary map (0.05 and 0.025)
FSLDIR = "/usr/local/fsl"
import re
sub = nib.load(f"{FSLDIR}/data/atlases/HarvardOxford/HarvardOxford-sub-maxprob-thr0-2mm.nii.gz")
a = np.asanyarray(sub.dataobj).astype(int)
labs = dict((int(i), n) for i, n in re.findall(
    r'<label index="(\d+)"[^>]*>([^<]+)</label>',
    open(f"{FSLDIR}/data/atlases/HarvardOxford-Subcortical.xml", encoding="utf-8", errors="replace").read()))
vent_ids = [i for i, nm in labs.items() if "Lateral Ventricle" in nm]
vent_gm = np.isin(a, vent_ids)[gmb]
for al, key in (("0.05", "005"), ("0.025", "0025")):
    m = variants["primary"][f"mask_{key}"]
    ve = variants.setdefault("ventricle_exclusion", {})
    ve[f"sig_{key}"] = int((m & ~vent_gm).sum())
    ve[f"vent_fraction_{key}"] = round(float((m & vent_gm).sum()) / max(int(m.sum()), 1), 3)
    ve[f"primary_{key}"] = int(m.sum())
    print(f"ventricle exclusion @ {al}/tail: non-vent {ve[f'sig_{key}']:,} "
          f"of {int(m.sum()):,} (vent frac {ve[f'vent_fraction_{key}']})")

# occupancy retention from the production loc_conn array (provisional; canonical rerun queued)
loc = np.load(f"{LNM}/stats/loc_conn_fwe_p.npy")
for al, name in ((0.05, "005"), (0.025, "0025")):
    ml = loc < al
    mp = variants["primary"][f"mask_{name}"]
    inter = int((ml & mp).sum())
    print(f"occupancy retention @{name}: {inter:,} of {int(mp.sum()):,} "
          f"({inter/max(int(mp.sum()),1):.1%}) [production stream, provisional]")
    variants.setdefault("occupancy_provisional", {})[name] = {
        "retained": inter, "primary": int(mp.sum()),
        "fraction": round(inter / max(int(mp.sum()), 1), 3)}

# gm04 lives on the stricter 168,561-voxel mask: map to 3D for Dice vs primary
th04 = np.asanyarray(nib.load(f"{LNM}/template/gm_mask_2mm_th04.nii.gz").dataobj) > 0
th04_flat = np.where(th04.ravel())[0]

idx_p = gmflat[variants["primary"]["mask_0025"]]
idx_g = th04_flat[variants["gm04"]["mask_0025"]]
inter = len(np.intersect1d(idx_p, idx_g))
d = 2 * inter / (len(idx_p) + len(idx_g))
variants["gm04"]["dice_vs_primary_0025"] = round(d, 3)
print(f"Dice(primary, gm04) @0.025/tail (GM>0.4 space) = {d:.3f}")

# serialise (drop numpy masks)
outj = {}
for k2, v2 in variants.items():
    outj[k2] = {kk: vv for kk, vv in v2.items() if not kk.startswith("mask_")}
json.dump(outj, open(f"{OUT}/canonical_numbers.json", "w"), indent=1)
np.savez(f"{OUT}/canonical_masks.npz",
         **{f"{k2}_{kk}": v2[kk] for k2, v2 in variants.items()
            for kk in v2 if kk.startswith("mask_")})
print("CANONICAL_DONE")
