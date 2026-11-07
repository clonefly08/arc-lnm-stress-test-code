#!/usr/bin/env python3
"""T1 (D-a spatial diagnostic) + T4 (lesion voxel counts) + true prevalence bounds for D7.
Run in WSL: python3 /mnt/n/ARC/analysis/t1t4_spatial_overlap.py"""
import os, json
os.environ.setdefault("OMP_NUM_THREADS", "4")
import numpy as np, pandas as pd, nibabel as nib
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = "/mnt/n/ARC"; LNM = f"{BASE}/lnm"; OUT = f"{BASE}/analysis/out"
os.makedirs(OUT, exist_ok=True)

gm_img = nib.load(f"{LNM}/template/gm_mask_2mm.nii.gz")
gmb = np.asanyarray(gm_img.dataobj) > 0
gmflat = np.where(gmb.ravel())[0]

summ = pd.read_csv(f"{LNM}/connmaps/gsp_sbc_summary.csv")
ph0 = pd.read_csv(f"{LNM}/stats/phenotype.csv")
names = [t for t in summ["lesion"] if t in set(ph0["tag"])]
n = len(names)
print(f"patients: {n}")

overlap = np.zeros(len(gmflat), dtype=np.int32)
Ns = {}
for t in names:
    m = np.asanyarray(nib.load(f"{LNM}/masks_mni/{t}/mask_mni.nii.gz").dataobj) > 0
    mg = m[gmb]
    overlap += mg.astype(np.int32)
    Ns[t] = int(mg.sum())
ov_frac = overlap.astype(np.float64) / n

tmap = np.asanyarray(nib.load(f"{LNM}/stats/obs_tmap.nii.gz").dataobj).astype(np.float64)
t_flat = tmap[gmb]

# ---- T1: spatial diagnostic ----
r_p, p_p = stats.pearsonr(ov_frac, t_flat)
r_s, p_s = stats.spearmanr(ov_frac, t_flat)
# deciles among voxels with overlap > 0 (zeros collapse quantiles)
pos = overlap > 0
r_p_pos, _ = stats.pearsonr(ov_frac[pos], t_flat[pos])
r_s_pos, _ = stats.spearmanr(ov_frac[pos], t_flat[pos])
qs = np.full(len(ov_frac), -1)
qs[pos] = pd.qcut(ov_frac[pos], 10, labels=False, duplicates="drop")
rows = []
for q in sorted(pd.unique(qs[qs >= 0])):
    sel = qs == q
    rows.append(dict(decile=int(q) + 1, n=int(sel.sum()),
                     mean_overlap_frac=float(ov_frac[sel].mean()),
                     mean_t=float(t_flat[sel].mean()),
                     mean_abs_t=float(np.abs(t_flat[sel]).mean()),
                     frac_abs_t_gt4=float((np.abs(t_flat[sel]) > 4).mean())))
json.dump(dict(overlap_gt0_voxels=int(pos.sum())), open(f"{OUT}/_cache.json", "w"))
print(f"T1: Pearson r = {r_p:.3f} (p={p_p:.2e}), Spearman rho = {r_s:.3f}")
for rr in rows:
    print(f"  D{rr['decile']}: ovf={rr['mean_overlap_frac']:.3f} "
          f"mean_t={rr['mean_t']:+.2f} mean|t|={rr['mean_abs_t']:.2f} "
          f"|t|>4: {rr['frac_abs_t_gt4']:.3f}")

fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.9),
                         gridspec_kw=dict(wspace=0.32, bottom=0.28, top=0.93))
ax = axes[0]
ax.hexbin(ov_frac, t_flat, gridsize=45, bins="log", cmap="Blues", linewidths=0)
ax.set_xlabel("lesion overlap fraction", fontsize=7)
ax.set_ylabel("t-statistic", fontsize=7)
ax.tick_params(labelsize=6)
ax.set_title(f"r = {r_p:.3f}, rho = {r_s:.3f}", fontsize=7)
ax = axes[1]
xs = [rr["mean_overlap_frac"] for rr in rows]
ax.plot(xs, [rr["mean_t"] for rr in rows], "o-", color="#31527a", lw=1, ms=3, label="mean t")
ax.plot(xs, [rr["mean_abs_t"] for rr in rows], "s--", color="#b5563f", lw=1, ms=3, label="mean |t|")
ax.set_xlabel("lesion overlap fraction (decile mean)", fontsize=7)
ax.set_ylabel("t-statistic", fontsize=7)
ax.tick_params(labelsize=6)
ax.legend(fontsize=6, frameon=False)
fig.savefig(f"{OUT}/figS_overlap_vs_t.png", dpi=300)
plt.close(fig)

json.dump(dict(n=n, pearson_r=round(float(r_p), 4), pearson_p=float(p_p),
               spearman_rho=round(float(r_s), 4),
               pearson_r_overlapgt0=round(float(r_p_pos), 4),
               spearman_rho_overlapgt0=round(float(r_s_pos), 4),
               n_overlap_gt0=int(pos.sum()),
               deciles=rows),
          open(f"{OUT}/t1_spatial_r.json", "w"), indent=1)

# ---- T4: lesion voxel counts ----
Nv = np.array([Ns[t] for t in names])
vol = ph0.set_index("tag").loc[names]["vol_mm3"].to_numpy(float)
t4 = dict(n=n, median_N=float(np.median(Nv)), min_N=int(Nv.min()), max_N=int(Nv.max()),
          n_lt_300=int((Nv < 300).sum()), n_lt_100=int((Nv < 100).sum()),
          corr_N_logvol=round(float(stats.pearsonr(Nv, np.log10(vol))[0]), 4))
print("T4:", t4)
pd.DataFrame({"tag": names, "N_gm_voxels": Nv, "vol_mm3": vol}).to_csv(
    f"{OUT}/t4_n_voxels.csv", index=False)
fig, ax = plt.subplots(figsize=(3.2, 2.4))
ax.hist(np.log10(Nv), bins=30, color="#7a93b5", lw=0)
ax.axvline(np.log10(300), color="#b5563f", lw=1, ls="--")
ax.text(np.log10(300) + 0.05, ax.get_ylim()[1] * 0.9, "N=300", fontsize=6, color="#b5563f")
ax.set_xlabel(r"log$_{10}$ lesion voxels in GM (MNI)", fontsize=7)
ax.set_ylabel("Patients", fontsize=7)
ax.tick_params(labelsize=6)
fig.tight_layout()
fig.savefig(f"{OUT}/t4_hist.png", dpi=300)
plt.close(fig)

# ---- prevalence bounds among tested VLSM voxels (for D7 fix) ----
tested = ov_frac >= 0.05
n_tested = int(tested.sum())
cnt = overlap[tested]
json.dump(dict(n_tested=n_tested, min_damaged=int(cnt.min()), max_damaged=int(cnt.max()),
               min_spared=int(n - cnt.max()), max_spared=int(n - cnt.min())),
          open(f"{OUT}/prevalence_bounds.json", "w"), indent=1)
print(f"tested voxels: {n_tested} (expect 45,109); damaged {cnt.min()}-{cnt.max()}; "
      f"spared {n-cnt.max()}-{n-cnt.min()}")
print("T1T4_DONE")
