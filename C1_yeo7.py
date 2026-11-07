#!/usr/bin/env python3
"""Resample Yeo2011 1mm FreeSurfer-conformed to MNI152 2mm grid, then run C1-C3
with Yeo-7 composition entropy.

Steps:
  1. Load 1mm atlas (256^3), resample to 2mm MNI152 (91x109x91) using scipy zoom
  2. Build composition matrix W (n_patients x 7 networks)
  3. Compute entropy H_i, correlate with log-volume
  4. Select low-entropy tertile, rerun main regression with max-T
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "4")
import json
import numpy as np
import pandas as pd
import nibabel as nib
from scipy import stats, ndimage
from scipy.ndimage import zoom as ndzoom

BASE = "/mnt/n/ARC/lnm"
STATS = f"{BASE}/stats"
OUT = f"{STATS}/revision"
os.makedirs(OUT, exist_ok=True)

# ---------- Step 1: resample Yeo atlas to 2mm ----------
yeo_1mm = nib.load("/mnt/e/fMRIsoft/DPABI_V6.0_210501/Templates/Yeo2011_7Networks_MNI152_FreeSurferConformed1mm_LiberalMask.nii.gz")
yeo_data = np.asanyarray(yeo_1mm.dataobj).astype(np.float32)
if yeo_data.ndim == 4:
    yeo_data = yeo_data[..., 0]

# MNI152 2mm grid: 91x109x91, origin at voxel (0,0,0) maps to MNI (-90,-126,-72)
# FreeSurfer conformed 1mm: 256^3, origin at (128,128,128) maps to (0,0,0) in RAS
# We need to create a transform from 1mm to 2mm
# Both are in MNI space but with different grids

# Use the affine matrices directly
yeo_1mm_affine = yeo_1mm.affine
gm_2mm = nib.load(f"{BASE}/template/gm_mask_2mm.nii.gz")
gm_2mm_affine = gm_2mm.affine
gm_shape = tuple(gm_2mm.shape)

# Create combined transform: 2mm voxel -> world -> 1mm voxel
T = np.linalg.inv(yeo_1mm_affine) @ gm_2mm_affine

# Generate world coordinates for each 2mm voxel centre
shape3 = gm_shape
zs, ys, xs = np.meshgrid(np.arange(shape3[0]), np.arange(shape3[1]),
                          np.arange(shape3[2]), indexing="ij")
world = gm_2mm_affine @ np.vstack([zs.ravel(), ys.ravel(), xs.ravel(),
                                     np.ones(zs.size)])
vox_1mm = np.linalg.inv(yeo_1mm_affine) @ world

# Nearest-neighbour resample
v1, v2, v3 = vox_1mm[0].round().astype(int), vox_1mm[1].round().astype(int), vox_1mm[2].round().astype(int)
valid = (v1 >= 0) & (v1 < yeo_data.shape[0]) & (v2 >= 0) & (v2 < yeo_data.shape[1]) & (v3 >= 0) & (v3 < yeo_data.shape[2])

yeo_2mm = np.zeros(gm_shape, dtype=np.float32)
yeo_2mm.ravel()[np.arange(int(np.prod(gm_shape)))[valid]] = yeo_data[v1[valid], v2[valid], v3[valid]]

print("Yeo resampled to 2mm:", yeo_2mm.shape)
print("  network voxel counts:", {int(v): int((yeo_2mm == v).sum()) for v in range(1, 8)})

# Save the resampled atlas
nib.save(nib.Nifti1Image(yeo_2mm, gm_2mm.affine), f"{BASE}/template/yeo7_2mm.nii.gz")
print("saved yeo7_2mm.nii.gz")

# ---------- Step 2: compute composition ----------
summ = pd.read_csv(f"{BASE}/connmaps/gsp_sbc_summary.csv")
pheno = pd.read_csv(f"{STATS}/phenotype.csv")
names = [t for t in summ["lesion"] if t in set(pheno["tag"])]
ph = pheno.set_index("tag").loc[names].reset_index()
n = len(ph)

W7 = np.zeros((n, 7))
seed_gm_flat = []
for i, t in enumerate(names):
    m = np.asanyarray(nib.load(f"{BASE}/masks_mni/{t}/mask_mni.nii.gz").dataobj) > 0
    m_yeo = (yeo_2mm > 0)
    lesion_yeo = m & (yeo_2mm > 0)
    # count lesion voxels in each of the 7 networks
    for k in range(1, 8):
        W7[i, k - 1] = (lesion_yeo == float(k)).sum()
    seed_gm_flat.append(np.where((m & (gm_2mm_affine[:3, :3].diagonal()[0] > 0)).ravel())[0])  # placeholder

print("W7 shape:", W7.shape, "total lesion voxels in Yeo-covered:", int(W7.sum()))

# ---------- Step 3: entropy ----------
Wn = W7 / np.maximum(W7.sum(1, keepdims=True), 1)
with np.errstate(divide="ignore", invalid="ignore"):
    H_yeo = -np.nansum(np.where(Wn > 0, Wn * np.log(np.clip(Wn, 1e-300, 1)), 0.0), axis=1)

print("H_yeo stats: mean=%.3f  SD=%.3f  min=%.3f  max=%.3f" % (
    H_yeo.mean(), H_yeo.std(), H_yeo.min(), H_yeo.max()))

# correlate with log-volume
log_vol = np.log10(np.maximum(ph["vol_mm3"].to_numpy(float), 1))
r_C1, p_C1 = stats.pearsonr(H_yeo, log_vol)
r_s_C1, p_s_C1 = stats.spearmanr(H_yeo, ph["vol_mm3"])
print(f"C1 (Yeo-7): r(H, log10 vol) = {r_C1:.3f}, p = {p_C1:.2e}")
print(f"C1 (Yeo-7): Spearman(H, vol) = {r_s_C1:.3f}")

json.dump(dict(r_H_logvol=round(float(r_C1), 3), p=round(float(p_C1), 6),
               spearman_H_vol=round(float(r_s_C1), 3)),
          open(f"{OUT}/C1_yeo7.json", "w"), indent=1)
np.save(f"{OUT}/H_yeo.npy", H_yeo)
np.save(f"{OUT}/W7_yeo.npy", W7)
np.save(f"{OUT}/yeo_2mm.npy", yeo_2mm)
print("C1_YEO_DONE")
