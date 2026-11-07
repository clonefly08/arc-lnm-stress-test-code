#!/usr/bin/env python3
"""Stats-level ventricle exclusion: of the 2,469 significant voxels, how many and which
regions survive when HarvardOxford Lateral Ventricle territory is excluded?
(FWE threshold unchanged — post-hoc subset reporting is conservative.)"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "4")
import numpy as np
import pandas as pd
import nibabel as nib

BASE = "/mnt/n/ARC/lnm"
STATS = f"{BASE}/stats"
FSLDIR = "/usr/local/fsl"

gm_img = nib.load(f"{BASE}/template/gm_mask_2mm.nii.gz")
gm_bool = np.asanyarray(gm_img.dataobj) > 0
fwe = np.load(f"{STATS}/fwe_p_maxT.npy")
sig = np.zeros(gm_bool.shape, dtype=bool)
sig.ravel()[np.where(gm_bool.ravel())[0]] = fwe < 0.05
n_all = int(sig.sum())

sub = nib.load(f"{FSLDIR}/data/atlases/HarvardOxford/HarvardOxford-sub-maxprob-thr0-2mm.nii.gz")
a = np.asanyarray(sub.dataobj).astype(int)
import re
labs = dict((int(i), n) for i, n in re.findall(
    r'<label index="(\d+)"[^>]*>([^<]+)</label>',
    open(f"{FSLDIR}/data/atlases/HarvardOxford-Subcortical.xml", encoding="utf-8", errors="replace").read()))
vent_ids = [i for i, n in labs.items() if "Lateral Ventricle" in n]
vent_mask = np.isin(a, vent_ids)

sig_nv = sig & ~vent_mask
n_nv = int(sig_nv.sum())
print(f"significant voxels: all={n_all}  ventricle-territory={int((sig & vent_mask).sum())}  "
      f"non-ventricle={n_nv}", flush=True)

cort = nib.load(f"{FSLDIR}/data/atlases/HarvardOxford/HarvardOxford-cort-maxprob-thr0-2mm.nii.gz")
ac = np.asanyarray(cort.dataobj).astype(int)
labs_c = dict((int(i), n) for i, n in re.findall(
    r'<label index="(\d+)"[^>]*>([^<]+)</label>',
    open(f"{FSLDIR}/data/atlases/HarvardOxford-Cortical.xml", encoding="utf-8", errors="replace").read()))
rows = []
for li, cnt in zip(*np.unique(ac[sig_nv], return_counts=True)):
    if li != 0 and cnt >= 5:
        rows.append(dict(region=labs_c.get(li, f"idx{li}"), voxels=int(cnt)))
for li, cnt in zip(*np.unique(a[sig_nv], return_counts=True)):
    if li != 0 and cnt >= 5 and li not in vent_ids:
        rows.append(dict(region="[sub] " + labs.get(li, f"idx{li}"), voxels=int(cnt)))
df = pd.DataFrame(rows).sort_values("voxels", ascending=False)
df.to_csv(f"{STATS}/network_labels_nonvent.csv", index=False)
print("\nnon-ventricle significant network, top 12:"); print(df.head(12).to_string(index=False))

t = np.asanyarray(nib.load(f"{STATS}/obs_tmap.nii.gz").dataobj)
for sign, nm in ((1, "positive"), (-1, "negative")):
    tt = np.where(sig_nv, t * sign, -np.inf)
    pk = np.unravel_index(np.argmax(tt), tt.shape)
    print(f"non-vent peak {nm}: t={t[pk]:.2f} at MNI {np.round(nib.affines.apply_affine(gm_img.affine, pk), 0)}")

nib.save(nib.Nifti1Image(sig_nv.astype(np.uint8), gm_img.affine),
         f"{STATS}/sig_network_nonventricle.nii.gz")
