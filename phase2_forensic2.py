#!/usr/bin/env python3
"""Forensic repair for the 2 irreducible QC failures (M2046, M2254).

Root cause hypothesis: pathological T1 (huge lesion hole, no fill on grid mismatch)
breaks bet and metric-masked SyN. Fix: whole-head pipeline —
  1. resample lesion mask into T1 grid with the existing rigid T2->T1 matrix
  2. enantiomorphic fill of the whole-head T1 via that mask
  3. ANTs Rigid+Affine+SyN (CC) whole-head T1 -> MNI152_T1_2mm (whole head), no -x
  4. apply composed transform to the mask; QC volume ratio + in-brain fraction
Container-run: python3 /work/scripts/phase2_forensic2.py
"""
import os, subprocess, traceback

os.environ.setdefault("MPLBACKEND", "Agg")
for v in ("OMP_NUM_THREADS", "ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"):
    os.environ.setdefault(v, "1")
import numpy as np
import nibabel as nib
from scipy import ndimage

BASE = os.environ.get("LNM_BASE", "/work")
OUT = f"{BASE}/masks_mni"
TMP = f"{BASE}/tmp"; os.makedirs(TMP, exist_ok=True)
os.environ["TMPDIR"] = TMP

FSLDIR = os.environ.get("FSLDIR", "/opt/fsl-6.0.7.6")
os.environ["FSLDIR"] = FSLDIR
os.environ["FSLOUTPUTTYPE"] = "NIFTI_GZ"
os.environ["PATH"] = f"{FSLDIR}/bin:{os.environ['PATH']}"
ANTS = "/opt/ants-2.4.1/antsRegistration"
MNI_WHOLE = f"{FSLDIR}/data/standard/MNI152_T1_2mm.nii.gz"
MNI_BRAIN_MASK = f"{FSLDIR}/data/standard/MNI152_T1_2mm_brain_mask.nii.gz"

CASES = ["sub-M2046_ses-2499", "sub-M2254_ses-814"]


def run(cmd, log):
    with open(log, "a") as f:
        f.write("\n$ " + " ".join(cmd) + "\n"); f.flush()
        p = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
    if p.returncode != 0:
        raise RuntimeError(f"rc={p.returncode}: {cmd[:3]}...")


def repair(tag, vol_native):
    wd = f"{OUT}/{tag}"
    log = f"{wd}/log_forensic2.txt"
    open(log, "w").close()
    mask = f"{wd}/mask_std.nii.gz"
    t1 = f"{wd}/t1_std.nii.gz"
    for p in (mask, t1, f"{wd}/t2_to_t1_0GenericAffine.mat"):
        assert os.path.exists(p), p

    # 1. mask -> T1 grid via existing rigid
    run(["antsApplyTransforms", "-i", mask, "-r", t1,
         "-t", f"{wd}/t2_to_t1_0GenericAffine.mat", "-n", "NearestNeighbor",
         "-o", f"{wd}/mask_t1.nii.gz"], log)

    # 2. enantiomorphic fill of whole-head T1
    t1_img = nib.load(t1)
    a = np.asanyarray(t1_img.dataobj).astype(np.float32)
    m = np.asanyarray(nib.load(f"{wd}/mask_t1.nii.gz").dataobj) > 0
    m = ndimage.binary_dilation(m, iterations=2)
    filled = np.where(m, a[:, ::-1, :], a)
    nib.save(nib.Nifti1Image(filled, t1_img.affine, t1_img.header),
             f"{wd}/t1_whole_filled.nii.gz")

    # 3. whole-head -> MNI whole-head
    run([ANTS, "--dimensionality", "3", "--float", "0",
         "--output", f"[{wd}/wh_to_mni_,{wd}/t1_in_mni_wh.nii.gz]",
         "--initial-moving-transform", f"[{MNI_WHOLE},{wd}/t1_whole_filled.nii.gz,1]",
         "--transform", "Rigid[0.1]",
         "--metric", f"MI[{MNI_WHOLE},{wd}/t1_whole_filled.nii.gz,1,32,Regular,0.25]",
         "--convergence", "[2000x2000x2000,1e-6,10]",
         "--shrink-factors", "4x2x1", "--smoothing-sigmas", "2x1x0vox",
         "--transform", "Affine[0.1]",
         "--metric", f"MI[{MNI_WHOLE},{wd}/t1_whole_filled.nii.gz,1,32,Regular,0.25]",
         "--convergence", "[1200x1200x100,1e-6,10]",
         "--shrink-factors", "4x2x1", "--smoothing-sigmas", "2x1x0vox",
         "--transform", "SyN[0.1,3,0]",
         "--metric", f"CC[{MNI_WHOLE},{wd}/t1_whole_filled.nii.gz,1,4]",
         "--convergence", "[100x70x50x20,1e-6,10]",
         "--shrink-factors", "6x4x2x1", "--smoothing-sigmas", "3x2x1x0vox"], log)

    # 4. mask -> MNI (whole-head transform composed with rigid)
    run(["antsApplyTransforms", "-i", mask, "-r", MNI_WHOLE,
         "-t", f"{wd}/wh_to_mni_1Warp.nii.gz",
         "-t", f"{wd}/wh_to_mni_0GenericAffine.mat",
         "-t", f"{wd}/t2_to_t1_0GenericAffine.mat",
         "-n", "NearestNeighbor", "-o", f"{wd}/mask_mni_forensic.nii.gz"], log)

    mm = nib.load(f"{wd}/mask_mni_forensic.nii.gz")
    mv = np.asanyarray(mm.dataobj) > 0
    vol = float(mv.sum() * np.prod(mm.header.get_zooms()[:3]))
    bm = np.asanyarray(nib.load(MNI_BRAIN_MASK).dataobj) > 0
    in_brain = float((mv & bm).sum() / max(mv.sum(), 1))
    ratio = vol / max(vol_native, 1)
    # hemisphere sanity: fraction of mask left of midline (MNI x<0)
    xs = np.where(mv.any(axis=(1, 2)))[0]
    frac_left = float(mv[xs[: len(xs) // 2]].sum() / max(mv.sum(), 1)) if len(xs) else 0
    verdict = "PASS" if (0.7 < ratio < 1.4 and in_brain >= 0.90) else "QC_WARN"
    if verdict == "PASS":
        subprocess.run(["cp", f"{wd}/mask_mni_forensic.nii.gz", f"{wd}/mask_mni.nii.gz"], check=True)
        open(f"{wd}/qc_pass.txt", "w").write("forensic2\n")
    print(f"{tag}: ratio={ratio:.3f} in_brain={in_brain:.3f} frac_left~{frac_left:.2f} -> {verdict}", flush=True)


def main():
    import pandas as pd
    audit = pd.read_csv(f"{BASE}/qc/phase0_mask_audit.csv")
    vols = {f"{r['sub']}_{r['ses']}": r["vol_mm3"] for _, r in audit.iterrows()}
    for tag in CASES:
        try:
            repair(tag, vols[tag])
        except Exception as e:
            print(f"{tag}: FAIL {type(e).__name__}: {e}", flush=True)
            traceback.print_exc()


if __name__ == "__main__":
    main()
