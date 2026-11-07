#!/usr/bin/env python3
"""Phase 2 retry: re-run only the 24 QC_WARN cases with two fixes:
  1. T1-side enantiomorphic fill (using the rigid-transformed mask_t1 -> grids match)
  2. T2->T1 upgraded to Rigid+Affine two-stage (scanner gradient scale differences)
Container-run like phase2_norm.py:  python3 /work/scripts/phase2_retry.py [--workers 7]
"""
import os, sys, glob, subprocess, argparse, traceback

os.environ.setdefault("MPLBACKEND", "Agg")
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"):
    os.environ.setdefault(v, "1")

import numpy as np
import pandas as pd
import nibabel as nib
from scipy import ndimage

DS = os.environ.get("LNM_DS", "/ds")
BASE = os.environ.get("LNM_BASE", "/work")
OUT = f"{BASE}/masks_mni"
TMP = f"{BASE}/tmp"
os.makedirs(TMP, exist_ok=True)
os.environ["TMPDIR"] = TMP


def find_tool(name):
    p = subprocess.run(["bash", "-c", f"command -v {name}"], capture_output=True, text=True)
    return p.stdout.strip()


FSLDIR = os.environ.get("FSLDIR") or "/opt/fsl-6.0.7.6"
os.environ["FSLDIR"] = FSLDIR
os.environ["FSLOUTPUTTYPE"] = "NIFTI_GZ"
os.environ["PATH"] = f"{FSLDIR}/bin:{os.environ['PATH']}"
ANTS = find_tool("antsRegistration")
MNI = f"{FSLDIR}/data/standard/MNI152_T1_2mm_brain"
MNI_MASK = MNI + "_mask.nii.gz"
MNI = MNI + ".nii.gz"
print(f"ANTS={ANTS} MNI={MNI}", flush=True)


def run(cmd, log_path):
    with open(log_path, "a") as f:
        f.write("\n$ " + " ".join(cmd) + "\n")
        f.flush()
        p = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, env=dict(os.environ, TMPDIR=TMP))
    if p.returncode != 0:
        raise RuntimeError(f"cmd failed rc={p.returncode}: {' '.join(cmd)}")


def mirror_fill_arr(arr, mask_bool, dilate=1):
    m = ndimage.binary_dilation(mask_bool, iterations=dilate) if dilate else mask_bool
    return np.where(m, arr[:, ::-1, :], arr)


def process(row):
    tag = row["tag"]
    wd = f"{OUT}/{tag}"
    log_path = f"{wd}/log_retry.txt"
    status = {"tag": tag, "status": "", "vol_ratio": np.nan, "in_brain_frac": np.nan, "note": ""}
    try:
        open(log_path, "w").close()
        mask = f"{wd}/mask_std.nii.gz"
        t2f = f"{wd}/t2_filled.nii.gz"
        t1 = f"{wd}/t1_std.nii.gz"
        for p in (mask, t2f, t1):
            if not os.path.exists(p):
                status.update(status="FAIL", note=f"missing {os.path.basename(p)}")
                return status

        # --- T2->T1: Rigid + Affine, metric-masked ---
        run([ANTS, "--dimensionality", "3", "--float", "0",
             "--output", f"[{wd}/t2_to_t1_,{wd}/t2_in_t1.nii.gz]",
             "--initial-moving-transform", f"[{wd}/t1_brain.nii.gz,{t2f},1]",
             "--transform", "Rigid[0.1]",
             "--metric", f"MI[{wd}/t1_brain.nii.gz,{t2f},1,32,Regular,0.25]",
             "--convergence", "[2000x2000x2000,1e-6,10]",
             "--shrink-factors", "4x2x1", "--smoothing-sigmas", "2x1x0vox",
             "--transform", "Affine[0.1]",
             "--metric", f"MI[{wd}/t1_brain.nii.gz,{t2f},1,32,Regular,0.25]",
             "--convergence", "[1200x1200x100,1e-6,10]",
             "--shrink-factors", "4x2x1", "--smoothing-sigmas", "2x1x0vox",
             "-x", f"[{wd}/t1_brain_mask.nii.gz,{mask}]"], log_path)
        run(["antsApplyTransforms", "-i", mask, "-r", f"{wd}/t1_brain.nii.gz",
             "-t", f"{wd}/t2_to_t1_0GenericAffine.mat", "-n", "NearestNeighbor",
             "-o", f"{wd}/mask_t1.nii.gz"], log_path)

        # --- T1-side enantiomorphic fill (grids now match via mask_t1) ---
        t1_img = nib.load(t1)
        t1_arr = np.asanyarray(t1_img.dataobj).astype(np.float32)
        m_t1 = np.asanyarray(nib.load(f"{wd}/mask_t1.nii.gz").dataobj) > 0
        t1f = np.where(m_t1, t1_arr[:, ::-1, :], t1_arr)
        nib.save(nib.Nifti1Image(t1f, t1_img.affine, t1_img.header), f"{wd}/t1_filled.nii.gz")
        run(["bet", f"{wd}/t1_filled.nii.gz", f"{wd}/t1_brain", "-R", "-f", "0.45"], log_path)
        if not os.path.exists(f"{wd}/t1_brain_mask.nii.gz"):
            run(["fslmaths", f"{wd}/t1_brain.nii.gz", "-bin", "-fillh",
                 f"{wd}/t1_brain_mask.nii.gz"], log_path)

        # --- T1->MNI rigid+affine+SyN ---
        run([ANTS, "--dimensionality", "3", "--float", "0",
             "--output", f"[{wd}/t1_to_mni_,{wd}/t1_in_mni.nii.gz]",
             "--initial-moving-transform", f"[{MNI},{wd}/t1_brain.nii.gz,1]",
             "--transform", "Rigid[0.1]",
             "--metric", f"MI[{MNI},{wd}/t1_brain.nii.gz,1,32,Regular,0.25]",
             "--convergence", "[2000x2000x2000,1e-6,10]",
             "--shrink-factors", "4x2x1", "--smoothing-sigmas", "2x1x0vox",
             "--transform", "Affine[0.1]",
             "--metric", f"MI[{MNI},{wd}/t1_brain.nii.gz,1,32,Regular,0.25]",
             "--convergence", "[1200x1200x100,1e-6,10]",
             "--shrink-factors", "4x2x1", "--smoothing-sigmas", "2x1x0vox",
             "--transform", "SyN[0.1,3,0]",
             "--metric", f"CC[{MNI},{wd}/t1_brain.nii.gz,1,4]",
             "--convergence", "[100x70x50x20,1e-6,10]",
             "--shrink-factors", "6x4x2x1", "--smoothing-sigmas", "3x2x1x0vox",
             "-x", f"[{MNI_MASK},{wd}/mask_t1.nii.gz]"], log_path)

        run(["antsApplyTransforms", "-i", mask, "-r", MNI,
             "-t", f"{wd}/t1_to_mni_1Warp.nii.gz",
             "-t", f"{wd}/t1_to_mni_0GenericAffine.mat",
             "-t", f"{wd}/t2_to_t1_0GenericAffine.mat",
             "-n", "NearestNeighbor", "-o", f"{wd}/mask_mni.nii.gz"], log_path)

        mm = nib.load(f"{wd}/mask_mni.nii.gz")
        mv = np.asanyarray(mm.dataobj) > 0
        vol_mni = float(mv.sum() * np.prod(mm.header.get_zooms()[:3]))
        bm = np.asanyarray(nib.load(MNI_MASK).dataobj) > 0
        in_brain = float((mv & bm).sum() / max(mv.sum(), 1))
        ratio = vol_mni / max(row["vol_native"], 1)
        status.update(vol_ratio=round(ratio, 3), in_brain_frac=round(in_brain, 3))
        problems = []
        if mv.sum() == 0:
            problems.append("empty")
        if not 0.7 < ratio < 1.4:
            problems.append(f"vol_ratio={ratio:.2f}")
        if in_brain < 0.90:
            problems.append(f"in_brain={in_brain:.2f}")
        if problems:
            status.update(status="QC_WARN", note=";".join(problems))
        else:
            status.update(status="PASS")
            open(f"{wd}/qc_pass.txt", "w").write("ok\n")
    except Exception as e:
        status["status"] = "FAIL"
        status["note"] = f"{type(e).__name__}: {e}"
        traceback.print_exc()
    return status


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=7)
    args = ap.parse_args()
    rep = pd.read_csv(f"{BASE}/qc/phase2_full_report.csv")
    warned = rep[rep["status"] == "QC_WARN"].to_dict("records")
    print(f"retrying {len(warned)} QC_WARN cases", flush=True)
    from multiprocessing import Pool
    results = []
    with Pool(args.workers) as pool:
        for n, res in enumerate(pool.imap_unordered(process, warned), 1):
            results.append(res)
            print(f"[{n}/{len(warned)}] {res['tag']} {res['status']} {res['note']}", flush=True)
    pd.DataFrame(results).to_csv(f"{BASE}/qc/phase2_retry_report.csv", index=False)
    print(pd.DataFrame(results)["status"].value_counts().to_string(), flush=True)


if __name__ == "__main__":
    main()
