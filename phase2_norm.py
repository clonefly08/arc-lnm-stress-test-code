#!/usr/bin/env python3
"""Phase 2 (container edition): normalize 228 lesion masks (T2w native) to MNI152 2mm.

Runs INSIDE the compneuro-dwiproc container:
  docker run --rm --init --cpus 15 --memory 28g \
    -e TMPDIR=/work/tmp \
    -v "E:/download/ds004884-download:/ds:ro" \
    -v "N:/ARC/lnm:/work" \
    compneurobilbaolab/compneuro-dwiproc:1.0.0 \
    python3 /work/scripts/phase2_norm.py [--smoke] [--workers 7]

Chain per subject (ANTs throughout, lesion metric-masked with -x):
  1. enantiomorphic mirror-fill of lesion on T2 (and T1 when grids match)
  2. bet T1 (FSL, robust)
  3. antsRegistration: T2_filled -> T1_brain, Rigid+MI, -x [T1brainmask, lesion]
  4. antsRegistration: T1_brain -> MNI152, Rigid+Affine+SyN, -x [MNIbrainmask, lesion_in_T1]
  5. antsApplyTransforms: T2 mask -> MNI (composite, nearest neighbour)
  6. QC: volume ratio, in-brain fraction; per-subject log; skip-if-done
"""
import os, sys, glob, subprocess, argparse, traceback

os.environ.setdefault("MPLBACKEND", "Agg")
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS", "NSLOTS"):
    os.environ.setdefault(v, "1")

import numpy as np
import pandas as pd
import nibabel as nib
from scipy import ndimage

DS = os.environ.get("LNM_DS", "/ds")
BASE = os.environ.get("LNM_BASE", "/work")
OUT = f"{BASE}/masks_mni"
QC_DIR = f"{BASE}/qc/phase2_png"
TMP = f"{BASE}/tmp"
os.makedirs(OUT, exist_ok=True)
os.makedirs(QC_DIR, exist_ok=True)
os.makedirs(TMP, exist_ok=True)
os.environ["TMPDIR"] = TMP


def find_tool(name):
    # non-login shell on purpose: container env PATH carries /opt/ants-2.4.1;
    # `bash -l` would source /etc/profile and reset PATH, hiding ANTs
    p = subprocess.run(["bash", "-c", f"command -v {name}"], capture_output=True, text=True)
    return p.stdout.strip()


def find_fSLdir():
    for c in (os.environ.get("FSLDIR"), "/usr/local/fsl", "/opt/fsl", "/usr/share/fsl"):
        if c and os.path.isdir(c):
            return c
    return None


FSLDIR = find_fSLdir()
if FSLDIR:
    os.environ["FSLDIR"] = FSLDIR
    os.environ["FSLOUTPUTTYPE"] = "NIFTI_GZ"  # force: image env may preset uncompressed NIFTI
    os.environ["PATH"] = f"{FSLDIR}/bin:{os.environ['PATH']}"
MNI = os.environ.get("LNM_MNI") or (f"{FSLDIR}/data/standard/MNI152_T1_2mm_brain" if FSLDIR else "")
ANTS = find_tool("antsRegistration")
BET = find_tool("bet")
if not ANTS:
    sys.exit("FATAL: antsRegistration not found in container")
if not BET:
    sys.exit("FATAL: bet (FSL) not found in container")
if not os.path.exists(MNI + ".nii.gz") or not os.path.exists(MNI + "_mask.nii.gz"):
    sys.exit(f"FATAL: MNI152 standard brain/mask missing under {MNI}")
MNI = MNI + ".nii.gz"
MNI_MASK = MNI.replace("_brain.nii.gz", "_brain_mask.nii.gz")
print(f"FSLDIR={FSLDIR}  ANTs={ANTS}  MNI={MNI}", flush=True)


def run(cmd, log_path):
    with open(log_path, "a") as f:
        f.write("\n$ " + " ".join(cmd) + "\n")
        f.flush()
        env = dict(os.environ, TMPDIR=TMP)
        p = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, env=env)
    if p.returncode != 0:
        raise RuntimeError(f"cmd failed rc={p.returncode}: {' '.join(cmd)}")


T1_INV = {}


def _init_worker(inv):
    global T1_INV
    T1_INV = inv


def pick_t1(sub, ses):
    """Same-session T1w preferred; else temporally nearest (ses ids are day numbers).
    Uses the single-pass inventory built in main(); mount re-enumeration only as last resort."""
    inv = T1_INV.get(sub) or []
    same = sorted(p for s, p in inv if s == ses)
    if same:
        return same[0], True
    try:
        s0 = int(ses)
        cands = sorted((abs(int(s) - s0), s, p) for s, p in inv if s != ses)
    except ValueError:
        cands = sorted((0, s, p) for s, p in inv)
    if cands:
        return cands[0][2], False
    for t1 in glob.glob(f"{DS}/{sub}/ses-*/anat/{sub}_ses-*_T1w.nii.gz"):
        return t1, False
    return None, False


def mirror_fill(img_path, mask_path, out_path, dilate=1):
    img = nib.load(img_path)
    msk = nib.load(mask_path)
    a = np.asanyarray(img.dataobj).astype(np.float32)
    m = (np.asanyarray(msk.dataobj) > 0)
    if dilate:
        m = ndimage.binary_dilation(m, iterations=dilate)
    filled = np.where(m, a[:, ::-1, :], a)
    nib.save(nib.Nifti1Image(filled, img.affine, img.header), out_path)


def com(a):
    return ndimage.center_of_mass(a)


def process(row):
    sub, ses = str(row["sub"]), str(row["ses"])
    tag = f"{sub}_{ses}"
    wd = f"{OUT}/{tag}"
    log_path = f"{wd}/log.txt"
    status = {"tag": tag, "status": "", "vol_native": row["vol_mm3"], "vol_mni": np.nan,
              "vol_ratio": np.nan, "in_brain_frac": np.nan, "t1_ses": "", "note": ""}
    if os.path.exists(f"{wd}/mask_mni.nii.gz") and os.path.exists(f"{wd}/qc_pass.txt"):
        status["status"] = "SKIPPED(done)"
        return status
    os.makedirs(wd, exist_ok=True)
    try:
        open(log_path, "w").close()

        def to_c(p):
            # audit CSV stores WSL-view paths (/mnt/e/...); container sees the dataset at /ds
            return p.replace("/mnt/e/download/ds004884-download", DS)

        mask_src, t2_src = to_c(row["mask_path"]), to_c(row["t2w_path"])
        mask = f"{wd}/mask_std.nii.gz"
        t2 = f"{wd}/t2_std.nii.gz"
        run(["fslreorient2std", mask_src, mask], log_path)
        run(["fslreorient2std", t2_src, t2], log_path)
        t1_path, _ = pick_t1(sub, ses)
        if t1_path is None:
            status.update(status="FAIL", note="no T1w anywhere")
            return status
        status["t1_ses"] = os.path.basename(t1_path).split("_")[1]
        t1 = f"{wd}/t1_std.nii.gz"
        run(["fslreorient2std", t1_path, t1], log_path)

        mirror_fill(t2, mask, f"{wd}/t2_filled.nii.gz", dilate=1)
        if nib.load(t1).shape == nib.load(t2).shape:
            mirror_fill(t1, mask, f"{wd}/t1_filled.nii.gz", dilate=1)
            t1_reg = f"{wd}/t1_filled.nii.gz"
        else:
            t1_reg = t1
            status["note"] += "no_t1_fill(grid_mismatch);"

        run(["bet", t1_reg, f"{wd}/t1_brain", "-R", "-f", "0.45"], log_path)
        # this image's bet does not emit the mask file; derive it (brain image is zero outside)
        if not os.path.exists(f"{wd}/t1_brain_mask.nii.gz"):
            run(["fslmaths", f"{wd}/t1_brain.nii.gz", "-bin", "-fillh",
                 f"{wd}/t1_brain_mask.nii.gz"], log_path)

        # --- T2 -> T1 rigid (metric masked: fixed=T1 brain mask, moving=lesion) ---
        run([ANTS,
             "--dimensionality", "3", "--float", "0",
             "--output", f"[{wd}/t2_to_t1_,{wd}/t2_in_t1.nii.gz]",
             "--initial-moving-transform", f"[{wd}/t1_brain.nii.gz,{wd}/t2_filled.nii.gz,1]",
             "--transform", "Rigid[0.1]",
             "--metric", f"MI[{wd}/t1_brain.nii.gz,{wd}/t2_filled.nii.gz,1,32,Regular,0.25]",
             "--convergence", "[2000x2000x2000,1e-6,10]",
             "--shrink-factors", "4x2x1", "--smoothing-sigmas", "2x1x0vox",
             "-x", f"[{wd}/t1_brain_mask.nii.gz,{mask}]",
             ], log_path)
        run(["antsApplyTransforms", "-i", mask, "-r", f"{wd}/t1_brain.nii.gz",
             "-t", f"{wd}/t2_to_t1_0GenericAffine.mat", "-n", "NearestNeighbor",
             "-o", f"{wd}/mask_t1.nii.gz"], log_path)

        # --- T1 -> MNI rigid+affine+SyN (lesion metric-masked) ---
        run([ANTS,
             "--dimensionality", "3", "--float", "0",
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
             "-x", f"[{MNI_MASK},{wd}/mask_t1.nii.gz]",
             ], log_path)

        # --- mask -> MNI (final stage leftmost) ---
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
        ratio = vol_mni / max(status["vol_native"], 1)
        status.update(vol_mni=round(vol_mni, 1), vol_ratio=round(ratio, 3),
                      in_brain_frac=round(in_brain, 3))
        problems = []
        if mv.sum() == 0:
            problems.append("empty")
        if not 0.7 < ratio < 1.4:
            problems.append(f"vol_ratio={ratio:.2f}")
        if in_brain < 0.90:
            problems.append(f"in_brain={in_brain:.2f}")
        if problems:
            status.update(status="QC_WARN", note=";".join([status["note"]] + problems))
        else:
            status.update(status="PASS")
            open(f"{wd}/qc_pass.txt", "w").write("ok\n")
    except Exception as e:
        status["status"] = "FAIL"
        status["note"] += f"{type(e).__name__}: {e}"
        traceback.print_exc()
    return status


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--workers", type=int, default=7)
    ap.add_argument("--tags", type=str, default="",
                    help="comma-separated sub-XXX_ses-YYY list to (re)process")
    args = ap.parse_args()

    df = pd.read_csv(f"{BASE}/qc/phase0_mask_audit.csv")
    df["ses"] = df["ses"].astype(str)
    if args.tags:
        tags = [t.strip() for t in args.tags.split(",") if t.strip()]
        subs = [t.split("_ses-")[0] for t in tags]
        sess = ["ses-" + t.split("_ses-")[1] for t in tags]  # audit CSV keeps the "ses-" prefix
        df = df[df["sub"].isin(subs) & df["ses"].isin(sess)]
        pairs = set(zip(subs, sess))
        df = df[[ (r["sub"], r["ses"]) in pairs for _, r in df.iterrows() ]]
        print(f"TAGS mode: {len(df)} rows", flush=True)
    if args.smoke:
        keep = ["sub-M2001", "sub-M2131", "sub-M2150", "sub-M2269"]
        df = df[df["sub"].isin(keep)]
        print("SMOKE subjects:", df["sub"].tolist(), flush=True)

    # single-pass T1w inventory (avoids per-worker mount enumeration flakes)
    t1_inv = {}
    for p in glob.glob(f"{DS}/sub-*/ses-*/anat/*_T1w.nii.gz"):
        parts = p.split("/")
        t1_inv.setdefault(parts[-4], []).append((parts[-3].replace("ses-", ""), p))
    print(f"T1 inventory: {sum(len(v) for v in t1_inv.values())} files / {len(t1_inv)} subjects", flush=True)

    rows = df.to_dict("records")
    from multiprocessing import Pool
    results = []
    with Pool(args.workers, initializer=_init_worker, initargs=(t1_inv,)) as pool:
        for n, res in enumerate(pool.imap_unordered(process, rows), 1):
            results.append(res)
            print(f"[{n}/{len(rows)}] {res['tag']} {res['status']} {res['note']}", flush=True)

    out_csv = f"{BASE}/qc/phase2_norm_report.csv"
    pd.DataFrame(results).to_csv(out_csv, index=False)
    st = pd.DataFrame(results)["status"].value_counts()
    print("\nstatus:\n" + st.to_string(), flush=True)
    print("report ->", out_csv, flush=True)


if __name__ == "__main__":
    main()
