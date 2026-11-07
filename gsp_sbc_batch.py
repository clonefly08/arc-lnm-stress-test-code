#!/usr/bin/env python3
"""Phase 3c: batch seed-based-connectivity (SBC) driver — GSP1000 normative × 228 lesions.

Math identical to compneurobilbao/lnm src/lnm_functional.py (zscore within GM mask,
seed = mean lesion timeseries, Pearson r -> Fisher-z, one-sample t-test across
normative subjects), but vectorized: each normative subject is loaded ONCE and all
228 lesion seeds are computed in one matmul. Runs (bld001/bld002) are averaged in
Fisher-z space. Resumable: accumulator checkpoint every 50 subjects.

Run in WSL: python3 scripts/gsp_sbc_batch.py
Outputs:
  connmaps/gsp_sbc_zmaps_4D.nii.gz   (228 volumes, 2mm MNI grid)
  connmaps/gsp_sbc_summary.csv
  connmaps/.sbc_checkpoint.npz       (resumable)
"""
import os, glob, sys, time, traceback

os.environ.setdefault("MPLBACKEND", "Agg")
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(v, "4")

import numpy as np
import pandas as pd
import nibabel as nib
from scipy import stats

BASE = "/mnt/n/ARC/lnm"
GSP = f"{BASE}/template/gsp1000"
TAG = os.environ.get("SBC_TAG", "")
GM = os.environ.get("SBC_GM") or f"{BASE}/template/gm_mask_2mm.nii.gz"
MASKS_DIR = f"{BASE}/masks_mni"
OUT_DIR = f"{BASE}/connmaps"
CKPT = f"{OUT_DIR}/.sbc_checkpoint{TAG}.npz"
OUT_4D = f"{OUT_DIR}/gsp_sbc_zmaps_4D{TAG}.nii.gz"
OUT_CSV = f"{OUT_DIR}/gsp_sbc_summary{TAG}.csv"
os.makedirs(OUT_DIR, exist_ok=True)

CLIP = 1.0 - 1e-6
CKPT_EVERY = 50


def find_subjects():
    """Map sub-XXXX -> sorted run files (bld001, bld002, ...)."""
    runs = {}
    for p in sorted(glob.glob(f"{GSP}/sub-*/func/*_finalmask.nii.gz")):
        sid = p.split("/")[-3]
        runs.setdefault(sid, []).append(p)
    return {k: v for k, v in sorted(runs.items()) if v}


def load_lesions(gm_bool):
    """Load all mask_mni files; binarize; intersect GM; return (names, seed_masks VxL bool)."""
    rows = pd.read_csv(f"{BASE}/qc/phase2_full_report.csv")
    names, cols = [], []
    for _, r in rows.iterrows():
        p = f"{MASKS_DIR}/{r['tag']}/mask_mni.nii.gz"
        if not os.path.exists(p):
            continue
        m = np.asanyarray(nib.load(p).dataobj) > 0
        if m.shape != gm_bool.shape:
            print(f"  SKIP {r['tag']}: shape {m.shape} != {gm_bool.shape}", flush=True)
            continue
        seed = m & gm_bool
        if seed.sum() == 0:
            print(f"  SKIP {r['tag']}: no GM voxels", flush=True)
            continue
        names.append(r["tag"])
        cols.append(seed[gm_bool])
    return names, np.stack(cols, axis=1)


def process_run(path, gm_idx_bool, ref_affine):
    img = nib.load(path)
    if tuple(img.shape[:3]) != tuple(gm_idx_bool.shape):
        raise ValueError(f"grid mismatch {img.shape[:3]}")
    data = np.asanyarray(img.dataobj, dtype=np.float32)
    if data.ndim == 4:
        ts = data[gm_idx_bool]          # V x T
    else:
        raise ValueError(f"not 4D: {path}")
    del data
    ts = ts[:, np.any(np.isfinite(ts), axis=0)]
    # GSP finalmask 外的 GM 先验体素可能是 NaN/常量：归零成中性信号，防 NaN 毒化累积器
    ts = np.nan_to_num(ts, nan=0.0, posinf=0.0, neginf=0.0)
    mu = ts.mean(axis=1, keepdims=True)
    sd = ts.std(axis=1, ddof=1, keepdims=True)
    sd[sd == 0] = 1.0
    Z = (ts - mu) / sd                  # V x T, zscore_sample
    return Z


def main():
    gm_img = nib.load(GM)
    gm_bool = np.asanyarray(gm_img.dataobj) > 0
    V = int(gm_bool.sum())
    print(f"GM voxels: {V}", flush=True)
    gm_idx_bool = np.zeros(gm_bool.shape, dtype=bool)
    gm_idx_bool[gm_bool] = True
    flat_idx = np.where(gm_bool.ravel())[0]

    subjects = find_subjects()
    print(f"normative subjects found: {len(subjects)}", flush=True)

    names, seeds = load_lesions(gm_bool)          # seeds: V x L
    L = len(names)
    print(f"lesion seeds: {L}", flush=True)
    seeds_f = seeds.astype(np.float32)
    seed_counts = seeds.sum(axis=0).astype(np.float32)

    # checkpoint resume
    acc_n, acc_sum, acc_sumsq, done_ids = 0, None, None, []
    if os.path.exists(CKPT):
        z = np.load(CKPT, allow_pickle=True)
        acc_n, acc_sum, acc_sumsq = int(z["n"]), z["sum"], z["sumsq"]
        done_ids = list(z["done"])
        if acc_sum.shape[0] != L:
            print("checkpoint lesion set changed; starting fresh", flush=True)
            acc_n, acc_sum, acc_sumsq, done_ids = 0, None, None, []
    if acc_sum is None:
        acc_sum = np.zeros((L, V), dtype=np.float64)
        acc_sumsq = np.zeros((L, V), dtype=np.float64)
    done = set(done_ids)

    t0 = time.time()
    n_proc = 0
    for i, (sid, files) in enumerate(subjects.items(), 1):
        if sid in done:
            continue
        try:
            F_list = []
            for rp in files:
                Z = process_run(rp, gm_idx_bool, gm_img.affine)      # V x T (z-scored voxels)
                seed_ts = seeds_f.T @ Z / seed_counts[:, None]       # L x T (mean z in lesion)
                # true Pearson: re-standardize the seed mean series per lesion
                # (reference lnm_functional.py skips this -> lesion-size-dependent scale bias)
                mu_s = seed_ts.mean(axis=1, keepdims=True)
                sd_s = seed_ts.std(axis=1, ddof=1, keepdims=True)
                sd_s[sd_s == 0] = 1.0
                seed_z = (seed_ts - mu_s) / sd_s
                corr = (seed_z @ Z.T) / Z.shape[1]                   # L x V
                F_list.append(np.arctanh(np.clip(corr, -CLIP, CLIP)).astype(np.float32))
                del Z, corr, seed_z
            F = np.mean(F_list, axis=0)                              # L x V
            F = np.nan_to_num(F, nan=0.0, posinf=0.0, neginf=0.0)
            acc_n += 1
            acc_sum += F
            acc_sumsq += F.astype(np.float64) ** 2
            done.add(sid)
            n_proc += 1
            if n_proc % 10 == 0:
                rate = n_proc * (time.time() - t0) / max(n_proc, 1)
                print(f"[{i}/{len(subjects)}] {sid} runs={len(files)} "
                      f"elapsed={time.time()-t0:.0f}s", flush=True)
            if n_proc % CKPT_EVERY == 0:
                np.savez(CKPT, n=acc_n, sum=acc_sum, sumsq=acc_sumsq,
                                    done=np.array(sorted(done)))
                print("  checkpoint saved", flush=True)
        except Exception as e:
            print(f"  FAIL {sid}: {type(e).__name__}: {e}", flush=True)
            traceback.print_exc()

    np.savez(CKPT, n=acc_n, sum=acc_sum, sumsq=acc_sumsq,
                        done=np.array(sorted(done)))
    print(f"\naccumulated over {acc_n} subjects", flush=True)

    mean = acc_sum / acc_n
    var = acc_sumsq / acc_n - mean ** 2
    var[var < 0] = 0
    t_stat = mean / np.sqrt(var / acc_n + 1e-12)
    p_vals = stats.t.sf(np.abs(t_stat), df=acc_n - 1) * 2.0
    z_maps = stats.norm.isf(np.clip(p_vals / 2, 1e-300, 1) ).astype(np.float32)
    z_maps = np.sign(t_stat) * z_maps                        # L x V signed z

    # write 4D z-map on the 2mm grid
    gm_img = nib.load(GM)
    vol = np.zeros((gm_bool.sum(),), dtype=np.float32)
    out = np.zeros(tuple(gm_bool.shape) + (L,), dtype=np.float32)
    flat = out.reshape(-1, L)
    flat[flat_idx, :] = z_maps.T
    img4d = nib.Nifti1Image(out, gm_img.affine)
    img4d.header.set_zooms(tuple(gm_img.header.get_zooms()[:3]) + (1.0,))
    nib.save(img4d, OUT_4D)
    print(f"saved -> {OUT_4D}", flush=True)

    pd.DataFrame({
        "lesion": names,
        "n_subjects": acc_n,
        "z_max": z_maps.max(axis=1),
        "z_min": z_maps.min(axis=1),
    }).to_csv(OUT_CSV, index=False)
    print(f"saved -> {OUT_CSV}", flush=True)
    print("SBC_BATCH_DONE", flush=True)


if __name__ == "__main__":
    main()
