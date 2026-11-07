#!/usr/bin/env python3
"""Phase 0: ARC lesion mask audit — volume, laterality, components, T2w intensity QC.

Runs in WSL; reads E: via /mnt/e, writes N: via /mnt/n. Single process, capped threads.
Output: /mnt/n/ARC/lnm/qc/phase0_mask_audit.csv + phase0_png/*.png
"""
import os, glob, traceback, sys

os.environ.setdefault("MPLBACKEND", "Agg")
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(v, "4")

import numpy as np
import pandas as pd
import nibabel as nib
from scipy import ndimage
import matplotlib.pyplot as plt

DS = "/mnt/e/download/ds004884-download"
OUT_CSV = "/mnt/n/ARC/lnm/qc/phase0_mask_audit.csv"
PNG_DIR = "/mnt/n/ARC/lnm/qc/phase0_png"
os.makedirs(PNG_DIR, exist_ok=True)


def source_t2w(mask_path):
    """Mask: <DS>/derivatives/lesion_masks/<sub>/<ses>/anat/... ; source: <DS>/<sub>/<ses>/anat/<stem>.nii.gz"""
    parts = mask_path.split("/")
    i = max(j for j, p in enumerate(parts[:-1]) if p.startswith("sub-"))
    stem = os.path.basename(mask_path).replace("_desc-lesion_mask.nii.gz", "")
    ds = parts[: parts.index("derivatives")]
    cand = "/".join(ds + [parts[i], parts[i + 1], "anat", stem + ".nii.gz"])
    return cand if os.path.exists(cand) else None


def world_x_of_voxels(affine, idx):
    homo = np.vstack([idx, np.ones(idx.shape[1], dtype=np.float64)])
    return (affine @ homo)[0]


def qc_png(t2w, mask, sub, ses):
    if t2w is None:
        return
    try:
        t2w = np.asanyarray(t2w.dataobj).astype(np.float32)
        i, j, k = [int(round(c)) for c in ndimage.center_of_mass(mask)]
        nz = t2w > 0
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        for ax, sl_t, sl_m, ttl in (
            (axes[0], t2w[:, :, k], mask[:, :, k], f"axial k={k}"),
            (axes[1], t2w[:, j, :], mask[:, j, :], f"coronal j={j}"),
            (axes[2], t2w[i, :, :], mask[i, :, :], f"sagittal i={i}"),
        ):
            ax.imshow(sl_t.T, cmap="gray", origin="lower",
                      vmin=np.percentile(sl_t[sl_t > 0], 5) if (sl_t > 0).any() else 0,
                      vmax=np.percentile(sl_t[sl_t > 0], 99.5) if (sl_t > 0).any() else 1)
            if sl_m.any():
                ax.contour(sl_m.T, levels=[0.5], colors="red", linewidths=0.8, origin="lower")
            ax.set_title(ttl, fontsize=8)
            ax.axis("off")
        fig.suptitle(f"{sub} {ses}", fontsize=10)
        fig.tight_layout()
        fig.savefig(os.path.join(PNG_DIR, f"{sub}_{ses}.png"), dpi=80)
        plt.close(fig)
    except Exception as e:
        print(f"  PNG failed for {sub} {ses}: {type(e).__name__}: {e}", flush=True)


def main():
    masks = sorted(glob.glob(f"{DS}/derivatives/lesion_masks/sub-*/ses-*/anat/*_T2w_desc-lesion_mask.nii.gz"))
    print(f"found {len(masks)} masks", flush=True)
    rows = []
    for n, mp in enumerate(masks, 1):
        sub = mp.split("/")[-4]
        ses = mp.split("/")[-3]
        row = dict(sub=sub, ses=ses, mask_path=mp, flags="")
        try:
            m_img = nib.load(mp)
            m = np.asanyarray(m_img.dataobj)
            binary = set(np.unique(m).tolist()) <= {0, 1} or set(np.unique(m).tolist()) <= {0, 255}
            mb = m > 0
            nvox = int(mb.sum())
            zooms = np.asarray(m_img.header.get_zooms()[:3], dtype=float)
            vol_mm3 = float(nvox * np.prod(zooms))

            idx = np.array(np.nonzero(mb))
            comp_lab, n_comp = ndimage.label(mb, structure=np.ones((3, 3, 3), dtype=int))
            comp_sizes = np.bincount(comp_lab.ravel())[1:]
            largest_frac = float(comp_sizes.max() / nvox) if nvox else np.nan

            axc = nib.aff2axcodes(m_img.affine)
            wx = world_x_of_voxels(np.asarray(m_img.affine, dtype=float), idx)
            # +x direction meaning: 'R' -> +x is right; 'L' -> +x is left
            left_sign = -1.0 if axc[0] == "R" else 1.0
            frac_left_x0 = float(np.mean(left_sign * wx > 0))
            frac_right_x0 = float(np.mean(left_sign * wx < 0))
            com_x0 = float(np.mean(left_sign * wx))
            if frac_left_x0 >= 0.95:
                side_x0 = "L"
            elif frac_right_x0 >= 0.95:
                side_x0 = "R"
            elif abs(com_x0) > 10:
                side_x0 = "BL" if com_x0 < 0 else "BR"
            else:
                side_x0 = "MID"
        except Exception:
            traceback.print_exc()
            rows.append({**row, "flags": "LOAD_FAIL"})
            continue

        t2w_path = source_t2w(mp)
        hyper_ratio = np.nan
        if t2w_path:
            try:
                t_img = nib.load(t2w_path)
                t2w = np.asanyarray(t_img.dataobj).astype(np.float32)
                nz = t2w > 0
                if nz.any() and nvox:
                    med = float(np.median(t2w[nz]))
                    hyper_ratio = float(t2w[mb & nz].mean() / med) if med > 0 else np.nan
                    # midline proxy: intensity-weighted head COM
                    try:
                        com_vox = np.asarray(ndimage.center_of_mass(t2w), dtype=float).reshape(3, 1)
                        head_com_x = float(np.mean(left_sign * world_x_of_voxels(
                            np.asarray(m_img.affine, dtype=float), com_vox)))
                    except Exception:
                        head_com_x = np.nan
                else:
                    head_com_x = np.nan
                del t2w, nz
            except Exception:
                traceback.print_exc()
                head_com_x = np.nan
                t2w_path = None
        else:
            head_com_x = np.nan

        flags = []
        if not binary: flags.append("NOT_BINARY")
        if nvox == 0: flags.append("EMPTY")
        if vol_mm3 < 100: flags.append("TINY<100mm3")
        if vol_mm3 > 200000: flags.append("HUGE>200mL")
        if n_comp > 1: flags.append(f"MULTI_COMP({n_comp})")
        if largest_frac < 0.8 and n_comp > 1: flags.append("FRAGMENTED")
        if side_x0 in ("MID", "BL", "BR"): flags.append(f"NON_UNILAT({side_x0})")
        if t2w_path and np.isfinite(hyper_ratio) and hyper_ratio < 1.1: flags.append("NOT_HYPERINT")
        if not t2w_path: flags.append("NO_SRC_T2W")

        row.update(dict(
            t2w_path=t2w_path or "", found_t2w=bool(t2w_path),
            vox_n=nvox, vol_mm3=round(vol_mm3, 1), vol_mL=round(vol_mm3 / 1000, 2),
            shape="x".join(map(str, m_img.shape)), voxel_size="x".join(f"{z:.2f}" for z in zooms),
            dtype=str(m.dtype), binary=binary, n_components=int(n_comp),
            largest_frac=round(largest_frac, 3),
            axcodes="".join(axc), frac_left_x0=round(frac_left_x0, 3),
            frac_right_x0=round(frac_right_x0, 3), com_x0_signed=round(com_x0, 1),
            side_x0=side_x0, head_com_x=round(head_com_x, 1) if np.isfinite(head_com_x) else np.nan,
            hyper_ratio=round(hyper_ratio, 2) if np.isfinite(hyper_ratio) else np.nan,
            flags=";".join(flags),
        ))
        rows.append(row)
        if t2w_path:
            qc_png(nib.load(t2w_path), mb, sub, ses)

        if n % 20 == 0 or n == len(masks):
            print(f"  [{n}/{len(masks)}] last={sub} {ses} vol={vol_mm3:.0f}mm3 side={side_x0}", flush=True)
        del m, mb
        import gc; gc.collect()

    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)

    print("\n===== SUMMARY =====", flush=True)
    print(f"masks: {len(df)}  unique subjects: {df['sub'].nunique()}")
    dup = df[df.duplicated("sub", keep=False)]
    if len(dup):
        print(f"DUPLICATE subjects (multi-session masks): {sorted(dup['sub'].unique())}")
    print("\nside distribution (x0 midline):\n", df.side_x0.value_counts().to_string())
    v = df.vol_mm3.describe(percentiles=[.05, .25, .5, .75, .95])
    print("\nvolume mm3:\n", v.to_string())
    print("\nflag counts:")
    for f in sorted({fl for s in df.flags.fillna("") for fl in s.split(";") if fl}):
        print(f"  {f}: {df.flags.fillna('').str.contains(f, regex=False).sum()}")
    agree = (df.side_x0.isin(["L", "R"])) & (
        (df.head_com_x < 0) == (df.side_x0 == "L"))
    print(f"\nside vs head-COM-midline agreement: {agree.sum()}/{agree.notna().sum()} among unilateral")
    hr = df.hyper_ratio.dropna()
    print(f"hyper ratio: median={hr.median():.2f}  <1.1: {(hr < 1.1).sum()}  min={hr.min():.2f}")
    print(f"\ncsv -> {OUT_CSV}\npngs -> {PNG_DIR}")


if __name__ == "__main__":
    main()
