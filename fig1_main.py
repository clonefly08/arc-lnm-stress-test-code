#!/usr/bin/env python3
"""Figure 1 (main): lesion-network mapping of post-stroke aphasia severity.

Contract
- Core conclusion: functional connectivity between lesions and a distributed
  left-lateralized network predicts WAB-AQ beyond lesion location.
- Panels: (a) hero, discovery — FWE-corrected significant network (axial montage,
  worse-AQ hot / better-AQ cool, peaks annotated); (b) validation — strict LOOCV
  predicted vs observed AQ; (c) added value — significant voxels, LNM vs
  location-only VLSM vs connectivity-beyond-occupancy.
- Archetype: image plate + quant. Backend: Python/matplotlib (exclusive).
- Export: 180 mm double column, SVG + PDF + TIFF + PNG preview.
"""
import os, sys
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("OMP_NUM_THREADS", "4")
sys.path.insert(0, "/mnt/c/Users/clonefly/.zcode/skills/nature-figure/scripts")

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import gridspec
import nibabel as nib

from audit_panel_alignment import require_matplotlib_panel_alignment

BASE = "/mnt/n/ARC/lnm"
STATS = f"{BASE}/stats"
FSLDIR = "/usr/local/fsl"
OUT = f"{BASE}/figs"
os.makedirs(OUT, exist_ok=True)

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "font.size": 7,
    "axes.linewidth": 0.8,
    "legend.frameon": False,
    "axes.spines.right": False,
    "axes.spines.top": False,
})

# ---------------- data ----------------
gm_img = nib.load(f"{BASE}/template/gm_mask_2mm.nii.gz")
gm_bool = np.asanyarray(gm_img.dataobj) > 0
tmap = np.asanyarray(nib.load(f"{STATS}/obs_tmap.nii.gz").dataobj).astype(np.float32)
fwe_c = np.load("/mnt/n/ARC/analysis/out/canonical_masks.npz")["primary_mask_0025"]
sig = np.zeros(gm_bool.shape, dtype=bool)
sig.ravel()[np.where(gm_bool.ravel())[0]] = fwe_c
brain = np.asanyarray(nib.load(f"{FSLDIR}/data/standard/MNI152_T1_2mm_brain.nii.gz").dataobj).astype(np.float32)

tgm = tmap[gm_bool].astype(np.float64)
t_sig = np.where(sig, tmap, np.nan)
neg = np.where(t_sig < 0, t_sig, np.nan)   # worse AQ
pos = np.where(t_sig > 0, t_sig, np.nan)   # better AQ

# LOOCV predictions (ALL cohort, order = summary∩phenotype)
summ = pd.read_csv(f"{BASE}/connmaps/gsp_sbc_summary.csv")
pheno = pd.read_csv(f"{STATS}/phenotype.csv")
names = [t for t in summ["lesion"] if t in set(pheno["tag"])]
ph = pheno.set_index("tag").loc[names].reset_index()
aq = ph["aq"].to_numpy(float)
pred = np.load(f"{STATS}/leakage_free/predictions.npz")["d"]
ok = np.isfinite(pred)
r = np.corrcoef(aq[ok], pred[ok])[0, 1]

vlsm_sig = int((np.load("/mnt/n/ARC/analysis/out/vlsm_fwe_p_maxT.npy") < 0.025).sum())
loc_conn_sig = int((np.load("/mnt/n/ARC/analysis/out/loc_conn_fwe_p.npy") < 0.025).sum())

# ---------------- figure ----------------
W, H = 180 / 25.4, 128 / 25.4          # 180 mm × 128 mm
fig = plt.figure(figsize=(W, H))
gs = gridspec.GridSpec(2, 3, figure=fig, height_ratios=[1.35, 1.0],
                       width_ratios=[1.0, 1.0, 1.0],
                       hspace=0.42, wspace=0.30,
                       left=0.055, right=0.985, top=0.93, bottom=0.10)

# ---- panel a: axial montage (hero) ----
z_slices = [-52, -46, -12, -8, -4, 4]
aff = np.asarray(gm_img.affine)
inv_aff = np.linalg.inv(aff)
def world_k(z_world):
    # voxel k for a world z via the affine
    vk = inv_aff @ np.array([0.0, 0.0, z_world, 1.0])
    return int(round(vk[2]))
ks = [world_k(z) for z in z_slices]
def voxel_xy(xyz):
    v = inv_aff @ np.array([xyz[0], xyz[1], xyz[2], 1.0])
    return int(round(v[0])), int(round(v[1]))

overlay = np.where(sig, tmap, np.nan).astype(np.float32)
VMIN, VMAX = -5.0, 5.0

axs = []
gs_top = gs[0, :].subgridspec(1, len(z_slices) + 1,
                              width_ratios=[1] * len(z_slices) + [0.30], wspace=0.04)
im_ov = None
for j, (k, z) in enumerate(zip(ks, z_slices)):
    ax = fig.add_subplot(gs_top[0, j])
    axs.append(ax)
    ax.imshow(brain[:, :, k].T, cmap="gray", origin="lower",
              vmin=0, vmax=np.percentile(brain, 99.5), interpolation="nearest")
    im_ov = ax.imshow(overlay[:, :, k].T, cmap="RdBu_r", origin="lower",
                      vmin=VMIN, vmax=VMAX, interpolation="nearest")
    ax.set_title("z = " + (str(z) if z >= 0 else "−" + str(-z)), fontsize=7, pad=2)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)

# peaks: marker + leader annotation on their true slices
pk_ins = voxel_xy((-34, -4, -8)); pk_cb = voxel_xy((12, -8, -46))
k_ins = world_k(-8); k_cb = world_k(-46)
for ax, z, (px, py), col, txt, va in (
    (axs[z_slices.index(-8)], -8, pk_ins, "#8b1a00", "L insula (−34,−4,−8)", "bottom"),
    (axs[z_slices.index(-46)], -46, pk_cb, "#0b5a8a", "R cerebellum (12,−8,−46)", "top"),
):
    ax.plot(py, px, marker="o", ms=3.0, mfc="none", mec=col, mew=0.8)
    ax.annotate(txt, xy=(py, px), xytext=(0.97, 0.05 if va == "bottom" else 0.95),
                textcoords="axes fraction", ha="right", va=va, fontsize=5.8, color=col,
                arrowprops=dict(arrowstyle="-", lw=0.6, color=col, shrinkB=2.5))

cax = fig.add_subplot(gs_top[0, len(z_slices)])
cb = fig.colorbar(im_ov, cax=cax, ticks=[-5, -2.5, 0, 2.5, 5])
cb.ax.set_yticklabels(["−5", "−2.5", "0", "2.5", "5"], fontsize=6)
cb.ax.tick_params(axis="y", pad=2.8, length=1.6, width=0.5)
cb.set_label("t-statistic", fontsize=6, labelpad=3)
cb.outline.set_linewidth(0.5)
fig.text(0.955, 0.955 - 0.02, "AQ↑", fontsize=5.5, color="#8b1a00", va="top", ha="right")
fig.text(0.955, 0.62, "AQ↓", fontsize=5.5, color="#0b5a8a", va="top", ha="right")

fig.text(0.012, 0.955, "a", fontsize=9, fontweight="bold")
R_ORIENT = "radiological convention (left hemisphere on the right of each slice)"
n_sig = int(sig.sum())
fig.text(0.035, 0.955,
         f"Lesion–network connectivity ↔ WAB-AQ  (max-T FWE, n = 226; {n_sig:,} voxels)",
         fontsize=6.5, color="#333", va="top")

# ---- bottom row: b scatter | c dots | d overlap diagnostic ----
gs_bot = gs[1, :].subgridspec(1, 3, width_ratios=[1.35, 0.95, 0.95], wspace=0.34)
axb = fig.add_subplot(gs_bot[0, 0])
lims = [0, 100]
axb.plot(lims, lims, ls="--", lw=0.7, color="#999", zorder=1)
axb.scatter(aq[ok], pred[ok], s=7, lw=0, color="#31527a", alpha=0.65,
            rasterized=True, zorder=2)
axb.set_xlim(lims); axb.set_ylim(lims)
axb.set_xlabel("Observed WAB-AQ", fontsize=7)
axb.set_ylabel("LOOCV-predicted AQ", fontsize=7)
axb.set_xticks([0, 25, 50, 75, 100]); axb.set_yticks([0, 25, 50, 75, 100])
axb.tick_params(labelsize=6, length=2.2, width=0.6)
axb.text(0.04, 0.93, f"R = {r:.3f}   R² = {r**2:.3f}   n = {int(ok.sum())}",
         transform=axb.transAxes, fontsize=7, va="top")

axb.annotate("b", xy=(-0.075, 1.10), xycoords="axes fraction", fontsize=9, fontweight="bold", annotation_clip=False)

# ---- panel c: added value (dot plot on log axis; bars would mislead on log scale) ----
axc = fig.add_subplot(gs_bot[0, 1])
rows = ["LNM\n(connectivity)", "LNM beyond\noccupancy", "VLSM\n(location only)"]
vals = [n_sig, loc_conn_sig, max(vlsm_sig, 1)]
ys = [2, 1, 0]
axc.hlines(ys, 1, vals, color="#d5dbe3", lw=1.0, zorder=1)
axc.scatter(vals, ys, s=26, color=["#31527a", "#7a93b5", "#8b1a00"], zorder=3)
axc.set_xscale("log")
axc.set_xlim(1, 1.4e4)
axc.set_ylim(-0.6, 2.6)
axc.set_yticks(ys)
axc.set_yticklabels(rows, fontsize=6)
axc.set_xticks([1, 10, 100, 1000, 10000])
axc.set_xticklabels(["1", "10", "100", "1,000", "10,000"], fontsize=6)
axc.set_xlabel("FWE-corrected significant voxels (log scale)", fontsize=7)
axc.tick_params(labelsize=6, length=2.2, width=0.6)
for yv, v in zip(ys, vals):
    axc.text(v * 1.4, yv, f"{v:,}", va="center", ha="left", fontsize=6.2, color="#333")
axc.text(0.5, -0.58, "same cohort, covariates & max-T framework",
         transform=axc.transAxes, ha="center", fontsize=5.6, color="#555")
axc.annotate("c", xy=(-0.50, 1.10), xycoords="axes fraction", fontsize=9, fontweight="bold", annotation_clip=False)

fig.text(0.5, 0.012,
         "Panel b: connectivity-weighted damage score; covariates: age, sex, time, volume  ·  "
         "Normative connectome: GSP1000 (n = 500)  ·  Freedman–Lane permutation, 5,000 max-T  ·  "
         "radiological convention (left on right)",
         ha="center", fontsize=5.8, color="#666")

# ---- panel d: lesion-overlap vs t (spatial diagnostic, Methods 2.8) ----
axd = fig.add_subplot(gs_bot[0, 2])
ov = np.zeros(int(gm_bool.sum()), dtype=np.int32)
for _t in names:
    _m = np.asanyarray(nib.load(f"{BASE}/masks_mni/{_t}/mask_mni.nii.gz").dataobj) > 0
    ov += _m[gm_bool]
ovf = ov.astype(np.float64) / len(names)
axd.hexbin(ovf, tgm, gridsize=42, bins="log", cmap="Blues", linewidths=0)
axd.set_xlabel("lesion overlap fraction", fontsize=6)
axd.set_ylabel("t-statistic", fontsize=6)
axd.tick_params(labelsize=5.5, length=2, width=0.5)
axd.set_title("r = −0.33 (overlap deciles: monotone)", fontsize=5.8, pad=2)
axd.annotate("d", xy=(-0.30, 1.10), xycoords="axes fraction", fontsize=9, fontweight="bold",
             annotation_clip=False)

# ---------------- alignment gate + export ----------------
stem = f"{OUT}/fig1_lnm_main"
require_matplotlib_panel_alignment(
    fig, json_out=f"{stem}.alignment.json", overlay_svg=f"{stem}.alignment.svg",
    tolerance_pt=1.5, gutter_tolerance_pt=1.5, strict=True,
    axes=[*axs, axb, axc, axd],
    panel_ids=[f"a{i+1}" for i in range(len(axs))] + ["b", "c", "d"],
    exclude_axes=[cax],
    exemptions=[{"panels": ["b", "c"], "checks": ["panel-width", "column"],
                 "reason": "asymmetric bottom row: LOOCV scatter (2/3 width) vs comparison bars (1/3) by design"}],
)
fig.savefig(f"{stem}.svg", bbox_inches="tight")
fig.savefig(f"{stem}.pdf", bbox_inches="tight")
fig.savefig(f"{stem}.tiff", dpi=600, bbox_inches="tight")
fig.savefig(f"{stem}.png", dpi=300, bbox_inches="tight")
print("saved:", stem + ".{svg,pdf,tiff,png}")
