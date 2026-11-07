#!/usr/bin/env python3
"""Figure 3 (paper): stress-test panel. Layout 2x2:
(a) significant voxels across analysis variants, sorted descending;
(b) leakage-free LOOCV R across scores;
(c) degeneracy diagnostic histogram;
(d) composition-entropy vs log-volume scatter with low-entropy tertile."""
import os, sys, json
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("OMP_NUM_THREADS", "4")
sys.path.insert(0, "/mnt/c/Users/clonefly/.zcode/skills/nature-figure/scripts")
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from audit_panel_alignment import require_matplotlib_panel_alignment
from matplotlib.ticker import FuncFormatter

BASE = "/mnt/n/ARC/lnm"
STATS = f"{BASE}/stats"
OUT = f"{BASE}/figs"

mpl.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "svg.fonttype": "none", "pdf.fonttype": 42, "font.size": 7,
    "axes.linewidth": 0.8, "legend.frameon": False,
    "axes.spines.right": False, "axes.spines.top": False,
})

n_base = int((np.load(f"{STATS}/fwe_p_maxT.npy") < 0.05).sum())
n_gm04 = int((np.load(f"{STATS}/fwe_p_maxT_gm04.npy") < 0.05).sum())
pred = pd.read_csv(f"{STATS}/phase5_prediction.csv")
deg = pd.read_csv(f"{STATS}/degeneracy.csv")

# cohort rows aligned with the connectivity checkpoint (summary ∩ phenotype order)
summ_df = pd.read_csv(f"{BASE}/connmaps/gsp_sbc_summary.csv")
ph0 = pd.read_csv(f"{STATS}/phenotype.csv")
names = [t for t in summ_df["lesion"] if t in set(ph0["tag"])]
phv = ph0.set_index("tag").loc[names].reset_index()
vol = phv["vol_mm3"].to_numpy(float)
H7 = np.load(f"{STATS}/revision/H_yeo.npy")
r_H_vol = float(np.corrcoef(H7, np.log10(np.maximum(vol, 1.0)))[0, 1])
tert = float(np.quantile(H7, 1 / 3))
n_low_pat = int((H7 <= tert).sum())

W, H = 180 / 25.4 * 0.999, 112 / 25.4
fig, axes = plt.subplots(2, 2, figsize=(W, H),
                         gridspec_kw=dict(left=0.115, right=0.985, top=0.955, bottom=0.095,
                                          wspace=0.42, hspace=0.62))
axa, axb, axc, axd = axes.ravel()

# a) voxels across variants from the canonical rerun (single permutation stream)
NL = chr(10)
CN = json.load(open("/mnt/n/ARC/analysis/out/canonical_numbers.json"))
n_base = CN["primary"]["sig_0025"]
n_Lonly = CN["left_only"]["sig_0025"]
n_gm04 = CN["gm04"]["sig_0025"]
n_vollog = CN["log_volume"]["sig_0025"]
n_nocere = CN["drop_cerebellar"]["sig_0025"]
n_occ = int((np.load("/mnt/n/ARC/analysis/out/loc_conn_fwe_p.npy") < 0.025).sum())
n_low_vox = CN["C3_low_entropy"]["sig_0025"]
labels = ["Baseline" + NL + "(n=226)", "Left-only" + NL + "(n=219)", "GM prior" + NL + ">0.4" + NL + "(n=225)", "Log" + NL + "volume" + NL + "(n=226)", "Drop cereb." + NL + "(n=194)", "Occupancy" + NL + "joint (n=226)", "Low-entropy" + NL + "tertile (n=76)"]
vals = [n_base, n_Lonly, n_gm04, n_vollog, n_nocere, n_occ, n_low_vox]
order = [5, 6, 3, 4, 2, 1, 0]  # bottom→top: occ(62), low(0), log(36), nocere(0)... sorted ascending
pairs = sorted(zip(vals, labels), key=lambda x: x[0])
vals_s = [v for v, _ in pairs]; labels_s = [l for _, l in pairs]
RED, B1, B2, B3 = "#b5563f", "#31527a", "#7a93b5", "#9db4cd"
colmap = {}
for lab, v in zip(labels, vals):
    colmap[lab] = RED if ("Occupancy" in lab or "Low-entropy" in lab or "Log" in lab or "cereb" in lab) else (B1 if lab.startswith("Baseline") else (B2 if "Left" in lab else B3))
bars = axa.barh(labels_s, vals_s, color=[colmap[l] for l in labels_s], height=0.62)
for b, v in zip(bars, vals_s):
    axa.text(v + max(vals_s) * 0.02, b.get_y() + b.get_height() / 2, f"{v:,}",
             va="center", ha="left", fontsize=6.3)
axa.xaxis.set_major_formatter(FuncFormatter(lambda v, _: "{:,}".format(int(v))))
axa.set_xlabel("Significant voxels (per-tail FWE < 0.025)", fontsize=7)
axa.set_xlim(0, max(vals_s) * 1.24)
axa.tick_params(labelsize=5.8, length=2.2, width=0.6)
axa.annotate("a", xy=(-0.38, 1.10), xycoords="axes fraction", fontsize=9, fontweight="bold")

# b) LOOCV R across scores
import json
_r = json.load(open(f"{STATS}/leakage_free/leakage_free_results.json"))
tiers = ["Covariates" + NL + "only", "Connectivity" + NL + "weighted", "Lesion" + NL + "overlap", "Negative" + NL + "lobe"]
rs = [_r["covariates"]["R"], _r["conn_weighted"]["R"], _r["lesion_overlap"]["R"], _r["neg_lobe"]["R"]]
bars = axb.bar(tiers, rs, color=["#c9cdd4", "#31527a", "#7a93b5", "#9db4cd"], width=0.58)
for b, v in zip(bars, rs):
    axb.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.3f}", ha="center", va="bottom", fontsize=6.5)
axb.set_ylabel("LOOCV R (AQ)", fontsize=7)
axb.set_ylim(0, 0.80)
axb.tick_params(labelsize=6, length=2.2, width=0.6)
axb.annotate("b", xy=(-0.28, 1.10), xycoords="axes fraction", fontsize=9, fontweight="bold")

# c) degeneracy histogram
cc = deg["corr_with_groupmean"].to_numpy()
axc.hist(cc, bins=24, color="#7a93b5", lw=0)
med = float(np.median(cc))
axc.axvline(med, color="#8b1a00", lw=1.0)
axc.text(med - 0.02, axc.get_ylim()[1] * 0.92, f"median = {med:.3f}",
         ha="right", fontsize=6.3, color="#8b1a00")
axc.xaxis.set_major_formatter(FuncFormatter(lambda v, _: ("−" if v < 0 else "") + ("%g" % abs(v))))
axc.set_xlabel("corr(patient map, group mean)", fontsize=7)
axc.set_ylabel("Patients", fontsize=7)
axc.tick_params(labelsize=6, length=2.2, width=0.6)
axc.annotate("c", xy=(-0.30, 1.10), xycoords="axes fraction", fontsize=9, fontweight="bold")

# d) composition entropy vs log-volume
x = np.log10(np.maximum(vol, 1.0))
y = H7
ylo = y.min() - 0.08
axd.axhspan(ylo, tert, color="#f5e3de", zorder=0)
axd.scatter(x, y, s=7, lw=0, color="#31527a", alpha=0.65, zorder=2, rasterized=True)
coef = np.polyfit(x, y, 1)
xs = np.linspace(x.min(), x.max(), 50)
axd.plot(xs, np.polyval(coef, xs), color="#8b1a00", lw=0.9, ls="-", zorder=3)
axd.axhline(tert, color="#8b1a00", lw=0.8, ls="--", zorder=3)
axd.set_xlabel(r"log$_{10}$ lesion volume (mL)", fontsize=7)
axd.set_ylabel("Composition entropy H", fontsize=7)
axd.set_xlim(x.min() - 0.05, x.max() + 0.05)
axd.set_ylim(ylo, y.max() + 0.08)
axd.tick_params(labelsize=6, length=2.2, width=0.6)
axd.text(0.04, 0.97, f"r = {r_H_vol:.3f}", transform=axd.transAxes, fontsize=6.5, va="top")
axd.text(0.04, 0.30, f"low-entropy tertile (n = {n_low_pat}):"
         + NL + "0 voxels survive FWE",
         transform=axd.transAxes, fontsize=5.8, va="bottom", color="#8b1a00")
axd.annotate("d", xy=(-0.28, 1.10), xycoords="axes fraction", fontsize=9, fontweight="bold")

stem = f"{OUT}/fig2_robustness"
require_matplotlib_panel_alignment(
    fig, json_out=f"{stem}.alignment.json", overlay_svg=f"{stem}.alignment.svg",
    tolerance_pt=1.5, gutter_tolerance_pt=1.5, strict=True,
    panel_ids=["a", "b", "c", "d"],
)
fig.savefig(f"{stem}.svg", bbox_inches="tight")
fig.savefig(f"{stem}.pdf", bbox_inches="tight")
fig.savefig(f"{stem}_hi.png", dpi=450, bbox_inches="tight")
fig.savefig(f"{stem}.png", dpi=300, bbox_inches="tight")
print("saved:", stem)
