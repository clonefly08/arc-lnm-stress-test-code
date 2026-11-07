#!/usr/bin/env python3
"""Re-threshold all existing max-T FWE arrays at per-tail alpha = 0.025 (total FWER <= 0.05),
alongside the current 0.05/tail convention. No permutations are re-run: the stored null
distributions already support the stricter percentile. Note the stored arrays are
p-values of the per-tail statistic (Eq. 8), so thresholding at 0.025 is exact.
Run in WSL: python3 /mnt/n/ARC/analysis/rethreshold_alpha025.py"""
import os, json
import numpy as np

STATS = "/mnt/n/ARC/lnm/stats"
OUT = "/mnt/n/ARC/analysis/out"
os.makedirs(OUT, exist_ok=True)

FILES = {
    "primary": "fwe_p_maxT.npy",
    "left_only": "fwe_p_maxT_Lonly.npy",
    "gm04": "fwe_p_maxT_gm04.npy",
    "log_volume": "fwe_p_maxT_vollog.npy",
    "drop_cerebellar": "fwe_p_maxT_nocere.npy",
    "drop_unrepairable": "fwe_p_maxT_noext.npy",
    "demean_check": "fwe_p_maxT_demean.npy",
    "connectivity_beyond_occupancy": "loc_conn_fwe_p.npy",
    "vlsm": "vlsm_fwe_p_maxT.npy",
}
res = {}
for name, f in FILES.items():
    p = f"{STATS}/{f}"
    if not os.path.exists(p):
        res[name] = "MISSING"
        continue
    arr = np.load(p)
    n05 = int((arr < 0.05).sum())
    n025 = int((arr < 0.025).sum())
    res[name] = {"alpha_005_per_tail": n05, "alpha_0025_per_tail": n025}
    print(f"{name:32s} 0.05/tail: {n05:6,}   0.025/tail: {n025:6,}")

# occupancy share of the primary map at 0.025 (190 is the 0.05 value)
res["occupancy_retained_fraction_0025"] = round(
    res["connectivity_beyond_occupancy"]["alpha_0025_per_tail"]
    / max(res["primary"]["alpha_0025_per_tail"], 1), 4)
res["occupancy_retained_fraction_005"] = round(
    res["connectivity_beyond_occupancy"]["alpha_005_per_tail"]
    / max(res["primary"]["alpha_005_per_tail"], 1), 4)
json.dump(res, open(f"{OUT}/rethreshold_alpha025.json", "w"), indent=1)
print("RETHRESHOLD_DONE")
