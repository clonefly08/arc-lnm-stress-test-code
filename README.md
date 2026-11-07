# ARC LNM stress test - analysis code

Code for: "A systematic stress test of lesion network mapping in 226 patients
with post-stroke aphasia" (corresponding author: Fei Xiong, clonefly08@163.com).

## Run order (see file headers for details)
1. phase0_audit.py            mask audit
2. phase2_norm.py / _retry.py / _phase2_forensic2.py   lesion normalization + QC
3. gsp_sbc_batch.py           normative seed-based connectivity (checkpointed)
4. phase4_regression.py       voxelwise regression + Freedman-Lane 5,000 max-T
5. phase4b_vlsm.py / loc_conn_test.py   lesion-location baselines
6. leakage_free_loocv.py      out-of-sample prediction
7. C1_yeo7.py / C2_C3_yeo.py  composition diagnostics
8. canonical_rerun.py         all variants under one shared permutation stream
   occ_canonical.py / vlsm_canonical.py / rethreshold_alpha025.py
9. t1t4_spatial_overlap.py / t2_power_calib.py / t3_selfcorr.py /
   t5_tail.py / t6_nested_boot.py / verify_c_numbers.py   diagnostics
fig1_main.py / fig2_robustness.py   figures
config.yaml                   all parameters and random seeds

## Data (public, not included here)
- Aphasia Recovery Cohort: OpenNeuro ds004884 (CC0)
- GSP1000 Preprocessed Connectome: Harvard Dataverse doi:10.7910/DVN/ILXIKS

## Notes
- Derived statistical maps (t maps, significance masks, FWE arrays) are not in
  this repository; they are available from the corresponding author and are
  being archived with a persistent DOI.
- Software: Python 3.10 (NumPy, SciPy, pandas, nibabel, matplotlib),
  FSL 6.0.7.6, ANTs 2.4.1.

## License
MIT
