# Gowa Rice-Field Methane Mitigation Potential (MMPI) — Reproducibility Code

This repository contains the final computational workflow supporting the manuscript **Remote Sensing-Based Environmental Assessment of Methane Mitigation Potential in Tropical Rice Fields Using Sentinel-1/2 and Multi-Source Data**.

## Run order

1. `01_rf_rice_mapping_2026.py` — Sentinel-1/Sentinel-2 Random Forest rice-field mapping for January–July 2026.
2. `02_rf_mask_calibration_2026.py` — Final conservative calibration of the RF-derived 10 m rice mask.
3. `03_rf_internal_class_separability_2026.py` — Expanded seed-derived internal RF diagnostic assessment.
4. `04_export_acatama_validation_raster.py` — Export the final two-class raster for independent QGIS/AcATaMa validation.
5. `05_final_mmpi_reproducibility.py` — Canonical final MMPI reconstruction using the calibrated 2026 rice mask and 2021–2025 environmental predictor stacks; includes sensitivity and CFR ablation.
6. `06_generate_manuscript_figures.py` — Generate the final manuscript figures at 600 dpi.

## Independent validation

The Python code only prepares the thematic raster for AcATaMa. The 400-point stratified validation and visual reference labeling were performed independently in QGIS/AcATaMa using high-resolution reference imagery.

## Important methodological distinction

The map-producing RF workflow and the expanded internal class-separability assessment are separate RF fits. The internal assessment is reported only as a seed-derived diagnostic and not as independent final-map accuracy. The AcATaMa area-adjusted assessment is the primary independent map-accuracy reference.

## MMPI interpretation

MMPI is a spatial screening index, not a validated methane-emission model. Fixed-score MMPI classes are interpretive score intervals and are not empirically calibrated methane-emission or methane-reduction thresholds.

## Input data

The workflow uses Sentinel-1, Sentinel-2, CHIRPS, ERA5-Land, SoilGrids, Copernicus DEM, JRC Global Surface Water, a 2024 rice-field reference raster, and the final calibrated 2026 rice-field mask. See the manuscript for data-source details.

## Not included

The successful script used for the external irrigation plausibility assessment was not present in the uploaded notebook, so it is intentionally not fabricated here. Add that script only after recovering or independently reproducing the exact workflow that generated the manuscript's irrigation results.
