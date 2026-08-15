"""
Canonical Final MMPI Reproducibility Workflow
=============================================

Reconstructs the final 30 m rice domain, MHSI, AWD proxy, CFR, uncertainty, final MMPI, fixed-score/relative classes, correlations, 33-scenario OAT sensitivity analysis, no-CFR ablation, and reproducibility tables. It starts from the final calibrated 2026 RF mask and the 2021–2025 predictor stacks.

This file was selected from the final successful workflow in the uploaded analysis notebook.
Superseded/failed notebook cells were intentionally excluded.
"""

# ============================================================
# FINAL MMPI REPRODUCIBILITY RERUN — GOWA RICE 2026
# Purpose:
#   1) Rebuild all MMPI sub-indices from the exact logic used in the
#      final notebook (2021–2025 environmental predictor stack +
#      January–July 2026 RF-derived rice-field mask).
#   2) Apply the final Climate/Topography hybrid patch.
#   3) Apply final soil gap-filling.
#   4) Recalculate MHSI, MMPI, classes, correlations.
#   5) Recalculate 33-scenario OAT weight sensitivity analysis.
#   6) Recalculate no-CFR ablation.
#   7) Export a publication-ready reproducibility parameter workbook.
#
# IMPORTANT:
# - This is NOT a re-run of Random Forest classification or AcATaMa.
# - It starts from the final calibrated RF rice mask and GEE predictor stack.
# - Final CFR penalty = 0.25; uncertainty penalty = 0.15.
# ============================================================

import os, glob, warnings, subprocess, sys
warnings.filterwarnings('ignore')

# Install only if needed (Google Colab compatible)
try:
    import rasterio
    import numpy as np
    import pandas as pd
    from scipy.stats import pearsonr, spearmanr
except Exception:
    subprocess.check_call([
        sys.executable, '-m', 'pip', 'install', '-q',
        'rasterio', 'numpy', 'pandas', 'scipy', 'openpyxl'
    ])
    import rasterio
    import numpy as np
    import pandas as pd
    from scipy.stats import pearsonr, spearmanr

from rasterio.warp import reproject, Resampling

try:
    from google.colab import drive
    drive.mount('/content/drive')
except Exception:
    print('Google Colab drive.mount is unavailable. Adjust the input paths when running locally.')

# ============================================================
# 1. PATHS — aligned with the final analysis workflow
# ============================================================

DRIVE_ROOT = os.environ.get("MMPI_DRIVE_ROOT", "/content/drive/MyDrive")

RICE2026_FINAL_10M_PATH = os.path.join(
    DRIVE_ROOT,
    'Gowa_Rice_RF_2026_Calibrated_Final_FIXED',
    'Raster_Output',
    'Gowa_Rice_RF_2026_Calibrated_Final_FIXED_10m.tif'
)

GEE_STACK_DIR = os.path.join(
    DRIVE_ROOT,
    'MMPI_Gowa_GEE_30m_2021_2025'
)

OUT_DIR = os.path.join(
    DRIVE_ROOT,
    'MANUSCRIPT_FINAL_MMPI_Rice2026_Gowa_REPRODUCIBILITY_RERUN'
)
RASTER_DIR = os.path.join(OUT_DIR, 'Raster_Output')
TABLE_DIR = os.path.join(OUT_DIR, 'Tables')
REPORT_DIR = os.path.join(OUT_DIR, 'Reports')
for d in [OUT_DIR, RASTER_DIR, TABLE_DIR, REPORT_DIR]:
    os.makedirs(d, exist_ok=True)

RICE_FRACTION_THRESHOLD_30M = 0.50

# Expected manuscript values — used only as diagnostics, not hard-coded results.
EXPECTED = {
    'Valid_area_ha': 26203.68,
    'Mean_MMPI': 41.7439,
    'Median_MMPI': 40.7539,
    'High_VeryHigh_area_ha': 689.76,
}

# ============================================================
# 2. EXPECTED PREDICTOR BANDS
# ============================================================

PREDICTOR_BANDS = [
    'S1_VV_median',
    'S1_VH_median',
    'S1_VH_minus_VV_median',
    'S1_Wet_Count',
    'S1_Observation_Count',
    'S1_Flooding_Frequency',
    'S1_Wet_Dry_Transition_Count',
    'S2_NDVI_median',
    'S2_NDWI_median',
    'S2_LSWI_median',
    'S2_EVI_median',
    'S2_NDVI_max',
    'S2_LSWI_max',
    'S2_NDVI_min',
    'S2_LSWI_min',
    'S2_Observation_Count',
    'CHIRPS_Rainfall_Total_mm',
    'CHIRPS_Rainfall_Mean_Daily_mm',
    'CHIRPS_Wet_Days_gt1mm',
    'ERA5_Temperature_2m_C_mean',
    'ERA5_SoilWater_Layer1_mean',
    'ERA5_Total_Precipitation_mm',
    'ERA5_Total_Evaporation_mm',
    'Soil_pH_H2O_0_30cm',
    'Soil_SOC_0_30cm',
    'Soil_Nitrogen_0_30cm',
    'Soil_Clay_0_30cm',
    'Soil_Sand_0_30cm',
    'Soil_Silt_0_30cm',
    'Soil_CEC_0_30cm',
    'Soil_BulkDensity_0_30cm',
    'DEM_Elevation_m',
    'DEM_Slope_degree',
    'JRC_Water_Occurrence',
    'JRC_Water_Seasonality',
    'JRC_Water_Recurrence'
]

# ============================================================
# 3. HELPERS
# ============================================================

def find_first(roots, patterns, required=True, label='file'):
    if isinstance(roots, str):
        roots = [roots]
    if isinstance(patterns, str):
        patterns = [patterns]
    found = []
    for root in roots:
        if not os.path.exists(root):
            continue
        for pat in patterns:
            found.extend(glob.glob(os.path.join(root, '**', pat), recursive=True))
    found = sorted(set(found))
    print(f'\nCandidate {label}:')
    for p in found[:10]:
        print(' -', p)
    if not found:
        if required:
            raise FileNotFoundError(f'{label} was not found.')
        return None
    return found[0]


def get_band_names(src, expected=None):
    desc = list(src.descriptions)
    if len(desc) == src.count and all(d is not None and str(d).strip() for d in desc):
        return [str(d) for d in desc]
    if expected is not None and len(expected) >= src.count:
        return expected[:src.count]
    return [f'band_{i}' for i in range(1, src.count + 1)]


def find_band_index(band_names, candidates, required=True):
    if isinstance(candidates, str):
        candidates = [candidates]
    low_names = [str(b).lower() for b in band_names]
    for cand in candidates:
        c = cand.lower()
        for i, name in enumerate(low_names):
            if name == c:
                return i + 1
    for cand in candidates:
        c = cand.lower()
        for i, name in enumerate(low_names):
            if c in name:
                return i + 1
    if required:
        raise ValueError('Band not found: ' + ', '.join(candidates))
    return None


def read_band(src, band_names, candidates, required=True):
    idx = find_band_index(band_names, candidates, required=required)
    if idx is None:
        return np.full((src.height, src.width), np.nan, dtype='float32')
    arr = src.read(idx).astype('float32')
    if src.nodata is not None:
        arr[arr == src.nodata] = np.nan
    arr[~np.isfinite(arr)] = np.nan
    return arr


def write_raster(path, arr, profile, dtype='float32', nodata=-9999):
    p = profile.copy()
    p.update(driver='GTiff', count=1, dtype=dtype, nodata=nodata, compress='lzw')
    out = arr.copy()
    if np.issubdtype(np.dtype(dtype), np.floating):
        out = out.astype(dtype)
        out[~np.isfinite(out)] = nodata
    else:
        out = out.astype(dtype)
    with rasterio.open(path, 'w', **p) as dst:
        dst.write(out, 1)
    print('Saved:', path)


def aggregate_binary_to_fraction(mask_path, ref_src, positive_value=1, nodata_value=255):
    with rasterio.open(mask_path) as src:
        arr = src.read(1)
        source = np.full(arr.shape, -9999, dtype='float32')
        source[arr == positive_value] = 1.0
        source[(arr != positive_value) & (arr != nodata_value)] = 0.0
        dst = np.full((ref_src.height, ref_src.width), -9999, dtype='float32')
        reproject(
            source=source,
            destination=dst,
            src_transform=src.transform,
            src_crs=src.crs,
            src_nodata=-9999,
            dst_transform=ref_src.transform,
            dst_crs=ref_src.crs,
            dst_nodata=-9999,
            resampling=Resampling.average
        )
        dst[dst == -9999] = np.nan
        dst[~np.isfinite(dst)] = np.nan
        return dst


def to_fraction_if_percent(arr):
    out = arr.astype('float32').copy()
    vals = out[np.isfinite(out)]
    if vals.size and np.nanpercentile(vals, 95) > 1.5:
        out /= 100.0
    return out


def standardize_soil_units(name, arr):
    out = arr.astype('float32').copy()
    vals = out[np.isfinite(out)]
    if not vals.size:
        return out
    med = np.nanmedian(vals)
    if name == 'pH':
        if med > 14:
            out /= 10.0
    elif name in ['clay', 'sand', 'silt']:
        if med > 100:
            out /= 10.0
        vals2 = out[np.isfinite(out)]
        if vals2.size and np.nanmedian(vals2) > 100:
            out /= 10.0
    elif name == 'bulk':
        if med > 100:
            out /= 1000.0
        elif med > 10:
            out /= 100.0
    elif name in ['soc', 'nitrogen', 'cec']:
        if med > 500:
            out /= 10.0
    return out


def clip_0_100(arr):
    return np.clip(arr, 0, 100).astype('float32')


def increasing_score(x, low, high):
    return clip_0_100((x - low) / (high - low) * 100.0)


def decreasing_score(x, low, high):
    return clip_0_100((high - x) / (high - low) * 100.0)


def range_score(x, min_v, opt_low, opt_high, max_v):
    score = np.full_like(x, np.nan, dtype='float32')
    left = (x >= min_v) & (x < opt_low)
    mid = (x >= opt_low) & (x <= opt_high)
    right = (x > opt_high) & (x <= max_v)
    score[left] = (x[left] - min_v) / (opt_low - min_v) * 100.0
    score[mid] = 100.0
    score[right] = (max_v - x[right]) / (max_v - opt_high) * 100.0
    score[(x < min_v) | (x > max_v)] = 0.0
    return clip_0_100(score)


def percentile_increasing_score(x, mask, p_low=5, p_high=95):
    vals = x[mask & np.isfinite(x)]
    if vals.size < 50:
        return np.full_like(x, np.nan, dtype='float32')
    low = np.nanpercentile(vals, p_low)
    high = np.nanpercentile(vals, p_high)
    if not np.isfinite(low) or not np.isfinite(high) or np.isclose(high, low):
        return np.full_like(x, 50, dtype='float32')
    return increasing_score(x, low, high)


def local_decreasing_score(x, mask, p_low=5, p_high=95):
    vals = x[mask & np.isfinite(x)]
    if vals.size < 50:
        return np.full_like(x, np.nan, dtype='float32')
    low, high = np.nanpercentile(vals, [p_low, p_high])
    if not np.isfinite(low) or not np.isfinite(high) or np.isclose(high, low):
        return np.full_like(x, 50, dtype='float32')
    return clip_0_100((high - x) / (high - low) * 100.0)


def local_midrange_score(x, mask, p_low=5, p_mid_low=35, p_mid_high=65, p_high=95):
    vals = x[mask & np.isfinite(x)]
    if vals.size < 50:
        return np.full_like(x, np.nan, dtype='float32')
    v_low, v_mid_low, v_mid_high, v_high = np.nanpercentile(
        vals, [p_low, p_mid_low, p_mid_high, p_high]
    )
    if not np.isfinite(v_low) or not np.isfinite(v_high) or np.isclose(v_high, v_low):
        return np.full_like(x, 50, dtype='float32')
    score = np.full_like(x, np.nan, dtype='float32')
    left = (x >= v_low) & (x < v_mid_low)
    mid = (x >= v_mid_low) & (x <= v_mid_high)
    right = (x > v_mid_high) & (x <= v_high)
    score[left] = (x[left] - v_low) / max(v_mid_low - v_low, 1e-6) * 100.0
    score[mid] = 100.0
    score[right] = (v_high - x[right]) / max(v_high - v_mid_high, 1e-6) * 100.0
    score[x < v_low] = 0.0
    score[x > v_high] = 0.0
    return clip_0_100(score)


def weighted_mean(arrays, weights):
    stack = np.stack(arrays).astype('float32')
    w = np.asarray(weights, dtype='float32').reshape(-1, 1, 1)
    valid = np.isfinite(stack)
    weighted = np.where(valid, stack * w, 0.0)
    wsum = np.where(valid, w, 0.0).sum(axis=0)
    return (weighted.sum(axis=0) / np.where(wsum == 0, np.nan, wsum)).astype('float32')


def safe_blend(abs_score, local_score, mask, w_abs=0.60, w_local=0.40):
    out = weighted_mean([abs_score, local_score], [w_abs, w_local])
    return np.where(mask, clip_0_100(out), np.nan)


def classify_fixed_score(mmpi, valid_mask):
    out = np.full(mmpi.shape, 255, dtype='uint8')
    out[(mmpi >= 0) & (mmpi < 20) & valid_mask] = 1
    out[(mmpi >= 20) & (mmpi < 40) & valid_mask] = 2
    out[(mmpi >= 40) & (mmpi < 60) & valid_mask] = 3
    out[(mmpi >= 60) & (mmpi < 80) & valid_mask] = 4
    out[(mmpi >= 80) & (mmpi <= 100) & valid_mask] = 5
    return out


def classify_relative(mmpi, valid_mask):
    out = np.full(mmpi.shape, 255, dtype='uint8')
    vals = mmpi[valid_mask & np.isfinite(mmpi)]
    q20, q40, q60, q80 = np.nanpercentile(vals, [20, 40, 60, 80])
    out[(mmpi <= q20) & valid_mask] = 1
    out[(mmpi > q20) & (mmpi <= q40) & valid_mask] = 2
    out[(mmpi > q40) & (mmpi <= q60) & valid_mask] = 3
    out[(mmpi > q60) & (mmpi <= q80) & valid_mask] = 4
    out[(mmpi > q80) & valid_mask] = 5
    return out, [q20, q40, q60, q80]


def corr_pair(a, b, mask, method='pearson'):
    v = mask & np.isfinite(a) & np.isfinite(b)
    if v.sum() < 5:
        return np.nan
    x, y = a[v].astype(float), b[v].astype(float)
    if method == 'spearman':
        return float(spearmanr(x, y).statistic)
    return float(pearsonr(x, y).statistic)


def normalize_weights(weights):
    w = np.asarray(weights, dtype=float)
    return (w / w.sum()).tolist()


def perturb_and_normalize(base_weights, target_idx, delta):
    w = np.asarray(base_weights, dtype=float).copy()
    w[target_idx] *= (1.0 + delta)
    return normalize_weights(w)


def summarize(name, arr, mask, pixel_area_ha):
    vals = arr[mask & np.isfinite(arr)]
    return {
        'Variable': name,
        'Pixels': int(vals.size),
        'Area_ha': float(vals.size * pixel_area_ha),
        'Mean': float(np.nanmean(vals)),
        'Median': float(np.nanmedian(vals)),
        'Std': float(np.nanstd(vals)),
        'Min': float(np.nanmin(vals)),
        'Max': float(np.nanmax(vals)),
        'P25': float(np.nanpercentile(vals, 25)),
        'P75': float(np.nanpercentile(vals, 75)),
    }


def percentile_values(arr, mask, ps):
    vals = arr[mask & np.isfinite(arr)]
    if vals.size < 1:
        return [np.nan] * len(ps)
    return [float(x) for x in np.nanpercentile(vals, ps)]

# ============================================================
# 4. LOCATE INPUT STACKS
# ============================================================

if not os.path.exists(RICE2026_FINAL_10M_PATH):
    raise FileNotFoundError('Final RF rice mask was not found: ' + RICE2026_FINAL_10M_PATH)

SEARCH_DIRS = [GEE_STACK_DIR, DRIVE_ROOT]
PREDICTOR_MEAN_PATH = find_first(
    SEARCH_DIRS,
    [
        'Gowa_MMPI_Predictor_Stack_30m_Mean_2021_2025*.tif',
        '*Predictor_Stack_30m_Mean_2021_2025*.tif',
        '*30m_Mean_2021_2025*.tif'
    ],
    label='predictor mean 30 m'
)
PREDICTOR_STD_PATH = find_first(
    SEARCH_DIRS,
    [
        'Gowa_MMPI_Predictor_Stack_30m_StdDev_2021_2025*.tif',
        '*Predictor_Stack_30m_StdDev_2021_2025*.tif',
        '*30m_StdDev_2021_2025*.tif'
    ],
    required=False,
    label='predictor StdDev 30 m'
)

# ============================================================
# 5. READ PREDICTOR STACK + AGGREGATE RICE MASK
# ============================================================

pred_src = rasterio.open(PREDICTOR_MEAN_PATH)
pred_profile = pred_src.profile.copy()
pred_profile.update(driver='GTiff', compress='lzw')
band_names = get_band_names(pred_src, PREDICTOR_BANDS)

rice_fraction_30m = aggregate_binary_to_fraction(RICE2026_FINAL_10M_PATH, pred_src)
rice_mask_30m = np.isfinite(rice_fraction_30m) & (rice_fraction_30m >= RICE_FRACTION_THRESHOLD_30M)
pixel_area_ha = abs(pred_src.transform.a * pred_src.transform.e) / 10000.0

print('\n=== RICE DOMAIN ===')
print('Rice pixels 30 m:', int(rice_mask_30m.sum()))
print('Rice area 30 m (ha):', rice_mask_30m.sum() * pixel_area_ha)

# ============================================================
# 6. READ VARIABLES
# ============================================================

S1_Flood = read_band(pred_src, band_names, 'S1_Flooding_Frequency', required=False)
S1_WetDry = read_band(pred_src, band_names, 'S1_Wet_Dry_Transition_Count', required=False)
S1_Obs = read_band(pred_src, band_names, 'S1_Observation_Count', required=False)

S2_NDVI = read_band(pred_src, band_names, ['S2_NDVI_median', 'NDVI_median'], required=False)
S2_NDVI_max = read_band(pred_src, band_names, 'S2_NDVI_max', required=False)
S2_NDVI_min = read_band(pred_src, band_names, 'S2_NDVI_min', required=False)
S2_LSWI_max = read_band(pred_src, band_names, 'S2_LSWI_max', required=False)
S2_LSWI_min = read_band(pred_src, band_names, 'S2_LSWI_min', required=False)
S2_Obs = read_band(pred_src, band_names, 'S2_Observation_Count', required=False)

Rain_Total = read_band(pred_src, band_names, 'CHIRPS_Rainfall_Total_mm', required=False)
Wet_Days = read_band(pred_src, band_names, 'CHIRPS_Wet_Days_gt1mm', required=False)
Temp_C = read_band(pred_src, band_names, 'ERA5_Temperature_2m_C_mean', required=False)
SoilWater = read_band(pred_src, band_names, 'ERA5_SoilWater_Layer1_mean', required=False)

pH = read_band(pred_src, band_names, 'Soil_pH_H2O_0_30cm', required=False)
SOC = read_band(pred_src, band_names, 'Soil_SOC_0_30cm', required=False)
Nitrogen = read_band(pred_src, band_names, 'Soil_Nitrogen_0_30cm', required=False)
Clay = read_band(pred_src, band_names, 'Soil_Clay_0_30cm', required=False)
Sand = read_band(pred_src, band_names, 'Soil_Sand_0_30cm', required=False)
CEC = read_band(pred_src, band_names, 'Soil_CEC_0_30cm', required=False)
Bulk = read_band(pred_src, band_names, 'Soil_BulkDensity_0_30cm', required=False)

Elevation = read_band(pred_src, band_names, 'DEM_Elevation_m', required=False)
Slope = read_band(pred_src, band_names, 'DEM_Slope_degree', required=False)
JRC_Occ = read_band(pred_src, band_names, 'JRC_Water_Occurrence', required=False)

# Standardize units exactly as final notebook
S1_Flood_F = to_fraction_if_percent(S1_Flood)
JRC_Occ_F = to_fraction_if_percent(JRC_Occ)
pH = standardize_soil_units('pH', pH)
Clay = standardize_soil_units('clay', Clay)
Sand = standardize_soil_units('sand', Sand)
SOC = standardize_soil_units('soc', SOC)
Nitrogen = standardize_soil_units('nitrogen', Nitrogen)
CEC = standardize_soil_units('cec', CEC)
Bulk = standardize_soil_units('bulk', Bulk)

NDVI_Amp = S2_NDVI_max - S2_NDVI_min
LSWI_Amp = S2_LSWI_max - S2_LSWI_min
valid_base = rice_mask_30m & np.isfinite(S2_NDVI)

# ============================================================
# 7. BASE SUB-INDICES — exact final notebook logic
# ============================================================

# Hydrology
flood_suit = range_score(S1_Flood_F, 0.02, 0.10, 0.45, 0.85)
wetdry_suit = increasing_score(S1_WetDry, 1, 6)
jrc_occ_suit = decreasing_score(JRC_Occ_F, 0.20, 0.75)
soilwater_suit = range_score(SoilWater, 0.08, 0.18, 0.45, 0.65)
HYDRO = weighted_mean(
    [flood_suit, wetdry_suit, jrc_occ_suit, soilwater_suit],
    [0.35, 0.30, 0.20, 0.15]
)

# Soil
ph_score = range_score(pH, 4.8, 5.5, 7.2, 8.2)
clay_score = range_score(Clay, 5, 15, 45, 65)
sand_score = decreasing_score(Sand, 65, 90)
soc_score = percentile_increasing_score(SOC, valid_base, 5, 95)
nitrogen_score = percentile_increasing_score(Nitrogen, valid_base, 5, 95)
cec_score = percentile_increasing_score(CEC, valid_base, 5, 95)
bulk_score = range_score(Bulk, 0.70, 0.90, 1.45, 1.85)
SOIL = weighted_mean(
    [ph_score, clay_score, sand_score, soc_score, nitrogen_score, cec_score, bulk_score],
    [0.18, 0.18, 0.10, 0.18, 0.12, 0.14, 0.10]
)

# Climate — initial absolute version, used to define pre-revision MMPI domain
rain_score_old = range_score(Rain_Total, 500, 1200, 3000, 4500)
wetdays_score_old = range_score(Wet_Days, 60, 100, 240, 320)
temp_score_old = range_score(Temp_C, 20, 24, 30, 34)
CLIMATE_OLD = weighted_mean(
    [rain_score_old, wetdays_score_old, temp_score_old],
    [0.40, 0.30, 0.30]
)

# Topography — initial absolute version
slope_score_old = decreasing_score(Slope, 2, 12)
elev_score_old = decreasing_score(Elevation, 500, 1200)
TOPO_OLD = weighted_mean([slope_score_old, elev_score_old], [0.75, 0.25])

# AWD proxy
wetdry_cycle_score = increasing_score(S1_WetDry, 1, 6)
ndvi_amp_score = range_score(NDVI_Amp, 0.04, 0.10, 0.45, 0.75)
lswi_amp_score = range_score(LSWI_Amp, 0.03, 0.08, 0.40, 0.70)
flooding_balance_score = range_score(S1_Flood_F, 0.02, 0.08, 0.45, 0.80)
AWD = weighted_mean(
    [wetdry_cycle_score, ndvi_amp_score, lswi_amp_score, flooding_balance_score],
    [0.35, 0.25, 0.25, 0.15]
)

# Continuous flooding risk
FLOOD = weighted_mean(
    [
        increasing_score(S1_Flood_F, 0.45, 0.90),
        increasing_score(JRC_Occ_F, 0.35, 0.80),
        decreasing_score(S1_WetDry, 1, 5)
    ],
    [0.45, 0.35, 0.20]
)

# Uncertainty
obs_s1_score = increasing_score(S1_Obs, 5, 30)
obs_s2_score = increasing_score(S2_Obs, 3, 20)
observation_quality = weighted_mean([obs_s1_score, obs_s2_score], [0.55, 0.45])
UNCERT = 100.0 - observation_quality

# Optional StdDev-based variability uncertainty
std_threshold_records = []
if PREDICTOR_STD_PATH is not None:
    try:
        std_src = rasterio.open(PREDICTOR_STD_PATH)
        std_names = get_band_names(std_src, PREDICTOR_BANDS)
        std_flood = read_band(std_src, std_names, 'S1_Flooding_Frequency', required=False)
        std_ndvi = read_band(std_src, std_names, ['S2_NDVI_median', 'NDVI_median'], required=False)
        std_lswi = read_band(std_src, std_names, ['S2_LSWI_median', 'LSWI_median'], required=False)

        std_flood_score = percentile_increasing_score(std_flood, valid_base, 5, 95)
        std_ndvi_score = percentile_increasing_score(std_ndvi, valid_base, 5, 95)
        std_lswi_score = percentile_increasing_score(std_lswi, valid_base, 5, 95)
        variability_uncertainty = weighted_mean(
            [std_flood_score, std_ndvi_score, std_lswi_score],
            [0.40, 0.35, 0.25]
        )
        UNCERT = weighted_mean([UNCERT, variability_uncertainty], [0.55, 0.45])

        for varname, arr in [
            ('StdDev S1 flooding frequency', std_flood),
            ('StdDev NDVI median', std_ndvi),
            ('StdDev LSWI median', std_lswi)
        ]:
            p5, p95 = percentile_values(arr, valid_base, [5, 95])
            std_threshold_records.append({
                'Variable': varname,
                'P5': p5,
                'P95': p95,
                'Purpose': 'Percentile increasing score for variability uncertainty'
            })
    except Exception as e:
        print('StdDev stack was not used because of an error:', e)

UNCERT = clip_0_100(UNCERT)

# Mask base components to rice domain
for name in ['HYDRO', 'SOIL', 'CLIMATE_OLD', 'TOPO_OLD', 'AWD', 'FLOOD', 'UNCERT']:
    arr = locals()[name]
    locals()[name] = np.where(rice_mask_30m, arr, np.nan)

# Initial MMPI used by final notebook patch to define valid_mmpi
MHSI_OLD = weighted_mean([HYDRO, SOIL, CLIMATE_OLD, TOPO_OLD], [0.42, 0.33, 0.15, 0.10])
MMPI_RAW_OLD = weighted_mean([MHSI_OLD, AWD], [0.68, 0.32])
MMPI_OLD = clip_0_100(
    MMPI_RAW_OLD * (1 - 0.25 * FLOOD / 100.0) * (1 - 0.15 * UNCERT / 100.0)
)
MMPI_OLD = np.where(rice_mask_30m, MMPI_OLD, np.nan)
valid_mmpi_old = rice_mask_30m & np.isfinite(MMPI_OLD)

# ============================================================
# 8. FINAL CLIMATE + TOPOGRAPHY HYBRID PATCH
# ============================================================

# Climate absolute component, final version
rain_abs = range_score(Rain_Total, 500, 1200, 3000, 4500)
wetdays_abs = range_score(Wet_Days, 60, 100, 240, 320)
temp_abs = range_score(Temp_C, 20, 24, 30, 34)
soilwater_abs = range_score(SoilWater, 0.08, 0.18, 0.45, 0.65)
climate_abs = weighted_mean(
    [rain_abs, wetdays_abs, temp_abs, soilwater_abs],
    [0.30, 0.25, 0.25, 0.20]
)

# Climate local component, P5-P35-P65-P95 midrange scoring
rain_local = local_midrange_score(Rain_Total, valid_mmpi_old)
wetdays_local = local_midrange_score(Wet_Days, valid_mmpi_old)
temp_local = local_midrange_score(Temp_C, valid_mmpi_old)
soilwater_local = local_midrange_score(SoilWater, valid_mmpi_old)
climate_local = weighted_mean(
    [rain_local, wetdays_local, temp_local, soilwater_local],
    [0.30, 0.25, 0.25, 0.20]
)
CLIMATE = safe_blend(climate_abs, climate_local, rice_mask_30m, 0.60, 0.40)

# Topography hybrid
slope_abs = decreasing_score(Slope, 2, 12)
elev_abs = decreasing_score(Elevation, 500, 1200)
topography_abs = weighted_mean([slope_abs, elev_abs], [0.75, 0.25])
slope_local = local_decreasing_score(Slope, valid_mmpi_old, 5, 95)
elev_local = local_decreasing_score(Elevation, valid_mmpi_old, 5, 95)
topography_local = weighted_mean([slope_local, elev_local], [0.75, 0.25])
TOPO = safe_blend(topography_abs, topography_local, rice_mask_30m, 0.60, 0.40)

# Same diagnostic rule as final patch
TOPO_STD = float(np.nanstd(TOPO[valid_mmpi_old & np.isfinite(TOPO)]))
USE_TOPOGRAPHY_IN_MHSI = np.isfinite(TOPO_STD) and TOPO_STD >= 0.50
print('\nTopography std revised:', TOPO_STD)
print('Topography used in MHSI:', USE_TOPOGRAPHY_IN_MHSI)

if USE_TOPOGRAPHY_IN_MHSI:
    MHSI_REVISED = weighted_mean([HYDRO, SOIL, CLIMATE, TOPO], [0.42, 0.33, 0.15, 0.10])
else:
    MHSI_REVISED = weighted_mean([HYDRO, SOIL, CLIMATE], [0.47, 0.38, 0.15])

MMPI_RAW_REVISED = weighted_mean([MHSI_REVISED, AWD], [0.68, 0.32])
MMPI_REVISED = clip_0_100(
    MMPI_RAW_REVISED * (1 - 0.25 * FLOOD / 100.0) * (1 - 0.15 * UNCERT / 100.0)
)
MMPI_REVISED = np.where(rice_mask_30m, MMPI_REVISED, np.nan)
valid_revised = rice_mask_30m & np.isfinite(MMPI_REVISED)

# ============================================================
# 9. FINAL SOIL GAP-FILLING + FINAL MMPI
# ============================================================

soil_valid = valid_revised & np.isfinite(SOIL)
soil_missing = valid_revised & ~np.isfinite(SOIL)
soil_median = float(np.nanmedian(SOIL[soil_valid]))
SOIL_FILLED = SOIL.copy()
SOIL_FILLED[soil_missing] = soil_median

# Final manuscript structure uses four MHSI components (topography retained in final notebook output)
MHSI_FINAL = weighted_mean(
    [HYDRO, SOIL_FILLED, CLIMATE, TOPO],
    [0.42, 0.33, 0.15, 0.10]
)
MMPI_RAW_FINAL = weighted_mean([MHSI_FINAL, AWD], [0.68, 0.32])
MMPI_FINAL = clip_0_100(
    MMPI_RAW_FINAL * (1 - 0.25 * FLOOD / 100.0) * (1 - 0.15 * UNCERT / 100.0)
)

for name in ['HYDRO', 'SOIL_FILLED', 'CLIMATE', 'TOPO', 'AWD', 'FLOOD', 'UNCERT', 'MHSI_FINAL', 'MMPI_RAW_FINAL', 'MMPI_FINAL']:
    arr = locals()[name]
    locals()[name] = np.where(rice_mask_30m, arr, np.nan)

valid_final = rice_mask_30m & np.isfinite(MMPI_FINAL)
FIXED_SCORE_FINAL = classify_fixed_score(MMPI_FINAL, valid_final)
REL_FINAL, REL_BREAKS = classify_relative(MMPI_FINAL, valid_final)

# ============================================================
# 10. SAVE CORE RASTERS
# ============================================================

core_rasters = {
    'Final_Rice_Fraction_30m': rice_fraction_30m,
    'Final_Hydrology_Suitability_30m': HYDRO,
    'Final_Soil_Suitability_GapFilled_30m': SOIL_FILLED,
    'Final_Climate_Suitability_Hybrid_30m': CLIMATE,
    'Final_Topography_Suitability_Hybrid_30m': TOPO,
    'Final_AWD_Proxy_30m': AWD,
    'Final_Continuous_Flooding_Risk_30m': FLOOD,
    'Final_Uncertainty_Index_30m': UNCERT,
    'Final_MHSI_30m': MHSI_FINAL,
    'Final_MMPI_Raw_30m': MMPI_RAW_FINAL,
    'Final_MMPI_30m': MMPI_FINAL,
}
for name, arr in core_rasters.items():
    write_raster(os.path.join(RASTER_DIR, name + '.tif'), arr, pred_profile)
write_raster(os.path.join(RASTER_DIR, 'Final_MMPI_Fixed_Score_Class_30m.tif'), FIXED_SCORE_FINAL, pred_profile, dtype='uint8', nodata=255)
write_raster(os.path.join(RASTER_DIR, 'Final_MMPI_Relative_Priority_Class_30m.tif'), REL_FINAL, pred_profile, dtype='uint8', nodata=255)

# ============================================================
# 11. BASELINE SUMMARY + CORRELATION
# ============================================================

summary_df = pd.DataFrame([
    summarize('MMPI', MMPI_FINAL, valid_final, pixel_area_ha),
    summarize('MHSI', MHSI_FINAL, valid_final, pixel_area_ha),
    summarize('AWD proxy', AWD, valid_final, pixel_area_ha),
    summarize('Hydrology suitability', HYDRO, valid_final, pixel_area_ha),
    summarize('Soil suitability', SOIL_FILLED, valid_final, pixel_area_ha),
    summarize('Climate suitability', CLIMATE, valid_final, pixel_area_ha),
    summarize('Topography suitability', TOPO, valid_final, pixel_area_ha),
    summarize('Continuous flooding risk', FLOOD, valid_final, pixel_area_ha),
    summarize('Uncertainty index', UNCERT, valid_final, pixel_area_ha),
])

corr_arrays = {
    'MMPI': MMPI_FINAL,
    'MHSI': MHSI_FINAL,
    'AWD proxy': AWD,
    'Hydrology suitability': HYDRO,
    'Soil suitability': SOIL_FILLED,
    'Climate suitability': CLIMATE,
    'Topography suitability': TOPO,
    'Continuous flooding risk': FLOOD,
    'Uncertainty index': UNCERT,
}
corr_df = pd.DataFrame({k: v[valid_final] for k, v in corr_arrays.items()}).corr(method='pearson')

# Class areas
class_labels = {1:'Very low',2:'Low',3:'Moderate',4:'High',5:'Very high'}
area_rows = []
for ctype, carr in [('Fixed-score', FIXED_SCORE_FINAL), ('Relative priority', REL_FINAL)]:
    denom = sum(((carr == c) & valid_final).sum() for c in range(1,6))
    for c in range(1,6):
        n = int(((carr == c) & valid_final).sum())
        area_rows.append({
            'Class_Type': ctype,
            'Class': c,
            'Class_Label': class_labels[c],
            'Pixels': n,
            'Area_ha': n * pixel_area_ha,
            'Percent': n / denom * 100.0 if denom else np.nan,
        })
area_df = pd.DataFrame(area_rows)

# ============================================================
# 12. REPRODUCIBILITY PARAMETER TABLES
# ============================================================

parameter_rows = [
    # Hydrology
    ['Hydrology', 'S1 flooding frequency', 'Range/trapezoid', '0.02', '0.10', '0.45', '0.85', 0.35, 'Fraction (0–1)'],
    ['Hydrology', 'S1 wet-dry transition count', 'Increasing', '1', '6', '', '', 0.30, 'Count'],
    ['Hydrology', 'JRC water occurrence', 'Decreasing', '0.20', '0.75', '', '', 0.20, 'Fraction (0–1)'],
    ['Hydrology', 'ERA5 soil water layer 1', 'Range/trapezoid', '0.08', '0.18', '0.45', '0.65', 0.15, 'Native stack units'],
    # Soil
    ['Soil', 'pH', 'Range/trapezoid', '4.8', '5.5', '7.2', '8.2', 0.18, 'pH'],
    ['Soil', 'Clay', 'Range/trapezoid', '5', '15', '45', '65', 0.18, '% after standardization'],
    ['Soil', 'Sand', 'Decreasing', '65', '90', '', '', 0.10, '% after standardization'],
    ['Soil', 'SOC', 'Percentile increasing', 'P5', 'P95', '', '', 0.18, 'Data-derived'],
    ['Soil', 'Nitrogen', 'Percentile increasing', 'P5', 'P95', '', '', 0.12, 'Data-derived'],
    ['Soil', 'CEC', 'Percentile increasing', 'P5', 'P95', '', '', 0.14, 'Data-derived'],
    ['Soil', 'Bulk density', 'Range/trapezoid', '0.70', '0.90', '1.45', '1.85', 0.10, 'g cm-3 after standardization'],
    # Climate final hybrid
    ['Climate absolute', 'Rainfall total', 'Range/trapezoid', '500', '1200', '3000', '4500', 0.30, 'mm'],
    ['Climate absolute', 'Wet days >1 mm', 'Range/trapezoid', '60', '100', '240', '320', 0.25, 'days'],
    ['Climate absolute', 'Temperature', 'Range/trapezoid', '20', '24', '30', '34', 0.25, 'deg C'],
    ['Climate absolute', 'ERA5 soil water layer 1', 'Range/trapezoid', '0.08', '0.18', '0.45', '0.65', 0.20, 'Native stack units'],
    ['Climate local', 'All climate variables', 'Local midrange', 'P5', 'P35', 'P65', 'P95', '', 'Per variable'],
    ['Climate final', 'Absolute + local', 'Weighted blend', '0.60 absolute', '0.40 local', '', '', '', 'Score 0–100'],
    # Topography final hybrid
    ['Topography absolute', 'Slope', 'Decreasing', '2', '12', '', '', 0.75, 'degrees'],
    ['Topography absolute', 'Elevation', 'Decreasing', '500', '1200', '', '', 0.25, 'm'],
    ['Topography local', 'Slope', 'Local decreasing', 'P5', 'P95', '', '', 0.75, 'degrees'],
    ['Topography local', 'Elevation', 'Local decreasing', 'P5', 'P95', '', '', 0.25, 'm'],
    ['Topography final', 'Absolute + local', 'Weighted blend', '0.60 absolute', '0.40 local', '', '', '', 'Score 0–100'],
    # AWD
    ['AWD proxy', 'S1 wet-dry transition count', 'Increasing', '1', '6', '', '', 0.35, 'Count'],
    ['AWD proxy', 'NDVI amplitude', 'Range/trapezoid', '0.04', '0.10', '0.45', '0.75', 0.25, 'Unitless'],
    ['AWD proxy', 'LSWI amplitude', 'Range/trapezoid', '0.03', '0.08', '0.40', '0.70', 0.25, 'Unitless'],
    ['AWD proxy', 'S1 flooding frequency balance', 'Range/trapezoid', '0.02', '0.08', '0.45', '0.80', 0.15, 'Fraction (0–1)'],
    # CFR
    ['CFR', 'S1 flooding frequency', 'Increasing', '0.45', '0.90', '', '', 0.45, 'Fraction (0–1)'],
    ['CFR', 'JRC occurrence', 'Increasing', '0.35', '0.80', '', '', 0.35, 'Fraction (0–1)'],
    ['CFR', 'S1 wet-dry transition count', 'Decreasing', '1', '5', '', '', 0.20, 'Count'],
    # Uncertainty
    ['Uncertainty observation quality', 'S1 observation count', 'Increasing', '5', '30', '', '', 0.55, 'Count'],
    ['Uncertainty observation quality', 'S2 observation count', 'Increasing', '3', '20', '', '', 0.45, 'Count'],
    ['Uncertainty variability', 'StdDev S1 flooding frequency', 'Percentile increasing', 'P5', 'P95', '', '', 0.40, 'Data-derived'],
    ['Uncertainty variability', 'StdDev NDVI median', 'Percentile increasing', 'P5', 'P95', '', '', 0.35, 'Data-derived'],
    ['Uncertainty variability', 'StdDev LSWI median', 'Percentile increasing', 'P5', 'P95', '', '', 0.25, 'Data-derived'],
    ['Uncertainty final', '100 - observation quality + variability', 'Weighted mean', '0.55 base', '0.45 variability', '', '', '', 'Score 0–100'],
    # Final model
    ['MHSI', 'Hydrology', 'Weighted mean', '', '', '', '', 0.42, 'Score 0–100'],
    ['MHSI', 'Soil gap-filled', 'Weighted mean', '', '', '', '', 0.33, 'Score 0–100'],
    ['MHSI', 'Climate hybrid', 'Weighted mean', '', '', '', '', 0.15, 'Score 0–100'],
    ['MHSI', 'Topography hybrid', 'Weighted mean', '', '', '', '', 0.10, 'Score 0–100'],
    ['MMPI raw', 'MHSI', 'Weighted mean', '', '', '', '', 0.68, 'Score 0–100'],
    ['MMPI raw', 'AWD proxy', 'Weighted mean', '', '', '', '', 0.32, 'Score 0–100'],
    ['MMPI final', 'Continuous flooding risk', 'Multiplicative penalty', 'Coefficient 0.25', '', '', '', '', 'MMPI_raw*(1-0.25*CFR/100)'],
    ['MMPI final', 'Uncertainty', 'Multiplicative penalty', 'Coefficient 0.15', '', '', '', '', '...*(1-0.15*U/100)'],
]
parameter_df = pd.DataFrame(
    parameter_rows,
    columns=['Domain','Variable','Scoring_Function','Param_1','Param_2','Param_3','Param_4','Weight','Units_or_Note']
)

# Actual data-derived percentiles used by the final run
derived_rows = []
for name, arr, mask, ps, purpose in [
    ('SOC', SOC, valid_base, [5,95], 'Soil percentile-increasing score'),
    ('Nitrogen', Nitrogen, valid_base, [5,95], 'Soil percentile-increasing score'),
    ('CEC', CEC, valid_base, [5,95], 'Soil percentile-increasing score'),
    ('Rainfall total', Rain_Total, valid_mmpi_old, [5,35,65,95], 'Climate local midrange score'),
    ('Wet days >1mm', Wet_Days, valid_mmpi_old, [5,35,65,95], 'Climate local midrange score'),
    ('Temperature', Temp_C, valid_mmpi_old, [5,35,65,95], 'Climate local midrange score'),
    ('Soil water', SoilWater, valid_mmpi_old, [5,35,65,95], 'Climate local midrange score'),
    ('Slope', Slope, valid_mmpi_old, [5,95], 'Topography local decreasing score'),
    ('Elevation', Elevation, valid_mmpi_old, [5,95], 'Topography local decreasing score'),
]:
    vals = percentile_values(arr, mask, ps)
    rec = {'Variable': name, 'Purpose': purpose}
    for p, v in zip(ps, vals):
        rec[f'P{p}'] = v
    derived_rows.append(rec)
derived_rows.extend(std_threshold_records)
derived_df = pd.DataFrame(derived_rows)

# ============================================================
# 13. OAT SENSITIVITY — 33 scenarios including baseline
# ============================================================

BASE_MHSI_W = [0.42, 0.33, 0.15, 0.10]
BASE_RAW_W = [0.68, 0.32]
BASE_CFR = 0.25
BASE_UNCERT = 0.15
PERTURBATIONS = [-0.15, -0.10, 0.10, 0.15]


def build_scenario(mhsi_w, raw_w, cfr_coef, uncert_coef):
    mhsi = weighted_mean([HYDRO, SOIL_FILLED, CLIMATE, TOPO], mhsi_w)
    raw = weighted_mean([mhsi, AWD], raw_w)
    final = clip_0_100(raw * (1 - cfr_coef * FLOOD / 100.0) * (1 - uncert_coef * UNCERT / 100.0))
    final = np.where(rice_mask_30m, final, np.nan)
    return mhsi, raw, final

baseline_mhsi, baseline_raw, baseline = build_scenario(BASE_MHSI_W, BASE_RAW_W, BASE_CFR, BASE_UNCERT)
baseline_mask = rice_mask_30m & np.isfinite(baseline)
baseline_class = classify_fixed_score(baseline, baseline_mask)
base_high_area = int(((baseline_class >= 4) & (baseline_class <= 5) & baseline_mask).sum()) * pixel_area_ha
base_mean = float(np.nanmean(baseline[baseline_mask]))
base_median = float(np.nanmedian(baseline[baseline_mask]))

sens_rows = []
scenario_weights_rows = []


def add_sensitivity_row(label, component, delta, mhsi_w, raw_w, cfr_coef, uncert_coef):
    _, _, arr = build_scenario(mhsi_w, raw_w, cfr_coef, uncert_coef)
    mask = baseline_mask & np.isfinite(arr)
    cls = classify_fixed_score(arr, mask)
    high_area = int(((cls >= 4) & (cls <= 5) & mask).sum()) * pixel_area_ha
    agreement = float(np.mean(cls[mask] == baseline_class[mask]) * 100.0)
    pr = corr_pair(baseline, arr, mask, 'pearson')
    sr = corr_pair(baseline, arr, mask, 'spearman')
    high_share = high_area / (mask.sum() * pixel_area_ha) * 100.0
    base_share = base_high_area / (baseline_mask.sum() * pixel_area_ha) * 100.0
    status = 'Baseline' if label == 'Baseline' else ('Check' if (agreement < 95 or pr < 0.98 or sr < 0.98) else 'Stable')
    sens_rows.append({
        'Scenario': label,
        'Perturbed_component': component,
        'Perturbation_percent': delta * 100 if delta is not None else 0,
        'Mean_MMPI': float(np.nanmean(arr[mask])),
        'Median_MMPI': float(np.nanmedian(arr[mask])),
        'Delta_mean_MMPI': float(np.nanmean(arr[mask]) - base_mean),
        'High_VeryHigh_area_ha': high_area,
        'Delta_High_VeryHigh_area_ha': high_area - base_high_area,
        'High_VeryHigh_share_percent': high_share,
        'Delta_High_VeryHigh_share_pp': high_share - base_share,
        'Fixed_score_class_agreement_percent': agreement,
        'Pearson_vs_baseline': pr,
        'Spearman_vs_baseline': sr,
        'Status': status,
    })
    scenario_weights_rows.append({
        'Scenario': label,
        'Hydrology_w': mhsi_w[0],
        'Soil_w': mhsi_w[1],
        'Climate_w': mhsi_w[2],
        'Topography_w': mhsi_w[3],
        'MHSI_w_in_MMPIraw': raw_w[0],
        'AWD_w_in_MMPIraw': raw_w[1],
        'CFR_penalty': cfr_coef,
        'Uncertainty_penalty': uncert_coef,
    })

add_sensitivity_row('Baseline', 'None', None, BASE_MHSI_W, BASE_RAW_W, BASE_CFR, BASE_UNCERT)

# 4 MHSI weights
mhsi_names = ['Hydrology', 'Soil', 'Climate', 'Topography']
for idx, name in enumerate(mhsi_names):
    for delta in PERTURBATIONS:
        w = perturb_and_normalize(BASE_MHSI_W, idx, delta)
        add_sensitivity_row(
            f'{name}_{delta*100:+.0f}pct', name, delta,
            w, BASE_RAW_W, BASE_CFR, BASE_UNCERT
        )

# 2 raw MMPI weights
raw_names = ['MHSI_raw_weight', 'AWD_raw_weight']
for idx, name in enumerate(raw_names):
    for delta in PERTURBATIONS:
        w = perturb_and_normalize(BASE_RAW_W, idx, delta)
        add_sensitivity_row(
            f'{name}_{delta*100:+.0f}pct', name, delta,
            BASE_MHSI_W, w, BASE_CFR, BASE_UNCERT
        )

# 2 penalty coefficients — direct perturbation
for name, base_coef in [('CFR_penalty', BASE_CFR), ('Uncertainty_penalty', BASE_UNCERT)]:
    for delta in PERTURBATIONS:
        cfr_coef = BASE_CFR * (1 + delta) if name == 'CFR_penalty' else BASE_CFR
        unc_coef = BASE_UNCERT * (1 + delta) if name == 'Uncertainty_penalty' else BASE_UNCERT
        add_sensitivity_row(
            f'{name}_{delta*100:+.0f}pct', name, delta,
            BASE_MHSI_W, BASE_RAW_W, cfr_coef, unc_coef
        )

sensitivity_df = pd.DataFrame(sens_rows)
scenario_weights_df = pd.DataFrame(scenario_weights_rows)
nonbase = sensitivity_df[sensitivity_df['Scenario'] != 'Baseline'].copy()

sensitivity_summary_df = pd.DataFrame([
    ['Sensitivity approach', 'One-at-a-time perturbation'],
    ['Weight perturbation range', '±10% and ±15%'],
    ['Number of scenarios, including baseline', int(len(sensitivity_df))],
    ['Number of stable scenarios', int((nonbase['Status'] == 'Stable').sum())],
    ['Number of scenarios requiring checking', int((nonbase['Status'] == 'Check').sum())],
    ['Baseline mean MMPI', base_mean],
    ['Baseline High + Very high area (ha)', base_high_area],
    ['Maximum absolute change in mean MMPI', float(nonbase['Delta_mean_MMPI'].abs().max())],
    ['Maximum absolute change in High + Very high area (ha)', float(nonbase['Delta_High_VeryHigh_area_ha'].abs().max())],
    ['Maximum absolute change in High + Very high share (pp)', float(nonbase['Delta_High_VeryHigh_share_pp'].abs().max())],
    ['Minimum fixed-score class agreement with baseline (%)', float(nonbase['Fixed_score_class_agreement_percent'].min())],
    ['Minimum Pearson correlation with baseline MMPI', float(nonbase['Pearson_vs_baseline'].min())],
    ['Minimum Spearman correlation with baseline MMPI', float(nonbase['Spearman_vs_baseline'].min())],
], columns=['Indicator','Result'])

# ============================================================
# 14. NO-CFR ABLATION
# ============================================================

MMPI_NO_CFR = clip_0_100(MMPI_RAW_FINAL * (1 - BASE_UNCERT * UNCERT / 100.0))
MMPI_NO_CFR = np.where(rice_mask_30m, MMPI_NO_CFR, np.nan)
mask_ab = valid_final & np.isfinite(MMPI_NO_CFR)
CLASS_NO_CFR = classify_fixed_score(MMPI_NO_CFR, mask_ab)
base_class_ab = classify_fixed_score(MMPI_FINAL, mask_ab)
no_cfr_high_area = int(((CLASS_NO_CFR >= 4) & (CLASS_NO_CFR <= 5) & mask_ab).sum()) * pixel_area_ha
base_high_area_ab = int(((base_class_ab >= 4) & (base_class_ab <= 5) & mask_ab).sum()) * pixel_area_ha
agreement_ab = float(np.mean(CLASS_NO_CFR[mask_ab] == base_class_ab[mask_ab]) * 100.0)
changed_area = float(np.sum(CLASS_NO_CFR[mask_ab] != base_class_ab[mask_ab]) * pixel_area_ha)
diff = MMPI_NO_CFR - MMPI_FINAL

ablation_df = pd.DataFrame([
    ['Valid analysis area', mask_ab.sum() * pixel_area_ha],
    ['Baseline mean MMPI', np.nanmean(MMPI_FINAL[mask_ab])],
    ['No-CFR mean MMPI', np.nanmean(MMPI_NO_CFR[mask_ab])],
    ['Change in mean MMPI', np.nanmean(MMPI_NO_CFR[mask_ab]) - np.nanmean(MMPI_FINAL[mask_ab])],
    ['Baseline median MMPI', np.nanmedian(MMPI_FINAL[mask_ab])],
    ['No-CFR median MMPI', np.nanmedian(MMPI_NO_CFR[mask_ab])],
    ['Baseline High + Very high area', base_high_area_ab],
    ['No-CFR High + Very high area', no_cfr_high_area],
    ['Change in High + Very high area', no_cfr_high_area - base_high_area_ab],
    ['Class agreement with baseline (%)', agreement_ab],
    ['Changed class area (ha)', changed_area],
    ['Pearson r, baseline vs no-CFR', corr_pair(MMPI_FINAL, MMPI_NO_CFR, mask_ab, 'pearson')],
    ['Spearman rho, baseline vs no-CFR', corr_pair(MMPI_FINAL, MMPI_NO_CFR, mask_ab, 'spearman')],
    ['Pearson r, difference vs CFR', corr_pair(diff, FLOOD, mask_ab, 'pearson')],
    ['Spearman rho, difference vs CFR', corr_pair(diff, FLOOD, mask_ab, 'spearman')],
], columns=['Indicator','Result'])

write_raster(os.path.join(RASTER_DIR, 'Ablation_MMPI_No_CFR_30m.tif'), MMPI_NO_CFR, pred_profile)
write_raster(os.path.join(RASTER_DIR, 'Ablation_MMPI_No_CFR_Fixed_Score_Class_30m.tif'), CLASS_NO_CFR, pred_profile, dtype='uint8', nodata=255)

# ============================================================
# 15. DIAGNOSTIC AGAINST MANUSCRIPT VALUES
# ============================================================

valid_area = valid_final.sum() * pixel_area_ha
mean_mmpi = float(np.nanmean(MMPI_FINAL[valid_final]))
median_mmpi = float(np.nanmedian(MMPI_FINAL[valid_final]))
high_area = int(((FIXED_SCORE_FINAL >= 4) & (FIXED_SCORE_FINAL <= 5) & valid_final).sum()) * pixel_area_ha

check_df = pd.DataFrame([
    ['Valid area (ha)', valid_area, EXPECTED['Valid_area_ha'], valid_area - EXPECTED['Valid_area_ha']],
    ['Mean MMPI', mean_mmpi, EXPECTED['Mean_MMPI'], mean_mmpi - EXPECTED['Mean_MMPI']],
    ['Median MMPI', median_mmpi, EXPECTED['Median_MMPI'], median_mmpi - EXPECTED['Median_MMPI']],
    ['High + Very high area (ha)', high_area, EXPECTED['High_VeryHigh_area_ha'], high_area - EXPECTED['High_VeryHigh_area_ha']],
], columns=['Metric','Rerun_value','Manuscript_expected','Difference'])

print('\n===== FINAL REPRODUCIBILITY CHECK =====')
print(check_df.to_string(index=False))
print('\n===== SENSITIVITY SUMMARY =====')
print(sensitivity_summary_df.to_string(index=False))
print('\n===== ABLATION SUMMARY =====')
print(ablation_df.to_string(index=False))

# ============================================================
# 16. EXPORT TABLES AND THE FINAL REPRODUCIBILITY WORKBOOK
# ============================================================

summary_df.to_csv(os.path.join(TABLE_DIR, 'T01_Final_MMPI_Summary_Statistics.csv'), index=False)
area_df.to_csv(os.path.join(TABLE_DIR, 'T02_Final_MMPI_Class_Area.csv'), index=False)
corr_df.to_csv(os.path.join(TABLE_DIR, 'T03_Final_MMPI_Correlation_Matrix.csv'))
parameter_df.to_csv(os.path.join(TABLE_DIR, 'S01_MMPI_Reproducibility_Parameters.csv'), index=False)
derived_df.to_csv(os.path.join(TABLE_DIR, 'S02_Data_Derived_Thresholds.csv'), index=False)
sensitivity_df.to_csv(os.path.join(TABLE_DIR, 'S03_Weight_Sensitivity_33_Scenarios.csv'), index=False)
scenario_weights_df.to_csv(os.path.join(TABLE_DIR, 'S04_Sensitivity_Scenario_Weights.csv'), index=False)
sensitivity_summary_df.to_csv(os.path.join(TABLE_DIR, 'S05_Sensitivity_Summary.csv'), index=False)
ablation_df.to_csv(os.path.join(TABLE_DIR, 'S06_No_CFR_Ablation.csv'), index=False)
check_df.to_csv(os.path.join(TABLE_DIR, 'S07_Reproducibility_Check_vs_Manuscript.csv'), index=False)

xlsx_path = os.path.join(TABLE_DIR, 'MMPI_FINAL_REPRODUCIBILITY_SUPPLEMENT.xlsx')
with pd.ExcelWriter(xlsx_path, engine='openpyxl') as writer:
    summary_df.to_excel(writer, sheet_name='Final_Summary', index=False)
    area_df.to_excel(writer, sheet_name='Class_Area', index=False)
    corr_df.to_excel(writer, sheet_name='Correlation')
    parameter_df.to_excel(writer, sheet_name='Model_Parameters', index=False)
    derived_df.to_excel(writer, sheet_name='Derived_Thresholds', index=False)
    sensitivity_df.to_excel(writer, sheet_name='Sensitivity_33', index=False)
    scenario_weights_df.to_excel(writer, sheet_name='Scenario_Weights', index=False)
    sensitivity_summary_df.to_excel(writer, sheet_name='Sensitivity_Summary', index=False)
    ablation_df.to_excel(writer, sheet_name='Ablation_No_CFR', index=False)
    check_df.to_excel(writer, sheet_name='Repro_Check', index=False)

# ============================================================
# 17. METHODS FORMULA REPORT
# ============================================================

report = f"""
MMPI FINAL REPRODUCIBILITY RERUN — GOWA
=======================================

Temporal design used by the final notebook:
- Rice-field domain: calibrated RF rice-field mask from January–July 2026 Sentinel observations.
- MMPI environmental predictor stack: multi-year 2021–2025 30 m predictor stack.
- Soil: static SoilGrids variables.
- Topography: static DEM-derived variables.
- JRC: long-term surface-water context.

Final model:
MHSI = 0.42*Hydrology + 0.33*Soil_filled + 0.15*Climate_hybrid + 0.10*Topography_hybrid
MMPI_raw = 0.68*MHSI + 0.32*AWD_proxy
MMPI = MMPI_raw*(1 - 0.25*CFR/100)*(1 - 0.15*Uncertainty/100)

Climate hybrid:
Climate_final = 0.60*Climate_absolute + 0.40*Climate_local
where local scores use P5-P35-P65-P95 midrange membership.

Topography hybrid:
Topography_final = 0.60*Topography_absolute + 0.40*Topography_local
where local slope/elevation scores use P5-P95 decreasing membership.

Soil gap filling:
- Missing soil suitability within the revised valid MMPI domain is filled with the median soil suitability of valid rice pixels.
- Median used in this rerun: {soil_median:.6f}
- Gap-filled area: {soil_missing.sum() * pixel_area_ha:.2f} ha
- Gap-filled share: {soil_missing.sum()/valid_revised.sum()*100:.4f}%

Final rerun results:
- Valid area: {valid_area:.2f} ha
- Mean MMPI: {mean_mmpi:.4f}
- Median MMPI: {median_mmpi:.4f}
- High + Very high area: {high_area:.2f} ha
- Mean MHSI: {np.nanmean(MHSI_FINAL[valid_final]):.4f}
- Mean AWD proxy: {np.nanmean(AWD[valid_final]):.4f}

Sensitivity:
- 33 scenarios including baseline.
- One-at-a-time changes: -15%, -10%, +10%, +15%.
- MHSI and MMPI-raw weights are renormalized to sum to 1 after perturbation.
- CFR and uncertainty penalties are perturbed directly.

Important interpretation:
The 30 m grid is a common computational grid. Resampling coarse CHIRPS, ERA5-Land,
and SoilGrids inputs does not create independent 30 m information content.
""".strip()

report_path = os.path.join(REPORT_DIR, 'MMPI_FINAL_REPRODUCIBILITY_REPORT.txt')
with open(report_path, 'w', encoding='utf-8') as f:
    f.write(report)

print('\n============================================================')
print('REPRODUCIBILITY RERUN COMPLETED')
print('Output folder :', OUT_DIR)
print('Workbook      :', xlsx_path)
print('GitHub file   : MMPI_FINAL_REPRODUCIBILITY_SUPPLEMENT.xlsx')
print('Report        :', report_path)
print('============================================================')
