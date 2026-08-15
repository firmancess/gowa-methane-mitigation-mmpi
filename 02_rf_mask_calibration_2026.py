"""
Final Calibration of the 2026 RF-Derived Rice Mask
==================================================

Applies the final conservative probability and spectral filters used to create the calibrated 10 m rice-field mask. This is the final calibration cell; the earlier failed slope-filter attempt is excluded.

This file was selected from the final successful workflow in the uploaded analysis notebook.
Superseded/failed notebook cells were intentionally excluded.
"""

# ============================================================
# FIXED CALIBRATION SCRIPT
# KALIBRASI ULANG SAWAH RF 2026 GOWA
# Mengatasi masalah slope filter yang membuat hasil menjadi 0 ha
# ============================================================


import os
import glob
import subprocess
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio

from rasterio.warp import reproject, Resampling
from rasterio.plot import plotting_extent
from rasterio.transform import xy

from scipy.ndimage import label as ndi_label
from scipy.ndimage import binary_opening, binary_closing, binary_dilation

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch
from matplotlib.ticker import ScalarFormatter, MaxNLocator

from shapely.geometry import Point

try:
    from google.colab import drive
    drive.mount('/content/drive')
except Exception:
    pass


# ============================================================
# 1. FOLDER DAN INPUT
# ============================================================

DRIVE_ROOT = os.environ.get("MMPI_DRIVE_ROOT", "/content/drive/MyDrive")

SENTINEL_DIR = os.path.join(
    DRIVE_ROOT,
    "Gowa_Sentinel_2026_RF_Input"
)

RF_OUT_DIR = os.path.join(
    DRIVE_ROOT,
    "Gowa_Rice_RF_Classification_2026"
)

POSTFILTER_DIR = os.path.join(
    DRIVE_ROOT,
    "Gowa_Rice_RF_2026_PostFiltered"
)

MMPI_GEE_DIR = os.path.join(
    DRIVE_ROOT,
    "MMPI_Gowa_GEE_30m_2021_2025"
)

SAWAH_2024_MASK_PATH = os.path.join(
    DRIVE_ROOT,
    "MMPI_Gowa_Analysis_Output_30m",
    "Sawah_2024_Mamminasata_Gowa_FINAL",
    "05_Sawah_2024_Gowa_binary_mask_aligned_to_MMPI_30m.tif"
)

OUT_DIR = os.path.join(
    DRIVE_ROOT,
    "Gowa_Rice_RF_2026_Calibrated_Final_FIXED"
)

RASTER_DIR = os.path.join(OUT_DIR, "Raster_Output")
TABLE_DIR = os.path.join(OUT_DIR, "Tables")
FIG_DIR = os.path.join(OUT_DIR, "Figures_DPI600")
VALIDATION_DIR = os.path.join(OUT_DIR, "Validation_Points")

for d in [OUT_DIR, RASTER_DIR, TABLE_DIR, FIG_DIR, VALIDATION_DIR]:
    os.makedirs(d, exist_ok=True)

RAW_CLASS_PATH = os.path.join(
    RF_OUT_DIR,
    "Gowa_Rice_RF_2026_Binary_Cleaned_10m.tif"
)

PROB_PATH = os.path.join(
    RF_OUT_DIR,
    "Gowa_Rice_RF_2026_Rice_Probability_10m.tif"
)

PREVIOUS_FILTERED_PATH = os.path.join(
    POSTFILTER_DIR,
    "Raster_Output",
    "Gowa_Rice_RF_2026_Conservative_Filtered_10m.tif"
)

RF_STACK_SINGLE = os.path.join(
    SENTINEL_DIR,
    "Gowa_Sentinel_2026_RF_Input_Stack_10m.tif"
)

RF_STACK_TILE_PATTERN = os.path.join(
    SENTINEL_DIR,
    "Gowa_Sentinel_2026_RF_Input_Stack_10m-*.tif"
)

print("Raw RF class      :", RAW_CLASS_PATH)
print("RF probability    :", PROB_PATH)
print("Previous filtered :", PREVIOUS_FILTERED_PATH)
print("Sawah 2024 ref    :", SAWAH_2024_MASK_PATH)
print("Output            :", OUT_DIR)


# ============================================================
# 2. PARAMETER KALIBRASI
# ============================================================

PROB_THRESHOLDS = [
    0.55, 0.60, 0.65, 0.70, 0.75,
    0.80, 0.85, 0.88, 0.90, 0.92,
    0.94, 0.96, 0.98
]

TARGET_RICE_AREA_HA = 24096
TARGET_MIN_AREA_HA = 18000
TARGET_MAX_AREA_HA = 35000

# Filter spektral sawah
NDVI_MIN = 0.20
NDVI_MAX = 0.78

NDVI_AMP_MIN_OUTSIDE = 0.08
LSWI_AMP_MIN_OUTSIDE = 0.05

# Filter hutan/perkebunan permanen
FOREST_NDVI_MIN = 0.76
FOREST_NDVI_STD_MAX = 0.08
FOREST_NDVI_AMP_MAX = 0.10

# Filter air dan terbangun
MNDWI_WATER_THRESHOLD = 0.25
NDBI_BUILT_THRESHOLD = 0.15

# Slope hanya digunakan jika valid
SLOPE_STRICT_MAX = 12.0
SLOPE_MIN_VALID_PIXEL = 50000

# Referensi sawah 2024 dipertahankan secara lebih longgar
REF_PROB_MIN = 0.35

# Minimum patch sawah
MIN_PATCH_AREA_HA = 0.50

N_VALIDATION_POINTS = 150
RANDOM_SEED = 2026
DPI = 600


# ============================================================
# 3. FUNGSI DASAR
# ============================================================

def build_vrt_from_tiles(tile_files, vrt_path):
    if len(tile_files) == 0:
        raise FileNotFoundError("Tile RF stack tidak ditemukan.")

    tile_list_path = vrt_path.replace(".vrt", "_tile_list.txt")

    with open(tile_list_path, "w") as f:
        for tile in tile_files:
            f.write(tile + "\n")

    cmd = [
        "/usr/bin/gdalbuildvrt",
        "-overwrite",
        "-input_file_list",
        tile_list_path,
        vrt_path
    ]

    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    if result.returncode != 0:
        print(result.stderr)
        raise RuntimeError("Gagal membuat VRT.")

    return vrt_path


def expected_rf_band_names_75():
    names = []

    names += [
        "S2_B2_median",
        "S2_B3_median",
        "S2_B4_median",
        "S2_B8_median",
        "S2_B11_median",
        "S2_B12_median",
        "S2_NDVI_median",
        "S2_EVI_median",
        "S2_NDWI_median",
        "S2_MNDWI_median",
        "S2_LSWI_median",
        "S2_NDBI_median"
    ]

    for var in ["NDVI", "LSWI", "NDWI", "MNDWI"]:
        names += [
            f"{var}_mean",
            f"{var}_min",
            f"{var}_max",
            f"{var}_stdDev"
        ]

    for var in ["VV", "VH", "VV_minus_VH", "VV_div_VH", "RVI"]:
        names += [
            f"{var}_mean",
            f"{var}_min",
            f"{var}_max",
            f"{var}_stdDev"
        ]

    for q in ["Q1", "Q2", "Q3_partial"]:
        names += [
            f"S2_NDVI_{q}",
            f"S2_LSWI_{q}",
            f"S2_NDWI_{q}",
            f"S2_MNDWI_{q}",
            f"S2_EVI_{q}",
            f"S1_VV_{q}",
            f"S1_VH_{q}",
            f"S1_VV_minus_VH_{q}",
            f"S1_RVI_{q}"
        ]

    return names


def get_band_names(src):
    names = []

    for i, desc in enumerate(src.descriptions, start=1):
        if desc is None or str(desc).strip() == "":
            names.append(f"band_{i}")
        else:
            names.append(str(desc))

    if all(n.startswith("band_") for n in names) and src.count == 75:
        names = expected_rf_band_names_75()

    return names


def find_band_index(band_names, candidates, required=True):
    if isinstance(candidates, str):
        candidates = [candidates]

    lower = [b.lower() for b in band_names]

    for cand in candidates:
        cand_l = cand.lower()
        for i, b in enumerate(lower):
            if b == cand_l:
                return i + 1

    for cand in candidates:
        cand_l = cand.lower()
        for i, b in enumerate(lower):
            if cand_l in b:
                return i + 1

    if required:
        raise ValueError(
            "Band tidak ditemukan: "
            + ", ".join(candidates)
            + "\n\nBand tersedia:\n"
            + "\n".join(band_names)
        )

    return None


def read_band(src, idx):
    if idx is None:
        return None

    arr = src.read(idx).astype("float32")

    if src.nodata is not None:
        arr[arr == src.nodata] = np.nan

    arr[np.isinf(arr)] = np.nan

    return arr


def find_file(roots, patterns):
    if isinstance(roots, str):
        roots = [roots]
    if isinstance(patterns, str):
        patterns = [patterns]

    found = []

    for root in roots:
        if not os.path.exists(root):
            continue

        for p in patterns:
            found.extend(
                glob.glob(
                    os.path.join(root, "**", p),
                    recursive=True
                )
            )

    return sorted(list(set(found)))


def align_binary_mask(mask_path, ref_src, positive_values=[1]):
    with rasterio.open(mask_path) as src:
        arr = src.read(1)
        mask = np.isin(arr, positive_values).astype("uint8")

        same_grid = (
            src.width == ref_src.width and
            src.height == ref_src.height and
            src.crs == ref_src.crs and
            src.transform == ref_src.transform
        )

        if same_grid:
            return mask == 1

        aligned = np.zeros(
            (ref_src.height, ref_src.width),
            dtype="uint8"
        )

        reproject(
            source=mask,
            destination=aligned,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=ref_src.transform,
            dst_crs=ref_src.crs,
            resampling=Resampling.nearest
        )

        return aligned == 1


def align_float_raster(path, ref_src, band_index=1, resampling=Resampling.bilinear):
    with rasterio.open(path) as src:
        arr = src.read(band_index).astype("float32")

        if src.nodata is not None:
            arr[arr == src.nodata] = np.nan

        same_grid = (
            src.width == ref_src.width and
            src.height == ref_src.height and
            src.crs == ref_src.crs and
            src.transform == ref_src.transform
        )

        if same_grid:
            return arr

        src_arr = arr.copy()
        src_arr[~np.isfinite(src_arr)] = -9999

        dst = np.full(
            (ref_src.height, ref_src.width),
            -9999,
            dtype="float32"
        )

        reproject(
            source=src_arr,
            destination=dst,
            src_transform=src.transform,
            src_crs=src.crs,
            src_nodata=-9999,
            dst_transform=ref_src.transform,
            dst_crs=ref_src.crs,
            dst_nodata=-9999,
            resampling=resampling
        )

        dst[dst == -9999] = np.nan
        return dst


def write_raster(path, arr, ref_profile, dtype="uint8", nodata=255):
    profile = ref_profile.copy()
    profile.update({
        "driver": "GTiff",
        "count": 1,
        "dtype": dtype,
        "nodata": nodata,
        "compress": "lzw"
    })

    out = arr.copy()

    if np.issubdtype(np.dtype(dtype), np.floating):
        out = out.astype(dtype)
        out[~np.isfinite(out)] = nodata
    else:
        out = out.astype(dtype)

    with rasterio.open(path, "w", **profile) as dst:
        dst.write(out, 1)

    print("Raster tersimpan:", path)


def remove_small_patches(binary_mask, pixel_area_ha, min_area_ha):
    min_pixels = max(1, int(min_area_ha / pixel_area_ha))

    labeled, ncomp = ndi_label(binary_mask)

    counts = np.bincount(labeled.ravel())

    keep = np.zeros_like(counts, dtype=bool)
    keep[counts >= min_pixels] = True
    keep[0] = False

    cleaned = keep[labeled]

    return cleaned, ncomp, min_pixels


def format_axis(ax):
    ax.set_xlabel("Easting (m)", fontsize=8)
    ax.set_ylabel("Northing (m)", fontsize=8)

    xf = ScalarFormatter(useOffset=False)
    xf.set_scientific(False)

    yf = ScalarFormatter(useOffset=False)
    yf.set_scientific(False)

    ax.xaxis.set_major_formatter(xf)
    ax.yaxis.set_major_formatter(yf)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4))

    ax.tick_params(axis="both", labelsize=7)
    ax.set_aspect("equal", adjustable="box")


def save_fig(fig, filename):
    path = os.path.join(FIG_DIR, filename)
    fig.savefig(path, dpi=DPI, bbox_inches="tight", pad_inches=0.03)
    plt.show()
    print("Figure tersimpan:", path)
    return path


def sample_points_from_mask(mask, n, label, transform, prob_arr, class_arr, seed):
    rng = np.random.default_rng(seed)

    rows, cols = np.where(mask)

    if len(rows) == 0:
        return pd.DataFrame()

    n_take = min(n, len(rows))
    idx = rng.choice(len(rows), size=n_take, replace=False)

    recs = []

    for i, k in enumerate(idx, start=1):
        r = int(rows[k])
        c = int(cols[k])
        x, y = xy(transform, r, c, offset="center")

        recs.append({
            "point_id": f"{label}_{i:03d}",
            "group": label,
            "row": r,
            "col": c,
            "x": x,
            "y": y,
            "rice_probability": float(prob_arr[r, c]) if np.isfinite(prob_arr[r, c]) else np.nan,
            "final_class": int(class_arr[r, c]) if class_arr[r, c] != 255 else 255,
            "visual_label": "",
            "confidence": "",
            "notes": ""
        })

    return pd.DataFrame(recs)


# ============================================================
# 4. BUKA RF STACK SENTINEL 2026
# ============================================================

tiles = sorted(glob.glob(RF_STACK_TILE_PATTERN))

if os.path.exists(RF_STACK_SINGLE):
    RF_STACK_PATH = RF_STACK_SINGLE
else:
    RF_STACK_PATH = os.path.join(
        SENTINEL_DIR,
        "Gowa_Sentinel_2026_RF_Input_Stack_10m_MOSAIC.vrt"
    )

    if not os.path.exists(RF_STACK_PATH):
        RF_STACK_PATH = build_vrt_from_tiles(tiles, RF_STACK_PATH)

src = rasterio.open(RF_STACK_PATH)
profile = src.profile.copy()
profile.update({
    "driver": "GTiff",
    "compress": "lzw"
})

band_names = get_band_names(src)

print("\nRF Stack:", RF_STACK_PATH)
print("Shape:", src.height, src.width)
print("Bands:", src.count)


# ============================================================
# 5. BACA RASTER RF DAN BAND DIAGNOSTIK
# ============================================================

with rasterio.open(RAW_CLASS_PATH) as cls_src:
    raw_class = cls_src.read(1).astype("uint8")
    raw_profile = cls_src.profile.copy()

with rasterio.open(PROB_PATH) as prob_src:
    prob = prob_src.read(1).astype("float32")
    if prob_src.nodata is not None:
        prob[prob == prob_src.nodata] = np.nan
    prob[np.isinf(prob)] = np.nan

if os.path.exists(PREVIOUS_FILTERED_PATH):
    with rasterio.open(PREVIOUS_FILTERED_PATH) as pf:
        previous_filtered = pf.read(1).astype("uint8")
else:
    previous_filtered = None

valid = raw_class != 255

ndvi_med_idx = find_band_index(band_names, ["S2_NDVI_median", "NDVI_median", "NDVI"], True)
ndvi_min_idx = find_band_index(band_names, ["NDVI_min"], False)
ndvi_max_idx = find_band_index(band_names, ["NDVI_max"], False)
ndvi_std_idx = find_band_index(band_names, ["NDVI_stdDev"], False)

lswi_med_idx = find_band_index(band_names, ["S2_LSWI_median", "LSWI_median", "LSWI"], False)
lswi_min_idx = find_band_index(band_names, ["LSWI_min"], False)
lswi_max_idx = find_band_index(band_names, ["LSWI_max"], False)

mndwi_med_idx = find_band_index(band_names, ["S2_MNDWI_median", "MNDWI_median", "MNDWI"], False)
ndbi_med_idx = find_band_index(band_names, ["S2_NDBI_median", "NDBI_median", "NDBI"], False)

NDVI_MED = read_band(src, ndvi_med_idx)
NDVI_MIN_ARR = read_band(src, ndvi_min_idx)
NDVI_MAX_ARR = read_band(src, ndvi_max_idx)
NDVI_STD = read_band(src, ndvi_std_idx)

LSWI_MED = read_band(src, lswi_med_idx)
LSWI_MIN_ARR = read_band(src, lswi_min_idx)
LSWI_MAX_ARR = read_band(src, lswi_max_idx)

MNDWI_MED = read_band(src, mndwi_med_idx)
NDBI_MED = read_band(src, ndbi_med_idx)

if NDVI_STD is None:
    NDVI_STD = np.full_like(NDVI_MED, np.nan)

if LSWI_MED is None:
    LSWI_MED = np.full_like(NDVI_MED, np.nan)

if MNDWI_MED is None:
    MNDWI_MED = np.full_like(NDVI_MED, np.nan)

if NDBI_MED is None:
    NDBI_MED = np.full_like(NDVI_MED, np.nan)

if NDVI_MIN_ARR is not None and NDVI_MAX_ARR is not None:
    NDVI_AMP = NDVI_MAX_ARR - NDVI_MIN_ARR
else:
    q_arrays = []
    for q in ["Q1", "Q2", "Q3_partial"]:
        idx = find_band_index(band_names, [f"S2_NDVI_{q}"], False)
        if idx is not None:
            q_arrays.append(read_band(src, idx))

    if len(q_arrays) >= 2:
        q_stack = np.stack(q_arrays)
        NDVI_AMP = np.nanmax(q_stack, axis=0) - np.nanmin(q_stack, axis=0)
    else:
        NDVI_AMP = np.full_like(NDVI_MED, np.nan)

if LSWI_MIN_ARR is not None and LSWI_MAX_ARR is not None:
    LSWI_AMP = LSWI_MAX_ARR - LSWI_MIN_ARR
else:
    q_arrays = []
    for q in ["Q1", "Q2", "Q3_partial"]:
        idx = find_band_index(band_names, [f"S2_LSWI_{q}"], False)
        if idx is not None:
            q_arrays.append(read_band(src, idx))

    if len(q_arrays) >= 2:
        q_stack = np.stack(q_arrays)
        LSWI_AMP = np.nanmax(q_stack, axis=0) - np.nanmin(q_stack, axis=0)
    else:
        LSWI_AMP = np.full_like(NDVI_MED, np.nan)

print("Band diagnostik selesai dibaca.")


# ============================================================
# 6. BACA SLOPE, TETAPI JANGAN JADIKAN WAJIB
# ============================================================

MMPI_EXPECTED_BANDS = [
    "S1_VV_median",
    "S1_VH_median",
    "S1_VH_minus_VV_median",
    "S1_Wet_Count",
    "S1_Observation_Count",
    "S1_Flooding_Frequency",
    "S1_Wet_Dry_Transition_Count",
    "S2_NDVI_median",
    "S2_NDWI_median",
    "S2_LSWI_median",
    "S2_EVI_median",
    "S2_NDVI_max",
    "S2_LSWI_max",
    "S2_NDVI_min",
    "S2_LSWI_min",
    "S2_Observation_Count",
    "CHIRPS_Rainfall_Total_mm",
    "CHIRPS_Rainfall_Mean_Daily_mm",
    "CHIRPS_Wet_Days_gt1mm",
    "ERA5_Temperature_2m_C_mean",
    "ERA5_SoilWater_Layer1_mean",
    "ERA5_Total_Precipitation_mm",
    "ERA5_Total_Evaporation_mm",
    "Soil_pH_H2O_0_30cm",
    "Soil_SOC_0_30cm",
    "Soil_Nitrogen_0_30cm",
    "Soil_Clay_0_30cm",
    "Soil_Sand_0_30cm",
    "Soil_Silt_0_30cm",
    "Soil_CEC_0_30cm",
    "Soil_BulkDensity_0_30cm",
    "DEM_Elevation_m",
    "DEM_Slope_degree",
    "JRC_Water_Occurrence",
    "JRC_Water_Seasonality",
    "JRC_Water_Recurrence"
]

def get_stack_band_names(src_stack, expected):
    desc = list(src_stack.descriptions)

    if all(d not in [None, ""] for d in desc) and len(desc) == src_stack.count:
        return desc

    return expected[:src_stack.count]


slope_10m = None
slope_available = False
slope_used = False

mean_stack_candidates = find_file(
    [MMPI_GEE_DIR, DRIVE_ROOT],
    [
        "Gowa_MMPI_Predictor_Stack_30m_Mean_2021_2025*.tif",
        "*30m*Mean*2021_2025*.tif"
    ]
)

if len(mean_stack_candidates) > 0:
    mean_stack_path = mean_stack_candidates[0]

    try:
        with rasterio.open(mean_stack_path) as mmpi_src:
            mmpi_band_names = get_stack_band_names(mmpi_src, MMPI_EXPECTED_BANDS)

            slope_idx = find_band_index(
                mmpi_band_names,
                ["DEM_Slope_degree", "Slope"],
                required=True
            )

            slope_10m = align_float_raster(
                mean_stack_path,
                src,
                band_index=slope_idx,
                resampling=Resampling.bilinear
            )

            slope_available = True

    except Exception as e:
        print("Slope gagal dibaca:", e)

if slope_10m is None:
    slope_10m = np.full_like(NDVI_MED, np.nan)

slope_valid_count = np.sum(np.isfinite(slope_10m) & valid)
slope_low_count = np.sum(np.isfinite(slope_10m) & valid & (slope_10m <= SLOPE_STRICT_MAX))

print("\n=== DIAGNOSIS SLOPE ===")
print("Slope available       :", slope_available)
print("Slope valid pixel     :", int(slope_valid_count))
print("Slope <= threshold px :", int(slope_low_count))

if slope_available and slope_low_count >= SLOPE_MIN_VALID_PIXEL:
    slope_used = True
    print("Slope digunakan sebagai filter.")
else:
    slope_used = False
    print("Slope TIDAK digunakan karena tidak lolos sanity check.")

if slope_used:
    slope_ok = slope_10m <= SLOPE_STRICT_MAX
else:
    slope_ok = np.ones_like(valid, dtype=bool)


# ============================================================
# 7. BACA REFERENSI SAWAH 2024
# ============================================================

if os.path.exists(SAWAH_2024_MASK_PATH):
    rice2024_mask = align_binary_mask(
        SAWAH_2024_MASK_PATH,
        src,
        positive_values=[1]
    )
    rice2024_available = True
else:
    rice2024_mask = np.zeros_like(valid, dtype=bool)
    rice2024_available = False

rice2024_mask = rice2024_mask & valid

print("\nRice 2024 available:", rice2024_available)
print("Luas referensi sawah 2024 grid 10 m:", round(rice2024_mask.sum() * 0.01, 2), "ha")


# ============================================================
# 8. FILTER SPEKTRAL DAN TEMPORAL
# ============================================================

pixel_area_ha = abs(src.transform.a * src.transform.e) / 10000.0
valid_area_ha = valid.sum() * pixel_area_ha

raw_rice_area_ha = np.sum((raw_class == 1) & valid) * pixel_area_ha

if previous_filtered is not None:
    previous_filtered_mask = (previous_filtered == 1) & valid
    previous_filtered_area_ha = previous_filtered_mask.sum() * pixel_area_ha
else:
    previous_filtered_mask = (raw_class == 1) & valid
    previous_filtered_area_ha = np.nan

ndvi_ok = (
    np.isfinite(NDVI_MED) &
    (NDVI_MED >= NDVI_MIN) &
    (NDVI_MED <= NDVI_MAX)
)

seasonality_outside_ok = (
    (
        np.isfinite(NDVI_AMP) &
        (NDVI_AMP >= NDVI_AMP_MIN_OUTSIDE)
    ) |
    (
        np.isfinite(LSWI_AMP) &
        (LSWI_AMP >= LSWI_AMP_MIN_OUTSIDE)
    )
)

forest_like = (
    np.isfinite(NDVI_MED) &
    np.isfinite(NDVI_STD) &
    np.isfinite(NDVI_AMP) &
    (NDVI_MED >= FOREST_NDVI_MIN) &
    (NDVI_STD <= FOREST_NDVI_STD_MAX) &
    (NDVI_AMP <= FOREST_NDVI_AMP_MAX)
)

water_like = (
    np.isfinite(MNDWI_MED) &
    (MNDWI_MED > MNDWI_WATER_THRESHOLD) &
    (NDVI_MED < 0.50)
)

built_like = (
    np.isfinite(NDBI_MED) &
    (NDBI_MED > NDBI_BUILT_THRESHOLD) &
    (NDVI_MED < 0.55)
)

basic_ok = (
    valid &
    ndvi_ok &
    (~forest_like) &
    (~water_like) &
    (~built_like) &
    slope_ok
)

print("\n=== DIAGNOSIS FILTER ===")
print("Valid area ha                 :", round(valid_area_ha, 2))
print("Raw RF rice area ha           :", round(raw_rice_area_ha, 2))
print("Previous filtered area ha     :", round(previous_filtered_area_ha, 2))
print("NDVI ok pixel                 :", int(ndvi_ok.sum()))
print("Seasonality outside ok pixel  :", int(seasonality_outside_ok.sum()))
print("Forest-like excluded pixel    :", int(forest_like.sum()))
print("Water-like excluded pixel     :", int(water_like.sum()))
print("Built-like excluded pixel     :", int(built_like.sum()))
print("Basic ok pixel                :", int(basic_ok.sum()))


# ============================================================
# 9. BUAT KANDIDAT MASK PER THRESHOLD
# ============================================================

def make_candidate(threshold):
    inside_ref = rice2024_mask & valid
    outside_ref = (~rice2024_mask) & valid

    # Komponen referensi 2024: dipertahankan jika sinyal RF dan spektral masih masuk akal.
    # Tidak wajib seasonality karena referensi 2024 sudah menjadi prior.
    ref_component = (
        inside_ref &
        (prob >= REF_PROB_MIN) &
        basic_ok
    )

    # Komponen luar referensi 2024: harus lebih ketat.
    outside_component = (
        outside_ref &
        previous_filtered_mask &
        (prob >= threshold) &
        basic_ok &
        seasonality_outside_ok
    )

    mask = ref_component | outside_component

    mask = binary_opening(mask, structure=np.ones((2, 2)))
    mask = binary_closing(mask, structure=np.ones((2, 2)))

    mask_clean, ncomp, min_pixels = remove_small_patches(
        mask,
        pixel_area_ha,
        MIN_PATCH_AREA_HA
    )

    return mask_clean, ref_component, outside_component, ncomp, min_pixels


threshold_rows = []
candidate_masks = {}

for th in PROB_THRESHOLDS:
    mask_th, ref_comp, out_comp, ncomp, min_pixels = make_candidate(th)

    area_ha = mask_th.sum() * pixel_area_ha
    ref_area_ha = (mask_th & rice2024_mask).sum() * pixel_area_ha
    outside_area_ha = (mask_th & (~rice2024_mask)).sum() * pixel_area_ha

    threshold_rows.append({
        "Probability_Threshold": th,
        "Rice_Area_ha": area_ha,
        "Rice_Area_km2": area_ha / 100,
        "Percent_of_valid_area": area_ha / valid_area_ha * 100,
        "Area_inside_2024_ref_ha": ref_area_ha,
        "Area_outside_2024_ref_ha": outside_area_ha,
        "Difference_from_target_ha": area_ha - TARGET_RICE_AREA_HA,
        "Abs_difference_from_target_ha": abs(area_ha - TARGET_RICE_AREA_HA),
        "Patch_count_before_cleaning": ncomp,
        "Minimum_patch_pixels": min_pixels
    })

    candidate_masks[th] = mask_th

threshold_df = pd.DataFrame(threshold_rows)

threshold_table_path = os.path.join(
    TABLE_DIR,
    "T01_FIXED_Calibrated_Rice_Area_By_Threshold.csv"
)

threshold_df.to_csv(threshold_table_path, index=False)

display(threshold_df)


# ============================================================
# 10. PILIH THRESHOLD TERBAIK
# ============================================================

within_range = threshold_df[
    (threshold_df["Rice_Area_ha"] >= TARGET_MIN_AREA_HA) &
    (threshold_df["Rice_Area_ha"] <= TARGET_MAX_AREA_HA)
].copy()

if len(within_range) > 0:
    selected_row = within_range.sort_values("Abs_difference_from_target_ha").iloc[0]
else:
    selected_row = threshold_df.sort_values("Abs_difference_from_target_ha").iloc[0]

SELECTED_THRESHOLD = float(selected_row["Probability_Threshold"])
selected_mask = candidate_masks[SELECTED_THRESHOLD]
selected_area_ha = selected_mask.sum() * pixel_area_ha

print("\n=== THRESHOLD TERPILIH ===")
print("Selected threshold:", SELECTED_THRESHOLD)
print("Selected rice area:", round(selected_area_ha, 2), "ha")
print("Difference from target:", round(selected_area_ha - TARGET_RICE_AREA_HA, 2), "ha")

if selected_area_ha < TARGET_MIN_AREA_HA or selected_area_ha > TARGET_MAX_AREA_HA:
    print("PERINGATAN: luas masih di luar rentang kewajaran.")
    print("Perlu cek visual atau perbaiki training RF.")
else:
    print("Luas berada dalam rentang kewajaran awal.")


# ============================================================
# 11. SIMPAN RASTER FINAL
# ============================================================

final_class = np.full(raw_class.shape, 255, dtype="uint8")
final_class[valid] = 0
final_class[selected_mask] = 1

final_path = os.path.join(
    RASTER_DIR,
    "Gowa_Rice_RF_2026_Calibrated_Final_FIXED_10m.tif"
)

write_raster(
    final_path,
    final_class,
    profile,
    dtype="uint8",
    nodata=255
)

prob_final = np.where(selected_mask, prob, np.nan).astype("float32")

prob_final_path = os.path.join(
    RASTER_DIR,
    "Gowa_Rice_RF_2026_Calibrated_Final_FIXED_Probability_10m.tif"
)

write_raster(
    prob_final_path,
    prob_final,
    profile,
    dtype="float32",
    nodata=-9999
)

ndvi_amp_path = os.path.join(
    RASTER_DIR,
    "Gowa_Rice_RF_2026_NDVI_Amplitude_10m.tif"
)

write_raster(
    ndvi_amp_path,
    NDVI_AMP,
    profile,
    dtype="float32",
    nodata=-9999
)

if slope_available:
    slope_path = os.path.join(
        RASTER_DIR,
        "Gowa_Rice_RF_2026_Slope_Aligned_10m.tif"
    )

    write_raster(
        slope_path,
        slope_10m,
        profile,
        dtype="float32",
        nodata=-9999
    )
else:
    slope_path = ""


# ============================================================
# 12. SIMPAN MASK KANDIDAT UNTUK CEK VISUAL
# ============================================================

for th in PROB_THRESHOLDS:
    if th in candidate_masks:
        arr = np.full(raw_class.shape, 255, dtype="uint8")
        arr[valid] = 0
        arr[candidate_masks[th]] = 1

        out_path = os.path.join(
            RASTER_DIR,
            f"Gowa_Rice_RF_2026_Candidate_FIXED_TH{str(th).replace('.', '_')}_10m.tif"
        )

        write_raster(
            out_path,
            arr,
            profile,
            dtype="uint8",
            nodata=255
        )


# ============================================================
# 13. PERBANDINGAN DENGAN REFERENSI 2024
# ============================================================

stable_ref_final = rice2024_mask & selected_mask
final_outside_ref = selected_mask & (~rice2024_mask)
ref_not_final = rice2024_mask & (~selected_mask)
outside_both = (~rice2024_mask) & (~selected_mask) & valid

comparison_map = np.full(raw_class.shape, 255, dtype="uint8")
comparison_map[outside_both] = 0
comparison_map[stable_ref_final] = 1
comparison_map[final_outside_ref] = 2
comparison_map[ref_not_final] = 3

comparison_path = os.path.join(
    RASTER_DIR,
    "Gowa_Rice_RF_2026_Calibrated_FIXED_vs_Reference2024_10m.tif"
)

write_raster(
    comparison_path,
    comparison_map,
    profile,
    dtype="uint8",
    nodata=255
)

comparison_df = pd.DataFrame([
    {
        "Class": 1,
        "Label": "Rice in both 2024 reference and calibrated 2026",
        "Area_ha": stable_ref_final.sum() * pixel_area_ha
    },
    {
        "Class": 2,
        "Label": "Calibrated 2026 rice outside 2024 reference",
        "Area_ha": final_outside_ref.sum() * pixel_area_ha
    },
    {
        "Class": 3,
        "Label": "2024 reference not retained in calibrated 2026",
        "Area_ha": ref_not_final.sum() * pixel_area_ha
    },
    {
        "Class": 0,
        "Label": "Outside both",
        "Area_ha": outside_both.sum() * pixel_area_ha
    }
])

comparison_df["Area_km2"] = comparison_df["Area_ha"] / 100
comparison_df["Percent"] = comparison_df["Area_ha"] / comparison_df["Area_ha"].sum() * 100

comparison_table_path = os.path.join(
    TABLE_DIR,
    "T03_FIXED_Calibrated_2026_vs_Reference2024.csv"
)

comparison_df.to_csv(comparison_table_path, index=False)

display(comparison_df)


# ============================================================
# 14. TABEL RINGKASAN FINAL
# ============================================================

summary_df = pd.DataFrame([
    {
        "Item": "Raw RF rice area",
        "Area_ha": raw_rice_area_ha,
        "Percent_of_valid_area": raw_rice_area_ha / valid_area_ha * 100
    },
    {
        "Item": "Previous conservative filtered area",
        "Area_ha": previous_filtered_area_ha,
        "Percent_of_valid_area": previous_filtered_area_ha / valid_area_ha * 100
    },
    {
        "Item": "Calibrated final fixed rice area",
        "Area_ha": selected_area_ha,
        "Percent_of_valid_area": selected_area_ha / valid_area_ha * 100
    },
    {
        "Item": "Target reference area",
        "Area_ha": TARGET_RICE_AREA_HA,
        "Percent_of_valid_area": TARGET_RICE_AREA_HA / valid_area_ha * 100
    },
    {
        "Item": "Slope used",
        "Area_ha": np.nan,
        "Percent_of_valid_area": np.nan
    }
])

summary_df.loc[summary_df["Item"] == "Slope used", "Area_ha"] = 1 if slope_used else 0

summary_path = os.path.join(
    TABLE_DIR,
    "T02_FIXED_Calibrated_Final_Summary.csv"
)

summary_df.to_csv(summary_path, index=False)

display(summary_df)


# ============================================================
# 15. TITIK VALIDASI VISUAL
# ============================================================

high_conf_rice = (
    selected_mask &
    np.isfinite(prob) &
    (prob >= max(SELECTED_THRESHOLD, 0.80))
)

uncertain_edge = (
    valid &
    np.isfinite(prob) &
    (prob >= SELECTED_THRESHOLD - 0.05) &
    (prob <= SELECTED_THRESHOLD + 0.05)
)

high_conf_nonrice = (
    valid &
    (~selected_mask) &
    np.isfinite(prob) &
    (prob <= 0.15)
)

n_each = N_VALIDATION_POINTS // 3

val1 = sample_points_from_mask(
    high_conf_rice,
    n_each,
    "high_conf_rice",
    src.transform,
    prob,
    final_class,
    RANDOM_SEED
)

val2 = sample_points_from_mask(
    uncertain_edge,
    n_each,
    "uncertain_edge",
    src.transform,
    prob,
    final_class,
    RANDOM_SEED + 1
)

val3 = sample_points_from_mask(
    high_conf_nonrice,
    n_each,
    "high_conf_nonrice",
    src.transform,
    prob,
    final_class,
    RANDOM_SEED + 2
)

validation_df = pd.concat([val1, val2, val3], ignore_index=True)

validation_csv_path = os.path.join(
    VALIDATION_DIR,
    "Gowa_Rice_RF_2026_Calibrated_FIXED_Visual_Validation_Points.csv"
)

validation_df.to_csv(validation_csv_path, index=False)

validation_gdf = gpd.GeoDataFrame(
    validation_df,
    geometry=[
        Point(xy_pair) for xy_pair in zip(validation_df["x"], validation_df["y"])
    ],
    crs=src.crs
)

validation_geojson_path = os.path.join(
    VALIDATION_DIR,
    "Gowa_Rice_RF_2026_Calibrated_FIXED_Visual_Validation_Points.geojson"
)

validation_gdf.to_file(validation_geojson_path, driver="GeoJSON")

print("Validation CSV    :", validation_csv_path)
print("Validation GeoJSON:", validation_geojson_path)


# ============================================================
# 16. BATAS GOWA UNTUK VISUALISASI
# ============================================================

boundary_candidates = find_file(
    [SENTINEL_DIR, MMPI_GEE_DIR, DRIVE_ROOT],
    [
        "Gowa_Boundary_GAUL2015_2026.geojson",
        "*Gowa*Boundary*.geojson",
        "*Gowa*boundary*.geojson",
        "*Gowa*Boundary*.shp"
    ]
)

gowa_admin = None

if len(boundary_candidates) > 0:
    boundary_path = boundary_candidates[0]
    gowa_admin = gpd.read_file(boundary_path)

    if gowa_admin.crs is None:
        gowa_admin = gowa_admin.set_crs("EPSG:4326")

    gowa_admin = gowa_admin.to_crs(src.crs)
    print("Boundary:", boundary_path)
else:
    print("Batas Gowa tidak ditemukan.")


def plot_boundary(ax):
    if gowa_admin is not None:
        gowa_admin.boundary.plot(
            ax=ax,
            edgecolor="black",
            linewidth=0.8,
            zorder=10
        )


# ============================================================
# 17. VISUALISASI 1 - RAW, PREVIOUS, FIXED FINAL
# ============================================================

extent = plotting_extent(src)

cmap_binary = ListedColormap(["#f0f0f0", "#238b45", "#ffffff"])
norm_binary = BoundaryNorm([-0.5, 0.5, 1.5, 255.5], cmap_binary.N)

if previous_filtered is None:
    previous_filtered_plot = np.full_like(raw_class, 255, dtype="uint8")
else:
    previous_filtered_plot = previous_filtered

fig, axes = plt.subplots(1, 3, figsize=(16, 5.8))

maps = [
    (raw_class, "Raw RF classification"),
    (previous_filtered_plot, "Previous conservative filter"),
    (final_class, "Fixed calibrated final mask")
]

for ax, (arr, title) in zip(axes, maps):
    ax.imshow(
        arr,
        extent=extent,
        origin="upper",
        cmap=cmap_binary,
        norm=norm_binary
    )
    plot_boundary(ax)
    format_axis(ax)
    ax.set_title(title, fontsize=10)

legend_elements = [
    Patch(facecolor="#238b45", edgecolor="black", label="Rice"),
    Patch(facecolor="#f0f0f0", edgecolor="black", label="Non-rice")
]

axes[0].legend(handles=legend_elements, loc="lower left", fontsize=8)

fig.tight_layout()

fig1_path = save_fig(
    fig,
    "01_FIXED_Raw_Previous_Filter_Calibrated_Final_DPI600.png"
)


# ============================================================
# 18. VISUALISASI 2 - PROBABILITY, NDVI AMP, FINAL
# ============================================================

fig, axes = plt.subplots(1, 3, figsize=(16, 5.8))

im1 = axes[0].imshow(
    prob,
    extent=extent,
    origin="upper",
    cmap="viridis",
    vmin=0,
    vmax=1
)
plot_boundary(axes[0])
format_axis(axes[0])
axes[0].set_title("Rice probability", fontsize=10)
cbar1 = plt.colorbar(im1, ax=axes[0], orientation="horizontal", fraction=0.046, pad=0.07)
cbar1.set_label("Probability", fontsize=8)
cbar1.ax.tick_params(labelsize=7)

vmax_amp = np.nanpercentile(NDVI_AMP[np.isfinite(NDVI_AMP)], 98)
im2 = axes[1].imshow(
    NDVI_AMP,
    extent=extent,
    origin="upper",
    cmap="magma",
    vmin=0,
    vmax=vmax_amp
)
plot_boundary(axes[1])
format_axis(axes[1])
axes[1].set_title("NDVI temporal amplitude", fontsize=10)
cbar2 = plt.colorbar(im2, ax=axes[1], orientation="horizontal", fraction=0.046, pad=0.07)
cbar2.set_label("NDVI amplitude", fontsize=8)
cbar2.ax.tick_params(labelsize=7)

axes[2].imshow(
    final_class,
    extent=extent,
    origin="upper",
    cmap=cmap_binary,
    norm=norm_binary
)
plot_boundary(axes[2])
format_axis(axes[2])
axes[2].set_title("Fixed calibrated final rice mask", fontsize=10)
axes[2].legend(handles=legend_elements, loc="lower left", fontsize=8)

fig.tight_layout()

fig2_path = save_fig(
    fig,
    "02_FIXED_Probability_NDVIAmplitude_FinalMask_DPI600.png"
)


# ============================================================
# 19. VISUALISASI 3 - SENSITIVITAS THRESHOLD
# ============================================================

fig, ax = plt.subplots(figsize=(8.8, 5.4))

ax.plot(
    threshold_df["Probability_Threshold"],
    threshold_df["Rice_Area_ha"],
    marker="o",
    linewidth=1.5
)

ax.axhline(
    TARGET_RICE_AREA_HA,
    linestyle="--",
    linewidth=1.2,
    label=f"Reference target: {TARGET_RICE_AREA_HA:,.0f} ha"
)

ax.axhspan(
    TARGET_MIN_AREA_HA,
    TARGET_MAX_AREA_HA,
    alpha=0.15,
    label=f"Reasonable range: {TARGET_MIN_AREA_HA:,.0f}–{TARGET_MAX_AREA_HA:,.0f} ha"
)

ax.scatter(
    [SELECTED_THRESHOLD],
    [selected_area_ha],
    s=80,
    zorder=5,
    label=f"Selected: {SELECTED_THRESHOLD}"
)

ax.set_xlabel("Rice probability threshold", fontsize=10)
ax.set_ylabel("Calibrated rice area (ha)", fontsize=10)
ax.grid(alpha=0.30)
ax.tick_params(axis="both", labelsize=9)
ax.legend(fontsize=8)

for x, y in zip(
    threshold_df["Probability_Threshold"],
    threshold_df["Rice_Area_ha"]
):
    ax.text(
        x,
        y,
        f"{y:,.0f}",
        fontsize=7,
        ha="center",
        va="bottom"
    )

fig.tight_layout()

fig3_path = save_fig(
    fig,
    "03_FIXED_Rice_Area_Sensitivity_By_Threshold_DPI600.png"
)


# ============================================================
# 20. VISUALISASI 4 - PERBANDINGAN 2024 DAN FINAL 2026
# ============================================================

comparison_cmap = ListedColormap([
    "#f0f0f0",
    "#238b45",
    "#2b8cbe",
    "#d95f02",
    "#ffffff"
])

comparison_norm = BoundaryNorm(
    [-0.5, 0.5, 1.5, 2.5, 3.5, 255.5],
    comparison_cmap.N
)

fig, ax = plt.subplots(figsize=(8.5, 7.5))

ax.imshow(
    comparison_map,
    extent=extent,
    origin="upper",
    cmap=comparison_cmap,
    norm=comparison_norm
)

plot_boundary(ax)
format_axis(ax)

legend_items = [
    Patch(facecolor="#238b45", edgecolor="black", label="Rice in both 2024 reference and fixed 2026"),
    Patch(facecolor="#2b8cbe", edgecolor="black", label="Fixed 2026 rice outside 2024 reference"),
    Patch(facecolor="#d95f02", edgecolor="black", label="2024 reference not retained"),
    Patch(facecolor="#f0f0f0", edgecolor="black", label="Outside both")
]

ax.legend(
    handles=legend_items,
    loc="lower left",
    fontsize=7,
    frameon=True
)

fig.tight_layout()

fig4_path = save_fig(
    fig,
    "04_FIXED_Calibrated_2026_vs_Reference2024_DPI600.png"
)


# ============================================================
# 21. SIMPAN EXCEL DAN INDEX
# ============================================================

excel_path = os.path.join(
    OUT_DIR,
    "Gowa_Rice_RF_2026_Calibrated_Final_FIXED_Result.xlsx"
)

with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
    threshold_df.to_excel(writer, sheet_name="Threshold_Sensitivity", index=False)
    summary_df.to_excel(writer, sheet_name="Final_Summary", index=False)
    comparison_df.to_excel(writer, sheet_name="Comparison_2024_2026", index=False)
    validation_df.to_excel(writer, sheet_name="Validation_Points", index=False)

index_df = pd.DataFrame([
    {
        "No": "R01",
        "Name": "Fixed calibrated final rice mask 2026",
        "File": final_path,
        "Use": "Final 10 m rice mask candidate for manuscript/MMPI analysis"
    },
    {
        "No": "R02",
        "Name": "Final probability masked raster",
        "File": prob_final_path,
        "Use": "Rice probability values within fixed calibrated mask"
    },
    {
        "No": "T01",
        "Name": "Threshold sensitivity table",
        "File": threshold_table_path,
        "Use": "Justification for threshold selection"
    },
    {
        "No": "T02",
        "Name": "Final summary table",
        "File": summary_path,
        "Use": "Main area summary"
    },
    {
        "No": "T03",
        "Name": "2024 versus 2026 comparison table",
        "File": comparison_table_path,
        "Use": "Reference comparison"
    },
    {
        "No": "V01",
        "Name": "Visual validation GeoJSON",
        "File": validation_geojson_path,
        "Use": "Open in QGIS/Google Earth for visual checking"
    },
    {
        "No": "F01",
        "Name": "Raw, previous, and fixed final map",
        "File": fig1_path,
        "Use": "Main diagnostic figure"
    },
    {
        "No": "F02",
        "Name": "Probability, NDVI amplitude, and final mask",
        "File": fig2_path,
        "Use": "Diagnostic figure"
    },
    {
        "No": "F03",
        "Name": "Area sensitivity by threshold",
        "File": fig3_path,
        "Use": "Threshold calibration figure"
    },
    {
        "No": "F04",
        "Name": "Fixed calibrated 2026 versus 2024 reference",
        "File": fig4_path,
        "Use": "Reference comparison figure"
    },
    {
        "No": "X01",
        "Name": "Excel result",
        "File": excel_path,
        "Use": "All tables"
    }
])

index_path = os.path.join(
    OUT_DIR,
    "OUTPUT_INDEX_Gowa_Rice_RF_2026_Calibrated_Final_FIXED.csv"
)

index_df.to_csv(index_path, index=False)

display(index_df)


# ============================================================
# 22. LAPORAN TEKS
# ============================================================

report = f"""
FIXED CALIBRATED RICE-FIELD CLASSIFICATION REPORT
Gowa Regency, Sentinel-1/Sentinel-2, 2026

1. Diagnosis of previous calibration
The previous calibration produced zero rice area because the slope filter eliminated all valid candidate pixels.
In this fixed version, slope is used only if it passes a sanity check.

2. Initial condition
- Raw RF rice area: {raw_rice_area_ha:,.2f} ha
- Previous conservative filtered area: {previous_filtered_area_ha:,.2f} ha
- 2024 reference rice area on 10 m grid: {rice2024_mask.sum() * pixel_area_ha:,.2f} ha

3. Fixed calibration approach
The final mask combines:
- RF rice probability
- 2024 rice reference as prior information
- NDVI range filtering
- NDVI/LSWI temporal amplitude for pixels outside the 2024 reference
- forest-like vegetation exclusion
- water-like pixel exclusion
- built-up-like pixel exclusion
- optional slope filtering only if valid
- small patch removal

4. Selected result
- Selected probability threshold: {SELECTED_THRESHOLD}
- Fixed calibrated final rice area: {selected_area_ha:,.2f} ha
- Reference target area: {TARGET_RICE_AREA_HA:,.2f} ha
- Difference from target: {selected_area_ha - TARGET_RICE_AREA_HA:,.2f} ha
- Slope used: {slope_used}

5. Main output
{final_path}

6. Interpretation
This raster should be treated as a calibrated RF-derived rice-field mask for 2026.
It is suitable for further manuscript/MMPI analysis only if the spatial pattern appears plausible after visual inspection.
"""

report_path = os.path.join(
    OUT_DIR,
    "REPORT_Gowa_Rice_RF_2026_Calibrated_Final_FIXED.txt"
)

with open(report_path, "w", encoding="utf-8") as f:
    f.write(report)

print(report)
print("Report:", report_path)


# ============================================================
# 23. RINGKASAN AKHIR
# ============================================================

print("============================================================")
print("KALIBRASI FIXED SAWAH 2026 SELESAI")
print("============================================================")
print("Raster final:")
print(final_path)
print("")
print("Luas sawah final fixed:")
print(round(selected_area_ha, 2), "ha")
print("")
print("Threshold terpilih:")
print(SELECTED_THRESHOLD)
print("")
print("Slope digunakan:")
print(slope_used)
print("")
print("Tabel sensitivitas:")
print(threshold_table_path)
print("")
print("Figure DPI600:")
print(FIG_DIR)
print("")
print("Validasi visual:")
print(validation_geojson_path)
print("============================================================")
