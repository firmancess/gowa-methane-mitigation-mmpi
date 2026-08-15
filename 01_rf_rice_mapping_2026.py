"""
Random Forest Rice-Field Mapping — Gowa, 2026
=============================================

Builds the initial January–July 2026 Sentinel-1/Sentinel-2 Random Forest rice probability and binary rice-field rasters. The 2024 rice raster is used only as a positive training seed.

This file was selected from the final successful workflow in the uploaded analysis notebook.
Superseded/failed notebook cells were intentionally excluded.
"""

# ============================================================
# RANDOM FOREST CLASSIFICATION
# GOWA RICE-FIELD MAPPING 2026 BASED ON SENTINEL-1 + SENTINEL-2
# FINAL VERSION FOR GEE EXPORTS SPLIT INTO TILES
# ============================================================

# ============================================================
# 0. INSTALL LIBRARIES
# ============================================================


# Install the GDAL command-line utility used to build a VRT mosaic



# ============================================================
# 1. IMPORT LIBRARIES
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
from rasterio.windows import Window
from rasterio.plot import plotting_extent
from rasterio.transform import xy

from shapely.geometry import Point

from scipy.ndimage import binary_erosion, binary_dilation
from scipy.ndimage import label as ndi_label

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    f1_score
)
from sklearn.impute import SimpleImputer

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch
from matplotlib.ticker import ScalarFormatter, MaxNLocator

try:
    from google.colab import drive
    drive.mount('/content/drive')
except Exception:
    pass


# ============================================================
# 2. FOLDER SETTINGS
# ============================================================

DRIVE_ROOT = os.environ.get("MMPI_DRIVE_ROOT", "/content/drive/MyDrive")

# Folder containing downloads from Google Earth Engine
SENTINEL_DIR = os.path.join(
    DRIVE_ROOT,
    "Gowa_Sentinel_2026_RF_Input"
)

# Classification output folder
OUT_DIR = os.path.join(
    DRIVE_ROOT,
    "Gowa_Rice_RF_Classification_2026"
)

FIG_DIR = os.path.join(
    OUT_DIR,
    "Figures_DPI600"
)

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

# Main RF-stack filename exported from GEE
RF_STACK_SINGLE_PATH = os.path.join(
    SENTINEL_DIR,
    "Gowa_Sentinel_2026_RF_Input_Stack_10m.tif"
)

# Tile filename pattern when a GEE export is split
RF_STACK_TILE_PATTERN = os.path.join(
    SENTINEL_DIR,
    "Gowa_Sentinel_2026_RF_Input_Stack_10m-*.tif"
)

# 2024 rice mask used as the positive training seed
RICE_2024_MASK_PATH = os.path.join(
    DRIVE_ROOT,
    "MMPI_Gowa_Analysis_Output_30m",
    "Sawah_2024_Mamminasata_Gowa_FINAL",
    "05_Sawah_2024_Gowa_binary_mask_aligned_to_MMPI_30m.tif"
)

# Fallback if the final mask is not found
SAWAH_2024_ORIGINAL_PATH = os.path.join(
    DRIVE_ROOT,
    "InputTiff",
    "Sawah_2024.tif"
)

print("Folder Sentinel :", SENTINEL_DIR)
print("Output          :", OUT_DIR)
print("RF stack single :", RF_STACK_SINGLE_PATH)
print("2024 rice mask :", RICE_2024_MASK_PATH)


# ============================================================
# 3. ANALYSIS PARAMETERS
# ============================================================

RANDOM_SEED = 2026

N_RICE_SAMPLE = 7000
N_NONRICE_SAMPLE = 7000

RICE_PROB_THRESHOLD = 0.50

ERODE_RICE_ITER = 1
NONRICE_BUFFER_PIXEL = 10

TILE_SIZE = 512

REMOVE_SMALL_PATCHES = True
MIN_PATCH_AREA_HA = 0.25

print("Random seed:", RANDOM_SEED)


# ============================================================
# 4. GENERAL FUNCTIONS
# ============================================================

def find_file(root, patterns):
    if isinstance(patterns, str):
        patterns = [patterns]

    found = []

    for p in patterns:
        found.extend(
            glob.glob(
                os.path.join(root, "**", p),
                recursive=True
            )
        )

    return sorted(list(set(found)))


def find_first_file(root, patterns, required=True, label="file"):
    files = find_file(root, patterns)

    print(f"\nCandidate {label}:")
    for f in files[:20]:
        print("-", f)

    if len(files) == 0:
        if required:
            raise FileNotFoundError(f"{label} was not found.")
        else:
            return None

    return files[0]


def build_vrt_from_tiles(tile_files, vrt_path):
    """
    Build a VRT mosaic from a GEE export that was split into multiple tiles.
    The VRT does not duplicate data; it only creates a mosaic reference.
    """

    if len(tile_files) == 0:
        raise FileNotFoundError("No RF-stack tiles were found for VRT creation.")

    gdalbuildvrt_path = "/usr/bin/gdalbuildvrt"

    if not os.path.exists(gdalbuildvrt_path):
        raise FileNotFoundError(
            "gdalbuildvrt is unavailable. Ensure that gdal-bin is installed."
        )

    tile_list_path = vrt_path.replace(".vrt", "_tile_list.txt")

    with open(tile_list_path, "w") as f:
        for tile in tile_files:
            f.write(tile + "\n")

    cmd = [
        gdalbuildvrt_path,
        "-overwrite",
        "-input_file_list",
        tile_list_path,
        vrt_path
    ]

    print("\nBuilding a VRT mosaic from RF-stack tiles...")
    print("Number of tiles:", len(tile_files))
    print("VRT path:", vrt_path)

    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    print(result.stdout)

    if result.returncode != 0:
        print(result.stderr)
        raise RuntimeError("Failed to build the RF-stack VRT.")

    if not os.path.exists(vrt_path):
        raise FileNotFoundError("VRT creation failed.")

    return vrt_path


def expected_rf_band_names_75():
    """
    Fallback band names if the VRT loses band descriptions.
    Matches the rfStack order used in the GEE script:
    s2Median + s2Temporal + s1Stats + Q1 + Q2 + Q3_partial.
    """

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


def get_band_names(src, fallback_tile=None):
    names = []

    for i, desc in enumerate(src.descriptions, start=1):
        if desc is None or str(desc).strip() == "":
            names.append(f"band_{i}")
        else:
            names.append(str(desc))

    # If the VRT loses band names, try reading band names from the first tile
    names_are_generic = all(
        name.startswith("band_") for name in names
    )

    if names_are_generic and fallback_tile is not None:
        try:
            with rasterio.open(fallback_tile) as tile_src:
                tile_names = []

                for i, desc in enumerate(tile_src.descriptions, start=1):
                    if desc is None or str(desc).strip() == "":
                        tile_names.append(f"band_{i}")
                    else:
                        tile_names.append(str(desc))

                tile_generic = all(
                    name.startswith("band_") for name in tile_names
                )

                if not tile_generic and len(tile_names) == src.count:
                    names = tile_names
        except Exception:
            pass

    # If names are still generic and the stack has 75 bands, use the expected band names
    names_are_generic = all(
        name.startswith("band_") for name in names
    )

    if names_are_generic and src.count == 75:
        names = expected_rf_band_names_75()

    return names


def find_band_index(band_names, candidates, required=True):
    """
    Mengembalikan index band rasterio 1-based.
    """

    if isinstance(candidates, str):
        candidates = [candidates]

    lower_names = [b.lower() for b in band_names]

    # Exact match
    for cand in candidates:
        cand_lower = cand.lower()
        for i, name in enumerate(lower_names):
            if name == cand_lower:
                return i + 1

    # Contains match
    for cand in candidates:
        cand_lower = cand.lower()
        for i, name in enumerate(lower_names):
            if cand_lower in name:
                return i + 1

    if required:
        raise ValueError(
            "Band was not found. Candidates: "
            + ", ".join(candidates)
            + "\n\nAvailable bands:\n"
            + "\n".join(band_names)
        )

    return None


def read_band_as_float(src, band_index):
    arr = src.read(band_index).astype("float32")
    nodata = src.nodata

    if nodata is not None:
        arr[arr == nodata] = np.nan

    arr[np.isinf(arr)] = np.nan

    return arr


def align_mask_to_ref(mask_path, ref_src, rice_values=[1]):
    with rasterio.open(mask_path) as src_mask:
        mask_arr = src_mask.read(1)

        mask_bin = np.isin(mask_arr, rice_values).astype("uint8")

        same_grid = (
            src_mask.width == ref_src.width and
            src_mask.height == ref_src.height and
            src_mask.crs == ref_src.crs and
            src_mask.transform == ref_src.transform
        )

        if same_grid:
            return mask_bin == 1

        aligned = np.zeros(
            (ref_src.height, ref_src.width),
            dtype="uint8"
        )

        reproject(
            source=mask_bin,
            destination=aligned,
            src_transform=src_mask.transform,
            src_crs=src_mask.crs,
            dst_transform=ref_src.transform,
            dst_crs=ref_src.crs,
            resampling=Resampling.nearest
        )

        return aligned == 1


def random_sample_from_mask(mask, n, seed=2026):
    rng = np.random.default_rng(seed)

    rows, cols = np.where(mask)

    if len(rows) == 0:
        return np.array([], dtype=int), np.array([], dtype=int)

    n_take = min(n, len(rows))

    idx = rng.choice(
        len(rows),
        size=n_take,
        replace=False
    )

    return rows[idx], cols[idx]


def rows_cols_to_coords(transform, rows, cols):
    coords = []

    for r, c in zip(rows, cols):
        x_val, y_val = xy(transform, r, c, offset="center")
        coords.append((x_val, y_val))

    return coords


def sample_stack_values(src, rows, cols):
    coords = rows_cols_to_coords(src.transform, rows, cols)
    values = np.array(list(src.sample(coords)), dtype="float32")
    return values, coords


def window_grid(width, height, tile_size):
    for row_off in range(0, height, tile_size):
        for col_off in range(0, width, tile_size):
            win_width = min(tile_size, width - col_off)
            win_height = min(tile_size, height - row_off)

            yield Window(
                col_off=col_off,
                row_off=row_off,
                width=win_width,
                height=win_height
            )


def format_map_axis(ax):
    ax.set_xlabel("Easting (m)", fontsize=8)
    ax.set_ylabel("Northing (m)", fontsize=8)

    x_formatter = ScalarFormatter(useOffset=False)
    x_formatter.set_scientific(False)

    y_formatter = ScalarFormatter(useOffset=False)
    y_formatter.set_scientific(False)

    ax.xaxis.set_major_formatter(x_formatter)
    ax.yaxis.set_major_formatter(y_formatter)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4))

    ax.tick_params(axis="both", labelsize=7)
    ax.set_aspect("equal", adjustable="box")


def read_downsample(path, max_size=1800):
    with rasterio.open(path) as rsrc:
        scale = max(rsrc.width / max_size, rsrc.height / max_size, 1)

        out_h = int(rsrc.height / scale)
        out_w = int(rsrc.width / scale)

        arr = rsrc.read(
            1,
            out_shape=(out_h, out_w),
            resampling=Resampling.nearest
        )

        ext = plotting_extent(rsrc)

    return arr, ext


# ============================================================
# 5. CHECK AND OPEN RF STACK
# ============================================================

rf_stack_tiles = sorted(glob.glob(RF_STACK_TILE_PATTERN))

print("\n=== CHECK RF STACK FILES ===")
print("Single RF stack:", RF_STACK_SINGLE_PATH)
print("Single file exists:", os.path.exists(RF_STACK_SINGLE_PATH))
print("Number of RF-stack tiles:", len(rf_stack_tiles))

for f in rf_stack_tiles:
    print("-", f)

fallback_tile_for_bandnames = None

if os.path.exists(RF_STACK_SINGLE_PATH):
    RF_STACK_PATH = RF_STACK_SINGLE_PATH
    print("\nUsing single RF stack:")
    print(RF_STACK_PATH)

elif len(rf_stack_tiles) > 0:
    fallback_tile_for_bandnames = rf_stack_tiles[0]

    RF_STACK_PATH = os.path.join(
        SENTINEL_DIR,
        "Gowa_Sentinel_2026_RF_Input_Stack_10m_MOSAIC.vrt"
    )

    RF_STACK_PATH = build_vrt_from_tiles(
        rf_stack_tiles,
        RF_STACK_PATH
    )

    print("\nUsing RF-stack VRT mosaic:")
    print(RF_STACK_PATH)

else:
    raise FileNotFoundError(
        "RF stack utama was not found.\n"
        "Search targets:\n"
        "1. Gowa_Sentinel_2026_RF_Input_Stack_10m.tif\n"
        "2. Gowa_Sentinel_2026_RF_Input_Stack_10m-*.tif\n\n"
        "Ensure that the GEE RF Input Stack export has completed."
    )


src = rasterio.open(RF_STACK_PATH)
ref_profile = src.profile.copy()

# If the source is a VRT, keep the output as GeoTIFF
ref_profile.update({
    "driver": "GTiff",
    "compress": "lzw"
})

band_names = get_band_names(
    src,
    fallback_tile=fallback_tile_for_bandnames
)

print("\n=== RF STACK INFORMATION ===")
print("Path     :", RF_STACK_PATH)
print("Driver   :", src.driver)
print("CRS      :", src.crs)
print("Resolution:", src.res)
print("Width    :", src.width)
print("Height   :", src.height)
print("Bands    :", src.count)
print("Nodata   :", src.nodata)

print("\nBand list:")
for i, b in enumerate(band_names, start=1):
    print(i, b)

band_names_lower = [b.lower() for b in band_names]

has_ndvi = any("ndvi" in b for b in band_names_lower)
has_s1 = any("vv" in b or "vh" in b or "rvi" in b for b in band_names_lower)

if not has_ndvi:
    raise ValueError(
        "The loaded RF stack does not contain an NDVI band.\n"
        "The loaded file may not be the combined RF Input Stack.\n\n"
        "Available bands:\n"
        + "\n".join(band_names)
    )

if not has_s1:
    print("WARNING: Sentinel-1 bands were not detected in the RF stack.")

print("\nThe RF stack is valid and ready for classification.")


# ============================================================
# 6. SELECT KEY BANDS FOR TRAINING SEEDS
# ============================================================

ndvi_idx = find_band_index(
    band_names,
    [
        "S2_NDVI_median",
        "NDVI_median",
        "NDVI_mean",
        "NDVI"
    ],
    required=True
)

lswi_idx = find_band_index(
    band_names,
    [
        "S2_LSWI_median",
        "LSWI_median",
        "LSWI_mean",
        "LSWI"
    ],
    required=False
)

mndwi_idx = find_band_index(
    band_names,
    [
        "S2_MNDWI_median",
        "MNDWI_median",
        "MNDWI_mean",
        "MNDWI"
    ],
    required=False
)

ndwi_idx = find_band_index(
    band_names,
    [
        "S2_NDWI_median",
        "NDWI_median",
        "NDWI_mean",
        "NDWI"
    ],
    required=False
)

ndbi_idx = find_band_index(
    band_names,
    [
        "S2_NDBI_median",
        "NDBI_median",
        "NDBI_mean",
        "NDBI"
    ],
    required=False
)

rvi_idx = find_band_index(
    band_names,
    [
        "RVI_mean",
        "S1_RVI",
        "RVI"
    ],
    required=False
)

print("\nBand NDVI :", ndvi_idx, band_names[ndvi_idx - 1])
print("Band LSWI :", lswi_idx, band_names[lswi_idx - 1] if lswi_idx else None)
print("Band MNDWI:", mndwi_idx, band_names[mndwi_idx - 1] if mndwi_idx else None)
print("Band NDWI :", ndwi_idx, band_names[ndwi_idx - 1] if ndwi_idx else None)
print("Band NDBI :", ndbi_idx, band_names[ndbi_idx - 1] if ndbi_idx else None)
print("Band RVI  :", rvi_idx, band_names[rvi_idx - 1] if rvi_idx else None)


# ============================================================
# 7. READ BANDS FOR AUTOMATED SAMPLE GENERATION
# ============================================================

NDVI = read_band_as_float(src, ndvi_idx)

LSWI = read_band_as_float(src, lswi_idx) if lswi_idx else np.full_like(NDVI, np.nan)
MNDWI = read_band_as_float(src, mndwi_idx) if mndwi_idx else np.full_like(NDVI, np.nan)
NDWI = read_band_as_float(src, ndwi_idx) if ndwi_idx else np.full_like(NDVI, np.nan)
NDBI = read_band_as_float(src, ndbi_idx) if ndbi_idx else np.full_like(NDVI, np.nan)
RVI = read_band_as_float(src, rvi_idx) if rvi_idx else np.full_like(NDVI, np.nan)

valid_feature_mask = np.isfinite(NDVI)

print("\nValid feature pixel:", int(valid_feature_mask.sum()))


# ============================================================
# 8. READ 2024 RICE DATA AS POSITIVE SEED
# ============================================================

if os.path.exists(RICE_2024_MASK_PATH):
    seed_path = RICE_2024_MASK_PATH

elif os.path.exists(SAWAH_2024_ORIGINAL_PATH):
    seed_path = SAWAH_2024_ORIGINAL_PATH

else:
    candidates = find_file(
        DRIVE_ROOT,
        [
            "05_Sawah_2024_Gowa_binary_mask_aligned_to_MMPI_30m.tif",
            "Sawah_2024.tif",
            "*Sawah_2024*.tif"
        ]
    )

    print("\nCandidate 2024 rice datasets:")
    for c in candidates[:20]:
        print("-", c)

    if len(candidates) == 0:
        raise FileNotFoundError("The 2024 rice dataset was not found.")

    seed_path = candidates[0]

print("\n2024 rice dataset used as the positive seed:")
print(seed_path)

rice_seed_raw = align_mask_to_ref(
    seed_path,
    src,
    rice_values=[1]
)

rice_seed_raw = rice_seed_raw & valid_feature_mask

print("Raw rice-seed pixels:", int(rice_seed_raw.sum()))

rice_seed_eroded = binary_erosion(
    rice_seed_raw,
    structure=np.ones((3, 3)),
    iterations=ERODE_RICE_ITER
)

if rice_seed_eroded.sum() < 100:
    print("Too few rice-seed pixels remain after erosion. Using rice_seed_raw.")
    rice_seed = rice_seed_raw.copy()
else:
    rice_seed = rice_seed_eroded.copy()

print("Final rice-seed pixels:", int(rice_seed.sum()))


# ============================================================
# 9. CREATE AUTOMATED NON-RICE SEEDS
# ============================================================

rice_buffer = binary_dilation(
    rice_seed_raw,
    structure=np.ones((3, 3)),
    iterations=NONRICE_BUFFER_PIXEL
)

not_near_rice = ~rice_buffer

# Water
water_candidate = np.zeros_like(valid_feature_mask, dtype=bool)

if mndwi_idx is not None:
    water_candidate |= (
        (MNDWI > 0.20) &
        (NDVI < 0.45)
    )

if ndwi_idx is not None:
    water_candidate |= (
        (NDWI > 0.20) &
        (NDVI < 0.45)
    )

# Built-up
built_candidate = np.zeros_like(valid_feature_mask, dtype=bool)

if ndbi_idx is not None:
    built_candidate |= (
        (NDBI > 0.05) &
        (NDVI < 0.50)
    )

# Bare/open land
bare_candidate = (
    (NDVI < 0.25) &
    ((np.isnan(MNDWI)) | (MNDWI < 0.10))
)

# Permanent vegetation / forest / dense plantations
forest_candidate = (
    (NDVI > 0.78) &
    ((np.isnan(LSWI)) | (LSWI < 0.35))
)

nonrice_seed = (
    water_candidate |
    built_candidate |
    bare_candidate |
    forest_candidate
)

nonrice_seed = (
    nonrice_seed &
    valid_feature_mask &
    not_near_rice &
    (~rice_seed_raw)
)

print("\n=== NUMBER OF NON-RICE CANDIDATES ===")
print("Water candidate :", int((water_candidate & valid_feature_mask).sum()))
print("Built candidate :", int((built_candidate & valid_feature_mask).sum()))
print("Bare candidate  :", int((bare_candidate & valid_feature_mask).sum()))
print("Forest candidate:", int((forest_candidate & valid_feature_mask).sum()))
print("Non-rice seed final:", int(nonrice_seed.sum()))

if rice_seed.sum() < 100:
    raise ValueError(
        "Too few rice samples were generated. Check the 2024 rice dataset."
    )

if nonrice_seed.sum() < 100:
    raise ValueError(
        "Too few non-rice samples were generated. Consider relaxing the threshold."
    )


# ============================================================
# 10. DRAW TRAINING SAMPLES
# ============================================================

rice_rows, rice_cols = random_sample_from_mask(
    rice_seed,
    N_RICE_SAMPLE,
    seed=RANDOM_SEED
)

nonrice_rows, nonrice_cols = random_sample_from_mask(
    nonrice_seed,
    N_NONRICE_SAMPLE,
    seed=RANDOM_SEED + 1
)

print("\nNumber of rice samples     :", len(rice_rows))
print("Number of non-rice samples :", len(nonrice_rows))

all_rows = np.concatenate([rice_rows, nonrice_rows])
all_cols = np.concatenate([rice_cols, nonrice_cols])

all_y = np.concatenate([
    np.ones(len(rice_rows), dtype=np.uint8),
    np.zeros(len(nonrice_rows), dtype=np.uint8)
])

X_all, coords_all = sample_stack_values(
    src,
    all_rows,
    all_cols
)

valid_sample = np.any(np.isfinite(X_all), axis=1)

X = X_all[valid_sample]
y = all_y[valid_sample]
rows_valid = all_rows[valid_sample]
cols_valid = all_cols[valid_sample]
coords_valid = [coords_all[i] for i in np.where(valid_sample)[0]]

print("Total valid samples:", len(y))
print("Valid rice samples    :", int(np.sum(y == 1)))
print("Valid non-rice samples:", int(np.sum(y == 0)))


# ============================================================
# 11. SAVE TRAINING POINTS
# ============================================================

training_df = pd.DataFrame({
    "class_value": y,
    "class_label": ["rice" if v == 1 else "non_rice" for v in y],
    "row": rows_valid.astype(int),
    "col": cols_valid.astype(int),
    "x": [c[0] for c in coords_valid],
    "y_coord": [c[1] for c in coords_valid]
})

training_csv_path = os.path.join(
    OUT_DIR,
    "Gowa_Rice_RF_2026_Training_Samples.csv"
)

training_df.to_csv(training_csv_path, index=False)

training_gdf = gpd.GeoDataFrame(
    training_df,
    geometry=[
        Point(xy) for xy in zip(training_df["x"], training_df["y_coord"])
    ],
    crs=src.crs
)

training_geojson_path = os.path.join(
    OUT_DIR,
    "Gowa_Rice_RF_2026_Training_Samples.geojson"
)

training_gdf.to_file(training_geojson_path, driver="GeoJSON")

print("Training samples CSV    :", training_csv_path)
print("Training samples GeoJSON:", training_geojson_path)


# ============================================================
# 12. TRAIN-TEST SPLIT AND RANDOM FOREST
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=RANDOM_SEED,
    stratify=y
)

imputer = SimpleImputer(strategy="median")

X_train_imp = imputer.fit_transform(X_train)
X_test_imp = imputer.transform(X_test)

rf = RandomForestClassifier(
    n_estimators=500,
    max_features="sqrt",
    min_samples_leaf=2,
    class_weight="balanced_subsample",
    oob_score=True,
    n_jobs=-1,
    random_state=RANDOM_SEED
)

rf.fit(X_train_imp, y_train)

y_pred = rf.predict(X_test_imp)

acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

cm = confusion_matrix(y_test, y_pred)

report = classification_report(
    y_test,
    y_pred,
    target_names=["non_rice", "rice"],
    output_dict=True
)

print("\n=== SEED-DERIVED INTERNAL ACCURACY ===")
print("Accuracy :", acc)
print("F1 rice  :", f1)
print("OOB score:", rf.oob_score_)
print("\nConfusion matrix:")
print(cm)

report_df = pd.DataFrame(report).T

report_path = os.path.join(
    OUT_DIR,
    "Gowa_Rice_RF_2026_Internal_Accuracy_Report.csv"
)

report_df.to_csv(report_path)

cm_df = pd.DataFrame(
    cm,
    index=["Actual_non_rice", "Actual_rice"],
    columns=["Pred_non_rice", "Pred_rice"]
)

cm_path = os.path.join(
    OUT_DIR,
    "Gowa_Rice_RF_2026_Internal_Confusion_Matrix.csv"
)

cm_df.to_csv(cm_path)

print("Accuracy report:", report_path)
print("Confusion matrix:", cm_path)


# ============================================================
# 13. FEATURE IMPORTANCE
# ============================================================

importance_df = pd.DataFrame({
    "Band": band_names,
    "Importance": rf.feature_importances_
}).sort_values("Importance", ascending=False)

importance_path = os.path.join(
    OUT_DIR,
    "Gowa_Rice_RF_2026_Feature_Importance.csv"
)

importance_df.to_csv(importance_path, index=False)

print("\nTop 20 feature importance:")
display(importance_df.head(20))

print("Feature importance:", importance_path)


# ============================================================
# 14. TILE-BASED PREDICTION FOR ALL OF GOWA
# ============================================================

prob_path = os.path.join(
    OUT_DIR,
    "Gowa_Rice_RF_2026_Rice_Probability_10m.tif"
)

class_raw_path = os.path.join(
    OUT_DIR,
    "Gowa_Rice_RF_2026_Binary_Raw_10m.tif"
)

prob_profile = ref_profile.copy()
prob_profile.update({
    "driver": "GTiff",
    "count": 1,
    "dtype": "float32",
    "nodata": -9999,
    "compress": "lzw"
})

class_profile = ref_profile.copy()
class_profile.update({
    "driver": "GTiff",
    "count": 1,
    "dtype": "uint8",
    "nodata": 255,
    "compress": "lzw"
})

rice_class_index = list(rf.classes_).index(1)

with rasterio.open(prob_path, "w", **prob_profile) as prob_dst, \
     rasterio.open(class_raw_path, "w", **class_profile) as cls_dst:

    total_windows = 0

    for win in window_grid(src.width, src.height, TILE_SIZE):
        total_windows += 1

        data = src.read(window=win).astype("float32")

        if src.nodata is not None:
            data[data == src.nodata] = np.nan

        data[np.isinf(data)] = np.nan

        bands, h, w = data.shape

        valid_tile = np.any(np.isfinite(data), axis=0)

        X_tile = data.reshape(bands, h * w).T
        valid_flat = valid_tile.reshape(h * w)

        prob_tile = np.full(h * w, -9999, dtype="float32")
        class_tile = np.full(h * w, 255, dtype="uint8")

        if np.sum(valid_flat) > 0:
            X_valid = X_tile[valid_flat]
            X_valid_imp = imputer.transform(X_valid)

            prob_valid = rf.predict_proba(X_valid_imp)[:, rice_class_index]
            class_valid = (prob_valid >= RICE_PROB_THRESHOLD).astype("uint8")

            prob_tile[valid_flat] = prob_valid.astype("float32")
            class_tile[valid_flat] = class_valid

        prob_tile = prob_tile.reshape(h, w)
        class_tile = class_tile.reshape(h, w)

        prob_dst.write(prob_tile, 1, window=win)
        cls_dst.write(class_tile, 1, window=win)

        if total_windows % 20 == 0:
            print("Windows processed:", total_windows)

print("\nPrediction completed.")
print("Probability raster:", prob_path)
print("Raw binary raster :", class_raw_path)


# ============================================================
# 15. SMALL-PATCH POST-PROCESSING
# ============================================================

class_clean_path = os.path.join(
    OUT_DIR,
    "Gowa_Rice_RF_2026_Binary_Cleaned_10m.tif"
)

if REMOVE_SMALL_PATCHES:
    with rasterio.open(class_raw_path) as cls_src:
        cls_arr = cls_src.read(1)
        cls_profile = cls_src.profile.copy()

    rice_arr = cls_arr == 1

    pixel_area_ha = abs(src.transform.a * src.transform.e) / 10000.0
    min_patch_pixels = max(1, int(MIN_PATCH_AREA_HA / pixel_area_ha))

    labeled, ncomp = ndi_label(rice_arr)

    counts = np.bincount(labeled.ravel())

    keep = np.zeros_like(counts, dtype=bool)
    keep[counts >= min_patch_pixels] = True
    keep[0] = False

    cleaned_rice = keep[labeled]

    cleaned_arr = np.where(
        cls_arr == 255,
        255,
        cleaned_rice.astype("uint8")
    )

    with rasterio.open(class_clean_path, "w", **cls_profile) as dst:
        dst.write(cleaned_arr.astype("uint8"), 1)

    print("\nPost-processing completed.")
    print("Initial number of patches:", ncomp)
    print("Minimum patch pixels:", min_patch_pixels)
    print("Cleaned raster:", class_clean_path)

else:
    class_clean_path = class_raw_path
    print("Post-processing skipped.")


# ============================================================
# 16. CALCULATE 2026 RICE-FIELD AREA
# ============================================================

with rasterio.open(class_clean_path) as cls_src:
    cls_clean = cls_src.read(1)

pixel_area_ha = abs(src.transform.a * src.transform.e) / 10000.0

rice_pixel = int(np.sum(cls_clean == 1))
nonrice_pixel = int(np.sum(cls_clean == 0))
nodata_pixel = int(np.sum(cls_clean == 255))

rice_area_ha = rice_pixel * pixel_area_ha
nonrice_area_ha = nonrice_pixel * pixel_area_ha
nodata_area_ha = nodata_pixel * pixel_area_ha

area_summary = pd.DataFrame([
    {
        "Class": 1,
        "Class_Label": "Rice",
        "Pixel_Count": rice_pixel,
        "Area_ha": rice_area_ha,
        "Area_km2": rice_area_ha / 100.0
    },
    {
        "Class": 0,
        "Class_Label": "Non-rice",
        "Pixel_Count": nonrice_pixel,
        "Area_ha": nonrice_area_ha,
        "Area_km2": nonrice_area_ha / 100.0
    },
    {
        "Class": 255,
        "Class_Label": "NoData",
        "Pixel_Count": nodata_pixel,
        "Area_ha": nodata_area_ha,
        "Area_km2": nodata_area_ha / 100.0
    }
])

area_path = os.path.join(
    OUT_DIR,
    "Gowa_Rice_RF_2026_Area_Summary.csv"
)

area_summary.to_csv(area_path, index=False)

print("\n=== 2026 RF-DERIVED RICE-FIELD AREA ===")
display(area_summary)

print("Area summary:", area_path)


# ============================================================
# 17. READ GOWA BOUNDARY FOR VISUALIZATION
# ============================================================

boundary_path = find_first_file(
    SENTINEL_DIR,
    [
        "Gowa_Boundary_GAUL2015_2026.geojson",
        "*Gowa*Boundary*.geojson",
        "*Gowa*boundary*.geojson"
    ],
    required=False,
    label="Gowa boundary"
)

gowa_admin = None

if boundary_path is not None:
    gowa_admin = gpd.read_file(boundary_path)

    if gowa_admin.crs is None:
        gowa_admin = gowa_admin.set_crs("EPSG:4326")

    gowa_admin = gowa_admin.to_crs(src.crs)

print("Boundary used for visualization:", boundary_path)


def plot_boundary(ax):
    if gowa_admin is not None:
        gowa_admin.boundary.plot(
            ax=ax,
            edgecolor="black",
            linewidth=0.8
        )


# ============================================================
# 18. FIGURE 1 - TRAINING SEED MAP
# ============================================================

fig, ax = plt.subplots(figsize=(9, 8))

seed_plot = np.full(rice_seed_raw.shape, np.nan, dtype="float32")
seed_plot[nonrice_seed] = 0
seed_plot[rice_seed_raw] = 1

seed_cmap = ListedColormap(["#d95f02", "#1b9e77"])
seed_norm = BoundaryNorm([-0.5, 0.5, 1.5], seed_cmap.N)

im = ax.imshow(
    seed_plot,
    extent=plotting_extent(src),
    origin="upper",
    cmap=seed_cmap,
    norm=seed_norm
)

plot_boundary(ax)
format_map_axis(ax)

legend_elements = [
    Patch(facecolor="#1b9e77", edgecolor="black", label="Rice seed from 2024 mask"),
    Patch(facecolor="#d95f02", edgecolor="black", label="High-confidence non-rice seed")
]

ax.legend(
    handles=legend_elements,
    loc="lower left",
    fontsize=8,
    frameon=True
)

fig.tight_layout()

fig1_path = os.path.join(
    FIG_DIR,
    "01_Training_Seed_Rice_NonRice_DPI600.png"
)

plt.savefig(fig1_path, dpi=600, bbox_inches="tight")
plt.show()

print("Figure 1:", fig1_path)


# ============================================================
# 19. FIGURE 2 - PROBABILITY MAP
# ============================================================

prob_plot, prob_extent = read_downsample(prob_path, max_size=1800)
prob_plot = prob_plot.astype("float32")
prob_plot[prob_plot == -9999] = np.nan

fig, ax = plt.subplots(figsize=(9, 8))

im = ax.imshow(
    prob_plot,
    extent=prob_extent,
    origin="upper",
    cmap="viridis",
    vmin=0,
    vmax=1
)

plot_boundary(ax)
format_map_axis(ax)

cbar = plt.colorbar(
    im,
    ax=ax,
    orientation="horizontal",
    fraction=0.045,
    pad=0.06
)

cbar.set_label("Rice probability", fontsize=9)
cbar.ax.tick_params(labelsize=8)

fig.tight_layout()

fig2_path = os.path.join(
    FIG_DIR,
    "02_Rice_Probability_2026_DPI600.png"
)

plt.savefig(fig2_path, dpi=600, bbox_inches="tight")
plt.show()

print("Figure 2:", fig2_path)


# ============================================================
# 20. FIGURE 3 - BINARY RICE MAP
# ============================================================

class_plot, class_extent = read_downsample(class_clean_path, max_size=1800)

class_cmap = ListedColormap(["#f0f0f0", "#238b45", "#ffffff"])
class_norm = BoundaryNorm([-0.5, 0.5, 1.5, 255.5], class_cmap.N)

fig, ax = plt.subplots(figsize=(9, 8))

im = ax.imshow(
    class_plot,
    extent=class_extent,
    origin="upper",
    cmap=class_cmap,
    norm=class_norm
)

plot_boundary(ax)
format_map_axis(ax)

legend_elements = [
    Patch(facecolor="#238b45", edgecolor="black", label="Rice 2026"),
    Patch(facecolor="#f0f0f0", edgecolor="black", label="Non-rice")
]

ax.legend(
    handles=legend_elements,
    loc="lower left",
    fontsize=8,
    frameon=True
)

fig.tight_layout()

fig3_path = os.path.join(
    FIG_DIR,
    "03_Rice_Classification_2026_Binary_DPI600.png"
)

plt.savefig(fig3_path, dpi=600, bbox_inches="tight")
plt.show()

print("Figure 3:", fig3_path)


# ============================================================
# 21. FIGURE 4 - FEATURE IMPORTANCE TOP 20
# ============================================================

top_imp = importance_df.head(20).sort_values("Importance", ascending=True)

fig, ax = plt.subplots(figsize=(8, 7))

ax.barh(
    top_imp["Band"],
    top_imp["Importance"]
)

ax.set_xlabel("Random Forest feature importance", fontsize=10)
ax.tick_params(axis="both", labelsize=8)
ax.grid(axis="x", alpha=0.25)

fig.tight_layout()

fig4_path = os.path.join(
    FIG_DIR,
    "04_Feature_Importance_Top20_DPI600.png"
)

plt.savefig(fig4_path, dpi=600, bbox_inches="tight")
plt.show()

print("Figure 4:", fig4_path)


# ============================================================
# 22. FIGURE 5 - CONFUSION MATRIX INTERNAL
# ============================================================

fig, ax = plt.subplots(figsize=(5.5, 5))

im = ax.imshow(cm, cmap="Blues")

ax.set_xticks([0, 1])
ax.set_yticks([0, 1])

ax.set_xticklabels(["Non-rice", "Rice"])
ax.set_yticklabels(["Non-rice", "Rice"])

ax.set_xlabel("Predicted label", fontsize=10)
ax.set_ylabel("Reference label", fontsize=10)

for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(
            j,
            i,
            str(cm[i, j]),
            ha="center",
            va="center",
            color="black",
            fontsize=11
        )

cbar = plt.colorbar(
    im,
    ax=ax,
    fraction=0.046,
    pad=0.04
)

cbar.ax.tick_params(labelsize=8)

fig.tight_layout()

fig5_path = os.path.join(
    FIG_DIR,
    "05_Internal_Confusion_Matrix_DPI600.png"
)

plt.savefig(fig5_path, dpi=600, bbox_inches="tight")
plt.show()

print("Figure 5:", fig5_path)


# ============================================================
# 23. SAVE METADATA
# ============================================================

metadata = pd.DataFrame([
    {
        "Item": "RF_stack",
        "Value": RF_STACK_PATH
    },
    {
        "Item": "Rice_seed_source",
        "Value": seed_path
    },
    {
        "Item": "Rice_probability_threshold",
        "Value": RICE_PROB_THRESHOLD
    },
    {
        "Item": "N_rice_training_sample",
        "Value": int(np.sum(y == 1))
    },
    {
        "Item": "N_nonrice_training_sample",
        "Value": int(np.sum(y == 0))
    },
    {
        "Item": "Internal_accuracy",
        "Value": acc
    },
    {
        "Item": "Internal_F1_rice",
        "Value": f1
    },
    {
        "Item": "OOB_score",
        "Value": rf.oob_score_
    },
    {
        "Item": "Rice_area_ha_2026",
        "Value": rice_area_ha
    },
    {
        "Item": "Nonrice_area_ha_2026",
        "Value": nonrice_area_ha
    }
])

metadata_path = os.path.join(
    OUT_DIR,
    "Gowa_Rice_RF_2026_Analysis_Metadata.csv"
)

metadata.to_csv(metadata_path, index=False)

print("Metadata:", metadata_path)


# ============================================================
# 24. SAVE ALL TABLES TO EXCEL
# ============================================================

excel_path = os.path.join(
    OUT_DIR,
    "Gowa_Rice_RF_2026_Classification_Result.xlsx"
)

with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
    metadata.to_excel(writer, sheet_name="Metadata", index=False)
    area_summary.to_excel(writer, sheet_name="Area_Summary", index=False)
    importance_df.to_excel(writer, sheet_name="Feature_Importance", index=False)
    report_df.to_excel(writer, sheet_name="Accuracy_Report", index=True)
    cm_df.to_excel(writer, sheet_name="Confusion_Matrix", index=True)
    training_df.to_excel(writer, sheet_name="Training_Samples", index=False)

print("Excel result:", excel_path)


# ============================================================
# 25. FINAL SUMMARY
# ============================================================

print("============================================================")
print("GOWA 2026 RICE-FIELD CLASSIFICATION COMPLETED")
print("============================================================")
print("Rice-probability raster:")
print(prob_path)
print("")
print("Raster binary raw:")
print(class_raw_path)
print("")
print("Raster binary cleaned:")
print(class_clean_path)
print("")
print("2026 rice-field area:")
print(round(rice_area_ha, 2), "ha")
print("")
print("600-dpi figure folder:")
print(FIG_DIR)
print("")
print("Important note:")
print("The reported accuracy is an internal seed-derived diagnostic,")
print("not a full independent field/map validation. The manuscript therefore requires")
print("independent visual/field reference data or independent validation points.")
print("============================================================")
