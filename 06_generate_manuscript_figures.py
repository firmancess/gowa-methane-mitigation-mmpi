"""
Final Manuscript Figure Generation
==================================

Generates the final 600-dpi manuscript figures from the final MMPI and rice-mask outputs. Figure labels should follow the manuscript terminology (fixed-score MMPI classes).

This file was selected from the final successful workflow in the uploaded analysis notebook.
Superseded/failed notebook cells were intentionally excluded.
"""

# ============================================================
# FINAL VISUALIZATION SCRIPT
# MMPI GOWA RICE 2026
# DPI 600 | ORDERED FIGURES | FIXED FILE SEARCH
# ============================================================


import os
import glob
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
import matplotlib.pyplot as plt

from rasterio.plot import plotting_extent
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch, FancyBboxPatch, FancyArrowPatch
from mpl_toolkits.axes_grid1 import make_axes_locatable
try:
    from google.colab import drive
    drive.mount('/content/drive')
except Exception:
    pass


# ============================================================
# 1. MOUNT DRIVE
# ============================================================



# ============================================================
# 2. FOLDER SETTING
# ============================================================

DRIVE_ROOT = os.environ.get("MMPI_DRIVE_ROOT", "/content/drive/MyDrive")

BASE_FINAL = os.path.join(
    DRIVE_ROOT,
    "MANUSCRIPT_FINAL_MMPI_Rice2026_Gowa_SOIL_GAPFILLED_FINAL"
)

BASE_REVISED = os.path.join(
    DRIVE_ROOT,
    "MANUSCRIPT_FINAL_MMPI_Rice2026_Gowa_REVISED"
)

BASE_ORIGINAL = os.path.join(
    DRIVE_ROOT,
    "MANUSCRIPT_FINAL_MMPI_Rice2026_Gowa"
)

BASE_PUBLICATION_FINAL = os.path.join(
    DRIVE_ROOT,
    "MANUSCRIPT_FINAL_MMPI_Rice2026_Gowa_REVISED_PUBLICATION_FINAL"
)

BASE_PUBLICATION = os.path.join(
    DRIVE_ROOT,
    "MANUSCRIPT_FINAL_MMPI_Rice2026_Gowa_REVISED_PUBLICATION"
)

BASE_CONTINUATION = os.path.join(
    DRIVE_ROOT,
    "Gowa_Rice2026_MMPI_Continuation_Analysis"
)

BASE_OLD_2024 = os.path.join(
    DRIVE_ROOT,
    "MMPI_Mamminasata_Gowa_Sawah2024_Analysis_Output_30m"
)

BASE_OLD_30M = os.path.join(
    DRIVE_ROOT,
    "MMPI_Gowa_Analysis_Output_30m"
)

RASTER_DIR_FINAL = os.path.join(BASE_FINAL, "Raster_Output")
TABLE_DIR = os.path.join(BASE_FINAL, "Tables")

FIG_DIR = os.path.join(
    BASE_FINAL,
    "Figures_DPI600_FINAL_ORDERED_FIXED_V3"
)

os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(TABLE_DIR, exist_ok=True)

DPI = 600
EXPECTED_RICE_AREA_HA = 26208.20

print("Output figure folder:")
print(FIG_DIR)


# ============================================================
# 3. SEARCH ROOTS
# ============================================================

SEARCH_ROOTS_COMPONENTS = [
    os.path.join(BASE_FINAL, "Raster_Output"),
    BASE_FINAL,
    os.path.join(BASE_REVISED, "Raster_Output"),
    BASE_REVISED,
    os.path.join(BASE_ORIGINAL, "Raster_Output"),
    BASE_ORIGINAL,
    os.path.join(BASE_PUBLICATION_FINAL, "Raster_Output"),
    BASE_PUBLICATION_FINAL,
    os.path.join(BASE_PUBLICATION, "Raster_Output"),
    BASE_PUBLICATION,
    os.path.join(BASE_CONTINUATION, "Raster_Output"),
    BASE_CONTINUATION,
    os.path.join(BASE_OLD_2024, "Raster_Output"),
    BASE_OLD_2024,
    BASE_OLD_30M,
]

SEARCH_ROOTS_COMPONENTS = [
    p for p in SEARCH_ROOTS_COMPONENTS
    if os.path.exists(p)
]

print("\nComponent search folders:")
for p in SEARCH_ROOTS_COMPONENTS:
    print("-", p)


# ============================================================
# 4. HELPER FUNCTIONS
# ============================================================

def glob_many(search_roots, patterns):
    found = []

    for root in search_roots:
        if not os.path.exists(root):
            continue

        for pat in patterns:
            found.extend(
                glob.glob(
                    os.path.join(root, "**", pat),
                    recursive=True
                )
            )

    return sorted(list(set(found)))


def read_raster_float(path):
    with rasterio.open(path) as src:
        arr = src.read(1).astype("float32")
        profile = src.profile.copy()
        crs = src.crs
        transform = src.transform
        extent = plotting_extent(src)
        nodata = src.nodata

    if nodata is not None:
        arr[arr == nodata] = np.nan

    arr[~np.isfinite(arr)] = np.nan

    return arr, profile, transform, crs, extent


def is_same_grid(path, ref_shape, ref_crs, ref_transform):
    try:
        with rasterio.open(path) as src:
            if src.shape != ref_shape:
                return False

            if src.crs != ref_crs:
                return False

            if not src.transform.almost_equals(ref_transform, precision=6):
                return False

        return True

    except Exception:
        return False


def find_raster_grid(patterns, label, ref_shape=None, ref_crs=None, ref_transform=None):
    candidates = glob_many(SEARCH_ROOTS_COMPONENTS, patterns)

    if len(candidates) == 0:
        raise FileNotFoundError(
            f"{label} was not found.\n"
            f"Search patterns: {patterns}\n\n"
            "Search folders:\n" + "\n".join(SEARCH_ROOTS_COMPONENTS)
        )

    if ref_shape is not None:
        candidates_same_grid = [
            p for p in candidates
            if is_same_grid(p, ref_shape, ref_crs, ref_transform)
        ]

        if len(candidates_same_grid) == 0:
            info = []

            for p in candidates:
                try:
                    with rasterio.open(p) as src:
                        info.append({
                            "label": label,
                            "file": os.path.basename(p),
                            "path": p,
                            "shape": str(src.shape),
                            "crs": str(src.crs),
                            "transform": str(src.transform)
                        })
                except Exception:
                    pass

            pd.DataFrame(info).to_csv(
                os.path.join(TABLE_DIR, f"ERROR_{label.replace(' ', '_')}_candidate_grid_check.csv"),
                index=False
            )

            raise FileNotFoundError(
                f"{label} were found, but none matches the final MMPI grid.\n"
                f"Check the candidate-file diagnostics in the Tables folder."
            )

        selected = candidates_same_grid[0]

    else:
        selected = candidates[0]

    print(f"{label}: {selected}")
    return selected


def read_binary_mask(path):
    with rasterio.open(path) as src:
        arr = src.read(1).astype("float32")
        profile = src.profile.copy()
        crs = src.crs
        transform = src.transform
        extent = plotting_extent(src)
        nodata = src.nodata
        pixel_width = abs(src.transform.a)
        pixel_height = abs(src.transform.e)

    if nodata is not None:
        arr[arr == nodata] = np.nan

    arr[~np.isfinite(arr)] = np.nan

    # Safeguard for value 255 when it represents NoData but is not encoded as raster NoData
    vals = arr[np.isfinite(arr)]

    if vals.size == 0:
        raise ValueError(f"Raster is empty: {path}")

    unique_small = np.unique(vals)
    unique_small = unique_small[:20]

    if np.nanmax(vals) == 255 and np.nanmin(vals) >= 0:
        arr[arr == 255] = np.nan
        vals = arr[np.isfinite(arr)]

    if vals.size == 0:
        raise ValueError(f"Raster is empty after NoData cleaning: {path}")

    vmax = np.nanmax(vals)

    # Probability range 0-1
    if vmax <= 1.0:
        mask = np.where(arr >= 0.5, 1.0, np.nan)
    else:
        mask = np.where(arr > 0, 1.0, np.nan)

    pixel_area_ha = (pixel_width * pixel_height) / 10000
    area_ha = np.nansum(mask == 1) * pixel_area_ha

    return mask, profile, transform, crs, extent, area_ha


def score_mask_candidate(path):
    try:
        mask, profile, transform, crs, extent, area_ha = read_binary_mask(path)

        fname = os.path.basename(path).lower()
        folder = os.path.dirname(path).lower()

        score = -abs(area_ha - EXPECTED_RICE_AREA_HA)

        bonus_terms = [
            "gowa",
            "rice",
            "rf",
            "2026",
            "calibrated",
            "final",
            "fixed",
            "binary",
            "cleaned"
        ]

        for term in bonus_terms:
            if term in fname or term in folder:
                score += 1500

        penalty_terms = [
            "2024",
            "mamminasata",
            "probability",
            "prob",
            "stack",
            "input",
            "mmpi",
            "mhsi",
            "awd",
            "hydrology",
            "soil",
            "climate",
            "topography",
            "flood",
            "uncert",
            "absolute",
            "relative"
        ]

        for term in penalty_terms:
            if term in fname:
                score -= 4000

        if area_ha < 5000:
            score -= 25000

        if area_ha > 80000:
            score -= 25000

        return {
            "path": path,
            "file": os.path.basename(path),
            "area_ha": area_ha,
            "score": score
        }

    except Exception as e:
        return {
            "path": path,
            "file": os.path.basename(path),
            "area_ha": np.nan,
            "score": -999999999,
            "error": str(e)
        }


def find_best_rice_mask():
    explicit_candidates = [
        os.path.join(
            DRIVE_ROOT,
            "Gowa_Rice_RF_2026_Calibrated_Final_FIXED",
            "Raster_Output",
            "Gowa_Rice_RF_2026_Calibrated_Final_FIXED_10m.tif"
        ),
        os.path.join(
            DRIVE_ROOT,
            "Gowa_Rice_RF_Classification_2026",
            "Raster_Output",
            "Gowa_Rice_RF_2026_Calibrated_Final_FIXED_10m.tif"
        ),
        os.path.join(
            DRIVE_ROOT,
            "Gowa_Rice_RF_Classification_2026",
            "Gowa_Rice_RF_2026_Calibrated_Final_FIXED_10m.tif"
        ),
        os.path.join(
            DRIVE_ROOT,
            "Gowa_Rice_RF_Classification_2026",
            "Gowa_Rice_RF_2026_Binary_Cleaned_10m.tif"
        ),
        os.path.join(
            BASE_FINAL,
            "Raster_Output",
            "FINAL_Rice2026_Binary_Mask_30m.tif"
        ),
        os.path.join(
            BASE_REVISED,
            "Raster_Output",
            "FINAL_Rice2026_Binary_Mask_30m.tif"
        ),
        os.path.join(
            BASE_ORIGINAL,
            "Raster_Output",
            "FINAL_Rice2026_Binary_Mask_30m.tif"
        ),
    ]

    existing_explicit = [p for p in explicit_candidates if os.path.exists(p)]

    patterns = [
        "**/*Gowa*Rice*RF*2026*.tif",
        "**/*Gowa*Rice*2026*Final*.tif",
        "**/*Gowa*Rice*2026*Calibrated*.tif",
        "**/*Rice_RF*2026*.tif",
        "**/*Rice*RF*2026*.tif",
        "**/*Rice2026*Binary*Mask*.tif",
        "**/*Rice*2026*Binary*.tif",
        "**/*Rice*2026*Cleaned*.tif",
        "**/*Rice*2026*Final*.tif",
        "**/*FINAL_Rice2026_Binary_Mask_30m.tif",
    ]

    auto_candidates = []

    for pat in patterns:
        auto_candidates.extend(
            glob.glob(
                os.path.join(DRIVE_ROOT, pat),
                recursive=True
            )
        )

    candidates = sorted(list(set(existing_explicit + auto_candidates)))

    if len(candidates) == 0:
        return None, None

    infos = [score_mask_candidate(p) for p in candidates]
    df = pd.DataFrame(infos).sort_values("score", ascending=False)

    df.to_csv(
        os.path.join(TABLE_DIR, "Rice_mask_candidate_check_FINAL.csv"),
        index=False
    )

    print("\nTop rice-mask candidates:")
    print(df.head(15)[["file", "area_ha", "score", "path"]].to_string(index=False))

    best_path = df.iloc[0]["path"]
    best_area = df.iloc[0]["area_ha"]

    return best_path, best_area


def clean_map_axis(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.grid(False)

    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.set_aspect("equal", adjustable="box")


def plot_boundary(ax, gdf, linewidth=0.8):
    gdf.boundary.plot(
        ax=ax,
        color="black",
        linewidth=linewidth,
        zorder=10
    )


def get_admin_extent(gdf, buffer_m=1200):
    minx, miny, maxx, maxy = gdf.total_bounds
    return (
        minx - buffer_m,
        maxx + buffer_m,
        miny - buffer_m,
        maxy + buffer_m
    )


def apply_extent(ax, extent_tuple):
    xmin, xmax, ymin, ymax = extent_tuple
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)


def add_horizontal_colorbar(fig, ax, im, label, ticks=[0, 50, 100]):
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("bottom", size="4.2%", pad=0.12)

    cbar = fig.colorbar(
        im,
        cax=cax,
        orientation="horizontal",
        ticks=ticks
    )

    cbar.set_label(label, fontsize=7, labelpad=2)
    cbar.ax.tick_params(labelsize=6, length=2, pad=1)

    return cbar


def save_fig(fig, filename, pad=0.08):
    out_path = os.path.join(FIG_DIR, filename)

    fig.savefig(
        out_path,
        dpi=DPI,
        bbox_inches="tight",
        pad_inches=pad
    )

    plt.show()
    print("Saved:", out_path)

    return out_path


# ============================================================
# 5. BOUNDARY
# ============================================================

boundary_candidates = [
    os.path.join(DRIVE_ROOT, "Gowa_Sentinel_2026_RF_Input", "Gowa_Boundary_GAUL2015_2026.geojson"),
    os.path.join(BASE_FINAL, "Gowa_Boundary_GAUL2015_2026.geojson"),
    os.path.join(BASE_REVISED, "Gowa_Boundary_GAUL2015_2026.geojson"),
    os.path.join(BASE_ORIGINAL, "Gowa_Boundary_GAUL2015_2026.geojson"),
    os.path.join(DRIVE_ROOT, "MMPI_Gowa", "non_GEE", "Gowa_boundary_GADM41.geojson"),
]

boundary_candidates += glob.glob(
    os.path.join(DRIVE_ROOT, "**", "*Gowa*Boundary*.geojson"),
    recursive=True
)

boundary_candidates += glob.glob(
    os.path.join(DRIVE_ROOT, "**", "*Gowa*boundary*.geojson"),
    recursive=True
)

boundary_candidates = sorted(list(set([p for p in boundary_candidates if os.path.exists(p)])))

if len(boundary_candidates) == 0:
    raise FileNotFoundError("Gowa boundary was not found.")

BOUNDARY_PATH = boundary_candidates[0]
print("\nGowa boundary:", BOUNDARY_PATH)


# ============================================================
# 6. LOAD FINAL MMPI AND MHSI FIRST
# ============================================================

MMPI_PATH = find_raster_grid(
    [
        "Final MMPI 30m.tif",
        "FINAL_Gowa_Rice2026_MMPI_REVISED_30m.tif",
        "*MMPI*REVISED*30m*.tif"
    ],
    label="MMPI final"
)

MMPI, profile, transform, crs, raster_extent = read_raster_float(MMPI_PATH)

ref_shape = MMPI.shape
ref_crs = crs
ref_transform = transform

MHSI_PATH = find_raster_grid(
    [
        "Final MHSI 30m.tif",
        "FINAL_Gowa_Rice2026_MHSI_REVISED_30m.tif",
        "*MHSI*REVISED*30m*.tif"
    ],
    label="MHSI final",
    ref_shape=ref_shape,
    ref_crs=ref_crs,
    ref_transform=ref_transform
)

MHSI, _, _, _, _ = read_raster_float(MHSI_PATH)

valid_mask = np.isfinite(MMPI)

print("\nMMPI final area check:")
print("Valid MMPI pixels:", int(np.sum(valid_mask)))
print("Valid MMPI area ha:", round(np.sum(valid_mask) * 0.09, 2))


# ============================================================
# 7. LOAD COMPONENTS FROM SOURCE FOLDERS
# ============================================================

AWD_PATH = find_raster_grid(
    [
        "FINAL_Gowa_Rice2026_AWD_Proxy_30m.tif",
        "*AWD*Proxy*30m*.tif",
        "*AWD*30m*.tif"
    ],
    label="AWD proxy",
    ref_shape=ref_shape,
    ref_crs=ref_crs,
    ref_transform=ref_transform
)

HYDRO_PATH = find_raster_grid(
    [
        "FINAL_Gowa_Rice2026_Hydrology_Suitability_30m.tif",
        "*Hydrology*Suitability*30m*.tif"
    ],
    label="Hydrology suitability",
    ref_shape=ref_shape,
    ref_crs=ref_crs,
    ref_transform=ref_transform
)

SOIL_PATH = find_raster_grid(
    [
        "Final Soil suitability gap filled 30m.tif",
        "*Soil*suitability*gap*filled*30m*.tif",
        "FINAL_Gowa_Rice2026_Soil_Suitability_30m.tif",
        "*Soil*Suitability*30m*.tif"
    ],
    label="Soil suitability",
    ref_shape=ref_shape,
    ref_crs=ref_crs,
    ref_transform=ref_transform
)

CLIMATE_PATH = find_raster_grid(
    [
        "FINAL_Gowa_Rice2026_Climate_Suitability_REVISED_30m.tif",
        "*Climate*Suitability*REVISED*30m*.tif",
        "*Climate*Suitability*30m*.tif"
    ],
    label="Climate suitability",
    ref_shape=ref_shape,
    ref_crs=ref_crs,
    ref_transform=ref_transform
)

TOPO_PATH = find_raster_grid(
    [
        "FINAL_Gowa_Rice2026_Topography_Suitability_REVISED_30m.tif",
        "*Topography*Suitability*REVISED*30m*.tif",
        "*Topography*Suitability*30m*.tif"
    ],
    label="Topography suitability",
    ref_shape=ref_shape,
    ref_crs=ref_crs,
    ref_transform=ref_transform
)

FLOOD_PATH = find_raster_grid(
    [
        "FINAL_Gowa_Rice2026_Continuous_Flooding_Risk_30m.tif",
        "*Continuous*Flooding*Risk*30m*.tif",
        "*Flooding*Risk*30m*.tif"
    ],
    label="Continuous flooding risk",
    ref_shape=ref_shape,
    ref_crs=ref_crs,
    ref_transform=ref_transform
)

UNCERT_PATH = find_raster_grid(
    [
        "FINAL_Gowa_Rice2026_Uncertainty_Index_30m.tif",
        "*Uncertainty*Index*30m*.tif"
    ],
    label="Uncertainty index",
    ref_shape=ref_shape,
    ref_crs=ref_crs,
    ref_transform=ref_transform
)

AWD, _, _, _, _ = read_raster_float(AWD_PATH)
HYDRO, _, _, _, _ = read_raster_float(HYDRO_PATH)
SOIL, _, _, _, _ = read_raster_float(SOIL_PATH)
CLIMATE, _, _, _, _ = read_raster_float(CLIMATE_PATH)
TOPO, _, _, _, _ = read_raster_float(TOPO_PATH)
FLOOD, _, _, _, _ = read_raster_float(FLOOD_PATH)
UNCERT, _, _, _, _ = read_raster_float(UNCERT_PATH)


# ============================================================
# 8. FIND RICE MASK FOR FIGURE 1
# ============================================================

best_rice_path, best_rice_area = find_best_rice_mask()

if best_rice_path is None:
    print("\nThe RF rice mask was not found. Figure 1 uses the valid MMPI mask as a fallback.")
    rice_fig1_mask = np.where(valid_mask, 1.0, np.nan)
    rice_fig1_extent = raster_extent
    rice_fig1_crs = crs
    rice_fig1_label = "Valid rice-field analysis mask 2026"
    rice_area_shown = np.sum(valid_mask) * 0.09
else:
    diff_pct = abs(best_rice_area - EXPECTED_RICE_AREA_HA) / EXPECTED_RICE_AREA_HA * 100

    print("\nSelected rice mask:")
    print(best_rice_path)
    print("Candidate area (ha):", round(best_rice_area, 2))
    print("Difference from target (%):", round(diff_pct, 2))

    if diff_pct > 35:
        print("\nCandidate area is too far from the expected value. Figure 1 uses the valid MMPI mask as a fallback.")
        rice_fig1_mask = np.where(valid_mask, 1.0, np.nan)
        rice_fig1_extent = raster_extent
        rice_fig1_crs = crs
        rice_fig1_label = "Valid rice-field analysis mask 2026"
        rice_area_shown = np.sum(valid_mask) * 0.09
    else:
        rice_fig1_mask, _, _, rice_fig1_crs, rice_fig1_extent, rice_area_shown = read_binary_mask(best_rice_path)
        rice_fig1_label = "RF-derived rice-field mask 2026"

print("\nFigure 1 mask used:")
print(rice_fig1_label)
print("Area shown ha:", round(rice_area_shown, 2))


# ============================================================
# 9. BOUNDARY CRS
# ============================================================

gowa_admin_30m = gpd.read_file(BOUNDARY_PATH)
if gowa_admin_30m.crs is None:
    gowa_admin_30m = gowa_admin_30m.set_crs("EPSG:4326")
gowa_admin_30m = gowa_admin_30m.to_crs(crs)
admin_extent_30m = get_admin_extent(gowa_admin_30m, buffer_m=1200)

gowa_admin_fig1 = gpd.read_file(BOUNDARY_PATH)
if gowa_admin_fig1.crs is None:
    gowa_admin_fig1 = gowa_admin_fig1.set_crs("EPSG:4326")
gowa_admin_fig1 = gowa_admin_fig1.to_crs(rice_fig1_crs)
admin_extent_fig1 = get_admin_extent(gowa_admin_fig1, buffer_m=1200)


# ============================================================
# 10. CLASSIFICATION FROM FINAL MMPI
# ============================================================

FIXED_SCORE_CLASS = np.full(MMPI.shape, np.nan, dtype="float32")
FIXED_SCORE_CLASS[(MMPI >= 0) & (MMPI < 20) & valid_mask] = 1
FIXED_SCORE_CLASS[(MMPI >= 20) & (MMPI < 40) & valid_mask] = 2
FIXED_SCORE_CLASS[(MMPI >= 40) & (MMPI < 60) & valid_mask] = 3
FIXED_SCORE_CLASS[(MMPI >= 60) & (MMPI < 80) & valid_mask] = 4
FIXED_SCORE_CLASS[(MMPI >= 80) & (MMPI <= 100) & valid_mask] = 5

q20, q40, q60, q80 = np.nanpercentile(MMPI[valid_mask], [20, 40, 60, 80])

REL_CLASS = np.full(MMPI.shape, np.nan, dtype="float32")
REL_CLASS[(MMPI <= q20) & valid_mask] = 1
REL_CLASS[(MMPI > q20) & (MMPI <= q40) & valid_mask] = 2
REL_CLASS[(MMPI > q40) & (MMPI <= q60) & valid_mask] = 3
REL_CLASS[(MMPI > q60) & (MMPI <= q80) & valid_mask] = 4
REL_CLASS[(MMPI > q80) & valid_mask] = 5

print("\nRelative priority thresholds:")
print("Q20:", round(q20, 2))
print("Q40:", round(q40, 2))
print("Q60:", round(q60, 2))
print("Q80:", round(q80, 2))


# ============================================================
# 11. COLORS
# ============================================================

score_cmap = "viridis"
risk_cmap = "magma"

class_colors = [
    "#2c7fb8",
    "#41ab5d",
    "#fec44f",
    "#fc8d59",
    "#d7301f"
]

class_labels = [
    "Very low",
    "Low",
    "Moderate",
    "High",
    "Very high"
]

class_cmap = ListedColormap(class_colors)
class_norm = BoundaryNorm(
    [0.5, 1.5, 2.5, 3.5, 4.5, 5.5],
    class_cmap.N
)


# ============================================================
# FIGURE 1
# STUDY AREA AND 2026 RICE-FIELD MASK
# ============================================================

fig, ax = plt.subplots(figsize=(8.0, 8.8))

ax.imshow(
    rice_fig1_mask,
    extent=rice_fig1_extent,
    cmap=ListedColormap(["#238b45"]),
    interpolation="nearest",
    zorder=2
)

plot_boundary(ax, gowa_admin_fig1, linewidth=1.05)
apply_extent(ax, admin_extent_fig1)
clean_map_axis(ax)

ax.set_title(
    "Study area and 2026 rice-field mask",
    fontsize=12,
    fontweight="bold",
    pad=18
)

legend_handles = [
    Patch(facecolor="#238b45", edgecolor="black", label=rice_fig1_label),
    Patch(facecolor="white", edgecolor="black", label="Gowa boundary")
]

legend = ax.legend(
    handles=legend_handles,
    loc="lower center",
    bbox_to_anchor=(0.5, -0.085),
    ncol=2,
    frameon=True,
    fontsize=8,
    handlelength=2.0,
    columnspacing=1.6,
    borderpad=0.8
)

legend.get_frame().set_edgecolor("black")
legend.get_frame().set_linewidth(0.9)
legend.get_frame().set_facecolor("white")
legend.get_frame().set_alpha(1.0)

fig.subplots_adjust(left=0.035, right=0.965, top=0.925, bottom=0.145)

save_fig(
    fig,
    "01_Study_Area_and_2026_Rice_Field_Mask_DPI600.png",
    pad=0.06
)


# ============================================================
# FIGURE 2
# OVERALL METHODOLOGICAL FRAMEWORK
# ============================================================

fig, ax = plt.subplots(figsize=(11.8, 5.9))
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis("off")

ax.set_title(
    "Overall methodological framework",
    fontsize=13,
    fontweight="bold",
    pad=18
)

boxes = [
    ((0.045, 0.60), "Sentinel-1 and\nSentinel-2 imagery\n(2026)"),
    ((0.255, 0.60), "Random Forest\nrice-field mapping"),
    ((0.465, 0.60), "Calibrated 2026\nrice-field mask"),
    ((0.675, 0.60), "Aggregation to\n30 m grid"),
    ((0.255, 0.22), "Hydrology, soil,\nclimate, topography"),
    ((0.465, 0.22), "Environmental\nsuitability indicators"),
    ((0.675, 0.22), "MHSI, AWD proxy,\nand final MMPI"),
]

box_w = 0.155
box_h = 0.15

for (x, y), text in boxes:
    box = FancyBboxPatch(
        (x, y),
        box_w,
        box_h,
        boxstyle="round,pad=0.018,rounding_size=0.025",
        linewidth=1.15,
        edgecolor="black",
        facecolor="white"
    )
    ax.add_patch(box)

    ax.text(
        x + box_w / 2,
        y + box_h / 2,
        text,
        ha="center",
        va="center",
        fontsize=9,
        linespacing=1.25
    )

arrow_pairs = [
    ((0.200, 0.675), (0.255, 0.675)),
    ((0.410, 0.675), (0.465, 0.675)),
    ((0.620, 0.675), (0.675, 0.675)),
    ((0.752, 0.600), (0.752, 0.390)),
    ((0.255 + box_w, 0.295), (0.465, 0.295)),
    ((0.465 + box_w, 0.295), (0.675, 0.295)),
]

for start, end in arrow_pairs:
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=14,
        linewidth=1.1,
        color="black"
    )
    ax.add_patch(arrow)

fig.subplots_adjust(left=0.03, right=0.97, top=0.86, bottom=0.08)

save_fig(
    fig,
    "02_Overall_Methodological_Framework_DPI600.png",
    pad=0.08
)


# ============================================================
# FIGURE 3
# SPATIAL DISTRIBUTION OF MMPI AND COMPONENT INDICES
# ============================================================

layers = [
    ("(a)", "MMPI", MMPI, "Score", score_cmap),
    ("(b)", "MHSI", MHSI, "Score", score_cmap),
    ("(c)", "AWD proxy", AWD, "Score", score_cmap),
    ("(d)", "Hydrology suitability", HYDRO, "Score", score_cmap),
    ("(e)", "Soil suitability", SOIL, "Score", score_cmap),
    ("(f)", "Climate suitability", CLIMATE, "Score", score_cmap),
    ("(g)", "Topography suitability", TOPO, "Score", score_cmap),
    ("(h)", "Continuous flooding risk", FLOOD, "Risk", risk_cmap),
    ("(i)", "Uncertainty index", UNCERT, "Score", risk_cmap),
]

fig, axes = plt.subplots(3, 3, figsize=(12.8, 14.8))

for ax, (letter, title, arr, cbar_label, cmap_name) in zip(axes.flat, layers):

    arr_plot = np.where(valid_mask, arr, np.nan)

    im = ax.imshow(
        arr_plot,
        extent=raster_extent,
        cmap=cmap_name,
        vmin=0,
        vmax=100,
        interpolation="nearest",
        zorder=2
    )

    plot_boundary(ax, gowa_admin_30m, linewidth=0.65)
    apply_extent(ax, admin_extent_30m)
    clean_map_axis(ax)

    ax.set_title(
        f"{letter} {title}",
        fontsize=9.5,
        fontweight="bold",
        pad=10
    )

    add_horizontal_colorbar(fig, ax, im, cbar_label, ticks=[0, 50, 100])

fig.subplots_adjust(
    left=0.035,
    right=0.985,
    top=0.965,
    bottom=0.045,
    wspace=0.25,
    hspace=0.32
)

save_fig(
    fig,
    "03_Spatial_Distribution_MMPI_and_Component_Indices_DPI600.png",
    pad=0.05
)


# ============================================================
# FIGURE 4
# FIXED-SCORE MMPI CLASS AND RELATIVE PRIORITY CLASS
# ============================================================

fig, axes = plt.subplots(1, 2, figsize=(12.4, 7.4))

class_maps = [
    ("(a) Fixed-score MMPI class", FIXED_SCORE_CLASS),
    ("(b) Relative priority class", REL_CLASS),
]

rice_background_30m = np.where(valid_mask, 1.0, np.nan)

for ax, (title, arr) in zip(axes, class_maps):

    ax.imshow(
        rice_background_30m,
        extent=raster_extent,
        cmap=ListedColormap(["#eeeeee"]),
        interpolation="nearest",
        zorder=1
    )

    arr_plot = np.where(valid_mask, arr, np.nan)

    ax.imshow(
        arr_plot,
        extent=raster_extent,
        cmap=class_cmap,
        norm=class_norm,
        interpolation="nearest",
        zorder=2
    )

    plot_boundary(ax, gowa_admin_30m, linewidth=0.75)
    apply_extent(ax, admin_extent_30m)
    clean_map_axis(ax)

    ax.set_title(title, fontsize=11, fontweight="bold", pad=12)

legend_handles = [
    Patch(facecolor=class_colors[i], edgecolor="black", label=class_labels[i])
    for i in range(5)
]

legend = fig.legend(
    handles=legend_handles,
    loc="lower center",
    bbox_to_anchor=(0.5, 0.045),
    ncol=5,
    frameon=True,
    fontsize=8,
    handlelength=1.6,
    columnspacing=1.3,
    borderpad=0.8
)

legend.get_frame().set_edgecolor("black")
legend.get_frame().set_linewidth(1.0)
legend.get_frame().set_facecolor("white")
legend.get_frame().set_alpha(1.0)

fig.subplots_adjust(
    left=0.035,
    right=0.985,
    top=0.90,
    bottom=0.17,
    wspace=0.09
)

save_fig(
    fig,
    "04_Fixed_Score_MMPI_Class_and_Relative_Priority_Class_DPI600.png",
    pad=0.06
)


# ============================================================
# FIGURE 5
# AREA DISTRIBUTION AND MEAN MMPI BY RELATIVE CLASS
# ============================================================

abs_rows = []

for k in [1, 2, 3, 4, 5]:
    m = (FIXED_SCORE_CLASS == k) & valid_mask
    n_pix = int(np.sum(m))
    area_ha = n_pix * 0.09

    abs_rows.append({
        "Class": k,
        "Class label": class_labels[k - 1],
        "Pixels": n_pix,
        "Area ha": area_ha,
        "Percent": area_ha / (np.sum(valid_mask) * 0.09) * 100
    })

abs_df = pd.DataFrame(abs_rows)

rel_rows = []

for k in [1, 2, 3, 4, 5]:
    m = (REL_CLASS == k) & valid_mask & np.isfinite(MMPI)

    rel_rows.append({
        "Class": k,
        "Class label": class_labels[k - 1],
        "Pixels": int(np.sum(m)),
        "Area ha": float(np.sum(m) * 0.09),
        "Mean MMPI": float(np.nanmean(MMPI[m])),
        "Median MMPI": float(np.nanmedian(MMPI[m]))
    })

rel_df = pd.DataFrame(rel_rows)

abs_df.to_csv(
    os.path.join(TABLE_DIR, "Area_by_fixed_score_MMPI_class_final.csv"),
    index=False
)

rel_df.to_csv(
    os.path.join(TABLE_DIR, "Mean_MMPI_by_relative_priority_class_final.csv"),
    index=False
)

fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.6))

ax = axes[0]

bars = ax.bar(
    abs_df["Class label"],
    abs_df["Area ha"],
    color=class_colors,
    edgecolor="black",
    linewidth=0.8
)

ax.set_title(
    "(a) Area by fixed-score MMPI class",
    fontsize=11,
    fontweight="bold",
    pad=13
)

ax.set_ylabel("Area (ha)", fontsize=9)
ax.tick_params(axis="x", labelsize=8, rotation=25)
ax.tick_params(axis="y", labelsize=8)
ax.grid(axis="y", linestyle="--", alpha=0.30)

ymax = abs_df["Area ha"].max() * 1.24
ax.set_ylim(0, ymax)

for bar, pct in zip(bars, abs_df["Percent"]):
    h = bar.get_height()

    ax.text(
        bar.get_x() + bar.get_width() / 2,
        h + ymax * 0.025,
        f"{h:,.0f} ha\n{pct:.1f}%",
        ha="center",
        va="bottom",
        fontsize=7
    )

ax = axes[1]

bars = ax.bar(
    rel_df["Class label"],
    rel_df["Mean MMPI"],
    color=class_colors,
    edgecolor="black",
    linewidth=0.8
)

ax.plot(
    rel_df["Class label"],
    rel_df["Mean MMPI"],
    color="black",
    marker="o",
    linewidth=1.1,
    markersize=3.8
)

ax.set_title(
    "(b) Mean MMPI by relative priority class",
    fontsize=11,
    fontweight="bold",
    pad=13
)

ax.set_ylabel("Mean MMPI score", fontsize=9)
ax.tick_params(axis="x", labelsize=8, rotation=25)
ax.tick_params(axis="y", labelsize=8)
ax.grid(axis="y", linestyle="--", alpha=0.30)

ymax = rel_df["Mean MMPI"].max() * 1.24
ax.set_ylim(0, ymax)

for bar in bars:
    h = bar.get_height()

    ax.text(
        bar.get_x() + bar.get_width() / 2,
        h + ymax * 0.025,
        f"{h:.2f}",
        ha="center",
        va="bottom",
        fontsize=7
    )

fig.subplots_adjust(
    left=0.07,
    right=0.98,
    top=0.88,
    bottom=0.25,
    wspace=0.30
)

save_fig(
    fig,
    "05_Area_Distribution_and_Mean_MMPI_by_Class_DPI600.png",
    pad=0.08
)


# ============================================================
# FIGURE 6
# CORRELATION MATRIX HEATMAP
# ============================================================

corr_layers = {
    "MMPI": MMPI,
    "MHSI": MHSI,
    "AWD proxy": AWD,
    "Hydrology": HYDRO,
    "Soil": SOIL,
    "Climate": CLIMATE,
    "Topography": TOPO,
    "Flooding risk": FLOOD,
    "Uncertainty": UNCERT,
}

names = list(corr_layers.keys())

stack = np.vstack([
    corr_layers[name][valid_mask].reshape(-1)
    for name in names
]).T

valid_rows = np.all(np.isfinite(stack), axis=1)
stack_valid = stack[valid_rows]

corr = np.corrcoef(stack_valid, rowvar=False)
corr_df = pd.DataFrame(corr, index=names, columns=names)

corr_csv = os.path.join(TABLE_DIR, "Correlation_matrix_for_Figure_06.csv")
corr_df.to_csv(corr_csv)

fig, ax = plt.subplots(figsize=(9.7, 8.5))

im = ax.imshow(
    corr,
    cmap="coolwarm",
    vmin=-1,
    vmax=1
)

ax.set_xticks(np.arange(len(names)))
ax.set_yticks(np.arange(len(names)))

ax.set_xticklabels(
    names,
    rotation=45,
    ha="right",
    fontsize=8
)

ax.set_yticklabels(
    names,
    fontsize=8
)

ax.set_title(
    "Correlation matrix of MMPI components",
    fontsize=12,
    fontweight="bold",
    pad=18
)

for i in range(len(names)):
    for j in range(len(names)):
        ax.text(
            j,
            i,
            f"{corr[i, j]:.2f}",
            ha="center",
            va="center",
            fontsize=7,
            color="black"
        )

divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="4%", pad=0.22)

cbar = fig.colorbar(im, cax=cax)
cbar.set_label("Pearson correlation coefficient", fontsize=8, labelpad=9)
cbar.ax.tick_params(labelsize=7)

fig.subplots_adjust(
    left=0.20,
    right=0.88,
    top=0.88,
    bottom=0.24
)

save_fig(
    fig,
    "06_Correlation_Matrix_MMPI_Components_DPI600.png",
    pad=0.08
)


# ============================================================
# 12. FIGURE INDEX
# ============================================================

figure_index = pd.DataFrame([
    {
        "Figure": "Figure 1",
        "Caption": "Study area and 2026 rice-field mask in Gowa Regency, South Sulawesi, Indonesia.",
        "File": "01_Study_Area_and_2026_Rice_Field_Mask_DPI600.png"
    },
    {
        "Figure": "Figure 2",
        "Caption": "Overall methodological framework for mapping methane mitigation potential in rice fields.",
        "File": "02_Overall_Methodological_Framework_DPI600.png"
    },
    {
        "Figure": "Figure 3",
        "Caption": "Spatial distribution of MMPI and component indices.",
        "File": "03_Spatial_Distribution_MMPI_and_Component_Indices_DPI600.png"
    },
    {
        "Figure": "Figure 4",
        "Caption": "Fixed-score MMPI class and relative priority class.",
        "File": "04_Fixed_Score_MMPI_Class_and_Relative_Priority_Class_DPI600.png"
    },
    {
        "Figure": "Figure 5",
        "Caption": "Area distribution by fixed-score MMPI class and mean MMPI score by relative priority class.",
        "File": "05_Area_Distribution_and_Mean_MMPI_by_Class_DPI600.png"
    },
    {
        "Figure": "Figure 6",
        "Caption": "Correlation matrix of MMPI components.",
        "File": "06_Correlation_Matrix_MMPI_Components_DPI600.png"
    }
])

index_path = os.path.join(
    FIG_DIR,
    "MANUSCRIPT_FIGURE_INDEX_FINAL_ORDERED_FIXED_V3.csv"
)

figure_index.to_csv(index_path, index=False)

print("\n============================================================")
print("FINAL VISUALIZATION SCRIPT COMPLETED")
print("All figures were saved in:")
print(FIG_DIR)
print("\nFigure index:")
print(index_path)
print("============================================================")
