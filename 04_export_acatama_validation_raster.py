"""
Export Final RF Map for AcATaMa Validation
==========================================

Converts the final RF raster to a two-class AcATaMa-compatible raster: 1 = Non-rice, 2 = Rice, 0 = NoData.

This file was selected from the final successful workflow in the uploaded analysis notebook.
Superseded/failed notebook cells were intentionally excluded.
"""

# ============================================================
# EXPORT RF FINAL RICE CLASSIFICATION FOR ACATAMA QGIS
# TWO VALIDATION CLASSES ONLY:
# 1 = Non-rice
# 2 = Rice
#
# NoData = 0 and is excluded from the AcATaMa class legend
# ============================================================

try:
    from google.colab import drive
    drive.mount('/content/drive')
except Exception:
    pass

import os
import numpy as np
import rasterio

# ============================================================
# 1. INPUT FINAL RF CLASSIFICATION
# ============================================================

INPUT_RASTER = (
    "/content/drive/MyDrive/"
    "Gowa_Rice_RF_2026_Calibrated_Final_FIXED/"
    "Raster_Output/"
    "Gowa_Rice_RF_2026_Calibrated_Final_FIXED_10m.tif"
)

OUTPUT_DIR = (
    "/content/drive/MyDrive/"
    "Gowa_Rice_RF_2026_Calibrated_Final_FIXED/"
    "AcATaMa_Validation"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

OUTPUT_TIF = os.path.join(
    OUTPUT_DIR,
    "Gowa_RF_2026_Rice_NonRice_AcATaMa_2Classes_10m.tif"
)

OUTPUT_CSV = os.path.join(
    OUTPUT_DIR,
    "Gowa_RF_2026_Rice_NonRice_AcATaMa_2Classes_Legend.csv"
)

# ============================================================
# 2. CLASS SETTING
# ============================================================

# Final RF input raster:
# 0   = Non-rice
# 1   = Rice
# 255 = NoData / luar area valid

INPUT_NON_RICE = 0
INPUT_RICE = 1
INPUT_NODATA = 255

# Output raster for AcATaMa:
# 1 = Non-rice
# 2 = Rice
# 0 = NoData and is excluded as a validation class

CLASS_NON_RICE = 1
CLASS_RICE = 2
NODATA_VALUE = 0

# ============================================================
# 3. READ INPUT RASTER
# ============================================================

if not os.path.exists(INPUT_RASTER):
    raise FileNotFoundError(f"Input raster was not found:\n{INPUT_RASTER}")

with rasterio.open(INPUT_RASTER) as src:
    data = src.read(1)
    profile = src.profile.copy()
    transform = src.transform
    crs = src.crs
    nodata_in = src.nodata

print("Input raster found:")
print(INPUT_RASTER)
print("CRS:", crs)
print("Shape:", data.shape)
print("Input NoData:", nodata_in)
print("Unique input values:", np.unique(data))

# ============================================================
# 4. CONVERT CLASS TO TWO-CLASS ACATAMA FORMAT
# ============================================================

# Initialize the output as NoData
out = np.full(data.shape, NODATA_VALUE, dtype=np.uint8)

# Convert valid classes
out[data == INPUT_NON_RICE] = CLASS_NON_RICE
out[data == INPUT_RICE] = CLASS_RICE

# Preserve input NoData as output NoData
out[data == INPUT_NODATA] = NODATA_VALUE

# Convert any values other than 0, 1, and 255 to NoData
valid_input_values = np.isin(data, [INPUT_NON_RICE, INPUT_RICE, INPUT_NODATA])
out[~valid_input_values] = NODATA_VALUE

print("Unique output values including NoData:", np.unique(out))
print("Unique valid AcATaMa classes:", np.unique(out[out != NODATA_VALUE]))

# ============================================================
# 5. UPDATE RASTER PROFILE
# ============================================================

profile.update(
    dtype=rasterio.uint8,
    count=1,
    nodata=NODATA_VALUE,
    compress="lzw",
    tiled=True,
    blockxsize=256,
    blockysize=256
)

# ============================================================
# 6. SAVE OUTPUT RASTER
# ============================================================

with rasterio.open(OUTPUT_TIF, "w", **profile) as dst:
    dst.write(out, 1)

    # Colormap for the two valid classes only
    dst.write_colormap(
        1,
        {
            CLASS_NON_RICE: (220, 220, 220, 255),  # gray = non-rice
            CLASS_RICE: (0, 150, 70, 255),         # green = rice
            NODATA_VALUE: (255, 255, 255, 0)       # transparan = NoData
        }
    )

print("\nTwo-class AcATaMa raster saved:")
print(OUTPUT_TIF)

# ============================================================
# 7. SAVE CLASS LEGEND - TWO CLASSES ONLY
# ============================================================

legend_text = """class_id,class_name,description
1,Non-rice,Area classified as non-rice by the final 2026 Random Forest map
2,Rice,Area classified as rice by the final 2026 Random Forest map
"""

with open(OUTPUT_CSV, "w", encoding="utf-8") as f:
    f.write(legend_text)

print("\nTwo-class legend saved:")
print(OUTPUT_CSV)

# ============================================================
# 8. SUMMARY AREA
# ============================================================

pixel_area_m2 = abs(transform.a * transform.e)
pixel_area_ha = pixel_area_m2 / 10000

non_rice_pixels = int(np.sum(out == CLASS_NON_RICE))
rice_pixels = int(np.sum(out == CLASS_RICE))
nodata_pixels = int(np.sum(out == NODATA_VALUE))

non_rice_area_ha = non_rice_pixels * pixel_area_ha
rice_area_ha = rice_pixels * pixel_area_ha

print("\n===== SUMMARY OUTPUT =====")
print(f"Pixel area        : {pixel_area_m2:.2f} m²")
print(f"Pixel area        : {pixel_area_ha:.4f} ha")
print(f"Non-rice pixels  : {non_rice_pixels:,}")
print(f"Non-rice area    : {non_rice_area_ha:,.2f} ha")
print(f"Rice pixels      : {rice_pixels:,}")
print(f"Rice area        : {rice_area_ha:,.2f} ha")
print(f"NoData pixels     : {nodata_pixels:,}")

print("\nThe file is ready for RF map validation in QGIS AcATaMa.")
print("Valid AcATaMa classes:")
print("1 = Non-rice")
print("2 = Rice")
