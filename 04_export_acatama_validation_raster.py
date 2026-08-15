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
# 1 = Non-sawah
# 2 = Sawah
#
# NoData = 0, tetapi tidak dimasukkan ke legend AcATaMa
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

# Input raster RF final:
# 0   = Non-sawah
# 1   = Sawah
# 255 = NoData / luar area valid

INPUT_NON_RICE = 0
INPUT_RICE = 1
INPUT_NODATA = 255

# Output raster untuk AcATaMa:
# 1 = Non-sawah
# 2 = Sawah
# 0 = NoData, tidak dimasukkan sebagai kelas validasi

CLASS_NON_RICE = 1
CLASS_RICE = 2
NODATA_VALUE = 0

# ============================================================
# 3. READ INPUT RASTER
# ============================================================

if not os.path.exists(INPUT_RASTER):
    raise FileNotFoundError(f"Input raster tidak ditemukan:\n{INPUT_RASTER}")

with rasterio.open(INPUT_RASTER) as src:
    data = src.read(1)
    profile = src.profile.copy()
    transform = src.transform
    crs = src.crs
    nodata_in = src.nodata

print("Input raster ditemukan:")
print(INPUT_RASTER)
print("CRS:", crs)
print("Shape:", data.shape)
print("Input NoData:", nodata_in)
print("Unique values input:", np.unique(data))

# ============================================================
# 4. CONVERT CLASS TO TWO-CLASS ACATAMA FORMAT
# ============================================================

# Buat output awal sebagai NoData
out = np.full(data.shape, NODATA_VALUE, dtype=np.uint8)

# Konversi kelas valid
out[data == INPUT_NON_RICE] = CLASS_NON_RICE
out[data == INPUT_RICE] = CLASS_RICE

# NoData input tetap NoData output
out[data == INPUT_NODATA] = NODATA_VALUE

# Jika ada nilai lain selain 0, 1, 255, jadikan NoData
valid_input_values = np.isin(data, [INPUT_NON_RICE, INPUT_RICE, INPUT_NODATA])
out[~valid_input_values] = NODATA_VALUE

print("Unique values output termasuk NoData:", np.unique(out))
print("Unique kelas valid untuk AcATaMa:", np.unique(out[out != NODATA_VALUE]))

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

    # Colormap hanya untuk dua kelas valid
    dst.write_colormap(
        1,
        {
            CLASS_NON_RICE: (220, 220, 220, 255),  # abu-abu = non-sawah
            CLASS_RICE: (0, 150, 70, 255),         # hijau = sawah
            NODATA_VALUE: (255, 255, 255, 0)       # transparan = NoData
        }
    )

print("\nRaster dua kelas untuk AcATaMa berhasil disimpan:")
print(OUTPUT_TIF)

# ============================================================
# 7. SAVE CLASS LEGEND - HANYA DUA KELAS
# ============================================================

legend_text = """class_id,class_name,description
1,Non-sawah,Area bukan sawah berdasarkan hasil klasifikasi Random Forest final 2026
2,Sawah,Area sawah berdasarkan hasil klasifikasi Random Forest final 2026
"""

with open(OUTPUT_CSV, "w", encoding="utf-8") as f:
    f.write(legend_text)

print("\nLegenda dua kelas berhasil disimpan:")
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
print(f"Non-sawah pixels  : {non_rice_pixels:,}")
print(f"Non-sawah area    : {non_rice_area_ha:,.2f} ha")
print(f"Sawah pixels      : {rice_pixels:,}")
print(f"Sawah area        : {rice_area_ha:,.2f} ha")
print(f"NoData pixels     : {nodata_pixels:,}")

print("\nFile siap digunakan untuk validasi klasifikasi RF di QGIS AcATaMa.")
print("Kelas valid AcATaMa hanya:")
print("1 = Non-sawah")
print("2 = Sawah")
