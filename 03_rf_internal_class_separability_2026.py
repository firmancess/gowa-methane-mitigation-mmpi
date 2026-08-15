"""
Internal Random Forest Class-Separability Assessment
====================================================

Runs the separate expanded internal assessment using up to 30,000 seed-derived samples per class. These metrics are internal diagnostics and are not the independent map accuracy.

This file was selected from the final successful workflow in the uploaded analysis notebook.
Superseded/failed notebook cells were intentionally excluded.
"""

# ============================================================
# RF MODEL INTERNAL ACCURACY ASSESSMENT
# GOWA RICE-FIELD CLASSIFICATION 2026
#
# Computes:
# - OOB score
# - Train-test confusion matrix
# - Overall accuracy
# - Balanced accuracy
# - Precision / User's accuracy
# - Recall / Producer's accuracy
# - F1-score
# - Kappa
# - ROC-AUC
# - Feature importance
#
# Note:
# This is an internal RF diagnostic, not the final AcATaMa map-accuracy assessment.
# ============================================================

try:
    from google.colab import drive
    drive.mount('/content/drive')
except Exception:
    pass


import os
import glob
import subprocess
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
import joblib

from rasterio.warp import reproject, Resampling
from scipy.ndimage import binary_dilation
from shapely.geometry import Point

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    balanced_accuracy_score,
    precision_recall_fscore_support,
    cohen_kappa_score,
    roc_auc_score,
    classification_report
)

from IPython.display import display

# ============================================================
# 1. FOLDERS AND INPUTS
# ============================================================

DRIVE_ROOT = os.environ.get("MMPI_DRIVE_ROOT", "/content/drive/MyDrive")

SENTINEL_DIR = os.path.join(
    DRIVE_ROOT,
    "Gowa_Sentinel_2026_RF_Input"
)

RICE_2024_MASK_PATH = os.path.join(
    DRIVE_ROOT,
    "MMPI_Gowa_Analysis_Output_30m",
    "Sawah_2024_Mamminasata_Gowa_FINAL",
    "05_Sawah_2024_Gowa_binary_mask_aligned_to_MMPI_30m.tif"
)

OUT_DIR = os.path.join(
    DRIVE_ROOT,
    "Gowa_Rice_RF_2026_Model_Accuracy"
)

TABLE_DIR = os.path.join(OUT_DIR, "Tables")
MODEL_DIR = os.path.join(OUT_DIR, "Model")
POINT_DIR = os.path.join(OUT_DIR, "Training_Testing_Points")

for d in [OUT_DIR, TABLE_DIR, MODEL_DIR, POINT_DIR]:
    os.makedirs(d, exist_ok=True)

RF_STACK_SINGLE = os.path.join(
    SENTINEL_DIR,
    "Gowa_Sentinel_2026_RF_Input_Stack_10m.tif"
)

RF_STACK_TILE_PATTERN = os.path.join(
    SENTINEL_DIR,
    "Gowa_Sentinel_2026_RF_Input_Stack_10m-*.tif"
)

RF_STACK_VRT = os.path.join(
    SENTINEL_DIR,
    "Gowa_Sentinel_2026_RF_Input_Stack_10m_MOSAIC.vrt"
)

print("Sentinel RF stack single :", RF_STACK_SINGLE)
print("Sentinel RF stack tiles  :", RF_STACK_TILE_PATTERN)
print("2024 rice mask          :", RICE_2024_MASK_PATH)
print("Output                  :", OUT_DIR)

# ============================================================
# 2. MODEL AND SAMPLING PARAMETERS
# ============================================================

RANDOM_SEED = 2026

# Maximum number of samples per class for internal training/testing
# This can be increased, but larger samples require more processing time.
N_SAMPLES_PER_CLASS = 30000

TEST_SIZE = 0.30

N_TREES = 500
MIN_SAMPLES_LEAF = 2
MAX_FEATURES = "sqrt"

# Default RF decision threshold
DEFAULT_THRESHOLD = 0.50

# Final threshold used in mask calibration
SELECTED_FINAL_THRESHOLD = 0.96

# Spectral filters used to construct training seeds
NDVI_MIN = 0.20
NDVI_MAX = 0.78

# Non-rice stable conditions
FOREST_NDVI_MIN = 0.76
FOREST_NDVI_STD_MAX = 0.08
FOREST_NDVI_AMP_MAX = 0.10

MNDWI_WATER_THRESHOLD = 0.25
NDBI_BUILT_THRESHOLD = 0.15
LOW_VEG_NDVI_MAX = 0.18

# Buffer to keep non-rice seeds away from rice-reference pixels
RICE_BUFFER_ITERATIONS = 3

# ============================================================
# 3. HELPER FUNCTIONS
# ============================================================

def build_vrt_from_tiles(tile_files, vrt_path):
    if len(tile_files) == 0:
        raise FileNotFoundError("Tile RF stack was not found.")

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
        raise RuntimeError("Failed to build a VRT from the tiles.")

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
            "Band was not found: "
            + ", ".join(candidates)
            + "\n\nAvailable bands:\n"
            + "\n".join(band_names)
        )

    return None


def read_band(src, idx):
    arr = src.read(idx).astype("float32")

    if src.nodata is not None:
        arr[arr == src.nodata] = np.nan

    arr[np.isinf(arr)] = np.nan
    return arr


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


def sample_from_mask(mask, n_samples, label_id, label_name, transform, seed):
    rng = np.random.default_rng(seed)

    rows, cols = np.where(mask)

    if len(rows) == 0:
        raise ValueError(f"No pixels are available for class {label_name}")

    n_take = min(n_samples, len(rows))
    idx = rng.choice(len(rows), size=n_take, replace=False)

    rows_sel = rows[idx].astype(int)
    cols_sel = cols[idx].astype(int)

    records = []

    for k, (r, c) in enumerate(zip(rows_sel, cols_sel), start=1):
        x, y = rasterio.transform.xy(transform, r, c, offset="center")

        records.append({
            "sample_id": f"{label_name}_{k:05d}",
            "row": int(r),
            "col": int(c),
            "x": float(x),
            "y": float(y),
            "label_id": int(label_id),
            "label_name": label_name
        })

    return pd.DataFrame(records)


def extract_predictor_values(src, sample_df, band_names):
    rows = sample_df["row"].values.astype(int)
    cols = sample_df["col"].values.astype(int)

    X = np.empty((len(sample_df), src.count), dtype="float32")

    for b in range(1, src.count + 1):
        arr = src.read(b).astype("float32")

        if src.nodata is not None:
            arr[arr == src.nodata] = np.nan

        arr[np.isinf(arr)] = np.nan

        X[:, b - 1] = arr[rows, cols]

        if b % 10 == 0 or b == src.count:
            print(f"Band extraction {b}/{src.count} completed.")

    X_df = pd.DataFrame(X, columns=band_names)

    return X_df


def make_metrics_table(y_true, y_pred, y_prob=None, label_text="Test"):
    class_ids = [0, 1]
    class_names = ["Non-rice", "Rice"]

    cm = confusion_matrix(y_true, y_pred, labels=class_ids)

    acc = accuracy_score(y_true, y_pred)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    kappa = cohen_kappa_score(y_true, y_pred)

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=class_ids,
        zero_division=0
    )

    # Producer's accuracy = recall
    producer_accuracy = recall

    # User's accuracy = precision
    user_accuracy = precision

    class_df = pd.DataFrame({
        "Assessment": label_text,
        "Class_id": class_ids,
        "Class_name": class_names,
        "Support_reference": support,
        "Producer_accuracy_recall": producer_accuracy,
        "User_accuracy_precision": user_accuracy,
        "F1_score": f1,
        "Omission_error": 1 - producer_accuracy,
        "Commission_error": 1 - user_accuracy
    })

    auc = np.nan

    if y_prob is not None:
        try:
            auc = roc_auc_score(y_true, y_prob)
        except Exception:
            auc = np.nan

    summary_df = pd.DataFrame([
        {
            "Assessment": label_text,
            "Metric": "Total samples",
            "Value": len(y_true)
        },
        {
            "Assessment": label_text,
            "Metric": "Overall accuracy",
            "Value": acc
        },
        {
            "Assessment": label_text,
            "Metric": "Balanced accuracy",
            "Value": bal_acc
        },
        {
            "Assessment": label_text,
            "Metric": "Kappa coefficient",
            "Value": kappa
        },
        {
            "Assessment": label_text,
            "Metric": "ROC-AUC rice class",
            "Value": auc
        }
    ])

    cm_df = pd.DataFrame(
        cm,
        index=["Reference_Non-rice", "Reference_Rice"],
        columns=["Predicted_Non-rice", "Predicted_Rice"]
    )

    return cm_df, class_df, summary_df


def predict_from_probability(prob, threshold):
    return (prob >= threshold).astype(int)


# ============================================================
# 4. OPEN RF STACK
# ============================================================

tiles = sorted(glob.glob(RF_STACK_TILE_PATTERN))

if os.path.exists(RF_STACK_SINGLE):
    RF_STACK_PATH = RF_STACK_SINGLE
elif os.path.exists(RF_STACK_VRT):
    RF_STACK_PATH = RF_STACK_VRT
else:
    RF_STACK_PATH = build_vrt_from_tiles(tiles, RF_STACK_VRT)

src = rasterio.open(RF_STACK_PATH)
band_names = get_band_names(src)

print("\nRF Stack:", RF_STACK_PATH)
print("Shape:", src.height, src.width)
print("Bands:", src.count)
print("CRS:", src.crs)

# ============================================================
# 5. READ DIAGNOSTIC BANDS TO BUILD TRAINING SEEDS
# ============================================================

ndvi_med_idx = find_band_index(
    band_names,
    ["S2_NDVI_median", "NDVI_median", "NDVI"],
    required=True
)

ndvi_min_idx = find_band_index(
    band_names,
    ["NDVI_min"],
    required=False
)

ndvi_max_idx = find_band_index(
    band_names,
    ["NDVI_max"],
    required=False
)

ndvi_std_idx = find_band_index(
    band_names,
    ["NDVI_stdDev"],
    required=False
)

mndwi_med_idx = find_band_index(
    band_names,
    ["S2_MNDWI_median", "MNDWI_median", "MNDWI"],
    required=False
)

ndbi_med_idx = find_band_index(
    band_names,
    ["S2_NDBI_median", "NDBI_median", "NDBI"],
    required=False
)

NDVI_MED = read_band(src, ndvi_med_idx)

if ndvi_min_idx is not None and ndvi_max_idx is not None:
    NDVI_MIN_ARR = read_band(src, ndvi_min_idx)
    NDVI_MAX_ARR = read_band(src, ndvi_max_idx)
    NDVI_AMP = NDVI_MAX_ARR - NDVI_MIN_ARR
else:
    NDVI_AMP = np.full_like(NDVI_MED, np.nan)

if ndvi_std_idx is not None:
    NDVI_STD = read_band(src, ndvi_std_idx)
else:
    NDVI_STD = np.full_like(NDVI_MED, np.nan)

if mndwi_med_idx is not None:
    MNDWI_MED = read_band(src, mndwi_med_idx)
else:
    MNDWI_MED = np.full_like(NDVI_MED, np.nan)

if ndbi_med_idx is not None:
    NDBI_MED = read_band(src, ndbi_med_idx)
else:
    NDBI_MED = np.full_like(NDVI_MED, np.nan)

valid = np.isfinite(NDVI_MED)

print("\nDiagnostic bands loaded.")
print("Valid pixels:", int(valid.sum()))

# ============================================================
# 6. BUILD RICE AND NON-RICE TRAINING SEEDS
# ============================================================

if not os.path.exists(RICE_2024_MASK_PATH):
    raise FileNotFoundError(f"2024 rice mask was not found:\n{RICE_2024_MASK_PATH}")

rice2024_mask = align_binary_mask(
    RICE_2024_MASK_PATH,
    src,
    positive_values=[1]
)

rice2024_mask = rice2024_mask & valid

ndvi_ok = (
    np.isfinite(NDVI_MED) &
    (NDVI_MED >= NDVI_MIN) &
    (NDVI_MED <= NDVI_MAX)
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

low_veg_like = (
    np.isfinite(NDVI_MED) &
    (NDVI_MED <= LOW_VEG_NDVI_MAX)
)

# Rice positives: 2024 rice-reference pixels that remain plausible according to NDVI
rice_seed = (
    rice2024_mask &
    ndvi_ok &
    (~forest_like) &
    (~water_like) &
    (~built_like)
)

# Buffer to keep non-rice seeds away from the rice reference
rice_buffer = binary_dilation(
    rice2024_mask,
    iterations=RICE_BUFFER_ITERATIONS
)

# Non-rice: stable areas that are clearly not rice fields
nonrice_seed = (
    valid &
    (~rice_buffer) &
    (
        forest_like |
        water_like |
        built_like |
        low_veg_like
    )
)

print("\n===== TRAINING SEED DIAGNOSTIC =====")
print("Rice 2024 reference pixels :", int(rice2024_mask.sum()))
print("Rice seed pixels           :", int(rice_seed.sum()))
print("Non-rice seed pixels       :", int(nonrice_seed.sum()))
print("Forest-like pixels         :", int(forest_like.sum()))
print("Water-like pixels          :", int(water_like.sum()))
print("Built-like pixels          :", int(built_like.sum()))
print("Low vegetation pixels      :", int(low_veg_like.sum()))

if rice_seed.sum() == 0 or nonrice_seed.sum() == 0:
    raise ValueError("One training-seed class contains zero pixels. Check the NDVI/non-rice thresholds.")

# ============================================================
# 7. SAMPLE TRAINING DATA
# ============================================================

rice_samples = sample_from_mask(
    rice_seed,
    N_SAMPLES_PER_CLASS,
    1,
    "Rice",
    src.transform,
    RANDOM_SEED
)

nonrice_samples = sample_from_mask(
    nonrice_seed,
    N_SAMPLES_PER_CLASS,
    0,
    "Non-rice",
    src.transform,
    RANDOM_SEED + 1
)

sample_df = pd.concat(
    [rice_samples, nonrice_samples],
    ignore_index=True
)

sample_df = sample_df.sample(
    frac=1,
    random_state=RANDOM_SEED
).reset_index(drop=True)

sample_csv = os.path.join(
    POINT_DIR,
    "RF_Internal_Model_Samples.csv"
)

sample_geojson = os.path.join(
    POINT_DIR,
    "RF_Internal_Model_Samples.geojson"
)

sample_df.to_csv(sample_csv, index=False)

sample_gdf = gpd.GeoDataFrame(
    sample_df,
    geometry=[Point(xy_pair) for xy_pair in zip(sample_df["x"], sample_df["y"])],
    crs=src.crs
)

sample_gdf.to_file(sample_geojson, driver="GeoJSON")

print("\nModel samples saved:")
print(sample_csv)
print(sample_geojson)

print("\nSample distribution:")
display(sample_df["label_name"].value_counts())

# ============================================================
# 8. EXTRACT PREDICTOR VALUES
# ============================================================

print("\nStarting predictor extraction from the RF stack...")

X_raw_df = extract_predictor_values(src, sample_df, band_names)
y = sample_df["label_id"].values.astype(int)

# Drop bands containing only NaN values
valid_band_mask = ~X_raw_df.isna().all(axis=0)
X_raw_df = X_raw_df.loc[:, valid_band_mask]

used_band_names = list(X_raw_df.columns)

print("\nInitial number of bands:", len(band_names))
print("Number of bands used:", len(used_band_names))

# Median imputation
imputer = SimpleImputer(strategy="median")
X = imputer.fit_transform(X_raw_df)

# Drop any rows that remain NaN after imputation
valid_rows = np.isfinite(X).all(axis=1)

X = X[valid_rows]
y = y[valid_rows]

sample_used_df = sample_df.loc[valid_rows].reset_index(drop=True)

print("\nNumber of samples after extraction and imputation:", len(y))
print("Class distribution after imputation:")
print(pd.Series(y).map({0: "Non-rice", 1: "Rice"}).value_counts())

# ============================================================
# 9. TRAIN-TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test, train_idx, test_idx = train_test_split(
    X,
    y,
    np.arange(len(y)),
    test_size=TEST_SIZE,
    random_state=RANDOM_SEED,
    stratify=y
)

print("\n===== TRAIN-TEST SPLIT =====")
print("Train samples:", len(y_train))
print("Test samples :", len(y_test))
print("Train class distribution:")
print(pd.Series(y_train).map({0: "Non-rice", 1: "Rice"}).value_counts())
print("Test class distribution:")
print(pd.Series(y_test).map({0: "Non-rice", 1: "Rice"}).value_counts())

# ============================================================
# 10. TRAIN RANDOM FOREST
# ============================================================

rf = RandomForestClassifier(
    n_estimators=N_TREES,
    max_features=MAX_FEATURES,
    min_samples_leaf=MIN_SAMPLES_LEAF,
    class_weight="balanced_subsample",
    oob_score=True,
    bootstrap=True,
    n_jobs=-1,
    random_state=RANDOM_SEED
)

print("\nTraining Random Forest...")
rf.fit(X_train, y_train)

print("Training completed.")
print("OOB score:", rf.oob_score_)

# ============================================================
# 11. PREDICT TEST SET
# ============================================================

y_pred_test_default = rf.predict(X_test)
y_prob_test = rf.predict_proba(X_test)[:, 1]

y_pred_test_thr050 = predict_from_probability(
    y_prob_test,
    DEFAULT_THRESHOLD
)

y_pred_test_thr096 = predict_from_probability(
    y_prob_test,
    SELECTED_FINAL_THRESHOLD
)

# ============================================================
# 12. OOB PREDICTION
# ============================================================

oob_prob_all = rf.oob_decision_function_

# Use only training samples with valid OOB predictions
oob_valid = (
    np.isfinite(oob_prob_all).all(axis=1) &
    (oob_prob_all.sum(axis=1) > 0)
)

y_train_oob = y_train[oob_valid]
oob_prob_rice = oob_prob_all[oob_valid, 1]

y_pred_oob_default = predict_from_probability(
    oob_prob_rice,
    DEFAULT_THRESHOLD
)

y_pred_oob_thr096 = predict_from_probability(
    oob_prob_rice,
    SELECTED_FINAL_THRESHOLD
)

# ============================================================
# 13. CALCULATE METRICS
# ============================================================

cm_test_default, class_test_default, summary_test_default = make_metrics_table(
    y_test,
    y_pred_test_default,
    y_prob_test,
    label_text="Test_RF_predict_default"
)

cm_test_thr050, class_test_thr050, summary_test_thr050 = make_metrics_table(
    y_test,
    y_pred_test_thr050,
    y_prob_test,
    label_text="Test_probability_threshold_0.50"
)

cm_test_thr096, class_test_thr096, summary_test_thr096 = make_metrics_table(
    y_test,
    y_pred_test_thr096,
    y_prob_test,
    label_text="Test_probability_threshold_0.96"
)

cm_oob_default, class_oob_default, summary_oob_default = make_metrics_table(
    y_train_oob,
    y_pred_oob_default,
    oob_prob_rice,
    label_text="OOB_probability_threshold_0.50"
)

cm_oob_thr096, class_oob_thr096, summary_oob_thr096 = make_metrics_table(
    y_train_oob,
    y_pred_oob_thr096,
    oob_prob_rice,
    label_text="OOB_probability_threshold_0.96"
)

summary_all = pd.concat(
    [
        summary_test_default,
        summary_test_thr050,
        summary_test_thr096,
        summary_oob_default,
        summary_oob_thr096,
        pd.DataFrame([
            {
                "Assessment": "RF_internal",
                "Metric": "sklearn_oob_score",
                "Value": rf.oob_score_
            }
        ])
    ],
    ignore_index=True
)

class_all = pd.concat(
    [
        class_test_default,
        class_test_thr050,
        class_test_thr096,
        class_oob_default,
        class_oob_thr096
    ],
    ignore_index=True
)

print("\n===== TEST CONFUSION MATRIX DEFAULT RF =====")
display(cm_test_default)

print("\n===== TEST METRICS DEFAULT RF =====")
display(summary_test_default)

print("\n===== TEST ACCURACY BY CLASS DEFAULT RF =====")
display(class_test_default)

print("\n===== OOB METRICS THRESHOLD 0.50 =====")
display(summary_oob_default)
display(class_oob_default)

print("\n===== TEST METRICS THRESHOLD 0.96 =====")
display(summary_test_thr096)
display(class_test_thr096)

# ============================================================
# 14. FEATURE IMPORTANCE
# ============================================================

importance_df = pd.DataFrame({
    "Feature": used_band_names,
    "Importance": rf.feature_importances_
}).sort_values(
    "Importance",
    ascending=False
).reset_index(drop=True)

print("\n===== TOP 30 FEATURE IMPORTANCE =====")
display(importance_df.head(30))

# ============================================================
# 15. SAVE MODEL AND OUTPUTS
# ============================================================

model_path = os.path.join(
    MODEL_DIR,
    "RF_Rice_Model_Internal_Accuracy.joblib"
)

imputer_path = os.path.join(
    MODEL_DIR,
    "RF_Rice_Model_Imputer.joblib"
)

joblib.dump(rf, model_path)
joblib.dump(imputer, imputer_path)

test_points_df = sample_used_df.loc[test_idx].copy()
test_points_df["split"] = "test"
test_points_df["reference_label"] = y_test
test_points_df["pred_default"] = y_pred_test_default
test_points_df["prob_rice"] = y_prob_test

train_points_df = sample_used_df.loc[train_idx].copy()
train_points_df["split"] = "train"

train_test_points_df = pd.concat(
    [train_points_df, test_points_df],
    ignore_index=True
)

train_test_points_csv = os.path.join(
    POINT_DIR,
    "RF_Internal_Train_Test_Samples_With_Prediction.csv"
)

train_test_points_df.to_csv(train_test_points_csv, index=False)

excel_path = os.path.join(
    TABLE_DIR,
    "RF_Model_Internal_Accuracy_Assessment.xlsx"
)

with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
    pd.DataFrame([
        {
            "Item": "RF stack",
            "Value": RF_STACK_PATH
        },
        {
            "Item": "2024 rice seed",
            "Value": RICE_2024_MASK_PATH
        },
        {
            "Item": "Samples per class requested",
            "Value": N_SAMPLES_PER_CLASS
        },
        {
            "Item": "Train size",
            "Value": len(y_train)
        },
        {
            "Item": "Test size",
            "Value": len(y_test)
        },
        {
            "Item": "Number of trees",
            "Value": N_TREES
        },
        {
            "Item": "Max features",
            "Value": MAX_FEATURES
        },
        {
            "Item": "Min samples leaf",
            "Value": MIN_SAMPLES_LEAF
        },
        {
            "Item": "OOB score",
            "Value": rf.oob_score_
        },
        {
            "Item": "Default threshold",
            "Value": DEFAULT_THRESHOLD
        },
        {
            "Item": "Selected final threshold",
            "Value": SELECTED_FINAL_THRESHOLD
        }
    ]).to_excel(writer, sheet_name="Model_Setting", index=False)

    sample_df.to_excel(writer, sheet_name="Sample_All", index=False)
    train_test_points_df.to_excel(writer, sheet_name="Train_Test_Points", index=False)

    cm_test_default.to_excel(writer, sheet_name="CM_Test_Default")
    cm_test_thr050.to_excel(writer, sheet_name="CM_Test_Thr050")
    cm_test_thr096.to_excel(writer, sheet_name="CM_Test_Thr096")
    cm_oob_default.to_excel(writer, sheet_name="CM_OOB_Thr050")
    cm_oob_thr096.to_excel(writer, sheet_name="CM_OOB_Thr096")

    summary_all.to_excel(writer, sheet_name="Summary_All", index=False)
    class_all.to_excel(writer, sheet_name="Accuracy_By_Class_All", index=False)
    importance_df.to_excel(writer, sheet_name="Feature_Importance", index=False)

summary_csv = os.path.join(
    TABLE_DIR,
    "RF_Model_Internal_Accuracy_Summary.csv"
)

class_csv = os.path.join(
    TABLE_DIR,
    "RF_Model_Internal_Accuracy_By_Class.csv"
)

importance_csv = os.path.join(
    TABLE_DIR,
    "RF_Model_Feature_Importance.csv"
)

summary_all.to_csv(summary_csv, index=False)
class_all.to_csv(class_csv, index=False)
importance_df.to_csv(importance_csv, index=False)

# ============================================================
# 16. TEXT REPORT
# ============================================================

def fmt_pct(x):
    if pd.isna(x):
        return "NA"
    return f"{x * 100:.2f}%"

def get_metric(summary_df, name):
    row = summary_df[summary_df["Metric"] == name]
    if len(row) == 0:
        return np.nan
    return float(row["Value"].iloc[0])

oa_test = get_metric(summary_test_default, "Overall accuracy")
ba_test = get_metric(summary_test_default, "Balanced accuracy")
kappa_test = get_metric(summary_test_default, "Kappa coefficient")
auc_test = get_metric(summary_test_default, "ROC-AUC rice class")

oa_oob_050 = get_metric(summary_oob_default, "Overall accuracy")
kappa_oob_050 = get_metric(summary_oob_default, "Kappa coefficient")
auc_oob_050 = get_metric(summary_oob_default, "ROC-AUC rice class")

report_lines = []

report_lines.append("RF INTERNAL MODEL ACCURACY REPORT")
report_lines.append("Gowa Regency rice-field classification, Sentinel-1/Sentinel-2, 2026")
report_lines.append("")
report_lines.append("Important interpretation:")
report_lines.append("These metrics represent internal Random Forest model performance based on training/test seed samples.")
report_lines.append("They are not a substitute for map accuracy assessment using independent interpreted samples or AcATaMa.")
report_lines.append("")
report_lines.append(f"RF stack: {RF_STACK_PATH}")
report_lines.append(f"2024 rice positive seed: {RICE_2024_MASK_PATH}")
report_lines.append("")
report_lines.append("Model settings:")
report_lines.append(f"- Number of trees: {N_TREES}")
report_lines.append(f"- Max features: {MAX_FEATURES}")
report_lines.append(f"- Min samples leaf: {MIN_SAMPLES_LEAF}")
report_lines.append(f"- Class weight: balanced_subsample")
report_lines.append(f"- OOB enabled: True")
report_lines.append("")
report_lines.append("Samples:")
report_lines.append(f"- Total samples used: {len(y)}")
report_lines.append(f"- Train samples: {len(y_train)}")
report_lines.append(f"- Test samples: {len(y_test)}")
report_lines.append("")
report_lines.append("Main internal RF accuracy:")
report_lines.append(f"- sklearn OOB score: {fmt_pct(rf.oob_score_)}")
report_lines.append(f"- OOB overall accuracy, threshold 0.50: {fmt_pct(oa_oob_050)}")
report_lines.append(f"- OOB kappa, threshold 0.50: {kappa_oob_050:.4f}")
report_lines.append(f"- OOB ROC-AUC, rice class: {auc_oob_050:.4f}")
report_lines.append(f"- Test overall accuracy: {fmt_pct(oa_test)}")
report_lines.append(f"- Test balanced accuracy: {fmt_pct(ba_test)}")
report_lines.append(f"- Test kappa: {kappa_test:.4f}")
report_lines.append(f"- Test ROC-AUC, rice class: {auc_test:.4f}")
report_lines.append("")
report_lines.append("Test accuracy by class, default RF prediction:")

for _, row in class_test_default.iterrows():
    report_lines.append(
        f"- {row['Class_name']}: "
        f"Producer's accuracy/Recall = {fmt_pct(row['Producer_accuracy_recall'])}, "
        f"User's accuracy/Precision = {fmt_pct(row['User_accuracy_precision'])}, "
        f"F1-score = {row['F1_score']:.4f}"
    )

report_lines.append("")
report_lines.append("Top 15 feature importance:")
for _, row in importance_df.head(15).iterrows():
    report_lines.append(
        f"- {row['Feature']}: {row['Importance']:.5f}"
    )

report_text = "\n".join(report_lines)

report_path = os.path.join(
    OUT_DIR,
    "RF_Internal_Model_Accuracy_Report.txt"
)

with open(report_path, "w", encoding="utf-8") as f:
    f.write(report_text)

print("\n============================================================")
print("RF INTERNAL CLASS-SEPARABILITY ASSESSMENT COMPLETED")
print("============================================================")
print(report_text)
print("")
print("Excel output:")
print(excel_path)
print("")
print("Report:")
print(report_path)
print("")
print("Model:")
print(model_path)
print("============================================================")
