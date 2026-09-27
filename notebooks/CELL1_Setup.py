# Kerala 2018 Flood Detection — GY7720 Dissertation
# Student: VRM8 | University Of Leicester
#
# This notebook uses Sentinel-1 SAR imagery to detect the 2018 Kerala
# floods. Kerala received the worst monsoon rainfall since 1924 that
# year — I chose this event because the data is well-documented and
# the flood extent was severe enough to be clearly visible in SAR.
#
# The workflow follows the supervisor's guidance:
#   - Two SAR images (pre-flood and flood period)
#   - 300+ observation points across three land cover classes
#   - Land use regression with terrain and climate predictors
#   - Random Forest as the primary prediction model

import subprocess, sys

# Only three packages are missing from Colab's default environment
for pkg in ["imbalanced-learn", "rasterio", "scikit-image"]:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", pkg],
                   capture_output=True)

import os, json, warnings
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import seaborn as sns
import rasterio
from rasterio.transform import from_bounds
from scipy.ndimage import distance_transform_edt, gaussian_filter, binary_opening, binary_closing
from scipy.stats import pearsonr
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, roc_curve, classification_report)
from imblearn.over_sampling import SMOTE
from IPython.display import display
warnings.filterwarnings("ignore")

# Mount Drive so results survive Colab session resets
try:
    from google.colab import drive
    drive.mount("/content/drive", force_remount=False)
    ROOT = Path("/content/drive/MyDrive/VRM8_Kerala_Flood")
    print("Drive mounted")
except Exception:
    ROOT = Path("/content/VRM8_Kerala_Flood")
    print("Running without Drive — saving locally")

# Folder structure — one folder per type of output so nothing gets mixed up
dirs = {
    "sar":        ROOT / "data" / "sar",
    "processed":  ROOT / "data" / "processed",
    "points":     ROOT / "data" / "points",
    "predictors": ROOT / "data" / "predictors",
    "models":     ROOT / "models",
    "figures":    ROOT / "results" / "figures",
    "maps":       ROOT / "results" / "maps",
    "logs":       ROOT / "logs",
}
for d in dirs.values():
    d.mkdir(parents=True, exist_ok=True)

# Study area: Kerala bounding box covers the most severely flooded
# districts — Ernakulam, Idukki, Thrissur, Alappuzha, Pathanamthitta
kerala_bbox = [76.0, 9.5, 77.5, 11.0]  # lon_min, lat_min, lon_max, lat_max

# I use July 2018 as the pre-flood baseline because the southwest monsoon
# typically starts in June — July is active monsoon but before the August
# extreme event. The flood peak was 15–25 August 2018 (Kerala SDMA, 2018)
pre_start  = "2018-07-01"
pre_end    = "2018-07-31"
flood_start = "2018-08-15"
flood_end   = "2018-08-25"

n_obs   = 350   # markers asked for minimum 300 — I use 350 for safety
seed    = 42
test_sz = 0.25  # 75% train, 25% test

config = {
    "root": str(ROOT),
    "kerala_bbox": kerala_bbox,
    "pre_flood_start": pre_start,
    "pre_flood_end":   pre_end,
    "flood_start":     flood_start,
    "flood_end":       flood_end,
    "n_obs_points":    n_obs,
    "random_state":    seed,
    "test_size":       test_sz,
    "created":         datetime.now().isoformat(),
}
with open(dirs["logs"] / "config.json", "w") as f:
    json.dump(config, f, indent=2)

print(f"Study area: Kerala {kerala_bbox}")
print(f"Pre-flood: {pre_start} to {pre_end}")
print(f"Flood peak: {flood_start} to {flood_end}")
print(f"Observation points: {n_obs}")
print(f"Saved config to: {dirs['logs']}/config.json")
print("Setup complete")
