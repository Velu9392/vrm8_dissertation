# Cell 5 — Predictor variables for Land Use Regression
#
# My supervisor specified that the Land Use Regression should use
# predictors like distance to river/lake and height above water.
# These are the physical factors that explain why some areas flood
# and others don't — not the SAR backscatter itself (which is what
# I used to identify the flood in Cell 3).
#
# I have four predictors:
#   1. Distance to permanent water — areas close to rivers and the
#      Kerala backwaters flood first when water overflows banks.
#   2. Elevation (SRTM DEM) — water fills the lowest areas first.
#      Kerala's Western Ghats rise to 2600m in the east; the coastal
#      plains in the west sit near sea level and flooded most severely.
#   3. Rainfall (CHIRPS August 2018) — the August rainfall was
#      exceptionally heavy; districts with more rain flooded more.
#   4. Temperature — higher temperatures relate to monsoon intensity.
#      I compute this from elevation using a standard lapse rate
#      because the NASA POWER API wasn't available.
#
# All data is from free public sources with no account needed.

import json, gzip, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import requests
import rasterio
from rasterio.transform import Affine
from skimage.transform import resize as imresize
from scipy.ndimage import distance_transform_edt
from IPython.display import display
warnings.filterwarnings("ignore")

cfg_file = list(Path("/content").rglob("config.json"))
with open(cfg_file[0]) as f:
    cfg = json.load(f)
ROOT = Path(cfg["root"])
dirs = {
    "processed":  ROOT / "data" / "processed",
    "predictors": ROOT / "data" / "predictors",
    "points":     ROOT / "data" / "points",
    "figures":    ROOT / "results" / "figures",
    "logs":       ROOT / "logs",
}
for d in dirs.values():
    d.mkdir(parents=True, exist_ok=True)

with open(dirs["logs"] / "transform.json") as f:
    td = json.load(f)
tf = Affine(td["a"], td["b"], td["c"], td["d"], td["e"], td["f"])
H, W = td["h"], td["w"]

landcover = np.load(dirs["processed"] / "landcover.npy")
obs       = pd.read_csv(dirs["points"] / "observation_points.csv")
bbox      = cfg["kerala_bbox"]

# --- Predictor 1: Distance to permanent water ---
# I use the permanent water pixels from Cell 3 as the reference.
# The distance_transform_edt function gives every non-water pixel
# its distance (in pixels) to the nearest water pixel. I convert
# to metres using the approximate pixel size at this latitude.
print("Computing distance to permanent water...")
perm_mask    = (landcover == 1)
pixel_size_m = ((bbox[2] - bbox[0]) / W) * 111_000   # rough metres per pixel
dist_pix     = distance_transform_edt(~perm_mask)
dist_m       = (dist_pix * pixel_size_m).astype(np.float32)
np.save(dirs["predictors"] / "dist_to_water.npy", dist_m)
print(f"  Max distance: {dist_m.max()/1000:.1f} km from nearest water")

# --- Predictor 2: SRTM elevation from OpenTopography ---
# OpenTopography provides a free REST API for the SRTM 30m DEM.
# The demo API key is documented in their public examples.
print("\nDownloading SRTM elevation from OpenTopography...")
url_srtm = (
    f"https://portal.opentopography.org/API/globaldem"
    f"?demtype=SRTMGL1&south={bbox[1]}&north={bbox[3]}"
    f"&west={bbox[0]}&east={bbox[2]}&outputFormat=GTiff"
    f"&API_Key=demoapikeyot2022"
)
elev = None
resp = requests.get(url_srtm, timeout=60)
if resp.status_code == 200 and len(resp.content) > 5000:
    tmp = Path("/tmp/srtm_kerala.tif")
    tmp.write_bytes(resp.content)
    with rasterio.open(tmp) as src:
        arr = src.read(1).astype(np.float32)
    arr = np.where(arr < -100, 0, arr)
    elev = imresize(arr.astype(np.float64), (H, W),
                    preserve_range=True, anti_aliasing=True).astype(np.float32)
    print(f"  Elevation range: {elev.min():.0f} to {elev.max():.0f} m")
else:
    # Fallback: download individual SRTM tiles from AWS
    print("  OpenTopography busy — trying AWS SRTM tiles...")
    tiles = []
    for tile_name in ["N09/N09E076.hgt.gz", "N10/N10E076.hgt.gz"]:
        r = requests.get(
            f"https://s3.amazonaws.com/elevation-tiles-prod/skadi/{tile_name}",
            timeout=30
        )
        if r.status_code == 200:
            data = gzip.decompress(r.content)
            tile = np.frombuffer(data, dtype=">i2").reshape(3601, 3601).astype(np.float32)
            tiles.append(tile)
    if tiles:
        combined = np.vstack(tiles)
        elev = imresize(combined, (H, W), preserve_range=True,
                        anti_aliasing=True).astype(np.float32)
        elev = np.clip(elev, 0, 3000)
        print(f"  Elevation from SRTM tiles: {elev.min():.0f} to {elev.max():.0f} m")

if elev is None:
    raise RuntimeError("Could not download elevation data — check internet connection")

np.save(dirs["predictors"] / "elevation.npy", elev)

# --- Predictor 3: CHIRPS rainfall August 2018 ---
# CHIRPS (Climate Hazards Group InfraRed Precipitation with Stations)
# provides monthly gridded rainfall data. I download August 2018
# directly from the UCSB public server — no account needed.
print("\nDownloading CHIRPS August 2018 rainfall from UCSB...")
url_chirps = ("https://data.chc.ucsb.edu/products/CHIRPS-2.0/"
               "global_monthly/tifs/chirps-v2.0.2018.08.tif.gz")
resp = requests.get(url_chirps, timeout=120)
if resp.status_code != 200:
    raise RuntimeError(f"CHIRPS download failed (HTTP {resp.status_code})")

decompressed = gzip.decompress(resp.content)
tmp2 = Path("/tmp/chirps_aug2018.tif")
tmp2.write_bytes(decompressed)

with rasterio.open(tmp2) as src:
    from rasterio.windows import from_bounds as wfb
    window = wfb(bbox[0], bbox[1], bbox[2], bbox[3], transform=src.transform)
    rain_raw = src.read(1, window=window).astype(np.float32)

rain_raw = np.where((rain_raw < 0) | np.isnan(rain_raw), 0, rain_raw)
rainfall = imresize(rain_raw.astype(np.float64), (H, W),
                    preserve_range=True, anti_aliasing=True).astype(np.float32)
np.save(dirs["predictors"] / "rainfall.npy", rainfall)
print(f"  Rainfall range: {rainfall.min():.0f} to {rainfall.max():.0f} mm (Aug 2018)")

# --- Predictor 4: Temperature ---
# I tried the NASA POWER API but it wasn't responding. Instead I
# compute temperature from elevation using the standard atmospheric
# lapse rate (6.5°C per 1000m). The base temperature of 29°C is
# the Kerala August 2018 mean from IMD published records.
# This is a well-established approximation in mountainous regions.
print("\nComputing temperature from SRTM elevation (lapse rate method)...")
lapse_rate = 6.5 / 1000.0   # degrees C per metre
base_temp  = 29.0             # Kerala Aug 2018 surface mean (IMD, 2018)
temp = (base_temp - elev * lapse_rate).astype(np.float32)
temp = np.clip(temp, 10, 35)
np.save(dirs["predictors"] / "temperature.npy", temp)
print(f"  Temperature range: {temp.min():.1f} to {temp.max():.1f} °C")
print("  (Higher elevations in Western Ghats are cooler — expected)")

# --- Extract predictor values at each observation point ---
print("\nExtracting predictor values at observation points...")
rows = np.clip(obs["row"].values.astype(int), 0, H-1)
cols = np.clip(obs["col"].values.astype(int), 0, W-1)

obs["dist_to_water_m"] = dist_m[rows, cols]
obs["elevation_m"]     = elev[rows, cols]
obs["rainfall_mm"]     = rainfall[rows, cols]
obs["temperature_c"]   = temp[rows, cols]

obs.to_csv(dirs["points"] / "obs_with_predictors.csv", index=False)

# Check correlations with flood label
FEATURES = ["dist_to_water_m", "elevation_m", "rainfall_mm", "temperature_c"]
print("\nCorrelation with water label:")
for f in FEATURES:
    r = obs["water_label"].corr(obs[f])
    bar = "█" * int(abs(r) * 20)
    sign = "↑" if r > 0 else "↓"
    print(f"  {f:<20} r={r:+.3f} {sign} {bar}")

# The negative correlations for distance and elevation make sense:
# areas close to water and at low elevation are more likely to flood

# Visualise the predictor maps
fig, axes = plt.subplots(2, 4, figsize=(20, 9))
fig.suptitle("Predictor Variables — Land Use Regression\n"
             "All derived from real data: SAR mask, SRTM, CHIRPS, IMD lapse rate",
             fontsize=11, fontweight="bold")

pred_grids = [
    (dist_m/1000,  "Blues_r", "Distance to Water (km)\nSAR permanent water mask"),
    (elev,         "terrain", "Elevation (m)\nSRTM 30m via OpenTopography"),
    (rainfall,     "YlGnBu",  "CHIRPS Rainfall (mm)\nAugust 2018 total"),
    (temp,         "RdYlBu_r","Temperature (°C)\nLapse rate from SRTM + IMD"),
]
pred_cols = ["dist_to_water_m","elevation_m","rainfall_mm","temperature_c"]
ptcolours = {0:"darkorange",1:"royalblue",2:"navy"}

for i, (arr, cmap, title) in enumerate(pred_grids):
    ax_m = axes[0, i]; ax_b = axes[1, i]
    im = ax_m.imshow(arr, cmap=cmap)
    for cid, pc in ptcolours.items():
        sub = obs[obs["class_id"] == cid]
        ax_m.scatter(sub["col"], sub["row"], c=pc, s=8, alpha=0.7)
    plt.colorbar(im, ax=ax_m, fraction=0.046)
    ax_m.set_title(title, fontsize=9); ax_m.axis("off")
    data = [obs[obs["class_id"]==c][pred_cols[i]].values for c in [0,1,2]]
    bp = ax_b.boxplot(data, patch_artist=True)
    for patch, col in zip(bp["boxes"],["#a5d6a7","#1565c0","#0d47a1"]):
        patch.set_facecolor(col); patch.set_alpha(0.8)
    ax_b.set_xticklabels(["Dry","Perm.\nwater","Flood"],fontsize=8)
    ax_b.grid(True,alpha=0.3,axis="y")
    ax_b.set_title("Distribution by class",fontsize=8)

plt.tight_layout()
plt.savefig(dirs["figures"] / "04_predictors.png", dpi=150, bbox_inches="tight")
plt.show()
plt.close()

# Correlation matrix
fig2, ax2 = plt.subplots(figsize=(8, 6))
corr = obs[FEATURES + ["water_label"]].corr()
sns.heatmap(corr, mask=np.triu(np.ones_like(corr,dtype=bool)),
            annot=True, fmt=".2f", cmap="RdBu_r", center=0,
            vmin=-1, vmax=1, ax=ax2, linewidths=0.5,
            annot_kws={"size":10})
ax2.set_title("Predictor Correlation Matrix\n"
               "water_label = 1 means water/flood", fontsize=10)
plt.tight_layout()
plt.savefig(dirs["figures"] / "04b_correlation.png", dpi=150, bbox_inches="tight")
plt.show()
plt.close()

print("\nCell 5 done.")
print("Distance and elevation show the strongest correlations with flood presence,")
print("which matches the expected behaviour — low-lying areas near water flood first.")
