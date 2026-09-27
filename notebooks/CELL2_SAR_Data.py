# Cell 2 — Downloading real Sentinel-1 SAR data
#
# I'm using Microsoft Planetary Computer to get the actual Sentinel-1
# imagery. It doesn't require an account and it holds the full archive
# going back to 2014. The library I use (odc-stac) is important because
# it only downloads the Kerala portion of each file rather than the full
# satellite swath — without this the download would be many gigabytes.
#
# Sentinel-1 records in two polarisations: VH (vertical transmit,
# horizontal receive) and VV (vertical transmit, vertical receive).
# VH is the main one I'll use for flood detection because calm water
# produces a very low VH signal — the flat surface bounces radar
# away from the satellite rather than back to it.
#
# The data from Planetary Computer is stored in linear power units,
# not decibels. I convert to dB using 10*log10() because dB values
# are easier to interpret and compare — flood water is around -23 dB
# while vegetated land is around -12 dB.

import subprocess, sys
for pkg in ["planetary-computer", "pystac-client", "odc-stac", "odc-geo",
            "rasterio", "scikit-image"]:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", pkg],
                   capture_output=True)

import json, warnings
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import rasterio
from rasterio.transform import from_bounds
from skimage.transform import resize as imresize
import planetary_computer
from pystac_client import Client
from odc.stac import stac_load
warnings.filterwarnings("ignore")

cfg_file = list(Path("/content").rglob("config.json"))
if not cfg_file:
    raise FileNotFoundError("Run Cell 1 first")
with open(cfg_file[0]) as f:
    cfg = json.load(f)

ROOT = Path(cfg["root"])
dirs = {
    "sar":      ROOT / "data" / "sar",
    "processed": ROOT / "data" / "processed",
    "figures":  ROOT / "results" / "figures",
    "logs":     ROOT / "logs",
}
for d in dirs.values():
    d.mkdir(parents=True, exist_ok=True)

bbox        = cfg["kerala_bbox"]
pre_start   = cfg["pre_flood_start"]
pre_end     = cfg["pre_flood_end"]
flood_start = cfg["flood_start"]
flood_end   = cfg["flood_end"]
GRID = 512       # output grid size in pixels
RES  = 0.003     # ~300m per pixel in degrees

print("Connecting to Microsoft Planetary Computer...")
catalog = Client.open(
    "https://planetarycomputer.microsoft.com/api/stac/v1",
    modifier=planetary_computer.sign_inplace
)
print("Connected — no account needed")

def get_scenes(start, end, label):
    results = catalog.search(
        collections=["sentinel-1-rtc"],
        bbox=bbox,
        datetime=f"{start}/{end}",
        max_items=10,
    )
    items = list(results.items())
    print(f"{label}: {len(items)} scenes")
    if not items:
        raise RuntimeError(f"No scenes found for {label}")
    return items

print("\nSearching for Sentinel-1 scenes over Kerala...")
scenes_pre   = get_scenes(pre_start,   pre_end,   "Pre-flood Jul 2018")
scenes_flood = get_scenes(flood_start, flood_end, "Flood Aug 2018")

def load_composite(scenes, label):
    # Sign each item so Planetary Computer lets us download it
    signed = [planetary_computer.sign(item) for item in scenes]

    # odc-stac downloads only the Kerala tiles from each COG file
    # and reprojects from UTM to WGS84 automatically
    ds = stac_load(
        signed,
        bands=["vh", "vv"],
        crs="EPSG:4326",
        resolution=RES,
        bbox=bbox,
        dtype="float32",
        groupby="solar_day",
    )

    print(f"Loading {label} ({len(scenes)} scenes)...")
    composite = ds.mean("time").compute()

    vh_lin = composite["vh"].values.astype(np.float32)
    vv_lin = composite["vv"].values.astype(np.float32)

    # Remove band dimension if present
    if vh_lin.ndim == 3:
        vh_lin = vh_lin[0]
        vv_lin = vv_lin[0]

    # Values <= 0 are no-data (can't take log of zero or negative)
    # Everything else gets converted to dB
    eps = 1e-6
    vh_db = np.where(vh_lin > eps, 10 * np.log10(vh_lin), np.nan).astype(np.float32)
    vv_db = np.where(vv_lin > eps, 10 * np.log10(vv_lin), np.nan).astype(np.float32)

    # Fill any NaN at edges with the scene median
    vh_db = np.where(np.isnan(vh_db), np.nanmedian(vh_db), vh_db)
    vv_db = np.where(np.isnan(vv_db), np.nanmedian(vv_db), vv_db)

    print(f"  VH mean: {vh_db.mean():.1f} dB  shape: {vh_db.shape}")
    return vh_db, vv_db

vh_pre,   vv_pre   = load_composite(scenes_pre,   "pre-flood")
vh_flood, vv_flood = load_composite(scenes_flood, "flood period")

# Resize both to the same 512x512 grid
def resize_to_grid(arr, h=GRID, w=GRID):
    if arr.shape == (h, w):
        return arr
    return imresize(arr.astype(np.float64), (h, w),
                    preserve_range=True, anti_aliasing=True).astype(np.float32)

vh_pre   = resize_to_grid(vh_pre)
vv_pre   = resize_to_grid(vv_pre)
vh_flood = resize_to_grid(vh_flood)
vv_flood = resize_to_grid(vv_flood)

tf = from_bounds(bbox[0], bbox[1], bbox[2], bbox[3], GRID, GRID)

print(f"\nPre-flood VH: {vh_pre.mean():.2f} dB")
print(f"Flood VH:     {vh_flood.mean():.2f} dB")
print(f"Mean drop:    {(vh_flood - vh_pre).mean():.2f} dB")

# Save for the next cells
np.savez_compressed(
    dirs["processed"] / "sar_scenes.npz",
    vh_pre=vh_pre, vv_pre=vv_pre,
    vh_flood=vh_flood, vv_flood=vv_flood,
)

# Save the transform so other cells can convert pixel coords to lat/lon
tf_dict = {"a":tf.a,"b":tf.b,"c":tf.c,"d":tf.d,"e":tf.e,"f":tf.f,
           "h":GRID,"w":GRID}
with open(dirs["logs"] / "transform.json","w") as f:
    json.dump(tf_dict, f)

def save_tif(arr, path):
    with rasterio.open(path, "w", driver="GTiff", height=GRID, width=GRID,
                       count=1, dtype=rasterio.float32,
                       crs="EPSG:4326", transform=tf, compress="lzw") as dst:
        dst.write(arr, 1)

save_tif(vh_pre,   dirs["sar"] / "vh_pre.tif")
save_tif(vv_pre,   dirs["sar"] / "vv_pre.tif")
save_tif(vh_flood, dirs["sar"] / "vh_flood.tif")
save_tif(vv_flood, dirs["sar"] / "vv_flood.tif")

# Visualise the two scenes side by side
vh_change = vh_flood - vh_pre

fig, axes = plt.subplots(1, 4, figsize=(18, 5))
fig.suptitle("Sentinel-1 VH Backscatter — Kerala 2018\nSource: Microsoft Planetary Computer",
             fontsize=11, fontweight="bold")

vmin = min(np.percentile(vh_pre,5), np.percentile(vh_flood,5))
vmax = max(np.percentile(vh_pre,95),np.percentile(vh_flood,95))

axes[0].imshow(vh_pre, cmap="gray", vmin=vmin, vmax=vmax)
axes[0].set_title(f"Pre-Flood VH (Jul 2018)\nmean = {vh_pre.mean():.1f} dB")
axes[0].axis("off")

axes[1].imshow(vh_flood, cmap="gray", vmin=vmin, vmax=vmax)
axes[1].set_title(f"Flood Period VH (Aug 2018)\nmean = {vh_flood.mean():.1f} dB")
axes[1].axis("off")

im = axes[2].imshow(vh_change, cmap="RdBu", vmin=-12, vmax=6)
axes[2].set_title("VH Change (flood − pre)\nBlue areas = backscatter dropped")
axes[2].axis("off")
plt.colorbar(im, ax=axes[2], fraction=0.046, label="dB")

axes[3].hist(vh_pre.ravel(), bins=80, density=True, alpha=0.6,
              color="darkorange", label=f"Pre-flood ({vh_pre.mean():.1f} dB)")
axes[3].hist(vh_flood.ravel(), bins=80, density=True, alpha=0.6,
              color="steelblue", label=f"Flood period ({vh_flood.mean():.1f} dB)")
axes[3].set_xlabel("VH Backscatter (dB)")
axes[3].set_ylabel("Density")
axes[3].set_title("Backscatter distributions")
axes[3].legend(fontsize=9)
axes[3].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(dirs["figures"] / "01_sar_scenes.png", dpi=150, bbox_inches="tight")
plt.show()
plt.close()

print("\nCell 2 done. SAR data saved.")
print("The mean VH drop of", round((vh_flood-vh_pre).mean(),2),
      "dB indicates the flood signal is present across Kerala.")
