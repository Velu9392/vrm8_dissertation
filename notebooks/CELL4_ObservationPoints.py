# Cell 4 — Creating the observation points
#
# The Land Use Regression and Random Forest models need a table of
# individual sample locations where I know the land cover class. This
# is the "ground truth" dataset that the models will learn from.
#
# My supervisor asked for at least 300 points. I'm creating 350 to
# give some headroom in case a few points end up in edge areas.
#
# I use stratified sampling — equal numbers from each class — because
# flood water only covers about 8-9% of the image. If I just sampled
# randomly, I'd get very few flood pixels and the model would struggle
# to learn the flood pattern. Balanced classes give both models a fair
# chance to learn the differences.
#
# Classes:
#   permanent water = label 1  (rivers, backwaters — always wet)
#   flood water     = label 1  (land that flooded in August 2018)
#   dry land        = label 0  (areas that stayed dry)
#
# I combine permanent water and flood water into the same label (1)
# because the question I'm answering is "is this pixel water or not?"

import json, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.geometry import Point
from rasterio.transform import Affine
warnings.filterwarnings("ignore")

cfg_file = list(Path("/content").rglob("config.json"))
with open(cfg_file[0]) as f:
    cfg = json.load(f)
ROOT = Path(cfg["root"])
dirs = {
    "processed": ROOT / "data" / "processed",
    "points":    ROOT / "data" / "points",
    "figures":   ROOT / "results" / "figures",
    "logs":      ROOT / "logs",
}

with open(dirs["logs"] / "transform.json") as f:
    td = json.load(f)
tf = Affine(td["a"], td["b"], td["c"], td["d"], td["e"], td["f"])
H, W = td["h"], td["w"]

d = np.load(dirs["processed"] / "sar_scenes.npz")
vh_pre   = d["vh_pre"];   vv_pre   = d["vv_pre"]
vh_flood = d["vh_flood"]; vv_flood = d["vv_flood"]
vh_change = np.load(dirs["processed"] / "vh_change.npy")
vv_change = np.load(dirs["processed"] / "vv_change.npy")
landcover = np.load(dirs["processed"] / "landcover.npy")

n_total = cfg["n_obs_points"]  # 350
seed    = cfg["random_state"]
rng     = np.random.default_rng(seed)

# Get the pixel locations for each class
dry_px   = np.argwhere(landcover == 0)
perm_px  = np.argwhere(landcover == 1)
flood_px = np.argwhere(landcover == 2)

n_each = n_total // 3

print(f"Available pixels per class:")
print(f"  Dry land:       {len(dry_px):,}")
print(f"  Permanent water:{len(perm_px):,}")
print(f"  Flood water:    {len(flood_px):,}")

# Take as many as available (permanent water might be limited in my area)
n_dry   = min(n_each, len(dry_px))
n_perm  = min(n_each, len(perm_px))
n_flood = min(n_each, len(flood_px))

def spread_sample(pixels, n, rng):
    # I divide the pixel array into n chunks and pick one from each.
    # This spreads the sample across the whole image rather than
    # clustering all points in one corner.
    if len(pixels) <= n:
        return pixels
    idx = rng.permutation(len(pixels))
    chunks = np.array_split(pixels[idx], n)
    return np.vstack([c[rng.integers(len(c))] for c in chunks if len(c) > 0])[:n]

dry_sample   = spread_sample(dry_px,   n_dry,   rng)
perm_sample  = spread_sample(perm_px,  n_perm,  rng)
flood_sample = spread_sample(flood_px, n_flood, rng)

print(f"\nSampled:")
print(f"  Dry land:       {len(dry_sample)}")
print(f"  Permanent water:{len(perm_sample)}")
print(f"  Flood water:    {len(flood_sample)}")

# Build the observation points table
records = []
for class_id, class_name, samples in [
    (0, "dry_land",        dry_sample),
    (1, "permanent_water", perm_sample),
    (2, "flood_water",     flood_sample),
]:
    for row, col in samples:
        # Convert pixel to geographic coordinates using the affine transform
        lon = tf.c + (col + 0.5) * tf.a
        lat = tf.f + (row + 0.5) * tf.e
        records.append({
            "point_id":   len(records),
            "row":        int(row),
            "col":        int(col),
            "longitude":  round(float(lon), 6),
            "latitude":   round(float(lat), 6),
            "class_id":   class_id,
            "class_name": class_name,
            "water_label": 1 if class_id >= 1 else 0,
            "vh_pre":     float(vh_pre[row, col]),
            "vv_pre":     float(vv_pre[row, col]),
            "vh_flood":   float(vh_flood[row, col]),
            "vv_flood":   float(vv_flood[row, col]),
            "vh_change":  float(vh_change[row, col]),
            "vv_change":  float(vv_change[row, col]),
        })

obs = pd.DataFrame(records)
obs.to_csv(dirs["points"] / "observation_points.csv", index=False)

# Save as GeoPackage too so it can be opened in QGIS
gdf = gpd.GeoDataFrame(
    obs,
    geometry=[Point(r.longitude, r.latitude) for r in obs.itertuples()],
    crs="EPSG:4326"
)
gdf.to_file(dirs["points"] / "observation_points.gpkg", driver="GPKG")

print(f"\nTotal observation points saved: {len(obs)}")

# Visualise the points
colours = {0: "darkorange", 1: "royalblue", 2: "navy"}
labels  = {0: "Dry land", 1: "Permanent water", 2: "Flood water"}

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle(f"Observation Points: {len(obs)} points across 3 classes\n"
             "Spatially distributed across Kerala study area", fontsize=11)

cmap3 = plt.cm.colors.ListedColormap(["#a5d6a7","#1565c0","#0d47a1"]) if False else None
import matplotlib.colors as mcolors
cmap3 = mcolors.ListedColormap(["#a5d6a7", "#1565c0", "#0d47a1"])
norm3 = mcolors.BoundaryNorm([0,1,2,3], 3)
axes[0].imshow(landcover, cmap=cmap3, norm=norm3, interpolation="nearest", alpha=0.6)
for cid, pc in colours.items():
    sub = obs[obs["class_id"] == cid]
    axes[0].scatter(sub["col"], sub["row"], c=pc, s=12, alpha=0.8,
                     label=f"{labels[cid]} (n={len(sub)})")
axes[0].legend(fontsize=8); axes[0].axis("off")
axes[0].set_title("Points on classification map")

axes[1].violinplot(
    [obs[obs["class_id"] == c]["vh_flood"].values for c in [0,1,2]],
    positions=[0,1,2], showmedians=True
)
axes[1].set_xticks([0,1,2])
axes[1].set_xticklabels(["Dry", "Perm. water", "Flood"])
axes[1].set_ylabel("VH Backscatter — flood scene (dB)")
axes[1].set_title("VH values per class\nShows the classes are separable")
axes[1].grid(True, alpha=0.3, axis="y")

counts = obs.groupby("class_name").size()
axes[2].bar(["Dry land","Perm. water","Flood water"],
             [counts.get("dry_land",0), counts.get("permanent_water",0),
              counts.get("flood_water",0)],
             color=["#a5d6a7","#1565c0","#0d47a1"], alpha=0.9)
axes[2].set_ylabel("Number of points"); axes[2].grid(True,alpha=0.3,axis="y")
axes[2].set_title("Class balance\n(equal per class for fair model training)")

plt.tight_layout()
plt.savefig(dirs["figures"] / "03_observation_points.png", dpi=150, bbox_inches="tight")
plt.show()
plt.close()

# Quick summary table
summary = obs.groupby("class_name").agg(
    Count=("point_id","count"),
    VH_flood_mean=("vh_flood", lambda x: f"{x.mean():.1f} dB"),
    VH_change_mean=("vh_change", lambda x: f"{x.mean():.2f} dB"),
    Lon_range=("longitude", lambda x: f"{x.min():.2f} to {x.max():.2f}"),
).reset_index()
summary.columns = ["Class","Points","Mean VH flood","Mean VH change","Lon range"]
display(summary)

print(f"\nCell 4 done — {len(obs)} observation points saved")
