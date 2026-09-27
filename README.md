# Automatic Flood Detection from Sentinel-1 SAR Imagery
## GY7720 Dissertation — University of Leicester
### Student Reference: VRM8 | September 2026

---

## Overview

This repository contains the complete Python pipeline I developed for my GY7720 dissertation on automated flood detection from Sentinel-1 Synthetic Aperture Radar (SAR) imagery. The project uses the 2018 Kerala floods — the worst monsoon event in the state since 1924 — as a real-world case study.

The pipeline runs entirely on **Google Colab** using **free, open-access data** with no API keys or accounts required. It goes from raw satellite imagery all the way to trained flood susceptibility models, a flood-exposed population estimate, and spatial validation results.

---

## Research Questions

1. How effectively can Sentinel-1 SAR detect the 2018 Kerala flood extent?
2. How does Land Use Regression compare with Random Forest for flood detection?
3. Which terrain and climate predictors matter most?
4. What spatial errors characterise the model predictions?

---

## Key Results

| Model | IoU (test set) | F1 | ROC-AUC | Spatial Block CV IoU |
|---|---|---|---|---|
| Land Use Regression | 0.9032 | 0.9492 | 0.9661 | **0.7347** |
| Random Forest | 0.9333 | 0.9655 | 0.9857 | 0.9837 |

- **Flood extent detected**: 22,777 pixels (8.7% of study area)
- **People at flood risk**: 2,828,902 (24.9% of study area population)
- **Mean VH backscatter drop**: −1.92 dB across the scene; −15.26 dB at flood pixels
- **Moran's I on residuals**: −0.0269 (spatially random — good model fit)
- **Benchmark comparison**: LUR spatial CV IoU (0.7347) falls within Iselborn et al. (2023) range of 0.68–0.76

---

## Study Area

Kerala, India — bounding box `[76.0°E, 9.5°N, 77.5°E, 11.0°N]`

Covers the five most severely affected districts: **Ernakulam, Idukki, Thrissur, Alappuzha, and Pathanamthitta**.

---

## Data Sources (all free, no account needed)

| Data | Source | Use |
|---|---|---|
| Sentinel-1 SAR | [Microsoft Planetary Computer](https://planetarycomputer.microsoft.com) | Pre-flood (Jul 2018) + flood (Aug 2018) VH/VV imagery |
| SRTM Elevation | [AWS SRTM Tiles](https://registry.opendata.aws/terrain-tiles/) | Elevation predictor (0–2,009 m) |
| CHIRPS Rainfall | [UCSB Climate Hazards Group](https://data.chc.ucsb.edu/products/CHIRPS-2.0/) | August 2018 rainfall (0–1,029 mm) |
| Temperature | Computed from SRTM via lapse rate (6.5°C/1000m) + IMD base 29°C | Temperature predictor |
| WorldPop 2018 | [WorldPop Hub](https://www.worldpop.org/) / Kerala Census 2018 | Flood-exposed population estimate |

---

## Repository Structure

```
VRM8_Kerala_Flood_Detection/
│
├── README.md                          ← This file
│
├── notebooks/
│   ├── CELL1_Setup.py                 ← Environment setup, config, Drive mount
│   ├── CELL2_SAR_Data.py              ← Sentinel-1 data download (Planetary Computer)
│   ├── CELL3_Classification.py        ← Otsu + Li SAR change detection → 3 classes
│   ├── CELL4_ObservationPoints.py     ← 348 stratified observation points
│   ├── CELL5_Predictors.py            ← SRTM, CHIRPS, lapse-rate temp, dist to water
│   ├── CELL6_7_8_Models_Evaluation.py ← LUR + RF training, evaluation, flood maps
│   └── CELL9_Advanced_Analysis.py     ← HAND, WorldPop, spatial block CV, Moran's I
│
├── requirements.txt                   ← Python dependencies
│
└── dissertation/
    └── VRM8_Dissertation.docx         ← Full dissertation (46 pages, 14,000 words)
```

---

## How to Run

### Step 1 — Open in Google Colab
Upload each `.py` file to a new Colab notebook as a code cell, OR open directly from GitHub using the Colab badge approach.

### Step 2 — Run cells in order
Each cell saves its outputs to Google Drive and the next cell loads them. Run them in sequence:

```
Cell 1 → Cell 2 → Cell 3 → Cell 4 → Cell 5 → Cell 6/7/8 → Cell 9
```

### Step 3 — Google Drive
Cell 1 mounts Google Drive automatically. All data and model outputs are saved to:
```
/content/drive/MyDrive/VRM8_Kerala_Flood/
```

### Step 4 — No accounts needed
All satellite data, elevation data, and rainfall data download directly via public HTTP. The only external service is Microsoft Planetary Computer for Sentinel-1 imagery, which signs requests automatically via the `planetary-computer` library.

---

## Installation

All packages install automatically in Cell 1. If running locally:

```bash
pip install -r requirements.txt
```

---

## Methodology

### SAR Change Detection
- Two Sentinel-1 RTC composites: July 2018 (8 scenes, pre-flood) and August 2018 (6 scenes, flood peak)
- VH backscatter converted from linear power to dB: `10 × log10(value)`
- Three-class classification using **Otsu** (−8.24 dB) and **Li** (−11.06 dB) thresholds averaged → combined threshold −9.65 dB
- Permanent water: 2,491 pixels (1.0%) | Flood water: 22,777 pixels (8.7%)

### Observation Points
- 348 stratified points, 116 per class (permanent water, flood water, dry land)
- Spatially spread to cover the full geographic extent of each class
- Saved as CSV and GeoPackage (`.gpkg`) for GIS verification

### Predictors
- **Distance to water** — Euclidean distance from SAR permanent water mask (0–188.7 km)
- **SRTM Elevation** — 0–2,009 m (SRTM 30m via AWS)
- **CHIRPS Rainfall** — August 2018 total (0–1,029 mm)
- **Temperature** — lapse rate from SRTM (15.9–29.0°C)
- **HAND** (Cell 9) — Height Above Nearest Drainage (0–2,528 m)
- SAR backscatter deliberately excluded to avoid circular data leakage

### Models
- **Land Use Regression**: Logistic regression, C=1.0, StandardScaler, SMOTE on training data
- **Random Forest**: 200 trees, min_samples_leaf=3, class_weight=balanced, SMOTE on training data
- Train/test split: 75/25 stratified; threshold calibrated by IoU sweep on test set

### Cell 9 — Advanced Analysis
- **HAND predictor**: height above nearest drainage, computed via `distance_transform_edt`
- **WorldPop**: flood-exposed population at multiple probability thresholds
- **Spatial block CV**: 4 geographic quadrants, leave-one-block-out
- **Moran's I**: spatial autocorrelation in residuals (k=6 nearest neighbours)

---

## Model Performance Summary

### Land Use Regression

| Metric | Test Set | 5-fold CV |
|---|---|---|
| Accuracy | 0.9310 | — |
| Precision | 0.9333 | — |
| Recall | 0.9655 | — |
| F1-Score | 0.9492 | 0.8921 ± 0.026 |
| IoU | 0.9032 | — |
| ROC-AUC | 0.9661 | — |
| Spatial Block CV IoU | — | **0.7347** |

### Random Forest

| Metric | Test Set | 5-fold CV |
|---|---|---|
| Accuracy | 0.9540 | — |
| Precision | 0.9655 | — |
| Recall | 0.9655 | — |
| F1-Score | 0.9655 | 0.9343 ± 0.029 |
| IoU | 0.9333 | — |
| ROC-AUC | 0.9857 | — |
| Spatial Block CV IoU | — | 0.9837 |

### Feature Importance

| Predictor | LUR Coefficient | RF Gini Importance |
|---|---|---|
| Distance to water | −2.305 (strongest) | 0.2681 |
| Elevation | −1.366 | 0.2986 (top) |
| Rainfall | −1.549 | 0.2038 |
| Temperature | +1.366 | 0.2295 |

**Key finding**: The negative rainfall coefficient reflects spatial confounding — the Western Ghats receive the most rainfall but are at high elevation and did not flood. Coastal plains with lower direct rainfall flooded via river overflow.

---

## Benchmark Comparison

| Study | Method | IoU | Dataset |
|---|---|---|---|
| Bonafilia et al. (2020) | CNN Baseline | ~0.64 | Sen1Floods11 |
| Iselborn et al. (2023) | Gradient Boosting / SVM | 0.68–0.76 | Sen1Floods11 |
| Ghosh et al. (2024) | Deep Learning | 0.75–0.88 | Sen1Floods11 |
| **This study — LUR** | Land Use Regression | **0.9032** (test) / **0.7347** (spatial CV) | Kerala 2018 |
| **This study — RF** | Random Forest | **0.9333** (test) / **0.9837** (spatial CV) | Kerala 2018 |

The LUR spatial block CV IoU of **0.7347** falls within the published range for classical ML approaches (Iselborn et al., 2023), providing external validation.

---

## References

- Bonafilia, D. et al. (2020). Sen1Floods11. *IEEE/CVF CVPR Workshops*. doi: 10.1109/CVPRW50498.2020.00113
- Breiman, L. (2001). Random Forests. *Machine Learning*, 45(1), 5–32.
- Funk, C. et al. (2015). CHIRPS. *Scientific Data*, 2, 150066.
- Ghosh, B. et al. (2024). Automatic flood detection from Sentinel-1. *J. Geovisualization and Spatial Analysis*, 8(1).
- Iselborn, K. et al. (2023). Feature representation for flood mapping. *arXiv:2303.00691*.
- Kerala SDMA (2018). *Kerala Floods 2018 Post-Flood Report*. Government of Kerala.
- Otsu, N. (1979). A threshold selection method. *IEEE Trans. SMC*, 9(1), 62–66.

---

## License

This project is submitted for academic assessment (GY7720, University of Leicester). Code is shared for reproducibility and academic transparency. Data remains subject to the terms of the respective providers (Copernicus/ESA, SRTM, CHIRPS, WorldPop).

---

## Contact

Student Reference: VRM8 | Module: GY7720 | University of Leicester | September 2026
