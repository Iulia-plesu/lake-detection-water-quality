# CopernicusDataFinder

Companion code package for the manuscript:

"Lake Detection and Water Quality Estimation in Sentinel-2 Data"

This repository provides a local, script-based pipeline for:

- water-body segmentation (U-Net and/or DeepLabV3),
- mask-constrained water-quality metric computation,
- reproducible export of masks, overlays, and text reports.

## 1. What This Package Produces

Given a 6-band Sentinel-2 style GeoTIFF (B02, B03, B04, B08, B11, B12), the pipeline:

1. runs segmentation inference,
2. exports a binary mask and overlay image,
3. computes summary water-quality indicators on water pixels,
4. writes a readable report.

Current default CLI behavior is report-oriented and does not generate map galleries.

## 2. Repository Contents (Core Files)

- `pipeline.py`: main CLI entrypoint for local runs
- `unet_predict.py`: U-Net inference
- `deeplab_predict.py`: DeepLabV3 inference
- `water_analysis.py`: index computation and report interpretation
- `models.py`: model definitions
- `models/`: trained model weights (`.pth`, tracked with Git LFS)
- `requirements.txt`: Python dependencies

### Model Files (Git LFS)

Model weight files in `models/` are stored with Git LFS.

After cloning, run:

```bash
git lfs install
git lfs pull
```

## 3. Environment Setup

Recommended Python version: 3.12 (64-bit).

Python 3.13 is not recommended for this pinned dependency set.

### Windows (PowerShell)

```powershell
py -3.12 -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

### Linux/macOS

```bash
python3.12 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

## 4. How to Run

Run both models:

```bash
python pipeline.py --tif path/to/input.tif --model both
```

Run one model only:

```bash
python pipeline.py --tif path/to/input.tif --model unet
python pipeline.py --tif path/to/input.tif --model deeplab
```

Run with explicit checkpoints and output folder:

```bash
python pipeline.py \
   --tif path/to/input.tif \
   --model both \
   --unet-model models/best_water_segmentation_model_unet.pth \
   --deeplab-model models/best_water_segmentation_model_deeplabv3.pth \
   --output-root outputs_local
```

## 5. Output Structure

Outputs are written under `outputs_local/`.

For each executed model (`unet`, `deeplab`), expected artifacts are:

- `<model>_mask.png`
- `<model>_overlay.png`
- `report.txt`


## 6. Metrics Included in Report

The report summarizes:

- Turbidity proxy (mean and std)
- Algae proxy (mean and std)
- Spectral depth proxy (mean and std)
- NDOSI (mean and std)
- Estimated oil-affected pixel ratio
- Detected water area (km^2)

## 7. Troubleshooting

- If `rasterio` fails on Windows, use a wheel compatible with your Python version.
- If CUDA is unavailable, inference runs on CPU.
- If checkpoint paths are wrong, the pipeline raises `FileNotFoundError`.
- If input TIFF has fewer than 6 bands, preprocessing fails.

## 8. Citation

If you reference this software, cite it as the companion code package of:

"Lake Detection and Water Quality Estimation in Sentinel-2 Data".
