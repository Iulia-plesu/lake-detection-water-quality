"""Core local water-analysis and visualization utilities.

This module is intentionally UI-agnostic and designed for local batch runs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import rasterio
from PIL import Image
from scipy.ndimage import generic_filter


def load_multispectral_tiff(path: str):
    """Load the first 6 Sentinel-2 bands from a GeoTIFF.

    Expected order: B02, B03, B04, B08, B11, B12.
    """
    with rasterio.open(path) as src:
        bands = src.read([1, 2, 3, 4, 5, 6]).astype(np.float32)

    return bands[0], bands[1], bands[2], bands[3], bands[4], bands[5]


def load_mask(mask_path: str) -> np.ndarray:
    mask = np.array(Image.open(mask_path).convert("L"))
    return (mask > 127).astype(np.uint8)


def compute_indices(B02, B03, B04, B08, B11, B12, mask) -> Dict[str, float]:
    eps = 1e-6
    w = mask == 1

    W2 = B02[w]
    W3 = B03[w]
    W4 = B04[w]
    W8 = B08[w]
    W11 = B11[w]

    if W2.size == 0:
        return {
            "Turbidity": 0.0,
            "Turbidity_std": 0.0,
            "Algae": 0.0,
            "Algae_std": 0.0,
            "Depth": 0.0,
            "Depth_std": 0.0,
            "NDOSI": 0.0,
            "NDOSI_std": 0.0,
            "OilPixelRatio": 0.0,
            "WaterPixels": 0,
            "Area_km2": 0.0,
        }

    turbidity = (W4 + W3) / (W2 + eps)
    algae = (W3 - W4) / (W3 + W4 + eps)
    depth = 1 - (W8 / (W8.max() + eps))
    ndosi = (W11 - (W2 + W3 + W4)) / (W11 + (W2 + W3 + W4) + eps)

    oil_pixels = np.sum(ndosi > 0.20)
    oil_ratio = float(oil_pixels) / float(ndosi.size)

    area_km2 = (W2.size * 100.0) / 1_000_000.0

    return {
        "Turbidity": float(np.nanmean(turbidity)),
        "Turbidity_std": float(np.nanstd(turbidity)),
        "Algae": float(np.nanmean(algae)),
        "Algae_std": float(np.nanstd(algae)),
        "Depth": float(np.nanmean(depth)),
        "Depth_std": float(np.nanstd(depth)),
        "NDOSI": float(np.nanmean(ndosi)),
        "NDOSI_std": float(np.nanstd(ndosi)),
        "OilPixelRatio": float(oil_ratio),
        "WaterPixels": int(W2.size),
        "Area_km2": float(area_km2),
    }


def interpret_results(res: Dict[str, float]) -> str:
    turb = res["Turbidity"]
    turb_std = res["Turbidity_std"]
    algae = res["Algae"]
    algae_std = res["Algae_std"]
    depth = res["Depth"]
    depth_std = res["Depth_std"]
    ndosi = res["NDOSI"]
    ndosi_std = res["NDOSI_std"]
    oil_ratio = res["OilPixelRatio"]
    area = res["Area_km2"]

    lines = ["WATER QUALITY SUMMARY", ""]

    lines.append(f"Turbidity: {turb:.3f} +/- {turb_std:.3f}")
    lines.append(f"Algae proxy: {algae:.3f} +/- {algae_std:.3f}")
    lines.append(f"Depth proxy: {depth:.3f} +/- {depth_std:.3f}")
    lines.append(f"NDOSI: {ndosi:.3f} +/- {ndosi_std:.3f}")
    lines.append(f"Estimated oil-affected ratio: {oil_ratio * 100.0:.2f}%")
    lines.append(f"Detected water area: {area:.4f} km^2")

    lines.append("")
    if ndosi > 0.15 and oil_ratio > 0.01:
        lines.append("Interpretation: potential pollution signal detected.")
    else:
        lines.append("Interpretation: no strong oil-pollution signal detected.")

    return "\n".join(lines)


def _masked(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    out = np.full(values.shape, np.nan, dtype=np.float32)
    out[mask == 1] = values[mask == 1]
    return out


def _save_heatmap(values: np.ndarray, mask: np.ndarray, output_path: str, title: str, cmap: str, vmin=None, vmax=None):
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    masked = _masked(values, mask)
    plt.figure(figsize=(8, 6), dpi=180)
    im = plt.imshow(masked, cmap=cmap, vmin=vmin, vmax=vmax)
    plt.title(title)
    plt.axis("off")
    cbar = plt.colorbar(im, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=8)
    plt.tight_layout()
    plt.savefig(output, bbox_inches="tight")
    plt.close()


def create_ndwi_visualization(B03, B08, mask, output_path):
    eps = 1e-6
    ndwi = (B03 - B08) / (B03 + B08 + eps)
    _save_heatmap(ndwi, mask, output_path, "NDWI", "Blues", vmin=-0.5, vmax=1.0)


def create_mndwi_visualization(B03, B11, mask, output_path):
    eps = 1e-6
    mndwi = (B03 - B11) / (B03 + B11 + eps)
    _save_heatmap(mndwi, mask, output_path, "MNDWI", "Blues", vmin=-0.5, vmax=1.0)


def create_turbidity_visualization(B02, B03, B04, mask, output_path):
    eps = 1e-6
    turbidity = (B04 + B03) / (B02 + eps)
    _save_heatmap(turbidity, mask, output_path, "Turbidity Proxy", "YlOrBr")


def create_algae_visualization(B03, B04, mask, output_path):
    eps = 1e-6
    algae = (B03 - B04) / (B03 + B04 + eps)
    _save_heatmap(algae, mask, output_path, "Algae Proxy", "RdYlGn_r", vmin=-1.0, vmax=1.0)


def create_depth_visualization(B08, mask, output_path):
    eps = 1e-6
    depth = 1 - (B08 / (np.nanmax(B08) + eps))
    _save_heatmap(depth, mask, output_path, "Spectral Depth Proxy", "Blues", vmin=0.0, vmax=1.0)


def create_ndosi_visualization(B02, B03, B04, B11, mask, output_path):
    eps = 1e-6
    ndosi = (B11 - (B02 + B03 + B04)) / (B11 + (B02 + B03 + B04) + eps)
    _save_heatmap(ndosi, mask, output_path, "NDOSI", "RdBu_r", vmin=-1.0, vmax=1.0)


def create_oil_ratio_visualization(B02, B03, B04, B11, mask, output_path):
    eps = 1e-6
    ndosi = (B11 - (B02 + B03 + B04)) / (B11 + (B02 + B03 + B04) + eps)
    oil = np.zeros_like(ndosi, dtype=np.float32)
    oil[(mask == 1) & (ndosi > 0.20)] = 1.0
    _save_heatmap(oil, mask, output_path, "Oil Alert Map", "Reds", vmin=0.0, vmax=1.0)


def create_all_visualizations(B02, B03, B04, B08, B11, B12, mask, output_dir="outputs_visualizations"):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    create_turbidity_visualization(B02, B03, B04, mask, str(out / "turbidity_map.png"))
    create_algae_visualization(B03, B04, mask, str(out / "algae_map.png"))
    create_depth_visualization(B08, mask, str(out / "depth_map.png"))
    create_ndosi_visualization(B02, B03, B04, B11, mask, str(out / "ndosi_map.png"))
    create_oil_ratio_visualization(B02, B03, B04, B11, mask, str(out / "oil_ratio_map.png"))


def _local_std(values: np.ndarray, size: int = 5) -> np.ndarray:
    def std_fn(window):
        return float(np.nanstd(window))

    return generic_filter(values, std_fn, size=size, mode="nearest")


def create_variance_visualizations(B02, B03, B04, B08, B11, B12, mask, output_dir="outputs_visualizations"):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    eps = 1e-6
    turbidity = (B04 + B03) / (B02 + eps)
    algae = (B03 - B04) / (B03 + B04 + eps)
    depth = 1 - (B08 / (np.nanmax(B08) + eps))
    ndosi = (B11 - (B02 + B03 + B04)) / (B11 + (B02 + B03 + B04) + eps)

    _save_heatmap(_local_std(turbidity), mask, str(out / "turbidity_variance.png"), "Turbidity Local Std", "viridis")
    _save_heatmap(_local_std(algae), mask, str(out / "algae_variance.png"), "Algae Local Std", "viridis")
    _save_heatmap(_local_std(depth), mask, str(out / "depth_variance.png"), "Depth Local Std", "viridis")
    _save_heatmap(_local_std(ndosi), mask, str(out / "ndosi_variance.png"), "NDOSI Local Std", "viridis")
