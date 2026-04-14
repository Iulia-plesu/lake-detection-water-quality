from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Tuple

from deeplab_predict import predict_deeplab
from unet_predict import predict_unet
from water_analysis import (
    compute_indices,
    interpret_results,
    load_mask,
    load_multispectral_tiff,
)


def _resolve_path(path_str: str) -> Path:
    path = Path(path_str).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Path does not exist: {path}")
    return path


def _run_single_model(
    model_name: str,
    model_path: Path,
    tif_path: Path,
    output_root: Path,
) -> Dict[str, str]:
    model_name = model_name.lower().strip()
    model_out_dir = output_root / model_name
    model_out_dir.mkdir(parents=True, exist_ok=True)

    if model_name == "unet":
        mask_path, overlay_path = predict_unet(str(model_path), str(tif_path), save_dir=model_out_dir)
    elif model_name == "deeplab":
        mask_path, overlay_path = predict_deeplab(str(model_path), str(tif_path), out_dir=model_out_dir)
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    b02, b03, b04, b08, b11, b12 = load_multispectral_tiff(str(tif_path))
    mask = load_mask(mask_path)

    metrics = compute_indices(b02, b03, b04, b08, b11, b12, mask)
    report_txt = interpret_results(metrics)
    report_path = model_out_dir / "report.txt"
    report_path.write_text(report_txt + "\n", encoding="utf-8")

    return {
        "model": model_name,
        "mask": str(Path(mask_path).resolve()),
        "overlay": str(Path(overlay_path).resolve()),
        "report": str(report_path.resolve()),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Local segmentation + report pipeline for Sentinel-2 6-band TIFF data."
    )
    parser.add_argument("--tif", required=True, help="Path to input 6-band GeoTIFF.")
    parser.add_argument(
        "--model",
        default="both",
        choices=["unet", "deeplab", "both"],
        help="Model to run.",
    )
    parser.add_argument(
        "--unet-model",
        default="models/best_water_segmentation_model_unet.pth",
        help="Path to UNet .pth weights.",
    )
    parser.add_argument(
        "--deeplab-model",
        default="models/best_water_segmentation_model_deeplabv3.pth",
        help="Path to DeepLabV3 .pth weights.",
    )
    parser.add_argument(
        "--output-root",
        default="outputs_local",
        help="Root folder for local pipeline outputs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tif_path = _resolve_path(args.tif)
    output_root = Path(args.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    run_models = ["unet", "deeplab"] if args.model == "both" else [args.model]

    results = []
    for model_name in run_models:
        model_path = _resolve_path(args.unet_model if model_name == "unet" else args.deeplab_model)
        result = _run_single_model(
            model_name=model_name,
            model_path=model_path,
            tif_path=tif_path,
            output_root=output_root,
        )
        results.append(result)

    print("\n=== LOCAL PIPELINE COMPLETED ===")
    for result in results:
        print(f"Model: {result['model']}")
        print(f"  Mask: {result['mask']}")
        print(f"  Overlay: {result['overlay']}")
        print(f"  Report: {result['report']}")


if __name__ == "__main__":
    main()
