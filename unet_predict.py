import torch
import torch.nn as nn
import numpy as np
import rasterio
from PIL import Image
import matplotlib.pyplot as plt
import shutil
from pathlib import Path
from datetime import datetime
from threading import Lock
from models import UNet

BASE_DIR = Path(__file__).resolve().parent
_MODEL_CACHE_LOCK = Lock()
_UNET_MODEL_CACHE = {}


def _get_cached_unet_model(model_path, device):
    cache_key = (str(Path(model_path).resolve()), device)
    cached_model = _UNET_MODEL_CACHE.get(cache_key)
    if cached_model is not None:
        return cached_model

    with _MODEL_CACHE_LOCK:
        cached_model = _UNET_MODEL_CACHE.get(cache_key)
        if cached_model is not None:
            return cached_model

        model = UNet(in_channels=6, num_classes=2)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        _UNET_MODEL_CACHE[cache_key] = model
        return model


def warm_unet_model(model_path, device=None):
    resolved_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    _get_cached_unet_model(model_path, resolved_device)


def load_and_normalize_s2(path):
    with rasterio.open(path) as src:
        data = src.read([1, 2, 3, 4, 5, 6]).astype(np.float32)

    data = np.moveaxis(data, 0, -1)
    out = np.zeros_like(data)

    for i in range(6):
        b = data[:, :, i]
        mn, mx = b.min(), b.max()
        out[:, :, i] = (b - mn) / (mx - mn) if mx > mn else 0

    return torch.from_numpy(out).permute(2, 0, 1).unsqueeze(0).float()


def pad_to_16(tensor):
    _, _, H, W = tensor.shape
    ph = (16 - H % 16) % 16
    pw = (16 - W % 16) % 16
    padded = nn.functional.pad(tensor, (0, pw, 0, ph), mode="reflect")
    return padded, ph, pw


def predict_unet(model_path, tif_path, save_dir=None):
    if save_dir is None:
        save_dir = BASE_DIR / "outputs_unet"
    else:
        save_dir = Path(save_dir)

    save_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = _get_cached_unet_model(model_path, device)

    img = load_and_normalize_s2(tif_path)
    img, ph, pw = pad_to_16(img)
    img = img.to(device)

    with torch.inference_mode():
        logits = model(img)
        prob = torch.softmax(logits, dim=1)[0, 1].cpu().numpy()

    mask = prob > 0.5

    if ph: mask = mask[:-ph, :]
    if pw: mask = mask[:, :-pw]

    mask_path = save_dir / "unet_mask.png"
    Image.fromarray((mask * 255).astype(np.uint8)).save(str(mask_path))

    rgb = img[0, :3].cpu().numpy()
    rgb = np.moveaxis(rgb, 0, 2)
    rgb = rgb[:mask.shape[0], :mask.shape[1], :]

    overlay = rgb.copy()
    overlay[mask] = [0, 0, 1]

    overlay_path = save_dir / "unet_overlay.png"
    plt.imsave(str(overlay_path), overlay)

    try:
        date_tag = Path(tif_path).stem.split("_")[-1] or datetime.now().date().isoformat()
    except Exception:
        date_tag = datetime.now().date().isoformat()

    archive_dir = BASE_DIR / "outputs_unet"
    archive_dir.mkdir(parents=True, exist_ok=True)
    try:
        shutil.copyfile(str(mask_path), str(archive_dir / f"unet_mask_{date_tag}.png"))
        shutil.copyfile(str(overlay_path), str(archive_dir / f"unet_overlay_{date_tag}.png"))
    except Exception:
        pass

    return str(mask_path), str(overlay_path)