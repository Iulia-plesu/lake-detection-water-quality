import torch
import numpy as np
import rasterio
from PIL import Image
import matplotlib.pyplot as plt
import shutil
from pathlib import Path
from datetime import datetime
from threading import Lock
from models import MultiSpectralDeepLabV3

BASE_DIR = Path(__file__).resolve().parent
_MODEL_CACHE_LOCK = Lock()
_DEEPLAB_MODEL_CACHE = {}


def _get_cached_deeplab_model(model_path, device):
    cache_key = (str(Path(model_path).resolve()), device)
    cached_model = _DEEPLAB_MODEL_CACHE.get(cache_key)
    if cached_model is not None:
        return cached_model

    with _MODEL_CACHE_LOCK:
        cached_model = _DEEPLAB_MODEL_CACHE.get(cache_key)
        if cached_model is not None:
            return cached_model

        model = MultiSpectralDeepLabV3()
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        _DEEPLAB_MODEL_CACHE[cache_key] = model
        return model


def warm_deeplab_model(model_path, device=None):
    resolved_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    _get_cached_deeplab_model(model_path, resolved_device)


def predict_deeplab(model_path, image_path, out_dir=None):
    if out_dir is None:
        out_dir = BASE_DIR / "outputs_deeplab"
    else:
        out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = _get_cached_deeplab_model(model_path, device)

    with rasterio.open(image_path) as src:
        img = src.read().astype(np.float32)

    img = np.moveaxis(img, 0, -1)
    img = (img - img.min()) / (img.max() - img.min() + 1e-6)

    tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)

    if tensor.shape[1] > 6:
        tensor = tensor[:, :6, :, :]

    tensor = tensor.to(device)

    with torch.inference_mode():
        logits = model(tensor)
        mask = (torch.softmax(logits, 1)[0, 1] > 0.5).cpu().numpy()

    mask_path = out_dir / "deeplab_mask.png"
    Image.fromarray((mask * 255).astype(np.uint8)).save(str(mask_path))

    rgb = img[:, :, :3]
    overlay = rgb.copy()
    overlay[:, :, 2] = np.maximum(overlay[:, :, 2], mask)

    overlay_path = out_dir / "deeplab_overlay.png"
    plt.imsave(str(overlay_path), overlay)

    try:
        date_tag = Path(image_path).stem.split("_")[-1] or datetime.now().date().isoformat()
    except Exception:
        date_tag = datetime.now().date().isoformat()

    archive_dir = BASE_DIR / "outputs_deeplab"
    archive_dir.mkdir(parents=True, exist_ok=True)
    try:
        shutil.copyfile(str(mask_path), str(archive_dir / f"deeplab_mask_{date_tag}.png"))
        shutil.copyfile(str(overlay_path), str(archive_dir / f"deeplab_overlay_{date_tag}.png"))
    except Exception:
        pass

    return str(mask_path), str(overlay_path)