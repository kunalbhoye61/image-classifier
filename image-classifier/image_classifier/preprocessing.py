"""Image loading and metadata (no TensorFlow needed)."""
import os

import numpy as np
from PIL import Image


def load_and_prepare_image(image_path, target_size=(224, 224)):
    """Return (original PIL image, float32 batch array of shape (1, H, W, 3))."""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: '{image_path}'.")
    try:
        original = Image.open(image_path).convert("RGB")
    except Exception as exc:
        raise ValueError(f"'{image_path}' exists but is not a valid image file.") from exc

    resized = original.resize(target_size)
    array = np.array(resized).astype(np.float32)
    return original, np.expand_dims(array, axis=0)


def get_image_metadata(image_path, original_image):
    size_kb = os.path.getsize(image_path) / 1024
    return {
        "File name": os.path.basename(image_path),
        "File size": f"{size_kb:.1f} KB",
        "Dimensions": f"{original_image.width} x {original_image.height} px",
        "Mode": original_image.mode,
        "Format (on disk)": os.path.splitext(image_path)[1].upper().lstrip("."),
    }
