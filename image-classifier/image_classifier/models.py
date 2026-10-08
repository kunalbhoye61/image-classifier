"""Model registry and loading. TensorFlow is imported lazily so the rest of the
package (and the tests) work without it."""
import os
import ssl

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

MODEL_KEYS = ("mobilenet", "resnet50", "efficientnet")
_CACHE = {}


def _maybe_disable_ssl_verification():
    """Opt-in workaround for networks that break the weight download.
    Enable with:  ALLOW_INSECURE_SSL=1   (off by default - it weakens security)."""
    if os.getenv("ALLOW_INSECURE_SSL", "0") == "1":
        ssl._create_default_https_context = ssl._create_unverified_context


def _registry():
    from tensorflow.keras.applications import EfficientNetB0, MobileNetV2, ResNet50
    from tensorflow.keras.applications import efficientnet, mobilenet_v2, resnet50

    return {
        "mobilenet": {
            "builder": MobileNetV2, "input_size": (224, 224),
            "preprocess": mobilenet_v2.preprocess_input,
            "decode": mobilenet_v2.decode_predictions,
            "display_name": "MobileNetV2",
        },
        "resnet50": {
            "builder": ResNet50, "input_size": (224, 224),
            "preprocess": resnet50.preprocess_input,
            "decode": resnet50.decode_predictions,
            "display_name": "ResNet50",
        },
        "efficientnet": {
            "builder": EfficientNetB0, "input_size": (224, 224),
            "preprocess": efficientnet.preprocess_input,
            "decode": efficientnet.decode_predictions,
            "display_name": "EfficientNetB0",
        },
    }


def load_model(model_key):
    if model_key not in MODEL_KEYS:
        raise ValueError(f"Unknown model '{model_key}'. Choose from: {list(MODEL_KEYS)}")
    if model_key in _CACHE:
        return _CACHE[model_key]

    _maybe_disable_ssl_verification()
    config = _registry()[model_key]
    print(f"[INFO] Loading {config['display_name']} (pretrained on ImageNet)...")
    model = config["builder"](weights="imagenet")
    print("[INFO] Model loaded.\n")
    _CACHE[model_key] = (model, config)
    return model, config
