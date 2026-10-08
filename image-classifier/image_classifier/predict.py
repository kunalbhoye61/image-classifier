"""Run a model on a prepared image."""
from .models import load_model
from .preprocessing import get_image_metadata, load_and_prepare_image


def predict(model, config, img_array, top_k=5):
    processed = config["preprocess"](img_array.copy())
    preds = model.predict(processed, verbose=0)
    return config["decode"](preds, top=top_k)[0]


def analyze(image_path, model_name="mobilenet", top_k=5):
    """Return (original_image, metadata, display_name, predictions)."""
    model, config = load_model(model_name)
    original, array = load_and_prepare_image(image_path, config["input_size"])
    metadata = get_image_metadata(image_path, original)
    predictions = predict(model, config, array, top_k=top_k)
    return original, metadata, config["display_name"], predictions
