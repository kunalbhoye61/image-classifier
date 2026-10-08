import numpy as np
import pytest
from PIL import Image

from image_classifier.preprocessing import get_image_metadata, load_and_prepare_image
from image_classifier.report import readable


@pytest.fixture
def sample_image(tmp_path):
    path = tmp_path / "sample.jpg"
    Image.new("RGB", (300, 200), color=(120, 80, 40)).save(path)
    return str(path)


def test_load_and_prepare_shape(sample_image):
    original, array = load_and_prepare_image(sample_image, (224, 224))
    assert original.size == (300, 200)
    assert array.shape == (1, 224, 224, 3)
    assert array.dtype == np.float32


def test_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        load_and_prepare_image("does_not_exist.jpg")


def test_invalid_image_raises(tmp_path):
    bad = tmp_path / "bad.jpg"
    bad.write_text("not an image")
    with pytest.raises(ValueError):
        load_and_prepare_image(str(bad))


def test_metadata(sample_image):
    original, _ = load_and_prepare_image(sample_image)
    meta = get_image_metadata(sample_image, original)
    assert meta["File name"] == "sample.jpg"
    assert meta["Dimensions"] == "300 x 200 px"
    assert meta["Format (on disk)"] == "JPG"


def test_readable_label():
    assert readable("golden_retriever") == "Golden Retriever"
