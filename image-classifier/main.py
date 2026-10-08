"""CLI: python main.py path/to/image.jpg [--model mobilenet] [--top-k 5] [--save out.png] [--no-show]"""
import argparse
import sys

from image_classifier import MODEL_NAMES
from image_classifier.predict import analyze
from image_classifier.report import print_report, visualize


def main():
    parser = argparse.ArgumentParser(description="Classify an image with a pretrained ImageNet model.")
    parser.add_argument("image", help="Path to the image file (jpg, png, jfif, ...)")
    parser.add_argument("--model", choices=MODEL_NAMES, default="mobilenet")
    parser.add_argument("--top-k", type=int, default=5, help="Number of predictions to show")
    parser.add_argument("--save", metavar="PATH", help="Save the visualization to this file")
    parser.add_argument("--no-show", action="store_true", help="Do not open a plot window")
    args = parser.parse_args()

    try:
        original, metadata, name, predictions = analyze(args.image, args.model, args.top_k)
    except (FileNotFoundError, ValueError) as err:
        sys.exit(f"[ERROR] {err}")

    print_report(metadata, name, predictions)
    visualize(original, metadata, name, predictions, save_path=args.save, show=not args.no_show)


if __name__ == "__main__":
    main()
