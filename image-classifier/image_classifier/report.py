"""Console report and matplotlib visualization."""


def readable(label):
    return label.replace("_", " ").title()


def print_report(metadata, model_name, predictions):
    print("=" * 50)
    print("IMAGE ANALYSIS REPORT")
    print("=" * 50)
    print("\n--- Image Details ---")
    for key, value in metadata.items():
        print(f"{key:<18}: {value}")
    print(f"\n--- Model Used ---\n{model_name}")
    print("\n--- Top Predictions ---")
    for i, (_, label, score) in enumerate(predictions, start=1):
        print(f"{i}. {readable(label):<25} confidence: {score * 100:.2f}%")
    print("=" * 50)


def visualize(original_image, metadata, model_name, predictions, save_path=None, show=True):
    import matplotlib.pyplot as plt

    if not show:
        plt.switch_backend("Agg")

    fig = plt.figure(figsize=(11, 5.5))
    grid = fig.add_gridspec(2, 2, height_ratios=[1, 1])
    ax_img = fig.add_subplot(grid[:, 0])
    ax_text = fig.add_subplot(grid[0, 1])
    ax_bar = fig.add_subplot(grid[1, 1])

    ax_img.imshow(original_image)
    ax_img.axis("off")
    ax_img.set_title("Input Image", fontsize=13, fontweight="bold")

    ax_text.axis("off")
    lines = ["IMAGE DETAILS", "-" * 28]
    lines += [f"{k}: {v}" for k, v in metadata.items()]
    lines += ["", f"MODEL: {model_name}", "-" * 28, "TOP PREDICTIONS"]
    for i, (_, label, score) in enumerate(predictions, start=1):
        lines.append(f"{i}. {readable(label)} - {score * 100:.2f}%")
    ax_text.text(0, 1, "\n".join(lines), fontsize=10, va="top", ha="left",
                 family="monospace", transform=ax_text.transAxes)

    labels = [readable(p[1]) for p in predictions][::-1]
    scores = [p[2] * 100 for p in predictions][::-1]
    ax_bar.barh(labels, scores, color="#4C72B0")
    ax_bar.set_xlim(0, 100)
    ax_bar.set_xlabel("Confidence (%)")
    ax_bar.set_title("Prediction Confidence", fontsize=10)
    for side in ("top", "right"):
        ax_bar.spines[side].set_visible(False)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[INFO] Saved visualization to {save_path}")
    if show:
        plt.show()
    plt.close(fig)
