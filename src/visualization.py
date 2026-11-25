import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np


CLASS_COLORS = {
    "liver": (0.15, 0.80, 0.45),
    "liver_tumor": (0.95, 0.20, 0.20),
    "kidney": (0.20, 0.65, 0.95),
    "kidney_tumor": (1.00, 0.72, 0.10),
    "tumor": (0.95, 0.20, 0.20),
}

FALLBACK_COLORS = [
    (0.0, 0.0, 0.0),
    (0.15, 0.80, 0.45),
    (0.95, 0.20, 0.20),
    (0.20, 0.65, 0.95),
    (1.00, 0.72, 0.10),
]


def _get_palette(num_classes, class_names=None):
    if class_names is None:
        return [FALLBACK_COLORS[i % len(FALLBACK_COLORS)] for i in range(num_classes)]
    return [CLASS_COLORS.get(name, FALLBACK_COLORS[i % len(FALLBACK_COLORS)]) for i, name in enumerate(class_names)]


def colorize_mask(mask, num_classes=3, class_names=None):
    rgb = np.zeros((*mask.shape, 3), dtype=np.float32)
    palette = _get_palette(num_classes, class_names)
    for cls in range(1, num_classes):
        rgb[mask == cls] = palette[cls]
    return rgb


def overlay_mask(image, mask, num_classes=3, alpha=0.48, class_names=None):
    gray = np.clip(image.astype(np.float32), 0.0, 1.0)
    base = np.stack([gray, gray, gray], axis=-1)
    color = colorize_mask(mask, num_classes=num_classes, class_names=class_names)
    fg = (mask > 0)[..., None]
    return np.clip(np.where(fg, (1.0 - alpha) * base + alpha * color, base), 0.0, 1.0)


def _add_legend(fig, class_names, num_classes):
    names = class_names or [f"class_{i}" for i in range(num_classes)]
    palette = _get_palette(num_classes, names)
    handles = [
        mpatches.Patch(color=palette[i], label=names[i].replace("_", " ").title())
        for i in range(1, num_classes)
    ]
    if handles:
        fig.legend(handles=handles, loc="lower center", ncol=len(handles), frameon=True, fontsize=9)


def plot_training_history(history, out_path):
    epochs = [h["epoch"] for h in history]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4), dpi=140)

    axes[0].plot(epochs, [h["train_loss"] for h in history], label="Train Loss", linewidth=2)
    axes[0].plot(epochs, [h["val_loss"] for h in history], label="Val Loss", linewidth=2)
    axes[0].set_title("Loss Curve")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Dice + CE Loss")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    axes[1].plot(epochs, [h["tumor_dice"] for h in history], label="Mean Tumor Dice", linewidth=2)
    axes[1].plot(epochs, [h["mean_fg_dice"] for h in history], label="Mean FG Dice", linewidth=2, linestyle="--")
    if "liver_tumor_dice" in history[0]:
        axes[1].plot(epochs, [h["liver_tumor_dice"] for h in history], label="Liver Tumor Dice", alpha=0.85)
    if "kidney_tumor_dice" in history[0]:
        axes[1].plot(epochs, [h["kidney_tumor_dice"] for h in history], label="Kidney Tumor Dice", alpha=0.85)
    axes[1].set_title("Validation Dice")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Dice")
    axes[1].set_ylim(0.0, 1.02)
    axes[1].grid(alpha=0.3)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_prediction_grid(images, gt_masks, pred_masks, out_path, num_classes=3, model_label="Prediction", class_names=None):
    n = len(images)
    if n == 0:
        return
    fig, axes = plt.subplots(n, 3, figsize=(10, 3.2 * n + 0.5), dpi=130)
    if n == 1:
        axes = np.expand_dims(axes, axis=0)

    titles = ["CT Slice", "Ground Truth", model_label]
    for r in range(n):
        panels = [
            images[r],
            overlay_mask(images[r], gt_masks[r], num_classes=num_classes, class_names=class_names),
            overlay_mask(images[r], pred_masks[r], num_classes=num_classes, class_names=class_names),
        ]
        for c, panel in enumerate(panels):
            ax = axes[r, c]
            ax.imshow(panel, cmap="gray", vmin=0.0, vmax=1.0) if c == 0 else ax.imshow(panel)
            if r == 0:
                ax.set_title(titles[c])
            ax.axis("off")

    _add_legend(fig, class_names, num_classes)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(out_path)
    plt.close(fig)


def plot_model_comparison(images, gt_masks, model_preds, out_path, num_classes=3, class_names=None):
    n_rows = len(images)
    if n_rows == 0:
        return
    names = list(model_preds.keys())
    n_cols = 2 + len(names)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.2 * n_cols, 3.2 * n_rows + 0.5), dpi=135)
    if n_rows == 1:
        axes = np.expand_dims(axes, axis=0)

    headers = ["CT Slice", "Ground Truth"] + [m.upper() for m in names]
    for r in range(n_rows):
        panels = [
            images[r],
            overlay_mask(images[r], gt_masks[r], num_classes=num_classes, class_names=class_names),
        ] + [
            overlay_mask(images[r], model_preds[m][r], num_classes=num_classes, class_names=class_names)
            for m in names
        ]
        for c, panel in enumerate(panels):
            ax = axes[r, c]
            ax.imshow(panel, cmap="gray", vmin=0.0, vmax=1.0) if c == 0 else ax.imshow(panel)
            if r == 0:
                ax.set_title(headers[c])
            ax.axis("off")

    _add_legend(fig, class_names, num_classes)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(out_path)
    plt.close(fig)
