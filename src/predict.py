import argparse
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import torch
from PIL import Image

from src.dataset import _infer_axial_axis, get_class_names, window_ct
from src.evaluate import load_model_from_checkpoint
from src.visualization import overlay_mask


def predict_slice(model, slice_norm, img_size, device):
    h, w = slice_norm.shape
    resized = cv2.resize(slice_norm.astype(np.float32), (img_size, img_size), interpolation=cv2.INTER_LINEAR)
    x = torch.from_numpy(resized).unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        pred = model(x).argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
    if (h, w) != (img_size, img_size):
        pred = cv2.resize(pred, (w, h), interpolation=cv2.INTER_NEAREST)
    return pred


def save_overlay_figure(image, pred_mask, out_path, num_classes, class_names, title):
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6), dpi=140)
    axes[0].imshow(image, cmap="gray", vmin=0.0, vmax=1.0)
    axes[0].set_title("CT Input")
    axes[1].imshow(pred_mask, cmap="viridis", vmin=0, vmax=max(num_classes - 1, 1))
    axes[1].set_title("Predicted Mask")
    axes[2].imshow(overlay_mask(image, pred_mask, num_classes=num_classes, class_names=class_names))
    axes[2].set_title(title)
    for ax in axes:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def parse_args():
    p = argparse.ArgumentParser(description="Run Liver & Kidney Tumor Segmentation Inference")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--input", required=True, help="2D slice (png/jpg/npy) or 3D NIfTI (.nii/.nii.gz)")
    p.add_argument("--out-dir", default="outputs/preds")
    p.add_argument("--hu-min", type=float, default=-100.0)
    p.add_argument("--hu-max", type=float, default=200.0)
    return p.parse_args()


def main():
    args = parse_args()
    in_path = Path(args.input)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, ckpt = load_model_from_checkpoint(args.checkpoint, device)
    img_size = int(ckpt.get("img_size", 256))
    num_classes = int(ckpt.get("num_classes", 3))
    organ = str(ckpt.get("organ", "both"))
    task = str(ckpt.get("task", "multiclass"))
    class_names = list(ckpt.get("class_names", get_class_names(organ, task)))
    model_name = str(ckpt.get("model_name", "model")).upper()
    tumor_ids = [i for i, c in enumerate(class_names) if "tumor" in c] or [num_classes - 1]

    if in_path.name.lower().endswith((".nii", ".nii.gz")):
        nii = nib.load(str(in_path))
        vol = nii.get_fdata(dtype=np.float32)
        axis = _infer_axial_axis(vol.shape, "kits" if organ == "kidney" else "lits")
        pred_vol = np.zeros(vol.shape, dtype=np.uint8)

        best_z, best_px, best_img = vol.shape[axis] // 2, -1, None
        for z in range(vol.shape[axis]):
            sl_norm = window_ct(np.take(vol, z, axis=axis), args.hu_min, args.hu_max)
            pred_sl = predict_slice(model, sl_norm, img_size, device)
            if axis == 0:
                pred_vol[z, :, :] = pred_sl
            elif axis == 1:
                pred_vol[:, z, :] = pred_sl
            else:
                pred_vol[:, :, z] = pred_sl

            px = int(np.isin(pred_sl, tumor_ids).sum())
            if px > best_px:
                best_px, best_z, best_img = px, z, sl_norm

        stem = in_path.name.replace(".nii.gz", "").replace(".nii", "")
        out_nii = out_dir / f"{stem}_pred.nii.gz"
        nib.save(nib.Nifti1Image(pred_vol, affine=nii.affine, header=nii.header), str(out_nii))

        if best_img is not None:
            save_overlay_figure(
                best_img,
                np.take(pred_vol, best_z, axis=axis),
                out_dir / f"{stem}_slice{best_z}_overlay.png",
                num_classes,
                class_names,
                f"{model_name} (Slice {best_z})",
            )
        print(f"Saved 3D prediction to {out_nii} (peak tumor slice={best_z}, pixels={best_px})")
    else:
        if in_path.suffix.lower() == ".npy":
            arr = np.load(in_path).astype(np.float32)
            img_norm = window_ct(arr, args.hu_min, args.hu_max) if (arr.max() > 1.5 or arr.min() < 0.0) else arr
        else:
            arr = np.array(Image.open(in_path).convert("L"), dtype=np.float32)
            img_norm = arr / 255.0 if arr.max() > 1.0 else arr

        pred = predict_slice(model, img_norm, img_size, device)
        Image.fromarray(pred).save(out_dir / f"{in_path.stem}_mask.png")
        save_overlay_figure(
            img_norm, pred, out_dir / f"{in_path.stem}_overlay.png", num_classes, class_names, f"{model_name} Prediction"
        )
        print(f"Saved 2D prediction mask and overlay to {out_dir}")


if __name__ == "__main__":
    main()
