import argparse
import json
import time
from pathlib import Path

import torch

from src.dataset import build_dataloaders, get_class_names
from src.losses import SegmentationMetrics
from src.models import build_model, count_parameters
from src.visualization import plot_model_comparison


def load_model_from_checkpoint(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    model = build_model(
        name=ckpt["model_name"],
        in_channels=ckpt.get("in_channels", 1),
        num_classes=ckpt.get("num_classes", 3),
        base_channels=ckpt.get("base_channels", 32),
        dropout=ckpt.get("dropout", 0.1),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, ckpt


def evaluate_checkpoint(ckpt_path, val_loader, device, max_vis=4):
    model, ckpt = load_model_from_checkpoint(ckpt_path, device)
    num_classes = int(ckpt.get("num_classes", 3))
    class_names = list(
        ckpt.get("class_names", get_class_names(ckpt.get("organ", "liver"), ckpt.get("task", "multiclass")))
    )
    metrics = SegmentationMetrics(num_classes=num_classes, class_names=class_names)

    vis_imgs, vis_gts, vis_preds = [], [], []
    elapsed, n_slices = 0.0, 0

    with torch.no_grad():
        for images, masks in val_loader:
            x = images.to(device)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            logits = model(x)
            if device.type == "cuda":
                torch.cuda.synchronize()
            elapsed += time.perf_counter() - t0
            n_slices += images.shape[0]

            metrics.update(logits.cpu(), masks)
            preds = logits.argmax(dim=1).cpu().numpy()

            if len(vis_preds) < max_vis:
                for i in range(min(max_vis - len(vis_preds), images.shape[0])):
                    vis_imgs.append(images[i, 0].numpy())
                    vis_gts.append(masks[i].numpy())
                    vis_preds.append(preds[i])

    summary = {
        "model": ckpt["model_name"],
        "organ": ckpt.get("organ", "both"),
        "checkpoint": str(ckpt_path),
        "params": count_parameters(model),
        "ms_per_slice": (elapsed / max(n_slices, 1)) * 1000.0,
        **metrics.compute(),
    }
    return summary, vis_imgs, vis_gts, vis_preds, class_names


def parse_args():
    p = argparse.ArgumentParser(description="Compare trained CNN / U-Net checkpoints")
    p.add_argument("--checkpoints", nargs="+", required=True)
    p.add_argument("--organ", default=None, choices=["liver", "kidney", "both"])
    p.add_argument("--dataset", default="synthetic", choices=["synthetic", "lits", "kits", "slices"])
    p.add_argument("--data-dir", default=None)
    p.add_argument("--task", default=None, choices=["multiclass", "tumor"])
    p.add_argument("--img-size", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--synthetic-val-size", type=int, default=60)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-dir", default="outputs/compare")
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    first_ckpt = torch.load(args.checkpoints[0], map_location="cpu", weights_only=True)
    organ = args.organ or first_ckpt.get("organ", "both")
    task = args.task or first_ckpt.get("task", "multiclass")
    class_names = get_class_names(organ=organ, task=task)

    _, val_loader = build_dataloaders(
        dataset_type=args.dataset,
        data_dir=args.data_dir,
        organ=organ,
        img_size=args.img_size,
        batch_size=args.batch_size,
        task=task,
        synthetic_train_size=16,
        synthetic_val_size=args.synthetic_val_size,
        seed=args.seed,
    )

    results = []
    model_preds = {}
    ref_imgs, ref_gts = [], []

    for ckpt_str in args.checkpoints:
        summary, imgs, gts, preds, ckpt_classes = evaluate_checkpoint(Path(ckpt_str), val_loader, device)
        results.append(summary)
        label = str(summary["model"])
        if label in model_preds:
            label = f"{label}_{len(model_preds)}"
        model_preds[label] = preds
        if not ref_imgs:
            ref_imgs, ref_gts = imgs, gts
            class_names = ckpt_classes

    print(f"\n=== Evaluation Summary (organ={organ}, task={task}) ===")
    tumor_cols = [c for c in class_names if "tumor" in c]
    extra_hdr = " ".join(f"{c[:12]:>13}" for c in tumor_cols)
    hdr = f"{'Model':<10} {'Params':>10} {extra_hdr} {'MeanTumorDice':>14} {'MeanFGDice':>11} {'ms/slice':>9}"
    print(hdr)
    print("-" * len(hdr))
    for r in results:
        extra_vals = " ".join(f"{r[f'{c}_dice']:>13.4f}" for c in tumor_cols)
        print(
            f"{str(r['model']):<10} {int(r['params']):>10,} {extra_vals} "
            f"{r['tumor_dice']:>14.4f} {r['mean_fg_dice']:>11.4f} {r['ms_per_slice']:>9.2f}"
        )

    (out_dir / "comparison_metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    plot_model_comparison(
        ref_imgs, ref_gts, model_preds, out_dir / "comparison.png", num_classes=len(class_names), class_names=class_names
    )
    print(f"\nSaved comparison plot and metrics to {out_dir}")


if __name__ == "__main__":
    main()
