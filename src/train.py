import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from src.dataset import build_dataloaders, get_class_names, get_class_weights
from src.losses import DiceCELoss, SegmentationMetrics
from src.models import build_model, count_parameters
from src.visualization import plot_prediction_grid, plot_training_history


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_epoch(model, loader, criterion, device, class_names, optimizer=None, scaler=None, use_amp=False):
    is_train = optimizer is not None
    model.train(is_train)
    metrics = SegmentationMetrics(num_classes=len(class_names), class_names=class_names)
    total_loss, steps = 0.0, 0

    with torch.set_grad_enabled(is_train):
        for images, masks in loader:
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            if is_train:
                optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                logits = model(images)
                loss = criterion(logits, masks)

            if is_train:
                if scaler is not None and use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

            total_loss += float(loss.item())
            steps += 1
            metrics.update(logits.detach(), masks.detach())

    return total_loss / max(steps, 1), metrics.compute()


def save_checkpoint(path, model, args, epoch, metrics, class_names, organ):
    torch.save(
        {
            "model_name": args.model,
            "organ": organ,
            "class_names": class_names,
            "num_classes": len(class_names),
            "in_channels": 1,
            "base_channels": args.base_channels,
            "dropout": args.dropout,
            "img_size": args.img_size,
            "task": args.task,
            "epoch": epoch,
            "metrics": metrics,
            "model_state_dict": model.state_dict(),
        },
        path,
    )


def parse_args():
    p = argparse.ArgumentParser(description="Train CNN / U-Net for Liver & Kidney Tumor Segmentation")
    p.add_argument("--model", default="unet", choices=["cnn", "unet", "cnn_unet"])
    p.add_argument("--organ", default="both", choices=["liver", "kidney", "both"])
    p.add_argument("--dataset", default="synthetic", choices=["synthetic", "lits", "kits", "slices"])
    p.add_argument("--data-dir", default=None)
    p.add_argument("--task", default="multiclass", choices=["multiclass", "tumor"])
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--img-size", type=int, default=256)
    p.add_argument("--base-channels", type=int, default=32)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--synthetic-train-size", type=int, default=240)
    p.add_argument("--synthetic-val-size", type=int, default=60)
    p.add_argument("--max-cases", type=int, default=None, help="Limit real NIfTI training to the first N cases")
    p.add_argument("--max-slices", type=int, default=None, help="Limit slices per train/validation split")
    p.add_argument(
        "--max-slices-per-case", type=int, default=None, help="Limit sampled slices from each real NIfTI case"
    )
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--amp", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save-dir", default="outputs/run")
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    organ = "liver" if args.dataset == "lits" else ("kidney" if args.dataset == "kits" else args.organ)
    class_names = get_class_names(organ=organ, task=args.task)
    num_classes = len(class_names)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = bool(args.amp and device.type == "cuda")

    train_loader, val_loader = build_dataloaders(
        dataset_type=args.dataset,
        data_dir=args.data_dir,
        organ=organ,
        img_size=args.img_size,
        batch_size=args.batch_size,
        task=args.task,
        num_workers=args.num_workers,
        synthetic_train_size=args.synthetic_train_size,
        synthetic_val_size=args.synthetic_val_size,
        seed=args.seed,
        max_cases=args.max_cases,
        max_slices=args.max_slices,
        max_slices_per_case=args.max_slices_per_case,
    )

    model = build_model(
        args.model, in_channels=1, num_classes=num_classes, base_channels=args.base_channels, dropout=args.dropout
    ).to(device)

    criterion = DiceCELoss(num_classes=num_classes, class_weights=get_class_weights(class_names))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)

    print(
        f"Training {args.model} | organ={organ} | classes={class_names} | "
        f"params={count_parameters(model):,} | device={device}"
    )

    best_tumor_dice = -1.0
    history = []

    for epoch in tqdm(range(1, args.epochs + 1), desc=args.model.upper()):
        train_loss, train_m = run_epoch(model, train_loader, criterion, device, class_names, optimizer, scaler, use_amp)
        val_loss, val_m = run_epoch(model, val_loader, criterion, device, class_names, use_amp=use_amp)
        scheduler.step()

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_tumor_dice": train_m["tumor_dice"],
            "tumor_dice": val_m["tumor_dice"],
            "mean_fg_dice": val_m["mean_fg_dice"],
            "tumor_iou": val_m["tumor_iou"],
        }
        for cname in class_names[1:]:
            row[f"{cname}_dice"] = val_m[f"{cname}_dice"]
        history.append(row)

        scores = " | ".join(f"{c}={val_m[f'{c}_dice']:.3f}" for c in class_names[1:])
        print(
            f"Epoch {epoch:03d}/{args.epochs:03d} | train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | {scores} | tumor_dice={val_m['tumor_dice']:.4f}"
        )

        if val_m["tumor_dice"] > best_tumor_dice:
            best_tumor_dice = val_m["tumor_dice"]
            save_checkpoint(save_dir / "best_model.pt", model, args, epoch, val_m, class_names, organ)

    save_checkpoint(save_dir / "last_model.pt", model, args, args.epochs, val_m, class_names, organ)
    (save_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    plot_training_history(history, save_dir / "training_curves.png")

    # Render sample overlays using the best checkpoint
    ckpt = torch.load(save_dir / "best_model.pt", map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    vis_imgs, vis_gts, vis_preds = [], [], []
    with torch.no_grad():
        for images, masks in val_loader:
            preds = model(images.to(device)).argmax(dim=1).cpu().numpy()
            for i in range(min(4 - len(vis_imgs), images.shape[0])):
                vis_imgs.append(images[i, 0].numpy())
                vis_gts.append(masks[i].numpy())
                vis_preds.append(preds[i])
            if len(vis_imgs) >= 4:
                break

    plot_prediction_grid(
        vis_imgs,
        vis_gts,
        vis_preds,
        save_dir / "val_predictions.png",
        num_classes=num_classes,
        model_label=args.model.upper(),
        class_names=class_names,
    )
    print(f"Done. Best val tumor_dice={best_tumor_dice:.4f} -> {save_dir}")


if __name__ == "__main__":
    main()
