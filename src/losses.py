import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftDiceLoss(nn.Module):
    """Soft multi-class Dice loss; skips background by default so small tumors get strong gradients."""

    def __init__(self, include_background=False, smooth=1e-5):
        super().__init__()
        self.include_background = include_background
        self.smooth = smooth

    def forward(self, logits, targets):
        num_classes = logits.shape[1]
        probs = F.softmax(logits, dim=1)
        one_hot = F.one_hot(targets.long(), num_classes=num_classes).permute(0, 3, 1, 2).float()

        start = 0 if self.include_background else 1
        probs = probs[:, start:]
        one_hot = one_hot[:, start:]

        intersection = (probs * one_hot).sum(dim=(0, 2, 3))
        union = (probs + one_hot).sum(dim=(0, 2, 3))
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1.0 - dice.mean()


class DiceCELoss(nn.Module):
    """Weighted Cross-Entropy + Soft Dice loss."""

    def __init__(self, num_classes=3, ce_weight=0.5, dice_weight=0.5, class_weights=None, include_background_in_dice=False):
        super().__init__()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        w = torch.tensor(class_weights, dtype=torch.float32) if class_weights is not None else None
        self.ce = nn.CrossEntropyLoss(weight=w)
        self.dice = SoftDiceLoss(include_background=include_background_in_dice)

    def forward(self, logits, targets):
        if self.ce.weight is not None and self.ce.weight.device != logits.device:
            self.ce.weight = self.ce.weight.to(logits.device)
        return self.ce_weight * self.ce(logits, targets.long()) + self.dice_weight * self.dice(logits, targets)


class SegmentationMetrics:
    """Tracks TP/FP/FN across batches to compute per-class Dice, IoU, precision, and recall."""

    def __init__(self, num_classes, class_names=None):
        self.num_classes = num_classes
        if class_names is not None:
            self.class_names = list(class_names)
        elif num_classes == 5:
            self.class_names = ["background", "liver", "liver_tumor", "kidney", "kidney_tumor"]
        elif num_classes == 3:
            self.class_names = ["background", "liver", "tumor"]
        elif num_classes == 2:
            self.class_names = ["background", "tumor"]
        else:
            self.class_names = [f"class_{i}" for i in range(num_classes)]
        self.reset()

    def reset(self):
        self.tp = torch.zeros(self.num_classes, dtype=torch.float64)
        self.fp = torch.zeros(self.num_classes, dtype=torch.float64)
        self.fn = torch.zeros(self.num_classes, dtype=torch.float64)

    @torch.no_grad()
    def update(self, logits, targets):
        preds = logits.argmax(dim=1).view(-1)
        targs = targets.long().view(-1)
        for c in range(self.num_classes):
            p = preds == c
            t = targs == c
            self.tp[c] += (p & t).sum().item()
            self.fp[c] += (p & ~t).sum().item()
            self.fn[c] += (~p & t).sum().item()

    def compute(self):
        eps = 1e-7
        out = {}
        fg_dice, fg_iou = [], []
        tumor_dice, tumor_iou = [], []

        for c, name in enumerate(self.class_names):
            tp, fp, fn = self.tp[c].item(), self.fp[c].item(), self.fn[c].item()
            dice = (2.0 * tp + eps) / (2.0 * tp + fp + fn + eps)
            iou = (tp + eps) / (tp + fp + fn + eps)
            prec = (tp + eps) / (tp + fp + eps)
            rec = (tp + eps) / (tp + fn + eps)

            out[f"{name}_dice"] = dice
            out[f"{name}_iou"] = iou
            out[f"{name}_precision"] = prec
            out[f"{name}_recall"] = rec
            if c > 0:
                fg_dice.append(dice)
                fg_iou.append(iou)
                if "tumor" in name:
                    tumor_dice.append(dice)
                    tumor_iou.append(iou)

        out["mean_fg_dice"] = sum(fg_dice) / max(len(fg_dice), 1)
        out["mean_fg_iou"] = sum(fg_iou) / max(len(fg_iou), 1)
        out["tumor_dice"] = sum(tumor_dice) / len(tumor_dice) if tumor_dice else out["mean_fg_dice"]
        out["tumor_iou"] = sum(tumor_iou) / len(tumor_iou) if tumor_iou else out["mean_fg_iou"]
        return out
