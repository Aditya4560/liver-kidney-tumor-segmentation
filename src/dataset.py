import re
from pathlib import Path

import cv2
import nibabel as nib
import numpy as np
import torch
from PIL import Image
from scipy.ndimage import gaussian_filter
from torch.utils.data import DataLoader, Dataset


def get_class_names(organ="both", task="multiclass"):
    if task == "tumor":
        if organ == "liver":
            return ["background", "liver_tumor"]
        if organ == "kidney":
            return ["background", "kidney_tumor"]
        return ["background", "liver_tumor", "kidney_tumor"]

    if organ == "liver":
        return ["background", "liver", "liver_tumor"]
    if organ == "kidney":
        return ["background", "kidney", "kidney_tumor"]
    return ["background", "liver", "liver_tumor", "kidney", "kidney_tumor"]


def get_class_weights(class_names):
    # Downweight background and upweight small tumor pixels during CE loss
    weights = []
    for name in class_names:
        if name == "background":
            weights.append(0.2)
        elif "tumor" in name:
            weights.append(3.0)
        else:
            weights.append(1.0)
    return weights


def window_ct(volume, hu_min=-100.0, hu_max=200.0):
    """Clip CT Hounsfield Units to an abdominal soft-tissue window and scale to [0, 1]."""
    clipped = np.clip(volume.astype(np.float32), hu_min, hu_max)
    return (clipped - hu_min) / max(hu_max - hu_min, 1e-6)


def _remap_mask(mask, organ, task, source_kind="unified"):
    m = mask.astype(np.int64)
    if source_kind == "lits":
        unified = np.clip(m, 0, 2)
    elif source_kind == "kits":
        # KiTS uses 1=kidney, 2=tumor, 3=cyst; treat cysts as kidney parenchyma
        unified = np.zeros_like(m)
        unified[(m == 1) | (m == 3)] = 3
        unified[m == 2] = 4
    else:
        unified = np.clip(m, 0, 4)

    if task == "tumor":
        out = np.zeros_like(unified)
        if organ == "liver":
            out[unified == 2] = 1
        elif organ == "kidney":
            out[unified == 4] = 1
        else:
            out[unified == 2] = 1
            out[unified == 4] = 2
        return out

    if organ == "liver":
        out = np.zeros_like(unified)
        out[unified == 1] = 1
        out[unified == 2] = 2
        return out
    if organ == "kidney":
        out = np.zeros_like(unified)
        out[unified == 3] = 1
        out[unified == 4] = 2
        return out
    return unified


def _augment_pair(image, mask, rng):
    if rng.random() < 0.5:
        image = np.fliplr(image).copy()
        mask = np.fliplr(mask).copy()
    if rng.random() < 0.25:
        image = np.flipud(image).copy()
        mask = np.flipud(mask).copy()
    k = int(rng.integers(0, 4))
    if k > 0:
        image = np.rot90(image, k).copy()
        mask = np.rot90(mask, k).copy()
    if rng.random() < 0.5:
        alpha = float(rng.uniform(0.9, 1.1))
        beta = float(rng.uniform(-0.05, 0.05))
        image = np.clip(image * alpha + beta, 0.0, 1.0)
    return image, mask


def _resize_pair(image, mask, img_size):
    if image.shape[:2] == (img_size, img_size):
        return image.astype(np.float32), mask.astype(np.int64)
    img_r = cv2.resize(image.astype(np.float32), (img_size, img_size), interpolation=cv2.INTER_LINEAR)
    mask_r = cv2.resize(mask.astype(np.uint8), (img_size, img_size), interpolation=cv2.INTER_NEAREST)
    return img_r.astype(np.float32), mask_r.astype(np.int64)


def _paint_tumors(hu, mask, organ_mask, tumor_label, xn, yn, rng, count_range=(1, 2), radius_range=(0.04, 0.11)):
    coords = np.argwhere(organ_mask)
    if len(coords) == 0:
        return
    h, w = xn.shape
    n_tumors = int(rng.integers(count_range[0], count_range[1] + 1))
    for _ in range(n_tumors):
        py, px = coords[int(rng.integers(0, len(coords)))]
        ty = (py - h / 2.0) / (h / 2.0)
        tx = (px - w / 2.0) / (w / 2.0)
        rx = rng.uniform(*radius_range)
        ry = rng.uniform(*radius_range)

        blob = gaussian_filter(rng.normal(0.0, 1.0, (h, w)).astype(np.float32), sigma=max(h / 32.0, 1.5))
        blob /= np.max(np.abs(blob)) + 1e-6
        t_mask = (((xn - tx) / rx) ** 2 + ((yn - ty) / ry) ** 2 + 0.22 * blob <= 1.0) & organ_mask
        if np.any(t_mask):
            # Most CT liver/kidney lesions are hypodense (~18-36 HU); some are hypervascular (~95-125 HU)
            base_hu = rng.uniform(18.0, 36.0) if rng.random() < 0.78 else rng.uniform(95.0, 125.0)
            hu[t_mask] = base_hu + rng.normal(0.0, 4.0, size=int(t_mask.sum())).astype(np.float32)
            mask[t_mask] = tumor_label


class SyntheticAbdominalTumorDataset(Dataset):
    """Synthetic abdominal CT slices with liver, bilateral kidneys, and tumors for quick experiments."""

    def __init__(self, num_samples=240, img_size=256, organ="both", task="multiclass", augment=True, seed=42):
        self.num_samples = num_samples
        self.img_size = img_size
        self.organ = organ
        self.task = task
        self.augment = augment
        self.seed = seed
        self.samples = [self._make_slice(i) for i in range(num_samples)]

    def _make_slice(self, idx):
        rng = np.random.default_rng(self.seed + idx * 7919)
        h = w = self.img_size
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        yn = (yy - h / 2.0) / (h / 2.0)
        xn = (xx - w / 2.0) / (w / 2.0)

        hu = np.full((h, w), -950.0, dtype=np.float32)
        body = (xn / rng.uniform(0.76, 0.86)) ** 2 + (yn / rng.uniform(0.62, 0.74)) ** 2 <= 1.0
        hu[body] = rng.uniform(-35.0, 18.0)

        spine = ((xn - rng.uniform(-0.03, 0.03)) / 0.085) ** 2 + ((yn - 0.36) / 0.095) ** 2 <= 1.0
        hu[spine] = rng.uniform(320.0, 550.0)

        spleen = ((xn - 0.44) / 0.16) ** 2 + ((yn + 0.08) / 0.19) ** 2 <= 1.0
        hu[spleen & body] = rng.uniform(38.0, 54.0)

        mask = np.zeros((h, w), dtype=np.int64)

        # Liver (anatomical right -> left side of axial CT)
        cx, cy = rng.uniform(-0.34, -0.20), rng.uniform(-0.22, -0.05)
        rx, ry = rng.uniform(0.30, 0.39), rng.uniform(0.24, 0.32)
        ang = rng.uniform(-0.35, 0.35)
        xr = (xn - cx) * np.cos(ang) + (yn - cy) * np.sin(ang)
        yr = -(xn - cx) * np.sin(ang) + (yn - cy) * np.cos(ang)

        l_warp = gaussian_filter(rng.normal(0.0, 1.0, (h, w)).astype(np.float32), sigma=max(h / 18.0, 2.0))
        l_warp /= np.max(np.abs(l_warp)) + 1e-6
        liver = (((xr / rx) ** 2 + (yr / ry) ** 2 + 0.16 * l_warp) <= 1.0) & body & ~spine
        l_tex = gaussian_filter(rng.normal(0.0, 6.0, (h, w)).astype(np.float32), sigma=2.0)
        hu[liver] = rng.uniform(60.0, 76.0) + l_tex[liver]
        mask[liver] = 1

        if self.organ in ("liver", "both"):
            _paint_tumors(hu, mask, liver, 2, xn, yn, rng, (1, 2), (0.045, 0.11))

        # Left and right kidneys flanking the spine
        kidneys = np.zeros((h, w), dtype=bool)
        for side in (-1.0, 1.0):
            kx = side * rng.uniform(0.23, 0.31)
            ky = rng.uniform(0.18, 0.28)
            krx, kry = rng.uniform(0.10, 0.135), rng.uniform(0.13, 0.175)
            kang = side * rng.uniform(0.15, 0.45)
            kxr = (xn - kx) * np.cos(kang) + (yn - ky) * np.sin(kang)
            kyr = -(xn - kx) * np.sin(kang) + (yn - ky) * np.cos(kang)

            k_warp = gaussian_filter(rng.normal(0.0, 1.0, (h, w)).astype(np.float32), sigma=max(h / 24.0, 1.5))
            k_warp /= np.max(np.abs(k_warp)) + 1e-6
            hilum = ((xn - (kx - side * krx * 0.65)) / (krx * 0.45)) ** 2 + ((yn - ky) / (kry * 0.45)) ** 2 <= 1.0
            k_one = (((kxr / krx) ** 2 + (kyr / kry) ** 2 + 0.12 * k_warp) <= 1.0) & ~hilum & body & ~spine & ~liver

            k_tex = gaussian_filter(rng.normal(0.0, 4.0, (h, w)).astype(np.float32), sigma=1.5)
            hu[k_one] = rng.uniform(42.0, 56.0) + k_tex[k_one]
            kidneys |= k_one

        mask[kidneys] = 3
        if self.organ in ("kidney", "both"):
            _paint_tumors(hu, mask, kidneys, 4, xn, yn, rng, (1, 2), (0.035, 0.085))

        hu = gaussian_filter(hu, sigma=0.65) + rng.normal(0.0, 6.5, (h, w)).astype(np.float32)
        return window_ct(hu, -100.0, 200.0), mask

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        image, mask = self.samples[idx]
        image = image.copy()
        mask = _remap_mask(mask, self.organ, self.task, source_kind="unified")
        if self.augment:
            image, mask = _augment_pair(image, mask, np.random.default_rng())
        return torch.from_numpy(image).unsqueeze(0).float(), torch.from_numpy(mask).long()


SyntheticLiverTumorDataset = SyntheticAbdominalTumorDataset


def _fallback_img_mask_pairs(root):
    img_dir, msk_dir = root / "images", root / "masks"
    if not (img_dir.is_dir() and msk_dir.is_dir()):
        return []
    pairs = []
    for vf in sorted(img_dir.glob("*.nii*")):
        mf = msk_dir / vf.name
        if mf.exists():
            pairs.append((vf, mf))
    return pairs


def discover_lits_pairs(data_dir):
    root = Path(data_dir)
    if not root.exists():
        raise FileNotFoundError(f"LiTS directory not found: {root}")

    pairs = []
    for vf in sorted(root.rglob("volume-*.nii*")):
        m = re.search(r"volume-(\d+)", vf.name)
        if not m:
            continue
        vid = m.group(1)
        cands = list(vf.parent.glob(f"segmentation-{vid}.nii*")) or list(root.rglob(f"segmentation-{vid}.nii*"))
        if cands:
            pairs.append((vf, cands[0]))

    pairs = pairs or _fallback_img_mask_pairs(root)
    if not pairs:
        raise FileNotFoundError(f"No LiTS pairs found in {root} (expected volume-<id>.nii* / segmentation-<id>.nii*)")
    return pairs


def discover_kits_pairs(data_dir):
    root = Path(data_dir)
    if not root.exists():
        raise FileNotFoundError(f"KiTS directory not found: {root}")

    pairs = []
    for img_path in sorted(root.rglob("imaging.nii*")):
        cands = list(img_path.parent.glob("segmentation.nii*"))
        if cands:
            pairs.append((img_path, cands[0]))

    pairs = pairs or _fallback_img_mask_pairs(root)
    if not pairs:
        raise FileNotFoundError(f"No KiTS pairs found in {root} (expected case_*/imaging.nii* / segmentation.nii*)")
    return pairs


def _infer_axial_axis(shape, source_kind):
    if len(shape) != 3:
        return 2
    if shape[1] == shape[2] and shape[0] != shape[1]:
        return 0
    if shape[0] == shape[1] and shape[2] != shape[0]:
        return 2
    return 0 if source_kind == "kits" else 2


class MedicalNiftiDataset(Dataset):
    """Extracts 2D axial slices from 3D LiTS or KiTS NIfTI CT volumes."""

    def __init__(
        self,
        pairs,
        source_kind="lits",
        organ=None,
        img_size=256,
        task="multiclass",
        hu_min=-100.0,
        hu_max=200.0,
        augment=False,
        keep_empty_ratio=0.05,
        seed=42,
        max_slices=None,
        max_slices_per_case=None,
    ):
        self.source_kind = source_kind
        self.organ = organ or ("kidney" if source_kind == "kits" else "liver")
        self.img_size = img_size
        self.task = task
        self.hu_min = hu_min
        self.hu_max = hu_max
        self.augment = augment

        rng = np.random.default_rng(seed)
        self.slices = []
        self._cache = {}
        self._seg_cache = {}

        for vol_path, seg_path in pairs:
            seg = np.asarray(nib.load(str(seg_path)).dataobj)
            axis = _infer_axial_axis(seg.shape, source_kind)
            other_axes = tuple(i for i in range(seg.ndim) if i != axis)
            has_labels = np.max(seg, axis=other_axes) > 0
            positive_z = np.flatnonzero(has_labels)
            empty_z = np.flatnonzero(~has_labels)
            empty_count = min(len(empty_z), int(round(len(positive_z) * keep_empty_ratio)))
            if empty_count:
                empty_z = rng.choice(empty_z, size=empty_count, replace=False)
            else:
                empty_z = np.empty(0, dtype=int)

            candidates = np.concatenate((positive_z, empty_z))
            if max_slices_per_case is not None and len(candidates) > max_slices_per_case:
                candidates = rng.choice(candidates, size=max_slices_per_case, replace=False)
            if max_slices is not None:
                remaining = max_slices - len(self.slices)
                if remaining <= 0:
                    break
                if len(candidates) > remaining:
                    candidates = rng.choice(candidates, size=remaining, replace=False)
            self.slices.extend((vol_path, seg_path, axis, int(z)) for z in candidates)
            if len(candidates):
                if len(self._seg_cache) >= 8:
                    self._seg_cache.clear()
                self._seg_cache[seg_path] = seg

    def _get_vol(self, path):
        if path not in self._cache:
            if len(self._cache) >= 8:
                self._cache.clear()
            self._cache[path] = nib.load(str(path)).get_fdata(dtype=np.float32)
        return self._cache[path]

    def _get_seg(self, path):
        if path not in self._seg_cache:
            if len(self._seg_cache) >= 8:
                self._seg_cache.clear()
            self._seg_cache[path] = np.asarray(nib.load(str(path)).dataobj)
        return self._seg_cache[path]

    def __len__(self):
        return len(self.slices)

    def __getitem__(self, idx):
        vol_path, seg_path, axis, z = self.slices[idx]
        vol = self._get_vol(vol_path)
        seg = self._get_seg(seg_path)

        image = window_ct(np.take(vol, z, axis=axis), self.hu_min, self.hu_max)
        mask = _remap_mask(np.take(seg, z, axis=axis), self.organ, self.task, self.source_kind)
        image, mask = _resize_pair(image, mask, self.img_size)
        if self.augment:
            image, mask = _augment_pair(image, mask, np.random.default_rng())
        return torch.from_numpy(image).unsqueeze(0).float(), torch.from_numpy(mask).long()


LiTSNiftiDataset = MedicalNiftiDataset


class SliceFolderDataset(Dataset):
    """Loads pre-extracted 2D slice pairs from `images/` and `masks/` folders."""

    def __init__(self, root_dir, img_size=256, organ="both", task="multiclass", source_kind="unified", augment=False):
        root = Path(root_dir)
        img_dir, msk_dir = root / "images", root / "masks"
        if not img_dir.is_dir() or not msk_dir.is_dir():
            raise FileNotFoundError(f"Expected 'images/' and 'masks/' inside {root}")

        self.img_size = img_size
        self.organ = organ
        self.task = task
        self.source_kind = source_kind
        self.augment = augment
        self.pairs = []

        exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".npy"}
        for img_path in sorted(img_dir.iterdir()):
            if img_path.suffix.lower() not in exts:
                continue
            cands = [msk_dir / img_path.name] + [msk_dir / f"{img_path.stem}{e}" for e in exts]
            match = next((c for c in cands if c.exists()), None)
            if match is not None:
                self.pairs.append((img_path, match))

        if not self.pairs:
            raise FileNotFoundError(f"No matching image/mask pairs found in {root}")

    @staticmethod
    def _read(path, is_mask):
        if path.suffix.lower() == ".npy":
            arr = np.load(path)
            return arr.astype(np.int64 if is_mask else np.float32)
        arr = np.array(Image.open(path).convert("L"))
        if is_mask:
            return arr.astype(np.int64)
        arr = arr.astype(np.float32)
        return arr / 255.0 if arr.max() > 1.0 else arr

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_path, msk_path = self.pairs[idx]
        image = self._read(img_path, is_mask=False)
        mask = _remap_mask(self._read(msk_path, is_mask=True), self.organ, self.task, self.source_kind)
        image, mask = _resize_pair(image, mask, self.img_size)
        if self.augment:
            image, mask = _augment_pair(image, mask, np.random.default_rng())
        return torch.from_numpy(image).unsqueeze(0).float(), torch.from_numpy(mask).long()


def build_dataloaders(
    dataset_type="synthetic",
    data_dir=None,
    organ="both",
    img_size=256,
    batch_size=8,
    task="multiclass",
    val_split=0.2,
    num_workers=0,
    synthetic_train_size=240,
    synthetic_val_size=60,
    seed=42,
    max_cases=None,
    max_slices=None,
    max_slices_per_case=None,
):
    dtype = dataset_type.lower().strip()

    if dtype == "synthetic":
        train_ds = SyntheticAbdominalTumorDataset(
            num_samples=synthetic_train_size, img_size=img_size, organ=organ, task=task, augment=True, seed=seed
        )
        val_ds = SyntheticAbdominalTumorDataset(
            num_samples=synthetic_val_size, img_size=img_size, organ=organ, task=task, augment=False, seed=seed + 10000
        )
    elif dtype in ("lits", "kits"):
        if not data_dir:
            raise ValueError(f"--data-dir is required for --dataset {dtype}")
        pairs = discover_lits_pairs(data_dir) if dtype == "lits" else discover_kits_pairs(data_dir)
        if max_cases is not None:
            pairs = pairs[:max_cases]
        perm = np.random.default_rng(seed).permutation(len(pairs))
        n_val = max(1, int(round(len(pairs) * val_split)))
        val_pairs = [pairs[i] for i in perm[:n_val]]
        train_pairs = [pairs[i] for i in perm[n_val:]] or val_pairs
        target_organ = "liver" if dtype == "lits" else "kidney"
        train_ds = MedicalNiftiDataset(
            train_pairs,
            dtype,
            target_organ,
            img_size,
            task,
            augment=True,
            seed=seed,
            max_slices=max_slices,
            max_slices_per_case=max_slices_per_case,
        )
        val_ds = MedicalNiftiDataset(
            val_pairs,
            dtype,
            target_organ,
            img_size,
            task,
            augment=False,
            seed=seed,
            max_slices=max_slices,
            max_slices_per_case=max_slices_per_case,
        )
    elif dtype == "slices":
        if not data_dir:
            raise ValueError("--data-dir is required for --dataset slices")
        root = Path(data_dir)
        if (root / "train").is_dir() and (root / "val").is_dir():
            train_ds = SliceFolderDataset(root / "train", img_size=img_size, organ=organ, task=task, augment=True)
            val_ds = SliceFolderDataset(root / "val", img_size=img_size, organ=organ, task=task, augment=False)
        else:
            full_ds = SliceFolderDataset(root, img_size=img_size, organ=organ, task=task, augment=False)
            n_val = max(1, int(round(len(full_ds) * val_split)))
            n_train = max(1, len(full_ds) - n_val)
            gen = torch.Generator().manual_seed(seed)
            train_ds, val_ds = torch.utils.data.random_split(full_ds, [n_train, len(full_ds) - n_train], generator=gen)
    else:
        raise ValueError(f"Unknown dataset '{dataset_type}'. Choose from: synthetic, lits, kits, slices")

    pin = torch.cuda.is_available()
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin)
    return train_loader, val_loader
