import nibabel as nib
import numpy as np
import pytest
import torch
from PIL import Image

from src.dataset import (
    MedicalNiftiDataset,
    SliceFolderDataset,
    SyntheticAbdominalTumorDataset,
    build_dataloaders,
    discover_kits_pairs,
    discover_lits_pairs,
    get_class_names,
    window_ct,
)
from src.losses import DiceCELoss, SegmentationMetrics
from src.models import build_model


@pytest.mark.parametrize("model_name", ["cnn", "unet", "cnn_unet"])
@pytest.mark.parametrize("num_classes", [2, 3, 5])
def test_model_forward_and_backward(model_name, num_classes):
    model = build_model(model_name, in_channels=1, num_classes=num_classes, base_channels=16)
    x = torch.randn(2, 1, 128, 128)
    y = torch.randint(0, num_classes, (2, 128, 128))

    logits = model(x)
    assert logits.shape == (2, num_classes, 128, 128)

    loss = DiceCELoss(num_classes=num_classes)(logits, y)
    assert torch.isfinite(loss)
    loss.backward()


def test_window_ct_bounds():
    arr = np.array([-1000.0, -100.0, 50.0, 200.0, 1000.0], dtype=np.float32)
    out = window_ct(arr, hu_min=-100.0, hu_max=200.0)
    assert np.allclose(out, [0.0, 0.0, 0.5, 1.0, 1.0])


def test_segmentation_metrics_multiorgan():
    names = get_class_names(organ="both", task="multiclass")
    metrics = SegmentationMetrics(num_classes=5, class_names=names)
    target = torch.tensor([[[0, 1, 2, 3, 4], [0, 1, 2, 3, 4]]])
    logits = torch.full((1, 5, 2, 5), -10.0)
    for r in range(2):
        for c in range(5):
            logits[0, target[0, r, c], r, c] = 10.0

    metrics.update(logits, target)
    res = metrics.compute()
    assert res["liver_tumor_dice"] > 0.999
    assert res["kidney_tumor_dice"] > 0.999
    assert res["tumor_dice"] > 0.999
    assert res["mean_fg_dice"] > 0.999


@pytest.mark.parametrize(
    "organ,task,expected",
    [
        ("liver", "multiclass", {0, 1, 2}),
        ("kidney", "multiclass", {0, 1, 2}),
        ("both", "multiclass", {0, 1, 2, 3, 4}),
        ("liver", "tumor", {0, 1}),
        ("kidney", "tumor", {0, 1}),
        ("both", "tumor", {0, 1, 2}),
    ],
)
def test_synthetic_dataset_organs_and_tasks(organ, task, expected):
    ds = SyntheticAbdominalTumorDataset(num_samples=4, img_size=128, organ=organ, task=task, augment=False)
    img, mask = ds[0]
    assert img.shape == (1, 128, 128)
    assert mask.shape == (128, 128)
    unique = set(torch.unique(mask).tolist())
    assert unique.issubset(expected)
    assert len(unique) >= 2


def test_slice_folder_dataset(tmp_path):
    (tmp_path / "images").mkdir()
    (tmp_path / "masks").mkdir()
    img = (np.random.rand(64, 64) * 255).astype(np.uint8)
    msk = np.zeros((64, 64), dtype=np.uint8)
    msk[8:30, 8:30] = 1
    msk[14:22, 14:22] = 2
    msk[35:55, 35:55] = 3
    msk[40:48, 40:48] = 4

    Image.fromarray(img).save(tmp_path / "images" / "case01.png")
    Image.fromarray(msk).save(tmp_path / "masks" / "case01.png")

    ds = SliceFolderDataset(tmp_path, img_size=96, organ="both", task="multiclass", augment=False)
    assert len(ds) == 1
    t_img, t_msk = ds[0]
    assert t_img.shape == (1, 96, 96)
    assert t_msk.shape == (96, 96)
    assert set(torch.unique(t_msk).tolist()) == {0, 1, 2, 3, 4}


def test_lits_nifti_dataset(tmp_path):
    vol = np.random.uniform(-100, 200, size=(64, 64, 6)).astype(np.float32)
    seg = np.zeros((64, 64, 6), dtype=np.uint8)
    seg[12:48, 12:48, 2:5] = 1
    seg[22:32, 22:32, 3] = 2

    nib.save(nib.Nifti1Image(vol, np.eye(4)), str(tmp_path / "volume-0.nii.gz"))
    nib.save(nib.Nifti1Image(seg, np.eye(4)), str(tmp_path / "segmentation-0.nii.gz"))

    pairs = discover_lits_pairs(tmp_path)
    assert len(pairs) == 1
    ds = MedicalNiftiDataset(pairs, source_kind="lits", img_size=128, task="multiclass", keep_empty_ratio=0.0)
    assert len(ds) == 3
    img, msk = ds[0]
    assert img.shape == (1, 128, 128)
    assert msk.shape == (128, 128)


def test_kits_nifti_dataset(tmp_path):
    case_dir = tmp_path / "case_00000"
    case_dir.mkdir()
    vol = np.random.uniform(-100, 200, size=(6, 64, 64)).astype(np.float32)
    seg = np.zeros((6, 64, 64), dtype=np.uint8)
    seg[1:4, 15:45, 15:45] = 1
    seg[2, 20:30, 20:30] = 2
    seg[3, 30:36, 30:36] = 3  # cyst

    nib.save(nib.Nifti1Image(vol, np.eye(4)), str(case_dir / "imaging.nii.gz"))
    nib.save(nib.Nifti1Image(seg, np.eye(4)), str(case_dir / "segmentation.nii.gz"))

    pairs = discover_kits_pairs(tmp_path)
    assert len(pairs) == 1
    ds = MedicalNiftiDataset(pairs, source_kind="kits", img_size=128, task="multiclass", keep_empty_ratio=0.0)
    assert len(ds) == 3
    img, msk = ds[2]
    assert img.shape == (1, 128, 128)
    assert set(torch.unique(msk).tolist()).issubset({0, 1, 2})


def test_dataloader_smoke():
    train_loader, val_loader = build_dataloaders(
        dataset_type="synthetic",
        organ="both",
        img_size=128,
        batch_size=4,
        synthetic_train_size=8,
        synthetic_val_size=4,
    )
    imgs, masks = next(iter(train_loader))
    assert imgs.shape == (4, 1, 128, 128)
    assert masks.shape == (4, 128, 128)
    assert len(val_loader) == 1
