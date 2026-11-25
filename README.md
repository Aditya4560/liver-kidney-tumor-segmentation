# Liver & Kidney Tumor Segmentation (CNN & U-Net)

PyTorch segmentation pipeline for **Liver, Kidney, Liver Tumor, and Kidney Tumor** detection on abdominal CT scans using **CNN** and **U-Net** models.

## Models

- `cnn` (`LiverSegCNN`): Encoder-decoder CNN without skip connections (baseline).
- `unet` (`UNet`): Standard 2D U-Net with encoder-decoder skip connections.
- `cnn_unet` (`CNNUNet`): Residual CNN blocks inside a U-Net skip-connection architecture.

## Organ & Label Modes

Use `--organ` and `--task` to switch targets without changing any model code:

- `--organ both` (default): `0=background`, `1=liver`, `2=liver_tumor`, `3=kidney`, `4=kidney_tumor`
- `--organ liver`: `0=background`, `1=liver`, `2=liver_tumor`
- `--organ kidney`: `0=background`, `1=kidney`, `2=kidney_tumor`
- `--task tumor`: Predict only tumor masks (`liver_tumor`, `kidney_tumor`, or both)

## Supported Datasets

- **Synthetic CT** (`--dataset synthetic`): Built-in abdominal CT generator (body, spine, liver, bilateral kidneys, and tumors) for quick runs and testing.
- **LiTS** (`--dataset lits`): 3D NIfTI liver & liver tumor volumes (`volume-*.nii*` / `segmentation-*.nii*`).
- **KiTS19 / KiTS21 / KiTS23** (`--dataset kits`): 3D NIfTI kidney & kidney tumor volumes (`case_*/imaging.nii*` / `segmentation.nii*`).
- **2D Slices** (`--dataset slices`): Folder with `images/` and `masks/` (`.png`, `.jpg`, `.npy`).

## Usage

### Train & Compare CNN vs U-Net (Synthetic Liver + Kidney CT)

```bash
python -m src.train --model cnn --organ both --dataset synthetic --epochs 12 --batch-size 16 --save-dir outputs/cnn
python -m src.train --model unet --organ both --dataset synthetic --epochs 12 --batch-size 16 --save-dir outputs/unet
python -m src.train --model cnn_unet --organ both --dataset synthetic --epochs 12 --batch-size 16 --save-dir outputs/cnn_unet

python -m src.evaluate --checkpoints outputs/cnn/best_model.pt outputs/unet/best_model.pt outputs/cnn_unet/best_model.pt --organ both --dataset synthetic --out-dir outputs/compare
```

### Train on Real LiTS or KiTS Data

```bash
# Liver tumor (LiTS)
python -m src.train --model unet --dataset lits --data-dir path/to/lits --epochs 50 --batch-size 8 --amp --save-dir outputs/lits_unet

# Kidney tumor (KiTS)
python -m src.train --model unet --dataset kits --data-dir path/to/kits23/dataset --epochs 50 --batch-size 8 --amp --save-dir outputs/kits_unet
```

### Predict on a 2D Slice or 3D NIfTI Volume

```bash
python -m src.predict --checkpoint outputs/unet/best_model.pt --input path/to/slice.png --out-dir outputs/preds
python -m src.predict --checkpoint outputs/kits_unet/best_model.pt --input path/to/case_00000/imaging.nii.gz --out-dir outputs/preds
```

### Run Tests

```bash
python -m pytest -v
```
