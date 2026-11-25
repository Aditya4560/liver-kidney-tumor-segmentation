from src.models.cnn import LiverSegCNN
from src.models.unet import CNNUNet, UNet

MODEL_REGISTRY = {
    "cnn": LiverSegCNN,
    "unet": UNet,
    "cnn_unet": CNNUNet,
}


def build_model(name, in_channels=1, num_classes=3, base_channels=32, dropout=0.1):
    key = name.lower().strip()
    if key not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{name}'. Expected one of {list(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[key](
        in_channels=in_channels,
        num_classes=num_classes,
        base_channels=base_channels,
        dropout=dropout,
    )


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


__all__ = ["LiverSegCNN", "UNet", "CNNUNet", "MODEL_REGISTRY", "build_model", "count_parameters"]
