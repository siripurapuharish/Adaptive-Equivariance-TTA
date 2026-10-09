
import os
import torch
from collections import OrderedDict
from models.DRNSeg import DRNSeg

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHECKPOINT_DIR = os.path.join(ROOT, "checkpoints")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

def load_weights(path, is_adversarial=False):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)

    if is_adversarial:
        weights = checkpoint["state_dict"]

        # Remove DataParallel prefix for a plain DRNSeg model.
        weights = OrderedDict(
            (key.removeprefix("module."), value)
            for key, value in weights.items()
        )
    else:
        weights = checkpoint

    return weights

models_to_test = [
    ("Clean", "clean_drn_d_22_cityscapes.pth", False),
    ("Adversarial", "advtrain_drn_d_22_cityscapes.pth.tar", True),
]

for name, filename, is_adv in models_to_test:
    print(f"\nTesting {name} checkpoint...")

    path = os.path.join(CHECKPOINT_DIR, filename)

    try:
        weights = load_weights(path, is_adv)

        model = DRNSeg(
            model_name="drn_d_22",
            classes=19,
            pretrained=False
        )

        result = model.load_state_dict(weights, strict=True)
        model = model.to(device).eval()

        print("Strict loading: SUCCESS")
        print("Missing keys:", result.missing_keys)
        print("Unexpected keys:", result.unexpected_keys)

        with torch.no_grad():
            sample = torch.randn(1, 3, 340, 680, device=device)
            output = model(sample)

        print("Output shape:", tuple(output[0].shape))
        print(f"✅ {name} model loaded and forward pass succeeded.")

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    except Exception as error:
        print(f"❌ {name} model failed:")
        print(type(error).__name__ + ":", error)
