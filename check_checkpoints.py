
import os
import torch
from collections import OrderedDict

# Project root folder
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")

files = [
    "clean_drn_d_22_cityscapes.pth",
    "advtrain_drn_d_22_cityscapes.pth.tar",
]

for filename in files:
    path = os.path.join(CHECKPOINT_DIR, filename)

    print("\n" + "=" * 60)
    print("Checking:", filename)

    if not os.path.exists(path):
        print("❌ FILE NOT FOUND")
        print("Expected path:", path)
        continue

    print(f"File size: {os.path.getsize(path) / (1024 * 1024):.2f} MB")

    try:
        checkpoint = torch.load(
            path,
            map_location="cpu",
            weights_only=False
        )

        # Clean checkpoint is directly a state_dict.
        # Adversarial checkpoint contains a state_dict inside a dictionary.
        if isinstance(checkpoint, (dict, OrderedDict)):
            if "state_dict" in checkpoint:
                weights = checkpoint["state_dict"]
                print("Checkpoint type: Training checkpoint")
                print("Epoch:", checkpoint.get("epoch", "Not available"))
                print("Architecture:", checkpoint.get("arch", "Not available"))
            else:
                weights = checkpoint
                print("Checkpoint type: Direct model weights")
        else:
            print("❌ Unexpected checkpoint format")
            continue

        if not isinstance(weights, (dict, OrderedDict)):
            print("❌ Model weights are not a dictionary")
            continue

        print("Number of weight entries:", len(weights))
        print("First 5 keys:")
        for key in list(weights.keys())[:5]:
            print(" ", key)

        print("Segmentation head present:", any(
            key.endswith("seg.weight") for key in weights
        ))
        print("Upsampling weights present:", any(
            key.endswith("up.weight") for key in weights
        ))

        if len(weights) > 0:
            print("✅ Checkpoint loaded successfully")
        else:
            print("❌ Empty checkpoint")

    except Exception as e:
        print("❌ FAILED TO LOAD")
        print("Error:", e)

print("\nCheckpoint inspection completed.")
