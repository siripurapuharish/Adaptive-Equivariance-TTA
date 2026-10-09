#!/usr/bin/env python3
"""
Cityscapes Dataset Preprocessing for "Robust Perception through Equivariance" (ICML 2023)

Preprocesses raw Cityscapes data for faithful reproduction:
  1. Downsamples RGB images to 680x340 using bilinear interpolation.
  2. Converts raw Cityscapes labelIds (0-33) to 19 standard trainIds (0-18, ignore=255)
     and downsamples to 680x340 using nearest-neighbor interpolation.
  3. Downsamples disparity maps to 680x340 using nearest-neighbor interpolation,
     preserving 16-bit disparity fidelity for the dataloader.
  4. Generates val_images.txt, val_labels.txt, and info.json required by Equi4Rob/dataloaders/dataloader.py.
"""

import os
import sys
import json
import argparse
import numpy as np
from PIL import Image

# Pillow interpolation compatibility (supports Pillow 9.x, 10.x, and legacy)
RESAMPLE_BILINEAR = getattr(getattr(Image, 'Resampling', Image), 'BILINEAR')
RESAMPLE_NEAREST = getattr(getattr(Image, 'Resampling', Image), 'NEAREST')

# Official Cityscapes labelId -> trainId mapping
# Trainable classes: 0 to 18; All void/ignored/unlabeled classes: 255
CITYSCAPES_LABEL_TO_TRAIN_ID = {
    -1: 255,  # license plate / ignored
     0: 255,  # unlabeled
     1: 255,  # ego vehicle
     2: 255,  # rectification border
     3: 255,  # out of roi
     4: 255,  # static
     5: 255,  # dynamic
     6: 255,  # ground
     7: 0,    # road
     8: 1,    # sidewalk
     9: 255,  # parking
    10: 255,  # rail track
    11: 2,    # building
    12: 3,    # wall
    13: 4,    # fence
    14: 255,  # guard rail
    15: 255,  # bridge
    16: 255,  # tunnel
    17: 5,    # pole
    18: 255,  # polegroup
    19: 6,    # traffic light
    20: 7,    # traffic sign
    21: 8,    # vegetation
    22: 9,    # terrain
    23: 10,   # sky
    24: 11,   # person
    25: 12,   # rider
    26: 13,   # car
    27: 14,   # truck
    28: 15,   # bus
    29: 255,  # caravan
    30: 255,  # trailer
    31: 16,   # train
    32: 17,   # motorcycle
    33: 18,   # bicycle
}

# Precompute lookup array for fast vectorized mapping
LOOKUP_TABLE = np.full(256, 255, dtype=np.uint8)
for label_id, train_id in CITYSCAPES_LABEL_TO_TRAIN_ID.items():
    if label_id >= 0:
        LOOKUP_TABLE[label_id] = train_id

# Expected dataset normalization constants from Equi4Rob/dataloaders/dataloader.py
NORMALIZATION_INFO = {
    "mean": [0.29010095242892997, 0.32808144844279574, 0.28696394422942517],
    "std": [0.1829540508368939, 0.18656561047509476, 0.18447508988480435]
}


def find_triples(input_dir, split="val", sample_limit=None):
    """
    Finds matching (RGB, labelIds, disparity) triples for the given split.
    Returns a sorted list of relative path dictionaries.
    """
    left_dir = os.path.join(input_dir, "leftImg8bit", split)
    if not os.path.exists(left_dir):
        raise FileNotFoundError(f"Directory not found: {left_dir}")

    triples = []
    cities = sorted(os.listdir(left_dir))
    for city in cities:
        city_p = os.path.join(left_dir, city)
        if not os.path.isdir(city_p):
            continue
        for fname in sorted(os.listdir(city_p)):
            if not fname.endswith("_leftImg8bit.png"):
                continue

            base_name = fname.replace("_leftImg8bit.png", "")
            rgb_rel = os.path.join("leftImg8bit", split, city, f"{base_name}_leftImg8bit.png").replace("\\", "/")
            lbl_rel = os.path.join("gtFine", split, city, f"{base_name}_gtFine_labelIds.png").replace("\\", "/")
            disp_rel = os.path.join("disparity", split, city, f"{base_name}_disparity.png").replace("\\", "/")

            # Verify raw files exist
            rgb_full = os.path.join(input_dir, rgb_rel)
            lbl_full = os.path.join(input_dir, lbl_rel)
            disp_full = os.path.join(input_dir, disp_rel)

            if not os.path.exists(lbl_full):
                raise FileNotFoundError(f"Missing label for {fname}: {lbl_full}")
            if not os.path.exists(disp_full):
                raise FileNotFoundError(f"Missing disparity for {fname}: {disp_full}")

            triples.append({
                "base_name": base_name,
                "city": city,
                "split": split,
                "rgb_rel": rgb_rel,
                "lbl_in_rel": lbl_rel,
                "disp_in_rel": disp_rel,
                # Output label filename uses _gtFine_labelTrainIds.png
                "lbl_out_rel": os.path.join("gtFine", split, city, f"{base_name}_gtFine_labelTrainIds.png").replace("\\", "/"),
                "disp_out_rel": disp_rel
            })

            if sample_limit is not None and len(triples) >= sample_limit:
                return triples

    return triples


def process_single_triple(triple, input_dir, output_dir, target_size=(680, 340), overwrite=False):
    """
    Processes one (RGB, label, disparity) triple with resizing and trainId remapping.
    Resumable: skips files that already exist unless overwrite=True.
    """
    w, h = target_size
    rgb_out_p = os.path.join(output_dir, triple["rgb_rel"])
    lbl_out_p = os.path.join(output_dir, triple["lbl_out_rel"])
    disp_out_p = os.path.join(output_dir, triple["disp_out_rel"])

    os.makedirs(os.path.dirname(rgb_out_p), exist_ok=True)
    os.makedirs(os.path.dirname(lbl_out_p), exist_ok=True)
    os.makedirs(os.path.dirname(disp_out_p), exist_ok=True)

    # Check if all three exist
    all_exist = os.path.exists(rgb_out_p) and os.path.exists(lbl_out_p) and os.path.exists(disp_out_p)
    if not overwrite and all_exist:
        return False  # Skipped

    # 1. Process RGB
    if overwrite or not os.path.exists(rgb_out_p):
        rgb_in_p = os.path.join(input_dir, triple["rgb_rel"])
        with Image.open(rgb_in_p) as im:
            if im.mode != "RGB":
                im = im.convert("RGB")
            im_resized = im.resize((w, h), resample=RESAMPLE_BILINEAR)
            im_resized.save(rgb_out_p)

    # 2. Process Label (remap to trainId, then nearest-neighbor resize)
    if overwrite or not os.path.exists(lbl_out_p):
        lbl_in_p = os.path.join(input_dir, triple["lbl_in_rel"])
        with Image.open(lbl_in_p) as lbl_im:
            lbl_arr = np.array(lbl_im, dtype=np.uint8)
            train_id_arr = LOOKUP_TABLE[lbl_arr]
            train_id_im = Image.fromarray(train_id_arr, mode="L")
            train_id_resized = train_id_im.resize((w, h), resample=RESAMPLE_NEAREST)
            train_id_resized.save(lbl_out_p)

    # 3. Process Disparity (nearest-neighbor resize, preserve 16-bit integer depth)
    if overwrite or not os.path.exists(disp_out_p):
        disp_in_p = os.path.join(input_dir, triple["disp_in_rel"])
        with Image.open(disp_in_p) as disp_im:
            disp_resized = disp_im.resize((w, h), resample=RESAMPLE_NEAREST)
            disp_resized.save(disp_out_p)

    return True  # Processed


def write_manifests(triples, output_dir, split="val"):
    """
    Writes val_images.txt and val_labels.txt manifests using relative POSIX paths.
    """
    img_manifest_p = os.path.join(output_dir, f"{split}_images.txt")
    lbl_manifest_p = os.path.join(output_dir, f"{split}_labels.txt")

    with open(img_manifest_p, "w", encoding="utf-8") as f_img:
        for t in triples:
            f_img.write(f"{t['rgb_rel']}\n")

    with open(lbl_manifest_p, "w", encoding="utf-8") as f_lbl:
        for t in triples:
            f_lbl.write(f"{t['lbl_out_rel']}\n")

    return img_manifest_p, lbl_manifest_p


def write_info_json(output_dir):
    """
    Writes info.json with exact normalization constants expected by dataloader.py.
    """
    info_p = os.path.join(output_dir, "info.json")
    with open(info_p, "w", encoding="utf-8") as f:
        json.dump(NORMALIZATION_INFO, f, indent=2)
    return info_p


def verify_processed_dataset(output_dir, split="val", expected_size=(680, 340)):
    """
    Runs automated sanity verification on the processed dataset.
    """
    print("\n" + "=" * 60)
    print("RUNNING AUTOMATED PREPROCESSING VERIFICATION")
    print("=" * 60)

    w_exp, h_exp = expected_size
    info_p = os.path.join(output_dir, "info.json")
    img_manifest_p = os.path.join(output_dir, f"{split}_images.txt")
    lbl_manifest_p = os.path.join(output_dir, f"{split}_labels.txt")

    assert os.path.exists(info_p), f"Missing {info_p}"
    assert os.path.exists(img_manifest_p), f"Missing {img_manifest_p}"
    assert os.path.exists(lbl_manifest_p), f"Missing {lbl_manifest_p}"

    info = json.load(open(info_p, "r"))
    assert "mean" in info and "std" in info, "info.json missing mean or std"

    with open(img_manifest_p, "r") as f:
        img_lines = [line.strip() for line in f if line.strip()]
    with open(lbl_manifest_p, "r") as f:
        lbl_lines = [line.strip() for line in f if line.strip()]

    assert len(img_lines) == len(lbl_lines), f"Manifest length mismatch: {len(img_lines)} vs {len(lbl_lines)}"
    total = len(img_lines)
    print(f"Manifest entry count: {total}")

    for idx, (img_rel, lbl_rel) in enumerate(zip(img_lines, lbl_lines)):
        img_p = os.path.join(output_dir, img_rel)
        lbl_p = os.path.join(output_dir, lbl_rel)
        disp_rel = img_rel.replace("leftImg8bit", "disparity")
        disp_p = os.path.join(output_dir, disp_rel)

        assert os.path.exists(img_p), f"Image not found: {img_p}"
        assert os.path.exists(lbl_p), f"Label not found: {lbl_p}"
        assert os.path.exists(disp_p), f"Disparity not found: {disp_p}"

        with Image.open(img_p) as im:
            assert im.size == (w_exp, h_exp), f"Wrong image size: {im.size} in {img_p}"
            assert im.mode == "RGB", f"Wrong image mode: {im.mode} in {img_p}"

        with Image.open(lbl_p) as lbl:
            assert lbl.size == (w_exp, h_exp), f"Wrong label size: {lbl.size} in {lbl_p}"
            lbl_arr = np.array(lbl)
            unique_vals = set(np.unique(lbl_arr))
            invalid_vals = unique_vals - (set(range(19)) | {255})
            assert not invalid_vals, f"Invalid trainId values found: {invalid_vals} in {lbl_p}"

        with Image.open(disp_p) as disp:
            assert disp.size == (w_exp, h_exp), f"Wrong disparity size: {disp.size} in {disp_p}"
            assert disp.mode in ("I;16", "I"), f"Wrong disparity mode: {disp.mode} in {disp_p}"

    print(f"[OK] All {total} processed samples verified successfully.")
    print(f"     Dimensions: ({w_exp}, {h_exp})")
    print(f"     Image format: RGB PNG")
    print(f"     Label format: 19 TrainIDs (0-18, 255 ignore)")
    print(f"     Disparity format: 16-bit depth (mode I;16)")
    print(f"     Manifests: relative POSIX paths, 1:1 match")
    print(f"     info.json: Validated")
    print("=" * 60 + "\n")
    return True


def main():
    parser = argparse.ArgumentParser(description="Preprocess Cityscapes for Robust Perception Reproduction")
    parser.add_argument("--input_dir", type=str, default="cityscapes_dataset", help="Input raw dataset path")
    parser.add_argument("--output_dir", type=str, default="cityscapes_dataset_subsampled", help="Output subsampled dataset path")
    parser.add_argument("--split", type=str, default="val", choices=["val", "train"], help="Dataset split to process")
    parser.add_argument("--sample", type=int, default=None, help="Process only the first N samples (e.g. 10)")
    parser.add_argument("--width", type=int, default=680, help="Target image width")
    parser.add_argument("--height", type=int, default=340, help="Target image height")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output files")
    parser.add_argument("--verify", action="store_true", help="Run verification check after processing")
    args = parser.parse_args()

    print(f"Cityscapes Preprocessing Configuration:")
    print(f"  Input Directory:  {args.input_dir}")
    print(f"  Output Directory: {args.output_dir}")
    print(f"  Split:            {args.split}")
    print(f"  Sample Limit:     {args.sample if args.sample is not None else 'Full split'}")
    print(f"  Target Size:      ({args.width}, {args.height})")
    print(f"  Overwrite:        {args.overwrite}\n")

    triples = find_triples(args.input_dir, split=args.split, sample_limit=args.sample)
    total = len(triples)
    print(f"Found {total} {args.split} triples to process.")

    errors = []
    processed_count = 0
    skipped_count = 0

    for idx, triple in enumerate(triples, 1):
        try:
            was_processed = process_single_triple(triple, args.input_dir, args.output_dir,
                                                  target_size=(args.width, args.height),
                                                  overwrite=args.overwrite)
            if was_processed:
                processed_count += 1
            else:
                skipped_count += 1

            if idx % 50 == 0 or idx == total or total <= 10:
                print(f"  [{idx}/{total}] Progress update (newly processed: {processed_count}, skipped: {skipped_count})")
        except Exception as e:
            err_msg = f"Failed on {triple['base_name']}: {str(e)}"
            print(f"[ERROR] {err_msg}", file=sys.stderr)
            errors.append(err_msg)

    print(f"\nProcessing Summary:")
    print(f"  Total items:     {total}")
    print(f"  Newly processed: {processed_count}")
    print(f"  Skipped (cached):{skipped_count}")
    print(f"  Errors:          {len(errors)}")

    if errors:
        print(f"\n[ERROR] Preprocessing completed with {len(errors)} error(s):", file=sys.stderr)
        for err in errors:
            print(f"   - {err}", file=sys.stderr)
        sys.exit(1)

    # Write manifests and info.json
    img_m, lbl_m = write_manifests(triples, args.output_dir, split=args.split)
    info_p = write_info_json(args.output_dir)
    print(f"\nManifests generated:")
    print(f"  - {img_m}")
    print(f"  - {lbl_m}")
    print(f"  - {info_p}")

    # Automated verification
    if args.verify or args.sample is not None:
        verify_processed_dataset(args.output_dir, split=args.split, expected_size=(args.width, args.height))

    print("Preprocessing completed successfully.")


if __name__ == "__main__":
    main()
