#!/usr/bin/env python3
"""
DRN-D-22 Vanilla Baseline Evaluation on Subsampled Cityscapes.

Evaluates the clean pre-trained DRN-D-22 model on the Cityscapes validation set
(500 subsampled 680x340 images) with no adversarial attack and no test-time defense.

Computes:
  - Global mean Intersection-over-Union (mIoU) across 19 Cityscapes classes
  - Per-class IoU for all 19 semantic categories
  - Overall pixel accuracy (ignoring void label 255)
  - Inference runtime and throughput (FPS)
  - Colorized segmentation visualizations and comparison panels
"""

import os
import sys
import time
import json
import argparse
from typing import Dict, List, Tuple

import numpy as np
import torch
from PIL import Image

# Add Equi4Rob to path for models and utilities
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
EQUI4ROB_DIR = os.path.join(ROOT_DIR, "Equi4Rob")
if EQUI4ROB_DIR not in sys.path:
    sys.path.insert(0, EQUI4ROB_DIR)

from models.DRNSeg import DRNSeg
from learning.utils_learn import fast_hist, per_class_iu

# Cityscapes 19 evaluation classes
CITYSCAPES_CLASSES: List[str] = [
    "road",          # 0
    "sidewalk",      # 1
    "building",      # 2
    "wall",          # 3
    "fence",         # 4
    "pole",          # 5
    "traffic light", # 6
    "traffic sign",  # 7
    "vegetation",    # 8
    "terrain",       # 9
    "sky",           # 10
    "person",        # 11
    "rider",         # 12
    "car",           # 13
    "truck",         # 14
    "bus",           # 15
    "train",         # 16
    "motorcycle",    # 17
    "bicycle",       # 18
]

# Official Cityscapes 19-class RGB color palette
CITYSCAPES_PALETTE: np.ndarray = np.array([
    [128,  64, 128],  # 0: road
    [244,  35, 232],  # 1: sidewalk
    [ 70,  70,  70],  # 2: building
    [102, 102, 156],  # 3: wall
    [190, 153, 153],  # 4: fence
    [153, 153, 153],  # 5: pole
    [250, 170,  30],  # 6: traffic light
    [220, 220,   0],  # 7: traffic sign
    [107, 142,  35],  # 8: vegetation
    [152, 251, 152],  # 9: terrain
    [  0, 130, 180],  # 10: sky
    [220,  20,  60],  # 11: person
    [255,   0,   0],  # 12: rider
    [  0,   0, 142],  # 13: car
    [  0,   0,  70],  # 14: truck
    [  0,  60, 100],  # 15: bus
    [  0,  80, 100],  # 16: train
    [  0,   0, 230],  # 17: motorcycle
    [119,  11,  32],  # 18: bicycle
], dtype=np.uint8)


def colorize_segmentation(mask: np.ndarray) -> Image.Image:
    """Colorize a 2D integer class mask using Cityscapes palette (void 255 -> black)."""
    h, w = mask.shape
    color_img = np.zeros((h, w, 3), dtype=np.uint8)
    for class_id in range(len(CITYSCAPES_PALETTE)):
        color_img[mask == class_id] = CITYSCAPES_PALETTE[class_id]
    return Image.fromarray(color_img)


def create_comparison_panel(
    rgb: Image.Image,
    gt: np.ndarray,
    pred: np.ndarray,
    sample_id: str
) -> Image.Image:
    """
    Creates a 4-panel side-by-side comparison image:
      [Original RGB | Ground Truth | Prediction | Error Overlay]
    """
    gt_color = colorize_segmentation(gt)
    pred_color = colorize_segmentation(pred)

    # Build error overlay (dim valid pixels, highlight errors in red, void in black)
    rgb_np = np.array(rgb, dtype=np.float32)
    error_np = rgb_np * 0.4  # dimmed background
    valid_mask = (gt >= 0) & (gt < 19)
    error_mask = valid_mask & (pred != gt)
    correct_mask = valid_mask & (pred == gt)

    # Green tint for correct, red highlight for errors
    error_np[error_mask] = [230, 40, 40]
    error_np[correct_mask] = error_np[correct_mask] * 0.7 + np.array([20, 140, 20], dtype=np.float32) * 0.3
    error_img = Image.fromarray(np.clip(error_np, 0, 255).astype(np.uint8))

    w, h = rgb.size
    panel = Image.new("RGB", (w * 4, h))
    panel.paste(rgb, (0, 0))
    panel.paste(gt_color, (w, 0))
    panel.paste(pred_color, (w * 2, 0))
    panel.paste(error_img, (w * 3, 0))
    return panel


def load_model(checkpoint_path: str, device: torch.device) -> torch.nn.Module:
    """Instantiate DRNSeg and strictly load pretrained clean weights."""
    model = DRNSeg("drn_d_22", classes=19, pretrained=False)
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    if "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    model.load_state_dict(state_dict, strict=True)
    model.to(device).eval()
    return model


def run_smoke_test(
    data_dir: str,
    checkpoint_path: str,
    num_samples: int = 5,
    device: torch.device = torch.device("cuda")
) -> bool:
    """
    Runs a detailed smoke test on 2-5 validation samples to verify:
      1. Image-label pairing and existence.
      2. Normalization metadata and tensor scales.
      3. Input and output shapes, including the 344 -> 340 height resolution.
      4. Class IDs and ignore-label masking.
      5. Correct confusion matrix and mIoU computation.
    """
    print("=" * 70)
    print("RUNNING BASELINE SMOKE TEST (2-5 SAMPLES)")
    print("=" * 70)

    # 1. Load info.json normalization
    info_path = os.path.join(data_dir, "info.json")
    assert os.path.exists(info_path), f"Missing info.json at {info_path}"
    with open(info_path, "r") as f:
        info = json.load(f)
    mean = torch.tensor(info["mean"], dtype=torch.float32, device=device).view(1, 3, 1, 1)
    std = torch.tensor(info["std"], dtype=torch.float32, device=device).view(1, 3, 1, 1)
    print(f"[OK] Normalization loaded: mean={info['mean']}, std={info['std']}")

    # 2. Check manifests
    images_txt = os.path.join(data_dir, "val_images.txt")
    labels_txt = os.path.join(data_dir, "val_labels.txt")
    assert os.path.exists(images_txt), f"Missing {images_txt}"
    assert os.path.exists(labels_txt), f"Missing {labels_txt}"

    with open(images_txt, "r") as f:
        img_lines = [line.strip() for line in f if line.strip()]
    with open(labels_txt, "r") as f:
        lbl_lines = [line.strip() for line in f if line.strip()]
    assert len(img_lines) == len(lbl_lines), "Manifest length mismatch!"
    print(f"[OK] Manifests verified: {len(img_lines)} total validation pairs.")

    # 3. Load model
    print(f"Loading checkpoint: {checkpoint_path}...")
    model = load_model(checkpoint_path, device)
    print("[OK] Model instantiated and weights loaded strictly.")

    # 4. Test sample processing
    test_samples = min(num_samples, len(img_lines))
    hist = np.zeros((19, 19), dtype=np.int64)

    print(f"\nEvaluating first {test_samples} validation samples:")
    for idx in range(test_samples):
        img_rel = img_lines[idx]
        lbl_rel = lbl_lines[idx]
        img_full = os.path.join(data_dir, img_rel)
        lbl_full = os.path.join(data_dir, lbl_rel)

        assert os.path.exists(img_full), f"Image not found: {img_full}"
        assert os.path.exists(lbl_full), f"Label not found: {lbl_full}"

        pil_img = Image.open(img_full).convert("RGB")
        pil_lbl = Image.open(lbl_full)
        lbl_np = np.array(pil_lbl, dtype=np.int32)

        # Dimension checks
        w_img, h_img = pil_img.size
        w_lbl, h_lbl = pil_lbl.size
        assert (w_img, h_img) == (680, 340), f"Unexpected image size: {(w_img, h_img)}"
        assert (w_lbl, h_lbl) == (680, 340), f"Unexpected label size: {(w_lbl, h_lbl)}"

        # Prepare tensor
        img_t = torch.from_numpy(np.array(pil_img, dtype=np.float32).transpose(2, 0, 1) / 255.0).unsqueeze(0).to(device)
        img_norm = (img_t - mean) / std

        with torch.no_grad():
            raw_out = model(img_norm)[0]

        # Raw output check
        assert raw_out.shape == (1, 19, 344, 680), f"Unexpected raw output shape: {raw_out.shape}"

        # Resizing verification: slice bottom 4 padding rows
        sliced_out = raw_out[:, :, :340, :680]
        assert sliced_out.shape == (1, 19, 340, 680), f"Sliced output shape incorrect: {sliced_out.shape}"

        pred_mask = sliced_out.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.int32)
        assert pred_mask.shape == (340, 680), f"Pred mask shape incorrect: {pred_mask.shape}"

        # Class ID checks
        pred_unique = np.unique(pred_mask)
        lbl_unique = np.unique(lbl_np)
        assert np.all((pred_unique >= 0) & (pred_unique < 19)), f"Predictions out of range: {pred_unique}"
        valid_lbl = (lbl_unique >= 0) & (lbl_unique < 19) | (lbl_unique == 255)
        assert np.all(valid_lbl), f"Invalid label IDs found: {lbl_unique}"

        # Accumulate confusion matrix
        step_hist = fast_hist(pred_mask.flatten(), lbl_np.flatten(), 19)
        hist += step_hist

        sample_iou = np.nanmean(per_class_iu(step_hist)) * 100
        print(f"  Sample {idx + 1}/{test_samples} [{os.path.basename(img_rel)}]: "
              f"Input={img_norm.shape}, RawOut={raw_out.shape}, Sliced={sliced_out.shape}, "
              f"Sample mIoU={sample_iou:.2f}%")

    smoke_miou = np.nanmean(per_class_iu(hist)) * 100
    print(f"\n[OK] Smoke test PASSED: Accumulated mIoU ({test_samples} samples) = {smoke_miou:.2f}%")
    print("=" * 70)
    return True


def evaluate_full_dataset(
    data_dir: str,
    checkpoint_path: str,
    output_dir: str,
    batch_size: int = 4,
    device: torch.device = torch.device("cuda"),
    num_visualizations: int = 5
) -> Dict:
    """
    Evaluates all 500 validation images:
      - Measures total runtime, batch latency, throughput
      - Calculates mIoU, per-class IoU, pixel accuracy
      - Saves results to JSON, formatted summary text, and visualizations
    """
    print("\n" + "=" * 70)
    print("STARTING FULL BASELINE EVALUATION (500 SAMPLES)")
    print("=" * 70)

    os.makedirs(output_dir, exist_ok=True)
    vis_dir = os.path.join(output_dir, "visualizations")
    os.makedirs(vis_dir, exist_ok=True)

    # 1. Load normalization & manifests
    info_path = os.path.join(data_dir, "info.json")
    with open(info_path, "r") as f:
        info = json.load(f)
    mean = torch.tensor(info["mean"], dtype=torch.float32, device=device).view(1, 3, 1, 1)
    std = torch.tensor(info["std"], dtype=torch.float32, device=device).view(1, 3, 1, 1)

    images_txt = os.path.join(data_dir, "val_images.txt")
    labels_txt = os.path.join(data_dir, "val_labels.txt")
    with open(images_txt, "r") as f:
        img_lines = [line.strip() for line in f if line.strip()]
    with open(labels_txt, "r") as f:
        lbl_lines = [line.strip() for line in f if line.strip()]
    total_images = len(img_lines)
    print(f"Total samples to evaluate: {total_images} (Batch size = {batch_size})")

    # 2. Load model
    print(f"Loading checkpoint: {checkpoint_path}...")
    model = load_model(checkpoint_path, device)

    # 3. Inference loop
    hist = np.zeros((19, 19), dtype=np.int64)
    total_correct = 0
    total_valid_pixels = 0
    saved_vis_count = 0

    start_time = time.time()
    batch_latencies = []

    num_batches = (total_images + batch_size - 1) // batch_size

    for b_idx in range(num_batches):
        b_start = b_idx * batch_size
        b_end = min(b_start + batch_size, total_images)
        current_bs = b_end - b_start

        batch_imgs = []
        batch_lbls = []
        pil_imgs_raw = []

        t0 = time.time()
        for idx in range(b_start, b_end):
            img_path = os.path.join(data_dir, img_lines[idx])
            lbl_path = os.path.join(data_dir, lbl_lines[idx])

            pil_img = Image.open(img_path).convert("RGB")
            pil_lbl = Image.open(lbl_path)

            img_arr = np.array(pil_img, dtype=np.float32).transpose(2, 0, 1) / 255.0
            lbl_arr = np.array(pil_lbl, dtype=np.int32)

            batch_imgs.append(img_arr)
            batch_lbls.append(lbl_arr)
            if saved_vis_count < num_visualizations:
                pil_imgs_raw.append((pil_img, lbl_arr, img_lines[idx]))

        batch_tensor = torch.from_numpy(np.stack(batch_imgs, axis=0)).to(device)
        batch_norm = (batch_tensor - mean) / std

        with torch.no_grad():
            out = model(batch_norm)[0]  # (B, 19, 344, 680)

        # Slice height from 344 -> 340
        out_sliced = out[:, :, :340, :680]
        preds = out_sliced.argmax(dim=1).cpu().numpy().astype(np.int32)  # (B, 340, 680)

        t1 = time.time()
        batch_latencies.append((t1 - t0) / current_bs)

        # Accumulate metrics
        for i in range(current_bs):
            p = preds[i]
            y = batch_lbls[i]

            step_hist = fast_hist(p.flatten(), y.flatten(), 19)
            hist += step_hist

            # Pixel accuracy
            valid = (y >= 0) & (y < 19)
            total_correct += int(np.sum((p == y) & valid))
            total_valid_pixels += int(np.sum(valid))

        # Save visualizations for the first few samples
        for pil_img, gt_arr, img_name in pil_imgs_raw:
            if saved_vis_count >= num_visualizations:
                break
            sample_pred = preds[saved_vis_count - b_start]
            base_name = os.path.splitext(os.path.basename(img_name))[0]

            # Save individual colored prediction and comparison panel
            pred_color = colorize_segmentation(sample_pred)
            pred_color.save(os.path.join(vis_dir, f"{base_name}_pred.png"))

            panel = create_comparison_panel(pil_img, gt_arr, sample_pred, base_name)
            panel.save(os.path.join(vis_dir, f"{base_name}_comparison.png"))
            saved_vis_count += 1

        if (b_idx + 1) % 25 == 0 or (b_idx + 1) == num_batches:
            current_miou = np.nanmean(per_class_iu(hist)) * 100
            elapsed = time.time() - start_time
            print(f"Processed [{b_end}/{total_images}] images | "
                  f"Elapsed: {elapsed:.1f}s | "
                  f"Current mIoU: {current_miou:.2f}%")

    total_time = time.time() - start_time
    avg_latency = (total_time / total_images) * 1000  # ms/image
    fps = total_images / total_time

    # Compute final metrics
    ious = per_class_iu(hist) * 100
    mean_iou = float(np.nanmean(ious))
    pixel_acc = (total_correct / total_valid_pixels) * 100 if total_valid_pixels > 0 else 0.0

    per_class_results = {}
    for idx, name in enumerate(CITYSCAPES_CLASSES):
        per_class_results[name] = round(float(ious[idx]), 2)

    results = {
        "experiment": "Vanilla DRN-D-22 Baseline Evaluation",
        "dataset": "Cityscapes Subsampled (val)",
        "num_samples": total_images,
        "input_resolution": "680x340",
        "checkpoint": os.path.basename(checkpoint_path),
        "mean_iou_percent": round(mean_iou, 2),
        "pixel_accuracy_percent": round(pixel_acc, 2),
        "total_runtime_seconds": round(total_time, 2),
        "latency_ms_per_image": round(avg_latency, 2),
        "throughput_fps": round(fps, 2),
        "per_class_iou_percent": per_class_results,
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    }

    # 4. Save results JSON
    json_path = os.path.join(output_dir, "baseline_drn_d_22_metrics.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[Saved] Metrics JSON: {json_path}")

    # 5. Save human-readable summary text
    summary_path = os.path.join(output_dir, "baseline_drn_d_22_summary.txt")
    with open(summary_path, "w") as f:
        f.write("=" * 60 + "\n")
        f.write("    VANILLA DRN-D-22 BASELINE EVALUATION SUMMARY\n")
        f.write("=" * 60 + "\n")
        f.write(f"Dataset:            Cityscapes Subsampled (Validation Split)\n")
        f.write(f"Checkpoint:         {checkpoint_path}\n")
        f.write(f"Evaluation Samples: {total_images}\n")
        f.write(f"Input Resolution:   680 x 340\n")
        f.write(f"Device:             {results['device']}\n")
        f.write(f"Total Runtime:      {total_time:.2f} s ({avg_latency:.1f} ms/sample, {fps:.1f} FPS)\n")
        f.write("-" * 60 + "\n")
        f.write(f"Global mIoU:        {mean_iou:.2f}%\n")
        f.write(f"Pixel Accuracy:     {pixel_acc:.2f}%\n")
        f.write("-" * 60 + "\n")
        f.write("PER-CLASS IoU (%):\n")
        for idx, (name, val) in enumerate(per_class_results.items()):
            f.write(f"  {idx:2d}. {name:<16s}: {val:6.2f}%\n")
        f.write("=" * 60 + "\n")
    print(f"[Saved] Summary report: {summary_path}")
    print(f"[Saved] Visualizations: {vis_dir} ({saved_vis_count} sample comparisons saved)")

    print("\n" + "=" * 60)
    print(f"FINAL RESULT: Global mIoU = {mean_iou:.2f}% | Pixel Acc = {pixel_acc:.2f}%")
    print("=" * 60)

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate Vanilla DRN-D-22 on Subsampled Cityscapes.")
    parser.add_argument("--data-dir", type=str, default="cityscapes_dataset_subsampled",
                        help="Path to preprocessed subsampled Cityscapes dataset.")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/clean_drn_d_22_cityscapes.pth",
                        help="Path to clean DRN-D-22 checkpoint.")
    parser.add_argument("--output-dir", type=str, default="results",
                        help="Directory to save evaluation results and visualizations.")
    parser.add_argument("--batch-size", type=int, default=4,
                        help="Inference batch size.")
    parser.add_argument("--smoke-test", action="store_true",
                        help="Run verification smoke test on first few samples only.")
    parser.add_argument("--num-samples", type=int, default=5,
                        help="Number of samples to run during smoke test.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.smoke_test:
        success = run_smoke_test(
            data_dir=args.data_dir,
            checkpoint_path=args.checkpoint,
            num_samples=args.num_samples,
            device=device
        )
        if not success:
            sys.exit(1)
    else:
        # Step 1: Run smoke test first
        print("Executing preliminary smoke test verification...")
        success = run_smoke_test(
            data_dir=args.data_dir,
            checkpoint_path=args.checkpoint,
            num_samples=5,
            device=device
        )
        if not success:
            print("Smoke test failed! Aborting full evaluation.")
            sys.exit(1)

        # Step 2: Run full evaluation
        evaluate_full_dataset(
            data_dir=args.data_dir,
            checkpoint_path=args.checkpoint,
            output_dir=args.output_dir,
            batch_size=args.batch_size,
            device=device,
            num_visualizations=5
        )


if __name__ == "__main__":
    main()
