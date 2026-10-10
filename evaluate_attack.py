#!/usr/bin/env python3
"""
DRN-D-22 Adversarial PGD Attack Evaluation on Subsampled Cityscapes.

Evaluates the adversarially trained DRN-D-22 model (or clean pretrained model) under
an untargeted L-infinity PGD attack on the 500 Cityscapes validation images (680x340).

Experiment Configuration:
  - Checkpoint: checkpoints/advtrain_drn_d_22_cityscapes.pth.tar
  - Attack: Untargeted PGD, L-infinity norm
  - Epsilon: 4/255 (~0.015686)
  - Step size: 1/255 (~0.003922)
  - Steps: 50 iterations
  - Random start: Uniform in [-epsilon, +epsilon]
  - Defense: None (Vanilla PGD baseline)
  - Output alignment: Differentiable bilinear interpolation to (340, 680)
    inside the attack loss and final prediction evaluation
  - Metrics: Global confusion-matrix mIoU across 19 Cityscapes classes (ignore label 255)
    and overall pixel accuracy.

Paper Reference:
  - ICML 2023 ("Robust Perception through Equivariance", Mao et al.), Table 1:
    Adversarially trained DRN-22-d on Cityscapes under Vanilla PGD: 26.61% mIoU.
"""

import os
import sys
import time
import json
import argparse
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

# Ensure Equi4Rob modules are accessible
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
EQUI4ROB_DIR = os.path.join(ROOT_DIR, "Equi4Rob")
if EQUI4ROB_DIR not in sys.path:
    sys.path.insert(0, EQUI4ROB_DIR)

# Ensure line-buffered output so logs are immediately flushed
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

import traceback

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


def align_output(out: torch.Tensor, target_h: int = 340, target_w: int = 680, mode: str = "bilinear") -> torch.Tensor:
    """
    Aligns raw model output (B, C, 344, 680) with target label dimensions (target_h, target_w).
    Uses differentiable bilinear interpolation by default to preserve gradient backpropagation.
    """
    if out.shape[2] == target_h and out.shape[3] == target_w:
        return out

    if mode == "bilinear":
        return F.interpolate(out, size=(target_h, target_w), mode="bilinear", align_corners=False)
    elif mode == "slice":
        return out[:, :, :target_h, :target_w]
    else:
        raise ValueError(f"Unsupported alignment mode '{mode}'. Choose 'bilinear' or 'slice'.")


def load_model(checkpoint_path: str, device: torch.device) -> Tuple[torch.nn.Module, Dict]:
    """
    Instantiate DRNSeg and strictly load pretrained weights.
    Supports both direct state_dict (.pth) and training checkpoints (.pth.tar with 'module.' prefixes).
    Returns (model, metadata_dict).
    """
    model = DRNSeg("drn_d_22", classes=19, pretrained=False)
    raw = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    metadata = {
        "checkpoint_path": checkpoint_path,
        "checkpoint_filename": os.path.basename(checkpoint_path),
        "epoch": None,
        "arch": "drn_d_22",
        "best_prec1": None,
        "has_module_prefix": False,
        "missing_keys": [],
        "unexpected_keys": [],
    }

    if isinstance(raw, dict) and "state_dict" in raw:
        state_dict = raw["state_dict"]
        metadata["epoch"] = raw.get("epoch")
        metadata["arch"] = raw.get("arch", "drn_d_22")
        bp1 = raw.get("best_prec1")
        metadata["best_prec1"] = float(bp1) if bp1 is not None else None
    elif isinstance(raw, dict):
        state_dict = raw
    else:
        raise ValueError(f"Unexpected checkpoint format in {checkpoint_path}: {type(raw)}")

    clean_state_dict = {}
    has_module_prefix = False
    for k, v in state_dict.items():
        if k.startswith("module."):
            clean_state_dict[k[7:]] = v
            has_module_prefix = True
        else:
            clean_state_dict[k] = v
    metadata["has_module_prefix"] = has_module_prefix

    load_result = model.load_state_dict(clean_state_dict, strict=True)
    metadata["missing_keys"] = list(load_result.missing_keys)
    metadata["unexpected_keys"] = list(load_result.unexpected_keys)

    model.to(device).eval()

    print(f"[Model Load] Checkpoint: {os.path.basename(checkpoint_path)}")
    if metadata.get("epoch") is not None:
        print(f"             Epoch: {metadata['epoch']}, best_prec1: {metadata.get('best_prec1')}")
    print(f"             Strict load OK (0 missing, 0 unexpected keys)")

    return model, metadata


def generate_pgd_perturbation(
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
    epsilon: float = 4.0 / 255.0,
    step_size: float = 1.0 / 255.0,
    steps: int = 50,
    target_h: int = 340,
    target_w: int = 680,
    random_start: bool = True,
    alignment: str = "bilinear",
    track_history: bool = False
) -> Tuple[torch.Tensor, List[float]]:
    """
    Generates untargeted L-infinity PGD adversarial perturbation.

    Args:
      model: Evaluation model (DRNSeg returning LogSoftmax in index 0).
      images: Unnormalized input tensor in [0.0, 1.0], shape (B, 3, H, W).
      labels: Ground truth semantic class labels, shape (B, H, W), ignore_index=255.
      mean, std: Normalization constants (1, 3, 1, 1).
      epsilon: L-infinity perturbation bound (e.g. 4/255).
      step_size: Step size per iteration (e.g. 1/255).
      steps: Number of PGD iterations (e.g. 50).
      target_h, target_w: Target spatial resolution for loss computation.
      random_start: Whether to initialize with uniform noise in [-eps, eps].
      alignment: Interpolation mode ('bilinear').
      track_history: Whether to record loss at each step.

    Returns:
      perturbed_images: Adversarial images in [0.0, 1.0], shape (B, 3, H, W).
      loss_history: List of loss values across iterations.
    """
    criterion = nn.NLLLoss(ignore_index=255)
    x = images.clone().detach()

    delta = torch.zeros_like(x)
    if random_start:
        delta.uniform_(-epsilon, epsilon)
    # Project initial perturbation onto valid image space [0, 1]
    delta = torch.clamp(delta, 0.0 - x, 1.0 - x)

    loss_history: List[float] = []

    for step in range(steps):
        delta = delta.detach().requires_grad_(True)
        x_adv = x + delta

        # Apply dataset normalization inside forward pass
        x_norm = (x_adv - mean) / std

        # Forward pass through DRNSeg; index 0 contains LogSoftmax predictions
        raw_out = model(x_norm)[0]

        # Differentiable spatial alignment to label dimensions (344 -> 340)
        aligned_out = align_output(raw_out, target_h=target_h, target_w=target_w, mode=alignment)

        loss = criterion(aligned_out, labels)
        if track_history:
            loss_history.append(loss.item())

        model.zero_grad()
        loss.backward()

        # Sign gradient ascent step
        grad_sign = delta.grad.detach().sign()
        delta = delta.detach() + step_size * grad_sign

        # Projection onto L-infinity ball [-epsilon, +epsilon]
        delta = torch.clamp(delta, -epsilon, epsilon)

        # Projection onto valid image space [0.0, 1.0]
        delta = torch.clamp(delta, 0.0 - x, 1.0 - x)

    # Exact projection guarantees
    delta = torch.clamp(delta, -epsilon, epsilon)
    perturbed_images = torch.clamp(x + delta, 0.0, 1.0).detach()
    return perturbed_images, loss_history


def run_smoke_test(
    data_dir: str,
    checkpoint_path: str,
    epsilon: float = 4.0 / 255.0,
    step_size: float = 1.0 / 255.0,
    steps: int = 50,
    num_samples: int = 5,
    alignment: str = "bilinear",
    device: torch.device = torch.device("cuda"),
    seed: int = 42
) -> bool:
    """
    Runs a rigorous 5-image smoke test to verify:
      1. Image bounds and label validity.
      2. Finite loss and loss trajectory across 50 PGD steps.
      3. Non-zero gradient flow through differentiable bilinear alignment.
      4. Strict L-infinity bounds (<= 4/255) and pixel clamping ([0.0, 1.0]).
      5. Impact on mIoU and pixel accuracy under attack.
    """
    print("=" * 75)
    print(f"RUNNING PGD ATTACK SMOKE TEST ({num_samples} SAMPLES)")
    print(f"Attack: L-inf PGD, eps={epsilon:.6f} ({epsilon*255:.1f}/255), "
          f"step_size={step_size:.6f} ({step_size*255:.1f}/255), steps={steps}, seed={seed}")
    print("=" * 75)

    torch.manual_seed(seed)
    np.random.seed(seed)

    info_path = os.path.join(data_dir, "info.json")
    with open(info_path, "r") as f:
        info = json.load(f)
    mean = torch.tensor(info["mean"], dtype=torch.float32, device=device).view(1, 3, 1, 1)
    std = torch.tensor(info["std"], dtype=torch.float32, device=device).view(1, 3, 1, 1)

    images_txt = os.path.join(data_dir, "val_images.txt")
    labels_txt = os.path.join(data_dir, "val_labels.txt")
    with open(images_txt, "r") as f:
        img_lines = [line.strip() for line in f if line.strip()][:num_samples]
    with open(labels_txt, "r") as f:
        lbl_lines = [line.strip() for line in f if line.strip()][:num_samples]

    model, metadata = load_model(checkpoint_path, device)
    criterion = nn.NLLLoss(ignore_index=255)

    clean_hist = np.zeros((19, 19), dtype=np.int64)
    adv_hist = np.zeros((19, 19), dtype=np.int64)

    clean_total_correct = 0
    clean_total_pixels = 0
    adv_total_correct = 0
    adv_total_pixels = 0

    print(f"\nProcessing {num_samples} validation samples individually:")
    for idx in range(num_samples):
        img_p = os.path.join(data_dir, img_lines[idx])
        lbl_p = os.path.join(data_dir, lbl_lines[idx])

        pil_img = Image.open(img_p).convert("RGB")
        pil_lbl = Image.open(lbl_p)
        lbl_np = np.array(pil_lbl, dtype=np.int32)

        img_t = torch.from_numpy(np.array(pil_img, dtype=np.float32).transpose(2, 0, 1) / 255.0).unsqueeze(0).to(device)
        lbl_t = torch.from_numpy(lbl_np).unsqueeze(0).long().to(device)

        # 1. Clean evaluation
        with torch.no_grad():
            clean_out = model((img_t - mean) / std)[0]
            clean_aligned = align_output(clean_out, target_h=340, target_w=680, mode=alignment)
            clean_loss = criterion(clean_aligned, lbl_t).item()

        clean_pred = clean_aligned.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.int32)
        c_step_hist = fast_hist(clean_pred.flatten(), lbl_np.flatten(), 19)
        clean_hist += c_step_hist
        c_sample_miou = np.nanmean(per_class_iu(c_step_hist)) * 100

        valid_m = (lbl_np >= 0) & (lbl_np < 19)
        clean_total_correct += np.sum((clean_pred == lbl_np) & valid_m)
        clean_total_pixels += np.sum(valid_m)

        # 2. PGD Attack generation
        t0 = time.time()
        adv_img_t, loss_hist = generate_pgd_perturbation(
            model=model,
            images=img_t,
            labels=lbl_t,
            mean=mean,
            std=std,
            epsilon=epsilon,
            step_size=step_size,
            steps=steps,
            target_h=340,
            target_w=680,
            random_start=True,
            alignment=alignment,
            track_history=True
        )
        attack_time = time.time() - t0

        # Perturbation checks
        delta_np = (adv_img_t - img_t).squeeze(0).cpu().numpy()
        max_linf = np.max(np.abs(delta_np))
        assert max_linf <= epsilon + 1e-6, f"Max perturbation exceeds epsilon: {max_linf} > {epsilon}"
        assert np.all(np.isfinite(loss_hist)), "Non-finite loss encountered in attack loop!"

        # 3. Attacked evaluation
        with torch.no_grad():
            adv_out = model((adv_img_t - mean) / std)[0]
            adv_aligned = align_output(adv_out, target_h=340, target_w=680, mode=alignment)
            adv_loss = criterion(adv_aligned, lbl_t).item()

        adv_pred = adv_aligned.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.int32)
        a_step_hist = fast_hist(adv_pred.flatten(), lbl_np.flatten(), 19)
        adv_hist += a_step_hist
        a_sample_miou = np.nanmean(per_class_iu(a_step_hist)) * 100

        adv_total_correct += np.sum((adv_pred == lbl_np) & valid_m)
        adv_total_pixels += np.sum(valid_m)

        print(f"  Sample {idx+1}/{num_samples} [{os.path.basename(img_lines[idx])}]: "
              f"Time={attack_time:.2f}s | "
              f"CleanLoss={clean_loss:.3f} -> AdvLoss={adv_loss:.3f} | "
              f"Clean-mIoU={c_sample_miou:.2f}% -> Adv-mIoU={a_sample_miou:.2f}% | "
              f"MaxLinf={max_linf*255:.3f}/255")

    clean_miou = np.nanmean(per_class_iu(clean_hist)) * 100
    adv_miou = np.nanmean(per_class_iu(adv_hist)) * 100
    clean_acc = 100.0 * clean_total_correct / clean_total_pixels
    adv_acc = 100.0 * adv_total_correct / adv_total_pixels

    print("\n" + "-" * 75)
    print("SMOKE TEST SUMMARY (5 Samples):")
    print(f"  Clean Input:    mIoU = {clean_miou:.2f}%, Pixel Acc = {clean_acc:.2f}%")
    print(f"  PGD-50 Attacked: mIoU = {adv_miou:.2f}%, Pixel Acc = {adv_acc:.2f}%")
    print(f"  Relative Drop:  {clean_miou - adv_miou:.2f} percentage points mIoU")
    print("-" * 75)

    assert adv_miou < clean_miou, f"Attack did not degrade performance ({adv_miou:.2f}% vs {clean_miou:.2f}%)!"
    print("[OK] Smoke test checks PASSED successfully.")
    print("=" * 75)
    return True


def evaluate_full_dataset(
    data_dir: str,
    checkpoint_path: str,
    output_dir: str,
    output_prefix: str = "baseline_advtrain_drn_d_22_pgd",
    epsilon: float = 4.0 / 255.0,
    step_size: float = 1.0 / 255.0,
    steps: int = 50,
    batch_size: int = 4,
    alignment: str = "bilinear",
    device: torch.device = torch.device("cuda"),
    seed: int = 42,
    num_visualizations: int = 5
) -> Dict:
    """
    Evaluates all 500 validation images under untargeted L-infinity PGD attack.
    Saves metrics JSON and formatted summary TXT.
    """
    print("\n" + "=" * 75)
    print("STARTING FULL PGD-50 ATTACK EVALUATION (500 SAMPLES)")
    print(f"Model: {os.path.basename(checkpoint_path)}")
    print(f"Attack: L-inf PGD, eps={epsilon:.6f} ({epsilon*255:.1f}/255), "
          f"step_size={step_size:.6f} ({step_size*255:.1f}/255), steps={steps}, bs={batch_size}")
    print("=" * 75)

    torch.manual_seed(seed)
    np.random.seed(seed)

    os.makedirs(output_dir, exist_ok=True)
    vis_dir = os.path.join(output_dir, "visualizations")
    os.makedirs(vis_dir, exist_ok=True)

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
    print(f"Total samples: {total_images} (Batch size = {batch_size})")

    model, metadata = load_model(checkpoint_path, device)
    criterion = nn.NLLLoss(ignore_index=255)

    hist = np.zeros((19, 19), dtype=np.int64)
    total_correct = 0
    total_valid_pixels = 0
    total_attack_loss = 0.0

    batch_latencies = []
    saved_vis_count = 0
    num_batches = (total_images + batch_size - 1) // batch_size

    start_time = time.time()

    for b_idx in range(num_batches):
        b_start = b_idx * batch_size
        b_end = min(b_start + batch_size, total_images)
        current_bs = b_end - b_start

        batch_imgs = []
        batch_lbls = []
        batch_raw_pil = []

        for idx in range(b_start, b_end):
            img_p = os.path.join(data_dir, img_lines[idx])
            lbl_p = os.path.join(data_dir, lbl_lines[idx])

            pil_img = Image.open(img_p).convert("RGB")
            pil_lbl = Image.open(lbl_p)

            img_arr = np.array(pil_img, dtype=np.float32).transpose(2, 0, 1) / 255.0
            lbl_arr = np.array(pil_lbl, dtype=np.int32)

            batch_imgs.append(img_arr)
            batch_lbls.append(lbl_arr)
            if saved_vis_count < num_visualizations:
                batch_raw_pil.append((pil_img, lbl_arr, img_lines[idx]))

        t_batch_start = time.time()

        batch_tensor = torch.from_numpy(np.stack(batch_imgs, axis=0)).to(device)
        batch_lbl_tensor = torch.from_numpy(np.stack(batch_lbls, axis=0)).long().to(device)

        # Generate PGD perturbation for batch
        adv_batch_tensor, _ = generate_pgd_perturbation(
            model=model,
            images=batch_tensor,
            labels=batch_lbl_tensor,
            mean=mean,
            std=std,
            epsilon=epsilon,
            step_size=step_size,
            steps=steps,
            target_h=340,
            target_w=680,
            random_start=True,
            alignment=alignment,
            track_history=False
        )

        # Forward pass on adversarial batch
        with torch.no_grad():
            adv_norm = (adv_batch_tensor - mean) / std
            raw_out = model(adv_norm)[0]
            aligned_out = align_output(raw_out, target_h=340, target_w=680, mode=alignment)
            batch_loss = criterion(aligned_out, batch_lbl_tensor).item()

        t_batch_end = time.time()
        batch_latencies.append(t_batch_end - t_batch_start)
        total_attack_loss += batch_loss * current_bs

        pred_batch = aligned_out.argmax(dim=1).cpu().numpy().astype(np.int32)

        for i in range(current_bs):
            pred_i = pred_batch[i]
            lbl_i = batch_lbls[i]

            hist += fast_hist(pred_i.flatten(), lbl_i.flatten(), 19)

            valid_mask = (lbl_i >= 0) & (lbl_i < 19)
            total_correct += np.sum((pred_i == lbl_i) & valid_mask)
            total_valid_pixels += np.sum(valid_mask)

            # Save sample visualizations for the first few batches
            if saved_vis_count < num_visualizations:
                raw_pil, orig_lbl, rel_path = batch_raw_pil[i]
                sample_name = os.path.splitext(os.path.basename(rel_path))[0]
                adv_img_np = np.clip(adv_batch_tensor[i].cpu().numpy().transpose(1, 2, 0) * 255.0, 0, 255).astype(np.uint8)
                adv_pil = Image.fromarray(adv_img_np)

                # Save 4-panel: [Clean RGB | Adv RGB | Ground Truth | Adv Prediction]
                w, h = raw_pil.size
                gt_color = colorize_segmentation(orig_lbl)
                pred_color = colorize_segmentation(pred_i)

                panel = Image.new("RGB", (w * 4, h))
                panel.paste(raw_pil, (0, 0))
                panel.paste(adv_pil, (w, 0))
                panel.paste(gt_color, (w * 2, 0))
                panel.paste(pred_color, (w * 3, 0))

                panel_path = os.path.join(vis_dir, f"{output_prefix}_vis_{saved_vis_count+1}_{sample_name}.png")
                panel.save(panel_path)
                saved_vis_count += 1

        if (b_idx + 1) % 10 == 0 or (b_idx + 1) == num_batches:
            processed = b_end
            elapsed = time.time() - start_time
            running_miou = np.nanmean(per_class_iu(hist)) * 100
            running_acc = 100.0 * total_correct / total_valid_pixels
            print(f"  Batch [{b_idx+1}/{num_batches}] "
                  f"Samples: {processed}/{total_images} ({100.0*processed/total_images:.1f}%) | "
                  f"Elapsed: {elapsed:.1f}s | "
                  f"Running mIoU: {running_miou:.2f}% | "
                  f"Pixel Acc: {running_acc:.2f}%")

    total_time = time.time() - start_time
    mean_batch_lat = np.mean(batch_latencies)
    throughput_fps = total_images / total_time
    avg_loss = total_attack_loss / total_images

    per_class_ious = per_class_iu(hist) * 100
    overall_miou = float(np.nanmean(per_class_ious))
    overall_pixel_acc = float(100.0 * total_correct / total_valid_pixels)

    print("\n" + "=" * 75)
    print("FULL EVALUATION COMPLETED")
    print(f"Total time: {total_time:.2f}s ({total_time/60:.2f} min) | Throughput: {throughput_fps:.2f} FPS")
    print(f"Adversarial mIoU:       {overall_miou:.2f}%")
    print(f"Adversarial Pixel Acc:  {overall_pixel_acc:.2f}%")
    print(f"Average NLL Loss:       {avg_loss:.4f}")
    print("=" * 75)

    per_class_dict = {}
    for i, c_name in enumerate(CITYSCAPES_CLASSES):
        iou_val = float(per_class_ious[i])
        per_class_dict[c_name] = round(iou_val, 2)
        print(f"  Class {i:02d} ({c_name:<14}): {iou_val:6.2f}%")

    # Paper Table 1 reference is 26.61%
    paper_reference_miou = 26.61
    discrepancy = round(overall_miou - paper_reference_miou, 2)

    results_dict = {
        "experiment": "Adversarially Trained DRN-D-22 PGD Attack Baseline",
        "attack_type": "untargeted_pgd_linf",
        "checkpoint": os.path.basename(checkpoint_path),
        "dataset_split": "val",
        "sample_count": total_images,
        "input_resolution": [340, 680],
        "alignment_mode": alignment,
        "epsilon": float(epsilon),
        "epsilon_scaled_255": round(float(epsilon * 255.0), 4),
        "step_size": float(step_size),
        "step_size_scaled_255": round(float(step_size * 255.0), 4),
        "steps": steps,
        "random_start": True,
        "defense": "none",
        "seed": seed,
        "mean_iou_percent": round(overall_miou, 2),
        "pixel_accuracy_percent": round(overall_pixel_acc, 2),
        "average_loss": round(avg_loss, 4),
        "paper_table_1_reference_miou": paper_reference_miou,
        "discrepancy_vs_paper_points": discrepancy,
        "total_runtime_seconds": round(total_time, 2),
        "throughput_fps": round(throughput_fps, 2),
        "mean_batch_latency_seconds": round(mean_batch_lat, 4),
        "per_class_iou": per_class_dict
    }

    # Save JSON metrics
    json_path = os.path.join(output_dir, f"{output_prefix}_metrics.json")
    with open(json_path, "w") as f:
        json.dump(results_dict, f, indent=2)
    print(f"\n[Saved JSON Metrics] -> {json_path}")

    # Save Summary TXT
    txt_path = os.path.join(output_dir, f"{output_prefix}_summary.txt")
    with open(txt_path, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("ADVERSARIALLY TRAINED DRN-D-22 PGD ATTACK BASELINE (NO DEFENSE)\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Checkpoint:                {checkpoint_path}\n")
        f.write(f"Checkpoint Filename:       {os.path.basename(checkpoint_path)}\n")
        f.write(f"Dataset:                   Cityscapes Validation Set (Subsampled)\n")
        f.write(f"Samples Evaluated:         {total_images}\n")
        f.write(f"Input Resolution:          340 x 680\n")
        f.write(f"Raw Output Shape:          19 x 344 x 680\n")
        f.write(f"Alignment Method:          Differentiable Bilinear Interpolation (344 -> 340)\n\n")

        f.write("ATTACK CONFIGURATION:\n")
        f.write(f"  Attack Mode:             Untargeted L-infinity PGD (Vanilla, No Defense)\n")
        f.write(f"  Perturbation Budget:     L_inf eps = {epsilon:.6f} ({epsilon*255:.1f}/255)\n")
        f.write(f"  Step Size:               alpha = {step_size:.6f} ({step_size*255:.1f}/255)\n")
        f.write(f"  Iterations:              T = {steps} steps\n")
        f.write(f"  Random Initialization:   Uniform in [-eps, +eps]\n")
        f.write(f"  Image Space Bounds:      [0.0, 1.0] (clamped at every step)\n")
        f.write(f"  Loss Function:           nn.NLLLoss(ignore_index=255) on aligned LogSoftmax\n")
        f.write(f"  Random Seed:             {seed}\n\n")

        f.write("PERFORMANCE SUMMARY:\n")
        f.write(f"  Adversarial mIoU:        {overall_miou:.2f}%\n")
        f.write(f"  Adversarial Pixel Acc:   {overall_pixel_acc:.2f}%\n")
        f.write(f"  Average NLL Loss:        {avg_loss:.4f}\n")
        f.write(f"  Clean-Image Baseline:    51.34% mIoU (measured previously on 680x340 bilinear)\n")
        f.write(f"  Absolute Degradation:    {51.34 - overall_miou:.2f} percentage points\n\n")

        f.write("PAPER BENCHMARK COMPARISON:\n")
        f.write(f"  Paper Table 1 Reference: 26.61% mIoU (ICML 2023, Mao et al.)\n")
        f.write(f"  Our Measured Result:     {overall_miou:.2f}% mIoU\n")
        f.write(f"  Discrepancy:             {discrepancy:+.2f} percentage points\n\n")

        f.write("PER-CLASS IoU BREAKDOWN (19 Classes):\n")
        f.write("-" * 50 + "\n")
        f.write(f"{'Class ID':<10}{'Class Name':<20}{'IoU (%)':<10}\n")
        f.write("-" * 50 + "\n")
        for i, c_name in enumerate(CITYSCAPES_CLASSES):
            f.write(f"{i:<10}{c_name:<20}{per_class_ious[i]:.2f}%\n")
        f.write("-" * 50 + "\n\n")

        f.write("RUNTIME & SYSTEM METRICS:\n")
        f.write(f"  Total Wall Time:         {total_time:.2f} seconds ({total_time/60:.2f} minutes)\n")
        f.write(f"  Throughput:              {throughput_fps:.2f} FPS\n")
        f.write(f"  Batch Latency:           {mean_batch_lat:.4f} s/batch (Batch size = {batch_size})\n")
        f.write(f"  Device:                  {torch.cuda.get_device_name(0)}\n")

    print(f"[Saved Summary Text]  -> {txt_path}")

    return results_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate DRN-D-22 under PGD Attack on Subsampled Cityscapes.")
    parser.add_argument("--data-dir", type=str, default="cityscapes_dataset_subsampled", help="Subsampled Cityscapes root.")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/advtrain_drn_d_22_cityscapes.pth.tar", help="Model checkpoint path.")
    parser.add_argument("--output-dir", type=str, default="results", help="Directory to save outputs.")
    parser.add_argument("--output-prefix", type=str, default="baseline_advtrain_drn_d_22_pgd", help="Prefix for result files.")
    parser.add_argument("--epsilon", type=float, default=4.0, help="L-inf perturbation budget in [0, 255] scale.")
    parser.add_argument("--step-size", type=float, default=1.0, help="PGD step size in [0, 255] scale.")
    parser.add_argument("--steps", type=int, default=50, help="Number of PGD iterations.")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size for evaluation.")
    parser.add_argument("--alignment", type=str, choices=["bilinear", "slice"], default="bilinear", help="Output alignment mode.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for repeatability.")
    parser.add_argument("--smoke-test", action="store_true", help="Run 5-sample smoke test and exit.")
    parser.add_argument("--smoke-samples", type=int, default=5, help="Number of samples for smoke test.")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    eps_normalized = args.epsilon / 255.0
    step_size_normalized = args.step_size / 255.0

    if args.smoke_test:
        success = run_smoke_test(
            data_dir=args.data_dir,
            checkpoint_path=args.checkpoint,
            epsilon=eps_normalized,
            step_size=step_size_normalized,
            steps=args.steps,
            num_samples=args.smoke_samples,
            alignment=args.alignment,
            device=device,
            seed=args.seed
        )
        sys.exit(0 if success else 1)
    else:
        # First run smoke test as safety gate
        print("Running preliminary smoke test verification...")
        smoke_ok = run_smoke_test(
            data_dir=args.data_dir,
            checkpoint_path=args.checkpoint,
            epsilon=eps_normalized,
            step_size=step_size_normalized,
            steps=args.steps,
            num_samples=args.smoke_samples,
            alignment=args.alignment,
            device=device,
            seed=args.seed
        )
        if not smoke_ok:
            print("[FATAL] Smoke test failed! Aborting full evaluation.")
            sys.exit(1)

        print("\nSmoke test succeeded. Proceeding to full 500-sample evaluation...\n")
        evaluate_full_dataset(
            data_dir=args.data_dir,
            checkpoint_path=args.checkpoint,
            output_dir=args.output_dir,
            output_prefix=args.output_prefix,
            epsilon=eps_normalized,
            step_size=step_size_normalized,
            steps=args.steps,
            batch_size=args.batch_size,
            alignment=args.alignment,
            device=device,
            seed=args.seed
        )
