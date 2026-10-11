#!/usr/bin/env python3
"""
Task 5: Adversarially Trained DRN-D-22 under BPDA Attack (No Defense / Vanilla)
Evaluates the adversarially trained model under BPDA adaptive formulation on undefended baseline.

Paper Reference (ICML 2023, Table 1):
  - Model: Adversarially Trained DRN-22-d
  - Attack: BPDA (Straight-through estimator on vanilla undefended model, eps=4/255, step=1/255, T=50)
  - Defense: None (Vanilla)
  - Paper Result: 26.61% mIoU (matching standard PGD vanilla 26.61% baseline)
"""

import os
import sys
import time
import json
import argparse
import random
import warnings
from typing import Dict, List, Tuple

warnings.filterwarnings("ignore")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
EQUI4ROB_DIR = os.path.join(ROOT_DIR, "Equi4Rob")
if EQUI4ROB_DIR not in sys.path:
    sys.path.insert(0, EQUI4ROB_DIR)

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

from models.DRNSeg import DRNSeg
from learning.utils_learn import fast_hist, per_class_iu

CITYSCAPES_CLASSES: List[str] = [
    "road", "sidewalk", "building", "wall", "fence",
    "pole", "traffic light", "traffic sign", "vegetation", "terrain",
    "sky", "person", "rider", "car", "truck",
    "bus", "train", "motorcycle", "bicycle"
]


def align_output(out: torch.Tensor, target_h: int = 340, target_w: int = 680, mode: str = "bilinear") -> torch.Tensor:
    if out.shape[2] == target_h and out.shape[3] == target_w:
        return out
    if mode == "bilinear":
        return F.interpolate(out, size=(target_h, target_w), mode="bilinear", align_corners=False)
    elif mode == "slice":
        return out[:, :, :target_h, :target_w]
    else:
        raise ValueError(f"Unsupported alignment mode '{mode}'")


def load_model(checkpoint_path: str, device: torch.device) -> Tuple[torch.nn.Module, Dict]:
    model = DRNSeg("drn_d_22", classes=19, pretrained=False)
    raw = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    metadata = {"checkpoint_path": checkpoint_path, "arch": "drn_d_22"}
    state_dict = raw["state_dict"] if isinstance(raw, dict) and "state_dict" in raw else raw

    clean_state_dict = {}
    for k, v in state_dict.items():
        clean_state_dict[k[7:] if k.startswith("module.") else k] = v

    model.load_state_dict(clean_state_dict, strict=True)
    for p in model.parameters():
        p.requires_grad = False
    model.to(device).eval()
    return model, metadata


def generate_bpda_nodefense_perturbation(
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    normalize,
    epsilon: float = 4.0 / 255.0,
    step_size: float = 1.0 / 255.0,
    steps: int = 50,
    alignment: str = "bilinear"
) -> torch.Tensor:
    """
    BPDA on undefended model (purification = identity function):
    Straight-through estimator backpropagates directly through identity transform.
    """
    criterion = nn.NLLLoss(ignore_index=255)
    x = images.clone().detach()

    delta = torch.zeros_like(x)
    delta.uniform_(-epsilon, epsilon)
    delta = torch.clamp(delta, 0.0 - x, 1.0 - x)

    for step in range(steps):
        delta = delta.detach().requires_grad_(True)
        # Identity forward purification: X_pfy = x + delta
        x_adv = x + delta
        x_norm = normalize(x_adv)

        raw_out = model(x_norm)[0]
        aligned_out = align_output(raw_out, target_h=labels.shape[1], target_w=labels.shape[2], mode=alignment)

        cost = criterion(aligned_out, labels)
        model.zero_grad()
        cost.backward()

        grad_sign = delta.grad.detach().sign()
        delta = delta.detach() + step_size * grad_sign
        delta = torch.clamp(delta, -epsilon, epsilon)
        delta = torch.clamp(delta, 0.0 - x, 1.0 - x)

    return delta.detach()


def evaluate_bpda_nodefense(
    data_dir: str,
    checkpoint_path: str,
    output_dir: str = "results",
    output_prefix: str = "baseline_advtrain_drn_d_22_bpda_nodefense",
    batch_size: int = 4,
    num_samples: int = None,
    epsilon: float = 4.0 / 255.0,
    step_size: float = 1.0 / 255.0,
    steps: int = 50,
    alignment: str = "bilinear",
    device: torch.device = None,
    seed: int = 42
):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    print("=" * 75)
    print(" TASK 5: ADVERSARIALLY TRAINED DRN-D-22 + BPDA ATTACK (NO DEFENSE) ")
    print("=" * 75)
    print(f"Device:             {torch.cuda.get_device_name(0)}")
    print(f"Checkpoint:         {checkpoint_path}")
    print(f"Attack:             BPDA Straight-Through (steps={steps}, eps={epsilon*255:.1f}/255, step={step_size*255:.1f}/255)")
    print(f"Defense:            None (Vanilla)")
    print(f"Batch Size:         {batch_size}")
    print(f"Output Alignment:   {alignment}")

    model, _ = load_model(checkpoint_path, device)

    mean = torch.tensor([0.290101, 0.328081, 0.286964]).view(1, 3, 1, 1).to(device)
    std = torch.tensor([0.182954, 0.186566, 0.184475]).view(1, 3, 1, 1).to(device)
    normalize = lambda x: (x - mean) / std

    images_list_file = os.path.join(data_dir, "val_images.txt")
    labels_list_file = os.path.join(data_dir, "val_labels.txt")
    with open(images_list_file, "r") as f:
        image_rel_paths = [line.strip() for line in f if line.strip()]
    with open(labels_list_file, "r") as f:
        label_rel_paths = [line.strip() for line in f if line.strip()]

    total_available = len(image_rel_paths)
    if num_samples is not None and num_samples < total_available:
        image_rel_paths = image_rel_paths[:num_samples]
        label_rel_paths = label_rel_paths[:num_samples]
    total_samples = len(image_rel_paths)
    num_batches = (total_samples + batch_size - 1) // batch_size
    print(f"Evaluating:         {total_samples} samples across {num_batches} batches\n")

    hist_clean = np.zeros((19, 19), dtype=np.int64)
    hist_adv = np.zeros((19, 19), dtype=np.int64)

    start_time = time.time()
    batch_latencies = []

    for b_idx in range(num_batches):
        b_start = time.time()
        start_i = b_idx * batch_size
        end_i = min(start_i + batch_size, total_samples)
        cur_bs = end_i - start_i

        imgs_list, lbls_list = [], []
        for idx in range(start_i, end_i):
            im_path = os.path.join(data_dir, image_rel_paths[idx])
            lb_path = os.path.join(data_dir, label_rel_paths[idx])
            im = Image.open(im_path).convert("RGB")
            lb = Image.open(lb_path)
            im_arr = np.array(im, dtype=np.float32) / 255.0
            lb_arr = np.array(lb, dtype=np.int64)
            imgs_list.append(torch.from_numpy(im_arr).permute(2, 0, 1))
            lbls_list.append(torch.from_numpy(lb_arr))

        imgs_tensor = torch.stack(imgs_list, dim=0).to(device)
        lbls_tensor = torch.stack(lbls_list, dim=0).to(device)

        # 1. Clean Prediction
        with torch.no_grad():
            clean_out = model(normalize(imgs_tensor))[0]
            clean_aligned = align_output(clean_out, target_h=340, target_w=680, mode=alignment)
            clean_pred = clean_aligned.argmax(dim=1).cpu().numpy()

        # 2. BPDA Attack without defense
        bpda_delta = generate_bpda_nodefense_perturbation(
            model=model,
            images=imgs_tensor,
            labels=lbls_tensor,
            normalize=normalize,
            epsilon=epsilon,
            step_size=step_size,
            steps=steps,
            alignment=alignment
        )
        x_adv = imgs_tensor + bpda_delta

        # 3. Adversarial Prediction
        with torch.no_grad():
            adv_out = model(normalize(x_adv))[0]
            adv_aligned = align_output(adv_out, target_h=340, target_w=680, mode=alignment)
            adv_pred = adv_aligned.argmax(dim=1).cpu().numpy()

        # Update confusion matrices
        gt_np = lbls_tensor.cpu().numpy()
        for i in range(cur_bs):
            hist_clean += fast_hist(clean_pred[i].flatten(), gt_np[i].flatten(), 19)
            hist_adv += fast_hist(adv_pred[i].flatten(), gt_np[i].flatten(), 19)

        b_time = time.time() - b_start
        batch_latencies.append(b_time)

        cur_clean_miou = np.nanmean(per_class_iu(hist_clean)) * 100
        cur_adv_miou = np.nanmean(per_class_iu(hist_adv)) * 100
        processed = end_i
        elapsed = time.time() - start_time
        fps = processed / elapsed
        rem_batches = num_batches - (b_idx + 1)
        eta_sec = rem_batches * np.mean(batch_latencies)

        print(f"[{b_idx+1:3d}/{num_batches:3d}] Samples: {processed:3d}/{total_samples} | "
              f"Clean: {cur_clean_miou:5.2f}% | BPDA-NoDef: {cur_adv_miou:5.2f}% | "
              f"Drop: {cur_clean_miou - cur_adv_miou:5.2f}% | Batch Lat: {b_time:.2f}s | ETA: {eta_sec/60:4.1f}m")

        # Live progress file updated every batch
        os.makedirs(output_dir, exist_ok=True)
        progress_path = os.path.join(output_dir, f"{output_prefix}_progress.txt")
        with open(progress_path, "w") as pf:
            pf.write(f"Task 5 Live Progress ({output_prefix}):\n")
            pf.write(f"  Batch:             {b_idx+1}/{num_batches} | Samples: {processed}/{total_samples} ({(processed/total_samples)*100:.1f}%)\n")
            pf.write(f"  Clean mIoU:        {cur_clean_miou:.2f}%\n")
            pf.write(f"  BPDA-NoDef mIoU:   {cur_adv_miou:.2f}%\n")
            pf.write(f"  Degradation:       {cur_clean_miou - cur_adv_miou:.2f} percentage points\n")
            pf.write(f"  Batch Latency:     {b_time:.2f}s | Speed: {fps:.2f} FPS\n")
            pf.write(f"  Elapsed:           {elapsed/60:.1f} min | ETA: {eta_sec/60:.1f} min\n")

        # Save checkpoint metrics every 25 batches
        if (b_idx + 1) % 25 == 0 or (b_idx + 1) == num_batches:
            interim_json = os.path.join(output_dir, f"{output_prefix}_metrics.json")
            with open(interim_json, "w") as f:
                json.dump({
                    "task": "Task 5: Adversarially Trained DRN-D-22 under BPDA Attack (No Defense)",
                    "checkpoint": checkpoint_path,
                    "samples_evaluated": processed,
                    "total_samples": total_samples,
                    "clean_miou": round(cur_clean_miou, 2),
                    "bpda_nodefense_miou": round(cur_adv_miou, 2),
                    "paper_target_miou": 26.61,
                    "status": "in_progress" if processed < total_samples else "completed",
                    "elapsed_minutes": round(elapsed / 60, 2)
                }, f, indent=2)

    total_time = time.time() - start_time
    clean_ious = per_class_iu(hist_clean) * 100
    adv_ious = per_class_iu(hist_adv) * 100

    clean_miou = float(np.nanmean(clean_ious))
    adv_miou = float(np.nanmean(adv_ious))

    print("\n" + "=" * 75)
    print(" FINAL EVALUATION RESULTS ")
    print("=" * 75)
    print(f"Clean Baseline mIoU:       {clean_miou:.2f}% (Paper Table 1: 53.23%)")
    print(f"BPDA Attack (No Defense):  {adv_miou:.2f}% (Paper Table 1: 26.61%)")
    print(f"Difference vs Paper:       {adv_miou - 26.61:+.2f} percentage points")
    print(f"Degradation Drop:          {clean_miou - adv_miou:.2f} percentage points")
    print(f"Total Runtime:             {total_time:.2f}s ({total_time/60:.2f}m) | Speed: {total_samples/total_time:.2f} FPS")
    print("=" * 75)

    os.makedirs(output_dir, exist_ok=True)
    summary_path = os.path.join(output_dir, f"{output_prefix}_summary.txt")
    json_path = os.path.join(output_dir, f"{output_prefix}_metrics.json")

    results_data = {
        "task": "Task 5: Adversarially Trained DRN-D-22 under BPDA Attack (No Defense)",
        "checkpoint": checkpoint_path,
        "samples_evaluated": total_samples,
        "clean_miou": clean_miou,
        "bpda_nodefense_miou": adv_miou,
        "paper_target_miou": 26.61,
        "discrepancy": adv_miou - 26.61,
        "runtime_seconds": total_time,
        "fps": total_samples / total_time,
        "per_class_ious": {CITYSCAPES_CLASSES[i]: round(float(adv_ious[i]), 2) for i in range(19)}
    }
    with open(json_path, "w") as f:
        json.dump(results_data, f, indent=2)

    with open(summary_path, "w") as f:
        f.write("=" * 75 + "\n")
        f.write("TASK 5: ADVERSARIALLY TRAINED DRN-D-22 + BPDA ATTACK (NO DEFENSE)\n")
        f.write("=" * 75 + "\n\n")
        f.write(f"Dataset:                   Cityscapes Validation Subsampled (680x340)\n")
        f.write(f"Checkpoint:                {checkpoint_path}\n")
        f.write(f"Samples Evaluated:         {total_samples}\n")
        f.write(f"Attack:                    BPDA Straight-Through (50 steps, eps=4/255, step=1/255, L_inf)\n")
        f.write(f"Defense:                   None (Vanilla)\n\n")
        f.write("RESULTS SUMMARY:\n")
        f.write(f"  Clean mIoU:              {clean_miou:.2f}%\n")
        f.write(f"  BPDA Attack (No Defense):{adv_miou:.2f}% (Paper Table 1: 26.61%)\n")
        f.write(f"  Discrepancy vs Paper:    {adv_miou - 26.61:+.2f} percentage points\n")
        f.write(f"  Degradation:             {clean_miou - adv_miou:.2f} percentage points\n\n")
        f.write("PER-CLASS ATTACKED IoU (%):\n")
        for i, c in enumerate(CITYSCAPES_CLASSES):
            f.write(f"  {i:2d}. {c:<18}: {adv_ious[i]:.2f}%\n")
        f.write(f"\nTotal Wall Time:           {total_time:.2f} s ({total_time/60:.2f} min)\n")

    print(f"[Saved] Metrics JSON: {json_path}")
    print(f"[Saved] Summary Text: {summary_path}")
    return results_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Task 5: BPDA Attack (No Defense)")
    parser.add_argument("--data-dir", type=str, default="cityscapes_dataset_subsampled")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/advtrain_drn_d_22_cityscapes.pth.tar")
    parser.add_argument("--output-dir", type=str, default="results")
    parser.add_argument("--output-prefix", type=str, default="baseline_advtrain_drn_d_22_bpda_nodefense")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-samples", type=int, default=None)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_samples = 4 if args.smoke_test else args.num_samples

    evaluate_bpda_nodefense(
        data_dir=args.data_dir,
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        output_prefix=args.output_prefix,
        batch_size=args.batch_size,
        num_samples=num_samples,
        steps=args.steps,
        device=device
    )
