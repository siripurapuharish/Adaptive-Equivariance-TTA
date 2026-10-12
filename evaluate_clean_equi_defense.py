#!/usr/bin/env python3
"""
Task 2: Adversarially Trained DRN-D-22 on Clean Images + Equivariance TTA Defense
Evaluates clean-image accuracy under test-time equivariance adaptation (measuring clean accuracy trade-off).

Paper Target (ICML 2023, Table 1):
  - Model: Adversarially Trained DRN-22-d
  - Input: Clean Images (No Adversarial Attack)
  - Defense: Equivariance TTA (20 steps, eps_v=10.0/255, 8 views)
  - Paper Result: 48.74% mIoU (vs 53.23% clean vanilla baseline)
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
from torchvision.transforms import transforms

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
EQUI4ROB_DIR = os.path.join(ROOT_DIR, "Equi4Rob")
if EQUI4ROB_DIR not in sys.path:
    sys.path.insert(0, EQUI4ROB_DIR)

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

from models.DRNSeg import DRNSeg
from learning.utils_learn import fast_hist, per_class_iu
from learning.attack_new_single import Equi_Set_Defense_good_inloop

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


def evaluate_clean_equi_defense(
    data_dir: str,
    checkpoint_path: str,
    output_dir: str = "results",
    output_prefix: str = "baseline_clean_drn_d_22_clean_equidefense",
    batch_size: int = 2,
    num_samples: int = None,
    epsilon_defense: float = 6.0 / 255.0,
    num_views: int = 7,
    defense_steps: int = 20,
    alignment: str = "bilinear",
    device: torch.device = None,
    seed: int = 42
):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    is_clean_model = "clean" in os.path.basename(checkpoint_path).lower()
    task_id = "TASK 8" if is_clean_model else "TASK 2"
    task_name = "CLEAN PRETRAINED DRN-D-22 (CLEAN) + EQUIVARIANCE DEFENSE" if is_clean_model else "ADVERSARIALLY TRAINED DRN-D-22 (CLEAN) + EQUIVARIANCE TTA DEFENSE"
    paper_target = 37.95 if is_clean_model else 48.74
    vanilla_target = 58.29 if is_clean_model else 53.23

    print("=" * 75)
    print(f" {task_id}: {task_name} ")
    print("=" * 75)
    print(f"Device:             {torch.cuda.get_device_name(0)}")
    print(f"Checkpoint:         {checkpoint_path}")
    print(f"Attack:             None (Clean Images)")
    print(f"Defense:            Equivariance TTA ({defense_steps} steps, eps_v={epsilon_defense*255:.1f}/255, {num_views} views)")
    print(f"Batch Size:         {batch_size}")
    print(f"Output Alignment:   {alignment}")

    model, _ = load_model(checkpoint_path, device)
    net_list = [model] * num_views

    mean = torch.tensor([0.290101, 0.328081, 0.286964]).view(1, 3, 1, 1).to(device)
    std = torch.tensor([0.182954, 0.186566, 0.184475]).view(1, 3, 1, 1).to(device)
    normalize = lambda x: (x - mean) / std

    s = 1
    transforms_customized = torch.nn.Sequential(
        transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s),
        transforms.RandomGrayscale(p=0.2),
    )
    scripted_transforms = torch.jit.script(transforms_customized).to(device)
    criterion = nn.NLLLoss(ignore_index=255)

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
    hist_defended = np.zeros((19, 19), dtype=np.int64)

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

        # 1. Clean Vanilla Prediction
        with torch.no_grad():
            clean_out = model(normalize(imgs_tensor))[0]
            clean_aligned = align_output(clean_out, target_h=340, target_w=680, mode=alignment)
            clean_pred = clean_aligned.argmax(dim=1).cpu().numpy()

        # 2. Equivariance TTA Defense applied to Clean Image
        defense_delta = Equi_Set_Defense_good_inloop(
            x=imgs_tensor,
            y=lbls_tensor,
            net_list=net_list,
            Loss=criterion,
            epsilon=epsilon_defense,
            steps=defense_steps,
            dataset='cityscape',
            step_size=255.0 / 255.0,
            info=None,
            using_noise=True,
            SGD_wN=True,
            attack_type='equi',
            innormalize=normalize,
            norm='l_inf',
            scripted_transforms=scripted_transforms,
            transform_delta=True
        )
        x_defended = torch.clamp(imgs_tensor + defense_delta, 0.0, 1.0)

        # 3. Defended Prediction
        with torch.no_grad():
            def_out = model(normalize(x_defended))[0]
            def_aligned = align_output(def_out, target_h=340, target_w=680, mode=alignment)
            def_pred = def_aligned.argmax(dim=1).cpu().numpy()

        # Update confusion matrices
        gt_np = lbls_tensor.cpu().numpy()
        for i in range(cur_bs):
            hist_clean += fast_hist(clean_pred[i].flatten(), gt_np[i].flatten(), 19)
            hist_defended += fast_hist(def_pred[i].flatten(), gt_np[i].flatten(), 19)

        b_time = time.time() - b_start
        batch_latencies.append(b_time)

        cur_clean_miou = np.nanmean(per_class_iu(hist_clean)) * 100
        cur_def_miou = np.nanmean(per_class_iu(hist_defended)) * 100
        processed = end_i
        elapsed = time.time() - start_time
        fps = processed / elapsed
        rem_batches = num_batches - (b_idx + 1)
        eta_sec = rem_batches * (np.mean(batch_latencies))

        print(f"[{b_idx+1:3d}/{num_batches:3d}] Samples: {processed:3d}/{total_samples} | "
              f"Clean (No Def): {cur_clean_miou:5.2f}% | Defended (Equi): {cur_def_miou:5.2f}% | "
              f"Diff: {cur_def_miou - cur_clean_miou:+5.2f}% | Latency: {b_time:5.2f}s | ETA: {eta_sec/60:4.1f}m")

        # Live progress file updated every batch
        os.makedirs(output_dir, exist_ok=True)
        progress_path = os.path.join(output_dir, f"{output_prefix}_progress.txt")
        with open(progress_path, "w") as pf:
            pf.write(f"{task_id} Live Progress:\n")
            pf.write(f"  Batch:             {b_idx+1}/{num_batches} | Samples: {processed}/{total_samples} ({(processed/total_samples)*100:.1f}%)\n")
            pf.write(f"  Clean (No Def):    {cur_clean_miou:.2f}%\n")
            pf.write(f"  Clean + Equi Def:  {cur_def_miou:.2f}%\n")
            pf.write(f"  Trade-off Diff:    {cur_def_miou - cur_clean_miou:+.2f} percentage points\n")
            pf.write(f"  Batch Latency:     {b_time:.2f}s | Speed: {fps:.2f} FPS\n")
            pf.write(f"  Elapsed:           {elapsed/60:.1f} min | ETA: {eta_sec/60:.1f} min\n")

        # Save checkpoint metrics every 25 batches
        if (b_idx + 1) % 25 == 0 or (b_idx + 1) == num_batches:
            interim_json = os.path.join(output_dir, f"{output_prefix}_metrics.json")
            with open(interim_json, "w") as f:
                json.dump({
                    "task": f"{task_id}: {task_name}",
                    "checkpoint": checkpoint_path,
                    "samples_evaluated": processed,
                    "total_samples": total_samples,
                    "clean_vanilla_miou": round(cur_clean_miou, 2),
                    "clean_equi_defended_miou": round(cur_def_miou, 2),
                    "difference": round(cur_def_miou - cur_clean_miou, 2),
                    "paper_target_miou": paper_target,
                    "epsilon_defense": f"{epsilon_defense*255:.2f}/255",
                    "num_views": num_views,
                    "status": "in_progress" if processed < total_samples else "completed",
                    "elapsed_minutes": round(elapsed / 60, 2)
                }, f, indent=2)

    total_time = time.time() - start_time
    clean_ious = per_class_iu(hist_clean) * 100
    def_ious = per_class_iu(hist_defended) * 100

    clean_miou = float(np.nanmean(clean_ious))
    def_miou = float(np.nanmean(def_ious))

    print("\n" + "=" * 75)
    print(" FINAL EVALUATION RESULTS ")
    print("=" * 75)
    print(f"Clean Vanilla Baseline:      {clean_miou:.2f}% (Paper Target: {vanilla_target:.2f}%)")
    print(f"Clean + Equivariance Defense:{def_miou:.2f}% (Paper Target: {paper_target:.2f}%)")
    print(f"Difference (Trade-off):      {def_miou - clean_miou:+.2f} percentage points")
    print(f"Total Runtime:               {total_time:.2f}s ({total_time/60:.2f}m) | Speed: {total_samples/total_time:.2f} FPS")
    print("=" * 75)

    os.makedirs(output_dir, exist_ok=True)
    summary_path = os.path.join(output_dir, f"{output_prefix}_summary.txt")
    json_path = os.path.join(output_dir, f"{output_prefix}_metrics.json")

    results_data = {
        "task": f"{task_id}: {task_name}",
        "checkpoint": checkpoint_path,
        "samples_evaluated": total_samples,
        "clean_vanilla_miou": clean_miou,
        "clean_equi_defended_miou": def_miou,
        "difference": def_miou - clean_miou,
        "paper_target_miou": paper_target,
        "epsilon_defense": f"{epsilon_defense*255:.2f}/255",
        "num_views": num_views,
        "runtime_seconds": total_time,
        "fps": total_samples / total_time,
        "per_class_ious": {CITYSCAPES_CLASSES[i]: round(float(def_ious[i]), 2) for i in range(19)}
    }
    with open(json_path, "w") as f:
        json.dump(results_data, f, indent=2)

    with open(summary_path, "w") as f:
        f.write("=" * 75 + "\n")
        f.write(f"{task_id}: {task_name}\n")
        f.write("=" * 75 + "\n\n")
        f.write(f"Dataset:                     Cityscapes Validation Subsampled (680x340)\n")
        f.write(f"Checkpoint:                  {checkpoint_path}\n")
        f.write(f"Samples Evaluated:           {total_samples}\n")
        f.write(f"Attack:                      None (Clean Images)\n")
        f.write(f"Defense:                     Equivariance TTA ({defense_steps} steps, eps_v={epsilon_defense*255:.1f}/255, {num_views} views)\n\n")
        f.write("RESULTS SUMMARY:\n")
        f.write(f"  Clean Vanilla mIoU:        {clean_miou:.2f}% (Paper Target: {vanilla_target:.2f}%)\n")
        f.write(f"  Clean + Equi Defended mIoU:{def_miou:.2f}% (Paper Target: {paper_target:.2f}%)\n")
        f.write(f"  Trade-off Difference:      {def_miou - clean_miou:+.2f} percentage points\n\n")
        f.write("PER-CLASS DEFENDED IoU (%):\n")
        for i, c in enumerate(CITYSCAPES_CLASSES):
            f.write(f"  {i:2d}. {c:<18}: {def_ious[i]:.2f}%\n")
        f.write(f"\nTotal Wall Time:             {total_time:.2f} s ({total_time/60:.2f} min)\n")

    print(f"[Saved] Metrics JSON: {json_path}")
    print(f"[Saved] Summary Text: {summary_path}")
    return results_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Clean Images + Equivariance Defense")
    parser.add_argument("--data-dir", type=str, default="cityscapes_dataset_subsampled")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/clean_drn_d_22_cityscapes.pth")
    parser.add_argument("--output-dir", type=str, default="results")
    parser.add_argument("--output-prefix", type=str, default="baseline_clean_drn_d_22_clean_equidefense")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-samples", type=int, default=None)
    parser.add_argument("--epsilon-defense", type=float, default=6.0 / 255.0, help="Defense perturbation bound (default: 6/255 for Table 2)")
    parser.add_argument("--num-views", type=int, default=7, help="Number of views for equivariance defense (default: 7 for Table 2)")
    parser.add_argument("--defense-steps", type=int, default=20, help="Number of defense steps (default: 20)")
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_samples = 2 if args.smoke_test else args.num_samples

    evaluate_clean_equi_defense(
        data_dir=args.data_dir,
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        output_prefix=args.output_prefix,
        batch_size=args.batch_size,
        num_samples=num_samples,
        epsilon_defense=args.epsilon_defense,
        num_views=args.num_views,
        defense_steps=args.defense_steps,
        device=device
    )
