"""
Data Normalization Module for H1 Motion Diffusion (Step 2 - Fixed).
Computes feature-wise mean and std on TRAIN SPLIT ONLY (1,530 training sequences).
Handles constant/near-zero variance features by setting mean=0.0, std=1.0 and clamping min std to 0.01.
Preserves binary contact channels (features 45 & 46) unscaled (mean=0, std=1).
Outputs:
  - data/normalization/h1_train_mean.pt
  - data/normalization/h1_train_std.pt
  - reports/diffusion_normalization_report.json
"""

import os
import json
import torch
import numpy as np


def compute_train_normalization():
    data_dir = "c:/Thesis/data"
    norm_dir = os.path.join(data_dir, "normalization")
    reports_dir = "c:/Thesis/reports"

    os.makedirs(norm_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)

    pt_path = os.path.join(data_dir, "h1_locomotion_dataset.pt")
    train_idx_path = os.path.join(data_dir, "splits", "train_indices.txt")

    assert os.path.exists(pt_path), f"Dataset tensor missing at {pt_path}"
    assert os.path.exists(train_idx_path), f"Train indices file missing at {train_idx_path}"

    full_tensor = torch.load(pt_path)  # (1942, 196, 47)
    train_indices = [int(l.strip()) for l in open(train_idx_path) if l.strip()]

    train_tensor = full_tensor[train_indices]  # (1530, 196, 47)
    N_train, T, D = train_tensor.shape
    assert D == 47, f"Expected 47 features, got {D}"

    # Reshape to (N_train * T, 47) for feature-wise statistics
    flat_train = train_tensor.view(-1, D)

    mean = flat_train.mean(dim=0)
    std = flat_train.std(dim=0)

    # Detect near-zero variance features (std < 1e-2)
    zero_var_mask = (std < 1e-2)
    zero_var_count = int(zero_var_mask.sum().item())

    # For zero-variance features, keep mean=0 and std=1 to avoid division by near-zero values
    mean[zero_var_mask] = 0.0
    std[zero_var_mask] = 1.0

    # Minimum std clamp for remaining features
    std = torch.clamp(std, min=1e-2)

    # Preserve binary contact features (45 & 46) unscaled: mean=0, std=1
    mean[45] = 0.0
    mean[46] = 0.0
    std[45] = 1.0
    std[46] = 1.0

    # Save torch normalization files
    torch.save(mean, os.path.join(norm_dir, "h1_train_mean.pt"))
    torch.save(std, os.path.join(norm_dir, "h1_train_std.pt"))

    # Export JSON report
    norm_report = {
        "train_sequence_count": N_train,
        "sequence_length": T,
        "feature_dimension": D,
        "near_zero_variance_features_count": zero_var_count,
        "binary_contacts_handling": "Features 45 & 46 (left/right foot contact) preserved unscaled (mean=0.0, std=1.0)",
        "feature_means": [float(x) for x in mean.numpy()],
        "feature_stds": [float(x) for x in std.numpy()],
        "feature_ranges": {
            "root_lin_vel_mean": [float(x) for x in mean[0:3].numpy()],
            "root_ang_vel_mean": [float(x) for x in mean[3:6].numpy()],
            "root_height_mean": float(mean[6]),
            "joint_angles_mean": [float(x) for x in mean[7:26].numpy()],
            "joint_velocities_mean": [float(x) for x in mean[26:45].numpy()],
            "foot_contacts_mean": [float(x) for x in mean[45:47].numpy()]
        }
    }

    report_path = os.path.join(reports_dir, "diffusion_normalization_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(norm_report, f, indent=4, ensure_ascii=False)

    print("\n==================================================")
    print("=== TRAIN DATA NORMALIZATION COMPLETE (FIXED) ===")
    print("==================================================")
    print(f"Train Sequences Used:               {N_train}")
    print(f"Near-Zero Variance Features Reset:  {zero_var_count}")
    print(f"Mean Root Height:                   {float(mean[6]):.4f} m")
    print(f"Mean Joint Angle:                   {float(mean[7:26].mean()):.4f} rad")
    print(f"Saved Normalization Stats to:\n  - data/normalization/h1_train_mean.pt\n  - data/normalization/h1_train_std.pt\n  - reports/diffusion_normalization_report.json")

    return mean, std


if __name__ == "__main__":
    compute_train_normalization()
