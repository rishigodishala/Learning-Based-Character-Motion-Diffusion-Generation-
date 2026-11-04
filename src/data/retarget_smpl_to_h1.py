"""
Bilateral Contact-Aware Optimization IK Solver for SMPL-to-Unitree-H1 Retargeting.
Replaces independent 2D leg IK with a joint PyTorch optimization over:
  q_leg in R^10 (both legs) + root_z in R^1
enforcing double-support foot ground consistency (|z_left| < 0.02m, |z_right| < 0.02m),
0.05 rad safe joint margins, pitch-only foot flatness, and temporal smoothness.
"""

import os
import json
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Tuple, List
from src.data.coordinate_transform import transform_positions_torch


H1_JOINT_NAMES = [
    "left_hip_yaw", "left_hip_roll", "left_hip_pitch", "left_knee", "left_ankle",
    "right_hip_yaw", "right_hip_roll", "right_hip_pitch", "right_knee", "right_ankle",
    "torso",
    "left_shoulder_pitch", "left_shoulder_roll", "left_shoulder_yaw", "left_elbow",
    "right_shoulder_pitch", "right_shoulder_roll", "right_shoulder_yaw", "right_elbow"
]

# Official URDF Joint Limits
H1_JOINT_LIMITS_LOWER = np.array([
    -0.43, -0.43, -1.57, -0.26, -0.87,
    -0.43, -0.43, -1.57, -0.26, -0.87,
    -2.35,
    -2.87, -0.34, -1.30, -1.25,
    -2.87, -3.11, -4.45, -1.25
], dtype=np.float32)

H1_JOINT_LIMITS_UPPER = np.array([
    0.43, 0.43, 1.57, 2.05, 0.52,
    0.43, 0.43, 1.57, 2.05, 0.52,
    2.35,
    2.87, 3.11, 4.45, 2.61,
    2.87, 0.34, 1.30, 2.61
], dtype=np.float32)

# Step 7: Configurable Safe Joint Margins (0.05 rad)
SAFETY_MARGIN_RAD = 0.05
H1_JOINT_SAFE_LOWER = H1_JOINT_LIMITS_LOWER + SAFETY_MARGIN_RAD
H1_JOINT_SAFE_UPPER = H1_JOINT_LIMITS_UPPER - SAFETY_MARGIN_RAD


def compute_h1_forward_kinematics_pytorch(q_leg: torch.Tensor, root_z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Differentiable Forward Kinematics for H1 legs in PyTorch.
    Args:
        q_leg: (10,) tensor [l_yaw, l_roll, l_pitch, l_knee, l_ankle, r_yaw, r_roll, r_pitch, r_knee, r_ankle]
        root_z: scalar tensor for pelvis height
    Returns:
        (foot_left_xyz, foot_right_xyz)
    """
    # Left Leg FK
    q_l_roll = q_leg[1]
    q_l_pitch = q_leg[2]
    q_l_knee = q_leg[3]
    q_l_ankle = q_leg[4]

    x_l = 0.40 * torch.sin(q_l_pitch) + 0.40 * torch.sin(q_l_pitch + q_l_knee) + 0.05 * torch.sin(q_l_pitch + q_l_knee + q_l_ankle)
    y_l = 0.10 + 0.40 * torch.sin(q_l_roll)
    z_l = root_z - 0.1742 - 0.40 * torch.cos(q_l_pitch) - 0.40 * torch.cos(q_l_pitch + q_l_knee) - 0.05 * torch.cos(q_l_pitch + q_l_knee + q_l_ankle)

    foot_l = torch.stack([x_l, y_l, z_l])

    # Right Leg FK
    q_r_roll = q_leg[6]
    q_r_pitch = q_leg[7]
    q_r_knee = q_leg[8]
    q_r_ankle = q_leg[9]

    x_r = 0.40 * torch.sin(q_r_pitch) + 0.40 * torch.sin(q_r_pitch + q_r_knee) + 0.05 * torch.sin(q_r_pitch + q_r_knee + q_r_ankle)
    y_r = -0.10 + 0.40 * torch.sin(q_r_roll)
    z_r = root_z - 0.1742 - 0.40 * torch.cos(q_r_pitch) - 0.40 * torch.cos(q_r_pitch + q_r_knee) - 0.05 * torch.cos(q_r_pitch + q_r_knee + q_r_ankle)

    foot_r = torch.stack([x_r, y_r, z_r])

    return foot_l, foot_r


def compute_h1_forward_kinematics_foot_heights(q: np.ndarray, root_z: float) -> Tuple[float, float, float]:
    """Computes exact H1 foot sole heights using forward kinematics geometry."""
    q_leg_t = torch.tensor(q[:10], dtype=torch.float32)
    root_z_t = torch.tensor(root_z, dtype=torch.float32)
    foot_l, foot_r = compute_h1_forward_kinematics_pytorch(q_leg_t, root_z_t)

    left_foot_z = float(foot_l[2].item())
    right_foot_z = float(foot_r[2].item())
    lowest_foot_z = min(left_foot_z, right_foot_z)
    return left_foot_z, right_foot_z, lowest_foot_z


def compute_ground_aligned_root_height(q0: np.ndarray, orig_root_z: float, contacts_2d: np.ndarray = None) -> Tuple[float, Dict]:
    """Calculates ground-aligned root height based on intended stance foot contact labels."""
    l_z_orig, r_z_orig, lowest_orig = compute_h1_forward_kinematics_foot_heights(q0, orig_root_z)

    is_left_stance = True
    is_right_stance = True

    if contacts_2d is not None and len(contacts_2d) > 0:
        c_l, c_r = contacts_2d[0], contacts_2d[1]
        is_left_stance = bool(c_l > 0.5)
        is_right_stance = bool(c_r > 0.5)

    if is_left_stance and is_right_stance:
        target_stance_z = lowest_orig
        stance_label = "DOUBLE_SUPPORT"
    elif is_left_stance:
        target_stance_z = l_z_orig
        stance_label = "LEFT_STANCE"
    elif is_right_stance:
        target_stance_z = r_z_orig
        stance_label = "RIGHT_STANCE"
    else:
        target_stance_z = lowest_orig
        stance_label = "DOUBLE_SUPPORT_FALLBACK"

    ground_offset = -target_stance_z
    corrected_root_z = orig_root_z + ground_offset

    l_z_final, r_z_final, lowest_final = compute_h1_forward_kinematics_foot_heights(q0, corrected_root_z)

    audit_log = {
        "stance_label": stance_label,
        "original_root_height_m": float(orig_root_z),
        "original_left_foot_height_m": float(l_z_orig),
        "original_right_foot_height_m": float(r_z_orig),
        "calculated_ground_offset_m": float(ground_offset),
        "corrected_root_height_m": float(corrected_root_z),
        "final_left_foot_height_m": float(l_z_final),
        "final_right_foot_height_m": float(r_z_final),
        "lowest_foot_height_at_ground": float(lowest_final)
    }

    return float(corrected_root_z), audit_log


class BilateralContactAwareIKSolver:
    """
    Step 2 - Step 10: Joint PyTorch Bilateral Optimization IK Solver for Unitree H1.
    Optimizes q_leg in R^10 + root_z in R^1 jointly per frame using L-BFGS/Adam optimizer.
    """
    def __init__(self, num_iters: int = 50, lr: float = 0.05):
        self.num_iters = num_iters
        self.lr = lr
        self.safe_lower = torch.tensor(H1_JOINT_SAFE_LOWER[:10], dtype=torch.float32)
        self.safe_upper = torch.tensor(H1_JOINT_SAFE_UPPER[:10], dtype=torch.float32)
        self.lower_limits = torch.tensor(H1_JOINT_LIMITS_LOWER[:10], dtype=torch.float32)
        self.upper_limits = torch.tensor(H1_JOINT_LIMITS_UPPER[:10], dtype=torch.float32)

    def solve_frame(
        self, 
        smpl_l_ankle_target: np.ndarray, 
        smpl_r_ankle_target: np.ndarray, 
        contact_2d: np.ndarray, 
        prev_q_leg: torch.Tensor = None, 
        prev_root_z: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        Solves single frame bilateral optimization over q_leg (10) and root_z (1).
        """
        target_l = torch.tensor(smpl_l_ankle_target, dtype=torch.float32)
        target_r = torch.tensor(smpl_r_ankle_target, dtype=torch.float32)
        c_l, c_r = float(contact_2d[0]), float(contact_2d[1])

        is_double_support = (c_l > 0.5 and c_r > 0.5)
        is_left_support = (c_l > 0.5 and c_r <= 0.5)
        is_right_support = (c_r > 0.5 and c_l <= 0.5)

        # Initial parameter estimates
        if prev_q_leg is not None:
            q_leg_param = nn.Parameter(prev_q_leg.clone())
            root_z_param = nn.Parameter(prev_root_z.clone())
        else:
            q_leg_param = nn.Parameter(torch.tensor([0.0, 0.0, 0.2, 0.4, -0.6, 0.0, 0.0, 0.2, 0.4, -0.6], dtype=torch.float32))
            root_z_param = nn.Parameter(torch.tensor(0.72, dtype=torch.float32))

        optimizer = torch.optim.Adam([q_leg_param, root_z_param], lr=self.lr)

        for it in range(self.num_iters):
            optimizer.zero_grad()

            foot_l, foot_r = compute_h1_forward_kinematics_pytorch(q_leg_param, root_z_param)

            # 1. Target Foot Tracking Error (XY position matching relative stride)
            e_pose_l = torch.sum((foot_l[:2] - target_l[:2])**2)
            e_pose_r = torch.sum((foot_r[:2] - target_r[:2])**2)

            # 2. Contact-Aware Ground Consistency Loss (Step 5)
            if is_double_support:
                e_ground = 150.0 * (foot_l[2]**2 + foot_r[2]**2)
            elif is_left_support:
                e_ground = 150.0 * (foot_l[2]**2) + 10.0 * ((foot_r[2] - target_r[2])**2)
            elif is_right_support:
                e_ground = 150.0 * (foot_r[2]**2) + 10.0 * ((foot_l[2] - target_l[2])**2)
            else:
                e_ground = 10.0 * ((foot_l[2] - target_l[2])**2 + (foot_r[2] - target_r[2])**2)

            # 3. Step 6: Pitch-Only Ankle Orientation & Foot Flatness
            l_pitch_sum = q_leg_param[2] + q_leg_param[3] + q_leg_param[4]
            r_pitch_sum = q_leg_param[7] + q_leg_param[8] + q_leg_param[9]
            e_flatness = 20.0 * (l_pitch_sum**2 + r_pitch_sum**2)

            # 4. Step 3: Safe Joint Margin Penalty (0.05 rad)
            underflow = torch.clamp(self.safe_lower - q_leg_param, min=0.0)
            overflow = torch.clamp(q_leg_param - self.safe_upper, min=0.0)
            e_limit = 500.0 * torch.sum(underflow**2 + overflow**2)

            # 5. Knee extension prior (discourage extreme knee bending > 1.2 rad)
            e_knee = 10.0 * (torch.clamp(q_leg_param[3] - 1.2, min=0.0)**2 + torch.clamp(q_leg_param[8] - 1.2, min=0.0)**2)

            # 6. Step 10: Temporal Smoothness Penalty
            e_smooth = 0.0
            if prev_q_leg is not None:
                e_smooth = 30.0 * torch.sum((q_leg_param - prev_q_leg)**2) + 30.0 * (root_z_param - prev_root_z)**2

            loss = e_pose_l + e_pose_r + e_ground + e_flatness + e_limit + e_knee + e_smooth

            loss.backward()
            optimizer.step()

            # Enforce hard limits during step
            with torch.no_grad():
                q_leg_param.clamp_(self.lower_limits, self.upper_limits)
                root_z_param.clamp_(0.45, 0.95)

        with torch.no_grad():
            final_foot_l, final_foot_r = compute_h1_forward_kinematics_pytorch(q_leg_param, root_z_param)
            res_l = float(torch.norm(final_foot_l[:2] - target_l[:2]).item())
            res_r = float(torch.norm(final_foot_r[:2] - target_r[:2]).item())
            z_err_l = abs(float(final_foot_l[2].item()))
            z_err_r = abs(float(final_foot_r[2].item()))

            # Step 11: Feasibility Classification
            if is_double_support:
                feasible = (z_err_l < 0.02 and z_err_r < 0.02 and res_l < 0.05 and res_r < 0.05)
            elif is_left_support:
                feasible = (z_err_l < 0.02 and res_l < 0.05)
            elif is_right_support:
                feasible = (z_err_r < 0.02 and res_r < 0.05)
            else:
                feasible = (res_l < 0.05 and res_r < 0.05)

        diag = {
            "left_foot_z_m": float(final_foot_l[2].item()),
            "right_foot_z_m": float(final_foot_r[2].item()),
            "left_foot_z_error_m": z_err_l,
            "right_foot_z_error_m": z_err_r,
            "left_residual_m": res_l,
            "right_residual_m": res_r,
            "root_z_m": float(root_z_param.item()),
            "contact_state": "DOUBLE_SUPPORT" if is_double_support else ("LEFT_SUPPORT" if is_left_support else ("RIGHT_SUPPORT" if is_right_support else "FLIGHT")),
            "feasible_pass": feasible
        }

        return q_leg_param.detach(), root_z_param.detach(), diag

    def solve_sequence(self, smpl_positions: torch.Tensor, foot_contacts: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor, List[Dict]]:
        """
        Solves complete temporal sequence bilaterally across all frames.
        """
        T = smpl_positions.shape[0]

        # Transform to Z-Up
        smpl_z_up = transform_positions_torch(smpl_positions).cpu().numpy()
        pelvis = smpl_z_up[:, 0, :]
        l_ankle = smpl_z_up[:, 7, :] - pelvis
        r_ankle = smpl_z_up[:, 8, :] - pelvis

        # Morphology scaling (Step 8: Global leg scaling ratio 0.90)
        l_ankle_scaled = l_ankle * 0.90
        r_ankle_scaled = r_ankle * 0.90

        if foot_contacts is None:
            foot_contacts = torch.ones((T, 2), dtype=torch.float32)

        q_legs = []
        root_zs = []
        diagnostics = []

        prev_q = None
        prev_rz = None

        for t in range(T):
            q_leg_t, rz_t, diag = self.solve_frame(
                smpl_l_ankle_target=l_ankle_scaled[t],
                smpl_r_ankle_target=r_ankle_scaled[t],
                contact_2d=foot_contacts[t].cpu().numpy(),
                prev_q_leg=prev_q,
                prev_root_z=prev_rz
            )

            # Arms & Torso Mapping
            l_elbow = smpl_z_up[t, 18, :] - smpl_z_up[t, 16, :]
            r_elbow = smpl_z_up[t, 19, :] - smpl_z_up[t, 17, :]

            l_s_pitch = np.clip(np.arctan2(l_elbow[0], -l_elbow[2]), H1_JOINT_SAFE_LOWER[11], H1_JOINT_SAFE_UPPER[11])
            l_s_roll = np.clip(np.arctan2(l_elbow[1], -l_elbow[2]), H1_JOINT_SAFE_LOWER[12], H1_JOINT_SAFE_UPPER[12])
            r_s_pitch = np.clip(np.arctan2(r_elbow[0], -r_elbow[2]), H1_JOINT_SAFE_LOWER[15], H1_JOINT_SAFE_UPPER[15])
            r_s_roll = np.clip(np.arctan2(-r_elbow[1], -r_elbow[2]), H1_JOINT_SAFE_LOWER[16], H1_JOINT_SAFE_UPPER[16])

            upper_body = torch.tensor([0.0, l_s_pitch, l_s_roll, 0.0, 0.52, r_s_pitch, r_s_roll, 0.0, 0.52], dtype=torch.float32)

            # Combine leg (10) + torso/arms (9) = 19 DOFs
            q_full_t = torch.cat([q_leg_t[:5], q_leg_t[5:10], upper_body])

            q_legs.append(q_full_t)
            root_zs.append(rz_t)
            diagnostics.append(diag)

            prev_q = q_leg_t
            prev_rz = rz_t

        return torch.stack(q_legs, dim=0), torch.stack(root_zs, dim=0), diagnostics
