"""
H1 Motion Dataset Builder (47-Dimensional Feature Vector at 20 FPS).
Uses Bilateral Contact-Aware IK Solver to retarget SMPL motion to Unitree H1.
Preserves 196 frames at 20 FPS (9.8 seconds duration).
"""

import os
import json
import torch
import numpy as np
from typing import Dict, List, Tuple
from src.data.retarget_smpl_to_h1 import BilateralContactAwareIKSolver, H1_JOINT_NAMES


def build_47d_h1_feature_matrix_20fps(
    q_20fps: np.ndarray, 
    qdot_20fps: np.ndarray, 
    root_vel_20fps: np.ndarray,
    root_height_20fps: np.ndarray,
    foot_contacts_20fps: np.ndarray,
    target_length: int = 196
) -> np.ndarray:
    """
    Constructs the 47-dimensional H1 motion feature matrix X of shape (target_length, 47) at 20 FPS.
    """
    T = q_20fps.shape[0]
    
    if T > target_length:
        q_20fps = q_20fps[:target_length]
        qdot_20fps = qdot_20fps[:target_length]
        root_vel_20fps = root_vel_20fps[:target_length]
        root_height_20fps = root_height_20fps[:target_length]
        foot_contacts_20fps = foot_contacts_20fps[:target_length]
    elif T < target_length:
        pad_len = target_length - T
        q_20fps = np.pad(q_20fps, ((0, pad_len), (0, 0)), mode="edge")
        qdot_20fps = np.pad(qdot_20fps, ((0, pad_len), (0, 0)), mode="constant")
        root_vel_20fps = np.pad(root_vel_20fps, ((0, pad_len), (0, 0)), mode="constant")
        root_height_20fps = np.pad(root_height_20fps, ((0, pad_len), (0, 0)), mode="edge")
        foot_contacts_20fps = np.pad(foot_contacts_20fps, ((0, pad_len), (0, 0)), mode="edge")

    omega_root_20fps = np.zeros_like(root_vel_20fps)

    X = np.concatenate([
        root_vel_20fps,       # 3
        omega_root_20fps,     # 3
        root_height_20fps,    # 1
        q_20fps,              # 19
        qdot_20fps,           # 19
        foot_contacts_20fps   # 2
    ], axis=-1)

    return X.astype(np.float32)


def process_batch_h1_dataset(
    humanml3d_dir: str,
    locomotion_ids_file: str,
    output_pt_path: str,
    output_report_path: str,
    max_samples: int = None
) -> Dict:
    """
    Batch processes locomotion motion IDs into 47D dataset tensor at 20 FPS (196 frames = 9.8s).
    """
    os.makedirs(os.path.dirname(output_pt_path), exist_ok=True)
    solver = BilateralContactAwareIKSolver(num_iters=40)

    vecs_dir = os.path.join(humanml3d_dir, "HumanML3D", "new_joint_vecs")
    if not os.path.exists(vecs_dir):
        vecs_dir = os.path.join(humanml3d_dir, "new_joint_vecs")

    available_files = [os.path.splitext(f)[0] for f in os.listdir(vecs_dir) if f.endswith(".npy")]

    if max_samples and available_files:
        available_files = available_files[:max_samples]

    h1_sequences = []
    accepted_ids = []

    print(f"[Phase 6-7 Dataset Build] Processing {len(available_files)} locomotion sequences at 20 FPS (196 frames = 9.8s)...")

    for idx, m_id in enumerate(available_files):
        vec_path = os.path.join(vecs_dir, f"{m_id}.npy")
        if not os.path.exists(vec_path):
            continue

        data = np.load(vec_path)
        T_orig = data.shape[0]

        r_dot_x = data[:, 1:2]
        r_dot_z = data[:, 2:3]
        r_dot_y = np.zeros_like(r_dot_x)
        root_vel_20 = np.concatenate([r_dot_z, -r_dot_x, r_dot_y], axis=-1)

        j_pos_rel = data[:, 4:67].reshape(T_orig, 21, 3)
        pelvis = np.zeros((T_orig, 1, 3))
        smpl_positions = np.concatenate([pelvis, j_pos_rel], axis=1)

        if data.shape[1] >= 263:
            c_foot_4d = data[:, 259:263]
            c_foot_2d = torch.tensor(np.stack([
                (c_foot_4d[:, 0] + c_foot_4d[:, 1]) > 0.5,
                (c_foot_4d[:, 2] + c_foot_4d[:, 3]) > 0.5
            ], axis=-1), dtype=torch.float32)
        else:
            c_foot_2d = torch.ones((T_orig, 2), dtype=torch.float32)

        smpl_tensor = torch.tensor(smpl_positions, dtype=torch.float32)
        q_h1_t, root_zs_t, diags = solver.solve_sequence(smpl_tensor, c_foot_2d)

        q_20 = q_h1_t.numpy()
        root_h_20 = root_zs_t.numpy()[:, None]  # (T, 1)

        # Calculate joint velocities at 20 FPS via finite differences
        qdot_20 = np.zeros_like(q_20)
        qdot_20[1:-1] = (q_20[2:] - q_20[:-2]) / (2.0 * 0.05)
        qdot_20[0] = (q_20[1] - q_20[0]) / 0.05
        qdot_20[-1] = (q_20[-1] - q_20[-2]) / 0.05

        X_47d = build_47d_h1_feature_matrix_20fps(
            q_20fps=q_20,
            qdot_20fps=qdot_20,
            root_vel_20fps=root_vel_20,
            root_height_20fps=root_h_20,
            foot_contacts_20fps=c_foot_2d.numpy(),
            target_length=196
        )

        h1_sequences.append(X_47d)
        accepted_ids.append(m_id)

    # Save dataset tensor
    if h1_sequences:
        dataset_tensor = torch.tensor(np.stack(h1_sequences, axis=0), dtype=torch.float32)
        torch.save(dataset_tensor, output_pt_path)
        print(f"[Dataset Build Complete] Saved {dataset_tensor.shape} dataset tensor (20 FPS, 9.8s) to: {output_pt_path}")
    else:
        dataset_tensor = torch.empty((0, 196, 47))

    dataset_report = {
        "successfully_retargeted": len(accepted_ids),
        "dataset_tensor_shape": list(dataset_tensor.shape),
        "dataset_fps": 20,
        "sequence_duration_seconds": 9.8,
        "feature_dimension": 47,
        "sequence_length": 196
    }

    with open(output_report_path, "w") as f:
        json.dump(dataset_report, f, indent=4)

    return dataset_report


if __name__ == "__main__":
    humanml3d = "c:/Thesis/HumanML3D"
    locomotion_ids = "c:/Thesis/reports/locomotion_motion_ids.txt"
    out_pt = "c:/Thesis/data/h1_locomotion_dataset.pt"
    out_rep = "c:/Thesis/reports/h1_dataset_build_report.json"

    process_batch_h1_dataset(humanml3d, locomotion_ids, out_pt, out_rep)
