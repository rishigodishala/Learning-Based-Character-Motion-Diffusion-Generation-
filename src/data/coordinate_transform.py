"""
HumanML3D to Isaac Lab / Unitree H1 Coordinate Transformation Module.
Phase 3: Converts Y-Up, XZ-Ground (HumanML3D) coordinates to Z-Up, XY-Ground (Isaac Lab / H1) coordinates.
"""

import torch
import numpy as np
from typing import Union


# Standard Transformation Matrix: HumanML3D (Y-Up) -> Isaac Lab / H1 (Z-Up)
# X_isaac = Z_humanml3d
# Y_isaac = -X_humanml3d
# Z_isaac = Y_humanml3d
R_HUMANML3D_TO_ISAAC_NP = np.array([
    [0.0, 0.0, 1.0],
    [-1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0]
], dtype=np.float32)

R_HUMANML3D_TO_ISAAC_TORCH = torch.tensor(R_HUMANML3D_TO_ISAAC_NP, dtype=torch.float32)


def transform_positions_np(pos: np.ndarray) -> np.ndarray:
    """
    Transforms 3D position array from HumanML3D (Y-up) to Isaac Lab (Z-up).
    
    Args:
        pos: Array of shape (..., 3)
    Returns:
        Transformed array of shape (..., 3)
    """
    return pos @ R_HUMANML3D_TO_ISAAC_NP.T


def transform_positions_torch(pos: torch.Tensor) -> torch.Tensor:
    """
    Transforms 3D position tensor from HumanML3D (Y-up) to Isaac Lab (Z-up).
    
    Args:
        pos: Tensor of shape (..., 3)
    Returns:
        Transformed tensor of shape (..., 3)
    """
    R = R_HUMANML3D_TO_ISAAC_TORCH.to(pos.device)
    return torch.matmul(pos, R.T)


def transform_velocities_torch(vel: torch.Tensor) -> torch.Tensor:
    """
    Transforms 3D velocity tensor from HumanML3D (Y-up) to Isaac Lab (Z-up).
    """
    return transform_positions_torch(vel)


def transform_orientations_torch(quat_y_up: torch.Tensor) -> torch.Tensor:
    """
    Transforms quaternions (w, x, y, z) from HumanML3D frame to Isaac Lab frame.
    """
    # Simple conversion via rotation matrix sandwich
    # R_new = R_trans * R_old * R_trans^T
    pass
