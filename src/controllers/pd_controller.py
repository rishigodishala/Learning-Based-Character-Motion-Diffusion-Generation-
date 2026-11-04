"""
Explicit Project Joint PD Controller (Option B).
Phases 10-13: Fixed explicit joint position PD controller executing target commands into motor torques.
"""

import torch
import torch.nn as nn
from typing import Tuple


class ExplicitJointPDController(nn.Module):
    """
    Computes motor joint torques using documented nominal Unitree H1 gains.
    Formula: tau = clip( Kp * (q_target - q_current) - Kd * qdot_current, -tau_max, tau_max )
    """
    def __init__(self, device: str = "cpu"):
        super().__init__()
        
        # Nominal H1 PD Gains & Limits (19 Joints)
        # Order: [left_hip_yaw, left_hip_roll, left_hip_pitch, left_knee, left_ankle,
        #        right_hip_yaw, right_hip_roll, right_hip_pitch, right_knee, right_ankle,
        #        torso,
        #        left_shoulder_pitch, left_shoulder_roll, left_shoulder_yaw, left_elbow,
        #        right_shoulder_pitch, right_shoulder_roll, right_shoulder_yaw, right_elbow]
        
        kp_gains = [
            200.0, 200.0, 200.0, 300.0, 80.0,  # Left Leg
            200.0, 200.0, 200.0, 300.0, 80.0,  # Right Leg
            200.0,                              # Torso
            40.0, 40.0, 40.0, 40.0,            # Left Arm
            40.0, 40.0, 40.0, 40.0             # Right Arm
        ]

        kd_gains = [
            5.0, 5.0, 5.0, 6.0, 2.0,  # Left Leg
            5.0, 5.0, 5.0, 6.0, 2.0,  # Right Leg
            5.0,                      # Torso
            1.0, 1.0, 1.0, 1.0,       # Left Arm
            1.0, 1.0, 1.0, 1.0        # Right Arm
        ]

        tau_limits = [
            200.0, 200.0, 200.0, 300.0, 40.0,  # Left Leg
            200.0, 200.0, 200.0, 300.0, 40.0,  # Right Leg
            200.0,                              # Torso
            40.0, 40.0, 40.0, 40.0,            # Left Arm
            40.0, 40.0, 40.0, 40.0             # Right Arm
        ]

        self.kp = nn.Parameter(torch.tensor(kp_gains, dtype=torch.float32, device=device), requires_grad=False)
        self.kd = nn.Parameter(torch.tensor(kd_gains, dtype=torch.float32, device=device), requires_grad=False)
        self.tau_max = nn.Parameter(torch.tensor(tau_limits, dtype=torch.float32, device=device), requires_grad=False)

    def compute_torques(
        self, 
        q_target: torch.Tensor, 
        q_current: torch.Tensor, 
        qdot_current: torch.Tensor
    ) -> torch.Tensor:
        """
        Computes joint torques for single state or batched environment states.
        
        Args:
            q_target: Target joint positions of shape (..., 19)
            q_current: Current joint positions of shape (..., 19)
            qdot_current: Current joint velocities of shape (..., 19)
        Returns:
            tau: Computed joint torques of shape (..., 19)
        """
        pos_error = q_target - q_current
        raw_tau = self.kp * pos_error - self.kd * qdot_current
        clipped_tau = torch.clamp(raw_tau, -self.tau_max, self.tau_max)
        return clipped_tau
