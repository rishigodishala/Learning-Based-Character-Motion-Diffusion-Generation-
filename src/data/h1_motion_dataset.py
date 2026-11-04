"""
PyTorch Dataset Loader for H1 Motion Diffusion (Step 3).
Loads dataset tensor C:/Thesis/data/h1_locomotion_dataset.pt [1942, 196, 47] and metadata.
Supports split="train", split="val", split="test" via official split index files.
Normalizes features using train split statistics (h1_train_mean.pt, h1_train_std.pt).
"""

import os
import json
import torch
from torch.utils.data import Dataset
from typing import Tuple, Dict, Any


class H1MotionDataset(Dataset):
    def __init__(self, split: str = "train", data_dir: str = "c:/Thesis/data", normalize: bool = True):
        super().__init__()
        self.split = split
        self.data_dir = data_dir
        self.normalize = normalize

        pt_path = os.path.join(data_dir, "h1_locomotion_dataset.pt")
        meta_path = os.path.join(data_dir, "h1_locomotion_metadata.json")

        assert os.path.exists(pt_path), f"Dataset tensor missing at {pt_path}"
        assert os.path.exists(meta_path), f"Metadata missing at {meta_path}"

        self.full_tensor = torch.load(pt_path)  # (1942, 196, 47)
        with open(meta_path, "r", encoding="utf-8") as f:
            self.metadata_list = json.load(f)

        split_file_map = {
            "train": os.path.join(data_dir, "splits", "train_indices.txt"),
            "val": os.path.join(data_dir, "splits", "val_indices.txt"),
            "test": os.path.join(data_dir, "splits", "test_indices.txt")
        }

        split_idx_path = split_file_map.get(split)
        assert os.path.exists(split_idx_path), f"Split index file missing: {split_idx_path}"

        self.indices = [int(l.strip()) for l in open(split_idx_path, encoding="utf-8") if l.strip()]

        # Load normalization stats
        norm_dir = os.path.join(data_dir, "normalization")
        mean_path = os.path.join(norm_dir, "h1_train_mean.pt")
        std_path = os.path.join(norm_dir, "h1_train_std.pt")

        assert os.path.exists(mean_path) and os.path.exists(std_path), "Normalization statistics missing. Run compute_normalization.py first."

        self.mean = torch.load(mean_path)  # (47,)
        self.std = torch.load(std_path)    # (47,)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, Any]]:
        tensor_idx = self.indices[idx]
        motion = self.full_tensor[tensor_idx]  # (196, 47)
        meta = self.metadata_list[tensor_idx]

        if self.normalize:
            motion = (motion - self.mean) / self.std

        return motion, meta

    def denormalize(self, motion_normalized: torch.Tensor) -> torch.Tensor:
        """
        Reverts normalized tensor back to original physical feature space.
        Args:
            motion_normalized: Tensor of shape (..., 196, 47) or (..., 47)
        """
        device = motion_normalized.device
        mean = self.mean.to(device)
        std = self.std.to(device)
        return motion_normalized * std + mean


if __name__ == "__main__":
    ds_train = H1MotionDataset(split="train")
    ds_val = H1MotionDataset(split="val")
    ds_test = H1MotionDataset(split="test")

    print(f"[Dataset Loader Test] Train samples: {len(ds_train)}, Val samples: {len(ds_val)}, Test samples: {len(ds_test)}")
    sample_tensor, sample_meta = ds_train[0]
    print(f"Sample tensor shape: {sample_tensor.shape}, HumanML3D ID: {sample_meta['humanml3d_id']}")
    print(f"Sample tensor mean: {sample_tensor.mean():.4f}, std: {sample_tensor.std():.4f}")
