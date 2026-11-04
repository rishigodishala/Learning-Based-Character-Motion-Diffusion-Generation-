"""
HumanML3D Locomotion Data Extractor & Filter.
Phase 2: Multi-stage locomotion dataset extraction and statistics generation.
"""

import os
import csv
import json
import argparse
from typing import Dict, List, Set, Tuple
import numpy as np


# Configurable keyword categories for locomotion text filtering
DEFAULT_LOCOMOTION_KEYWORDS = {
    "walking": ["walk", "walking", "stroll", "pacing"],
    "turning": ["turn", "turning", "rotate", "curving"],
    "jogging_running": ["jog", "jogging", "run", "running", "sprint", "sprinting"],
    "backward": ["backward", "backstep", "back stepping", "reverse walk"],
    "locomotion_general": ["stride", "striding", "march", "marching", "stepping"]
}


def load_text_annotations(humanml3d_dir: str) -> Dict[str, List[str]]:
    """
    Loads text descriptions from texts/ directory as well as index.csv source_paths.
    
    Returns:
        Dict mapping motion_id (e.g. '000000') -> list of descriptions/metadata strings
    """
    motion_texts = {}
    
    # 1. Search texts/ subdirectories (e.g. humanml3d_dir/texts or humanml3d_dir/HumanML3D/texts)
    possible_text_dirs = [
        os.path.join(humanml3d_dir, "texts"),
        os.path.join(humanml3d_dir, "HumanML3D", "texts")
    ]
    
    texts_dir = None
    for d in possible_text_dirs:
        if os.path.exists(d) and len(os.listdir(d)) > 0:
            texts_dir = d
            break

    if texts_dir:
        print(f"[Info] Loading text annotations from: {texts_dir}")
        for filename in os.listdir(texts_dir):
            if not filename.endswith(".txt"):
                continue
            motion_id = filename.replace(".txt", "")
            file_path = os.path.join(texts_dir, filename)
            
            descriptions = []
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    parts = line.strip().split("#")
                    if parts:
                        desc = parts[0].strip().lower()
                        if desc:
                            descriptions.append(desc)
                            
            if descriptions:
                motion_texts[motion_id] = descriptions
    else:
        print("[Info] texts/ directory not yet extracted or empty. Falling back to index.csv source paths...")

    # 2. Augment / Fallback with index.csv if available
    index_csv = os.path.join(humanml3d_dir, "index.csv")
    if os.path.exists(index_csv):
        with open(index_csv, "r", encoding="utf-8", errors="ignore") as f:
            reader = csv.DictReader(f)
            for row in reader:
                new_name = row.get("new_name", "").replace(".npy", "")
                source_path = row.get("source_path", "").lower()
                if new_name and source_path:
                    if new_name not in motion_texts:
                        motion_texts[new_name] = [source_path]
                    else:
                        motion_texts[new_name].append(source_path)

    return motion_texts


def check_text_locomotion_match(
    descriptions: List[str], 
    keyword_map: Dict[str, List[str]]
) -> Tuple[bool, List[str]]:
    """
    Checks if any description matches any of the locomotion keywords.
    
    Returns:
        (is_locomotion_match, list of matched categories)
    """
    matched_categories = set()
    for desc in descriptions:
        for category, keywords in keyword_map.items():
            for kw in keywords:
                if kw in desc:
                    matched_categories.add(category)
                    
    return len(matched_categories) > 0, sorted(list(matched_categories))


def check_kinematic_velocity_threshold(
    vec_path: str, 
    min_velocity_threshold: float = 0.3
) -> Tuple[bool, float]:
    """
    Checks if the motion clip has root linear speed exceeding min_velocity_threshold.
    
    HumanML3D 263-dim vector format:
        Index 1: r_dot_x (root linear velocity X)
        Index 2: r_dot_z (root linear velocity Z)
    """
    if not os.path.exists(vec_path):
        return False, 0.0

    try:
        data = np.load(vec_path)
        if len(data.shape) != 2 or data.shape[1] < 3:
            return False, 0.0

        r_dot_x = data[:, 1]
        r_dot_z = data[:, 2]
        root_speed = np.sqrt(r_dot_x**2 + r_dot_z**2)
        mean_speed = float(np.mean(root_speed))

        return mean_speed >= min_velocity_threshold, mean_speed
    except Exception as e:
        print(f"[Error] Failed reading vector file {vec_path}: {e}")
        return False, 0.0


def filter_dataset(
    humanml3d_dir: str,
    output_dir: str,
    min_velocity_threshold: float = 0.3,
    keywords_config: Dict[str, List[str]] = DEFAULT_LOCOMOTION_KEYWORDS
) -> Dict:
    """
    Runs multi-stage dataset filtering and exports filtered motion list & report.
    """
    os.makedirs(output_dir, exist_ok=True)

    text_annos = load_text_annotations(humanml3d_dir)
    all_motion_ids = sorted(list(text_annos.keys()))
    total_original = len(all_motion_ids)

    passed_text_filter = []
    text_matched_categories = {cat: 0 for cat in keywords_config.keys()}

    for m_id, descs in text_annos.items():
        is_match, categories = check_text_locomotion_match(descs, keywords_config)
        if is_match:
            passed_text_filter.append(m_id)
            for cat in categories:
                text_matched_categories[cat] += 1

    # Search for vecs dir in root or nested HumanML3D
    possible_vecs_dirs = [
        os.path.join(humanml3d_dir, "new_joint_vecs"),
        os.path.join(humanml3d_dir, "HumanML3D", "new_joint_vecs")
    ]
    vecs_dir = None
    for d in possible_vecs_dirs:
        if os.path.exists(d):
            vecs_dir = d
            break

    passed_velocity_filter = []
    speeds = []

    for m_id in passed_text_filter:
        if vecs_dir:
            vec_path = os.path.join(vecs_dir, f"{m_id}.npy")
            if os.path.exists(vec_path):
                is_valid_vel, mean_speed = check_kinematic_velocity_threshold(vec_path, min_velocity_threshold)
                if is_valid_vel:
                    passed_velocity_filter.append(m_id)
                    speeds.append(mean_speed)
            else:
                passed_velocity_filter.append(m_id)
        else:
            passed_velocity_filter.append(m_id)

    report = {
        "dataset_statistics": {
            "total_original_sequences": total_original,
            "passed_text_filtering": len(passed_text_filter),
            "passed_velocity_filtering": len(passed_velocity_filter),
            "velocity_rejection_count": len(passed_text_filter) - len(passed_velocity_filter),
            "min_root_velocity_threshold_m_s": min_velocity_threshold,
            "average_root_speed_m_s": float(np.mean(speeds)) if speeds else 0.0
        },
        "category_distribution": text_matched_categories,
        "selected_motion_ids": passed_velocity_filter
    }

    # Save output list and report
    output_list_path = os.path.join(output_dir, "locomotion_motion_ids.txt")
    with open(output_list_path, "w") as f:
        for m_id in passed_velocity_filter:
            f.write(f"{m_id}\n")

    json_report_path = os.path.join(output_dir, "dataset_filtering_report.json")
    with open(json_report_path, "w") as f:
        json.dump(report, f, indent=4)

    txt_report_path = os.path.join(output_dir, "dataset_filtering_report.txt")
    with open(txt_report_path, "w") as f:
        f.write("=== HumanML3D Locomotion Filtering Report ===\n")
        f.write(f"Total Original Motions: {total_original}\n")
        f.write(f"Passed Text Filtering:  {len(passed_text_filter)}\n")
        f.write(f"Passed Velocity Filter: {len(passed_velocity_filter)}\n")
        f.write(f"Velocity Threshold:     {min_velocity_threshold} m/s\n")
        f.write(f"Average Root Speed:     {report['dataset_statistics']['average_root_speed_m_s']:.3f} m/s\n\n")
        f.write("Category Distribution:\n")
        for cat, count in text_matched_categories.items():
            f.write(f"  - {cat:20s}: {count}\n")

    print(f"[Phase 2 Complete] Report saved to: {json_report_path}")
    print(f"Filtered {len(passed_velocity_filter)} locomotion motion IDs to: {output_list_path}")

    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Filter HumanML3D dataset for locomotion motions.")
    parser.add_argument("--humanml3d_dir", type=str, default="c:/Thesis/HumanML3D", help="Path to HumanML3D repo")
    parser.add_argument("--output_dir", type=str, default="c:/Thesis/reports", help="Output directory for reports")
    parser.add_argument("--min_velocity", type=float, default=0.3, help="Minimum root velocity threshold (m/s)")
    args = parser.parse_args()

    filter_dataset(args.humanml3d_dir, args.output_dir, args.min_velocity)
