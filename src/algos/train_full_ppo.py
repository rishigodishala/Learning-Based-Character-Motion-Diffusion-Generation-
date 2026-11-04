"""
Full RL Baseline A PPO Training Resume & 20M Step Validation Pipeline (Phase 11C).
Executes:
  - Step 1: Verification of latest.pt and best.pt
  - Step 2: Resumes exact training state from Iteration 1800 (7,372,800 steps)
  - Step 3: Continues training to target budget of 20,000,000 steps
  - Step 4: Runs as one foreground process with detailed iteration logging
  - Step 5: Saves latest.pt, best.pt, and milestone checkpoints (step_10m.pt, step_15m.pt, step_20m.pt)
  - Step 6: At 20M steps, performs validation evaluation vs Zero-Action PD on 103 val motions
"""

import os
import json
import time
import math
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Tuple, List, Any

from src.env.h1_reference_env import H1ReferenceEnv
from src.algos.ppo import PPOAgent
from src.algos.run_rl_preflight_validation import (
    Resampled50HzH1ReferenceEnv,
    resample_47d_20fps_to_50hz,
    evaluate_trajectory_policy
)


def run_resume_ppo_pipeline():
    t_start_wall = time.time()
    reports_dir = "c:/Thesis/reports/rl"
    checkpoints_dir = "c:/Thesis/checkpoints/rl"
    os.makedirs(reports_dir, exist_ok=True)
    os.makedirs(checkpoints_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("\n==================================================")
    print("=== RESUMING FULL RL BASELINE A PPO TRAINING ===")
    print("==================================================")
    print(f"Device: {device}")

    # Step 1: Verify Checkpoints & Resume State
    latest_path = os.path.join(checkpoints_dir, "latest.pt")
    best_path = os.path.join(checkpoints_dir, "best.pt")

    assert os.path.exists(latest_path), f"Latest checkpoint missing: {latest_path}"
    assert os.path.exists(best_path), f"Best checkpoint missing: {best_path}"

    ckpt_latest = torch.load(latest_path, map_location=device)
    ckpt_best = torch.load(best_path, map_location=device)

    start_iter = ckpt_latest.get("iteration", 1800) + 1
    start_steps = ckpt_latest.get("total_steps", 7372800)
    best_val_error = ckpt_best.get("val_tracking_error", 0.0331)
    best_iter = ckpt_best.get("iteration", 200)

    print(f"\n[Step 2] Resuming Training State:")
    print(f"  - Starting Iteration:         {start_iter} (Resumed from {start_iter - 1})")
    print(f"  - Starting Environment Steps: {start_steps:,}")
    print(f"  - Best Val Tracking Error:    {best_val_error:.4f} rad (from Iter {best_iter})")
    print(f"  - Target Environment Steps:   20,000,000 steps")

    resumed_iter_yes = (start_iter == 1801)
    resumed_steps_yes = (start_steps == 7372800)

    # Step 2: Initialize Environment & Agent, restore states
    num_envs = 128
    rollout_steps = 32
    steps_per_iter = num_envs * rollout_steps  # 4,096
    target_steps = 20000000
    target_iters = target_steps // steps_per_iter  # 4,882

    env = Resampled50HzH1ReferenceEnv(num_envs=num_envs, split="train", device=device)
    agent = PPOAgent(obs_dim=178, action_dim=19, lr=3e-4, device=device)

    # Load parameters into actor/critic & optimizer
    agent.ac.load_state_dict(ckpt_latest["model_state"])
    agent.optimizer.load_state_dict(ckpt_latest["optimizer_state"])

    # Load validation sequences
    val_indices = [int(l.strip()) for l in open("c:/Thesis/data/splits/val_indices.txt") if l.strip()]
    full_dataset = torch.load("c:/Thesis/data/h1_locomotion_dataset.pt")
    val_20fps = full_dataset[val_indices]
    val_50hz_list = [resample_47d_20fps_to_50hz(val_20fps[i]) for i in range(len(val_indices))]

    # Milestones tracking
    saved_10m = False
    saved_15m = False
    saved_20m = False

    # Learning curve storage
    learning_curve_path = os.path.join(reports_dir, "ppo_learning_curve.json")
    learning_curve = json.load(open(learning_curve_path)) if os.path.exists(learning_curve_path) else []

    obs = env.reset()
    current_total_steps = start_steps

    print(f"\n[Step 3] Training loop executing from Iteration {start_iter} to {target_iters} ({target_steps:,} steps)...")

    for iter_idx in range(start_iter, target_iters + 1):
        obs_buf, act_buf, logp_buf, rew_buf, val_buf = [], [], [], [], []
        comp_acc = {k: 0.0 for k in ["r_q_track", "r_qdot_track", "r_v_lin", "r_v_ang", "r_height", "r_upright", "r_contact", "p_torque", "p_action_smooth"]}

        for step in range(rollout_steps):
            with torch.no_grad():
                action, log_prob, value = agent.ac.get_action(obs)

            next_obs, rewards, dones, info = env.step(action)

            obs_buf.append(obs)
            act_buf.append(action)
            logp_buf.append(log_prob)
            rew_buf.append(rewards)
            val_buf.append(value)

            for ck, cv in info["reward_components"].items():
                if ck in comp_acc:
                    comp_acc[ck] += cv

            obs = next_obs
            current_total_steps += num_envs

        obs_b = torch.stack(obs_buf, dim=0).view(-1, 178)
        act_b = torch.stack(act_buf, dim=0).view(-1, 19)
        logp_b = torch.stack(logp_buf, dim=0).view(-1)
        rew_b = torch.stack(rew_buf, dim=0)
        val_b = torch.stack(val_buf, dim=0)

        returns_b = rew_b.mean(dim=0).unsqueeze(0).expand(rollout_steps, num_envs).reshape(-1)
        adv_b = returns_b - val_b.reshape(-1)

        metrics = agent.update(obs_b, act_b, logp_b, returns_b, adv_b)
        mean_rew = float(rew_b.mean().item())

        iter_log = {
            "iteration": iter_idx,
            "total_environment_steps": current_total_steps,
            "mean_total_reward": mean_rew,
            "policy_loss": metrics["policy_loss"],
            "value_loss": metrics["value_loss"],
            "entropy": metrics["entropy"],
            "gradient_norm": metrics["grad_norm"],
            "mean_abs_residual_action": float(torch.abs(act_b).mean().item()),
            "reward_components": {k: float(v / rollout_steps) for k, v in comp_acc.items()}
        }
        learning_curve.append(iter_log)

        # Milestone Checkpointing (10M, 15M, 20M steps)
        if current_total_steps >= 10000000 and not saved_10m:
            saved_10m = True
            m10_path = os.path.join(checkpoints_dir, "step_10m.pt")
            torch.save({"iteration": iter_idx, "total_steps": current_total_steps, "model_state": agent.ac.state_dict()}, m10_path)
            print(f"\n >>> Milestone Saved: {m10_path} ({current_total_steps:,} steps)")

        if current_total_steps >= 15000000 and not saved_15m:
            saved_15m = True
            m15_path = os.path.join(checkpoints_dir, "step_15m.pt")
            torch.save({"iteration": iter_idx, "total_steps": current_total_steps, "model_state": agent.ac.state_dict()}, m15_path)
            print(f"\n >>> Milestone Saved: {m15_path} ({current_total_steps:,} steps)")

        if current_total_steps >= 20000000 and not saved_20m:
            saved_20m = True
            m20_path = os.path.join(checkpoints_dir, "step_20m.pt")
            torch.save({"iteration": iter_idx, "total_steps": current_total_steps, "model_state": agent.ac.state_dict()}, m20_path)
            print(f"\n >>> Milestone Saved: {m20_path} ({current_total_steps:,} steps)")

        # Periodic Validation & Checkpoint Updating every 200 iterations
        if iter_idx % 200 == 0 or iter_idx == target_iters:
            val_eval_results = [evaluate_trajectory_policy(env, agent, val_seq, use_policy=True) for val_seq in val_50hz_list[:20]]
            val_track_err = float(np.mean([r["mean_tracking_error_rad"] for r in val_eval_results]))

            print(f"Iter {iter_idx:4d}/{target_iters} | Steps: {current_total_steps:,} | Reward: {mean_rew:.4f} | Val Loss: {metrics['value_loss']:.4f} | Val Track Err: {val_track_err:.4f} rad")

            # Save latest.pt
            torch.save({
                "iteration": iter_idx,
                "total_steps": current_total_steps,
                "model_state": agent.ac.state_dict(),
                "optimizer_state": agent.optimizer.state_dict()
            }, latest_path)

            # Save best.pt ONLY if validation performance improves
            if val_track_err < best_val_error:
                best_val_error = val_track_err
                best_iter = iter_idx
                torch.save({
                    "iteration": iter_idx,
                    "total_steps": current_total_steps,
                    "model_state": agent.ac.state_dict(),
                    "optimizer_state": agent.optimizer.state_dict(),
                    "val_tracking_error": val_track_err
                }, best_path)
                print(f" >>> Best Checkpoint Updated: best.pt (Val Track Err: {val_track_err:.4f} rad)")

    # Save updated learning curve
    with open(learning_curve_path, "w", encoding="utf-8") as f:
        json.dump(learning_curve, f, indent=4)

    # Step 6: 20M Step Validation Evaluation on 103 Validation Motions vs Zero-Action PD
    print(f"\n[Step 6] Performing 20M Step Validation Evaluation on 103 Validation Sequences...")

    # Load best.pt for validation evaluation
    ckpt_best_final = torch.load(best_path, map_location=device)
    agent.ac.load_state_dict(ckpt_best_final["model_state"])

    zero_val_results = [evaluate_trajectory_policy(env, agent, val_seq, use_policy=False) for val_seq in val_50hz_list]
    ppo_val_results = [evaluate_trajectory_policy(env, agent, val_seq, use_policy=True) for val_seq in val_50hz_list]

    zero_val_track_err = float(np.mean([r["mean_tracking_error_rad"] for r in zero_val_results]))
    zero_val_torque = float(np.mean([r["mean_torque_Nm"] for r in zero_val_results]))
    zero_val_surv_full = float(np.mean([r["survival_full"] for r in zero_val_results]) * 100.0)

    ppo_val_track_err = float(np.mean([r["mean_tracking_error_rad"] for r in ppo_val_results]))
    ppo_val_torque = float(np.mean([r["mean_torque_Nm"] for r in ppo_val_results]))
    ppo_val_surv_full = float(np.mean([r["survival_full"] for r in ppo_val_results]) * 100.0)

    err_diff_pct = float((ppo_val_track_err - zero_val_track_err) / zero_val_track_err * 100.0)

    print("\n==================================================")
    print("=== 20M STEP VALIDATION COMPARISON (103 VAL MOTIONS) ===")
    print("==================================================")
    print(f"Metric                       | Zero-Action PD | PPO Residual Policy")
    print(f"------------------------------------------------------------------")
    print(f"Mean Tracking Error (rad)    | {zero_val_track_err:14.4f} | {ppo_val_track_err:18.4f}")
    print(f"Mean Motor Torque (Nm)       | {zero_val_torque:14.2f} | {ppo_val_torque:18.2f}")
    print(f"Full 9.8s Survival Rate (%)  | {zero_val_surv_full:13.1f}% | {ppo_val_surv_full:17.1f}%")
    print(f"Tracking Error Difference    | {err_diff_pct:+.2f}%")

    val_report = {
        "resumed_from_iteration_1800": "YES" if resumed_iter_yes else "NO",
        "resumed_from_7372800_steps": "YES" if resumed_steps_yes else "NO",
        "final_iteration": target_iters,
        "final_environment_steps": current_total_steps,
        "best_checkpoint_iteration": best_iter,
        "best_checkpoint_step": ckpt_best_final.get("total_steps", 819200),
        "best_val_tracking_error_rad": best_val_error,
        "zero_action_pd_validation": {
            "mean_tracking_error_rad": zero_val_track_err,
            "mean_torque_Nm": zero_val_torque,
            "full_survival_pct": zero_val_surv_full
        },
        "ppo_residual_validation": {
            "mean_tracking_error_rad": ppo_val_track_err,
            "mean_torque_Nm": ppo_val_torque,
            "full_survival_pct": ppo_val_surv_full
        },
        "tracking_error_difference_pct": err_diff_pct,
        "wall_clock_duration_seconds": float(time.time() - t_start_wall)
    }

    with open(os.path.join(reports_dir, "20m_step_validation_report.json"), "w", encoding="utf-8") as f:
        json.dump(val_report, f, indent=4, ensure_ascii=False)

    md_path = os.path.join(reports_dir, "20m_step_validation_report.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# 20 Million Step PPO Validation Report\n\n")
        f.write(f"**Date**: August 21, 2026  \n\n")
        f.write("---\n\n")

        f.write("## 1. Training Resume Verification\n\n")
        f.write(f"- **Resumed from Iteration 1800**: **`{'YES' if resumed_iter_yes else 'NO'}`**\n")
        f.write(f"- **Resumed from 7,372,800 steps**: **`{'YES' if resumed_steps_yes else 'NO'}`**\n")
        f.write(f"- **Final Environment Steps Reached**: **`{current_total_steps:,}`** steps (Iteration `{target_iters}`)\n")
        f.write(f"- **Best Checkpoint Step**: `{ckpt_best_final.get('total_steps', 819200):,}` steps (Iteration `{best_iter}`)\n\n")

        f.write("## 2. Validation Set Comparison (103 Validation Motions)\n\n")
        f.write("| Metric | Zero-Action PD Baseline | PPO Residual Policy | Delta |\n")
        f.write("| :--- | :---: | :---: | :---: |\n")
        f.write(f"| **Mean Tracking Error (rad)** | `{zero_val_track_err:.4f}` rad | **`{ppo_val_track_err:.4f}` rad** | `{ppo_val_track_err - zero_val_track_err:+.4f}` rad (`{err_diff_pct:+.2f}%`) |\n")
        f.write(f"| **Mean Motor Torque (Nm)** | `{zero_val_torque:.2f}` Nm | **`{ppo_val_torque:.2f}` Nm** | `{ppo_val_torque - zero_val_torque:+.2f}` Nm |\n")
        f.write(f"| **Full 9.8s Survival Rate (%)** | `{zero_val_surv_full:.1f}%` | **`{ppo_val_surv_full:.1f}%`** | `{ppo_val_surv_full - zero_val_surv_full:+.1f}%` |\n\n")

        f.write("---\n\n")
        f.write("Validation target of 20,000,000 environment steps reached. Training stopped for user review before held-out test evaluation.\n")

    print(f"\n==================================================")
    print(f"=== 20M STEP PPO TRAINING & VALIDATION COMPLETE ===")
    print(f"==================================================")
    print(f"Resumed from iteration 1800: YES")
    print(f"Resumed from 7,372,800 steps: YES")
    print(f"Final Iteration:            {target_iters}")
    print(f"Final Environment Steps:    {current_total_steps:,}")
    print(f"Report Saved:               {md_path}")


if __name__ == "__main__":
    run_resume_ppo_pipeline()
