# Learning-Based Character Locomotion: Comparative Evaluation of Reinforcement Learning, Diffusion Models, and Hybrid Execution under Challenging Physical Conditions

**Master of Science in Computer Science Thesis Project**  
*Blekinge Institute of Technology (BTH), Karlskrona, Sweden*  
**Authors**: Rishi Godishala, Devi Harsh Dhanukonda  
**Supervisor**: Dr. Suejb Memeti  

---

## Overview

This repository contains the official, clean implementation of the end-to-end research pipeline for humanoid locomotion on the **Unitree H1 (19 actuated degrees of freedom)**. The framework integrates:
1. **Motion Data Processing & Semantic Locomotion Filtering**: Semantic filtering of HumanML3D motion descriptions and mapping to raw AMASS mocap sequences.
2. **Bilateral Contact-Aware Inverse Kinematics (IK)**: Morphological scaling and constrained lower-body IK with contact preservation and joint safety margins.
3. **Motion Diffusion Model (MDM)**: An unconditional 8-layer Transformer denoising diffusion model ($26.68\text{M}$ trainable parameters, 1000 DDPM steps) generating 47-dimensional H1 motion trajectories.
4. **PPO Residual Controller (Baseline A)**: Reference-conditioned reinforcement learning controller operating on 178D observation space with a 19D residual action space ($s_a = 0.25\text{ rad}$).
5. **Low-Level Motor PD Control**: Explicit joint-space PD controller enforcing nominal H1 stiffness/damping gains ($K_p, K_d$) and actuator saturation torque limits ($\tau_{\text{max}}$).
6. **Controlled Physics & Perturbation Evaluation**: Systematic evaluation comparing direct PD tracking (**Baseline B**) against PPO-augmented execution (**Hybrid Baseline C**) across eleven physical perturbation conditions (friction variations, terrain slopes, rough ground, external pushes, and initial state perturbations).

---

## Repository Structure

```text
h1-diffusion-rl-locomotion/
│
├── configs/
│   └── diffusion.yaml                  # Model hyperparameters and diffusion training configuration
│
├── data/
│   └── splits/                         # Official train / val / test dataset split indices
│       ├── train_indices.txt           # 1,530 training motion indices
│       ├── val_indices.txt             # 103 validation motion indices
│       └── test_indices.txt            # 309 held-out test motion indices
│
├── src/
│   ├── data/                           # 1. DATA PREPROCESSING & RETARGETING
│   │   ├── filter_locomotion.py        # Semantic filtering of locomotion from HumanML3D text
│   │   ├── coordinate_transform.py     # Coordinate transformation (HumanML3D Y-up -> Isaac Z-up)
│   │   ├── retarget_smpl_to_h1.py      # Bilateral contact-aware inverse kinematics (IK) solver
│   │   ├── build_full_h1_dataset.py    # AMASS .npz -> FK -> IK -> 47D tensor [1942, 196, 47]
│   │   ├── compute_normalization.py    # Training split mean & std computation for diffusion
│   │   ├── h1_motion_dataset.py        # PyTorch Dataset class for loading 47D motion sequences
│   │   └── dataset_h1.py               # Feature matrix assembly and padding utilities
│   │
│   ├── models/                         # 2. DIFFUSION MODEL TRAINING & SAMPLING
│   │   ├── diffusion_mdm.py            # MotionDiffusionTransformer & GaussianDiffusionManager
│   │   └── train_diffusion.py          # Diffusion training pipeline & 100-trajectory offline generator
│   │
│   ├── algos/                          # 3. REINFORCEMENT LEARNING (PPO RESIDUAL CONTROLLER)
│   │   ├── ppo.py                      # ActorCriticMLP architecture & PPOAgent optimization
│   │   ├── train_full_ppo.py           # PPO policy training pipeline (20M environment steps)
│   │   └── run_rl_preflight_validation.py # Vectorized 50Hz Resampled Env & observation building
│   │
│   ├── env/                            # 4. SIMULATION ENVIRONMENTS
│   │   └── h1_reference_env.py         # Vectorized Reference-Conditioned H1 Locomotion Environment
│   │
│   ├── controllers/                    # 5. LOW-LEVEL MOTOR CONTROL
│   │   └── pd_controller.py            # ExplicitJointPDController (19-DOF Unitree H1 gains & torque limits)
│   │
│   └── evaluation/                     # 6. EXPERIMENTAL EVALUATION
│       ├── run_phase_13_final_evaluation.py  # Master evaluation (Baseline B vs Hybrid C across 11 conditions)
│       ├── run_baseline_b_evaluation.py      # Standalone Baseline B (Diffusion -> PD) evaluation
│       ├── run_phase_11d_robustness_evaluation.py # Baseline A (Ground-Truth -> PPO -> PD) robustness evaluation
│       └── compute_bh_correction.py          # Benjamini–Hochberg FDR statistical correction
│
├── .gitignore                          # Standard Python / PyTorch / Isaac Sim gitignore
├── requirements.txt                    # Python environment dependencies
└── README.md                           # Documentation, pipeline execution guide, and citation
```

---

## 47-Dimensional Motion Representation

Each frame is represented by a 47-dimensional feature vector at 20 Hz (duration: 196 frames = 9.80 seconds):
* **0–2** (3D): Reserved root linear velocity channels (zero placeholders)
* **3–5** (3D): Reserved root angular velocity channels (zero placeholders)
* **6** (1D): Optimized root height ($z_{\text{root}}$)
* **7–25** (19D): Actuated H1 joint positions ($\mathbf{q}_t$)
* **26–44** (19D): Actuated H1 joint velocities ($\dot{\mathbf{q}}_t$)
* **45–46** (2D): Binary left and right foot contact indicators ($c_{\text{left}}, c_{\text{right}}$)

Joint ordering (19 actuated joints):
`left_hip_yaw`, `left_hip_roll`, `left_hip_pitch`, `left_knee`, `left_ankle`, `right_hip_yaw`, `right_hip_roll`, `right_hip_pitch`, `right_knee`, `right_ankle`, `torso`, `left_shoulder_pitch`, `left_shoulder_roll`, `left_shoulder_yaw`, `left_elbow`, `right_shoulder_pitch`, `right_shoulder_roll`, `right_shoulder_yaw`, `right_elbow`.

---

## Getting Started

### 1. Installation

Create a virtual environment and install the required dependencies:
```bash
python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On Linux / macOS:
source venv/bin/activate

pip install -r requirements.txt
```

---

## Execution Pipeline

### Stage 1: Data Preprocessing & Retargeting
1. **Semantic Locomotion Filtering**:
   Filter HumanML3D text annotations into walking, turning, jogging/running, backward, and general locomotion:
   ```bash
   python src/data/filter_locomotion.py
   ```
2. **Build Full Retargeted H1 Dataset**:
   Convert raw AMASS `.npz` files through forward kinematics and bilateral contact-aware IK into `[1942, 196, 47]` tensor:
   ```bash
   python src/data/build_full_h1_dataset.py
   ```
3. **Compute Normalization Statistics**:
   Calculate training split mean and standard deviation:
   ```bash
   python src/data/compute_normalization.py
   ```

---

### Stage 2: Motion Diffusion Training & Offline Trajectory Generation
Train the 8-layer Transformer denoising model on the 1,530 training split motions and generate 100 unconditional 9.8-second evaluation trajectories:
```bash
python src/models/train_diffusion.py
```
* **Configuration**: `configs/diffusion.yaml`
* **Output Checkpoints**: `checkpoints/diffusion/best.pt`, `checkpoints/diffusion/latest.pt`
* **Generated Trajectories**: `data/generated/diffusion_samples.pt` (`[100, 196, 47]`)

---

### Stage 3: PPO Residual Policy Training
Train the reference-conditioned PPO residual policy on retargeted motions (20,000,000 environment interaction steps):
```bash
python src/algos/train_full_ppo.py
```
* **Observation Space**: 178D (68D simulated state + 110D multi-horizon reference block)
* **Action Space**: 19D residual joint targets ($\pm 0.25\text{ rad}$)
* **Output Checkpoints**: `checkpoints/rl/best.pt` (validation-selected at Iteration 200, 819.2K steps)

---

### Stage 4: Comparative Evaluation Across 11 Physical Conditions
Run the authoritative evaluation comparing **Baseline B** (Diffusion $\to$ PD) and **Hybrid Baseline C** (Diffusion $\to$ PPO $\to$ PD) across 2,200 total physics rollouts:
```bash
python src/evaluation/run_phase_13_final_evaluation.py
```

Apply Benjamini–Hochberg False Discovery Rate (FDR) statistical corrections across the evaluation condition family:
```bash
python src/evaluation/compute_bh_correction.py
```

---

## Key Experimental Results

* **Nominal Execution ($N=100$)**:
  * **Baseline B (Diffusion $\to$ PD)**: 100% full-horizon survival, $0.0304\text{ rad}$ ($1.74^\circ$) mean joint tracking error, $4.89\text{ Nm}$ mean motor torque.
  * **Hybrid C (Diffusion $\to$ PPO $\to$ PD)**: 100% full-horizon survival, $0.0355\text{ rad}$ ($2.03^\circ$) mean joint tracking error, $5.77\text{ Nm}$ mean motor torque.
  * **Paired Difference**: $\bar{\Delta E} = +0.0051\text{ rad}$ ($+16.78\%$, $p = 1.85 \times 10^{-63}$, $d_z = 4.074$).
* **Robustness Parity**:
  * Both controllers achieved **100% identical survival rates** across all eleven physical perturbation conditions (including 94% under $\mu = 0.2$ low friction and 0% under $5.0^\circ$ terrain slope).
  * The frozen, independently trained PPO residual policy added no full-horizon survival advantage over explicit PD control alone.

---

## License & Attribution

This research code is developed for academic reproducibility. Third-party datasets (HumanML3D, AMASS) and robot models (Unitree H1) remain subject to their respective licenses and terms of use.
