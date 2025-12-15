# Group 7 - Pong World Tournament Submission

This repository contains the implementation for **Part 2** of the Machine Learning paradigms project. Our submission focuses on solving the **PettingZoo Pong** multi-agent environment using **Proximal Policy Optimization (PPO)**.

The project utilizes the standard **AEC API** and **SuperSuit wrappers** as specified in the course task requirements.

## 📂 Project Structure

The project is organized into three main directories distinguishing between baselines, training experiments, and the final tournament submission.

```text
/Part 2
├── Baseline/
│   ├── baseline_Naive_random_vs_random.py  # Benchmark: Training from scratch (Self-play)
│   ├── baseline_teacher.py                 # Benchmark: Transfer learning from Atari AI
│   └── baseline_teacher_sweep.py           # Hyperparameter optimization (WandB)
│
├── Tournament/
│   └── Group_7_model/                      # FINAL SUBMISSION PACKAGE
│       ├── __init__.py
│       ├── load_agents.py                  # Script to load agents for play.py
│       ├── round_200_Left.zip              # Final trained Left Agent
│       └── round_200_Right.zip             # Final trained Right Agent
│
└── Training/
    ├── mirror_training.py                  # Experiment A: Generational Ladder
    ├── left_rigth_training.py              # Experiment: Species-Specific Ladder
    └── train_simultaneous.py               # Experiment B: Simultaneous Rapid-Switching (Selected)
```

## 🛠️ Environment & Dependencies

This implementation strictly adheres to the environment specifications provided in the task description. We use **PettingZoo** with **SuperSuit** preprocessing.
1. Run the Tournament Submission (Verification)
To verify the final agents using the teacher's evaluation script (play.py), follow these steps:

Ensure play.py, wrappers.py, and utils.py are in your root directory.

Place the Group_7_model folder in the same directory.

## 🚀 How to Run
Run the following command to pitch the agent against itself:

```text

# Syntax: python play.py <left_package> <right_package> <match_id>
python play.py Group_7_model Group_7_model 1
Results: Scores will be saved to logs/ and a replay video to videos/.
```

### 2. Run Training (Method B)

To reproduce our selected training methodology (**Simultaneous Rapid-Switching**):

```bash
# Run the training loop (Method B)
python Training/train_simultaneous.py
```
Output: This script generates models in a models/ directory and logs in logs/.

Note: Uses DynamicOpponentWrapper to switch training between Left and Right agents every 50,000 steps.

### 3. Run Baselines
To train the "Teacher" agent (Single-player transfer learning):

```Bash

python Baseline/baseline_teacher.py

```
### 🧠 Model & Strategy Overview
Architecture: We utilize PPO (Proximal Policy Optimization) from Stable Baselines 3 due to its stability in non-stationary multi-agent environments.

Training Method: The submitted agents were trained using Method B (Simultaneous Rapid-Switching), which mitigates overfitting by frequently updating the opponent.

Custom Wrappers:

Safety Masking: We mask the top 10 pixels of the observation to hide the scoreboard, preventing confusion from asymmetric numbers.

Anti-Freeze: A heuristic in load_agents.py forces the agent to serve if inactive for 100 frames, preventing "Safety Stalemates."