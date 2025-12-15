## Part 3: Solving a “complex” ALE environment

This part of the project focuses on **comparing multiple Reinforcement Learning algorithms** on the **Atari Air Raid environment**. Three different algorithms are used:

1. **Proximal Policy Optimization (PPO)**
2. **Advantage Actor-Critic (A2C)**
3. **Deep Q-Network (DQN)**

The main objectives of this part are:

- To train and evaluate different RL algorithms on the same environment.
- To log training metrics and evaluation results in a standardized way.
- To visually compare agent performance through recorded gameplay videos.

All algorithms follow a **shared logging structure** and are executed using common utilities located in the `scripts/` folder.

-----

## Folder Structure Overview

```
├── part 3/
│   ├── ppo/
│   │   ├── logs_ppo_air_raid/
│   │   │   ├── best_model/
│   │   │   │   ├── best_model.zip
│   │   │   │   ├── evaluations.npz
│   │   │   ├── videos/
│   │   │   │   ├── step_600016.mp4
│   │   │   │   ├── step_3000016.mp4
│   │   │   │   ├── ...
│   ├── a2c/
│   │   ├── logs_a2c_air_raid/
│   ├── dqn/
│   │   ├── logs_dqn_air_raid/
│   ├── scripts/
│   │   ├── 3models.py
│   │   ├── wrappers.py
│   └── requirements.txt
```

> **Note:** The `ppo/`, `a2c/`, and `dqn/` folders follow the same internal structure.

-----

## PPO / A2C / DQN Experiment Folders

Each algorithm folder (`ppo/`, `a2c/`, `dqn/`) contains logs and artifacts generated during training and evaluation on the **Air Raid** environment.

### `logs_<algorithm>_air_raid/`

This directory stores **all outputs related to training and evaluation** for a specific algorithm.

Include:

- Saved best-performing models.
- Evaluation files.
- Gameplay videos recorded at different training steps.

-----

### `best_model/`

Contains the **best model checkpoint**, selected according to an evaluation criterion (e.g., average reward).

- `best_model.zip`  
  Serialized model checkpoint corresponding to the best evaluation performance.

- `evaluations.npz`  
  NumPy archive storing evaluation results over time (e.g., episode rewards, timesteps).

-----

### `videos/`

Contains **recorded evaluation episodes** of the trained agent.

- `step_600016.mp4`  
  Evaluation video recorded after 600,016 environment steps.

- `step_3000016.mp4`  
  Evaluation video recorded after 3,000,016 environment steps.

- Additional video files correspond to other evaluation checkpoints.

These videos allow for **qualitative comparison** of agent performance across algorithms and training stages.

-----

## `scripts/` Folder

This folder contains **shared utilities and execution scripts** used by all three algorithms.

### `3models.py`

This script is responsible for **running and/or comparing the three RL algorithms (PPO, A2C, and DQN)** on the Air Raid environment.

**Key responsibilities:**

- Creating the Air Raid environment.
- Initializing PPO, A2C, and DQN agents with comparable settings.
- Launching training for each algorithm.
- Configuring logging, evaluation callbacks, and video recording.
- Saving models and logs to their respective folders and in wandb.

-----

### `wrappers.py`

This script defines **custom environment wrappers** applied to the Air Raid environment.

**Key responsibilities:**

- Frame preprocessing (resizing, grayscaling, cropping...).


These wrappers ensure consistent preprocessing across PPO, A2C, and DQN experiments.
Not all wrappers can be implemented at the same time due to incompatibilities, for detailed information about this refer to the Notes section of this file.

-----

## `requirements.txt`

Lists all Python dependencies required to run Part 3 of the project.

-----

## How to Run

1. **Install dependencies**:

```bash
pip install -r requirements.txt
```

2. **Run the training / comparison script**:

```bash
python scripts/3models.py
```

This will train and evaluate PPO, A2C, and DQN on the Air Raid environment, saving logs, models, and videos in their respective folders.

3. **Inspect results**:

- Best models are stored under `*/logs_*_air_raid/best_model/`.
- Evaluation metrics are saved in `evaluations.npz`.
- Gameplay videos can be found in the `videos/` subfolders.
- You can also check performance in your wandb account.

-----

## Notes

- All algorithms are evaluated under comparable conditions to ensure a fair comparison.
- Quantitative results (from `evaluations.npz`) should be complemented with qualitative inspection of gameplay videos.
- This structure allows easy extension to additional environments or algorithms.
- Wrapper Interactions and Design Constraints: 
    CropTopBar removes the score area from the Atari frame and produces an 84×84 grayscale image that preserves all gameplay-relevant visual information.

    SemanticMaskObs, on the other hand, replaces the raw image with a three-channel binary mask representing enemies, bombs, and buildings. This removes pixel intensity information. As a result, it is incompatible with wrappers that rely on grayscale pixel values, like AirRaidBuildingPenaltyWrapper. These two wrappers cannot be combined.

    Reward shaping and normalization.
    AirRaidBuildingPenaltyWrapper applies a small negative reward when building damage is detected. When this wrapper is active, reward normalization (VecNormalize) is disabled. Normalizing rewards would dynamically rescale the penalty.
    Wrapper ordering.
    Wrapper order matters. Observation preprocessing must be applied before any reward-shaping wrapper that depends on visual content. Action wrappers must be applied before environment vectorization.

- Rely on the report for more detailed information about our custom wrappers. Just take this considerations into account if you want to train the models.

-----

