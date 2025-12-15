## Project Overview

This part of the project focuses on **Reinforcement Learning (RL)** approaches applied to the Atari Pong environment. Two different RL algorithms are implemented, trained, and evaluated:

1. **Deep Q-Network (DQN)**
2. **Advantage Actor-Critic (A2C)**

Both approaches follow a similar workflow:

1. **Training** the agent on the Pong environment.
2. **Hyperparameter tuning** using sweep scripts.
3. **Saving trained models** (best and final checkpoints).
4. **Evaluation and visualization** through recorded gameplay videos.

The code is organized into two main folders (`DQN/` and `A2C/`), each containing algorithm-specific scripts, trained models, and evaluation outputs.

-----

## Folder Structure Overview

```
├── A2C/
│   ├── models/
│   │   ├── best_actor.dat
│   │   ├── best_critic.dat
│   │   ├── final_actor.dat
│   │   ├── final_critic.dat
│   ├── videos/
│   │   ├── eval_ep_200.mp4
│   │   ├── ...
│   ├── A2C_sweep.py
│   ├── A2C_train.py
│   └── requirements.txt
│
├── DQN/
│   ├── models/
│   │   ├── best_model_pong.dat
│   │   ├── final_pong_model.dat
│   ├── videos/
│   │   ├── eval_ep_200.mp4
│   │   ├── ...
│   ├── DQN_sweep.py
│   ├── train_dqn.py
│   └── requirements.txt
```

-----

## A2C (Advantage Actor-Critic)

The `A2C/` folder contains the implementation of the **Advantage Actor-Critic** algorithm. A2C uses two neural networks:

- **Actor**: learns the policy.
- **Critic**: estimates the value function used to reduce variance during training.

### `A2C_train.py`

This script is responsible for **training the A2C agent** on the Pong environment.

**Key responsibilities:**

- Environment initialization and preprocessing.
- Definition of the **actor** and **critic** neural networks.
- Implementation of the A2C update rule using advantage estimates.
- Training loop over multiple episodes.
- Periodic evaluation of the agent.
- Saving model checkpoints.
- Saves and uploads results in our wandb account, for visualization of the results and performance.

**Outputs:**

- Saves trained models into the `models/` directory:
  - `best_actor.dat`
  - `best_critic.dat`
  - `final_actor.dat`
  - `final_critic.dat`

-----

### `A2C_sweep.py`

This script is used for **hyperparameter tuning** of the A2C algorithm.

**Key responsibilities:**

- Running multiple training sessions with different hyperparameter configurations.
- Sweeping over parameters such as:
  - Learning rate
  - Entropy coefficient
- Tracking performance metrics across runs.
- Selecting and saving the best-performing models.


-----

### `models/` (A2C)

This directory stores **serialized neural network weights** for the A2C agent.

- `best_actor.dat`  
  Weights of the actor network corresponding to the best evaluation performance.

- `best_critic.dat`  
  Weights of the critic network corresponding to the best evaluation performance.

- `final_actor.dat`  
  Actor network weights after the final training episode.

- `final_critic.dat`  
  Critic network weights after the final training episode.

-----

### `videos/` (A2C)

This folder contains **recorded evaluation episodes** of the trained A2C agent.

- `eval_ep_200.mp4`  
  Recorded performance of the agent at evaluation episode 200.

- Additional videos correspond to other evaluation checkpoints.

-----

### `requirements.txt` (A2C)

Lists all Python dependencies required to run the A2C code.

-----

## DQN (Deep Q-Network)

The `DQN/` folder contains the implementation of a **Deep Q-Network** agent for Pong.

### `train_dqn.py`

This script is responsible for **training the DQN agent**.

**Key responsibilities:**

- Environment setup and frame preprocessing.
- Definition of the Q-network architecture.
- Implementation of experience replay.
- Epsilon-greedy action selection.
- Target network updates.
- Training loop and evaluation.
- Saving trained models (Includes saving and visualization at wandb)

**Outputs:**

- Model checkpoints saved to `models/`:
  - `best_model_pong.dat`
  - `final_pong_model.dat`

-----

### `DQN_sweep.py`

This script performs **hyperparameter sweeps** for the DQN algorithm.

**Key responsibilities:**

- Running multiple DQN training sessions with different configurations.
- Sweeping parameters such as:
  - Learning rate
  - Replay buffer size
  - Epsilon decay schedule
- Logging and comparing performance across runs.
- Selecting the best-performing model.

-----

### `models/` (DQN)

Stores trained DQN model weights.

- `best_model_pong.dat`  
  Q-network weights corresponding to the best evaluation performance.

- `final_pong_model.dat`  
  Q-network weights after the final training episode.

-----

### `videos/` (DQN)

Contains gameplay recordings of the DQN agent during evaluation.

- `eval_ep_200.mp4`  
  Recorded performance of the agent at evaluation episode 200.

- Additional videos correspond to other evaluation checkpoints.

-----

### `requirements.txt` (DQN)

Lists Python dependencies required for running the DQN implementation.

-----

## How to Run

1. **Install dependencies** (inside either `A2C/` or `DQN/`):

```bash
pip install -r requirements.txt
```

2. **Train the agent**:

- A2C:
```bash
python A2C_train.py
```

- DQN:
```bash
python train_dqn.py
```

3. **Run hyperparameter sweeps**:

- A2C:
```bash
python A2C_sweep.py
```

- DQN:
```bash
python DQN_sweep.py
```

4. **Evaluate results**:

- Check saved models in `models/`.
- Watch evaluation videos in `videos/` to visually assess agent performance.
- For visualization of the model performance go to the runs in your wandb account.

-----

