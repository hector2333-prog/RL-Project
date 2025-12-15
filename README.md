# Deep Reinforcement Learning Project

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python&logoColor=white)
![RL](https://img.shields.io/badge/Reinforcement_Learning-DQN_%26_A2C-orange?style=for-the-badge)
![WandB](https://img.shields.io/badge/Logging-WandB-yellow?style=for-the-badge&logo=weightsandbiases)
![Gymnasium](https://img.shields.io/badge/Gymnasium-Farama-black?style=for-the-badge&logo=openai&logoColor=white)
[![Atari](https://img.shields.io/badge/Atari-Pong-red?style=for-the-badge&logo=atari&logoColor=white)](https://gymnasium.farama.org/environments/atari/pong/)

<br>

> **Course:** Paradigms of Machine Learning (Bachelor's Degree in AI)  
> **Institution:** Universitat Autònoma de Barcelona (UAB)  
> **Academic Year:** 2025-2026
<br>

## 📖 Introduction

Welcome to our repository for the **Deep Reinforcement Learning (DRL)** project. This project explores the training of intelligent agents capable of playing Atari games from raw pixel inputs. We implement algorithms from scratch as well as leverage state-of-the-art libraries like **Stable Baselines 3**.

The project focuses on two main environments from the **Arcade Learning Environment (ALE)**:
*   **Pong** (Single & Multi-agent)
*   **AirRaid** (Complex Environment)

<br>
<br>

<table align="center" style="border: none;">
  <tr>
    <td align="center" style="border: none;">
      <img src="./Part 1/DQN/videos/gif.gif" width="200" alt="Pong DQN" />
      <br>
      <b>Pong (DQN)</b>
    </td>
    <td align="center" style="border: none; padding-left: 20px;"> <!-- padding adds the gap -->
      <img src="./Part 3/ppo/logs_ppo_air_raid/videos/gif2.gif" width="200" alt="AirRaid PPO" />
      <br>
      <b>AirRaid (PPO)</b>
    </td>
  </tr>
</table>

-----

## 🎯 Project Components

This repository is organized into three distinct parts, each tackling a specific RL challenge:

| Part | Environment | Goal | Key Technologies |
| :--- | :--- | :--- | :--- |
| **01** | `PongNoFrameskip-v4` | Train an agent to beat the built-in CPU using custom implementations. | PyTorch, Wrappers |
| **02** | `Pong (PettingZoo)` | **Multi-Agent Tournament**: Fine-tune an agent to compete against other students' agents. | PettingZoo, SuperSuit |
| **03** | `AirRaid-v5` | Solve a complex environment using advanced, optimized libraries. | Stable Baselines 3, custom Wrappers |

---

## 📂 Repository Structure

<br>

```
├── Part 1/ # Code for the custom Pong implementation (Part 1)
├── Part 2/ # Code for the MARL Pong environment using SB3 (Part 2)
├── Part 3/ # Code for the AirRaid environment using SB3 (Part 3)
├── README.md # Project documentation
└── .gitattributes # Git configuration
```

---

## 🛠️ Installation & Usage

To replicate our results, ensure you have the required dependencies installed.

### 1. Clone the repository

```
git clone https://github.com/hector2333-prog/RL-Project.git
cd RL-Project
```

### 2. Install Dependencies
We recommend using a virtual environment (Conda or venv).
The exact dependencies for each specific part of the project are located inside their respective folder (e.g., installation instructions for part 1 are explained in the README file inside the folder Part 1).

---

## 👥 Authors

*   **Víctor Brao** - *Developer & Researcher*
*   **David Piera** - *Developer & Researcher*
*   **Hector Salguero** - *Developer & Researcher*

---
