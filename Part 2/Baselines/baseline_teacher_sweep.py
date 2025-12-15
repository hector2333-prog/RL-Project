import gymnasium as gym
import numpy as np
import os
import torch
import torch.nn as nn
import wandb
import supersuit as ss
import ale_py
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from wandb.integration.sb3 import WandbCallback

# --- 1. FIXED CONFIGURATION ---
NUM_ENVS = 16
TOTAL_TIMESTEPS_SWEEP = 5_000_000 
MODELS_DIR = "sweep_models"
os.makedirs(MODELS_DIR, exist_ok=True)


class FireResetEnv(gym.Wrapper):
    def __init__(self, env): super().__init__(env)
    def reset(self, **kwargs):
        self.env.reset(**kwargs)
        obs, _, term, trunc, _ = self.env.step(1)
        if term or trunc: 
            self.env.reset(**kwargs)
            obs, _, term, trunc, _ = self.env.step(1)
        return obs, {}

class MaskWrapper(gym.ObservationWrapper):
    def __init__(self, env): super().__init__(env)
    def observation(self, obs):
        obs = obs.copy()
        obs[:, 0:10, :] = 0 
        return obs

def make_bootcamp_env():
    env = gym.make("ALE/Pong-v5", render_mode="rgb_array", frameskip=1, repeat_action_probability=0)
    env = FireResetEnv(env)
    env = ss.sticky_actions_v0(env, repeat_action_probability=0.25)
    env = ss.color_reduction_v0(env, mode="B")
    env = ss.resize_v1(env, x_size=84, y_size=84)
    env = ss.frame_stack_v1(env, 4, stack_dim=0)
    env = ss.dtype_v0(env, dtype=np.float32)
    env = ss.normalize_obs_v0(env, env_min=0, env_max=1)
    env = ss.reshape_v0(env, (4, 84, 84))
    env = MaskWrapper(env)
    return Monitor(env)

#SWEEP TRAINING FUNCTION
def train():
    run = wandb.init(sync_tensorboard=True, monitor_gym=True)
    config = wandb.config
    
    activation_fn = nn.ReLU if config.activation_fn == "ReLU" else nn.Tanh
    
    # Handle "None" for target_kl (wandb passes strings or nulls)
    target_kl = None if config.target_kl == "None" else config.target_kl
    
    print(f">>> SWEEP: LR={config.learning_rate} | Steps={config.n_steps} | Act={config.activation_fn}")
    
    env = SubprocVecEnv([make_bootcamp_env for _ in range(NUM_ENVS)])

    model = PPO(
        "CnnPolicy",
        env,
        verbose=1,

        learning_rate=config.learning_rate,
        n_steps=config.n_steps,
        batch_size=config.batch_size,
        ent_coef=config.ent_coef,
        gamma=config.gamma,
        clip_range=config.clip_range,
        n_epochs=config.n_epochs,
        gae_lambda=config.gae_lambda,
        target_kl=target_kl,
        max_grad_norm=config.max_grad_norm,
        vf_coef=config.vf_coef,
        policy_kwargs={
            "normalize_images": False, 
            "activation_fn": activation_fn
        },
        
        tensorboard_log=f"runs/{run.id}"
    )

    wandb_callback = WandbCallback(
        gradient_save_freq=100_000,
        model_save_path=f"models/{run.id}",
        verbose=2,
    )
    
    try:
        model.learn(total_timesteps=TOTAL_TIMESTEPS_SWEEP, callback=wandb_callback)
    except KeyboardInterrupt:
        print("Interrupted.")
        
    run.finish()

# --- 4. SWEEP CONFIGURATION ---
if __name__ == "__main__":
    sweep_configuration = {
        "method": "bayes", 
        "metric": {"name": "rollout/ep_rew_mean", "goal": "maximize"},
        "parameters": {
            # --- PREVIOUS PARAMS ---
            "learning_rate": {"values": [1e-4, 2.5e-4, 4e-4]},
            "n_steps": {"values": [512, 1024]},
            "batch_size": {"values": [2048, 4096]},
            "ent_coef": {"values": [0.005, 0.01, 0.02]},
            "gamma": {"values": [0.99, 0.995]},
            "n_epochs": {"values": [4, 8]},
            "clip_range": {"values": [0.1, 0.2]},
            "gae_lambda": {"values": [0.95, 0.98]},
            
            # --- NEW PARAMS ---
            "max_grad_norm": {"values": [0.5, 1.0, 5.0]},
            "vf_coef": {"values": [0.5, 1.0]},
            "target_kl": {"values": [0.01, 0.05, "None"]}, # "None" means OFF
            "activation_fn": {"values": ["Tanh", "ReLU"]}
        }
    }

    sweep_id = wandb.sweep(sweep_configuration, project="Pong-Bootcamp-Raw-Sweep-V2")
    wandb.agent(sweep_id, function=train, count=20) 