#agains machine 1 but parameteers from the sweeep

import gymnasium as gym
import numpy as np
import os
import torch
import wandb
import supersuit as ss
import ale_py
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback, CallbackList
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from wandb.integration.sb3 import WandbCallback

# --- CONFIGURATION ---
TOTAL_TIMESTEPS = 15_000_000 
NUM_ENVS = 16


# HYPERPARAMETERS MODIFIED BASED ON SWEEP RESULTS
BATCH_SIZE = 2048                
LEARNING_RATE = 2.5e-4           

# PPO-Specific Parameters
N_STEPS = 1024                   
ENT_COEF = 0.01                  
VF_COEF = 1.0                    
GAMMA = 0.995                    
GAE_LAMBDA = 0.98                
N_EPOCHS = 8                    
CLIP_RANGE = 0.2 

MODELS_DIR = ""
RUN_NAME = ""
WANDB_PROJECT = ""
VIDEO_FREQ = 200_000

os.makedirs(MODELS_DIR, exist_ok=True)

# HELPERS
class FireResetEnv(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

    def reset(self, **kwargs):
        self.env.reset(**kwargs)
        obs, _, terminated, truncated, _ = self.env.step(1)
        if terminated or truncated:
            self.env.reset(**kwargs)
            obs, _, terminated, truncated, _ = self.env.step(1)
        return obs, {}

class MaskWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        
    def observation(self, obs):
        obs = obs.copy()
        obs[:, 0:10, :] = 0 
        return obs

# ENVIRONMENT
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

#VIDEO CALLBACK
class BootcampVideoCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.eval_env = make_bootcamp_env()
        self.video_freq_calls = VIDEO_FREQ // NUM_ENVS

    def _on_step(self) -> bool:
        if self.n_calls % self.video_freq_calls == 0:
            screens = []
            obs, _ = self.eval_env.reset()
            done = False
            truncated = False
            total_rew = 0
            
            # Limit video length because Raw Pong is VERY long
            steps = 0
            MAX_VIDEO_STEPS = 4000 

            while not (done or truncated) and steps < MAX_VIDEO_STEPS:
                raw_frame = self.eval_env.unwrapped.render()
                screens.append(raw_frame)

                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, done, truncated, _ = self.eval_env.step(action)
                total_rew += reward
                steps += 1
            
            if len(screens) > 0:
                screens = np.array(screens)
                screens = np.transpose(screens, (0, 3, 1, 2))
                
                wandb.log({
                    "bootcamp_gameplay": wandb.Video(screens, fps=60, format="mp4", caption=f"Score: {total_rew}"),
                    "bootcamp_score": total_rew
                })
        return True

# EXECUTION
if __name__ == "__main__":
    run = wandb.init(
        project=WANDB_PROJECT,
        name=RUN_NAME,
        sync_tensorboard=True,
        monitor_gym=True
    )
    
    env = SubprocVecEnv([make_bootcamp_env for _ in range(NUM_ENVS)])

    model = PPO(
        "CnnPolicy",
        env,
        verbose=1,
        # --- OPTIMIZED HYPERPARAMETERS ---
        learning_rate=LEARNING_RATE,
        n_steps=N_STEPS,
        batch_size=BATCH_SIZE,
        ent_coef=ENT_COEF,
        vf_coef=VF_COEF,
        gamma=GAMMA,
        gae_lambda=GAE_LAMBDA,
        n_epochs=N_EPOCHS,
        clip_range=CLIP_RANGE,
        
        
        policy_kwargs={"normalize_images": False},
        tensorboard_log=f"runs/{run.id}"
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=1_000_000 // NUM_ENVS,
        save_path=MODELS_DIR,
        name_prefix=""
    )
    
    wandb_callback = WandbCallback(
        gradient_save_freq=100_000,
        model_save_path=f"models/{run.id}",
        verbose=2,
    )

    video_callback = BootcampVideoCallback()

    try:
        model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=CallbackList([checkpoint_callback, wandb_callback, video_callback]))
    except KeyboardInterrupt:
        print("Interrupted. Saving...")

    final_path = os.path.join(MODELS_DIR, "gen_0_teacher.zip")
    model.save(final_path)
    print(f"Saved Agent to: {final_path}")
    run.finish()