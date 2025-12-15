
import numpy as np
import gymnasium as gym
import supersuit as ss
import wandb
import os
import torch

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, CallbackList
from stable_baselines3.common.vec_env import SubprocVecEnv
from wandb.integration.sb3 import WandbCallback
from pettingzoo.utils import conversions 

# Import your provided wrappers

from wrappers import make_env



NUM_ENVS = 16           
TOTAL_TIMESTEPS = 10_000_000 
LEARNING_RATE = 2.5e-4
BATCH_SIZE = 4096       
N_STEPS = 128           
VIDEO_FREQ = 200_000    

# WandB Config
WANDB_PROJECT = ""
RUN_NAME = ""

#OBSERVATION MODIFIERS

def mirror_obs(obs, obs_space, agent):
    """
    Flips the observation for the Left player ('second_0').
    We flip axis 2 (Width).
    """
    if agent == "second_0":
        return np.flip(obs, axis=2).copy()
    return obs

def mask_obs(obs, obs_space, agent):
    """
    Blacks out the top 12 pixels for ALL agents.
    This hides the scoreboard and prevents 'backwards numbers' confusion.
    """
    obs = obs.copy()
    obs[:, 0:10, :] = 0 # Mask Top 10 rows
    return obs

# VIDEO CALLBACK
class WandbVideoCallback(BaseCallback):
    """
    Pauses training to play ONE match in a separate environment, 
    records it, and uploads to WandB.
    """
    def __init__(self, verbose=0):
        super().__init__(verbose)
      
        self.eval_env = make_env(render_mode="rgb_array")
        
        # Apply the SAME wrappers to the evaluation env!
        # 1. Mirror
        self.eval_env = ss.observation_lambda_v0(self.eval_env, mirror_obs, lambda s, a: s)
        # 2. Mask
        self.eval_env = ss.observation_lambda_v0(self.eval_env, mask_obs, lambda s, a: s)

    def _on_step(self) -> bool:
        if self.num_timesteps % VIDEO_FREQ == 0:
            screens = []
            self.eval_env.reset()
       
            for agent in self.eval_env.agent_iter():
                obs, reward, term, trunc, info = self.eval_env.last()
                
                if term or trunc:
                    action = None 
                else:
                    action, _ = self.model.predict(obs, deterministic=True)

                self.eval_env.step(action)
                
                
                if agent == 'first_0': 
                    screens.append(self.eval_env.render())

                if all(self.eval_env.terminations.values()) or all(self.eval_env.truncations.values()):
                    break
            
            if len(screens) > 0:
                screens = np.array(screens)
        
                screens = np.transpose(screens, (0, 3, 1, 2))
                
                wandb.log({
                    "gameplay_video": wandb.Video(screens, fps=30, format="mp4", caption=f"Step: {self.num_timesteps}")
                })
                print(f">>> Video recorded and uploaded at step {self.num_timesteps}")

        return True

# MAIN TRAINING LOOP
if __name__ == "__main__":

    # Initialize WandB
    run = wandb.init(
        project=WANDB_PROJECT, 
        name=RUN_NAME,
        sync_tensorboard=True,
        monitor_gym=True
    )

    print(f"--- Starting Training on {torch.cuda.get_device_name(0)} ---")
    print(f"--- Environments: {NUM_ENVS} | Batch Size: {BATCH_SIZE} ---")

    # Setup Base Environment
    env = make_env()
    
    # Apply Mirror Wrapper (Left player flips)
    env = ss.observation_lambda_v0(env, mirror_obs, lambda s, a: s)

    # Apply Mask Wrapper (EVERYONE gets the black bar)
    env = ss.observation_lambda_v0(env, mask_obs, lambda s, a: s)
    
    # Vectorization Setup
    env = conversions.aec_to_parallel(env) 
    env = ss.pettingzoo_env_to_vec_env_v1(env)
    

    env = ss.concat_vec_envs_v1(env, num_vec_envs=NUM_ENVS, num_cpus=16, base_class="stable_baselines3")

    # 5. Define the Model
    model = PPO(
        "CnnPolicy",
        env,
        verbose=1,
        learning_rate=LEARNING_RATE,
        n_steps=N_STEPS,       
        batch_size=BATCH_SIZE, 
        n_epochs=4,
        clip_range=0.1,
        ent_coef=0.01,         
        vf_coef=0.5,
        policy_kwargs={"normalize_images": False},
        tensorboard_log=f"runs/{run.id}"
    )

    # Callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=1_000_000 // NUM_ENVS, 
        save_path='./checkpoints_gen0/',
        name_prefix='gen0_masked'
    )
    
    wb_callback = WandbCallback(
        gradient_save_freq=100_000,
        model_save_path=f"models/{run.id}",
        verbose=2,
    )
    
    vid_callback = WandbVideoCallback()

    callbacks = CallbackList([checkpoint_callback, wb_callback, vid_callback])

    # 7. Train
    try:
        model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=callbacks)
    except KeyboardInterrupt:
        print("Training interrupted manually. Saving current model...")
    
    # 8. Save Final Model
    model.save("gen_0_naive")
    print("Training Complete. Model saved as 'gen_0_naive'.")
    run.finish()