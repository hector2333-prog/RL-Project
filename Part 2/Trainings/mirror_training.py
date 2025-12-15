import os
#PERFORMANCE OPTIMIZATIONS
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["TORCH_NUM_THREADS"] = "1"

import numpy as np
import gymnasium as gym
import supersuit as ss
import wandb
import shutil
from typing import Callable
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, EvalCallback
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from wandb.integration.sb3 import WandbCallback
from pettingzoo.utils import conversions

from wrappers import make_env

#HYPERPARAMETERS
NUM_ENVS = 16              
STEPS_PER_GEN = 10_000_000  
NUM_GENERATIONS = 12       
BATCH_SIZE = 1000          
VIDEO_FREQ = 1_000_000     
EVAL_FREQ = 500_000        
EVAL_EPISODES = 20          

WANDB_PROJECT = "Pong-Tournament-B"
MODELS_DIR = "genereations_B"
LOGS_DIR = "logs_pro"

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)


def linear_schedule(initial_value: float) -> Callable[[float], float]:
    def func(progress_remaining: float) -> float:
        return progress_remaining * initial_value
    return func

class EntropyScheduleCallback(BaseCallback):
    def __init__(self, initial_ent_coef=0.01, final_ent_coef=0.001, total_steps=STEPS_PER_GEN, verbose=0):
        super().__init__(verbose)
        self.initial_ent = initial_ent_coef
        self.final_ent = final_ent_coef
        self.total_steps = total_steps
        
    def _on_step(self) -> bool:
        progress = min(1.0, self.num_timesteps / (self.total_steps * 0.7)) 
        current_ent = self.initial_ent + progress * (self.final_ent - self.initial_ent)
        self.model.ent_coef = current_ent
        return True

#OBSERVATION HELPERS
def mirror_obs(obs, obs_space, agent):
    # Flips width for Left player
    if agent == "second_0":
        return np.flip(obs, axis=2).copy()
    return obs

def mask_obs(obs):
    # Applies the black bar to the top 12 pixels
    # Works on single observation (C, H, W)
    obs = obs.copy()
    obs[:, 0:10, :] = 0
    return obs

#DETERMINISTIC OPPONENT WRAPPER
class SingleOpponentWrapper(gym.Env):
    def __init__(self, opponent_path):
        super().__init__()
        #Load Env
        self.pz_env = make_env(render_mode="rgb_array")
        self.learner = 'first_0'
        
        #Load Opponent
        self.opponent = PPO.load(opponent_path, device="cpu")
        
        #Setup Spaces
        self.pz_env.reset()
        self.observation_space = self.pz_env.observation_space(self.learner)
        self.action_space = self.pz_env.action_space(self.learner)
        self.render_mode = "rgb_array"
        
        self.current_steps = 0
        self.max_steps = 4000 

    def _process_obs(self, obs, agent):
        """Helper to apply Mirroring AND Masking correctly."""
        #Mirror if it's the Left Agent
        if agent == 'second_0':
            obs = np.flip(obs, axis=2).copy()
        
        #ALWAYS Apply Mask
        obs = mask_obs(obs)
        return obs

    def reset(self, seed=None, options=None):
        self.current_steps = 0
        self.pz_env.reset(seed=seed)
        for agent in self.pz_env.agent_iter():
            obs, reward, term, trunc, info = self.pz_env.last()
            
            # Apply processing (Mirror + Mask)
            obs_processed = self._process_obs(obs, agent)
            
            if term or trunc: 
                return obs_processed, {}
            
            if agent == self.learner: 
                return obs_processed, {}
            
            # Opponent acts
            act, _ = self.opponent.predict(obs_processed, deterministic=True)
            self.pz_env.step(act)
            
        return obs_processed, {}

    def step(self, action):
        self.current_steps += 1
        self.pz_env.step(action)
        accumulated_reward = 0
        
        force_quit = self.current_steps >= self.max_steps
        
        for agent in self.pz_env.agent_iter():
            obs, reward, term, trunc, info = self.pz_env.last()
            
            # Apply processing (Mirror + Mask)
            obs_processed = self._process_obs(obs, agent)

            if agent == self.learner: accumulated_reward += reward
            
            if force_quit:
                return obs_processed, accumulated_reward, False, True, {} 

            if term or trunc:
                return obs_processed, accumulated_reward, term, trunc, info
            
            if agent == self.learner:
                return obs_processed, accumulated_reward, term, trunc, info
            else:
                act, _ = self.opponent.predict(obs_processed, deterministic=True)
                self.pz_env.step(act)
                
        return obs_processed, 0, True, False, {}
    
    def render(self):
        return self.pz_env.render()
    
    def close(self):
        self.pz_env.close()

#VIDEO CALLBACK
class LeagueVideoCallback(BaseCallback):
    def __init__(self, opponent_path, verbose=0):
        super().__init__(verbose)
        self.eval_env = SingleOpponentWrapper(opponent_path)
        self.video_freq_calls = VIDEO_FREQ // NUM_ENVS

    def _on_step(self) -> bool:
        if self.n_calls % self.video_freq_calls == 0:
            screens = []
            obs, _ = self.eval_env.reset()
            done = False
            truncated = False
            step_count = 0
            while not (done or truncated):
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, done, truncated, info = self.eval_env.step(action)
                screens.append(self.eval_env.render())
                step_count += 1
                if step_count >= 5000: break

            if len(screens) > 0:
                screens = np.array(screens)
                screens = np.transpose(screens, (0, 3, 1, 2))
                wandb.log({"gameplay_vs_opponent": wandb.Video(screens, fps=30, format="mp4")})
        return True

#TRAINING FUNCTIONS
def train_gen_0():
    print("\n>>> STARTING GENERATION 0: Pure Self-Play (Scratch)")
    run = wandb.init(project=WANDB_PROJECT, name="Gen_0_SelfPlay", reinit=True, sync_tensorboard=True)
    
    #Create Env
    env = make_env()
    #Apply Mirror (Left flips)
    env = ss.observation_lambda_v0(env, mirror_obs, lambda s, a: s)
    #Apply Mask (EVERYONE gets black bar)
    env = ss.observation_lambda_v0(env, lambda obs, _, __: mask_obs(obs), lambda s, a: s)
    
    env = conversions.aec_to_parallel(env)
    env = ss.pettingzoo_env_to_vec_env_v1(env)
    env = ss.concat_vec_envs_v1(env, NUM_ENVS, num_cpus=12, base_class="stable_baselines3")
    
    model = PPO("CnnPolicy", env, verbose=1, 
                learning_rate=2.5e-4, 
                n_steps=128,          
                batch_size=BATCH_SIZE, 
                policy_kwargs={"normalize_images": False}, 
                tensorboard_log=f"runs/{run.id}")
    
    model.learn(total_timesteps=STEPS_PER_GEN, callback=WandbCallback(verbose=2))
    save_path = os.path.join(MODELS_DIR, "gen_0.zip")
    model.save(save_path)
    run.finish()
    return save_path

def train_gen_curriculum(gen_idx, learner_base_path, opponent_path, is_weak_round):
    if is_weak_round:
        run_name = f"Gen_{gen_idx}_WEAK"
        final_filename = f"gen_{gen_idx}_weak.zip"
        print(f"\n>>> ROUND {gen_idx}: ROBUSTNESS (vs Weak)")
    else:
        run_name = f"Gen_{gen_idx}_LADDER"
        final_filename = f"gen_{gen_idx}.zip"
        print(f"\n>>> ROUND {gen_idx}: LADDER (vs Previous Best)")

    run = wandb.init(project=WANDB_PROJECT, name=run_name, reinit=True, sync_tensorboard=True)
    
    def make_curriculum_env():
        return Monitor(SingleOpponentWrapper(opponent_path))
    
    env = SubprocVecEnv([make_curriculum_env for _ in range(NUM_ENVS)], start_method="spawn")
    eval_env = Monitor(SingleOpponentWrapper(opponent_path))

    model = PPO.load(learner_base_path, env=env, custom_objects={
        "learning_rate": 4.0e-5,
        "n_steps": 128,
        "batch_size": BATCH_SIZE
    })
    
    model.learning_rate = linear_schedule(4.0e-5) 
    ent_callback = EntropyScheduleCallback(initial_ent_coef=0.03, final_ent_coef=0.005)
    
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(MODELS_DIR, f"temp_best_{gen_idx}"),
        log_path=LOGS_DIR,
        eval_freq=EVAL_FREQ // NUM_ENVS,
        deterministic=True,
        render=False,
        n_eval_episodes=EVAL_EPISODES
    )
    
    vid_callback = LeagueVideoCallback(opponent_path)
    wb_callback = WandbCallback(verbose=2)
    
    model.learn(total_timesteps=STEPS_PER_GEN, callback=CallbackList([eval_callback, ent_callback, vid_callback, wb_callback]))
    
    # Save
    best_model_path = os.path.join(MODELS_DIR, f"temp_best_{gen_idx}", "best_model.zip")
    final_save_path = os.path.join(MODELS_DIR, final_filename)

    if os.path.exists(best_model_path):
        print(f">>> VICTORY: Found a High-Score model! Promoting to {final_filename}")
        shutil.copy(best_model_path, final_save_path)
        shutil.rmtree(os.path.join(MODELS_DIR, f"temp_best_{gen_idx}"))
    else:
        print(f">>> WARNING: No 'Best Model' record found. Saving Final State as fallback.")
        model.save(final_save_path)
        
    run.finish()
    return final_save_path

if __name__ == "__main__":
    gen0_path = os.path.join(MODELS_DIR, "gen_0.zip")
    dumb_path = os.path.join(MODELS_DIR, "gen_0_masked.zip")
    
    # Logic to select start
    if os.path.exists(gen0_path):
        last_strong_model = gen0_path
        gen0_abs_path = gen0_path
    elif os.path.exists(dumb_path):
        last_strong_model = dumb_path
        gen0_abs_path = dumb_path
    else:
        last_strong_model = train_gen_0()
        gen0_abs_path = last_strong_model
    
    for i in range(1, NUM_GENERATIONS + 1):
        if i % 3 == 0:
            last_strong_model = train_gen_curriculum(i, last_strong_model, gen0_abs_path, is_weak_round=True)
        else:
            last_strong_model = train_gen_curriculum(i, last_strong_model, last_strong_model, is_weak_round=False)

    print("\n>>> PRO CURRICULUM TRAINING COMPLETE! <<<")