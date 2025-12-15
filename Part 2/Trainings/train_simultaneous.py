import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["TORCH_NUM_THREADS"] = "1"

import numpy as np
import gymnasium as gym
import supersuit as ss
import wandb
import shutil
import torch as th
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
# CHANGED: Import DummyVecEnv instead of Subproc
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv 
from stable_baselines3.common.monitor import Monitor
from wandb.integration.sb3 import WandbCallback
from pettingzoo.atari import pong_v3


#CONFIGURATION
WANDB_PROJECT = ""
MODELS_DIR = ""
os.makedirs(MODELS_DIR, exist_ok=True)

LEFT_START_PATH = ""   
RIGHT_START_PATH = "" 

STEPS_PER_ROUND = 50_000    
TOTAL_ROUNDS = 200          

# use ggressive configuration
NUM_ENVS = 4
BATCH_SIZE = 2048
N_STEPS = 2048              
LEARNING_RATE = 5.0e-4      
ENT_COEF = 0.02             
VF_COEF = 1.0               
CLIP_RANGE = 0.2
ACTIVATION_FN = th.nn.Tanh  


#WRAPPERS
def mask_obs_fn(obs, obs_space, agent):
    obs = obs.copy()
    obs[:, 0:10, :] = 0 
    return obs

def make_tournament_env():
    env = pong_v3.env(num_players=2, render_mode="rgb_array")
    env = ss.sticky_actions_v0(env, repeat_action_probability=0.25)
    env = ss.color_reduction_v0(env, mode="B")
    env = ss.resize_v1(env, x_size=84, y_size=84)
    env = ss.frame_stack_v1(env, 4, stack_dim=0)
    env = ss.dtype_v0(env, dtype=np.float32)
    env = ss.normalize_obs_v0(env, env_min=0, env_max=1)
    env = ss.reshape_v0(env, (4, 84, 84))
    env = ss.observation_lambda_v0(env, mask_obs_fn, lambda obs, agent: obs)
    return env

#DYNAMIC OPPONENT WRAPPER
class DynamicOpponentWrapper(gym.Env):
    def __init__(self, opponent_path, learner_is_left: bool):
        super().__init__()
        self.pz_env = make_tournament_env()
        self.learner_is_left = learner_is_left
        
        if learner_is_left:
            self.learner = 'second_0'
            self.opponent = 'first_0'
        else:
            self.learner = 'first_0'
            self.opponent = 'second_0'
            
        try:
            self.opponent_model = PPO.load(opponent_path, device="cpu", custom_objects={"activation_fn": ACTIVATION_FN})
        except:
            self.opponent_model = PPO.load(opponent_path, device="cpu")
        
        self.pz_env.reset()
        self.observation_space = self.pz_env.observation_space(self.learner)
        self.action_space = self.pz_env.action_space(self.learner)
        self.render_mode = "rgb_array"
        self.max_steps = 4000
        self.current_steps = 0
        self.opponent_noop = 0

    def reset(self, seed=None, options=None):
        self.current_steps = 0
        self.opponent_noop = 0
        self.pz_env.reset(seed=seed)
        for agent in self.pz_env.agent_iter():
            obs, _, term, trunc, _ = self.pz_env.last()
            if term or trunc: return obs, {}
            if agent == self.learner: return obs, {}
            else:
                act, _ = self.opponent_model.predict(obs, deterministic=False)
                if act == 0: self.opponent_noop += 1
                else: self.opponent_noop = 0
                if self.opponent_noop > 12: 
                    act = np.random.choice([1, 2, 3, 2, 3])
                    self.opponent_noop = 0
                self.pz_env.step(act)
        return obs, {}

    def step(self, action):
        self.current_steps += 1
        self.pz_env.step(action)
        accumulated_reward = 0
        for agent in self.pz_env.agent_iter():
            obs, reward, term, trunc, _ = self.pz_env.last()
            if agent == self.learner: accumulated_reward += reward
            if term or trunc: return obs, accumulated_reward, term, trunc, {}
            if self.current_steps >= self.max_steps: return obs, accumulated_reward, False, True, {}
            
            if agent == self.learner: return obs, accumulated_reward, term, trunc, {}
            else:
                act, _ = self.opponent_model.predict(obs, deterministic=False)
                if act == 0: self.opponent_noop += 1
                else: self.opponent_noop = 0
                if self.opponent_noop > 12: 
                    act = np.random.choice([1, 2, 3, 2, 3])
                    self.opponent_noop = 0
                self.pz_env.step(act)
        return obs, 0, True, False, {}
    
    def close(self): self.pz_env.close()

#INIT WRAPPER
class RandomGen0Wrapper(gym.Env):
    def __init__(self):
        super().__init__()
        self.pz_env = make_tournament_env()
        self.learner = 'first_0'
        self.opponent = 'second_0'
        self.pz_env.reset()
        self.observation_space = self.pz_env.observation_space(self.learner)
        self.action_space = self.pz_env.action_space(self.learner)
        self.render_mode = "rgb_array"

    def reset(self, seed=None, options=None):
        self.pz_env.reset(seed=seed)
        for agent in self.pz_env.agent_iter():
            obs, _, term, trunc, _ = self.pz_env.last()
            if term or trunc: return obs, {}
            if agent == self.learner: return obs, {}
            else:
                self.pz_env.step(self.pz_env.action_space(agent).sample())
        return obs, {}

    def step(self, action):
        self.pz_env.step(action)
        for agent in self.pz_env.agent_iter():
            obs, _, term, trunc, _ = self.pz_env.last()
            if term or trunc: return obs, 0, term, trunc, {}
            if agent == self.learner: return obs, 0, term, trunc, {}
            else:
                self.pz_env.step(self.pz_env.action_space(agent).sample())
        return obs, 0, True, False, {}
    
    def close(self): self.pz_env.close()

#VIDEO CALLBACK
class SimultaneousVideoCallback(BaseCallback):
    def __init__(self, opponent_path, learner_is_left, verbose=0):
        super().__init__(verbose)
        self.opponent_path = opponent_path
        self.learner_is_left = learner_is_left
        # Record roughly once per round
        self.record_freq = (STEPS_PER_ROUND // NUM_ENVS) - 100 
        # For evaluation, we can keep using DynamicOpponentWrapper directly (single process)
        self.eval_env = DynamicOpponentWrapper(self.opponent_path, self.learner_is_left)

    def _on_step(self) -> bool:
        if self.n_calls % self.record_freq == 0:
            screens = []
            obs, _ = self.eval_env.reset()
            done = False
            truncated = False
            step_count = 0
            
            while not (done or truncated) and step_count < 4000:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, done, truncated, info = self.eval_env.step(action)
                screens.append(self.eval_env.pz_env.render())
                step_count += 1

            if len(screens) > 0:
                screens = np.array(screens)
                screens = np.transpose(screens, (0, 3, 1, 2))
                side_str = "LEFT" if self.learner_is_left else "RIGHT"
                wandb.log({
                    f"Video_{side_str}_Learning": wandb.Video(screens, fps=30, format="mp4", caption=f"Learner: {side_str}"),
                    "total_timesteps": self.model.num_timesteps
                })
        return True
    
    def close(self):
        self.eval_env.close()

# MAIN LOOP
if __name__ == "__main__":
    wandb.init(project=WANDB_PROJECT, name="Safe_Mode_Training", sync_tensorboard=True)

    #AUTO-INIT
    current_left = os.path.join(MODELS_DIR, "latest_left.zip")
    current_right = os.path.join(MODELS_DIR, "latest_right.zip")
    
    if not os.path.exists(current_left):
        if os.path.exists(LEFT_START_PATH):
            shutil.copy(LEFT_START_PATH, current_left)
        else:
            print("Creating Random Left...")
            def make_dummy(): return Monitor(RandomGen0Wrapper())
            # Use DummyVecEnv for init too (Safe Mode)
            dummy = DummyVecEnv([make_dummy for _ in range(4)])
            model = PPO("CnnPolicy", dummy, policy_kwargs={"normalize_images": False, "activation_fn": ACTIVATION_FN})
            model.save(current_left)
            dummy.close()

    if not os.path.exists(current_right):
        if os.path.exists(RIGHT_START_PATH):
            shutil.copy(RIGHT_START_PATH, current_right)
        else:
            print("Creating Random Right...")
            def make_dummy(): return Monitor(RandomGen0Wrapper())
            dummy = DummyVecEnv([make_dummy for _ in range(4)])
            model = PPO("CnnPolicy", dummy, policy_kwargs={"normalize_images": False, "activation_fn": ACTIVATION_FN})
            model.save(current_right)
            dummy.close()

    print(f"STARTING SAFE MODE TRAINING (DummyVecEnv)")
    
    load_params = {
        "learning_rate": LEARNING_RATE,
        "n_steps": N_STEPS,
        "batch_size": BATCH_SIZE,
        "ent_coef": ENT_COEF,
        "vf_coef": VF_COEF,
        "clip_range": CLIP_RANGE
    }
    policy_args = {"normalize_images": False, "activation_fn": ACTIVATION_FN}

    for round_idx in range(1, TOTAL_ROUNDS + 1):
        print(f"\n=== ROUND {round_idx}/{TOTAL_ROUNDS} ===")
        
        #TRAIN LEFT
        def make_env_left(): return Monitor(DynamicOpponentWrapper(current_right, learner_is_left=True))
        env_left = DummyVecEnv([make_env_left for _ in range(NUM_ENVS)])
        
        model_left = PPO.load(current_left, env=env_left, custom_objects=load_params, policy_kwargs=policy_args)
        
        vid_cb_left = SimultaneousVideoCallback(current_right, learner_is_left=True)
        model_left.learn(total_timesteps=STEPS_PER_ROUND, callback=CallbackList([vid_cb_left, WandbCallback(verbose=2)]))
        
        model_left.save(current_left)
        vid_cb_left.close()
        env_left.close()
        
        #TRAIN RIGHT
        def make_env_right(): return Monitor(DynamicOpponentWrapper(current_left, learner_is_left=False))
        env_right = DummyVecEnv([make_env_right for _ in range(NUM_ENVS)])
        
        model_right = PPO.load(current_right, env=env_right, custom_objects=load_params, policy_kwargs=policy_args)
        
        vid_cb_right = SimultaneousVideoCallback(current_left, learner_is_left=False)
        model_right.learn(total_timesteps=STEPS_PER_ROUND, callback=CallbackList([vid_cb_right, WandbCallback(verbose=2)]))
        
        model_right.save(current_right)
        vid_cb_right.close()
        env_right.close()
        
        if round_idx % 10 == 0:
            shutil.copy(current_left, os.path.join(MODELS_DIR, f"round_{round_idx}_Left.zip"))
            shutil.copy(current_right, os.path.join(MODELS_DIR, f"round_{round_idx}_Right.zip"))
            print(f"Saved Checkpoint Round {round_idx}")
            
    wandb.finish()