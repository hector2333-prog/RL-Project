import os
# PERFORMANCE OPTIMIZATIONS
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
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, CallbackList
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from wandb.integration.sb3 import WandbCallback
from pettingzoo.atari import pong_v3


RUN_ID = ""
WANDB_PROJECT = ""
MODELS_DIR = ""
LOGS_DIR = ""

#HYPERPARAMETERS
NUM_ENVS = 16              
STEPS_PER_GEN = 5_000_000   
NUM_GENERATIONS = 12       
BATCH_SIZE = 2048           
N_STEPS = 1024              
LEARNING_RATE = 4.0e-5     
ENT_COEF = 0.02             
VF_COEF = 1.0               
CLIP_RANGE = 0.2            
ACTIVATION_FN = th.nn.ReLU 

VIDEO_FREQ = 500_000  


os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

#HELPERS and WRAPPERS

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

#OPPONENT WRAPPER ---
class SingleSideWrapper(gym.Env):
    def __init__(self, opponent_path, learner_is_left: bool, activation_fn):
        super().__init__()
        self.pz_env = make_tournament_env()
        if learner_is_left:
            self.learner = 'second_0'    # Left Agent
            self.opponent = 'first_0'    # Right Agent
        else:
            self.learner = 'first_0'     # Right Agent
            self.opponent = 'second_0'   # Left Agent
            
        # Load opponent
        try:
            self.opponent_model = PPO.load(opponent_path, device="cpu", custom_objects={"activation_fn": activation_fn})
        except:
            self.opponent_model = PPO.load(opponent_path, device="cpu", custom_objects={"activation_fn": th.nn.Tanh})

        self.pz_env.reset()
        self.observation_space = self.pz_env.observation_space(self.learner)
        self.action_space = self.pz_env.action_space(self.learner)
        self.render_mode = "rgb_array"
        self.max_steps = 4000
        self.current_steps = 0
        self.opponent_noop_counter = 0

    def reset(self, seed=None, options=None):
        self.current_steps = 0
        self.opponent_noop_counter = 0
        self.pz_env.reset(seed=seed)
        for agent in self.pz_env.agent_iter():
            obs, reward, term, trunc, info = self.pz_env.last()
            if term or trunc: return obs, {}
            if agent == self.learner: return obs, {}
            else:
                # Opponent Turn
                # We use deterministic=False to allow it to wake up easier
                act, _ = self.opponent_model.predict(obs, deterministic=False)
                
                #ANTI-FREEZE LOGIC (Simulates FireResetEnv)
                if act == 0: # If standing still
                    self.opponent_noop_counter += 1
                else:
                    self.opponent_noop_counter = 0
                
                # If stuck for 12 frames, force a random move/fire
                if self.opponent_noop_counter > 12:
                    act = np.random.choice([1, 2, 3, 2, 3]) 
                    self.opponent_noop_counter = 0

                self.pz_env.step(act)
        return obs, {}

    def step(self, action):
        self.current_steps += 1
        self.pz_env.step(action)
        accumulated_reward = 0
        for agent in self.pz_env.agent_iter():
            obs, reward, term, trunc, info = self.pz_env.last()
            if agent == self.learner: accumulated_reward += reward
            if term or trunc: return obs, accumulated_reward, term, trunc, info
            if self.current_steps >= self.max_steps: return obs, accumulated_reward, False, True, {}
            if agent == self.learner: return obs, accumulated_reward, term, trunc, info
            else:
                # Opponent Turn
                act, _ = self.opponent_model.predict(obs, deterministic=False)
                
                #ANTI-FREEZE LOGIC (Simulates FireResetEnv)
                if act == 0:
                    self.opponent_noop_counter += 1
                else:
                    self.opponent_noop_counter = 0
                
                if self.opponent_noop_counter > 12:
                    act = np.random.choice([1, 2, 3, 2, 3])
                    self.opponent_noop_counter = 0

                self.pz_env.step(act)
        return obs, 0, True, False, {}
    
    def close(self): self.pz_env.close()
#VIDEO CALLBACK
class LeagueVideoCallback(BaseCallback):
    def __init__(self, opponent_path, learner_is_left, activation_fn, verbose=0):
        super().__init__(verbose)
        self.learner_is_left = learner_is_left
        self.opponent_path = opponent_path
        self.activation_fn = activation_fn
        self.video_freq_calls = VIDEO_FREQ // NUM_ENVS
        self.eval_env = SingleSideWrapper(self.opponent_path, self.learner_is_left, self.activation_fn)

    def _on_step(self) -> bool:
        if self.n_calls % self.video_freq_calls == 0:
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
                wandb.log({
                    f"Gen{self.learner_is_left}_Video": wandb.Video(screens, fps=30, format="mp4", caption=f"Score: {self.model.num_timesteps}"),
                    "total_timesteps": self.model.num_timesteps
                })
        return True
    
    def close_eval_env(self):
        self.eval_env.close()

#TRAINING LOOP

def train_generation(gen_idx, learner_side_str, opponent_model_path, previous_learner_path):
    is_left = (learner_side_str == "Left")
    run_name = f"{RUN_ID}_Gen{gen_idx}_{learner_side_str}"
    print(f"\n>>> STARTING {run_name} vs {opponent_model_path}")

    run = wandb.init(project=WANDB_PROJECT, name=run_name, reinit=True, sync_tensorboard=True, config={"gen": gen_idx, "side": learner_side_str, "run_id": RUN_ID})
    
    def make_env_fn(): return Monitor(SingleSideWrapper(opponent_model_path, is_left, ACTIVATION_FN))
    env = SubprocVecEnv([make_env_fn for _ in range(NUM_ENVS)], start_method="spawn")

    print(f"Loading weights from: {previous_learner_path}")
    

    if gen_idx == 1:
        load_activation = th.nn.Tanh
        load_kwargs = {"normalize_images": False} 
    else:
        load_activation = ACTIVATION_FN
        load_kwargs = {"normalize_images": False, "activation_fn": load_activation}

    model = PPO.load(
        previous_learner_path, 
        env=env,
        custom_objects={
            "n_steps": N_STEPS,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "ent_coef": ENT_COEF,
            "vf_coef": VF_COEF,
            "clip_range": CLIP_RANGE
        },
        policy_kwargs=load_kwargs
    )
    
    checkpoint_callback = CheckpointCallback(save_freq=1_000_000 // NUM_ENVS, save_path=os.path.join(MODELS_DIR, f"checkpoints_gen{gen_idx}"), name_prefix=f"gen_{gen_idx}_{learner_side_str}")
    vid_callback = LeagueVideoCallback(opponent_model_path, is_left, ACTIVATION_FN)
    wandb_callback = WandbCallback(verbose=2)

    try:
        model.learn(total_timesteps=STEPS_PER_GEN, callback=CallbackList([checkpoint_callback, vid_callback, wandb_callback]))
    except KeyboardInterrupt:
        print("Interrupted. Saving current state.")
        vid_callback.close_eval_env()

    final_name = f"gen_{gen_idx}_{learner_side_str}.zip"
    final_path = os.path.join(MODELS_DIR, final_name)
    model.save(final_path)
    run.finish()
    env.close()
    vid_callback.close_eval_env()
    return final_path


if __name__ == "__main__":
    # --- EXACT FILENAME ENFORCEMENT ---
    # The file MUST be named 'gen_0_teacher.zip' and act as the Right Agent
    TARGET_FILENAME = "gen_0_teacher.zip"
    
    target_path = os.path.join(MODELS_DIR, TARGET_FILENAME)
    
    if os.path.exists(target_path):
        print(f">>> Found Target Model in {MODELS_DIR}: {target_path}")
        initial_source = target_path
    elif os.path.exists(TARGET_FILENAME):
        print(f">>> Found Target Model in Root. Copying to {MODELS_DIR}...")
        shutil.copy(TARGET_FILENAME, target_path)
        initial_source = target_path
    else:
        # 3. Fallback search (just in case)
        print(f">>> WARNING: {TARGET_FILENAME} not found.")
       

    last_right_model = initial_source
    last_left_model = initial_source # Gen 1 Left starts as a clone of Gen 0 Right
    
    for i in range(1, NUM_GENERATIONS + 1):
        if i % 2 != 0:
            last_left_model = train_generation(i, "Left", last_right_model, last_left_model)
        else:
            last_right_model = train_generation(i, "Right", last_left_model, last_right_model)

    print("\n>>> FAST TRACK CO-EVOLUTION COMPLETE! <<<")