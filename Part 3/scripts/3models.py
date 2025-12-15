import os
import argparse
import wandb
import imageio
import numpy as np
import gymnasium as gym
import ale_py
import datetime
import glob
print("hola")

import wandb

from stable_baselines3 import DQN, A2C, PPO
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv, VecVideoRecorder, SubprocVecEnv, VecNormalize, VecTransposeImage
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, BaseCallback
from stable_baselines3.ppo.policies import MlpPolicy
from wandb.integration.sb3 import WandbCallback
from stable_baselines3.common.monitor import Monitor
from gymnasium.wrappers import FrameStackObservation

#Custom wrappers
from wrappers import AirRaidActionWrapper, AirRaidBuildingPenaltyWrapper, CropTopBar, StickyActions, SemanticMaskObs

gym.register_envs(ale_py)

ENV_ID = "ALE/AirRaid-v5"
N_ENVS = 16
EVAL_FREQ = 100_000
N_EVAL_EPISODES = 10
VIDEO_EVERY = 3
FINAL_EVAL_EPISODES = 30

# --- Fixed Video Recording Callback ---
class VideoRecorderCallback(BaseCallback):
    def __init__(self, eval_env, render_freq: int, video_folder: str):
        super().__init__(verbose=1)
        self.eval_env = eval_env
        self.render_freq = render_freq
        self.video_folder = video_folder
        self.last_video_step = -1  # Force video at start
        
        os.makedirs(self.video_folder, exist_ok=True)

    def _on_step(self) -> bool:
        # We record if we passed the frequency threshold
        if self.num_timesteps - self.last_video_step >= self.render_freq or self.last_video_step == -1:
            print(f">>> Recording match at step {self.num_timesteps}...")
            self.last_video_step = self.num_timesteps
            
            screens = []
            
            obs = self.eval_env.reset()
            done = False
            
            # Play one full episode
            while not done:
                try:
                    screen = self.eval_env.render()
                    screens.append(screen)
                except Exception as e:
                    print(f"Render Error: {e}")
                    break
                
                action, _ = self.model.predict(obs, deterministic=False)
                
                obs, reward, done, info = self.eval_env.step(action)
                
            # Save and Upload
            if len(screens) > 0:
                video_filename = f"{self.video_folder}/step_{self.num_timesteps}.mp4"
                try:
                    imageio.mimsave(video_filename, screens, fps=30, macro_block_size=16)
                    print(f">>> Video saved locally: {video_filename}")
                    
                    # Upload to W&B
                    video_array = np.array(screens)
                    video_array = np.transpose(video_array, (0, 3, 1, 2))
                    
                    wandb.log({
                        "gameplay_video": wandb.Video(video_array, fps=30, format="mp4", caption=f"Step {self.num_timesteps}")
                    })
                    print(">>> Video uploaded to Weights & Biases!")
                    
                except Exception as e:
                    print(f"Video Saving Error: {e}")

        return True


def make_single_wrapped_atari_env(env_id: str, seed: int = 0, render_mode: str = None):
    """
    Create a single Atari env with SB3's atari_wrappers, then apply
    custom AirRaid wrappers.
    """
    vec = make_atari_env(env_id, n_envs=1, seed=seed)
    base_env = vec.envs[0]

    # Apply your custom wrappers on top
    env = CropTopBar(base_env)
    #env = SemanticMaskObs(env)        # TO USE THIS WRAPPER DO NOT USE AIRRAIDBUILDINGPENALTYWRAPPER
    #env = StickyActions(env)
    #env = AirRaidActionWrapper(env)
    env = AirRaidBuildingPenaltyWrapper(base_env)

    # If render_mode requested, try to enable rgb_array (used by video recorder).
    if render_mode == "rgb_array":
        try:
            env.render_mode = "rgb_array"
        except Exception:
            pass

    return env


def make_wrapped_vec_env(env_id: str, n_envs: int, seed: int = 0, subprocess: bool = False, render_mode: str = None, norm_obs: bool = False, norm_reward: bool = False, training: bool = True, n_stack: int = 4):
    """
    VecEnv over n_envs copies of the single wrapped Atari env.
    """
    env_fns = [lambda s=seed + i: make_single_wrapped_atari_env(env_id, s) for i in range(n_envs)]
    if subprocess:
        env = SubprocVecEnv(env_fns)
    else:
        env = DummyVecEnv(env_fns)

    # If using AirRaidBuildingPenaltyWrapper, norm_obs and norm_reward should be False
    env = VecNormalize(env, training=training, norm_obs=norm_obs, norm_reward=norm_reward)
    env = VecFrameStack(env, n_stack=n_stack)
    return env


def record_final_video(model, env_id: str, video_dir: str, video_name: str = "final_eval", video_length: int = 3000):
    """
    Record a video of the trained agent after training.
    """
    vec_env = make_wrapped_vec_env(env_id, n_envs=1, seed=1234, subprocess=False)

    vec_env = VecVideoRecorder(
        vec_env,
        video_dir,
        record_video_trigger=lambda step: step == 0,
        video_length=video_length,
        name_prefix=video_name,
    )

    obs = vec_env.reset()
    for _ in range(video_length):
        action, _ = model.predict(obs, deterministic=True)
        obs, rewards, dones, infos = vec_env.step(action)
        if dones[0]:
            break

    vec_env.close()

    video_file = os.path.join(video_dir, f"final_eval-step-0-to-step-{video_length}.mp4")
    return video_file


def evaluate_trained_agent(model, env, n_episodes: int):
    """
    Final fixed-episodes evaluation (no training).
    """
    episode_rewards = []
    for _ in range(n_episodes):
        obs = env.reset()
        done = False
        total_r = 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done_vec, info = env.step(action)
            done = bool(done_vec[0])
            total_r += float(reward[0])
        episode_rewards.append(total_r)
    mean_r = float(np.mean(episode_rewards))
    std_r = float(np.std(episode_rewards))
    print(f"[FINAL EVAL] over {n_episodes} episodes: mean={mean_r:.2f}, std={std_r:.2f}")
    return mean_r, std_r, episode_rewards


def train(model_name: str, algo_cfg: dict):
    run = wandb.init(
        project="Task3_Airraid_3Models_Wrappers",
        name=f"{model_name}_run_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}",
        entity=None,
        config={
            "env_id": ENV_ID,
            "model": model_name,
            "total_timesteps": algo_cfg["total_timesteps"],
            "n_envs": N_ENVS,
            "eval_freq": 100_000,
            "n_eval_episodes": 20,
            **algo_cfg["hyperparams"],
        },
        sync_tensorboard=True,
    )

    #Root direcotory
    ROOT = "XXXXXX" #TODO
    ROOT = os.path.join(ROOT, model_name)
    MODEL_PATH = os.path.join(ROOT, model_name+"_model")
    LOG_DIR = os.path.join(ROOT, "logs_"+model_name+"_air_raid")
    VIDEO_DIR = os.path.join(LOG_DIR, "videos")
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(VIDEO_DIR, exist_ok=True)

    tb_log_dir = os.path.join(LOG_DIR, run.id)
    best_model_dir = os.path.join(LOG_DIR, run.id, "best_model")
    os.makedirs(best_model_dir, exist_ok=True)

    vec_env = make_wrapped_vec_env(ENV_ID, n_envs=N_ENVS, seed=0, subprocess=False, norm_obs=False, norm_reward=False)

    eval_env = make_wrapped_vec_env(ENV_ID, n_envs=1, seed=123, subprocess=False, training=False)
    
    # if there is a saved VecNormalize stats, load them
    vecnormalize_path = os.path.join(LOG_DIR, "vecnormalize.pkl")
    
    if os.path.exists(vecnormalize_path):
        eval_env = VecNormalize.load(vecnormalize_path, eval_env)
    
    eval_env.training = False
    eval_env.norm_reward = False
    eval_env = VecTransposeImage(eval_env)

    #models
    print("\n>>> Creating and traininig model '{}'...".format(model_name))
    
    algo_class = algo_cfg["class"]
    model = algo_class(
        algo_cfg["policy"],
        vec_env,
        verbose=1,
        tensorboard_log=tb_log_dir,
        **algo_cfg["hyperparams"]
    )

    # Callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=1000000 // N_ENVS, 
        save_path=LOG_DIR,
        name_prefix="airraid"
    )

    wandb_callback = WandbCallback(
        gradient_save_freq=1000,
        model_save_path=f"{LOG_DIR}/wandb_chkpt",
        verbose=2
    )

    # EvalCallback that saves best_model.zip in LOG_DIR [web:3][web:31]
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=best_model_dir,
        log_path=best_model_dir,
        eval_freq=EVAL_FREQ // N_ENVS,
        n_eval_episodes=N_EVAL_EPISODES,
        deterministic=True,
        render=False,
    )
    
    video_callback = VideoRecorderCallback(
        eval_env,
        render_freq=200_000, 
        video_folder=VIDEO_DIR
    )

    #training
    model.learn(
        total_timesteps=algo_cfg["total_timesteps"],
        callback=[checkpoint_callback, wandb_callback, eval_callback, video_callback],
        progress_bar=True
    )

    model.save(MODEL_PATH)

    # Save VecNormalize statistics so we can reload them for later evaluation / inference.
    try:
        vec_env.save(vecnormalize_path)
        print(f"Saved VecNormalize stats to: {vecnormalize_path}")
    except Exception as e:
        print(f"Warning: could not save VecNormalize stats: {e}")

    # load model
    best_model_path = os.path.join(best_model_dir, "best_model.zip")
    print("Loading and evaluating model '{}'...".format(model_name))
    
    if model_name == "dqn":
        model = DQN.load(best_model_path, env=vec_env)
    elif model_name == "a2c":
        model = A2C.load(best_model_path, env=vec_env)
    elif model_name == "ppo":
        model = PPO.load(best_model_path, env=vec_env)
    else:
        print("Error, unknown model ({})".format(model_name))


    # Final fixed evaluation
    FINAL_EVAL_EPISODES = 50

    final_eval_env = make_wrapped_vec_env(ENV_ID, n_envs=1, seed=999, subprocess=False, norm_obs=False, norm_reward=False, training=False, render_mode="rgb_array")

    final_eval_env = VecNormalize.load(vecnormalize_path, final_eval_env)
    final_eval_env.training = False
    final_eval_env.norm_reward = False

    final_eval_env = VecTransposeImage(final_eval_env)

    mean_r, std_r, episode_rewards = evaluate_trained_agent(
        model, final_eval_env, n_episodes=FINAL_EVAL_EPISODES
    )

    # Log final eval metrics to W&B
    wandb.log({
        "final_eval/mean_reward": mean_r,
        "final_eval/std_reward": std_r,
        "final_eval/episodes": FINAL_EVAL_EPISODES,
    })

    # Record and log final evaluation video
    final_video_file = record_final_video(
        model,
        ENV_ID,
        video_dir=VIDEO_DIR,
        video_name="final_eval",
        video_length=3000,
    )

    if os.path.exists(final_video_file):
        wandb.log({
            "final_eval/video": wandb.Video(final_video_file, fps=30, format="mp4")
        })

    vec_env.close()
    eval_env.close()
    final_eval_env.close()
    wandb.finish()


if __name__ == "__main__":

    ALGO_CONFIGS = {
    "dqn": {
        "class": DQN,
        "policy": "CnnPolicy",
        "total_timesteps": 15_000_000,
        "hyperparams": {
            "learning_rate": 1.53e-4,
            "buffer_size": 200_000 ,
            "batch_size": 64,
            "gamma": 0.98,
            "target_update_interval": 1_000,
            "exploration_final_eps": 0.01,
        },
    },
    "a2c": {
        "class": A2C,
        "policy": "CnnPolicy",
        "total_timesteps": 15_000_000,
        "hyperparams": {
            "learning_rate": 2.5e-4,
            "n_steps": 10,    
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "ent_coef": 0.05,
        },
    },
    "ppo": {
        "class": PPO,
        "policy": "CnnPolicy",
        "total_timesteps": 15_000_000,
        "hyperparams": {
            "learning_rate": 1.34e-4,
            "n_steps": 1024,
            "batch_size": 512,
            "n_epochs": 4,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "clip_range": 0.1,
            "ent_coef": 0.01,
            "vf_coef": 0.5,
        },
    },
}

    # list of models
    models = ["dqn","a2c","ppo"]

    for model_name in models:
        algo_cfg = ALGO_CONFIGS.get(model_name)
        train(model_name, algo_cfg)
