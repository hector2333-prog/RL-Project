import gymnasium as gym
import ale_py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import collections
import time
import datetime
import wandb
import os
import glob 
import cv2
from gymnasium.wrappers import RecordVideo, MaxAndSkipObservation, ResizeObservation, GrayscaleObservation, FrameStackObservation

# --- CONFIGURATION ---
ENV_NAME = "PongNoFrameskip-v4"
MEAN_REWARD_BOUND = 19.0 
NUMBER_OF_REWARDS_TO_AVERAGE = 10          
GAMMA = 0.99       
BATCH_SIZE = 32  
LEARNING_RATE = 1e-4           
EXPERIENCE_REPLAY_SIZE = 50000 
SYNC_TARGET_NETWORK = 1000     
EPS_START = 1.0
EPS_DECAY = 0.999985
EPS_MIN = 0.02
EVAL_EVERY_FRAMES = 10000      
EVAL_EPISODES = 20              

SAVE_PATH = "/home/hsalguero/RL/models_DQN"
VIDEO_PATH = os.path.join(SAVE_PATH, "videos_DQN") # Path to save videos locally
#SAVE_PATH = "/fhome/pmlai07/task1/task1_2n/models"
#VIDEO_PATH = os.path.join(SAVE_PATH, "videos") # Path to save videos locally
#SAVE_PATH = "./DQN_sweep/saved_models"
#VIDEO_PATH = os.path.join(SAVE_PATH, "videos_DQN") # Path to save videos locally
os.makedirs(VIDEO_PATH, exist_ok=True)
os.makedirs(SAVE_PATH, exist_ok=True)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Register ALE environments
gym.register_envs(ale_py)

# Ensure video directory exists
os.makedirs(VIDEO_PATH, exist_ok=True)

# --- PREPROCESSING WRAPPERS ---
class ScaledFloatFrame(gym.ObservationWrapper):
    def observation(self, obs):
        return np.array(obs).astype(np.float32) / 255.0
class SkipFrame(gym.Wrapper):
    """
    Replaces MaxAndSkipObservation.
    1. Repeats the action for 'skip' number of frames.
    2. Sums the rewards over those frames.
    3. Returns the max-pooled observation of the last two frames (to deal with Atari flickering).
    """
    def __init__(self, env, skip=4):
        super().__init__(env)
        self._skip = skip

    def step(self, action):
        total_reward = 0.0
        done = False
        # Buffer to hold the last two frames for max pooling
        obs_buffer = collections.deque(maxlen=2)
        
        for _ in range(self._skip):
            obs, reward, terminated, truncated, info = self.env.step(action)
            obs_buffer.append(obs)
            total_reward += reward
            done = terminated or truncated
            if done:
                break
        
        # Atari specific: Max-pool over the last two observations to remove flickering
        # If we only have 1 frame (died immediately), use that.
        if len(obs_buffer) > 1:
            max_frame = np.maximum(obs_buffer[0], obs_buffer[1])
        else:
            max_frame = obs_buffer[0]
            
        return max_frame, total_reward, terminated, truncated, info

class GrayScaleResize(gym.ObservationWrapper):
    """
    Combines ResizeObservation and GrayscaleObservation.
    1. Converts to Grayscale (RGB -> Y).
    2. Resizes to 84x84.
    """
    def __init__(self, env, shape=(84, 84)):
        super().__init__(env)
        self.shape = shape
        # Update the observation space to match new shape (H, W) - no channels yet
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=self.shape, dtype=np.uint8
        )

    def observation(self, observation):
        # 1. Convert to Grayscale using OpenCV
        # (If observation is already gray, skip this)
        if observation.shape[-1] == 3:
            observation = cv2.cvtColor(observation, cv2.COLOR_RGB2GRAY)
        
        # 2. Resize
        observation = cv2.resize(observation, self.shape, interpolation=cv2.INTER_AREA)
        return observation

class FrameStack(gym.Wrapper):
    """
    Replaces FrameStackObservation.
    Stacks 'k' last frames. Returns shape (k, H, W).
    This fits PyTorch's expectation of (Channels, Height, Width).
    """
    def __init__(self, env, k=4):
        super().__init__(env)
        self.k = k
        self.frames = collections.deque(maxlen=k)
        
        # Update observation space: (k, 84, 84)
        old_space = env.observation_space
        self.observation_space = gym.spaces.Box(
            low=0, 
            high=255, 
            shape=(k, old_space.shape[0], old_space.shape[1]), 
            dtype=np.uint8
        )

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        # Fill the buffer with the initial frame k times
        for _ in range(self.k):
            self.frames.append(obs)
        return self._get_obs(), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.frames.append(obs)
        return self._get_obs(), reward, terminated, truncated, info

    def _get_obs(self):
        # Stack frames along the 0th axis -> (4, 84, 84)
        return np.array(self.frames)

# Keep this custom wrapper you already had
class ScaledFloatFrame(gym.ObservationWrapper):
    def observation(self, obs):
        return np.array(obs).astype(np.float32) / 255.0

def make_env(env_name, render_mode=None):
    # 1. Base Environment
    env = gym.make(env_name, render_mode=render_mode)
    
    # 2. Skip Frames (Replaces MaxAndSkipObservation)
    env = SkipFrame(env, skip=4)
    
    # 3. Grayscale and Resize (Replaces ResizeObservation & GrayscaleObservation)
    env = GrayScaleResize(env, shape=(84, 84))
    
    # 4. Stack Frames (Replaces FrameStackObservation)
    env = FrameStack(env, k=4) # Output shape: (4, 84, 84)
    
    # 5. Normalize to float (Your wrapper)
    env = ScaledFloatFrame(env)
    
    return env

# --- NEURAL NETWORK ---
def make_DQN(input_shape, output_shape):
    return nn.Sequential(
        nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
        nn.ReLU(),
        nn.Conv2d(32, 64, kernel_size=4, stride=2),
        nn.ReLU(),
        nn.Conv2d(64, 64, kernel_size=3, stride=1),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(64*7*7, 512),
        nn.ReLU(),
        nn.Linear(512, output_shape)
    )

# --- REPLAY BUFFER ---
Experience = collections.namedtuple('Experience', field_names=['state', 'action', 'reward', 'done', 'new_state'])

class ExperienceReplay:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def __len__(self):
        return len(self.buffer)

    def append(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, actions, rewards, dones, next_states = zip(*[self.buffer[idx] for idx in indices])
        
        return np.array(states), np.array(actions), np.array(rewards, dtype=np.float32), \
               np.array(dones, dtype=np.uint8), np.array(next_states)

# --- AGENT ---
class Agent:
    def __init__(self, env, exp_replay_buffer):
        self.env = env
        self.exp_replay_buffer = exp_replay_buffer
        self._reset()

    def _reset(self):
        self.current_state = self.env.reset()[0]
        self.total_reward = 0.0

    def step(self, net, epsilon=0.0, device="cpu"):
        done_reward = None
        
        if np.random.random() < epsilon:
            action = self.env.action_space.sample()
        else:
            state_a = np.array([self.current_state], copy=False)
            state_v = torch.tensor(state_a).to(device)
            q_vals = net(state_v)
            _, act_v = torch.max(q_vals, dim=1)
            action = int(act_v.item())

        new_state, reward, terminated, truncated, _ = self.env.step(action)
        is_done = terminated or truncated
        self.total_reward += reward

        exp = Experience(self.current_state, action, reward, is_done, new_state)
        self.exp_replay_buffer.append(exp)
        self.current_state = new_state
        
        if is_done:
            done_reward = self.total_reward
            self._reset()
        
        return done_reward

# --- EVALUATION FUNCTION WITH VIDEO LOGGING ---
def evaluate_model(env, net, device, episodes=5, step_num=0):
    """Runs the agent with epsilon=0 to test true performance and uploads video"""
    total_rewards = []
    
    # Run evaluation episodes
    for _ in range(episodes):
        state, _ = env.reset()
        episode_reward = 0
        done = False
        while not done:
            state_t = torch.tensor(np.array([state])).to(device)
            q_values = net(state_t)
            action = torch.argmax(q_values, dim=1).item() 
            state, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward
            done = terminated or truncated
        total_rewards.append(episode_reward)

    # VIDEO UPLOAD LOGIC
    # RecordVideo wrapper saves mp4 files to VIDEO_PATH. We find the newest one.
    try:
        # Find all mp4 files in the folder
        mp4_files = glob.glob(f"{VIDEO_PATH}/*.mp4")
        if mp4_files:
            # Select the most recently created video
            latest_video = max(mp4_files, key=os.path.getctime)
            print(f"Uploading video to WandB: {latest_video}")
            
            # Log the video to WandB
            wandb.log({"gameplay_video": wandb.Video(latest_video, fps=30, format="mp4")}, step=step_num)
            
            # Optional: Delete uploaded video to save cluster space
            # os.remove(latest_video) 
    except Exception as e:
        print(f"Video upload failed: {e}")

    return np.mean(total_rewards)


def train(config=None):
    with wandb.init(project="Pong_DQN_sweep", config=config):
        cfg = wandb.config

        # Hyperparameters from sweep / defaults
        GAMMA = cfg.get("gamma", 0.99)
        BATCH_SIZE = cfg.get("batch_size", 32)
        LEARNING_RATE = cfg.get("learning_rate", 1e-4)
        EXPERIENCE_REPLAY_SIZE = cfg.get("buffer_size", 50000)
        SYNC_TARGET_NETWORK = cfg.get("target_update", 1000)
        EPS_START = 1.0
        EPS_MIN = cfg.get("epsilon_end", 0.02)
        EPS_DECAY = cfg.get("epsilon_decay_frames", 0.999985)
        OPTIMIZER_NAME = cfg.get("optimizer", "adam")

        MAX_ENV_STEPS = 200000

        print(">>> Training starts at ", datetime.datetime.now())
        print("Config:", dict(cfg))

        # Environments
        env = make_env(ENV_NAME)
        test_env = make_env(ENV_NAME, render_mode='rgb_array')
        test_env = RecordVideo(
            test_env,
            video_folder=VIDEO_PATH,
            episode_trigger=lambda ep: ep % 10 == 0,
            name_prefix="eval"
        )

        net = make_DQN(env.observation_space.shape, env.action_space.n).to(device)
        target_net = make_DQN(env.observation_space.shape, env.action_space.n).to(device)
        target_net.load_state_dict(net.state_dict())

        buffer = ExperienceReplay(EXPERIENCE_REPLAY_SIZE)
        agent = Agent(env, buffer)

        if OPTIMIZER_NAME.lower() == "adam":
            optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)
        elif OPTIMIZER_NAME.lower() == "rmsprop":
            optimizer = optim.RMSprop(net.parameters(), lr=LEARNING_RATE, alpha=0.95, eps=0.01)
        else:
            raise ValueError(f"Unknown optimizer: {OPTIMIZER_NAME}")

        total_rewards = []
        frame_number = 0
        epsilon = EPS_START
        best_eval_reward = -float("inf")

        # Main training loop
        while True:
            frame_number += 1

            epsilon = max(epsilon * EPS_DECAY, EPS_MIN)

            # 1. Agent step
            reward = agent.step(net, epsilon, device=device)

            if reward is not None:
                total_rewards.append(reward)
                mean_reward = np.mean(total_rewards[-NUMBER_OF_REWARDS_TO_AVERAGE:])
                wandb.log(
                    {
                        "epsilon": epsilon,
                        "train_reward_10": mean_reward,
                        "last_reward": reward,
                        "frame": frame_number
                    },
                    step=frame_number
                )

            if frame_number >= MAX_ENV_STEPS:
                print(f"Reached max steps: {MAX_ENV_STEPS}. Ending training.")
                break

            # 2. Evaluation step
            if frame_number % EVAL_EVERY_FRAMES == 0:
                print(f"Running evaluation at frame {frame_number}...")

                eval_score = evaluate_model(
                    test_env, net, device,
                    episodes=EVAL_EPISODES,
                    step_num=frame_number
                )
                print(f"Evaluation Score (avg over {EVAL_EPISODES} games): {eval_score:.2f}")
                wandb.log({"eval_score": eval_score}, step=frame_number)

                if eval_score > best_eval_reward:
                    best_eval_reward = eval_score
                    save_file = os.path.join(SAVE_PATH, "best_model_pong_sweep.dat")
                    torch.save(net.state_dict(), save_file)
                    print(f"New best model saved with score: {best_eval_reward}")

            # Optional: early stop if solved
            #if best_eval_reward >= MEAN_REWARD_BOUND:
            #    print(f"SOLVED! Reached reward {best_eval_reward} in {frame_number} frames.")
            #    break

            # 3. Learning step
            if len(buffer) < EXPERIENCE_REPLAY_SIZE // 10:
                continue

            batch = buffer.sample(BATCH_SIZE)
            states_, actions_, rewards_, dones_, next_states_ = batch

            states = torch.tensor(states_).to(device)
            next_states = torch.tensor(next_states_).to(device)
            actions = torch.tensor(actions_).to(device)
            rewards = torch.tensor(rewards_).to(device)
            dones_t = torch.BoolTensor(dones_).to(device)

            Q_values = net(states).gather(1, actions.unsqueeze(-1)).squeeze(-1)

            with torch.no_grad():
                next_state_values = target_net(next_states).max(1)[0]
                next_state_values[dones_t] = 0.0

            expected_Q_values = rewards + GAMMA * next_state_values
            loss = nn.MSELoss()(Q_values, expected_Q_values)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            wandb.log({"loss": loss.item()}, step=frame_number)

            # 4. Sync target network
            if frame_number % SYNC_TARGET_NETWORK == 0:
                target_net.load_state_dict(net.state_dict())

        # Save final model
        final_file = os.path.join(SAVE_PATH, "final_pong_model_final.dat")
        torch.save(net.state_dict(), final_file)
        print(">>> Training ends at ", datetime.datetime.now())

# --- MAIN TRAINING LOOP ---

if __name__ == "__main__":
    wandb.login()

    # Define the sweep configuration (you can also load from YAML)
    sweep_config = {
        "method": "bayes",
        "metric": {"name": "eval_score", "goal": "maximize"},
        "parameters": {
            "learning_rate": {
                "distribution": "log_uniform_values",
                "min": 1e-5,
                "max": 5e-4
            },
            "gamma": {"values": [0.98, 0.99]},
            "batch_size": {"values": [32, 64]},
            "buffer_size": {"values": [50000, 100000, 200000]},
            "target_update": {"values": [1000, 5000, 10000]},
            "epsilon_end": {"values": [0.01, 0.02, 0.1]},
            "epsilon_decay_frames": {"values": [0.99995, 0.999985]},
            "optimizer": {"values": ["adam", "rmsprop"]},
        }
    }

    # Create the sweep on W&B
    sweep_id = wandb.sweep(sweep_config, project="Pong_DQN_sweep")

    # Launch the sweep agent (this process will run multiple train() calls)
    # count=None → run forever; set an int for a fixed number of runs
    wandb.agent(sweep_id, function=train, count=30)

