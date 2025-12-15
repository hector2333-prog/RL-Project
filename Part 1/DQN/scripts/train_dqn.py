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
from gymnasium.wrappers import RecordVideo, MaxAndSkipObservation, ResizeObservation, GrayscaleObservation, FrameStackObservation

# --- CONFIGURATION ---
ENV_NAME = "PongNoFrameskip-v4"
MEAN_REWARD_BOUND = 19.0 
NUMBER_OF_REWARDS_TO_AVERAGE = 10          
GAMMA = 0.98       
BATCH_SIZE = 64  
LEARNING_RATE = 1.53e-4           
EXPERIENCE_REPLAY_SIZE = 200000 
SYNC_TARGET_NETWORK = 1000     
EPS_START = 1.0
EPS_DECAY = 0.99995
EPS_MIN = 0.01
EVAL_EVERY_FRAMES = 10000      
EVAL_EPISODES = 20              

SAVE_PATH = "XXXXXXX" #TODO
VIDEO_PATH = os.path.join(SAVE_PATH, "videos_DQN")
os.makedirs(VIDEO_PATH, exist_ok=True)
os.makedirs(SAVE_PATH, exist_ok=True)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

gym.register_envs(ale_py)

# --- PREPROCESSING WRAPPERS ---
class ScaledFloatFrame(gym.ObservationWrapper):
    def observation(self, obs):
        return np.array(obs).astype(np.float32) / 255.0

def make_env(env_name, render_mode=None):
    # Added render_mode argument to support video recording
    env = gym.make(env_name, render_mode=render_mode)
    env = MaxAndSkipObservation(env, skip=4)
    env = ResizeObservation(env, (84, 84))
    env = GrayscaleObservation(env, keep_dim=False) 
    env = FrameStackObservation(env, stack_size=4) 
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
    # RecordVideo wrapper saves mp4 files to VIDEO_PATH.
    try:
        mp4_files = glob.glob(f"{VIDEO_PATH}/*.mp4")
        if mp4_files:
            latest_video = max(mp4_files, key=os.path.getctime)
            print(f"Uploading video to WandB: {latest_video}")
            
            # Log the video to WandB
            wandb.log({"gameplay_video": wandb.Video(latest_video, fps=30, format="mp4")}, step=step_num)
 
    except Exception as e:
        print(f"Video upload failed: {e}")

    return np.mean(total_rewards)


# --- MAIN TRAINING LOOP ---
if __name__ == "__main__":
    # WandB Init
    wandb.login()
    wandb.init(project="Pong_DQN", config={
        "learning_rate": LEARNING_RATE,
        "batch_size": BATCH_SIZE,
        "buffer_size": EXPERIENCE_REPLAY_SIZE,
        "epsilon_decay": EPS_DECAY
    })

    MAX_ENV_STEPS = 500000

    print(">>> Training starts at ", datetime.datetime.now())

    # Setup Training Environment
    env = make_env(ENV_NAME)

    # Setup Evaluation Environment (WITH Video recording)
    # We enable 'rgb_array' mode so the wrapper can capture frames
    test_env = make_env(ENV_NAME, render_mode='rgb_array')
    
    # We wrap it with RecordVideo
    test_env = RecordVideo(
        test_env, 
        video_folder=VIDEO_PATH, 
        episode_trigger=lambda x: True, 
        name_prefix="eval"
    )

    # Neural Networks
    net = make_DQN(env.observation_space.shape, env.action_space.n).to(device)
    target_net = make_DQN(env.observation_space.shape, env.action_space.n).to(device)
    target_net.load_state_dict(net.state_dict()) 
    
    # Experience Replay Buffer and Agent
    buffer = ExperienceReplay(EXPERIENCE_REPLAY_SIZE)
    agent = Agent(env, buffer)

    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)
    
    total_rewards = []
    frame_number = 0 
    epsilon = EPS_START
    best_eval_reward = -float('inf')

    while True:
        frame_number += 1
        epsilon = max(epsilon * EPS_DECAY, EPS_MIN)

        # Agent Step
        reward = agent.step(net, epsilon, device=device)
        
        if reward is not None:
            total_rewards.append(reward)
            mean_reward = np.mean(total_rewards[-NUMBER_OF_REWARDS_TO_AVERAGE:])
            wandb.log({"epsilon": epsilon, "train_reward_10": mean_reward, "last_reward": reward}, step=frame_number)

        if frame_number >= MAX_ENV_STEPS:
            print(f"Reached max steps: {MAX_ENV_STEPS}. Ending training.")
            break


        # Evaluation Step (Every X frames)
        if frame_number % EVAL_EVERY_FRAMES == 0:
            print(f"Running evaluation at frame {frame_number}...")
            
            eval_score = evaluate_model(test_env, net, device, episodes=EVAL_EPISODES, step_num=frame_number)
            
            print(f"Evaluation Score (avg over {EVAL_EPISODES} games): {eval_score:.2f}")
            wandb.log({"eval_score": eval_score}, step=frame_number)

            # Save Best Model
            if eval_score > best_eval_reward:
                best_eval_reward = eval_score
                save_file = os.path.join(SAVE_PATH, "best_model_pong.dat")
                torch.save(net.state_dict(), save_file)
                print(f"New best model saved with score: {best_eval_reward}")

        # 4. Learning Step 
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

        # 5. Sync Target Network
        if frame_number % SYNC_TARGET_NETWORK == 0:
            target_net.load_state_dict(net.state_dict())

    # Save final model
    final_file = os.path.join(SAVE_PATH, "final_pong_model.dat")
    torch.save(net.state_dict(), final_file)
    print(">>> Training ends at ", datetime.datetime.now())
    wandb.finish()
