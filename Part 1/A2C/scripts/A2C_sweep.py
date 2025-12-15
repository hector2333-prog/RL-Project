import gymnasium as gym
import ale_py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import time
import datetime
import wandb
import os
import glob
from gymnasium.wrappers import RecordVideo, MaxAndSkipObservation, ResizeObservation, GrayscaleObservation, FrameStackObservation, TransformReward

# --- RELIABLE CONFIGURATION ---
ENV_NAME = "PongNoFrameskip-v4"
N_ENVS = 16                  # Keep high for speed (Data collection)
N_STEPS = 10                 # Changed to 10 for better stability with 16 envs
MEAN_REWARD_BOUND = 19.5     
GAMMA = 0.99
GAE_LAMBDA = 0.95            
ENTROPY_BETA = 0.05          # INCREASED: Prevents the "Entropy 0" collapse
LEARNING_RATE = 2.5e-4       # LOWERED: Prevents the network from panicking
EVAL_EVERY_FRAMES = 50000
EVAL_EPISODES = 20

SAVE_PATH = "/home/hsalguero/RL/A2C_good/models"
VIDEO_PATH = os.path.join(SAVE_PATH, "videos_A2C") # Path to save videos locally
os.makedirs(VIDEO_PATH, exist_ok=True)
os.makedirs(SAVE_PATH, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
gym.register_envs(ale_py)

# -------- PREPROCESSING ----------
class ScaledFloatFrame(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        old_shape = env.observation_space.shape
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=old_shape, dtype=np.float32
        )

    def observation(self, obs):
        return np.array(obs).astype(np.float32) / 255.0

def make_env(env_name, render_mode=None):
    env = gym.make(env_name, render_mode=render_mode)
    env = MaxAndSkipObservation(env, skip=4)
    env = ResizeObservation(env, (84, 84))
    env = GrayscaleObservation(env, keep_dim=False)
    # Clip rewards to stabilize training
    env = TransformReward(env, lambda r: np.clip(r, -1, 1))
    env = FrameStackObservation(env, stack_size=4)
    env = ScaledFloatFrame(env)
    return env

def make_vec_envs(env_name, n_envs):
    return gym.vector.SyncVectorEnv([lambda: make_env(env_name) for _ in range(n_envs)])

# -------- Neural Networks (Orthogonal Init) ----------
def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class ActorNet(nn.Module):
    def __init__(self, input_shape, output_shape):
        super().__init__()
        self.conv = nn.Sequential(
            layer_init(nn.Conv2d(input_shape[0], 32, 8, stride=4)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, 4, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, 3, stride=1)),
            nn.ReLU(),
            nn.Flatten()
        )
        with torch.no_grad():
            dummy = torch.zeros(1, *input_shape)
            n_flatten = self.conv(dummy).shape[1]

        self.fc = nn.Sequential(
            layer_init(nn.Linear(n_flatten, 512)),
            nn.ReLU()
        )
        # Output layer scaled by 0.01 to ensure initial policy is near-random
        self.logits = layer_init(nn.Linear(512, output_shape), std=0.01)

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return self.logits(x)

class CriticNet(nn.Module):
    def __init__(self, input_shape):
        super().__init__()
        self.conv = nn.Sequential(
            layer_init(nn.Conv2d(input_shape[0], 32, 8, stride=4)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, 4, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, 3, stride=1)),
            nn.ReLU(),
            nn.Flatten()
        )
        with torch.no_grad():
            dummy = torch.zeros(1, *input_shape)
            n_flatten = self.conv(dummy).shape[1]

        self.fc = nn.Sequential(
            layer_init(nn.Linear(n_flatten, 512)),
            nn.ReLU()
        )
        self.value = layer_init(nn.Linear(512, 1), std=1)

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return self.value(x).squeeze(-1)

# ------------ AGENT -------------
class Agent:
    def __init__(self, actor, critic, device):
        self.actor = actor
        self.critic = critic
        self.device = device

    @torch.no_grad()
    def get_action_and_value(self, state):
        state_t = torch.tensor(state, dtype=torch.float32).to(self.device)
        logits = self.actor(state_t)
        dist = torch.distributions.Categorical(logits=logits)
        actions = dist.sample()
        log_probs = dist.log_prob(actions)
        values = self.critic(state_t)
        return actions.cpu().numpy(), log_probs.cpu().numpy(), values.cpu().numpy()

    @torch.no_grad()
    def value(self, state):
        state_t = torch.tensor(state, dtype=torch.float32).to(self.device)
        return self.critic(state_t).cpu().numpy()

# --------- ROLLOUT & GAE -----------
def collect_rollout(envs, agent, obs, n_steps, device):
    obs_lst, actions_lst, rewards_lst, dones_lst, values_lst, log_probs_lst = [], [], [], [], [], []

    for _ in range(n_steps):
        actions, log_probs, values = agent.get_action_and_value(obs)

        next_obs, rewards, terminated, truncated, _ = envs.step(actions)
        dones = np.logical_or(terminated, truncated)

        obs_lst.append(obs)
        actions_lst.append(actions)
        rewards_lst.append(rewards)
        dones_lst.append(dones)
        values_lst.append(values)
        log_probs_lst.append(log_probs)

        obs = next_obs

    last_value = agent.value(obs)

    return (
        np.array(obs_lst),
        np.array(actions_lst),
        np.array(rewards_lst),
        np.array(dones_lst),
        np.array(values_lst),
        np.array(log_probs_lst),
        last_value,
        obs 
    )

def compute_gae(rewards, dones, values, last_value, gamma=GAMMA, lam=GAE_LAMBDA):
    n_steps, n_envs = rewards.shape
    advantages = np.zeros_like(rewards, dtype=np.float32)
    last_gae_lam = 0
    
    for step in reversed(range(n_steps)):
        if step == n_steps - 1:
            next_non_terminal = 1.0 - dones[step]
            next_value = last_value
        else:
            next_non_terminal = 1.0 - dones[step]
            next_value = values[step + 1]
        
        delta = rewards[step] + gamma * next_value * next_non_terminal - values[step]
        advantages[step] = last_gae_lam = delta + gamma * lam * next_non_terminal * last_gae_lam
    
    returns = advantages + values
    return returns, advantages

# ---------- Evaluation ------------
def evaluate_model(env, actor, device, episodes=5, step_num=0):
    total_rewards = []
    for _ in range(episodes):
        state, _ = env.reset()
        done = False
        reward_sum = 0
        while not done:
            state_t = torch.tensor(np.array([state]), dtype=torch.float32).to(device)
            with torch.no_grad():
                logits = actor(state_t)
            action = torch.argmax(logits, dim=1).item()
            state, r, terminated, truncated, _ = env.step(action)
            reward_sum += r
            done = terminated or truncated
        total_rewards.append(reward_sum)
    
    try:
        mp4_files = glob.glob(f"{VIDEO_PATH}/*.mp4")
        if mp4_files:
            latest_video = max(mp4_files, key=os.path.getctime)
            wandb.log({"gameplay_video": wandb.Video(latest_video, fps=30, format="mp4")}, step=step_num)
    except Exception as e:
        print(f"Video error: {e}")

    return np.mean(total_rewards)

def train(config = None):
    with wandb.init(project="Pong_A2C_Sweep", config=config):
        cfg = wandb.config

        # Hyperparameters from sweep / defaults
        LEARNING_RATE = cfg.get("learning_rate", 2.5e-4)
        ENTROPY_BETA = cfg.get("entropy_beta", 0.05)
        GAE_LAMBDA = cfg.get("gae_lambda", 0.95)
        N_ENVS = cfg.get("n_envs", 16)
        N_STEPS = cfg.get("n_steps", 10)
        OPTIMIZER_NAME = cfg.get("optimizer", "rmsprop")
        
        MAX_ENV_STEPS = 3750000

        print(">>> Training starts at ", datetime.datetime.now())
        print("Config:", dict(cfg))

        envs = make_vec_envs(ENV_NAME, N_ENVS)
    
        test_env = make_env(ENV_NAME, render_mode='rgb_array')
        test_env = RecordVideo(
            test_env,
            video_folder=VIDEO_PATH,
            episode_trigger=lambda ep: ep % 10 == 0,
            name_prefix="eval"
        )

        actor = ActorNet(envs.single_observation_space.shape, envs.single_action_space.n).to(device)
        critic = CriticNet(envs.single_observation_space.shape).to(device)
        agent = Agent(actor, critic, device)

        if OPTIMIZER_NAME.lower() == "adam":
            optimizer_actor = optim.Adam(actor.parameters(), lr=LEARNING_RATE)
            optimizer_critic = optim.Adam(critic.parameters(), lr=LEARNING_RATE)
        elif OPTIMIZER_NAME.lower() == "rmsprop":
            optimizer_actor = optim.RMSprop(actor.parameters(), lr=LEARNING_RATE, eps=1e-5, alpha=0.99)
            optimizer_critic = optim.RMSprop(critic.parameters(), lr=LEARNING_RATE, eps=1e-5, alpha=0.99)
        else:
            raise ValueError(f"Unknown optimizer: {OPTIMIZER_NAME}")
        
        frame_number = 0
        best_eval_reward = -999
        obs, _ = envs.reset()

        episode_rewards = np.zeros(envs.num_envs, dtype=np.float32)

        while True:
            # 1) Rollout
            roll = collect_rollout(envs, agent, obs, N_STEPS, device)
            obs_b, act_b, rew_b, done_b, val_b, logp_b, last_val, obs = roll
            
            frame_number += N_ENVS * N_STEPS

            # Log episodic rewards
            for t in range(rew_b.shape[0]):
                episode_rewards += rew_b[t]
                finished = done_b[t]
                if np.any(finished):
                    for idx in np.where(finished)[0]:
                        wandb.log(
                            {"episodic_reward": episode_rewards[idx]},
                            step=frame_number
                        )
                        episode_rewards[idx] = 0.0

            # Stop if we hit max env steps
            if frame_number >= MAX_ENV_STEPS:
                print(f"Reached max_env_steps={MAX_ENV_STEPS}, stopping run.")
                break

            # 2) Compute GAE
            returns, advantages = compute_gae(rew_b, done_b, val_b, last_val, gamma=GAMMA, lam=GAE_LAMBDA)

            # Flatten
            obs_t = torch.tensor(obs_b.reshape(-1, *obs_b.shape[2:]), dtype=torch.float32).to(device)
            actions_t = torch.tensor(act_b.flatten(), dtype=torch.long).to(device)
            returns_t = torch.tensor(returns.flatten(), dtype=torch.float32).to(device)
            adv_t = torch.tensor(advantages.flatten(), dtype=torch.float32).to(device)

            # Normalize Advantages
            adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)

            # ---- ACTOR LOSS ----
            logits = actor(obs_t)
            dist = torch.distributions.Categorical(logits=logits)
            log_probs = dist.log_prob(actions_t)
            entropy = dist.entropy().mean()

            actor_loss = -(log_probs * adv_t).mean() - ENTROPY_BETA * entropy

            # ---- CRITIC LOSS ----
            values = critic(obs_t)
            critic_loss = nn.MSELoss()(values, returns_t)

            # ---- OPTIMIZE ----
            optimizer_actor.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(actor.parameters(), 0.5)
            optimizer_actor.step()

            optimizer_critic.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(critic.parameters(), 0.5)
            optimizer_critic.step()

            # ---- EVALUATION ----
            if frame_number % EVAL_EVERY_FRAMES == 0:
                eval_score = evaluate_model(test_env, actor, device, EVAL_EPISODES, frame_number)
                print(f"Frame {frame_number}, Eval score: {eval_score:.2f}")
                
                wandb.log({
                    "eval_score": eval_score,
                    "actor_loss": actor_loss.item(),
                    "critic_loss": critic_loss.item(),
                    "entropy": entropy.item()
                }, step=frame_number)

                if eval_score > best_eval_reward:
                    best_eval_reward = eval_score
                    torch.save(actor.state_dict(), os.path.join(SAVE_PATH, "best_actor.dat"))
                    torch.save(critic.state_dict(), os.path.join(SAVE_PATH, "best_critic.dat"))
                    print(f"New best saved: {best_eval_reward}")

            #if best_eval_reward >= MEAN_REWARD_BOUND:
                #print(f"SOLVED! Reward {best_eval_reward} in {frame_number} frames.")
                #break
        
        torch.save(actor.state_dict(), os.path.join(SAVE_PATH, "final_actor.dat"))
        torch.save(critic.state_dict(), os.path.join(SAVE_PATH, "final_critic.dat"))
        print(">>> Training ends at ", datetime.datetime.now())

if __name__ == "__main__":
    #wandb.login()

    sweep_config = {
        "method": "bayes",
        "metric": {"name": "eval_score", "goal": "maximize"},
        "parameters": {
            "learning_rate": {
                "distribution": "log_uniform_values",
                "min": 1e-5,
                "max": 5e-4
            },
            "entropy_beta": {
                "values": [0.02, 0.05, 0.075]
            },
            "gae_lambda": {
                "values": [0.90, 0.95]
            },
            "n_envs": {
                "values": [8, 16]
            },
            "n_steps": {
                "values": [5, 10, 20]
            },
            "optimizer": {
                "values": ["adam", "rmsprop"]
            }
        }
    }

    sweep_id = wandb.sweep(sweep_config, project="Pong_A2C_Sweep")

    wandb.agent(sweep_id, function=train, count=30)