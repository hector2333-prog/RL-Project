import numpy as np
from stable_baselines3 import PPO
import os

# --- CONFIGURATION ---
# These must match the filenames you submit exactly.
LEFT_MODEL_PATH = "round_200_Left.zip"
RIGHT_MODEL_PATH = "round_200_Right.zip"
class TournamentAgent:
    def __init__(self, model_path):
        # Path handling to ensure we find the zip file relative to this script
        script_dir = os.path.dirname(os.path.abspath(__file__))
        full_path = os.path.join(script_dir, model_path)
        
        try:
            self.model = PPO.load(full_path, device="cpu")
        except FileNotFoundError:
            self.model = PPO.load(model_path, device="cpu")
            
        self.steps_since_fire = 0 

    def predict(self, obs, deterministic=True):
        # --- 1. SAFETY MASKING (Critical Fix) ---
        # The tournament environment might show the score, but our agent 
        # was trained without it. We manually mask the top 10 pixels here 
        # to ensure the agent sees exactly what it expects.
        masked_obs = obs.copy()
        # Shape is usually (4, 84, 84) or (84, 84, 4) depending on processing
        # We assume channel-first (4, 84, 84) based on training, but handle both just in case
        if masked_obs.shape[0] == 4: 
            masked_obs[:, 0:10, :] = 0 
        elif masked_obs.shape[2] == 4:
            masked_obs[0:10, :, :] = 0

        # --- 2. PREDICTION ---
        action, state = self.model.predict(masked_obs, deterministic=deterministic)
        
        # --- 3. ANTI-FREEZE LOGIC ---
        if action == 1: 
            self.steps_since_fire = 0
        else:
            self.steps_since_fire += 1
        
        if self.steps_since_fire > 100:
            action = 1 
            self.steps_since_fire = 0
            
        return action, state

def load_left_agent():
    return TournamentAgent(LEFT_MODEL_PATH)

def load_right_agent():
    return TournamentAgent(RIGHT_MODEL_PATH)