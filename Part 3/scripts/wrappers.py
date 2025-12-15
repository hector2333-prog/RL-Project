import gymnasium as gym
from gymnasium import ObservationWrapper, ActionWrapper
from gymnasium import spaces
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

class AirRaidActionWrapper(gym.ActionWrapper):
    """Reduce Air Raid minimal action set to 3 actions: FIRE, LEFT, RIGHT."""
    
    def __init__(self, env):
        super().__init__(env)
        # New action space: 3 actions
        self.action_space = spaces.Discrete(3)

    def action(self, act):
        # act in {0,1,2}
        if act == 0:      # FIRE
            return 1      # original FIRE
        elif act == 1:    # LEFT
            return 3      # original LEFT
        elif act == 2:    # RIGHT
            return 2      # original RIGHT
        else:
            raise ValueError(f"Invalid action {act}")


class AirRaidBuildingPenaltyWrapper(gym.Wrapper):
    """
    Reward-shaping wrapper for Air Raid:
    - Assumes grayscale 84x84 observations (after Atari preprocessing).
    - Monitors a rectangular region where the buildings are.
    - Adds a negative reward when the number of "building pixels" in that
      region decreases (interpreted as a building being hit).
    """

    def __init__(self, env, penalty=-0.05,
                 y_min=59, y_max=71, x_min=0, x_max=83,
                 pixel_threshold=100):
        """
        penalty: negative reward added per detected building hit.
        (y_min, y_max, x_min, x_max): crop defining the building region
                                     in the 84x84 frame.
        pixel_threshold: grayscale value above which a pixel is considered
                         part of a building (tune empirically).
        """
        super().__init__(env)
        self.penalty = penalty
        self.y_min = y_min
        self.y_max = y_max
        self.x_min = x_min
        self.x_max = x_max
        self.pixel_threshold = pixel_threshold

        self._prev_building_pixels = None


    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._prev_building_pixels = self._count_building_pixels(obs)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        current_pixels = self._count_building_pixels(obs)

        if self._prev_building_pixels is not None:
            effective_prev = self._prev_building_pixels
            if current_pixels > effective_prev:
                current_pixels = effective_prev

            delta = current_pixels - effective_prev
            if delta < 0:
                reward += self.penalty
            self._prev_building_pixels = current_pixels
        else:
            self._prev_building_pixels = current_pixels

        return obs, reward, terminated, truncated, info
    

    def _count_building_pixels(self, obs):
        """
        obs: expected shape (84, 84) or (84, 84, 1) grayscale uint8.
        Returns the number of "bright" pixels in the building region.
        """
        if obs.ndim == 3:
            if obs.shape == (84, 84, 1):
                frame = obs[:, :, 0]        # (84,84)
            elif obs.shape[0] == 4:          # (4,H,W)
                frame = obs[-1]            # (H,W)
            elif obs.shape[2] == 4:        # (H,W,4)
                frame = obs[:, :, -1]      # (H,W)
            else:
                raise ValueError(f"Unexpected obs shape {obs.shape}")
        else:
            frame = obs

        # Crop building region
        crop = frame[self.y_min:self.y_max+1, self.x_min:self.x_max+1]

        # Count bright pixels (building blocks)
        count = np.sum(crop >= self.pixel_threshold)
        return int(count)
    

class CropTopBar(ObservationWrapper):
    """Crops the top score bar from Atari observations, then resizes to 84x84."""

    def __init__(self, env, crop_top=40):
        super().__init__(env)
        self.crop_top = crop_top
        self.height = 84
        self.width = 84
        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(self.height, self.width, 1),
            dtype=np.uint8
        )

    def observation(self, obs):
        import cv2
        cropped = obs[self.crop_top:, :, :]

        resized = cv2.resize(cropped, (self.width, self.height), interpolation=cv2.INTER_AREA)

        if resized.ndim == 2:
            resized = resized[:, :, None]

        return resized.astype(np.uint8)


class StickyActions(ActionWrapper):
    """Repeat last action with probability p (default 0.25)."""
    def __init__(self, env, p=0.25):
        super().__init__(env)
        self.p = p
        self.last_action_applied = 0

    def action(self, action):
        if np.random.rand() < self.p:
            return self.last_action_applied
        self.last_action_applied = action
        return action


class SemanticMaskObs(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.cluster_centers = np.array([
            [0.392, 0.392, 0.392],
            [44.0, 44.0, 44.0],
            [88.0, 88.0, 88.0],
            [113.0, 113.0, 113.0],
            [144.0, 144.0, 144.0],
        ], dtype=np.float32)

        self.enemy_cluster = 4
        self.bomb_cluster  = 2
        self.asset_cluster = 1

        self.height = 84
        self.width = 84

        #Now the shape is (C,H,W) for SB3
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=(3, self.height, self.width), dtype=np.float32
        )

    def observation(self, obs):
        import cv2
        #convert to 3 channels if 2D
        if obs.ndim == 2:
            obs = np.stack([obs]*3, axis=-1)  # (H,W) -> (H,W,3)
        
        # Resize
        obs_resized = cv2.resize(obs, (self.width, self.height), interpolation=cv2.INTER_NEAREST)
        
        #ensure 3 channels
        if obs_resized.ndim == 2: 
            obs_resized = np.stack([obs_resized]*3, axis=-1)  # (H,W) -> (H,W,3)
        elif obs_resized.shape[2] != 3:
            #in case of a different number of channels
            obs_resized = obs_resized[:, :, :3]

        h, w, _ = obs_resized.shape
        flat = obs_resized.reshape(-1, 3).astype(np.float32)
        dists = np.linalg.norm(flat[:, None, :] - self.cluster_centers[None, :, :], axis=-1)
        labels = dists.argmin(axis=1).reshape(h, w)

        enemy = (labels == self.enemy_cluster).astype(np.float32)
        bomb  = (labels == self.bomb_cluster).astype(np.float32)
        asset = (labels == self.asset_cluster).astype(np.float32)

        mask = np.stack([enemy, bomb, asset], axis=-1)  # (H,W,C)
        mask_resized = np.transpose(mask, (2,0,1))       # (C,H,W)
        return mask_resized.astype(np.float32)





