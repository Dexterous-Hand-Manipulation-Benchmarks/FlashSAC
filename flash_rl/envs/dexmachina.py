import gymnasium as gym
from gymnasium import spaces
import torch
import numpy as np


class GenesisFlashSACWrapper(gym.Env):
    """
    Wraps Dexmachina's BaseEnv to look exactly like a Gymnasium vectorized
    environment so FlashSAC can interact with it smoothly.
    """

    def __init__(self, env, rl_device="cuda:0"):
        super().__init__()
        self.env = env
        self.rl_device = rl_device
        self._clip_actions = self.env.action_clip

        self.num_envs = self.env.num_envs

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.env.obs_dim,)
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.env.action_dim,)
        )

    def reset(self, seed=None, options=None):
        obs_dict, extras = self.env.reset()
        obs = obs_dict["policy"] if isinstance(obs_dict, dict) else obs_dict

        if extras is None:
            extras = {}

        # --- FlashSAC requires these to build its neural networks ---
        extras["actor_observation_size"] = (self.env.obs_dim,)
        extras["critic_observation_size"] = (self.env.obs_dim,)

        return obs, extras

    def step(self, actions):
        if not isinstance(actions, torch.Tensor):
            actions = torch.tensor(actions, dtype=torch.float32, device=self.rl_device)
        else:
            actions = actions.to(self.rl_device)

        if actions.ndim == 1:
            batch_size = actions.numel() // self.env.action_dim
            actions = actions.view(batch_size, self.env.action_dim)

        actions = torch.clamp(actions, -self._clip_actions, self._clip_actions)

        obs_dict, rew, terminated, truncated, extras = self.env.step(actions)
        obs = obs_dict["policy"] if isinstance(obs_dict, dict) else obs_dict

        if extras is None:
            extras = {}

        extras["final_obs"] = obs.clone()
        if "log" in extras:
            extras["episode_info"] = extras["log"]

        return obs, rew, terminated, truncated, extras

    def close(self):
        pass