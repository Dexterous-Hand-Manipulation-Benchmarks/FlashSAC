import gymnasium as gym
from gymnasium.vector import VectorEnv
from gymnasium.vector.utils import batch_space
import torch
import numpy as np


class GenesisFlashSACWrapper(VectorEnv):
    """
    Wraps Dexmachina's BaseEnv as a Gymnasium VectorEnv compatible with FlashSAC.
    """

    def __init__(self, env, rl_device="cuda:0"):
        self.env = env
        self.rl_device = rl_device
        self._clip_actions = self.env.action_clip
        self.num_envs = self.env.num_envs

        # Define single and batched spaces
        single_obs_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.env.obs_dim,)
        )
        single_action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(self.env.action_dim,)
        )

        # Initialize VectorEnv attributes without passing invalid args to super().__init__()
        super().__init__()
        self.num_envs = self.num_envs
        self.single_observation_space = single_obs_space
        self.single_action_space = single_action_space
        self.observation_space = batch_space(single_obs_space, self.num_envs)
        self.action_space = batch_space(single_action_space, self.num_envs)

    def reset(self, seed=None, options=None):
        obs_dict, extras = self.env.reset()
        obs = obs_dict["policy"] if isinstance(obs_dict, dict) else obs_dict

        if extras is None:
            extras = {}

        extras["actor_observation_size"] = (self.env.obs_dim,)
        extras["critic_observation_size"] = (self.env.obs_dim,)
        extras["num_envs"] = self.num_envs

        return obs, extras

    def step(self, actions):
        if not isinstance(actions, torch.Tensor):
            actions = torch.tensor(actions, dtype=torch.float32, device=self.rl_device)
        else:
            actions = actions.to(self.rl_device)

        # Handle flattened or 1D actions
        if actions.ndim == 1:
            batch_size = actions.numel() // self.env.action_dim
            actions = actions.view(batch_size, self.env.action_dim)

        # Safety fallback: If FlashSAC passes a single-env action batch (size 1)
        # while num_envs is 1024, repeat it to match Dexmachina's parallel requirement
        if actions.shape[0] == 1 and self.num_envs > 1:
            actions = actions.repeat(self.num_envs, 1)

        assert actions.shape[0] == self.num_envs, f"Expected action batch size {self.num_envs}, got {actions.shape[0]}"

        actions = torch.clamp(actions, -self._clip_actions, self._clip_actions)

        obs_dict, rew, terminated, truncated, extras = self.env.step(actions)
        obs = obs_dict["policy"] if isinstance(obs_dict, dict) else obs_dict

        if extras is None:
            extras = {}

        extras["final_obs"] = obs.clone()
        if "log" in extras:
            extras["episode_info"] = extras["log"]

        return obs, rew, terminated, truncated, extras

    def render(self):
        """
        Delegates rendering to Genesis/Dexmachina base environment.
        Includes a safety fallback to prevent numpy AxisErrors if the camera is disabled.
        """
        if hasattr(self.env, "render"):
            image = self.env.render()
            if isinstance(image, torch.Tensor):
                image = image.cpu().numpy()

            if image is not None:
                # Ensure batch dimension exists for FlashSAC (B, C, H, W)
                if image.ndim == 3:
                    # If channels are last (H, W, C), move them to front (C, H, W) for WandB
                    if image.shape[-1] in [3, 4]:
                        image = np.transpose(image, (2, 0, 1))
                    image = image[np.newaxis, ...]
                return image

        # Fallback: If camera is disabled, return a blank black frame (Batch, Channels, Height, Width)
        return np.zeros((1, 3, 256, 256), dtype=np.uint8)

    def close(self):
        pass