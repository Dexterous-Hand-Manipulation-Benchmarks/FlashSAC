import math
import torch
import numpy as np
import gymnasium as gym


class GenesisFlashSACWrapper(gym.Wrapper):
    """Adapts Dexmachina Genesis environment for FlashSAC (Gymnasium 1.1 API)"""

    def __init__(self, env, rl_device: str, clip_obs: float = math.inf, clip_actions: float = math.inf):
        # Initialize standard Gym wrapper
        super().__init__(env)

        self.env = env
        self.rl_device = rl_device
        self.num_obs = env.num_obs
        self.num_actions = env.num_actions

        self._clip_obs = clip_obs
        self._clip_actions = clip_actions

        self.asymmetric_obs = True
        self.env_type = "dexmachina"
        self.max_episode_steps = env.max_episode_length

        # FlashSAC expects standard Gymnasium spaces to initialize neural networks
        self.observation_space = gym.spaces.Box(
            low=-self._clip_obs, high=self._clip_obs, shape=(self.num_obs,), dtype=np.float32
        )
        self.action_space = gym.spaces.Box(
            low=-self._clip_actions, high=self._clip_actions, shape=(self.num_actions,), dtype=np.float32
        )

    @property
    def num_envs(self) -> int:
        """Returns the number of vectorized environments."""
        return self.env.num_envs

    @property
    def device(self) -> str:
        """Returns the device for RL computations."""
        return self.rl_device

    def seed(self, seed=None):
        # Genesis handles seeding internally, but Gym API expects this signature
        pass

    def reset(self, seed=None, options=None):
        obs_dict, _ = self.env.reset()

        # Process observations and states
        obs, critic_obs = self._process_obs(obs_dict)

        # FlashSAC expects (obs, info) for reset
        info = {'observations': {'critic': critic_obs}} if self.asymmetric_obs else {}
        return obs, info

    def step(self, actions):
        # Clip the actions
        actions = torch.clamp(actions, -self._clip_actions, self._clip_actions)

        # Perform environment step
        obs_dict, rew, terminated, truncated, extras = self.env.step(actions)

        # Move time out information to the extras dict (for value bootstrapping)
        if not self.env.is_finite_horizon:
            extras["time_outs"] = truncated.to(device=self.rl_device)

        # Process observations and states
        obs, critic_obs = self._process_obs(obs_dict)

        # Move buffers to rl-device safely
        rew = rew.to(device=self.rl_device)
        terminated = terminated.to(device=self.rl_device)
        truncated = truncated.to(device=self.rl_device)

        extras = {
            k: v.to(device=self.rl_device, non_blocking=True) if hasattr(v, "to") else v
            for k, v in extras.items()
        }

        if self.asymmetric_obs:
            extras['observations'] = {'critic': critic_obs}

        # Remap extras from "log" to "episode" for logging compatibility
        if "log" in extras:
            extras["episode"] = extras.pop("log")

        # FlashSAC expects Gymnasium 1.1 output: obs, reward, terminated, truncated, info
        return obs, rew, terminated, truncated, extras

    def close(self):
        return self.env.close()

    @property
    def unwrapped(self):
        """Returns pure Genesis dexmachina env"""
        return self.env

    def _process_obs(self, obs_dict) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Processing of the observations and states from the environment."""
        # Process policy obs
        obs = obs_dict["policy"]
        obs = torch.clamp(obs, -self._clip_obs, self._clip_obs)

        # Check if asymmetric actor-critic or not
        if self.asymmetric_obs:
            states = obs_dict["critic"]
            states = torch.clamp(states, -self._clip_obs, self._clip_obs)
            return obs, states
        else:
            return obs, None