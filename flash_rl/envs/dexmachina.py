import math

import torch

from diffusion_rl.utils.wrappers.wrapper import TorchVecEnv


class GenesisEnv(TorchVecEnv):
    """Adapts dexmachina Genesis environment"""

    def __init__(self, env, rl_device: str, clip_obs: float = math.inf, clip_actions: float = math.inf):
        super().__init__(env, rl_device)
        self.num_obs = env.num_obs
        self.num_actions = env.num_actions

        self._clip_obs = clip_obs
        self._clip_actions = clip_actions

        self.asymmetric_obs = True

        self.env_type = "dexmachina"
        self.max_episode_steps = env.max_episode_length

    def seed(self):
        raise NotImplementedError

    def reset(self):
        obs_dict, _ = self.env.reset()
        # process observations and states
        obs, critic_obs = self._process_obs(obs_dict)
        return obs, {'observations': {'critic': critic_obs}}

    def step(self, actions):
        # clip the actions
        actions = torch.clamp(actions, -self._clip_actions, self._clip_actions)
        # perform environment step
        obs_dict, rew, terminated, truncated, extras = self.env.step(actions)

        # move time out information to the extras dict
        # this is only needed for infinite horizon tasks
        # note: only useful when `value_bootstrap` is True in the agent configuration
        if not self.env.is_finite_horizon:
            extras["time_outs"] = truncated.to(device=self.rl_device)

        # process observations and states
        obs, critic_obs = self._process_obs(obs_dict)
        # move buffers to rl-device
        # note: we perform clone to prevent issues when rl-device and sim-device are the same.
        rew = rew.to(device=self.rl_device)
        dones = (terminated | truncated).to(device=self.rl_device)
        extras = {
            k: v.to(device=self.rl_device, non_blocking=True) if hasattr(v, "to") else v for k, v in extras.items()
        }

        extras['observations'] = {'critic': critic_obs}

        # remap extras from "log" to "episode"
        if "log" in extras:
            extras["episode"] = extras.pop("log")

        return obs, rew, dones, truncated, extras

    def close(self):
        return self.env.close()

    @property
    def unwrapped(self):
        # returns pure Genesis dexmachina env
        return self.env

    def _process_obs(self, obs_dict) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Processing of the observations and states from the environment.

        Note:
            States typically refers to privileged observations for the critic function. It is typically used in
            asymmetric actor-critic algorithms.

        Args:
            obs_dict: The current observations from environment.

        Returns:
            If environment provides states, then a dictionary containing the observations and states is returned.
            Otherwise just the observations tensor is returned.
        """
        # self.obs_dict = obs_dict
        # process policy obs
        obs = obs_dict["policy"]
        # clip the observations
        obs = torch.clamp(obs, -self._clip_obs, self._clip_obs)

        # check if asymmetric actor-critic or not
        if self.asymmetric_obs:
            states = obs_dict["critic"]

            # convert to dictionary
            return obs, states
        else:
            return obs, None
